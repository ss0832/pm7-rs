// SPDX-License-Identifier: GPL-3.0-or-later

//! k-resolved periodic SCF.
//!
//! # What changes relative to the Γ point
//!
//! At the Γ point the real-space density matrix is the same for every lattice translation, so
//! everything collapses to one matrix and the molecular SCF loop can be reused verbatim (see
//! [`crate::hamiltonian::build_core_periodic`]). With a k-point mesh the translation index
//! survives: the Hamiltonian and the density are held as **real-space blocks** `H(T)`, `P(T)`,
//! the Bloch matrix `H(k) = Σ_T H(T) e^{ik·T}` is diagonalized per k, and the density comes
//! back by `P(T) = Σ_k w_k Re[e^{−ik·T} P(k)]`.
//!
//! Two structural facts keep this manageable:
//!
//! * `H(−T) = H(T)ᵀ` by construction, which makes every `H(k)` Hermitian exactly rather than to
//!   rounding, and makes `P(T)` real.
//! * `H(T)` is real, so `H(−k) = H(k)*`: the two k points have identical eigenvalues and
//!   conjugate eigenvectors. [`crate::pbc::KPointSet`] folds them onto one representative with
//!   double weight, which halves the diagonalizations with no approximation.
//!
//! # Which terms carry the translation index
//!
//! | term | where it lives |
//! |---|---|
//! | one-centre `U`, electron–core `e1b`/`e2a` | `H(0)` |
//! | resonance `β·S` | `H(T)` |
//! | Coulomb `J` | `F(0)` — it contracts the *on-site* density block of each atom, which is `P(0)` |
//! | long-range Coulomb (Ewald) | `F(0)` — built from the Mulliken charges, which are `P(0)` quantities |
//! | exchange `K` | `F(T)` |
//! | long-range exchange tail | `F(0)`; see the note in [`crate::hamiltonian::exchange_potential`] |
//!
//! The long-range exchange tail is the one term whose k-dispersion is neglected: it is taken as
//! `−P^σ(0)[μ_A, λ_B] · M̃_AB`, the T = 0 density block times the divergence-corrected lattice
//! sum. That is exact in the molecular limit, reduces *exactly* to the Γ-point treatment when the
//! mesh is 1×1×1 (so `KMesh::Gamma` and `KMesh::grid(1,1,1)` agree), and is small elsewhere
//! because `P(0)[μ_A, λ_B]` decays exponentially with `R_AB` while `M̃_AB` only goes as `1/R`.

use std::collections::HashMap;

use crate::cmatrix::CMatrix;
use crate::error::{Pm7Error, Result};
use crate::linalg::Matrix;
use crate::pbc::{KPoint, KPointSet, Smearing};

/// Real-space matrix blocks indexed by lattice translation.
///
/// The translation list is fixed at construction and closed under negation, so `block(−T)` is
/// always available — which is what lets a pair contribution be written into both `H(T)` and
/// `H(−T)` and keeps every `H(k)` Hermitian.
#[derive(Clone, Debug)]
pub struct BlochBlocks {
    translations: Vec<[i32; 3]>,
    blocks: Vec<Matrix>,
    index: HashMap<[i32; 3], usize>,
}

impl BlochBlocks {
    /// Allocate zero blocks for each translation. The list must contain `[0,0,0]` and be closed
    /// under negation; both are checked, because a violation would silently produce a
    /// non-Hermitian `H(k)`.
    pub fn new(translations: Vec<[i32; 3]>, nao: usize) -> Result<Self> {
        if !translations.contains(&[0, 0, 0]) {
            return Err(Pm7Error::InvalidInput(
                "translation set must contain the zero translation".into(),
            ));
        }
        for t in &translations {
            let neg = [-t[0], -t[1], -t[2]];
            if !translations.contains(&neg) {
                return Err(Pm7Error::InvalidInput(format!(
                    "translation set is not closed under negation: {t:?} present, {neg:?} missing"
                )));
            }
        }
        let index = translations
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, i))
            .collect();
        let blocks = translations
            .iter()
            .map(|_| Matrix::zeros(nao, nao))
            .collect();
        Ok(Self {
            translations,
            blocks,
            index,
        })
    }

    pub fn translations(&self) -> &[[i32; 3]] {
        &self.translations
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn nao(&self) -> usize {
        self.blocks.first().map(|b| b.rows).unwrap_or(0)
    }

    /// Index of a translation, or `None` when it is outside the set.
    #[inline]
    pub fn position(&self, t: [i32; 3]) -> Option<usize> {
        self.index.get(&t).copied()
    }

    #[inline]
    pub fn block(&self, i: usize) -> &Matrix {
        &self.blocks[i]
    }

    #[inline]
    pub fn block_mut(&mut self, i: usize) -> &mut Matrix {
        &mut self.blocks[i]
    }

    /// The block for translation `t`, or `None` when it is outside the set.
    #[inline]
    pub fn get(&self, t: [i32; 3]) -> Option<&Matrix> {
        self.position(t).map(|i| &self.blocks[i])
    }

    /// The block for translation `t`, folded into the Born–von Kármán residue range of an
    /// `n₁×n₂×n₃` mesh when `t` itself is not stored.
    ///
    /// Under Born–von Kármán boundary conditions `P(T + n_i a_i) = P(T)` exactly, so the fold is
    /// an identity rather than an approximation. It matters because the gradient's pair list can
    /// reach translations the density's block set does not name — the set covers the *core's*
    /// pairs and the residue representatives, and a wider correction cutoff can go further.
    pub fn folded(&self, t: [i32; 3], n: [usize; 3]) -> Option<&Matrix> {
        if let Some(b) = self.get(t) {
            return Some(b);
        }
        let fold = |x: i32, m: usize| -> i32 {
            let m = m as i32;
            let r = ((x % m) + m) % m;
            if 2 * r > m {
                r - m
            } else {
                r
            }
        };
        self.get([fold(t[0], n[0]), fold(t[1], n[1]), fold(t[2], n[2])])
    }

    /// The block for translation `t`, panicking if it is not in the set — a caller writing into
    /// a translation the list does not cover is a bug, not a runtime condition.
    #[inline]
    pub fn at_mut(&mut self, t: [i32; 3]) -> &mut Matrix {
        let i = self
            .position(t)
            .unwrap_or_else(|| panic!("translation {t:?} is not in the Bloch block set"));
        &mut self.blocks[i]
    }

    /// Set every element back to zero, reusing the allocation.
    pub fn clear(&mut self) {
        for b in &mut self.blocks {
            for v in b.as_mut_slice() {
                *v = 0.0;
            }
        }
    }

    /// `Σ_T block(T)` — the Γ-point matrix. Useful as a consistency check against the real Γ
    /// path, and as the starting guess.
    pub fn gamma_sum(&self) -> Matrix {
        let mut out = Matrix::zeros(self.nao(), self.nao());
        for b in &self.blocks {
            for (dst, src) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *dst += src;
            }
        }
        out
    }

    /// `M(k) = Σ_T M(T) e^{ik·T}`.
    pub fn at_k(&self, k: &KPoint) -> CMatrix {
        let mut out = CMatrix::zeros(self.nao());
        for (t, block) in self.translations.iter().zip(&self.blocks) {
            let phase = k.phase(*t);
            out.add_phase(block, phase.cos(), phase.sin());
        }
        out
    }

    /// Accumulate `w_k · Re[e^{−ik·T} P(k)]` into every block.
    ///
    /// With time-reversal folding the weight already stands for both `k` and `−k`, and
    /// `P(−k) = P(k)*` makes their sum exactly twice the real part — which is why only the real
    /// part is taken and no separate `−k` pass is needed.
    pub fn accumulate_from_k(&mut self, p_k: &CMatrix, k: &KPoint) {
        for (t, block) in self.translations.iter().zip(&mut self.blocks) {
            let phase = -k.phase(*t);
            let (c, s) = (phase.cos(), phase.sin());
            for (dst, (re, im)) in block
                .as_mut_slice()
                .iter_mut()
                .zip(p_k.re.iter().zip(&p_k.im))
            {
                *dst += k.weight * (c * re - s * im);
            }
        }
    }

    /// `Σ_T Σ_{μν} a(T)[μν] b(T)[μν]` — the trace that the periodic energy expression needs.
    pub fn frobenius_dot(&self, other: &BlochBlocks) -> f64 {
        self.blocks
            .iter()
            .zip(&other.blocks)
            .map(|(a, b)| a.frobenius_dot(b))
            .sum()
    }

    /// Root-mean-square difference, for SCF convergence testing.
    pub fn rms_diff(&self, other: &BlochBlocks) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (a, b) in self.blocks.iter().zip(&other.blocks) {
            for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
                sum += (x - y) * (x - y);
                count += 1;
            }
        }
        (sum / count.max(1) as f64).sqrt()
    }

    /// Enforce `M(−T) = M(T)ᵀ` by averaging the two, and report how far the input was from it.
    ///
    /// Contributions are written into both `T` and `−T` as they are generated, so the property
    /// should already hold; symmetrizing removes the accumulated rounding and returns the
    /// residual so a caller can assert on it instead of assuming.
    pub fn symmetrize(&mut self) -> f64 {
        let n = self.nao();
        let mut worst = 0.0_f64;
        for i in 0..self.translations.len() {
            let t = self.translations[i];
            let neg = [-t[0], -t[1], -t[2]];
            let j = self.position(neg).expect("closed under negation");
            if j < i {
                continue;
            }
            for a in 0..n {
                for b in 0..n {
                    let x = self.blocks[i][(a, b)];
                    let y = self.blocks[j][(b, a)];
                    worst = worst.max((x - y).abs());
                    let mean = 0.5 * (x + y);
                    self.blocks[i][(a, b)] = mean;
                    self.blocks[j][(b, a)] = mean;
                }
            }
        }
        worst
    }
}

/// Occupation numbers and the Fermi level for one SCF iteration.
#[derive(Clone, Debug)]
pub struct Occupations {
    /// `occupations[k][band]`, in `[0, 1]` per spin channel.
    pub occupations: Vec<Vec<f64>>,
    /// Fermi level in eV.
    pub fermi_ev: f64,
    /// Electronic entropy contribution `−TS` in eV (zero without smearing).
    pub entropy_ev: f64,
}

/// Fill `n_electrons` states across a k-point mesh.
///
/// `energies[k]` are the eigenvalues at k point `k`, and `weights[k]` its integration weight.
/// `spin_factor` is 2 for a spin-restricted calculation (each state holds two electrons) and 1
/// for one spin channel of an unrestricted one.
///
/// Without smearing this is a plain aufbau fill, and a metal will simply refuse to converge —
/// which is the honest outcome, not a silently wrong number. With smearing the Fermi level is
/// found by bisection on the total electron count, which is monotone in `E_F` and therefore
/// always brackets.
pub fn fill_bands(
    energies: &[Vec<f64>],
    weights: &[f64],
    n_electrons: f64,
    spin_factor: f64,
    smearing: Smearing,
) -> Result<Occupations> {
    if energies.is_empty() {
        return Err(Pm7Error::InvalidInput("no k points to fill".into()));
    }
    let target = n_electrons / spin_factor;

    if matches!(smearing, Smearing::None) {
        // Aufbau: sort every (k, band) state by energy and fill in order.
        let mut states: Vec<(f64, usize, usize)> = Vec::new();
        for (ik, e) in energies.iter().enumerate() {
            for (ib, &v) in e.iter().enumerate() {
                states.push((v, ik, ib));
            }
        }
        states.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut occ: Vec<Vec<f64>> = energies.iter().map(|e| vec![0.0; e.len()]).collect();
        let mut remaining = target;
        let mut fermi = states.first().map(|s| s.0).unwrap_or(0.0);
        for (energy, ik, ib) in states {
            if remaining <= 1.0e-12 {
                break;
            }
            let take = weights[ik].min(remaining);
            occ[ik][ib] = take / weights[ik];
            remaining -= take;
            fermi = energy;
        }
        if remaining > 1.0e-8 {
            return Err(Pm7Error::InvalidInput(format!(
                "not enough bands to hold {n_electrons} electrons ({remaining} left over)"
            )));
        }
        return Ok(Occupations {
            occupations: occ,
            fermi_ev: fermi,
            entropy_ev: 0.0,
        });
    }

    // Smeared filling: bisect on E_F. The electron count is monotone non-decreasing in E_F, so
    // widening the bracket until it straddles `target` always terminates.
    let all: Vec<f64> = energies.iter().flatten().copied().collect();
    let mut lo =
        all.iter().cloned().fold(f64::INFINITY, f64::min) - 10.0 * smearing.width_ev() - 1.0;
    let mut hi =
        all.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 10.0 * smearing.width_ev() + 1.0;
    let count = |ef: f64| -> f64 {
        energies
            .iter()
            .zip(weights)
            .map(|(e, w)| w * e.iter().map(|&x| occupation(x, ef, smearing)).sum::<f64>())
            .sum()
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1.0e-13 {
            break;
        }
    }
    let fermi = 0.5 * (lo + hi);
    let occupations: Vec<Vec<f64>> = energies
        .iter()
        .map(|e| e.iter().map(|&x| occupation(x, fermi, smearing)).collect())
        .collect();
    let entropy_ev: f64 = energies
        .iter()
        .zip(weights)
        .map(|(e, w)| {
            w * spin_factor
                * e.iter()
                    .map(|&x| entropy_term(x, fermi, smearing))
                    .sum::<f64>()
        })
        .sum();
    Ok(Occupations {
        occupations,
        fermi_ev: fermi,
        entropy_ev,
    })
}

/// Occupation of a state at energy `e` for the given Fermi level and broadening.
pub fn occupation(e: f64, fermi: f64, smearing: Smearing) -> f64 {
    match smearing {
        Smearing::None => {
            if e < fermi {
                1.0
            } else {
                0.0
            }
        }
        Smearing::FermiDirac { width_ev } => {
            let x = (e - fermi) / width_ev;
            // Written to avoid overflow at large |x|.
            if x > 0.0 {
                let z = (-x).exp();
                z / (1.0 + z)
            } else {
                1.0 / (1.0 + x.exp())
            }
        }
        // Gaussian broadening is Methfessel–Paxton of order 0, so one implementation serves
        // both and they cannot drift apart.
        Smearing::Gaussian { width_ev } => methfessel_paxton((e - fermi) / width_ev, 0),
        Smearing::MethfesselPaxton { width_ev, order } => {
            methfessel_paxton((e - fermi) / width_ev, order)
        }
    }
}

/// Physicists' Hermite polynomials `H_0 … H_m` at `x`, from `H_{k+1} = 2x H_k − 2k H_{k−1}`.
fn hermite_up_to(m: usize, x: f64) -> Vec<f64> {
    let mut h = Vec::with_capacity(m + 1);
    h.push(1.0);
    if m == 0 {
        return h;
    }
    h.push(2.0 * x);
    for k in 1..m {
        let next = 2.0 * x * h[k] - 2.0 * (k as f64) * h[k - 1];
        h.push(next);
    }
    h
}

/// `A_n = (−1)ⁿ / (n! 4ⁿ √π)`, the Methfessel–Paxton expansion coefficients.
fn mp_coefficients(order: usize) -> Vec<f64> {
    let mut a = Vec::with_capacity(order + 1);
    let mut value = 1.0 / std::f64::consts::PI.sqrt();
    a.push(value);
    for n in 1..=order {
        value *= -1.0 / (4.0 * n as f64);
        a.push(value);
    }
    a
}

/// Methfessel–Paxton occupation of order `N`:
/// `f_N(x) = ½ erfc(x) + Σ_{n=1..N} A_n H_{2n−1}(x) e^{−x²}`.
///
/// Order 0 is plain Gaussian broadening. Higher orders make the occupation approach a step
/// function faster in the sense that the *integrated* density of states is accurate to higher
/// order in the width — at the price of occupations that can fall slightly outside `[0, 1]`,
/// which is expected and is left unclamped so a caller can see it.
fn methfessel_paxton(x: f64, order: usize) -> f64 {
    let mut f = 0.5 * crate::special::erfc(x);
    if order == 0 {
        return f;
    }
    let gauss = (-x * x).exp();
    let h = hermite_up_to(2 * order, x);
    let a = mp_coefficients(order);
    for n in 1..=order {
        f += a[n] * h[2 * n - 1] * gauss;
    }
    f
}

/// `−TS`-style entropy contribution of one state, in eV (already carrying the sign that makes
/// it lower the free energy).
///
/// For Fermi–Dirac this is the usual `k_BT [f ln f + (1−f) ln(1−f)]`. For Methfessel–Paxton of
/// order `N` (Gaussian being `N = 0`) it is `−½ width · A_N H_{2N}(x) e^{−x²}`, which for `N = 0`
/// reduces to the familiar `−width e^{−x²} / (2√π)`.
fn entropy_term(e: f64, fermi: f64, smearing: Smearing) -> f64 {
    match smearing {
        Smearing::None => 0.0,
        Smearing::FermiDirac { width_ev } => {
            let f = occupation(e, fermi, smearing);
            let s = |x: f64| if x > 1.0e-14 { x * x.ln() } else { 0.0 };
            width_ev * (s(f) + s(1.0 - f))
        }
        Smearing::Gaussian { width_ev } => {
            let x = (e - fermi) / width_ev;
            -0.5 * width_ev * mp_coefficients(0)[0] * (-x * x).exp()
        }
        Smearing::MethfesselPaxton { width_ev, order } => {
            let x = (e - fermi) / width_ev;
            let h = hermite_up_to(2 * order, x);
            let a = mp_coefficients(order);
            -0.5 * width_ev * a[order] * h[2 * order] * (-x * x).exp()
        }
    }
}

/// Bands and weights of a k-point set, in the order [`KPointSet`] stores them.
pub fn kpoint_weights(set: &KPointSet) -> Vec<f64> {
    set.points.iter().map(|p| p.weight).collect()
}

/// The converged state of a k-resolved SCF.
pub struct KScfState {
    /// Real-space total density blocks `P(T)`.
    pub density: BlochBlocks,
    /// Real-space spin-density blocks `Pα(T) − Pβ(T)`, or `None` for a closed shell.
    pub spin_density: Option<BlochBlocks>,
    /// Band energies per k point, ascending.
    pub band_energies: Vec<Vec<f64>>,
    /// Electronic energy per unit cell, in eV.
    pub electronic_ev: f64,
    pub fermi_ev: f64,
    pub entropy_ev: f64,
    pub iterations: usize,
    pub converged: bool,
    pub density_error: f64,
    pub unrestricted: bool,
    /// Number of k points diagonalized (after time-reversal folding).
    pub n_kpoints: usize,
}

/// One spin channel's Γ-point orbitals: energies and coefficients.
pub struct GammaOrbitals {
    pub energies: Vec<f64>,
    pub coefficients: Matrix,
    pub energies_beta: Option<Vec<f64>>,
    pub coefficients_beta: Option<Matrix>,
}

/// The Γ-point eigenpair of the **converged** Hamiltonian, for molecular-style orbital reporting.
///
/// This exists because the obvious shortcuts are both wrong. Reporting the pre-SCF Γ solve's
/// coefficients gives eigenvectors of a Hamiltonian built from the *starting* density; pairing
/// them with `band_energies[0]` — the first expanded k point, which on a shifted mesh is not even
/// Γ — describes two different matrices at once. Rebuilding the Fock from the converged density
/// and diagonalizing it costs one Fock build and one diagonalization, which is nothing next to a
/// k-point SCF, and produces an eigenpair that actually belongs together.
///
/// The gap it reports is the **Γ** gap, not a zone-wide one; [`band_structure`] is the k-resolved
/// answer and `fermi_ev` the mesh-wide one. [`crate::scf::OrbitalSource`] labels which is which
/// so a caller cannot mistake them.
pub fn gamma_orbitals(
    molecule: &crate::system::Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    state: &KScfState,
) -> Result<GammaOrbitals> {
    use crate::linalg::symmetric_eigen;

    match &state.spin_density {
        None => {
            let half = scale_blocks(&state.density, 0.5);
            let fock = crate::fock::build_fock_spin_bloch(
                molecule,
                basis,
                params,
                core,
                &state.density,
                &half,
            )?;
            let (energies, coefficients) = symmetric_eigen(&fock.gamma_sum())?;
            Ok(GammaOrbitals {
                energies,
                coefficients,
                energies_beta: None,
                coefficients_beta: None,
            })
        }
        Some(spin) => {
            let alpha = scale_blocks(&add_blocks(&state.density, spin), 0.5);
            let beta = scale_blocks(&sub_blocks(&state.density, spin), 0.5);
            let fa = crate::fock::build_fock_spin_bloch(
                molecule,
                basis,
                params,
                core,
                &state.density,
                &alpha,
            )?;
            let fb = crate::fock::build_fock_spin_bloch(
                molecule,
                basis,
                params,
                core,
                &state.density,
                &beta,
            )?;
            let (energies, coefficients) = symmetric_eigen(&fa.gamma_sum())?;
            let (eb, cb) = symmetric_eigen(&fb.gamma_sum())?;
            Ok(GammaOrbitals {
                energies,
                coefficients,
                energies_beta: Some(eb),
                coefficients_beta: Some(cb),
            })
        }
    }
}

/// Diagonalize `F(k)` at every k, fill the bands, and rebuild the real-space density.
///
/// `spin_factor` is 2 for a spin-restricted run and 1 for one channel of an unrestricted one.
///
/// `per_k` adds a Hermitian operator to `H(k)`, one matrix per k point in the order of
/// `kpoints.points`. Only the Berry-phase finite field passes anything: its electric-enthalpy
/// term couples **neighbouring** k points, so it is a `k`-dependent operator that is not a Bloch
/// sum of any `H(T)` and cannot live where every other term does. `None` is the ordinary path.
#[allow(clippy::type_complexity)]
fn density_from_fock_with(
    fock: &BlochBlocks,
    kpoints: &KPointSet,
    n_electrons: f64,
    spin_factor: f64,
    smearing: Smearing,
    per_k: Option<&[CMatrix]>,
) -> Result<(BlochBlocks, Vec<Vec<f64>>, Occupations)> {
    use rayon::prelude::*;
    if let Some(terms) = per_k {
        if terms.len() != kpoints.points.len() {
            return Err(Pm7Error::InvalidInput(format!(
                "the per-k operator has {} matrices against {} k points",
                terms.len(),
                kpoints.points.len()
            )));
        }
    }
    // The k points are independent; diagonalizing them in parallel and collecting in order
    // keeps the result independent of the thread count.
    let solved: Vec<(Vec<f64>, CMatrix)> = kpoints
        .points
        .par_iter()
        .enumerate()
        .map(|(index, k)| {
            let mut hk = fock.at_k(k);
            if let Some(terms) = per_k {
                let extra = &terms[index];
                for i in 0..hk.n {
                    for j in 0..hk.n {
                        let (ar, ai) = hk.get(i, j);
                        let (br, bi) = extra.get(i, j);
                        hk.set(i, j, ar + br, ai + bi);
                    }
                }
            }
            hk.hermitian_eigen()
        })
        .collect::<Result<Vec<_>>>()?;
    let energies: Vec<Vec<f64>> = solved.iter().map(|(e, _)| e.clone()).collect();
    let weights = kpoint_weights(kpoints);
    let occ = fill_bands(&energies, &weights, n_electrons, spin_factor, smearing)?;

    let mut density = BlochBlocks::new(fock.translations().to_vec(), fock.nao())?;
    for (ik, k) in kpoints.points.iter().enumerate() {
        let pk = solved[ik]
            .1
            .weighted_density(&occ.occupations[ik], spin_factor);
        density.accumulate_from_k(&pk, k);
    }
    density.symmetrize();
    Ok((density, energies, occ))
}

/// Total number of electrons represented by a real-space density's `T = 0` block.
pub fn electron_count(density: &BlochBlocks) -> f64 {
    let zero = density.position([0, 0, 0]).expect("zero translation");
    let block = density.block(zero);
    (0..block.rows).map(|i| block[(i, i)]).sum()
}

/// The density residual split into the three channels that fail for different reasons.
///
/// One RMS number cannot tell a mixing problem from a preconditioning one, or either from a fill
/// that is flipping an occupation. These can:
///
/// * `charge` is the per-atom Mulliken residual `r_A = Σ_{μ∈A} R(0)[μμ]`. It is the **only** part
///   the long-range Ewald term couples to — that term reaches the Fock matrix through the Mulliken
///   charges of `P(0)` and nothing else — so it is the only channel that can slosh, and the only
///   one a Kerker preconditioner could act on.
/// * `onsite` is the `T = 0` block with each atom's diagonal trace projected out: on-site
///   polarization and intra-cell bond order.
/// * `intercell` is the worst `T ≠ 0` block, the bond-order tail between cells.
///
/// # What this measured, and what it ruled out
///
/// `docs/performance.md` diagnosed the periodic SCF's slow cases as charge sloshing and prescribed
/// Kerker. On the case that actually stalls here — zinc-blende ZnS on a 3×3×3 mesh — that is not
/// what is happening. At iteration 200 of 200:
///
/// ```text
/// dP 2.739e-7   charge 5.300e-9   onsite 4.636e-6   intercell 3.605e-6   fractional 0
/// ```
///
/// The charge channel has fallen nine orders and is the **best** converged of the three. What is
/// stuck is bond order, creeping down 0.07 % per iteration — a spectral radius near 0.9993. The
/// occupations are integral at every iteration and the gap is 6.4 eV, so it is not an occupation
/// flip either, and Fermi smearing at 0.02 and 0.05 eV converges the same cell in 27 and 19
/// iterations to the same energy to 2e-11 eV.
///
/// So Kerker would precondition a channel that is already converged. The tool that measures the
/// stuck one is the commutator `[F(k), P(k)]`, and wiring that in needs the mixing scheme reworked
/// first: the loop mixes in *real space*, so the density a Fock was built from is not a projector
/// at any k, and CDIIS's (Fock, error) pairing has nothing consistent to stand on. Attempted and
/// backed out rather than shipped — it turned a monotone creep into an oscillation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidualChannels {
    pub charge: f64,
    pub onsite: f64,
    pub intercell: f64,
}

/// Decompose `new − old` into [`ResidualChannels`].
pub fn residual_channels(
    old: &BlochBlocks,
    new: &BlochBlocks,
    basis: &crate::basis::Basis,
) -> ResidualChannels {
    let mut out = ResidualChannels::default();
    let home = old.position([0, 0, 0]);
    for (i, (a, b)) in old.blocks.iter().zip(&new.blocks).enumerate() {
        if Some(i) == home {
            let mut diagonal = vec![0.0; basis.atom_offset.len()];
            for (atom, slot) in diagonal.iter_mut().enumerate() {
                let start = basis.atom_offset[atom];
                for orbital in 0..basis.atom_norb[atom] {
                    let mu = start + orbital;
                    *slot += b[(mu, mu)] - a[(mu, mu)];
                }
            }
            out.charge = diagonal.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let mut sum = 0.0;
            for (atom, &trace) in diagonal.iter().enumerate() {
                let start = basis.atom_offset[atom];
                let n = basis.atom_norb[atom];
                let mean = if n > 0 { trace / n as f64 } else { 0.0 };
                for orbital in 0..n {
                    let mu = start + orbital;
                    for nu in 0..a.cols {
                        let d = b[(mu, nu)] - a[(mu, nu)] - if mu == nu { mean } else { 0.0 };
                        sum += d * d;
                    }
                }
            }
            out.onsite = sum.sqrt();
        } else {
            let mut sum = 0.0;
            for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
                sum += (y - x) * (y - x);
            }
            out.intercell = out.intercell.max(sum.sqrt());
        }
    }
    out
}

/// Periodic electronic energy per unit cell, `½ Σ_T Σ_{μν} P(T)[μν] (H(T) + F(T))[μν]`.
pub fn electronic_energy(density: &BlochBlocks, core: &BlochBlocks, fock: &BlochBlocks) -> f64 {
    0.5 * (density.frobenius_dot(core) + density.frobenius_dot(fock))
}

/// Spin-resolved periodic electronic energy,
/// `½ Σ_T [ P_tot(T)·H(T) + Σ_σ P^σ(T)·F^σ(T) ]`.
pub fn electronic_energy_spin(
    p_tot: &BlochBlocks,
    p_alpha: &BlochBlocks,
    p_beta: &BlochBlocks,
    core: &BlochBlocks,
    f_alpha: &BlochBlocks,
    f_beta: &BlochBlocks,
) -> f64 {
    0.5 * (p_tot.frobenius_dot(core)
        + p_alpha.frobenius_dot(f_alpha)
        + p_beta.frobenius_dot(f_beta))
}

/// Add two block sets elementwise into a fresh one.
pub fn add_blocks(a: &BlochBlocks, b: &BlochBlocks) -> BlochBlocks {
    let mut out = a.clone();
    for i in 0..out.len() {
        let src = b.block(i).as_slice().to_vec();
        for (dst, s) in out.block_mut(i).as_mut_slice().iter_mut().zip(src) {
            *dst += s;
        }
    }
    out
}

/// Subtract `b` from `a` elementwise into a fresh set.
pub fn sub_blocks(a: &BlochBlocks, b: &BlochBlocks) -> BlochBlocks {
    let mut out = a.clone();
    for i in 0..out.len() {
        let src = b.block(i).as_slice().to_vec();
        for (dst, s) in out.block_mut(i).as_mut_slice().iter_mut().zip(src) {
            *dst -= s;
        }
    }
    out
}

/// Linear mix `(1 − w)·old + w·new`.
pub fn mix_blocks(old: &BlochBlocks, new: &BlochBlocks, w: f64) -> BlochBlocks {
    let mut out = old.clone();
    for i in 0..out.len() {
        let src = new.block(i).as_slice().to_vec();
        for (dst, s) in out.block_mut(i).as_mut_slice().iter_mut().zip(src) {
            *dst = (1.0 - w) * *dst + w * s;
        }
    }
    out
}

/// Pulay (Anderson) mixing of the real-space density blocks.
///
/// Plain linear damping does not converge the k-point SCF for a covalent solid: silicon stalls
/// at a density error of `3e-4` even after 200 iterations, and gets *worse* as the mesh is
/// refined, because each added k point adds another slowly-decaying mode. Pulay mixing on the
/// density residual `R_i = P_out(i) − P_in(i)` fixes that without needing a commutator error
/// vector per k point — the residual is already the natural error measure for a fixed-point
/// iteration on the density.
///
/// The next input is `Σ_i c_i (P_in(i) + β R_i)` with `c` minimizing `|Σ c_i R_i|²` subject to
/// `Σ c_i = 1`.
pub struct DensityMixer {
    inputs: Vec<BlochBlocks>,
    residuals: Vec<BlochBlocks>,
    /// How much of the residual to fold in alongside the extrapolation.
    beta: f64,
    /// Longest history kept. Too long and the least-squares system goes singular.
    depth: usize,
    /// Smallest residual norm seen since the last restart, used to detect a history that has
    /// stopped helping.
    best: f64,
    /// Consecutive iterations without improvement.
    stalled: usize,
}

impl DensityMixer {
    pub fn new(beta: f64, depth: usize) -> Self {
        Self {
            inputs: Vec::new(),
            residuals: Vec::new(),
            beta,
            depth: depth.max(1),
            best: f64::INFINITY,
            stalled: 0,
        }
    }

    /// Feed one iteration's input and output densities, and get the next input.
    pub fn next(&mut self, input: &BlochBlocks, output: &BlochBlocks) -> BlochBlocks {
        let residual = sub_blocks(output, input);
        let norm = residual.frobenius_dot(&residual).sqrt();
        // A history whose vectors have gone nearly linearly dependent extrapolates into noise
        // and the residual stops falling. Restarting costs a few iterations and is far better
        // than stalling just short of the tolerance, which is what happens without this.
        //
        // The 5 % threshold is deliberately kept. Relaxing it to "any new minimum is progress"
        // was tried, on the reasoning that a stiff iteration converging at 0.993 per step is
        // making progress and should not have its history wiped every six iterations. It does
        // help the stiff cases -- ZnS at 4x4x4 went from 135 iterations to 37 -- but it breaks
        // the *wandering* ones, where the residual keeps touching new minima while the energy is
        // still moving by 1e-3 eV, so the counter never fires and the run never recovers. ZnS at
        // 2x2x2 and 6x6x6 both converge with this threshold and both fail without it.
        //
        // What actually caused the fixed-rate creep is below, in `pulay_coefficients` and in the
        // pruning: the history was being rejected for being *small* rather than dependent, and
        // one rejection dropped the mixer to plain damping for the rest of the run. With those
        // fixed this threshold is doing the job it was written for.
        if norm < self.best * 0.95 {
            self.best = norm;
            self.stalled = 0;
        } else {
            self.stalled += 1;
            if self.stalled >= 6 {
                self.inputs.clear();
                self.residuals.clear();
                self.best = norm;
                self.stalled = 0;
            }
        }
        self.inputs.push(input.clone());
        self.residuals.push(residual);
        if self.inputs.len() > self.depth {
            self.inputs.remove(0);
            self.residuals.remove(0);
        }
        let n = self.residuals.len();
        if n == 1 {
            // Nothing to extrapolate from yet: a plain damped step.
            return mix_blocks(input, output, self.beta);
        }
        // **Drop the oldest vector and retry rather than giving up on the whole history.** A
        // rejected solve means the *oldest* vectors have gone nearly dependent with the newest,
        // and the newest are the ones worth keeping. Falling straight back to a damped step, as
        // this did through v0.2.2, throws away a history that is still perfectly usable at depth
        // three or four -- and because the history then keeps growing, the next iteration is
        // rejected too, so one bad solve degraded the mixer to plain damping for the rest of the
        // run. That is the other half of the fixed-rate creep.
        let mut first = 0usize;
        let coefficients = loop {
            if n - first < 2 {
                break None;
            }
            if let Some(c) = pulay_coefficients(&self.residuals[first..]) {
                break Some(c);
            }
            first += 1;
        };
        let Some(c) = coefficients else {
            return mix_blocks(input, output, self.beta);
        };
        // Whatever prefix was dropped is dropped for good; keeping it would have the next
        // iteration re-derive the same rejection.
        if first > 0 {
            self.inputs.drain(..first);
            self.residuals.drain(..first);
        }
        let nao = input.nao();
        let mut out =
            BlochBlocks::new(input.translations().to_vec(), nao).expect("same translation set");
        for (i, &ci) in c.iter().enumerate() {
            for b in 0..out.len() {
                let inp = self.inputs[i].block(b).as_slice().to_vec();
                let res = self.residuals[i].block(b).as_slice().to_vec();
                for (k, dst) in out.block_mut(b).as_mut_slice().iter_mut().enumerate() {
                    *dst += ci * (inp[k] + self.beta * res[k]);
                }
            }
        }
        out
    }

    /// Drop the history, e.g. after a step that made things worse.
    pub fn reset(&mut self) {
        self.inputs.clear();
        self.residuals.clear();
        self.best = f64::INFINITY;
        self.stalled = 0;
    }
}

/// Solve the Pulay least-squares problem `min |Σ c_i R_i|²` with `Σ c_i = 1`.
///
/// Returns `None` when the bordered system is singular or the coefficients come out
/// unreasonably large, which is the signal that the history has gone linearly dependent and the
/// caller should fall back to a damped step.
fn pulay_coefficients(residuals: &[BlochBlocks]) -> Option<Vec<f64>> {
    let n = residuals.len();
    // **Scale the Gram matrix to a unit diagonal.** Its raw entries are `<r_i, r_j>`, which
    // shrink like the residual squared -- by the time the SCF is at 1e-12 they are around 1e-24,
    // and `solve_linear`'s pivot guard is an *absolute* test. So a perfectly well-conditioned
    // history was being rejected for being small rather than for being dependent, `next` fell
    // back to a plain damped step, and the run crept to the iteration limit at a fixed rate.
    //
    // `docs/performance.md` records this exact fix being made for the molecular CDIIS ("the `B`
    // matrix is now scaled to a unit diagonal so its pivot guard is a relative test"); the
    // periodic mixer never received it. With the scaling the test measures conditioning, which
    // is what it was always meant to measure.
    let diagonal: Vec<f64> = (0..n)
        .map(|i| residuals[i].frobenius_dot(&residuals[i]))
        .collect();
    if diagonal.iter().any(|d| !d.is_finite() || *d <= 0.0) {
        return None;
    }
    let scale: Vec<f64> = diagonal.iter().map(|d| d.sqrt()).collect();

    let mut a = crate::linalg::Matrix::zeros(n + 1, n + 1);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = residuals[i].frobenius_dot(&residuals[j]) / (scale[i] * scale[j]);
        }
        a[(i, n)] = -1.0;
        a[(n, i)] = -1.0;
    }
    a[(n, n)] = 0.0;
    let mut b = vec![0.0; n + 1];
    b[n] = -1.0;
    let solution = crate::linalg::solve_linear(&a, &b).ok()?;
    // Undo the scaling: the constraint is `sum c_i = 1` on the *unscaled* coefficients.
    let mut c: Vec<f64> = (0..n).map(|i| solution[i] / scale[i]).collect();
    let total: f64 = c.iter().sum();
    if !total.is_finite() || total.abs() < 1.0e-12 {
        return None;
    }
    for v in c.iter_mut() {
        *v /= total;
    }
    if c.iter().any(|v| !v.is_finite() || v.abs() > 1.0e3) {
        return None;
    }
    Some(c)
}

/// Scale every block by `s`.
pub fn scale_blocks(blocks: &BlochBlocks, s: f64) -> BlochBlocks {
    let mut out = blocks.clone();
    for i in 0..out.len() {
        for v in out.block_mut(i).as_mut_slice() {
            *v *= s;
        }
    }
    out
}

/// Run the k-resolved SCF to convergence.
///
/// Restricted and unrestricted are handled by the same loop: RHF is the special case
/// `Pα = Pβ = ½P`, and the only branch is whether the two spin channels get their own Fock and
/// their own filling. Both spins share a **single Fermi level**, determined from the combined
/// state list, so a partially occupied band is filled consistently across spins.
#[allow(clippy::too_many_arguments)]
pub fn run_kpoint_scf(
    molecule: &crate::system::Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    kpoints: &KPointSet,
    n_alpha: usize,
    n_beta: usize,
    options: &crate::scf::Pm7Options,
) -> Result<KScfState> {
    run_kpoint_scf_with_terms(
        molecule, basis, params, core, kpoints, n_alpha, n_beta, options, None,
    )
}

/// [`run_kpoint_scf`], with an optional Hermitian operator added to `H(k)` at every k point.
///
/// One matrix per point of `kpoints`, in order. Only the Berry-phase finite field uses this: its
/// electric-enthalpy term couples **neighbouring** k points, so it is a `k`-dependent operator
/// that is not a Bloch sum of any `H(T)` and cannot live where every other term does. The operator
/// is held fixed across the SCF and updated by an outer loop; see [`crate::pbc::finite_field`].
///
/// `None` is the ordinary path and is bit-identical to not having the parameter.
///
/// # The smearing fallback
///
/// If the unsmeared run does not converge, this retries down a ladder of Fermi widths and keeps
/// the first that does. It can only ever fire on a run that would otherwise have **failed**, so
/// nothing that converges today can move, and the fallback reports itself on stderr.
///
/// It is here because an unsmeared aufbau fill does not merely converge slowly on a hard cell --
/// it can converge to a solution that **breaks the crystal's symmetry**. Zinc-blende ZnS at a
/// 4x4x4 mesh is the case that found it. Its zone-centre valence triplet, which cubic Td requires
/// to be threefold degenerate, comes out as
///
/// ```text
/// unsmeared   -7.68230600  -7.40549900  -7.40549900     split by 0.28 eV
/// smeared     -7.35428100  -7.35428100  -7.35428100     degenerate
/// ```
///
/// and the splitting wanders with the mesh (1.15 eV at 2x2x2, 0.016 eV at 6x6x6) rather than
/// converging away. Everything downstream inherits it: the dynamical matrix then fails
/// `R D R^T = D` by more than its own norm, and the triply degenerate optical phonon splits into a
/// singlet and a doublet. Diamond, GaAs and CdS are exactly symmetric on the same tests.
///
/// The gap is 6.4 eV, so smearing cannot change an occupation *in the converged solution* -- the
/// entropy term is zero and the answer is the unsmeared one. What it changes is the path: the
/// aufbau fill is discontinuous in the band energies, and on the way to convergence it makes hard
/// choices between states that are momentarily near-degenerate, which is what locks in the broken
/// solution. This is the standard remedy and the standard reason for it.
///
/// The entropy is checked rather than assumed. A cell that genuinely needs the smearing -- a metal
/// -- keeps a non-zero `entropy_ev`, and that is reported instead of being passed off as a
/// zero-temperature answer.
#[allow(clippy::too_many_arguments)]
pub fn run_kpoint_scf_with_terms(
    molecule: &crate::system::Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    kpoints: &KPointSet,
    n_alpha: usize,
    n_beta: usize,
    options: &crate::scf::Pm7Options,
    per_k: Option<&[CMatrix]>,
) -> Result<KScfState> {
    let first = run_kpoint_scf_once(
        molecule, basis, params, core, kpoints, n_alpha, n_beta, options, per_k,
    )?;
    let asked_for_smearing = options
        .pbc_for(molecule)
        .map(|p| !matches!(p.smearing, Smearing::None))
        .unwrap_or(false);
    if first.converged || asked_for_smearing {
        return Ok(first);
    }

    // Widest first: a wide window is the one most likely to converge, and each rung narrows the
    // approximation. The ladder stops at the first success rather than walking to the end,
    // because a converged wide run says nothing a converged narrow one does not.
    for width in [0.20_f64, 0.10, 0.05, 0.02, 0.01] {
        let mut retry = options.clone();
        if let Some(mut pbc) = retry.pbc_for(molecule) {
            pbc.smearing = Smearing::FermiDirac { width_ev: width };
            retry.pbc = Some(pbc);
        } else {
            break;
        }
        let Ok(state) = run_kpoint_scf_once(
            molecule, basis, params, core, kpoints, n_alpha, n_beta, &retry, per_k,
        ) else {
            continue;
        };
        if !state.converged {
            continue;
        }
        // **A rung that converges is not automatically an answer.** Only a solution whose entropy
        // vanishes has every state fully occupied or empty, and only then is the smeared result
        // *identical* to the zero-temperature one the caller asked for -- the smearing having
        // changed the path and not the result. That is the whole justification for substituting
        // it silently, so it is checked rather than assumed.
        //
        // The check is not academic. On ZnS at a 4x4x4 mesh the 0.20 eV rung converges in 20
        // iterations with a *non-zero* entropy, and its zone-centre triplet is split by 1.19 eV;
        // the 0.10 eV rung converges to the symmetric solution with zero entropy. Accepting the
        // first rung that merely converged would have returned a free energy dressed as an energy,
        // and a broken spectrum with it.
        if state.entropy_ev.abs() > 1.0e-9 {
            continue;
        }
        if std::env::var_os("PM7_QUIET").is_none() {
            eprintln!(
                "pm7-rs: the unsmeared k-point SCF stopped at {:.3e} after {} iterations; a Fermi \
                 smearing of {width} eV converged it in {} iterations, with zero entropy -- every \
                 state is fully occupied or empty, so this is the zero-temperature answer and the \
                 smearing changed the path rather than the result. An unsmeared aufbau fill is \
                 discontinuous in the band energies and can settle into a symmetry-broken \
                 solution; that is what this avoids. Pass --smearing to choose the width yourself.",
                first.density_error, first.iterations, state.iterations
            );
        }
        return Ok(state);
    }
    // Nothing on the ladder helped; hand back the original so the caller's error says what the
    // run actually did rather than what the last rung did.
    Ok(first)
}

#[allow(clippy::too_many_arguments)]
fn run_kpoint_scf_once(
    molecule: &crate::system::Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    kpoints: &KPointSet,
    n_alpha: usize,
    n_beta: usize,
    options: &crate::scf::Pm7Options,
    per_k: Option<&[CMatrix]>,
) -> Result<KScfState> {
    let core_bloch = core.bloch.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput("k-point SCF needs the translation-resolved core".into())
    })?;
    let pbc = options
        .pbc_for(molecule)
        .ok_or_else(|| Pm7Error::InvalidInput("k-point SCF needs a periodic cell".into()))?;
    let translations = core_bloch.translations().to_vec();
    let nao = basis.nao;
    let unrestricted =
        n_alpha != n_beta || options.reference == crate::scf::ScfReference::Unrestricted;

    // Initial guess: MOPAC's smeared atomic density on the T = 0 block and nothing between
    // cells. Starting with `P(T) = P(0)` for every T would be the Γ-point guess, which for a
    // small cell is a *worse* starting point than no inter-cell density at all.
    let n_electrons = (n_alpha + n_beta) as f64;
    let sad = crate::scf::sad_density(molecule, basis, params, n_electrons)?;
    let mut p_alpha = BlochBlocks::new(translations.clone(), nao)?;
    let mut p_beta = BlochBlocks::new(translations.clone(), nao)?;
    {
        let scale = if unrestricted {
            (
                n_alpha as f64 / n_electrons.max(1.0),
                n_beta as f64 / n_electrons.max(1.0),
            )
        } else {
            (0.5, 0.5)
        };
        let za = p_alpha.at_mut([0, 0, 0]);
        for (dst, src) in za.as_mut_slice().iter_mut().zip(sad.as_slice()) {
            *dst = scale.0 * src;
        }
        let zb = p_beta.at_mut([0, 0, 0]);
        for (dst, src) in zb.as_mut_slice().iter_mut().zip(sad.as_slice()) {
            *dst = scale.1 * src;
        }
    }

    let mut energies_alpha: Vec<Vec<f64>> = Vec::new();
    let mut fermi_ev = 0.0;
    let mut entropy_ev = 0.0;
    let mut converged = false;
    let mut density_error = f64::INFINITY;
    let mut iterations = 0usize;
    // Pulay mixing on the density residual. Plain damping does not converge a covalent solid
    // here — silicon stalls at 3e-4 and degrades as the mesh is refined.
    let mut mixer_alpha = DensityMixer::new(0.4, 12);
    let mut mixer_beta = DensityMixer::new(0.4, 12);
    // Per-iteration trace to stderr, the k-point twin of the molecular one in `scf.rs`, and
    // hoisted out of the loop for the same reason. It reports the residual **by channel** and the
    // fractional-occupation count, because a stalled periodic SCF has three distinct causes and
    // one RMS number distinguishes none of them. See [`ResidualChannels`].
    let trace = std::env::var_os("PM7_SCF_TRACE").is_some();
    let mut previous_energy = f64::NAN;
    let mut last_channels = ResidualChannels::default();
    let mut last_energy_step = f64::NAN;
    let mut last_fractional = 0usize;

    for iteration in 1..=options.max_scf {
        iterations = iteration;
        let p_tot = add_blocks(&p_alpha, &p_beta);
        let f_alpha =
            crate::fock::build_fock_spin_bloch(molecule, basis, params, core, &p_tot, &p_alpha)?;
        let (new_alpha, e_alpha, occ_alpha) =
            density_from_fock_with(&f_alpha, kpoints, n_alpha as f64, 1.0, pbc.smearing, per_k)?;
        // The β Fock is needed only to fill that channel; the reported energy is rebuilt at the
        // converged density after the loop, so nothing here has to survive the iteration.
        let (new_beta, occ_beta) = if unrestricted {
            let f_beta =
                crate::fock::build_fock_spin_bloch(molecule, basis, params, core, &p_tot, &p_beta)?;
            let (nb, _, ob) =
                density_from_fock_with(&f_beta, kpoints, n_beta as f64, 1.0, pbc.smearing, per_k)?;
            (nb, ob)
        } else {
            (new_alpha.clone(), occ_alpha.clone())
        };

        let new_tot = add_blocks(&new_alpha, &new_beta);
        density_error = new_tot.rms_diff(&p_tot);
        energies_alpha = e_alpha;
        fermi_ev = occ_alpha.fermi_ev;
        entropy_ev = occ_alpha.entropy_ev
            + if unrestricted {
                occ_beta.entropy_ev
            } else {
                0.0
            };

        // The diagnosis is collected whether or not it is printed, because it is what the failure
        // message needs and re-running a 200-iteration stall to find out is not a diagnosis.
        last_channels = residual_channels(&p_tot, &new_tot, basis);
        last_fractional = occ_alpha
            .occupations
            .iter()
            .flatten()
            .filter(|&&f| f > 1.0e-9 && f < 1.0 - 1.0e-9)
            .count();
        let e_elec = electronic_energy(&p_tot, core_bloch, &f_alpha);
        last_energy_step = (e_elec - previous_energy).abs();
        previous_energy = e_elec;
        if trace {
            eprintln!(
                "pm7-rs kscf {iteration:4}  dP {density_error:.6e}  dE {last_energy_step:.3e}  \
                 charge {:.3e}  onsite {:.3e}  intercell {:.3e}  \
                 E_F {fermi_ev:10.6}  fractional {last_fractional}",
                last_channels.charge, last_channels.onsite, last_channels.intercell
            );
        }

        if density_error < options.p_tol {
            p_alpha = new_alpha;
            p_beta = new_beta;
            converged = true;
            break;
        }
        let next_alpha = mixer_alpha.next(&p_alpha, &new_alpha);
        let next_beta = if unrestricted {
            mixer_beta.next(&p_beta, &new_beta)
        } else {
            next_alpha.clone()
        };
        p_alpha = next_alpha;
        p_beta = next_beta;
    }

    if !converged && std::env::var_os("PM7_QUIET").is_none() {
        // "did not converge (error=2.739e-7)" says nothing a user can act on. The channel
        // decomposition does: it names which of the three failure modes this is, and each has a
        // different remedy. Printed here rather than folded into `Pm7Error::ScfNotConverged`,
        // whose shape is shared with the molecular path.
        let ResidualChannels {
            charge,
            onsite,
            intercell,
        } = last_channels;
        let culprit = if last_fractional > 0 {
            "fractional occupations: states are partly filled, so the fill is deciding an \
             occupation that the k mesh cannot resolve. Add `--smearing fermi 0.05` (or a width \
             suited to the band width) -- for a metal that is not a workaround but the definition"
        } else if charge > onsite.max(intercell) {
            "the Mulliken charge channel, which is long-wavelength charge transfer between atoms \
             -- the mode a denser k mesh and a smaller mixing weight help with"
        } else {
            "the bond-order channels, not the charge channel. Density mixing accelerates this \
             poorly; a Fermi smearing of 0.02-0.05 eV usually converges the same cell in a tenth \
             of the iterations and, across a gap, to the same energy"
        };
        eprintln!(
            "pm7-rs: the k-point SCF stopped at {density_error:.3e} after {iterations} \
             iterations, against p_tol {:.1e}. The last energy step was {last_energy_step:.3e} eV, \
             and the residual splits as charge {charge:.3e}, on-site {onsite:.3e}, inter-cell \
             {intercell:.3e}, with {last_fractional} fractionally occupied states. What is stuck \
             is {culprit}. Set PM7_QUIET to silence this, or PM7_SCF_TRACE=1 for the whole \
             iteration history.",
            options.p_tol
        );
    }

    let density = add_blocks(&p_alpha, &p_beta);
    let spin_density = unrestricted.then(|| sub_blocks(&p_alpha, &p_beta));

    // One more Fock build, at the density actually being reported.
    //
    // The loop above contracts the **output** density against a Fock built from the **input** one,
    // and those differ by the density step. `E[P] = ½ Tr[P(H + F[P])]` is stationary only on the
    // idempotent manifold, so evaluating it with a mismatched pair leaves an error that is
    // **first order** in the residual, not second — and it is invisible, because the number looks
    // like an energy either way. The molecular path has always spent this extra build
    // (`scf.rs`, `f_final`); the k-point path did not, so the two were not computing the same
    // quantity.
    //
    // At the default tolerance the difference is small, which is why it survived. It is not small
    // for a loose `p_tol` — the case molecular dynamics runs on purpose — and it is exactly the
    // inconsistency that shows up as a gap between the analytic gradient and a finite difference
    // of the energy.
    let electronic_ev = {
        let f_alpha =
            crate::fock::build_fock_spin_bloch(molecule, basis, params, core, &density, &p_alpha)?;
        let f_beta = if unrestricted {
            crate::fock::build_fock_spin_bloch(molecule, basis, params, core, &density, &p_beta)?
        } else {
            f_alpha.clone()
        };
        electronic_energy_spin(&density, &p_alpha, &p_beta, core_bloch, &f_alpha, &f_beta)
    };

    Ok(KScfState {
        density,
        spin_density,
        band_energies: energies_alpha,
        electronic_ev,
        fermi_ev,
        entropy_ev,
        iterations,
        converged,
        density_error,
        unrestricted,
        n_kpoints: kpoints.len(),
    })
}

/// Band energies along an arbitrary path in the Brillouin zone.
#[derive(Clone, Debug)]
pub struct BandStructure {
    /// The path, in fractional reciprocal coordinates, as given.
    pub kpoints: Vec<[f64; 3]>,
    /// `energies[k]` are the eigenvalues at `kpoints[k]`, ascending, in eV.
    pub energies: Vec<Vec<f64>>,
    /// Spin-β eigenvalues for an unrestricted run; `None` for a restricted one.
    pub energies_beta: Option<Vec<Vec<f64>>>,
    /// The Fermi level from the **sampling mesh**, in eV.
    pub fermi_ev: f64,
}

/// Diagonalize the converged Fock along a k path.
///
/// This is deliberately **not** an SCF. A band path runs along high-symmetry lines, which is the
/// wrong set of points to build a density from — it weights those lines as though they were the
/// whole zone. The density and the Fermi level come from the sampling mesh in `options`, and the
/// path only asks the converged Hamiltonian what its eigenvalues are elsewhere. That is what a
/// band structure is, and doing it the other way produces a plot that is subtly wrong everywhere.
pub fn band_structure(
    molecule: &crate::system::Molecule,
    params: &crate::params::Pm7Parameters,
    options: &crate::scf::Pm7Options,
    path: &[[f64; 3]],
) -> Result<BandStructure> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput("a band structure needs a periodic cell".to_string())
    })?;
    let scf = crate::scf::run_pm7(molecule, params, options)?;
    let density = scf.bloch_density.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput(
            "a band structure needs the translation-resolved density; run with a k mesh"
                .to_string(),
        )
    })?;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let pbc = options
        .pbc_for(molecule)
        .ok_or_else(|| Pm7Error::InvalidInput("a band structure needs PBC options".to_string()))?;
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        &basis,
        params,
        false,
        &pbc,
        options.active_field(),
    )?;

    // The same Fock the SCF converged to, rebuilt once from its own density.
    let build = |spin: &BlochBlocks| {
        crate::fock::build_fock_spin_bloch(molecule, &basis, params, &core, density, spin)
    };
    let (fock_alpha, fock_beta) = match scf.bloch_spin_density.as_ref() {
        Some(spin) => {
            // `spin` holds `P_α − P_β`, so the two channels are `(P ± Δ)/2`.
            let alpha = scale_blocks(&add_blocks(density, spin), 0.5);
            let beta = scale_blocks(&sub_blocks(density, spin), 0.5);
            (build(&alpha)?, Some(build(&beta)?))
        }
        None => (build(&scale_blocks(density, 0.5))?, None),
    };

    let b = cell.reciprocal_2pi();
    let mut energies = Vec::with_capacity(path.len());
    let mut energies_beta = fock_beta.as_ref().map(|_| Vec::with_capacity(path.len()));
    for frac in path {
        let cart = b[0] * frac[0] + b[1] * frac[1] + b[2] * frac[2];
        let k = crate::pbc::kpoints::KPoint {
            frac: *frac,
            cart,
            weight: 1.0,
            time_reversal_pair: false,
        };
        energies.push(fock_alpha.at_k(&k).hermitian_eigen()?.0);
        if let (Some(fb), Some(store)) = (fock_beta.as_ref(), energies_beta.as_mut()) {
            store.push(fb.at_k(&k).hermitian_eigen()?.0);
        }
    }

    Ok(BandStructure {
        kpoints: path.to_vec(),
        energies,
        energies_beta,
        // A Γ-only run has no Fermi level of its own; the HOMO is the honest stand-in there.
        fermi_ev: scf
            .fermi_ev
            .unwrap_or_else(|| scf.homo_ev.unwrap_or(f64::NAN)),
    })
}

#[cfg(test)]
mod scf_helpers_tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::KMesh;

    #[test]
    fn density_from_fock_conserves_electrons_and_reproduces_a_known_band_structure() {
        // A one-orbital-per-cell tight-binding chain: H(0) = ε, H(±1) = t. Its exact band is
        // ε + 2t cos(k a), so the eigenvalues from the Bloch machinery can be checked against a
        // closed form rather than against another run of the same code.
        let a = 5.0_f64;
        let cell = Cell::new(&[crate::math::Vec3::new(a, 0.0, 0.0)]).unwrap();
        let (eps, t) = (-2.0_f64, -0.5_f64);
        let ts = vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0]];
        let mut h = BlochBlocks::new(ts, 1).unwrap();
        h.at_mut([0, 0, 0])[(0, 0)] = eps;
        h.at_mut([1, 0, 0])[(0, 0)] = t;
        h.at_mut([-1, 0, 0])[(0, 0)] = t;

        let set = KMesh::grid(8, 1, 1).expand(&cell).unwrap();
        let (density, energies, occ) =
            density_from_fock_with(&h, &set, 1.0, 2.0, Smearing::None, None).unwrap();

        for (k, e) in set.points.iter().zip(&energies) {
            let ka = k.phase([1, 0, 0]);
            let exact = eps + 2.0 * t * ka.cos();
            assert!(
                (e[0] - exact).abs() < 1e-12,
                "band at k={:?}: {} vs exact {exact}",
                k.frac,
                e[0]
            );
        }
        assert!(
            (electron_count(&density) - 1.0).abs() < 1e-10,
            "electron count {} should be 1",
            electron_count(&density)
        );
        // Half filling of a single band puts the Fermi level at the band centre.
        assert!(occ.fermi_ev <= eps + 1e-9 && occ.fermi_ev >= eps + 2.0 * t - 1e-9);
    }

    #[test]
    fn block_arithmetic_round_trips() {
        let ts = vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0]];
        let mut a = BlochBlocks::new(ts.clone(), 2).unwrap();
        let mut b = BlochBlocks::new(ts, 2).unwrap();
        for i in 0..a.len() {
            for k in 0..4 {
                a.block_mut(i).as_mut_slice()[k] = (i * 4 + k) as f64;
                b.block_mut(i).as_mut_slice()[k] = (k as f64) * 0.5;
            }
        }
        let sum = add_blocks(&a, &b);
        let back = sub_blocks(&sum, &b);
        assert!(back.rms_diff(&a) < 1e-15);
        let mixed = mix_blocks(&a, &b, 1.0);
        assert!(mixed.rms_diff(&b) < 1e-15);
        let unmixed = mix_blocks(&a, &b, 0.0);
        assert!(unmixed.rms_diff(&a) < 1e-15);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::math::Vec3;
    use crate::pbc::KMesh;

    /// Build a residual history whose blocks carry the given values, for the Pulay tests.
    fn residuals(values: &[[f64; 4]]) -> Vec<BlochBlocks> {
        values
            .iter()
            .map(|v| {
                let mut b = BlochBlocks::new(vec![[0, 0, 0]], 2).unwrap();
                let block = b.at_mut([0, 0, 0]);
                block[(0, 0)] = v[0];
                block[(0, 1)] = v[1];
                block[(1, 0)] = v[2];
                block[(1, 1)] = v[3];
                b
            })
            .collect()
    }

    /// The Pulay coefficients minimize `|Σ c_i r_i|²` subject to `Σ c_i = 1`, so scaling every
    /// residual by a common factor cannot change them: the objective is homogeneous and the
    /// constraint does not involve the residuals at all.
    ///
    /// Through v0.2.2 it *did* change them, because the Gram matrix was built from raw
    /// `<r_i, r_j>` and handed to a solver whose pivot guard is an absolute test. By the time an
    /// SCF reaches a residual of `1e-12` those entries are around `1e-24`, so a perfectly
    /// well-conditioned history was rejected for being small. The mixer then fell back to a plain
    /// damped step -- and, because the history kept growing, went on being rejected -- which is a
    /// residual that creeps toward the tolerance at a fixed rate and never arrives.
    ///
    /// A scale sweep is the sharpest statement of the fix: nothing about the answer may depend on
    /// how far the SCF has already converged.
    #[test]
    fn the_pulay_coefficients_do_not_depend_on_how_small_the_residuals_are() {
        let base = [
            [1.0, 0.5, 0.5, -0.25],
            [0.6, -0.3, -0.3, 0.4],
            [0.2, 0.15, 0.15, -0.05],
        ];
        let reference =
            pulay_coefficients(&residuals(&base)).expect("a well-conditioned history solves");
        assert!((reference.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);

        for scale in [1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12] {
            let scaled: Vec<[f64; 4]> = base
                .iter()
                .map(|r| [r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale])
                .collect();
            let c = pulay_coefficients(&residuals(&scaled))
                .unwrap_or_else(|| panic!("history rejected at scale {scale:e}"));
            for (a, b) in reference.iter().zip(&c) {
                assert!(
                    (a - b).abs() < 1.0e-9,
                    "coefficients moved when the residuals were scaled by {scale:e}: \
                     {reference:?} against {c:?}"
                );
            }
        }
    }

    /// A genuinely dependent history is still refused, so the scaling did not simply disable the
    /// guard: two identical residuals leave the bordered system singular at any scale.
    #[test]
    fn a_linearly_dependent_pulay_history_is_still_refused() {
        let repeated = [[1.0, 0.5, 0.5, -0.25], [1.0, 0.5, 0.5, -0.25]];
        assert!(pulay_coefficients(&residuals(&repeated)).is_none());
    }

    #[test]
    fn bloch_blocks_reject_a_set_that_is_not_closed_under_negation() {
        assert!(BlochBlocks::new(vec![[0, 0, 0], [1, 0, 0]], 2).is_err());
        assert!(BlochBlocks::new(vec![[1, 0, 0], [-1, 0, 0]], 2).is_err());
        assert!(BlochBlocks::new(vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0]], 2).is_ok());
    }

    #[test]
    fn the_gamma_point_of_the_bloch_sum_is_the_plain_block_sum() {
        let cell = Cell::cubic(8.0).unwrap();
        let mut b = BlochBlocks::new(vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0]], 2).unwrap();
        b.at_mut([0, 0, 0])[(0, 0)] = 1.0;
        b.at_mut([1, 0, 0])[(0, 1)] = 0.5;
        b.at_mut([-1, 0, 0])[(1, 0)] = 0.5;
        let set = KMesh::Gamma.expand(&cell).unwrap();
        let hk = b.at_k(&set.points[0]);
        let gamma = b.gamma_sum();
        for i in 0..2 {
            for j in 0..2 {
                let (re, im) = hk.get(i, j);
                assert!((re - gamma[(i, j)]).abs() < 1e-14);
                assert!(im.abs() < 1e-14, "Γ must be real");
            }
        }
    }

    #[test]
    fn bloch_matrices_are_hermitian_at_every_k() {
        let cell = Cell::new(&[
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.0, 5.5, 0.0),
            Vec3::new(0.0, 0.5, 7.0),
        ])
        .unwrap();
        let ts = vec![
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [1, 1, 0],
            [-1, -1, 0],
        ];
        let n = 4;
        let mut b = BlochBlocks::new(ts.clone(), n).unwrap();
        // Fill with H(-T) = H(T)^T, as the builders do.
        for (idx, t) in ts.iter().enumerate() {
            let neg = [-t[0], -t[1], -t[2]];
            for i in 0..n {
                for j in 0..n {
                    let v = ((idx * 3 + i * 5 + j * 7) % 13) as f64 - 6.0;
                    b.at_mut(*t)[(i, j)] += v;
                    b.at_mut(neg)[(j, i)] += v;
                }
            }
        }
        for mesh in [KMesh::grid(3, 3, 3), KMesh::grid(4, 2, 1)] {
            let set = mesh.expand(&cell).unwrap();
            for k in &set.points {
                let hk = b.at_k(k);
                assert!(
                    hk.hermiticity_error() < 1e-12,
                    "H(k) not Hermitian at {:?}: {:.3e}",
                    k.frac,
                    hk.hermiticity_error()
                );
            }
        }
    }

    #[test]
    fn aufbau_filling_conserves_the_electron_count() {
        let energies = vec![
            vec![-5.0, -3.0, 1.0, 4.0],
            vec![-4.5, -2.0, 0.5, 3.0],
            vec![-4.0, -2.5, 1.5, 2.5],
        ];
        let weights = vec![0.25, 0.5, 0.25];
        for n_elec in [2.0_f64, 4.0, 5.0, 8.0] {
            let occ = fill_bands(&energies, &weights, n_elec, 2.0, Smearing::None).unwrap();
            let total: f64 = occ
                .occupations
                .iter()
                .zip(&weights)
                .map(|(o, w)| 2.0 * w * o.iter().sum::<f64>())
                .sum();
            assert!(
                (total - n_elec).abs() < 1e-9,
                "aufbau filled {total} electrons, wanted {n_elec}"
            );
            assert!(occ
                .occupations
                .iter()
                .flatten()
                .all(|f| (0.0..=1.0).contains(f)));
            assert_eq!(occ.entropy_ev, 0.0);
        }
        // Too many electrons for the available bands must be an error, not a silent overfill.
        assert!(fill_bands(&energies, &weights, 100.0, 2.0, Smearing::None).is_err());
    }

    #[test]
    fn smeared_filling_conserves_the_electron_count_and_lowers_the_free_energy() {
        let energies = vec![
            vec![-5.0, -3.0, -0.2, 0.2, 4.0],
            vec![-4.5, -2.0, -0.1, 0.4, 3.0],
        ];
        let weights = vec![0.5, 0.5];
        for smearing in [
            Smearing::FermiDirac { width_ev: 0.2 },
            Smearing::Gaussian { width_ev: 0.2 },
        ] {
            let occ = fill_bands(&energies, &weights, 6.0, 2.0, smearing).unwrap();
            let total: f64 = occ
                .occupations
                .iter()
                .zip(&weights)
                .map(|(o, w)| 2.0 * w * o.iter().sum::<f64>())
                .sum();
            assert!(
                (total - 6.0).abs() < 1e-8,
                "{smearing:?}: filled {total} electrons, wanted 6"
            );
            // With states straddling E_F the entropy term must be non-zero and negative.
            assert!(
                occ.entropy_ev < 0.0,
                "{smearing:?}: entropy {} should lower the free energy",
                occ.entropy_ev
            );
        }
    }

    #[test]
    fn occupations_are_monotone_and_have_the_right_limits() {
        for smearing in [
            Smearing::FermiDirac { width_ev: 0.3 },
            Smearing::Gaussian { width_ev: 0.3 },
        ] {
            let mut previous = 1.1_f64;
            let mut e = -5.0_f64;
            while e < 5.0 {
                let f = occupation(e, 0.0, smearing);
                assert!((0.0..=1.0).contains(&f), "{smearing:?}: f({e}) = {f}");
                assert!(f <= previous + 1e-12, "{smearing:?}: not monotone at {e}");
                previous = f;
                e += 0.05;
            }
            assert!(occupation(-50.0, 0.0, smearing) > 1.0 - 1e-12);
            assert!(occupation(50.0, 0.0, smearing) < 1e-12);
            assert!((occupation(0.0, 0.0, smearing) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn methfessel_paxton_reduces_to_gaussian_at_order_zero_and_sharpens_above_it() {
        // Order 0 must be exactly the Gaussian form, and higher orders must approach a step
        // function: the integrated occupancy error against a hard step shrinks with the order.
        for x in [-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            let g = occupation(x, 0.0, Smearing::Gaussian { width_ev: 1.0 });
            let mp0 = occupation(
                x,
                0.0,
                Smearing::MethfesselPaxton {
                    width_ev: 1.0,
                    order: 0,
                },
            );
            assert!(
                (g - mp0).abs() < 1e-15,
                "order 0 differs from Gaussian at {x}"
            );
        }
        // ∫ f(x) dx over a symmetric window approximates the step's ∫ = window/2 better as the
        // order rises; this is the property the expansion is built to have.
        let window = 6.0;
        let n = 4001;
        let integral = |order: usize| -> f64 {
            let mut s = 0.0;
            for i in 0..n {
                let x = -window + 2.0 * window * i as f64 / (n - 1) as f64;
                let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
                s += w * occupation(
                    x,
                    0.0,
                    Smearing::MethfesselPaxton {
                        width_ev: 1.0,
                        order,
                    },
                );
            }
            s * 2.0 * window / (n - 1) as f64
        };
        let exact = window; // ∫_{-6}^{0} 1 dx
        let e0 = (integral(0) - exact).abs();
        let e2 = (integral(2) - exact).abs();
        assert!(
            e2 <= e0 + 1e-12,
            "order 2 ({e2:.3e}) should not be worse than order 0 ({e0:.3e})"
        );
    }

    #[test]
    fn accumulating_over_a_k_mesh_gives_a_real_density_obeying_p_minus_t_equals_p_t_transposed() {
        // Sanity for the k → real-space transform: whatever comes back must be a real matrix
        // with P(−T) = P(T)ᵀ, or the periodic energy expression is not even well defined.
        let cell = Cell::cubic(7.0).unwrap();
        let ts = vec![[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]];
        let n = 3;
        let mut h = BlochBlocks::new(ts.clone(), n).unwrap();
        for (idx, t) in ts.iter().enumerate() {
            let neg = [-t[0], -t[1], -t[2]];
            for i in 0..n {
                for j in 0..n {
                    let v = ((idx * 5 + i * 3 + j) % 7) as f64 - 3.0;
                    h.at_mut(*t)[(i, j)] += v;
                    h.at_mut(neg)[(j, i)] += v;
                }
            }
        }
        let set = KMesh::grid(4, 4, 1).expand(&cell).unwrap();
        let mut p = BlochBlocks::new(ts.clone(), n).unwrap();
        let mut energies = Vec::new();
        let mut vectors = Vec::new();
        for k in &set.points {
            let (e, c) = h.at_k(k).hermitian_eigen().unwrap();
            energies.push(e);
            vectors.push(c);
        }
        let weights = kpoint_weights(&set);
        let occ = fill_bands(&energies, &weights, 2.0, 2.0, Smearing::None).unwrap();
        for (ik, k) in set.points.iter().enumerate() {
            let pk = vectors[ik].weighted_density(&occ.occupations[ik], 2.0);
            p.accumulate_from_k(&pk, k);
        }
        let residual = p.symmetrize();
        assert!(
            residual < 1e-10,
            "P(-T) != P(T)^T by {residual:.3e}; the k-space transform is inconsistent"
        );
        // The T = 0 trace is the electron count per cell.
        let trace: f64 = (0..n).map(|i| p.at_mut([0, 0, 0])[(i, i)]).sum();
        assert!(
            (trace - 2.0).abs() < 1e-9,
            "P(0) trace {trace} should be the 2 electrons per cell"
        );
    }
}
