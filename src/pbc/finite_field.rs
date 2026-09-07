// SPDX-License-Identifier: GPL-3.0-or-later

//! A finite electric field **along** a periodic direction, by the Berry-phase electric enthalpy.
//!
//! # Why `𝓔·R` cannot be used here
//!
//! Under a lattice, `𝓔·R` shifts by `𝓔·T` on translation by `T`, so it is lattice-periodic exactly
//! when `𝓔·T = 0` for every lattice vector. A field orthogonal to all of them — normal to a slab,
//! transverse to a chain — is an ordinary calculation and goes through [`Pm7Options::field`],
//! which already checks that condition (`src/field.rs`). Along a periodic direction the potential
//! is unbounded, the spectrum has no lower bound, and no care in the assembly repairs it: the
//! ground state of `H − 𝓔·R` on a lattice does not exist.
//!
//! # What replaces it
//!
//! The **electric enthalpy** of Nunes and Gonze, minimized in place of the energy:
//!
//! ```text
//! F[ψ, 𝓔] = E[ψ] − Ω 𝓔·P[ψ]
//! ```
//!
//! with `P` the Berry-phase polarization of [`crate::pbc::berry`] rather than `⟨r⟩`. Because `P`
//! is built from overlaps between **neighbouring** k points, its derivative couples them: the
//! field term at `k` reads the coefficients at `k ± b`. The k points can no longer be solved one
//! at a time, which is the structural reason this is not a small change to the SCF — and why
//! `scf_pbc::run_kpoint_scf_with_terms` exists to take a `k`-dependent operator that is not a Bloch
//! sum of anything.
//!
//! # The coupling, derived here rather than quoted
//!
//! Sign and factor conventions for a Berry phase differ between sources, and a wrong factor does
//! not fail — it returns a plausible polarizability. So this is derived from *this crate's own*
//! polarization convention, fixed and documented in [`crate::pbc::berry`]:
//!
//! ```text
//! P_el = (f/Ω) Σ_α a_α φ_α,   φ_α = (1/2π)(1/N⊥) Σ_{k⊥} Im ln Π_j det S_j
//! S_j  = C_j† Δ C_{j+1},      Δ = diag(e^{−i b·τ_μ}),   b = G_α/J,   C_J ≡ C_0
//! ```
//!
//! `f` is the occupancy (2, restricted). `C_J ≡ C_0` with no extra phase is the cell gauge; see
//! [`crate::pbc::berry`] for why the closure factor a textbook derivation calls for is already
//! accounted for by the `J` steps.
//!
//! Differentiate, treating `C` and `C*` as independent. From `S_j` alone,
//! `∂ ln det S_j/∂C*(k_j) = Δ C_{j+1} S_j⁻¹`; the conjugate half picks up `S_{j−1}`, giving
//!
//! ```text
//! ∂(Im ln Z)/∂C*(k_j) = (1/2i) [ Δ C_{j+1} S_j⁻¹ − Δ† C_{j−1} (S_{j−1}⁻¹)† ]  ≡ (1/2i)(W₊ − W₋)
//! ```
//!
//! The energy's own gradient is `w_k f H C(k_j)` with `w_k = 1/(J N⊥)`. Dividing the enthalpy's
//! gradient through by that same `w_k f` turns it into an operator — the `N⊥` cancels against the
//! transverse average already in `φ`, and `1/i = −i` flips the sign:
//!
//! ```text
//! ΔH C(k_j) = i λ_α (W₊ − W₋),     λ_α = (𝓔·a_α) J / 4π
//! ```
//!
//! `ΔH` is recovered by projecting onto the occupied manifold, `M = i λ (W₊ − W₋) C_j†`, and made
//! Hermitian as **`M + M†`** — not `½(M + M†)`.
//!
//! The half would be right for a general matrix and is wrong here, because `M` is one-sided:
//! `M|v⟩ = G C†|v⟩ = 0` for any virtual `v`, since `C†` annihilates everything outside the
//! occupied span. So `M` carries the whole virtual–occupied block and none of the
//! occupied–virtual one, `M†` is its mirror, and adding them fills two disjoint blocks once each.
//! Averaging instead halves the occupied–virtual block — which is the block the linear response is
//! made of, so the polarizability comes out at exactly half.
//!
//! The occupied–occupied block is doubled by this and it does not matter: it mixes occupied
//! orbitals among themselves, leaving the density and hence the polarization untouched.
//!
//! # What says the factor is right
//!
//! Not the derivation. `tests/pbc_finite_field.rs` takes `α = Ω ∂P/∂𝓔` by finite differences of
//! this and compares it against the **CPHF** polarizability from [`crate::polarizability`] — two
//! formalisms sharing only the SCF. A factor of two, a missing `J`, or a sign shows up there and
//! nowhere else.
//!
//! This is a semiempirical model, so neither number is a prediction of experiment; what is checked
//! is that the crate computes its own model's polarizability consistently by two routes.

// The loop variables are Cartesian directions, k-point indices, or matrix rows and columns; the
// index is the meaning.
#![allow(clippy::needless_range_loop)]

use crate::basis::Basis;
use crate::cmatrix::CMatrix;
use crate::error::{Pm7Error, Result};
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::pbc::kpoints::{KPoint, KPointSet};
use crate::scf::Pm7Options;
use crate::system::Molecule;

/// A converged finite-field state and the polarization it carries.
#[derive(Clone, Debug)]
pub struct FiniteFieldResult {
    /// Total energy of the converged state, in eV. **Not** the quantity that was minimized; see
    /// `enthalpy_ev`.
    pub energy_ev: f64,
    /// `E − Ω 𝓔·P`, the electric enthalpy actually minimized, in eV.
    pub enthalpy_ev: f64,
    /// The applied field, in the same units [`Pm7Options::field`] uses.
    pub field: Vec3,
    /// The Berry phase along each lattice direction, in units of `2π`.
    pub phase: [f64; 3],
    /// Electronic polarization, `e/Bohr²`.
    pub electronic_polarization: Vec3,
    /// Ionic polarization, `e/Bohr²`.
    pub ionic_polarization: Vec3,
    /// Their sum, defined modulo the quantum of [`crate::pbc::berry`].
    pub polarization: Vec3,
    /// Outer-loop iterations taken.
    pub iterations: usize,
    pub converged: bool,
    /// Which axes had enough k points for a phase. An axis with fewer than three is reported
    /// unresolved rather than as zero, which is a different claim.
    pub resolved: [bool; 3],
}

/// Controls for the outer field loop.
#[derive(Clone, Copy, Debug)]
pub struct FiniteFieldOptions {
    /// Convergence threshold on the **field operator** itself, which is what the outer loop
    /// solves for.
    pub tol: f64,
    pub max_iterations: usize,
    /// Linear mixing of the field operator between outer iterations.
    ///
    /// The enthalpy is not a minimum of the energy, so the plain fixed point oscillates for a
    /// field of any size; damping is what makes it a contraction. Below the point where it
    /// converges at all this only changes how long it takes.
    pub mixing: f64,
}

impl Default for FiniteFieldOptions {
    fn default() -> Self {
        Self {
            tol: 1.0e-9,
            max_iterations: 200,
            mixing: 0.5,
        }
    }
}

/// Inverse of a small complex matrix by Gauss–Jordan with partial pivoting, `[re, im]` pairs.
///
/// Only ever applied to an occupied-by-occupied overlap block.
fn invert(a: &[Vec<[f64; 2]>]) -> Result<Vec<Vec<[f64; 2]>>> {
    let n = a.len();
    let mul = |x: [f64; 2], y: [f64; 2]| [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
    let div = |x: [f64; 2], y: [f64; 2]| {
        let d = y[0] * y[0] + y[1] * y[1];
        [
            (x[0] * y[0] + x[1] * y[1]) / d,
            (x[1] * y[0] - x[0] * y[1]) / d,
        ]
    };
    let mut m: Vec<Vec<[f64; 2]>> = a.to_vec();
    let mut inv: Vec<Vec<[f64; 2]>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { [1.0, 0.0] } else { [0.0, 0.0] })
                .collect()
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        let mut best = (m[col][col][0].powi(2) + m[col][col][1].powi(2)).sqrt();
        for row in (col + 1)..n {
            let size = (m[row][col][0].powi(2) + m[row][col][1].powi(2)).sqrt();
            if size > best {
                best = size;
                pivot = row;
            }
        }
        if best == 0.0 {
            return Err(Pm7Error::InvalidInput(
                "the overlap between adjacent points on a Berry string is singular, so the \
                 occupied manifold cannot be followed from one to the next; use more k points \
                 along that direction"
                    .into(),
            ));
        }
        m.swap(pivot, col);
        inv.swap(pivot, col);
        let diagonal = m[col][col];
        for j in 0..n {
            m[col][j] = div(m[col][j], diagonal);
            inv[col][j] = div(inv[col][j], diagonal);
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            if factor == [0.0, 0.0] {
                continue;
            }
            for j in 0..n {
                let a1 = mul(m[col][j], factor);
                m[row][j][0] -= a1[0];
                m[row][j][1] -= a1[1];
                let a2 = mul(inv[col][j], factor);
                inv[row][j][0] -= a2[0];
                inv[row][j][1] -= a2[1];
            }
        }
    }
    Ok(inv)
}

/// `S_mn = Σ_μ c*_{μm}(left) e^{−i b·τ_μ} c_{μn}(right)`, over the lowest `n_occ` bands.
fn overlap(
    left: &CMatrix,
    right: &CMatrix,
    phase: &[[f64; 2]],
    n_occ: usize,
) -> Vec<Vec<[f64; 2]>> {
    let nao = phase.len();
    let mut s = vec![vec![[0.0_f64; 2]; n_occ]; n_occ];
    for m in 0..n_occ {
        for n in 0..n_occ {
            let mut acc = [0.0_f64; 2];
            for mu in 0..nao {
                let (lr, li) = left.get(mu, m);
                let (rr, ri) = right.get(mu, n);
                let p = phase[mu];
                // conj(left) * p
                let a = [lr * p[0] + li * p[1], lr * p[1] - li * p[0]];
                acc[0] += a[0] * rr - a[1] * ri;
                acc[1] += a[0] * ri + a[1] * rr;
            }
            s[m][n] = acc;
        }
    }
    s
}

fn determinant(mut a: Vec<Vec<[f64; 2]>>) -> [f64; 2] {
    let n = a.len();
    let mul = |x: [f64; 2], y: [f64; 2]| [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
    let div = |x: [f64; 2], y: [f64; 2]| {
        let d = y[0] * y[0] + y[1] * y[1];
        [
            (x[0] * y[0] + x[1] * y[1]) / d,
            (x[1] * y[0] - x[0] * y[1]) / d,
        ]
    };
    let mut det = [1.0, 0.0];
    for col in 0..n {
        let mut pivot = col;
        let mut best = (a[col][col][0].powi(2) + a[col][col][1].powi(2)).sqrt();
        for row in (col + 1)..n {
            let size = (a[row][col][0].powi(2) + a[row][col][1].powi(2)).sqrt();
            if size > best {
                best = size;
                pivot = row;
            }
        }
        if best == 0.0 {
            return [0.0, 0.0];
        }
        if pivot != col {
            a.swap(pivot, col);
            det = [-det[0], -det[1]];
        }
        det = mul(det, a[col][col]);
        let diagonal = a[col][col];
        for row in (col + 1)..n {
            let factor = div(a[row][col], diagonal);
            if factor == [0.0, 0.0] {
                continue;
            }
            for k in col..n {
                let value = mul(a[col][k], factor);
                a[row][k][0] -= value[0];
                a[row][k][1] -= value[1];
            }
        }
    }
    det
}

/// The k-point index of grid position `i` in the unfolded mesh this module builds.
fn index_of(divisions: [usize; 3], i: [usize; 3]) -> usize {
    (i[0] * divisions[1] + i[1]) * divisions[2] + i[2]
}

/// The full, **unfolded** Monkhorst–Pack grid.
///
/// Unfolded deliberately: a string needs every point along its direction in order, and
/// time-reversal folding removes half of them.
fn grid(divisions: [usize; 3], cell: &crate::cell::Cell) -> KPointSet {
    let reciprocal = cell.reciprocal_2pi();
    let total = divisions[0] * divisions[1] * divisions[2];
    let weight = 1.0 / total as f64;
    let mut points = Vec::with_capacity(total);
    for i0 in 0..divisions[0] {
        for i1 in 0..divisions[1] {
            for i2 in 0..divisions[2] {
                let frac = [
                    i0 as f64 / divisions[0] as f64,
                    i1 as f64 / divisions[1] as f64,
                    i2 as f64 / divisions[2] as f64,
                ];
                let mut cart = Vec3::zero();
                for d in 0..3 {
                    cart += reciprocal[d] * frac[d];
                }
                points.push(KPoint {
                    frac,
                    cart,
                    weight,
                    time_reversal_pair: false,
                });
            }
        }
    }
    KPointSet {
        points,
        unfolded_count: total,
    }
}

/// Converge a cell in a finite field applied **along** a periodic direction.
///
/// `divisions` is the k mesh, and doubles as the string length: the phase along axis `α` is
/// discretized over `divisions[α]` points, so that is the convergence parameter for the
/// polarization as well as for the Brillouin-zone integral.
pub fn run_finite_field(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    divisions: [usize; 3],
    field: Vec3,
    ff: &FiniteFieldOptions,
) -> Result<FiniteFieldResult> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput("a finite-field calculation needs a periodic cell".into())
    })?;
    if cell.vectors().len() != 3 {
        return Err(Pm7Error::InvalidInput(
            "the Berry-phase finite field is implemented for a three-dimensional cell, matching \
             `pbc::berry`. A field orthogonal to every lattice vector needs none of this \
             machinery: set `Pm7Options::field`, which checks that condition."
                .into(),
        ));
    }
    if options.field.is_some() {
        return Err(Pm7Error::InvalidInput(
            "`Pm7Options::field` and the Berry-phase finite field are two treatments of the same \
             perturbation; pass the field to `run_finite_field` only. The E.R form is for a field \
             orthogonal to every lattice vector, this one for a field along a periodic one."
                .into(),
        ));
    }
    if divisions.contains(&0) {
        return Err(Pm7Error::InvalidInput(
            "a k-mesh division of zero is not a mesh".into(),
        ));
    }
    if options.multiplicity > 1 {
        return Err(Pm7Error::InvalidInput(
            "the Berry-phase finite field is restricted-only, matching `pbc::berry`: an \
             open-shell cell needs the phase of each spin manifold separately"
                .into(),
        ));
    }

    let reciprocal = cell.reciprocal_2pi();

    // Which axes the field couples to, and which the mesh can resolve a phase along. These are
    // **not** the same set: an axis the field does not touch still carries polarization, so its
    // phase has to be computed even though it contributes no `ΔH`. Computing only the coupled axes
    // would make a zero field report an electronic polarization of exactly zero, which is not its
    // value.
    //
    // The threshold is relative to `|𝓔||a|` rather than an exact zero: in a non-orthogonal cell a
    // field the caller placed perpendicular to a lattice vector lands at `1e-17` rather than `0`,
    // and an exact test would then demand three k points along an axis contributing nothing.
    let mut active: Vec<usize> = Vec::new();
    for axis in 0..3 {
        let a = cell.vectors()[axis];
        let scale = field.norm() * a.norm();
        if scale > 0.0 && field.dot(a).abs() > 1.0e-12 * scale {
            active.push(axis);
        }
    }
    for axis in &active {
        if divisions[*axis] < 3 {
            return Err(Pm7Error::InvalidInput(format!(
                "the field has a component along lattice vector {axis}, whose string has {} k \
                 points. A discretized Berry phase is a product of nearest-neighbour overlaps and \
                 two points cannot resolve a winding; use at least 3.",
                divisions[*axis]
            )));
        }
    }
    let resolved = [divisions[0] >= 3, divisions[1] >= 3, divisions[2] >= 3];

    let basis = Basis::build(molecule, params)?;
    let nao = basis.nao;
    let kpoints = grid(divisions, &cell);
    let pbc = options
        .pbc_for(molecule)
        .ok_or_else(|| Pm7Error::InvalidInput("a finite field needs periodic options".into()))?;
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        &basis,
        params,
        options.force_dpath,
        &pbc,
        None,
    )?;

    // Orbital positions: the NDDO placement the dipole operator and the Berry phase both use.
    let mut tau = vec![Vec3::zero(); nao];
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let start = basis.atom_offset[a];
        for slot in tau[start..start + basis.atom_norb[a]].iter_mut() {
            *slot = atom.position;
        }
    }

    let n_elec: f64 = molecule
        .atoms
        .iter()
        .map(|atom| params.element(atom.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum::<f64>()
        - options.charge;
    let n_occ = (n_elec / 2.0).round() as usize;
    if n_occ == 0 || n_occ >= nao {
        return Err(Pm7Error::InvalidInput(format!(
            "the occupied manifold has {n_occ} bands against {nao} orbitals; a Berry phase needs a \
             filled manifold to follow"
        )));
    }
    let occupancy = 2.0;

    let mut k_terms: Vec<CMatrix> = (0..kpoints.points.len())
        .map(|_| CMatrix::zeros(nao))
        .collect();
    let mut phase = [0.0_f64; 3];
    let mut converged = false;
    let mut iterations = 0usize;
    // Kept outside the loop so a failure reports how far off it was, which is the difference
    // between "needs more iterations" and "is not converging".
    let mut largest_change = f64::INFINITY;
    let mut energy_ev = 0.0_f64;

    for iteration in 0..ff.max_iterations {
        iterations = iteration + 1;
        let state = crate::scf_pbc::run_kpoint_scf_with_terms(
            molecule,
            &basis,
            params,
            &core,
            &kpoints,
            n_occ,
            n_occ,
            options,
            Some(&k_terms),
        )?;
        // Electronic plus core-core, the same two pieces `run_pm7` adds for a periodic cell.
        energy_ev = state.electronic_ev
            + crate::repulsion::core_core_energy_periodic(molecule, params, &pbc)?;

        // The coefficients from the same Hamiltonian the SCF diagonalized: its own Fock rebuilt at
        // the converged density, plus the field term that was held fixed through it.
        let half = crate::scf_pbc::scale_blocks(&state.density, 0.5);
        let fock = crate::fock::build_fock_spin_bloch(
            molecule,
            &basis,
            params,
            &core,
            &state.density,
            &half,
        )?;
        let mut coefficients: Vec<CMatrix> = Vec::with_capacity(kpoints.points.len());
        for (index, k) in kpoints.points.iter().enumerate() {
            let mut hk = fock.at_k(k);
            for i in 0..nao {
                for j in 0..nao {
                    let (ar, ai) = hk.get(i, j);
                    let (br, bi) = k_terms[index].get(i, j);
                    hk.set(i, j, ar + br, ai + bi);
                }
            }
            let (_, vectors) = hk.hermitian_eigen()?;
            coefficients.push(vectors);
        }

        let mut next: Vec<CMatrix> = (0..kpoints.points.len())
            .map(|_| CMatrix::zeros(nao))
            .collect();
        let mut new_phase = [0.0_f64; 3];

        // Every axis the mesh resolves, not only the ones the field couples to. `lambda` is zero
        // on the axes the field misses, so those contribute no operator.
        for axis in (0..3).filter(|axis| resolved[*axis]) {
            let j_count = divisions[axis];
            let step = reciprocal[axis] / j_count as f64;
            let delta: Vec<[f64; 2]> = tau
                .iter()
                .map(|position| {
                    let angle = -step.dot(*position);
                    [angle.cos(), angle.sin()]
                })
                .collect();
            let lambda =
                field.dot(cell.vectors()[axis]) * j_count as f64 / (4.0 * std::f64::consts::PI);

            let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut axis_phase = 0.0_f64;
            let mut strings = 0usize;

            for a1 in 0..divisions[t1] {
                for a2 in 0..divisions[t2] {
                    let mut line = Vec::with_capacity(j_count);
                    for j in 0..j_count {
                        let mut i = [0usize; 3];
                        i[axis] = j;
                        i[t1] = a1;
                        i[t2] = a2;
                        line.push(index_of(divisions, i));
                    }

                    let mut s_inv = Vec::with_capacity(j_count);
                    let mut running = [1.0_f64, 0.0];
                    for j in 0..j_count {
                        let right = line[(j + 1) % j_count];
                        let s =
                            overlap(&coefficients[line[j]], &coefficients[right], &delta, n_occ);
                        let d = determinant(s.clone());
                        let size = (d[0] * d[0] + d[1] * d[1]).sqrt();
                        if size == 0.0 {
                            return Err(Pm7Error::InvalidInput(format!(
                                "the overlap along axis {axis} is singular; the string is too \
                                 coarse to follow the occupied manifold"
                            )));
                        }
                        let product = [
                            running[0] * d[0] - running[1] * d[1],
                            running[0] * d[1] + running[1] * d[0],
                        ];
                        let n = (product[0] * product[0] + product[1] * product[1]).sqrt();
                        running = [product[0] / n, product[1] / n];
                        s_inv.push(invert(&s)?);
                    }
                    axis_phase += running[1].atan2(running[0]) / std::f64::consts::TAU;
                    strings += 1;

                    for j in 0..j_count {
                        let here = line[j];
                        let ahead = line[(j + 1) % j_count];
                        let behind = line[(j + j_count - 1) % j_count];
                        let s_here = &s_inv[j];
                        let s_back = &s_inv[(j + j_count - 1) % j_count];

                        // W₊ = Δ C_{j+1} S_j⁻¹ ; W₋ = Δ† C_{j−1} (S_{j−1}⁻¹)†
                        let mut w = vec![vec![[0.0_f64; 2]; n_occ]; nao];
                        for mu in 0..nao {
                            for m in 0..n_occ {
                                let mut plus = [0.0_f64; 2];
                                let mut minus = [0.0_f64; 2];
                                for n in 0..n_occ {
                                    let (cr, ci) = coefficients[ahead].get(mu, n);
                                    let s = s_here[n][m];
                                    plus[0] += cr * s[0] - ci * s[1];
                                    plus[1] += cr * s[1] + ci * s[0];
                                    let (br, bi) = coefficients[behind].get(mu, n);
                                    // `(S⁻¹)†` is the conjugate transpose: index [m][n]
                                    // conjugated rather than [n][m].
                                    let t = s_back[m][n];
                                    minus[0] += br * t[0] + bi * t[1];
                                    minus[1] += bi * t[0] - br * t[1];
                                }
                                let d = delta[mu];
                                let dp = [
                                    d[0] * plus[0] - d[1] * plus[1],
                                    d[0] * plus[1] + d[1] * plus[0],
                                ];
                                let dm = [
                                    d[0] * minus[0] + d[1] * minus[1],
                                    d[0] * minus[1] - d[1] * minus[0],
                                ];
                                w[mu][m] = [dp[0] - dm[0], dp[1] - dm[1]];
                            }
                        }

                        // `M = i λ W C_j†`, then `ΔH += M + M†` — **not** `½(M + M†)`; see the
                        // module note for why the conventional half halves the polarizability.
                        for mu in 0..nao {
                            for nu in 0..nao {
                                let mut value = [0.0_f64; 2];
                                for m in 0..n_occ {
                                    let (cr, ci) = coefficients[here].get(nu, m);
                                    let x = w[mu][m];
                                    // x * conj(c)
                                    value[0] += x[0] * cr + x[1] * ci;
                                    value[1] += x[1] * cr - x[0] * ci;
                                }
                                // multiply by `i λ`
                                let entry = [-lambda * value[1], lambda * value[0]];
                                let (ar, ai) = next[here].get(mu, nu);
                                next[here].set(mu, nu, ar + entry[0], ai + entry[1]);
                                let (br, bi) = next[here].get(nu, mu);
                                next[here].set(nu, mu, br + entry[0], bi - entry[1]);
                            }
                        }
                    }
                }
            }
            new_phase[axis] = axis_phase / strings as f64;
        }

        // Convergence on the operator itself, which is what the outer loop is solving for.
        let mut largest = 0.0_f64;
        for (fresh, old) in next.iter().zip(&k_terms) {
            for row in 0..nao {
                for col in 0..nao {
                    let (ar, ai) = fresh.get(row, col);
                    let (br, bi) = old.get(row, col);
                    largest = largest.max(((ar - br).powi(2) + (ai - bi).powi(2)).sqrt());
                }
            }
        }
        phase = new_phase;
        largest_change = largest;
        if largest < ff.tol {
            converged = true;
            break;
        }
        for (target, fresh) in k_terms.iter_mut().zip(&next) {
            for row in 0..nao {
                for col in 0..nao {
                    let (ar, ai) = target.get(row, col);
                    let (br, bi) = fresh.get(row, col);
                    target.set(
                        row,
                        col,
                        ar * (1.0 - ff.mixing) + br * ff.mixing,
                        ai * (1.0 - ff.mixing) + bi * ff.mixing,
                    );
                }
            }
        }
    }

    if !converged {
        return Err(Pm7Error::ScfNotConverged {
            iterations,
            error: largest_change,
        });
    }

    let volume = cell.measure();
    let mut electronic = Vec3::zero();
    for axis in 0..3 {
        // The same sign `pbc::berry` derives, with the occupancy carried explicitly.
        electronic += cell.vectors()[axis] * (occupancy * phase[axis] / volume);
    }
    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        ionic += atom.position * (params.element(atom.z)?.core_charge / volume);
    }
    let polarization = electronic + ionic;

    Ok(FiniteFieldResult {
        energy_ev,
        enthalpy_ev: energy_ev - volume * field.dot(polarization),
        field,
        phase,
        electronic_polarization: electronic,
        ionic_polarization: ionic,
        polarization,
        iterations,
        converged,
        resolved,
    })
}
