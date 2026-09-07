// SPDX-License-Identifier: GPL-3.0-or-later
//! Is the converged SCF solution a **minimum**, or only a stationary point?
//!
//! An SCF iteration finds a solution of `[F, P] = 0`. That is a stationary condition, not a
//! minimum one, and nothing in the iteration distinguishes the two: a saddle point converges as
//! cleanly as a minimum, to as tight a residual, and reports the same `converged: true`.
//!
//! The 176-case oracle made the cost of that concrete. `pm7-rs` converges CuCl to
//! **13.39233 kcal/mol** and MOPAC to **8.21456**, 5.18 lower, with Mulliken charges differing by
//! 0.019 e — which is what a different SCF solution looks like. AgCl is the same by 3.82.
//!
//! **The analysis was built to test that reading, and refuted it.** CuCl's lowest orbital-Hessian
//! eigenvalue is `+6.05 eV`: the solution is a genuine local minimum, not a saddle, so there is
//! nothing here to follow. Thirty-six combinations of level shift, accelerator and spin reference
//! reach the same number (`examples/cucl_basin.rs`) and so do forty-eight perturbed starting
//! densities (`examples/cucl_search.rs`). A basin no perturbation and no negative curvature can
//! reach is not a basin, and `docs/fidelity.md` reclassifies those two cases accordingly.
//!
//! So this module exists having answered one question with a "no". That is worth keeping: the
//! alternative was to ship a guess about CuCl, and an SCF that cannot tell a minimum from a saddle
//! cannot tell anyone else either.
//!
//! # It does find a real one, and the real one was hiding in the UHF guess
//!
//! Two channels have to be asked, not one. The **singlet** channel asks whether the solution is a
//! minimum among closed-shell solutions; the **triplet** channel asks whether it is still one when
//! the α and β orbitals are allowed to differ. Stretched H₂ is the case every textbook uses, and it
//! is stable in the first and unstable in the second.
//!
//! | H₂ | singlet | triplet | ΔHf |
//! |---|---|---|---|
//! | 0.74 Å | +13.64 | **+8.23** | −31.748, unchanged |
//! | 1.50 Å | +10.36 | **−2.04** | 85.93 → 78.17 |
//! | 2.50 Å | +9.63 | **−7.99** | 187.08 → 103.48 |
//! | 4.00 Å | +10.70 | **−10.60** | 225.79 → **104.19** |
//!
//! That last number is the check. Two hydrogen atoms are `2 × 52.102 = 104.204` kcal/mol, so the
//! escaped solution dissociates correctly, while the restricted one sits 121.6 kcal/mol above it —
//! the classic RHF dissociation failure, found and removed.
//!
//! **And it could not have been removed without a change to the SCF's guess.** `uhf_loop` scales
//! the atomic density by `n_α/n` and `n_β/n`, which for a closed shell is the *same* number twice:
//! the spin density starts at exactly zero and the UHF equations keep it there forever. So
//! `reference = "uhf"` on stretched H₂ returned the restricted energy, and no amount of level
//! shifting or DIIS tuning changed that. [`crate::scf::Pm7Options::initial_spin_density`] is the
//! way in, and the triplet eigenvector is what to put there — the analysis does not merely report
//! the instability, it supplies the direction out of it.
//!
//! # The test
//!
//! At a stationary point the energy's second derivative with respect to occupied–virtual orbital
//! rotations is the **orbital Hessian**
//!
//! ```text
//! A U = (ε_a − ε_i) U + [G(ΔP(U))]_ov
//! ```
//!
//! which is exactly the operator the CPHF already applies — [`crate::hessian::cphf_ov_with_kernel`]
//! solves `A U = −G_skel` with it. So the stability analysis needs no new integrals and no new
//! kernel: it needs the **lowest eigenvalue** of an operator the crate can already multiply by.
//! Positive means a minimum; negative means a saddle, and its eigenvector is the direction out.
//!
//! # Which of Seeger and Pople's channels this is
//!
//! The framework is theirs — R. Seeger and J. A. Pople, *J. Chem. Phys.* **66** (1977) 3045,
//! "Self-consistent molecular orbital methods. XVIII. Constraints and stability in Hartree–Fock
//! theory". They classify the ways a converged solution can fail to be a minimum by which
//! constraint the lower solution breaks, and this module implements two of the three:
//!
//! | channel | constraint broken | operator | here |
//! |---|---|---|---|
//! | singlet, real | none — RHF→RHF | `A + B`, full `G` | [`restricted_stability`], `lowest_ev` |
//! | triplet, real | `α = β` — RHF→UHF | `A + B`, exchange only | [`restricted_stability`], `lowest_triplet_ev` |
//! | complex | real orbitals — RHF→CHF | `A − B` | **not implemented** |
//! | UHF internal | none — UHF→UHF | `A + B`, both spin blocks | [`unrestricted_stability`] |
//!
//! The operator above is `A + B`, which is the second derivative with respect to **real** orbital
//! rotations; `A − B` governs imaginary ones and is what would find a complex instability. That
//! channel is left out deliberately rather than overlooked: the rest of the crate is real
//! throughout — the molecular Fock matrix, the CPHF, the Hessian — so a complex solution is not
//! one any other part of pm7-rs could carry, and reporting an instability that cannot be followed
//! would be worse than not reporting it. `docs/scope.md` says so in the same words.
//!
//! Two departures from the 1977 paper, both mechanical. The lowest eigenpair comes from a
//! preconditioned Rayleigh-quotient iteration rather than a full diagonalization of the rotation
//! space, because the question is a sign. And **following** the instability — rotating along the
//! eigenvector and re-converging — is not in that paper at all; it is the standard practice that
//! grew up around it.
//!
//! # Following it
//!
//! Given an unstable direction `U`, rotating the occupied orbitals by `exp(κ)` with `κ` the
//! antisymmetric extension of `U` moves off the saddle. Rather than exponentiate, this mixes a step
//! of the virtual into the occupied and re-orthonormalizes, which is the same motion to first order
//! and cannot leave the manifold. The SCF then re-converges from that density; if it lands lower,
//! that is the answer, and if it does not, the original stands. **The lower of the two is always
//! what is returned**, so following an instability can never make an answer worse.

use crate::error::Result;
use crate::linalg::Matrix;
use crate::scf::{Pm7Options, Pm7Result};
use crate::system::Molecule;

/// What to do about an SCF solution that is not a minimum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScfStability {
    /// Do not look. The solution is whatever the iteration converged to.
    #[default]
    Off,
    /// Look and report through [`Pm7Result::stability`], but keep the solution.
    Check,
    /// Look, and if the solution is a saddle, follow the unstable direction and re-converge.
    Follow,
}

/// What the analysis found, in both channels.
///
/// **Both, because they are different questions.** The singlet channel asks whether the solution is
/// a minimum among *closed-shell* solutions; the triplet channel asks whether it is a minimum once
/// the α and β orbitals are allowed to differ. A closed-shell molecule can be perfectly stable in
/// the first and unstable in the second — stretched H₂ is the textbook case — and an analysis that
/// only ran the first would report it as a minimum.
#[derive(Clone, Copy, Debug)]
pub struct Stability {
    /// Lowest eigenvalue of the **singlet** (spin-preserving, RHF→RHF) orbital Hessian, in eV.
    pub lowest_ev: f64,
    /// Lowest eigenvalue of the **triplet** (spin-breaking, RHF→UHF) orbital Hessian, in eV.
    ///
    /// `None` for a solution that is already unrestricted, where the question does not arise.
    pub lowest_triplet_ev: Option<f64>,
    /// Whether either eigenvalue is negative beyond the tolerance.
    pub unstable: bool,
    /// Whether following the instability reached a lower solution, and by how much (eV).
    pub lowered_ev: Option<f64>,
}

/// Say that the spin reference changed underneath the caller. `PM7_QUIET` silences it.
///
/// A `follow` run that escapes through the triplet channel returns an **unrestricted** solution
/// where the caller asked for a restricted one, and the two are not interchangeable: the UHF
/// solution is lower, but it is a broken-symmetry state rather than a spin eigenfunction, and a
/// geometry scan that switches at one point and not the next has a discontinuous energy. The
/// switch is the right answer and it is also a change of model, so it is reported rather than
/// implied by `result.unrestricted`.
fn warn_switched_to_unrestricted(triplet_ev: f64, lowered_ev: f64) {
    if std::env::var_os("PM7_QUIET").is_some() {
        return;
    }
    eprintln!(
        "pm7-rs: the restricted solution is triplet-unstable ({triplet_ev:.4} eV in the \
         RHF->UHF channel), so `stability = follow` escaped along that direction and this result \
         is now UNRESTRICTED, {:.4} eV ({:.4} kcal/mol) below the restricted one. It is a \
         broken-symmetry solution, not a spin eigenfunction, so <S^2> exceeds S(S+1) and the \
         energy is not comparable with a restricted energy at a neighbouring geometry -- a scan \
         that switches partway through is discontinuous there. Use `stability = check` to be told \
         without being moved, or `off` to keep the restricted solution. Set PM7_QUIET to silence \
         this.",
        lowered_ev,
        lowered_ev * crate::constants::EV_TO_KCAL,
    );
}

/// Below this the solution is a saddle rather than a minimum, in eV.
///
/// A converged minimum's lowest orbital-Hessian eigenvalue is the HOMO–LUMO gap plus a
/// two-electron term, which is electron-volts for anything with a gap; a saddle's is negative by a
/// comparable amount. Nothing sits near this line, so it discriminates rather than decides — and it
/// is a *curvature*, not a magnitude test on an energy difference.
const INSTABILITY_TOLERANCE_EV: f64 = -1.0e-4;

/// Operator applications the eigenvalue search may spend.
const SEARCH_ITERATIONS: usize = 40;

/// The lowest eigenvalue of an orbital Hessian, and its eigenvector, on a flat vector.
///
/// Preconditioned steepest descent on the Rayleigh quotient, with the energy-denominator
/// preconditioner the CPHF already uses. Not Davidson: the question is a *sign*, the operator is
/// symmetric, and a dozen applications settle a sign long before they settle a value.
///
/// Flat rather than a `Matrix` because the unrestricted rotation space is two blocks of different
/// shapes — `n_vir^α × n_occ^α` and `n_vir^β × n_occ^β` — and one vector holds both without the
/// caller pretending they are one matrix.
fn lowest_eigenpair(
    apply: &dyn Fn(&[f64]) -> Result<Vec<f64>>,
    diagonal: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let n = diagonal.len();
    // A deterministic, non-symmetric start: a symmetric one can be orthogonal to the unstable
    // direction and stay there, which would report a saddle as a minimum.
    let mut x: Vec<f64> = (0..n).map(|k| ((k * 7 + 3) % 13) as f64 - 6.0).collect();
    normalize(&mut x);

    let mut eigenvalue = f64::INFINITY;
    let mut best = x.clone();
    for _ in 0..SEARCH_ITERATIONS {
        let ax = apply(&x)?;
        let rayleigh = dot(&x, &ax);
        if rayleigh < eigenvalue {
            eigenvalue = rayleigh;
            best.copy_from_slice(&x);
        }
        // Residual `r = A x − λ x`, preconditioned by the operator's diagonal.
        let mut residual = ax;
        for ((r, xv), d) in residual.iter_mut().zip(&x).zip(diagonal) {
            *r -= rayleigh * *xv;
            *r /= d.abs().max(1.0e-6);
        }
        if norm(&residual) < 1.0e-8 {
            break;
        }
        // Steepest descent with a fixed short step: the Rayleigh quotient is bounded below by the
        // eigenvalue being sought, so a small step cannot diverge, and the minimum over the
        // iterates is kept regardless.
        for (xv, r) in x.iter_mut().zip(&residual) {
            *xv -= 0.5 * *r;
        }
        normalize(&mut x);
    }
    Ok((eigenvalue, best))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: &mut [f64]) {
    let n = norm(a);
    if n > 0.0 {
        for v in a.iter_mut() {
            *v /= n;
        }
    }
}

/// A flat slice viewed as an `rows × cols` matrix.
fn as_matrix(flat: &[f64], rows: usize, cols: usize) -> Matrix {
    let mut m = Matrix::zeros(rows, cols);
    m.as_mut_slice().copy_from_slice(&flat[..rows * cols]);
    m
}

/// The pieces one spin channel contributes to the orbital Hessian.
struct Channel {
    /// Occupied and virtual MO coefficient blocks.
    occupied: Matrix,
    virtual_: Matrix,
    /// `ε_i − ε_a`, so the Hessian's diagonal is its negation.
    denominators: Matrix,
    n_occ: usize,
    n_vir: usize,
}

impl Channel {
    fn of(coefficients: &Matrix, energies: &[f64], n_occ: usize, nao: usize) -> Self {
        let n_vir = nao.saturating_sub(n_occ);
        Self {
            occupied: crate::hessian::submatrix_cols(coefficients, 0, n_occ),
            virtual_: crate::hessian::submatrix_cols(coefficients, n_occ, n_vir),
            denominators: crate::hessian::ov_denominators(energies, n_occ, n_vir),
            n_occ,
            n_vir,
        }
    }

    fn len(&self) -> usize {
        self.n_occ * self.n_vir
    }

    /// The AO-basis density this rotation produces, at the given per-spin weight.
    fn density(&self, u: &Matrix, weight: f64) -> Matrix {
        crate::hessian::ao_response_density_w(u, &self.virtual_, &self.occupied, weight)
    }

    fn project(&self, g: &Matrix) -> Matrix {
        crate::hessian::project_ov(g, &self.virtual_, &self.occupied)
    }
}

/// **Measure** the stability of a converged SCF solution, without changing it.
///
/// The standalone form of the analysis: `run_pm7` calls it through
/// [`Pm7Options::stability`](crate::scf::Pm7Options::stability), and a caller who already has a
/// `Pm7Result` can ask the same question of it directly. `Ok(None)` means the question does not
/// apply — an unconverged solve, a periodic cell, or a system with no virtual orbitals to rotate
/// into — and is deliberately not an error, so a caller can ask about anything.
pub fn check(
    molecule: &Molecule,
    params: &crate::params::Pm7Parameters,
    options: &Pm7Options,
    result: &Pm7Result,
) -> Result<Option<Stability>> {
    if !result.converged || molecule.is_periodic() {
        return Ok(None);
    }
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let core = crate::hamiltonian::build_core_field(
        molecule,
        &basis,
        params,
        false,
        options.active_field(),
    )?;

    if result.unrestricted {
        return unrestricted_stability(molecule, &basis, params, &core, options, result);
    }
    restricted_stability(molecule, &basis, params, &core, options, result, nao).map(Some)
}

/// Both channels of a **closed-shell** solution.
fn restricted_stability(
    molecule: &Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    options: &Pm7Options,
    result: &Pm7Result,
    nao: usize,
) -> Result<Stability> {
    let channel = Channel::of(&result.mo_coeff, &result.mo_energies, result.n_occ, nao);
    if channel.len() == 0 {
        return Ok(Stability {
            lowest_ev: f64::INFINITY,
            lowest_triplet_ev: None,
            unstable: false,
            lowered_ev: None,
        });
    }
    let (rows, cols) = (channel.n_vir, channel.n_occ);
    let diagonal: Vec<f64> = channel.denominators.as_slice().iter().map(|d| -d).collect();

    // Singlet: the closed-shell response kernel, which is the CPHF's own operator.
    let singlet = |flat: &[f64]| -> Result<Vec<f64>> {
        let u = as_matrix(flat, rows, cols);
        let rho = channel.density(&u, 2.0);
        let mut g = crate::fock::build_fock_x(
            molecule,
            basis,
            params,
            core,
            &rho,
            options.exchange_cutoff,
        )?;
        for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        let projected = channel.project(&g);
        Ok(projected
            .as_slice()
            .iter()
            .zip(flat)
            .zip(&diagonal)
            .map(|((p, uu), d)| p + d * uu)
            .collect())
    };

    // Triplet: `J` cancels between `A` and `B` for a spin-flip rotation, leaving exchange alone —
    // which `build_fock_spin_x` gives by being handed a zero *total* density and the perturbation
    // as the *spin* one. Weight one, not two: that kernel applies `−K` with no compensating half.
    let zero = Matrix::zeros(basis.nao, basis.nao);
    let triplet = |flat: &[f64]| -> Result<Vec<f64>> {
        let u = as_matrix(flat, rows, cols);
        let rho = channel.density(&u, 1.0);
        let mut g = crate::fock::build_fock_spin_x(
            molecule,
            basis,
            params,
            core,
            &zero,
            &rho,
            options.exchange_cutoff,
        )?;
        for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        let projected = channel.project(&g);
        Ok(projected
            .as_slice()
            .iter()
            .zip(flat)
            .zip(&diagonal)
            .map(|((p, uu), d)| p + d * uu)
            .collect())
    };

    let (lowest_ev, _) = lowest_eigenpair(&singlet, &diagonal)?;
    let (triplet_ev, _) = lowest_eigenpair(&triplet, &diagonal)?;
    Ok(Stability {
        lowest_ev,
        lowest_triplet_ev: Some(triplet_ev),
        unstable: lowest_ev < INSTABILITY_TOLERANCE_EV || triplet_ev < INSTABILITY_TOLERANCE_EV,
        lowered_ev: None,
    })
}

/// The **internal** stability of an unrestricted solution.
///
/// One question rather than two: with the spins already free to differ there is no separate
/// spin-breaking channel, and what is left is whether the solution is a minimum among unrestricted
/// ones. The rotation space is both spins at once and the Coulomb term couples them, which is why
/// the operator below builds the *total* response density from both blocks and the exchange from
/// each block on its own — the same structure the unrestricted CPHF uses.
fn unrestricted_stability(
    molecule: &Molecule,
    basis: &crate::basis::Basis,
    params: &crate::params::Pm7Parameters,
    core: &crate::hamiltonian::CoreHamiltonian,
    options: &Pm7Options,
    result: &Pm7Result,
) -> Result<Option<Stability>> {
    let (Some(coefficients_beta), Some(energies_beta), Some(n_occ_beta)) = (
        result.mo_coeff_beta.as_ref(),
        result.mo_energies_beta.as_ref(),
        result.n_occ_beta,
    ) else {
        return Ok(None);
    };
    let nao = basis.nao;
    let alpha = Channel::of(&result.mo_coeff, &result.mo_energies, result.n_occ, nao);
    let beta = Channel::of(coefficients_beta, energies_beta, n_occ_beta, nao);
    if alpha.len() == 0 && beta.len() == 0 {
        return Ok(None);
    }

    let mut diagonal: Vec<f64> = alpha.denominators.as_slice().iter().map(|d| -d).collect();
    diagonal.extend(beta.denominators.as_slice().iter().map(|d| -d));
    let split = alpha.len();

    let apply = |flat: &[f64]| -> Result<Vec<f64>> {
        let ua = as_matrix(&flat[..split], alpha.n_vir, alpha.n_occ);
        let ub = as_matrix(&flat[split..], beta.n_vir, beta.n_occ);
        // Each spin's own density at weight one, and their sum as the total the Coulomb term sees.
        let da = alpha.density(&ua, 1.0);
        let db = beta.density(&ub, 1.0);
        let mut total = da.clone();
        for (t, v) in total.as_mut_slice().iter_mut().zip(db.as_slice()) {
            *t += *v;
        }
        let mut out = Vec::with_capacity(flat.len());
        for (channel, own) in [(&alpha, &da), (&beta, &db)] {
            let mut g = crate::fock::build_fock_spin_x(
                molecule,
                basis,
                params,
                core,
                &total,
                own,
                options.exchange_cutoff,
            )?;
            for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
                *xv -= *hv;
            }
            out.extend(channel.project(&g).as_slice().iter().copied());
        }
        for ((v, uu), d) in out.iter_mut().zip(flat).zip(&diagonal) {
            *v += d * uu;
        }
        Ok(out)
    };

    let (lowest_ev, _) = lowest_eigenpair(&apply, &diagonal)?;
    Ok(Some(Stability {
        lowest_ev,
        // Already unrestricted: there is no further spin symmetry left to break.
        lowest_triplet_ev: None,
        unstable: lowest_ev < INSTABILITY_TOLERANCE_EV,
        lowered_ev: None,
    }))
}

/// Analyse — and optionally escape — the SCF solution in `result`.
///
/// Returns the solution to keep. Under [`ScfStability::Follow`] that is the lower of the original
/// and whatever following the instability reached, so this can lower an answer and never raise one.
pub fn analyse(
    molecule: &Molecule,
    params: &crate::params::Pm7Parameters,
    options: &Pm7Options,
    result: Pm7Result,
) -> Result<Pm7Result> {
    if options.stability == ScfStability::Off {
        return Ok(result);
    }
    let Some(mut stability) = check(molecule, params, options, &result)? else {
        return Ok(result);
    };
    if !stability.unstable || options.stability != ScfStability::Follow {
        let mut kept = result;
        kept.stability = Some(stability);
        return Ok(kept);
    }

    // Escaping. Which direction depends on which channel complained, and the two are followed
    // differently: a singlet instability is a rotation inside the closed-shell manifold, and a
    // triplet one is a rotation *out* of it, reachable only by letting the spins differ.
    let basis = crate::basis::Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_field(
        molecule,
        &basis,
        params,
        false,
        options.active_field(),
    )?;
    let nao = basis.nao;
    let mut best = result;

    if !best.unrestricted {
        let channel = Channel::of(&best.mo_coeff, &best.mo_energies, best.n_occ, nao);
        let diagonal: Vec<f64> = channel.denominators.as_slice().iter().map(|d| -d).collect();
        let (rows, cols) = (channel.n_vir, channel.n_occ);

        if stability.lowest_triplet_ev.unwrap_or(0.0) < INSTABILITY_TOLERANCE_EV {
            let zero = Matrix::zeros(nao, nao);
            let triplet = |flat: &[f64]| -> Result<Vec<f64>> {
                let u = as_matrix(flat, rows, cols);
                let rho = channel.density(&u, 1.0);
                let mut g = crate::fock::build_fock_spin_x(
                    molecule,
                    &basis,
                    params,
                    &core,
                    &zero,
                    &rho,
                    options.exchange_cutoff,
                )?;
                for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
                    *xv -= *hv;
                }
                let projected = channel.project(&g);
                Ok(projected
                    .as_slice()
                    .iter()
                    .zip(flat)
                    .zip(&diagonal)
                    .map(|((p, uu), d)| p + d * uu)
                    .collect())
            };
            let (_, direction) = lowest_eigenpair(&triplet, &diagonal)?;
            let u = as_matrix(&direction, rows, cols);
            for step in [0.05_f64, 0.15, 0.4] {
                let spin = channel.density(&u, step);
                let mut restarted = options.clone();
                restarted.reference = crate::scf::ScfReference::Unrestricted;
                restarted.initial_spin_density = Some(spin);
                restarted.stability = ScfStability::Off; // one level of following, not recursion
                let Ok(candidate) = crate::scf::run_pm7(molecule, params, &restarted) else {
                    continue;
                };
                if candidate.converged && candidate.total_ev < best.total_ev - 1.0e-9 {
                    stability.lowered_ev = Some(best.total_ev - candidate.total_ev);
                    best = candidate;
                }
            }
            // The spin reference **changed**, and silence about that would be the worst outcome
            // here: the number that comes back is no longer the restricted one the caller asked
            // for, it is not comparable with a restricted number for a neighbouring geometry, and
            // a broken-symmetry UHF solution is not a spin eigenfunction. Say so once, in the
            // crate's usual place, and give the eigenvalue that forced it.
            if best.unrestricted {
                warn_switched_to_unrestricted(
                    stability.lowest_triplet_ev.unwrap_or(f64::NAN),
                    stability.lowered_ev.unwrap_or(0.0),
                );
            }
        }

        if stability.lowest_ev < INSTABILITY_TOLERANCE_EV && !best.unrestricted {
            let singlet = |flat: &[f64]| -> Result<Vec<f64>> {
                let u = as_matrix(flat, rows, cols);
                let rho = channel.density(&u, 2.0);
                let mut g = crate::fock::build_fock_x(
                    molecule,
                    &basis,
                    params,
                    &core,
                    &rho,
                    options.exchange_cutoff,
                )?;
                for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
                    *xv -= *hv;
                }
                let projected = channel.project(&g);
                Ok(projected
                    .as_slice()
                    .iter()
                    .zip(flat)
                    .zip(&diagonal)
                    .map(|((p, uu), d)| p + d * uu)
                    .collect())
            };
            let (_, direction) = lowest_eigenpair(&singlet, &diagonal)?;
            let u = as_matrix(&direction, rows, cols);
            // Mix the rotation into the occupied orbitals. Several step lengths, because the
            // quadratic model that produced the direction says nothing about how far to go along
            // it, and a step that overshoots comes back to the same saddle.
            for step in [0.1_f64, 0.3, 0.6] {
                let mut guess = Matrix::zeros(nao, nao);
                for i in 0..cols {
                    for mu in 0..nao {
                        let mut mixed = 0.0;
                        for a in 0..rows {
                            mixed += u[(a, i)] * channel.virtual_[(mu, a)];
                        }
                        guess[(mu, i)] = channel.occupied[(mu, i)] + step * mixed;
                    }
                }
                // `P = 2 C_occ C_occᵀ`. Not orthonormalized: the SCF's first diagonalization does
                // that, and a guess only has to be in the right basin.
                let mut density = Matrix::zeros(nao, nao);
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut sum = 0.0;
                        for i in 0..cols {
                            sum += guess[(mu, i)] * guess[(nu, i)];
                        }
                        density[(mu, nu)] = 2.0 * sum;
                    }
                }
                let mut restarted = options.clone();
                restarted.initial_density = Some(density);
                restarted.stability = ScfStability::Off;
                let Ok(candidate) = crate::scf::run_pm7(molecule, params, &restarted) else {
                    continue;
                };
                if candidate.converged && candidate.total_ev < best.total_ev - 1.0e-9 {
                    stability.lowered_ev = Some(
                        stability.lowered_ev.unwrap_or(0.0) + (best.total_ev - candidate.total_ev),
                    );
                    best = candidate;
                }
            }
        }
    }

    best.stability = Some(stability);
    Ok(best)
}
