// SPDX-License-Identifier: GPL-3.0-or-later

//! Dipole derivatives and infrared intensities.
//!
//! # Why this is nearly free
//!
//! An IR spectrum needs `∂mu/∂R`, which needs the first-order density response — and the analytic
//! Hessian already solves for exactly that and throws it away. So the spectrum costs one extra
//! dipole-operator build and `3N` traces on top of a Hessian that was going to be computed
//! anyway. That is why [`ir_spectrum`] computes both in **one** call: asking for a Hessian and a
//! spectrum separately would pay for the CPHF twice.
//!
//! # The two derivative terms
//!
//! ```text
//! ∂mu_a/∂R_{B,b} = q_B delta_ab  +  Tr[ D_a · ∂P/∂R_{B,b} ]
//! ```
//!
//! an **explicit** term from the operator's own coordinate dependence (only the point-charge part
//! has one) and an **implicit** term through the density. Dropping either leaves a plausible
//! spectrum, which is why both finite-difference checks in `tests/ir.rs` are against quantities
//! that share no code with this file.
//!
//! # Origin
//!
//! The raw tensor is always taken about the input coordinate origin, whatever
//! [`crate::dipole::DipoleOrigin`] says — convention C-8 in `docs/theory.md`. A moving origin
//! would add `−q_tot · m_B/M` and make the tensor convention-dependent for an ion. MOPAC arrives
//! at the same place by disabling its recentring under `FORCE` (`dipole.F90:86-88`).

use crate::basis::Basis;
use crate::constants::{
    AU_DIPOLE_TO_DEBYE, E_IN_DEBYE_PER_ANGSTROM, IR_E2_PER_AMU_TO_KM_PER_MOL, PM7_A0,
};
use crate::data_tables::MASS;
use crate::dipole::{dipole_operator, DipoleTerms};
use crate::error::Result;
use crate::hessian::{analytic_hessian_with, HessianRequest, VibrationalModes};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::projection::Projection;
use crate::scf::{Pm7Options, Pm7Result};
use crate::system::Molecule;

/// A harmonic infrared spectrum, in both of the forms a caller normally wants.
///
/// **Seven fields are indexed by mode and move together**: `frequencies_cm`, `modes` and
/// `cartesian_modes` (by column), `mode_dipole_derivatives` and `mopac_trdip` (by row),
/// `intensities_km_per_mol` and `mopac_dipt`. Since 0.2.3 that count is `3N − 6` for a molecule
/// (`3N − 5` linear) rather than `3N`, because the rigid-body subspace is projected out — see
/// [`crate::projection`]. `dipole_derivatives` is **not** one of them: it is indexed by Cartesian
/// degree of freedom and stays `3 × 3N`, as does `hessian` at `3N × 3N`.
#[derive(Clone, Debug)]
pub struct IrSpectrum {
    /// **The dense raw tensor**: `∂mu_a/∂x_t`, `3 × 3N`, in atomic units (e·Bohr per Bohr = e).
    /// Row `a` is a Cartesian dipole component, column `t = 3A + b` a nuclear degree of freedom.
    pub dipole_derivatives: Matrix,
    /// Harmonic frequencies (cm⁻¹), ascending — the same ordering every other field uses.
    pub frequencies_cm: Vec<f64>,
    /// Mass-weighted normal modes, as columns.
    pub modes: Matrix,
    /// Cartesian normal modes, unit-normalized per column.
    pub cartesian_modes: Matrix,
    /// **The normal-mode form**: `∂mu/∂Q_n`, one row per mode, in e·amu^(−1/2).
    pub mode_dipole_derivatives: Matrix,
    /// Double-harmonic IR intensity per mode, km/mol.
    pub intensities_km_per_mol: Vec<f64>,
    /// **What MOPAC prints as `DIPT`** under `FORCE LARGE`, reproduced so the two can be compared
    /// directly. Debye/Ångström.
    ///
    /// Two things make this *not* an IR intensity, and both are properties of MOPAC's definition
    /// rather than choices made here:
    ///
    /// * it projects on the **Cartesian** normal mode, renormalized without mass weighting
    ///   (`force.F90:425-441`), so it is not proportional to `|∂mu/∂Q|²`; and
    /// * it is **half the actual derivative**. `fmat.F90:197-252` evaluates the dipole at
    ///   `+δ/2` and `−δ/2` — a separation of `δ` — and then divides the difference by `2δ`. The
    ///   factor was confirmed empirically before being confirmed in the source: water's three
    ///   vibrations gave ratios of 1.995, 2.002 and 1.995 against this implementation.
    ///
    /// Use [`Self::intensities_km_per_mol`] for spectroscopy and this only for the oracle.
    pub mopac_dipt: Vec<f64>,
    /// MOPAC's `DIPX`/`DIPY`/`DIPZ`, one row per mode, Debye/Ångström.
    ///
    /// Component-wise comparison against MOPAC is **ill-posed**: the sign of a normal-mode
    /// eigenvector is arbitrary and degenerate modes mix arbitrarily. Compare [`Self::mopac_dipt`],
    /// or a sum over a degenerate set.
    pub mopac_trdip: Matrix,
    /// The converged SCF, returned so a caller need not repeat it.
    pub scf: Pm7Result,
    /// The Hessian the modes came from (eV/Bohr²).
    pub hessian: Matrix,
}

/// The `3 × 3N` dipole-derivative tensor alone, when the modes are not wanted.
///
/// `terms` selects the operator: [`DipoleTerms::Full`] for a physical spectrum,
/// [`DipoleTerms::FieldConjugate`] for the quantity that equals `∂²E/∂f∂R` and can therefore be
/// checked against a finite difference of the field gradient. They differ for any d-bearing atom;
/// see `docs/theory.md` convention C-2.
pub fn dipole_derivatives(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
    terms: DipoleTerms,
) -> Result<Matrix> {
    refuse_periodic(molecule, "dipole derivatives")?;
    let out = analytic_hessian_with(
        molecule,
        params,
        options,
        step,
        &HessianRequest::with_response(),
    )?;
    let basis = Basis::build(molecule, params)?;
    derivatives_from(molecule, params, &basis, &out, terms)
}

/// Refuse a periodic cell here, by name, before the molecular machinery gets a chance to fail.
///
/// `∂μ/∂R` and the intensities built on it are **molecular** quantities: the dipole of a periodic
/// cell is not a function of the density alone, it depends on the surface termination, which is
/// exactly the problem the Born effective charge exists to solve. The periodic answer is
/// `born_and_dielectric`, whose `Z*` is the same derivative done properly.
///
/// Without this, a periodic cell reached the molecular Hessian and died several layers down on
/// "dipole derivatives need the retained CPHF orbital response" — true, in that the periodic path
/// does not retain one, and useless, in that it names an internal and not the thing to do instead.
/// The ASE calculator refused this correctly and the library did not, so the CLI and anyone
/// calling `native.vibrations(ir=True)` got the internal message.
fn refuse_periodic(molecule: &Molecule, what: &str) -> Result<()> {
    if molecule.cell.is_some() {
        return Err(crate::error::Pm7Error::InvalidInput(format!(
            "{what} are a molecular quantity and this system has a cell. The dipole of a periodic \
             cell depends on how the crystal is terminated, not on the density alone. Use \
             `born_and_dielectric` (Python: `born_charges`, CLI: `born`) — the Born effective \
             charge is this derivative done in a way that survives periodicity."
        )));
    }
    Ok(())
}

/// Assemble `∂mu/∂R` from an already-solved Hessian plus its retained response.
fn derivatives_from(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    hessian: &crate::hessian::HessianResult,
    terms: DipoleTerms,
) -> Result<Matrix> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut out = Matrix::zeros(3, ndof);

    // Explicit term: only the point-charge part of the operator moves with the nuclei, and it
    // moves as `q_B delta_ab`.
    for (b, charge) in hessian.scf.charges.iter().enumerate() {
        for axis in 0..3 {
            out[(axis, 3 * b + axis)] += *charge;
        }
    }

    // Implicit term: the operator contracted with the density response. Always about the
    // coordinate origin (convention C-8).
    let Some(response) = &hessian.response else {
        return Err(crate::error::Pm7Error::InvalidInput(
            "dipole derivatives need the retained CPHF orbital response".into(),
        ));
    };
    let d = dipole_operator(molecule, basis, params, Vec3::zero(), terms)?;
    for t in 0..ndof.min(response.len()) {
        let dp = response.density_derivative(t);
        for (axis, d_axis) in d.iter().enumerate() {
            out[(axis, t)] += dp.frobenius_dot(d_axis);
        }
    }
    Ok(out)
}

/// Frequencies, normal modes, dipole derivatives and IR intensities from **one** CPHF solve.
///
/// Projects out the rigid-body subspace, so a non-linear molecule returns `3N − 6` modes. See
/// [`ir_spectrum_projected`] to ask for the raw set instead.
pub fn ir_spectrum(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<IrSpectrum> {
    ir_spectrum_projected(molecule, params, options, step, Projection::default())
}

/// [`ir_spectrum`], choosing what to project out.
///
/// [`Projection::None`] returns the unprojected `3N` set, with the translations and rotations left
/// in and each carrying whatever intensity the raw modes give them. That is a diagnostic — it is
/// what shows *which* numbers the projector removed and how far from zero they were — and not a
/// spectrum: a rigid translation of a neutral molecule has no infrared activity, so any intensity
/// on those rows is Hessian error made visible.
pub fn ir_spectrum_projected(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
    projection: Projection,
) -> Result<IrSpectrum> {
    refuse_periodic(molecule, "infrared intensities")?;
    let solved = analytic_hessian_with(
        molecule,
        params,
        options,
        step,
        &HessianRequest::with_response(),
    )?;
    let basis = Basis::build(molecule, params)?;
    // IR intensities use the **physical** dipole, p–d term included.
    let dipole_derivatives =
        derivatives_from(molecule, params, &basis, &solved, DipoleTerms::Full)?;
    let modes =
        crate::hessian::vibrational_modes_projected(molecule, solved.hessian.clone(), projection)?;
    Ok(assemble(molecule, dipole_derivatives, modes, solved))
}

fn assemble(
    molecule: &Molecule,
    dipole_derivatives: Matrix,
    vibrations: VibrationalModes,
    solved: crate::hessian::HessianResult,
) -> IrSpectrum {
    let ndof = 3 * molecule.atoms.len();
    let mass_of = |dof: usize| MASS[molecule.atoms[dof / 3].z as usize];
    // The two are no longer the same number. `ndof` indexes Cartesian degrees of freedom, which the
    // inner contractions run over; `n_modes` indexes the spectrum, which the rigid-body projection
    // has shortened for a molecule (`3N − 6`) and left at `3N` for a cell. Every mode-indexed field
    // below is `n_modes` long, `dipole_derivatives` stays `3 × ndof`, and confusing the two is the
    // one way to get a silently misaligned intensity.
    let n_modes = vibrations.frequencies_cm.len();

    // `∂mu/∂Q_n = Σ_i (∂mu/∂x_i) m_i^(−1/2) L_in`, in e·amu^(−1/2) with `x` in Bohr.
    let mut mode_dipole_derivatives = Matrix::zeros(n_modes, 3);
    for n in 0..n_modes {
        for axis in 0..3 {
            let mut total = 0.0;
            for i in 0..ndof {
                total +=
                    dipole_derivatives[(axis, i)] * vibrations.modes[(i, n)] / mass_of(i).sqrt();
            }
            mode_dipole_derivatives[(n, axis)] = total;
        }
    }
    let intensities_km_per_mol = (0..n_modes)
        .map(|n| {
            let squared: f64 = (0..3)
                .map(|a| mode_dipole_derivatives[(n, a)].powi(2))
                .sum();
            squared * IR_E2_PER_AMU_TO_KM_PER_MOL
        })
        .collect();

    // MOPAC's quantity: the derivative along the **unit Cartesian** mode, in Debye/Angstrom, and
    // then halved.
    //
    // The halving reproduces `fmat.F90:197-252`, which evaluates the dipole at `+δ/2` and `−δ/2`
    // — a separation of `δ` — and divides the difference by `2δ`. So MOPAC's printed `DIPT` is
    // half the true derivative. Matching it is the point: this field exists only so a MOPAC
    // `FORCE LARGE` run can be compared number for number, and a field that does not reproduce
    // the number it is named after would be worse than not having one.
    let mut mopac_trdip = Matrix::zeros(n_modes, 3);
    let mut mopac_dipt = vec![0.0; n_modes];
    for n in 0..n_modes {
        let mut squared = 0.0;
        for axis in 0..3 {
            let mut total = 0.0;
            for i in 0..ndof {
                total += dipole_derivatives[(axis, i)] * vibrations.cartesian_modes[(i, n)];
            }
            let value = 0.5 * total * E_IN_DEBYE_PER_ANGSTROM;
            mopac_trdip[(n, axis)] = value;
            squared += value * value;
        }
        mopac_dipt[n] = squared.sqrt();
    }

    IrSpectrum {
        dipole_derivatives,
        frequencies_cm: vibrations.frequencies_cm,
        modes: vibrations.modes,
        cartesian_modes: vibrations.cartesian_modes,
        mode_dipole_derivatives,
        intensities_km_per_mol,
        mopac_dipt,
        mopac_trdip,
        scf: solved.scf,
        hessian: solved.hessian,
    }
}

/// The dipole in Debye from a converged result, recomputed about the coordinate origin.
///
/// Used by the finite-difference tests, which must differentiate a *fixed-origin* dipole for the
/// comparison against [`dipole_derivatives`] to mean anything (convention C-8).
pub fn dipole_at_coordinate_origin(result: &Pm7Result) -> Vec3 {
    // The hybrid parts are origin-independent; only the point-charge part needs undoing, and
    // `origin` records exactly what was subtracted.
    let mut total = result.dipole.total();
    let shift: f64 = result.charges.iter().sum();
    total += result.dipole.origin * (shift * AU_DIPOLE_TO_DEBYE);
    total
}

/// Bohr per Ångström, for callers converting the raw tensor into spectroscopic units.
pub const BOHR_PER_ANGSTROM: f64 = 1.0 / PM7_A0;
