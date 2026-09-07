// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic stress tensor for periodic systems.
//!
//! # Why the virial is exact here
//!
//! Every term in the PM7 energy — resonance `β·S`, electron–core attraction, the two-electron
//! block, core–core repulsion, dispersion, the EH+ hydrogen bond, and the PM7-HH repulsion —
//! is a function of **interatomic displacement vectors alone**. Under a strain `ε` the atoms and
//! the lattice both transform as `d → (1 + ε) d`, so
//!
//! ```text
//! σ_αβ = (1/Ω) ∂E/∂ε_αβ = (1/Ω) Σ_pairs (∂E/∂d_α) d_β
//! ```
//!
//! with `Ω` the cell measure (volume in 3-D, area in 2-D, length in 1-D). Nothing is
//! approximated and nothing extra has to be evaluated: the same forward-mode derivatives the
//! analytic gradient already computes, contracted against the displacement instead of scattered
//! onto atoms, give the stress. The NDDO basis is orthonormal and carries no dependence on the
//! cell, so — unlike a Gaussian-basis code — there is no Pulay/basis-set stress term at all.
//!
//! The long-range part is the one place that needs more than the pair virial: the Ewald
//! reciprocal sum depends on the cell through `1/Ω` and through the reciprocal vectors, and for a
//! charged cell the neutralizing background depends on `Ω` too. Those derivatives are computed in
//! [`crate::pbc::ewald`] and added here.
//!
//! # Sign and units
//!
//! `σ = (1/Ω) ∂E/∂ε` — positive under tension, the convention ASE and MOPAC's `voigt` both use.
//! Units are eV/Bohr³ in 3-D (eV/Bohr² in 2-D, eV/Bohr in 1-D, since `Ω` is then an area or a
//! length).

use crate::basis::Basis;
use crate::error::{Pm7Error, Result};
use crate::math::Mat3;
use crate::params::Pm7Parameters;
use crate::scf::{Pm7Options, Pm7Result};
use crate::system::Molecule;

/// A stress tensor with the pieces it was built from, so a caller can see where a pressure came
/// from instead of only that it exists.
#[derive(Clone, Copy, Debug)]
pub struct StressResult {
    /// Total stress `σ = (1/Ω) ∂E/∂ε`.
    pub stress: Mat3,
    /// Short-range electronic contribution (resonance, electron–core, two-electron).
    pub electronic: Mat3,
    /// Core–core repulsion contribution.
    pub core: Mat3,
    /// Long-range Ewald contribution, including the charged-cell background.
    pub ewald: Mat3,
    /// Post-SCF corrections (dispersion, EH+, PM7-HH).
    pub correction: Mat3,
    /// The cell measure the virial was divided by (Bohr³ / Bohr² / Bohr).
    pub measure: f64,
}

impl StressResult {
    /// Hydrostatic pressure `P = −Tr σ / 3`, in eV/Bohr³. Only meaningful in 3-D.
    pub fn pressure(&self) -> f64 {
        -self.stress.trace() / 3.0
    }

    /// Pressure in GPa. `None` unless the cell is 3-D, where alone a volume exists.
    pub fn pressure_gpa(&self, molecule: &Molecule) -> Option<f64> {
        molecule.cell?.volume()?;
        Some(self.pressure() * EV_PER_BOHR3_TO_GPA)
    }

    /// Voigt form `[xx, yy, zz, yz, xz, xy]`, the order ASE and MOPAC both use.
    pub fn voigt(&self) -> [f64; 6] {
        self.stress.to_voigt()
    }
}

/// eV/Bohr³ → GPa. `1 eV = 1.602176634e-19 J` (exact, 2019 SI) and `1 Bohr = PM7_A0 × 1e-10 m`.
pub const EV_PER_BOHR3_TO_GPA: f64 = {
    let bohr_m = crate::constants::PM7_A0 * 1.0e-10;
    1.602_176_634e-19 / (bohr_m * bohr_m * bohr_m) / 1.0e9
};

/// Analytic stress at a converged SCF solution.
///
/// `scf` must be the result of running `run_pm7` on this exact `molecule` and `options`: the
/// stress is a fixed-density (Hellmann–Feynman) quantity, so feeding it a density from a
/// different geometry silently gives a wrong answer rather than an error.
pub fn analytic_stress(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    scf: &Pm7Result,
) -> Result<StressResult> {
    let Some(cell) = molecule.cell else {
        return Err(Pm7Error::InvalidInput(
            "stress is only defined for a periodic system; this molecule has no cell".into(),
        ));
    };
    let pbc = options.pbc_for(molecule).expect("cell implies pbc options");
    pbc.validate()?;
    // A field is admissible on a periodic cell only along a free direction, and there it does
    // shift the energy and the forces — but its strain derivative is a separate question this
    // code does not answer: the coupling of a free Cartesian axis to a strain of the periodic
    // axes is not in the virial assembled below. Refusing is the honest answer; returning the
    // field-free stress under a field would look like a number and be one.
    if options.active_field().is_some() {
        return Err(Pm7Error::InvalidInput(
            "stress under an external field is not implemented: the field's strain derivative is \
             not part of the virial, so the result would silently be the field-free stress. Use \
             `closed_form_gradient` for forces under a field, or drop the field for a stress."
                .into(),
        ));
    }
    let basis = Basis::build(molecule, params)?;

    // Short-range electronic virial, from the same pair enumeration the energy used. Open-shell
    // systems take the spin-resolved contraction; everything else about the periodic treatment
    // is identical, so UHF stresses are supported on the same terms as RHF.
    // A k-point run's density decays with distance, so the pair terms have to see `P(T)` rather
    // than `P(0)`; `TranslatedDensity` carries whichever the run produced.
    let divisions = pbc.kmesh.divisions();
    let half_blocks = scf
        .bloch_density
        .as_ref()
        .map(|b| crate::scf_pbc::scale_blocks(b, 0.5));
    let (pa_flat, pb_flat) = spin_densities(scf);
    let spin_blocks = scf
        .bloch_density
        .as_ref()
        .map(|total| crate::gradient::bloch_spin_densities(total, scf.bloch_spin_density.as_ref()));
    let total_density = match &scf.bloch_density {
        Some(b) => crate::gradient::TranslatedDensity::Bloch(b, divisions),
        None => crate::gradient::TranslatedDensity::Uniform(&scf.density),
    };
    let (alpha, beta) = match (&spin_blocks, &half_blocks) {
        (Some((a, b)), _) if scf.unrestricted => (
            crate::gradient::TranslatedDensity::Bloch(a, divisions),
            crate::gradient::TranslatedDensity::Bloch(b, divisions),
        ),
        (_, Some(h)) => (
            crate::gradient::TranslatedDensity::Bloch(h, divisions),
            crate::gradient::TranslatedDensity::Bloch(h, divisions),
        ),
        _ => (
            crate::gradient::TranslatedDensity::Uniform(&pa_flat),
            crate::gradient::TranslatedDensity::Uniform(&pb_flat),
        ),
    };

    let electronic = if scf.unrestricted {
        let (_, v) = spin_electronic_virial(molecule, params, options, scf)?;
        v
    } else {
        crate::gradient::electronic_gradient_periodic(
            molecule,
            params,
            &basis,
            total_density,
            &pbc,
        )?
        .1
    };
    let (_, core) = crate::repulsion::core_core_gradient_periodic(molecule, params, &pbc)?;

    // Long-range monopoles, Coulomb and exchange. The Ewald virial covers the reciprocal-space
    // cell dependence and the charged-cell background, neither of which is a pair term.
    let ewald = match pbc.mode {
        crate::pbc::PbcMode::Ewald => {
            let positions: Vec<crate::math::Vec3> =
                molecule.atoms.iter().map(|a| a.position).collect();
            let ep = crate::pbc::EwaldParameters::new(
                &cell,
                positions.len(),
                pbc.ewald_accuracy,
                pbc.ewald_alpha,
            );
            let coulomb =
                crate::pbc::ewald::ewald(&cell, &positions, &scf.charges, &ep, &pbc)?.virial;
            let (_, exchange) = crate::gradient::long_range_exchange_derivatives(
                &cell,
                &positions,
                &basis,
                &[alpha, beta],
                &pbc,
            )?;
            coulomb.plus(&exchange)
        }
        crate::pbc::PbcMode::MopacCluster => Mat3::zero(),
    };

    let (_, correction) = crate::gradient::correction_gradient_and_virial(molecule, options);

    let measure = cell.measure();
    let total = electronic
        .plus(&core)
        .plus(&ewald)
        .plus(&correction)
        .symmetrized()
        .scaled(1.0 / measure);
    Ok(StressResult {
        stress: total,
        electronic: electronic.symmetrized().scaled(1.0 / measure),
        core: core.symmetrized().scaled(1.0 / measure),
        ewald: ewald.symmetrized().scaled(1.0 / measure),
        correction: correction.symmetrized().scaled(1.0 / measure),
        measure,
    })
}

/// The α and β density matrices, reconstructed from the total and spin densities. For a closed
/// shell both are half the total.
fn spin_densities(scf: &Pm7Result) -> (crate::linalg::Matrix, crate::linalg::Matrix) {
    let mut pa = scf.density.clone();
    let mut pb = scf.density.clone();
    match &scf.spin_density {
        None => {
            for v in pa.as_mut_slice() {
                *v *= 0.5;
            }
            for v in pb.as_mut_slice() {
                *v *= 0.5;
            }
        }
        Some(s) => {
            let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
            let (pts, ss) = (scf.density.as_slice(), s.as_slice());
            for i in 0..pts.len() {
                pas[i] = 0.5 * (pts[i] + ss[i]);
                pbs[i] = 0.5 * (pts[i] - ss[i]);
            }
        }
    }
    (pa, pb)
}

/// The open-shell short-range electronic virial.
///
/// `fixed_density_gradient_uhf_with` returns the *total* periodic virial; the core–core and
/// long-range parts are subtracted back out so that [`StressResult`]'s breakdown stays
/// meaningful and the caller can add them once.
fn spin_electronic_virial(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    scf: &Pm7Result,
) -> Result<(Vec<crate::math::Vec3>, Mat3)> {
    let pbc = options.pbc_for(molecule).expect("periodic");
    let basis = Basis::build(molecule, params)?;
    let (g, total) =
        crate::gradient::fixed_density_gradient_uhf_with(molecule, params, scf, options)?;
    let (_, core) = crate::repulsion::core_core_gradient_periodic(molecule, params, &pbc)?;
    let long_range = if pbc.mode == crate::pbc::PbcMode::Ewald {
        let cell = molecule.cell.expect("periodic");
        let positions: Vec<crate::math::Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        let ep = crate::pbc::EwaldParameters::new(
            &cell,
            positions.len(),
            pbc.ewald_accuracy,
            pbc.ewald_alpha,
        );
        let coulomb = crate::pbc::ewald::ewald(&cell, &positions, &scf.charges, &ep, &pbc)?.virial;
        let divisions = pbc.kmesh.divisions();
        let (pa_flat, pb_flat) = spin_densities(scf);
        let blocks = scf.bloch_density.as_ref().map(|total| {
            crate::gradient::bloch_spin_densities(total, scf.bloch_spin_density.as_ref())
        });
        let (alpha, beta) = match &blocks {
            Some((a, b)) => (
                crate::gradient::TranslatedDensity::Bloch(a, divisions),
                crate::gradient::TranslatedDensity::Bloch(b, divisions),
            ),
            None => (
                crate::gradient::TranslatedDensity::Uniform(&pa_flat),
                crate::gradient::TranslatedDensity::Uniform(&pb_flat),
            ),
        };
        let (_, exchange) = crate::gradient::long_range_exchange_derivatives(
            &cell,
            &positions,
            &basis,
            &[alpha, beta],
            &pbc,
        )?;
        coulomb.plus(&exchange)
    } else {
        Mat3::zero()
    };
    Ok((
        g,
        total
            .plus(&core.scaled(-1.0))
            .plus(&long_range.scaled(-1.0)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpa_conversion_matches_the_defining_constants() {
        // 1 Hartree/Bohr³ is 29421.015697 GPa (CODATA "atomic unit of pressure"). Going through
        // eV/Bohr³ must reproduce it, which pins both the eV and the Bohr in the constant.
        let hartree_per_bohr3_to_gpa = crate::constants::HARTREE_TO_EV * EV_PER_BOHR3_TO_GPA;
        assert!(
            (hartree_per_bohr3_to_gpa - 29_421.015_697).abs() < 1.0e-3,
            "1 Hartree/Bohr^3 came out as {hartree_per_bohr3_to_gpa} GPa"
        );
    }
}
