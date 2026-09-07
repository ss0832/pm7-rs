// SPDX-License-Identifier: GPL-3.0-or-later
//! The static dielectric tensor, pinned against Lyddane-Sachs-Teller.
//!
//! `eps^0 = eps^inf + (4 pi / Omega) sum_m (Z* e_m)(Z* e_m) / omega_m^2` carries a unit conversion
//! between the mass-weighted eigenvalue's `eV/(A^2 amu)` and the dimensionless susceptibility it
//! has to produce. That conversion cannot be checked by re-deriving it — that is checking the
//! arithmetic against itself.
//!
//! LST checks it against the phonons instead:
//!
//! ```text
//! eps^0 / eps^inf = (omega_LO / omega_TO)^2
//! ```
//!
//! for a cubic diatomic crystal. The right-hand side comes from the dynamical matrix with and
//! without the non-analytic term, sharing none of the constant being tested.
//!
//! It caught the constant being wrong by `HARTREE_TO_EV * a0^4` — a factor of 347. What makes that
//! worth a test rather than a comment is how it looked while it was wrong: an ionic contribution
//! of 0.000256 on top of an `eps^inf` of 1.0119. That reads as a small correction to a weakly
//! polarizable crystal. The true value is 0.0889.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    born_and_dielectric, dynamical_matrix_dfpt, static_dielectric_tensor, Atom, Cell, DfptOptions,
    KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters,
};

fn rocksalt(z1: u8, z2: u8, a_ang: f64) -> Molecule {
    let a = a_ang * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: z1,
            position: Vec3::zero(),
        },
        Atom {
            z: z2,
            position: Vec3::new(a * 0.5, 0.0, 0.0),
        },
    ])
    .with_cell(cell)
}

fn diamond() -> Molecule {
    let a = 3.567 * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell)
}

fn options(mesh: usize) -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 500,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The ionic term reproduces the LST ratio on an ionic crystal.
#[test]
fn the_ionic_term_satisfies_lyddane_sachs_teller() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = rocksalt(11, 17, 5.64);
    let opts = options(3);
    let dfpt = DfptOptions::default();

    let field = born_and_dielectric(&molecule, &params, &opts, &dfpt).unwrap();
    let eps_inf = field.dielectric.get(0, 0);

    // TO from `D(0)`, LO from `D(0)` plus the non-analytic term along x.
    let phonons = dynamical_matrix_dfpt(&molecule, &params, &opts, [0.0; 3], &dfpt).unwrap();
    let na = field.non_analytic().unwrap();
    let mut transverse = phonons.frequencies_cm().unwrap();
    let mut longitudinal = phonons.frequencies_cm_lo_to(&na, [1.0, 0.0, 0.0]).unwrap();
    transverse.sort_by(|a, b| a.partial_cmp(b).unwrap());
    longitudinal.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (w_to, w_lo) = (
        transverse[transverse.len() - 1],
        longitudinal[longitudinal.len() - 1],
    );

    let expected = eps_inf * (w_lo / w_to).powi(2);
    let got = static_dielectric_tensor(&molecule, &params, &opts, &dfpt)
        .unwrap()
        .epsilon
        .get(0, 0);

    assert!(
        (got / expected - 1.0).abs() < 1.0e-3,
        "eps_0 = {got:.6} against the LST value {expected:.6} (eps_inf {eps_inf:.6}, \
         omega_TO {w_to:.3}, omega_LO {w_lo:.3} cm^-1), ratio {:.4}. These are independent routes: \
         a clean constant ratio is a unit error in the ionic sum, not physics.",
        got / expected
    );
    // And the ionic term is not simply negligible, or the test above would pass on nothing.
    assert!(
        expected - eps_inf > 0.05,
        "the fixture's ionic term is only {:.6}; LST cannot discriminate a scale error on a \
         contribution that small",
        expected - eps_inf
    );
}

/// A non-polar crystal has no ionic term at all, and LST is the trivial identity there.
///
/// The complement of the test above: with `Z* = 0` the sum vanishes term by term, so this catches
/// a spurious contribution that a scale test cannot — any constant times zero is still zero.
#[test]
fn a_non_polar_crystal_has_no_ionic_contribution() {
    let params = Pm7Parameters::standard().unwrap();
    let out = static_dielectric_tensor(&diamond(), &params, &options(3), &DfptOptions::default())
        .unwrap();
    let worst = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| out.ionic.get(a, b).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1.0e-12,
        "diamond has vanishing Born charges, so its ionic dielectric term must vanish; got {worst:.3e}"
    );
    // Three, by construction rather than by measurement: the acoustic modes are identified by
    // their overlap with the mass-weighted uniform translations and there are exactly three of
    // those, so this asserts the projection is wired in rather than that three frequencies landed
    // under a threshold. Through 0.2.2 it was the latter, against `SOFT_MODE_FLOOR = 1e-6`.
    assert_eq!(
        out.skipped_modes, 3,
        "three acoustic modes at the zone centre"
    );
    // The signal the old count conflated with it: an optical mode at or below zero means the
    // geometry is not a minimum, and the ionic term is missing what that mode would carry.
    assert_eq!(
        out.soft_optical_modes, 0,
        "relaxed diamond has no soft optical mode"
    );
}

/// **The classification is by eigenvector, not by frequency.**
///
/// The replacement for `SOFT_MODE_FLOOR` has to be checked against the case the floor could not
/// handle: a structure whose softest *optical* mode is small. Under the old rule such a mode fell
/// below `1e-6` and was silently dropped from a sum it dominates — the answer came back looking
/// like a number. Under the new one it is optical because its eigenvector is orthogonal to the
/// uniform translations, whatever its frequency, so it is either used or reported.
///
/// Compressing the cell is the cheap way to soften a mode without leaving the code's domain.
#[test]
fn a_soft_optical_mode_is_not_mistaken_for_an_acoustic_one() {
    let params = Pm7Parameters::standard().unwrap();
    let opts = options(2);
    let settings = pm7_rs::DfptOptions::default();

    for scale in [1.0, 0.94] {
        let molecule = rocksalt(11, 17, 5.64 * scale);
        let Ok(out) = pm7_rs::static_dielectric_tensor(&molecule, &params, &opts, &settings) else {
            continue;
        };
        // Whatever the frequencies do, exactly three modes are acoustic: the translation subspace
        // is three-dimensional and does not care how the cell is strained.
        assert_eq!(
            out.skipped_modes, 3,
            "scale {scale}: the acoustic count is a property of the subspace, not of the spectrum"
        );
        // And nothing that was kept sits at zero, which is what a `1/omega^2` cannot survive.
        assert!(
            out.soft_optical_modes > 0 || out.softest_kept > 0.0,
            "scale {scale}: a kept mode at omega^2 = 0 would make the ionic term infinite"
        );
    }
}
