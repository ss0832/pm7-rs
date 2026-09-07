// SPDX-License-Identifier: GPL-3.0-or-later
//! Dipole moments against MOPAC v23.2.5.
//!
//! These numbers are transcribed from runs of the vendored MOPAC binary
//! (`PM7 1SCF PRECISE GRADIENTS RELSCF=0.0001`), so the fidelity claim stays checkable by anyone
//! running `cargo test`, with or without MOPAC installed. `tools/oracle/baseline.py` re-derives
//! them from a live MOPAC when it is available.
//!
//! The d-shell cases are the point. Before v0.2.1 the one-centre **p–d** hybridization term
//! (MOPAC's `ddp(5)`, `dipole.F90:118-148`) was missing entirely, so for H2S the reported hybrid
//! contribution was (0.917, 1.202) D where MOPAC gives (0.285, 0.374) D — not a rounding
//! difference but a missing term of the same size as the answer.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::dipole::DipoleOrigin;
use pm7_rs::math::Vec3;
use pm7_rs::{run_pm7, Atom, Molecule, Pm7Options, Pm7Parameters};

fn at(z: u8, x: f64, y: f64, z_: f64) -> Atom {
    Atom {
        z,
        position: Vec3::new(
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z_ * ANGSTROM_TO_BOHR,
        ),
    }
}

fn tight(charge: f64) -> Pm7Options {
    Pm7Options {
        charge,
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..Default::default()
    }
}

/// `(name, atoms, charge, MOPAC point-charge, MOPAC hybrid, MOPAC sum)`, all Debye.
///
/// MOPAC prints one `HYBRID` row covering both the s–p and p–d terms, so the reference is
/// compared against `sp_hybrid + pd_hybrid`.
type Case = (&'static str, Vec<Atom>, f64, [f64; 3], [f64; 3], [f64; 3]);

fn cases() -> Vec<Case> {
    vec![
        (
            "water",
            vec![
                at(8, 0.0, 0.0, 0.0),
                at(1, 0.96, 0.0, 0.0),
                at(1, -0.24, 0.93, 0.0),
            ],
            0.0,
            [1.115, 1.441, 0.000],
            [0.197, 0.254, 0.000],
            [1.312, 1.695, 0.000],
        ),
        (
            "hydrogen_sulfide",
            vec![
                at(16, 0.0, 0.0, 0.0),
                at(1, 1.34, 0.0, 0.0),
                at(1, -0.35, 1.29, 0.0),
            ],
            0.0,
            [0.824, 1.069, 0.000],
            [0.285, 0.374, 0.000],
            [1.109, 1.443, 0.000],
        ),
        (
            "sulfur_dioxide",
            vec![
                at(16, 0.0, 0.0, 0.0),
                at(8, 1.24, 0.0, 0.72),
                at(8, -1.24, 0.0, 0.72),
            ],
            0.0,
            [0.000, 0.000, -4.764],
            [0.000, 0.000, 1.472],
            [0.000, 0.000, -3.292],
        ),
        (
            "phosphorus_trifluoride",
            vec![
                at(15, 0.0, 0.0, 0.0),
                at(9, 1.24, 0.0, 0.51),
                at(9, -0.62, 1.07, 0.51),
                at(9, -0.62, -1.07, 0.51),
            ],
            0.0,
            [-0.004, 0.000, -1.677],
            [-0.001, 0.000, 0.883],
            [-0.005, 0.000, -0.795],
        ),
    ]
}

/// MOPAC prints the dipole to three decimals, so the threshold sits just above that.
const TOLERANCE: f64 = 2.0e-3;

#[test]
fn the_dipole_breakdown_matches_mopac() {
    let params = Pm7Parameters::standard().unwrap();
    for (name, atoms, charge, point_charge, hybrid, sum) in cases() {
        let molecule = Molecule::new(atoms);
        let result = run_pm7(&molecule, &params, &tight(charge)).unwrap();
        let parts = result.dipole;
        for axis in 0..3 {
            let combined_hybrid = parts.sp_hybrid.get(axis) + parts.pd_hybrid.get(axis);
            assert!(
                (parts.point_charge.get(axis) - point_charge[axis]).abs() < TOLERANCE,
                "{name} point charge axis {axis}: {} vs MOPAC {}",
                parts.point_charge.get(axis),
                point_charge[axis]
            );
            assert!(
                (combined_hybrid - hybrid[axis]).abs() < TOLERANCE,
                "{name} hybrid axis {axis}: {combined_hybrid} vs MOPAC {}",
                hybrid[axis]
            );
            assert!(
                (parts.total().get(axis) - sum[axis]).abs() < TOLERANCE,
                "{name} sum axis {axis}: {} vs MOPAC {}",
                parts.total().get(axis),
                sum[axis]
            );
        }
    }
}

/// The p–d term is not a small correction: without it H2S's hybrid dipole is off by more than
/// the value MOPAC reports. This asserts the size, so a regression that silently drops the term
/// cannot pass by being "close enough".
#[test]
fn the_pd_term_is_large_enough_that_omitting_it_would_be_obvious() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = Molecule::new(vec![
        at(16, 0.0, 0.0, 0.0),
        at(1, 1.34, 0.0, 0.0),
        at(1, -0.35, 1.29, 0.0),
    ]);
    let parts = run_pm7(&molecule, &params, &tight(0.0)).unwrap().dipole;
    assert!(
        parts.pd_hybrid.norm() > 0.5,
        "the p-d hybrid term should be ~1 D for H2S, got {:?}",
        parts.pd_hybrid
    );
    // Dropping it would leave the s-p part alone, which is nowhere near MOPAC's hybrid row.
    let without_pd = parts.sp_hybrid.norm();
    let with_pd = (parts.sp_hybrid + parts.pd_hybrid).norm();
    assert!(
        (without_pd - with_pd).abs() > 0.5,
        "omitting the p-d term must change the hybrid dipole substantially"
    );
}

/// A charged molecule's dipole is origin-dependent, so it is only comparable once an origin is
/// stated. `pm7-rs` defaults to the centre of mass, which is what MOPAC uses, and MOPAC's
/// hydroxide dipole is the check.
#[test]
fn a_charged_species_uses_the_centre_of_mass_like_mopac() {
    let params = Pm7Parameters::standard().unwrap();
    let hydroxide = Molecule::new(vec![at(8, 0.0, 0.0, 0.0), at(1, 0.96, 0.0, 0.0)]);

    let default = run_pm7(&hydroxide, &params, &tight(-1.0)).unwrap();
    assert!(
        default.dipole.origin.norm() > 1.0e-6,
        "a charged molecule must be recentred; origin was {:?}",
        default.dipole.origin
    );

    // The pre-0.2.1 convention is still reachable, and it differs — which is the whole reason the
    // default changed.
    let mut options = tight(-1.0);
    options.dipole_origin = DipoleOrigin::Coordinates;
    let raw = run_pm7(&hydroxide, &params, &options).unwrap();
    assert!(
        (raw.dipole_magnitude - default.dipole_magnitude).abs() > 0.1,
        "the two origin conventions should disagree for an ion"
    );
}

/// A neutral molecule cannot see the origin setting at all — not even in the last bit.
#[test]
fn a_neutral_species_is_bit_identical_across_origins() {
    let params = Pm7Parameters::standard().unwrap();
    let water = Molecule::new(vec![
        at(8, 0.0, 0.0, 0.0),
        at(1, 0.96, 0.0, 0.0),
        at(1, -0.24, 0.93, 0.0),
    ]);
    let mut options = tight(0.0);
    let reference = run_pm7(&water, &params, &options).unwrap().dipole_debye;
    for choice in [
        DipoleOrigin::Coordinates,
        DipoleOrigin::CentreOfMass,
        DipoleOrigin::CentreOfCharge,
    ] {
        options.dipole_origin = choice;
        let other = run_pm7(&water, &params, &options).unwrap().dipole_debye;
        assert_eq!(reference.x.to_bits(), other.x.to_bits(), "{choice:?}");
        assert_eq!(reference.y.to_bits(), other.y.to_bits(), "{choice:?}");
        assert_eq!(reference.z.to_bits(), other.z.to_bits(), "{choice:?}");
    }
}
