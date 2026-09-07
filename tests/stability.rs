// SPDX-License-Identifier: GPL-3.0-or-later
//! The SCF stability analysis, against the one case whose answer is known independently.
//!
//! Every other test in this crate compares `pm7-rs` against `pm7-rs` or against MOPAC. This one
//! compares it against a textbook: restricted Hartree–Fock dissociates H₂ incorrectly, the failure
//! is a **triplet** (spin-symmetry-breaking) instability, and the correct dissociation limit is two
//! hydrogen atoms. All three are statements nobody had to run this program to know.

use pm7_rs::math::Vec3;
use pm7_rs::stability::ScfStability;
use pm7_rs::{run_pm7, Atom, Molecule, Pm7Method, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::method(Pm7Method::Pm7).expect("PM7 parameters")
}

fn options(stability: ScfStability) -> Pm7Options {
    Pm7Options {
        stability,
        p_tol: 1.0e-10,
        e_tol: 1.0e-11,
        max_scf: 400,
        ..Pm7Options::default()
    }
}

fn h2(r_angstrom: f64) -> Molecule {
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(r_angstrom * a0, 0.0, 0.0),
        },
    ])
}

/// H₂ is stable at equilibrium and triplet-unstable when stretched, and the onset is between.
#[test]
fn stretched_h2_is_triplet_unstable_and_equilibrium_h2_is_not() {
    let at_equilibrium = run_pm7(&h2(0.74), &params(), &options(ScfStability::Check))
        .expect("scf")
        .stability
        .expect("the analysis ran");
    assert!(
        at_equilibrium.lowest_ev > 0.0,
        "equilibrium H2 should be singlet-stable, got {:.4} eV",
        at_equilibrium.lowest_ev
    );
    let triplet = at_equilibrium.lowest_triplet_ev.expect("triplet channel");
    assert!(
        triplet > 0.0,
        "equilibrium H2 is a closed-shell minimum and must be triplet-stable; got {triplet:.4} eV"
    );

    // Stretched, and progressively more unstable. The monotonicity matters as much as the sign: a
    // detector that fired on everything would pass a sign test and fail this one.
    let mut previous = f64::INFINITY;
    for r in [1.5, 2.5, 4.0] {
        let stability = run_pm7(&h2(r), &params(), &options(ScfStability::Check))
            .expect("scf")
            .stability
            .expect("the analysis ran");
        let triplet = stability.lowest_triplet_ev.expect("triplet channel");
        assert!(
            triplet < 0.0,
            "H2 at {r} A must be triplet-unstable, got {triplet:.4} eV"
        );
        assert!(
            stability.lowest_ev > 0.0,
            "H2 at {r} A is unstable in the *spin* channel, not the singlet one; singlet came \
             back {:.4} eV",
            stability.lowest_ev
        );
        assert!(
            triplet < previous,
            "the instability must deepen as the bond stretches: {triplet:.4} at {r} A is not \
             below {previous:.4}"
        );
        previous = triplet;
    }
}

/// Following the instability reaches the correct dissociation limit: two hydrogen atoms.
///
/// This is the test that says the escape produces the *right* answer rather than merely a lower
/// one. PM7's heat of formation for a hydrogen atom is 52.102 kcal/mol, so a dissociated H₂ is
/// 104.204 — a number that comes from the parameter table and not from this calculation.
#[test]
fn following_the_instability_dissociates_h2_correctly() {
    let far = h2(4.0);
    let restricted = run_pm7(&far, &params(), &options(ScfStability::Off)).expect("scf");
    let escaped =
        run_pm7(&far, &params(), &options(ScfStability::Follow)).expect("scf + stability");

    let two_atoms = 2.0 * 52.102;
    let restricted_error = restricted.heat_of_formation_kcal - two_atoms;
    let escaped_error = escaped.heat_of_formation_kcal - two_atoms;
    assert!(
        restricted_error > 100.0,
        "restricted H2 at 4 A should sit far above two atoms; it is {restricted_error:.2} above"
    );
    assert!(
        escaped_error.abs() < 1.0,
        "the escaped solution should dissociate to two hydrogen atoms ({two_atoms:.3} kcal/mol); \
         it gives {:.3}, off by {escaped_error:.3}",
        escaped.heat_of_formation_kcal
    );
}

/// `⟨S²⟩` against two numbers that are not this program's to choose.
///
/// The first is MOPAC's, on the two open-shell cases where it can be read straight out of a
/// `(S**2)` line: CH₃ gives `0.753039` and O₂ `2.002664`, both at PM7/UHF. The second is a limit —
/// a broken-symmetry singlet made of two separated hydrogen atoms has `⟨S²⟩ = 1` exactly, which
/// follows from the two determinants it is an equal mixture of and from nothing in this crate. It
/// is the sharper of the two: it says the contamination is being *counted*, not merely computed.
#[test]
fn spin_contamination_matches_mopac_and_the_dissociation_limit() {
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;

    let methyl = Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.079 * a0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5395 * a0, 0.9345 * a0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.5395 * a0, -0.9345 * a0, 0.0),
        },
    ]);
    let doublet = Pm7Options {
        multiplicity: 2,
        ..options(ScfStability::Off)
    };
    let s2 = run_pm7(&methyl, &params(), &doublet)
        .expect("scf")
        .spin_squared()
        .expect("an unrestricted run reports <S^2>");
    assert!(
        (s2 - 0.753_039).abs() < 5.0e-6,
        "CH3 doublet: MOPAC prints <S^2> = 0.753039, this gives {s2:.6}"
    );

    let dioxygen = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 8,
            position: Vec3::new(1.21 * a0, 0.0, 0.0),
        },
    ]);
    let triplet = Pm7Options {
        multiplicity: 3,
        ..options(ScfStability::Off)
    };
    let s2 = run_pm7(&dioxygen, &params(), &triplet)
        .expect("scf")
        .spin_squared()
        .expect("an unrestricted run reports <S^2>");
    assert!(
        (s2 - 2.002_664).abs() < 5.0e-6,
        "O2 triplet: MOPAC prints <S^2> = 2.002664, this gives {s2:.6}"
    );

    // A restricted solution has no contamination to report, and reporting `S(S+1)` there would
    // make a tautology look like a measurement.
    let bound = run_pm7(&h2(0.74), &params(), &options(ScfStability::Follow)).expect("scf");
    assert!(!bound.unrestricted, "H2 at equilibrium stays restricted");
    assert!(
        bound.spin_squared().is_none(),
        "a restricted solution must report no <S^2>, not S(S+1)"
    );

    // Following the triplet instability to dissociation. The escape is progressive — half-broken at
    // 1.5 Å, essentially complete by 2.5 — and lands on exactly 1.
    let mut previous = 0.0;
    for (r, expected) in [(1.5, None), (2.5, None), (4.0, Some(1.0))] {
        let escaped = run_pm7(&h2(r), &params(), &options(ScfStability::Follow)).expect("scf");
        assert!(
            escaped.unrestricted,
            "H2 at {r} A is triplet-unstable, so `follow` must return an unrestricted solution"
        );
        let s2 = escaped.spin_squared().expect("<S^2>");
        assert!(
            s2 > previous,
            "contamination must grow as the bond stretches: {s2:.4} at {r} A is not above \
             {previous:.4}"
        );
        if let Some(limit) = expected {
            assert!(
                (s2 - limit).abs() < 1.0e-3,
                "two separated hydrogen atoms are <S^2> = 1 exactly; at {r} A this gives {s2:.6}"
            );
        }
        previous = s2;
    }
}

/// An ordinary closed shell is a minimum in both channels and is left exactly alone.
///
/// The property that makes the option safe to switch on: if the analysis moved a well-behaved
/// answer, it would be a bug rather than a feature.
#[test]
fn a_stable_molecule_is_not_moved() {
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let water = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.96 * a0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.24 * a0, 0.93 * a0, 0.0),
        },
    ]);
    let plain = run_pm7(&water, &params(), &options(ScfStability::Off)).expect("scf");
    let checked = run_pm7(&water, &params(), &options(ScfStability::Follow)).expect("scf");
    let stability = checked.stability.expect("the analysis ran");
    assert!(
        !stability.unstable,
        "water is a minimum; the analysis called it unstable (singlet {:.4}, triplet {:?})",
        stability.lowest_ev, stability.lowest_triplet_ev
    );
    assert_eq!(
        plain.heat_of_formation_kcal, checked.heat_of_formation_kcal,
        "a stable solution must come back bit-identical"
    );
}
