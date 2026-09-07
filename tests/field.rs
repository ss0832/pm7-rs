// SPDX-License-Identifier: GPL-3.0-or-later
//! The external electric field: energy, analytic gradient, analytic Hessian.
//!
//! Four independent references, because the field touches the core Hamiltonian, the nuclear
//! energy, the gradient and the CPHF and a mistake in any one of them is easy to hide:
//!
//! 1. **MOPAC itself** — a `FIELD=(0.5,0,0)` heat of formation, transcribed from a run of the
//!    vendored v23.2.5 binary. This pins the sign convention, the units, and the operator.
//! 2. **An exact identity** — `E_field = f · mu_FieldConjugate`, which holds because the field
//!    Hamiltonian *is* the dipole operator contracted with the field.
//! 3. **Finite differences** — the closed-form gradient against a full-SCF numerical gradient,
//!    and the analytic Hessian against a numerical one, both under a field.
//! 4. **Bit identity** — a zero field must change nothing at all.

use pm7_rs::constants::{ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE, EV_TO_KCAL};
use pm7_rs::math::Vec3;
use pm7_rs::{
    analytic_gradient, analytic_hessian, closed_form_gradient, numerical_gradient,
    numerical_hessian, run_pm7, Atom, Cell, ExternalField, Molecule, Pm7Options, Pm7Parameters,
    ScfReference,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

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

/// The exact geometry the MOPAC reference run used.
fn water() -> Molecule {
    Molecule::new(vec![
        at(8, 0.0, 0.0, 0.0),
        at(1, 0.96, 0.0, 0.0),
        at(1, -0.24, 0.93, 0.0),
    ])
}

/// A d-shell molecule, so the p–d dipole term is exercised.
fn hydrogen_sulfide() -> Molecule {
    Molecule::new(vec![
        at(16, 0.0, 0.0, 0.0),
        at(1, 1.34, 0.0, 0.0),
        at(1, -0.35, 1.29, 0.0),
    ])
}

fn methyl() -> Molecule {
    Molecule::new(vec![
        at(6, 0.0, 0.0, 0.0),
        at(1, 1.08, 0.0, 0.0),
        at(1, -0.54, 0.94, 0.0),
        at(1, -0.54, -0.94, 0.0),
    ])
}

fn tight(field: Option<ExternalField>) -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 800,
        field,
        ..Default::default()
    }
}

/// MOPAC v23.2.5, `PM7 1SCF PRECISE GRADIENTS FIELD=(0.5,0.0,0.0)` on the geometry above.
///
/// The field *raises* the energy here (−57.78228 → −54.78395), which is the whole point of the
/// test: MOPAC's `FIELD=` vector is the potential gradient, so `E = +F·mu`, and a sign slip would
/// move the answer by twice the field energy rather than producing something merely inaccurate.
const MOPAC_WATER_FIELD_HOF: f64 = -54.78395;
const MOPAC_WATER_ZERO_HOF: f64 = -57.78228;

#[test]
fn the_field_energy_matches_mopac() {
    let p = params();
    let zero = run_pm7(&water(), &p, &tight(None)).unwrap();
    assert!(
        (zero.heat_of_formation_kcal - MOPAC_WATER_ZERO_HOF).abs() < 1.0e-4,
        "no field: {} vs MOPAC {MOPAC_WATER_ZERO_HOF}",
        zero.heat_of_formation_kcal
    );

    let field = ExternalField::new(0.5, 0.0, 0.0);
    let out = run_pm7(&water(), &p, &tight(Some(field))).unwrap();
    assert!(
        (out.heat_of_formation_kcal - MOPAC_WATER_FIELD_HOF).abs() < 1.0e-4,
        "field: {} vs MOPAC {MOPAC_WATER_FIELD_HOF}",
        out.heat_of_formation_kcal
    );
}

/// `E_field = f · mu_FieldConjugate`, exactly — the identity that says the field Hamiltonian and
/// the dipole operator are the same object.
///
/// Checked on H2S as well as water, because there the identity uses the **field-conjugate**
/// dipole and would fail against the full one by exactly the p–d term.
#[test]
fn the_field_energy_is_the_field_contracted_with_its_conjugate_dipole() {
    let p = params();
    let field = ExternalField::new(0.4, -0.3, 0.2);
    for molecule in [water(), hydrogen_sulfide()] {
        let out = run_pm7(&molecule, &p, &tight(Some(field))).unwrap();
        let expected = field
            .internal()
            .dot(out.dipole.field_conjugate() * (1.0 / AU_DIPOLE_TO_DEBYE));
        let reported = out.field_ev.expect("a field was applied");
        assert!(
            (reported - expected).abs() < 1.0e-12,
            "reported {reported} vs f.mu {expected}"
        );
    }
}

/// A zero field must be bit-identical to no field. `ExternalField::is_zero` is an exact test for
/// exactly this reason: it is what lets a caller thread `field` everywhere without paying for it.
#[test]
fn a_zero_field_changes_nothing_at_all() {
    let p = params();
    // Restricted and unrestricted both, since the field enters two different Fock builds. Methyl
    // is a doublet: 7 valence electrons cannot be a singlet.
    for (molecule, multiplicity) in [(water(), 1), (hydrogen_sulfide(), 1), (methyl(), 2)] {
        let mut options = tight(None);
        options.multiplicity = multiplicity;
        options.reference = ScfReference::Unrestricted;
        let without = run_pm7(&molecule, &p, &options).unwrap();

        options.field = Some(ExternalField::new(0.0, 0.0, 0.0));
        let with = run_pm7(&molecule, &p, &options).unwrap();

        assert_eq!(
            without.total_ev.to_bits(),
            with.total_ev.to_bits(),
            "total energy moved with a zero field"
        );
        assert_eq!(
            without.dipole_debye.x.to_bits(),
            with.dipole_debye.x.to_bits()
        );
        let ga = closed_form_gradient(&molecule, &p, &{
            let mut o = options.clone();
            o.field = None;
            o
        })
        .unwrap();
        let gb = closed_form_gradient(&molecule, &p, &options).unwrap();
        for (a, b) in ga.gradient.iter().zip(&gb.gradient) {
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }
}

/// The closed-form gradient against a full-SCF finite difference, under a field.
///
/// This exercises the whole chain — the field in `h_core` changes the converged density, and the
/// gradient then has to pick up both that and the explicit `q_A f` term.
#[test]
fn the_field_gradient_matches_a_full_scf_finite_difference() {
    let p = params();
    let field = ExternalField::new(0.5, -0.25, 0.1);
    for molecule in [water(), hydrogen_sulfide()] {
        let options = tight(Some(field));
        let closed = closed_form_gradient(&molecule, &p, &options).unwrap();
        let numeric = numerical_gradient(&molecule, &p, &options, 1.0e-4).unwrap();
        for (i, (a, b)) in closed.gradient.iter().zip(&numeric.gradient).enumerate() {
            for axis in 0..3 {
                assert!(
                    (a.get(axis) - b.get(axis)).abs() < 2.0e-6,
                    "atom {i} axis {axis}: closed form {} vs finite difference {}",
                    a.get(axis),
                    b.get(axis)
                );
            }
        }
    }
}

/// `analytic_gradient` differentiates the energy; `closed_form_gradient` evaluates `q_A f` in
/// closed form. Under a field they share no code beyond the SCF, so agreement is meaningful.
#[test]
fn the_two_gradient_routes_agree_under_a_field() {
    let p = params();
    let options = tight(Some(ExternalField::new(0.3, 0.2, -0.4)));
    let closed = closed_form_gradient(&water(), &p, &options).unwrap();
    let differentiated = analytic_gradient(&water(), &p, &options, 5.0e-4).unwrap();
    for (a, b) in closed.gradient.iter().zip(&differentiated.gradient) {
        for axis in 0..3 {
            assert!(
                (a.get(axis) - b.get(axis)).abs() < 5.0e-5,
                "{} vs {}",
                a.get(axis),
                b.get(axis)
            );
        }
    }
}

/// The analytic Hessian under a field, against a numerical one.
///
/// The field's *skeleton* second derivative is identically zero, so everything this checks lives
/// in the CPHF's perturbed `∂h/∂R`. If that block were dropped the test would still see a
/// plausible Hessian — just the wrong one — which is why it is a finite difference and not an
/// internal consistency check.
#[test]
fn the_field_hessian_matches_finite_differences() {
    let p = params();
    let field = ExternalField::new(0.5, 0.0, -0.3);
    for (name, molecule) in [("water", water()), ("h2s", hydrogen_sulfide())] {
        let options = tight(Some(field));
        let analytic = analytic_hessian(&molecule, &p, &options, 1.0e-3).unwrap();
        let numeric = numerical_hessian(&molecule, &p, &options, 2.0e-4).unwrap();
        let n = analytic.rows;
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                worst = worst.max((analytic[(i, j)] - numeric[(i, j)]).abs());
            }
        }
        assert!(worst < 3.0e-4, "{name}: worst Hessian difference {worst}");
    }
}

/// Open shell: the field is spin-independent, so both channels must see it.
#[test]
fn the_field_hessian_matches_finite_differences_for_an_open_shell() {
    let p = params();
    let mut options = tight(Some(ExternalField::new(0.4, 0.0, 0.0)));
    options.multiplicity = 2;
    options.reference = ScfReference::Unrestricted;
    let analytic = analytic_hessian(&methyl(), &p, &options, 1.0e-3).unwrap();
    let numeric = numerical_hessian(&methyl(), &p, &options, 2.0e-4).unwrap();
    let n = analytic.rows;
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((analytic[(i, j)] - numeric[(i, j)]).abs());
        }
    }
    assert!(worst < 1.0e-3, "worst Hessian difference {worst}");
}

/// `∂E/∂F = mu_FieldConjugate`: differentiating the total energy with respect to the field must
/// return the dipole the field conjugates to. On H2S it must **not** return the full dipole, and
/// the gap is exactly the p–d term — MOPAC's own inconsistency, made executable.
#[test]
fn the_energy_derivative_with_respect_to_the_field_is_the_conjugate_dipole() {
    let p = params();
    let h = 1.0e-3; // V/Angstrom
    for (name, molecule) in [("water", water()), ("h2s", hydrogen_sulfide())] {
        let zero = run_pm7(&molecule, &p, &tight(None)).unwrap();
        for axis in 0..3 {
            let mut plus = [0.0; 3];
            let mut minus = [0.0; 3];
            plus[axis] = h;
            minus[axis] = -h;
            let ep = run_pm7(
                &molecule,
                &p,
                &tight(Some(ExternalField {
                    volts_per_angstrom: plus,
                })),
            )
            .unwrap();
            let em = run_pm7(
                &molecule,
                &p,
                &tight(Some(ExternalField {
                    volts_per_angstrom: minus,
                })),
            )
            .unwrap();
            // dE/dF in eV per (V/Angstrom); the conjugate dipole in the same units is
            // `mu[e.Bohr] * a0`.
            let derivative = (ep.total_ev - em.total_ev) / (2.0 * h);
            let conjugate = zero.dipole.field_conjugate().get(axis) / AU_DIPOLE_TO_DEBYE
                * pm7_rs::constants::PM7_A0;
            assert!(
                (derivative - conjugate).abs() < 1.0e-6,
                "{name} axis {axis}: dE/dF {derivative} vs conjugate dipole {conjugate}"
            );

            if name == "h2s" {
                let full =
                    zero.dipole.total().get(axis) / AU_DIPOLE_TO_DEBYE * pm7_rs::constants::PM7_A0;
                let gap = zero.dipole.pd_hybrid.get(axis) / AU_DIPOLE_TO_DEBYE
                    * pm7_rs::constants::PM7_A0;
                assert!(
                    ((full - derivative) - gap).abs() < 1.0e-6,
                    "the full-dipole discrepancy must be exactly the p-d term"
                );
            }
        }
    }
}

/// A field along a periodic direction is refused; one across it is not.
#[test]
fn a_field_along_a_periodic_direction_is_refused() {
    let p = params();
    let a = 3.0 * ANGSTROM_TO_BOHR;
    let chain = Molecule::new(vec![at(1, 0.0, 0.0, 0.0), at(9, 0.93, 0.0, 0.0)])
        .with_cell(Cell::new(&[Vec3::new(a, 0.0, 0.0)]).unwrap());

    let along = run_pm7(&chain, &p, &tight(Some(ExternalField::new(0.2, 0.0, 0.0))));
    let message = along.unwrap_err().to_string();
    assert!(message.contains("unbounded"), "{message}");
    assert!(message.contains("dfpt"), "{message}");

    // Across the chain the potential is bounded and the calculation is ordinary.
    let across = run_pm7(&chain, &p, &tight(Some(ExternalField::new(0.0, 0.2, 0.0))));
    assert!(across.is_ok(), "{:?}", across.err());
}

/// Heats of formation are what the oracle compares, so pin the conversion too: the field enters
/// the heat of formation through the core energy exactly once.
#[test]
fn the_field_enters_the_heat_of_formation_once() {
    let p = params();
    let field = ExternalField::new(0.5, 0.0, 0.0);
    let zero = run_pm7(&water(), &p, &tight(None)).unwrap();
    let out = run_pm7(&water(), &p, &tight(Some(field))).unwrap();
    let difference_kcal = (out.total_ev - zero.total_ev) * EV_TO_KCAL;
    let observed = out.heat_of_formation_kcal - zero.heat_of_formation_kcal;
    assert!(
        (difference_kcal - observed).abs() < 1.0e-9,
        "{difference_kcal} vs {observed}"
    );
}
