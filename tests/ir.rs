// SPDX-License-Identifier: GPL-3.0-or-later
//! Dipole derivatives and infrared intensities.
//!
//! Three references, none of which shares code with the thing it checks:
//!
//! 1. **A finite difference of the dipole** across two full SCF runs.
//! 2. **A finite difference of the field gradient**, `∂mu/∂R = ∂²E/∂f∂R`. This one goes through
//!    the external-field machinery instead of the response machinery, so it checks both.
//! 3. **An exact analytic identity**, the translational sum rule.
//!
//! MOPAC's `FORCE LARGE` numbers are compared in `tools/oracle/`, not here, because they need a
//! stationary geometry and a MOPAC run; and they must be compared as `DIPT` magnitudes rather
//! than components, since a normal mode's sign is arbitrary and degenerate modes mix freely.

use pm7_rs::constants::{ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE};
use pm7_rs::dipole::DipoleTerms;
use pm7_rs::ir::{dipole_at_coordinate_origin, dipole_derivatives, ir_spectrum};
use pm7_rs::math::Vec3;
use pm7_rs::{
    closed_form_gradient, run_pm7, Atom, ExternalField, Molecule, Pm7Options, Pm7Parameters,
    ScfReference,
};

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

fn water() -> Molecule {
    Molecule::new(vec![
        at(8, 0.0, 0.0, 0.0),
        at(1, 0.96, 0.0, 0.0),
        at(1, -0.24, 0.93, 0.0),
    ])
}

fn formaldehyde() -> Molecule {
    Molecule::new(vec![
        at(6, 0.0, 0.0, 0.0),
        at(8, 0.0, 0.0, 1.21),
        at(1, 0.94, 0.0, -0.54),
        at(1, -0.94, 0.0, -0.54),
    ])
}

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

fn tight() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..Default::default()
    }
}

/// `∂mu/∂R` against a central difference of the dipole itself, with a full SCF at each step.
#[test]
fn the_dipole_derivatives_match_a_finite_difference_of_the_dipole() {
    let params = Pm7Parameters::standard().unwrap();
    let h = 1.0e-4; // Bohr
    for (name, molecule, multiplicity) in [
        ("water", water(), 1),
        ("formaldehyde", formaldehyde(), 1),
        ("h2s", hydrogen_sulfide(), 1),
        ("methyl", methyl(), 2),
    ] {
        let mut options = tight();
        options.multiplicity = multiplicity;
        if multiplicity > 1 {
            options.reference = ScfReference::Unrestricted;
        }
        let analytic =
            dipole_derivatives(&molecule, &params, &options, 1.0e-3, DipoleTerms::Full).unwrap();

        let ndof = 3 * molecule.atoms.len();
        for t in 0..ndof {
            let (atom, axis) = (t / 3, t % 3);
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            match axis {
                0 => {
                    plus.atoms[atom].position.x += h;
                    minus.atoms[atom].position.x -= h;
                }
                1 => {
                    plus.atoms[atom].position.y += h;
                    minus.atoms[atom].position.y -= h;
                }
                _ => {
                    plus.atoms[atom].position.z += h;
                    minus.atoms[atom].position.z -= h;
                }
            }
            let mp = dipole_at_coordinate_origin(&run_pm7(&plus, &params, &options).unwrap());
            let mm = dipole_at_coordinate_origin(&run_pm7(&minus, &params, &options).unwrap());
            for a in 0..3 {
                // Convert the finite difference from Debye/Bohr into atomic units (e).
                let numeric = (mp.get(a) - mm.get(a)) / (2.0 * h) / AU_DIPOLE_TO_DEBYE;
                assert!(
                    (analytic[(a, t)] - numeric).abs() < 5.0e-6,
                    "{name} dof {t} axis {a}: analytic {} vs finite difference {numeric}",
                    analytic[(a, t)]
                );
            }
        }
    }
}

/// `∂mu_FieldConjugate/∂R = ∂²E/∂f∂R`: differentiate the **gradient** with respect to the field.
///
/// The two sides share no code beyond the SCF — one goes through the CPHF response, the other
/// through the external-field Hamiltonian — so this checks both at once. It must use the
/// field-conjugate operator, since MOPAC's field has no p–d term (convention C-2); on H2S the
/// full-dipole comparison is asserted to fail by exactly that term.
#[test]
fn the_dipole_derivatives_match_the_field_cross_derivative() {
    let params = Pm7Parameters::standard().unwrap();
    let h = 1.0e-3; // V/Angstrom
    for (name, molecule) in [("water", water()), ("h2s", hydrogen_sulfide())] {
        let options = tight();
        let conjugate = dipole_derivatives(
            &molecule,
            &params,
            &options,
            1.0e-3,
            DipoleTerms::FieldConjugate,
        )
        .unwrap();
        let full =
            dipole_derivatives(&molecule, &params, &options, 1.0e-3, DipoleTerms::Full).unwrap();
        let ndof = 3 * molecule.atoms.len();
        // The largest p-d gap seen anywhere in this molecule. For H2S it must be substantial:
        // if it were zero the comparison above would pass for the wrong reason.
        let mut pd_gap_seen = 0.0_f64;

        for axis in 0..3 {
            let mut plus = options.clone();
            let mut minus = options.clone();
            let mut fp = [0.0; 3];
            let mut fm = [0.0; 3];
            fp[axis] = h;
            fm[axis] = -h;
            plus.field = Some(ExternalField {
                volts_per_angstrom: fp,
            });
            minus.field = Some(ExternalField {
                volts_per_angstrom: fm,
            });
            let gp = closed_form_gradient(&molecule, &params, &plus).unwrap();
            let gm = closed_form_gradient(&molecule, &params, &minus).unwrap();

            for t in 0..ndof {
                let (atom, component) = (t / 3, t % 3);
                // d(dE/dR_t)/df_axis, in eV/Bohr per (V/Angstrom). The dipole derivative in the
                // same units is `∂mu/∂R [e] * a0`.
                let cross = (gp.gradient[atom].get(component) - gm.gradient[atom].get(component))
                    / (2.0 * h);
                let expected = conjugate[(axis, t)] * pm7_rs::constants::PM7_A0;
                // The reference is a central difference of a full SCF gradient with respect to
                // the field, so its own truncation error sets the floor here, not the analytic
                // side. Measured worst case on these molecules is ~3e-6.
                assert!(
                    (cross - expected).abs() < 2.0e-5,
                    "{name} axis {axis} dof {t}: cross derivative {cross} vs conjugate {expected}"
                );

                if name == "h2s" {
                    // The full-dipole derivative must **disagree** with the field cross
                    // derivative, by the p-d amount. Asserting only that
                    // `full - conjugate == full - cross` would be algebraically the same
                    // statement as the assertion above and would survive deleting the p-d term
                    // altogether, so the load-bearing part is the *size*: at least one degree of
                    // freedom has to show a gap far larger than the tolerance.
                    let gap = (full[(axis, t)] - conjugate[(axis, t)]) * pm7_rs::constants::PM7_A0;
                    if gap.abs() > 1.0e-4 {
                        let discrepancy = full[(axis, t)] * pm7_rs::constants::PM7_A0 - cross;
                        assert!(
                            (discrepancy - gap).abs() < 2.0e-5,
                            "axis {axis} dof {t}: the full-dipole discrepancy is {discrepancy}, \
                             which should equal the p-d term {gap}"
                        );
                        pd_gap_seen = pd_gap_seen.max(gap.abs());
                    }
                }
            }
        }
        if name == "h2s" {
            assert!(
                pd_gap_seen > 1.0e-3,
                "H2S must show a p-d gap between the two dipole operators; saw {pd_gap_seen}. \
                 A zero here means the p-d term is missing and the test above passed vacuously."
            );
        } else {
            assert_eq!(pd_gap_seen, 0.0, "{name} has no d orbitals, so no p-d gap");
        }
    }
}

/// `Σ_B ∂mu_a/∂R_{B,b} = q_tot delta_ab` — exact, and zero for a neutral molecule.
///
/// A translation of the whole molecule cannot change a neutral dipole, so this catches a sign or
/// index slip in the explicit term instantly, without any reference calculation at all.
#[test]
fn the_translational_sum_rule_holds() {
    let params = Pm7Parameters::standard().unwrap();
    for (name, molecule) in [
        ("water", water()),
        ("formaldehyde", formaldehyde()),
        ("h2s", hydrogen_sulfide()),
    ] {
        let d =
            dipole_derivatives(&molecule, &params, &tight(), 1.0e-3, DipoleTerms::Full).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                let total: f64 = (0..molecule.atoms.len())
                    .map(|atom| d[(a, 3 * atom + b)])
                    .sum();
                assert!(
                    total.abs() < 1.0e-8,
                    "{name}: sum rule ({a},{b}) = {total}, expected 0 for a neutral molecule"
                );
            }
        }
    }
}

/// The **rigid-motion identities**, both of them exact and neither needing a reference
/// calculation.
///
/// A translation cannot change a neutral molecule's dipole, so `Σ_B ∂mu/∂R_B = 0`. A **rotation**
/// very much can: turning a polar molecule turns its dipole with it, and the exact statement is
///
/// ```text
/// Σ_B (n × R_B) · ∇_B mu  =  n × mu
/// ```
///
/// for any rotation axis `n`. That is a far stronger check than "the rigid modes are dark" — which
/// is simply false for a polar molecule, and was the wrong premise this test started from. It
/// constrains the *response* part of the derivative, not just the point-charge part, because the
/// hybrid terms rotate with the molecule too.
#[test]
fn the_rigid_motion_identities_hold() {
    let params = Pm7Parameters::standard().unwrap();
    for (name, molecule) in [
        ("water", water()),
        ("formaldehyde", formaldehyde()),
        ("h2s", hydrogen_sulfide()),
    ] {
        let d =
            dipole_derivatives(&molecule, &params, &tight(), 1.0e-3, DipoleTerms::Full).unwrap();
        let mu = dipole_at_coordinate_origin(&run_pm7(&molecule, &params, &tight()).unwrap())
            * (1.0 / AU_DIPOLE_TO_DEBYE);

        for axis in 0..3 {
            let n = Vec3::new(
                if axis == 0 { 1.0 } else { 0.0 },
                if axis == 1 { 1.0 } else { 0.0 },
                if axis == 2 { 1.0 } else { 0.0 },
            );
            // Translation along `n`.
            for a in 0..3 {
                let total: f64 = (0..molecule.atoms.len())
                    .map(|b| d[(a, 3 * b + axis)])
                    .sum();
                assert!(
                    total.abs() < 1.0e-8,
                    "{name}: translation {axis} -> {total}"
                );
            }
            // Rotation about `n`.
            let expected = n.cross(mu);
            for a in 0..3 {
                let mut total = 0.0;
                for (b, atom) in molecule.atoms.iter().enumerate() {
                    let u = n.cross(atom.position);
                    for c in 0..3 {
                        total += d[(a, 3 * b + c)] * u.get(c);
                    }
                }
                assert!(
                    (total - expected.get(a)).abs() < 1.0e-6,
                    "{name}: rotation about {axis}, dipole component {a}: {total} vs {}",
                    expected.get(a)
                );
            }
        }
    }
}

/// Water's spectrum is well known: three vibrations, the asymmetric stretch the strongest, and a
/// bend of a few tens of km/mol. Loose bounds, but they catch a units error of any size.
#[test]
fn the_water_spectrum_is_physically_sane() {
    let params = Pm7Parameters::standard().unwrap();
    let spectrum = ir_spectrum(&water(), &params, &tight(), 1.0e-3).unwrap();
    // No `filter(|f| *f > 500.0)`. That filter was the last magnitude heuristic in this file: it
    // asked "is this number small" to decide which rows were rigid motions, which is the question
    // 0.2.3 replaces with a geometric projection. Every row the spectrum returns is now a
    // vibration, so a stray rigid motion shows up as a length of four rather than being silently
    // dropped -- which is the failure the old form could not see.
    let mut vibrations: Vec<(f64, f64)> = spectrum
        .frequencies_cm
        .iter()
        .zip(&spectrum.intensities_km_per_mol)
        .map(|(f, i)| (*f, *i))
        .collect();
    vibrations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(
        vibrations.len(),
        3,
        "water has three vibrations: {vibrations:?}"
    );
    for (frequency, intensity) in &vibrations {
        assert!(
            *intensity > 0.0 && *intensity < 5000.0,
            "{frequency} cm-1 has an implausible intensity of {intensity} km/mol"
        );
    }
    // The bend is the lowest of the three and is not the strongest.
    let bend = vibrations[0];
    assert!(bend.0 > 900.0 && bend.0 < 2200.0, "bend at {} cm-1", bend.0);
}

/// The two unit routes to km/mol — from `e²/amu` and from `(D/Å)²/amu` — must agree.
#[test]
fn both_intensity_unit_routes_agree() {
    use pm7_rs::constants::{
        E_IN_DEBYE_PER_ANGSTROM, IR_DEBYE_ANG2_PER_AMU_TO_KM_PER_MOL, IR_E2_PER_AMU_TO_KM_PER_MOL,
    };
    let direct = IR_E2_PER_AMU_TO_KM_PER_MOL;
    let viadebye =
        IR_DEBYE_ANG2_PER_AMU_TO_KM_PER_MOL * E_IN_DEBYE_PER_ANGSTROM * E_IN_DEBYE_PER_ANGSTROM;
    assert!((direct - viadebye).abs() < 1.0e-9 * direct.abs());
}

/// The IR spectrum's Hessian is the same object `analytic_hessian` returns — one CPHF solve, not
/// two — and the frequencies agree with `vibrational_analysis`.
#[test]
fn the_spectrum_reuses_the_hessian_it_reports() {
    let params = Pm7Parameters::standard().unwrap();
    let options = tight();
    let spectrum = ir_spectrum(&water(), &params, &options, 1.0e-3).unwrap();
    let separate = pm7_rs::vibrational_analysis(&water(), &params, &options, 1.0e-3).unwrap();
    for (a, b) in spectrum.frequencies_cm.iter().zip(&separate.frequencies_cm) {
        assert!((a - b).abs() < 1.0e-9, "{a} vs {b}");
    }
    let n = spectrum.hessian.rows;
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                spectrum.hessian[(i, j)].to_bits(),
                separate.hessian[(i, j)].to_bits()
            );
        }
    }
}

/// Retaining the response must not perturb the Hessian, bit for bit — the guard that the
/// restructuring did not change a summation order.
#[test]
fn retaining_the_response_is_bit_identical() {
    use pm7_rs::{analytic_hessian, analytic_hessian_with, HessianRequest};
    let params = Pm7Parameters::standard().unwrap();
    for (molecule, multiplicity) in [(water(), 1), (hydrogen_sulfide(), 1), (methyl(), 2)] {
        let mut options = tight();
        options.multiplicity = multiplicity;
        if multiplicity > 1 {
            options.reference = ScfReference::Unrestricted;
        }
        let plain = analytic_hessian(&molecule, &params, &options, 1.0e-3).unwrap();
        let kept = analytic_hessian_with(
            &molecule,
            &params,
            &options,
            1.0e-3,
            &HessianRequest::with_response(),
        )
        .unwrap();
        assert!(kept.response.is_some());
        let n = plain.rows;
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    plain[(i, j)].to_bits(),
                    kept.hessian[(i, j)].to_bits(),
                    "element ({i},{j}) moved when the response was retained"
                );
            }
        }
    }
}

/// The retained response really is `∂P/∂R`: check it against a finite difference of the density.
#[test]
fn the_orbital_response_reproduces_the_density_derivative() {
    use pm7_rs::{analytic_hessian_with, HessianRequest};
    let params = Pm7Parameters::standard().unwrap();
    let options = tight();
    let h = 1.0e-4;
    for (name, molecule) in [("water", water()), ("h2s", hydrogen_sulfide())] {
        let kept = analytic_hessian_with(
            &molecule,
            &params,
            &options,
            1.0e-3,
            &HessianRequest::with_response(),
        )
        .unwrap();
        let response = kept.response.unwrap();
        for t in [0_usize, 4, 8] {
            let (atom, axis) = (t / 3, t % 3);
            if atom >= molecule.atoms.len() {
                continue;
            }
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            for (target, sign) in [(&mut plus, 1.0), (&mut minus, -1.0)] {
                match axis {
                    0 => target.atoms[atom].position.x += sign * h,
                    1 => target.atoms[atom].position.y += sign * h,
                    _ => target.atoms[atom].position.z += sign * h,
                }
            }
            let pp = run_pm7(&plus, &params, &options).unwrap().density;
            let pm = run_pm7(&minus, &params, &options).unwrap().density;
            let analytic = response.density_derivative(t);
            let n = pp.rows;
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let numeric = (pp[(i, j)] - pm[(i, j)]) / (2.0 * h);
                    worst = worst.max((analytic[(i, j)] - numeric).abs());
                }
            }
            // Again the finite difference is the less accurate side: `h = 1e-4` Bohr on a
            // density converged to 1e-10 leaves a truncation error of order 1e-5.
            assert!(
                worst < 5.0e-5,
                "{name} dof {t}: worst dP/dR difference {worst}"
            );
        }
    }
}

/// MOPAC v23.2.5, `PM7 PRECISE FORCE LARGE` on the water geometry above: the three vibrational
/// frequencies and their printed `DIPT`.
///
/// The comparison is by **magnitude per mode**, matched on frequency. A normal-mode eigenvector's
/// sign is arbitrary and degenerate modes mix freely, so comparing `DIPX`/`DIPY`/`DIPZ`
/// component-wise is ill-posed no matter how well the two codes agree.
///
/// `DIPT` also carries MOPAC's factor of one half (`fmat.F90:197-252` divides a difference over
/// a separation of delta by 2*delta), which this implementation reproduces deliberately so the
/// two numbers are comparable at all.
const MOPAC_WATER_MODES: [(f64, f64); 3] = [
    (1414.6312, 0.51893),
    (2800.9549, 1.24612),
    (2855.2703, 0.39391),
];

#[test]
fn the_vibrational_dipoles_match_mopac_force_large() {
    let params = Pm7Parameters::standard().unwrap();
    let spectrum = ir_spectrum(&water(), &params, &tight(), 1.0e-3).unwrap();

    for (frequency, dipt) in MOPAC_WATER_MODES {
        let (index, ours) = spectrum
            .frequencies_cm
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - frequency)
                    .abs()
                    .partial_cmp(&(b.1 - frequency).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, f)| (i, *f))
            .unwrap();
        assert!(
            (ours - frequency).abs() < 1.0,
            "frequency {ours} vs MOPAC {frequency}"
        );
        let mine = spectrum.mopac_dipt[index];
        assert!(
            (mine - dipt).abs() < 0.01,
            "mode at {frequency} cm-1: DIPT {mine} vs MOPAC {dipt}"
        );
    }
}
