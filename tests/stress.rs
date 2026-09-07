// SPDX-License-Identifier: GPL-3.0-or-later
//! The analytic periodic stress and gradient, checked against finite differences of the **full
//! SCF energy**.
//!
//! This is the strongest statement available about a derivative: not that the virial matches
//! another analytic expression, but that it matches what actually happens to the converged
//! energy when the cell is strained. Because the density is re-converged at every displaced
//! geometry, a missing Pulay term, a dropped background derivative, or an inconsistency between
//! the energy's pair enumeration and the gradient's would all show up here.

use pm7_rs::cell::Cell;
use pm7_rs::math::{Mat3, Vec3};
use pm7_rs::pbc::PbcOptions;
use pm7_rs::{analytic_stress, closed_form_gradient, run_pm7, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

fn options() -> Pm7Options {
    Pm7Options {
        // Tighten the SCF: a finite difference of a loosely converged energy is dominated by the
        // convergence noise, not by the derivative.
        e_tol: 1.0e-12,
        p_tol: 1.0e-10,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    }
}

fn energy(molecule: &Molecule, options: &Pm7Options) -> f64 {
    run_pm7(molecule, &params(), options).expect("SCF").total_ev
}

/// A 1-D polyethylene-like chain: CH2 repeating along x.
fn ch2_chain() -> Molecule {
    let a = a0();
    let period = 2.55 * a;
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, 0.89 * a),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, -0.89 * a),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(period, 0.0, 0.0)]).unwrap())
}

/// A 2-D hexagonal boron-nitride-like sheet (two atoms, one cell).
fn bn_sheet() -> Molecule {
    let a = 2.5 * a0();
    let cell = Cell::new(&[
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(a * 0.5, a * 0.866_025_403_784_44, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 5,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 7,
            position: Vec3::new(a * 0.5, a * 0.288_675_134_594_81, 0.0),
        },
    ])
    .with_cell(cell)
}

/// A 3-D cell with two water molecules — polar, so the Ewald terms are exercised properly.
fn water_crystal() -> Molecule {
    let a = 5.6 * a0();
    let cell = Cell::cubic(a).unwrap();
    let mut atoms = Vec::new();
    for (shift, flip) in [
        (Vec3::zero(), 1.0),
        (Vec3::new(a * 0.5, a * 0.5, a * 0.5), -1.0),
    ] {
        atoms.push(pm7_rs::Atom {
            z: 8,
            position: shift,
        });
        atoms.push(pm7_rs::Atom {
            z: 1,
            position: shift + Vec3::new(0.9584 * a0() * flip, 0.0, 0.0),
        });
        atoms.push(pm7_rs::Atom {
            z: 1,
            position: shift + Vec3::new(-0.2400 * a0() * flip, 0.9278 * a0(), 0.0),
        });
    }
    Molecule::new(atoms).with_cell(cell)
}

/// [`options`] with a Monkhorst–Pack mesh instead of the Γ point.
fn kpoint_options(n: [usize; 3]) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: pm7_rs::KMesh::MonkhorstPack {
                n,
                shift: [0.0; 3],
                gamma_centred: true,
            },
            ..PbcOptions::default()
        }),
        ..options()
    }
}

/// Compare the analytic stress with a central finite difference of the SCF energy under strain.
fn check_stress(molecule: &Molecule, label: &str, tolerance: f64) {
    check_stress_with(molecule, label, tolerance, &options());
}

fn check_stress_with(molecule: &Molecule, label: &str, tolerance: f64, opts: &Pm7Options) {
    let opts = opts.clone();
    let scf = run_pm7(molecule, &params(), &opts).expect("SCF");
    let out = analytic_stress(molecule, &params(), &opts, &scf).expect("stress");
    let cell = molecule.cell.unwrap();
    let dim = cell.dim();
    let h = 2.0e-4;

    for i in 0..dim {
        for j in 0..dim {
            let mut eps = Mat3::zero();
            // Symmetric strain, matching the symmetrized virial.
            eps.set(i, j, eps.get(i, j) + 0.5 * h);
            eps.set(j, i, eps.get(j, i) + 0.5 * h);
            let strained = |s: f64| -> f64 {
                let e = eps.scaled(s);
                let mut m = molecule.clone();
                m.cell = Some(cell.strained(&e).unwrap());
                for atom in &mut m.atoms {
                    atom.position += e.mul_vec(atom.position);
                }
                energy(&m, &opts)
            };
            let fd = (strained(1.0) - strained(-1.0)) / (2.0 * h) / out.measure;
            let got = out.stress.get(i, j);
            assert!(
                (got - fd).abs() < tolerance,
                "{label}: sigma[{i}][{j}] = {got:.9} vs finite difference {fd:.9} \
                 (difference {:.2e}, tolerance {tolerance:.1e})",
                (got - fd).abs()
            );
        }
    }
    // The off-diagonal components must be symmetric — an asymmetric stress would mean the cell
    // exerts a net torque on itself.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (out.stress.get(i, j) - out.stress.get(j, i)).abs() < 1e-12,
                "{label}: stress is not symmetric"
            );
        }
    }
}

/// Compare the analytic gradient with a central finite difference of the SCF energy.
fn check_gradient(molecule: &Molecule, label: &str, tolerance: f64) {
    check_gradient_with(molecule, label, tolerance, &options());
}

fn check_gradient_with(molecule: &Molecule, label: &str, tolerance: f64, opts: &Pm7Options) {
    let opts = opts.clone();
    let g = closed_form_gradient(molecule, &params(), &opts).expect("gradient");
    let h = 2.0e-4;
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let shifted = |s: f64| -> f64 {
                let mut m = molecule.clone();
                let mut d = Vec3::zero();
                match axis {
                    0 => d.x = s * h,
                    1 => d.y = s * h,
                    _ => d.z = s * h,
                }
                m.atoms[atom].position += d;
                energy(&m, &opts)
            };
            let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
            let got = g.gradient[atom].get(axis);
            assert!(
                (got - fd).abs() < tolerance,
                "{label}: grad[{atom}][{axis}] = {got:.9} vs finite difference {fd:.9} \
                 (difference {:.2e})",
                (got - fd).abs()
            );
        }
    }
}

#[test]
fn periodic_gradient_matches_finite_difference_in_one_dimension() {
    check_gradient(&ch2_chain(), "1-D CH2 chain", 2.0e-4);
}

#[test]
fn periodic_gradient_matches_finite_difference_in_two_dimensions() {
    check_gradient(&bn_sheet(), "2-D BN sheet", 2.0e-4);
}

#[test]
fn periodic_gradient_matches_finite_difference_in_three_dimensions() {
    check_gradient(&water_crystal(), "3-D water crystal", 3.0e-4);
}

// ---------------------------------------------------------------------------------------------
// k-point derivatives.
//
// These are not a variation on the Γ tests, they close a different hole. At Γ the density block
// `P(T)` is the same matrix for every translation, so a gradient that contracts everything
// against `P(0)` is right by accident. A k mesh makes `P(T)` decay, and the resonance and
// exchange terms have to see the block at the translation they act on — with the long-range
// exchange resolved by Born–von Kármán residue class on top. Getting that wrong left forces of
// ~19 eV/Å on a perfect diamond crystal and a pressure off by 436 GPa, neither of which any
// Γ-point test could see.
// ---------------------------------------------------------------------------------------------

#[test]
fn kpoint_gradient_matches_finite_difference_in_one_dimension() {
    check_gradient_with(
        &ch2_chain(),
        "1-D CH2 chain, 4x1x1",
        3.0e-4,
        &kpoint_options([4, 1, 1]),
    );
}

#[test]
fn kpoint_gradient_matches_finite_difference_in_two_dimensions() {
    check_gradient_with(
        &bn_sheet(),
        "2-D BN sheet, 3x3x1",
        3.0e-4,
        &kpoint_options([3, 3, 1]),
    );
}

#[test]
fn kpoint_gradient_matches_finite_difference_in_three_dimensions() {
    check_gradient_with(
        &water_crystal(),
        "3-D water crystal, 2x2x2",
        4.0e-4,
        &kpoint_options([2, 2, 2]),
    );
}

#[test]
fn kpoint_stress_matches_finite_difference_in_one_dimension() {
    check_stress_with(
        &ch2_chain(),
        "1-D CH2 chain, 4x1x1",
        3.0e-4,
        &kpoint_options([4, 1, 1]),
    );
}

#[test]
fn kpoint_stress_matches_finite_difference_in_two_dimensions() {
    check_stress_with(
        &bn_sheet(),
        "2-D BN sheet, 3x3x1",
        3.0e-4,
        &kpoint_options([3, 3, 1]),
    );
}

#[test]
fn kpoint_stress_matches_finite_difference_in_three_dimensions() {
    check_stress_with(
        &water_crystal(),
        "3-D water crystal, 2x2x2",
        5.0e-4,
        &kpoint_options([2, 2, 2]),
    );
}

/// The forces on a perfect crystal vanish by site symmetry — in **any** cell that describes it.
///
/// This is the test the k-point gradient bug walked straight through. The two-atom primitive cell
/// cannot show it (its own symmetry pins the forces to zero whatever the code does); doubling one
/// axis makes the displacement of a single atom a zone-boundary perturbation, and only a gradient
/// that resolves the density by translation gets it right. The mesh is chosen so the sampling of
/// the *primitive* Brillouin zone stays isotropic, because a mesh that breaks the cubic point
/// group breaks the symmetry argument along with it.
#[test]
fn a_perfect_crystal_feels_no_force_in_a_doubled_cell() {
    let a = 3.567 * a0();
    let prim = [
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ];
    // Double along the first vector; the basis atom and its image come along.
    let cell = Cell::new(&[prim[0] * 2.0, prim[1], prim[2]]).unwrap();
    let basis = Vec3::new(a * 0.25, a * 0.25, a * 0.25);
    let atoms = [Vec3::zero(), basis, prim[0], prim[0] + basis]
        .into_iter()
        .map(|position| pm7_rs::Atom { z: 6, position })
        .collect();
    let molecule = Molecule::new(atoms).with_cell(cell);

    let opts = kpoint_options([2, 4, 4]);
    let g = closed_form_gradient(&molecule, &params(), &opts).expect("gradient");
    assert!(
        g.max_gradient < 1.0e-9,
        "a perfect diamond lattice has a force of {:.3e} eV/Bohr on it; the gradient is not \
         seeing the density at the right translation",
        g.max_gradient
    );

    // The stress of a cubic crystal must also be isotropic and free of shear, whatever cell it
    // is described in.
    let scf = run_pm7(&molecule, &params(), &opts).expect("SCF");
    let out = analytic_stress(&molecule, &params(), &opts, &scf).expect("stress");
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { out.stress.get(0, 0) } else { 0.0 };
            assert!(
                (out.stress.get(i, j) - expected).abs() < 1.0e-9,
                "cubic crystal has an anisotropic stress: sigma[{i}][{j}] = {:.3e}",
                out.stress.get(i, j)
            );
        }
    }
}

#[test]
fn periodic_forces_sum_to_zero() {
    // Translational invariance: no net force on a periodic cell, in any dimension.
    for (molecule, label) in [
        (ch2_chain(), "1-D"),
        (bn_sheet(), "2-D"),
        (water_crystal(), "3-D"),
    ] {
        let g = closed_form_gradient(&molecule, &params(), &options()).expect("gradient");
        let sum = g.gradient.iter().fold(Vec3::zero(), |acc, v| acc + *v);
        assert!(
            sum.norm() < 1e-8,
            "{label}: net force {sum:?} on a periodic cell"
        );
    }
}

#[test]
fn periodic_stress_matches_finite_difference_in_one_dimension() {
    check_stress(&ch2_chain(), "1-D CH2 chain", 2.0e-5);
}

#[test]
fn periodic_stress_matches_finite_difference_in_two_dimensions() {
    check_stress(&bn_sheet(), "2-D BN sheet", 2.0e-5);
}

#[test]
fn periodic_stress_matches_finite_difference_in_three_dimensions() {
    check_stress(&water_crystal(), "3-D water crystal", 2.0e-5);
}

#[test]
fn charged_cell_stress_matches_finite_difference() {
    // The neutralizing background scales as 1/V and so contributes an isotropic stress. Leaving
    // it out leaves the forces right and the stress quietly wrong, which no force test catches.
    let mut mol = water_crystal();
    mol.charge = 1.0;
    let opts = Pm7Options {
        charge: 1.0,
        multiplicity: 2,
        e_tol: 1.0e-12,
        p_tol: 1.0e-10,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    let scf = run_pm7(&mol, &params(), &opts).expect("charged SCF");
    let out = analytic_stress(&mol, &params(), &opts, &scf).expect("stress");
    let cell = mol.cell.unwrap();
    let h = 2.0e-4;
    for i in 0..3 {
        let mut eps = Mat3::zero();
        eps.set(i, i, h);
        let strained = |s: f64| -> f64 {
            let e = eps.scaled(s);
            let mut m = mol.clone();
            m.cell = Some(cell.strained(&e).unwrap());
            for atom in &mut m.atoms {
                atom.position += e.mul_vec(atom.position);
            }
            energy(&m, &opts)
        };
        let fd = (strained(1.0) - strained(-1.0)) / (2.0 * h) / out.measure;
        let got = out.stress.get(i, i);
        assert!(
            (got - fd).abs() < 3.0e-5,
            "charged cell: sigma[{i}][{i}] = {got:.9} vs finite difference {fd:.9}"
        );
    }
}

#[test]
fn hydrostatic_pressure_matches_minus_de_by_dv() {
    // An independent route to the same physics: compress the cell isotropically and compare
    // −dE/dV with the trace of the stress. This checks the *scale* of the stress, which a
    // component-wise comparison against the same strain machinery could in principle share an
    // error with.
    let mol = water_crystal();
    let opts = options();
    let scf = run_pm7(&mol, &params(), &opts).expect("SCF");
    let out = analytic_stress(&mol, &params(), &opts, &scf).expect("stress");
    let cell = mol.cell.unwrap();
    let v0 = cell.volume().unwrap();

    let scaled = |s: f64| -> f64 {
        let mut m = mol.clone();
        m.cell = Some(cell.scaled(s).unwrap());
        for atom in &mut m.atoms {
            atom.position = atom.position * s;
        }
        energy(&m, &opts)
    };
    let ds = 1.0e-4;
    let dedscale = (scaled(1.0 + ds) - scaled(1.0 - ds)) / (2.0 * ds);
    // V = s³ V₀ ⇒ dV/ds = 3 V₀ at s = 1, so dE/dV = (dE/ds) / (3 V₀).
    let dedv = dedscale / (3.0 * v0);
    let pressure = out.pressure();
    assert!(
        (pressure + dedv).abs() < 2.0e-5,
        "P = {pressure:.9} eV/Bohr^3 but -dE/dV = {:.9}",
        -dedv
    );
    // And the GPa conversion must be self-consistent.
    let gpa = out.pressure_gpa(&mol).unwrap();
    assert!(
        (gpa - pressure * pm7_rs::stress::EV_PER_BOHR3_TO_GPA).abs() < 1e-9,
        "GPa conversion inconsistent"
    );
}

#[test]
fn stress_is_only_defined_for_a_periodic_system() {
    let mol = Molecule::from_xyz_str("2\nH2\nH 0 0 0\nH 0.74 0 0\n", 0.0).unwrap();
    let opts = Pm7Options::default();
    let scf = run_pm7(&mol, &params(), &opts).unwrap();
    assert!(analytic_stress(&mol, &params(), &opts, &scf).is_err());
}
