// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic systems built from d-bearing elements.
//!
//! The MNDO/d kernel and the periodic machinery were written at different times, and the places
//! they meet are the ones nothing else exercises: `has_any_d` routing inside the image-pair loop,
//! the 45×45 packed one-centre block under a Bloch sum, the 9-orbital electron–core blocks in
//! `H(T)`, and the `Dual2` d-path inside the periodic skeleton Hessian. A d-bearing *molecule*
//! test does not reach any of that, and neither does an s/p *crystal*.
//!
//! Silicon, sulfur, phosphorus, chlorine and zinc all carry d orbitals in MNDO/d.

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{
    analytic_stress, closed_form_gradient, force_constants, numerical_gradient, run_pm7, Atom,
    Molecule, Pm7Options, Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

fn options(mesh: KMesh) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: mesh,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// Silicon in the diamond structure: two d-bearing atoms per primitive cell, 18 AOs.
fn silicon() -> Molecule {
    let a = 5.431 * a0();
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 14,
            position: Vec3::zero(),
        },
        Atom {
            z: 14,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell)
}

/// A 1-D chain of H2S molecules: sulfur carries d, hydrogen does not, so the mixed sp/spd pair
/// path is exercised as well as the spd/spd one.
fn h2s_chain() -> Molecule {
    let a = a0();
    Molecule::new(vec![
        Atom {
            z: 16,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.96 * a, 0.86 * a, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.96 * a, 0.86 * a, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(4.5 * a, 0.0, 0.0)]).unwrap())
}

/// A 2-D sheet of ZnO in the flat hexagonal arrangement: two different d-bearing elements.
fn zno_sheet() -> Molecule {
    let a = 3.28 * a0();
    let cell = Cell::new(&[
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(-0.5 * a, 0.8660254 * a, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 30,
            position: Vec3::zero(),
        },
        Atom {
            z: 8,
            position: Vec3::new(0.0, a / 1.7320508, 0.0),
        },
    ])
    .with_cell(cell)
}

#[test]
fn a_d_bearing_crystal_runs_at_gamma_and_on_a_k_mesh() {
    let molecule = silicon();
    let mut previous = None;
    for mesh in [KMesh::Gamma, KMesh::grid(2, 2, 2), KMesh::grid(3, 3, 3)] {
        let label = format!("{mesh:?}");
        let result = run_pm7(&molecule, &params(), &options(mesh)).expect("silicon SCF");
        let per_atom = result.total_ev / molecule.atoms.len() as f64;
        assert!(
            per_atom.is_finite() && per_atom < 0.0,
            "silicon energy per atom is {per_atom}"
        );
        assert!(result.converged, "silicon SCF did not converge on {label}");
        previous = Some(per_atom);
    }
    // Γ over-counts the long-range exchange, so refining the mesh has to move the energy — if it
    // did not, the d path would be quietly ignoring the k dependence.
    let gamma = run_pm7(&molecule, &params(), &options(KMesh::Gamma))
        .unwrap()
        .total_ev
        / 2.0;
    assert!(
        (gamma - previous.unwrap()).abs() > 1.0e-3,
        "the k mesh changed nothing for a d-bearing crystal, which cannot be right"
    );
}

#[test]
fn the_d_path_analytic_gradient_matches_a_finite_difference_in_every_dimension() {
    for (name, molecule, mesh) in [
        (
            "H2S chain (1-D, mixed sp/spd)",
            h2s_chain(),
            KMesh::grid(2, 1, 1),
        ),
        (
            "ZnO sheet (2-D, two d elements)",
            zno_sheet(),
            KMesh::grid(2, 2, 1),
        ),
        ("silicon (3-D, spd/spd)", silicon(), KMesh::grid(2, 2, 2)),
    ] {
        let opts = options(mesh);
        // Displace so no atom sits on a symmetry point, where a vanishing gradient would let a
        // wrong one pass unnoticed.
        let mut shifted = molecule.clone();
        shifted.atoms[0].position += Vec3::new(0.03, -0.02, 0.015) * a0();
        let analytic = closed_form_gradient(&shifted, &params(), &opts)
            .unwrap_or_else(|e| panic!("{name}: analytic gradient failed: {e}"));
        let numerical = numerical_gradient(&shifted, &params(), &opts, 3.0e-3)
            .unwrap_or_else(|e| panic!("{name}: numerical gradient failed: {e}"));
        let mut worst = 0.0_f64;
        let mut scale = 1.0_f64;
        for (a, n) in analytic.gradient.iter().zip(&numerical.gradient) {
            for k in 0..3 {
                scale = scale.max(n.get(k).abs());
                worst = worst.max((a.get(k) - n.get(k)).abs());
            }
        }
        assert!(
            worst < 5.0e-3 * scale,
            "{name}: analytic gradient differs from a finite difference by {worst:.3e} \
             (values up to {scale:.3e})"
        );
    }
}

#[test]
fn the_d_path_analytic_stress_matches_a_strain_finite_difference() {
    let molecule = silicon();
    let opts = options(KMesh::grid(2, 2, 2));
    let scf = run_pm7(&molecule, &params(), &opts).expect("silicon SCF");
    let analytic = analytic_stress(&molecule, &params(), &opts, &scf).expect("silicon stress");

    // Isotropic strain: `Tr σ · Ω` is `dE/dε` for `ε = s·I`, and the energy route knows nothing
    // about how the virial is assembled.
    let cell = molecule.cell.unwrap();
    let volume = cell.measure();
    let h = 1.0e-3;
    let energy_at = |s: f64| -> f64 {
        let eps = pm7_rs::Mat3::from_rows(
            Vec3::new(s, 0.0, 0.0),
            Vec3::new(0.0, s, 0.0),
            Vec3::new(0.0, 0.0, s),
        );
        let mut strained = molecule.clone();
        strained.cell = Some(cell.strained(&eps).unwrap());
        for atom in &mut strained.atoms {
            atom.position += Vec3::new(
                s * atom.position.x,
                s * atom.position.y,
                s * atom.position.z,
            );
        }
        run_pm7(&strained, &params(), &opts)
            .expect("strained SCF")
            .total_ev
    };
    let de_deps = (energy_at(h) - energy_at(-h)) / (2.0 * h);
    let analytic_trace = analytic.stress.trace() * volume;
    assert!(
        (analytic_trace - de_deps).abs() < 2.0e-2 * de_deps.abs().max(1.0),
        "d-path stress trace {analytic_trace:.6} vs dE/de {de_deps:.6} eV"
    );
}

#[test]
fn the_d_path_periodic_force_constants_are_real_and_obey_the_acoustic_sum_rule() {
    let molecule = silicon();
    let opts = options(KMesh::Gamma);
    let mut fc =
        force_constants(&molecule, &params(), &opts, [2, 1, 1]).expect("silicon force constants");
    // The residual measures how well the d-path second derivatives respect translational
    // invariance; it is a property of the integrals, not of the projection.
    assert!(
        fc.acoustic_residual() < 1.0e-4,
        "silicon acoustic residual {:.3e} is too large for a correct d-path Hessian",
        fc.acoustic_residual()
    );
    fc.enforce_acoustic_sum_rule();
    let mut frequencies = fc.frequencies_cm([0.0, 0.0, 0.0]).expect("frequencies");
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for acoustic in &frequencies[..3] {
        assert!(
            acoustic.abs() < 1.0,
            "acoustic mode {acoustic:.3} cm^-1 should vanish after the projection"
        );
    }
    // Silicon's Raman line is 520 cm^-1; PM7 need not hit it, but a real optical branch must be
    // present and in the right decade rather than imaginary or absurd.
    let optical = frequencies[frequencies.len() - 1];
    assert!(
        (100.0..2000.0).contains(&optical),
        "silicon's optical mode came out at {optical:.1} cm^-1"
    );
}
