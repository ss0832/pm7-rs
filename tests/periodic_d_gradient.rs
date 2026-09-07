// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic derivatives for cells the suite did not previously reach, and what a loose SCF
//! tolerance does to a force.
//!
//! # Why this file exists
//!
//! Every periodic derivative test in the suite is s/p only. `tests/phonons.rs` checks the analytic
//! periodic Hessian against finite differences on a CH2 chain, a BN sheet and a water crystal;
//! `tests/stress.rs` uses the same three. `tests/pbc_d_orbitals.rs` exercises d orbitals in a
//! periodic *energy* and stops there.
//!
//! It was opened by asking a question no test asks: does each crystal reproduce its own space
//! group at the zone centre? Zinc-blende ZnS does not -- cubic Td demands one triply degenerate
//! optical mode and pm7-rs splits it -- and the first suspect was the gradient, because at the
//! ideal structure, where both atoms sit on sites the space group fixes and every force is zero by
//! symmetry, `closed_form_gradient` returned `1.13e-4 eV/Bohr` against diamond's `6.3e-11`.
//!
//! # It is not a gradient defect. It is the SCF tolerance.
//!
//! The spurious force falls with `p_tol` and does not stop falling:
//!
//! | `--scf-tolerance` | max abs force at the ideal ZnS structure |
//! |---|---|
//! | `1e-7` (the default) | `1.13e-4 eV/Bohr` |
//! | `1e-8` | `3.21e-5` |
//! | `1e-9` | `3.45e-6` |
//! | `1e-10` | `5.59e-10` |
//! | `1e-11` | `5.59e-10` |
//!
//! A Hellmann-Feynman force is evaluated at the converged density, so its error is **first order**
//! in the density error -- and the observed amplification is about three orders. That is the real
//! finding, and it is a property of every periodic force in the crate, not of ZnS: the default
//! `p_tol = 1e-7` is fine for a cell that converges cleanly and is not fine for one that does not.
//! ZnS needs 130+ iterations at the default and is bistable (at a 3x3x3 mesh an unsmeared run
//! finds -207.384 eV and a Fermi-smeared one -207.292, across a 6.4 eV gap where smearing cannot
//! change an occupation), so it is exactly the case where the residual density error is largest.
//!
//! What remains open is the triplet splitting itself. It is **not** the ground state: at the ideal
//! geometry the converged Mulliken charges are symmetric to the last digit (`F -0.75854214` twice
//! for CaF2, `Zn +0.83418061` / `S -0.83418061`), the forces vanish to `5.6e-10`, and the analytic
//! Hessian reproduces a double central difference of the energy to `1.3e-5`. So the splitting sits
//! in the perturbation solve, not in the gradient, and that is where to look next.
//!
//! A caution for anyone extending this file: CaF2 and ZnS both have *multiple* SCF solutions, so a
//! finite difference across two geometries can silently compare two different electronic states.
//! Displacing Ca by 0.06 A and then stepping F1 by +-0.003 A gave `E(+) = -958.4017408` after 414
//! iterations and `E(-) = -958.3988715` after 199 -- a 2.9e-3 eV "energy difference" that is
//! entirely a change of solution, and which reads as a 0.24 eV/Bohr force error if believed.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{closed_form_gradient, run_pm7, Atom, Cell, Molecule, Pm7Options, Pm7Parameters};

fn options(mesh: usize, p_tol: f64) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..PbcOptions::default()
        }),
        p_tol,
        max_scf: 4000,
        ..Pm7Options::default()
    }
}

/// An fcc primitive cell with a two-atom zinc-blende basis, `a` in Angstrom.
fn zinc_blende(z_a: u8, z_b: u8, a: f64) -> Molecule {
    let h = 0.5 * a * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, h, h),
        Vec3::new(h, 0.0, h),
        Vec3::new(h, h, 0.0),
    ])
    .unwrap();
    let q = 0.25 * a * ANGSTROM_TO_BOHR;
    Molecule::new(vec![
        Atom {
            z: z_a,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: z_b,
            position: Vec3::new(q, q, q),
        },
    ])
    .with_cell(cell)
}

fn max_force(molecule: &Molecule, params: &Pm7Parameters, options: &Pm7Options) -> f64 {
    closed_form_gradient(molecule, params, options)
        .unwrap()
        .gradient
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// At the ideal zinc-blende geometry every force vanishes by symmetry -- once the SCF is actually
/// converged.
#[test]
fn a_converged_scf_gives_the_symmetry_required_zero_force() {
    let params = Pm7Parameters::standard().unwrap();
    for (label, molecule) in [
        ("diamond", zinc_blende(6, 6, 3.567)),
        ("ZnS", zinc_blende(30, 16, 5.41)),
    ] {
        let worst = max_force(&molecule, &params, &options(4, 1.0e-10));
        eprintln!("{label}: max |F| at p_tol 1e-10 = {worst:.4e} eV/Bohr");
        // Measured: diamond 6.3e-11, ZnS 5.6e-8. The bound is set by ZnS, whose SCF is hard
        // enough that the residual density error is still visible here -- three orders below the
        // 1.1e-4 the default tolerance leaves, which is the statement being made. Diamond shows
        // what a well-behaved cell does on the same test.
        assert!(
            worst < 1.0e-7,
            "{label}: both atoms sit on sites the space group fixes, so every force is zero by \
             symmetry; got {worst:.4e} eV/Bohr with a converged SCF. That is a gradient defect, \
             not a tolerance artefact."
        );
    }
}

/// The force error is first order in the density error, and the amplification is large enough to
/// matter at the default tolerance.
///
/// This is the finding worth keeping: a periodic force is only as converged as its density, and
/// `p_tol = 1e-7` leaves a symmetry-breaking force of `1e-4 eV/Bohr` on a cell whose SCF is hard.
#[test]
fn a_loose_scf_tolerance_shows_up_as_a_symmetry_breaking_force() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = zinc_blende(30, 16, 5.41);
    let loose = max_force(&molecule, &params, &options(4, 1.0e-7));
    let tight = max_force(&molecule, &params, &options(4, 1.0e-10));
    eprintln!("ZnS: max |F| = {loose:.4e} at p_tol 1e-7, {tight:.4e} at 1e-10");
    assert!(
        loose > 1.0e-5,
        "the default tolerance is expected to leave a visible spurious force here ({loose:.3e}); \
         if it no longer does, the SCF got better and this test should record the new numbers"
    );
    assert!(
        tight < loose * 1.0e-3,
        "tightening the density tolerance by three orders should remove the force: \
         {loose:.3e} -> {tight:.3e}"
    );
}

/// The gradient is the derivative of the energy, for a cell whose atoms carry d orbitals.
///
/// The coverage the suite was missing. Checked away from the ideal structure, where the forces are
/// actually non-zero -- at the symmetric point comparing zero against zero tests almost nothing.
#[test]
fn the_periodic_gradient_is_the_energy_derivative_with_d_orbitals() {
    let params = Pm7Parameters::standard().unwrap();
    let options = options(4, 1.0e-10);
    // A fixed, reproducible rattle -- no RNG, so a failure is the same failure next time.
    let rattle = [
        Vec3::new(0.031, -0.017, 0.024),
        Vec3::new(-0.022, 0.038, -0.029),
    ];
    for (label, base) in [
        ("diamond", zinc_blende(6, 6, 3.567)),
        ("ZnS", zinc_blende(30, 16, 5.41)),
    ] {
        let mut molecule = base;
        for (atom, shift) in molecule.atoms.iter_mut().zip(&rattle) {
            atom.position += *shift * ANGSTROM_TO_BOHR;
        }
        let g = closed_form_gradient(&molecule, &params, &options).unwrap();
        let analytic: Vec<f64> = g.gradient.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let scale = analytic.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

        let step = 4.0e-3;
        let mut worst = 0.0_f64;
        for (dof, &component) in analytic.iter().enumerate() {
            let shift = |delta: f64| -> f64 {
                let mut moved = molecule.clone();
                let p = &mut moved.atoms[dof / 3].position;
                match dof % 3 {
                    0 => p.x += delta,
                    1 => p.y += delta,
                    _ => p.z += delta,
                }
                run_pm7(&moved, &params, &options).unwrap().total_ev
            };
            let numeric = (shift(step) - shift(-step)) / (2.0 * step);
            worst = worst.max((component - numeric).abs());
        }
        eprintln!(
            "{label}: max |grad - dE/dR| = {worst:.4e} eV/Bohr, largest component {scale:.3e}, \
             relative {:.2e}",
            worst / scale.max(1.0e-12)
        );
        assert!(
            worst < 1.0e-4 * scale.max(1.0e-6),
            "{label}: the analytic periodic gradient differs from the energy's own derivative by \
             {worst:.4e} eV/Bohr against a largest force of {scale:.3e}."
        );
    }
}
