// SPDX-License-Identifier: GPL-3.0-or-later
//! Variable-cell optimization: the lattice as a degree of freedom.
//!
//! Through 0.2.2 `optimize` was L-BFGS over `3·nat` Cartesian coordinates and never mentioned
//! `src/stress.rs`, so a periodic run relaxed the atoms and reported success with whatever stress
//! the fixed cell implied. Measured on diamond at `a = 3.75 Å`, 2×2×2: `converged: true after 1
//! iterations`, max gradient `2e-14 eV/Bohr`, and **−29.6 GPa** of pressure. Both statements are
//! true, which is what makes it a trap rather than a bug — the atoms are at their minimum for that
//! cell, and the cell is 2 % too big.
//!
//! The only cell relaxation in the project was external, through ASE's `FrechetCellFilter`, which
//! covers ASE users and nobody else: not the Rust library, not either command line, not a non-ASE
//! Python caller.
//!
//! The tests here are identities rather than regression values. A relaxed cell is defined by its
//! stress vanishing, and two different starting cells that relax to the same structure must agree
//! — neither statement depends on what PM7 thinks a lattice constant is.

use pm7_rs::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM};
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{
    analytic_stress, optimize, Atom, Cell, Molecule, OptOptions, Pm7Options, Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().expect("PM7 parameters")
}

/// Diamond, with the lattice constant scaled so a run can start off its minimum.
fn diamond(a_angstrom: f64) -> Molecule {
    let a = a_angstrom * ANGSTROM_TO_BOHR;
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

/// A hydrogen chain along x, in a nine-number cell with only the first vector periodic.
fn chain(a_angstrom: f64) -> Molecule {
    let a = a_angstrom * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[Vec3::new(a, 0.0, 0.0)]).unwrap();
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.74 * ANGSTROM_TO_BOHR, 0.0, 0.0),
        },
    ])
    .with_cell(cell)
}

fn options(mesh: usize) -> Pm7Options {
    Pm7Options {
        p_tol: 1.0e-9,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn relaxing() -> OptOptions {
    OptOptions {
        relax_cell: true,
        max_iter: 60,
        ..OptOptions::default()
    }
}

fn measure_angstrom(molecule: &Molecule) -> f64 {
    let dim = molecule.cell.unwrap().dim() as i32;
    molecule.cell.unwrap().volume().unwrap() * BOHR_TO_ANGSTROM.powi(dim)
}

/// **A stretched cell and a compressed cell reach the same volume.**
///
/// The statement that makes this a cell optimizer rather than a cell perturber. Neither number is
/// pinned to a reference: the two runs approach the minimum from opposite sides and have to meet.
#[test]
fn a_stretched_and_a_compressed_cell_relax_to_the_same_volume() {
    let params = params();
    let scf = options(2);
    let opt = relaxing();

    let stretched = optimize(&diamond(3.75), &params, &scf, &opt).expect("stretched");
    let compressed = optimize(&diamond(3.40), &params, &scf, &opt).expect("compressed");

    let (a, b) = (
        measure_angstrom(&stretched.molecule),
        measure_angstrom(&compressed.molecule),
    );
    assert!(
        (a - b).abs() / a < 2.0e-3,
        "the two runs disagree on the volume: {a:.6} from 3.75 A, {b:.6} from 3.40 A"
    );
    // And both moved: a run that did nothing would also "agree".
    let start = measure_angstrom(&diamond(3.75));
    assert!(
        (a - start).abs() / start > 1.0e-3,
        "the stretched cell did not relax at all: {a:.6} against {start:.6}"
    );
    assert!(
        (stretched.energy_ev - compressed.energy_ev).abs() < 1.0e-3,
        "same structure, different energies: {} vs {}",
        stretched.energy_ev,
        compressed.energy_ev
    );
}

/// **A relaxed cell has no stress left**, which is the definition rather than a symptom.
///
/// Checked against `analytic_stress` on the returned structure — the same quantity the optimizer
/// drove on, but recomputed from the final geometry rather than carried over from the last step.
#[test]
fn the_free_stress_components_vanish_at_the_end() {
    let params = params();
    let scf = options(2);
    let out = optimize(&diamond(3.70), &params, &scf, &relaxing()).expect("relax");
    assert!(
        out.converged,
        "did not converge in {} steps",
        out.iterations
    );

    let result = pm7_rs::run_pm7(&out.molecule, &params, &scf).expect("SCF");
    let stress = analytic_stress(&out.molecule, &params, &scf, &result).expect("stress");
    let worst = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| stress.stress.get(a, b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 5.0e-5,
        "a converged variable-cell run left {worst:.3e} eV/Bohr^3 of stress standing"
    );
}

/// **A chain acquires no strain in the directions it does not repeat in.**
///
/// The 1-D cell has one lattice vector and therefore one strain degree of freedom. If the
/// generators were built over all six Voigt components instead of the periodic subspace, the
/// optimizer would try to strain vacuum — and `Cell` would either refuse or silently accept a
/// meaningless number.
#[test]
fn a_chain_relaxes_only_along_its_own_axis() {
    let params = params();
    let scf = Pm7Options {
        p_tol: 1.0e-9,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(4, 1, 1),
            ..Default::default()
        }),
        ..Default::default()
    };
    let start = chain(3.2);
    let out = optimize(&start, &params, &scf, &relaxing()).expect("relax the chain");

    let cell = out.molecule.cell.expect("still periodic");
    assert_eq!(cell.dim(), 1, "the chain stayed one-dimensional");
    let axis = cell.vectors()[0];
    assert!(
        axis.y.abs() < 1.0e-9 && axis.z.abs() < 1.0e-9,
        "the lattice vector rotated out of x: {axis:?}"
    );
    assert!(
        (axis.x - start.cell.unwrap().vectors()[0].x).abs() > 1.0e-6,
        "the chain's one free direction did not move"
    );
}

/// The default is still atoms-only, and it says so by leaving the cell exactly alone.
///
/// The opt-in was a decision, not an oversight: turning cell relaxation on by default would move
/// every published periodic `optimize` number.
#[test]
fn the_cell_is_fixed_unless_asked_for() {
    let params = params();
    let scf = options(2);
    let start = diamond(3.75);
    let out = optimize(&start, &params, &scf, &OptOptions::default()).expect("atoms only");
    assert_eq!(
        out.molecule.cell.unwrap(),
        start.cell.unwrap(),
        "an atoms-only optimization must not touch the lattice"
    );
    for step in &out.trajectory {
        assert_eq!(
            step.max_stress, 0.0,
            "no stress degree of freedom, no stress"
        );
    }
}

/// `relax_cell` on a molecule is refused, and the message says what to do.
///
/// The alternative — running atoms-only and reporting success — is the failure mode this release
/// keeps removing: a flag accepted and not honoured.
#[test]
fn relaxing_the_cell_of_a_molecule_is_refused() {
    let water = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    let error = optimize(&water, &params(), &Pm7Options::default(), &relaxing())
        .expect_err("a molecule has no lattice to relax");
    let text = error.to_string();
    assert!(
        text.contains("cell") && text.contains("drop the flag"),
        "the refusal should name the cell and say what to do: {text}"
    );
}

/// The trajectory records the cell, so a variable-cell run can be replayed.
#[test]
fn the_trajectory_carries_the_cell_at_every_step() {
    let params = params();
    let out = optimize(
        &diamond(3.70),
        &params,
        &options(2),
        &OptOptions {
            max_iter: 3,
            ..relaxing()
        },
    )
    .expect("a few steps");
    assert!(out.trajectory.len() > 1);
    let first = out.trajectory[0].cell.expect("periodic");
    let last = out.trajectory.last().unwrap().cell.expect("periodic");
    assert_ne!(
        first, last,
        "the recorded cell never changed, so the trajectory cannot be replayed"
    );
}
