// SPDX-License-Identifier: GPL-3.0-or-later
//! SCF convergence on the cases that are genuinely hard.
//!
//! Two failure modes are pinned here, both found by running molecular dynamics rather than
//! single points — a trajectory has to converge at *every* step, so it samples geometries a
//! curated test set never reaches.
//!
//! 1. A Γ-point periodic cell couples every atom's charge to every other through the Madelung
//!    potential, and Γ-only sampling of a small cell also over-weights long-range exchange
//!    (`P(T)` does not decay at Γ). Together these can flatten the SCF surface enough that
//!    plain iteration oscillates and DIIS settles into a limit cycle around `1e-6` — converged
//!    for the energy, not converged by the density criterion. The level-shift controller in
//!    [`pm7_rs::scf`] exists for this.
//! 2. A hard *molecular* SCF can creep along a shallow valley for hundreds of iterations with
//!    a flat density step while the energy falls steadily. That must **not** be mistaken for a
//!    stall: shifting it only slows the descent. This file keeps both honest at once, because
//!    a controller tuned on either one alone breaks the other.

use pm7_rs::cell::Cell;
use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::params::Pm7Parameters;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::scf::{run_pm7, Pm7Options, Pm7Result};
use pm7_rs::system::{Atom, Molecule};

/// Primitive diamond-silicon cell doubled along **a**, as `ase.build.bulk("Si").repeat((2,1,1))`.
const ROWS: [[f64; 3]; 3] = [[0.0, 5.43, 5.43], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]];

/// A geometry 12 steps into a 300 K trajectory. Nothing is special about it beyond being the
/// first one the old SCF could not converge; the displacements are ~0.01 Å.
const RATTLED: [[f64; 3]; 4] = [
    [-0.005367078382250, 0.008993767458821, 0.012838173008614],
    [1.359256235542085, 1.329720965720638, 1.334153009317239],
    [-0.003632789715548, 2.743646977674292, 2.724142378738188],
    [1.364743632555712, 4.062638289146248, 4.073866438935957],
];

const PERFECT: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.3575, 1.3575, 1.3575],
    [0.0, 2.715, 2.715],
    [1.3575, 4.0725, 4.0725],
];

fn atoms(coords: &[[f64; 3]]) -> Vec<Atom> {
    coords
        .iter()
        .map(|c| Atom {
            z: 14,
            position: Vec3::new(c[0], c[1], c[2]) * ANGSTROM_TO_BOHR,
        })
        .collect()
}

fn silicon(coords: &[[f64; 3]]) -> Molecule {
    Molecule::new(atoms(coords)).with_cell(Cell::from_angstrom_rows(&ROWS).unwrap())
}

fn converge(molecule: &Molecule, options: &Pm7Options) -> Pm7Result {
    let params = Pm7Parameters::method(options.method).unwrap();
    run_pm7(molecule, &params, options).expect("SCF did not converge")
}

fn periodic(kmesh: KMesh) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

#[test]
fn a_rattled_gamma_point_cell_converges_within_the_default_iteration_budget() {
    let r = converge(&silicon(&RATTLED), &periodic(KMesh::Gamma));
    assert!(r.converged, "Γ-point SCF reported non-convergence");
    // The undistorted cell converges in single digits; this one needs the level shift and
    // lands around 120. The bound is the default `max_scf`, i.e. "it converges at all".
    assert!(
        r.iterations <= 200,
        "Γ-point SCF took {} iterations",
        r.iterations
    );
}

#[test]
fn the_undistorted_cell_still_converges_immediately() {
    // The stall controller must not slow down the easy case: nothing should fire here, because
    // this run finishes before two comparison windows have even accumulated.
    let r = converge(&silicon(&PERFECT), &periodic(KMesh::Gamma));
    assert!(r.converged);
    assert!(
        r.iterations <= 15,
        "the perfect cell regressed to {} iterations",
        r.iterations
    );
}

#[test]
fn k_point_sampling_makes_the_same_cell_easy() {
    // Γ-only exchange is what makes the rattled cell hard, so a k mesh should not merely
    // converge — it should converge *fast*. This is the numerical half of the advice in
    // `docs/pbc.md` to use k points for small cells.
    for n in [2usize, 3] {
        let mesh = KMesh::MonkhorstPack {
            n: [n, n, n],
            shift: [0.0; 3],
            gamma_centred: true,
        };
        let r = converge(&silicon(&RATTLED), &periodic(mesh));
        assert!(r.converged, "{n}x{n}x{n} SCF reported non-convergence");
        assert!(
            r.iterations <= 40,
            "{n}x{n}x{n} took {} iterations",
            r.iterations
        );
    }
}

#[test]
fn a_slow_molecular_descent_is_not_mistaken_for_a_stall() {
    // The same four silicon atoms with no cell: a dangling-bond cluster whose RHF singlet is a
    // poor description, so the SCF crawls downhill for ~275 iterations with a density step
    // pinned near 6e-4 while the energy falls by 0.45 eV. A stall controller that watches only
    // the step size shifts this run and never lets it finish.
    let options = Pm7Options {
        max_scf: 400,
        ..Pm7Options::default()
    };
    let r = converge(&Molecule::new(atoms(&RATTLED)), &options);
    assert!(r.converged, "molecular SCF reported non-convergence");
}
