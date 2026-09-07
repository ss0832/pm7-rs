// SPDX-License-Identifier: GPL-3.0-or-later
//! Where the Born-charge path's time actually goes.
//!
//! `PM7_PROFILE=1 cargo run --release --example born_profile`
//!
//! Decomposed the way it has to be to be answerable: the SCF, then the field response alone
//! (three perturbations, no nuclear bare terms), then the full field result, then a phonon solve
//! at the same geometry (`3N` perturbations). Per-solve costs come out of the ratio, and a field
//! solve costing many nuclear solves would say the parallelism or the convergence is wrong rather
//! than the arithmetic.

use std::time::Instant;

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    born_and_dielectric, dynamical_matrix_dfpt, run_pm7, Atom, Cell, DfptOptions, KMesh, Molecule,
    PbcOptions, Pm7Options, Pm7Parameters,
};

fn lif(cells: usize) -> Molecule {
    let a = 4.03 * ANGSTROM_TO_BOHR;
    let base = [
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ];
    let mut atoms = Vec::new();
    for image in 0..cells {
        let shift = base[0] * image as f64;
        atoms.push(Atom {
            z: 3,
            position: shift,
        });
        atoms.push(Atom {
            z: 9,
            position: Vec3::new(a * 0.5, 0.0, 0.0) + shift,
        });
    }
    let cell = Cell::new(&[base[0] * cells as f64, base[1], base[2]]).unwrap();
    Molecule::new(atoms).with_cell(cell)
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64())
}

fn main() {
    let params = Pm7Parameters::standard().unwrap();
    println!(
        "{:>6} {:>8} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "cells", "atoms", "scf s", "born s", "phonon s", "per field", "per nuclear"
    );
    for cells in [1usize, 2, 3] {
        let molecule = lif(cells);
        let nat = molecule.atoms.len();
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(3, 3, 3),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dfpt = DfptOptions::default();

        let (_, scf_s) = time(|| run_pm7(&molecule, &params, &options).unwrap());
        // The field result carries `3N` nuclear solves *and* three field solves, because the Born
        // charge is the cross term between them.
        let (_, born_s) =
            time(|| born_and_dielectric(&molecule, &params, &options, &dfpt).unwrap());
        // A phonon solve at the same geometry is the `3N` nuclear half on its own.
        let (_, phonon_s) =
            time(|| dynamical_matrix_dfpt(&molecule, &params, &options, [0.0; 3], &dfpt).unwrap());

        let nuclear = (phonon_s - scf_s).max(0.0);
        let field = (born_s - phonon_s).max(0.0);
        println!(
            "{cells:>6} {nat:>8} {scf_s:>10.3} {born_s:>12.3} {phonon_s:>12.3} {:>12.4} {:>12.4}",
            field / 3.0,
            nuclear / (3 * nat) as f64
        );
    }
    if std::env::var_os("PM7_PROFILE").is_some() {
        println!();
        pm7_rs::profile::report_and_reset();
    } else {
        println!("\n(set PM7_PROFILE=1 for the stage breakdown)");
    }
}
