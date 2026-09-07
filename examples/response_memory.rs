// SPDX-License-Identifier: GPL-3.0-or-later
//! How much memory the perturbation solver retains, and how it scales.
//!
//! `cargo run --release --example response_memory`
//!
//! The response densities are `3N x n_k x nao^2` complex numbers, all held at once because the
//! assembly contracts every perturbation against every other. Whether that is worth restructuring
//! is a question about the actual number, not about the exponent: at the sizes the test suite
//! runs it is kilobytes, and the exponent alone would say to rewrite it anyway.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{Atom, Cell, KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters};

fn diamond_supercell(n: usize) -> Molecule {
    let a = 3.567 * ANGSTROM_TO_BOHR;
    let base = [
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ];
    let mut atoms = Vec::new();
    for image in 0..n {
        let shift = base[0] * image as f64;
        atoms.push(Atom {
            z: 6,
            position: shift,
        });
        atoms.push(Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25) + shift,
        });
    }
    let cell = Cell::new(&[base[0] * n as f64, base[1], base[2]]).unwrap();
    Molecule::new(atoms).with_cell(cell)
}

fn main() {
    let params = Pm7Parameters::standard().unwrap();
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>14} {:>14}",
        "atoms", "3N", "n_k", "nao", "responses MB", "bare MB"
    );
    for (cells, mesh) in [(1usize, 2usize), (1, 3), (2, 3), (3, 3), (4, 4), (6, 4)] {
        let molecule = diamond_supercell(cells);
        let nat = molecule.atoms.len();
        let basis = pm7_rs::basis::Basis::build(&molecule, &params).unwrap();
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(mesh, mesh, mesh),
                ..Default::default()
            }),
            ..Default::default()
        };
        // The response k set is the SCF's, unfolded: `k` and `k + q` are two origins of the same
        // zone, so time-reversal folding does not apply and every point costs.
        let cell = molecule.cell.unwrap();
        let pbc = options.pbc_for(&molecule).unwrap();
        let n_k = pbc.kmesh.expand_unfolded(&cell).unwrap().points.len();
        let ndof = 3 * nat;
        let nao = basis.nao;

        // `Vec<Vec<CMatrix>>`: ndof x n_k, each a complex nao x nao (two f64 arrays).
        let responses = ndof as f64 * n_k as f64 * 2.0 * (nao * nao) as f64 * 8.0;
        // The bare perturbation: ndof `ComplexBlocks` over the translation set.
        let translations = pm7_rs::hamiltonian::bvk_representatives(pbc.kmesh.divisions()).len();
        let bare = ndof as f64 * translations as f64 * 2.0 * (nao * nao) as f64 * 8.0;
        println!(
            "{nat:>6} {ndof:>6} {n_k:>6} {nao:>6} {:>14.1} {:>14.1}",
            responses / 1.048576e6,
            bare / 1.048576e6
        );
    }
    println!("\nBoth are held simultaneously; the assembly contracts each bare column against");
    println!("every response, so neither side can be dropped before the other is finished.");
    println!("`DfptOptions::response_block` caps the response side; the cost is that the phased");
    println!("columns are rebuilt once per block. Measured on a 4-atom chain, 2x2x2 mesh:\n");

    use std::time::Instant;
    let molecule = diamond_supercell(2);
    let options = Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(2, 2, 2),
            ..Default::default()
        }),
        ..Default::default()
    };
    let ndof = 3 * molecule.atoms.len();
    println!("{:>8} {:>14} {:>16}", "block", "responses MB", "seconds");
    for block in [Some(1usize), Some(3), Some(6), None] {
        let settings = pm7_rs::DfptOptions {
            response_block: block,
            ..Default::default()
        };
        let start = Instant::now();
        let out = pm7_rs::dynamical_matrix_dfpt(
            &molecule,
            &params,
            &options,
            [0.25, 0.0, 0.0],
            &settings,
        )
        .unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let held = block.unwrap_or(ndof).min(ndof);
        let basis = pm7_rs::basis::Basis::build(&molecule, &params).unwrap();
        let cell = molecule.cell.unwrap();
        let n_k = options
            .pbc_for(&molecule)
            .unwrap()
            .kmesh
            .expand_unfolded(&cell)
            .unwrap()
            .points
            .len();
        let mb = held as f64 * n_k as f64 * 2.0 * (basis.nao * basis.nao) as f64 * 8.0 / 1.048576e6;
        let label = block.map(|b| b.to_string()).unwrap_or_else(|| "all".into());
        println!("{label:>8} {mb:>14.2} {seconds:>16.3}");
        let _ = out.converged;
    }
}
