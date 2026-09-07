// SPDX-License-Identifier: GPL-3.0-or-later
//! Blocking the response assembly changes the memory and nothing else.
//!
//! `D_{jj'} = Σ_k Tr[Δh^j(k)† ΔP^{j'}(k)]` pairs every bare column with every response, so neither
//! side can be released before the other is finished and both are `3N × n_k × nao²`. That grows as
//! `N³ n_k`: measured on diamond supercells at a 4×4×4 mesh it is 24 MB apiece at 8 atoms and 81 MB
//! at 12, which extrapolates to about 650 MB at 24 atoms and 5 GB at 48 — the size at which a
//! calculation stops being possible rather than merely slow.
//!
//! `DfptOptions::response_block` caps the response side. What has to be true is that it is a
//! **memory** knob and not a numerical one: the same sum, reassociated only in the order the outer
//! loop visits it.
//!
//! The default is `None`, and that has to stay bit-identical to the unblocked code, because
//! everything else in the crate is validated against it.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    dynamical_matrix_dfpt, Atom, Cell, DfptOptions, KMesh, Molecule, PbcOptions, Pm7Options,
    Pm7Parameters,
};

/// An `n`-cell diamond chain: enough degrees of freedom that a block size below `3N` is a
/// different code path rather than the same one under another name.
fn diamond_chain(n: usize) -> Molecule {
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
    Molecule::new(atoms).with_cell(Cell::new(&[base[0] * n as f64, base[1], base[2]]).unwrap())
}

fn options() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 500,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(2, 2, 2),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Every block size gives the same force constants.
///
/// Not bit-identical between *different* block sizes, and it should not be: a block boundary moves
/// where the `j'` accumulation starts, which reassociates a floating-point sum. What must hold is
/// that the difference stays at rounding — anything larger means a block boundary is dropping or
/// double-counting a term, which is the failure mode a partial sum invites.
#[test]
fn the_block_size_is_a_memory_knob_and_not_a_numerical_one() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = diamond_chain(2); // 4 atoms, 12 degrees of freedom
    let opts = options();
    let q = [0.25, 0.0, 0.0];

    let reference = dynamical_matrix_dfpt(&molecule, &params, &opts, q, &DfptOptions::default())
        .unwrap()
        .force_constants;
    let scale = (0..reference.n)
        .flat_map(|i| (0..reference.n).map(move |j| (i, j)))
        .map(|(i, j)| reference.get(i, j).0.abs())
        .fold(0.0, f64::max);

    for block in [1usize, 2, 5, 12, 64] {
        let settings = DfptOptions {
            response_block: Some(block),
            ..Default::default()
        };
        let got = dynamical_matrix_dfpt(&molecule, &params, &opts, q, &settings)
            .unwrap()
            .force_constants;
        let mut worst = 0.0_f64;
        for i in 0..reference.n {
            for j in 0..reference.n {
                let (ar, ai) = reference.get(i, j);
                let (br, bi) = got.get(i, j);
                worst = worst.max((ar - br).abs().max((ai - bi).abs()));
            }
        }
        assert!(
            worst < 1.0e-10 * scale.max(1.0),
            "block {block}: D(q) differs from the unblocked result by {worst:.3e} against a \
             largest element of {scale:.3e}. A block boundary is meant to reassociate the sum, \
             not to change which terms are in it."
        );
    }
}

/// The default is the unblocked path, bit for bit.
///
/// `response_block: None` has to be the code everything else in the crate is validated against —
/// a memory option that quietly perturbed every published number would be a poor trade.
#[test]
fn the_default_is_bit_identical_to_the_unblocked_path() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = diamond_chain(2);
    let opts = options();
    let ndof = 3 * molecule.atoms.len();

    let default = dynamical_matrix_dfpt(
        &molecule,
        &params,
        &opts,
        [0.25, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .unwrap()
    .force_constants;
    // One block covering every degree of freedom is the same traversal the default takes.
    let whole = dynamical_matrix_dfpt(
        &molecule,
        &params,
        &opts,
        [0.25, 0.0, 0.0],
        &DfptOptions {
            response_block: Some(ndof),
            ..Default::default()
        },
    )
    .unwrap()
    .force_constants;

    for i in 0..default.n {
        for j in 0..default.n {
            let (ar, ai) = default.get(i, j);
            let (br, bi) = whole.get(i, j);
            assert_eq!(
                (ar.to_bits(), ai.to_bits()),
                (br.to_bits(), bi.to_bits()),
                "element ({i},{j}) differs between `None` and one full block, which are the same \
                 traversal and must produce the same bits"
            );
        }
    }
}
