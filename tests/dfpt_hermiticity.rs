// SPDX-License-Identifier: GPL-3.0-or-later
//! How Hermitian `D(q)` comes out, and why the threshold that guards it is where it is.
//!
//! `D(q)` is Hermitian by construction, so the departure measured before it is symmetrized is a
//! useful invariant: a phase error, a transposed index or a diverged response all break it badly.
//! What it is *not* is a fixed small number. The residual asymmetry is set by how well the
//! eigenvectors are determined, and inside a near-degenerate manifold that is not very well:
//! any unitary rotation of a degenerate block is an equally valid eigenbasis, so the response is
//! invariant under it algebraically and only to rounding numerically.
//!
//! Measured on two-atom rocksalt cells at one lattice constant, changing only the elements:
//!
//! | cell | smallest level splitting | asymmetry of `D(q)` |
//! |------|--------------------------|---------------------|
//! | NaCl | 1.4e-14 eV               | 3.5e-19             |
//! | CsI  | 2.7e-14 eV               | 5.8e-18             |
//! | MgO  | 9.5e-10 eV               | 7e-11               |
//! | ZnS  | 4.7e-10 eV               | 7e-9                |
//!
//! Nine orders of magnitude between the two groups, for the same code. A flat `1e-8` threshold
//! therefore refused correct answers -- a cubic CsPbI3 cell went past it outright, with a response
//! converged to 1e-10 and a value reproducible to three significant figures across four decades of
//! SCF tolerance, which is what a conditioning limit looks like and is not what a defect looks
//! like.
//!
//! These tests pin **both** ends. Relaxing the threshold alone would trade one blind spot for
//! another: the well-conditioned case is held to 1e-12, far tighter than the global threshold, and
//! that is where a construction error would show.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    dynamical_matrix_dfpt, Atom, Cell, DfptOptions, KMesh, Molecule, PbcOptions, Pm7Options,
    Pm7Parameters,
};

/// Two-atom rocksalt at a fixed lattice constant, so the only thing that varies is the elements.
fn rocksalt(z1: u8, z2: u8, a_ang: f64) -> Molecule {
    let a = a_ang * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: z1,
            position: Vec3::zero(),
        },
        Atom {
            z: z2,
            position: Vec3::new(a * 0.5, 0.0, 0.0),
        },
    ])
    .with_cell(cell)
}

/// `p_tol` is `1e-8`, not the `1e-9` the rest of the periodic tests use.
///
/// The near-degenerate cell this file exists to exercise is a hard SCF case for the same reason it
/// is a hard eigenvector case: levels a nanoelectronvolt apart make the density hard to pin down.
/// MgO at 5 A reaches 2.03e-8 after 800 iterations and stops. Tightening the SCF is not the fix --
/// it does not converge at all below about 1e-8, and this file's subject is what happens *given* a
/// converged density, not how tightly one can be had.
fn options(mesh: usize) -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-8,
        max_scf: 800,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn hermiticity_of(molecule: &Molecule, mesh: usize) -> f64 {
    let params = Pm7Parameters::standard().unwrap();
    dynamical_matrix_dfpt(
        molecule,
        &params,
        &options(mesh),
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("D(q) at the zone centre")
    .hermiticity
}

/// A well-separated spectrum gives a `D(q)` Hermitian to machine precision.
///
/// This is the tight end, and the one that matters: it is held to `1e-12`, six orders below the
/// threshold that guards the general case, because on a cell with no near-degeneracy there is
/// nothing to excuse a larger number. A construction error shows up here first.
#[test]
fn a_well_conditioned_cell_is_hermitian_to_machine_precision() {
    for (name, z1, z2) in [("NaCl", 11u8, 17u8), ("CsI", 55, 53), ("LiF", 3, 9)] {
        let value = hermiticity_of(&rocksalt(z1, z2, 5.0), 3);
        assert!(
            value < 1.0e-12,
            "{name}: D(q) departs from Hermiticity by {value:.3e}, and a cell whose levels are \
             separated has no conditioning excuse for it. This is where a phase error or a \
             transposed index surfaces."
        );
    }
}

/// A near-degenerate manifold does not, and that is not a defect.
///
/// The fixture assertion is the important half: if MgO ever becomes well conditioned -- a
/// parameter change, a different eigensolver -- this test stops testing anything, and saying so
/// out loud is cheaper than discovering it years later.
#[test]
fn a_near_degenerate_cell_is_not_refused_for_its_conditioning() {
    let value = hermiticity_of(&rocksalt(12, 8, 4.21), 3);
    assert!(
        value > 1.0e-14,
        "MgO no longer has the near-degenerate manifold this test exists to exercise (asymmetry \
         {value:.3e}); find another cell whose level splittings are ~1e-9 eV, or this is now a \
         duplicate of the well-conditioned test."
    );
    assert!(
        value < 1.0e-6,
        "MgO's asymmetry {value:.3e} is past the threshold; conditioning explains ~1e-10, so this \
         is something else."
    );
}

/// The reported number is a measurement, not a constant.
///
/// A diagnostic field that always carries the same value tells nobody anything, and one filled
/// with a placeholder is worse than absent because it reads as information. Two cells that differ
/// by orders of magnitude are the cheapest possible proof that this one is real.
#[test]
fn the_reported_hermiticity_varies_with_the_system() {
    let clean = hermiticity_of(&rocksalt(11, 17, 5.0), 3);
    let conditioned = hermiticity_of(&rocksalt(12, 8, 4.21), 3);
    assert!(
        conditioned > 1.0e3 * clean.max(f64::MIN_POSITIVE),
        "NaCl reports {clean:.3e} and MgO {conditioned:.3e}; these should differ by orders of \
         magnitude, and a field that reports the same number for both is not measuring anything."
    );
}
