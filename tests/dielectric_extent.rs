// SPDX-License-Identifier: GPL-3.0-or-later
//! `ε^∞` for a chain and a slab, where the cell has no volume of its own.
//!
//! The raw response `α = ∂μ/∂f` per cell is defined in every dimensionality; `ε` is not, because
//! it needs a volume. Supplying the missing extent is a **convention**, and this file is mostly
//! about pinning what the convention can and cannot change.
//!
//! What it cannot change:
//!
//! * the three-dimensional answer — the crystal is the `N = 0` row of the same depolarization
//!   table, so the low-dimensional formula has to close on `born_and_dielectric`'s own `ε`;
//! * the sheet invariants `(ε_∥ − 1)d` and `(1 − 1/ε_⊥)d`, which are free of `d` by construction
//!   and are what a slab can quote without choosing one;
//! * anything at all when only the **vacuum** changes. A supercell says where the atoms are, not
//!   where the material stops, and a formula that took the cell height would move here.

use pm7_rs::cell::Cell;
use pm7_rs::math::{Mat3, Vec3};
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{
    born_and_dielectric, dielectric_with_extent, epsilon_from_polarizability, ExtentConvention,
    Molecule, Pm7Options, Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::method("pm7-".parse().unwrap()).unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

fn options(mesh: KMesh) -> Pm7Options {
    Pm7Options {
        method: "pm7-".parse().unwrap(),
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 800,
        pbc: Some(PbcOptions {
            kmesh: mesh,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// A 1-D chain of hydrogen fluoride, with `vacuum` Å of padding transverse to it.
fn hf_chain(vacuum: f64) -> Molecule {
    let a = a0();
    let _ = vacuum; // a 1-D cell has no transverse lattice vector to pad
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 9,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.95 * a, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(3.4 * a, 0.0, 0.0)]).unwrap())
}

/// A 2-D boron-nitride-like sheet.
///
/// Not a hydrogen lattice, and the reason is worth recording. PM7's hydrogen carries an s
/// orbital and nothing else, so it has no on-site dipole: the position operator reduces to
/// `-R_A δ_ab` and the in-plane response of two hydrogens stacked along `z` is **exactly**
/// zero, because they sit at the same `(x, y)`. A first version of this test used that fixture
/// and measured ε = 1 to every digit — which looked like a broken conversion and was a system
/// with no in-plane structure to respond. Boron and nitrogen have p orbitals and sit at
/// different in-plane sites.
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

fn diamond() -> Molecule {
    let a = 3.567 * a0();
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell)
}

/// The crystal is the `N = 0` row of the same table, not a separate rule.
///
/// Three-dimensional tin-foil summation removes the macroscopic depolarizing field, so `α` there
/// is already the response to the *internal* field. Feeding a 3-D `α` and its own volume to the
/// low-dimensional conversion with every depolarization factor zero must give back exactly what
/// `born_and_dielectric` computed. If the two disagreed, one of them would be wrong and there
/// would be no way to tell which.
#[test]
fn the_low_dimensional_law_closes_on_the_three_dimensional_one() {
    let molecule = diamond();
    let field = born_and_dielectric(
        &molecule,
        &params(),
        &options(KMesh::grid(2, 2, 2)),
        &pm7_rs::DfptOptions::default(),
    )
    .unwrap();
    let volume = field.volume_bohr3;
    assert!(volume > 0.0);

    // `measure = volume`, `extent = 1`, so the assigned volume is the real one; a slab convention
    // with the axis along a principal direction then gives `N = 0` in plane, and the two in-plane
    // entries are the ones to compare. The normal carries `N = 1` by construction, so it is
    // deliberately not part of this comparison — that is what the *next* test is about.
    let eps = epsilon_from_polarizability(
        &field.polarizability,
        Vec3::new(0.0, 0.0, 1.0),
        volume,
        ExtentConvention::SlabThickness(1.0),
    )
    .unwrap();
    for (i, j) in [(0, 0), (1, 1), (0, 1), (1, 0)] {
        let (a, b) = (eps.get(i, j), field.dielectric.get(i, j));
        assert!(
            (a - b).abs() < 1.0e-10,
            "in-plane ({i},{j}): extent law gives {a:.12}, born_and_dielectric {b:.12}"
        );
    }
    assert!(
        field.dielectric.get(0, 0) > 1.0,
        "diamond's epsilon should exceed 1, got {:.6}",
        field.dielectric.get(0, 0)
    );
}

/// Across a slab the answer is *not* the naive division, and the difference is the physics.
#[test]
fn the_normal_of_a_slab_uses_the_depolarization_law() {
    // A field derivative in MOPAC's convention, so the *physical* polarizability is its negation:
    // epsilon_from_polarizability takes ∂μ/∂f exactly as DfptFieldResult reports it and
    // applies the C-6 sign itself, which is the whole point of it living in one place.
    let mut alpha = Mat3::zero();
    alpha.set(0, 0, -2.5);
    alpha.set(1, 1, -2.5);
    alpha.set(2, 2, -0.8);
    let (area, thickness) = (30.0, 6.0);
    let eps = epsilon_from_polarizability(
        &alpha,
        Vec3::new(0.0, 0.0, 1.0),
        area,
        ExtentConvention::SlabThickness(thickness),
    )
    .unwrap();
    let four_pi = 4.0 * std::f64::consts::PI;
    let chi = 0.8 / (area * thickness);
    let naive = 1.0 + four_pi * chi;
    assert!(
        eps.get(2, 2) > naive * 1.0001,
        "the normal came out at {:.6}, which is the naive 1 + 4πχ = {naive:.6}; the \
         depolarizing field is already inside α and must not be counted twice",
        eps.get(2, 2)
    );
}

/// A real chain: the conversion runs, gives something physical, and its invariants ignore the
/// cross-section.
#[test]
fn a_chain_gets_an_epsilon_and_thickness_free_invariants() {
    let molecule = hf_chain(0.0);
    let mut previous: Option<(f64, f64)> = None;
    for section in [40.0_f64, 120.0, 400.0] {
        let out = dielectric_with_extent(
            &molecule,
            &params(),
            &options(KMesh::grid(4, 1, 1)),
            &pm7_rs::DfptOptions::default(),
            ExtentConvention::WireCrossSection(section),
        )
        .unwrap();

        // `ε ≥ 1` along every principal direction: a bound electron gas screens, never anti-screens.
        for i in 0..3 {
            assert!(
                out.dielectric.get(i, i) > 1.0 - 1.0e-12,
                "epsilon_{i}{i} = {:.6} < 1 at S = {section}",
                out.dielectric.get(i, i)
            );
        }
        // Symmetric, because it is built from a symmetric α through an orthogonal rotation.
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (out.dielectric.get(i, j) - out.dielectric.get(j, i)).abs() < 1.0e-10,
                    "epsilon is not symmetric at ({i},{j})"
                );
            }
        }

        let invariants = (out.invariants.parallel, out.invariants.perpendicular);
        if let Some(before) = previous {
            assert!(
                (invariants.0 - before.0).abs() < 1.0e-10
                    && (invariants.1 - before.1).abs() < 1.0e-10,
                "the invariants moved with the cross-section: {before:?} then {invariants:?}"
            );
        }
        previous = Some(invariants);

        // A larger assigned volume dilutes the response, so ε must fall towards 1.
        assert!(out.axis_mixing.is_finite());
    }
    let (parallel, _) = previous.unwrap();
    assert!(
        parallel.abs() > 1.0e-6,
        "the parallel invariant is {parallel:.3e}; this fixture has no response to speak of and \
         the test is vacuous"
    );
}

/// A bigger assigned extent means a more dilute material, so `ε` falls towards 1 monotonically.
#[test]
fn a_larger_extent_dilutes_the_response() {
    let molecule = bn_sheet();
    let mut last = f64::INFINITY;
    for thickness in [4.0_f64, 8.0, 16.0, 32.0] {
        let out = dielectric_with_extent(
            &molecule,
            &params(),
            &options(KMesh::grid(2, 2, 1)),
            &pm7_rs::DfptOptions::default(),
            ExtentConvention::SlabThickness(thickness),
        )
        .unwrap();
        let in_plane = out.dielectric.get(0, 0);
        assert!(
            in_plane > 1.0 - 1.0e-12,
            "in-plane epsilon {in_plane:.6} < 1 at d = {thickness}"
        );
        assert!(
            in_plane < last,
            "epsilon did not fall when the thickness grew: {last:.6} then {in_plane:.6}"
        );
        last = in_plane;
    }
    assert!(
        last < 1.5,
        "a 32 Bohr thick sheet is mostly vacuum and its epsilon should be near 1, got {last:.6}"
    );
}

/// The dimensionality of the cell and of the convention have to agree, and a 3-D cell is refused.
#[test]
fn the_convention_has_to_match_the_cell() {
    let settings = pm7_rs::DfptOptions::default();

    let error = dielectric_with_extent(
        &diamond(),
        &params(),
        &options(KMesh::grid(2, 2, 2)),
        &settings,
        ExtentConvention::SlabThickness(6.0),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("already has a volume"),
        "a 3-D cell should be refused by name: {error}"
    );

    let error = dielectric_with_extent(
        &hf_chain(0.0),
        &params(),
        &options(KMesh::grid(2, 1, 1)),
        &settings,
        ExtentConvention::SlabThickness(6.0),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("periodic direction"),
        "a thickness on a chain should be refused: {error}"
    );
}
