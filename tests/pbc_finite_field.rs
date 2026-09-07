// SPDX-License-Identifier: GPL-3.0-or-later
//! A finite field along a periodic direction, and the polarizability it gives against CPHF's.
//!
//! `α = Ω ∂P/∂𝓔` is reachable two ways: by linear response (`polarizability`, a CPHF solve) and by
//! finite-differencing a self-consistent calculation in an actual field. The two share the SCF and
//! nothing else — one solves a linear equation for the response, the other minimizes a different
//! functional and never linearizes anything.
//!
//! That comparison is the only thing that can check the coupling factor. The derivation in
//! `pbc::finite_field` fixes a `λ = (𝓔·a) J / 4π` and Hermitizes as `M + M†` rather than
//! `½(M + M†)`, and **neither choice fails visibly**: a wrong factor of two, a missing `J` or a
//! flipped sign all give a converged calculation and a plausible polarizability. Only the number
//! says which.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::finite_field::{run_finite_field, FiniteFieldOptions};
use pm7_rs::{
    polarizability, Atom, Cell, DfptOptions, KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters,
};

/// A hydrogen crystal: two atoms, no d orbitals, and a wide gap, so the SCF is easy in a field.
fn hydrogen(a_ang: f64) -> Molecule {
    let a = a_ang * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, a),
    ])
    .unwrap();
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

fn options() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 500,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(4, 1, 1),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A zero field reproduces the ordinary polarization, and converges immediately.
#[test]
fn a_zero_field_is_the_field_free_state() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hydrogen(6.0);
    let out = run_finite_field(
        &molecule,
        &params,
        &options(),
        [4, 1, 1],
        Vec3::zero(),
        &FiniteFieldOptions::default(),
    )
    .unwrap();
    assert!(out.converged);
    // The phase is still computed on every axis the mesh resolves, even with no field to couple
    // to. Reporting zero there instead would be a different claim than "unresolved".
    assert!(out.resolved[0], "the 4-point axis should be resolved");
    assert!(
        !out.resolved[1] && !out.resolved[2],
        "a single-point axis cannot carry a phase and must say so rather than report zero"
    );
    assert!(
        out.electronic_polarization.norm().is_finite(),
        "the field-free polarization must still be a number"
    );
}

/// A field along a periodic direction and one orthogonal to every lattice vector are two different
/// treatments, and asking for both at once is refused.
#[test]
fn the_two_field_treatments_refuse_to_be_combined() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hydrogen(6.0);
    let mut opts = options();
    opts.field = Some(pm7_rs::ExternalField::new(0.0, 0.1, 0.0));
    let err = run_finite_field(
        &molecule,
        &params,
        &opts,
        [4, 1, 1],
        Vec3::new(1.0e-4, 0.0, 0.0),
        &FiniteFieldOptions::default(),
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("only") || message.contains("two treatments"),
        "the refusal should say why, got: {message}"
    );
}

/// A string of two points cannot resolve a winding, and that is refused rather than averaged.
#[test]
fn a_field_along_an_unresolvable_axis_is_refused() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hydrogen(6.0);
    let err = run_finite_field(
        &molecule,
        &params,
        &options(),
        [2, 1, 1],
        Vec3::new(1.0e-4, 0.0, 0.0),
        &FiniteFieldOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("at least 3"), "got: {err}");
}

/// `Ω ∂P/∂𝓔` from a finite field reproduces the CPHF polarizability.
///
/// The independent-route check, and the only thing that pins the coupling factor. A central
/// difference in the field, so the leading error is `O(𝓔²)` and the field can be small enough to
/// stay linear without the difference drowning in it.
#[test]
fn the_finite_field_polarizability_agrees_with_the_cphf_one() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hydrogen(6.0);
    let opts = options();
    let ff = FiniteFieldOptions::default();
    let field = 2.0e-4;

    let plus = run_finite_field(
        &molecule,
        &params,
        &opts,
        [4, 1, 1],
        Vec3::new(field, 0.0, 0.0),
        &ff,
    )
    .unwrap();
    let minus = run_finite_field(
        &molecule,
        &params,
        &opts,
        [4, 1, 1],
        Vec3::new(-field, 0.0, 0.0),
        &ff,
    )
    .unwrap();

    let volume = molecule.cell.unwrap().measure();
    let finite = volume * (plus.polarization.x - minus.polarization.x) / (2.0 * field);
    let cphf = polarizability(&molecule, &params, &opts, &DfptOptions::default())
        .unwrap()
        .get(0, 0);

    // The CPHF tensor is in the MOPAC `FIELD=` convention (C-1), whose sign is opposite to the
    // physical polarizability the enthalpy route produces.
    let cphf = -cphf;
    let ratio = finite / cphf;
    println!("alpha_xx: finite field {finite:.6}, CPHF {cphf:.6}, ratio {ratio:.4}");
    println!(
        "  plus:  P = {:?}  iters {}",
        plus.polarization, plus.iterations
    );
    println!(
        "  minus: P = {:?}  iters {}",
        minus.polarization, minus.iterations
    );
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "finite-field alpha_xx = {finite:.6} against CPHF {cphf:.6}, ratio {ratio:.4}. These are \
         independent routes to the same derivative. A ratio near 0.5 is the Hermitization: \
         `M + M^dagger` was replaced by half of it, which halves the occupied-virtual block the \
         response is made of. A ratio near 2, or off by the string length, is the `lambda` factor."
    );
}
