// SPDX-License-Identifier: GPL-3.0-or-later
//! What `D(q)` does as `q → 0`, and what the k mesh has to say about it.
//!
//! The acoustic sum rule `Σ_B Φ_{Aα,Bβ} = 0` is exact at the zone centre — a rigid translation
//! costs no energy. Away from it there is no such identity, but `Σ_B Φ_{Aα,Bβ}(q)` must still
//! *approach* zero, because `Φ(q)` is a Bloch sum of the same real-space blocks and its `q → 0`
//! limit is the zone-centre matrix plus, for a polar crystal, the non-analytic term — and that
//! term's own `Σ_B` vanishes by `Σ_B Z*_B = 0`.
//!
//! So a residue that does **not** go away is a defect. This file pins that it does, and pins the
//! condition under which it does: `q` has to be something the k mesh can represent.
//!
//! # The trap this exists to document
//!
//! An `n × n × n` mesh samples wavevectors in steps of `1/n`. Ask for `q ≪ 1/n` and the response
//! couples `k` with a `k + q` the sampling cannot distinguish from `k`, and the answer stops
//! meaning anything — while still reporting `converged = true`, because the linear solve did
//! converge; it converged to the answer for a question the mesh could not pose. Measured on LiF
//! at `q = 0.00625` (against a mesh step of `1/3`, `1/5`, `1/7`):
//!
//! | mesh | `max |Σ_B Φ(q)|` |
//! |---|---|
//! | 3³ | 2.96e-1 |
//! | 5³ | 6.75e-2 |
//! | 7³ | 3.21e-2 |
//!
//! It is a sampling limit, not a bug in the construction, and the identities below hold to
//! machine precision at every mesh. But nothing in the API says so, which is why it is written
//! down here and in `docs/pbc.md`.

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{born_and_dielectric, dynamical_matrix_dfpt, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().expect("PM7 parameters")
}

fn rocksalt(z_a: u8, z_b: u8, a_ang: f64) -> Molecule {
    let a = a_ang * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: z_a,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: z_b,
            position: Vec3::new(a * 0.5, 0.0, 0.0),
        },
    ])
    .with_cell(cell)
}

fn diamond() -> Molecule {
    let a = 3.567 * pm7_rs::constants::ANGSTROM_TO_BOHR;
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

fn options(mesh: usize) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// `max_{A,α,β} |Σ_B M_{Aα,Bβ}|` for a complex `3N × 3N` matrix.
fn acoustic_residual(m: &pm7_rs::cmatrix::CMatrix) -> f64 {
    let nat = m.n / 3;
    let mut worst = 0.0_f64;
    for a in 0..nat {
        for alpha in 0..3 {
            for beta in 0..3 {
                let (mut re, mut im) = (0.0, 0.0);
                for b in 0..nat {
                    let (r, i) = m.get(3 * a + alpha, 3 * b + beta);
                    re += r;
                    im += i;
                }
                worst = worst.max(re.hypot(im));
            }
        }
    }
    worst
}

/// The largest element, to judge a residual against.
fn scale(m: &pm7_rs::cmatrix::CMatrix) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..m.n {
        for j in 0..m.n {
            let (re, im) = m.get(i, j);
            worst = worst.max(re.hypot(im));
        }
    }
    worst
}

/// At the zone centre the sum rule is exact, not approximate.
#[test]
fn the_acoustic_sum_rule_is_exact_at_the_zone_centre() {
    for (name, molecule) in [("LiF", rocksalt(3, 9, 4.03)), ("diamond", diamond())] {
        let result = dynamical_matrix_dfpt(
            &molecule,
            &params(),
            &options(3),
            [0.0, 0.0, 0.0],
            &pm7_rs::dfpt::DfptOptions::default(),
        )
        .unwrap();
        let residual = acoustic_residual(&result.force_constants);
        let size = scale(&result.force_constants);
        assert!(size > 0.1, "{name}: force constants are trivially small");
        assert!(
            residual < 1.0e-12,
            "{name}: acoustic sum rule at q = 0 is {residual:.3e}, not zero"
        );
    }
}

/// The non-analytic term carries no acoustic residue of its own.
///
/// `Σ_B D^NA_{Aα,Bβ}(q̂) = (4π/Ω)(q̂·Z*_A)_α [Σ_B (q̂·Z*_B)_β] / (q̂·ε·q̂)`, and the bracket is the
/// Born sum rule. This is what makes the `q → 0` limit of the *full* `D(q)` acoustically clean
/// even for a polar crystal, where `D(q)` is genuinely discontinuous at the zone centre.
#[test]
fn the_non_analytic_term_satisfies_the_sum_rule_by_construction() {
    let molecule = rocksalt(3, 9, 4.03);
    let field = born_and_dielectric(
        &molecule,
        &params(),
        &options(3),
        &pm7_rs::dfpt::DfptOptions::default(),
    )
    .unwrap();
    assert!(
        field.acoustic_residual() < 1.0e-12,
        "Born charges do not sum to zero: {:.3e}",
        field.acoustic_residual()
    );
    let na = field.non_analytic().unwrap();
    for q_hat in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]] {
        let matrix = na.matrix(q_hat).unwrap();
        assert!(
            scale(&matrix) > 1.0e-3,
            "the non-analytic term is trivially small for q_hat = {q_hat:?}, so this is vacuous"
        );
        assert!(
            acoustic_residual(&matrix) < 1.0e-12,
            "sum_B D^NA is {:.3e} for q_hat = {q_hat:?}",
            acoustic_residual(&matrix)
        );
    }
}

/// The term depends on the **direction** of `q_hat` and not on its length.
///
/// `(q·Z*)(q·Z*)/(q·ε·q)` is homogeneous of degree zero, so scaling the argument must change
/// nothing. If it did — if a `1/q²` survived the normalization — the LO–TO correction would
/// diverge as the zone centre is approached, which is the failure this pins against.
#[test]
fn the_non_analytic_term_is_scale_free_in_q_hat() {
    let molecule = rocksalt(3, 9, 4.03);
    let field = born_and_dielectric(
        &molecule,
        &params(),
        &options(3),
        &pm7_rs::dfpt::DfptOptions::default(),
    )
    .unwrap();
    let na = field.non_analytic().unwrap();
    let reference = na.matrix([1.0, 0.0, 0.0]).unwrap();
    for length in [1.0e-6_f64, 1.0e-3, 1.0, 1.0e3] {
        let scaled = na.matrix([length, 0.0, 0.0]).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..reference.n {
            for j in 0..reference.n {
                let (ar, ai) = reference.get(i, j);
                let (br, bi) = scaled.get(i, j);
                worst = worst.max((ar - br).hypot(ai - bi));
            }
        }
        assert!(
            worst < 1.0e-12,
            "|q_hat| = {length:.0e} changed the non-analytic term by {worst:.3e}"
        );
    }
}

/// Away from the zone centre the acoustic residue shrinks with `q`, on a mesh that can hold it.
///
/// The mesh has to resolve the wavevector: five divisions give a step of `1/5`, and the smallest
/// `q` tested here is `1/160`. The residue falls by roughly a factor of two per halving of `q`,
/// which is the `O(q)` an analytic `Φ(q)` gives — a **`1/q²` divergence, or any plateau, is what
/// this test exists to catch**.
#[test]
fn the_acoustic_residue_vanishes_linearly_with_q() {
    let molecule = diamond();
    let settings = pm7_rs::dfpt::DfptOptions::default();
    let mut previous: Option<(f64, f64)> = None;
    let mut last: Option<(f64, f64)> = None;
    for q in [0.1_f64, 0.05, 0.025, 0.0125] {
        let result =
            dynamical_matrix_dfpt(&molecule, &params(), &options(5), [q, 0.0, 0.0], &settings)
                .unwrap();
        let residual = acoustic_residual(&result.force_constants);
        last = Some((residual, scale(&result.force_constants)));
        if let Some((prev_q, prev_residual)) = previous {
            let shrink = prev_residual / residual;
            let step = prev_q / q;
            assert!(
                shrink > 0.7 * step,
                "halving q from {prev_q} to {q} shrank the acoustic residue only {shrink:.2}x \
                 ({prev_residual:.3e} -> {residual:.3e}); an O(q) approach should give about \
                 {step:.0}x, and anything that grows is a divergence"
            );
        }
        previous = Some((q, residual));
    }
    // Judged against the matrix's own scale, which is the only form that means anything: the
    // residue is an element of `Φ(q)` and "small" has to be relative to how large those are.
    // Measured 1.85e-1 against 9.24 — two percent — at `q = 1/80` on a five-division mesh.
    let (residual, size) = last.unwrap();
    assert!(
        residual < 0.03 * size,
        "the acoustic residue is {residual:.3e} at q = 0.0125, which is {:.1} % of the force \
         constants' scale {size:.3e}",
        100.0 * residual / size
    );
}

/// `analytic_hessian` on a k mesh is the zone-centre perturbation solve, not a refusal.
///
/// v0.2.1 refused and pointed at `numerical_hessian`, which differentiates the analytic gradient —
/// `3N` pairs of SCF runs where one response solve would do, and a different approximation. The
/// two must agree, and they must agree with a supercell at Γ, which is the same physics reached a
/// third way.
#[test]
fn the_k_mesh_hessian_is_the_zone_centre_response() {
    let molecule = diamond();
    let settings = pm7_rs::dfpt::DfptOptions::default();

    let hessian = pm7_rs::analytic_hessian(&molecule, &params(), &options(3), 1.0e-3).unwrap();
    let response =
        dynamical_matrix_dfpt(&molecule, &params(), &options(3), [0.0; 3], &settings).unwrap();

    assert_eq!(hessian.rows, response.force_constants.n);
    let mut worst = 0.0_f64;
    let mut size = 0.0_f64;
    for i in 0..hessian.rows {
        for j in 0..hessian.cols {
            let (re, _) = response.force_constants.get(i, j);
            worst = worst.max((hessian[(i, j)] - re).abs());
            size = size.max(re.abs());
        }
    }
    assert!(
        size > 1.0,
        "force constants are trivially small: {size:.3e}"
    );
    assert!(
        worst < 1.0e-10 * size,
        "analytic_hessian on a k mesh differs from D(q=0) by {worst:.3e} against a scale of \
         {size:.3e}; it is meant to be the same calculation"
    );

    // And the acoustic sum rule survives the route through `analytic_hessian`.
    let nat = hessian.rows / 3;
    let mut acoustic = 0.0_f64;
    for a in 0..nat {
        for alpha in 0..3 {
            for beta in 0..3 {
                let total: f64 = (0..nat)
                    .map(|b| hessian[(3 * a + alpha, 3 * b + beta)])
                    .sum();
                acoustic = acoustic.max(total.abs());
            }
        }
    }
    assert!(
        acoustic < 1.0e-10 * size,
        "acoustic sum rule is {acoustic:.3e} on the k-mesh Hessian"
    );
}

/// The LO–TO term raises exactly one branch, and leaves the transverse pair alone.
///
/// That is the whole observable content of the splitting, and it is the check that a sign or an
/// index transposition fails: a `D^NA` built with the displacement index contracted against `q̂`
/// instead of the field index would move the wrong branches, and `Σ|ω|` alone would not notice.
///
/// Both routes are checked, because until v0.2.2 neither could be reached from anywhere except
/// Rust — `DfptResult::frequencies_cm_lo_to` and `ForceConstants::frequencies_cm_lo_to` had no
/// caller in the entire repository, while `docs/properties.md` described them as the way to use
/// the term.
#[test]
fn the_lo_to_term_raises_one_branch_and_leaves_the_others() {
    let molecule = rocksalt(3, 9, 4.03);
    let settings = pm7_rs::dfpt::DfptOptions::default();
    let field = born_and_dielectric(&molecule, &params(), &options(3), &settings).unwrap();
    let na = field.non_analytic().unwrap();
    let q_hat = [1.0, 0.0, 0.0];

    let gamma =
        dynamical_matrix_dfpt(&molecule, &params(), &options(3), [0.0; 3], &settings).unwrap();
    let plain = sorted(gamma.frequencies_cm().unwrap());
    let split = sorted(gamma.frequencies_cm_lo_to(&na, q_hat).unwrap());

    assert_eq!(plain.len(), split.len());
    // Nothing may go down: the non-analytic term is a positive-semidefinite outer product.
    for (before, after) in plain.iter().zip(&split) {
        assert!(
            after >= &(before - 1.0e-6),
            "a branch fell from {before:.4} to {after:.4} cm^-1 under the LO-TO term"
        );
    }
    // Exactly one optical branch moves, by a visible amount.
    let moved: Vec<usize> = (0..plain.len())
        .filter(|&i| (split[i] - plain[i]).abs() > 1.0)
        .collect();
    assert_eq!(
        moved.len(),
        1,
        "expected one longitudinal branch to move, got {moved:?}: {plain:?} -> {split:?}"
    );
    assert!(
        split[moved[0]] - plain[moved[0]] > 10.0,
        "the splitting is only {:.3} cm^-1, which is too small to be distinguishable from noise",
        split[moved[0]] - plain[moved[0]]
    );

    // The supercell route carries the same term through its own method.
    let force_constants =
        pm7_rs::force_constants(&molecule, &params(), &options(1), [2, 2, 2]).unwrap();
    let plain = sorted(force_constants.frequencies_cm([0.0; 3]).unwrap());
    let split = sorted(force_constants.frequencies_cm_lo_to(&na, q_hat).unwrap());
    let moved: Vec<usize> = (0..plain.len())
        .filter(|&i| (split[i] - plain[i]).abs() > 1.0)
        .collect();
    assert_eq!(
        moved.len(),
        1,
        "supercell route: expected one branch to move, got {moved:?}"
    );
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// A coarse mesh cannot answer a long-wavelength question, and the residue shows it.
///
/// Not an assertion about a bug — it is the sampling limit, and it is here so that a future
/// change which *removes* the mesh dependence (or makes it worse) is visible rather than
/// discovered by someone asking for `q = 0.001` on a 3³ mesh and believing the answer.
#[test]
fn the_small_q_residue_is_a_k_mesh_artifact_and_converges_away() {
    let molecule = rocksalt(3, 9, 4.03);
    let settings = pm7_rs::dfpt::DfptOptions::default();
    let q = [0.00625, 0.0, 0.0];
    let mut residuals = Vec::new();
    for mesh in [3usize, 5, 7] {
        let result =
            dynamical_matrix_dfpt(&molecule, &params(), &options(mesh), q, &settings).unwrap();
        residuals.push(acoustic_residual(&result.force_constants));
    }
    assert!(
        residuals[1] < 0.4 * residuals[0],
        "5^3 did not improve on 3^3: {:.3e} vs {:.3e}",
        residuals[1],
        residuals[0]
    );
    assert!(
        residuals[2] < 0.7 * residuals[1],
        "7^3 did not improve on 5^3: {:.3e} vs {:.3e}",
        residuals[2],
        residuals[1]
    );
}
