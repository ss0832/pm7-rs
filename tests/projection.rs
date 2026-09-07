// SPDX-License-Identifier: GPL-3.0-or-later
//! Removing translations and rotations by projection.
//!
//! The unit tests in `src/projection.rs` check the construction against itself: that the generators
//! are orthonormal, that the complement is a complement, that the lattice rule counts the axes a
//! lattice admits. None of that would catch a *wrong generator* — a rotation built about the
//! coordinate origin instead of the centre of mass, or one missing its `sqrt(m)` weighting, is
//! still a perfectly orthonormal set of vectors spanning a perfectly good subspace.
//!
//! These tests check it against things that share no code with it:
//!
//! 1. **The analytic gradient.** Differentiating `E(Rot(theta w) R) = E(R)` twice gives a closed
//!    form for the curvature along a rotation generator, in terms of the gradient. That is the
//!    test that pins the generator itself.
//! 2. **A rigid motion.** Frequencies are invariant under one; a generator with the wrong origin
//!    or the wrong mass weighting is not.
//! 3. **The trace.** An algebraic identity of any orthonormal decomposition.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::hessian::vibrational_analysis;
use pm7_rs::math::Vec3;
use pm7_rs::projection::{Projection, RigidSubspace};
use pm7_rs::{analytic_hessian, closed_form_gradient, Atom, Molecule, Pm7Options, Pm7Parameters};

fn at(z: u8, x: f64, y: f64, z_: f64) -> Atom {
    Atom {
        z,
        position: Vec3::new(
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z_ * ANGSTROM_TO_BOHR,
        ),
    }
}

/// Water at its **experimental** geometry, which is not PM7's minimum. That is the point: the
/// rotations only have curvature away from a stationary point, so a relaxed structure would make
/// every one of these tests pass vacuously.
fn unrelaxed_water() -> Molecule {
    Molecule::new(vec![
        at(8, 0.0, 0.0, 0.0),
        at(1, 0.9584, 0.0, 0.0),
        at(1, -0.24, 0.9278, 0.0),
    ])
}

fn co2() -> Molecule {
    Molecule::new(vec![
        at(6, 0.0, 0.0, 0.0),
        at(8, 0.0, 0.0, 1.16),
        at(8, 0.0, 0.0, -1.16),
    ])
}

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

/// Rotate by `angle` about `axis` (Rodrigues), then translate.
fn moved(molecule: &Molecule, axis: Vec3, angle: f64, shift: Vec3) -> Molecule {
    let k = axis.normalized();
    let (s, c) = (angle.sin(), angle.cos());
    Molecule::new(
        molecule
            .atoms
            .iter()
            .map(|a| {
                let r = a.position;
                let rotated = r * c + k.cross(r) * s + k * (k.dot(r) * (1.0 - c));
                Atom {
                    z: a.z,
                    position: rotated + shift,
                }
            })
            .collect(),
    )
}

/// The curvature along a rotation generator, in closed form, from the **gradient**.
///
/// `E(Rot(theta w) R) = E(R)` for every theta. Differentiating twice at theta = 0 and contracting
/// with the normalized mass-weighted generator gives
///
/// ```text
/// lambda_rot(w) = a0^2 * (sum_A g_A . d_A_perp) / I_w
/// ```
///
/// with `d_A_perp = d_A - w (w . d_A)`, `I_w = sum_A m_A |w x d_A|^2`, `g` in eV/Bohr, `d` in Bohr
/// and `I` in amu*Bohr^2 — landing in eV/(A^2*amu), the units the mass-weighted Hessian is
/// diagonalized in (`src/hessian.rs`, the `a0_sq` factor).
///
/// This shares nothing with `RigidSubspace` but the geometry.
fn rotational_curvature_from_gradient(molecule: &Molecule, gradient: &[Vec3], axis: Vec3) -> f64 {
    let masses = pm7_rs::data_tables::MASS;
    let total: f64 = molecule.atoms.iter().map(|a| masses[a.z as usize]).sum();
    let mut com = Vec3::new(0.0, 0.0, 0.0);
    for a in &molecule.atoms {
        com += a.position * masses[a.z as usize];
    }
    com = com / total;

    let w = axis.normalized();
    let mut numerator = 0.0;
    let mut moment = 0.0;
    for (atom, g) in molecule.atoms.iter().zip(gradient) {
        let d = atom.position - com;
        let perp = d - w * w.dot(d);
        numerator += g.dot(perp);
        moment += masses[atom.z as usize] * w.cross(d).norm2();
    }
    let a0_sq = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    a0_sq * numerator / moment
}

/// The test that catches a wrong rotation generator, and the reason the others can be trusted.
#[test]
fn the_removed_rotational_curvature_is_the_gradient_contraction() {
    let molecule = unrelaxed_water();
    let params = params();
    let options = Pm7Options::default();

    let vib = vibrational_analysis(&molecule, &params, &options, 1.0e-3).unwrap();
    let removed = vib.removed.as_ref().expect("rigid subspace reported");
    assert_eq!(removed.subspace.n_rotations, 3);

    let gradient = closed_form_gradient(&molecule, &params, &options).unwrap();

    let want: Vec<f64> = removed
        .subspace
        .rotation_axes
        .iter()
        .map(|axis| rotational_curvature_from_gradient(&molecule, &gradient.gradient, *axis))
        .collect();
    let got = &removed.curvature[removed.subspace.n_translations..];
    eprintln!("projector {got:?}\ngradient  {want:?}");

    // The identity is exact for an exact Hessian, so what is being measured here is the *Hessian's*
    // accuracy, not the projector's. Measured on this geometry the two agree to 2.8e-5 and 2.6e-5
    // absolute on curvatures of 0.121 and 0.019 -- a constant absolute error, which is why the
    // bound is scaled by the largest curvature rather than applied per element. A relative bound
    // would be a test of how small the smallest rotation happens to be.
    //
    // Loose as a number, tight as a test: a generator built about the coordinate origin, or one
    // missing its sqrt(m) weighting, is wrong by a factor, not by a part in ten thousand.
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1.0e-3 * scale,
            "rotation {k}: projector says {g}, the gradient says {w}"
        );
    }

    // And they are real numbers, not noise: at this geometry v0.2.2 printed these three as part of
    // the spectrum, at 71, 121 and 181 cm^-1. Asserted on the largest, because which axis carries
    // the curvature depends on the principal frame and is not the claim being made.
    let rotational = &removed.frequencies_cm[removed.subspace.n_translations..];
    let worst = rotational.iter().fold(0.0_f64, |m, f| m.max(f.abs()));
    eprintln!("removed rotations (cm^-1): {rotational:?}");
    assert!(
        worst > 100.0,
        "the rotations should be visibly non-zero at an unrelaxed geometry -- that is the whole \
         reason a magnitude cutoff cannot find them: {rotational:?}"
    );
}

/// Translational invariance is unconditional, so this one holds at *any* geometry — including this
/// deliberately unrelaxed one. It is the molecular acoustic sum rule.
#[test]
fn the_removed_translational_curvature_is_zero_even_off_a_stationary_point() {
    let molecule = unrelaxed_water();
    let vib = vibrational_analysis(&molecule, &params(), &Pm7Options::default(), 1.0e-3).unwrap();
    let removed = vib.removed.as_ref().unwrap();
    let scale = vib
        .hessian
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    for c in &removed.curvature[..3] {
        assert!(c.abs() < 1.0e-9 * scale, "translation curvature {c}");
    }
}

#[test]
fn a_molecule_off_its_minimum_gives_exactly_three_vibrations() {
    let vib = vibrational_analysis(
        &unrelaxed_water(),
        &params(),
        &Pm7Options::default(),
        1.0e-3,
    )
    .unwrap();
    assert_eq!(
        vib.frequencies_cm.len(),
        3,
        "3N - 6, not 3N: {:?}",
        vib.frequencies_cm
    );
    assert_eq!(vib.modes.cols, 3);
    assert_eq!(vib.cartesian_modes.cols, 3);
    assert_eq!(vib.modes.rows, 9, "the rows stay Cartesian");
    // The three that survive are the vibrations, all far above the rotations that were removed.
    for f in &vib.frequencies_cm {
        assert!(*f > 1000.0, "{:?}", vib.frequencies_cm);
    }
}

#[test]
fn a_linear_molecule_gives_three_n_minus_five() {
    let vib = vibrational_analysis(&co2(), &params(), &Pm7Options::default(), 1.0e-3).unwrap();
    let removed = vib.removed.as_ref().unwrap();
    assert!(removed.subspace.linear);
    assert_eq!(
        removed.subspace.dimension(),
        5,
        "three translations, two rotations"
    );
    assert_eq!(
        vib.frequencies_cm.len(),
        4,
        "3N - 5 for a linear triatomic: {:?}",
        vib.frequencies_cm
    );
    // CO2's bend is doubly degenerate, which is the structure a wrong projector destroys.
    assert!(
        (vib.frequencies_cm[0] - vib.frequencies_cm[1]).abs() < 1.0,
        "the bend should be degenerate: {:?}",
        vib.frequencies_cm
    );
}

/// A rigid motion cannot change a frequency. A generator built about the coordinate origin instead
/// of the centre of mass, or one missing its `sqrt(m)` weighting, survives every self-consistency
/// check in `src/projection.rs` and fails this one.
///
/// The shift has to be large and the molecule mass-asymmetric for the test to bite: the removed
/// *span* is origin-independent even with a wrong origin, because `sqrt(m_A) (w x c)` is a fixed
/// combination of the translation generators. What a wrong origin changes is which vectors inside
/// the span are called rotations, and that only shows up against unequal masses.
#[test]
fn the_projected_frequencies_are_invariant_under_a_rigid_motion() {
    let params = params();
    let options = Pm7Options::default();
    let here = unrelaxed_water();
    let there = moved(
        &here,
        Vec3::new(1.0, -2.0, 3.0),
        37.0_f64.to_radians(),
        Vec3::new(137.0, -41.0, 9.0) * ANGSTROM_TO_BOHR,
    );

    let a = vibrational_analysis(&here, &params, &options, 1.0e-3).unwrap();
    let b = vibrational_analysis(&there, &params, &options, 1.0e-3).unwrap();
    assert_eq!(a.frequencies_cm.len(), b.frequencies_cm.len());
    // Measured: 1409.413827 / 2805.739256 / 2860.678155 here against 1409.415946 / 2805.738266 /
    // 2860.677231 there -- 2.1e-3 cm^-1 at worst, or 1.5e-6 relative, over a 137 A displacement.
    //
    // That residual is **not** the projector and **not** convergence. It is entirely the starting
    // orientation: `unrelaxed_water` has an O-H bond exactly along x and all three atoms in
    // z = 0, which is the singular two-centre frame `docs/singularities.md` describes, so that
    // pair's derivative comes from a localized finite difference here and from the analytic path
    // after the rotation. Two numerical routes, one small difference.
    // `the_projection_is_exactly_invariant_from_a_generic_orientation` below is the sharp version
    // and gets a bit-for-bit bound; this one keeps the loose bound because its fixture is the
    // singular case on purpose.
    //
    // A wrong origin would fail by a factor rather than in the sixth digit: with the centre of
    // mass 137 A from the coordinate origin, generators built about the origin are 99.99 %
    // translation, and the "rotations" removed would be three more copies of the translational
    // subspace.
    for (x, y) in a.frequencies_cm.iter().zip(&b.frequencies_cm) {
        assert!(
            (x - y).abs() < 1.0e-2,
            "a rigid motion moved a frequency: {:?} vs {:?}",
            a.frequencies_cm,
            b.frequencies_cm
        );
    }
    // The removed *rotational* curvatures are invariant too, which is the sharper statement: it
    // says the subspace travelled with the molecule rather than staying with the axes. Sorted,
    // because the principal frame comes back in whatever order the eigensolver produces and which
    // axis is which is not the claim. The translations are excluded: they are zero either way, and
    // comparing two numbers near 1e-12 with a relative tolerance tests the rounding, not the
    // physics -- `the_removed_translational_curvature_is_zero...` covers them properly.
    let (ra, rb) = (a.removed.as_ref().unwrap(), b.removed.as_ref().unwrap());
    let sorted = |r: &pm7_rs::hessian::RemovedSubspace| {
        let mut c = r.curvature[r.subspace.n_translations..].to_vec();
        c.sort_by(|p, q| p.partial_cmp(q).unwrap());
        c
    };
    let (ca, cb) = (sorted(ra), sorted(rb));
    let scale = ca.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (x, y) in ca.iter().zip(&cb) {
        assert!(
            (x - y).abs() < 1.0e-3 * scale,
            "removed curvature moved under a rigid motion: {ca:?} vs {cb:?}"
        );
    }
}

/// `sum(kept eigenvalues) + sum(removed curvature) = trace(H_mw)`, exactly, for any orthonormal
/// decomposition. Catches a non-orthonormal generator set, a wrong normalization, or a complement
/// that is not one.
#[test]
fn the_projection_conserves_the_trace() {
    let molecule = unrelaxed_water();
    let params = params();
    let options = Pm7Options::default();
    let vib = vibrational_analysis(&molecule, &params, &options, 1.0e-3).unwrap();
    let removed = vib.removed.as_ref().unwrap();

    // trace(H_mw) = sum_i H_ii * a0^2 / m_i, rebuilt here rather than taken from the module.
    let masses = pm7_rs::data_tables::MASS;
    let a0_sq = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    let trace: f64 = (0..3 * molecule.atoms.len())
        .map(|i| vib.hessian[(i, i)] * a0_sq / masses[molecule.atoms[i / 3].z as usize])
        .sum();

    let accounted: f64 =
        vib.eigenvalues.iter().sum::<f64>() + removed.curvature.iter().sum::<f64>();
    assert!(
        (accounted - trace).abs() < 1.0e-9 * trace.abs(),
        "trace {trace} but kept + removed = {accounted}"
    );
}

/// `Projection::None` reproduces v0.2.2: all `3N`, rotations included.
#[test]
fn the_raw_spectrum_is_still_reachable() {
    let molecule = unrelaxed_water();
    let params = params();
    let options = Pm7Options::default();
    let hessian = analytic_hessian(&molecule, &params, &options, 1.0e-3).unwrap();
    let raw =
        pm7_rs::hessian::vibrational_modes_projected(&molecule, hessian.clone(), Projection::None)
            .unwrap();
    assert_eq!(raw.frequencies_cm.len(), 9);
    assert!(raw.removed.is_none());

    // And the three that the projector calls rotations are in there, at the values v0.2.2 printed.
    let projected =
        pm7_rs::hessian::vibrational_modes_projected(&molecule, hessian, Projection::Rigid)
            .unwrap();
    let removed = projected.removed.unwrap();
    for f in &removed.frequencies_cm[3..] {
        assert!(
            raw.frequencies_cm.iter().any(|r| (r - f).abs() < 5.0),
            "removed rotation {f} is not in the raw spectrum {:?}",
            raw.frequencies_cm
        );
    }
}

/// A single atom has no vibrations at all, and asking for them is not an error.
#[test]
fn a_single_atom_has_no_vibrations() {
    let molecule = Molecule::new(vec![at(10, 0.0, 0.0, 0.0)]);
    let s = RigidSubspace::of(&molecule, Projection::Rigid).unwrap();
    assert_eq!(s.dimension(), 3);
    assert_eq!(s.complement().cols, 0);
}

/// **A chain keeps its own rotation, and a periodic spectrum keeps its `3N` length.**
///
/// Two claims in one, because they are easy to state wrongly in opposite directions.
///
/// A 1-D cell has one genuine free rotation at Γ — about its own axis, the only generator
/// satisfying `ω × T = 0` — and a 3-D acoustic sum rule does not remove it. `Projection::Rigid`
/// consults the lattice and removes four things; `Projection::Translations` removes three and
/// leaves the rotation standing. A slab has *no* admissible rotation, so both remove three.
///
/// And unlike a molecule, a periodic system's array does **not** shrink. The removed generators
/// are re-inserted at exactly zero, because a phonon branch index has to mean the same thing at
/// every `q` and a `q`-dependent length would break every band plot. So the shape is `3N` with
/// four exact zeros, not `3N − 4`.
#[test]
fn a_chain_has_one_admissible_rotation_and_still_reports_three_n_frequencies() {
    // A CH2 chain along x: 3 atoms per cell, 9 degrees of freedom.
    let cell = pm7_rs::Cell::from_angstrom_rows(&[[2.6, 0.0, 0.0]]).unwrap();
    let molecule = Molecule::new(vec![
        at(6, 0.0, 0.0, 0.0),
        at(1, 0.0, 0.63, 0.63),
        at(1, 0.0, -0.63, 0.63),
    ])
    .with_cell(cell);

    for (projection, removed_dim) in [
        (Projection::Rigid, 4),
        (Projection::Translations, 3),
        (Projection::None, 0),
    ] {
        let subspace = RigidSubspace::of(&molecule, projection).unwrap();
        assert_eq!(
            subspace.dimension(),
            removed_dim,
            "{projection:?} on a chain"
        );
    }

    let params = params();
    let options = Pm7Options::default();
    let hessian = analytic_hessian(&molecule, &params, &options, 1.0e-3).unwrap();
    let modes = pm7_rs::hessian::vibrational_modes_projected(&molecule, hessian, Projection::Rigid)
        .unwrap();
    assert_eq!(
        modes.frequencies_cm.len(),
        9,
        "a periodic spectrum keeps its 3N length: {:?}",
        modes.frequencies_cm
    );
    let zeros = modes
        .frequencies_cm
        .iter()
        .filter(|f| f.abs() < 1.0e-9)
        .count();
    assert_eq!(
        zeros, 4,
        "three acoustic modes plus the chain's own rotation, all exactly zero: {:?}",
        modes.frequencies_cm
    );

    // A slab has no admissible rotation: no axis is perpendicular to both lattice vectors and
    // parallel to the plane they span.
    let sheet = pm7_rs::Cell::from_angstrom_rows(&[[2.6, 0.0, 0.0], [0.0, 2.6, 0.0]]).unwrap();
    let slab = Molecule::new(vec![at(6, 0.0, 0.0, 0.0), at(1, 0.0, 0.0, 1.09)]).with_cell(sheet);
    assert_eq!(
        RigidSubspace::of(&slab, Projection::Rigid)
            .unwrap()
            .dimension(),
        3,
        "a slab has three translations and no admissible rotation"
    );
}

/// **From a generic orientation a rigid motion moves a frequency by round-off and nothing more.**
///
/// The sharp form of the test above, and the one that says what the projector's own invariance is.
/// The difference between the two fixtures is not the projector: `unrelaxed_water` sits with an
/// O-H bond exactly along x and all three atoms in z = 0, which is the singular two-centre frame
/// orientation `docs/singularities.md` describes, so one pair's derivative takes the
/// finite-difference fallback there and the analytic path after a rotation. Two numerical routes,
/// one small difference.
///
/// Measured both ways before this was written, worst case over the three vibrations:
///
/// | starting orientation | worst \|delta\| | relative |
/// |---|---|---|
/// | canonical (O-H along x) | 2.1e-3 cm^-1 | 7.4e-7 |
/// | generic | 1.1e-7 cm^-1 | 8.0e-11 |
///
/// Four orders of magnitude apart. The second is round-off in a rotated Cartesian frame -- the
/// `1e-10` the release plan expected of this check -- and the first is a pre-existing,
/// deliberately documented fallback that has nothing to do with projecting rigid motions out.
/// Without both numbers the 2.1e-3 reads as the projector's tolerance, which is what this rules
/// out.
///
/// The bound is relative, not absolute: an absolute one would be a different test for the 1409
/// mode than for the 2860 one.
#[test]
fn the_projection_is_exactly_invariant_from_a_generic_orientation() {
    let params = params();
    let options = Pm7Options::default();
    // Rotated off every symmetry axis first, so no bond lies on a Cartesian axis and no atom in a
    // coordinate plane.
    let here = moved(
        &unrelaxed_water(),
        Vec3::new(0.37, 0.61, -0.70),
        23.7_f64.to_radians(),
        Vec3::zero(),
    );
    let there = moved(
        &here,
        Vec3::new(0.31, -0.77, 0.56),
        37.0_f64.to_radians(),
        Vec3::new(3.1, -2.4, 7.9) * ANGSTROM_TO_BOHR,
    );

    let a = vibrational_analysis(&here, &params, &options, 1.0e-3).unwrap();
    let b = vibrational_analysis(&there, &params, &options, 1.0e-3).unwrap();
    assert_eq!(a.frequencies_cm.len(), b.frequencies_cm.len());
    for (x, y) in a.frequencies_cm.iter().zip(&b.frequencies_cm) {
        assert!(
            (x - y).abs() <= 1.0e-9 * x.abs(),
            "a rigid motion of a generically oriented molecule moved a frequency by more than \
             round-off: {:?} vs {:?}",
            a.frequencies_cm,
            b.frequencies_cm
        );
    }
}
