// SPDX-License-Identifier: GPL-3.0-or-later
//! The continuous dispersion `C6`, and the two ways of getting it wrong.
//!
//! `c6_atom` is a step on an integer bond count: a carbon with four neighbours takes 0.95 and
//! anything else 1.65. That is what MOPAC does, and it makes the energy discontinuous where a
//! distance crosses `1.3(r_i + r_j)` — measured at **0.086 meV** on a methane whose fourth C–H bond
//! is stretched past the threshold, against a local slope of 0.0002 meV per 0.002 Å. Small, and a
//! step: the force there is not the derivative of anything.
//!
//! Two failure modes are guarded here, and the second is the one worth having a test for.
//!
//! **Smoothing that is not faithful.** A soft switch removes the step and changes every answer.
//! The first attempt used a switch sharpness of 12 and was 0.18 % off the discrete coefficient at
//! an *integer* count — and raising the counting sharpness, the obvious knob, did not move that at
//! all, because the error was in the switch rather than the counting.
//!
//! **Smoothing without the chain rule.** Once `C6` depends on the geometry through a coordination
//! number, the gradient acquires a term that the pairwise loop does not contain: `∂E/∂cn_i` times
//! `∂cn_i/∂R_k`, for every `k` near `i`. Omitting it leaves a continuous energy with a gradient
//! that is not its derivative — a systematic force error wherever a coordination number is in
//! transition, in exchange for removing a rare discontinuity. That is a worse trade than doing
//! nothing, and nothing but a finite difference catches it.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{closed_form_gradient, run_pm7, Atom, Molecule, Pm7Options, Pm7Parameters};

/// Methane with its fourth C–H bond at `d` Å. The C–H bonding threshold is `1.3(0.76 + 0.31)`
/// = 1.391 Å, so `d` sweeping past that flips the carbon's neighbour count from 4 to 3.
fn methane(d: f64) -> Molecule {
    let a = ANGSTROM_TO_BOHR;
    let s = d / 1.09;
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.63 * a, 0.63 * a, 0.63 * a),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.63 * a, -0.63 * a, 0.63 * a),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.63 * a, -0.63 * a, -0.63 * a),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.63 * a * s, 0.63 * a * s, -0.63 * a * s),
        },
    ])
}

fn options(smooth: bool) -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-10,
        max_scf: 500,
        smooth_dispersion: smooth,
        ..Default::default()
    }
}

/// The discrete path steps and the smooth one does not — decided by how each scales, not by a
/// threshold.
///
/// A second difference cannot be read against a fixed bound, because a smooth function has one
/// too. What separates them is the **sample spacing**: a step keeps its second difference as `h`
/// shrinks, and curvature falls as `h²`. Measured on the dispersion energy alone (no SCF), across
/// the 1.391 Å C–H threshold:
///
/// | h (Å) | discrete | ratio | smooth | ratio |
/// |---|---|---|---|---|
/// | 0.0040 | 1.973e-3 | — | 6.236e-5 | — |
/// | 0.0020 | 1.973e-3 | **1.00** | 1.657e-5 | **3.76** |
/// | 0.0010 | 1.086e-8 | — | 4.208e-6 | 3.94 |
/// | 0.0005 | 2.714e-9 | 4.00 | 1.056e-6 | 3.98 |
///
/// The discrete row holding at 1.00 is the step. Its collapse at 0.0010 is the sample points
/// ceasing to straddle the threshold, after which it is curvature like everything else. The smooth
/// column is 4.00 throughout: no step at any spacing.
///
/// An earlier version of this test compared total energies and subtracted a "trend" sampled 0.03 Å
/// away. That isolates nothing — the bond-stretch energy is nonlinear, so it left 8e-4 eV of
/// curvature standing over an 8.6e-5 eV step and reported the smooth path as still stepping.
#[test]
fn the_smooth_coefficient_removes_the_step_the_discrete_one_has() {
    let centre = 1.3910_f64;
    let second_difference = |h: f64, smooth: bool| {
        let at = |x: f64| {
            if smooth {
                pm7_rs::dispersion::dispersion_energy_smooth_cut(&methane(x), f64::INFINITY)
            } else {
                pm7_rs::dispersion::dispersion_energy(&methane(x))
            }
        };
        (at(centre + h) - 2.0 * at(centre) + at(centre - h)).abs()
    };

    // The discrete path: the same second difference at two spacings that both straddle the step.
    let coarse = second_difference(0.0040, false);
    let fine = second_difference(0.0020, false);
    assert!(
        coarse > 1.0e-4 && (fine / coarse - 1.0).abs() < 0.05,
        "the discrete path should hold its second difference as h halves, which is what a step \
         does: {coarse:.3e} then {fine:.3e}. If this fixture stopped straddling the threshold the \
         rest of the test would prove nothing."
    );

    // The smooth path: falls as h squared, which is what curvature does and a step does not.
    let mut previous = second_difference(0.0040, true);
    for h in [0.0020_f64, 0.0010, 0.0005] {
        let value = second_difference(h, true);
        let ratio = previous / value;
        assert!(
            (3.6..=4.4).contains(&ratio),
            "halving h to {h:.4} changed the smooth path's second difference by {ratio:.2}; \
             curvature gives 4 and a residual step gives 1, so this is a step the smoothing did \
             not remove"
        );
        previous = value;
    }
}

/// At an ordinary geometry the smooth path reproduces the discrete one.
///
/// The switch is sharp enough that an integer coordination number gives the discrete coefficient
/// to nine digits, so this is the statement that smoothing did not quietly re-parameterize the
/// dispersion everywhere in order to fix it at one point.
#[test]
fn away_from_a_threshold_the_two_paths_agree() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = methane(1.09);
    let discrete = run_pm7(&molecule, &params, &options(false))
        .unwrap()
        .total_ev;
    let smooth = run_pm7(&molecule, &params, &options(true))
        .unwrap()
        .total_ev;
    assert!(
        (smooth - discrete).abs() < 1.0e-6,
        "relaxed methane moved by {:.3e} eV between the two paths; the smooth coefficient is meant \
         to differ only where the discrete one has no derivative",
        smooth - discrete
    );
}

/// The gradient is the derivative of the energy, including the coordination chain rule.
///
/// This is the test the whole exercise turns on. The pairwise loop gives `∂E/∂r`; the coordination
/// number contributes `∂E/∂cn · ∂cn/∂R`, which is many-body — every atom near `i` moves `cn_i`.
/// Dropping that second pass leaves the energy continuous and the gradient wrong, and only a
/// finite difference says so.
///
/// Checked **at** the threshold, where `∂cn/∂R` is largest and the omitted term would be too.
#[test]
fn the_smooth_gradient_matches_a_finite_difference_at_the_threshold() {
    let params = Pm7Parameters::standard().unwrap();
    let opts = options(true);
    let molecule = methane(1.391);
    let analytic = closed_form_gradient(&molecule, &params, &opts)
        .unwrap()
        .gradient;

    let step = 1.0e-4 * ANGSTROM_TO_BOHR;
    let mut worst = 0.0_f64;
    for (atom, reference) in analytic.iter().enumerate() {
        for axis in 0..3 {
            let shift = |sign: f64| {
                let mut m = molecule.clone();
                let mut p = m.atoms[atom].position;
                match axis {
                    0 => p.x += sign * step,
                    1 => p.y += sign * step,
                    _ => p.z += sign * step,
                }
                m.atoms[atom].position = p;
                run_pm7(&m, &params, &opts).unwrap().total_ev
            };
            let fd = (shift(1.0) - shift(-1.0)) / (2.0 * step);
            worst = worst.max((reference.get(axis) - fd).abs());
        }
    }
    assert!(
        worst < 2.0e-4,
        "the analytic gradient differs from a central finite difference by {worst:.3e} eV/Bohr at \
         the coordination threshold. That is where `d(cn)/dR` is largest, so this is what the \
         chain-rule pass in `dispersion_gradient_smooth_cut` exists to supply — a continuous \
         energy whose gradient is not its derivative would pass every continuity test and fail \
         here."
    );
}

/// Switching the option off changes nothing at all.
///
/// At the dispersion function rather than the total energy: comparing total energies compares two
/// SCF solutions, and a first version of this test set a tighter tolerance on one side and failed
/// on the last two bits of a converged density rather than on anything to do with dispersion.
#[test]
fn the_default_path_is_untouched() {
    for d in [1.09, 1.30, 1.3910, 1.60] {
        let molecule = methane(d);
        let a = pm7_rs::dispersion::dispersion_energy(&molecule);
        let b = pm7_rs::dispersion::dispersion_energy_cut(&molecule, f64::INFINITY);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the discrete path must be untouched by the smooth one existing beside it"
        );
    }
}
