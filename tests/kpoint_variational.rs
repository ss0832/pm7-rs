// SPDX-License-Identifier: GPL-3.0-or-later
//! The k-point analytic gradient is the derivative of the energy the SCF reports.
//!
//! Stated as a **convergence rate**, not as an absolute bound. A central difference carries a
//! truncation error going as `h²`, so the two agree if and only if the disagreement falls by four
//! when `h` is halved. An absolute tolerance cannot distinguish "the gradient is right and my step
//! was coarse" from "the gradient is wrong by a constant", and picking one is really picking a
//! step size and hoping.
//!
//! Measured on a hydrogen-fluoride chain, `4×1×1` mesh:
//!
//! | h (Bohr) | \|analytic − FD\| |
//! |---|---|
//! | 4.0e-3 | 1.010e-4 |
//! | 2.0e-3 | 2.525e-5 |
//! | 1.0e-3 | 6.311e-6 |
//! | 5.0e-4 | 1.577e-6 |
//! | 2.5e-4 | 3.936e-7 |
//!
//! Exactly four per halving, over sixteen-fold in `h`. There is no constant term to find.
//!
//! # What this file was written to look for, and did not find
//!
//! `E[P] = ½ Tr[P(H + F[P])]` is stationary only on the idempotent manifold, so contracting one
//! density against a Fock built from a *different* one leaves an error **first order** in the gap
//! between them. The k-point loop did contract the output density against `F[input]`, where the
//! molecular path has always spent one extra Fock build at the converged density — a real
//! structural difference between the two, and the sort that hides a first-order error behind a
//! number that still looks like an energy.
//!
//! It does not, at convergence: the loop exits when the two densities agree to `p_tol`, so the
//! first-order term is `p_tol`-sized and the reported energy was already the right one. Rebuilding
//! the Fock at the converged density moved this measurement by **3e-8 eV/Bohr**, which is nothing.
//! The rebuild is kept because it makes the invariant hold on the *unconverged* exit path too —
//! where the reported density is the mixed one and the energy came from the unmixed — and because
//! having the two SCF paths differ in that discipline is what made the question hard to answer.

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{closed_form_gradient, run_pm7, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::method("pm7-".parse().unwrap()).unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

/// A polar chain: the long-range channel carries real weight, so an inconsistency in the energy
/// would have somewhere to show up.
fn hf_chain(displacement: f64) -> Molecule {
    let a = a0();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 9,
            position: Vec3::new(displacement, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.95 * a, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(3.4 * a, 0.0, 0.0)]).unwrap())
}

fn options(p_tol: f64) -> Pm7Options {
    Pm7Options {
        method: "pm7-".parse().unwrap(),
        e_tol: 1.0e-12,
        p_tol,
        max_scf: 2000,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(4, 1, 1),
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// `|∂E/∂x_analytic − central difference of the reported energy|`, in eV/Bohr.
fn mismatch(p_tol: f64, step: f64) -> f64 {
    let opts = options(p_tol);
    let analytic = closed_form_gradient(&hf_chain(0.0), &params(), &opts)
        .expect("analytic gradient")
        .gradient[0]
        .x;
    let plus = run_pm7(&hf_chain(step), &params(), &opts)
        .expect("SCF at +h")
        .total_ev;
    let minus = run_pm7(&hf_chain(-step), &params(), &opts)
        .expect("SCF at -h")
        .total_ev;
    (analytic - (plus - minus) / (2.0 * step)).abs()
}

#[test]
fn the_gradient_is_the_derivative_of_the_reported_energy() {
    // Four halvings. Each must divide the disagreement by close to four; a constant offset — a
    // gradient that differentiates something other than the reported energy — would show as a
    // ratio heading for one.
    let steps = [4.0e-3_f64, 2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4];
    let values: Vec<f64> = steps.iter().map(|&h| mismatch(1.0e-10, h)).collect();
    assert!(
        values[0] > 1.0e-6,
        "the coarsest step already agrees to {:.3e}; this test cannot see a rate",
        values[0]
    );
    for window in values.windows(2).enumerate() {
        let (index, pair) = window;
        let ratio = pair[0] / pair[1];
        assert!(
            (3.0..5.2).contains(&ratio),
            "halving h from {:.2e} to {:.2e} changed the disagreement by {ratio:.2}x \
             ({:.3e} -> {:.3e}); a central difference must give 4, and anything near 1 is a \
             gradient that does not differentiate the reported energy",
            steps[index],
            steps[index + 1],
            pair[0],
            pair[1]
        );
    }
}

/// Loosening the SCF must not break the agreement.
///
/// A first-order inconsistency between the reported energy and the density it belongs to scales
/// with the density residual, so it would grow as `p_tol` is relaxed — while finite-difference
/// truncation does not care about `p_tol` at all. Molecular dynamics runs loose on purpose, which
/// is where such a defect would do its damage.
#[test]
fn loosening_the_scf_does_not_break_the_agreement() {
    let step = 1.0e-3;
    let tight = mismatch(1.0e-10, step);
    let loose = mismatch(1.0e-6, step);
    assert!(
        loose < tight.max(1.0e-8) * 20.0,
        "loosening p_tol from 1e-10 to 1e-6 took the disagreement from {tight:.3e} to \
         {loose:.3e} eV/Bohr; at this step it is truncation error, which p_tol cannot move"
    );
}
