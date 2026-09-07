// SPDX-License-Identifier: GPL-3.0-or-later
//! The CPHF has to converge, and it has to say so when it does not.
//!
//! Two separate claims, and until v0.2.2 neither was checked.
//!
//! The first is that the solver reaches the answer on systems harder than the well-behaved
//! organic molecules the Hessian tests use — heavy elements with `d` orbitals, conjugated
//! systems with a small frontier gap, charged species, open shells. These are where a
//! fixed-point iteration's spectral radius creeps towards one, and they are the reason the
//! solver is a preconditioned conjugate gradient rather than a damped iteration.
//!
//! The second is that a failure is *reported*. Through v0.2.1 both solvers ran their iteration
//! budget out and returned the last iterate as `Ok`, with nothing recording that it was not a
//! solution. The relaxation term is `4 G:U`, linear in `U`, so a Hessian built on an unconverged
//! response is wrong in proportion to the residual — and it comes back looking exactly like a
//! converged one. Frequencies derived from it are wrong by an amount nobody can see.
//!
//! The check used throughout is the analytic Hessian against `numerical_hessian`, which shares
//! the SCF and the gradient but **no** part of the CPHF: it central-differences the analytic
//! gradient instead of solving a response. A CPHF that stopped early disagrees with it.

use pm7_rs::{analytic_hessian, numerical_hessian, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().expect("PM7 parameters")
}

fn molecule(xyz: &str) -> Molecule {
    Molecule::from_xyz_str(xyz, 0.0).expect("geometry")
}

/// Tight enough that the CPHF is solving against a genuinely stationary density.
///
/// A loose SCF is the usual reason a response looks hard: the orbital Hessian is positive
/// definite *at a stationary point*, and a density that is not quite one makes the operator not
/// quite SPD.
fn tight() -> Pm7Options {
    Pm7Options {
        p_tol: 1.0e-10,
        e_tol: 1.0e-11,
        ..Pm7Options::default()
    }
}

/// Largest absolute disagreement between the two Hessians, and the scale to judge it against.
fn compare(molecule: &Molecule, options: &Pm7Options) -> (f64, f64) {
    // 1e-3 Bohr, the step the crate's own analytic-vs-numeric unit tests use. A smaller step is
    // not a better finite difference here: the truncation error falls as h^2 but the SCF noise
    // in the differenced gradient rises as 1/h, and below about 1e-3 the second term wins.
    let analytic =
        analytic_hessian(molecule, &params(), options, 1.0e-3).expect("analytic Hessian");
    let numeric = numerical_hessian(molecule, &params(), options, 1.0e-3).expect("numeric Hessian");
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (a, b) in analytic.as_slice().iter().zip(numeric.as_slice()) {
        worst = worst.max((a - b).abs());
        scale = scale.max(a.abs());
    }
    (worst, scale)
}

/// The smallest system in the set, and the one a damped iteration is most often assumed to find
/// easy — which is why it is worth checking rather than assuming.
#[test]
fn water_converges_and_matches_the_finite_difference() {
    let mol = molecule(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.958400 0.000000 0.000000\n\
         H -0.239987 0.927846 0.000000\n",
    );
    let (worst, scale) = compare(&mol, &tight());
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "analytic vs numeric Hessian differ by {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// `d` orbitals: nine functions per heavy atom, so the ov block and the response kernel are both
/// several times larger, and the frontier levels sit closer together.
#[test]
fn a_heavy_element_converges() {
    let mol = molecule(
        "3\nhydrogen sulfide\nS 0.000000 0.000000 0.000000\nH 1.340000 0.000000 0.000000\n\
         H -0.330000 1.290000 0.000000\n",
    );
    let (worst, scale) = compare(&mol, &tight());
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "H2S: analytic vs numeric differ by {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// A conjugated ring: the smallest frontier gap of the set, which is what makes the orbital
/// Hessian ill-conditioned — the preconditioner divides by `ε_a − ε_i` and those are small here.
#[test]
fn a_conjugated_system_converges() {
    let mol = molecule(
        "12\nbenzene\n\
         C  0.000000  1.396000 0.000000\nC  1.209000  0.698000 0.000000\n\
         C  1.209000 -0.698000 0.000000\nC  0.000000 -1.396000 0.000000\n\
         C -1.209000 -0.698000 0.000000\nC -1.209000  0.698000 0.000000\n\
         H  0.000000  2.479000 0.000000\nH  2.147000  1.240000 0.000000\n\
         H  2.147000 -1.240000 0.000000\nH  0.000000 -2.479000 0.000000\n\
         H -2.147000 -1.240000 0.000000\nH -2.147000  1.240000 0.000000\n",
    );
    let (worst, scale) = compare(&mol, &tight());
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "benzene: analytic vs numeric differ by {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// A charged species. The response of an ion is not the response of a neutral with a different
/// electron count: the long-range part of the kernel sees a net charge.
#[test]
fn a_charged_species_converges() {
    let mut mol = molecule(
        "5\nammonium\nN 0.000000 0.000000 0.000000\nH 0.589000 0.589000 0.589000\n\
         H -0.589000 -0.589000 0.589000\nH -0.589000 0.589000 -0.589000\n\
         H 0.589000 -0.589000 -0.589000\n",
    );
    mol.charge = 1.0;
    let options = Pm7Options {
        charge: 1.0,
        ..tight()
    };
    let (worst, scale) = compare(&mol, &options);
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "NH4+: analytic vs numeric differ by {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// Open shell, which goes through the coupled two-channel `ucphf_ov` rather than the restricted
/// solver — a different code path with the same convergence question.
#[test]
fn an_open_shell_radical_converges() {
    let mut mol = molecule(
        "4\nmethyl\nC 0.000000 0.000000 0.000000\nH 1.079000 0.000000 0.000000\n\
         H -0.539000 0.934000 0.000000\nH -0.539000 -0.934000 0.000000\n",
    );
    mol.multiplicity = 2;
    let options = Pm7Options {
        multiplicity: 2,
        ..tight()
    };
    let (worst, scale) = compare(&mol, &options);
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "methyl: analytic vs numeric differ by {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// The budget is the caller's, and running out of it is an error rather than an answer.
///
/// Two claims in one test, because they are the same claim from either side: a budget too small to
/// finish must **refuse** (naming the knob, so the reader knows what to change), and the default
/// must be enough for a molecule this size. The pair is what says the number is read at all — a
/// budget that was ignored would pass the second half and fail the first.
///
/// Water rather than something hard on purpose. The claim under test is the plumbing, and a system
/// that needs 90 iterations would put the test one tuning change away from being about the solver.
#[test]
fn the_cphf_budget_is_the_callers_and_running_out_is_an_error() {
    let mol = molecule(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.958400 0.000000 0.000000\n\
         H -0.239987 0.927846 0.000000\n",
    );
    let starved = Pm7Options {
        cphf_max_iterations: 2,
        ..tight()
    };
    let error = analytic_hessian(&mol, &params(), &starved, 1.0e-3)
        .expect_err("two applications cannot solve water's response");
    let message = error.to_string();
    assert!(
        message.contains("did not converge") && message.contains("cphf-max-iterations"),
        "the refusal must name the budget it wants raised: {message}"
    );

    // And the same system at the default budget, so the refusal above is about the number and not
    // about the molecule.
    let (worst, scale) = compare(&mol, &tight());
    assert!(scale > 1.0, "Hessian is trivially small: scale {scale}");
    assert!(
        worst < 1.0e-3,
        "water at the default budget: {worst:.3e} eV/Bohr^2 (Hessian scale {scale:.3e})"
    );
}

/// The open-shell solver refuses too — which it did not until 0.2.3.
///
/// `ucphf_ov` does not share a loop with the restricted solvers, so when they were made to refuse
/// an unconverged response it kept the old behaviour: a hard-coded hundred iterations and then
/// `Ok(last iterate)`. A Hessian built on that is wrong in proportion to the residual and is
/// indistinguishable from a converged one, which is the entire reason the restricted path refuses.
#[test]
fn the_open_shell_cphf_refuses_rather_than_returning_its_last_iterate() {
    let mut mol = molecule(
        "4\nmethyl\nC 0.000000 0.000000 0.000000\nH 1.079000 0.000000 0.000000\n\
         H -0.539000 0.934000 0.000000\nH -0.539000 -0.934000 0.000000\n",
    );
    mol.multiplicity = 2;
    let starved = Pm7Options {
        multiplicity: 2,
        cphf_max_iterations: 2,
        ..tight()
    };
    let error = analytic_hessian(&mol, &params(), &starved, 1.0e-3)
        .expect_err("two applications cannot solve the methyl radical's coupled response");
    assert!(
        error.to_string().contains("did not converge"),
        "the unrestricted solver must refuse, not return its last iterate: {error}"
    );
}
