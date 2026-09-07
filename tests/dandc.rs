// SPDX-License-Identifier: GPL-3.0-or-later
//! Divide and conquer: does it converge to the exact answer, and does it stay linear?
//!
//! The two questions are separate and both have to be answered. A method that scales linearly to
//! the wrong number is worthless, and one that gets the right number by quietly building a dense
//! matrix has not achieved anything. So: convergence in the buffer radius first, then the size of
//! what the method actually stores and solves.

use pm7_rs::dandc::{run_dandc, DandcOptions};
use pm7_rs::{optimize, run_pm7, Molecule, OptOptions, Pm7Options, Pm7Parameters};

/// Divide and conquer drives a geometry optimization to the same minimum as the exact SCF.
///
/// A relaxation is tens to hundreds of energy-and-gradient evaluations, so it is exactly the
/// workload linear scaling exists for -- and until 0.2.3 it was the one driver `--dandc` could not
/// reach, because `optimize` went straight to `closed_form_gradient`.
///
/// Water is small enough that a wide buffer makes the two paths agree to the SCF's own tolerance,
/// which is the point: it isolates *the optimizer plumbing* from the buffer approximation. The
/// accuracy of the approximation itself is what the rest of this file measures.
#[test]
fn divide_and_conquer_optimizes_to_the_same_minimum() {
    let molecule = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n",
        0.0,
    )
    .unwrap();
    let params = params();
    let scf = options();

    let exact = optimize(&molecule, &params, &scf, &OptOptions::default()).unwrap();
    let approximate = optimize(
        &molecule,
        &params,
        &scf,
        &OptOptions {
            dandc: Some(DandcOptions {
                buffer: 15.0 * pm7_rs::constants::ANGSTROM_TO_BOHR,
                ..DandcOptions::default()
            }),
            ..OptOptions::default()
        },
    )
    .unwrap();

    assert!(exact.converged && approximate.converged);
    assert!(
        approximate.scf.is_none(),
        "a divide-and-conquer run has no Pm7Result to report, and must say so rather than \
         inventing one"
    );
    eprintln!(
        "water dHf: exact {:.6}, divide and conquer {:.6} kcal/mol",
        exact.heat_of_formation_kcal, approximate.heat_of_formation_kcal
    );
    assert!(
        (exact.energy_ev - approximate.energy_ev).abs() < 1.0e-6,
        "energies {} vs {}",
        exact.energy_ev,
        approximate.energy_ev
    );
    // The heat of formation goes through a different code path for the two drivers --
    // `run_pm7` computes it inline, divide and conquer through `heat_of_formation_from_total` --
    // so agreeing here is what says the second transcription of the reference sums is right.
    assert!(
        (exact.heat_of_formation_kcal - approximate.heat_of_formation_kcal).abs() < 1.0e-4,
        "heats of formation {} vs {}",
        exact.heat_of_formation_kcal,
        approximate.heat_of_formation_kcal
    );
    for (a, b) in exact.molecule.atoms.iter().zip(&approximate.molecule.atoms) {
        assert!(
            (a.position - b.position).norm() < 1.0e-3,
            "the two drivers relaxed to different geometries"
        );
    }
}

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn options() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-9,
        ..Pm7Options::default()
    }
}

/// A linear alkane `C_nH_{2n+2}` in a flat all-anti conformation.
///
/// Deliberately one-dimensional: a chain is the case divide and conquer is designed for, and it
/// makes the subsystem count grow while the subsystem size does not.
fn alkane(n_carbon: usize) -> Molecule {
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let mut lines = Vec::new();
    let (dx, dy) = (1.26, 0.44); // zig-zag with a 1.53 Å C–C bond
    for i in 0..n_carbon {
        let x = i as f64 * dx;
        let y = if i % 2 == 0 { 0.0 } else { dy };
        lines.push(format!("C {x} {y} 0.0"));
        lines.push(format!(
            "H {x} {} 0.89",
            y + (if i % 2 == 0 { -0.5 } else { 0.5 })
        ));
        lines.push(format!(
            "H {x} {} -0.89",
            y + (if i % 2 == 0 { -0.5 } else { 0.5 })
        ));
    }
    // Cap both ends.
    lines.push(format!("H {} 0.0 0.0", -1.09));
    lines.push(format!(
        "H {} {} 0.0",
        (n_carbon - 1) as f64 * dx + 1.09,
        if (n_carbon - 1) % 2 == 0 { 0.0 } else { dy }
    ));
    let body = lines.join("\n");
    let xyz = format!("{}\nalkane\n{body}\n", lines.len());
    let mut molecule = Molecule::from_xyz_str(&xyz, 0.0).unwrap();
    // `from_xyz_str` reads Ångström; the constant is only here to make the units explicit.
    let _ = a;
    molecule.charge = 0.0;
    molecule
}

#[test]
fn the_energy_converges_to_the_exact_scf_as_the_buffer_grows() {
    // The defining property. The buffer is the only approximation, so widening it has to walk the
    // answer monotonically onto the exact one — and the test reports the sequence rather than
    // asserting a single tolerance, because the *shape* of the convergence is what says the method
    // is right.
    let molecule = alkane(10);
    let exact = run_pm7(&molecule, &params(), &options()).expect("exact SCF");

    let mut errors = Vec::new();
    for buffer in [6.0, 10.0, 14.0, 20.0, 30.0] {
        let dandc = DandcOptions {
            buffer,
            core_size: 6,
            p_tol: 1.0e-8,
            max_scf: 400,
            ..DandcOptions::default()
        };
        let result = run_dandc(&molecule, &params(), &options(), &dandc).expect("D&C SCF");
        assert!(
            result.converged,
            "D&C did not converge at buffer {buffer} (error {:.3e})",
            result.density_error
        );
        errors.push((buffer, (result.electronic_ev - exact.electronic_ev).abs()));
    }
    let report: Vec<String> = errors
        .iter()
        .map(|(b, e)| format!("{b:.0} Bohr: {e:.3e} eV"))
        .collect();
    // The widest buffer must be much closer than the narrowest, and close in absolute terms.
    let (_, first) = errors[0];
    let (_, last) = errors[errors.len() - 1];
    assert!(
        last < first,
        "widening the buffer did not improve the energy: {}",
        report.join(", ")
    );
    assert!(
        last < 0.05,
        "at the widest buffer the electronic energy is still {last:.3e} eV off: {}",
        report.join(", ")
    );
}

#[test]
fn the_density_is_stored_sparsely() {
    // The precondition for linear scaling, stated as a number rather than an intention: the stored
    // density must be a small fraction of the dense matrix, and the fraction must *shrink* as the
    // system grows.
    let mut fractions = Vec::new();
    for n in [10usize, 20, 40] {
        let molecule = alkane(n);
        let dandc = DandcOptions {
            buffer: 10.0,
            core_size: 6,
            max_scf: 1,
            ..DandcOptions::default()
        };
        let result = run_dandc(&molecule, &params(), &options(), &dandc).expect("D&C SCF");
        let basis = pm7_rs::basis::Basis::build(&molecule, &params()).unwrap();
        let dense = basis.nao * basis.nao;
        fractions.push((n, result.density.stored_elements() as f64 / dense as f64));
    }
    let report: Vec<String> = fractions
        .iter()
        .map(|(n, f)| format!("C{n}: {:.1}%", 100.0 * f))
        .collect();
    for window in fractions.windows(2) {
        assert!(
            window[1].1 < window[0].1,
            "the stored fraction did not shrink as the system grew: {}",
            report.join(", ")
        );
    }
    assert!(
        fractions.last().unwrap().1 < 0.35,
        "the density is not sparse enough to scale: {}",
        report.join(", ")
    );
}

#[test]
fn the_subsystem_size_does_not_grow_with_the_system() {
    // The other precondition. If the largest subsystem grew, the per-subsystem diagonalization
    // would too, and the method would be cubic with extra steps.
    let mut sizes = Vec::new();
    for n in [10usize, 20, 40, 80] {
        let dandc = DandcOptions {
            buffer: 10.0,
            core_size: 6,
            max_scf: 1,
            ..DandcOptions::default()
        };
        let result = run_dandc(&alkane(n), &params(), &options(), &dandc).expect("D&C SCF");
        sizes.push((n, result.subsystems, result.largest_subsystem));
    }
    let report: Vec<String> = sizes
        .iter()
        .map(|(n, count, largest)| format!("C{n}: {count} subsystems, largest {largest}"))
        .collect();
    let baseline = sizes[0].2;
    for (_, _, largest) in &sizes {
        assert!(
            *largest <= baseline + 4,
            "the largest subsystem grew from {baseline}: {}",
            report.join(", ")
        );
    }
    // And the count has to grow roughly in proportion.
    assert!(
        sizes.last().unwrap().1 >= 6 * sizes[0].1,
        "the subsystem count did not grow with the system: {}",
        report.join(", ")
    );
}

#[test]
fn the_gradient_converges_to_the_exact_one_as_the_buffer_grows() {
    // The gradient is the fixed-density expression evaluated at the divide-and-conquer density, so
    // it inherits the density's error and adds a non-variational residual of its own. Both go away
    // with the buffer, and this shows the sequence rather than asserting one bound — the shape is
    // the evidence.
    use pm7_rs::dandc::dandc_derivatives;
    let molecule = alkane(8);
    let exact = pm7_rs::closed_form_gradient(&molecule, &params(), &options()).expect("exact");

    let mut errors = Vec::new();
    for buffer in [8.0, 14.0, 22.0, 34.0] {
        let dandc = DandcOptions {
            buffer,
            core_size: 6,
            p_tol: 1.0e-8,
            max_scf: 400,
            ..DandcOptions::default()
        };
        let result = run_dandc(&molecule, &params(), &options(), &dandc).expect("D&C");
        let d = dandc_derivatives(&molecule, &params(), &options(), &result).expect("derivatives");
        let worst = d
            .gradient
            .iter()
            .zip(&exact.gradient)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        errors.push((buffer, worst));
    }
    let report: Vec<String> = errors
        .iter()
        .map(|(b, e)| format!("{b:.0} Bohr: {e:.3e}"))
        .collect();
    assert!(
        errors.last().unwrap().1 < errors[0].1,
        "widening the buffer did not improve the gradient: {}",
        report.join(", ")
    );
    assert!(
        errors.last().unwrap().1 < 2.0e-3,
        "at the widest buffer the gradient is still {:.3e} eV/Bohr off: {}",
        errors.last().unwrap().1,
        report.join(", ")
    );
}

#[test]
fn a_periodic_cell_gets_a_stress_and_a_molecule_does_not() {
    use pm7_rs::cell::Cell;
    use pm7_rs::dandc::dandc_derivatives;
    use pm7_rs::math::Vec3;
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let chain = Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, 0.89 * a),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, -0.89 * a),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(2.55 * a, 0.0, 0.0)]).unwrap());
    let mut opts = options();
    opts.pbc = Some(pm7_rs::PbcOptions::default());
    let dandc = DandcOptions {
        buffer: 14.0,
        core_size: 3,
        max_scf: 200,
        ..DandcOptions::default()
    };
    let result = run_dandc(&chain, &params(), &opts, &dandc).expect("periodic D&C");
    let d = dandc_derivatives(&chain, &params(), &opts, &result).expect("derivatives");
    let stress = d.stress.expect("a periodic cell has a stress");
    assert!(
        stress.max_abs().is_finite() && stress.max_abs() > 0.0,
        "the periodic stress came out as {stress:?}"
    );

    let free = Molecule::new(chain.atoms.clone());
    let free_result = run_dandc(&free, &params(), &options(), &dandc).expect("molecular D&C");
    let free_d =
        dandc_derivatives(&free, &params(), &options(), &free_result).expect("derivatives");
    assert!(
        free_d.stress.is_none(),
        "a molecule was given a stress tensor"
    );
}

/// Least-squares slope of `log y` against `log x`.
fn log_log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy): (f64, f64) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x.ln(), sy + y.ln()));
    let (mx, my) = (sx / n, sy / n);
    let num: f64 = points
        .iter()
        .map(|(x, y)| (x.ln() - mx) * (y.ln() - my))
        .sum();
    let den: f64 = points.iter().map(|(x, _)| (x.ln() - mx).powi(2)).sum();
    num / den
}

#[test]
fn the_diagonalization_work_grows_linearly() {
    // The linear-scaling gate, stated so that it does not depend on the machine. Wall-clock times
    // belong in `tests/dandc_scaling.rs`, which reports rather than asserts; what is *decided*
    // here is the arithmetic the method commits to.
    //
    // `Σ_α n_α³` is the cost of the subsystem diagonalizations, which is what divide and conquer
    // exists to make linear, and `Σ_α n_α²` stands in for the Fock builds. Both must grow like
    // `N`, not like `N³` or `N²`: it is the *number* of subsystems that grows, never their size.
    let mut cubic = Vec::new();
    let mut quadratic = Vec::new();
    let mut stored = Vec::new();
    for n in [20usize, 40, 80, 160, 320] {
        let molecule = alkane(n);
        let atoms = molecule.atoms.len() as f64;
        let parts = pm7_rs::dandc::partition::partition(&molecule, 8, 12.0);
        let c: f64 = parts
            .subsystems
            .iter()
            .map(|s| (s.len() as f64).powi(3))
            .sum();
        let q: f64 = parts
            .subsystems
            .iter()
            .map(|s| (s.len() as f64).powi(2))
            .sum();
        cubic.push((atoms, c));
        quadratic.push((atoms, q));

        let dandc = DandcOptions {
            buffer: 12.0,
            core_size: 8,
            max_scf: 1,
            ..DandcOptions::default()
        };
        let result = run_dandc(&molecule, &params(), &options(), &dandc).expect("D&C");
        stored.push((atoms, result.density.stored_elements() as f64));
    }
    for (label, points, bound) in [
        ("diagonalization work Σn³", &cubic, 1.2),
        ("Fock work Σn²", &quadratic, 1.2),
        ("stored density", &stored, 1.2),
    ] {
        let slope = log_log_slope(points);
        assert!(
            slope <= bound,
            "{label} scales as N^{slope:.2}, over the {bound} gate: {:?}",
            points
                .iter()
                .map(|(n, v)| format!("{n:.0}: {v:.3e}"))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn an_open_shell_system_runs_unrestricted() {
    // UHF on the same terms as RHF, which is the requirement. A doublet radical is the smallest
    // honest test: the spin density has to survive the partitioning and reassembly.
    let molecule = Molecule::from_xyz_str(
        "5\nethyl radical\nC 0.0 0.0 0.0\nC 1.50 0.0 0.0\nH -0.5 0.9 0.0\nH -0.5 -0.9 0.0\n\
         H 2.05 0.92 0.0\n",
        0.0,
    )
    .unwrap();
    let mut opts = options();
    opts.multiplicity = 2;
    let dandc = DandcOptions {
        buffer: 25.0,
        core_size: 3,
        max_scf: 400,
        p_tol: 1.0e-8,
        ..DandcOptions::default()
    };
    let result = run_dandc(&molecule, &params(), &opts, &dandc).expect("D&C UHF");
    assert!(result.unrestricted, "the doublet was not run unrestricted");
    let spin = result.spin_density.as_ref().expect("a spin density");
    let total: f64 = (0..molecule.atoms.len()).map(|a| spin.population(a)).sum();
    assert!(
        (total - 1.0).abs() < 0.15,
        "the unpaired electron count came out as {total:.4} instead of 1"
    );
}

#[test]
fn a_periodic_cell_is_partitioned_with_buffers_drawn_from_images() {
    // The periodic path differs only in where the buffer atoms come from, so the thing to check
    // is that they *do* come from images — a subsystem of a small cell must be larger than the
    // cell itself.
    use pm7_rs::cell::Cell;
    use pm7_rs::math::Vec3;
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let molecule = Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, 0.89 * a),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, -0.89 * a),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(2.55 * a, 0.0, 0.0)]).unwrap());
    let mut opts = options();
    opts.pbc = Some(pm7_rs::PbcOptions::default());
    let dandc = DandcOptions {
        buffer: 12.0,
        core_size: 3,
        max_scf: 1,
        ..DandcOptions::default()
    };
    let result = run_dandc(&molecule, &params(), &opts, &dandc).expect("periodic D&C");
    assert!(
        result.largest_subsystem > molecule.atoms.len(),
        "a periodic subsystem has only {} atoms, so no images were pulled in",
        result.largest_subsystem
    );
}

/// The sparse gradient equals the one the dense path produced, to the last bit that matters.
///
/// `dandc_derivatives` reads the `O(N)` sparse density directly instead of calling `to_dense()`
/// first, which removed an `N_ao²` allocation — about 512 MB at two thousand atoms — from the one
/// step *after* the linear-scaling SCF had finished. This is the invariant that change rests on:
/// every read the pair loop makes is block-local, so serving them from a block store must give
/// the same number as serving them from a matrix. A block missing from the sparse store is a zero
/// the SCF already converged with, not an approximation introduced by the gradient.
#[test]
fn the_sparse_gradient_matches_the_densified_one() {
    let molecule = alkane(14);
    let dandc = DandcOptions {
        buffer: 9.0,
        ..DandcOptions::default()
    };
    let result = run_dandc(&molecule, &params(), &options(), &dandc).expect("divide and conquer");

    let mine = pm7_rs::dandc::dandc_derivatives(&molecule, &params(), &options(), &result)
        .expect("sparse derivatives");

    // The dense reference, built the way the code used to: densify, then run the ordinary
    // fixed-density pair loop.
    let basis = pm7_rs::basis::Basis::build(&molecule, &params()).expect("basis");
    let dense = result.density.to_dense(&basis);
    let reference =
        pm7_rs::gradient::electronic_gradient_fixed_density(&molecule, &params(), &basis, &dense)
            .expect("dense electronic gradient");
    let core = pm7_rs::repulsion::core_core_gradient(&molecule, &params()).expect("core-core");
    let correction = pm7_rs::gradient::correction_gradient_and_virial(&molecule, &options()).0;

    let mut worst = 0.0_f64;
    let mut scale = 1.0e-6_f64;
    for i in 0..molecule.atoms.len() {
        let want = reference[i] + core[i] + correction[i];
        let got = mine.gradient[i];
        for (a, b) in [(want.x, got.x), (want.y, got.y), (want.z, got.z)] {
            scale = scale.max(a.abs());
            worst = worst.max((a - b).abs());
        }
    }
    assert!(
        worst < 1.0e-9 * scale,
        "the sparse gradient differs from the densified one by {worst:.3e} eV/Bohr \
         (values up to {scale:.3e})"
    );
    // Not a comparison of two zeros.
    assert!(scale > 1.0e-3, "an all-zero gradient would pass vacuously");
}

/// **A buffer too large for the machine is refused, not aborted.**
///
/// `run_pm7`'s pre-flight memory guard is on the whole molecule's basis, which is useless here:
/// the dense work is per *subsystem*, and for a periodic cell a subsystem is larger than the cell,
/// because the buffer pulls in lattice images. A two-atom diamond cell has 8 AOs and passes every
/// check `run_pm7` makes; ask for a 15 Å buffer on its 2.5 Å lattice and the largest subsystem is
/// 2674 atoms and 10696 orbitals, wanting an estimated 34 GB.
///
/// What that used to do was consume 11.5 GB over several minutes and then abort the process on a
/// 4 MB allocation — no `Pm7Error`, no message, nothing naming the flag. `tests/cli_matrix.rs`
/// found it, because its invariant is that a refusal is a message and never a crash.
///
/// The assertion is on the *message*, not on the failure: failing is easy, and failing usefully is
/// the property that was missing.
#[test]
fn a_buffer_that_will_not_fit_is_refused_with_the_numbers() {
    let diamond = Molecule::from_xyz_str(
        "2\nLattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" \
         Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n\
         C 0.000000 0.000000 0.000000\nC 0.891750 0.891750 0.891750\n",
        0.0,
    )
    .unwrap();
    let dandc = DandcOptions {
        buffer: 15.0 * pm7_rs::constants::ANGSTROM_TO_BOHR,
        ..DandcOptions::default()
    };
    // A budget small enough that the estimate cannot fit it on any machine this runs on, so the
    // test measures the guard rather than the tester's RAM.
    let scf = Pm7Options {
        max_memory_mb: Some(64),
        pbc: Some(pm7_rs::PbcOptions::default()),
        ..Pm7Options::default()
    };
    let error = run_dandc(&diamond, &params(), &scf, &dandc)
        .expect_err("a 15 A buffer on a 2.5 A lattice cannot fit in 64 MB");
    let text = error.to_string();
    for wanted in ["subsystem", "buffer", "exact SCF"] {
        assert!(
            text.contains(wanted),
            "the refusal should name {wanted:?} so the user knows which knob to turn: {text}"
        );
    }
    // And it names the actual sizes, not just "too big".
    assert!(
        text.contains("atoms") && text.contains("orbitals"),
        "the refusal should report the subsystem it could not fit: {text}"
    );
}
