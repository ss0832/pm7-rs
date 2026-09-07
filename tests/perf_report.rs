// SPDX-License-Identifier: GPL-3.0-or-later
//! Timings for `docs/performance.md`. Reports; asserts nothing.
//!
//! `cargo test --release --test perf_report -- --nocapture --ignored`

use std::time::Instant;

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::PbcOptions;
use pm7_rs::{
    analytic_hessian, closed_form_gradient, run_pm7, KMesh, Molecule, Pm7Options, Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// How many times a timed case is repeated; the report keeps the **minimum**.
///
/// A loaded machine can only slow a run down, never speed it up, so the minimum is the closest
/// thing to the machine's own number. The difference is not cosmetic: the same DFPT binary
/// measured 15.3 s and 19.2 s on consecutive runs of this file, which is wider than most of the
/// changes anyone would want to measure with it.
fn repeats() -> usize {
    std::env::var("PM7_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// [`time`], best of [`repeats`].
fn best<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let mut out = None;
    let mut seconds = f64::INFINITY;
    for _ in 0..repeats() {
        let (value, elapsed) = time(&mut f);
        seconds = seconds.min(elapsed);
        out = Some(value);
    }
    (out.expect("at least one repeat"), seconds)
}

fn load(path: &str) -> Molecule {
    let text = std::fs::read_to_string(path).unwrap();
    Molecule::from_xyz_str(&text, 0.0).unwrap()
}

fn diamond(a_ang: f64) -> Molecule {
    let a = a_ang * pm7_rs::constants::ANGSTROM_TO_BOHR;
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

#[test]
#[ignore = "reports timings; run explicitly"]
fn molecular() {
    println!("\n## Molecular (16 cores)\n");
    println!(
        "{:>16} {:>7} {:>12} {:>12} {:>12} {:>10}",
        "system", "atoms", "point (s)", "gradient (s)", "hessian (s)", "grad/pt"
    );
    for path in [
        "examples/water.xyz",
        "examples/ethanol.xyz",
        "examples/bench102.xyz",
    ] {
        let molecule = load(path);
        let options = Pm7Options::default();
        let (_, t_point) = time(|| run_pm7(&molecule, &params(), &options).unwrap());
        let (_, t_grad) = time(|| closed_form_gradient(&molecule, &params(), &options).unwrap());
        let (_, t_hess) = time(|| analytic_hessian(&molecule, &params(), &options, 1e-4).unwrap());
        println!(
            "{:>16} {:>7} {t_point:>12.4} {t_grad:>12.4} {t_hess:>12.4} {:>10.1}",
            path.trim_start_matches("examples/"),
            molecule.atoms.len(),
            t_grad / t_point
        );
    }
}

#[test]
#[ignore = "reports timings; run explicitly"]
fn periodic() {
    println!("\n## Periodic diamond, 2-atom primitive cell (16 cores)\n");
    println!(
        "{:>10} {:>6} {:>12} {:>12} {:>12} {:>12}",
        "mesh", "k pts", "point (s)", "grad (s)", "stress (s)", "E/atom (eV)"
    );
    let molecule = diamond(3.567);
    for n in [1usize, 2, 3, 4, 6] {
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: if n == 1 {
                    KMesh::Gamma
                } else {
                    KMesh::grid(n, n, n)
                },
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        let (scf, t_point) = time(|| run_pm7(&molecule, &params(), &options).unwrap());
        let (_, t_grad) = time(|| closed_form_gradient(&molecule, &params(), &options).unwrap());
        let (_, t_stress) = time(|| {
            let s = run_pm7(&molecule, &params(), &options).unwrap();
            pm7_rs::analytic_stress(&molecule, &params(), &options, &s).unwrap()
        });
        println!(
            "{:>10} {:>6} {t_point:>12.4} {t_grad:>12.4} {t_stress:>12.4} {:>12.4}",
            format!("{n}x{n}x{n}"),
            scf.n_kpoints.unwrap_or(1),
            scf.total_ev / 2.0
        );
    }
}

/// Where the analytic Hessian's time actually goes. Set `PM7_PROFILE=1` to get the table.
///
/// Reading the source and picking the loop that *looks* expensive is how you optimize a stage that
/// was 3 % of the run. This prints the ranking instead.
#[test]
#[ignore = "reports timings; run explicitly"]
fn hessian_profile() {
    let molecule = load("examples/bench102.xyz");
    let options = Pm7Options::default();
    let (_, seconds) = best(|| analytic_hessian(&molecule, &params(), &options, 1e-4).unwrap());
    println!(
        "\n## Analytic Hessian, {} atoms: {seconds:.3} s (best of {})\n",
        molecule.atoms.len(),
        repeats()
    );
    if std::env::var_os("PM7_PROFILE").is_some() {
        pm7_rs::profile::report_and_reset();
    } else {
        println!("(set PM7_PROFILE=1 for the stage breakdown)");
    }
}

/// A `n x 1 x 1` supercell of the diamond primitive cell, along the first lattice vector.
fn diamond_supercell(a_ang: f64, n: usize) -> Molecule {
    let base = diamond(a_ang);
    let cell = base.cell.unwrap();
    let vectors = cell.vectors();
    let a1 = vectors[0];
    let mut atoms = Vec::with_capacity(2 * n);
    for image in 0..n {
        let shift = a1 * image as f64;
        for atom in &base.atoms {
            atoms.push(pm7_rs::Atom {
                z: atom.z,
                position: atom.position + shift,
            });
        }
    }
    let super_cell = Cell::new(&[a1 * n as f64, vectors[1], vectors[2]]).unwrap();
    Molecule::new(atoms).with_cell(super_cell)
}

/// The perturbation path, which no benchmark reached before v0.2.2.
///
/// Two axes, because the response scales differently in each and the interesting term is the
/// product: the number of degrees of freedom (`3N`, one linear solve apiece) and the k mesh (the
/// response couples `k` with `k + q`, so the mesh is not folded and every point costs).
#[test]
#[ignore = "reports timings; run explicitly"]
fn dfpt() {
    use pm7_rs::dfpt::DfptOptions;

    println!("\n## Perturbation theory, diamond (16 cores)\n");
    println!(
        "{:>10} {:>7} {:>7} {:>10} {:>12} {:>7} {:>16}",
        "cell", "atoms", "mesh", "q", "D(q) (s)", "iters", "optical (cm-1)"
    );
    let settings = DfptOptions::default();
    for (cells, mesh) in [(1usize, 2usize), (1, 3), (1, 4), (2, 2), (3, 2), (4, 2)] {
        let molecule = diamond_supercell(3.567, cells);
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(mesh, mesh, mesh),
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        for q in [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0]] {
            let (result, seconds) = best(|| {
                pm7_rs::dynamical_matrix_dfpt(&molecule, &params(), &options, q, &settings).unwrap()
            });
            let mut freqs = result.frequencies_cm().unwrap();
            freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{:>10} {:>7} {:>7} {:>10} {seconds:>12.3} {:>7} {:>16.1}",
                format!("{cells}x1x1"),
                molecule.atoms.len(),
                format!("{mesh}^3"),
                if q[0] == 0.0 { "0" } else { "1/3" },
                result.iterations,
                freqs[freqs.len() - 1]
            );
        }
    }

    println!("\n## Born charges and the dielectric tensor, diamond (16 cores)\n");
    println!(
        "{:>10} {:>7} {:>7} {:>12} {:>7} {:>12}",
        "cell", "atoms", "mesh", "field (s)", "iters", "eps_xx"
    );
    for (cells, mesh) in [(1usize, 2usize), (1, 3), (2, 2), (3, 2)] {
        let molecule = diamond_supercell(3.567, cells);
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(mesh, mesh, mesh),
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        let (out, seconds) = best(|| {
            pm7_rs::born_and_dielectric(&molecule, &params(), &options, &settings).unwrap()
        });
        println!(
            "{:>10} {:>7} {:>7} {seconds:>12.3} {:>7} {:>12.4}",
            format!("{cells}x1x1"),
            molecule.atoms.len(),
            format!("{mesh}^3"),
            out.iterations,
            out.dielectric.get(0, 0)
        );
    }
    if std::env::var_os("PM7_PROFILE").is_some() {
        pm7_rs::profile::report_and_reset();
    } else {
        println!("\n(set PM7_PROFILE=1 for the stage breakdown)");
    }
}

#[test]
#[ignore = "reports timings; run explicitly"]
fn phonons() {
    println!("\n## Phonons, diamond (16 cores)\n");
    println!(
        "{:>10} {:>8} {:>14} {:>16}",
        "supercell", "atoms", "force csts (s)", "optical (cm-1)"
    );
    let molecule = diamond(3.567);
    let options = Pm7Options {
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    for n in [1usize, 2, 3] {
        let (fc, t) =
            time(|| pm7_rs::force_constants(&molecule, &params(), &options, [n, n, n]).unwrap());
        let mut freqs = fc.frequencies_cm([0.0, 0.0, 0.0]).unwrap();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:>10} {:>8} {t:>14.3} {:>16.1}",
            format!("{n}x{n}x{n}"),
            2 * n * n * n,
            freqs[freqs.len() - 1]
        );
    }
}

/// How the k-point gradient and stress scale with the **mesh**, at a fixed cell.
///
/// `docs/performance.md` names the long-range exchange derivative as the largest remaining
/// `O(mesh^2)` structure: it walks the Born-von Karman residue classes and, for each, runs a
/// shifted pair Ewald over the supercell, whose reciprocal lattice is itself proportional to the
/// class count. The claim is plausible and, until this test, unmeasured.
///
/// The atom count is held fixed and only the mesh moves, so the exponent reported here is in the
/// class count `C = n^3` alone. A slope near 2 confirms the `O(C^2)` reading; a slope near 1 says
/// the cost is somewhere else and the rewrite would buy nothing.
#[test]
#[ignore]
fn kpoint_derivative_scaling() {
    let params = params();
    let molecule = diamond(3.567);
    println!("k-point derivatives vs mesh (2 atoms, diamond); exponent is on the stress");
    println!(
        "{:>5}  {:>6}  {:>10}  {:>10}  {:>10}",
        "mesh", "C", "scf s", "grad s", "stress s"
    );

    let mut points: Vec<(f64, f64)> = Vec::new();
    for n in [2usize, 3, 4, 5, 6, 7] {
        let options = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(n, n, n),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (scf, scf_s) = best(|| run_pm7(&molecule, &params, &options).unwrap());
        // The stress takes the converged SCF, so this times derivative work and nothing else.
        // The gradient is reported too, but it runs its own SCF and so cannot be the measurement:
        // subtracting one best-of-N from another leaves the difference of two noise floors.
        let (_, stress_s) = best(|| {
            pm7_rs::analytic_stress(&molecule, &params, &options, &scf)
                .unwrap()
                .stress
                .get(0, 0)
        });
        let (_, grad_s) = best(|| {
            closed_form_gradient(&molecule, &params, &options)
                .unwrap()
                .max_gradient
        });
        let c = (n * n * n) as f64;
        println!(
            "{n:>5}  {:>6.0}  {scf_s:>10.4}  {grad_s:>10.4}  {stress_s:>10.4}",
            c
        );
        points.push((c, stress_s));
    }

    // **Local** slopes, not one fit over the whole range.
    //
    // A single least-squares line through all of these reports 1.01 and is wrong about the thing
    // it is being asked. The small meshes sit on a floor of costs that do not scale with the mesh
    // at all (the SCF, the short-range pair loop), so a global fit averages the floor together
    // with the asymptote and lands halfway between them. The exponent that matters is the one the
    // last pair of points shows.
    println!("{:>16}  {:>8}", "C ratio", "exponent");
    for pair in points.windows(2) {
        let (c0, t0) = pair[0];
        let (c1, t1) = pair[1];
        println!(
            "{:>7.0} -> {:<5.0}  {:>8.2}",
            c0,
            c1,
            (t1 / t0).ln() / (c1 / c0).ln()
        );
    }
}
