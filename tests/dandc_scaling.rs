// SPDX-License-Identifier: GPL-3.0-or-later
//! Reported (not asserted) numbers for the divide-and-conquer scaling record.
//!
//! Run with `cargo test --release --test dandc_scaling -- --nocapture --ignored`.

use std::time::Instant;

use pm7_rs::dandc::{run_dandc, DandcOptions};
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn options() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-10,
        p_tol: 1.0e-8,
        max_scf: 400,
        ..Pm7Options::default()
    }
}

fn alkane(n_carbon: usize) -> Molecule {
    let mut lines = Vec::new();
    let (dx, dy) = (1.26, 0.44);
    for i in 0..n_carbon {
        let x = i as f64 * dx;
        let y = if i % 2 == 0 { 0.0 } else { dy };
        let hy = y + (if i % 2 == 0 { -0.5 } else { 0.5 });
        lines.push(format!("C {x} {y} 0.0"));
        lines.push(format!("H {x} {hy} 0.89"));
        lines.push(format!("H {x} {hy} -0.89"));
    }
    lines.push("H -1.09 0.0 0.0".to_string());
    lines.push(format!(
        "H {} {} 0.0",
        (n_carbon - 1) as f64 * dx + 1.09,
        if (n_carbon - 1) % 2 == 0 { 0.0 } else { dy }
    ));
    let xyz = format!("{}\nalkane\n{}\n", lines.len(), lines.join("\n"));
    Molecule::from_xyz_str(&xyz, 0.0).unwrap()
}

/// Least-squares slope of `log(y)` against `log(x)`.
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
#[ignore = "reports timings; run explicitly"]
fn scaling_series() {
    let dandc = DandcOptions {
        // Past the 7 Å feather range, where every interaction the buffer excludes is *exactly* a
        // monopole. Below it the accuracy falls off a cliff — see `docs/divide_and_conquer.md`.
        buffer: 15.0,
        core_size: 8,
        p_tol: 1.0e-7,
        max_scf: 400,
        ..DandcOptions::default()
    };
    println!(
        "{:>6} {:>7} {:>12} {:>10} {:>12} {:>10} {:>10} {:>9}",
        "C", "atoms", "D&C (s)", "s/iter", "exact (s)", "dE (eV)", "sparse %", "subsys"
    );
    let mut dandc_points = Vec::new();
    let mut per_iter_points = Vec::new();
    let mut exact_points = Vec::new();
    for n in [10usize, 20, 40, 80, 160, 320, 640] {
        let molecule = alkane(n);
        let atoms = molecule.atoms.len() as f64;

        let t0 = Instant::now();
        let d = run_dandc(&molecule, &params(), &options(), &dandc).expect("D&C");
        let dt = t0.elapsed().as_secs_f64();
        dandc_points.push((atoms, dt));
        let per_iter = dt / d.iterations as f64;
        per_iter_points.push((atoms, per_iter));

        // The exact SCF is only run while it is still affordable.
        let (exact_time, delta) = if n <= 160 {
            let t1 = Instant::now();
            let e = run_pm7(&molecule, &params(), &options()).expect("exact");
            let et = t1.elapsed().as_secs_f64();
            exact_points.push((atoms, et));
            (et, (d.electronic_ev - e.electronic_ev).abs())
        } else {
            (f64::NAN, f64::NAN)
        };

        let basis = pm7_rs::basis::Basis::build(&molecule, &params()).unwrap();
        let sparse = 100.0 * d.density.stored_elements() as f64 / (basis.nao * basis.nao) as f64;
        println!(
            "{n:>6} {:>7} {dt:>12.3} {per_iter:>10.4} {exact_time:>12.3} {delta:>10.4} \
             {sparse:>9.1}% {:>9}   ({} iters)",
            molecule.atoms.len(),
            d.subsystems,
            d.iterations
        );
    }
    let big: Vec<(f64, f64)> = dandc_points
        .iter()
        .copied()
        .filter(|(n, _)| *n >= 200.0)
        .collect();
    let big_iter: Vec<(f64, f64)> = per_iter_points
        .iter()
        .copied()
        .filter(|(n, _)| *n >= 200.0)
        .collect();
    println!(
        "\nlog-log slope: D&C {:.2} (>=200 atoms: {:.2}), per iteration {:.2}, exact {:.2}",
        log_log_slope(&dandc_points),
        log_log_slope(&big),
        log_log_slope(&big_iter),
        log_log_slope(&exact_points)
    );
}
