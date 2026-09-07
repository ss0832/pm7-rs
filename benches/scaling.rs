// SPDX-License-Identifier: GPL-3.0-or-later
//! Criterion benchmarks for the paths that dominate real runs.
//!
//! ```bash
//! cargo bench --bench scaling
//! cargo bench --bench scaling -- hessian     # one group
//! ```
//!
//! These measure the *hot* operations rather than whole workflows, because criterion's value is
//! its statistics and a 30-second workflow gets three samples. The whole-workflow timings live in
//! `tests/perf_report.rs`, which reports and asserts nothing, and the scaling exponents in
//! `tests/dandc_scaling.rs`.
//!
//! A benchmark run on a loaded machine measures the load. If the numbers look wrong, check what
//! else is running before believing them — that mistake cost an afternoon here once.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::PbcOptions;
use pm7_rs::{
    analytic_hessian, closed_form_gradient, run_pm7, DandcOptions, KMesh, Molecule, Pm7Options,
    Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

fn load(path: &str) -> Molecule {
    Molecule::from_xyz_str(&std::fs::read_to_string(path).unwrap(), 0.0).unwrap()
}

/// A linear alkane `C_nH_{2n+2}` in a plausible zig-zag, for the size series.
fn alkane(carbons: usize) -> Molecule {
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let mut atoms = Vec::new();
    let (cc, rise) = (1.26_f64, 0.51_f64);
    for i in 0..carbons {
        let x = i as f64 * cc;
        let y = if i % 2 == 0 { 0.0 } else { rise };
        atoms.push(pm7_rs::Atom {
            z: 6,
            position: Vec3::new(x, y, 0.0) * a0,
        });
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        for z in [0.88_f64, -0.88] {
            atoms.push(pm7_rs::Atom {
                z: 1,
                position: Vec3::new(x, y + 0.63 * sign, z) * a0,
            });
        }
    }
    // Cap both ends so the formula is right and the ends are not radicals.
    for (i, dx) in [(0usize, -1.09_f64), (carbons - 1, 1.09)] {
        let base = atoms[3 * i].position;
        atoms.push(pm7_rs::Atom {
            z: 1,
            position: base + Vec3::new(dx * a0, 0.0, 0.0),
        });
    }
    Molecule::new(atoms)
}

fn diamond(a_angstrom: f64) -> Molecule {
    let a = a_angstrom * pm7_rs::constants::ANGSTROM_TO_BOHR;
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

fn molecular(c: &mut Criterion) {
    let mut group = c.benchmark_group("molecular");
    group.sample_size(20);
    for path in ["examples/water.xyz", "examples/ethanol.xyz"] {
        let molecule = load(path);
        let name = path.trim_start_matches("examples/");
        let options = Pm7Options::default();
        group.bench_with_input(BenchmarkId::new("single_point", name), &molecule, |b, m| {
            b.iter(|| run_pm7(black_box(m), &params(), &options).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("gradient", name), &molecule, |b, m| {
            b.iter(|| closed_form_gradient(black_box(m), &params(), &options).unwrap())
        });
    }
    group.finish();
}

/// Size series, for the exponent rather than the absolute time.
fn size_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("alkane_series");
    group.sample_size(10);
    let options = Pm7Options::default();
    for carbons in [4usize, 8, 16, 32] {
        let molecule = alkane(carbons);
        group.bench_with_input(
            BenchmarkId::new("single_point", molecule.atoms.len()),
            &molecule,
            |b, m| b.iter(|| run_pm7(black_box(m), &params(), &options).unwrap()),
        );
    }
    group.finish();
}

/// The analytic Hessian, which is where the CPHF lives and where the time goes.
fn hessian(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian");
    // A Hessian on anything sizeable takes seconds; criterion needs to be told not to try for 100.
    group.sample_size(10);
    let options = Pm7Options::default();
    for carbons in [4usize, 8] {
        let molecule = alkane(carbons);
        group.bench_with_input(
            BenchmarkId::new("analytic", molecule.atoms.len()),
            &molecule,
            |b, m| b.iter(|| analytic_hessian(black_box(m), &params(), &options, 1e-4).unwrap()),
        );
    }
    group.finish();
}

/// The periodic path, and how it scales with the k mesh.
fn periodic(c: &mut Criterion) {
    let mut group = c.benchmark_group("periodic");
    group.sample_size(10);
    let molecule = diamond(3.567);
    for n in [1usize, 2, 4] {
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
        group.bench_with_input(
            BenchmarkId::new("single_point", format!("{n}x{n}x{n}")),
            &options,
            |b, o| b.iter(|| run_pm7(black_box(&molecule), &params(), o).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("gradient", format!("{n}x{n}x{n}")),
            &options,
            |b, o| b.iter(|| closed_form_gradient(black_box(&molecule), &params(), o).unwrap()),
        );
    }
    group.finish();
}

/// Divide and conquer against the exact SCF, on the size where they meet.
fn dandc(c: &mut Criterion) {
    let mut group = c.benchmark_group("dandc");
    group.sample_size(10);
    let options = Pm7Options::default();
    let dandc_options = DandcOptions::default();
    for carbons in [20usize, 40] {
        let molecule = alkane(carbons);
        let label = molecule.atoms.len();
        group.bench_with_input(BenchmarkId::new("exact", label), &molecule, |b, m| {
            b.iter(|| run_pm7(black_box(m), &params(), &options).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("dandc", label), &molecule, |b, m| {
            b.iter(|| pm7_rs::run_dandc(black_box(m), &params(), &options, &dandc_options).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, molecular, size_series, hessian, periodic, dandc);
criterion_main!(benches);
