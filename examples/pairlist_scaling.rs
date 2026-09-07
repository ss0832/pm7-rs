// SPDX-License-Identifier: GPL-3.0-or-later
//! How long a periodic pair list takes as the cell grows, binned against the old nested scan.
//!
//! `cargo run --release --example pairlist_scaling`
//!
//! `tests/pair_list.rs` pins the *arithmetic* — the list is bit-identical to the nested-loop
//! enumeration and its length grows linearly. This prints the wall clock, which is the thing a
//! user notices and the thing no test should assert.
//!
//! **Compact `k × k × k` supercells**, not a cell elongated along one axis. The difference matters
//! more than it looks: `PairList` takes its image range from `cell_diameter`, the longest diagonal,
//! so a cell stretched 100× along `a₁` asks for ~10⁵ images in the *short* directions and both
//! implementations spend all their time there. That is a property of the margin heuristic, not of
//! the pair enumeration, and benchmarking it measures the wrong thing. (It is also a real
//! inefficiency for slab and wire cells, and is not fixed here.)
//!
//! The `build` column goes through `PairList::build`, which **dispatches**: it scans below 32
//! atoms and bins at or above. So the first two rows are the scan measured against itself plus the
//! cost of actually building the entries, and the speedup only becomes meaningful from the third
//! row on.

use std::time::Instant;

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::images::PairList;
use pm7_rs::{Atom, Cell, Molecule};

fn supercell(k: usize) -> Molecule {
    let a = 3.567 * ANGSTROM_TO_BOHR;
    let base = [
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ];
    let cell = Cell::new(&[base[0] * k as f64, base[1] * k as f64, base[2] * k as f64]).unwrap();
    let mut atoms = Vec::new();
    for i in 0..k {
        for j in 0..k {
            for l in 0..k {
                let shift = base[0] * i as f64 + base[1] * j as f64 + base[2] * l as f64;
                atoms.push(Atom {
                    z: 6,
                    position: shift,
                });
                atoms.push(Atom {
                    z: 6,
                    position: shift + (base[0] + base[1] + base[2]) * 0.25,
                });
            }
        }
    }
    Molecule::new(atoms).with_cell(cell)
}

fn main() {
    let cutoff = 7.0 * ANGSTROM_TO_BOHR;
    println!(
        "{:>7}  {:>10}  {:>12}  {:>12}  {:>8}",
        "atoms", "pairs", "build (ms)", "scan (ms)", "speedup"
    );
    for k in [1usize, 2, 3, 4, 5, 6] {
        let molecule = supercell(k);
        let n = molecule.atoms.len();
        let runs = if n < 100 { 20 } else { 3 };

        let start = Instant::now();
        let mut pairs = 0;
        for _ in 0..runs {
            pairs = PairList::build(&molecule, cutoff).pairs.len();
        }
        let binned = start.elapsed().as_secs_f64() * 1.0e3 / runs as f64;

        let start = Instant::now();
        let mut scanned = 0;
        for _ in 0..runs {
            scanned = nested_loop(&molecule, cutoff);
        }
        let scan = start.elapsed().as_secs_f64() * 1.0e3 / runs as f64;
        assert_eq!(pairs, scanned, "the two enumerations disagree");

        println!(
            "{n:>7}  {pairs:>10}  {binned:>12.3}  {scan:>12.3}  {:>8.2}x",
            scan / binned
        );
    }
}

/// The enumeration `PairList::periodic` used through 0.2.2: every `A ≤ B` pair against every image.
///
/// Kept here so the comparison is against a real implementation rather than an estimate.
///
/// It returns only the **count** and does not construct the `ImagePair` entries, so it is doing
/// strictly less work than `PairList::build` on the other side of the comparison. Every speedup
/// printed is therefore a lower bound. `tests/pair_list.rs` is what checks the two agree entry by
/// entry; this only asserts the counts match, as a guard against benchmarking two different things.
fn nested_loop(molecule: &Molecule, cutoff: f64) -> usize {
    let cell = molecule.cell.unwrap();
    let v = cell.vectors();
    let mut margin = 0.0_f64;
    for mask in 0..(1usize << v.len()) {
        let mut corner = Vec3::zero();
        for (k, a) in v.iter().enumerate() {
            if mask & (1 << k) != 0 {
                corner += *a;
            }
        }
        margin = margin.max(corner.norm());
    }
    let images = cell.image_indices(cutoff, margin);
    let cut2 = cutoff * cutoff;
    let positive = |t: [i32; 3]| t.iter().find(|&&c| c != 0).is_some_and(|&c| c > 0);
    let n = molecule.atoms.len();
    let mut count = 0;
    for a in 0..n {
        let pa = molecule.atoms[a].position;
        for b in a..n {
            let pb = molecule.atoms[b].position;
            let _ = cell.minimum_image(pb - pa);
            for &t in &images {
                if a == b && !positive(t) {
                    continue;
                }
                let d = pb + cell.translation(t) - pa;
                let r2 = d.norm2();
                if r2 > cut2 || r2 < 1.0e-20 {
                    continue;
                }
                count += 1;
            }
        }
    }
    count
}
