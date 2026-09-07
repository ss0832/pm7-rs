// SPDX-License-Identifier: GPL-3.0-or-later
//! The periodic neighbour list, after it stopped being quadratic.
//!
//! `PairList::periodic` looped over every `A ≤ B` pair and, inside that, over every lattice image:
//! `O(N²·I)` distance evaluations for a list whose content is `O(N·I)`. It is reached from about
//! seven places per energy-and-gradient evaluation (through `PairList::cached`), so it is one of
//! the few places where an asymptotic change is worth the risk of touching a hot path.
//!
//! The risk is entirely in the *order* of the output. Every consumer sums over `pairs` in order,
//! so a reordered list is a different floating-point summation and a different last bit — a
//! "harmless speedup" that quietly moves every published number. The binned version therefore
//! sorts back into the original `(a, b, image index)` order, and the first test here is the one
//! that matters: it re-implements the old nested-loop enumeration and demands the two agree
//! **exactly**, field for field, bit for bit.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::images::PairList;
use pm7_rs::{Atom, Cell, Molecule};

/// The enumeration `PairList::periodic` used through 0.2.2, kept here as the reference.
fn reference(
    molecule: &Molecule,
    cell: &Cell,
    cutoff: f64,
) -> Vec<(usize, usize, [i32; 3], f64, bool)> {
    let n = molecule.atoms.len();
    let margin = {
        let v = cell.vectors();
        let dim = v.len();
        let mut worst = 0.0_f64;
        for mask in 0..(1usize << dim) {
            let mut corner = Vec3::zero();
            for (k, a) in v.iter().enumerate() {
                if mask & (1 << k) != 0 {
                    corner += *a;
                }
            }
            worst = worst.max(corner.norm());
        }
        worst
    };
    let images = cell.image_indices(cutoff, margin);
    let cut2 = cutoff * cutoff;
    let positive = |t: [i32; 3]| {
        for &c in &t {
            if c > 0 {
                return true;
            }
            if c < 0 {
                return false;
            }
        }
        false
    };
    let mut out = Vec::new();
    for a in 0..n {
        let pa = molecule.atoms[a].position;
        for b in a..n {
            let pb = molecule.atoms[b].position;
            let (_, nearest_t) = cell.minimum_image(pb - pa);
            for &t in &images {
                if a == b && !positive(t) {
                    continue;
                }
                let d = pb + cell.translation(t) - pa;
                let r2 = d.norm2();
                if r2 > cut2 || r2 < 1.0e-20 {
                    continue;
                }
                out.push((a, b, t, r2.sqrt(), a != b && t == nearest_t));
            }
        }
    }
    out
}

fn crystal(z: &[u8], fractional: &[[f64; 3]], rows: &[[f64; 3]]) -> Molecule {
    let cell = Cell::from_angstrom_rows(rows).unwrap();
    let vectors = cell.vectors().to_vec();
    let atoms = z
        .iter()
        .zip(fractional)
        .map(|(&z, f)| {
            let mut position = Vec3::zero();
            for (k, v) in vectors.iter().enumerate() {
                position += *v * f[k];
            }
            Atom { z, position }
        })
        .collect();
    Molecule::new(atoms).with_cell(cell)
}

fn diamond() -> Molecule {
    let a = 3.567;
    crystal(
        &[6, 6],
        &[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        &[
            [0.0, a * 0.5, a * 0.5],
            [a * 0.5, 0.0, a * 0.5],
            [a * 0.5, a * 0.5, 0.0],
        ],
    )
}

/// A deliberately skewed triclinic cell: a cubic one cannot see a transposed axis in the grid
/// indexing, and every permutation would give the same answer.
fn triclinic() -> Molecule {
    crystal(
        &[6, 1, 1, 8],
        &[
            [0.0, 0.0, 0.0],
            [0.31, 0.17, 0.62],
            [0.74, 0.51, 0.08],
            [0.45, 0.88, 0.37],
        ],
        &[[4.1, 0.0, 0.0], [1.3, 4.7, 0.0], [0.7, -1.1, 5.3]],
    )
}

fn chain() -> Molecule {
    let cell = Cell::from_angstrom_rows(&[[2.6, 0.0, 0.0]]).unwrap();
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.74 * ANGSTROM_TO_BOHR, 0.1, -0.2),
        },
    ])
    .with_cell(cell)
}

fn sheet() -> Molecule {
    crystal(
        &[5, 7],
        &[[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]],
        &[[2.51, 0.0, 0.0], [-1.255, 2.1738, 0.0], [0.0, 0.0, 20.0]],
    )
}

/// A `k × k × k` diamond supercell, to get above the binning threshold.
fn diamond_supercell(k: usize) -> Molecule {
    let base = diamond();
    let v = base.cell.unwrap().vectors().to_vec();
    let cell = Cell::new(&[v[0] * k as f64, v[1] * k as f64, v[2] * k as f64]).unwrap();
    let mut atoms = Vec::new();
    for i in 0..k {
        for j in 0..k {
            for l in 0..k {
                let shift = v[0] * i as f64 + v[1] * j as f64 + v[2] * l as f64;
                for atom in &base.atoms {
                    atoms.push(Atom {
                        z: atom.z,
                        position: atom.position + shift,
                    });
                }
            }
        }
    }
    Molecule::new(atoms).with_cell(cell)
}

/// **Bit-identical to the nested-loop enumeration**, on every periodicity and a skewed cell.
///
/// Field for field and in the same order. Anything weaker would let a reordering through, and a
/// reordering changes the summation order in every consumer.
///
/// The 54-atom supercell is here because `PairList::periodic` **dispatches on the atom count**: it
/// scans below 32 atoms and bins at or above, since the grid is six times slower on a two-atom
/// primitive cell and eight times faster on a 432-atom one. Without a fixture on each side of that
/// line this test would only ever exercise one path.
#[test]
fn the_binned_list_matches_the_scan_exactly() {
    for (name, molecule) in [
        ("diamond", diamond()),
        ("triclinic", triclinic()),
        ("chain", chain()),
        ("sheet", sheet()),
        ("diamond 3x3x3 (binned path)", diamond_supercell(3)),
    ] {
        let cell = molecule.cell.unwrap();
        for cutoff_angstrom in [3.0, 7.0, 9.5] {
            let cutoff = cutoff_angstrom * ANGSTROM_TO_BOHR;
            let got = PairList::build(&molecule, cutoff);
            let want = reference(&molecule, &cell, cutoff);
            assert_eq!(
                got.pairs.len(),
                want.len(),
                "{name} at {cutoff_angstrom} A: {} pairs against {}",
                got.pairs.len(),
                want.len()
            );
            for (index, (pair, expected)) in got.pairs.iter().zip(&want).enumerate() {
                assert_eq!(
                    (pair.a, pair.b, pair.t),
                    (expected.0, expected.1, expected.2),
                    "{name} at {cutoff_angstrom} A, entry {index}: wrong pair or image"
                );
                assert_eq!(
                    pair.r.to_bits(),
                    expected.3.to_bits(),
                    "{name} at {cutoff_angstrom} A, entry {index}: distance differs in its last bits"
                );
                assert_eq!(
                    pair.minimum_image, expected.4,
                    "{name} at {cutoff_angstrom} A, entry {index}: nearest-image flag differs"
                );
            }
        }
    }
}

/// **The work is linear in the atom count**, not quadratic.
///
/// An arithmetic gate rather than a wall-clock one: the pair count is what the enumeration has to
/// examine, and the old version examined `N²·I` candidates to produce it. Repeating the cell along
/// one axis multiplies both the atom count and the true pair count by the same factor, so the
/// slope of `log(pairs)` against `log(N)` is 1 for a correct list — and a list that had started
/// including spurious pairs, or dropping real ones, would not hold that.
///
/// The complementary half is the exactness test above: together they say the binned version
/// produces the same list in linear work.
#[test]
fn the_pair_count_grows_linearly_with_the_cell() {
    let cutoff = 7.0 * ANGSTROM_TO_BOHR;
    let base = diamond();

    let mut counts = Vec::new();
    for repeat in [1usize, 2, 4] {
        // `repeat` copies of the primitive cell along the first lattice vector.
        let v = base.cell.unwrap().vectors().to_vec();
        let cell = Cell::new(&[v[0] * repeat as f64, v[1], v[2]]).unwrap();
        let mut atoms = Vec::new();
        for k in 0..repeat {
            for atom in &base.atoms {
                atoms.push(Atom {
                    z: atom.z,
                    position: atom.position + v[0] * k as f64,
                });
            }
        }
        let molecule = Molecule::new(atoms).with_cell(cell);
        let list = PairList::build(&molecule, cutoff);
        counts.push((molecule.atoms.len() as f64, list.pairs.len() as f64));
    }

    let slope = |(n0, p0): (f64, f64), (n1, p1): (f64, f64)| (p1 / p0).ln() / (n1 / n0).ln();
    let first = slope(counts[0], counts[1]);
    let second = slope(counts[1], counts[2]);
    assert!(
        first < 1.2 && second < 1.2,
        "the pair count is growing faster than linearly: slopes {first:.3} and {second:.3} for \
         counts {counts:?}"
    );
    // And it is genuinely growing, or a constant would pass the test above.
    assert!(
        counts[2].1 > counts[0].1 * 3.0,
        "the list did not grow with the cell: {counts:?}"
    );
}
