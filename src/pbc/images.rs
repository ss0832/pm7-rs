// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic image pair lists.
//!
//! Every two-centre term in PM7 — resonance `β·S`, electron–core `e1b`/`e2a`, the two-electron
//! block, core–core repulsion, dispersion, and the PM7-HH repulsion — depends only on the
//! displacement `d = R_B − R_A + T` for some lattice translation `T`. A [`PairList`] is the
//! enumeration of those `(A, B, T)` triples within a cutoff, computed once per geometry and
//! reused by the Fock build, the gradient, the stress virial, and the force constants.
//!
//! Two invariants make the periodic paths behave like the molecular ones:
//!
//! * **No double counting.** A pair is emitted once. For `A < B` every translation is kept; for
//!   `A == B` only half the translations are kept (`T > 0` in a fixed lexicographic order), which
//!   is how an atom interacts with its own images exactly once. `T = 0` with `A == B` is a
//!   self-interaction and is never emitted.
//! * **Deterministic order.** Entries are produced in a fixed order (`A`, then `B`, then `T`
//!   lexicographically), so a sum over the list accumulates identically on every run and thread
//!   count. This is the periodic counterpart of the bit-identical guarantee the molecular code
//!   already makes.
//!
//! The molecular case is not a special case bolted on afterwards: `Cell::image_range` returns
//! all-zero for a molecule, so a `None` cell yields exactly the `A < B, T = 0` list the v0.1.x
//! code enumerated by hand.

use crate::cell::Cell;
use crate::math::Vec3;
use crate::system::Molecule;

/// One periodic image pair.
#[derive(Clone, Copy, Debug)]
pub struct ImagePair {
    /// Index of the first atom, in the central cell.
    pub a: usize,
    /// Index of the second atom, in the cell displaced by `t`.
    pub b: usize,
    /// Integer lattice translation applied to atom `b`.
    pub t: [i32; 3],
    /// Displacement `R_b + T − R_a`, in Bohr.
    pub d: Vec3,
    /// `|d|`, in Bohr.
    pub r: f64,
    /// Weight for the energy sum.
    ///
    /// Always `1.0` with the enumeration used here: a self-pair `(A, A, T)` already stands for
    /// both `T` and `−T` because only half the translations are emitted, so no further halving
    /// is needed. (MOPAC reaches the same place differently — it sums *all* translations for the
    /// self pair and multiplies by `one = 0.5D0`, `solrot.F90:66-67`.) The field is kept so a
    /// caller that wants MOPAC's enumeration can supply its own weights without every consumer
    /// having to special-case self pairs.
    pub weight: f64,
    /// True when this is the **nearest** image of the `(a, b)` pair.
    ///
    /// Exchange is restricted to these, and the reason is correctness rather than speed. At the
    /// Γ point the real-space density matrix `P(T)` is the *same* for every lattice translation
    /// — a 1×1×1 Born–von Kármán cell has no decay to represent — so summing
    /// `−½ P(μ_A, λ_B) (μν|λσ)_T` over images adds a term that grows with the cutoff instead of
    /// converging. MOPAC does the same thing: `solrot.F90:94`, "K integrals apply only to
    /// nearest pair", and it zeroes the self-image exchange outright (`solrot.F90:101`).
    /// Coulomb is unaffected — that one *is* a convergent lattice sum, handled by Ewald.
    pub minimum_image: bool,
}

impl ImagePair {
    /// `true` when this entry is an atom interacting with one of its own images.
    #[inline]
    pub fn is_self_image(&self) -> bool {
        self.a == self.b
    }
}

/// All image pairs of a system within a cutoff.
/// What makes two `PairList` requests the same request.
///
/// Bit patterns rather than values, so `NaN` and `-0.0` compare the way a cache needs them to
/// (reflexively, and distinctly from `+0.0`) instead of the way `f64` comparison does. A geometry
/// is the same geometry only if every coordinate is bit-for-bit identical.
#[derive(PartialEq, Eq, Debug)]
struct PairKey {
    cutoff: u64,
    /// The lattice vectors, however many there are. **Not** a fixed `[u64; 9]`:
    /// [`Cell::vectors`] returns one entry per periodic direction, so a chain has one and a slab
    /// two, and indexing past that panics. Keeping it variable-length also makes the
    /// dimensionality part of the key, which it must be — a 1-D and a 2-D cell sharing their first
    /// lattice vector are different systems.
    cell: Option<Vec<[u64; 3]>>,
    positions: Vec<[u64; 3]>,
}

impl PairKey {
    fn of(molecule: &Molecule, cutoff: f64) -> Self {
        let bits = |v: &crate::math::Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
        Self {
            cutoff: cutoff.to_bits(),
            cell: molecule
                .cell
                .map(|c| c.vectors().iter().map(bits).collect()),
            positions: molecule.atoms.iter().map(|a| bits(&a.position)).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PairList {
    pub pairs: Vec<ImagePair>,
    /// The cutoff (Bohr) this list was built with.
    pub cutoff: f64,
}

impl PairList {
    /// Build the pair list for `molecule` within `cutoff` Bohr.
    ///
    /// For a molecule (`cell: None`) the cutoff is ignored and every `A < B` pair is emitted, so
    /// the existing all-pairs behaviour is preserved exactly. Screening a *molecular* run by
    /// distance would silently change published numbers; that is opt-in elsewhere, not here.
    pub fn build(molecule: &Molecule, cutoff: f64) -> Self {
        match molecule.cell {
            None => Self::molecular(molecule),
            Some(cell) => {
                // An infinite cutoff is meaningful for a molecule ("every pair") and meaningless
                // for a lattice, where it asks for infinitely many images. Catching it here
                // turns a caller that forgot to thread `PbcOptions` through into an immediate,
                // named failure instead of a multi-gigabyte allocation.
                assert!(
                    cutoff.is_finite() && cutoff > 0.0,
                    "a periodic pair list needs a finite positive cutoff, got {cutoff}; \
                     the caller is probably missing its PbcOptions"
                );
                Self::periodic(molecule, &cell, cutoff)
            }
        }
    }

    /// [`PairList::build`], memoized on the exact geometry and cutoff.
    ///
    /// One energy-and-gradient evaluation builds this list from about seven places — the core
    /// Hamiltonian, core–core repulsion and its gradient, dispersion energy and gradient, the H–H
    /// repulsion, and the electronic gradient — each rebuilding the same enumeration for the same
    /// atoms. For a molecule that is `O(N²)` distance evaluations apiece; for a periodic cell it
    /// also re-enumerates the images.
    ///
    /// # Why this is safe to memoize
    ///
    /// The key is the **exact bit pattern** of every position, the cell, and the cutoff, compared
    /// element by element — not a hash, so there is no collision to reason about. A geometry that
    /// differs in the last bit misses, which is the conservative direction: an optimizer or an MD
    /// step gets a fresh list, and only genuinely repeated geometries hit.
    ///
    /// The cache is **thread-local**, so there is no lock and no way for one thread's geometry to
    /// be handed to another. `PairList` has no `&mut self` method, so sharing one behind an `Arc`
    /// cannot let a caller mutate what another is reading.
    ///
    /// Four slots, because a single evaluation legitimately wants several different cutoffs at
    /// once (the short-range cutoff, the dispersion cutoff, the H–H cutoff) and one slot would
    /// thrash between them.
    pub fn cached(molecule: &Molecule, cutoff: f64) -> std::sync::Arc<Self> {
        const SLOTS: usize = 4;
        thread_local! {
            static CACHE: std::cell::RefCell<Vec<(PairKey, std::sync::Arc<PairList>)>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        let key = PairKey::of(molecule, cutoff);
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(index) = cache.iter().position(|(k, _)| *k == key) {
                // Move to front, so the four slots behave as an LRU rather than evicting whichever
                // entry happens to sit at the end.
                let hit = cache.remove(index);
                let list = std::sync::Arc::clone(&hit.1);
                cache.insert(0, hit);
                return list;
            }
            let list = std::sync::Arc::new(Self::build(molecule, cutoff));
            cache.insert(0, (key, std::sync::Arc::clone(&list)));
            cache.truncate(SLOTS);
            list
        })
    }

    fn molecular(molecule: &Molecule) -> Self {
        let n = molecule.atoms.len();
        let mut pairs = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for a in 0..n {
            for b in (a + 1)..n {
                let d = molecule.atoms[b].position - molecule.atoms[a].position;
                pairs.push(ImagePair {
                    a,
                    b,
                    t: [0, 0, 0],
                    d,
                    r: d.norm(),
                    weight: 1.0,
                    minimum_image: true,
                });
            }
        }
        Self {
            pairs,
            cutoff: f64::INFINITY,
        }
    }

    /// Below this many atoms, enumerate by scanning; at or above it, bin.
    ///
    /// **A measured crossover, not a guess.** Diamond supercells at a 7 Å cutoff, calling the two
    /// constructors directly (`cargo test --release --lib -- --ignored --nocapture
    /// pair_list_crossover`):
    ///
    /// | atoms | scan | grid | grid/scan |
    /// |---|---|---|---|
    /// | 2 | 0.159 ms | 1.294 ms | **8.16×** |
    /// | 16 | 2.81 ms | 3.33 ms | 1.18× |
    /// | 54 | 29.5 ms | 14.2 ms | 0.48× |
    /// | 128 | 119 ms | 39.9 ms | 0.33× |
    /// | 250 | 480 ms | 94.0 ms | 0.20× |
    ///
    /// The grid wins asymptotically and **loses badly on a small cell** — eight times slower on a
    /// two-atom primitive cell, which is what most of this crate's own tests and a great deal of
    /// ordinary use are. The reason is not subtle: the scan costs `N²/2` distance evaluations per
    /// image and the grid costs `N` bucket queries, and a bucket query is worth ten or twenty
    /// distance evaluations, so the grid only pays once `N/2` exceeds that. The crossover is
    /// between 16 and 54 atoms, which is where 32 sits. Shipping the grid unconditionally would
    /// have been a regression sold as an optimization.
    ///
    /// The table this used to carry was measured through `PairList::build`, which *dispatches* —
    /// so its sub-threshold rows were the scan timed against a count-only reimplementation of
    /// itself, and said nothing about the grid at all. They happened to point the same way. The
    /// numbers above are the two paths.
    ///
    /// Both paths produce **bit-identical** lists (`tests/pair_list.rs` checks it on either side of
    /// this threshold), so this is a pure performance dispatch: no answer depends on which side of
    /// it a system falls.
    const BINNING_THRESHOLD: usize = 32;

    fn periodic(molecule: &Molecule, cell: &Cell, cutoff: f64) -> Self {
        if molecule.atoms.len() < Self::BINNING_THRESHOLD {
            Self::periodic_by_scan(molecule, cell, cutoff)
        } else {
            Self::periodic_binned(molecule, cell, cutoff)
        }
    }

    /// The nested scan: every `A <= B` pair against every image. `O(N^2 * I)`.
    ///
    /// Kept, rather than replaced, because it is faster than the grid below
    /// [`Self::BINNING_THRESHOLD`] atoms. See there for the measurements.
    fn periodic_by_scan(molecule: &Molecule, cell: &Cell, cutoff: f64) -> Self {
        let n = molecule.atoms.len();
        // The margin covers the intra-cell separation of the two atoms: an atom at one corner
        // and an atom at the opposite corner are already `diameter` apart before any translation.
        let margin = cell_diameter(cell);
        let images = cell.image_indices(cutoff, margin);
        let cut2 = cutoff * cutoff;
        let mut pairs = Vec::new();
        for a in 0..n {
            let pa = molecule.atoms[a].position;
            for b in a..n {
                let pb = molecule.atoms[b].position;
                // The nearest image of this atom pair, found once so it can be flagged below.
                // Ties (two images at the same distance, as at a zone boundary) go to the first
                // in the fixed enumeration order, which keeps the flag deterministic.
                let (_, nearest_t) = cell.minimum_image(pb - pa);
                let first = pairs.len();
                for &t in &images {
                    if a == b {
                        // Half the translations, so each self-image pair appears once.
                        if !is_positive_translation(t) {
                            continue;
                        }
                    }
                    let d = pb + cell.translation(t) - pa;
                    let r2 = d.norm2();
                    if r2 > cut2 || r2 < 1.0e-20 {
                        continue;
                    }
                    pairs.push(ImagePair {
                        a,
                        b,
                        t,
                        d,
                        r: r2.sqrt(),
                        weight: 1.0,
                        // A self pair has no meaningful "nearest image" for exchange purposes
                        // -- the nearest image of an atom is itself -- so it is never flagged.
                        minimum_image: a != b && t == nearest_t,
                    });
                }
                // If the nearest image fell outside the cutoff nothing is flagged, which is
                // correct: there is then no exchange partner within range.
                debug_assert!(
                    pairs[first..].iter().filter(|p| p.minimum_image).count() <= 1,
                    "more than one nearest image flagged for pair ({a},{b})"
                );
            }
        }
        Self { pairs, cutoff }
    }

    /// Spatially binned: `O(N * I)`, and bit-identical to the scan.
    ///
    /// The rearrangement is one line of algebra. `|p_B + T - p_A| < r` is `|p_B - (p_A - T)| < r`,
    /// so **one** grid over the home-cell positions serves every translation -- the translation
    /// moves the query point, not the index. Each `(atom, image)` query touches 27 buckets holding
    /// a bounded number of atoms.
    ///
    /// The result is sorted back into the original `(a, b, image index)` order, and that is not
    /// cosmetic: every consumer sums over `pairs` in order, so a different order is a different
    /// floating-point summation and a different last bit. Sorting makes the change provably
    /// invisible, which `tests/pair_list.rs` checks against the scan entry by entry.
    fn periodic_binned(molecule: &Molecule, cell: &Cell, cutoff: f64) -> Self {
        let n = molecule.atoms.len();
        // The margin covers the intra-cell separation of the two atoms: an atom at one corner
        // and an atom at the opposite corner are already `diameter` apart before any translation.
        let margin = cell_diameter(cell);
        let images = cell.image_indices(cutoff, margin);
        let cut2 = cutoff * cutoff;

        let positions: Vec<Vec3> = molecule.atoms.iter().map(|atom| atom.position).collect();
        let grid = crate::spatial::Grid::build(&positions, cutoff);

        // `(sort key, pair)`. The key is `(a, b, image index)`, which is the order the nested-loop
        // version emitted.
        let mut found: Vec<((usize, usize, usize), ImagePair)> = Vec::new();
        let mut hits: Vec<usize> = Vec::new();
        for (image_index, &t) in images.iter().enumerate() {
            let shift = cell.translation(t);
            // Half the translations for a self pair, so each self-image pair appears once.
            let self_pairs = is_positive_translation(t);
            for a in 0..n {
                hits.clear();
                grid.near(positions[a] - shift, &mut hits);
                for &b in &hits {
                    if b < a || (a == b && !self_pairs) {
                        continue;
                    }
                    let d = positions[b] + shift - positions[a];
                    let r2 = d.norm2();
                    if r2 > cut2 || r2 < 1.0e-20 {
                        continue;
                    }
                    found.push((
                        (a, b, image_index),
                        ImagePair {
                            a,
                            b,
                            t,
                            d,
                            r: r2.sqrt(),
                            weight: 1.0,
                            // Filled in below, once per `(a, b)` rather than once per image.
                            minimum_image: false,
                        },
                    ));
                }
            }
        }
        found.sort_unstable_by_key(|(key, _)| *key);

        // The nearest image of each pair that has at least one image in range. Ties (two images at
        // the same distance, as at a zone boundary) go to the first in the fixed enumeration
        // order, which keeps the flag deterministic. A pair whose nearest image falls outside the
        // cutoff gets no flag, which is correct: there is then no exchange partner within range.
        let mut pairs: Vec<ImagePair> = Vec::with_capacity(found.len());
        let mut group_start = 0usize;
        while group_start < found.len() {
            let (a, b, _) = found[group_start].0;
            let mut group_end = group_start;
            while group_end < found.len()
                && (found[group_end].0 .0, found[group_end].0 .1) == (a, b)
            {
                group_end += 1;
            }
            // A self pair has no meaningful "nearest image" for exchange purposes — the nearest
            // image of an atom is itself — so it is never flagged.
            if a != b {
                let (_, nearest_t) = cell.minimum_image(positions[b] - positions[a]);
                for entry in &mut found[group_start..group_end] {
                    if entry.1.t == nearest_t {
                        entry.1.minimum_image = true;
                        break;
                    }
                }
            }
            debug_assert!(
                found[group_start..group_end]
                    .iter()
                    .filter(|(_, p)| p.minimum_image)
                    .count()
                    <= 1,
                "more than one nearest image flagged for pair ({a},{b})"
            );
            pairs.extend(found[group_start..group_end].iter().map(|(_, p)| *p));
            group_start = group_end;
        }
        Self { pairs, cutoff }
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The distinct lattice translations that appear in the list, in a deterministic order.
    ///
    /// The real-space Hamiltonian and density are stored per translation (`H(T)`, `P(T)`), and
    /// `H(k) = Σ_T H(T) e^{ik·T}` runs over exactly this set.
    pub fn translations(&self) -> Vec<[i32; 3]> {
        // A `BTreeSet` rather than a `Vec` with `contains`: that was a linear scan of everything
        // found so far, executed twice per pair, so the cost was `O(P·T)` in the number of pairs
        // and distinct translations. Membership is now `O(log T)`, and the set is ordered, so the
        // final `sort_unstable` is subsumed — the result is byte-for-byte the same list.
        let mut seen: std::collections::BTreeSet<[i32; 3]> = std::collections::BTreeSet::new();
        for p in &self.pairs {
            seen.insert(p.t);
            // A pair `(a, b, T)` also implies the conjugate block `(b, a, −T)`, which the
            // k-space assembly needs even though the energy sum only visits one of them.
            seen.insert([-p.t[0], -p.t[1], -p.t[2]]);
        }
        seen.insert([0, 0, 0]);
        seen.into_iter().collect()
    }
}

/// Whether `t` is in the "positive" half of translation space, used to keep exactly one of each
/// `±T` pair for self-images. Lexicographic sign: the first non-zero component decides.
#[inline]
fn is_positive_translation(t: [i32; 3]) -> bool {
    for &c in &t {
        if c > 0 {
            return true;
        }
        if c < 0 {
            return false;
        }
    }
    false // t == 0
}

/// Longest diagonal of the (completed) cell: the largest possible separation of two atoms that
/// both sit inside it.
fn cell_diameter(cell: &Cell) -> f64 {
    let v = cell.vectors();
    let mut worst = 0.0_f64;
    // Every ± combination of the periodic vectors; 2^dim corners, at most 8.
    let dim = v.len();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn atom(z: u8, x: f64, y: f64, z_: f64) -> Atom {
        Atom {
            z,
            position: Vec3::new(x, y, z_),
        }
    }

    /// Where the grid actually overtakes the scan, measured on the two paths themselves.
    ///
    /// `examples/pairlist_scaling.rs` cannot answer this: it goes through `PairList::build`, which
    /// *dispatches*, so below the threshold its "binned" column is the scan measured against a
    /// count-only reimplementation of itself. A table built from it says nothing about the grid at
    /// small `N` — and [`PairList::BINNING_THRESHOLD`] used to cite exactly that. From here the two
    /// constructors are in scope and can be called directly, which is the only way to see the
    /// crossover.
    ///
    /// `#[ignore]`d: it is a measurement, and wall clock is not something a test should assert.
    ///     cargo test --release -p pm7-rs --lib -- --ignored --nocapture pair_list_crossover
    #[test]
    #[ignore = "measurement: prints the scan/grid crossover"]
    fn pair_list_crossover() {
        use std::time::Instant;
        let a = 3.567 * crate::constants::ANGSTROM_TO_BOHR;
        let base = [
            Vec3::new(0.0, a * 0.5, a * 0.5),
            Vec3::new(a * 0.5, 0.0, a * 0.5),
            Vec3::new(a * 0.5, a * 0.5, 0.0),
        ];
        let cutoff = 7.0 * crate::constants::ANGSTROM_TO_BOHR;
        println!(
            "{:>7}  {:>12}  {:>12}  {:>9}",
            "atoms", "scan (ms)", "grid (ms)", "grid/scan"
        );
        for k in [1usize, 2, 3, 4, 5] {
            let cell =
                Cell::new(&[base[0] * k as f64, base[1] * k as f64, base[2] * k as f64]).unwrap();
            let mut atoms = Vec::new();
            for i in 0..k {
                for j in 0..k {
                    for l in 0..k {
                        let s = base[0] * i as f64 + base[1] * j as f64 + base[2] * l as f64;
                        atoms.push(atom(6, s.x, s.y, s.z));
                        let d = s + (base[0] + base[1] + base[2]) * 0.25;
                        atoms.push(atom(6, d.x, d.y, d.z));
                    }
                }
            }
            let molecule = Molecule::new(atoms).with_cell(cell);
            let n = molecule.atoms.len();
            let runs = if n < 100 { 50 } else { 5 };

            let start = Instant::now();
            let mut scanned = 0;
            for _ in 0..runs {
                scanned = PairList::periodic_by_scan(&molecule, &cell, cutoff)
                    .pairs
                    .len();
            }
            let scan = start.elapsed().as_secs_f64() * 1.0e3 / runs as f64;

            let start = Instant::now();
            let mut binned = 0;
            for _ in 0..runs {
                binned = PairList::periodic_binned(&molecule, &cell, cutoff)
                    .pairs
                    .len();
            }
            let grid = start.elapsed().as_secs_f64() * 1.0e3 / runs as f64;
            assert_eq!(scanned, binned, "the two paths disagree at {n} atoms");
            println!("{n:>7}  {scan:>12.4}  {grid:>12.4}  {:>8.2}x", grid / scan);
        }
    }

    #[test]
    fn molecular_list_matches_the_all_pairs_enumeration() {
        let mol = Molecule::new(vec![
            atom(8, 0.0, 0.0, 0.0),
            atom(1, 1.8, 0.0, 0.0),
            atom(1, -0.45, 1.75, 0.0),
        ]);
        let list = PairList::build(&mol, 1.0); // cutoff ignored for a molecule
        assert_eq!(list.len(), 3);
        assert!(list
            .pairs
            .iter()
            .all(|p| p.t == [0, 0, 0] && p.weight == 1.0));
        assert_eq!(list.translations(), vec![[0, 0, 0]]);
    }

    #[test]
    fn self_images_are_counted_once() {
        // One atom in a cubic cell: the only pairs are the atom with its own images.
        let cell = Cell::cubic(6.0).unwrap();
        let mol = Molecule::new(vec![atom(6, 0.0, 0.0, 0.0)]).with_cell(cell);
        let list = PairList::build(&mol, 6.5);
        assert!(!list.is_empty());
        assert!(list.pairs.iter().all(|p| p.is_self_image()));
        assert!(list.pairs.iter().all(|p| p.weight == 1.0));
        // Exactly one of each ±T survives.
        for p in &list.pairs {
            let neg = [-p.t[0], -p.t[1], -p.t[2]];
            assert!(
                !list.pairs.iter().any(|q| q.t == neg),
                "both +T and -T present for {:?}",
                p.t
            );
        }
        // The six nearest images at 6.0 Bohr must be represented once each (i.e. three entries,
        // each standing for a ±pair).
        let nearest = list
            .pairs
            .iter()
            .filter(|p| (p.r - 6.0).abs() < 1e-9)
            .count();
        assert_eq!(nearest, 3, "expected 3 of the 6 nearest images (±pairs)");
    }

    #[test]
    fn weighted_sum_reproduces_a_brute_force_lattice_sum() {
        // The weight convention must make `Σ_pairs weight · f(r)` equal to the physically
        // intended `½ Σ_{A≠B or T≠0} f(r)` over the full image set.
        let cell = Cell::new(&[
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.6, 4.4, 0.0),
            Vec3::new(0.0, 0.3, 5.7),
        ])
        .unwrap();
        let mol = Molecule::new(vec![
            atom(6, 0.0, 0.0, 0.0),
            atom(8, 1.3, 0.7, 2.1),
            atom(1, -1.1, 2.0, 0.4),
        ])
        .with_cell(cell);
        let cutoff = 11.0;
        let list = PairList::build(&mol, cutoff);
        let f = |r: f64| (-0.35 * r).exp() / r;

        let from_list: f64 = list.pairs.iter().map(|p| p.weight * f(p.r)).sum();

        // Brute force: every ordered (A, B, T) with (A,B,T) != (A,A,0), halved.
        let n = mol.atoms.len();
        let images = cell.image_indices(cutoff, 12.0);
        let mut brute = 0.0;
        for a in 0..n {
            for b in 0..n {
                for &t in &images {
                    if a == b && t == [0, 0, 0] {
                        continue;
                    }
                    let d = mol.atoms[b].position + cell.translation(t) - mol.atoms[a].position;
                    let r = d.norm();
                    if r <= cutoff {
                        brute += 0.5 * f(r);
                    }
                }
            }
        }
        assert!(
            (from_list - brute).abs() < 1e-12 * brute.abs().max(1.0),
            "pair-list sum {from_list} vs brute force {brute}"
        );
    }

    #[test]
    fn every_pair_is_within_the_cutoff_and_the_translation_is_consistent() {
        let cell = Cell::orthorhombic(7.0, 8.0, 9.0).unwrap();
        let mol = Molecule::new(vec![
            atom(6, 0.2, 0.1, 0.3),
            atom(1, 3.4, 2.2, 1.1),
            atom(8, 5.9, 6.6, 7.7),
        ])
        .with_cell(cell);
        let cutoff = 10.0;
        let list = PairList::build(&mol, cutoff);
        for p in &list.pairs {
            assert!(p.r <= cutoff + 1e-12);
            let rebuilt = mol.atoms[p.b].position + cell.translation(p.t) - mol.atoms[p.a].position;
            assert!((rebuilt - p.d).norm() < 1e-12);
            assert!((p.d.norm() - p.r).abs() < 1e-12);
        }
    }

    #[test]
    fn translations_are_closed_under_negation() {
        let cell = Cell::new(&[Vec3::new(4.5, 0.0, 0.0), Vec3::new(0.0, 5.5, 0.0)]).unwrap();
        let mol =
            Molecule::new(vec![atom(6, 0.0, 0.0, 0.0), atom(1, 1.9, 0.4, 0.0)]).with_cell(cell);
        let ts = PairList::build(&mol, 9.0).translations();
        assert!(ts.contains(&[0, 0, 0]));
        for t in &ts {
            assert!(
                ts.contains(&[-t[0], -t[1], -t[2]]),
                "translation set not closed under negation: {t:?}"
            );
        }
        // A 2-D cell must never generate a translation along the open direction.
        assert!(ts.iter().all(|t| t[2] == 0));
    }

    #[test]
    fn one_dimensional_cell_only_translates_along_its_axis() {
        let cell = Cell::new(&[Vec3::new(4.7, 0.0, 0.0)]).unwrap();
        let mol =
            Molecule::new(vec![atom(6, 0.0, 0.0, 0.0), atom(1, 0.0, 2.0, 0.0)]).with_cell(cell);
        let list = PairList::build(&mol, 15.0);
        assert!(list.pairs.iter().all(|p| p.t[1] == 0 && p.t[2] == 0));
        assert!(list.pairs.iter().any(|p| p.t[0] != 0));
    }
}
