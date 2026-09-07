// SPDX-License-Identifier: GPL-3.0-or-later

//! A density matrix stored as atom-pair blocks.
//!
//! This is what makes divide and conquer scale linearly, and it is not an optimisation that can
//! be added later. A dense `nao × nao` density costs `O(N²)` to *store* and `O(N²)` to assemble,
//! so a method whose every other step is `O(N)` still comes out quadratic if the density is dense
//! — and at 5 000 atoms the dense matrix alone is gigabytes.
//!
//! The representation exploits two facts. A divide-and-conquer density is built from subsystems of
//! bounded size, so it is only ever non-zero between atoms that share a subsystem; and every
//! consumer of a density in this crate — the Fock build, the gradient, the stress, the energy
//! contraction — already walks **atom pairs**, so a pair-block store is what they want to be
//! handed. Nothing has to be transposed into a different shape at the boundary.
//!
//! Blocks are held in a CSR-like layout: one sorted run of column atoms per row atom, with the
//! numerical data packed contiguously. Lookup is a binary search inside one short run.

use crate::basis::Basis;
use crate::linalg::Matrix;

/// A symmetric density matrix, stored as `na × nb` blocks for the atom pairs that carry weight.
#[derive(Clone, Debug, Default)]
pub struct SparseDensity {
    /// Number of atoms.
    n_atoms: usize,
    /// `row_start[a] .. row_start[a + 1]` indexes `columns` and `offsets` for row atom `a`.
    row_start: Vec<usize>,
    /// Column atom of each stored block, ascending within a row.
    columns: Vec<usize>,
    /// Where each block starts in `data`.
    offsets: Vec<usize>,
    /// Row-major `na × nb` blocks, concatenated.
    data: Vec<f64>,
    /// Orbital count per atom, so a block's shape is known without the basis.
    norb: Vec<usize>,
}

/// A read-only view of one `rows × cols` block.
#[derive(Clone, Copy, Debug)]
pub struct BlockView<'a> {
    values: &'a [f64],
    cols: usize,
}

impl BlockView<'_> {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.cols + j]
    }

    /// An all-zero view, for a pair the sparsity pattern does not cover.
    pub const EMPTY: BlockView<'static> = BlockView {
        values: &[],
        cols: 0,
    };

    /// The block's row length, so a caller can walk it without re-deriving the shape.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl SparseDensity {
    /// Allocate the pattern: for each row atom, the sorted list of column atoms it stores.
    ///
    /// The pattern is fixed for the life of the density. That is deliberate — it is decided once
    /// by the partitioning and then reused every SCF iteration, so no iteration pays for
    /// allocation or for discovering which pairs exist.
    pub fn with_pattern(norb: &[usize], neighbours: &[Vec<usize>]) -> Self {
        let n_atoms = norb.len();
        let mut row_start = Vec::with_capacity(n_atoms + 1);
        let mut columns = Vec::new();
        let mut offsets = Vec::new();
        let mut total = 0usize;
        row_start.push(0);
        for (a, list) in neighbours.iter().enumerate() {
            let mut sorted = list.clone();
            sorted.sort_unstable();
            sorted.dedup();
            for b in sorted {
                columns.push(b);
                offsets.push(total);
                total += norb[a] * norb[b];
            }
            row_start.push(columns.len());
        }
        offsets.push(total);
        Self {
            n_atoms,
            row_start,
            columns,
            offsets,
            data: vec![0.0; total],
            norb: norb.to_vec(),
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Number of stored atom-pair blocks — the honest measure of how sparse this actually is.
    pub fn stored_pairs(&self) -> usize {
        self.columns.len()
    }

    /// Stored floating-point elements. Compare against `nao²` to see the saving.
    pub fn stored_elements(&self) -> usize {
        self.data.len()
    }

    /// Index of the `(a, b)` block in the pattern, for callers that need to carry a scalar per
    /// stored pair alongside the density — the partition-weight totals, for instance.
    #[inline]
    pub fn pair_slot(&self, a: usize, b: usize) -> Option<usize> {
        self.slot(a, b)
    }

    /// Multiply each stored block by its own factor, indexed as [`Self::pair_slot`].
    pub fn scale_pairs(&mut self, factors: &[f64]) {
        debug_assert_eq!(factors.len(), self.columns.len());
        for (slot, factor) in factors.iter().enumerate() {
            let (start, end) = (self.offsets[slot], self.offsets[slot + 1]);
            for v in &mut self.data[start..end] {
                *v *= factor;
            }
        }
    }

    #[inline]
    fn slot(&self, a: usize, b: usize) -> Option<usize> {
        let (lo, hi) = (self.row_start[a], self.row_start[a + 1]);
        self.columns[lo..hi]
            .binary_search(&b)
            .ok()
            .map(|index| lo + index)
    }

    /// The `(a, b)` block, or an empty view when the pattern does not cover that pair.
    #[inline]
    pub fn block(&self, a: usize, b: usize) -> BlockView<'_> {
        match self.slot(a, b) {
            None => BlockView::EMPTY,
            Some(slot) => BlockView {
                values: &self.data[self.offsets[slot]..self.offsets[slot + 1]],
                cols: self.norb[b],
            },
        }
    }

    /// Mutable access to the `(a, b)` block, row-major, or `None` outside the pattern.
    #[inline]
    pub fn block_mut(&mut self, a: usize, b: usize) -> Option<&mut [f64]> {
        let slot = self.slot(a, b)?;
        let (start, end) = (self.offsets[slot], self.offsets[slot + 1]);
        Some(&mut self.data[start..end])
    }

    pub fn clear(&mut self) {
        for v in &mut self.data {
            *v = 0.0;
        }
    }

    /// `Σ_μ P_μμ` over the orbitals of atom `a` — its Mulliken population.
    pub fn population(&self, a: usize) -> f64 {
        let block = self.block(a, a);
        if block.is_empty() {
            return 0.0;
        }
        (0..self.norb[a]).map(|mu| block.get(mu, mu)).sum()
    }

    /// Total electron count, `Tr P`.
    pub fn trace(&self) -> f64 {
        (0..self.n_atoms).map(|a| self.population(a)).sum()
    }

    /// `‖self − other‖ / √n`, the same RMS measure the dense SCF converges on. The two must share
    /// a pattern.
    pub fn rms_diff(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        let n = self.data.len().max(1);
        (self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n as f64)
            .sqrt()
    }

    /// `self ← (1 − w)·self + w·other`, for linear mixing. The two must share a pattern.
    pub fn mix(&mut self, other: &Self, w: f64) {
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a = (1.0 - w) * *a + w * *b;
        }
    }

    /// Add another density of the **same pattern** into this one, block for block.
    ///
    /// For combining per-thread partial accumulations. Because the two share a pattern, the
    /// storage is index-for-index comparable and the sum is a flat vector add — and because the
    /// caller folds the partials back **in subsystem order**, the result does not depend on how
    /// the work was divided.
    pub fn add_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }

    /// Accumulate `weight × block` into the `(a, b)` block, ignoring pairs outside the pattern.
    ///
    /// Silently dropping out-of-pattern pairs is the intended behaviour: the pattern *is* the
    /// approximation, and a subsystem's density between two atoms that are not stored together is
    /// exactly what divide and conquer discards.
    pub fn accumulate(&mut self, a: usize, b: usize, weight: f64, block: &[f64]) {
        let cols = self.norb[b];
        if let Some(target) = self.block_mut(a, b) {
            debug_assert_eq!(target.len(), block.len());
            for (dst, src) in target.iter_mut().zip(block) {
                *dst += weight * src;
            }
        }
        let _ = cols;
    }

    /// Materialize the dense matrix. **For tests and small systems only** — this is the `O(N²)`
    /// step the whole representation exists to avoid, and calling it in an SCF loop would undo
    /// the linear scaling.
    pub fn to_dense(&self, basis: &Basis) -> Matrix {
        let mut m = Matrix::zeros(basis.nao, basis.nao);
        for a in 0..self.n_atoms {
            let (lo, hi) = (self.row_start[a], self.row_start[a + 1]);
            for slot in lo..hi {
                let b = self.columns[slot];
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (self.norb[a], self.norb[b]);
                let values = &self.data[self.offsets[slot]..self.offsets[slot + 1]];
                for i in 0..na {
                    for j in 0..nb {
                        m[(oa + i, ob + j)] = values[i * nb + j];
                    }
                }
            }
        }
        m
    }

    /// Fill from a dense matrix, keeping only the pattern. The round trip
    /// `dense → sparse → dense` is the identity exactly when the pattern covers every
    /// non-negligible pair, which is what `density_cutoff_keeps_what_matters` checks.
    pub fn fill_from_dense(&mut self, basis: &Basis, dense: &Matrix) {
        for a in 0..self.n_atoms {
            let (lo, hi) = (self.row_start[a], self.row_start[a + 1]);
            for slot in lo..hi {
                let b = self.columns[slot];
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (self.norb[a], self.norb[b]);
                let (start, end) = (self.offsets[slot], self.offsets[slot + 1]);
                let values = &mut self.data[start..end];
                for i in 0..na {
                    for j in 0..nb {
                        values[i * nb + j] = dense[(oa + i, ob + j)];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern() -> SparseDensity {
        // Three atoms: 0 and 1 share a subsystem, 1 and 2 share another, 0 and 2 do not meet.
        let norb = [4, 1, 4];
        let neighbours = vec![vec![0, 1], vec![0, 1, 2], vec![1, 2]];
        SparseDensity::with_pattern(&norb, &neighbours)
    }

    #[test]
    fn a_block_round_trips_and_a_missing_pair_reads_as_zero() {
        let mut p = pattern();
        let block: Vec<f64> = (0..4).map(|i| i as f64).collect();
        p.accumulate(0, 1, 2.0, &block);
        let view = p.block(0, 1);
        for i in 0..4 {
            assert_eq!(view.get(i, 0), 2.0 * i as f64);
        }
        assert!(p.block(0, 2).is_empty());
        assert_eq!(p.block(0, 2).values.len(), 0);
        // Accumulating into a pair outside the pattern is a no-op, not a panic: dropping it *is*
        // the divide-and-conquer approximation.
        p.accumulate(0, 2, 1.0, &[1.0; 16]);
        assert!(p.block(0, 2).is_empty());
    }

    #[test]
    fn the_population_and_trace_come_from_the_diagonal_blocks() {
        let mut p = pattern();
        let mut d = vec![0.0; 16];
        for i in 0..4 {
            d[i * 4 + i] = 1.5;
        }
        p.accumulate(0, 0, 1.0, &d);
        p.accumulate(1, 1, 1.0, &[0.5]);
        assert!((p.population(0) - 6.0).abs() < 1e-12);
        assert!((p.population(1) - 0.5).abs() < 1e-12);
        assert!((p.trace() - 6.5).abs() < 1e-12);
    }

    #[test]
    fn the_stored_size_is_the_pattern_and_not_the_square() {
        let p = pattern();
        // Row 0 holds (0,0) and (0,1): 4·4 + 4·1 = 20. Row 1 holds all three: 1·4 + 1 + 1·4 = 9.
        // Row 2 holds (2,1) and (2,2): 4·1 + 4·4 = 20. Against a dense 9×9 = 81.
        assert_eq!(p.stored_elements(), 49);
        assert_eq!(p.stored_pairs(), 7);
    }

    #[test]
    fn scaling_a_pair_touches_only_that_block() {
        let mut p = pattern();
        p.accumulate(0, 0, 1.0, &[2.0; 16]);
        p.accumulate(1, 1, 1.0, &[3.0]);
        let mut factors = vec![1.0; p.stored_pairs()];
        factors[p.pair_slot(1, 1).unwrap()] = 0.5;
        p.scale_pairs(&factors);
        assert_eq!(p.block(0, 0).get(2, 2), 2.0);
        assert_eq!(p.block(1, 1).get(0, 0), 1.5);
    }
}
