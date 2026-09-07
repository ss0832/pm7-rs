// SPDX-License-Identifier: GPL-3.0-or-later

//! A uniform spatial grid: "every atom within `r` of this point" in `O(1)` instead of a scan.
//!
//! Two callers, both of which were quadratic without it:
//!
//! * [`crate::dandc::partition`] grows a buffer around every subsystem core. Scanning every atom
//!   for every core is `O(N²)`, which is enough on its own to stop divide and conquer being
//!   linear — the partitioning happens once, but it happens over the whole system.
//! * [`crate::pbc::images::PairList::periodic`] enumerates neighbour pairs. It looped over every
//!   `A ≤ B` pair and, inside that, over every lattice image, so it was `O(N²·I)` for a list whose
//!   *content* is `O(N·I)`: at a 7 Å cutoff almost every pair it examined was rejected.
//!
//! The grid lived inside `partition.rs` until 0.2.3 and moved here to serve both. It is a spatial
//! *index* and nothing more: [`Grid::near`] returns a **superset** of what is within one edge
//! length, and the caller filters by exact distance. That is what makes it safe to coarsen the
//! edge when a caller asks for one so small the bucket count would not fit in memory — a larger
//! edge keeps the superset property, a smaller one would break it.

use crate::math::Vec3;

pub(crate) struct Grid {
    origin: Vec3,
    edge: f64,
    dims: [i64; 3],
    buckets: Vec<Vec<usize>>,
}

impl Grid {
    /// Bin `positions` into cells of side `edge`.
    ///
    /// `edge` is a *request*: it is enlarged if the bounding box divided by it would need more
    /// buckets than a small multiple of the atom count. A caller asking for `1e-9` on a 100-Bohr
    /// system would otherwise allocate `10²⁴` buckets, which aborts the process rather than
    /// returning an error.
    pub(crate) fn build(positions: &[Vec3], edge: f64) -> Self {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in positions {
            lo.x = lo.x.min(p.x);
            lo.y = lo.y.min(p.y);
            lo.z = lo.z.min(p.z);
            hi.x = hi.x.max(p.x);
            hi.y = hi.y.max(p.y);
            hi.z = hi.z.max(p.z);
        }
        if positions.is_empty() {
            lo = Vec3::zero();
            hi = Vec3::zero();
        }
        let extent = |a: f64, b: f64| (b - a).max(0.0);
        let span = [extent(lo.x, hi.x), extent(lo.y, hi.y), extent(lo.z, hi.z)];
        let cap = (8 * positions.len() + 64) as f64;
        let mut edge = edge.max(1.0e-6);
        let count = |e: f64| -> f64 { span.iter().map(|s| (s / e).floor() + 1.0).product() };
        for _ in 0..64 {
            let total = count(edge);
            if total <= cap {
                break;
            }
            // Three linear dimensions, so the bucket count falls roughly as `edge³`. Take the cube
            // root of the overshoot, with a floor so a degenerate (planar or collinear) geometry —
            // where only one or two factors actually shrink — still terminates.
            edge *= (total / cap).cbrt().max(1.25);
        }
        let dims = [
            (((hi.x - lo.x) / edge).floor() as i64 + 1).max(1),
            (((hi.y - lo.y) / edge).floor() as i64 + 1).max(1),
            (((hi.z - lo.z) / edge).floor() as i64 + 1).max(1),
        ];
        let mut grid = Self {
            origin: lo,
            edge,
            dims,
            buckets: vec![Vec::new(); (dims[0] * dims[1] * dims[2]) as usize],
        };
        for (index, p) in positions.iter().enumerate() {
            let cell = grid.cell_of(*p);
            let slot = grid.slot(cell);
            grid.buckets[slot].push(index);
        }
        grid
    }

    /// The edge actually used, which may be larger than the one requested. Only the superset test
    /// below needs it — a caller filters by its own cutoff, not by the grid's edge.
    #[cfg(test)]
    fn edge(&self) -> f64 {
        self.edge
    }

    #[inline]
    fn cell_of(&self, p: Vec3) -> [i64; 3] {
        [
            (((p.x - self.origin.x) / self.edge).floor() as i64).clamp(0, self.dims[0] - 1),
            (((p.y - self.origin.y) / self.edge).floor() as i64).clamp(0, self.dims[1] - 1),
            (((p.z - self.origin.z) / self.edge).floor() as i64).clamp(0, self.dims[2] - 1),
        ]
    }

    /// Like [`Self::cell_of`] but **without** the clamp, so a query point outside the box keeps
    /// its true (possibly out-of-range) cell index.
    ///
    /// The clamp is right for *binning* — every atom must land somewhere — and wrong for a
    /// *query*: a point far outside the box would be clamped onto the boundary cell and would come
    /// back with that cell's atoms as neighbours. `PairList::periodic` queries at `p - T(t)`,
    /// which is outside the box for every non-zero translation, so this distinction is the whole
    /// correctness of the periodic path.
    #[inline]
    fn unclamped_cell_of(&self, p: Vec3) -> [i64; 3] {
        [
            ((p.x - self.origin.x) / self.edge).floor() as i64,
            ((p.y - self.origin.y) / self.edge).floor() as i64,
            ((p.z - self.origin.z) / self.edge).floor() as i64,
        ]
    }

    #[inline]
    fn slot(&self, c: [i64; 3]) -> usize {
        ((c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]) as usize
    }

    /// Every atom in the 27 cells around `p`, appended to `out`. A superset of what is within one
    /// edge length; the caller filters by exact distance.
    pub(crate) fn near(&self, p: Vec3, out: &mut Vec<usize>) {
        let c = self.unclamped_cell_of(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let n = [c[0] + dx, c[1] + dy, c[2] + dz];
                    if (0..3).any(|k| n[k] < 0 || n[k] >= self.dims[k]) {
                        continue;
                    }
                    out.extend_from_slice(&self.buckets[self.slot(n)]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property everything else rests on: `near` never misses an atom within one edge length.
    ///
    /// Checked against a brute-force scan on a random-ish cloud, including query points well
    /// outside the bounding box — which is where `PairList::periodic` spends most of its queries,
    /// and where a clamped cell index would silently return the wrong bucket.
    #[test]
    fn near_returns_a_superset_of_everything_within_one_edge() {
        let mut positions = Vec::new();
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) * 20.0 - 10.0
        };
        for _ in 0..200 {
            positions.push(Vec3::new(next(), next(), next()));
        }
        let grid = Grid::build(&positions, 2.5);
        let edge = grid.edge();

        let mut found = Vec::new();
        for query in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(9.0, -9.0, 3.0),
            // Outside the box, by a lot: the periodic pair list queries here constantly.
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(-40.0, 25.0, -33.0),
        ] {
            found.clear();
            grid.near(query, &mut found);
            for (index, p) in positions.iter().enumerate() {
                if (*p - query).norm() <= edge {
                    assert!(
                        found.contains(&index),
                        "atom {index} at {p:?} is within {edge} of {query:?} and was not returned"
                    );
                }
            }
        }
    }

    /// An empty input is a grid with one bucket, not a panic.
    #[test]
    fn an_empty_cloud_is_not_a_panic() {
        let grid = Grid::build(&[], 1.0);
        let mut out = Vec::new();
        grid.near(Vec3::zero(), &mut out);
        assert!(out.is_empty());
    }
}
