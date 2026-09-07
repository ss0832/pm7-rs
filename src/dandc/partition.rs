// SPDX-License-Identifier: GPL-3.0-or-later

//! Spatial partitioning for divide and conquer.
//!
//! Each subsystem is a **core** of atoms plus a **buffer** shell of everything within `R_buf` of
//! the core. The core atoms are the ones whose density the subsystem is responsible for; the
//! buffer exists only so that the core's orbitals see a chemically complete environment.
//!
//! The one design rule that matters for scaling: **a subsystem's size must not grow with the
//! system**. Cores are sized by a target atom count and buffers by a fixed radius, so doubling the
//! system doubles the *number* of subsystems and leaves their size alone — which is what turns the
//! `O(n³)` diagonalization into `O(N)`. A partitioning that split the system into a fixed *number*
//! of pieces would look similar and scale cubically.
//!
//! For a periodic cell the buffer is drawn from periodic images, so a subsystem is a finite
//! cluster even though the system is not. Periodicity reaches the calculation through the buffer
//! atoms and through the long-range monopole field, never through complex arithmetic — which is
//! why divide and conquer works the same way at Γ and on a k mesh.

use crate::math::Vec3;
use crate::system::Molecule;

/// One subsystem: a core, its buffer, and the mapping back to the parent atoms.
#[derive(Clone, Debug)]
pub struct Subsystem {
    /// Parent atom index of each subsystem atom, core atoms first.
    pub parent: Vec<usize>,
    /// Position of each subsystem atom — a buffer atom drawn from an image carries the shifted
    /// position, so the cluster is geometrically faithful.
    pub positions: Vec<Vec3>,
    /// The lattice translation each subsystem atom came from. `[0, 0, 0]` for the home cell.
    pub translation: Vec<[i32; 3]>,
    /// How many of the leading entries are core atoms.
    pub n_core: usize,
}

impl Subsystem {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Dixon–Merz partition weight for a subsystem atom pair: 1 when both are core, ½ when one is,
    /// 0 when neither.
    ///
    /// Summed over subsystems these weights come to 1 for every pair that any subsystem holds,
    /// because a core–buffer pair is seen by exactly two subsystems — the one that owns each end.
    /// That is the whole content of the partitioning: it is a partition of unity over pairs.
    #[inline]
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        match (i < self.n_core, j < self.n_core) {
            (true, true) => 1.0,
            (true, false) | (false, true) => 0.5,
            (false, false) => 0.0,
        }
    }
}

/// The full partitioning of a system.
#[derive(Clone, Debug)]
pub struct Partitioning {
    pub subsystems: Vec<Subsystem>,
}

impl Partitioning {
    /// Atoms in the largest subsystem — the number that has to stay flat as the system grows.
    pub fn largest(&self) -> usize {
        self.subsystems.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    /// Mean subsystem size, for the scaling record.
    pub fn mean_size(&self) -> f64 {
        if self.subsystems.is_empty() {
            return 0.0;
        }
        self.subsystems.iter().map(|s| s.len()).sum::<usize>() as f64 / self.subsystems.len() as f64
    }
}

/// Split `molecule` into cores of about `core_size` atoms by recursive bisection along the widest
/// axis, then grow a `buffer`-radius shell around each.
///
/// Recursive bisection rather than a fixed grid: a molecular system is rarely a filled box, and a
/// grid over its bounding box produces empty cells and wildly uneven occupancy. Bisecting on the
/// median along the widest extent gives cores of even size whatever the shape.
pub fn partition(molecule: &Molecule, core_size: usize, buffer: f64) -> Partitioning {
    let n = molecule.atoms.len();
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let mut cores: Vec<Vec<usize>> = Vec::new();
    bisect((0..n).collect(), &positions, core_size.max(1), &mut cores);

    // One grid per image shell, built **once** for the whole system. Building them per subsystem
    // would put an `O(N)` pass inside an `O(N)` loop and hand back the quadratic scaling the grid
    // exists to remove.
    let images: Vec<[i32; 3]> = match molecule.cell {
        None => vec![[0, 0, 0]],
        Some(c) => c.image_indices(buffer, 0.0),
    };
    let shells: Vec<(([i32; 3], Vec<Vec3>), crate::spatial::Grid)> = images
        .iter()
        .map(|t| {
            let shifted: Vec<Vec3> = match molecule.cell {
                None => positions.clone(),
                Some(c) => {
                    let shift = c.translation(*t);
                    positions.iter().map(|p| *p + shift).collect()
                }
            };
            let grid = crate::spatial::Grid::build(&shifted, buffer);
            ((*t, shifted), grid)
        })
        .collect();

    let subsystems = cores
        .into_iter()
        .map(|core| grow_buffer(&core, &positions, &shells, buffer))
        .collect();
    Partitioning { subsystems }
}

/// Recursively halve `indices` until each piece is at most `target` atoms.
fn bisect(indices: Vec<usize>, positions: &[Vec3], target: usize, out: &mut Vec<Vec<usize>>) {
    if indices.len() <= target || indices.len() < 2 {
        out.push(indices);
        return;
    }
    // Widest Cartesian extent of this group.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in &indices {
        for axis in 0..3 {
            let v = positions[i].get(axis);
            lo[axis] = lo[axis].min(v);
            hi[axis] = hi[axis].max(v);
        }
    }
    // `unwrap_or(Ordering::Equal)` rather than `unwrap`: `run_dandc` validates its input, so a
    // NaN extent cannot reach here — but a comparator that panics on one is a landmine for the
    // next caller, and the rest of the crate (`linalg.rs`, `scf.rs`) already handles it this way.
    let axis = (0..3)
        .max_by(|a, b| {
            (hi[*a] - lo[*a])
                .partial_cmp(&(hi[*b] - lo[*b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut sorted = indices;
    sorted.sort_by(|a, b| {
        positions[*a]
            .get(axis)
            .partial_cmp(&positions[*b].get(axis))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let half = sorted.len() / 2;
    let right = sorted.split_off(half);
    bisect(sorted, positions, target, out);
    bisect(right, positions, target, out);
}

/// Everything within `buffer` of any core atom, taken from the pre-built image shells.
fn grow_buffer(
    core: &[usize],
    positions: &[Vec3],
    shells: &[(([i32; 3], Vec<Vec3>), crate::spatial::Grid)],
    buffer: f64,
) -> Subsystem {
    let core_coords: Vec<Vec3> = core.iter().map(|&i| positions[i]).collect();
    let n_core = core.len();
    let mut parent = core.to_vec();
    let mut coords = core_coords.clone();
    let mut translation: Vec<[i32; 3]> = vec![[0, 0, 0]; n_core];

    let in_core: std::collections::HashSet<usize> = core.iter().copied().collect();
    let cutoff2 = buffer * buffer;
    let mut seen: std::collections::HashSet<(usize, [i32; 3])> = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    for ((t, shifted), grid) in shells {
        for core_position in &core_coords {
            candidates.clear();
            grid.near(*core_position, &mut candidates);
            for &index in &candidates {
                if in_core.contains(&index) && *t == [0, 0, 0] {
                    continue;
                }
                if (shifted[index] - *core_position).norm2() > cutoff2 {
                    continue;
                }
                if seen.insert((index, *t)) {
                    parent.push(index);
                    coords.push(shifted[index]);
                    translation.push(*t);
                }
            }
        }
    }
    Subsystem {
        parent,
        positions: coords,
        translation,
        n_core,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    /// A straight chain of `n` atoms spaced 2 Bohr apart.
    fn chain(n: usize) -> Molecule {
        Molecule::new(
            (0..n)
                .map(|i| Atom {
                    z: 1,
                    position: Vec3::new(2.0 * i as f64, 0.0, 0.0),
                })
                .collect(),
        )
    }

    #[test]
    fn every_atom_is_the_core_of_exactly_one_subsystem() {
        let p = partition(&chain(40), 5, 6.0);
        let mut owned = vec![0usize; 40];
        for s in &p.subsystems {
            for &atom in &s.parent[..s.n_core] {
                owned[atom] += 1;
            }
        }
        assert!(
            owned.iter().all(|c| *c == 1),
            "core ownership is not a partition: {owned:?}"
        );
    }

    #[test]
    fn subsystem_size_stays_flat_as_the_system_grows() {
        // The property linear scaling rests on. Doubling the chain must double the subsystem
        // *count* and leave the largest subsystem alone.
        let small = partition(&chain(40), 5, 6.0);
        let large = partition(&chain(160), 5, 6.0);
        assert_eq!(large.subsystems.len(), 4 * small.subsystems.len());
        assert!(
            large.largest() <= small.largest() + 1,
            "the largest subsystem grew from {} to {} when the system quadrupled",
            small.largest(),
            large.largest()
        );
    }

    #[test]
    fn the_buffer_reaches_exactly_as_far_as_it_is_told() {
        let molecule = chain(40);
        let p = partition(&molecule, 4, 6.0);
        for s in &p.subsystems {
            for i in s.n_core..s.len() {
                let d = s.positions[..s.n_core]
                    .iter()
                    .map(|c| (s.positions[i] - *c).norm())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    d <= 6.0 + 1e-12,
                    "a buffer atom sits {d} Bohr from the core"
                );
            }
            // And nothing within the buffer was missed.
            for (atom, position) in molecule.atoms.iter().enumerate() {
                let d = s.positions[..s.n_core]
                    .iter()
                    .map(|c| (position.position - *c).norm())
                    .fold(f64::INFINITY, f64::min);
                if d <= 6.0 - 1e-12 {
                    assert!(
                        s.parent.contains(&atom),
                        "atom {atom} is {d} Bohr from the core but not in the subsystem"
                    );
                }
            }
        }
    }

    #[test]
    fn the_partition_weights_are_a_partition_of_unity_except_at_buffer_edges() {
        // Dixon–Merz weights come to 1 for a core–core pair, and for a core–buffer pair *whose two
        // ends see each other* — the two owning subsystems contribute ½ apiece. At the very edge
        // of a buffer that reciprocity can fail: an atom can lie inside another subsystem's buffer
        // without that subsystem's cores lying inside its own, and the pair then collects a single
        // ½. This pins the exception rather than papering over it, because the driver has to know
        // to normalize: `run_dandc` divides by the realized total.
        let p = partition(&chain(30), 5, 6.0);
        let mut total = std::collections::HashMap::<(usize, usize), f64>::new();
        for s in &p.subsystems {
            for i in 0..s.len() {
                for j in 0..s.len() {
                    // Only home-cell atoms; a molecule has no images anyway.
                    if s.translation[i] != [0, 0, 0] || s.translation[j] != [0, 0, 0] {
                        continue;
                    }
                    let w = s.weight(i, j);
                    if w > 0.0 {
                        *total.entry((s.parent[i], s.parent[j])).or_insert(0.0) += w;
                    }
                }
            }
        }
        let mut edges = 0;
        for (&(a, b), &w) in &total {
            assert!(
                w > 0.0 && w <= 1.0 + 1e-12,
                "pair ({a},{b}) carries total weight {w}, outside (0, 1]"
            );
            if (w - 1.0).abs() > 1e-12 {
                edges += 1;
            }
            // A diagonal entry is the atom's own population, and every atom is the core of
            // exactly one subsystem, so those must be exactly 1.
            if a == b {
                assert!(
                    (w - 1.0).abs() < 1e-12,
                    "atom {a} carries diagonal weight {w}, not 1"
                );
            }
        }
        assert!(
            edges > 0,
            "the buffer-edge case did not occur, so this test proves nothing"
        );
    }
}
