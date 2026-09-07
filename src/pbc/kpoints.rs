// SPDX-License-Identifier: GPL-3.0-or-later

//! Brillouin-zone sampling: Γ point, Monkhorst–Pack meshes, and explicit k-point lists.
//!
//! The real-space Hamiltonian blocks `H(T)` are **real**, so `H(−k) = H(k)*` and the two k points
//! give identical eigenvalues and complex-conjugate eigenvectors. Folding `k` and `−k` onto one
//! representative with double weight therefore halves the diagonalization work with no
//! approximation whatsoever. A k point that is its own negative modulo a reciprocal-lattice
//! vector (Γ, and the zone-boundary points of an even mesh) is *not* doubled — getting that
//! wrong silently misweights exactly the points that matter most for a metal.
//!
//! k points are stored as **fractional** coordinates of the reciprocal lattice, so `k·T` for a
//! lattice translation `t = [t₀, t₁, t₂]` is just `2π Σ_i frac_i t_i` — no Cartesian round trip,
//! and the phase is exact for integer translations.

use crate::cell::Cell;
use crate::error::{Pm7Error, Result};
use crate::math::Vec3;

/// One sampling point of the Brillouin zone.
#[derive(Clone, Copy, Debug)]
pub struct KPoint {
    /// Fractional coordinates along the reciprocal lattice, in `[-½, ½)`.
    pub frac: [f64; 3],
    /// Cartesian wavevector in Bohr⁻¹ (includes the `2π`).
    pub cart: Vec3,
    /// Integration weight; the weights of a set sum to 1.
    pub weight: f64,
    /// `true` when this point stands for both `k` and `−k` (time-reversal folded).
    pub time_reversal_pair: bool,
}

impl KPoint {
    /// Phase `k · T` (radians) for the integer lattice translation `t`.
    #[inline]
    pub fn phase(&self, t: [i32; 3]) -> f64 {
        std::f64::consts::TAU
            * (self.frac[0] * t[0] as f64 + self.frac[1] * t[1] as f64 + self.frac[2] * t[2] as f64)
    }

    /// `true` when every phase factor is `+1`, so the k-space problem is real and the Γ-point
    /// code path applies.
    #[inline]
    pub fn is_gamma(&self) -> bool {
        self.frac.iter().all(|f| f.abs() < 1.0e-12)
    }
}

/// How to sample the Brillouin zone.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum KMesh {
    /// The Γ point only. Equivalent to a `1×1×1` mesh; the electronic problem stays real.
    #[default]
    Gamma,
    /// Monkhorst–Pack mesh. `gamma_centred` selects the Γ-centred convention (as in
    /// `ase.dft.kpoints.monkhorst_pack` with an added Γ shift, and VASP's `Gamma` mode) versus
    /// the original Monkhorst–Pack offset grid.
    MonkhorstPack {
        n: [usize; 3],
        shift: [f64; 3],
        gamma_centred: bool,
    },
    /// An explicit list of fractional k points with weights (renormalized on use). Used for band
    /// structures along a path, and to pin a mesh in a test.
    Explicit(Vec<([f64; 3], f64)>),
}

impl KMesh {
    /// Γ-centred `n1 × n2 × n3` mesh.
    pub fn grid(n1: usize, n2: usize, n3: usize) -> Self {
        KMesh::MonkhorstPack {
            n: [n1, n2, n3],
            shift: [0.0; 3],
            gamma_centred: true,
        }
    }

    /// Mesh subdivisions along each direction (1 for Γ / an explicit list).
    pub fn divisions(&self) -> [usize; 3] {
        match self {
            KMesh::Gamma | KMesh::Explicit(_) => [1, 1, 1],
            KMesh::MonkhorstPack { n, .. } => *n,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            KMesh::Gamma => Ok(()),
            KMesh::MonkhorstPack { n, shift, .. } => {
                if n.contains(&0) {
                    return Err(Pm7Error::InvalidInput(
                        "Monkhorst-Pack divisions must be >= 1".into(),
                    ));
                }
                if shift.iter().any(|s| !s.is_finite()) {
                    return Err(Pm7Error::InvalidInput(
                        "k-point shift must be finite".into(),
                    ));
                }
                // The mesh is `{(i + s)/n}`, whose negation is `{-(i + s)/n}`. Closure under
                // `k → −k` needs `-(i + s) ≡ j + s (mod n)` for some integer `j`, i.e. `j =
                // −i − 2s`, which is an integer exactly when **2s is**. So `s ∈ {0, ½}` per
                // direction, and nothing else.
                //
                // This is not a stylistic restriction. The k-point density is assembled as
                // `P(T) = Σ_k w_k Re[e^{−ik·T} P(k)]`, and the real part is only the correct
                // Brillouin-zone average when the sampled set contains `−k` alongside `k`. It is
                // also what the Born–von Kármán exchange construction assumes: `bvk_representatives`
                // builds the residue classes of an *untwisted* `n₁×n₂×n₃` supercell.
                //
                // Left unchecked, an unsupported shift does not produce a wrong number loudly —
                // it makes the SCF stall at a small, non-zero residual (measured: 5.8e-6 on a 1-D
                // HF chain at `shift = 0.25`, unchanged after 2000 iterations and unaffected by
                // smearing, in a system with a 19 eV gap). That reads like a convergence problem
                // and is really an unrepresentable request.
                for (d, s) in shift.iter().enumerate() {
                    let doubled = 2.0 * s;
                    if (doubled - doubled.round()).abs() > 1.0e-12 {
                        return Err(Pm7Error::InvalidInput(format!(
                            "k-point shift {s} along direction {d} is not supported: only 0 and \
                             0.5 (in units of the mesh spacing) give a set closed under k -> -k, \
                             which the real-part density assembly and the Born-von Karman \
                             exchange both require. An unsupported shift stalls the SCF rather \
                             than converging to a wrong answer."
                        )));
                    }
                }
                Ok(())
            }
            KMesh::Explicit(list) => {
                if list.is_empty() {
                    return Err(Pm7Error::InvalidInput(
                        "an explicit k-point list must not be empty".into(),
                    ));
                }
                if list
                    .iter()
                    .any(|(k, w)| k.iter().any(|c| !c.is_finite()) || !w.is_finite() || *w < 0.0)
                {
                    return Err(Pm7Error::InvalidInput(
                        "explicit k points need finite coordinates and non-negative weights".into(),
                    ));
                }
                if list.iter().map(|(_, w)| *w).sum::<f64>() <= 0.0 {
                    return Err(Pm7Error::InvalidInput(
                        "explicit k-point weights must not sum to zero".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Expand into the irreducible (time-reversal folded) set for `cell`.
    ///
    /// Mesh divisions along a **non-periodic** direction are forced to 1: sampling a direction
    /// that has no reciprocal lattice would silently duplicate the same calculation.
    pub fn expand(&self, cell: &Cell) -> Result<KPointSet> {
        self.expand_impl(cell, true)
    }

    /// The same set, **without** time-reversal folding.
    ///
    /// Perturbation theory needs this. Folding pairs `k` with `−k`, while a response at `q ≠ 0`
    /// relates `k` to `k + q` — a different pairing entirely, and one that cannot be built from
    /// the folded representatives.
    ///
    /// It exists as a sibling of [`Self::expand`] rather than as a separate construction because
    /// the two must agree. `dfpt` used to rebuild a Γ-centred grid from `divisions()` alone, which
    /// discards `shift` and `gamma_centred` and reports `[1, 1, 1]` for [`KMesh::Explicit`] — so a
    /// shifted, original-Monkhorst–Pack, or explicit mesh silently ran perturbation theory on a
    /// different k set from the one the density was converged on.
    pub fn expand_unfolded(&self, cell: &Cell) -> Result<KPointSet> {
        self.expand_impl(cell, false)
    }

    fn expand_impl(&self, cell: &Cell, fold: bool) -> Result<KPointSet> {
        self.validate()?;
        let dim = cell.dim();
        let raw: Vec<([f64; 3], f64)> = match self {
            KMesh::Gamma => vec![([0.0; 3], 1.0)],
            KMesh::MonkhorstPack {
                n,
                shift,
                gamma_centred,
            } => {
                let mut n = *n;
                for (k, nk) in n.iter_mut().enumerate() {
                    if k >= dim {
                        *nk = 1;
                    }
                }
                let total = (n[0] * n[1] * n[2]) as f64;
                let mut out = Vec::with_capacity(n[0] * n[1] * n[2]);
                for i in 0..n[0] {
                    for j in 0..n[1] {
                        for l in 0..n[2] {
                            let idx = [i, j, l];
                            let mut frac = [0.0f64; 3];
                            for (d, f) in frac.iter_mut().enumerate() {
                                let nd = n[d] as f64;
                                *f = if *gamma_centred {
                                    idx[d] as f64 / nd
                                } else {
                                    // Original Monkhorst-Pack: (2r - n - 1) / 2n.
                                    (2.0 * idx[d] as f64 - nd + 1.0) / (2.0 * nd)
                                };
                                if d < dim {
                                    *f += shift[d] / nd;
                                } else {
                                    *f = 0.0;
                                }
                            }
                            out.push((wrap_to_bz(frac), 1.0 / total));
                        }
                    }
                }
                out
            }
            KMesh::Explicit(list) => {
                let total: f64 = list.iter().map(|(_, w)| *w).sum();
                list.iter()
                    .map(|(k, w)| {
                        let mut frac = *k;
                        for (d, f) in frac.iter_mut().enumerate() {
                            if d >= dim {
                                *f = 0.0;
                            }
                        }
                        (frac, w / total)
                    })
                    .collect()
            }
        };
        Ok(KPointSet::from_raw(
            raw,
            cell,
            fold && !matches!(self, KMesh::Explicit(_)),
        ))
    }
}

/// An expanded, weighted set of k points.
#[derive(Clone, Debug)]
pub struct KPointSet {
    pub points: Vec<KPoint>,
    /// Number of points before time-reversal folding — what the weights were normalized against.
    pub unfolded_count: usize,
}

impl KPointSet {
    /// The Γ-point-only set.
    pub fn gamma(cell: &Cell) -> Self {
        Self::from_raw(vec![([0.0; 3], 1.0)], cell, true)
    }

    fn from_raw(raw: Vec<([f64; 3], f64)>, cell: &Cell, fold: bool) -> Self {
        let unfolded_count = raw.len();
        let b = cell.reciprocal_2pi();
        let mut kept: Vec<KPoint> = Vec::with_capacity(raw.len());
        for (frac, weight) in raw {
            let frac = wrap_to_bz(frac);
            if fold {
                // Does `-frac` already have a representative? If so, merge into it. A point that
                // is its own negative (Γ and even-mesh zone-boundary points) must not merge with
                // itself, or it would be double counted.
                let neg = wrap_to_bz([-frac[0], -frac[1], -frac[2]]);
                if !same_k(neg, frac) {
                    if let Some(existing) = kept.iter_mut().find(|p| same_k(p.frac, neg)) {
                        existing.weight += weight;
                        existing.time_reversal_pair = true;
                        continue;
                    }
                }
            }
            let cart = b[0] * frac[0] + b[1] * frac[1] + b[2] * frac[2];
            kept.push(KPoint {
                frac,
                cart,
                weight,
                time_reversal_pair: false,
            });
        }
        Self {
            points: kept,
            unfolded_count,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// `true` when the set is Γ only, so the whole SCF can stay in real arithmetic.
    pub fn is_gamma_only(&self) -> bool {
        self.points.len() == 1 && self.points[0].is_gamma()
    }

    /// Sum of the weights. Always 1 up to rounding; used as an internal consistency check.
    pub fn weight_sum(&self) -> f64 {
        self.points.iter().map(|p| p.weight).sum()
    }
}

/// Wrap fractional k coordinates into `[-½, ½)`.
fn wrap_to_bz(mut frac: [f64; 3]) -> [f64; 3] {
    for f in frac.iter_mut() {
        *f -= (*f + 0.5).floor();
        // `-0.5` and `+0.5` are the same point; normalize to `-0.5` so `same_k` is exact.
        if (*f - 0.5).abs() < 1.0e-12 {
            *f = -0.5;
        }
        if f.abs() < 1.0e-12 {
            *f = 0.0;
        }
    }
    frac
}

/// Are two wrapped fractional k points the same point?
fn same_k(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1.0e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic() -> Cell {
        Cell::cubic(7.0).unwrap()
    }

    #[test]
    fn gamma_is_a_single_real_point() {
        let set = KMesh::Gamma.expand(&cubic()).unwrap();
        assert_eq!(set.len(), 1);
        assert!(set.is_gamma_only());
        assert!((set.weight_sum() - 1.0).abs() < 1e-15);
        assert!(set.points[0].phase([3, -2, 5]).abs() < 1e-12);
    }

    #[test]
    fn weights_survive_time_reversal_folding() {
        for n in [1usize, 2, 3, 4, 5] {
            let set = KMesh::grid(n, n, n).expand(&cubic()).unwrap();
            assert!(
                (set.weight_sum() - 1.0).abs() < 1e-12,
                "n={n}: weights sum to {}",
                set.weight_sum()
            );
            assert_eq!(set.unfolded_count, n * n * n);
            // Folding never keeps more points than the full mesh. It strictly reduces the count
            // only once the mesh contains a k that is not its own negative — for n = 1 and
            // n = 2 every Γ-centred point sits at 0 or −½ in each direction, so nothing folds.
            assert!(set.len() <= n * n * n);
            if n >= 3 {
                assert!(set.len() < n * n * n, "n={n}: nothing folded");
            } else {
                assert_eq!(
                    set.len(),
                    n * n * n,
                    "n={n}: self-negative mesh must not fold"
                );
            }
        }
    }

    #[test]
    fn self_negative_points_are_not_double_weighted() {
        // A 2×2×2 Γ-centred mesh is entirely self-negative points (0 and -1/2 in each
        // direction), so nothing may fold and every weight must stay 1/8.
        let set = KMesh::grid(2, 2, 2).expand(&cubic()).unwrap();
        assert_eq!(set.len(), 8);
        for p in &set.points {
            assert!(
                (p.weight - 0.125).abs() < 1e-12,
                "weight {} at {:?}",
                p.weight,
                p.frac
            );
            assert!(!p.time_reversal_pair);
        }
    }

    #[test]
    fn folding_pairs_k_with_minus_k() {
        // A 3×1×1 mesh has Γ (weight 1/3) and ±1/3 which fold into one point of weight 2/3.
        let set = KMesh::grid(3, 1, 1).expand(&cubic()).unwrap();
        assert_eq!(set.len(), 2);
        let mut weights: Vec<f64> = set.points.iter().map(|p| p.weight).collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!(set.points.iter().any(|p| p.time_reversal_pair));
    }

    #[test]
    fn non_periodic_directions_are_never_sampled() {
        let slab = Cell::new(&[Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 6.0, 0.0)]).unwrap();
        let set = KMesh::grid(3, 3, 4).expand(&slab).unwrap();
        // The third direction collapses to 1 point, so the mesh is 3×3×1 = 9 before folding.
        assert_eq!(set.unfolded_count, 9);
        assert!(set.points.iter().all(|p| p.frac[2] == 0.0));

        let wire = Cell::new(&[Vec3::new(4.0, 0.0, 0.0)]).unwrap();
        let set = KMesh::grid(5, 7, 7).expand(&wire).unwrap();
        assert_eq!(set.unfolded_count, 5);
        assert!(set
            .points
            .iter()
            .all(|p| p.frac[1] == 0.0 && p.frac[2] == 0.0));
    }

    #[test]
    fn cartesian_wavevector_reproduces_the_fractional_phase() {
        let cell = Cell::new(&[
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.2, 4.8, 0.0),
            Vec3::new(0.3, -0.7, 6.5),
        ])
        .unwrap();
        let set = KMesh::grid(4, 4, 4).expand(&cell).unwrap();
        for p in &set.points {
            for t in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, -1, 3]] {
                let from_cart = p.cart.dot(cell.translation(t));
                assert!(
                    (from_cart - p.phase(t)).abs() < 1e-9,
                    "phase mismatch at {:?} t={t:?}: {from_cart} vs {}",
                    p.frac,
                    p.phase(t)
                );
            }
        }
    }

    #[test]
    fn explicit_lists_renormalize_and_are_validated() {
        let set = KMesh::Explicit(vec![([0.0, 0.0, 0.0], 2.0), ([0.25, 0.0, 0.0], 2.0)])
            .expand(&cubic())
            .unwrap();
        assert_eq!(set.len(), 2, "an explicit list must not be folded");
        assert!((set.weight_sum() - 1.0).abs() < 1e-15);
        assert!(KMesh::Explicit(vec![]).validate().is_err());
        assert!(KMesh::Explicit(vec![([0.0; 3], -1.0)]).validate().is_err());
        assert!(KMesh::MonkhorstPack {
            n: [0, 1, 1],
            shift: [0.0; 3],
            gamma_centred: true
        }
        .validate()
        .is_err());
    }
}
