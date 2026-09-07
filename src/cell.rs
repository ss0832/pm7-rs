// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic lattice: 1-D (polymer), 2-D (layer/slab), and 3-D (crystal).
//!
//! A [`Cell`] stores up to three **translation vectors** in Bohr, matching MOPAC's `Tv`
//! convention, plus how many of them are periodic. The non-periodic directions are *not*
//! represented by a vacuum layer: a 1-D cell has exactly one translation vector and the other two
//! directions are genuinely open, so the electrostatics uses the 1-D lattice sum rather than a
//! 3-D sum with an arbitrary vacuum gap. That is why [`Periodicity`] is part of the type instead
//! of a flag the caller has to remember to honour.
//!
//! For the reciprocal lattice and cell "volume" the missing directions are completed with an
//! orthonormal basis of the complementary subspace (see [`Cell::completed_vectors`]). With that
//! completion the same reciprocal-lattice formula works in every dimension, and `volume()` means
//! length (1-D), area (2-D), or volume (3-D) — the quantity a stress or a charge density is
//! divided by.

use crate::error::{Pm7Error, Result};
use crate::math::{Mat3, Vec3};

/// Number of periodic lattice directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Periodicity {
    /// Molecule: no translation vectors.
    #[default]
    Molecule,
    /// Polymer / wire: one translation vector.
    OneD,
    /// Layer / slab: two translation vectors.
    TwoD,
    /// Crystal: three translation vectors.
    ThreeD,
}

impl Periodicity {
    /// Number of periodic directions (0..=3).
    pub const fn dim(self) -> usize {
        match self {
            Periodicity::Molecule => 0,
            Periodicity::OneD => 1,
            Periodicity::TwoD => 2,
            Periodicity::ThreeD => 3,
        }
    }

    /// Build from a count of translation vectors.
    pub fn from_dim(dim: usize) -> Result<Self> {
        Ok(match dim {
            0 => Periodicity::Molecule,
            1 => Periodicity::OneD,
            2 => Periodicity::TwoD,
            3 => Periodicity::ThreeD,
            other => {
                return Err(Pm7Error::InvalidInput(format!(
                    "periodicity must be 0, 1, 2, or 3 translation vectors, got {other}"
                )))
            }
        })
    }

    /// Build from ASE-style per-direction flags. The periodic directions must come first
    /// (`[true, true, false]` is a slab; `[true, false, true]` is rejected) because a `Cell`
    /// stores its periodic translation vectors contiguously.
    ///
    /// This is deliberately still strict: `Periodicity` is a *count*, and a count cannot record
    /// which axes the caller meant. [`AxisRotation::for_flags`] is the way to accept an arbitrary
    /// pattern — it says how to reorder the lattice vectors so this function will take them.
    pub fn from_flags(pbc: [bool; 3]) -> Result<Self> {
        let dim = pbc.iter().filter(|p| **p).count();
        if pbc[..dim].iter().any(|p| !*p) {
            return Err(Pm7Error::InvalidInput(format!(
                "periodic directions must be the leading ones: got {pbc:?}; \
                 reorder the lattice vectors so the periodic ones come first, or build the cell \
                 through `AxisRotation::for_flags`, which does the reordering for you"
            )));
        }
        Periodicity::from_dim(dim)
    }

    /// ASE-style per-direction flags.
    pub const fn flags(self) -> [bool; 3] {
        let d = self.dim();
        [d >= 1, d >= 2, d >= 3]
    }
}

/// A cyclic reordering of the three lattice-vector slots, so that any per-axis periodicity
/// pattern can be expressed against a [`Cell`] that stores its periodic vectors contiguously.
///
/// # Why a rotation and not a mask
///
/// The alternative is to let `Cell` hold a `[bool; 3]` mask. That would undo the argument the
/// module doc-comment makes: [`Periodicity`] is in the type precisely so no downstream module has
/// to remember to honour a flag, and a mask puts the flag back in ~25 dimension branches in
/// `pbc/ewald.rs` alone. A reordering is invisible to all of them, because they reach the lattice
/// through `cell.dim()` and `cell.vectors()` — a count and a slice.
///
/// **Every per-axis pattern is reachable by a cyclic rotation**, which is what makes this cheap
/// rather than merely cheaper. All eight patterns:
///
/// | requested | rotation | leading-periodic form |
/// |---|---|---|
/// | `FFF`, `TFF`, `TTF`, `TTT` | identity | unchanged |
/// | `FTF`, `FTT` | `(a₂, a₃, a₁)` | `TFF`, `TTF` |
/// | `FFT`, `TFT` | `(a₃, a₁, a₂)` | `TFF`, `TTF` |
///
/// Cyclic rotations preserve handedness; a transposition would not. Handedness is load-bearing —
/// [`Cell::reciprocal`] divides by a signed determinant, and the Berry-phase sign conventions
/// depend on it — so restricting to the three rotations means there is no sign to chase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AxisRotation {
    /// How many places to rotate left: slot `k` of the reordered triple is slot `(k + shift) % 3`
    /// of the original. `0`, `1` or `2`.
    shift: usize,
}

impl AxisRotation {
    /// The identity: the lattice vectors are already in an order `Cell` accepts.
    pub const IDENTITY: Self = Self { shift: 0 };

    /// The rotation that moves the periodic directions of `pbc` to the front.
    ///
    /// Never fails: the table above covers all eight patterns.
    pub fn for_flags(pbc: [bool; 3]) -> Self {
        for shift in 0..3 {
            let rotated = Self { shift }.apply(pbc);
            let dim = rotated.iter().filter(|p| **p).count();
            if !rotated[..dim].iter().any(|p| !*p) {
                return Self { shift };
            }
        }
        // Unreachable: `shift = 0` already succeeds for any pattern whose `true`s are contiguous
        // from the front, and the three rotations cover the rest. Returning the identity rather
        // than panicking keeps a hypothetical fourth case a wrong answer and not a crash — and
        // `every_pattern_is_reachable_by_a_rotation` in this module's tests is what rules it out.
        Self::IDENTITY
    }

    /// Whether this reorders anything.
    pub const fn is_identity(self) -> bool {
        self.shift == 0
    }

    /// Reorder a per-lattice-vector triple into the cell's storage order.
    ///
    /// Use it on everything the user indexes by lattice vector in their own axis labelling: the
    /// lattice rows themselves, `--kpoints`, `--kshift`, `--supercell`, and fractional `q`.
    pub fn apply<T: Copy>(self, x: [T; 3]) -> [T; 3] {
        [
            x[self.shift % 3],
            x[(self.shift + 1) % 3],
            x[(self.shift + 2) % 3],
        ]
    }

    /// The inverse of [`Self::apply`], for reporting a per-lattice-vector triple back in the
    /// caller's original axis order.
    pub fn undo<T: Copy>(self, y: [T; 3]) -> [T; 3] {
        let mut out = y;
        for k in 0..3 {
            out[(self.shift + k) % 3] = y[k];
        }
        out
    }

    /// The permutation as axis indices, for messages: `[2, 0, 1]` reads "a₃, a₁, a₂".
    pub fn order(self) -> [usize; 3] {
        self.apply([0, 1, 2])
    }
}

/// A periodic cell: up to three translation vectors in **Bohr**.
///
/// `vectors[0..dim]` are the periodic translation vectors; entries beyond `dim` are ignored and
/// stored as zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    vectors: [Vec3; 3],
    periodicity: Periodicity,
}

impl Cell {
    /// Build a cell from `dim` translation vectors in Bohr.
    ///
    /// Rejects a non-finite, zero-length, or linearly dependent set, because every downstream
    /// consumer (reciprocal lattice, minimum image, Ewald) would otherwise produce silent
    /// nonsense rather than an error.
    pub fn new(vectors: &[Vec3]) -> Result<Self> {
        let periodicity = Periodicity::from_dim(vectors.len())?;
        let mut stored = [Vec3::zero(); 3];
        for (slot, v) in stored.iter_mut().zip(vectors) {
            if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
                return Err(Pm7Error::InvalidInput(
                    "lattice vectors must be finite".into(),
                ));
            }
            if v.norm2() < 1.0e-12 {
                return Err(Pm7Error::InvalidInput(
                    "lattice vectors must have non-zero length".into(),
                ));
            }
            *slot = *v;
        }
        let cell = Self {
            vectors: stored,
            periodicity,
        };
        cell.check_independent()?;
        Ok(cell)
    }

    /// Build from Ångström row vectors (`rows[i]` is lattice vector `i`), the convention used by
    /// ASE, extended XYZ, and CIF.
    pub fn from_angstrom_rows(rows: &[[f64; 3]]) -> Result<Self> {
        let a0 = crate::constants::ANGSTROM_TO_BOHR;
        let v: Vec<Vec3> = rows
            .iter()
            .map(|r| Vec3::new(r[0], r[1], r[2]) * a0)
            .collect();
        Self::new(&v)
    }

    /// Build from three Ångström row vectors plus an ASE-style per-axis periodicity pattern,
    /// reordering the lattice vectors as needed.
    ///
    /// Returns the cell — `None` when nothing is periodic — together with the [`AxisRotation`]
    /// that was applied, so the caller can put its own per-lattice-vector inputs (`--kpoints`,
    /// `--supercell`, fractional `q`) into the same order.
    ///
    /// A slab built along *y* is an ordinary thing for an ASE user to have, and until 0.2.3 every
    /// surface refused it: `Cell::from_flags`, `native`'s `pbc=` argument, and the ASE calculator
    /// each said "reorder the lattice vectors so the periodic ones come first" and left the user
    /// to do it.
    pub fn from_angstrom_rows_pbc(
        rows: &[[f64; 3]; 3],
        pbc: [bool; 3],
    ) -> Result<(Option<Self>, AxisRotation)> {
        let rotation = AxisRotation::for_flags(pbc);
        let ordered = rotation.apply(*rows);
        let dim = pbc.iter().filter(|p| **p).count();
        if dim == 0 {
            return Ok((None, rotation));
        }
        Ok((Some(Self::from_angstrom_rows(&ordered[..dim])?), rotation))
    }

    /// An orthorhombic 3-D cell with the given edge lengths in Bohr.
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Result<Self> {
        Self::new(&[
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, b, 0.0),
            Vec3::new(0.0, 0.0, c),
        ])
    }

    /// A cubic 3-D cell with edge `a` in Bohr.
    pub fn cubic(a: f64) -> Result<Self> {
        Self::orthorhombic(a, a, a)
    }

    pub const fn periodicity(&self) -> Periodicity {
        self.periodicity
    }

    /// Number of periodic directions (0..=3).
    pub const fn dim(&self) -> usize {
        self.periodicity.dim()
    }

    /// The periodic translation vectors (length `dim()`), in Bohr.
    pub fn vectors(&self) -> &[Vec3] {
        &self.vectors[..self.periodicity.dim()]
    }

    /// Lattice vectors in Ångström row form, for I/O and the Python/ASE boundary.
    pub fn angstrom_rows(&self) -> Vec<[f64; 3]> {
        let a0 = crate::constants::BOHR_TO_ANGSTROM;
        self.vectors()
            .iter()
            .map(|v| [v.x * a0, v.y * a0, v.z * a0])
            .collect()
    }

    /// The `dim` periodic vectors completed to a full 3-D basis with an orthonormal basis of the
    /// complementary subspace.
    ///
    /// This is what lets one reciprocal-lattice and one minimum-image implementation serve every
    /// dimension: in a non-periodic direction the completion vector has unit length, so the
    /// corresponding reciprocal vector is the unit normal and the "cell measure" degenerates to
    /// length or area instead of volume.
    pub fn completed_vectors(&self) -> [Vec3; 3] {
        let dim = self.periodicity.dim();
        let mut basis = self.vectors;
        match dim {
            0 => {
                basis[0] = Vec3::new(1.0, 0.0, 0.0);
                basis[1] = Vec3::new(0.0, 1.0, 0.0);
                basis[2] = Vec3::new(0.0, 0.0, 1.0);
            }
            1 => {
                let a = basis[0].normalized();
                let (u, v) = orthonormal_complement(a);
                basis[1] = u;
                basis[2] = v;
            }
            2 => {
                let n = basis[0].cross(basis[1]);
                basis[2] = n.normalized();
            }
            _ => {}
        }
        basis
    }

    /// Cell measure: length (1-D, Bohr), area (2-D, Bohr²), or volume (3-D, Bohr³).
    ///
    /// This is the quantity an extensive energy is divided by to give a stress, and the quantity
    /// a charge is divided by to give the neutralizing background density. For a molecule it is
    /// `1.0` (the completion basis is the identity), which keeps the formulas total rather than
    /// intensive — callers must not divide molecular energies by it.
    pub fn measure(&self) -> f64 {
        let b = self.completed_vectors();
        b[0].dot(b[1].cross(b[2])).abs()
    }

    /// 3-D volume in Bohr³. Only meaningful for [`Periodicity::ThreeD`]; returns `None` otherwise
    /// so a caller that genuinely needs a volume (pressure, density) cannot silently get an area.
    pub fn volume(&self) -> Option<f64> {
        (self.periodicity == Periodicity::ThreeD).then(|| self.measure())
    }

    /// Reciprocal lattice vectors **without** the `2π` factor, i.e. `b_i · a_j = δ_ij`.
    ///
    /// Rows beyond `dim()` belong to the completion basis and are the unit normals of the
    /// non-periodic directions; they are what converts a Cartesian displacement into the
    /// fractional coordinate that the minimum-image search wraps.
    pub fn reciprocal(&self) -> [Vec3; 3] {
        let b = self.completed_vectors();
        let det = b[0].dot(b[1].cross(b[2]));
        let inv = 1.0 / det;
        [
            b[1].cross(b[2]) * inv,
            b[2].cross(b[0]) * inv,
            b[0].cross(b[1]) * inv,
        ]
    }

    /// Reciprocal lattice vectors **with** the `2π` factor, the convention for `k` and `q`
    /// vectors: `k = Σ_i frac_i · b2pi_i` and `exp(i k·T)` is 1 for every lattice vector `T`.
    pub fn reciprocal_2pi(&self) -> [Vec3; 3] {
        let two_pi = std::f64::consts::TAU;
        let r = self.reciprocal();
        [r[0] * two_pi, r[1] * two_pi, r[2] * two_pi]
    }

    /// Cartesian → fractional coordinates along the completed basis.
    pub fn to_fractional(&self, cart: Vec3) -> Vec3 {
        let r = self.reciprocal();
        Vec3::new(r[0].dot(cart), r[1].dot(cart), r[2].dot(cart))
    }

    /// Fractional (along the completed basis) → Cartesian.
    pub fn to_cartesian(&self, frac: Vec3) -> Vec3 {
        let b = self.completed_vectors();
        b[0] * frac.x + b[1] * frac.y + b[2] * frac.z
    }

    /// The lattice translation for integer cell indices `t`. Indices beyond `dim()` are ignored,
    /// so a caller can always pass a `[i32; 3]`.
    pub fn translation(&self, t: [i32; 3]) -> Vec3 {
        let mut out = Vec3::zero();
        for k in 0..self.periodicity.dim() {
            out += self.vectors[k] * t[k] as f64;
        }
        out
    }

    /// Wrap a Cartesian position into the cell (fractional coordinates in `[0, 1)` along the
    /// periodic directions; non-periodic directions are left alone).
    pub fn wrap(&self, cart: Vec3) -> Vec3 {
        let dim = self.periodicity.dim();
        if dim == 0 {
            return cart;
        }
        let frac = self.to_fractional(cart);
        let mut f = [frac.x, frac.y, frac.z];
        for slot in f.iter_mut().take(dim) {
            *slot -= slot.floor();
        }
        self.to_cartesian(Vec3::new(f[0], f[1], f[2]))
    }

    /// The minimum-image displacement equivalent to `d`, together with the integer lattice
    /// translation that realizes it.
    ///
    /// Returns `(d_min, t)` with `d_min = d + Σ_k t_k a_k`. The nearest-fractional-image guess is
    /// exact for an orthogonal cell but can miss for a strongly skewed one, so the guess is
    /// refined by an explicit ±1 search around it — cheap, and it removes a whole class of
    /// silent errors in triclinic cells.
    pub fn minimum_image(&self, d: Vec3) -> (Vec3, [i32; 3]) {
        let dim = self.periodicity.dim();
        if dim == 0 {
            return (d, [0; 3]);
        }
        let frac = self.to_fractional(d);
        let guess = [
            if dim >= 1 { -frac.x.round() as i32 } else { 0 },
            if dim >= 2 { -frac.y.round() as i32 } else { 0 },
            if dim >= 3 { -frac.z.round() as i32 } else { 0 },
        ];
        let mut best_t = guess;
        let mut best = d + self.translation(guess);
        let mut best_n2 = best.norm2();
        let span = |k: usize| if k < dim { -1..=1 } else { 0..=0 };
        for i in span(0) {
            for j in span(1) {
                for k in span(2) {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let t = [guess[0] + i, guess[1] + j, guess[2] + k];
                    let cand = d + self.translation(t);
                    let n2 = cand.norm2();
                    if n2 < best_n2 {
                        best_n2 = n2;
                        best = cand;
                        best_t = t;
                    }
                }
            }
        }
        (best, best_t)
    }

    /// Every integer cell index `t` whose translation could bring two atoms within `cutoff`
    /// (Bohr) of each other, given that both atoms already lie in the central cell.
    ///
    /// The bound per direction is `ceil(cutoff * |b_k| + margin)`, where `b_k` is the reciprocal
    /// vector without `2π`: `|b_k|⁻¹` is the interplanar spacing along direction `k`, so this is
    /// the exact number of layers that can reach within `cutoff`. `margin` covers the fact that
    /// the two atoms are not at the same point inside the cell.
    pub fn image_range(&self, cutoff: f64, margin: f64) -> [i32; 3] {
        let dim = self.periodicity.dim();
        let recip = self.reciprocal();
        let mut out = [0i32; 3];
        for k in 0..dim {
            let spacing_inv = recip[k].norm();
            out[k] = ((cutoff + margin) * spacing_inv).ceil() as i32;
        }
        out
    }

    /// All integer cell indices within [`Cell::image_range`], ordered deterministically.
    ///
    /// The order is fixed (lexicographic in `t`) so that a summation over images accumulates in
    /// the same order on every run and thread count — the periodic counterpart of the
    /// bit-identical guarantee the molecular paths already make.
    pub fn image_indices(&self, cutoff: f64, margin: f64) -> Vec<[i32; 3]> {
        let n = self.image_range(cutoff, margin);
        let mut out = Vec::new();
        for i in -n[0]..=n[0] {
            for j in -n[1]..=n[1] {
                for k in -n[2]..=n[2] {
                    out.push([i, j, k]);
                }
            }
        }
        out
    }

    /// Apply an infinitesimal strain: `a_k → (1 + ε) a_k`. Used to finite-difference the stress
    /// tensor in tests, and by the variable-cell optimizer.
    pub fn strained(&self, eps: &Mat3) -> Result<Self> {
        let f = |v: Vec3| v + eps.mul_vec(v);
        let v: Vec<Vec3> = self.vectors().iter().map(|a| f(*a)).collect();
        Self::new(&v)
    }

    /// Scale every lattice vector by `s` (isotropic volume change).
    pub fn scaled(&self, s: f64) -> Result<Self> {
        let v: Vec<Vec3> = self.vectors().iter().map(|a| *a * s).collect();
        Self::new(&v)
    }

    /// Build the `n[0] × n[1] × n[2]` Born–von Kármán supercell of this cell.
    ///
    /// Returns the supercell lattice and the list of translations of the original cell that fill
    /// it, in a fixed order. This is the construction behind the BvK identity that a `n1×n2×n3`
    /// k-mesh on this cell gives the same energy per cell as the Γ point of the supercell — the
    /// sharpest available test of the k-point machinery.
    pub fn supercell(&self, n: [usize; 3]) -> Result<(Self, Vec<[i32; 3]>)> {
        let dim = self.periodicity.dim();
        for (k, &nk) in n.iter().enumerate() {
            if nk == 0 {
                return Err(Pm7Error::InvalidInput(
                    "supercell repetitions must be >= 1".into(),
                ));
            }
            if k >= dim && nk != 1 {
                return Err(Pm7Error::InvalidInput(format!(
                    "cannot repeat non-periodic direction {k} of a {dim}-D cell {nk} times"
                )));
            }
        }
        let v: Vec<Vec3> = (0..dim).map(|k| self.vectors[k] * n[k] as f64).collect();
        let mut shifts = Vec::with_capacity(n[0] * n[1] * n[2]);
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    shifts.push([i as i32, j as i32, k as i32]);
                }
            }
        }
        Ok((Self::new(&v)?, shifts))
    }

    fn check_independent(&self) -> Result<()> {
        let dim = self.periodicity.dim();
        // Scale-free test: the completed basis' determinant against the product of the vector
        // lengths. A near-zero ratio means the vectors are (nearly) linearly dependent, which
        // would make the reciprocal lattice blow up.
        if dim == 0 {
            return Ok(());
        }
        let b = self.completed_vectors();
        let det = b[0].dot(b[1].cross(b[2])).abs();
        let scale: f64 = b.iter().map(|v| v.norm()).product();
        if scale <= 0.0 || det / scale < 1.0e-8 {
            return Err(Pm7Error::InvalidInput(
                "lattice vectors are linearly dependent (or nearly so)".into(),
            ));
        }
        Ok(())
    }
}

/// Two unit vectors completing `a` (already normalized) to a right-handed orthonormal basis.
fn orthonormal_complement(a: Vec3) -> (Vec3, Vec3) {
    // Pick the Cartesian axis least aligned with `a` so the cross product is well conditioned.
    let ax = a.x.abs();
    let ay = a.y.abs();
    let az = a.z.abs();
    let seed = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = a.cross(seed).normalized();
    let v = a.cross(u).normalized();
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn reciprocal_is_dual_to_the_lattice_in_every_dimension() {
        let cells = [
            Cell::new(&[Vec3::new(3.1, 0.4, -0.2)]).unwrap(),
            Cell::new(&[Vec3::new(4.0, 0.0, 0.0), Vec3::new(-2.0, 3.5, 0.0)]).unwrap(),
            Cell::new(&[
                Vec3::new(6.7, 0.0, 0.0),
                Vec3::new(1.3, 6.1, 0.0),
                Vec3::new(0.5, -0.9, 7.2),
            ])
            .unwrap(),
        ];
        for cell in cells {
            let a = cell.completed_vectors();
            let b = cell.reciprocal();
            for i in 0..3 {
                for j in 0..3 {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        approx(b[i].dot(a[j]), expect, 1e-12),
                        "dim {}: b{i}·a{j} = {}",
                        cell.dim(),
                        b[i].dot(a[j])
                    );
                }
            }
        }
    }

    #[test]
    fn phase_factor_is_unity_on_every_lattice_vector() {
        // `reciprocal_2pi` must satisfy exp(i·b_i·a_j) = 1, i.e. b_i·a_j ∈ 2πZ.
        let cell = Cell::new(&[
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(2.5, 4.33, 0.0),
            Vec3::new(0.0, 0.0, 9.1),
        ])
        .unwrap();
        let b = cell.reciprocal_2pi();
        for (i, bi) in b.iter().enumerate() {
            for (j, aj) in cell.vectors().iter().enumerate() {
                let phase = bi.dot(*aj) / std::f64::consts::TAU;
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(phase, expect, 1e-12), "b{i}·a{j}/2π = {phase}");
            }
        }
    }

    #[test]
    fn measure_is_length_area_or_volume() {
        let one = Cell::new(&[Vec3::new(0.0, 3.0, 4.0)]).unwrap();
        assert!(approx(one.measure(), 5.0, 1e-12));
        assert_eq!(one.volume(), None);

        let two = Cell::new(&[Vec3::new(4.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0)]).unwrap();
        assert!(approx(two.measure(), 12.0, 1e-12));
        assert_eq!(two.volume(), None);

        let three = Cell::orthorhombic(2.0, 3.0, 4.0).unwrap();
        assert!(approx(three.measure(), 24.0, 1e-12));
        assert!(approx(three.volume().unwrap(), 24.0, 1e-12));
    }

    #[test]
    fn minimum_image_beats_the_fractional_guess_in_a_skewed_cell() {
        // A strongly skewed 2-D cell: rounding fractional coordinates alone is not enough.
        let cell = Cell::new(&[Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.94, 0.35, 0.0)]).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..17 {
            for j in 0..17 {
                let d = Vec3::new(i as f64 * 0.11 - 0.9, j as f64 * 0.07 - 0.5, 0.0);
                let (dmin, t) = cell.minimum_image(d);
                // The returned translation must actually produce the returned displacement.
                let rebuilt = d + cell.translation(t);
                assert!((rebuilt - dmin).norm() < 1e-12);
                // Brute-force check against a wide image search.
                let mut brute = f64::INFINITY;
                for a in -4..=4 {
                    for b in -4..=4 {
                        let cand = d + cell.translation([a, b, 0]);
                        brute = brute.min(cand.norm2());
                    }
                }
                worst = worst.max(dmin.norm2() - brute);
                assert!(
                    dmin.norm2() <= brute + 1e-12,
                    "minimum image missed: {} vs {}",
                    dmin.norm2(),
                    brute
                );
            }
        }
    }

    #[test]
    fn image_range_covers_every_reachable_cell() {
        let cell = Cell::new(&[
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.0, 4.0, 0.0),
            Vec3::new(0.0, 1.0, 6.0),
        ])
        .unwrap();
        let cutoff = 12.0;
        let n = cell.image_range(cutoff, 0.0);
        // Any translation shorter than `cutoff` must lie inside the reported range.
        for i in -12..=12 {
            for j in -12..=12 {
                for k in -12..=12 {
                    if cell.translation([i, j, k]).norm() <= cutoff {
                        assert!(
                            i.abs() <= n[0] && j.abs() <= n[1] && k.abs() <= n[2],
                            "translation [{i},{j},{k}] within {cutoff} but outside range {n:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wrap_puts_atoms_in_the_cell_and_keeps_open_directions() {
        let cell = Cell::new(&[Vec3::new(4.0, 0.0, 0.0), Vec3::new(0.0, 5.0, 0.0)]).unwrap();
        let p = Vec3::new(9.5, -7.25, 3.75);
        let w = cell.wrap(p);
        let f = cell.to_fractional(w);
        assert!((0.0..1.0).contains(&f.x), "x fraction {f:?}");
        assert!((0.0..1.0).contains(&f.y), "y fraction {f:?}");
        // The open direction must be untouched, not folded into a fictitious vacuum cell.
        assert!(approx(w.z, 3.75, 1e-12));
        // Wrapping moves the atom by a lattice vector, so the difference is a lattice translation.
        let d = p - w;
        let df = cell.to_fractional(d);
        assert!(approx(df.x, df.x.round(), 1e-10));
        assert!(approx(df.y, df.y.round(), 1e-10));
    }

    #[test]
    fn supercell_multiplies_the_measure_and_enumerates_its_cells() {
        let cell = Cell::orthorhombic(3.0, 4.0, 5.0).unwrap();
        let (sc, shifts) = cell.supercell([2, 3, 1]).unwrap();
        assert!(approx(sc.measure(), cell.measure() * 6.0, 1e-12));
        assert_eq!(shifts.len(), 6);
        // A 2-D cell cannot be repeated along its open direction.
        let slab = Cell::new(&[Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.0, 4.0, 0.0)]).unwrap();
        assert!(slab.supercell([2, 2, 2]).is_err());
        assert!(slab.supercell([2, 2, 1]).is_ok());
    }

    #[test]
    fn degenerate_lattices_are_rejected() {
        assert!(Cell::new(&[Vec3::zero()]).is_err());
        assert!(Cell::new(&[Vec3::new(f64::NAN, 0.0, 0.0)]).is_err());
        // Two parallel vectors span no area.
        assert!(Cell::new(&[Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)]).is_err());
        // Three coplanar vectors span no volume.
        assert!(Cell::new(&[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ])
        .is_err());
        assert!(Periodicity::from_dim(4).is_err());
        assert!(Periodicity::from_flags([true, false, true]).is_err());
        assert_eq!(
            Periodicity::from_flags([true, true, false]).unwrap(),
            Periodicity::TwoD
        );
    }

    /// The claim the `AxisRotation` doc table makes: three cyclic rotations cover all eight
    /// patterns. If that were false, some pattern would be unrepresentable and `for_flags` would
    /// quietly return the identity for it.
    #[test]
    fn every_pattern_is_reachable_by_a_rotation() {
        for bits in 0..8u8 {
            let pbc = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
            let rotation = AxisRotation::for_flags(pbc);
            let rotated = rotation.apply(pbc);
            let dim = pbc.iter().filter(|p| **p).count();
            assert_eq!(
                rotated.iter().filter(|p| **p).count(),
                dim,
                "a rotation changed how many directions are periodic: {pbc:?}"
            );
            assert!(
                Periodicity::from_flags(rotated).is_ok(),
                "{pbc:?} rotated to {rotated:?}, which Cell still will not take"
            );
        }
    }

    /// `undo` inverts `apply`, which is what lets a caller report in the user's axis order.
    #[test]
    fn the_rotation_is_invertible() {
        for bits in 0..8u8 {
            let pbc = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
            let rotation = AxisRotation::for_flags(pbc);
            let labels = [10, 20, 30];
            assert_eq!(rotation.undo(rotation.apply(labels)), labels, "{pbc:?}");
            assert_eq!(rotation.apply(rotation.undo(labels)), labels, "{pbc:?}");
        }
    }

    /// A rotation reorders the lattice vectors and changes no physical quantity of the cell: the
    /// measure is the same, and it stays right-handed. A *transposition* would flip the sign of
    /// the determinant, which is why only the three rotations are admitted.
    #[test]
    fn a_rotation_preserves_the_measure_and_the_handedness() {
        let rows = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let (reference, _) = Cell::from_angstrom_rows_pbc(&rows, [true, true, true]).unwrap();
        let reference = reference.unwrap();
        for pbc in [
            [true, true, true],
            [true, false, true],
            [false, true, true],
            [false, false, true],
            [false, true, false],
        ] {
            let (cell, rotation) = Cell::from_angstrom_rows_pbc(&rows, pbc).unwrap();
            let cell = cell.unwrap();
            if cell.dim() == 3 {
                let volume = cell.volume().unwrap();
                assert!(
                    (volume - reference.volume().unwrap()).abs() < 1.0e-9,
                    "{pbc:?} changed the volume to {volume}"
                );
                let v = cell.vectors();
                assert!(
                    v[0].cross(v[1]).dot(v[2]) > 0.0,
                    "{pbc:?} produced a left-handed cell"
                );
            }
            // The vectors that survive are exactly the ones the caller marked periodic.
            let wanted: Vec<[f64; 3]> = rotation
                .apply(rows)
                .into_iter()
                .take(cell.dim())
                .collect::<Vec<_>>();
            for (k, row) in wanted.iter().enumerate() {
                let got = cell.vectors()[k] / crate::constants::ANGSTROM_TO_BOHR;
                assert!(
                    (got.x - row[0]).abs() < 1.0e-9
                        && (got.y - row[1]).abs() < 1.0e-9
                        && (got.z - row[2]).abs() < 1.0e-9,
                    "{pbc:?} vector {k}: got {got:?}, wanted {row:?}"
                );
            }
        }
    }

    #[test]
    fn strain_scales_the_measure_to_first_order() {
        let cell = Cell::orthorhombic(6.0, 7.0, 8.0).unwrap();
        let e = 1.0e-5;
        let eps = Mat3::from_columns(
            Vec3::new(e, 0.0, 0.0),
            Vec3::new(0.0, e, 0.0),
            Vec3::new(0.0, 0.0, e),
        );
        let strained = cell.strained(&eps).unwrap();
        let expect = cell.measure() * (1.0 + 3.0 * e);
        assert!(approx(strained.measure(), expect, 1e-6 * cell.measure()));
    }
}
