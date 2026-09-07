// SPDX-License-Identifier: GPL-3.0-or-later

//! Small 3-vector / 3×3-matrix algebra, ported from `gfn1-rs`'s `math.rs`.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    #[inline]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n <= f64::EPSILON {
            Self::zero()
        } else {
            self / n
        }
    }
    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    #[inline]
    pub fn get(self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// Column-major 3×3 matrix (three column vectors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub col: [Vec3; 3],
}

impl Mat3 {
    #[inline]
    pub const fn from_columns(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { col: [a, b, c] }
    }
    /// Build from three **row** vectors. `from_rows(r0, r1, r2)[(i, j)] == r_i[j]`.
    #[inline]
    pub fn from_rows(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self::from_columns(
            Vec3::new(a.x, b.x, c.x),
            Vec3::new(a.y, b.y, c.y),
            Vec3::new(a.z, b.z, c.z),
        )
    }
    #[inline]
    pub fn zero() -> Self {
        Self::from_columns(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }
    #[inline]
    pub fn identity() -> Self {
        Self::from_columns(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    }
    /// Outer product `a ⊗ b`, i.e. `out[(i, j)] = a_i b_j`. This is the per-pair building block
    /// of the virial stress `σ = (1/V) Σ (∂E/∂d) ⊗ d`.
    #[inline]
    pub fn outer(a: Vec3, b: Vec3) -> Self {
        Self::from_columns(a * b.x, a * b.y, a * b.z)
    }
    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        self.col[0] * v.x + self.col[1] * v.y + self.col[2] * v.z
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.col[j].get(i)
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        match i {
            0 => self.col[j].x = value,
            1 => self.col[j].y = value,
            _ => self.col[j].z = value,
        }
    }
    #[inline]
    pub fn add_assign_at(&mut self, i: usize, j: usize, value: f64) {
        self.set(i, j, self.get(i, j) + value);
    }
    #[inline]
    pub fn transpose(&self) -> Self {
        Self::from_rows(self.col[0], self.col[1], self.col[2])
    }
    /// `½ (M + Mᵀ)`. A virial accumulated pair-by-pair is symmetric only after the whole sum,
    /// so symmetrizing at the end is both correct and a useful residual check.
    #[inline]
    pub fn symmetrized(&self) -> Self {
        let t = self.transpose();
        let half = |a: Vec3, b: Vec3| (a + b) * 0.5;
        Self::from_columns(
            half(self.col[0], t.col[0]),
            half(self.col[1], t.col[1]),
            half(self.col[2], t.col[2]),
        )
    }
    #[inline]
    pub fn trace(&self) -> f64 {
        self.col[0].x + self.col[1].y + self.col[2].z
    }
    #[inline]
    pub fn scaled(&self, s: f64) -> Self {
        Self::from_columns(self.col[0] * s, self.col[1] * s, self.col[2] * s)
    }
    #[inline]
    pub fn plus(&self, other: &Self) -> Self {
        Self::from_columns(
            self.col[0] + other.col[0],
            self.col[1] + other.col[1],
            self.col[2] + other.col[2],
        )
    }
    /// Voigt order `[xx, yy, zz, yz, xz, xy]` — the convention used by ASE and by MOPAC's
    /// `voigt` array (`moldat.F90` `calculate_voigt`). Symmetrizes first, so an
    /// almost-symmetric input does not silently lose its antisymmetric part.
    #[inline]
    pub fn to_voigt(&self) -> [f64; 6] {
        let s = self.symmetrized();
        [
            s.get(0, 0),
            s.get(1, 1),
            s.get(2, 2),
            s.get(1, 2),
            s.get(0, 2),
            s.get(0, 1),
        ]
    }
    /// Inverse of [`Mat3::to_voigt`].
    #[inline]
    pub fn from_voigt(v: [f64; 6]) -> Self {
        Self::from_rows(
            Vec3::new(v[0], v[5], v[4]),
            Vec3::new(v[5], v[1], v[3]),
            Vec3::new(v[4], v[3], v[2]),
        )
    }
    /// Matrix product `self · other`.
    #[inline]
    pub fn mul_mat(&self, other: &Self) -> Self {
        Self::from_columns(
            self.mul_vec(other.col[0]),
            self.mul_vec(other.col[1]),
            self.mul_vec(other.col[2]),
        )
    }
    /// Largest absolute element — a convergence measure for a stress tensor.
    #[inline]
    pub fn max_abs(&self) -> f64 {
        self.col
            .iter()
            .flat_map(|c| c.to_array())
            .fold(0.0_f64, |m, v| m.max(v.abs()))
    }
}
