// SPDX-License-Identifier: GPL-3.0-or-later

//! Multi-variable second-order forward-mode AD (`Dual2N<N>`) and the [`HbScalar`] trait it and
//! `f64` share.
//!
//! The single-variable [`crate::dual2::Dual2`] tracks derivatives w.r.t. one 3-vector, which is
//! enough for the pairwise NDDO/dispersion terms but *not* for the many-body hydrogen-bond
//! correction: each EH+ term couples up to nine atoms, so its analytic Hessian needs the full
//! cross second derivatives ∂²E/∂rₐ∂r_b. `Dual2N<N>` carries the value, an `N`-vector gradient,
//! and the `N×N` Hessian, so instantiating the (geometric) H-bond energy at `Dual2N<27>` yields
//! the exact 27×27 local Hessian block for a bond's atoms in one pass — no finite differences.
//!
//! Only the operations the H-bond energy uses are provided (`HbScalar`); the branchy angle/
//! torsion logic keys off `.val()` and is smooth away from the (degenerate) branch boundaries,
//! exactly as the `f64` port is.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Scalar operations shared by the plain `f64` energy path and the `Dual2N` Hessian path.
pub trait HbScalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn cst(x: f64) -> Self;
    fn val(&self) -> f64;
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn exp(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, x: f64) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;
    fn abs(self) -> Self;
    /// Whichever of `self`/`other` has the smaller value, carrying its derivatives.
    fn min(self, other: Self) -> Self;
    /// Whichever of `self`/`other` has the larger value, carrying its derivatives.
    fn max(self, other: Self) -> Self;
    /// Clamp the value to `[lo, hi]`; a clamped result is a constant (zero derivative), matching
    /// the flat regions MOPAC's `min(max(..))` produces.
    fn clamp2(self, lo: f64, hi: f64) -> Self;
}

impl HbScalar for f64 {
    #[inline]
    fn cst(x: f64) -> Self {
        x
    }
    #[inline]
    fn val(&self) -> f64 {
        *self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn powf(self, x: f64) -> Self {
        f64::powf(self, x)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self <= other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn clamp2(self, lo: f64, hi: f64) -> Self {
        self.clamp(lo, hi)
    }
}

// First-order three-variable AD is sufficient when one atom's x/y/z coordinates are seeded.
// Reusing `Dual` lets the H-bond gradient evaluate each participating atom once instead of six
// energy evaluations for a central difference.
impl HbScalar for crate::dual::Dual {
    #[inline]
    fn cst(x: f64) -> Self {
        crate::dual::Dual::constant(x)
    }
    #[inline]
    fn val(&self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let value = self.v.sqrt();
        self.map(value, if value > 0.0 { 0.5 / value } else { 0.0 })
    }
    #[inline]
    fn recip(self) -> Self {
        let value = self.v.recip();
        self.map(value, -value * value)
    }
    #[inline]
    fn exp(self) -> Self {
        let value = self.v.exp();
        self.map(value, value)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        let value = self.v.powi(n);
        let derivative = if self.v == 0.0 {
            if n == 1 {
                1.0
            } else {
                0.0
            }
        } else {
            n as f64 * self.v.powi(n - 1)
        };
        self.map(value, derivative)
    }
    #[inline]
    fn powf(self, exponent: f64) -> Self {
        let value = self.v.powf(exponent);
        let derivative = if self.v > 0.0 {
            exponent * self.v.powf(exponent - 1.0)
        } else {
            0.0
        };
        self.map(value, derivative)
    }
    #[inline]
    fn cos(self) -> Self {
        self.map(self.v.cos(), -self.v.sin())
    }
    #[inline]
    fn acos(self) -> Self {
        let value = self.v.acos();
        let denominator = (1.0 - self.v * self.v).sqrt();
        self.map(
            value,
            if denominator > 0.0 {
                -1.0 / denominator
            } else {
                0.0
            },
        )
    }
    #[inline]
    fn abs(self) -> Self {
        if self.v >= 0.0 {
            self
        } else {
            -self
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.v <= other.v {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.v >= other.v {
            self
        } else {
            other
        }
    }
    #[inline]
    fn clamp2(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            crate::dual::Dual::constant(lo)
        } else if self.v > hi {
            crate::dual::Dual::constant(hi)
        } else {
            self
        }
    }
}

/// Value + `N`-vector gradient + `N×N` Hessian.
#[derive(Clone, Copy)]
pub struct Dual2N<const N: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [[f64; N]; N],
}

impl<const N: usize> Dual2N<N> {
    #[inline]
    pub fn constant(x: f64) -> Self {
        Self {
            v: x,
            g: [0.0; N],
            h: [[0.0; N]; N],
        }
    }
    /// Independent variable seeded along axis `i` (∂/∂xᵢ = 1, second derivatives 0).
    #[inline]
    pub fn var(x: f64, i: usize) -> Self {
        let mut g = [0.0; N];
        g[i] = 1.0;
        Self {
            v: x,
            g,
            h: [[0.0; N]; N],
        }
    }
    /// Chain rule for a smooth unary φ with `val = φ(v)`, `d1 = φ'(v)`, `d2 = φ''(v)`.
    #[inline]
    fn chain(self, val: f64, d1: f64, d2: f64) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for a in 0..N {
            g[a] = d1 * self.g[a];
        }
        for a in 0..N {
            let ga = self.g[a];
            for b in 0..N {
                h[a][b] = d2 * ga * self.g[b] + d1 * self.h[a][b];
            }
        }
        Self { v: val, g, h }
    }
}

impl<const N: usize> Add for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for a in 0..N {
            self.g[a] += o.g[a];
            for b in 0..N {
                self.h[a][b] += o.h[a][b];
            }
        }
        self
    }
}
impl<const N: usize> Sub for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        self.v -= o.v;
        for a in 0..N {
            self.g[a] -= o.g[a];
            for b in 0..N {
                self.h[a][b] -= o.h[a][b];
            }
        }
        self
    }
}
impl<const N: usize> Mul for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut g = [0.0; N];
        let mut h = [[0.0; N]; N];
        for a in 0..N {
            g[a] = self.g[a] * o.v + self.v * o.g[a];
        }
        for a in 0..N {
            for b in 0..N {
                h[a][b] = self.h[a][b] * o.v
                    + self.g[a] * o.g[b]
                    + self.g[b] * o.g[a]
                    + self.v * o.h[a][b];
            }
        }
        Self {
            v: self.v * o.v,
            g,
            h,
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Div for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        self * o.recip()
    }
}
impl<const N: usize> Neg for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.v = -self.v;
        for a in 0..N {
            self.g[a] = -self.g[a];
            for b in 0..N {
                self.h[a][b] = -self.h[a][b];
            }
        }
        self
    }
}
impl<const N: usize> Add<f64> for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}
impl<const N: usize> Sub<f64> for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}
impl<const N: usize> Mul<f64> for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn mul(mut self, o: f64) -> Self {
        self.v *= o;
        for a in 0..N {
            self.g[a] *= o;
            for b in 0..N {
                self.h[a][b] *= o;
            }
        }
        self
    }
}
impl<const N: usize> Div<f64> for Dual2N<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl<const N: usize> HbScalar for Dual2N<N> {
    #[inline]
    fn cst(x: f64) -> Self {
        Dual2N::constant(x)
    }
    #[inline]
    fn val(&self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        if s > 0.0 {
            self.chain(s, 0.5 / s, -0.25 / (self.v * s))
        } else {
            self.chain(s, 0.0, 0.0)
        }
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.v;
        let r2 = r * r;
        self.chain(r, -r2, 2.0 * r2 * r)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e, e)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        let val = self.v.powi(n);
        if self.v == 0.0 {
            // Our H-bond usage only ever hits n ≥ 1 (n = 2, 4). Give the exact derivatives at
            // v = 0 (x¹ → d1 = 1; x² → d2 = 2) instead of a blanket zero, and avoid the `inf`
            // that a negative `powi` exponent would produce.
            let d1 = if n == 1 { 1.0 } else { 0.0 };
            let d2 = if n == 2 { 2.0 } else { 0.0 };
            return self.chain(val, d1, d2);
        }
        let d1 = n as f64 * self.v.powi(n - 1);
        let d2 = n as f64 * (n - 1) as f64 * self.v.powi(n - 2);
        self.chain(val, d1, d2)
    }
    #[inline]
    fn powf(self, x: f64) -> Self {
        let val = self.v.powf(x);
        if self.v > 0.0 {
            let d1 = x * self.v.powf(x - 1.0);
            let d2 = x * (x - 1.0) * self.v.powf(x - 2.0);
            self.chain(val, d1, d2)
        } else {
            self.chain(val, 0.0, 0.0)
        }
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(c, -s, -c)
    }
    #[inline]
    fn acos(self) -> Self {
        let val = self.v.acos();
        let t = 1.0 - self.v * self.v;
        if t > 0.0 {
            let root = t.sqrt();
            let d1 = -1.0 / root;
            let d2 = -self.v / (t * root); // −x·(1−x²)^(−3/2)
            self.chain(val, d1, d2)
        } else {
            self.chain(val, 0.0, 0.0)
        }
    }
    #[inline]
    fn abs(self) -> Self {
        if self.v >= 0.0 {
            self
        } else {
            -self
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.v <= other.v {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.v >= other.v {
            self
        } else {
            other
        }
    }
    #[inline]
    fn clamp2(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Dual2N::constant(lo)
        } else if self.v > hi {
            Dual2N::constant(hi)
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual2n_second_derivatives_match_fd() {
        // f(x0..x5) = acos(clamp(x0·x3 + x1·x4 + x2·x5, -1, 1)) · exp(-(x0²+x1²)).sqrt-ish
        let f = |p: &[f64; 6]| {
            let d = (p[0] * p[3] + p[1] * p[4] + p[2] * p[5]).clamp(-0.9, 0.9);
            d.acos() * (-(p[0] * p[0] + p[1] * p[1])).exp() + (p[2] * p[2] + 1.0).sqrt()
        };
        let p = [0.3, -0.4, 0.5, 0.2, 0.6, -0.1];
        let mut xs = [Dual2N::<6>::constant(0.0); 6];
        for i in 0..6 {
            xs[i] = Dual2N::<6>::var(p[i], i);
        }
        let d = (xs[0] * xs[3] + xs[1] * xs[4] + xs[2] * xs[5]).clamp2(-0.9, 0.9);
        let val =
            d.acos() * (-(xs[0] * xs[0] + xs[1] * xs[1])).exp() + (xs[2] * xs[2] + 1.0).sqrt();
        assert!((val.v - f(&p)).abs() < 1e-12);
        let step = 1e-4;
        let mut maxd = 0.0f64;
        for a in 0..6 {
            for b in 0..6 {
                let mut pp = p;
                let mut pm = p;
                pp[a] += step;
                pm[a] -= step;
                let da = |q: &[f64; 6]| {
                    let mut qp = *q;
                    let mut qm = *q;
                    qp[b] += step;
                    qm[b] -= step;
                    (f(&qp) - f(&qm)) / (2.0 * step)
                };
                let fd = (da(&pp) - da(&pm)) / (2.0 * step);
                maxd = maxd.max((val.h[a][b] - fd).abs());
            }
        }
        assert!(maxd < 1e-5, "Dual2N Hessian vs FD {maxd:.2e}");
    }
}
