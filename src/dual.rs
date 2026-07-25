// SPDX-License-Identifier: GPL-3.0-or-later

//! A `Scalar` abstraction over `f64` and a forward-mode dual number carrying three spatial
//! partial derivatives. The integral kernels ([`crate::integrals`], [`crate::overlap`],
//! [`crate::repulsion`]) are written generically over `Scalar`, so instantiating them at
//! `f64` gives the (validated) energy path and at [`Dual`] gives **exact closed-form
//! derivatives** of the same expressions with respect to an interatomic displacement — the
//! basis of the fully analytic gradient (no finite differences).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Numeric type usable by the integral kernels: `f64` (values) or [`Dual`] (value + gradient).
pub trait Scalar:
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
    fn exp(self) -> Self;
    fn recip(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, x: f64) -> Self;
    fn abs(self) -> Self;
}

impl Scalar for f64 {
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
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
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
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

/// Forward-mode dual number: value plus three partial derivatives (∂/∂x, ∂/∂y, ∂/∂z).
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub v: f64,
    pub d: [f64; 3],
}

impl Dual {
    #[inline]
    pub fn constant(x: f64) -> Self {
        Self { v: x, d: [0.0; 3] }
    }
    /// A variable whose derivative is the unit vector along axis `i`.
    #[inline]
    pub fn var(x: f64, i: usize) -> Self {
        let mut d = [0.0; 3];
        d[i] = 1.0;
        Self { v: x, d }
    }
    #[inline]
    pub(crate) fn map(self, v: f64, factor: f64) -> Self {
        // chain rule: new value `v`, derivative = factor · self.d
        Self {
            v,
            d: [self.d[0] * factor, self.d[1] * factor, self.d[2] * factor],
        }
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            d: [self.d[0] + o.d[0], self.d[1] + o.d[1], self.d[2] + o.d[2]],
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            v: self.v - o.v,
            d: [self.d[0] - o.d[0], self.d[1] - o.d[1], self.d[2] - o.d[2]],
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: [
                self.d[0] * o.v + self.v * o.d[0],
                self.d[1] * o.v + self.v * o.d[1],
                self.d[2] * o.v + self.v * o.d[2],
            ],
        }
    }
}
impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let inv2 = inv * inv;
        Self {
            v: self.v * inv,
            d: [
                (self.d[0] * o.v - self.v * o.d[0]) * inv2,
                (self.d[1] * o.v - self.v * o.d[1]) * inv2,
                (self.d[2] * o.v - self.v * o.d[2]) * inv2,
            ],
        }
    }
}
impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: [-self.d[0], -self.d[1], -self.d[2]],
        }
    }
}
// Mixed ops with f64 (constant), for readable kernel code.
impl Add<f64> for Dual {
    type Output = Self;
    #[inline]
    fn add(self, o: f64) -> Self {
        Self {
            v: self.v + o,
            d: self.d,
        }
    }
}
impl Sub<f64> for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, o: f64) -> Self {
        Self {
            v: self.v - o,
            d: self.d,
        }
    }
}
impl Mul<f64> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, o: f64) -> Self {
        Self {
            v: self.v * o,
            d: [self.d[0] * o, self.d[1] * o, self.d[2] * o],
        }
    }
}
impl Div<f64> for Dual {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl Scalar for Dual {
    #[inline]
    fn cst(x: f64) -> Self {
        Dual::constant(x)
    }
    #[inline]
    fn val(&self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.map(s, if s > 0.0 { 0.5 / s } else { 0.0 })
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.v;
        self.map(r, -r * r)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        let p = self.v.powi(n);
        let dfac = if self.v != 0.0 {
            n as f64 * self.v.powi(n - 1)
        } else {
            0.0
        };
        self.map(p, dfac)
    }
    #[inline]
    fn powf(self, x: f64) -> Self {
        let p = self.v.powf(x);
        let dfac = if self.v > 0.0 {
            x * self.v.powf(x - 1.0)
        } else {
            0.0
        };
        self.map(p, dfac)
    }
    #[inline]
    fn abs(self) -> Self {
        if self.v >= 0.0 {
            self
        } else {
            -self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_matches_analytic_derivative() {
        // f(x) = exp(-x^2) / sqrt(x^2 + 4);  check f'(x) at x=1.3.
        let x0 = 1.3;
        let x = Dual::var(x0, 0);
        let f = (-(x * x)).exp() / (x * x + 4.0).sqrt();
        // analytic derivative
        let g = |x: f64| (-(x * x)).exp() / (x * x + 4.0).sqrt();
        let h = 1e-6;
        let fd = (g(x0 + h) - g(x0 - h)) / (2.0 * h);
        assert!((f.d[0] - fd).abs() < 1e-8, "{} vs {}", f.d[0], fd);
    }
}
