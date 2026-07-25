// SPDX-License-Identifier: GPL-3.0-or-later

//! Second-order forward-mode automatic differentiation scalar: value + gradient (3) +
//! Hessian (3×3) with respect to a 3-vector (an interatomic displacement).
//!
//! [`Dual2`] implements the same [`crate::dual::Scalar`] trait as the first-order
//! [`crate::dual::Dual`], so the generic integral kernels ([`crate::integrals`],
//! [`crate::overlap`], [`crate::repulsion`]) can be instantiated at `Dual2` to obtain the
//! **exact closed-form second derivatives** of the same expressions — the basis of the fully
//! analytic (finite-difference-free) skeleton Hessian. The Hessian array is kept full 3×3
//! (symmetric); the redundant upper/lower entries carry the same value.

use crate::dual::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Value, gradient, and Hessian of a scalar quantity w.r.t. a 3-vector variable.
#[derive(Clone, Copy, Debug)]
pub struct Dual2 {
    pub v: f64,
    pub g: [f64; 3],
    pub h: [[f64; 3]; 3],
}

impl Dual2 {
    #[inline]
    pub fn constant(x: f64) -> Self {
        Self {
            v: x,
            g: [0.0; 3],
            h: [[0.0; 3]; 3],
        }
    }
    /// Independent variable seeded along axis `i` (∂/∂x_i = 1, all second derivatives 0).
    #[inline]
    pub fn var(x: f64, i: usize) -> Self {
        let mut g = [0.0; 3];
        g[i] = 1.0;
        Self {
            v: x,
            g,
            h: [[0.0; 3]; 3],
        }
    }
    /// Chain rule for a smooth unary function φ evaluated at `self.v`, given
    /// `val = φ(v)`, `d1 = φ'(v)`, `d2 = φ''(v)`:
    /// `∂φ/∂i = d1·u_i`, `∂²φ/∂i∂j = d2·u_i·u_j + d1·u_ij`.
    #[inline]
    fn chain(self, val: f64, d1: f64, d2: f64) -> Self {
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                h[a][b] = d2 * self.g[a] * self.g[b] + d1 * self.h[a][b];
            }
        }
        Self {
            v: val,
            g: [d1 * self.g[0], d1 * self.g[1], d1 * self.g[2]],
            h,
        }
    }
}

impl Add for Dual2 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            g[a] = self.g[a] + o.g[a];
            for b in 0..3 {
                h[a][b] = self.h[a][b] + o.h[a][b];
            }
        }
        Self {
            v: self.v + o.v,
            g,
            h,
        }
    }
}
impl Sub for Dual2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            g[a] = self.g[a] - o.g[a];
            for b in 0..3 {
                h[a][b] = self.h[a][b] - o.h[a][b];
            }
        }
        Self {
            v: self.v - o.v,
            g,
            h,
        }
    }
}
impl Mul for Dual2 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        // Product rule: (fg)_i = f_i g + f g_i; (fg)_ij = f_ij g + f_i g_j + f_j g_i + f g_ij.
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            g[a] = self.g[a] * o.v + self.v * o.g[a];
        }
        for a in 0..3 {
            for b in 0..3 {
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
impl Div for Dual2 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        self * o.recip()
    }
}
impl Neg for Dual2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            g[a] = -self.g[a];
            for b in 0..3 {
                h[a][b] = -self.h[a][b];
            }
        }
        Self { v: -self.v, g, h }
    }
}

// Mixed operations with an f64 constant (derivatives of the constant are zero).
impl Add<f64> for Dual2 {
    type Output = Self;
    #[inline]
    fn add(self, o: f64) -> Self {
        Self {
            v: self.v + o,
            g: self.g,
            h: self.h,
        }
    }
}
impl Sub<f64> for Dual2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: f64) -> Self {
        Self {
            v: self.v - o,
            g: self.g,
            h: self.h,
        }
    }
}
impl Mul<f64> for Dual2 {
    type Output = Self;
    #[inline]
    fn mul(self, o: f64) -> Self {
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for a in 0..3 {
            g[a] = self.g[a] * o;
            for b in 0..3 {
                h[a][b] = self.h[a][b] * o;
            }
        }
        Self {
            v: self.v * o,
            g,
            h,
        }
    }
}
impl Div<f64> for Dual2 {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl Scalar for Dual2 {
    #[inline]
    fn cst(x: f64) -> Self {
        Dual2::constant(x)
    }
    #[inline]
    fn val(&self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        if s > 0.0 {
            let d1 = 0.5 / s;
            let d2 = -0.25 / (self.v * s); // -1/4 · v^(-3/2)
            self.chain(s, d1, d2)
        } else {
            self.chain(s, 0.0, 0.0)
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e, e)
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.v;
        let r2 = r * r;
        self.chain(r, -r2, 2.0 * r2 * r)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        let val = self.v.powi(n);
        if self.v != 0.0 {
            let d1 = n as f64 * self.v.powi(n - 1);
            let d2 = n as f64 * (n - 1) as f64 * self.v.powi(n - 2);
            self.chain(val, d1, d2)
        } else {
            self.chain(val, 0.0, 0.0)
        }
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

    /// Second derivative of a nontrivial composite against central finite differences.
    #[test]
    fn dual2_matches_fd_second_derivatives() {
        // f(x,y,z) = exp(-r) / sqrt(r^2 + 4),  r = |(x,y,z)|.
        let f = |p: [f64; 3]| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            (-r).exp() / (r * r + 4.0).sqrt()
        };
        let fd2 = |p: [f64; 3], a: usize, b: usize| {
            let h = 1e-4;
            let mut pp = p;
            let mut pm = p;
            pp[a] += h;
            pm[a] -= h;
            // central difference of the a-derivative w.r.t. b
            let da = |q: [f64; 3]| {
                let mut qp = q;
                let mut qm = q;
                qp[b] += h;
                qm[b] -= h;
                (f(qp) - f(qm)) / (2.0 * h)
            };
            (da(pp) - da(pm)) / (2.0 * h)
        };

        let p0 = [0.7, -1.1, 0.9];
        let x = Dual2::var(p0[0], 0);
        let y = Dual2::var(p0[1], 1);
        let z = Dual2::var(p0[2], 2);
        let r = (x * x + y * y + z * z).sqrt();
        let val = (-r).exp() / (r * r + 4.0).sqrt();

        assert!((val.v - f(p0)).abs() < 1e-12);
        let mut max_delta = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                max_delta = max_delta.max((val.h[a][b] - fd2(p0, a, b)).abs());
            }
        }
        eprintln!("Dual2 second-derivative max delta vs FD = {max_delta:.2e}");
        assert!(max_delta < 1e-5, "Dual2 Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn dual2_hessian_is_symmetric() {
        let x = Dual2::var(0.4, 0);
        let y = Dual2::var(-0.6, 1);
        let f = (x * y + x * x).exp() / (y * y + 1.0);
        for a in 0..3 {
            for b in 0..3 {
                assert!((f.h[a][b] - f.h[b][a]).abs() < 1e-12);
            }
        }
    }
}
