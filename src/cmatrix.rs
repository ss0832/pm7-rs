// SPDX-License-Identifier: GPL-3.0-or-later

//! Complex Hermitian matrices for the k-resolved SCF.
//!
//! Only what the Bloch problem needs: assemble `H(k) = Σ_T H(T) e^{ik·T}` from real
//! translation blocks, diagonalize it, and build a density from the occupied orbitals. Real and
//! imaginary parts are stored in **separate** contiguous arrays rather than interleaved, because
//! the assembly is `re += cos·block` / `im += sin·block` over real blocks — two independent
//! real AXPYs that vectorize, with no complex arithmetic at all.

use crate::error::{Pm7Error, Result};
use crate::linalg::{Matrix, Side};

/// A square complex matrix, stored as separate row-major real and imaginary parts.
#[derive(Clone, Debug)]
pub struct CMatrix {
    pub n: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl CMatrix {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            re: vec![0.0; n * n],
            im: vec![0.0; n * n],
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> (f64, f64) {
        let k = i * self.n + j;
        (self.re[k], self.im[k])
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, re: f64, im: f64) {
        let k = i * self.n + j;
        self.re[k] = re;
        self.im[k] = im;
    }

    /// Accumulate `cos·block + i·sin·block`, the contribution of one lattice translation to
    /// `H(k) = Σ_T H(T) e^{ik·T}`.
    pub fn add_phase(&mut self, block: &Matrix, cos: f64, sin: f64) {
        debug_assert_eq!(block.rows, self.n);
        debug_assert_eq!(block.cols, self.n);
        let b = block.as_slice();
        if cos != 0.0 {
            for (dst, src) in self.re.iter_mut().zip(b) {
                *dst += cos * src;
            }
        }
        if sin != 0.0 {
            for (dst, src) in self.im.iter_mut().zip(b) {
                *dst += sin * src;
            }
        }
    }

    /// `self · other`, through four real blocked GEMMs on the split storage.
    ///
    /// `(A_r + iA_i)(B_r + iB_i) = (A_r B_r − A_i B_i) + i(A_r B_i + A_i B_r)`. The split real/
    /// imaginary layout this type already uses is what makes that four `faer` calls rather than a
    /// complex kernel — and `faer`'s blocked GEMM is what the hand-written triple loop it replaces
    /// was not: it blocks for cache, uses FMA lanes, and reuses a packed panel across output
    /// columns.
    ///
    /// This runs once per k point, per DIIS iteration, per degree of freedom in the perturbation
    /// solver, so the scalar version was the dominant arithmetic there.
    pub fn matmul(&self, other: &Self) -> Self {
        let n = self.n;
        let a_re = Matrix::from_row_major(n, n, self.re.clone());
        let a_im = Matrix::from_row_major(n, n, self.im.clone());
        let b_re = Matrix::from_row_major(n, n, other.re.clone());
        let b_im = Matrix::from_row_major(n, n, other.im.clone());
        let rr = a_re.gemm(Side::Normal, &b_re, Side::Normal, false);
        let ii = a_im.gemm(Side::Normal, &b_im, Side::Normal, false);
        let ri = a_re.gemm(Side::Normal, &b_im, Side::Normal, false);
        let ir = a_im.gemm(Side::Normal, &b_re, Side::Normal, false);
        let mut out = Self::zeros(n);
        for (slot, (x, y)) in out
            .re
            .iter_mut()
            .zip(rr.as_slice().iter().zip(ii.as_slice()))
        {
            *slot = x - y;
        }
        for (slot, (x, y)) in out
            .im
            .iter_mut()
            .zip(ri.as_slice().iter().zip(ir.as_slice()))
        {
            *slot = x + y;
        }
        out
    }

    /// The conjugate transpose.
    pub fn adjoint(&self) -> Self {
        let n = self.n;
        let mut out = Self::zeros(n);
        for i in 0..n {
            for j in 0..n {
                out.re[i * n + j] = self.re[j * n + i];
                out.im[i * n + j] = -self.im[j * n + i];
            }
        }
        out
    }

    /// Largest deviation from Hermiticity, `max |M − M†|`. A Bloch matrix built from blocks
    /// satisfying `H(−T) = H(T)ᵀ` is Hermitian by construction, so this is a cheap invariant
    /// check rather than a repair.
    pub fn hermiticity_error(&self) -> f64 {
        let n = self.n;
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let (ar, ai) = self.get(i, j);
                let (br, bi) = self.get(j, i);
                worst = worst.max((ar - br).abs()).max((ai + bi).abs());
            }
        }
        worst
    }

    /// Eigenvalues (ascending) and eigenvectors (columns) of a Hermitian matrix.
    ///
    /// Only the lower triangle is read, so a matrix that is Hermitian up to rounding gives the
    /// same answer as one that is exactly Hermitian.
    pub fn hermitian_eigen(&self) -> Result<(Vec<f64>, CMatrix)> {
        let n = self.n;
        if n == 0 {
            return Ok((Vec::new(), CMatrix::zeros(0)));
        }
        let fa = faer::Mat::<faer::c64>::from_fn(n, n, |i, j| {
            let (re, im) = self.get(i, j);
            faer::c64::new(re, im)
        });
        let eigen = fa
            .self_adjoint_eigen(faer::Side::Lower)
            .map_err(|e| Pm7Error::LinearAlgebra(format!("faer Hermitian eigen failed: {e:?}")))?;
        let s = eigen.S();
        let u = eigen.U();
        // faer's ordering is not guaranteed ascending, and the aufbau filling depends on it.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            s[i].re
                .partial_cmp(&s[j].re)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let values: Vec<f64> = order.iter().map(|&k| s[k].re).collect();
        let mut vectors = CMatrix::zeros(n);
        for (new_col, &old_col) in order.iter().enumerate() {
            for i in 0..n {
                let v = u[(i, old_col)];
                vectors.set(i, new_col, v.re, v.im);
            }
        }
        Ok((values, vectors))
    }

    /// `P(k) = weight · Σ_{n} f_n c_n c_n†` for the given occupations, one column per orbital.
    ///
    /// The result is Hermitian; only its real part times `cos(k·T)` plus its imaginary part
    /// times `sin(k·T)` ever reaches the real-space density, so both parts are kept.
    pub fn weighted_density(&self, occupations: &[f64], weight: f64) -> CMatrix {
        let n = self.n;
        let mut p = CMatrix::zeros(n);
        for (band, &f) in occupations.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            let w = weight * f;
            for mu in 0..n {
                let (cr, ci) = self.get(mu, band);
                for nu in 0..n {
                    let (dr, di) = self.get(nu, band);
                    // c_mu · conj(c_nu)
                    let k = mu * n + nu;
                    p.re[k] += w * (cr * dr + ci * di);
                    p.im[k] += w * (ci * dr - cr * di);
                }
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hermitian_sample(n: usize) -> CMatrix {
        let mut m = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                // A deterministic Hermitian pattern.
                let re = ((i * 7 + j * 13) % 11) as f64 - 5.0;
                let im = if i == j {
                    0.0
                } else {
                    (((i * 3 + j * 5) % 7) as f64 - 3.0) * if i < j { 1.0 } else { -1.0 }
                };
                let (re, im) = if i <= j {
                    (re, im)
                } else {
                    (m.get(j, i).0, -m.get(j, i).1)
                };
                m.set(i, j, re, im);
            }
        }
        m
    }

    #[test]
    fn eigen_reconstructs_the_matrix() {
        for n in [1usize, 2, 5, 9] {
            let m = hermitian_sample(n);
            assert!(m.hermiticity_error() < 1e-14, "test matrix not Hermitian");
            let (vals, vecs) = m.hermitian_eigen().unwrap();
            assert_eq!(vals.len(), n);
            assert!(
                vals.windows(2).all(|w| w[0] <= w[1] + 1e-12),
                "not ascending"
            );
            // U diag(λ) U† must reproduce M.
            for i in 0..n {
                for j in 0..n {
                    let (mut re, mut im) = (0.0, 0.0);
                    for (k, &lam) in vals.iter().enumerate() {
                        let (ar, ai) = vecs.get(i, k);
                        let (br, bi) = vecs.get(j, k);
                        // λ · u_ik · conj(u_jk)
                        re += lam * (ar * br + ai * bi);
                        im += lam * (ai * br - ar * bi);
                    }
                    let (er, ei) = m.get(i, j);
                    assert!(
                        (re - er).abs() < 1e-10 && (im - ei).abs() < 1e-10,
                        "n={n} ({i},{j}): got ({re}, {im}) want ({er}, {ei})"
                    );
                }
            }
        }
    }

    #[test]
    fn a_real_matrix_gives_the_real_eigenvalues() {
        // With no imaginary part the Hermitian solver must agree with the real symmetric one,
        // which is what makes the Γ point of the k-point path match the real Γ code.
        let n = 6;
        let mut real = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = ((i * 5 + j * 3) % 9) as f64 - 4.0;
                real[(i, j)] = v;
                real[(j, i)] = v;
            }
        }
        let mut c = CMatrix::zeros(n);
        c.add_phase(&real, 1.0, 0.0);
        let (cv, _) = c.hermitian_eigen().unwrap();
        let (rv, _) = crate::linalg::symmetric_eigen(&real).unwrap();
        for (a, b) in cv.iter().zip(&rv) {
            assert!((a - b).abs() < 1e-10, "complex {a} vs real {b}");
        }
    }

    #[test]
    fn add_phase_builds_the_bloch_sum() {
        let n = 3;
        let mut block = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                block[(i, j)] = (i + 2 * j) as f64;
            }
        }
        let mut c = CMatrix::zeros(n);
        c.add_phase(&block, 0.5, -0.25);
        c.add_phase(&block, 0.25, 0.75);
        for i in 0..n {
            for j in 0..n {
                let (re, im) = c.get(i, j);
                assert!((re - 0.75 * block[(i, j)]).abs() < 1e-14);
                assert!((im - 0.5 * block[(i, j)]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn weighted_density_is_hermitian_and_traces_to_the_electron_count() {
        let n = 5;
        let m = hermitian_sample(n);
        let (_, vecs) = m.hermitian_eigen().unwrap();
        let occ = vec![1.0, 1.0, 0.5, 0.0, 0.0];
        let p = vecs.weighted_density(&occ, 2.0);
        assert!(p.hermiticity_error() < 1e-12, "density is not Hermitian");
        let trace: f64 = (0..n).map(|i| p.get(i, i).0).sum();
        let expect = 2.0 * occ.iter().sum::<f64>();
        assert!(
            (trace - expect).abs() < 1e-10,
            "trace {trace} vs expected electron count {expect}"
        );
        // The diagonal of a density must be real.
        for i in 0..n {
            assert!(p.get(i, i).1.abs() < 1e-12);
        }
    }
}
