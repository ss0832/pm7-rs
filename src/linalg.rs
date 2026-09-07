// SPDX-License-Identifier: GPL-3.0-or-later

//! Dense linear algebra: a row-major `Matrix` wrapper plus the solvers the SCF needs.
//!
//! Parallels `gfn1-rs`'s `linalg.rs`. The heavy O(n³) work — the symmetric eigendecomposition
//! and the LU solve — is delegated to **faer** (pure Rust; no LAPACK/BLAS). Only a tiny
//! pivot-guarded Gaussian elimination for the DIIS coefficient system lives in `scf.rs`,
//! where faer's non-erroring behaviour on the near-singular DIIS matrix is undesirable.

use crate::error::{Pm7Error, Result};

/// Row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "matrix data length mismatch");
        Self { rows, cols, data }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Standard `self · other`.
    ///
    /// Cache-friendly `ikj` accumulation, parallelized over output rows with rayon for large
    /// operands. Each output element accumulates in the same `k` order as the serial version,
    /// so the result is **bit-identical** regardless of thread count (no floating-point reorder).
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        let mut out = Matrix::zeros(self.rows, other.cols);
        if out.data.is_empty() {
            return out;
        }
        let (inner, ncol) = (self.cols, other.cols);
        let (a, b) = (&self.data, &other.data);
        let row = |i: usize, out_row: &mut [f64]| {
            let a_row = &a[i * inner..i * inner + inner];
            for (k, &aik) in a_row.iter().enumerate() {
                if aik == 0.0 {
                    continue;
                }
                let b_row = &b[k * ncol..k * ncol + ncol];
                for (out_j, &bkj) in out_row.iter_mut().zip(b_row) {
                    *out_j += aik * bkj;
                }
            }
        };
        // Parallelize only when there is enough work to amortize rayon's overhead.
        if self.rows.saturating_mul(ncol).saturating_mul(inner) >= 1 << 15 {
            use rayon::prelude::*;
            out.data
                .par_chunks_mut(ncol)
                .enumerate()
                .for_each(|(i, out_row)| row(i, out_row));
        } else {
            for (i, out_row) in out.data.chunks_mut(ncol).enumerate() {
                row(i, out_row);
            }
        }
        out
    }

    /// Always-sequential `self · other` (cache-friendly `ikj`). Use inside an outer rayon loop
    /// (e.g. the per-DOF CPHF solves) where the surrounding parallelism already saturates the
    /// cores, so an inner parallel matmul would only add work-stealing overhead. Bit-identical to
    /// [`Matrix::matmul`].
    pub fn matmul_seq(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        let mut out = Matrix::zeros(self.rows, other.cols);
        if out.data.is_empty() {
            return out;
        }
        let (inner, ncol) = (self.cols, other.cols);
        let (a, b) = (&self.data, &other.data);
        for (i, out_row) in out.data.chunks_mut(ncol).enumerate() {
            let a_row = &a[i * inner..i * inner + inner];
            for (k, &aik) in a_row.iter().enumerate() {
                if aik == 0.0 {
                    continue;
                }
                let b_row = &b[k * ncol..k * ncol + ncol];
                for (out_j, &bkj) in out_row.iter_mut().zip(b_row) {
                    *out_j += aik * bkj;
                }
            }
        }
        out
    }

    /// Frobenius inner product `Σ_ij A_ij B_ij` (used for energies `½Σ P(H+F)`).
    /// Overwrite this matrix's contents from another of the same shape, **without reallocating**.
    ///
    /// For reusing a buffer that is about to be overwritten anyway — an SCF history slot, say —
    /// instead of dropping its allocation and taking a fresh one for the clone that replaces it.
    pub fn copy_from(&mut self, other: &Matrix) {
        debug_assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        self.data.copy_from_slice(&other.data);
    }

    pub fn frobenius_dot(&self, other: &Matrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
}

/// How a factor enters a product: as itself, or transposed.
///
/// Transposing through a *view* rather than a copy is most of the point. `Cvᵀ F Co` and
/// `Cv U Coᵀ` are the two products the CPHF runs thousands of times, and building the transpose
/// with [`Matrix::transpose`] each time allocates and fills an `nao × n_occ` matrix for nothing —
/// faer reads a transposed view at zero cost by swapping the strides.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Normal,
    Transposed,
}

impl Matrix {
    fn faer_view(&self) -> faer::MatRef<'_, f64> {
        faer::MatRef::from_row_major_slice(&self.data, self.rows, self.cols)
    }

    /// `op(a) · op(b)`, through faer's blocked GEMM.
    ///
    /// The hand-written `ikj` loop this replaces is cache-friendly but scalar: it does not block
    /// for L2, does not use FMA lanes, and cannot reuse a packed panel across output columns. On
    /// the 102-atom analytic Hessian the two CPHF products were **48 % of the whole run** — more
    /// than the Fock builds — which is not a ratio any amount of loop tidying was going to fix.
    ///
    /// The summation order differs from the `ikj` loop, so results move in the last ulp or two.
    /// It is still deterministic for a given shape (faer's blocking depends on the dimensions, not
    /// on thread scheduling), so runs remain reproducible; only the comparison against the *old*
    /// code changes. That is the right trade here: this path feeds an iterative solver converged
    /// to `1e-9` whose output is checked against finite differences.
    pub fn gemm(&self, a_side: Side, b: &Matrix, b_side: Side, parallel: bool) -> Matrix {
        let a_view = match a_side {
            Side::Normal => self.faer_view(),
            Side::Transposed => self.faer_view().transpose(),
        };
        let b_view = match b_side {
            Side::Normal => b.faer_view(),
            Side::Transposed => b.faer_view().transpose(),
        };
        assert_eq!(
            a_view.ncols(),
            b_view.nrows(),
            "gemm dimension mismatch: {}x{} times {}x{}",
            a_view.nrows(),
            a_view.ncols(),
            b_view.nrows(),
            b_view.ncols()
        );
        let mut out = Matrix::zeros(a_view.nrows(), b_view.ncols());
        if out.data.is_empty() {
            return out;
        }
        let (rows, cols) = (out.rows, out.cols);
        let dst = faer::MatMut::from_row_major_slice_mut(&mut out.data, rows, cols);
        faer::linalg::matmul::matmul(
            dst,
            faer::Accum::Replace,
            a_view,
            b_view,
            1.0_f64,
            if parallel {
                faer::Par::rayon(0)
            } else {
                faer::Par::Seq
            },
        );
        out
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

/// Symmetric eigendecomposition (faer, pure-Rust — no LAPACK/BLAS).
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues in **ascending** order and
/// the eigenvectors as the **columns** of the returned matrix, so `A = V diag(λ) Vᵀ`.
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    let n = a.rows;
    if a.cols != n {
        return Err(Pm7Error::LinearAlgebra(
            "symmetric_eigen requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Matrix::zeros(0, 0)));
    }
    // A borrowed row-major view, not a copy. `Mat::from_fn` walked this matrix column-major over
    // a row-major buffer — an `N²` strided read on every diagonalization, and there is one of
    // those per SCF iteration, per k point, and per divide-and-conquer subsystem.
    let eigen = a
        .faer_view()
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| Pm7Error::LinearAlgebra(format!("faer eigendecomposition failed: {e:?}")))?;
    let s = eigen.S();
    let u = eigen.U();

    // The SCF aufbau occupies the lowest orbitals and faer's ordering is not guaranteed ascending,
    // so it has to be enforced. In practice it comes back ascending already, and checking for that
    // is `O(N)` against an `O(N²)` strided permutation copy — so the common case now writes the
    // eigenvectors out row by row, contiguously, instead of column by column.
    let ascending = (1..n).all(|i| s[i - 1] <= s[i]);
    if ascending {
        let values: Vec<f64> = (0..n).map(|k| s[k]).collect();
        let mut vectors = Matrix::zeros(n, n);
        for i in 0..n {
            let row = &mut vectors.as_mut_slice()[i * n..(i + 1) * n];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = u[(i, j)];
            }
        }
        return Ok((values, vectors));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = order.iter().map(|&k| s[k]).collect();
    let mut vectors = Matrix::zeros(n, n);
    for (new_col, &old_col) in order.iter().enumerate() {
        for i in 0..n {
            vectors[(i, new_col)] = u[(i, old_col)];
        }
    }
    Ok((values, vectors))
}

/// Solve `A x = b` via faer's partial-pivot LU (pure-Rust — no LAPACK/BLAS).
pub fn solve_linear(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(Pm7Error::LinearAlgebra(
            "solve_linear dimension mismatch".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    use faer::linalg::solvers::Solve;
    let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a[(i, j)]);
    let rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let lu = fa.partial_piv_lu();
    let x = lu.solve(&rhs);
    let sol: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
    // faer's LU does not error on a singular/near-singular system (it returns a huge,
    // low-residual solution). The old Gaussian-elimination path returned `Err` on a tiny
    // pivot so DIIS callers using `.ok()` fell back. Reproduce that: reject non-finite or
    // unreasonably large solutions (for a well-posed DIIS system the coefficients are O(1)).
    let bmax = b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    if sol.iter().any(|v| !v.is_finite()) || sol.iter().any(|v| v.abs() > 1.0e8 * bmax) {
        return Err(Pm7Error::LinearAlgebra(
            "singular or ill-conditioned linear solve".to_string(),
        ));
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes_known_matrix() {
        // [[2,1],[1,2]] -> eigenvalues 1, 3.
        let a = Matrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        // Reconstruct A = V Λ Vᵀ.
        let mut lam = Matrix::zeros(2, 2);
        lam[(0, 0)] = vals[0];
        lam[(1, 1)] = vals[1];
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        for i in 0..2 {
            for j in 0..2 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn eigen_reconstructs_asymmetric_eigenvectors() {
        // Distinct eigenvalues -> eigenvector matrix is NOT symmetric; catches transpose bugs.
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 1.0]);
        let (vals, vecs) = symmetric_eigen(&a).unwrap();
        // ascending
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        let mut lam = Matrix::zeros(3, 3);
        for i in 0..3 {
            lam[(i, i)] = vals[i];
        }
        let recon = vecs.matmul(&lam).matmul(&vecs.transpose());
        for i in 0..3 {
            for j in 0..3 {
                assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-9, "recon[{i}][{j}]");
            }
        }
        // Columns orthonormal.
        for j in 0..3 {
            let nj: f64 = (0..3).map(|i| vecs[(i, j)] * vecs[(i, j)]).sum();
            assert!((nj - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn solve_linear_basic() {
        let a = Matrix::from_row_major(2, 2, vec![3.0, 2.0, 1.0, 2.0]);
        let x = solve_linear(&a, &[7.0, 5.0]).unwrap();
        // 3x+2y=7, x+2y=5 -> x=1, y=2.
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn solve_diis_saddle_point() {
        // DIIS-like bordered system: B = [[1,0.5,-1],[0.5,1,-1],[-1,-1,0]], rhs=[0,0,-1].
        // Known solution: c0 = c1 = 0.5, lambda = 0.75.
        let a = Matrix::from_row_major(3, 3, vec![1.0, 0.5, -1.0, 0.5, 1.0, -1.0, -1.0, -1.0, 0.0]);
        let x = solve_linear(&a, &[0.0, 0.0, -1.0]).unwrap();
        eprintln!("DIIS solve: {x:?}");
        assert!((x[0] - 0.5).abs() < 1e-9, "c0={}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-9, "c1={}", x[1]);
    }
}
