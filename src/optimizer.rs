// SPDX-License-Identifier: GPL-3.0-or-later

//! L-BFGS geometry optimization on the PM7 energy surface (Rust-native), driven by the
//! Hellmann-Feynman nuclear gradient from [`crate::gradient`]. Positions are Bohr
//! internally; gradients are eV/Bohr. Mirrors gfn1-rs's `optimizer.rs` structure.

use crate::error::Result;
use crate::gradient::closed_form_gradient;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::scf::{run_pm7, Pm7Options, Pm7Result};
use crate::system::Molecule;

#[derive(Clone, Debug)]
pub struct OptOptions {
    pub max_iter: usize,
    /// Convergence on the max gradient component (eV/Bohr).
    pub gtol: f64,
    /// Finite-difference step for the gradient (Bohr).
    pub grad_step: f64,
    /// L-BFGS history length.
    pub history: usize,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1.0e-3,
            grad_step: 5.0e-4,
            history: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptStep {
    pub energy_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub max_gradient: f64,
    pub positions: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct OptResult {
    pub molecule: Molecule,
    pub scf: Pm7Result,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<OptStep>,
}

pub fn optimize(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf_options: &Pm7Options,
    opt: &OptOptions,
) -> Result<OptResult> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut mol = molecule.clone();

    let mut x = flatten(&mol);
    let grad0 = closed_form_gradient(&mol, params, scf_options)?;
    let mut g = flatten_grad(&grad0.gradient);
    let mut energy = grad0.energy_ev;
    let mut scf = grad0.scf;
    let mut max_grad = grad0.max_gradient;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut trajectory = vec![OptStep {
        energy_ev: energy,
        heat_of_formation_kcal: scf.heat_of_formation_kcal,
        max_gradient: max_grad,
        positions: unflatten(&x),
    }];

    let mut converged = max_grad < opt.gtol;
    let mut iterations = 0;

    for iter in 0..opt.max_iter {
        iterations = iter + 1;
        if converged {
            break;
        }

        // L-BFGS two-loop recursion -> search direction d = -H*g.
        let mut q = g.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling.
        let gamma = if m > 0 {
            let sy = dot(&s_hist[m - 1], &y_hist[m - 1]);
            let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
            if yy > 0.0 {
                sy / yy
            } else {
                1.0
            }
        } else {
            // Cautious first step.
            0.1 / max_grad.max(1.0e-6)
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_hist[i]);
        }
        let mut d: Vec<f64> = q.iter().map(|v| -v).collect();
        // Guard against uphill directions.
        if dot(&d, &g) > 0.0 {
            d = g.iter().map(|v| -v).collect();
        }

        // Backtracking Armijo line search.
        let g_dot_d = dot(&g, &d);
        let mut step = 1.0;
        let c1 = 1.0e-4;
        let mut x_new;
        let mut ok = false;
        loop {
            x_new = x.clone();
            axpy(&mut x_new, step, &d);
            set_positions(&mut mol, &x_new);
            if let Ok(r) = run_pm7(&mol, params, scf_options) {
                if r.total_ev <= energy + c1 * step * g_dot_d {
                    ok = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-8 {
                break;
            }
        }
        if !ok {
            // Could not make progress; stop at the current point.
            set_positions(&mut mol, &x);
            break;
        }

        let grad_new = closed_form_gradient(&mol, params, scf_options)?;
        let g_new = flatten_grad(&grad_new.gradient);

        // Update L-BFGS memory.
        let s: Vec<f64> = (0..ndof).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..ndof).map(|i| g_new[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1.0e-10 {
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
            if s_hist.len() > opt.history {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
        }

        x = x_new;
        g = g_new;
        energy = grad_new.energy_ev;
        scf = grad_new.scf;
        max_grad = grad_new.max_gradient;
        converged = max_grad < opt.gtol;

        trajectory.push(OptStep {
            energy_ev: energy,
            heat_of_formation_kcal: scf.heat_of_formation_kcal,
            max_gradient: max_grad,
            positions: unflatten(&x),
        });
    }

    set_positions(&mut mol, &x);
    Ok(OptResult {
        molecule: mol,
        scf,
        converged,
        iterations,
        trajectory,
    })
}

fn flatten(mol: &Molecule) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * mol.atoms.len());
    for a in &mol.atoms {
        v.push(a.position.x);
        v.push(a.position.y);
        v.push(a.position.z);
    }
    v
}
fn flatten_grad(g: &[Vec3]) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * g.len());
    for gi in g {
        v.push(gi.x);
        v.push(gi.y);
        v.push(gi.z);
    }
    v
}
fn unflatten(x: &[f64]) -> Vec<Vec3> {
    x.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}
fn set_positions(mol: &mut Molecule, x: &[f64]) {
    for (i, a) in mol.atoms.iter_mut().enumerate() {
        a.position = Vec3::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizes_water() {
        // Start from a distorted water; expect relaxation toward the PM7 minimum
        // (dHf about -59.24 kcal/mol, r(OH) about 0.96 A, angle about 103.5 deg).
        let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let res = optimize(
            &mol,
            &params,
            &Pm7Options::default(),
            &OptOptions::default(),
        )
        .unwrap();
        eprintln!(
            "opt H2O: converged={} iters={} dHf={:.3} kcal/mol maxgrad={:.2e}",
            res.converged,
            res.iterations,
            res.scf.heat_of_formation_kcal,
            res.trajectory.last().unwrap().max_gradient
        );
        assert!(res.converged);
        assert!((res.scf.heat_of_formation_kcal - (-57.8)).abs() < 0.5);
        assert!(res.trajectory.last().unwrap().energy_ev < res.trajectory[0].energy_ev);
    }
}
