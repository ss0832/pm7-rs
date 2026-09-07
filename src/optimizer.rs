// SPDX-License-Identifier: GPL-3.0-or-later

//! L-BFGS geometry optimization on the PM7 energy surface (Rust-native), driven by the
//! Hellmann-Feynman nuclear gradient from [`crate::gradient`]. Positions are Bohr
//! internally; gradients are eV/Bohr. Mirrors gfn1-rs's `optimizer.rs` structure.

use crate::cell::Cell;
use crate::error::{Pm7Error, Result};
use crate::gradient::closed_form_gradient;
use crate::math::{Mat3, Vec3};
use crate::params::Pm7Parameters;
use crate::scf::{run_pm7, Pm7Options, Pm7Result};
use crate::system::Molecule;

#[derive(Clone, Debug)]
pub struct OptOptions {
    pub max_iter: usize,
    /// Convergence on the max gradient component (eV/Bohr).
    pub gtol: f64,
    /// L-BFGS history length.
    pub history: usize,
    /// Run an SCF stability analysis every `n` steps, and switch to following instabilities for the
    /// rest of the run once one is found. `0` (the default) never looks.
    ///
    /// **Because a geometry optimization can walk into an instability it did not start with.** The
    /// classic case is a bond stretching: H₂ is a perfectly good closed shell at 0.74 Å and is
    /// triplet-unstable by 1.5 Å, so an optimization that begins stable can end on a solution
    /// 100 kcal/mol above the right one, with every step converged and every gradient consistent.
    /// Checking periodically costs about one CPHF solve per check, which is why it is periodic
    /// rather than every step — and why finding one instability turns following on permanently
    /// instead of re-deciding each time.
    pub stability_every: usize,
    /// Relax the **lattice** as well as the atoms. Opt-in, so no published number moves.
    ///
    /// Off, the optimizer relaxes atomic coordinates in a fixed cell and can leave an arbitrarily
    /// large stress standing. Measured: a diamond cell at `a = 3.75 Å` on a 2×2×2 mesh reports
    /// `converged: true after 1 iterations` with a max gradient of `2e-14 eV/Bohr` — and −29.6 GPa
    /// of pressure. Both statements are true; the atoms genuinely are at their minimum *for that
    /// cell*, and the cell is 2 % too big. Turning this on relaxes it in four iterations to
    /// `a = 3.6751 Å`, 0.076 eV lower — and in six from a compressed `a = 3.40 Å` start, to the
    /// same lattice constant within 6e-6 relative.
    ///
    /// See [`CellDof`] for the parameterization.
    ///
    /// Requires a stress, so it is refused for a molecule, for `PbcMode::MopacCluster`, and under
    /// an external field — [`crate::stress::analytic_stress`]'s own refusals, surfaced here rather
    /// than silently becoming an atoms-only run.
    pub relax_cell: bool,
    /// Convergence on the largest free stress component, eV/Bohr^dim. Only read when
    /// [`Self::relax_cell`] is on.
    ///
    /// **Separate from `gtol` on purpose.** The two quantities have different units and different
    /// magnitudes, so a single mixed norm over the combined L-BFGS vector would let a converged
    /// force hide an unconverged stress — the run would stop, report success, and leave a cell
    /// under pressure.
    pub stress_tol: f64,
    /// Drive the optimization with the **divide-and-conquer** SCF instead of the exact cubic one.
    ///
    /// The energy and the gradient then come from [`crate::dandc::run_dandc`] and
    /// [`crate::dandc::dandc_derivatives`], which is the whole point: a geometry optimization is
    /// tens to hundreds of energy-and-gradient evaluations, so it is exactly the workload where
    /// linear scaling pays, and it was the one driver divide and conquer could not reach.
    ///
    /// The assembled density is not the exact SCF density, so the energy and gradient carry a
    /// residual that shrinks with the buffer (`docs/divide_and_conquer.md`). Below a 7 Å buffer the
    /// accuracy falls off a cliff, and below roughly 350 atoms the exact SCF is both faster *and*
    /// exact — so this is opt-in and stays opt-in.
    pub dandc: Option<crate::dandc::DandcOptions>,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1.0e-3,
            history: 8,
            relax_cell: false,
            // eV/Bohr^3 is about 3.6 GPa, so this is roughly 0.02 GPa: tight enough that a relaxed
            // cell is relaxed, loose enough not to chase SCF noise, since a stress is a derivative
            // of the density and inherits its convergence.
            stress_tol: 5.0e-6,
            dandc: None,
            stability_every: 0,
        }
    }
}

/// The lattice degrees of freedom, when `--opt-cell` is on.
///
/// # Strain, not lattice vectors
///
/// The variables are the six (or three, or one) components of a symmetric strain `ε`, with the
/// cell `h = (I + ε)h₀` and the atoms carried along affinely, `R_A = (I + ε)q_A`. Two things make
/// that the right parameterization rather than "optimize the nine numbers in `h`":
///
/// * **the stress is exactly conjugate to it.** `σ = (1/Ω)∂E/∂ε` is the definition of the stress
///   tensor, so `∂E/∂ε = Ωσ` needs no chain rule and no finite difference — `analytic_stress`
///   already computes it. Against the lattice vectors themselves the derivative would have to be
///   assembled, and would be a second implementation of the same quantity.
/// * **fractional coordinates stay fixed under a pure cell step.** Optimizing `h` with Cartesian
///   atomic coordinates held fixed would shear the atoms out of the cell; carrying them affinely
///   is what makes a cell step and an atomic step independent directions rather than two
///   descriptions of the same motion.
///
/// # Which components are free
///
/// A slab has no lattice vector along its normal, so straining that direction is not a degree of
/// freedom — it is a request to stretch vacuum. The free components are those inside the span of
/// the periodic lattice vectors, so the generators are built in an **orthonormal basis of that
/// span** (Gram–Schmidt on the lattice vectors), giving `dim(dim+1)/2` of them: 6 in 3-D, 3 in
/// 2-D, 1 in 1-D.
///
/// # Units
///
/// The nuclear gradient is eV/Bohr and `∂E/∂ε` is an energy, so putting both into one L-BFGS
/// vector needs a length. The strain variables are `s_k = L·ε_k` with `L = Ω^(1/dim)`, which makes
/// `∂E/∂s = Ωσ/L` an eV/Bohr in **every** dimension — `Ω/L` is Bohr² in 3-D against a σ in
/// eV/Bohr³, Bohr in 2-D against eV/Bohr², and dimensionless in 1-D against eV/Bohr. The two
/// convergence tests stay separate regardless, because a mixed norm lets a converged force hide an
/// unconverged stress.
struct CellDof {
    /// The cell at `ε = 0`.
    reference: Cell,
    /// Symmetric, trace-orthonormal generators of the admissible strains.
    generators: Vec<Mat3>,
    /// `Ω^(1/dim)` at the reference cell, in Bohr.
    length: f64,
}

impl CellDof {
    fn new(molecule: &Molecule) -> Result<Option<Self>> {
        let Some(cell) = molecule.cell else {
            return Ok(None);
        };
        let dim = cell.dim();
        // `measure`, not `volume`: the latter returns `None` for anything but a 3-D cell, on
        // purpose, so a caller who needs a volume cannot silently be handed an area. Here the
        // dimension-appropriate measure is exactly what is wanted — `σ` is eV/Bohr^dim and
        // `∂E/∂ε = Ωσ` holds with Ω a length in 1-D and an area in 2-D.
        let measure = cell.measure();
        if !(measure.is_finite() && measure > 0.0) {
            return Err(Pm7Error::InvalidInput(
                "a variable-cell optimization needs a cell with a non-zero measure".into(),
            ));
        }

        // Gram-Schmidt on the lattice vectors: an orthonormal basis of the periodic subspace.
        let mut basis: Vec<Vec3> = Vec::with_capacity(dim);
        for a in cell.vectors() {
            let mut v = *a;
            for u in &basis {
                v = v - *u * u.dot(v);
            }
            let norm = v.norm();
            if norm < 1.0e-10 {
                return Err(Pm7Error::InvalidInput(
                    "the lattice vectors are linearly dependent; a strain basis cannot be built"
                        .into(),
                ));
            }
            basis.push(v / norm);
        }

        // Symmetric generators, orthonormal under the Frobenius inner product so the L-BFGS metric
        // is the obvious one.
        let mut generators = Vec::new();
        for i in 0..dim {
            for j in i..dim {
                let (u, v) = (basis[i], basis[j]);
                let mut g = Mat3::zero();
                for a in 0..3 {
                    for b in 0..3 {
                        let value = if i == j {
                            u.get(a) * v.get(b)
                        } else {
                            (u.get(a) * v.get(b) + v.get(a) * u.get(b)) / std::f64::consts::SQRT_2
                        };
                        g.set(a, b, value);
                    }
                }
                generators.push(g);
            }
        }

        Ok(Some(Self {
            reference: cell,
            generators,
            length: measure.powf(1.0 / dim as f64),
        }))
    }

    fn count(&self) -> usize {
        self.generators.len()
    }

    /// `ε` from the scaled variables.
    fn strain(&self, s: &[f64]) -> Mat3 {
        let mut eps = Mat3::zero();
        for (value, generator) in s.iter().zip(&self.generators) {
            for a in 0..3 {
                for b in 0..3 {
                    eps.set(
                        a,
                        b,
                        eps.get(a, b) + value / self.length * generator.get(a, b),
                    );
                }
            }
        }
        eps
    }

    /// `∂E/∂s_k = Ω⟨σ, G_k⟩ / L`, and the largest free stress component for the separate test.
    fn gradient(&self, stress: &Mat3, measure: f64) -> (Vec<f64>, f64) {
        let mut out = Vec::with_capacity(self.generators.len());
        let mut worst = 0.0_f64;
        for generator in &self.generators {
            let mut contracted = 0.0_f64;
            for a in 0..3 {
                for b in 0..3 {
                    contracted += stress.get(a, b) * generator.get(a, b);
                }
            }
            worst = worst.max(contracted.abs());
            out.push(measure * contracted / self.length);
        }
        (out, worst)
    }
}

/// One energy-and-gradient evaluation, from whichever driver `opt` selects.
struct Evaluation {
    energy_ev: f64,
    heat_of_formation_kcal: f64,
    gradient: Vec<Vec3>,
    max_gradient: f64,
    /// The exact SCF, when that is what produced this. `None` under divide and conquer, which has
    /// no `Pm7Result` to hand back.
    scf: Option<Pm7Result>,
    /// `σ` in eV/Bohr^dim, computed only when the cell is being relaxed. Both drivers can supply
    /// one for a periodic cell.
    stress: Option<Mat3>,
}

fn evaluate(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf_options: &Pm7Options,
    opt: &OptOptions,
    want_stress: bool,
) -> Result<Evaluation> {
    let Some(dandc) = opt.dandc.as_ref() else {
        let g = closed_form_gradient(molecule, params, scf_options)?;
        let stress = if want_stress {
            // `analytic_stress`'s refusals — an external field, `PbcMode::MopacCluster` — reach
            // the caller here rather than being swallowed into an atoms-only run that reports
            // success.
            Some(crate::stress::analytic_stress(molecule, params, scf_options, &g.scf)?.stress)
        } else {
            None
        };
        return Ok(Evaluation {
            energy_ev: g.energy_ev,
            heat_of_formation_kcal: g.scf.heat_of_formation_kcal,
            gradient: g.gradient,
            max_gradient: g.max_gradient,
            scf: Some(g.scf),
            stress,
        });
    };
    let result = crate::dandc::run_dandc(molecule, params, scf_options, dandc)?;
    if !result.converged {
        return Err(crate::error::Pm7Error::ScfNotConverged {
            iterations: result.iterations,
            error: result.density_error,
        });
    }
    let d = crate::dandc::dandc_derivatives(molecule, params, scf_options, &result)?;
    let max_gradient = d
        .gradient
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let stress = if want_stress {
        Some(d.stress.ok_or_else(|| {
            Pm7Error::InvalidInput(
                "a variable-cell optimization needs a stress, and divide and conquer returns one \
                 only for a periodic cell"
                    .into(),
            )
        })?)
    } else {
        None
    };
    Ok(Evaluation {
        heat_of_formation_kcal: crate::scf::heat_of_formation_from_total(
            molecule,
            params,
            scf_options,
            d.energy_ev,
        )?,
        energy_ev: d.energy_ev,
        gradient: d.gradient,
        max_gradient,
        scf: None,
        stress,
    })
}

/// The energy alone, for the line search.
fn evaluate_energy(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf_options: &Pm7Options,
    opt: &OptOptions,
) -> Result<f64> {
    let Some(dandc) = opt.dandc.as_ref() else {
        return Ok(run_pm7(molecule, params, scf_options)?.total_ev);
    };
    let result = crate::dandc::run_dandc(molecule, params, scf_options, dandc)?;
    if !result.converged {
        return Err(crate::error::Pm7Error::ScfNotConverged {
            iterations: result.iterations,
            error: result.density_error,
        });
    }
    Ok(crate::dandc::dandc_derivatives(molecule, params, scf_options, &result)?.energy_ev)
}

#[derive(Clone, Debug)]
pub struct OptStep {
    pub energy_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub max_gradient: f64,
    /// Largest free stress component at this step, eV/Bohr^dim. Zero when the cell is fixed —
    /// there is no stress *degree of freedom* then, and reporting the fixed-cell stress here would
    /// read as something the optimizer was working on.
    pub max_stress: f64,
    pub positions: Vec<Vec3>,
    /// The cell at this step, so a variable-cell trajectory can be replayed. `None` for a
    /// molecule, and constant through a fixed-cell run.
    pub cell: Option<Cell>,
}

#[derive(Clone, Debug)]
pub struct OptResult {
    pub molecule: Molecule,
    /// The exact SCF at the final geometry. `None` under [`OptOptions::dandc`], which produces no
    /// [`Pm7Result`] -- use `energy_ev` and `heat_of_formation_kcal`, which are always present.
    pub scf: Option<Pm7Result>,
    /// Total energy at the final geometry, eV. Available from either driver.
    pub energy_ev: f64,
    /// Heat of formation at the final geometry, kcal/mol. Available from either driver.
    pub heat_of_formation_kcal: f64,
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

    // The lattice degrees of freedom, when they were asked for. `None` leaves the whole thing
    // exactly as it was: the state vector is `3N` long and no stress is ever computed.
    let cell_dof = if opt.relax_cell {
        if molecule.cell.is_none() {
            return Err(Pm7Error::InvalidInput(
                "`relax_cell` asks for the lattice to be optimized and this system has none. Give \
                 it a cell, or drop the flag for an ordinary geometry optimization."
                    .into(),
            ));
        }
        CellDof::new(molecule)?
    } else {
        None
    };
    let n_strain = cell_dof.as_ref().map_or(0, CellDof::count);
    // The reference frame: `q` are the atomic coordinates at `ε = 0` and never change meaning, so
    // a strain step moves the atoms affinely with the cell rather than shearing them out of it.
    let reference = molecule.clone();

    let mut x = flatten(&reference);
    x.extend(std::iter::repeat_n(0.0, n_strain));
    let mut mol = materialize(&reference, cell_dof.as_ref(), &x, ndof)?;
    let grad0 = evaluate(&mol, params, scf_options, opt, n_strain > 0)?;
    let mut g = combined_gradient(cell_dof.as_ref(), &grad0, &x, ndof, &mol)?;
    let mut energy = grad0.energy_ev;
    let mut heat = grad0.heat_of_formation_kcal;
    let mut max_grad = grad0.max_gradient;
    let mut max_stress = worst_stress(cell_dof.as_ref(), &grad0, &mol);
    let mut scf = grad0.scf;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut trajectory = vec![OptStep {
        energy_ev: energy,
        heat_of_formation_kcal: heat,
        max_gradient: max_grad,
        max_stress,
        positions: mol.atoms.iter().map(|a| a.position).collect(),
        cell: mol.cell,
    }];

    // Two tests, both of which have to pass. A single mixed norm would let a converged force hide
    // an unconverged stress, which is the failure this whole item exists to remove.
    let mut converged = max_grad < opt.gtol && max_stress < opt.stress_tol;
    let mut iterations = 0;

    // A local copy, because finding an instability turns following on for the rest of the run.
    let mut live_options = scf_options.clone();

    for iter in 0..opt.max_iter {
        iterations = iter + 1;
        if converged {
            break;
        }

        // Ask, periodically, whether the solution the gradients are being taken at is a minimum.
        // Only while it still might not be: once following is on it stays on, and until an
        // instability is seen there is nothing to follow.
        if opt.stability_every > 0
            && iter % opt.stability_every == 0
            && live_options.stability != crate::stability::ScfStability::Follow
        {
            if let Some(current) = scf.as_ref() {
                if let Ok(Some(found)) =
                    crate::stability::check(&mol, params, &live_options, current)
                {
                    if found.unstable {
                        live_options.stability = crate::stability::ScfStability::Follow;
                    }
                }
            }
        }
        let scf_options = &live_options;

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
            // Cautious first step, scaled by the largest component of the **whole** vector.
            //
            // Using `max_grad` here was wrong the moment the state grew a strain block. A cell
            // near its equilibrium volume but with the atoms at a symmetric site has a nuclear
            // gradient of essentially zero and a large stress — diamond is exactly that — so
            // `0.1/max_grad` came out around `1e5` and the first strain step collapsed the cell.
            // The SCF then spent minutes on a lattice a hundred times too small before the line
            // search could reject it.
            let scale = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            0.1 / scale.max(1.0e-6)
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
            // A trial step can strain the cell into something degenerate, which is a bad step and
            // not an error: shrink it and try again, exactly as for a step that raised the energy.
            if let Ok(trial_mol) = materialize(&reference, cell_dof.as_ref(), &x_new, ndof) {
                mol = trial_mol;
                if let Ok(trial) = evaluate_energy(&mol, params, scf_options, opt) {
                    if trial <= energy + c1 * step * g_dot_d {
                        ok = true;
                        break;
                    }
                }
            }
            step *= 0.5;
            if step < 1.0e-8 {
                break;
            }
        }
        if !ok {
            // Could not make progress; stop at the current point. `mol` is restored from `x` after
            // the loop, so the failed trial geometry never escapes.
            break;
        }

        let grad_new = evaluate(&mol, params, scf_options, opt, n_strain > 0)?;
        let g_new = combined_gradient(cell_dof.as_ref(), &grad_new, &x_new, ndof, &mol)?;

        // Update L-BFGS memory.
        let total = ndof + n_strain;
        let s: Vec<f64> = (0..total).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..total).map(|i| g_new[i] - g[i]).collect();
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
        heat = grad_new.heat_of_formation_kcal;
        max_grad = grad_new.max_gradient;
        max_stress = worst_stress(cell_dof.as_ref(), &grad_new, &mol);
        scf = grad_new.scf;
        converged = max_grad < opt.gtol && max_stress < opt.stress_tol;

        trajectory.push(OptStep {
            energy_ev: energy,
            heat_of_formation_kcal: heat,
            max_gradient: max_grad,
            max_stress,
            positions: mol.atoms.iter().map(|a| a.position).collect(),
            cell: mol.cell,
        });
    }

    mol = materialize(&reference, cell_dof.as_ref(), &x, ndof)?;
    Ok(OptResult {
        molecule: mol,
        scf,
        energy_ev: energy,
        heat_of_formation_kcal: heat,
        converged,
        iterations,
        trajectory,
    })
}

/// Build the molecule the state vector describes: `h = (I + ε)h₀`, `R_A = (I + ε)q_A`.
///
/// With no lattice degrees of freedom this is the reference molecule with `x` written into its
/// coordinates, which is exactly what the fixed-cell optimizer always did.
fn materialize(
    reference: &Molecule,
    cell_dof: Option<&CellDof>,
    x: &[f64],
    ndof: usize,
) -> Result<Molecule> {
    let mut mol = reference.clone();
    let Some(dof) = cell_dof else {
        set_positions(&mut mol, &x[..ndof]);
        return Ok(mol);
    };
    let eps = dof.strain(&x[ndof..]);
    // A trial step big enough to invert or collapse the cell is a bad step, not a valid geometry.
    // `Cell::new` catches a *degenerate* lattice but not one that is merely a hundred times too
    // small, and that one is worse: it is accepted, and the SCF then works for minutes on a cell
    // whose image lists have exploded before the line search gets to reject it. Half a percent
    // over unit strain is far outside anything a relaxation should pass through.
    let largest = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .fold(0.0_f64, |m, (a, b)| m.max(eps.get(a, b).abs()));
    if largest > 0.5 {
        return Err(Pm7Error::InvalidInput(format!(
            "a strain component of {largest:.3} is outside the range a relaxation should pass \
             through; the step is being rejected rather than evaluated"
        )));
    }
    mol.cell = Some(dof.reference.strained(&eps)?);
    for (atom, q) in mol.atoms.iter_mut().zip(x[..ndof].chunks(3)) {
        let reference_position = Vec3::new(q[0], q[1], q[2]);
        atom.position = reference_position + eps.mul_vec(reference_position);
    }
    Ok(mol)
}

/// `[∂E/∂q ; ∂E/∂s]`.
///
/// The nuclear block is the Cartesian gradient pulled back into the reference frame,
/// `∂E/∂q_A = (I + ε)ᵀ ∂E/∂R_A`, because `q` is what the optimizer moves and `R` is where the
/// energy was evaluated. The strain block is `Ω⟨σ, G_k⟩ / L`.
fn combined_gradient(
    cell_dof: Option<&CellDof>,
    evaluation: &Evaluation,
    x: &[f64],
    ndof: usize,
    molecule: &Molecule,
) -> Result<Vec<f64>> {
    let mut out = flatten_grad(&evaluation.gradient);
    let Some(dof) = cell_dof else {
        return Ok(out);
    };
    let eps = dof.strain(&x[ndof..]);
    for (slot, force) in out.chunks_mut(3).zip(&evaluation.gradient) {
        // `(I + ε)ᵀ f`, with ε symmetric so the transpose is the same matrix.
        let pulled = *force + eps.mul_vec(*force);
        slot[0] = pulled.x;
        slot[1] = pulled.y;
        slot[2] = pulled.z;
    }
    let stress = evaluation.stress.ok_or_else(|| {
        Pm7Error::InvalidInput("a variable-cell step needs a stress and none was computed".into())
    })?;
    let measure = molecule
        .cell
        .map(|c| c.measure())
        .ok_or_else(|| Pm7Error::InvalidInput("the strained cell has no measure".into()))?;
    out.extend(dof.gradient(&stress, measure).0);
    Ok(out)
}

/// The largest free stress component, for the convergence test that is separate from `gtol`.
fn worst_stress(cell_dof: Option<&CellDof>, evaluation: &Evaluation, molecule: &Molecule) -> f64 {
    let (Some(dof), Some(stress)) = (cell_dof, evaluation.stress) else {
        return 0.0;
    };
    let measure = molecule.cell.map(|c| c.measure()).unwrap_or(1.0);
    dof.gradient(&stress, measure).1
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
            res.heat_of_formation_kcal,
            res.trajectory.last().unwrap().max_gradient
        );
        assert!(res.converged);
        assert!((res.heat_of_formation_kcal - (-57.8)).abs() < 0.5);
        assert!(res.trajectory.last().unwrap().energy_ev < res.trajectory[0].energy_ev);
    }
}
