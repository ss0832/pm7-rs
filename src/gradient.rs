// SPDX-License-Identifier: GPL-3.0-or-later

//! Nuclear gradients of the PM7 total energy.
//!
//! Because NDDO works in an orthonormal AO basis, the SCF energy is stationary with respect to
//! the density, so the nuclear gradient is the derivative of the energy expression at the
//! **fixed converged density** (there is no Pulay/overlap-constraint term). Three routines are
//! provided:
//!
//! * [`closed_form_gradient`] — the primary, **fully closed-form** gradient (forward-mode
//!   dual-number AD of every integral kernel; radial *and* angular overlap analytic for
//!   `n ≤ 3`). No SCF re-runs; only a local-frame singular orientation uses the documented
//!   symmetric pair-integral fallback in the internal `rotfix` module. This is what the optimizer uses.
//! * [`analytic_gradient`] — the same Hellmann–Feynman gradient with the electronic term taken
//!   by fixed-density central differences (the core-core term stays closed-form). Kept for the
//!   open-shell path and as a cross-check.
//! * [`numerical_gradient`] — a full-SCF central-difference gradient, kept as an independent
//!   correctness reference (each Cartesian component re-runs the SCF twice).

use crate::basis::Basis;
use crate::error::Result;
use crate::fock::build_fock;
use crate::hamiltonian::build_core;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::repulsion::core_core_energy;
use crate::scf::{run_pm7, Pm7Options, Pm7Result};
use crate::system::Molecule;

/// Post-SCF correction Hessian (dispersion + PM7-HH H–H repulsion) scattered into
/// `hess` (eV/Bohr²). The corrections are pairwise in the interatomic distance, so
/// their 3×3 blocks are second-order dual (`Dual2`) derivatives, scattered exactly
/// like the core-core term. No-op when the method disables corrections.
pub fn add_correction_hessian(
    molecule: &Molecule,
    options: &Pm7Options,
    hess: &mut crate::linalg::Matrix,
) {
    if !options.method.has_post_scf_corrections() {
        return;
    }
    use crate::dual::Scalar;
    use crate::dual2::Dual2;
    let nb = crate::dispersion::bond_counts(molecule);
    let hh = options.method.has_hh_repulsion();
    let kcal_to_ev = crate::constants::KCAL_TO_EV;
    let a0 = crate::constants::PM7_A0;
    let n = molecule.atoms.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (zi, zj) = (molecule.atoms[i].z, molecule.atoms[j].z);
            let d = molecule.atoms[j].position - molecule.atoms[i].position;
            let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
            let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
            let mut e = crate::dispersion::pair_dispersion_scalar::<Dual2>(zi, zj, nb[i], nb[j], r);
            if hh && zi == 1 && zj == 1 {
                e = e + crate::hh_rep::poly_scalar::<Dual2>(r * a0);
            }
            // e is in kcal/mol; convert its 3×3 Hessian block to eV/Bohr².
            for (ax, row) in e.h.iter().enumerate() {
                for (bx, &v) in row.iter().enumerate() {
                    let val = v * kcal_to_ev;
                    hess[(3 * i + ax, 3 * i + bx)] += val;
                    hess[(3 * j + ax, 3 * j + bx)] += val;
                    hess[(3 * i + ax, 3 * j + bx)] -= val;
                    hess[(3 * j + ax, 3 * i + bx)] -= val;
                }
            }
        }
    }
    // Many-body hydrogen-bond term (numerical, at fixed topology); no-op without H-bonds.
    crate::hbond::add_hbond_hessian(molecule, hess);
}

/// Post-SCF correction gradient (dispersion + H-bond) in eV/Bohr, or zeros when the
/// method disables corrections (PM7-minus).
fn correction_gradient(molecule: &Molecule, options: &Pm7Options) -> Vec<Vec3> {
    if !options.method.has_post_scf_corrections() {
        return vec![Vec3::zero(); molecule.atoms.len()];
    }
    let mut g = crate::dispersion::dispersion_gradient(molecule);
    if options.method.has_hh_repulsion() {
        for (gi, hi) in g
            .iter_mut()
            .zip(crate::hh_rep::hh_repulsion_gradient(molecule))
        {
            *gi += hi;
        }
    }
    // Hydrogen-bond gradient (kcal/mol per Bohr), same units as the dispersion term.
    for (gi, hi) in g
        .iter_mut()
        .zip(crate::hbond::hydrogen_bond_gradient(molecule))
    {
        *gi += hi;
    }
    g.into_iter()
        .map(|g| g * crate::constants::KCAL_TO_EV)
        .collect()
}

/// Resonance β for orbital index `orb` (0 = s, 1..3 = p, 4..8 = d).
#[inline]
fn beta_of(elem: &crate::params::Pm7Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

/// Dual-valued two-electron integrals + 9×9 overlap for an ordered pair (ea = first),
/// seeded on the displacement `R_b − R_a`. Uses the MNDO/d two-center + overlap kernel
/// when the molecule contains a d atom, else the sp path (padded to 9×9).
#[allow(clippy::type_complexity)]
pub(crate) fn pair_dual(
    ea: &crate::params::Pm7Element,
    eb: &crate::params::Pm7Element,
    pa: Vec3,
    pb: Vec3,
    has_any_d: bool,
) -> Result<(
    crate::integrals::PairTwoElecG<crate::dual::Dual>,
    [[crate::dual::Dual; 9]; 9],
)> {
    use crate::dual::Dual;
    let d = pb - pa;
    // A bond exactly aligned with the local-frame rotation's singular axis (sp: +x, d: ±z)
    // makes the forward-mode derivative of the two-center integrals collapse; fall back to a
    // finite-difference of the (always-correct) f64 integrals for that rare pair.
    if crate::rotfix::near_frame_singularity(d, has_any_d) {
        return Ok(crate::rotfix::pair_dual_fd(ea, eb, d, has_any_d));
    }
    if has_any_d {
        let dv = [Dual::var(d.x, 0), Dual::var(d.y, 1), Dual::var(d.z, 2)];
        let te = crate::mndod_twocenter::pair_two_electron_d_g::<Dual>(ea, eb, dv);
        let s = crate::overlap_d::diat_overlap::<Dual>(ea, eb, dv);
        Ok((te, s))
    } else {
        let te = crate::integrals::pair_two_electron_dual(ea, eb, d);
        let s4 = crate::overlap::diatom_overlap_dual(ea, pa, eb, pb)?;
        let mut s = [[Dual::constant(0.0); 9]; 9];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = s4[i][j];
            }
        }
        Ok((te, s))
    }
}

/// Electronic energy (eV) at a **fixed density** matrix (no SCF, no core-core term).
pub fn electronic_energy_at_fixed_density(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: &Matrix,
) -> Result<f64> {
    let basis = Basis::build(molecule, params)?;
    let core = build_core(molecule, &basis, params)?;
    let f = build_fock(molecule, &basis, params, &core, density)?;
    Ok(0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f)))
}

/// NDDO electronic + core-core energy (eV) at a **fixed density**.
///
/// This low-level function intentionally excludes method-specific post-SCF corrections because
/// it has no [`Pm7Options`] argument. Use [`run_pm7`] for the complete PM7-family total energy.
pub fn energy_at_fixed_density(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: &Matrix,
) -> Result<f64> {
    Ok(
        electronic_energy_at_fixed_density(molecule, params, density)?
            + core_core_energy(molecule, params)?,
    )
}

/// Hellmann–Feynman nuclear gradient. `step` is the displacement in Bohr (default 5e-4).
pub fn analytic_gradient(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<GradientResult> {
    use rayon::prelude::*;

    let scf = run_pm7(molecule, params, options)?;
    let energy_ev = scf.total_ev;
    let nat = molecule.atoms.len();
    let density = scf.density.clone();

    // Core-core repulsion: exact closed-form derivative.
    let mut gradient = crate::repulsion::core_core_gradient(molecule, params)?;

    // Electronic term: Hellmann-Feynman (fixed converged density) central difference of the
    // electronic energy only — the 3N components are independent, so run them on rayon.
    let comps: Vec<(usize, usize)> = (0..nat).flat_map(|a| (0..3).map(move |k| (a, k))).collect();
    let electronic: Vec<(usize, usize, f64)> = comps
        .par_iter()
        .map(|&(a, k)| -> Result<(usize, usize, f64)> {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            displace(&mut plus.atoms[a].position, k, step);
            displace(&mut minus.atoms[a].position, k, -step);
            let ep = electronic_energy_at_fixed_density(&plus, params, &density)?;
            let em = electronic_energy_at_fixed_density(&minus, params, &density)?;
            Ok((a, k, (ep - em) / (2.0 * step)))
        })
        .collect::<Result<Vec<_>>>()?;
    for (a, k, g) in electronic {
        match k {
            0 => gradient[a].x += g,
            1 => gradient[a].y += g,
            _ => gradient[a].z += g,
        }
    }

    // The fixed-density electronic difference above deliberately excludes post-SCF terms.
    // Add their geometry derivative so this public validation path differentiates scf.total_ev.
    for (g, correction) in gradient
        .iter_mut()
        .zip(correction_gradient(molecule, options))
    {
        *g += correction;
    }

    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

#[derive(Clone, Debug)]
pub struct GradientResult {
    /// Converged SCF result at the input geometry.
    pub scf: Pm7Result,
    /// Total energy (eV).
    pub energy_ev: f64,
    /// Gradient dE/dR in eV/Bohr (atomic-unit length).
    pub gradient: Vec<Vec3>,
    /// Forces = −gradient (eV/Bohr).
    pub forces: Vec<Vec3>,
    /// Largest gradient component magnitude (eV/Bohr).
    pub max_gradient: f64,
}

/// Finite-difference nuclear gradient. `step` is the displacement in Bohr (default 5e-4).
pub fn numerical_gradient(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<GradientResult> {
    let scf = run_pm7(molecule, params, options)?;
    let energy_ev = scf.total_ev;
    let nat = molecule.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];

    let energy_at = |m: &Molecule| -> Result<f64> { Ok(run_pm7(m, params, options)?.total_ev) };

    for a in 0..nat {
        for k in 0..3 {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            displace(&mut plus.atoms[a].position, k, step);
            displace(&mut minus.atoms[a].position, k, -step);
            let ep = energy_at(&plus)?;
            let em = energy_at(&minus)?;
            let g = (ep - em) / (2.0 * step);
            set_component(&mut gradient[a], k, g);
        }
    }

    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));

    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

/// Fully closed-form (dual-number) Hellmann–Feynman gradient. The two-electron and
/// core-attraction integral derivatives, the overlap (radial *and* angular, for valence shells
/// `n ≤ 3`), and the core-core term are all exact forward-mode AD — no SCF re-runs and no
/// finite differences. (Heavy elements, `n ≥ 4`, keep a tight 1-D radial overlap difference.)
/// Falls back to the fixed-density gradient for open-shell (UHF) systems.
pub fn closed_form_gradient(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
) -> Result<GradientResult> {
    let scf = run_pm7(molecule, params, options)?;
    // Post-SCF dispersion gradient (eV/Bohr), added when corrections are enabled.
    let correction_grad = correction_gradient(molecule, options);
    if scf.unrestricted {
        // Open-shell: spin-resolved closed-form fixed-density (Hellmann–Feynman) gradient.
        let energy_ev = scf.total_ev;
        let mut gradient = fixed_density_gradient_uhf(molecule, params, &scf)?;
        for (g, c) in gradient.iter_mut().zip(&correction_grad) {
            *g += *c;
        }
        let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
        let max_gradient = gradient
            .iter()
            .flat_map(|g| g.to_array())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        return Ok(GradientResult {
            scf,
            energy_ev,
            gradient,
            forces,
            max_gradient,
        });
    }
    let energy_ev = scf.total_ev;
    let mut gradient = fixed_density_gradient(molecule, params, &scf.density)?;
    for (g, c) in gradient.iter_mut().zip(&correction_grad) {
        *g += *c;
    }
    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(GradientResult {
        scf,
        energy_ev,
        gradient,
        forces,
        max_gradient,
    })
}

/// Total closed-form gradient (core-core + electronic) at an **arbitrary fixed density** `p`
/// (no SCF solve). Finite-differencing this over the nuclei at fixed `p` gives the skeleton
/// (fixed-density) second derivative used by the analytic Hessian.
pub fn fixed_density_gradient(
    molecule: &Molecule,
    params: &Pm7Parameters,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    let basis = Basis::build(molecule, params)?;
    let mut gradient = crate::repulsion::core_core_gradient(molecule, params)?;
    let elec = electronic_gradient_fixed_density(molecule, params, &basis, p)?;
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    Ok(gradient)
}

/// Electronic part of the closed-form gradient at fixed density `p` (dual-number contraction).
pub fn electronic_gradient_fixed_density(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));
    // Parallelize the expensive per-pair dual-integral build; scatter the small [f;3] force
    // contributions serially in pair order (bit-identical to the serial accumulation).
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let contribs: Result<Vec<(usize, usize, [f64; 3])>> = pairs
        .par_iter()
        .map(|&(u, v)| {
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            // Heavier atom (more AOs) first, matching the SCF core build.
            let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let (pa, pb) = (molecule.atoms[a].position, molecule.atoms[b].position);
            let (te, s) = pair_dual(ea, eb, pa, pb, has_any_d)?;
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            let mut f = [0.0_f64; 3];
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[ob + j].orb);
                    let coef = p[(oa + i, ob + j)] * (bi + bj);
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
            for i in 0..na {
                for j in 0..na {
                    let coef = p[(oa + i, oa + j)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e1b[i][j].d[ax];
                    }
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    let coef = p[(ob + k, ob + l)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e2a[k][l].d[ax];
                    }
                }
            }
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            let dw = te.two_e(mu, nu, la, si).d;
                            let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                            let exch = -0.5 * p[(oa + mu, ob + la)] * p[(oa + nu, ob + si)];
                            let coef = coul + exch;
                            for (ax, fx) in f.iter_mut().enumerate() {
                                *fx += coef * dw[ax];
                            }
                        }
                    }
                }
            }
            Ok((a, b, f))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); nat];
    for (a, b, f) in contribs? {
        gradient[b] += Vec3::new(f[0], f[1], f[2]);
        gradient[a] -= Vec3::new(f[0], f[1], f[2]);
    }
    Ok(gradient)
}

/// Total closed-form UHF gradient (core-core + spin-resolved electronic) at the converged
/// open-shell density. `Pα = (P_tot + S)/2`, `Pβ = (P_tot − S)/2` are reconstructed from the
/// total density and the spin density `S = Pα − Pβ`. Hellmann–Feynman (orthonormal basis).
pub fn fixed_density_gradient_uhf(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf: &Pm7Result,
) -> Result<Vec<Vec3>> {
    let basis = Basis::build(molecule, params)?;
    let pt = &scf.density;
    let spin = scf.spin_density.as_ref().ok_or_else(|| {
        crate::error::Pm7Error::InvalidInput("UHF gradient requires a spin density".into())
    })?;
    let mut pa = pt.clone();
    let mut pb = pt.clone();
    {
        let n = pt.as_slice().len();
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..n {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
    }
    let mut gradient = crate::repulsion::core_core_gradient(molecule, params)?;
    let elec = electronic_gradient_fixed_density_spin(molecule, params, &basis, pt, &pa, &pb)?;
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    Ok(gradient)
}

/// Spin-resolved electronic part of the closed-form gradient at fixed densities: resonance,
/// electron–core attraction, and Coulomb use the **total** density `P_tot`; exchange uses the
/// **same-spin** densities `Pα`, `Pβ` (`−[Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ](μν|λσ)`). Reduces to the
/// RHF form when `Pα = Pβ = P_tot/2`.
pub fn electronic_gradient_fixed_density_spin(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    pt: &Matrix,
    pa: &Matrix,
    pb: &Matrix,
) -> Result<Vec<Vec3>> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let contribs: Result<Vec<(usize, usize, [f64; 3])>> = pairs
        .par_iter()
        .map(|&(u, v)| {
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let (pos_a, pos_b) = (molecule.atoms[a].position, molecule.atoms[b].position);
            let (te, s) = pair_dual(ea, eb, pos_a, pos_b, has_any_d)?;
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            let mut f = [0.0_f64; 3];
            // Resonance β·S (total density).
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[ob + j].orb);
                    let coef = pt[(oa + i, ob + j)] * (bi + bj);
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
            // Electron–core attraction (total density).
            for i in 0..na {
                for j in 0..na {
                    let coef = pt[(oa + i, oa + j)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e1b[i][j].d[ax];
                    }
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    let coef = pt[(ob + k, ob + l)];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e2a[k][l].d[ax];
                    }
                }
            }
            // Two-electron: Coulomb from P_tot, exchange from same-spin Pα/Pβ.
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            let dw = te.two_e(mu, nu, la, si).d;
                            let coul = pt[(oa + mu, oa + nu)] * pt[(ob + la, ob + si)];
                            let exch = -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)]);
                            let coef = coul + exch;
                            for (ax, fx) in f.iter_mut().enumerate() {
                                *fx += coef * dw[ax];
                            }
                        }
                    }
                }
            }
            Ok((a, b, f))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); nat];
    for (a, b, f) in contribs? {
        gradient[b] += Vec3::new(f[0], f[1], f[2]);
        gradient[a] -= Vec3::new(f[0], f[1], f[2]);
    }
    Ok(gradient)
}

#[inline]
fn displace(p: &mut Vec3, k: usize, d: f64) {
    match k {
        0 => p.x += d,
        1 => p.y += d,
        _ => p.z += d,
    }
}

#[inline]
fn set_component(v: &mut Vec3, k: usize, val: f64) {
    match k {
        0 => v.x = val,
        1 => v.y = val,
        _ => v.z = val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_matches_full_scf_gradient() {
        // Hellmann–Feynman (fixed-density) gradient must match the full-SCF finite
        // difference on a molecule displaced away from equilibrium (nonzero forces).
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.02 0.0 0.0\nH -0.28 0.96 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let a = analytic_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (ga, gn) in a.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((ga.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("analytic-vs-numerical gradient max delta = {max_delta:.3e} eV/Bohr");
        assert!(max_delta < 1.0e-4, "gradient mismatch {max_delta:.3e}");
        // Forces must be nonzero for this distorted geometry.
        assert!(a.max_gradient > 1.0e-2);
    }

    #[test]
    fn closed_form_matches_numerical_gradient() {
        // The fully closed-form (dual-number) gradient must match the full-SCF finite
        // difference on a molecule with s and p atoms displaced from equilibrium.
        let mol = Molecule::from_xyz_str(
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.03 0.0 1.25\nH 0.95 0.02 -0.55\nH -0.94 -0.03 -0.52\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("closed-form-vs-numerical gradient max delta = {max_delta:.3e} eV/Bohr");
        assert!(
            max_delta < 5.0e-5,
            "closed-form gradient mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn d_shell_closed_form_gradient_matches_numerical() {
        // H2S (sulfur carries d orbitals): the fully closed-form MNDO/d gradient
        // (dual-number AD through the two-center kernel, overlap, and core-core)
        // must match the full-SCF finite difference.
        let mol = Molecule::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.9705 0.9430\nH 0.0 -0.9705 0.9430\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("d-shell closed-form-vs-numerical gradient max delta = {max_delta:.3e} eV/Bohr");
        assert!(
            max_delta < 5.0e-5,
            "d-shell gradient mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn closed_form_gradient_uhf_radical() {
        // Methyl radical (doublet, UHF), distorted from planar: the spin-resolved closed-form
        // gradient must match the full-SCF finite difference (no fixed-density FD fallback).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.05\nH 1.12 0.0 0.0\nH -0.55 0.95 0.0\nH -0.55 -0.95 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options {
            multiplicity: 2,
            ..Pm7Options::default()
        };
        let cf = closed_form_gradient(&mol, &params, &opts).unwrap();
        let n = numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
        assert!(cf.scf.unrestricted);
        let mut max_delta = 0.0_f64;
        for (gc, gn) in cf.gradient.iter().zip(&n.gradient) {
            for k in 0..3 {
                max_delta = max_delta.max((gc.get(k) - gn.get(k)).abs());
            }
        }
        eprintln!("UHF closed-form-vs-numerical gradient max delta = {max_delta:.3e}");
        assert!(max_delta < 5.0e-5, "UHF gradient mismatch {max_delta:.3e}");
        assert!(cf.max_gradient > 1.0e-2);
    }
}
