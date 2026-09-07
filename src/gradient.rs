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

/// [`add_correction_hessian`] over periodic images, with the same tapered cutoff the periodic
/// correction energy and gradient use.
///
/// The pair terms go through second-order duals in the **displacement** rather than in the
/// distance, so the taper's product rule (`w'' e + 2 w' e' + w e''`) comes out of the arithmetic
/// instead of being written by hand. A molecule takes the same path with an infinite cutoff, and
/// the taper is then the constant 1.
pub fn add_correction_hessian_periodic(
    molecule: &Molecule,
    options: &Pm7Options,
    hess: &mut crate::linalg::Matrix,
) {
    if !options.method.has_post_scf_corrections() {
        return;
    }
    let Some(pbc) = options.pbc_for(molecule) else {
        add_correction_hessian(molecule, options, hess);
        return;
    };
    use crate::dual::Scalar;
    use crate::dual2::Dual2;
    let cutoff = pbc.correction_cutoff;
    let r_on = crate::pbc::taper_onset(cutoff);
    let nb = crate::dispersion::bond_counts(molecule);
    let hh = options.method.has_hh_repulsion();
    let kcal_to_ev = crate::constants::KCAL_TO_EV;
    let a0 = crate::constants::PM7_A0;
    for pair in &crate::pbc::PairList::cached(molecule, cutoff).pairs {
        let (a, b) = (pair.a, pair.b);
        if a == b {
            // A self-image pair's separation does not move with the atom.
            continue;
        }
        let d = [
            Dual2::var(pair.d.x, 0),
            Dual2::var(pair.d.y, 1),
            Dual2::var(pair.d.z, 2),
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (za, zb) = (molecule.atoms[a].z, molecule.atoms[b].z);
        let mut e = crate::dispersion::pair_dispersion_scalar::<Dual2>(za, zb, nb[a], nb[b], r);
        if hh && za == 1 && zb == 1 {
            e = e + crate::hh_rep::poly_scalar::<Dual2>(r * a0);
        }
        let e = e * crate::pbc::taper_scalar::<Dual2>(r, r_on, cutoff);
        let scale = kcal_to_ev * pair.weight;
        for (ax, row) in e.h.iter().enumerate() {
            for (bx, &v) in row.iter().enumerate() {
                let val = v * scale;
                hess[(3 * a + ax, 3 * a + bx)] += val;
                hess[(3 * b + ax, 3 * b + bx)] += val;
                hess[(3 * a + ax, 3 * b + bx)] -= val;
                hess[(3 * b + ax, 3 * a + bx)] -= val;
            }
        }
    }
    // The EH+ hydrogen-bond Hessian is already image-aware: it builds the unwrapped cluster,
    // takes only the bonds this cell owns, and folds the block back onto the central atoms.
    crate::hbond::add_hbond_hessian(molecule, hess);
}

/// The post-SCF corrections' contribution to a dynamical matrix at wavevector `q`.
///
/// The phased twin of [`add_correction_hessian_periodic`]. Without it a `method = "pm7"` phonon
/// calculation through `dynamical_matrix_dfpt` silently omitted the dispersion, PM7-HH and EH+
/// force constants entirely — silently because every test in `tests/dfpt.rs` used `"pm7-"`, which
/// switches all three off.
///
/// The pairwise terms scatter exactly like the skeleton: diagonals unphased, off-diagonals with
/// `e^{±iq·T}`. The many-body EH+ term goes through
/// [`crate::hbond::add_hbond_hessian_phased`], which phases per cluster image.
pub fn add_correction_hessian_phased(
    molecule: &Molecule,
    options: &Pm7Options,
    q_cart: Vec3,
    out: &mut crate::cmatrix::CMatrix,
) {
    if !options.method.has_post_scf_corrections() {
        return;
    }
    let Some(pbc) = options.pbc_for(molecule) else {
        return;
    };
    let Some(cell) = molecule.cell else {
        return;
    };
    use crate::dual::Scalar;
    use crate::dual2::Dual2;
    let cutoff = pbc.correction_cutoff;
    let r_on = crate::pbc::taper_onset(cutoff);
    let nb = crate::dispersion::bond_counts(molecule);
    let hh = options.method.has_hh_repulsion();
    let kcal_to_ev = crate::constants::KCAL_TO_EV;
    let a0 = crate::constants::PM7_A0;
    for pair in &crate::pbc::PairList::cached(molecule, cutoff).pairs {
        let (a, b) = (pair.a, pair.b);
        let d = [
            Dual2::var(pair.d.x, 0),
            Dual2::var(pair.d.y, 1),
            Dual2::var(pair.d.z, 2),
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let (za, zb) = (molecule.atoms[a].z, molecule.atoms[b].z);
        let mut e = crate::dispersion::pair_dispersion_scalar::<Dual2>(za, zb, nb[a], nb[b], r);
        if hh && za == 1 && zb == 1 {
            e = e + crate::hh_rep::poly_scalar::<Dual2>(r * a0);
        }
        let e = e * crate::pbc::taper_scalar::<Dual2>(r, r_on, cutoff);
        let scale = kcal_to_ev * pair.weight;
        let angle = q_cart.dot(cell.translation(pair.t));
        let (cos, sin) = (angle.cos(), angle.sin());
        for (ax, row) in e.h.iter().enumerate() {
            for (bx, &v) in row.iter().enumerate() {
                let w = v * scale;
                // A self-image pair (`a == b`, `T ≠ 0`) cancels at `q = 0` and does not away from
                // it: the four terms below become `2w(1 − cos q·T)` on the diagonal block.
                let bump =
                    |m: &mut crate::cmatrix::CMatrix, r: usize, c: usize, re: f64, im: f64| {
                        let (x, y) = m.get(r, c);
                        m.set(r, c, x + re, y + im);
                    };
                bump(out, 3 * a + ax, 3 * a + bx, w, 0.0);
                bump(out, 3 * b + ax, 3 * b + bx, w, 0.0);
                bump(out, 3 * a + ax, 3 * b + bx, -w * cos, -w * sin);
                bump(out, 3 * b + ax, 3 * a + bx, -w * cos, w * sin);
            }
        }
    }
    crate::hbond::add_hbond_hessian_phased(molecule, &cell, q_cart, out);
}

/// Post-SCF correction gradient (dispersion + H-bond) in eV/Bohr, or zeros when the
/// method disables corrections (PM7-minus).
fn correction_gradient(molecule: &Molecule, options: &Pm7Options) -> Vec<Vec3> {
    correction_gradient_and_virial(molecule, options).0
}

/// Correction gradient (eV/Bohr) and virial (eV) — the latter `Σ (∂E/∂d) ⊗ d` over the same
/// image pairs, which is the exact strain derivative of a term that depends on the atoms only
/// through their separations.
pub fn correction_gradient_and_virial(
    molecule: &Molecule,
    options: &Pm7Options,
) -> (Vec<Vec3>, crate::math::Mat3) {
    let n = molecule.atoms.len();
    if !options.method.has_post_scf_corrections() {
        return (vec![Vec3::zero(); n], crate::math::Mat3::zero());
    }
    let cutoff = options
        .pbc_for(molecule)
        .map(|p| p.correction_cutoff)
        .unwrap_or(f64::INFINITY);
    let (mut g, mut virial) = if options.smooth_dispersion {
        crate::dispersion::dispersion_gradient_smooth_cut(molecule, cutoff)
    } else {
        crate::dispersion::dispersion_gradient_cut(molecule, cutoff)
    };
    if options.method.has_hh_repulsion() {
        let (hg, hv) = crate::hh_rep::hh_repulsion_gradient_cut(molecule, cutoff);
        for (gi, hi) in g.iter_mut().zip(hg) {
            *gi += hi;
        }
        virial = virial.plus(&hv);
    }
    // Hydrogen-bond gradient (kcal/mol per Bohr), same units as the dispersion term.
    let (hbg, hbv) = crate::hbond::hydrogen_bond_gradient_and_virial(molecule);
    for (gi, hi) in g.iter_mut().zip(hbg) {
        *gi += hi;
    }
    virial = virial.plus(&hbv);
    // The molecular-mechanics corrections, in lockstep with `scf::correction_energy`. Adding an
    // energy term without its derivative would leave `optimize` pulling against `energy`, which is
    // worse than omitting both.
    for (grad, vir) in [
        crate::mm_corrections::c_triple_bond_gradient_and_virial(molecule),
        crate::mm_corrections::si_o_h_gradient_and_virial(molecule),
    ] {
        for (gi, mi) in g.iter_mut().zip(grad) {
            *gi += mi;
        }
        virial = virial.plus(&vir);
    }
    let k = crate::constants::KCAL_TO_EV;
    (g.into_iter().map(|g| g * k).collect(), virial.scaled(k))
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
    electronic_energy_at_fixed_density_with(molecule, params, density, None)
}

/// [`electronic_energy_at_fixed_density`] with an optional external field folded into `h_core`.
///
/// The field-aware form exists so [`analytic_gradient`] stays a *genuinely independent* check of
/// [`closed_form_gradient`] when a field is applied: one side differentiates the energy, the
/// other evaluates `q_A f` in closed form, and they share no code beyond the SCF.
pub fn electronic_energy_at_fixed_density_with(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: &Matrix,
    field: Option<&crate::field::ExternalField>,
) -> Result<f64> {
    let basis = Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_field(molecule, &basis, params, false, field)?;
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
    energy_at_fixed_density_with(molecule, params, density, None)
}

/// [`energy_at_fixed_density`] with an optional external field, including its nuclear half.
pub fn energy_at_fixed_density_with(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: &Matrix,
    field: Option<&crate::field::ExternalField>,
) -> Result<f64> {
    let mut total = electronic_energy_at_fixed_density_with(molecule, params, density, field)?
        + core_core_energy(molecule, params)?;
    if let Some(f) = field {
        total += f.core_energy(molecule, params)?;
    }
    Ok(total)
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

    // The field's **nuclear** half, `Σ_A Z_A (f·R_A)`, differentiates to `Z_A f`. Its electronic
    // half is inside the finite difference below, and the two combine to `q_A f`. Splitting them
    // this way — rather than adding `q_A f` outright — is what keeps this an independent check of
    // `closed_form_gradient`, which does add `q_A f` outright.
    if let Some(f) = options.active_field() {
        let fi = f.internal();
        for (g, atom) in gradient.iter_mut().zip(&molecule.atoms) {
            *g += fi * params.element(atom.z)?.core_charge;
        }
    }

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
            let field = options.active_field();
            let ep = electronic_energy_at_fixed_density_with(&plus, params, &density, field)?;
            let em = electronic_energy_at_fixed_density_with(&minus, params, &density, field)?;
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
        let mut gradient = fixed_density_gradient_uhf_with(molecule, params, &scf, options)?.0;
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
    // A k-point run hands back `P(T)`; use it, because the resonance and exchange terms of the
    // gradient contract the block at the translation they act on.
    let divisions = options
        .pbc_for(molecule)
        .map(|p| p.kmesh.divisions())
        .unwrap_or([1, 1, 1]);
    let density = match &scf.bloch_density {
        Some(b) => TranslatedDensity::Bloch(b, divisions),
        None => TranslatedDensity::Uniform(&scf.density),
    };
    let mut gradient = fixed_density_gradient_blocks(molecule, params, density, options)?.0;
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

/// A density resolved by lattice translation wherever the translation matters.
///
/// The resonance (`β·S`) and exchange terms of a periodic energy contract the density block
/// **between the home cell and the image they act on**, `P(T)`. At Γ that is the same matrix for
/// every `T`, so one suffices and the distinction never arises; a k mesh makes `P(T)` decay with
/// distance, and contracting it against `P(0)` instead leaves the gradient and the stress wrong
/// by whole eV/Å — enough to put spurious forces on a perfect crystal. Everything else (on-site
/// blocks, Coulomb, Mulliken charges) is a `P(0)` quantity either way.
#[derive(Clone, Copy)]
pub enum TranslatedDensity<'a> {
    /// One matrix for every image: a molecule, or a Γ-point cell.
    Uniform(&'a Matrix),
    /// `P(T)` from a k-point run, with the Born–von Kármán mesh divisions used to fold a
    /// translation the block set does not name.
    Bloch(&'a crate::scf_pbc::BlochBlocks, [usize; 3]),
}

impl<'a> TranslatedDensity<'a> {
    /// The on-site block `P(0)`.
    pub fn onsite(self) -> &'a Matrix {
        match self {
            Self::Uniform(p) => p,
            Self::Bloch(b, _) => b
                .get([0, 0, 0])
                .expect("Bloch density blocks contain the zero translation"),
        }
    }

    /// The block at translation `t`.
    pub fn at(self, t: [i32; 3]) -> &'a Matrix {
        match self {
            Self::Uniform(p) => p,
            Self::Bloch(b, n) => b
                .folded(t, n)
                .expect("Bloch density blocks cover the pair list's translations"),
        }
    }

    /// Whether this is a k-point (translation-resolved) density.
    pub fn is_bloch(self) -> bool {
        matches!(self, Self::Bloch(..))
    }
}

/// The α and β densities of a k-point run, as translation-resolved blocks.
pub fn bloch_spin_densities(
    total: &crate::scf_pbc::BlochBlocks,
    spin: Option<&crate::scf_pbc::BlochBlocks>,
) -> (crate::scf_pbc::BlochBlocks, crate::scf_pbc::BlochBlocks) {
    let mut pa = total.clone();
    let mut pb = total.clone();
    for i in 0..total.len() {
        let t = total.block(i).as_slice().to_vec();
        let s = spin.map(|s| s.block(i).as_slice().to_vec());
        let a = pa.block_mut(i).as_mut_slice();
        for (k, v) in a.iter_mut().enumerate() {
            *v = 0.5 * (t[k] + s.as_ref().map(|m| m[k]).unwrap_or(0.0));
        }
        let b = pb.block_mut(i).as_mut_slice();
        for (k, v) in b.iter_mut().enumerate() {
            *v = 0.5 * (t[k] - s.as_ref().map(|m| m[k]).unwrap_or(0.0));
        }
    }
    (pa, pb)
}

/// Total closed-form gradient (core-core + electronic) at an **arbitrary fixed density** `p`
/// (no SCF solve). Finite-differencing this over the nuclei at fixed `p` gives the skeleton
/// (fixed-density) second derivative used by the analytic Hessian.
pub fn fixed_density_gradient(
    molecule: &Molecule,
    params: &Pm7Parameters,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    fixed_density_gradient_with(molecule, params, p, &Pm7Options::default())
}

/// [`fixed_density_gradient`] with explicit options, which is what carries the periodic
/// settings. A periodic system needs them: the pair enumeration, the monopole subtraction, and
/// the long-range Ewald force all come from there, and defaulting them silently would
/// differentiate a different energy than the one that was computed.
pub fn fixed_density_gradient_with(
    molecule: &Molecule,
    params: &Pm7Parameters,
    p: &Matrix,
    options: &Pm7Options,
) -> Result<Vec<Vec3>> {
    fixed_density_gradient_blocks(molecule, params, TranslatedDensity::Uniform(p), options)
        .map(|r| r.0)
}

/// [`fixed_density_gradient_with`] over a possibly translation-resolved density, returning the
/// virial alongside the gradient.
///
/// A k-point run must come through here with [`TranslatedDensity::Bloch`]: its `P(T)` decays with
/// distance, and the resonance and exchange terms contract the block at the translation they act
/// on rather than `P(0)`.
pub fn fixed_density_gradient_blocks(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: TranslatedDensity<'_>,
    options: &Pm7Options,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let basis = Basis::build(molecule, params)?;
    match options.pbc_for(molecule) {
        None => {
            let mut gradient = crate::repulsion::core_core_gradient(molecule, params)?;
            let elec =
                electronic_gradient_fixed_density(molecule, params, &basis, density.onsite())?;
            for (g, e) in gradient.iter_mut().zip(&elec) {
                *g += *e;
            }
            // The external field, if any: `∂E_field/∂R_A = q_A f`, closed form. MOPAC's
            // `dfield.F90`. Kept out of `electronic_gradient_fixed_density`, which enumerates
            // *pairs*; this is a one-centre term and folding it in there would put a per-atom
            // quantity inside a pair loop for no reason and cost the no-field path bit-identity.
            if let Some(f) = options.active_field() {
                let charges = mulliken_charges(molecule, params, &basis, density.onsite())?;
                for (g, e) in gradient.iter_mut().zip(f.gradient(&charges)) {
                    *g += e;
                }
            }
            Ok((gradient, crate::math::Mat3::zero()))
        }
        Some(pbc) => {
            let (mut gradient, mut virial) =
                crate::repulsion::core_core_gradient_periodic(molecule, params, &pbc)?;
            let (elec, elec_virial) =
                electronic_gradient_periodic(molecule, params, &basis, density, &pbc)?;
            for (g, e) in gradient.iter_mut().zip(&elec) {
                *g += *e;
            }
            virial = virial.plus(&elec_virial);
            // Long-range monopole forces, Coulomb and exchange. The short-range terms had their
            // monopoles removed pair by pair; this puts back the derivative of the whole
            // lattice sum. For RHF the same-spin density is half the total.
            if pbc.mode == crate::pbc::PbcMode::Ewald {
                let charges = mulliken_charges(molecule, params, &basis, density.onsite())?;
                let half = halved(density);
                let spin = half.as_density(density);
                let (lr, lr_virial) =
                    long_range_derivatives(molecule, &basis, &charges, &[spin, spin], &pbc)?;
                for (g, e) in gradient.iter_mut().zip(&lr) {
                    *g += *e;
                }
                virial = virial.plus(&lr_virial);
            }
            // The same one-centre field term as the molecular branch. A field is only ever
            // admissible along a **non-periodic** direction (`crate::field::validate_for`), but
            // that makes it no less real: it shifts `h_core`, it shifts the converged density,
            // and it exerts `q_A f` on every atom. Leaving it out here while `build_core` put it
            // in is what made a field on a periodic cell produce a field-free force.
            //
            // It contributes nothing to the **virial**: `E_field` depends on the Cartesian
            // position along a free axis, which no strain of the periodic axes moves. Callers
            // that want a stress under a field are refused upstream rather than handed this.
            if let Some(f) = options.active_field() {
                let charges = mulliken_charges(molecule, params, &basis, density.onsite())?;
                for (g, e) in gradient.iter_mut().zip(f.gradient(&charges)) {
                    *g += e;
                }
            }
            Ok((gradient, virial))
        }
    }
}

/// Half a density, kept in whichever representation it came in.
enum Halved {
    Uniform(Matrix),
    Bloch(crate::scf_pbc::BlochBlocks),
}

impl Halved {
    fn as_density<'a>(&'a self, like: TranslatedDensity<'_>) -> TranslatedDensity<'a> {
        match (self, like) {
            (Self::Uniform(m), _) => TranslatedDensity::Uniform(m),
            (Self::Bloch(b), TranslatedDensity::Bloch(_, n)) => TranslatedDensity::Bloch(b, n),
            (Self::Bloch(b), _) => TranslatedDensity::Bloch(b, [1, 1, 1]),
        }
    }
}

fn halved(density: TranslatedDensity<'_>) -> Halved {
    match density {
        TranslatedDensity::Uniform(p) => {
            let mut m = p.clone();
            for v in m.as_mut_slice() {
                *v *= 0.5;
            }
            Halved::Uniform(m)
        }
        TranslatedDensity::Bloch(b, _) => Halved::Bloch(crate::scf_pbc::scale_blocks(b, 0.5)),
    }
}

/// The coefficient matrix of the long-range **exchange** lattice sum.
///
/// The exchange operator's monopole part is `−P^σ(μ_A, λ_B) Σ'_T v(r_AB + T)`, so its energy is
/// `½ Σ_{A,B} c_AB M_AB` with
///
/// ```text
/// c_AB = −Σ_σ Σ_{μ∈A, λ∈B} P^σ(μ_A, λ_B)²
/// ```
///
/// Unlike the Coulomb coefficient `q_A q_B` this does not factorize, which is why the gradient
/// and stress go through [`crate::pbc::ewald::ewald_pair_matrix`] rather than the charge path.
pub fn exchange_coefficient_matrix(basis: &Basis, spin_densities: &[&Matrix]) -> Vec<Vec<f64>> {
    let n = basis.atom_offset.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for ia in 0..n {
        let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
        for ib in 0..n {
            let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
            let mut sum = 0.0;
            for p in spin_densities {
                for mu in 0..na {
                    for la in 0..nb {
                        let x = p[(oa + mu, ob + la)];
                        sum += x * x;
                    }
                }
            }
            c[ia][ib] = -sum;
        }
    }
    c
}

/// Long-range Coulomb **and** exchange forces and virial for a periodic system.
pub(crate) fn long_range_derivatives(
    molecule: &Molecule,
    basis: &Basis,
    charges: &[f64],
    spin_densities: &[TranslatedDensity<'_>],
    pbc: &crate::pbc::PbcOptions,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let cell = molecule.cell.expect("periodic");
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let ep = crate::pbc::EwaldParameters::new(
        &cell,
        positions.len(),
        pbc.ewald_accuracy,
        pbc.ewald_alpha,
    );
    let coulomb = crate::pbc::ewald::ewald(&cell, &positions, charges, &ep, pbc)?;
    let (ex_gradient, ex_virial) =
        long_range_exchange_derivatives(&cell, &positions, basis, spin_densities, pbc)?;
    let gradient: Vec<Vec3> = (0..positions.len())
        .map(|i| coulomb.gradient[i] + ex_gradient[i])
        .collect();
    Ok((gradient, coulomb.virial.plus(&ex_virial)))
}

/// Forces and virial of the long-range **exchange** lattice sum.
///
/// The energy this differentiates is
///
/// ```text
/// E = ½ Σ_t Σ_{A,B} c_AB(t) [Φ(d_AB + T_t) − Φ(0)]
/// ```
///
/// with `Φ` the regularized lattice sum of the Born–von Kármán supercell and
/// `c_AB(t) = −Σ_σ Σ_{μ∈A, λ∈B} P^σ(t)²_{μλ}`. At Γ there is one residue class, the supercell is
/// the cell, and this collapses to `½ Σ_AB c_AB (M_AB − M_self)` — the Γ expression, evaluated by
/// the very same code, so the two cannot drift apart.
pub(crate) fn long_range_exchange_derivatives(
    cell: &crate::cell::Cell,
    positions: &[Vec3],
    basis: &Basis,
    spin_densities: &[TranslatedDensity<'_>],
    pbc: &crate::pbc::PbcOptions,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let n = positions.len();
    let bloch = spin_densities.iter().any(|d| d.is_bloch());
    let divisions = if bloch {
        pbc.kmesh.divisions()
    } else {
        [1, 1, 1]
    };
    let classes = crate::hamiltonian::bvk_representatives(divisions);
    let (super_cell, _) = cell.supercell(divisions)?;
    let ep = crate::pbc::EwaldParameters::new(
        &super_cell,
        n * classes.len(),
        pbc.ewald_accuracy,
        pbc.ewald_alpha,
    );

    let mut gradient = vec![Vec3::zero(); n];
    let mut virial = crate::math::Mat3::zero();
    let mut total = 0.0_f64;

    // Every class's reciprocal sum in one pass over `G`, when the sum is three-dimensional.
    //
    // The class loop below then runs with the reciprocal kernel switched off: what is left in it
    // — the real-space `erfc` sum, the self term, the background — genuinely depends on the
    // shifted distance `|d_AB + T_t|` and does not factorize. Those are already linear in the
    // class count, because a class whose translation carries every pair past the real-space cutoff
    // contributes nothing.
    //
    // See [`crate::pbc::ewald::ewald_reciprocal_bvk`] for why this is `O(C)` where the class loop
    // was `O(C²)`, and why the answer is 3-D only.
    let folded = super_cell.dim() == 3 && classes.len() > 1;
    if folded {
        let per_class: Vec<Vec<Vec<f64>>> = classes
            .iter()
            .map(|t| {
                let blocks: Vec<&Matrix> = spin_densities.iter().map(|d| d.at(*t)).collect();
                exchange_coefficient_matrix(basis, &blocks)
            })
            .collect();
        let (_, g, v) = crate::pbc::ewald::ewald_reciprocal_bvk(
            cell,
            &super_cell,
            divisions,
            positions,
            &per_class,
            &ep,
        );
        for (dst, src) in gradient.iter_mut().zip(&g) {
            *dst += *src;
        }
        virial = virial.plus(&v);
    }
    let pair_matrix = |positions: &[Vec3], c: &[Vec<f64>]| {
        crate::pbc::ewald::ewald_pair_matrix_with(&super_cell, positions, c, &ep, !folded)
    };

    for t in &classes {
        let blocks: Vec<&Matrix> = spin_densities.iter().map(|d| d.at(*t)).collect();
        let c = exchange_coefficient_matrix(basis, &blocks);
        total += c.iter().flatten().sum::<f64>();
        if *t == [0, 0, 0] {
            let out = pair_matrix(positions, &c);
            for (g, v) in gradient.iter_mut().zip(&out.gradient) {
                *g += *v;
            }
            virial = virial.plus(&out.virial);
            continue;
        }
        // A non-zero residue class needs `Φ(d_AB + T_t)`, a *shifted* pair sum. Writing the shift
        // into a doubled atom list — the home atoms and a copy displaced by `T_t` — expresses it
        // as an ordinary pair-matrix Ewald, so the strain derivatives (which differ per
        // dimensionality and are the delicate part) stay in one validated place. The blocks that
        // do not correspond to a real interaction carry zero coefficients and are skipped.
        let shift = cell.translation(*t);
        let mut doubled = Vec::with_capacity(2 * n);
        doubled.extend_from_slice(positions);
        doubled.extend(positions.iter().map(|p| *p + shift));
        let mut c2 = vec![vec![0.0_f64; 2 * n]; 2 * n];
        for (a, row) in c.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                c2[a][n + b] = 0.5 * v;
                c2[n + b][a] = 0.5 * v;
            }
        }
        let out = pair_matrix(&doubled, &c2);
        for (i, g) in gradient.iter_mut().enumerate() {
            // The displaced copy moves with its home atom.
            *g += out.gradient[i] + out.gradient[n + i];
        }
        virial = virial.plus(&out.virial);
    }

    // The exchange divergence correction, `−½ (Σ_{t,A,B} c_AB(t)) Φ(0)`. `Φ(0)` depends only on
    // the supercell, so this exerts no force — but it does depend on the cell measure and the
    // reciprocal lattice, so it contributes to the stress. Expressing it as one more
    // `ewald_pair_matrix` call with a coefficient matrix supported on a single diagonal entry
    // reuses the validated strain derivatives instead of writing another variant of them.
    let mut correction = vec![vec![0.0_f64; n]; n];
    correction[0][0] = -total;
    let out = crate::pbc::ewald::ewald_pair_matrix(&super_cell, positions, &correction, &ep);
    for (g, v) in gradient.iter_mut().zip(&out.gradient) {
        *g += *v;
    }
    Ok((gradient, virial.plus(&out.virial)))
}

/// Net atomic charges `q_A = Z_A − P_A` from a density matrix.
fn mulliken_charges(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    p: &Matrix,
) -> Result<Vec<f64>> {
    molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(ia, atom)| {
            let off = basis.atom_offset[ia];
            let n = basis.atom_norb[ia];
            let pop: f64 = (0..n).map(|mu| p[(off + mu, off + mu)]).sum();
            Ok(params.element(atom.z)?.core_charge - pop)
        })
        .collect()
}

/// Electronic part of the closed-form gradient at fixed density `p` (dual-number contraction).
pub fn electronic_gradient_fixed_density(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    p: &Matrix,
) -> Result<Vec<Vec3>> {
    Ok(electronic_gradient_generic(
        molecule,
        params,
        basis,
        GradientDensity::Translated(TranslatedDensity::Uniform(p)),
        None,
    )?
    .0)
}

/// The molecular fixed-density gradient, straight off a divide-and-conquer density.
///
/// The point is the **memory order**. Divide and conquer converges an `O(N)` sparse density and
/// the gradient used to call `to_dense()` on it, allocating `N_ao²` — 512 MB for two thousand
/// atoms — at the one step after the linear-scaling SCF had already finished. Nothing about the
/// pair loop needed that: every read it makes is block-local.
///
/// Molecular only, because divide and conquer is. Core–core repulsion and the post-SCF
/// corrections depend on positions alone and are added by the caller.
pub fn electronic_gradient_sparse(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    density: &crate::dandc::SparseDensity,
) -> Result<Vec<Vec3>> {
    Ok(electronic_gradient_generic(
        molecule,
        params,
        basis,
        GradientDensity::Sparse(density),
        None,
    )?
    .0)
}

/// One density element, read without materializing an `N_ao × N_ao` matrix.
///
/// Every density read in the pair loop is **block-local**: `P(A,A)`, `P(B,B)` or `P(A,B)` with
/// offsets inside those blocks. So a sparse block store answers all of them, and a divide-and-
/// conquer gradient no longer has to densify a density whose whole point is that it is `O(N)`.
///
/// The dense arm indexes the matrix exactly as before, so nothing on that path moves.
#[derive(Clone, Copy)]
enum PairDensity<'a> {
    Dense(&'a Matrix),
    Sparse(&'a crate::dandc::SparseDensity),
}

impl PairDensity<'_> {
    #[inline]
    fn get(&self, basis: &Basis, i: usize, j: usize) -> f64 {
        match self {
            Self::Dense(m) => m[(i, j)],
            Self::Sparse(s) => {
                let (a, b) = (basis.aos[i].atom, basis.aos[j].atom);
                let block = s.block(a, b);
                if block.is_empty() {
                    // Outside the buffer the two atoms share no density block. That is not an
                    // approximation introduced here: it is the same zero the divide-and-conquer
                    // SCF already converged with, and densifying it produced an explicit zero in
                    // an `N_ao²` array.
                    0.0
                } else {
                    block.get(i - basis.atom_offset[a], j - basis.atom_offset[b])
                }
            }
        }
    }
}

/// What the pair loop reads its density from.
///
/// `Translated` carries the ordinary dense (and, for a k mesh, translation-resolved) density.
/// `Sparse` carries a divide-and-conquer density directly. Divide and conquer is molecular, so the
/// sparse arm needs no translation index — there is only `T = 0`.
#[derive(Clone, Copy)]
enum GradientDensity<'a> {
    Translated(TranslatedDensity<'a>),
    Sparse(&'a crate::dandc::SparseDensity),
}

impl<'a> GradientDensity<'a> {
    fn onsite(&self) -> PairDensity<'a> {
        match self {
            Self::Translated(d) => PairDensity::Dense(d.onsite()),
            Self::Sparse(s) => PairDensity::Sparse(s),
        }
    }

    fn at(&self, t: [i32; 3]) -> PairDensity<'a> {
        match self {
            Self::Translated(d) => PairDensity::Dense(d.at(t)),
            Self::Sparse(s) => PairDensity::Sparse(s),
        }
    }
}

/// Electronic gradient **and virial** at fixed density, over periodic images.
///
/// The virial `Σ (∂E/∂d) ⊗ d` is the exact strain derivative for free, because every term in the
/// NDDO energy depends on the nuclei only through the pair displacements. That is also why the
/// stress needs no Pulay term: the NDDO basis is orthonormal and carries no dependence on the
/// cell.
pub fn electronic_gradient_periodic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    density: TranslatedDensity<'_>,
    pbc: &crate::pbc::PbcOptions,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    electronic_gradient_generic(
        molecule,
        params,
        basis,
        GradientDensity::Translated(density),
        Some(pbc),
    )
}

fn electronic_gradient_generic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    density: GradientDensity<'_>,
    pbc: Option<&crate::pbc::PbcOptions>,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let p = density.onsite();
    use crate::pbc::{PairList, PbcMode};
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));

    // The same pair enumeration and the same filtering as the core/Fock build, so the gradient
    // differentiates exactly the energy that was computed rather than a nearby one.
    let cutoff = pbc.map(|o| o.short_range_cutoff).unwrap_or(f64::INFINITY);
    let list = PairList::cached(molecule, cutoff);
    let subtract_monopole = matches!(pbc.map(|o| o.mode), Some(PbcMode::Ewald));

    // Parallelize the expensive per-pair dual-integral build; scatter the small contributions
    // serially in pair order (bit-identical to the serial accumulation).
    let contribs: Result<Vec<(usize, usize, [f64; 3], Vec3)>> = list
        .pairs
        .par_iter()
        .map(|pair| {
            let (u, v) = (pair.a, pair.b);
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            // Heavier atom (more AOs) first, matching the SCF core build. Swapping the atoms
            // reverses the displacement — and the translation with it, since the density block
            // between the swapped pair is `P(−T)`.
            let (a, b, d, t) = if eu.n_orb >= ev.n_orb {
                (u, v, pair.d, pair.t)
            } else {
                (v, u, pair.d * -1.0, [-pair.t[0], -pair.t[1], -pair.t[2]])
            };
            // Resonance and exchange see the density block at *this* translation; on-site
            // populations and Coulomb see `P(0)`.
            let pt = density.at(t);
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let (te, s) = if pbc.is_some() && pair.r > crate::pbc::FEATHER_RANGE_BOHR {
                (
                    crate::integrals::point_charge_pair_dual(ea, eb, d),
                    [[crate::dual::Dual::constant(0.0); 9]; 9],
                )
            } else {
                pair_dual(ea, eb, crate::math::Vec3::zero(), d, has_any_d)?
            };
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
            // Derivative of the point-charge monopole `PM7_EV/r`, subtracted wherever the Ewald
            // sum re-supplies it: d(1/r)/dd = −d/r³.
            let dv: [f64; 3] = if subtract_monopole {
                let c = -crate::constants::PM7_EV / (pair.r * pair.r * pair.r);
                [d.x * c, d.y * c, d.z * c]
            } else {
                [0.0; 3]
            };
            // The three blocks this pair needs, copied out once. Two reasons rather than one:
            // it is what lets a sparse density serve the loop at all, and indexing a small local
            // array beats walking an `N_ao × N_ao` matrix with a stride of `N_ao` in the innermost
            // of four nested loops.
            let mut paa = [0.0_f64; 81];
            let mut pbb = [0.0_f64; 81];
            let mut pab = [0.0_f64; 81];
            for i in 0..na {
                for j in 0..na {
                    paa[i * na + j] = p.get(basis, oa + i, oa + j);
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    pbb[k * nb + l] = p.get(basis, ob + k, ob + l);
                }
            }
            for i in 0..na {
                for j in 0..nb {
                    pab[i * nb + j] = pt.get(basis, oa + i, ob + j);
                }
            }
            let pop_a: f64 = (0..na).map(|k| paa[k * na + k]).sum();
            let pop_b: f64 = (0..nb).map(|k| pbb[k * nb + k]).sum();

            let mut f = [0.0_f64; 3];
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[ob + j].orb);
                    let coef = pab[i * nb + j] * (bi + bj);
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * s[i][j].d[ax];
                    }
                }
            }
            for i in 0..na {
                for j in 0..na {
                    let coef = paa[i * na + j];
                    for (ax, fx) in f.iter_mut().enumerate() {
                        *fx += coef * te.e1b[i][j].d[ax];
                    }
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    let coef = pbb[k * nb + l];
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
                            let coul = paa[mu * na + nu] * pbb[la * nb + si];
                            let exch = -0.5 * pab[mu * nb + la] * pab[nu * nb + si];
                            let coef = coul + exch;
                            for (ax, fx) in f.iter_mut().enumerate() {
                                *fx += coef * dw[ax];
                            }
                        }
                    }
                }
            }
            if subtract_monopole {
                // Differentiate the monopole removed from e1b, e2a, and the two-electron block.
                // The `δ_μν δ_λσ` of the monopole picks the diagonal orbital pairs out of both
                // the Coulomb (`pop_a·pop_b`) and the exchange (`−½ Σ P(μ,λ)²`) contractions.
                let exch_mono: f64 = (0..na)
                    .flat_map(|mu| (0..nb).map(move |la| (mu, la)))
                    .map(|(mu, la)| {
                        let x = pab[mu * nb + la];
                        -0.5 * x * x
                    })
                    .sum();
                let c =
                    -(pop_a * pop_b + exch_mono) + eb.core_charge * pop_a + ea.core_charge * pop_b;
                for (ax, fx) in f.iter_mut().enumerate() {
                    *fx += c * dv[ax];
                }
            }
            for fx in f.iter_mut() {
                *fx *= pair.weight;
            }
            Ok((a, b, f, d))
        })
        .collect();

    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = crate::math::Mat3::zero();
    for (a, b, f, d) in contribs? {
        let force = Vec3::new(f[0], f[1], f[2]);
        gradient[b] += force;
        gradient[a] -= force;
        virial = virial.plus(&crate::math::Mat3::outer(force, d));
    }
    Ok((gradient, virial))
}

/// Total closed-form UHF gradient (core-core + spin-resolved electronic) at the converged
/// open-shell density. `Pα = (P_tot + S)/2`, `Pβ = (P_tot − S)/2` are reconstructed from the
/// total density and the spin density `S = Pα − Pβ`. Hellmann–Feynman (orthonormal basis).
pub fn fixed_density_gradient_uhf(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf: &Pm7Result,
) -> Result<Vec<Vec3>> {
    fixed_density_gradient_uhf_with(molecule, params, scf, &Pm7Options::default()).map(|r| r.0)
}

/// [`fixed_density_gradient_uhf`] with explicit options, and the virial alongside.
///
/// Open-shell periodic systems are handled on exactly the same terms as closed-shell ones: the
/// image enumeration, the monopole subtraction, the Ewald-regularized long-range exchange, and the
/// long-range Ewald force are all spin-independent — only the exchange contraction differs, and
/// it uses the same-spin densities here instead of half the total.
pub fn fixed_density_gradient_uhf_with(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf: &Pm7Result,
    options: &Pm7Options,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let basis = Basis::build(molecule, params)?;
    let pbc = options.pbc_for(molecule);
    let divisions = pbc
        .as_ref()
        .map(|p| p.kmesh.divisions())
        .unwrap_or([1, 1, 1]);

    // A k-point run carries `P(T)`; everything else carries one matrix that stands for every
    // translation. Both go down the same path from here.
    let bloch = scf.bloch_density.as_ref().map(|total| {
        let (a, b) = bloch_spin_densities(total, scf.bloch_spin_density.as_ref());
        (total, a, b)
    });
    let flat = if bloch.is_none() {
        let pt = &scf.density;
        let spin = scf.spin_density.as_ref().ok_or_else(|| {
            crate::error::Pm7Error::InvalidInput("UHF gradient requires a spin density".into())
        })?;
        let mut pa = pt.clone();
        let mut pb = pt.clone();
        let n = pt.as_slice().len();
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..n {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
        Some((pa, pb))
    } else {
        None
    };
    let (total, pa, pb) = match (&bloch, &flat) {
        (Some((t, a, b)), _) => (
            TranslatedDensity::Bloch(t, divisions),
            TranslatedDensity::Bloch(a, divisions),
            TranslatedDensity::Bloch(b, divisions),
        ),
        (None, Some((a, b))) => (
            TranslatedDensity::Uniform(&scf.density),
            TranslatedDensity::Uniform(a),
            TranslatedDensity::Uniform(b),
        ),
        (None, None) => unreachable!("one representation is always built"),
    };

    let (mut gradient, mut virial) = match &pbc {
        None => (
            crate::repulsion::core_core_gradient(molecule, params)?,
            crate::math::Mat3::zero(),
        ),
        Some(p) => crate::repulsion::core_core_gradient_periodic(molecule, params, p)?,
    };
    let (elec, elec_virial) =
        electronic_gradient_spin_generic(molecule, params, &basis, total, pa, pb, pbc.as_ref())?;
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    virial = virial.plus(&elec_virial);
    if let Some(p) = &pbc {
        if p.mode == crate::pbc::PbcMode::Ewald {
            let charges = mulliken_charges(molecule, params, &basis, total.onsite())?;
            let (lr, lr_virial) = long_range_derivatives(molecule, &basis, &charges, &[pa, pb], p)?;
            for (g, e) in gradient.iter_mut().zip(&lr) {
                *g += *e;
            }
            virial = virial.plus(&lr_virial);
        }
    }
    // The external field is spin-independent and couples only to the total Mulliken charge, so
    // the unrestricted term is identical to the restricted one.
    if let Some(f) = options.active_field() {
        let charges = mulliken_charges(molecule, params, &basis, total.onsite())?;
        for (g, e) in gradient.iter_mut().zip(f.gradient(&charges)) {
            *g += e;
        }
    }
    Ok((gradient, virial))
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
    Ok(electronic_gradient_spin_generic(
        molecule,
        params,
        basis,
        TranslatedDensity::Uniform(pt),
        TranslatedDensity::Uniform(pa),
        TranslatedDensity::Uniform(pb),
        None,
    )?
    .0)
}

fn electronic_gradient_spin_generic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    total: TranslatedDensity<'_>,
    alpha: TranslatedDensity<'_>,
    beta: TranslatedDensity<'_>,
    pbc: Option<&crate::pbc::PbcOptions>,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    let pt = total.onsite();
    use crate::pbc::{PairList, PbcMode};
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));
    let cutoff = pbc.map(|o| o.short_range_cutoff).unwrap_or(f64::INFINITY);
    let list = PairList::cached(molecule, cutoff);
    let subtract_monopole = matches!(pbc.map(|o| o.mode), Some(PbcMode::Ewald));
    let contribs: Result<Vec<(usize, usize, [f64; 3], Vec3)>> = list
        .pairs
        .par_iter()
        .map(|pair| {
            let (u, v) = (pair.a, pair.b);
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            let (a, b, d, t) = if eu.n_orb >= ev.n_orb {
                (u, v, pair.d, pair.t)
            } else {
                (v, u, pair.d * -1.0, [-pair.t[0], -pair.t[1], -pair.t[2]])
            };
            // Resonance and exchange see the blocks at *this* translation.
            let (ptt, pa, pb) = (total.at(t), alpha.at(t), beta.at(t));
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let (te, s) = if pbc.is_some() && pair.r > crate::pbc::FEATHER_RANGE_BOHR {
                (
                    crate::integrals::point_charge_pair_dual(ea, eb, d),
                    [[crate::dual::Dual::constant(0.0); 9]; 9],
                )
            } else {
                pair_dual(ea, eb, Vec3::zero(), d, has_any_d)?
            };
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
            let dv: [f64; 3] = if subtract_monopole {
                let c = -crate::constants::PM7_EV / (pair.r * pair.r * pair.r);
                [d.x * c, d.y * c, d.z * c]
            } else {
                [0.0; 3]
            };
            let pop_a: f64 = (0..na).map(|k| pt[(oa + k, oa + k)]).sum();
            let pop_b: f64 = (0..nb).map(|k| pt[(ob + k, ob + k)]).sum();

            let mut f = [0.0_f64; 3];
            // Resonance β·S (total density).
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[ob + j].orb);
                    let coef = ptt[(oa + i, ob + j)] * (bi + bj);
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
            if subtract_monopole {
                let exch_mono: f64 = (0..na)
                    .flat_map(|mu| (0..nb).map(move |la| (mu, la)))
                    .map(|(mu, la)| {
                        let (x, y) = (pa[(oa + mu, ob + la)], pb[(oa + mu, ob + la)]);
                        -(x * x + y * y)
                    })
                    .sum();
                let c =
                    -(pop_a * pop_b + exch_mono) + eb.core_charge * pop_a + ea.core_charge * pop_b;
                for (ax, fx) in f.iter_mut().enumerate() {
                    *fx += c * dv[ax];
                }
            }
            for fx in f.iter_mut() {
                *fx *= pair.weight;
            }
            Ok((a, b, f, d))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = crate::math::Mat3::zero();
    for (a, b, f, d) in contribs? {
        let force = Vec3::new(f[0], f[1], f[2]);
        gradient[b] += force;
        gradient[a] -= force;
        virial = virial.plus(&crate::math::Mat3::outer(force, d));
    }
    Ok((gradient, virial))
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
