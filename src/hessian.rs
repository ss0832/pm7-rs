// SPDX-License-Identifier: GPL-3.0-or-later

//! Nuclear Hessian and harmonic vibrational analysis.
//!
//! [`analytic_hessian`] is the primary path: an AD/CPHF RHF Hessian combining a
//! closed-form skeleton second derivative (second-order forward-AD, [`crate::dual2::Dual2`],
//! of the two-center integral kernels) with the CPHF orbital-relaxation response — no finite
//! differences except for the documented singular-frame pair fallback. [`numerical_hessian`]
//! (central differences of the analytic gradient, `3N`
//! columns in parallel on rayon) is retained as the independent validation reference. Both the
//! restricted and the unrestricted analytic paths are fully analytic; nothing here falls back to
//! finite differences except the documented singular-frame pair.
//!
//! Mass-weighting, **projection of the rigid-body subspace** ([`crate::projection`]) and
//! diagonalization (faer) give harmonic frequencies. Through v0.2.2 this module removed nothing and
//! claimed here that translations and rotations "appear as the ~6 (5 for linear molecules)
//! near-zero modes"; the translations do, by a theorem, and the rotations do not — at a geometry
//! that is not a stationary point they carry real curvature, and water at its experimental geometry
//! put three of them at 71, 121 and 181 cm⁻¹. They are removed by symmetry now, never by size.

use crate::data_tables::MASS;
use crate::dual::Scalar;
use crate::error::Result;
use crate::gradient::closed_form_gradient;
use crate::linalg::{symmetric_eigen, Matrix, Side};
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::projection::{Projection, RigidSubspace};
use crate::scf::Pm7Options;
use crate::system::Molecule;

/// `sqrt(eV / (Å²·amu))` → cm⁻¹ (standard vibrational conversion; 1 unit = 521.47 cm⁻¹).
pub const SQRT_EV_PER_ANG2_AMU_TO_CM: f64 = 521.470_9;

/// Resonance β for orbital index `orb` (0 = s, 1..3 = p, 4..8 = d).
#[inline]
pub(crate) fn beta_of(elem: &crate::params::Pm7Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

/// True when the molecule contains a d-bearing atom (uses the MNDO/d kernels).
fn molecule_has_d(molecule: &Molecule, params: &Pm7Parameters) -> bool {
    molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false))
}

/// Second-order (Dual2) two-electron integrals + 9×9 overlap for an ordered pair
/// (ea = first), seeded on `R_b − R_a`. MNDO/d kernel when `has_any_d`, else sp (padded).
#[allow(clippy::type_complexity)]
fn pair_dual2(
    ea: &crate::params::Pm7Element,
    eb: &crate::params::Pm7Element,
    pa: Vec3,
    pb: Vec3,
    has_any_d: bool,
) -> Result<(
    crate::integrals::PairTwoElecG<crate::dual2::Dual2>,
    [[crate::dual2::Dual2; 9]; 9],
    [crate::dual2::Dual2; 3],
)> {
    pair_dual2_at(ea, eb, pb - pa, has_any_d)
}

/// [`pair_dual2`] seeded on a **displacement** rather than a pair of positions.
///
/// The periodic Hessian works in displacements — an image pair's separation is
/// `R_b + T − R_a`, which is not the difference of two atoms' stored positions.
#[allow(clippy::type_complexity)]
pub(crate) fn pair_dual2_at(
    ea: &crate::params::Pm7Element,
    eb: &crate::params::Pm7Element,
    d: Vec3,
    has_any_d: bool,
) -> Result<(
    crate::integrals::PairTwoElecG<crate::dual2::Dual2>,
    [[crate::dual2::Dual2; 9]; 9],
    [crate::dual2::Dual2; 3],
)> {
    use crate::dual2::Dual2;
    let (pa, pb) = (Vec3::zero(), d);
    let dvec = [
        Dual2::var(pb.x - pa.x, 0),
        Dual2::var(pb.y - pa.y, 1),
        Dual2::var(pb.z - pa.z, 2),
    ];
    // Bond exactly on the local-frame rotation's singular axis (sp: +x, d: ±z): the analytic
    // second derivative of the two-center integrals collapses there, so finite-difference the
    // f64 integrals for that rare pair (`dvec` itself is exact and kept as the seeded vars).
    let d = pb - pa;
    if crate::rotfix::near_frame_singularity(d, has_any_d) {
        let (te, s) = crate::rotfix::pair_dual2_fd(ea, eb, d, has_any_d);
        return Ok((te, s, dvec));
    }
    let te = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
    let mut s = [[Dual2::constant(0.0); 9]; 9];
    if has_any_d {
        s = crate::overlap_d::diat_overlap::<Dual2>(ea, eb, dvec);
    } else {
        let s4 = crate::overlap::diatom_overlap_dual2(ea, pa, eb, pb)?;
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = s4[i][j];
            }
        }
    }
    Ok((te, s, dvec))
}

#[derive(Clone, Debug)]
pub struct VibrationalModes {
    /// Cartesian Hessian (eV/Bohr²), symmetric, size `3N × 3N`.
    pub hessian: Matrix,
    /// Harmonic frequencies (cm⁻¹), ascending; negative = imaginary (saddle/unconverged).
    pub frequencies_cm: Vec<f64>,
    /// Mass-weighted eigenvalues (eV/(Å²·amu)).
    pub eigenvalues: Vec<f64>,
    /// Mass-weighted normal modes as the **columns** of a `3N × 3N` matrix, in the same ascending
    /// order as `frequencies_cm`. Eigenvectors of `H_ij / sqrt(m_i m_j)`, orthonormal.
    ///
    /// These used to be discarded. Nothing downstream could then say *which way* a mode moves,
    /// which is exactly what an IR intensity needs.
    pub modes: Matrix,
    /// The same modes as Cartesian displacements, `m_i^{-1/2} L_in`, each column renormalized to
    /// unit Euclidean length. This is MOPAC's `cnorml` (`force.F90:425-441`).
    pub cartesian_modes: Matrix,
    /// The rigid-body subspace taken out before diagonalizing. `None` only under
    /// [`Projection::None`].
    pub removed: Option<RemovedSubspace>,
}

/// What the projection removed, kept rather than discarded.
///
/// The numbers in here are the ones v0.2.2 printed as part of the spectrum. They have not stopped
/// existing; they have stopped being *frequencies*, and become the diagnostic they always were.
#[derive(Clone, Debug)]
pub struct RemovedSubspace {
    /// The generators, and the geometry they were derived from.
    pub subspace: crate::projection::RigidSubspace,
    /// `<v|H_mw|v>` per generator, eV/(Å²·amu).
    ///
    /// Exactly zero for the translations at **any** geometry — translational invariance is
    /// unconditional, so this is the molecular acoustic sum rule and a non-zero value here is a
    /// defect in the Hessian, not a property of the structure. Non-zero for the rotations exactly
    /// to the extent that the geometry is not a stationary point: `<r|H|r> = sum_A g_A . d_A_perp`.
    pub curvature: Vec<f64>,
    /// The same numbers as signed cm⁻¹ — literally what v0.2.2 reported in place of these modes.
    pub frequencies_cm: Vec<f64>,
    /// Where the removed modes were re-inserted as exact zeros. Empty for a molecule, where they
    /// are dropped instead; see [`vibrational_modes_projected`] for why the two differ.
    pub zero_indices: Vec<usize>,
}

/// Cartesian Hessian (eV/Bohr²) by central differences of the analytic gradient.
pub fn numerical_hessian(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // Every finite-difference column owns two sequential gradient/SCF calculations. Bound the
    // number of columns running concurrently from the same calculation-time budget used by the
    // SCF guard; otherwise `rayon_threads × SCF_peak` can OOM even when one SCF fits.
    let basis = crate::basis::Basis::build(molecule, params)?;
    let n_pairs = nat * nat.saturating_sub(1) / 2;
    let has_d = molecule_has_d(molecule, params) || options.force_dpath;
    let per_column = crate::memory::estimate_peak_bytes(basis.nao, n_pairs, nat, has_d, false);
    let workers = crate::memory::parallel_task_limit(per_column, ndof, options.max_memory_mb);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("pm7-num-hess-{index}"))
        .build()
        .map_err(|error| {
            crate::error::Pm7Error::InvalidInput(format!(
                "failed to create numerical-Hessian worker pool: {error}"
            ))
        })?;

    // Column j = (grad(+step·e_j) − grad(−step·e_j)) / (2·step); columns are independent.
    let columns: Vec<Result<Vec<f64>>> = pool.install(|| {
        (0..ndof)
            .into_par_iter()
            .map(|j| {
                let (atom, k) = (j / 3, j % 3);
                let mut plus = molecule.clone();
                let mut minus = molecule.clone();
                displace(&mut plus.atoms[atom].position, k, step);
                displace(&mut minus.atoms[atom].position, k, -step);
                // Retain only the gradient between the two displaced calculations.  Keeping the
                // full first SCF state alive while solving the second displacement can almost
                // double a worker's peak memory on large systems.
                let gp = closed_form_gradient(&plus, params, options)?.gradient;
                let gm = closed_form_gradient(&minus, params, options)?.gradient;
                let mut col = vec![0.0; ndof];
                for a in 0..nat {
                    for c in 0..3 {
                        let idx = 3 * a + c;
                        col[idx] = (component(&gp[a], c) - component(&gm[a], c)) / (2.0 * step);
                    }
                }
                Ok(col)
            })
            .collect()
    });

    let mut h = Matrix::zeros(ndof, ndof);
    for (j, col) in columns.into_iter().enumerate() {
        let col = col?;
        for (i, &v) in col.iter().enumerate() {
            h[(i, j)] = v;
        }
    }
    // Symmetrize (FD asymmetry).
    let mut hs = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            hs[(i, j)] = 0.5 * (h[(i, j)] + h[(j, i)]);
        }
    }
    Ok(hs)
}

/// Harmonic vibrational analysis at the given geometry (should be a stationary point).
///
/// Returns `3N − 6` frequencies for a non-linear molecule; see [`vibrational_analysis_projected`]
/// for the raw `3N` set.
pub fn vibrational_analysis(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<VibrationalModes> {
    vibrational_analysis_projected(molecule, params, options, step, Projection::default())
}

/// [`vibrational_analysis`], choosing what to project out.
pub fn vibrational_analysis_projected(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
    projection: Projection,
) -> Result<VibrationalModes> {
    // CPHF analytic Hessian (no SCF re-runs); UHF falls back to finite differences internally.
    let hessian = analytic_hessian(molecule, params, options, step)?;
    vibrational_modes_projected(molecule, hessian, projection)
}

/// Mass-weight, project out the rigid-body subspace, diagonalize, and convert an already-computed
/// Cartesian Hessian.
///
/// Split out from [`vibrational_analysis`] so [`crate::ir`] can reuse the modes of the Hessian it
/// has already solved rather than solving a second one.
///
/// Uses [`Projection::Rigid`], which is what a harmonic spectrum means; see
/// [`vibrational_modes_projected`] for the other options and for why a molecule and a cell return
/// different lengths.
pub fn vibrational_modes_from(molecule: &Molecule, hessian: Matrix) -> Result<VibrationalModes> {
    vibrational_modes_projected(molecule, hessian, Projection::default())
}

/// `sqrt(lambda)` in cm⁻¹, negative for an imaginary mode. A sign test, which is exact.
fn frequency_of(lambda: f64) -> f64 {
    if lambda >= 0.0 {
        SQRT_EV_PER_ANG2_AMU_TO_CM * lambda.sqrt()
    } else {
        -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lambda).sqrt()
    }
}

/// As [`vibrational_modes_from`], with the rigid-body subspace to remove named explicitly.
///
/// # What comes back, and why it differs between a molecule and a cell
///
/// **A molecule returns `3N − k` modes.** There is no continuous parameter along which `k` changes,
/// so a length that depends on the geometry is a contract, not a hazard, and the six numbers a
/// caller would have had to slice off are exactly the ones that are not frequencies.
///
/// **A periodic cell returns `3N`, with the removed slots as the literal `0.0`.** A phonon spectrum
/// is a function of `q`, and at `q != 0` there is nothing to remove — so dropping at the zone
/// centre would make the array length depend on the wavevector, and every branch index, dispersion
/// plot and density of states downstream would have to special-case it. The zeros are produced *by
/// construction*: the generators are re-inserted at eigenvalue `0.0` after a diagonalization they
/// were never part of. Nothing is clamped, and no small number is rounded down.
///
/// Either way [`VibrationalModes::removed`] carries what was taken out.
pub fn vibrational_modes_projected(
    molecule: &Molecule,
    hessian: Matrix,
    projection: Projection,
) -> Result<VibrationalModes> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // Mass-weight: H'_ij = H_ij / sqrt(m_i m_j), converting eV/Bohr² → eV/Å².
    let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mass_of = |dof: usize| MASS[molecule.atoms[dof / 3].z as usize];
    let mut mw = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let mij = (mass_of(i) * mass_of(j)).sqrt();
            mw[(i, j)] = hessian[(i, j)] * a0_sq / mij; // eV/(Å²·amu)
        }
    }

    let rigid = RigidSubspace::of(molecule, projection)?;
    let removed_dim = rigid.dimension();
    // Diagonalize inside the complement. `Bᵀ (H B)` rather than `(Bᵀ H) B` so the larger product
    // runs first and the second is `M × 3N` against `3N × M`; `gemm` transposes through a view, so
    // neither transpose is materialized.
    let basis = rigid.complement();
    let reduced = basis.gemm(
        Side::Transposed,
        &mw.gemm(Side::Normal, &basis, Side::Normal, true),
        Side::Normal,
        true,
    );
    let (mut eigs, reduced_modes) = symmetric_eigen(&reduced)?;
    let mut modes = basis.gemm(Side::Normal, &reduced_modes, Side::Normal, true);

    let curvature = rigid.removed_curvature(&mw);
    let mut zero_indices = Vec::new();
    if removed_dim > 0 && molecule.cell.is_some() {
        // Re-insert the generators at exactly zero, keeping the ascending order. A removed mode
        // sorts below every positive eigenvalue and above every imaginary one, which is where a
        // zero belongs.
        let insert_at = eigs.iter().take_while(|&&lam| lam < 0.0).count();
        let mut merged = Matrix::zeros(ndof, ndof);
        let mut column = 0;
        let mut source = 0;
        for slot in 0..ndof {
            let take_generator = slot >= insert_at && column < removed_dim;
            if take_generator {
                for row in 0..ndof {
                    merged[(row, slot)] = rigid.generators[(row, column)];
                }
                zero_indices.push(slot);
                column += 1;
            } else {
                for row in 0..ndof {
                    merged[(row, slot)] = modes[(row, source)];
                }
                source += 1;
            }
        }
        for (offset, slot) in zero_indices.iter().enumerate() {
            let _ = offset;
            eigs.insert(*slot, 0.0);
        }
        modes = merged;
    }

    let frequencies_cm: Vec<f64> = eigs.iter().map(|&lam| frequency_of(lam)).collect();

    // Cartesian displacements: undo the mass weighting, then renormalize each column. MOPAC
    // renormalizes explicitly for the same reason (`force.F90:425-430`) — the mass-weighted
    // eigenvector is unit length, its Cartesian image is not.
    let n_modes = modes.cols;
    let mut cartesian_modes = Matrix::zeros(ndof, n_modes);
    for column in 0..n_modes {
        let mut norm = 0.0;
        for row in 0..ndof {
            let v = modes[(row, column)] / mass_of(row).sqrt();
            cartesian_modes[(row, column)] = v;
            norm += v * v;
        }
        let scale = if norm > 0.0 { 1.0 / norm.sqrt() } else { 0.0 };
        for row in 0..ndof {
            cartesian_modes[(row, column)] *= scale;
        }
    }

    let removed = (projection != Projection::None).then(|| RemovedSubspace {
        frequencies_cm: curvature.iter().map(|&c| frequency_of(c)).collect(),
        curvature,
        zero_indices,
        subspace: rigid,
    });

    Ok(VibrationalModes {
        hessian,
        frequencies_cm,
        eigenvalues: eigs,
        modes,
        cartesian_modes,
        removed,
    })
}

/// The CPHF **first-order orbital response**, retained rather than discarded.
///
/// `u[t]` is the occupied–virtual block for Cartesian degree of freedom `t = 3A + k`, an
/// `n_vir × n_occ` matrix in the MO basis of [`Self::mo_coeff`], in units of 1/Bohr:
///
/// ```text
/// ∂c_i/∂R_t = Σ_a u[t][(a, i)] · c_{n_occ + a}
/// ```
///
/// **Retaining this costs no arithmetic at all.** The Hessian already solves for it and then
/// throws it away; what the opt-in bounds is *memory*, `3N · n_vir · n_occ · 8` bytes (about
/// 98 MB for the 102-atom benchmark), plus the cost of handing it across a language boundary.
#[derive(Clone, Debug)]
pub struct OrbitalResponse {
    /// One `n_vir × n_occ` block per Cartesian degree of freedom. The α channel for an
    /// unrestricted run.
    pub u: Vec<Matrix>,
    pub u_beta: Option<Vec<Matrix>>,
    /// The MO basis `u` is expressed against.
    pub mo_coeff: Matrix,
    pub mo_coeff_beta: Option<Matrix>,
    pub n_occ: usize,
    pub n_occ_beta: Option<usize>,
}

impl OrbitalResponse {
    /// The AO-basis density derivative `∂P/∂R_t` (`nao × nao`), spin-summed.
    ///
    /// `O(nao² · n_occ)` through matrix products rather than the `O(n_occ · n_vir · nao²)`
    /// outer-product loop, and formed one degree of freedom at a time so nothing materializes a
    /// `3N × nao²` intermediate.
    pub fn density_derivative(&self, t: usize) -> Matrix {
        let nvir = self.mo_coeff.rows - self.n_occ;
        let cv = submatrix_cols(&self.mo_coeff, self.n_occ, nvir);
        let co = submatrix_cols(&self.mo_coeff, 0, self.n_occ);
        // Weight 2 spin-sums a restricted response; an unrestricted one adds the β channel below
        // at weight 1 each.
        let weight = if self.u_beta.is_some() { 1.0 } else { 2.0 };
        let mut out = ao_response_density_w(&self.u[t], &cv, &co, weight);
        if let (Some(ub), Some(cb), Some(occupied)) =
            (&self.u_beta, &self.mo_coeff_beta, self.n_occ_beta)
        {
            let nvb = cb.rows - occupied;
            let cvb = submatrix_cols(cb, occupied, nvb);
            let cob = submatrix_cols(cb, 0, occupied);
            let beta = ao_response_density_w(&ub[t], &cvb, &cob, 1.0);
            for (dst, src) in out.as_mut_slice().iter_mut().zip(beta.as_slice()) {
                *dst += *src;
            }
        }
        out
    }

    /// Number of Cartesian degrees of freedom.
    pub fn len(&self) -> usize {
        self.u.len()
    }

    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }
}

/// What to keep from an analytic Hessian besides the Hessian itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct HessianRequest {
    /// Retain the CPHF orbital response. **Storage only** — the solve happens either way.
    pub response: bool,
}

impl HessianRequest {
    pub fn with_response() -> Self {
        Self { response: true }
    }
}

/// An analytic Hessian plus whatever [`HessianRequest`] asked to keep alongside it.
#[derive(Clone, Debug)]
pub struct HessianResult {
    /// Cartesian Hessian in eV/Bohr² — exactly what [`analytic_hessian`] returns.
    pub hessian: Matrix,
    /// The converged SCF at this geometry, returned so the caller need not run it again.
    pub scf: crate::scf::Pm7Result,
    /// Present only when requested. Always `None` for a periodic system, whose response lives in
    /// a different (translation-resolved) object.
    pub response: Option<OrbitalResponse>,
}

/// **Analytic (CPHF) Cartesian Hessian** (eV/Bohr²).
///
/// A thin wrapper over [`analytic_hessian_with`] that keeps nothing but the Hessian.
///
/// `H_ab = E^(2,skel)_ab + Σ_μν F^a_μν (∂P/∂R_b)_μν`, where:
///
/// * the **skeleton** (fixed-density) second derivative `E^(2,skel)` is computed in **closed
///   form** by second-order forward-mode automatic differentiation ([`crate::dual2::Dual2`])
///   of the two-center integral kernels — resonance `β·S`, electron–core attraction, the
///   Dewar–Sabelli–Klopman two-electron integrals, and the PM7 core–core repulsion — with **no
///   finite differences**; and
/// * the density response `∂P/∂R_b` solves the coupled-perturbed (CPHF) equations, whose kernel
///   is the **orbital Hessian** (the same object a second-order SOSCF would use). This is done
///   entirely in the compact MO occupied–virtual subspace (`H_relax[a][b] = 4 G^a·U^b`), so the
///   working set is `O(ndof · n_occ · n_vir)` — no dense `ndof × nao²` derivative-Fock or
///   response intermediates are ever materialized (memory-lean and rayon-parallel over DOFs).
///
/// Fully analytic for both closed-shell RHF and open-shell UHF (the latter via the internal `analytic_hessian_uhf`
/// with coupled α/β CPHF), across all valence shells (`n ≤ 3` analytic kernel, `n ≥ 4` via AD
/// through the numerical overlap quadrature). No finite differences.
pub fn analytic_hessian(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<Matrix> {
    analytic_hessian_with(molecule, params, options, step, &HessianRequest::default())
        .map(|out| out.hessian)
}

/// The zone-centre Cartesian Hessian from the perturbation solver, with its ground state.
///
/// `Φ(q = 0)` is `D(q)` at `q = 0`, so this is one `dynamical_matrix_dfpt` call and a real-part
/// extraction — but the extraction *measures* the imaginary part rather than dropping it, which is
/// the only reason it is worth a function of its own.
fn zone_centre_from_response(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
) -> Result<HessianResult> {
    let result = crate::dfpt::dynamical_matrix_dfpt(
        molecule,
        params,
        options,
        [0.0; 3],
        &crate::dfpt::DfptOptions::default(),
    )?;
    let n = result.force_constants.n;
    let mut hessian = Matrix::zeros(n, n);
    let mut worst_imaginary = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (re, im) = result.force_constants.get(i, j);
            hessian[(i, j)] = re;
            worst_imaginary = worst_imaginary.max(im.abs());
            scale = scale.max(re.abs());
        }
    }
    // `Φ(q = 0) = Σ_T Φ(T)` is a sum of real blocks with unit phases, so it is real by
    // construction. Measure that rather than dropping the imaginary part: a phase error would
    // otherwise be silently discarded here, which is the one place it could hide.
    let allowed = 1.0e-9 * scale.max(1.0);
    if worst_imaginary > allowed {
        return Err(crate::error::Pm7Error::ResponseFailed(format!(
            "the zone-centre force constants came back with an imaginary part of \
             {worst_imaginary:.3e} against a real scale of {scale:.3e}. At q = 0 every Bloch phase \
             is 1 and Φ is a sum of real blocks, so this is a defect in the construction rather \
             than rounding."
        )));
    }
    Ok(HessianResult {
        hessian,
        scf: result.scf,
        response: None,
    })
}

/// [`analytic_hessian`], optionally keeping the CPHF orbital response.
///
/// The response is what IR intensities are built from, and it is free: the same solve produces
/// both, so asking for a Hessian and a spectrum separately would pay for the CPHF twice.
pub fn analytic_hessian_with(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
    request: &HessianRequest,
) -> Result<HessianResult> {
    use crate::dual2::Dual2;

    if molecule.is_periodic() {
        // Zone-centre (q = 0) force constants. The Γ-point restriction is on the *sampling*, not
        // on the cell: run the same call on a supercell to resolve `Φ(0A, TB)` and hence `D(q)`
        // at every commensurate `q`. See `crate::hessian_pbc`.
        let pbc = options.pbc_for(molecule).expect("periodic");
        // **A real k mesh has only one arm**, so nothing here needs a ground state to choose it —
        // and `dynamical_matrix_dfpt` runs its own and now hands it back. Through 0.2.2 an SCF ran
        // *before* this branch, purely to fill `HessianResult::scf` and to test `unrestricted`,
        // and then the perturbation solver ran the whole thing again: two full k-mesh SCFs for one
        // Hessian, the more expensive half of the calculation done twice. The options the solver
        // uses are the caller's own whenever the mesh is not Γ, so the state it returns is the one
        // that used to be computed here, to the last bit.
        //
        // A k mesh makes the response couple `k` with `k + q`, which is exactly what the
        // perturbation solver does; at `q = 0` its force constants **are** the k-point zone-centre
        // Hessian, so refusing here (as v0.2.1 did, pointing at `numerical_hessian`) was refusing
        // to make one call.
        if pbc.kmesh.divisions() != [1, 1, 1] {
            return zone_centre_from_response(molecule, params, options);
        }
        // A Γ mesh has two arms, and which one applies is a question the SCF has already answered:
        // `analytic_hessian_periodic` has no unrestricted CPHF at all. Re-deriving "is this cell
        // unrestricted" here from `reference` and the electron count would be a second copy of that
        // rule, in a place nothing would notice had drifted — so the SCF is run and asked.
        let scf = crate::scf::run_pm7(molecule, params, options)?;
        // The perturbation solver has carried a band set per spin since 0.2.2, so at `q = 0` on a Γ
        // mesh it produces exactly the matrix that refusal was standing in front of. A second UCPHF
        // written to reach the same number would be two implementations to keep in step, and a Γ
        // mesh is not a special case of the solver — it is the ordinary path with one k point.
        //
        // This is the one place a second SCF is still spent, and it is not the same SCF twice:
        // `KMesh::Gamma` does not build the translation-resolved Hamiltonian the response needs, so
        // the solver promotes the sampling to `grid(1, 1, 1)` and converges a different object.
        if scf.unrestricted {
            return zone_centre_from_response(molecule, params, options);
        }
        let _ = step;
        let hessian =
            crate::hessian_pbc::analytic_hessian_periodic(molecule, params, options, &scf)?;
        return Ok(HessianResult {
            hessian,
            scf,
            response: None,
        });
    }
    let scf = crate::scf::run_pm7(molecule, params, options)?;
    // Pre-flight OOM guard for the CPHF ov-block stacks (O(n_atoms · n_basis²)).
    {
        let nat = molecule.atoms.len();
        let basis = crate::basis::Basis::build(molecule, params)?;
        crate::memory::guard(
            basis.nao,
            nat * nat.saturating_sub(1) / 2,
            nat,
            molecule_has_d(molecule, params),
            true,
            options.max_memory_mb,
        )?;
    }
    if scf.unrestricted {
        let _ = step; // UHF path is fully analytic (no finite-difference step)
        return analytic_hessian_uhf(molecule, params, options, &scf, request);
    }

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_field(
        molecule,
        &basis,
        params,
        false,
        options.active_field(),
    )?;
    let p = scf.density.clone();
    let c = scf.mo_coeff.clone();
    let eps = scf.mo_energies.clone();
    let n_occ = scf.n_occ;

    // 1) Skeleton (fixed-density) second derivative — fully analytic via second-order AD
    //    (Dual2) of each two-center pair's energy contribution E_pair(R_ab). Since E_pair
    //    depends only on the displacement R_ab = R_b − R_a, its 3×3 Hessian block scatters as
    //    +H onto the (a,a) and (b,b) diagonal blocks and −H onto the (a,b)/(b,a) blocks.
    let mut hess = Matrix::zeros(ndof, ndof);
    let skeleton_timer = crate::profile::stage("hessian: skeleton (Dual2 pairs)");
    let has_any_d = molecule_has_d(molecule, params);
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let blocks: Vec<Result<(usize, usize, [[f64; 3]; 3])>> = {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(u, v)| -> Result<(usize, usize, [[f64; 3]; 3])> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (pa, pb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let (te, s, dvec) = pair_dual2(ea, eb, pa, pb, has_any_d)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                // Resonance β·S energy (both μν and νμ orderings → factor (β_i+β_j)).
                for i in 0..na {
                    let bi = beta_of(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = beta_of(eb, basis.aos[ob + j].orb);
                        let coef = p[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                // Electron–core attraction.
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * p[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * p[(ob + k, ob + l)];
                    }
                }
                // Two-electron Coulomb (J) + exchange (K), fixed density.
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                                let exch = -0.5 * p[(oa + mu, ob + la)] * p[(oa + nu, ob + si)];
                                epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                            }
                        }
                    }
                }
                // Core–core repulsion (function of |R_ab|).
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                        params,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    drop(skeleton_timer);
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 2+3) Orbital-relaxation (CPHF) term in the compact MO occupied–virtual subspace:
    //   H_relax[a][b] = 4 Σ_{ov} G^a_{ov} U^b_{ov},
    // where G^t is the skeleton derivative Fock projected to the occ–virt block (n_vir × n_occ)
    // and U^b solves the coupled-perturbed equations against the orbital Hessian. Keeping
    // everything in the n_vir × n_occ block (never ndof × nao²) makes this both fast and
    // memory-lean: the response density is formed by matrix products (O(nao²·n_occ)), not the
    // O(n_occ·n_vir·nao²) outer-product loop, and no 3N full Fock/response matrices are stored.
    let mut retained: Option<OrbitalResponse> = None;
    let nvir = basis.nao - n_occ;
    if nvir > 0 && n_occ > 0 {
        let cv = submatrix_cols(&c, n_occ, nvir); // virtual MOs, nao × n_vir
        let co = submatrix_cols(&c, 0, n_occ); // occupied MOs, nao × n_occ
        let denom = ov_denominators(&eps, n_occ, nvir); // ε_i − ε_a, n_vir × n_occ

        // Skeleton derivative Fock ov-blocks, built one atom at a time (peak memory O(nao²)).
        let gov = {
            let _t = crate::profile::stage("hessian: skeleton derivative Fock");
            skeleton_fock_ov(
                molecule,
                params,
                &basis,
                &p,
                &cv,
                &co,
                options.active_field(),
            )?
        };

        // CPHF response ov-blocks. Solved for all degrees of freedom together, so each sweep makes
        // **one** pass over the atom-pair integrals instead of `3N` of them — see
        // `cphf_ov_batched`.
        let uov: Vec<Matrix> = {
            let _t = crate::profile::stage("hessian: CPHF solve");
            use rayon::prelude::*;
            gov.par_iter()
                .map(|g| {
                    cphf_ov(
                        g,
                        &denom,
                        &cv,
                        &co,
                        molecule,
                        params,
                        &basis,
                        &core,
                        options.exchange_cutoff,
                        options.cphf_max_iterations,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Assemble H_relax[a][b] = 4 G^a : U^b (parallel over rows; no extra dense storage).
        let _assemble = crate::profile::stage("hessian: G:U assembly");
        use rayon::prelude::*;
        let rows: Vec<Vec<f64>> = (0..ndof)
            .into_par_iter()
            .map(|a| {
                (0..ndof)
                    .map(|b| 4.0 * gov[a].frobenius_dot(&uov[b]))
                    .collect()
            })
            .collect();
        for (a, row) in rows.into_iter().enumerate() {
            for (b, v) in row.into_iter().enumerate() {
                hess[(a, b)] += v;
            }
        }

        // Keep the response only if asked. The contraction above has already used it, so this is
        // purely about whether the caller wants to hold `3N · n_vir · n_occ` floats alive.
        if request.response {
            retained = Some(OrbitalResponse {
                u: uov,
                u_beta: None,
                mo_coeff: c.clone(),
                mo_coeff_beta: None,
                n_occ,
                n_occ_beta: None,
            });
        }
    }

    // Post-SCF correction (dispersion + PM7-HH) second derivative.
    crate::gradient::add_correction_hessian(molecule, options, &mut hess);

    // Symmetrize.
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(HessianResult {
        hessian: sym,
        scf,
        response: retained,
    })
}

/// Copy `count` columns of `c` starting at `start` into a fresh `nao × count` matrix.
pub(crate) fn submatrix_cols(c: &Matrix, start: usize, count: usize) -> Matrix {
    let nao = c.rows;
    let mut m = Matrix::zeros(nao, count);
    for mu in 0..nao {
        for k in 0..count {
            m[(mu, k)] = c[(mu, start + k)];
        }
    }
    m
}

/// Orbital-energy denominators `ε_i − ε_a` (occupied `i`, virtual `a`), as an `n_vir × n_occ`
/// matrix — the diagonal of the uncoupled orbital Hessian.
pub(crate) fn ov_denominators(eps: &[f64], n_occ: usize, nvir: usize) -> Matrix {
    let mut d = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            d[(a, i)] = eps[i] - eps[n_occ + a];
        }
    }
    d
}

/// Project an AO-basis matrix `f` onto the MO occupied–virtual block `Cvᵀ F Co` (n_vir × n_occ).
pub(crate) fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    // Sequential GEMMs: `project_ov` always runs inside an outer rayon loop (`skeleton_fock_ov`
    // over atoms, `cphf_ov` over DOF), so a parallel one would only oversubscribe. `Cvᵀ` is a
    // transposed *view*, not a copy — this used to allocate and fill an `nao × n_vir` matrix on
    // every one of several thousand calls.
    let m = f.gemm(Side::Normal, co, Side::Normal, false); // nao × n_occ
    cv.gemm(Side::Transposed, &m, Side::Normal, false) // n_vir × n_occ
}

/// Skeleton derivative Fock, projected to the MO occ–virt block, one entry per Cartesian DOF.
///
/// Built **one atom at a time**: for atom `c` its three axis-derivative Fock matrices are
/// accumulated from the pairs `{c, x}` and immediately projected to the compact `n_vir × n_occ`
/// block, so peak memory is `O(nao²)` (a few transient matrices per thread) rather than
/// `O(ndof · nao²)`. Each pair's dual integrals are evaluated twice overall (once per endpoint),
/// a negligible cost next to the CPHF solve.
fn skeleton_fock_ov(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &crate::basis::Basis,
    p: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    field: Option<&crate::field::ExternalField>,
) -> Result<Vec<Matrix>> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let has_any_d = molecule_has_d(molecule, params);

    // Per atom: the three projected ov-blocks (x, y, z).
    let per_atom: Vec<Result<[Matrix; 3]>> = (0..nat)
        .into_par_iter()
        .map(|c| -> Result<[Matrix; 3]> {
            let mut fmat = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            for x in 0..nat {
                if x == c {
                    continue;
                }
                let (u, v) = (c.min(x), c.max(x));
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (pa, pb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                // E_pair depends on R_ab = R_b − R_a; ∂/∂R_c = +∂/∂R_ab if c==b, else −.
                let sign = if c == b { 1.0 } else { -1.0 };
                let (te, s) = crate::gradient::pair_dual(ea, eb, pa, pb, has_any_d)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                for axis in 0..3 {
                    let fm = &mut fmat[axis];
                    // Resonance β·S.
                    for i in 0..na {
                        let bi = beta_of(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = beta_of(eb, basis.aos[ob + j].orb);
                            let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                            fm[(oa + i, ob + j)] += val;
                            fm[(ob + j, oa + i)] += val;
                        }
                    }
                    // Electron–core attraction.
                    for i in 0..na {
                        for j in 0..na {
                            fm[(oa + i, oa + j)] += sign * te.e1b[i][j].d[axis];
                        }
                    }
                    for k in 0..nb {
                        for l in 0..nb {
                            fm[(ob + k, ob + l)] += sign * te.e2a[k][l].d[axis];
                        }
                    }
                    // Two-electron Coulomb (J).
                    for mu in 0..na {
                        for nu in 0..na {
                            let mut acc = 0.0;
                            for la in 0..nb {
                                for si in 0..nb {
                                    acc += p[(ob + la, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            fm[(oa + mu, oa + nu)] += sign * acc;
                        }
                    }
                    for la in 0..nb {
                        for si in 0..nb {
                            let mut acc = 0.0;
                            for mu in 0..na {
                                for nu in 0..na {
                                    acc += p[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            fm[(ob + la, ob + si)] += sign * acc;
                        }
                    }
                    // Two-electron exchange (K).
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acc = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    acc += p[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            let val = sign * (-0.5 * acc);
                            fm[(oa + mu, ob + la)] += val;
                            fm[(ob + la, oa + mu)] += val;
                        }
                    }
                }
            }
            // The external field's contribution to `∂h/∂R_c`. It sits outside the neighbour loop
            // because it is a **one-centre** term: only atom `c`'s own diagonal block moves, and
            // the `sign = ±1` bookkeeping that a pair term needs does not apply to it.
            //
            // This is the *entire* field contribution to the Hessian. The skeleton second
            // derivative vanishes identically, because `E_field(R; P) = Σ_A q_A(P) (f·R_A)` is
            // linear in `R` at fixed density, and the s–p hybrid term has no coordinate
            // dependence at all (`dd` is a parameter).
            if let Some(f) = field {
                for (axis, fm) in fmat.iter_mut().enumerate() {
                    let value = -f.internal().get(axis);
                    let off = basis.atom_offset[c];
                    for mu in 0..basis.atom_norb[c] {
                        fm[(off + mu, off + mu)] += value;
                    }
                }
            }
            Ok([
                project_ov(&fmat[0], cv, co),
                project_ov(&fmat[1], cv, co),
                project_ov(&fmat[2], cv, co),
            ])
        })
        .collect();

    let mut gov: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let [gx, gy, gz] = res?;
        gov.push(gx);
        gov.push(gy);
        gov.push(gz);
    }
    Ok(gov)
}

/// AO-basis first-order density response from the MO occ–virt response coefficients `u`
/// (n_vir × n_occ): `R = Cv (w·U) Coᵀ + Co (w·U)ᵀ Cvᵀ`, built by matrix products
/// (O(nao²·n_occ)). The occupation weight `w` is 2 for RHF (spin-summed) and 1 for a single UHF
/// spin channel.
pub(crate) fn ao_response_density_w(u: &Matrix, cv: &Matrix, co: &Matrix, weight: f64) -> Matrix {
    let mut uw = u.clone();
    for x in uw.as_mut_slice() {
        *x *= weight;
    }
    // Sequential GEMMs: this runs inside the per-DOF CPHF `par_iter`, which already saturates the
    // cores, so a nested parallel one would only add overhead. `Coᵀ` is a view, not a copy.
    let a = cv.gemm(Side::Normal, &uw, Side::Normal, false); // nao × n_occ
    let b = a.gemm(Side::Normal, co, Side::Transposed, false); // nao × nao
    let bt = b.transpose();
    let mut r = b;
    for (rv, tv) in r.as_mut_slice().iter_mut().zip(bt.as_slice()) {
        *rv += *tv;
    }
    r
}

/// RHF response density (occupation weight 2).
pub(crate) fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    ao_response_density_w(u, cv, co, 2.0)
}

/// Solve the CPHF equations for one perturbation entirely in the MO occ-virt block: iterate
/// `U = (G_skel + [G(dP(U))]_ov) / (eps_i - eps_a)` to self-consistency. `G(dP) = F(dP) - H_core`
/// is the two-electron response Fock (the orbital-Hessian coupling); the fixed point is the
/// coupled response. Returns the converged `U` (n_vir x n_occ).
///
/// Solving all `3N` systems **together**, so that one pass over the atom-pair integrals serves
/// every degree of freedom, was implemented and **measured to be slower**: 5.2 s to 8.8 s on the
/// 102-atom Hessian. The premise was that the pair loop is memory bound, and at this size it is
/// not — 102 atoms is about 4 MB of integral blocks, which sits in L3 across calls, so batching
/// saved no traffic and cost the per-DOF parallelism this `par_iter` gets for free. It would be
/// the right shape for a system whose integrals exceed cache, but a dense CPHF is out of reach
/// there for other reasons. `docs/performance.md` records the numbers.
#[allow(clippy::too_many_arguments)]
fn cphf_ov(
    g_ov: &Matrix,
    denom: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
    exchange_cutoff: Option<(f64, f64)>,
    max_iterations: usize,
) -> Result<Matrix> {
    // The molecular response kernel: `F` is linear in the density, so `F(dP) - H_core` is `G(dP)`
    // exactly. (A *periodic* Fock is only affine - see `hessian_pbc::relaxation_hessian`.)
    let kernel = |r: &Matrix| -> Result<Matrix> {
        let mut g = crate::fock::build_fock_x(molecule, basis, params, core, r, exchange_cutoff)?;
        for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        Ok(g)
    };
    cphf_ov_with_kernel(g_ov, denom, cv, co, &kernel, max_iterations)
}
/// The CPHF fixed-point solver, over an arbitrary two-electron response kernel `G(ΔP)`.
///
/// Iterates `U = (G_skel + [G(ΔP(U))]_ov) / (ε_i − ε_a)` to self-consistency and returns the
/// converged `U` (n_vir × n_occ). Splitting the kernel out is what lets the molecular and
/// periodic Hessians share one solver: they differ only in how `G` is formed.
pub(crate) fn cphf_ov_with_kernel(
    g_ov: &Matrix,
    denom: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    kernel: &dyn Fn(&Matrix) -> Result<Matrix>,
    max_iterations: usize,
) -> Result<Matrix> {
    let context = CphfContext {
        g_ov,
        denom,
        cv,
        co,
        kernel,
        max_iterations: cphf_budget(max_iterations)?,
    };
    context.solve()
}

/// Everything one CPHF solve needs, so the conjugate-gradient and fixed-point solvers can share it.
struct CphfContext<'a> {
    g_ov: &'a Matrix,
    /// `ε_i − ε_a`, so it is **negative** for every occupied–virtual pair.
    denom: &'a Matrix,
    cv: &'a Matrix,
    co: &'a Matrix,
    kernel: &'a dyn Fn(&Matrix) -> Result<Matrix>,
    /// Operator applications this solve may spend, shared by both solvers below —
    /// [`crate::scf::Pm7Options::cphf_max_iterations`].
    max_iterations: usize,
}

/// Below this the orbital pair is treated as degenerate and dropped.
///
/// A rotation between degenerate orbitals is undetermined *and* does not change the density, so
/// dropping it removes nothing physical — but leaving it in divides by a number near zero.
const CPHF_DEGENERACY_FLOOR: f64 = 1.0e-10;

/// Convergence threshold on `‖M⁻¹ r‖`. See [`CphfContext::solve`] for why that measure.
const CPHF_TOLERANCE: f64 = 1.0e-9;

/// Refuse a response that did not converge, naming the numbers.
///
/// Through v0.2.1 every solver here ran out its iteration budget and returned the last iterate as
/// `Ok`, with nothing recording that it was not a solution. A Hessian assembled from an unconverged
/// `U` is not approximately right — the relaxation term is `4 G:U`, linear in the error — and it
/// came back looking like every other Hessian. Frequencies built on it are wrong by an amount
/// nobody could see.
///
/// A free function rather than a method because the restricted and unrestricted solvers do not
/// share a loop, and having refused through two different messages is how the unrestricted one kept
/// the old behaviour three releases longer than the restricted one.
fn refuse_unconverged(residual: f64, iterations: usize) -> crate::error::Pm7Error {
    crate::error::Pm7Error::ResponseFailed(format!(
        "the CPHF did not converge after {iterations} operator applications: residual \
          {residual:.3e} against a tolerance of {CPHF_TOLERANCE:.1e}. A near-degenerate frontier \
          pair makes the orbital Hessian ill-conditioned, so the solve converges slowly rather \
          than not at all, and more applications is the first thing to try: raise \
          `--cphf-max-iterations` on either command line, or `cphf_max_iterations` in Python and \
          on `Pm7Options`. It costs iterations and moves nothing else, since the tolerance is \
          unchanged. Expect to need a lot more rather than a little, and expect the residual to \
          wander on the way — measured on cubic SrTiO3 (a 0.28 eV PM7 gap), `phonons --supercell \
          2 2 2` gives 6e-9 at the default 100, 9e-7 at 200, 1e-7 at 400 and 2e-9 at 800, and \
          converges at 1600. Tightening `scf_tolerance` also helps, by improving the conditioning. \
          For a periodic cell `dfpt` is a different solver on the same response and is often much \
          cheaper: it converges that SrTiO3 zone centre to 8e-11 in fourteen iterations."
    ))
}

/// A CPHF budget of zero is a caller error, not a solve that fails instantly.
///
/// Every interface that carries the budget validates it, so this is the backstop for a Rust caller
/// who built `Pm7Options` by hand.
fn cphf_budget(max_iterations: usize) -> Result<usize> {
    if max_iterations == 0 {
        return Err(crate::error::Pm7Error::InvalidInput(
            "cphf_max_iterations must be at least 1: the orbital response is solved iteratively, \
             and a budget of zero asks for no iterations at all rather than for a cheap answer."
                .into(),
        ));
    }
    Ok(max_iterations)
}

fn frobenius_dot(a: &Matrix, b: &Matrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum()
}

impl CphfContext<'_> {
    /// `M⁻¹ x`: divide by `ε_a − ε_i`, with degenerate pairs zeroed.
    fn precondition(&self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for (v, d) in out.as_mut_slice().iter_mut().zip(self.denom.as_slice()) {
            // `denom` is `ε_i − ε_a`; the preconditioner wants `ε_a − ε_i`.
            *v = if d.abs() < CPHF_DEGENERACY_FLOOR {
                0.0
            } else {
                *v / -*d
            };
        }
        out
    }

    /// The orbital Hessian `A U = (ε_a − ε_i) U + [G(ΔP(U))]_ov`.
    fn apply(&self, u: &Matrix) -> Result<Matrix> {
        let rho = {
            let _t = crate::profile::stage("cphf: response density (AO)");
            ao_response_density(u, self.cv, self.co)
        };
        let g = {
            let _t = crate::profile::stage("cphf: response Fock");
            (self.kernel)(&rho)?
        };
        let mut out = {
            let _t = crate::profile::stage("cphf: project to ov");
            project_ov(&g, self.cv, self.co)
        };
        for ((v, uu), d) in out
            .as_mut_slice()
            .iter_mut()
            .zip(u.as_slice())
            .zip(self.denom.as_slice())
        {
            *v += -*d * *uu;
        }
        Ok(out)
    }

    /// Solve `A U = −G_skel` for the coupled first-order orbital response.
    ///
    /// # Why conjugate gradient
    ///
    /// The CPHF is a **linear** system, and at a stable closed-shell SCF solution its operator
    /// `A U = (ε_a − ε_i) U + [G(ΔP(U))]_ov` is symmetric and positive definite: `G` is a
    /// symmetric linear map on symmetric densities, and the diagonal part is positive because the
    /// orbitals are aufbau-filled. That is exactly the hypothesis conjugate gradient needs, and CG
    /// on an SPD system converges in a number of steps governed by `√κ` rather than by the
    /// spectral radius of a fixed-point map. The preconditioner is the same energy-denominator
    /// division the fixed point already applied.
    ///
    /// The measure is `‖M⁻¹ r‖`, and that choice is not cosmetic: with `r = −G_skel − A U` and
    /// `M = ε_a − ε_i`, `M⁻¹ r` **is** the fixed-point step `f(U) − U` that the previous solver
    /// tested, algebraically and not merely in spirit. So the tolerance means the same thing it
    /// always did, the two solvers are directly comparable iteration for iteration, and nothing
    /// downstream needed retuning.
    ///
    /// # Why the fixed point is still here
    ///
    /// `p·Ap ≤ 0` says the operator is not positive definite on this system — a saddle point, or
    /// an SCF solution that is not a minimum. CG has no meaning there and would take a step in a
    /// direction that increases the residual. The DIIS fixed point does not care, so the solve
    /// falls back to it rather than failing. The test is written `!(pap > 0.0)` so that a `NaN`
    /// takes the fallback too.
    fn solve(&self) -> Result<Matrix> {
        // The uncoupled response, which is also where the fixed point started.
        let rhs = {
            let mut b = self.g_ov.clone();
            for v in b.as_mut_slice() {
                *v = -*v;
            }
            b
        };
        let mut u = self.precondition(&rhs);
        let mut residual = {
            let au = self.apply(&u)?;
            let mut r = rhs.clone();
            for (rv, av) in r.as_mut_slice().iter_mut().zip(au.as_slice()) {
                *rv -= *av;
            }
            r
        };
        let mut z = self.precondition(&residual);
        let mut direction = z.clone();
        let mut rz = frobenius_dot(&residual, &z);
        let mut spent = 1usize;

        while spent < self.max_iterations {
            if norm(&z) < CPHF_TOLERANCE {
                return Ok(u);
            }
            let ad = self.apply(&direction)?;
            spent += 1;
            let dad = frobenius_dot(&direction, &ad);
            // Deliberately `!(x > 0)`, so a NaN from a diverged kernel also takes the fallback.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(dad > 0.0) {
                return self.solve_fixed_point();
            }
            let alpha = rz / dad;
            for (uv, dv) in u.as_mut_slice().iter_mut().zip(direction.as_slice()) {
                *uv += alpha * *dv;
            }
            for (rv, av) in residual.as_mut_slice().iter_mut().zip(ad.as_slice()) {
                *rv -= alpha * *av;
            }
            z = self.precondition(&residual);
            let rz_next = frobenius_dot(&residual, &z);
            let beta = rz_next / rz;
            rz = rz_next;
            for (dv, zv) in direction.as_mut_slice().iter_mut().zip(z.as_slice()) {
                *dv = *zv + beta * *dv;
            }
        }
        if norm(&z) < CPHF_TOLERANCE {
            return Ok(u);
        }
        // A CG that runs out of budget on an SPD system usually means the preconditioner is far
        // off, not that the answer is nearly there. Try the other solver before giving up: it is
        // the one that used to run here, so this cannot be worse than v0.2.1 was.
        self.solve_fixed_point()
    }

    /// Refuse a response that did not converge, naming the numbers.
    fn refuse_unconverged(&self, residual: f64, iterations: usize) -> crate::error::Pm7Error {
        refuse_unconverged(residual, iterations)
    }

    /// Pulay-DIIS-accelerated fixed point: `U ← (G_skel + [G(ΔP(U))]_ov) / (ε_i − ε_a)`.
    ///
    /// The solver v0.2.1 used, kept as the fallback for a system where the orbital Hessian is not
    /// positive definite. It stores a small history of ov-blocks (8 × n_vir·n_occ, freed per
    /// solve); on a singular DIIS system it falls back to the plain step, so it can never do worse
    /// than the un-accelerated iteration.
    fn solve_fixed_point(&self) -> Result<Matrix> {
        let elem_div = |num: &Matrix| -> Matrix {
            let mut u = num.clone();
            for (uv, dv) in u.as_mut_slice().iter_mut().zip(self.denom.as_slice()) {
                *uv = if dv.abs() < CPHF_DEGENERACY_FLOOR {
                    0.0
                } else {
                    *uv / *dv
                };
            }
            u
        };
        let step = |u: &Matrix| -> Result<Matrix> {
            let rho = {
                let _t = crate::profile::stage("cphf: response density (AO)");
                ao_response_density(u, self.cv, self.co)
            };
            let kr = {
                let _t = crate::profile::stage("cphf: response Fock");
                (self.kernel)(&rho)?
            };
            let g_resp = {
                let _t = crate::profile::stage("cphf: project to ov");
                project_ov(&kr, self.cv, self.co)
            };
            let mut rhs = self.g_ov.clone();
            for (rv, gv) in rhs.as_mut_slice().iter_mut().zip(g_resp.as_slice()) {
                *rv += *gv;
            }
            Ok(elem_div(&rhs))
        };
        let max_diis = 8;
        let mut u = elem_div(self.g_ov);
        let mut images: Vec<Matrix> = Vec::new(); // f(U_i)
        let mut errors: Vec<Matrix> = Vec::new(); // e_i = f(U_i) − U_i
        for _ in 0..self.max_iterations {
            let fu = step(&u)?;
            let mut e = fu.clone();
            for (ev, uv) in e.as_mut_slice().iter_mut().zip(u.as_slice()) {
                *ev -= *uv;
            }
            if norm(&e) < CPHF_TOLERANCE {
                return Ok(fu);
            }
            images.push(fu);
            errors.push(e);
            if images.len() > max_diis {
                images.remove(0);
                errors.remove(0);
            }
            u = cphf_diis(&images, &errors).unwrap_or_else(|| images.last().unwrap().clone());
        }
        let last = errors.last().map(norm).unwrap_or(f64::INFINITY);
        Err(self.refuse_unconverged(last, self.max_iterations))
    }
}

/// Frobenius norm.
fn norm(m: &Matrix) -> f64 {
    m.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Pulay-DIIS extrapolation for the CPHF fixed point: given stored images `f(U_i)` and errors
/// `e_i = f(U_i) − U_i`, solve `min ‖Σ c_i e_i‖` s.t. `Σ c_i = 1` and return `Σ c_i f(U_i)`.
/// `None` if the (near-singular) bordered system can't be solved reasonably → caller uses the
/// plain step.
fn cphf_diis(images: &[Matrix], errors: &[Matrix]) -> Option<Matrix> {
    let n = errors.len();
    if n < 2 {
        return None;
    }
    let dim = n + 1;
    let mut bmat = Matrix::zeros(dim, dim);
    for i in 0..n {
        for j in i..n {
            let v = errors[i].frobenius_dot(&errors[j]);
            bmat[(i, j)] = v;
            bmat[(j, i)] = v;
        }
        bmat[(i, n)] = -1.0;
        bmat[(n, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    let c = crate::linalg::solve_linear(&bmat, &rhs).ok()?;
    if c.iter().take(n).any(|v| !v.is_finite() || v.abs() > 1.0e6) {
        return None;
    }
    let mut u = Matrix::zeros(images[0].rows, images[0].cols);
    for (i, img) in images.iter().enumerate() {
        for (uv, iv) in u.as_mut_slice().iter_mut().zip(img.as_slice()) {
            *uv += c[i] * iv;
        }
    }
    Some(u)
}

/// **Analytic open-shell (UHF) Cartesian Hessian** (eV/Bohr²). Same structure as the RHF path
/// but spin-resolved: the skeleton second derivative uses same-spin exchange
/// `−[Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ]`, and the response solves the **coupled** α/β CPHF equations
/// (the α and β responses are coupled through the total-density Coulomb term). Everything stays
/// in the per-spin MO occ–virt blocks — memory-lean, rayon-parallel. No finite differences.
fn analytic_hessian_uhf(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    scf: &crate::scf::Pm7Result,
    request: &HessianRequest,
) -> Result<HessianResult> {
    use crate::dual2::Dual2;
    use crate::fock::build_fock_spin;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let core = crate::hamiltonian::build_core_field(
        molecule,
        &basis,
        params,
        false,
        options.active_field(),
    )?;

    // Spin densities Pα = (P_tot + S)/2, Pβ = (P_tot − S)/2.
    let pt = scf.density.clone();
    let spin = scf.spin_density.as_ref().ok_or_else(|| {
        crate::error::Pm7Error::InvalidInput("UHF Hessian requires a spin density".into())
    })?;
    let mut pa = pt.clone();
    let mut pb = pt.clone();
    {
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..pts.len() {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
    }
    // Recover both spin orbital sets by diagonalizing the converged spin Fock matrices.
    let fa = build_fock_spin(molecule, &basis, params, &core, &pt, &pa)?;
    let fb = build_fock_spin(molecule, &basis, params, &core, &pt, &pb)?;
    let (eps_a, ca) = symmetric_eigen(&fa)?;
    let (eps_b, cb) = symmetric_eigen(&fb)?;
    let n_alpha = scf.n_occ;
    let n_beta = scf.n_occ - (options.multiplicity - 1);

    // 1) Skeleton (fixed-density) second derivative — spin-resolved exchange.
    let mut hess = Matrix::zeros(ndof, ndof);
    let has_any_d = molecule_has_d(molecule, params);
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let blocks: Vec<Result<(usize, usize, [[f64; 3]; 3])>> = {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(u, v)| -> Result<(usize, usize, [[f64; 3]; 3])> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let (te, s, dvec) = pair_dual2(ea, eb, posa, posb, has_any_d)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                for i in 0..na {
                    let bi = beta_of(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = beta_of(eb, basis.aos[ob + j].orb);
                        let coef = pt[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * pt[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * pt[(ob + k, ob + l)];
                    }
                }
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = pt[(oa + mu, oa + nu)] * pt[(ob + la, ob + si)];
                                // Same-spin exchange: −(Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ).
                                let exch = -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                    + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)]);
                                epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                            }
                        }
                    }
                }
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                        params,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 2+3) Coupled UCPHF relaxation term: H_relax[a][b] = 2 Σ_σ Gσ^a · Uσ^b.
    let mut retained: Option<OrbitalResponse> = None;
    let nva = nao - n_alpha;
    let nvb = nao - n_beta;
    let have_a = nva > 0 && n_alpha > 0;
    let have_b = nvb > 0 && n_beta > 0;
    if have_a || have_b {
        let cva = submatrix_cols(&ca, n_alpha, nva);
        let coa = submatrix_cols(&ca, 0, n_alpha);
        let cvb = submatrix_cols(&cb, n_beta, nvb);
        let cob = submatrix_cols(&cb, 0, n_beta);
        let denom_a = ov_denominators(&eps_a, n_alpha, nva);
        let denom_b = ov_denominators(&eps_b, n_beta, nvb);

        let (gova, govb) = skeleton_fock_ov_spin(
            molecule,
            params,
            &basis,
            &pt,
            &pa,
            &pb,
            &cva,
            &coa,
            &cvb,
            &cob,
            options.active_field(),
        )?;

        let uovs: Vec<(Matrix, Matrix)> = {
            use rayon::prelude::*;
            (0..ndof)
                .into_par_iter()
                .map(|t| {
                    ucphf_ov(
                        &gova[t],
                        &govb[t],
                        &denom_a,
                        &denom_b,
                        &cva,
                        &coa,
                        &cvb,
                        &cob,
                        molecule,
                        params,
                        &basis,
                        &core,
                        options.exchange_cutoff,
                        options.cphf_max_iterations,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        use rayon::prelude::*;
        let rows: Vec<Vec<f64>> = (0..ndof)
            .into_par_iter()
            .map(|a| {
                (0..ndof)
                    .map(|b| {
                        2.0 * (gova[a].frobenius_dot(&uovs[b].0)
                            + govb[a].frobenius_dot(&uovs[b].1))
                    })
                    .collect()
            })
            .collect();
        for (a, row) in rows.into_iter().enumerate() {
            for (b, v) in row.into_iter().enumerate() {
                hess[(a, b)] += v;
            }
        }

        if request.response {
            let (mut ua, mut ub) = (Vec::with_capacity(ndof), Vec::with_capacity(ndof));
            for (a, b) in uovs {
                ua.push(a);
                ub.push(b);
            }
            retained = Some(OrbitalResponse {
                u: ua,
                u_beta: Some(ub),
                mo_coeff: ca.clone(),
                mo_coeff_beta: Some(cb.clone()),
                n_occ: n_alpha,
                n_occ_beta: Some(n_beta),
            });
        }
    }

    // Post-SCF correction (dispersion + PM7-HH) second derivative.
    crate::gradient::add_correction_hessian(molecule, options, &mut hess);

    // Symmetrize.
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(HessianResult {
        hessian: sym,
        scf: scf.clone(),
        response: retained,
    })
}

/// Spin-resolved skeleton derivative Fock ov-blocks: returns `(Gα, Gβ)` per DOF. The resonance,
/// electron–core, and Coulomb `J(P_tot)` parts are spin-independent (shared); the exchange
/// differs — `Kα(Pα)` into the α Fock, `Kβ(Pβ)` into the β Fock. Built one atom at a time and
/// projected to each spin's occ–virt block (peak memory `O(nao²)`).
#[allow(clippy::too_many_arguments)]
fn skeleton_fock_ov_spin(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &crate::basis::Basis,
    pt: &Matrix,
    pa: &Matrix,
    pb: &Matrix,
    cva: &Matrix,
    coa: &Matrix,
    cvb: &Matrix,
    cob: &Matrix,
    field: Option<&crate::field::ExternalField>,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let has_any_d = molecule_has_d(molecule, params);

    let per_atom: Vec<Result<[(Matrix, Matrix); 3]>> = (0..nat)
        .into_par_iter()
        .map(|c| -> Result<[(Matrix, Matrix); 3]> {
            let mut fa = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            let mut fb = [
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
                Matrix::zeros(nao, nao),
            ];
            for x in 0..nat {
                if x == c {
                    continue;
                }
                let (u, v) = (c.min(x), c.max(x));
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let sign = if c == b { 1.0 } else { -1.0 };
                let (te, s) = crate::gradient::pair_dual(ea, eb, posa, posb, has_any_d)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                for axis in 0..3 {
                    // Borrow both spin Fock matrices for this atom (distinct arrays fa/fb).
                    let fma = &mut fa[axis];
                    let fmb = &mut fb[axis];
                    // Shared (spin-independent): resonance, e-core, Coulomb J(P_tot) → into BOTH,
                    // per-pair (do NOT copy the running-accumulated matrix, which would
                    // re-add earlier neighbours' contributions).
                    for i in 0..na {
                        let bi = beta_of(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = beta_of(eb, basis.aos[ob + j].orb);
                            let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                            fma[(oa + i, ob + j)] += val;
                            fma[(ob + j, oa + i)] += val;
                            fmb[(oa + i, ob + j)] += val;
                            fmb[(ob + j, oa + i)] += val;
                        }
                    }
                    for i in 0..na {
                        for j in 0..na {
                            let val = sign * te.e1b[i][j].d[axis];
                            fma[(oa + i, oa + j)] += val;
                            fmb[(oa + i, oa + j)] += val;
                        }
                    }
                    for k in 0..nb {
                        for l in 0..nb {
                            let val = sign * te.e2a[k][l].d[axis];
                            fma[(ob + k, ob + l)] += val;
                            fmb[(ob + k, ob + l)] += val;
                        }
                    }
                    for mu in 0..na {
                        for nu in 0..na {
                            let mut acc = 0.0;
                            for la in 0..nb {
                                for si in 0..nb {
                                    acc +=
                                        pt[(ob + la, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            let val = sign * acc;
                            fma[(oa + mu, oa + nu)] += val;
                            fmb[(oa + mu, oa + nu)] += val;
                        }
                    }
                    for la in 0..nb {
                        for si in 0..nb {
                            let mut acc = 0.0;
                            for mu in 0..na {
                                for nu in 0..na {
                                    acc +=
                                        pt[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            let val = sign * acc;
                            fma[(ob + la, ob + si)] += val;
                            fmb[(ob + la, ob + si)] += val;
                        }
                    }
                    // Same-spin exchange Kσ (coefficient −1): Kα(Pα) → fa, Kβ(Pβ) → fb.
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acca = 0.0;
                            let mut accb = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    let dw = te.two_e(mu, nu, la, si).d[axis];
                                    acca += pa[(oa + nu, ob + si)] * dw;
                                    accb += pb[(oa + nu, ob + si)] * dw;
                                }
                            }
                            let va = sign * (-acca);
                            let vb = sign * (-accb);
                            fma[(oa + mu, ob + la)] += va;
                            fma[(ob + la, oa + mu)] += va;
                            fmb[(oa + mu, ob + la)] += vb;
                            fmb[(ob + la, oa + mu)] += vb;
                        }
                    }
                }
            }
            // The external field, added to **both** spin channels: it is a one-electron,
            // spin-independent operator. See `skeleton_fock_ov` for why this sits outside the
            // neighbour loop and why it is the whole field contribution to the Hessian.
            if let Some(f) = field {
                for axis in 0..3 {
                    let value = -f.internal().get(axis);
                    let off = basis.atom_offset[c];
                    for mu in 0..basis.atom_norb[c] {
                        fa[axis][(off + mu, off + mu)] += value;
                        fb[axis][(off + mu, off + mu)] += value;
                    }
                }
            }
            Ok([
                (project_ov(&fa[0], cva, coa), project_ov(&fb[0], cvb, cob)),
                (project_ov(&fa[1], cva, coa), project_ov(&fb[1], cvb, cob)),
                (project_ov(&fa[2], cva, coa), project_ov(&fb[2], cvb, cob)),
            ])
        })
        .collect();

    let mut gova: Vec<Matrix> = Vec::with_capacity(3 * nat);
    let mut govb: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let arr = res?;
        for (ga, gb) in arr {
            gova.push(ga);
            govb.push(gb);
        }
    }
    Ok((gova, govb))
}

/// Coupled α/β CPHF solve for one perturbation (MO occ–virt blocks). Iterate
/// `Uσ = (Gσ_skel + [J(ΔP_tot) − Kσ(ΔPσ)]_ov) / (εσ_i − εσ_a)` to self-consistency; the α and β
/// channels couple through the total response density `ΔP_tot = ΔPα + ΔPβ` in the Coulomb term.
///
/// **This solver used to return its last iterate as `Ok` when it ran out of budget**, which is the
/// defect `CphfContext::refuse_unconverged` was written to fix on the restricted path and which
/// survived here because the two solvers do not share a loop. A Hessian assembled from an
/// unconverged `U` is not approximately right — the relaxation term is linear in the error — and
/// it comes back looking like every other Hessian. It refuses now, through the same message, and
/// the budget it refuses at is the caller's.
#[allow(clippy::too_many_arguments)]
fn ucphf_ov(
    ga: &Matrix,
    gb: &Matrix,
    denom_a: &Matrix,
    denom_b: &Matrix,
    cva: &Matrix,
    coa: &Matrix,
    cvb: &Matrix,
    cob: &Matrix,
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
    exchange_cutoff: Option<(f64, f64)>,
    max_iterations: usize,
) -> Result<(Matrix, Matrix)> {
    let max_iterations = cphf_budget(max_iterations)?;
    let div = |num: &Matrix, denom: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    // Coupled α/β fixed-point step f(Uα,Uβ) = (div(Gα + Gα_resp), div(Gβ + Gβ_resp)).
    let step = |ua: &Matrix, ub: &Matrix| -> Result<(Matrix, Matrix)> {
        let dpa = ao_response_density_w(ua, cva, coa, 1.0);
        let dpb = ao_response_density_w(ub, cvb, cob, 1.0);
        let mut dpt = dpa.clone();
        for (t, x) in dpt.as_mut_slice().iter_mut().zip(dpb.as_slice()) {
            *t += *x;
        }
        // Response two-electron Focks Gσ(ΔP) = build_fock_spin(ΔP_tot, ΔPσ) − H_core.
        let mut fa_r = crate::fock::build_fock_spin_x(
            molecule,
            basis,
            params,
            core,
            &dpt,
            &dpa,
            exchange_cutoff,
        )?;
        let mut fb_r = crate::fock::build_fock_spin_x(
            molecule,
            basis,
            params,
            core,
            &dpt,
            &dpb,
            exchange_cutoff,
        )?;
        for (xv, hv) in fa_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        for (xv, hv) in fb_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        let ga_resp = project_ov(&fa_r, cva, coa);
        let gb_resp = project_ov(&fb_r, cvb, cob);
        let mut rhs_a = ga.clone();
        for (rv, gv) in rhs_a.as_mut_slice().iter_mut().zip(ga_resp.as_slice()) {
            *rv += *gv;
        }
        let mut rhs_b = gb.clone();
        for (rv, gv) in rhs_b.as_mut_slice().iter_mut().zip(gb_resp.as_slice()) {
            *rv += *gv;
        }
        Ok((div(&rhs_a, denom_a), div(&rhs_b, denom_b)))
    };
    // Pulay-DIIS on the combined (α,β) error (see [`cphf_diis`]); same solution, fewer iterations.
    let max_diis = 8;
    let mut ua = div(ga, denom_a);
    let mut ub = div(gb, denom_b);
    let mut ima: Vec<Matrix> = Vec::new();
    let mut imb: Vec<Matrix> = Vec::new();
    let mut erra: Vec<Matrix> = Vec::new();
    let mut errb: Vec<Matrix> = Vec::new();
    let mut last_error = f64::INFINITY;
    for _ in 0..max_iterations {
        let (fa, fb) = step(&ua, &ub)?;
        let mut ea = fa.clone();
        for (ev, uv) in ea.as_mut_slice().iter_mut().zip(ua.as_slice()) {
            *ev -= *uv;
        }
        let mut eb = fb.clone();
        for (ev, uv) in eb.as_mut_slice().iter_mut().zip(ub.as_slice()) {
            *ev -= *uv;
        }
        // The two channels' errors as one vector, which is also what `cphf_diis2` minimizes — the
        // same measure the restricted solver applies to its single channel.
        let diff = (ea.as_slice().iter().map(|x| x * x).sum::<f64>()
            + eb.as_slice().iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        if diff < CPHF_TOLERANCE {
            return Ok((fa, fb));
        }
        last_error = diff;
        ima.push(fa);
        imb.push(fb);
        erra.push(ea);
        errb.push(eb);
        if ima.len() > max_diis {
            ima.remove(0);
            imb.remove(0);
            erra.remove(0);
            errb.remove(0);
        }
        match cphf_diis2(&ima, &imb, &erra, &errb) {
            Some((na, nb)) => {
                ua = na;
                ub = nb;
            }
            None => {
                ua = ima.last().unwrap().clone();
                ub = imb.last().unwrap().clone();
            }
        }
    }
    Err(refuse_unconverged(last_error, max_iterations))
}

/// Combined-spin Pulay-DIIS for the coupled α/β CPHF: the error metric sums both channels
/// `B_ij = ⟨eα_i|eα_j⟩ + ⟨eβ_i|eβ_j⟩`, and both `Σ c_i fα_i`, `Σ c_i fβ_i` are returned.
#[allow(clippy::type_complexity)]
fn cphf_diis2(
    ima: &[Matrix],
    imb: &[Matrix],
    erra: &[Matrix],
    errb: &[Matrix],
) -> Option<(Matrix, Matrix)> {
    let n = erra.len();
    if n < 2 {
        return None;
    }
    let dim = n + 1;
    let mut bmat = Matrix::zeros(dim, dim);
    for i in 0..n {
        for j in i..n {
            let v = erra[i].frobenius_dot(&erra[j]) + errb[i].frobenius_dot(&errb[j]);
            bmat[(i, j)] = v;
            bmat[(j, i)] = v;
        }
        bmat[(i, n)] = -1.0;
        bmat[(n, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    let c = crate::linalg::solve_linear(&bmat, &rhs).ok()?;
    if c.iter().take(n).any(|v| !v.is_finite() || v.abs() > 1.0e6) {
        return None;
    }
    let mut ua = Matrix::zeros(ima[0].rows, ima[0].cols);
    let mut ub = Matrix::zeros(imb[0].rows, imb[0].cols);
    for i in 0..n {
        for (uv, iv) in ua.as_mut_slice().iter_mut().zip(ima[i].as_slice()) {
            *uv += c[i] * iv;
        }
        for (uv, iv) in ub.as_mut_slice().iter_mut().zip(imb[i].as_slice()) {
            *uv += c[i] * iv;
        }
    }
    Some((ua, ub))
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
fn component(v: &Vec3, k: usize) -> f64 {
    match k {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{optimize, OptOptions};

    /// Conjugate gradient and the fixed point solve the **same** linear system, so they must land
    /// on the same `U`.
    ///
    /// This is what lets CG carry the fixed point as a fallback for an operator that is not
    /// positive definite: whichever branch runs, the answer is the same one, and a Hessian does
    /// not depend on which solver reached it. It has to be a unit test — `CphfContext` is
    /// `pub(crate)`-adjacent and an integration test cannot select a solver, so a test outside
    /// this module can only compare one solver against a finite difference and call that
    /// solver-independence, which it is not.
    #[test]
    fn the_two_cphf_solvers_agree() {
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.958400 0.000000 0.000000\n\
             H -0.239987 0.927846 0.000000\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let options = Pm7Options {
            p_tol: 1.0e-10,
            ..Pm7Options::default()
        };
        let scf = crate::scf::run_pm7(&mol, &params, &options).unwrap();
        let basis = crate::basis::Basis::build(&mol, &params).unwrap();
        let core = crate::hamiltonian::build_core_with(&mol, &basis, &params, options.force_dpath)
            .unwrap();
        let n_occ = scf.n_occ;
        let nvir = basis.nao - n_occ;
        let cv = submatrix_cols(&scf.mo_coeff, n_occ, nvir);
        let co = submatrix_cols(&scf.mo_coeff, 0, n_occ);
        let denom = ov_denominators(&scf.mo_energies, n_occ, nvir);

        // Any right-hand side in the ov block will do; the claim is about the solver, not about
        // which perturbation produced it. A deterministic pattern keeps the test reproducible.
        let mut g_ov = Matrix::zeros(nvir, n_occ);
        for a in 0..nvir {
            for i in 0..n_occ {
                g_ov[(a, i)] = 0.01 * ((a * 7 + i * 3) % 11) as f64 - 0.05;
            }
        }
        let kernel = |r: &Matrix| -> Result<Matrix> {
            let mut g = crate::fock::build_fock_x(&mol, &basis, &params, &core, r, None)?;
            for (xv, hv) in g.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
                *xv -= *hv;
            }
            Ok(g)
        };
        let context = CphfContext {
            g_ov: &g_ov,
            denom: &denom,
            cv: &cv,
            co: &co,
            kernel: &kernel,
            max_iterations: options.cphf_max_iterations,
        };
        let cg = context.solve().unwrap();
        let fixed = context.solve_fixed_point().unwrap();

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for (a, b) in cg.as_slice().iter().zip(fixed.as_slice()) {
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }
        // A trivially zero `U` would satisfy any agreement test, so insist the solve did something.
        assert!(
            scale > 1.0e-4,
            "response is trivially small: scale {scale:.3e}"
        );
        assert!(
            worst < 1.0e-8,
            "the two CPHF solvers disagree by {worst:.3e} on a response of scale {scale:.3e}"
        );
    }

    #[test]
    fn analytic_hessian_matches_numerical() {
        // The CPHF analytic Hessian must match the finite-difference Hessian (FD of the
        // full-SCF gradient) — the independent ground truth.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.97 0.02 0.0\nH -0.25 0.94 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("analytic-vs-numerical Hessian max delta = {max_delta:.2e} eV/Bohr^2");
        assert!(max_delta < 1.0e-3, "Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn d_shell_analytic_hessian_matches_numerical() {
        // H2S (sulfur carries d orbitals): the analytic MNDO/d Hessian (Dual2 skeleton
        // + CPHF orbital relaxation) must match the finite-difference Hessian.
        let mol = Molecule::from_xyz_str(
            "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.9705 0.9430\nH 0.0 -0.9705 0.9430\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let a = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let n = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let mut max_delta = 0.0_f64;
        for i in 0..a.rows {
            for j in 0..a.cols {
                max_delta = max_delta.max((a[(i, j)] - n[(i, j)]).abs());
            }
        }
        eprintln!("d-shell analytic-vs-numerical Hessian max delta = {max_delta:.3e} eV/Bohr^2");
        assert!(
            max_delta < 1.0e-4,
            "d-shell Hessian mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn analytic_hessian_uhf_radical() {
        // Methyl radical (doublet, UHF): the coupled α/β CPHF (UCPHF) analytic Hessian must
        // match the finite-difference Hessian (FD of the analytic UHF gradient).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\nH -0.545 -0.944 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options {
            multiplicity: 2,
            ..Pm7Options::default()
        };
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("UHF analytic-vs-numerical Hessian max delta = {max_delta:.2e}");
        assert!(max_delta < 2.0e-3, "UHF Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn d_shell_uhf_hessian_matches_numerical() {
        // SH radical (doublet): sulfur carries d orbitals AND the system is open
        // shell, exercising the UCPHF MNDO/d Hessian path.
        let mol = Molecule::from_xyz_str("2\nSH\nS 0.0 0.0 0.0\nH 0.0 0.0 1.34\n", 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options {
            multiplicity: 2,
            ..Pm7Options::default()
        };
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let mut max_delta = 0.0_f64;
        for i in 0..ha.rows {
            for j in 0..ha.cols {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("SH UHF d-shell analytic-vs-numerical Hessian max delta = {max_delta:.2e}");
        assert!(
            max_delta < 2.0e-3,
            "SH UHF d Hessian mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn water_vibrations() {
        // Optimize water, then compute harmonic frequencies. Water is bent and non-linear, so
        // `3N − 6 = 3` modes come back: the bend and the two stretches, and nothing else.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options::default();
        let relaxed = optimize(&mol, &params, &opts, &OptOptions::default()).unwrap();
        let vib = vibrational_analysis(&relaxed.molecule, &params, &opts, 1.0e-3).unwrap();
        let freqs = &vib.frequencies_cm;
        eprintln!(
            "H2O frequencies (cm^-1): {:?}",
            freqs.iter().map(|f| f.round()).collect::<Vec<_>>()
        );
        assert_eq!(freqs.len(), 3, "3N - 6 for a bent triatomic: {freqs:?}");
        assert!(freqs[0] > 1200.0 && freqs[0] < 2200.0, "bend {}", freqs[0]);
        assert!(
            freqs[1] > 2000.0 && freqs[2] > 2000.0,
            "stretches {freqs:?}"
        );

        // Through v0.2.2 this test asserted only that the six lowest modes were below 300 cm^-1 --
        // a tolerance wide enough to admit a genuine vibration, and the strongest statement that
        // could honestly be made when nothing was projected. The projection replaces it with two
        // exact ones.
        let removed = vib
            .removed
            .as_ref()
            .expect("the rigid subspace is reported");
        assert_eq!(removed.subspace.dimension(), 6);
        let scale = vib
            .hessian
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));

        // Translational invariance is unconditional: `sum_B H[Aa,Bb] = 0` at any geometry, so the
        // translations carry *identically* zero curvature. A non-zero value here is a defect in the
        // Hessian, not a property of the structure.
        for c in &removed.curvature[..3] {
            assert!(c.abs() < 1.0e-9 * scale, "translation curvature {c}");
        }
        // The rotations only vanish at a stationary point, which is what the optimizer just found:
        // `<r|H|r> = sum_A g_A . d_A_perp`, and the residual gradient is what is left.
        for c in &removed.curvature[3..] {
            assert!(c.abs() < 1.0e-4 * scale, "rotation curvature {c}");
        }
    }
}
