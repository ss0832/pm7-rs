// SPDX-License-Identifier: GPL-3.0-or-later

//! Nuclear Hessian and harmonic vibrational analysis.
//!
//! [`analytic_hessian`] is the primary path: an AD/CPHF RHF Hessian combining a
//! closed-form skeleton second derivative (second-order forward-AD, [`crate::dual2::Dual2`],
//! of the two-center integral kernels) with the CPHF orbital-relaxation response — no finite
//! differences except for the documented singular-frame pair fallback. [`numerical_hessian`]
//! (central differences of the analytic gradient, `3N`
//! columns in parallel on rayon) is retained as the independent validation reference and as
//! the fallback for open-shell / heavy-element systems. Mass-weighting and diagonalization
//! (faer) give harmonic frequencies; overall translations and rotations appear as the ~`6`
//! (5 for linear molecules) near-zero modes.

use crate::data_tables::MASS;
use crate::dual::Scalar;
use crate::error::Result;
use crate::gradient::closed_form_gradient;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::scf::Pm7Options;
use crate::system::Molecule;

/// `sqrt(eV / (Å²·amu))` → cm⁻¹ (standard vibrational conversion; 1 unit = 521.47 cm⁻¹).
pub const SQRT_EV_PER_ANG2_AMU_TO_CM: f64 = 521.470_9;

/// Resonance β for orbital index `orb` (0 = s, 1..3 = p, 4..8 = d).
#[inline]
fn beta_of(elem: &crate::params::Pm7Element, orb: u8) -> f64 {
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
    use crate::dual2::Dual2;
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
pub fn vibrational_analysis(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    step: f64,
) -> Result<VibrationalModes> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    // CPHF analytic Hessian (no SCF re-runs); UHF falls back to finite differences internally.
    let hessian = analytic_hessian(molecule, params, options, step)?;

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
    let (eigs, _vecs) = symmetric_eigen(&mw)?;
    let frequencies_cm: Vec<f64> = eigs
        .iter()
        .map(|&lam| {
            if lam >= 0.0 {
                SQRT_EV_PER_ANG2_AMU_TO_CM * lam.sqrt()
            } else {
                -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lam).sqrt()
            }
        })
        .collect();

    Ok(VibrationalModes {
        hessian,
        frequencies_cm,
        eigenvalues: eigs,
    })
}

/// **Analytic (CPHF) Cartesian Hessian** (eV/Bohr²).
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
    use crate::dual2::Dual2;

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
        return analytic_hessian_uhf(molecule, params, options, &scf);
    }

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core(molecule, &basis, params)?;
    let p = scf.density.clone();
    let c = scf.mo_coeff.clone();
    let eps = scf.mo_energies.clone();
    let n_occ = scf.n_occ;

    // 1) Skeleton (fixed-density) second derivative — fully analytic via second-order AD
    //    (Dual2) of each two-center pair's energy contribution E_pair(R_ab). Since E_pair
    //    depends only on the displacement R_ab = R_b − R_a, its 3×3 Hessian block scatters as
    //    +H onto the (a,a) and (b,b) diagonal blocks and −H onto the (a,b)/(b,a) blocks.
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
    let nvir = basis.nao - n_occ;
    if nvir > 0 && n_occ > 0 {
        let cv = submatrix_cols(&c, n_occ, nvir); // virtual MOs, nao × n_vir
        let co = submatrix_cols(&c, 0, n_occ); // occupied MOs, nao × n_occ
        let denom = ov_denominators(&eps, n_occ, nvir); // ε_i − ε_a, n_vir × n_occ

        // Skeleton derivative Fock ov-blocks, built one atom at a time (peak memory O(nao²)).
        let gov = skeleton_fock_ov(molecule, params, &basis, &p, &cv, &co)?;

        // CPHF response ov-blocks — independent per DOF, solved in parallel.
        let uov: Vec<Matrix> = {
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
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Assemble H_relax[a][b] = 4 G^a : U^b (parallel over rows; no extra dense storage).
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
    Ok(sym)
}

/// Copy `count` columns of `c` starting at `start` into a fresh `nao × count` matrix.
fn submatrix_cols(c: &Matrix, start: usize, count: usize) -> Matrix {
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
fn ov_denominators(eps: &[f64], n_occ: usize, nvir: usize) -> Matrix {
    let mut d = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            d[(a, i)] = eps[i] - eps[n_occ + a];
        }
    }
    d
}

/// Project an AO-basis matrix `f` onto the MO occupied–virtual block `Cvᵀ F Co` (n_vir × n_occ).
fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    // Sequential matmuls: project_ov always runs inside an outer rayon loop
    // (skeleton_fock_ov over atoms, cphf_ov over DOF), so a parallel matmul would oversubscribe.
    let m = f.matmul_seq(co); // nao × n_occ
    cv.transpose().matmul_seq(&m) // n_vir × n_occ
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
fn ao_response_density_w(u: &Matrix, cv: &Matrix, co: &Matrix, weight: f64) -> Matrix {
    let mut uw = u.clone();
    for x in uw.as_mut_slice() {
        *x *= weight;
    }
    // Sequential matmuls: this runs inside the per-DOF CPHF par_iter, which already saturates
    // the cores, so a nested parallel matmul would only add overhead.
    let a = cv.matmul_seq(&uw); // nao × n_occ
    let b = a.matmul_seq(&co.transpose()); // nao × nao
    let bt = b.transpose();
    let mut r = b;
    for (rv, tv) in r.as_mut_slice().iter_mut().zip(bt.as_slice()) {
        *rv += *tv;
    }
    r
}

/// RHF response density (occupation weight 2).
fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    ao_response_density_w(u, cv, co, 2.0)
}

/// Solve the CPHF equations for one perturbation entirely in the MO occ–virt block: iterate
/// `U = (G_skel + [G(∂P(U))]_ov) / (ε_i − ε_a)` to self-consistency. `G(∂P) = F(∂P) − H_core`
/// is the two-electron response Fock (the orbital-Hessian coupling); the fixed point is the
/// coupled response. Returns the converged `U` (n_vir × n_occ).
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
) -> Result<Matrix> {
    // Uncoupled start: U0 = G / (ε_i − ε_a).
    let elem_div = |num: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    // One fixed-point step f(U) = elem_div(G_skel + [F(∂P(U)) − H_core]_ov).
    let step = |u: &Matrix| -> Result<Matrix> {
        let r = ao_response_density(u, cv, co);
        let vf = crate::fock::build_fock_x(molecule, basis, params, core, &r, exchange_cutoff)?;
        let mut g_resp_full = vf;
        for (xv, hv) in g_resp_full
            .as_mut_slice()
            .iter_mut()
            .zip(core.h_core.as_slice())
        {
            *xv -= *hv;
        }
        let g_resp = project_ov(&g_resp_full, cv, co);
        let mut rhs = g_ov.clone();
        for (rv, gv) in rhs.as_mut_slice().iter_mut().zip(g_resp.as_slice()) {
            *rv += *gv;
        }
        Ok(elem_div(&rhs))
    };
    // Pulay-DIIS-accelerated fixed point. The CPHF is a linear system, so DIIS converges to the
    // same U in far fewer iterations than the plain iteration; it stores only a small history of
    // ov-blocks (max_diis × n_vir·n_occ, freed per solve) → OOM-safe. On a singular DIIS system it
    // falls back to the plain step, so it can never do worse than the un-accelerated iteration.
    let max_diis = 8;
    let mut u = elem_div(g_ov);
    let mut images: Vec<Matrix> = Vec::new(); // f(U_i)
    let mut errors: Vec<Matrix> = Vec::new(); // e_i = f(U_i) − U_i
    for _ in 0..100 {
        let fu = step(&u)?;
        let mut e = fu.clone();
        for (ev, uv) in e.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *ev -= *uv;
        }
        let diff = e.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        if diff < 1.0e-9 {
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
    Ok(u)
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
) -> Result<Matrix> {
    use crate::dual2::Dual2;
    use crate::fock::build_fock_spin;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let core = crate::hamiltonian::build_core(molecule, &basis, params)?;

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
            molecule, params, &basis, &pt, &pa, &pb, &cva, &coa, &cvb, &cob,
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
    Ok(sym)
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
) -> Result<(Matrix, Matrix)> {
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
    for _ in 0..100 {
        let (fa, fb) = step(&ua, &ub)?;
        let mut ea = fa.clone();
        for (ev, uv) in ea.as_mut_slice().iter_mut().zip(ua.as_slice()) {
            *ev -= *uv;
        }
        let mut eb = fb.clone();
        for (ev, uv) in eb.as_mut_slice().iter_mut().zip(ub.as_slice()) {
            *ev -= *uv;
        }
        let diff = (ea.as_slice().iter().map(|x| x * x).sum::<f64>()
            + eb.as_slice().iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        if diff < 1.0e-9 {
            return Ok((fa, fb));
        }
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
    Ok((ua, ub))
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
        // Optimize water, then compute harmonic frequencies. Expect 3 real modes
        // (bend ~1600–1800, two stretches ~3700–3900 cm⁻¹ for PM7) plus ~6 near-zero.
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
        // The three highest are the real vibrational modes.
        let n = freqs.len();
        let high = &freqs[n - 3..];
        assert!(high[0] > 1200.0 && high[0] < 2200.0, "bend {}", high[0]);
        assert!(high[1] > 2000.0 && high[2] > 2000.0, "stretches {high:?}");
        // The six lowest (trans/rot) should be small in magnitude.
        let six_low_max = freqs[..6].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
        assert!(
            six_low_max < 300.0,
            "trans/rot not near zero: {six_low_max}"
        );
    }
}
