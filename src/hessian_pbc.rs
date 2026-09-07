// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic force constants for periodic systems.
//!
//! # What a Γ-point Hessian of a periodic cell is
//!
//! Displacing atom `A` of the home cell under periodic boundary conditions displaces its image in
//! *every* cell, so the second derivative this module computes is
//!
//! ```text
//! D(q = 0)_{Aα,Bβ} = Σ_T Φ(0Aα, TBβ)
//! ```
//!
//! the `q = 0` dynamical matrix — the zone-centre force constants. Its three exactly-zero
//! eigenvalues are the acoustic modes, and [`acoustic_residual`] measures how well that holds.
//! Running the same calculation on an `n₁×n₂×n₃` supercell resolves `Φ(0A, TB)` for every `T`
//! inside it, and hence `D(q)` at every `q` commensurate with that supercell — see
//! [`ForceConstants`].
//!
//! # Why it is analytic
//!
//! Every term of the PM7 energy depends on the nuclei only through pair displacements
//! `d = R_B + T − R_A`, so its second derivative is a 3×3 block per image pair that scatters onto
//! the four `(A,A)`, `(B,B)`, `(A,B)`, `(B,A)` positions exactly as the molecular Hessian's does.
//! A self pair (`A = B`, `T ≠ 0`) has a displacement that does not move with the atom at all, and
//! the four scattered terms cancel — which is right, and is why no special case is needed.
//!
//! The long-range monopole part is not a short-range pair sum, but it *is* a function of pair
//! displacements, so [`crate::pbc::ewald::ewald_pair_hessian`] supplies its 3×3 blocks on the same
//! footing.
//!
//! The orbital-relaxation (CPHF) half needs one change from the molecular case and no more: the
//! response kernel `G(ΔP) = F(P + ΔP) − F(P)` must be taken as a *difference*, because the
//! periodic Fock is affine rather than linear in the density — the Ewald potential is built from
//! `q_A = Z_A − P_A`, and that `Z_A` is a constant the molecular `F(ΔP) − H_core` shortcut would
//! wrongly keep.

use crate::basis::Basis;
use crate::dual::Scalar;
use crate::dual2::Dual2;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::pbc::{PairList, PbcMode, PbcOptions};
use crate::scf::{Pm7Options, Pm7Result};
use crate::system::Molecule;

/// Scatter a pair's 3×3 second-derivative block onto the four Hessian positions it belongs to.
///
/// `a == b` (a self-image pair) cancels to nothing, which is correct: the displacement of such a
/// pair is a lattice vector and does not move with the atom.
fn scatter_pair(hess: &mut Matrix, a: usize, b: usize, block: &[[f64; 3]; 3], weight: f64) {
    if a == b {
        return;
    }
    for (i, row) in block.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let val = weight * v;
            hess[(3 * a + i, 3 * a + j)] += val;
            hess[(3 * b + i, 3 * b + j)] += val;
            hess[(3 * a + i, 3 * b + j)] -= val;
            hess[(3 * b + i, 3 * a + j)] -= val;
        }
    }
}

/// The skeleton (fixed-density) second derivative of the short-range periodic energy.
///
/// One `Dual2` pass per image pair over exactly the terms
/// [`crate::gradient::electronic_gradient_periodic`] differentiates once, plus the core–core
/// repulsion, so the Hessian is the derivative of the gradient that is already checked against
/// finite differences rather than of a parallel re-derivation.
pub fn skeleton_hessian(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    density: crate::gradient::TranslatedDensity<'_>,
    spin: Option<(
        crate::gradient::TranslatedDensity<'_>,
        crate::gradient::TranslatedDensity<'_>,
    )>,
    pbc: &PbcOptions,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let p = density.onsite();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));
    let subtract_monopole = pbc.mode == PbcMode::Ewald;
    let list = PairList::cached(molecule, pbc.short_range_cutoff);

    let blocks: Result<Vec<(usize, usize, [[f64; 3]; 3], f64)>> = list
        .pairs
        .par_iter()
        .map(|pair| -> Result<(usize, usize, [[f64; 3]; 3], f64)> {
            let (u, v) = (pair.a, pair.b);
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            let (a, b, d, t) = if eu.n_orb >= ev.n_orb {
                (u, v, pair.d, pair.t)
            } else {
                (v, u, pair.d * -1.0, [-pair.t[0], -pair.t[1], -pair.t[2]])
            };
            let pt = density.at(t);
            let (pat, pbt) = match spin {
                Some((alpha, beta)) => (Some(alpha.at(t)), Some(beta.at(t))),
                None => (None, None),
            };
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let far = pair.r > crate::pbc::FEATHER_RANGE_BOHR;
            let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
            let (te, s) = if far {
                (
                    crate::integrals::point_charge_pair_dual2(ea, eb, d),
                    [[Dual2::constant(0.0); 9]; 9],
                )
            } else {
                let (te, s, _) = crate::hessian::pair_dual2_at(ea, eb, d, has_any_d)?;
                (te, s)
            };
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            let mut epair = Dual2::constant(0.0);
            // Resonance β·S — the density block at *this* translation.
            for i in 0..na {
                let bi = crate::hessian::beta_of(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = crate::hessian::beta_of(eb, basis.aos[ob + j].orb);
                    epair = epair + s[i][j] * (pt[(oa + i, ob + j)] * (bi + bj));
                }
            }
            // Electron–core attraction, on-site blocks.
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
            // Two-electron: Coulomb from the on-site blocks, exchange from this translation.
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                            let exch = match (pat, pbt) {
                                (Some(pa), Some(pb)) => {
                                    -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                        + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)])
                                }
                                _ => -0.5 * pt[(oa + mu, ob + la)] * pt[(oa + nu, ob + si)],
                            };
                            epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                        }
                    }
                }
            }
            let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
            if subtract_monopole {
                // The monopole the Ewald sum re-supplies, removed from e1b, e2a and the
                // two-electron block. Same coefficient as in the gradient.
                let pop_a: f64 = (0..na).map(|k| p[(oa + k, oa + k)]).sum();
                let pop_b: f64 = (0..nb).map(|k| p[(ob + k, ob + k)]).sum();
                let exch_mono: f64 = (0..na)
                    .flat_map(|mu| (0..nb).map(move |la| (mu, la)))
                    .map(|(mu, la)| match (pat, pbt) {
                        (Some(pa), Some(pb)) => {
                            let (x, y) = (pa[(oa + mu, ob + la)], pb[(oa + mu, ob + la)]);
                            -(x * x + y * y)
                        }
                        _ => {
                            let x = pt[(oa + mu, ob + la)];
                            -0.5 * x * x
                        }
                    })
                    .sum();
                let c =
                    -(pop_a * pop_b + exch_mono) + eb.core_charge * pop_a + ea.core_charge * pop_b;
                epair = epair + r.recip() * (c * crate::constants::PM7_EV);
            }
            // Core–core repulsion, with its own monopole removed in Ewald mode.
            epair = epair
                + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                    ea,
                    eb,
                    molecule.atoms[a].z,
                    molecule.atoms[b].z,
                    r,
                    params,
                );
            if subtract_monopole {
                let zz = ea.core_charge * eb.core_charge * crate::constants::PM7_EV;
                epair = epair - r.recip() * zz;
            }
            Ok((a, b, epair.h, pair.weight))
        })
        .collect();

    let mut hess = Matrix::zeros(3 * nat, 3 * nat);
    for (a, b, block, weight) in blocks? {
        scatter_pair(&mut hess, a, b, &block, weight);
    }
    Ok(hess)
}

/// Second derivative of the long-range monopole energy — Coulomb and the regularized exchange.
///
/// Both are `½ Σ_AB c_AB M_AB` with a different coefficient matrix, and neither the exchange
/// divergence correction nor the self term moves with the atoms, so this is two calls to
/// [`crate::pbc::ewald::ewald_pair_hessian`] and nothing else.
pub fn long_range_hessian(
    molecule: &Molecule,
    basis: &Basis,
    charges: &[f64],
    spin_densities: &[crate::gradient::TranslatedDensity<'_>],
    pbc: &PbcOptions,
) -> Result<Matrix> {
    let cell = molecule.cell.expect("periodic");
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let n = positions.len();
    let ep = crate::pbc::EwaldParameters::new(&cell, n, pbc.ewald_accuracy, pbc.ewald_alpha);
    let coulomb: Vec<Vec<f64>> = (0..n)
        .map(|a| (0..n).map(|b| charges[a] * charges[b]).collect())
        .collect();
    let mut hess = crate::pbc::ewald::ewald_pair_hessian(&cell, &positions, &coulomb, &ep);

    // Exchange, resolved by Born–von Kármán residue class. At Γ there is one class and the
    // supercell is the cell.
    let divisions = if spin_densities.iter().any(|d| d.is_bloch()) {
        pbc.kmesh.divisions()
    } else {
        [1, 1, 1]
    };
    let classes = crate::hamiltonian::bvk_representatives(divisions);
    let (super_cell, _) = cell.supercell(divisions)?;
    let super_ep = crate::pbc::EwaldParameters::new(
        &super_cell,
        n * classes.len(),
        pbc.ewald_accuracy,
        pbc.ewald_alpha,
    );
    // Every class's reciprocal second derivative in one pass over `G`, when the sum is 3-D.
    //
    // The same fold as the gradient's, for the same reason: done class by class this is `O(C²)`
    // in the mesh, because `|G|` grows with the supercell exactly as the class count does. What
    // stays in the class loop below is the real-space sum, whose distances genuinely change with
    // the translation. See [`crate::pbc::ewald::ewald_reciprocal_hessian_bvk`].
    let folded = super_cell.dim() == 3 && classes.len() > 1;
    if folded {
        let per_class: Vec<Vec<Vec<f64>>> = classes
            .iter()
            .map(|t| {
                let blocks: Vec<&Matrix> = spin_densities.iter().map(|d| d.at(*t)).collect();
                crate::gradient::exchange_coefficient_matrix(basis, &blocks)
            })
            .collect();
        let folded_hessian = crate::pbc::ewald::ewald_reciprocal_hessian_bvk(
            &cell,
            &super_cell,
            divisions,
            &positions,
            &per_class,
            &super_ep,
        );
        for (dst, src) in hess
            .as_mut_slice()
            .iter_mut()
            .zip(folded_hessian.as_slice())
        {
            *dst += *src;
        }
    }
    let pair_hessian = |positions: &[Vec3], c: &[Vec<f64>]| {
        crate::pbc::ewald::ewald_pair_hessian_with(&super_cell, positions, c, &super_ep, !folded)
    };

    for t in &classes {
        let blocks: Vec<&Matrix> = spin_densities.iter().map(|d| d.at(*t)).collect();
        let c = crate::gradient::exchange_coefficient_matrix(basis, &blocks);
        if *t == [0, 0, 0] {
            let h = pair_hessian(&positions, &c);
            for (dst, src) in hess.as_mut_slice().iter_mut().zip(h.as_slice()) {
                *dst += *src;
            }
            continue;
        }
        let shift = cell.translation(*t);
        let mut doubled = Vec::with_capacity(2 * n);
        doubled.extend_from_slice(&positions);
        doubled.extend(positions.iter().map(|p| *p + shift));
        let mut c2 = vec![vec![0.0_f64; 2 * n]; 2 * n];
        for (a, row) in c.iter().enumerate() {
            for (b, v) in row.iter().enumerate() {
                c2[a][n + b] = 0.5 * v;
                c2[n + b][a] = 0.5 * v;
            }
        }
        let h = pair_hessian(&doubled, &c2);
        // Fold the displaced copies back onto their home atoms: they move together.
        for a in 0..n {
            for i in 0..3 {
                for b in 0..n {
                    for j in 0..3 {
                        let v = h[(3 * a + i, 3 * b + j)]
                            + h[(3 * (n + a) + i, 3 * b + j)]
                            + h[(3 * a + i, 3 * (n + b) + j)]
                            + h[(3 * (n + a) + i, 3 * (n + b) + j)];
                        hess[(3 * a + i, 3 * b + j)] += v;
                    }
                }
            }
        }
    }
    Ok(hess)
}

/// The derivative Fock matrices `∂F/∂R_c` of a **Γ-point** periodic cell, at fixed density.
///
/// One `nao × nao` matrix per Cartesian degree of freedom. The short-range part mirrors the
/// molecular `skeleton_fock_ov` over image pairs; on top of it sit the two long-range terms the
/// molecular case does not have:
///
/// * the Ewald potential on the diagonal, `−V_A = −Σ_B M_AB q_B`, and
/// * the regularized long-range exchange, `−P_{μ_A λ_B}(M_AB − M_self)`,
///
/// both of which move with the atoms through `∂M_AB/∂R_C = (δ_CB − δ_CA) F_AB`, the field matrix
/// from [`crate::pbc::ewald::ewald_field_matrix`].
///
/// Γ point only: at a k mesh the derivative Fock acquires a translation index and the response
/// couples `k` with `k + q`, which is a different calculation rather than a bigger one.
pub(crate) fn derivative_fock(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    p: &Matrix,
    charges: &[f64],
    pbc: &PbcOptions,
) -> Result<Vec<Matrix>> {
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));
    let subtract_monopole = pbc.mode == PbcMode::Ewald;
    let list = PairList::cached(molecule, pbc.short_range_cutoff);
    let mut out: Vec<Matrix> = (0..3 * nat).map(|_| Matrix::zeros(nao, nao)).collect();

    for pair in &list.pairs {
        let (u, v) = (pair.a, pair.b);
        if u == v {
            // A self-image pair contributes to the Fock but not to its position derivative.
            continue;
        }
        let eu = params.element(molecule.atoms[u].z)?;
        let ev = params.element(molecule.atoms[v].z)?;
        let (a, b, d) = if eu.n_orb >= ev.n_orb {
            (u, v, pair.d)
        } else {
            (v, u, pair.d * -1.0)
        };
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let (te, s) = if pair.r > crate::pbc::FEATHER_RANGE_BOHR {
            (
                crate::integrals::point_charge_pair_dual(ea, eb, d),
                [[crate::dual::Dual::constant(0.0); 9]; 9],
            )
        } else {
            crate::gradient::pair_dual(ea, eb, Vec3::zero(), d, has_any_d)?
        };
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        // Derivative of the removed monopole `PM7_EV/r` with respect to the displacement.
        let dv: [f64; 3] = if subtract_monopole {
            let c = -crate::constants::PM7_EV / (pair.r * pair.r * pair.r);
            [d.x * c, d.y * c, d.z * c]
        } else {
            [0.0; 3]
        };
        for axis in 0..3 {
            // `E_pair` depends on `R_b − R_a`, so ∂/∂R_b = +∂/∂d and ∂/∂R_a = −∂/∂d.
            for (atom, sign) in [(b, pair.weight), (a, -pair.weight)] {
                let fm = &mut out[3 * atom + axis];
                // Resonance β·S.
                for i in 0..na {
                    let bi = crate::hessian::beta_of(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = crate::hessian::beta_of(eb, basis.aos[ob + j].orb);
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
                // Two-electron Coulomb, with the monopole removed where Ewald re-supplies it.
                for mu in 0..na {
                    for nu in 0..na {
                        let mut acc = 0.0;
                        for la in 0..nb {
                            for si in 0..nb {
                                acc += p[(ob + la, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                            }
                        }
                        if subtract_monopole && mu == nu {
                            let pop_b: f64 = (0..nb).map(|k| p[(ob + k, ob + k)]).sum();
                            acc -= dv[axis] * pop_b;
                            acc += dv[axis] * eb.core_charge;
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
                        if subtract_monopole && la == si {
                            let pop_a: f64 = (0..na).map(|k| p[(oa + k, oa + k)]).sum();
                            acc -= dv[axis] * pop_a;
                            acc += dv[axis] * ea.core_charge;
                        }
                        fm[(ob + la, ob + si)] += sign * acc;
                    }
                }
                // Two-electron exchange, likewise.
                for mu in 0..na {
                    for la in 0..nb {
                        let mut acc = 0.0;
                        for nu in 0..na {
                            for si in 0..nb {
                                acc += p[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si).d[axis];
                            }
                        }
                        if subtract_monopole {
                            acc -= p[(oa + mu, ob + la)] * dv[axis];
                        }
                        let val = sign * (-0.5 * acc);
                        fm[(oa + mu, ob + la)] += val;
                        fm[(ob + la, oa + mu)] += val;
                    }
                }
            }
        }
    }

    if subtract_monopole {
        let cell = molecule.cell.expect("periodic");
        let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        let ep = crate::pbc::EwaldParameters::new(&cell, nat, pbc.ewald_accuracy, pbc.ewald_alpha);
        let field = crate::pbc::ewald::ewald_field_matrix(&cell, &positions, &ep);
        for c in 0..nat {
            for axis in 0..3 {
                let fm = &mut out[3 * c + axis];
                // Ewald potential: −V_A = −Σ_B M_AB q_B on the diagonal of atom A's block.
                for ia in 0..nat {
                    let mut dv_a = 0.0;
                    for ib in 0..nat {
                        // ∂M_AB/∂R_c = (δ_cB − δ_cA) F_AB.
                        let mut factor = 0.0;
                        if c == ib {
                            factor += 1.0;
                        }
                        if c == ia {
                            factor -= 1.0;
                        }
                        if factor != 0.0 {
                            dv_a += factor * field[ia][ib].get(axis) * charges[ib];
                        }
                    }
                    let off = basis.atom_offset[ia];
                    for mu in 0..basis.atom_norb[ia] {
                        fm[(off + mu, off + mu)] -= dv_a;
                    }
                }
                // Long-range exchange: −P_{μ_A λ_B}(M_AB − M_self); only `M_AB` moves.
                for ia in 0..nat {
                    let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
                    for ib in 0..nat {
                        let mut factor = 0.0;
                        if c == ib {
                            factor += 1.0;
                        }
                        if c == ia {
                            factor -= 1.0;
                        }
                        if factor == 0.0 {
                            continue;
                        }
                        let dm = factor * field[ia][ib].get(axis);
                        let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
                        for mu in 0..na {
                            for la in 0..nb {
                                // RHF: the same-spin density is half the total, and the two spins
                                // sum back to a factor of one half overall.
                                fm[(oa + mu, ob + la)] -= 0.5 * p[(oa + mu, ob + la)] * dm;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A phonon spectrum at one wavevector: the frequencies, and the motions that go with them.
///
/// Both phonon routes produce this — [`ForceConstants::modes`] from a supercell and
/// [`crate::dfpt::DfptResult::modes`] from the perturbation solver — so a caller can read a mode
/// the same way whichever route produced it.
#[derive(Clone, Debug)]
pub struct PhononModes {
    /// cm⁻¹, ascending. A negative value is an imaginary frequency (`−|ω|`).
    pub frequencies_cm: Vec<f64>,
    /// Eigenvectors of the **mass-weighted** dynamical matrix, one mode per column, in the same
    /// order as `frequencies_cm`. Unitary, so `Σ_i |e_i|² = 1` down each column.
    ///
    /// Complex away from the zone centre: a phonon at `q` is `u_A ∝ e_A e^{iq·R_A}`, and the phase
    /// carried by `e` is what distinguishes the branches that share a `|q|`. At `q = 0` and at a
    /// zone-boundary `q` where every phase is `±1` the imaginary part is zero to rounding.
    pub eigenvectors: crate::cmatrix::CMatrix,
    /// The same modes as Cartesian displacements `m_A^{−1/2} e_A`, each column renormalized to
    /// unit length.
    ///
    /// This is the one to displace a structure along: the mass-weighted eigenvector is what the
    /// eigenproblem is posed in, and the atoms move as its mass-scaled image. The convention
    /// matches [`crate::hessian::VibrationalModes::cartesian_modes`] exactly, which is MOPAC's
    /// `cnorml` — so a molecular mode and a zone-centre phonon mean the same thing by the same
    /// rule.
    pub cartesian_modes: crate::cmatrix::CMatrix,
}

/// Diagonalize a mass-weighted dynamical matrix into [`PhononModes`].
///
/// Shared by both phonon routes so that "the frequencies" and "the modes" cannot come from two
/// diagonalizations that drift apart — `frequencies_cm` on either route is this function's output
/// with one field taken.
pub(crate) fn phonon_modes(d: &crate::cmatrix::CMatrix, masses: &[f64]) -> Result<PhononModes> {
    let (eigenvalues, eigenvectors) = d.hermitian_eigen()?;
    let frequencies_cm = eigenvalues
        .into_iter()
        .map(|lambda| {
            let w = lambda.abs().sqrt() * crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
            if lambda < 0.0 {
                -w
            } else {
                w
            }
        })
        .collect();

    // `m^{-1/2} e`, renormalized per column. The mass-weighted eigenvector is unit length and its
    // Cartesian image is not, so the normalization is not cosmetic — without it the length of a
    // "mode" would depend on the masses of the atoms it happens to move.
    let n = d.n;
    let mut cartesian_modes = crate::cmatrix::CMatrix::zeros(n);
    for column in 0..n {
        let mut norm = 0.0;
        for row in 0..n {
            let (re, im) = eigenvectors.get(row, column);
            let scale = masses[row / 3].sqrt();
            let (re, im) = (re / scale, im / scale);
            cartesian_modes.set(row, column, re, im);
            norm += re * re + im * im;
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for row in 0..n {
                let (re, im) = cartesian_modes.get(row, column);
                cartesian_modes.set(row, column, re / norm, im / norm);
            }
        }
    }

    Ok(PhononModes {
        frequencies_cm,
        eigenvectors,
        cartesian_modes,
    })
}

/// Real-space force constants `Φ(0A, TB)` and everything derived from them.
#[derive(Clone, Debug)]
pub struct ForceConstants {
    /// The cell the translations index.
    pub cell: crate::cell::Cell,
    /// Translations, closed under negation.
    pub translations: Vec<[i32; 3]>,
    /// `blocks[t]` is the `3N × 3N` matrix `Φ(0A, TB)` in eV/Bohr².
    pub blocks: Vec<Matrix>,
    /// Atomic masses in amu, in atom order.
    pub masses: Vec<f64>,
    /// The supercell these came from. Recorded because it is what decides which wavevectors are
    /// exact: `D(q)` is the true dynamical matrix only where `q·n` is integral on every periodic
    /// axis, and an interpolated `q` is otherwise indistinguishable from an exact one.
    pub supercell: [usize; 3],
}

impl ForceConstants {
    /// `Σ_T Φ(0A, TB)`, the `q = 0` dynamical matrix before mass weighting.
    pub fn gamma(&self) -> Matrix {
        let mut m = Matrix::zeros(self.blocks[0].rows, self.blocks[0].cols);
        for b in &self.blocks {
            for (dst, src) in m.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *dst += *src;
            }
        }
        m
    }

    /// How badly the acoustic sum rule `Σ_{T,B} Φ(0A, TB) = 0` is violated, in eV/Bohr².
    ///
    /// It follows exactly from every term depending only on displacement *differences*, so a
    /// non-zero value is numerical, not physical: it measures truncated image sums and SCF
    /// convergence, and is the natural sanity check on a set of force constants.
    pub fn acoustic_residual(&self) -> f64 {
        let gamma = self.gamma();
        let nat = self.masses.len();
        let mut worst = 0.0_f64;
        for a in 0..nat {
            for i in 0..3 {
                for j in 0..3 {
                    let sum: f64 = (0..nat).map(|b| gamma[(3 * a + i, 3 * b + j)]).sum();
                    worst = worst.max(sum.abs());
                }
            }
        }
        worst
    }

    /// Project the acoustic sum rule back onto the force constants, by removing the residual from
    /// the self block. Optional: it changes the numbers, so it is never applied silently.
    ///
    /// Two constraints have to survive together: the sum rule, and the symmetry of the `q = 0`
    /// matrix — without the first the acoustic branch does not reach zero, without the second
    /// `D(q)` stops being Hermitian. Simply subtracting the row sums from the self block satisfies
    /// the first and breaks the second, and alternating the two repairs only half the residual
    /// (its antisymmetric part is a fixed point of that iteration).
    ///
    /// So the correction is the two-sided projector `Φ(0) += P Γ P − Γ`, with
    /// `P = I − (1/N) J ⊗ I₃` removing the uniform translation. `P Γ P` is symmetric because `Γ`
    /// is, and has zero row sums because `P` annihilates the uniform vector — both exactly, in
    /// one pass.
    pub fn enforce_acoustic_sum_rule(&mut self) {
        let nat = self.masses.len();
        if nat == 0 {
            return;
        }
        let zero = self
            .translations
            .iter()
            .position(|t| *t == [0, 0, 0])
            .expect("the zero translation is always present");
        let gamma = self.gamma();
        let inv = 1.0 / nat as f64;
        // Row, column and grand means of each 3×3 Cartesian sub-block.
        let mut row = vec![[[0.0_f64; 3]; 3]; nat]; // Σ_b Γ(a,b)
        let mut col = vec![[[0.0_f64; 3]; 3]; nat]; // Σ_a Γ(a,b)
        let mut all = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for a in 0..nat {
                    for b in 0..nat {
                        let v = gamma[(3 * a + i, 3 * b + j)];
                        row[a][i][j] += v;
                        col[b][i][j] += v;
                        all[i][j] += v;
                    }
                }
            }
        }
        let block = &mut self.blocks[zero];
        for a in 0..nat {
            for b in 0..nat {
                for i in 0..3 {
                    for j in 0..3 {
                        block[(3 * a + i, 3 * b + j)] +=
                            -inv * row[a][i][j] - inv * col[b][i][j] + inv * inv * all[i][j];
                    }
                }
            }
        }
    }

    /// Mass-weighted dynamical matrix `D(q) = Σ_T Φ(0A,TB) e^{i q·T} / √(m_A m_B)`, in
    /// eV/(Å²·amu) — the same units [`crate::hessian::VibrationalModes::eigenvalues`] uses, so a
    /// Γ-point phonon and a molecular vibration are directly comparable numbers.
    ///
    /// `q` is in **fractional** reciprocal coordinates, so `q = (0.5, 0, 0)` is the zone boundary
    /// along the first lattice vector and the phase is `e^{2πi q·t}` with integer `t`.
    ///
    /// # Exact at the commensurate wavevectors, interpolated everywhere else
    ///
    /// A supercell holds `Φ(0A, TB)` exactly for the translations it contains, so `D(q)` is the
    /// true dynamical matrix precisely where `q·n` is an integer on every periodic axis. Anywhere
    /// else this is a Fourier interpolation between those points, and [`Self::is_commensurate`]
    /// says which case a given `q` is.
    ///
    /// Two things bound the interpolation, and the second is the one that surprises:
    ///
    /// 1. the force constants must have decayed inside the supercell; and
    /// 2. at a zone-boundary translation the supercell **cannot distinguish `+T` from `−T`**. The
    ///    folding and the `Φ(−T) = Φ(T)ᵀ` averaging below keep the symmetric part and discard the
    ///    antisymmetric one. That is exact at every commensurate `q`, where `e^{iq·T} = ±1` is
    ///    real and only the symmetric part is observable — and it is an assumption elsewhere,
    ///    which no tighter SCF and no larger cutoff repairs.
    ///
    /// Measured on diamond with a 2×2×2 supercell at `q = (¼,0,0)`: the flat TA branch interpolates
    /// to 372.5 cm⁻¹ against DFPT's 377.3, while the dispersive LA branch gives 798.5 against 670.0
    /// — 19 % on a number printed with the same authority as an exact one. Hence the warning.
    pub fn dynamical_matrix(&self, q_frac: [f64; 3]) -> crate::cmatrix::CMatrix {
        self.warn_if_incommensurate(q_frac);
        let n = self.blocks[0].rows;
        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let mut d = crate::cmatrix::CMatrix::zeros(n);
        for (block, t) in self.blocks.iter().zip(&self.translations) {
            let phase = std::f64::consts::TAU
                * (q_frac[0] * t[0] as f64 + q_frac[1] * t[1] as f64 + q_frac[2] * t[2] as f64);
            let (c, s) = (phase.cos(), phase.sin());
            for i in 0..n {
                for j in 0..n {
                    let w = a0_sq / (self.masses[i / 3] * self.masses[j / 3]).sqrt();
                    let v = block[(i, j)] * w;
                    let (re, im) = d.get(i, j);
                    d.set(i, j, re + v * c, im + v * s);
                }
            }
        }
        d
    }

    /// Does this supercell hold `q` exactly? True when `q·n` is integral on every periodic axis.
    ///
    /// A tolerance rather than an equality, because `1/3` is not representable and a caller who
    /// typed `0.333333` for a 3×3×3 supercell means the commensurate point.
    pub fn is_commensurate(&self, q_frac: [f64; 3]) -> bool {
        (0..self.cell.dim()).all(|k| {
            let x = q_frac[k] * self.supercell[k] as f64;
            (x - x.round()).abs() <= 1.0e-9
        })
    }

    /// Warn when `q` is being interpolated rather than evaluated. `PM7_QUIET` silences it.
    fn warn_if_incommensurate(&self, q_frac: [f64; 3]) {
        if self.is_commensurate(q_frac) || std::env::var_os("PM7_QUIET").is_some() {
            return;
        }
        // The smallest supercell that would hold this q exactly, per axis: the denominator of the
        // fraction, found by walking up rather than by a continued-fraction expansion, since the
        // useful answers are small.
        let needed: Vec<String> = (0..3)
            .map(|k| {
                if k >= self.cell.dim() {
                    return "1".to_string();
                }
                let q = q_frac[k].abs();
                for n in 1..=64_usize {
                    let x = q * n as f64;
                    if (x - x.round()).abs() <= 1.0e-9 {
                        return n.to_string();
                    }
                }
                ">64".to_string()
            })
            .collect();
        eprintln!(
            "pm7-rs: q = ({:.6}, {:.6}, {:.6}) is not commensurate with the {}x{}x{} \
             force-constant supercell, so D(q) here is a Fourier interpolation between the \
             wavevectors the supercell does hold, not an evaluation at this one. Two things limit \
             it: the force constants must have decayed inside the supercell, and at a \
             zone-boundary translation the supercell cannot distinguish +T from -T, so the \
             antisymmetric part of that block is discarded -- an error that does not shrink with a \
             tighter SCF. Use --supercell {} {} {}, a commensurate q, or `dfpt`, which needs no \
             supercell. Set PM7_QUIET to silence this.",
            q_frac[0],
            q_frac[1],
            q_frac[2],
            self.supercell[0],
            self.supercell[1],
            self.supercell[2],
            needed[0],
            needed[1],
            needed[2],
        );
    }

    /// Phonon frequencies at `q_frac`, in cm⁻¹, ascending. A negative value is an imaginary
    /// frequency (`−|ω|`), which is how an unstable mode is reported rather than hidden.
    pub fn frequencies_cm(&self, q_frac: [f64; 3]) -> Result<Vec<f64>> {
        Ok(self.modes(q_frac)?.frequencies_cm)
    }

    /// The frequencies **and the polarization vectors** at `q_frac`.
    ///
    /// The eigenvectors were always computed here and thrown away, which made a frequency the only
    /// thing this route could tell you — no way to say which atoms a soft branch moves, no way to
    /// displace a structure along a mode, no way to assign a symmetry label. They are the same
    /// diagonalization; the caller simply gets both halves of it.
    pub fn modes(&self, q_frac: [f64; 3]) -> Result<PhononModes> {
        phonon_modes(&self.dynamical_matrix(q_frac), &self.masses)
    }

    /// `Σ_T Φ(0A,TB) + Φ^NA(q_hat)` at the zone centre, in eV/Bohr², **before** mass weighting.
    ///
    /// The supercell twin of [`crate::dfpt::DfptResult::force_constants_with_lo_to`], so a caller
    /// can apply the same LO–TO correction to either phonon route. Convention C-7 throughout, and
    /// the `NonAnalytic` it takes still has to come from [`crate::dfpt::born_and_dielectric`] —
    /// the supercell path produces force constants, not Born charges.
    pub fn gamma_with_lo_to(
        &self,
        na: &crate::dfpt::NonAnalytic,
        q_hat: [f64; 3],
    ) -> Result<crate::cmatrix::CMatrix> {
        let extra = na.matrix(q_hat)?;
        let gamma = self.gamma();
        if extra.n != gamma.rows {
            return Err(crate::error::Pm7Error::InvalidInput(format!(
                "the non-analytic term is {}x{} but these force constants are {}x{}; they are not \
                 from the same cell",
                extra.n, extra.n, gamma.rows, gamma.cols
            )));
        }
        let mut out = crate::cmatrix::CMatrix::zeros(gamma.rows);
        for i in 0..gamma.rows {
            for j in 0..gamma.cols {
                let (br, bi) = extra.get(i, j);
                out.set(i, j, gamma[(i, j)] + br, bi);
            }
        }
        Ok(out)
    }

    /// Zone-centre frequencies in cm⁻¹ with the LO–TO term added along `q_hat`.
    pub fn frequencies_cm_lo_to(
        &self,
        na: &crate::dfpt::NonAnalytic,
        q_hat: [f64; 3],
    ) -> Result<Vec<f64>> {
        let phi = self.gamma_with_lo_to(na, q_hat)?;
        let n = phi.n;
        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let mut d = crate::cmatrix::CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let w = a0_sq / (self.masses[i / 3] * self.masses[j / 3]).sqrt();
                let (re, im) = phi.get(i, j);
                d.set(i, j, re * w, im * w);
            }
        }
        let (eigenvalues, _) = d.hermitian_eigen()?;
        Ok(eigenvalues
            .into_iter()
            .map(|lambda| {
                let w = lambda.abs().sqrt() * crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
                if lambda < 0.0 {
                    -w
                } else {
                    w
                }
            })
            .collect())
    }
}

/// Real-space force constants of a cell, from the analytic Hessian of an `n₁×n₂×n₃` supercell.
///
/// The supercell's Γ-point Hessian is `Φ_super(0A', 0B')` over its `N·n₁n₂n₃` atoms, and each of
/// its atoms is a `(cell atom, translation)` pair — so reading the supercell Hessian's blocks off
/// by translation *is* `Φ(0A, TB)`, with no interpolation and no finite displacement anywhere.
/// The result is exact for every `q` commensurate with the supercell, and Fourier-interpolates
/// between them as well as the force constants have decayed inside it.
///
/// `supercell = (1, 1, 1)` gives the single `T = 0` block, i.e. the zone-centre force constants
/// and nothing else.
/// The supercell *is* the Brillouin-zone sampling, so a k mesh asked for alongside it is not
/// coarsened — it is replaced. Say so, rather than quietly computing something else.
///
/// A warning and not a refusal, for the reason `warn_if_q_outruns_the_mesh` gives
/// (`src/dfpt.rs`): a deliberately coarse survey is a legitimate thing to run, and there is no
/// threshold at which the answer stops being useful. `PM7_QUIET` silences it.
fn warn_if_the_supercell_replaces_the_mesh(
    options: &Pm7Options,
    supercell: [usize; 3],
    dim: usize,
) {
    if std::env::var_os("PM7_QUIET").is_some() {
        return;
    }
    let Some(pbc) = options.pbc.as_ref() else {
        return;
    };
    let asked = pbc.kmesh.divisions();
    // Only the periodic axes carry a mesh; the rest are 1 by construction.
    let short: Vec<usize> = (0..dim).filter(|&k| asked[k] > supercell[k]).collect();
    if short.is_empty() {
        return;
    }
    let needed: Vec<String> = (0..3)
        .map(|k| asked[k].max(supercell[k]).to_string())
        .collect();
    eprintln!(
        "pm7-rs: --kpoints {} {} {} is finer than the {}x{}x{} force-constant supercell on \
         axis {:?}, and the supercell wins. A supercell's Gamma point *is* the cell's \
         n1xn2xn3 mesh, so the two cannot both apply, and the analytic periodic Hessian this \
         path uses is only defined at a Gamma mesh. What was computed is the {}x{}x{} sampling. \
         Use --supercell {} {} {} to get the mesh you asked for, or `dfpt`, which needs no \
         supercell. Set PM7_QUIET to silence this.",
        asked[0],
        asked[1],
        asked[2],
        supercell[0],
        supercell[1],
        supercell[2],
        short,
        supercell[0],
        supercell[1],
        supercell[2],
        needed[0],
        needed[1],
        needed[2],
    );
}

pub fn force_constants(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    supercell: [usize; 3],
) -> Result<ForceConstants> {
    let cell = molecule.cell.ok_or_else(|| {
        crate::error::Pm7Error::InvalidInput(
            "force constants are only defined for a periodic system; this molecule has no cell"
                .into(),
        )
    })?;
    if supercell.contains(&0) {
        return Err(crate::error::Pm7Error::InvalidInput(
            "each supercell division must be at least 1".into(),
        ));
    }
    let nat = molecule.atoms.len();
    let (super_cell, translations) = cell.supercell(supercell)?;
    let mut atoms = Vec::with_capacity(nat * translations.len());
    for t in &translations {
        let shift = cell.translation(*t);
        for atom in &molecule.atoms {
            atoms.push(crate::system::Atom {
                z: atom.z,
                position: atom.position + shift,
            });
        }
    }
    let expanded = Molecule::new(atoms)
        .with_charge(molecule.charge * translations.len() as f64)
        .with_multiplicity(molecule.multiplicity)
        .with_cell(super_cell);
    // Γ of the supercell is exactly the `n₁×n₂×n₃` mesh of the cell — the identity
    // `tests/pbc_equivalence.rs` pins — so the supercell *is* the Brillouin-zone sampling and the
    // caller's own k mesh cannot also apply. `analytic_hessian_periodic` is only trusted at a Γ
    // mesh anyway, which is why `analytic_hessian` delegates to DFPT whenever the mesh is not
    // `[1,1,1]` (`src/hessian.rs`).
    //
    // Silently substituting it is the problem: `--kpoints 6 6 6 --supercell 2 2 2` computed at
    // 2×2×2 and said nothing, and on diamond that is 1317.9 cm⁻¹ against the 1237.7 the same code
    // gives once the sampling is converged — an 80 cm⁻¹ error that looks like a result.
    warn_if_the_supercell_replaces_the_mesh(options, supercell, cell.dim());
    let mut super_options = options.clone();
    super_options.pbc = Some(crate::pbc::PbcOptions {
        kmesh: crate::pbc::KMesh::Gamma,
        ..options.pbc.clone().unwrap_or_default()
    });
    let scf = crate::scf::run_pm7(&expanded, params, &super_options)?;
    let full = analytic_hessian_periodic(&expanded, params, &super_options, &scf)?;

    // Slice the supercell Hessian back into per-translation blocks. Supercell atom `i` is
    // `(translations[i / nat], i % nat)`, so the `(0A, TB)` block is rows `A` of the first
    // replica against columns `B` of replica `T`.
    //
    // The translations then have to be **folded into the symmetric range**, because
    // `cell.supercell` numbers them `0..n` and that set is not closed under negation. Fourier
    // transforming `{0, 1, 2}` with phases `e^{iθt}` instead of `{0, 1, −1}` gives a `D(q)` that
    // is not even Hermitian away from the commensurate points — the n = 3 chain came out with a
    // hermiticity error of 3.7 and frequencies to match.
    let mut folded: Vec<([i32; 3], Matrix)> = Vec::new();
    for (index, shift) in translations.iter().enumerate() {
        let mut block = Matrix::zeros(3 * nat, 3 * nat);
        for a in 0..nat {
            for b in 0..nat {
                for i in 0..3 {
                    for j in 0..3 {
                        block[(3 * a + i, 3 * b + j)] =
                            full[(3 * a + i, 3 * (index * nat + b) + j)];
                    }
                }
            }
        }
        // A direction with `2t = n` folds onto itself: that block stands for both `+t` and `−t`,
        // so it is split evenly between them. Entering it once would break Hermiticity exactly
        // as an unfolded list does.
        let mut base = [0_i32; 3];
        let mut self_negative = Vec::new();
        for k in 0..3 {
            let n = supercell[k] as i32;
            let i = shift[k];
            if 2 * i == n && n > 1 {
                base[k] = i;
                self_negative.push(k);
            } else if 2 * i > n {
                base[k] = i - n;
            } else {
                base[k] = i;
            }
        }
        let count = 1_usize << self_negative.len();
        let weight = 1.0 / count as f64;
        for mask in 0..count {
            let mut t = base;
            for (bit, &k) in self_negative.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    t[k] = -base[k];
                }
            }
            let mut piece = block.clone();
            for v in piece.as_mut_slice() {
                *v *= weight;
            }
            match folded.iter_mut().find(|(other, _)| *other == t) {
                Some((_, existing)) => {
                    for (dst, src) in existing.as_mut_slice().iter_mut().zip(piece.as_slice()) {
                        *dst += *src;
                    }
                }
                None => folded.push((t, piece)),
            }
        }
    }
    let (translations, mut blocks): (Vec<_>, Vec<_>) = folded.into_iter().unzip();
    // Impose `Φ(−T) = Φ(T)ᵀ`. It holds exactly in the continuum of the model — the two are the
    // same physical force constant read from opposite ends — but the computed pair is only equal
    // to the extent that image sums are converged and the SCF is tight. Averaging them is what
    // makes `D(q)` Hermitian at *every* `q` rather than only where the phases happen to be real;
    // without it the acoustic modes of a doubled chain came out at ±9.5 cm⁻¹ instead of zero.
    let mirrored: Vec<Matrix> = translations
        .iter()
        .map(|t| {
            let neg = [-t[0], -t[1], -t[2]];
            let index = translations
                .iter()
                .position(|o| *o == neg)
                .expect("the folded translation set is closed under negation");
            blocks[index].transpose()
        })
        .collect();
    for (block, mirror) in blocks.iter_mut().zip(&mirrored) {
        for (v, m) in block.as_mut_slice().iter_mut().zip(mirror.as_slice()) {
            *v = 0.5 * (*v + *m);
        }
    }
    let masses = molecule
        .atoms
        .iter()
        .map(|a| crate::data_tables::MASS[a.z as usize])
        .collect();
    Ok(ForceConstants {
        cell,
        translations,
        blocks,
        masses,
        supercell,
    })
}

/// Analytic Γ-point (`q = 0`) Hessian of a periodic cell, in eV/Bohr².
pub fn analytic_hessian_periodic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    scf: &Pm7Result,
) -> Result<Matrix> {
    let pbc = options.pbc_for(molecule).expect("periodic");
    let basis = Basis::build(molecule, params)?;
    let divisions = pbc.kmesh.divisions();

    // Densities, translation-resolved when a k mesh made them so.
    let bloch_spin = scf
        .bloch_density
        .as_ref()
        .map(|total| crate::gradient::bloch_spin_densities(total, scf.bloch_spin_density.as_ref()));
    let half_blocks = scf
        .bloch_density
        .as_ref()
        .map(|b| crate::scf_pbc::scale_blocks(b, 0.5));
    let (pa_flat, pb_flat) = flat_spin_densities(scf);
    let total = match &scf.bloch_density {
        Some(b) => crate::gradient::TranslatedDensity::Bloch(b, divisions),
        None => crate::gradient::TranslatedDensity::Uniform(&scf.density),
    };
    let (alpha, beta) = match (&bloch_spin, &half_blocks) {
        (Some((a, b)), _) if scf.unrestricted => (
            crate::gradient::TranslatedDensity::Bloch(a, divisions),
            crate::gradient::TranslatedDensity::Bloch(b, divisions),
        ),
        (_, Some(h)) => (
            crate::gradient::TranslatedDensity::Bloch(h, divisions),
            crate::gradient::TranslatedDensity::Bloch(h, divisions),
        ),
        _ => (
            crate::gradient::TranslatedDensity::Uniform(&pa_flat),
            crate::gradient::TranslatedDensity::Uniform(&pb_flat),
        ),
    };
    let spin = if scf.unrestricted {
        Some((alpha, beta))
    } else {
        None
    };

    let mut hess = skeleton_hessian(molecule, params, &basis, total, spin, &pbc)?;
    if pbc.mode == PbcMode::Ewald {
        let lr = long_range_hessian(molecule, &basis, &scf.charges, &[alpha, beta], &pbc)?;
        for (dst, src) in hess.as_mut_slice().iter_mut().zip(lr.as_slice()) {
            *dst += *src;
        }
    }
    let relax = relaxation_hessian(molecule, params, options, scf, &basis, &pbc)?;
    for (dst, src) in hess.as_mut_slice().iter_mut().zip(relax.as_slice()) {
        *dst += *src;
    }
    crate::gradient::add_correction_hessian_periodic(molecule, options, &mut hess);

    let ndof = hess.rows;
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(sym)
}

/// The orbital-relaxation (CPHF) half of the Γ-point Hessian, `H_relax[a][b] = 4 G^a : U^b`.
///
/// Same compact occupied–virtual formulation as the molecular case; the two periodic differences
/// are that `G` comes from [`derivative_fock`] (which carries the Ewald terms) and that the
/// response kernel is a **difference** of Fock builds:
///
/// ```text
/// G(ΔP) = F(P + ΔP) − F(P)
/// ```
///
/// The molecular shortcut `F(ΔP) − H_core` is only valid because a molecular Fock is linear in
/// the density. A periodic one is *affine*: the Ewald potential is built from `q_A = Z_A − P_A`,
/// and feeding it `ΔP` alone would drag the nuclear charges along.
fn relaxation_hessian(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    scf: &Pm7Result,
    basis: &Basis,
    pbc: &PbcOptions,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut hess = Matrix::zeros(ndof, ndof);
    let n_occ = scf.n_occ;
    let nvir = basis.nao - n_occ;
    if nvir == 0 || n_occ == 0 {
        return Ok(hess);
    }
    if scf.unrestricted {
        return Err(crate::error::Pm7Error::InvalidInput(
            "the periodic analytic Hessian has no unrestricted CPHF yet; run the closed-shell \
             cell, or use `numerical_hessian`, which differentiates the (already unrestricted) \
             analytic gradient"
                .into(),
        ));
    }
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        basis,
        params,
        options.force_dpath,
        pbc,
        options.active_field(),
    )?;
    let p = &scf.density;
    let cv = crate::hessian::submatrix_cols(&scf.mo_coeff, n_occ, nvir);
    let co = crate::hessian::submatrix_cols(&scf.mo_coeff, 0, n_occ);
    let denom = crate::hessian::ov_denominators(&scf.mo_energies, n_occ, nvir);

    let dfock = derivative_fock(molecule, params, basis, p, &scf.charges, pbc)?;
    let gov: Vec<Matrix> = dfock
        .iter()
        .map(|f| crate::hessian::project_ov(f, &cv, &co))
        .collect();

    // The reference Fock at the converged density; subtracting it makes the response kernel
    // exactly linear in `ΔP` whether or not an Ewald term is present.
    let f_reference = crate::fock::build_fock(molecule, basis, params, &core, p)?;
    let kernel = |dp: &Matrix| -> Result<Matrix> {
        let mut shifted = p.clone();
        for (s, d) in shifted.as_mut_slice().iter_mut().zip(dp.as_slice()) {
            *s += *d;
        }
        let mut g = crate::fock::build_fock(molecule, basis, params, &core, &shifted)?;
        for (gv, fv) in g.as_mut_slice().iter_mut().zip(f_reference.as_slice()) {
            *gv -= *fv;
        }
        Ok(g)
    };

    let uov: Vec<Matrix> = gov
        .par_iter()
        .map(|g| {
            crate::hessian::cphf_ov_with_kernel(
                g,
                &denom,
                &cv,
                &co,
                &kernel,
                options.cphf_max_iterations,
            )
        })
        .collect::<Result<Vec<_>>>()?;

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
    let _ = options;
    Ok(hess)
}

/// α and β density matrices reconstructed from the total and spin densities.
fn flat_spin_densities(scf: &Pm7Result) -> (Matrix, Matrix) {
    let mut pa = scf.density.clone();
    let mut pb = scf.density.clone();
    match &scf.spin_density {
        None => {
            for v in pa.as_mut_slice() {
                *v *= 0.5;
            }
            for v in pb.as_mut_slice() {
                *v *= 0.5;
            }
        }
        Some(s) => {
            let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
            let (pts, ss) = (scf.density.as_slice(), s.as_slice());
            for i in 0..pts.len() {
                pas[i] = 0.5 * (pts[i] + ss[i]);
                pbs[i] = 0.5 * (pts[i] - ss[i]);
            }
        }
    }
    (pa, pb)
}
