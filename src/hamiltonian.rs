// SPDX-License-Identifier: GPL-3.0-or-later

//! Core (one-electron) Hamiltonian assembly.
//!
//! `H_core` holds the diagonal atomic energies `U_ss/U_pp`, the electron–core attraction
//! to every other atom (from the NDDO integrals), and the inter-atomic resonance
//! `H_μν = ½(β_μ + β_ν) S_μν`. The per-pair two-electron integrals are returned alongside
//! for reuse in the Fock build.

use crate::basis::Basis;
use crate::error::Result;
use crate::integrals::{pair_two_electron, PairTwoElec};
use crate::linalg::Matrix;
use crate::mndod_twocenter::pair_two_electron_d_g;
use crate::overlap::diatom_overlap;
use crate::overlap_d::diat_overlap;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// Rotated two-electron integrals for one atom pair, tagged with the ordered atom indices
/// (`a` is the heavy atom when the other is H).
pub struct PairIntegral {
    pub a: usize,
    pub b: usize,
    pub te: PairTwoElec,
}

pub struct CoreHamiltonian {
    pub h_core: Matrix,
    pub pairs: Vec<PairIntegral>,
}

/// Return the resonance β for orbital index `orb` (0 = s, 1..3 = p, 4..8 = d).
#[inline]
fn beta_of(elem: &crate::params::Pm7Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

pub fn build_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
) -> Result<CoreHamiltonian> {
    build_core_impl(molecule, basis, params, false)
}

/// Like [`build_core`] but with an explicit `force_dpath` diagnostic flag.
pub fn build_core_with(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
) -> Result<CoreHamiltonian> {
    build_core_impl(molecule, basis, params, force_dpath)
}

fn build_core_impl(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
) -> Result<CoreHamiltonian> {
    let nao = basis.nao;
    let mut h = Matrix::zeros(nao, nao);

    // Diagonal one-electron energies U_ss / U_pp / U_dd.
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h[(mu, mu)] = match ao.orb {
            0 => elem.u_ss,
            1..=3 => elem.u_pp,
            _ => elem.u_dd,
        };
    }

    use rayon::prelude::*;

    let nat = molecule.atoms.len();

    // A molecule containing any d-bearing atom is evaluated entirely in the
    // MNDO/d rotation frame (two-electron *and* overlap) so that H_core mixes no
    // frames; a pure sp molecule uses the faster sp Dewar–Sabelli–Klopman path.
    let has_any_d = force_dpath
        || molecule
            .atoms
            .iter()
            .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));

    // Enumerate atom pairs, then compute their (independent) integrals in parallel.
    let pair_indices: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let computed: Vec<(usize, usize, PairTwoElec, [[f64; 9]; 9])> = pair_indices
        .par_iter()
        .map(
            |&(u, v)| -> Result<(usize, usize, PairTwoElec, [[f64; 9]; 9])> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                // Ordered pair: heavier atom (more AOs) first.
                let (a, b) = if eu.n_orb >= ev.n_orb { (u, v) } else { (v, u) };
                let (ea, eb) = (
                    params.element(molecule.atoms[a].z)?,
                    params.element(molecule.atoms[b].z)?,
                );
                let pos_a = molecule.atoms[a].position;
                let pos_b = molecule.atoms[b].position;
                let d = pos_b - pos_a;
                let r = d.norm();
                let xij = d / r;
                let (te, s_block): (PairTwoElec, [[f64; 9]; 9]) = if has_any_d {
                    (
                        pair_two_electron_d_g::<f64>(ea, eb, [d.x, d.y, d.z]),
                        diat_overlap::<f64>(ea, eb, [d.x, d.y, d.z]),
                    )
                } else {
                    let mut s9 = [[0.0; 9]; 9];
                    let s4 = diatom_overlap(ea, pos_a, eb, pos_b)?;
                    for i in 0..4 {
                        s9[i][..4].copy_from_slice(&s4[i][..4]);
                    }
                    (pair_two_electron(ea, eb, xij, r), s9)
                };
                Ok((a, b, te, s_block))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    // Assemble H_core serially from the precomputed per-pair integrals.
    let mut pairs = Vec::with_capacity(computed.len());
    for (a, b, te, s_block) in computed {
        {
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let off_a = basis.atom_offset[a];
            let off_b = basis.atom_offset[b];
            let na = basis.atom_norb[a];
            let nb = basis.atom_norb[b];

            // Electron–core attraction: e1b onto atom a's block, e2a onto atom b's block.
            for i in 0..na {
                for j in 0..na {
                    h[(off_a + i, off_a + j)] += te.e1b[i][j];
                }
            }
            for i in 0..nb {
                for j in 0..nb {
                    h[(off_b + i, off_b + j)] += te.e2a[i][j];
                }
            }

            // Resonance β·S (inter-atomic, symmetric).
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[off_a + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[off_b + j].orb);
                    let value = 0.5 * (bi + bj) * s_block[i][j];
                    h[(off_a + i, off_b + j)] = value;
                    h[(off_b + j, off_a + i)] = value;
                }
            }

            pairs.push(PairIntegral { a, b, te });
        }
    }

    Ok(CoreHamiltonian { h_core: h, pairs })
}
