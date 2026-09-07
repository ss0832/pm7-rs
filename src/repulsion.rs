// SPDX-License-Identifier: GPL-3.0-or-later

//! PM7 core-core repulsion translated from MOPAC `ccrep.F90`.
//!
//! This is the PM7 NDDO core term, not a post-SCF correction: pair `alpb/xfac`
//! scaling, the MOPAC core Gaussians, and the negligible short-range `r^-12`
//! guard all belong to every PM7-family method, including PM7-minus.

use crate::constants::{PM7_A0, PM7_EV};
use crate::dual::{Dual, Scalar};
use crate::error::Result;
use crate::math::Vec3;
use crate::params::{Pm7Element, Pm7Parameters};
use crate::system::Molecule;

pub fn core_core_energy(molecule: &Molecule, params: &Pm7Parameters) -> Result<f64> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    // Compute per-pair energies in parallel; sum in pair order for a stable result.
    let terms: Result<Vec<f64>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            pair_core_energy(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                molecule.atoms[i].position,
                molecule.atoms[j].position,
                params,
            )
        })
        .collect();
    Ok(terms?.iter().sum())
}

/// Core–core repulsion of a periodic cell, per unit cell, in eV.
///
/// In [`crate::pbc::PbcMode::Ewald`] the `Z_A Z_B / r` monopole of every pair is removed here
/// because the Ewald sum supplies it for the whole lattice (through `q_A = Z_A − P_A`); what
/// remains is the short-ranged `alpb`/`xfac` scaling, the core Gaussians, and the `r⁻¹²` guard,
/// all of which have died out well before the pair cutoff.
///
/// The residual is not *identically* zero past 7 Å: an undefined pair keeps
/// `10·exp(−2.18 r)·Z_A Z_B/r`, which is ~2e-7 eV at 9 Å and falls by a further decade every
/// 1.1 Å. That is the truncation error of `PbcOptions::short_range_cutoff`, and it is why the
/// cutoff is a public, convergence-testable parameter rather than a hard-coded 7 Å.
pub fn core_core_energy_periodic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    pbc: &crate::pbc::PbcOptions,
) -> Result<f64> {
    use crate::pbc::{PairList, PbcMode};
    use rayon::prelude::*;
    let subtract_monopole = pbc.mode == PbcMode::Ewald;
    let list = PairList::cached(molecule, pbc.short_range_cutoff);
    let terms: Result<Vec<f64>> = list
        .pairs
        .par_iter()
        .map(|p| -> Result<f64> {
            let (zi, zj) = (molecule.atoms[p.a].z, molecule.atoms[p.b].z);
            let ei = params.element(zi)?;
            let ej = params.element(zj)?;
            let full = pair_core_energy_scalar(ei, ej, zi, zj, p.r, params);
            let mono = if subtract_monopole {
                ei.core_charge * ej.core_charge * PM7_EV / p.r
            } else {
                0.0
            };
            Ok(p.weight * (full - mono))
        })
        .collect();
    Ok(terms?.iter().sum())
}

/// Analytic Cartesian PM7 core-core gradient in eV/Bohr.
pub fn core_core_gradient(molecule: &Molecule, params: &Pm7Parameters) -> Result<Vec<Vec3>> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let contribs: Result<Vec<(usize, usize, Vec3)>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (_energy, dedr) = pair_core_energy_and_dr(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                molecule.atoms[i].position,
                molecule.atoms[j].position,
                params,
            )?;
            let displacement = molecule.atoms[j].position - molecule.atoms[i].position;
            let unit = displacement / displacement.norm();
            Ok((i, j, unit * dedr))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); n];
    for (i, j, force) in contribs? {
        gradient[i] -= force;
        gradient[j] += force;
    }
    Ok(gradient)
}

/// Core–core gradient and virial of a periodic cell, matching
/// [`core_core_energy_periodic`] term for term.
pub fn core_core_gradient_periodic(
    molecule: &Molecule,
    params: &Pm7Parameters,
    pbc: &crate::pbc::PbcOptions,
) -> Result<(Vec<Vec3>, crate::math::Mat3)> {
    use crate::pbc::{PairList, PbcMode};
    use rayon::prelude::*;
    let subtract_monopole = pbc.mode == PbcMode::Ewald;
    let list = PairList::cached(molecule, pbc.short_range_cutoff);
    let contribs: Result<Vec<(usize, usize, Vec3, Vec3)>> = list
        .pairs
        .par_iter()
        .map(|p| -> Result<(usize, usize, Vec3, Vec3)> {
            let (zi, zj) = (molecule.atoms[p.a].z, molecule.atoms[p.b].z);
            let ei = params.element(zi)?;
            let ej = params.element(zj)?;
            let full = pair_core_energy_scalar(ei, ej, zi, zj, Dual::var(p.r, 0), params);
            // d/dr of the removed monopole `Z_i Z_j · PM7_EV / r`.
            let dmono = if subtract_monopole {
                -ei.core_charge * ej.core_charge * PM7_EV / (p.r * p.r)
            } else {
                0.0
            };
            let dedd = p.d / p.r * (p.weight * (full.d[0] - dmono));
            Ok((p.a, p.b, dedd, p.d))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); molecule.atoms.len()];
    let mut virial = crate::math::Mat3::zero();
    for (a, b, dedd, d) in contribs? {
        gradient[a] -= dedd;
        gradient[b] += dedd;
        virial = virial.plus(&crate::math::Mat3::outer(dedd, d));
    }
    Ok((gradient, virial))
}

pub fn pair_core_energy_and_dr(
    zi: u8,
    zj: u8,
    pos_i: Vec3,
    pos_j: Vec3,
    params: &Pm7Parameters,
) -> Result<(f64, f64)> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    let distance = (pos_j - pos_i).norm();
    let dual = pair_core_energy_scalar(ei, ej, zi, zj, Dual::var(distance, 0), params);
    Ok((dual.v, dual.d[0]))
}

/// PM7 `ccrep.F90` as a generic scalar of the interatomic distance in Bohr.
///
/// Scalar generality gives the exact contribution to both the analytic gradient
/// and Hessian from the same expression used for an ordinary energy evaluation.
pub fn pair_core_energy_scalar<S: Scalar>(
    ei: &Pm7Element,
    ej: &Pm7Element,
    zi: u8,
    zj: u8,
    r_bohr: S,
    params: &Pm7Parameters,
) -> S {
    let r_angstrom = r_bohr * PM7_A0;
    let zz = ei.core_charge * ej.core_charge;
    // Core–core uses the PM7 *core* Klopman–Ohno radius po(9) (= `poc` when defined,
    // else rho0), NOT the electron-cloud rho0. MOPAC `mndod.F90:503-505`.
    let gab_nddo = (r_bohr * r_bohr + (ei.po9() + ej.po9()).powi(2))
        .sqrt()
        .recip()
        * PM7_EV;
    // PM7 feathers the core-core monopole to a bare point charge as the atoms separate,
    // exactly as the two-electron `gab` in MOPAC `reppd.F90:824` (`l_feather` is on for every
    // PM7 run).  Without it a pair such as F···F at ~3.6 Å is ~0.6 eV under-repulsive.
    let (cfrac, point) = crate::integrals::feather_to_point(r_bohr);
    let gab = gab_nddo * cfrac + point * (S::cst(1.0) - cfrac);
    let mut enuc = gab * zz;
    let pair = params.pair(zi, zj);
    let pair_defined = pair.xfac.abs() > 1.0e-5;

    if pair_defined {
        let alpb = if pair.alpb < 1.0e-6 { 1.2 } else { pair.alpb };
        let mut scale = S::cst(1.0)
            + (-(r_angstrom + r_angstrom.powi(6) * 0.0003) * alpb).exp() * (2.0 * pair.xfac);
        let (hi, lo) = (zi.max(zj), zi.min(zj));
        match (lo, hi) {
            (1, 6 | 7) => {
                scale = S::cst(1.0) + (-(r_angstrom * r_angstrom) * alpb).exp() * (2.0 * pair.xfac);
            }
            (1, 8) => {
                scale = S::cst(1.0) + (-(r_angstrom * r_angstrom) * alpb).exp() * (2.0 * pair.xfac)
                    - (r_angstrom * (-2.0 * params.vpar(4))).exp() * params.vpar(3);
            }
            (6, 6) => {
                scale = scale + (r_angstrom * (-params.vpar(2))).exp() * params.vpar(1);
            }
            (8, 14) => {
                scale = scale - (-(r_angstrom - 2.9).powi(2)).exp() * 0.0007;
            }
            _ => {}
        }
        enuc = enuc * scale;
    } else {
        let exponent = if (57..=71).contains(&zi) || (57..=71).contains(&zj) {
            -3.0
        } else {
            -2.18
        };
        let scale = (r_angstrom * exponent).exp() * 10.0;
        enuc = enuc * (S::cst(1.0) + scale);
    }

    // In MOPAC a defined pair receives the first core Gaussian for each atom;
    // an undefined pair receives that contribution plus the full four-slot loop.
    let mut correction = core_gaussian_slot(ei, ej, zz, r_angstrom, 0);
    correction = correction + core_gaussian_slot(ej, ei, zz, r_angstrom, 0);
    if !pair_defined {
        for slot in 0..4 {
            correction = correction + core_gaussian_slot(ei, ej, zz, r_angstrom, slot);
            correction = correction + core_gaussian_slot(ej, ei, zz, r_angstrom, slot);
        }
    }

    // The unpolarizable-core 12th-power guard is a PM7 base term.  It is far
    // below chemical accuracy at ordinary distances but prevents pathologies at
    // forced atom overlap, exactly as in MOPAC.
    let reduced = r_angstrom / ((zi as f64).powf(0.3333) + (zj as f64).powf(0.3333));
    if reduced.val() < 3.0 {
        let guard = reduced.powi(12).recip() * 1.0e-8;
        correction = correction
            + if guard.val() < 1.0e5 {
                guard
            } else {
                S::cst(1.0e5)
            };
    }
    enuc + correction
}

fn core_gaussian_slot<S: Scalar>(
    source: &Pm7Element,
    _other: &Pm7Element,
    zz: f64,
    r_angstrom: S,
    slot: usize,
) -> S {
    let Some(&(k, l, m)) = source.gauss.get(slot) else {
        return S::cst(0.0);
    };
    let ax = (r_angstrom - m).powi(2) * l;
    if ax.val() >= 25.0 {
        return S::cst(0.0);
    }
    (-ax).exp() * (zz * k) / r_angstrom
}

pub fn pair_core_energy(
    zi: u8,
    zj: u8,
    pos_i: Vec3,
    pos_j: Vec3,
    params: &Pm7Parameters,
) -> Result<f64> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    Ok(pair_core_energy_scalar(
        ei,
        ej,
        zi,
        zj,
        (pos_j - pos_i).norm(),
        params,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual2::Dual2;
    use crate::system::Molecule;

    #[test]
    fn analytic_core_core_gradient_matches_fd() {
        let molecule = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.02 0.05 0.0\nH -0.28 0.96 0.10\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let analytic = core_core_gradient(&molecule, &params).unwrap();
        let step = 1.0e-5;
        let mut maximum = 0.0_f64;
        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let mut plus = molecule.clone();
                let mut minus = molecule.clone();
                match axis {
                    0 => {
                        plus.atoms[atom].position.x += step;
                        minus.atoms[atom].position.x -= step;
                    }
                    1 => {
                        plus.atoms[atom].position.y += step;
                        minus.atoms[atom].position.y -= step;
                    }
                    _ => {
                        plus.atoms[atom].position.z += step;
                        minus.atoms[atom].position.z -= step;
                    }
                }
                let fd = (core_core_energy(&plus, &params).unwrap()
                    - core_core_energy(&minus, &params).unwrap())
                    / (2.0 * step);
                let value = match axis {
                    0 => analytic[atom].x,
                    1 => analytic[atom].y,
                    _ => analytic[atom].z,
                };
                maximum = maximum.max((fd - value).abs());
            }
        }
        assert!(
            maximum < 1.0e-6,
            "core-core gradient mismatch {maximum:.3e}"
        );
    }

    #[test]
    fn scalar_energy_and_second_derivative_are_consistent() {
        let params = Pm7Parameters::standard().unwrap();
        let (zi, zj) = (8u8, 1u8);
        let ei = params.element(zi).unwrap();
        let ej = params.element(zj).unwrap();
        let distance = 1.8;
        let scalar = pair_core_energy_scalar(ei, ej, zi, zj, distance, &params);
        let dual = pair_core_energy_scalar(ei, ej, zi, zj, Dual2::var(distance, 0), &params);
        let step = 1.0e-5;
        let plus = pair_core_energy_scalar(ei, ej, zi, zj, distance + step, &params);
        let minus = pair_core_energy_scalar(ei, ej, zi, zj, distance - step, &params);
        let fd2 = (plus - 2.0 * scalar + minus) / (step * step);
        assert!((dual.h[0][0] - fd2).abs() < 1.0e-3);
    }
}
