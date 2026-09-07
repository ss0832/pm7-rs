// SPDX-License-Identifier: GPL-3.0-or-later

//! PM7-HH hydrogen–hydrogen repulsion correction, ported verbatim from MOPAC
//! v23.2.5 `src/corrections/H_bonds4.F90` (`energy_corr_hh_rep` + `poly`).
//!
//! Every ordered H–H pair contributes `poly(r)` (kcal/mol), a piecewise function
//! of the H–H distance in Ångström: a constant plateau below 1 Å, a degree-5
//! polynomial fit on `[1, 1.5)` Å, and an exponential tail beyond. The exact
//! MOPAC coefficients are used; the term is generic over [`crate::dual::Scalar`]
//! so its analytic gradient follows.
//!
//! Provenance: MOPAC, Apache-2.0 (c) 2021 Virginia Tech. See `THIRD_PARTY_NOTICES.md`.

use crate::constants::PM7_A0;
use crate::dual::{Dual, Scalar};
use crate::math::{Mat3, Vec3};
use crate::system::Molecule;

/// MOPAC `poly`: the H–H repulsion for two hydrogens separated by `r` Å (kcal/mol),
/// generic over the scalar type. Branch selection uses the value; the derivative
/// within each branch is exact.
pub fn poly_scalar<S: Scalar>(r_ang: S) -> S {
    let r = r_ang.val();
    if r <= 1.0 {
        S::cst(25.462_936_031_476_93)
    } else if r < 1.5 {
        r_ang.powi(5) * (-2_714.952_351_603_469_7)
            + r_ang.powi(4) * 17_103.650_110_591_705
            + r_ang.powi(3) * (-42_511.857_982_217_96)
            + r_ang.powi(2) * 52_063.196_799_138_34
            + r_ang * (-31_430.658_335_972_29)
            + 7_516.084_696_095_14
    } else {
        (r_ang.powf(1.729_05) * (-1.539_65)).exp() * 118.7326
    }
}

/// Total PM7-HH H–H repulsion energy (kcal/mol), over every H–H pair of a molecule.
pub fn hh_repulsion_energy(molecule: &Molecule) -> f64 {
    hh_repulsion_energy_cut(molecule, f64::INFINITY)
}

/// H–H repulsion with an explicit image cutoff (Bohr), per unit cell for a periodic system.
///
/// The term is very short ranged — the exponential tail is already below 1e-6 kcal/mol past
/// about 5 Å — so any sane cutoff is exact in practice. It is still summed over images, because
/// a hydrogen near a cell face is genuinely close to its neighbour's image.
pub fn hh_repulsion_energy_cut(molecule: &Molecule, cutoff: f64) -> f64 {
    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    list.pairs
        .iter()
        .filter(|p| molecule.atoms[p.a].z == 1 && molecule.atoms[p.b].z == 1)
        .map(|p| {
            let (w, _) = crate::pbc::taper(p.r, r_on, cutoff);
            p.weight * w * poly_scalar::<f64>(p.r * PM7_A0)
        })
        .sum()
}

/// Analytic Cartesian gradient of the H–H repulsion (kcal/mol per Bohr).
pub fn hh_repulsion_gradient(molecule: &Molecule) -> Vec<Vec3> {
    hh_repulsion_gradient_cut(molecule, f64::INFINITY).0
}

/// H–H repulsion gradient and virial with an explicit image cutoff.
///
/// A self-image pair contributes nothing to the gradient — both ends are the same atom, so the
/// two contributions cancel — but it does contribute to the virial. That asymmetry is real:
/// dropping it would leave a periodic stress silently short.
pub fn hh_repulsion_gradient_cut(molecule: &Molecule, cutoff: f64) -> (Vec<Vec3>, Mat3) {
    let n = molecule.atoms.len();
    let mut grad = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();
    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    for p in &list.pairs {
        if molecule.atoms[p.a].z != 1 || molecule.atoms[p.b].z != 1 {
            continue;
        }
        // dE/dr_ang via a 1-D dual on the Å distance; chain to Bohr via PM7_A0.
        let e = poly_scalar::<Dual>(Dual::var(p.r * PM7_A0, 0));
        // Product rule through the taper (in Bohr): d(w·e)/dr = w' e + w · (de/dr).
        let (w, dw) = crate::pbc::taper(p.r, r_on, cutoff);
        let dedr = dw * e.v + w * e.d[0] * PM7_A0;
        let dedd = p.d / p.r * (p.weight * dedr);
        grad[p.a] -= dedd;
        grad[p.b] += dedd;
        virial = virial.plus(&Mat3::outer(dedd, p.d));
    }
    (grad, virial)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_matches_mopac_branches() {
        // Plateau, polynomial, and exponential branches at representative distances.
        assert!((poly_scalar::<f64>(0.8) - 25.462_936_031_476_93).abs() < 1e-9);
        // Exponential branch at 2.0 Å: 118.7326*exp(-1.53965*2^1.72905).
        let expo = 118.7326 * (-1.53965 * 2.0_f64.powf(1.72905)).exp();
        assert!((poly_scalar::<f64>(2.0) - expo).abs() < 1e-9);
        // The polynomial branch is continuous-ish and positive.
        assert!(poly_scalar::<f64>(1.25) > 0.0);
    }

    #[test]
    fn hh_gradient_matches_fd() {
        // Distances in the smooth exponential branch (r > 1.5 Å), the physically
        // relevant range; the stiff polynomial fit on [1,1.5) needs a tighter FD.
        let mol =
            Molecule::from_xyz_str("3\nH3\nH 0.0 0.0 0.0\nH 1.9 0.0 0.0\nH 0.8 2.1 0.0\n", 0.0)
                .unwrap();
        let g = hh_repulsion_gradient(&mol);
        let step = 1.0e-6;
        for atom in 0..mol.atoms.len() {
            for axis in 0..3 {
                let mut mp = mol.clone();
                let mut mm = mol.clone();
                match axis {
                    0 => {
                        mp.atoms[atom].position.x += step;
                        mm.atoms[atom].position.x -= step;
                    }
                    1 => {
                        mp.atoms[atom].position.y += step;
                        mm.atoms[atom].position.y -= step;
                    }
                    _ => {
                        mp.atoms[atom].position.z += step;
                        mm.atoms[atom].position.z -= step;
                    }
                }
                let fd = (hh_repulsion_energy(&mp) - hh_repulsion_energy(&mm)) / (2.0 * step);
                assert!(
                    (g[atom].get(axis) - fd).abs() < 1.0e-6,
                    "hh gradient mismatch atom {atom} axis {axis}"
                );
            }
        }
    }
}
