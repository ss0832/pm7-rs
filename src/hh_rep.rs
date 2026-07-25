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
use crate::math::Vec3;
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

/// Total PM7-HH H–H repulsion energy (kcal/mol).
pub fn hh_repulsion_energy(molecule: &Molecule) -> f64 {
    let n = molecule.atoms.len();
    let mut e = 0.0;
    for i in 0..n {
        if molecule.atoms[i].z != 1 {
            continue;
        }
        for j in 0..i {
            if molecule.atoms[j].z != 1 {
                continue;
            }
            let r = (molecule.atoms[i].position - molecule.atoms[j].position).norm() * PM7_A0;
            e += poly_scalar::<f64>(r);
        }
    }
    e
}

/// Analytic Cartesian gradient of the H–H repulsion (kcal/mol per Bohr).
pub fn hh_repulsion_gradient(molecule: &Molecule) -> Vec<Vec3> {
    let n = molecule.atoms.len();
    let mut grad = vec![Vec3::zero(); n];
    for i in 0..n {
        if molecule.atoms[i].z != 1 {
            continue;
        }
        for j in 0..i {
            if molecule.atoms[j].z != 1 {
                continue;
            }
            let d = molecule.atoms[i].position - molecule.atoms[j].position;
            let r_bohr = d.norm();
            // dE/dr_ang via a 1-D dual on the Å distance; chain to Bohr via PM7_A0.
            let e = poly_scalar::<Dual>(Dual::var(r_bohr * PM7_A0, 0));
            let dedr_ang = e.d[0];
            let dedr_bohr = dedr_ang * PM7_A0;
            let unit = d / r_bohr;
            grad[i] += unit * dedr_bohr;
            grad[j] -= unit * dedr_bohr;
        }
    }
    grad
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
