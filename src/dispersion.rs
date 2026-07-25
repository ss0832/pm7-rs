// SPDX-License-Identifier: GPL-3.0-or-later

//! PM7 post-SCF dispersion correction (D2-style), ported from MOPAC v23.2.5
//! `src/corrections/H_bond_correction_PM6_DH_Dispersion.F90`.
//!
//! `E_disp = −cscale · Σ_{i<j} C6_ij / R_ij⁶ · f_damp(R_ij)` with a Fermi damping
//! function and a Slater–Kirkwood C6 combination rule. The energy is a pairwise
//! function of the interatomic distance, so it is written generically over
//! [`crate::dual::Scalar`] and its analytic gradient/Hessian follow directly.
//!
//! Reference: Korth, Pitoňák, Řezáč, Hobza, *J. Chem. Theory Comput.* **6**, 344
//! (2010). Provenance: MOPAC, Apache-2.0 (c) 2021 Virginia Tech.

use crate::constants::PM7_A0;
use crate::dual::{Dual, Scalar};
use crate::math::Vec3;
use crate::system::Molecule;

// PM7 damping / scaling constants.
const ALPHA: f64 = 15.450118;
const S: f64 = 1.226593;
const CSCALE: f64 = 2.286419;

/// C6 dispersion coefficients (J·nm⁶/mol), index 1..=86; 0 = no parameters.
static C6: [f64; 87] = [
    0.0, 0.16, 0.084, 0.0, 0.0, 5.79, 1.65, 1.11, 0.70, 0.57, 0.45, 0.0, 0.0, 0.0, 0.0, 3.25, 5.79,
    5.97, 3.71, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    11.60, 4.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    25.80, 16.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// R0 van der Waals radii (pm), index 1..=86.
static R0: [f64; 87] = [
    0.0, 156.0, 140.0, 0.0, 0.0, 180.0, 170.0, 155.0, 152.0, 147.0, 154.0, 0.0, 0.0, 0.0, 0.0,
    180.0, 180.0, 175.0, 188.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 140.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 185.0, 202.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 198.0, 216.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Slater–Kirkwood effective electron numbers, index 1..=86.
static NEFF: [f64; 87] = [
    0.0, 0.80, 1.42, 0.0, 0.0, 2.16, 2.50, 2.82, 3.15, 3.48, 3.81, 0.0, 0.0, 0.0, 0.0, 4.50, 4.80,
    5.10, 5.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    6.00, 6.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    6.95, 7.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

#[inline]
fn has_params(z: u8) -> bool {
    let z = z as usize;
    (1..=86).contains(&z) && C6[z] != 0.0 && R0[z] != 0.0 && NEFF[z] != 0.0
}

/// Effective C6 of an atom: a carbon with four bonds (saturated sp³) uses 0.95,
/// otherwise 1.65 (MOPAC `nbonds`-dependent selection).
#[inline]
fn c6_atom(z: u8, nbonds: usize) -> f64 {
    if z == 6 {
        if nbonds == 4 {
            0.95
        } else {
            1.65
        }
    } else {
        C6[z as usize]
    }
}

/// Number of covalent neighbours of each atom (MOPAC `nbonds`), from covalent-radius
/// overlap. A saturated sp³ carbon (4 neighbours, e.g. CH₄) takes C6 = 0.95, an
/// unsaturated carbon (e.g. aromatic, 3 neighbours) takes 1.65.
pub fn bond_counts(molecule: &Molecule) -> Vec<usize> {
    let n = molecule.atoms.len();
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let ri = crate::constants::covalent_radius_angstrom(molecule.atoms[i].z);
            let rj = crate::constants::covalent_radius_angstrom(molecule.atoms[j].z);
            let d = (molecule.atoms[j].position - molecule.atoms[i].position).norm() * PM7_A0;
            if d < 1.3 * (ri + rj) {
                counts[i] += 1;
                counts[j] += 1;
            }
        }
    }
    counts
}

/// Pairwise dispersion energy `−cscale·C6/R⁶·damp` (kcal/mol) as a function of the
/// interatomic distance `r` (Bohr), generic over the scalar type. Returns 0 for a
/// pair lacking parameters.
pub fn pair_dispersion_scalar<T: Scalar>(zi: u8, zj: u8, nb_i: usize, nb_j: usize, r: T) -> T {
    if !has_params(zi) || !has_params(zj) {
        return T::cst(0.0);
    }
    let (ni, nj) = (NEFF[zi as usize], NEFF[zj as usize]);
    let (c6i, c6j) = (c6_atom(zi, nb_i), c6_atom(zj, nb_j));
    // Slater–Kirkwood combination rule.
    let c6 = 2.0 * (c6i * c6i * c6j * c6j * ni * nj).powf(1.0 / 3.0)
        / ((c6i * nj * nj).powf(1.0 / 3.0) + (c6j * ni * ni).powf(1.0 / 3.0));
    let (ri, rj) = (R0[zi as usize], R0[zj as usize]);
    // Effective R0 (pm → nm).
    let r0 = (ri * ri * ri + rj * rj * rj) / (ri * ri + rj * rj) / 1000.0 * 2.0;
    // r: Bohr → Angstrom → nm.
    let rij_nm = r * (PM7_A0 * 0.1);
    // Fermi damping denominator 1 + exp(−α·(R/(s·R0) − 1)); E ∝ 1/denominator.
    let damp_denom = ((rij_nm * (1.0 / (S * r0)) - 1.0) * (-ALPHA)).exp() + 1.0;
    let e = rij_nm.powi(6).recip() * (c6 / (1000.0 * 4.184)) / damp_denom;
    // E_disp = −cscale·Σ e.
    e * (-CSCALE)
}

/// Total PM7 dispersion energy (kcal/mol).
pub fn dispersion_energy(molecule: &Molecule) -> f64 {
    use rayon::prelude::*;
    let nb = bond_counts(molecule);
    let n = molecule.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    // Per-pair energies in parallel, summed in pair order for a stable result.
    let terms: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let r = (molecule.atoms[j].position - molecule.atoms[i].position).norm();
            pair_dispersion_scalar::<f64>(molecule.atoms[i].z, molecule.atoms[j].z, nb[i], nb[j], r)
        })
        .collect();
    terms.iter().sum()
}

/// Analytic Cartesian gradient of the dispersion energy (kcal/mol per Bohr).
pub fn dispersion_gradient(molecule: &Molecule) -> Vec<Vec3> {
    use rayon::prelude::*;
    let nb = bond_counts(molecule);
    let n = molecule.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let contribs: Vec<(usize, usize, Vec3)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let d = molecule.atoms[j].position - molecule.atoms[i].position;
            let r = d.norm();
            // dE/dr via a 1-D dual on the distance.
            let e = pair_dispersion_scalar::<Dual>(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                nb[i],
                nb[j],
                Dual::var(r, 0),
            );
            (i, j, d / r * e.d[0])
        })
        .collect();
    let mut grad = vec![Vec3::zero(); n];
    for (i, j, force) in contribs {
        grad[i] -= force;
        grad[j] += force;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispersion_is_attractive_and_small_for_water_dimer() {
        // Two water molecules ~2.9 Å apart: dispersion is negative (attractive).
        let mol = Molecule::from_xyz_str(
            "6\nwater dimer\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\nO 2.9 0.0 0.0\nH 3.2 0.9 0.0\nH 3.2 -0.9 0.0\n",
            0.0,
        )
        .unwrap();
        let e = dispersion_energy(&mol);
        assert!(e < 0.0, "dispersion should be attractive, got {e}");
        assert!(e > -10.0, "dispersion magnitude unreasonable: {e}");
    }

    #[test]
    fn pair_dispersion_exact_value() {
        // Isolated C–C at 4.0 Å (not bonded → carbon C6 = 1.65). Hand value from the
        // exact MOPAC formula: C6=1.65, R0=0.34 nm, damp=0.3479, cscale=2.286419.
        let mol = Molecule::from_xyz_str("2\nCC\nC 0.0 0.0 0.0\nC 4.0 0.0 0.0\n", 0.0).unwrap();
        let e = dispersion_energy(&mol);
        // Reference computed independently from the ported formula.
        let r_nm = 0.4;
        let r0 = 0.34;
        let damp = 1.0 / (1.0 + (-15.450118 * (r_nm / (1.226593 * r0) - 1.0)).exp());
        let ref_e = -2.286419 * 1.65 / r_nm.powi(6) * damp / (1000.0 * 4.184);
        assert!((e - ref_e).abs() < 1e-9, "disp {e} vs ref {ref_e}");
    }

    #[test]
    fn dispersion_gradient_matches_fd() {
        let mol = Molecule::from_xyz_str(
            "4\ntest\nC 0.0 0.0 0.0\nO 0.0 0.0 3.0\nN 2.5 0.0 0.5\nH 1.0 1.0 1.0\n",
            0.0,
        )
        .unwrap();
        let g = dispersion_gradient(&mol);
        let step = 1.0e-5;
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
                let fd = (dispersion_energy(&mp) - dispersion_energy(&mm)) / (2.0 * step);
                assert!(
                    (g[atom].get(axis) - fd).abs() < 1.0e-6,
                    "dispersion gradient mismatch atom {atom} axis {axis}: {} vs {fd}",
                    g[atom].get(axis)
                );
            }
        }
    }
}
