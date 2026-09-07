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
use crate::math::{Mat3, Vec3};
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

/// Sharpness of the smooth neighbour count, per unit of `r_cut/r − 1`.
///
/// This sets the **width** of the transition, and it is the only thing that does. At a C–H contact
/// (threshold 1.391 Å) the weight runs 0.984 at 1.20 Å, 0.688 at 1.35 Å, 0.500 at the threshold
/// and 0.131 at 1.50 Å — a crossing about 0.1 Å wide, which is broad enough that no force spike
/// replaces the step it removes.
///
/// It does **not** control fidelity to the discrete rule. At an equilibrium C–H of 1.09 Å the
/// weight is 1 − 7.6e-4, so a methane carbon counts 3.99696 rather than 4, and raising this
/// constant to 100 only moves that to 4.00000 while leaving the C6 error unchanged at 0.18 %.
/// That error is the switch below, not the counting, and it was worth measuring before turning a
/// knob that could not have fixed it.
const COUNT_SHARPNESS: f64 = 26.0;

/// Sharpness of the C6 switch, per neighbour.
///
/// This is what sets **fidelity** to the discrete rule, and it needs to be steep. Centred at 3.5
/// neighbours, between the saturated 4 and unsaturated 3 that MOPAC distinguishes:
///
/// | sharpness | C6 at count 4 | error against the discrete 0.95 |
/// |---|---|---|
/// | 12 | 0.951731 | 1.7e-3 |
/// | 25 | 0.950003 | 2.6e-6 |
/// | **40** | **0.950000001** | **1.4e-9** |
///
/// At 40 an integer count reproduces the discrete coefficient to nine digits, so the smooth path
/// differs from the discrete one only where the discrete one has no derivative. Steepness here
/// costs nothing in smoothness: the count it acts on is already spread over ~0.1 Å, so the
/// composite is differentiable in the geometry however sharp this is in count space.
const C6_SWITCH_SHARPNESS: f64 = 40.0;

/// A neighbour's smooth contribution to a coordination number, and its derivative in `r`.
///
/// `1/(1 + exp(−k(r_cut/r − 1)))`, which is 1 well inside the threshold, ½ at it, and 0 outside.
/// The reciprocal argument rather than a plain difference is what keeps the transition width
/// proportional to the bond length rather than fixed in Bohr.
#[inline]
fn count_weight(r: f64, r_cut: f64) -> (f64, f64) {
    let x = r_cut / r - 1.0;
    let e = (-COUNT_SHARPNESS * x).exp();
    if !e.is_finite() {
        // Far outside the threshold the exponential overflows; the weight and its derivative are
        // both zero there, which is the limit rather than an approximation of it.
        return (0.0, 0.0);
    }
    let w = 1.0 / (1.0 + e);
    // dw/dr = w(1−w) · k · d(r_cut/r)/dr = −w(1−w) k r_cut / r²
    (w, -w * (1.0 - w) * COUNT_SHARPNESS * r_cut / (r * r))
}

/// `C6` as a **continuous** function of a smooth neighbour count, and `dC6/dcount`.
///
/// The discrete [`c6_atom`] is a step: a carbon with four neighbours takes 0.95 and anything else
/// 1.65, so the energy jumps by 74 % of the carbon dispersion coefficient when a distance crosses
/// `1.3(r_i + r_j)` and the force is undefined there. Measured on a methane whose fourth C–H bond
/// is stretched across the threshold, the post-SCF correction steps by **0.086 meV** against a
/// smooth trend of 0.0002 meV per 0.002 Å — 430 times the local slope, at a single point.
///
/// That is small, and it is a step. An optimizer that lands on it sees a force that does not
/// describe the energy either side, and an MD run that crosses it repeatedly injects energy.
#[inline]
pub fn c6_atom_smooth(z: u8, count: f64) -> (f64, f64) {
    if z != 6 {
        return (C6[z as usize], 0.0);
    }
    let s = 1.0 / (1.0 + (-(count - 3.5) * C6_SWITCH_SHARPNESS).exp());
    let span = 0.95 - 1.65;
    (1.65 + span * s, span * s * (1.0 - s) * C6_SWITCH_SHARPNESS)
}

/// Smooth coordination numbers, the continuous counterpart of [`bond_counts`].
///
/// Same threshold, same pair enumeration; only the step is replaced by [`count_weight`]. At a
/// relaxed geometry every weight is within 1e-7 of 0 or 1, so this agrees with the integer count
/// to well past the precision anything downstream needs.
pub fn smooth_counts(molecule: &Molecule) -> Vec<f64> {
    let n = molecule.atoms.len();
    let mut counts = vec![0.0_f64; n];
    let cut = |i: usize, j: usize| {
        1.3 * (crate::constants::covalent_radius_angstrom(molecule.atoms[i].z)
            + crate::constants::covalent_radius_angstrom(molecule.atoms[j].z))
            / PM7_A0
    };
    match molecule.cell {
        None => {
            for i in 0..n {
                for j in (i + 1)..n {
                    let r = (molecule.atoms[j].position - molecule.atoms[i].position).norm();
                    let (w, _) = count_weight(r, cut(i, j));
                    counts[i] += w;
                    counts[j] += w;
                }
            }
        }
        Some(_) => {
            let cutoff = 5.0 * crate::constants::ANGSTROM_TO_BOHR;
            for p in &crate::pbc::PairList::cached(molecule, cutoff).pairs {
                let (w, _) = count_weight(p.r, cut(p.a, p.b));
                counts[p.a] += w;
                counts[p.b] += w;
            }
        }
    }
    counts
}
/// Number of covalent neighbours of each atom (MOPAC `nbonds`), from covalent-radius
/// overlap. A saturated sp³ carbon (4 neighbours, e.g. CH₄) takes C6 = 0.95, an
/// unsaturated carbon (e.g. aromatic, 3 neighbours) takes 1.65.
///
/// **Periodic systems count neighbours across cell boundaries.** Without that, every carbon in
/// diamond would be perceived as under-coordinated and take the unsaturated C6 of 1.65 instead
/// of the saturated 0.95 — a silent 74 % error in the dispersion coefficient of exactly the
/// systems where periodicity matters. MOPAC does the same, with a minimum-image search in
/// `set_up_dentate.F90:53-59`.
pub fn bond_counts(molecule: &Molecule) -> Vec<usize> {
    let n = molecule.atoms.len();
    let mut counts = vec![0usize; n];
    let bonded = |i: usize, j: usize, r_bohr: f64| {
        let ri = crate::constants::covalent_radius_angstrom(molecule.atoms[i].z);
        let rj = crate::constants::covalent_radius_angstrom(molecule.atoms[j].z);
        r_bohr * PM7_A0 < 1.3 * (ri + rj)
    };
    match molecule.cell {
        None => {
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = (molecule.atoms[j].position - molecule.atoms[i].position).norm();
                    if bonded(i, j, d) {
                        counts[i] += 1;
                        counts[j] += 1;
                    }
                }
            }
        }
        Some(_) => {
            // A covalent contact is at most a few Angstrom, so a short image list is enough —
            // and includes an atom bonded to its own image, which happens in a 1-D chain.
            let cutoff = 5.0 * crate::constants::ANGSTROM_TO_BOHR;
            let list = crate::pbc::PairList::cached(molecule, cutoff);
            for p in &list.pairs {
                if !bonded(p.a, p.b, p.r) {
                    continue;
                }
                counts[p.a] += 1;
                counts[p.b] += 1;
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

/// Total PM7 dispersion energy (kcal/mol), summing every pair of a molecule.
pub fn dispersion_energy(molecule: &Molecule) -> f64 {
    dispersion_energy_cut(molecule, f64::INFINITY)
}

/// Dispersion energy with an explicit image cutoff (Bohr), per unit cell for a periodic system.
///
/// `−C6/R⁶` converges absolutely, so the cutoff is a truncation whose error falls off as
/// `R_cut⁻³` rather than a convergence trick. For a molecule the cutoff is ignored and every
/// pair is summed, so the molecular result is unchanged.
pub fn dispersion_energy_cut(molecule: &Molecule, cutoff: f64) -> f64 {
    use rayon::prelude::*;
    let nb = bond_counts(molecule);
    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    // Per-pair energies in parallel, summed in list order for a stable result.
    let terms: Vec<f64> = list
        .pairs
        .par_iter()
        .map(|p| {
            let (w, _) = crate::pbc::taper(p.r, r_on, cutoff);
            if w == 0.0 {
                return 0.0;
            }
            p.weight
                * w
                * pair_dispersion_scalar::<f64>(
                    molecule.atoms[p.a].z,
                    molecule.atoms[p.b].z,
                    nb[p.a],
                    nb[p.b],
                    p.r,
                )
        })
        .collect();
    terms.iter().sum()
}

/// Analytic Cartesian gradient of the dispersion energy (kcal/mol per Bohr).
pub fn dispersion_gradient(molecule: &Molecule) -> Vec<Vec3> {
    dispersion_gradient_cut(molecule, f64::INFINITY).0
}

/// Dispersion gradient and virial with an explicit image cutoff.
///
/// The virial is `Σ (∂E/∂d) ⊗ d` over the same pairs, which is the exact strain derivative
/// because the term depends on the atoms only through their separations. A self-image pair
/// contributes nothing to the gradient (both ends are the same atom) but does contribute to the
/// virial — that asymmetry is real, and getting it wrong would leave a periodic dispersion
/// stress silently short.
pub fn dispersion_gradient_cut(molecule: &Molecule, cutoff: f64) -> (Vec<Vec3>, Mat3) {
    use rayon::prelude::*;
    let nb = bond_counts(molecule);
    let n = molecule.atoms.len();
    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    let contribs: Vec<(usize, usize, Vec3, Vec3)> = list
        .pairs
        .par_iter()
        .map(|p| {
            // dE/dr via a 1-D dual on the distance.
            let e = pair_dispersion_scalar::<Dual>(
                molecule.atoms[p.a].z,
                molecule.atoms[p.b].z,
                nb[p.a],
                nb[p.b],
                Dual::var(p.r, 0),
            );
            // Product rule through the taper: d(w·e)/dr = w' e + w e'.
            let (w, dw) = crate::pbc::taper(p.r, r_on, cutoff);
            let dedr = dw * e.v + w * e.d[0];
            let dedd = p.d / p.r * (p.weight * dedr);
            (p.a, p.b, dedd, p.d)
        })
        .collect();
    let mut grad = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();
    for (i, j, dedd, d) in contribs {
        grad[i] -= dedd;
        grad[j] += dedd;
        virial = virial.plus(&Mat3::outer(dedd, d));
    }
    (grad, virial)
}

/// [`pair_dispersion_scalar`] with the two `C6` coefficients supplied directly.
///
/// The discrete path derives them from an integer bond count; the smooth path from a continuous
/// coordination number. Both end here, so there is one copy of the Slater-Kirkwood rule and the
/// Fermi damping rather than two that could drift.
pub fn pair_dispersion_with_c6<T: Scalar>(zi: u8, zj: u8, c6i: f64, c6j: f64, r: T) -> T {
    if !has_params(zi) || !has_params(zj) {
        return T::cst(0.0);
    }
    let (ni, nj) = (NEFF[zi as usize], NEFF[zj as usize]);
    // Slater–Kirkwood combination rule.
    let c6 = 2.0 * (c6i * c6i * c6j * c6j * ni * nj).powf(1.0 / 3.0)
        / ((c6i * nj * nj).powf(1.0 / 3.0) + (c6j * ni * ni).powf(1.0 / 3.0));
    let (ri, rj) = (R0[zi as usize], R0[zj as usize]);
    let r0 = (ri * ri * ri + rj * rj * rj) / (ri * ri + rj * rj) / 1000.0 * 2.0;
    let rij_nm = r * (PM7_A0 * 0.1);
    let damp_denom = ((rij_nm * (1.0 / (S * r0)) - 1.0) * (-ALPHA)).exp() + 1.0;
    let e = rij_nm.powi(6).recip() * (c6 / (1000.0 * 4.184)) / damp_denom;
    e * (-CSCALE)
}

/// `∂c6/∂c6i` and `∂c6/∂c6j` for the Slater-Kirkwood combination, divided by `c6` itself.
///
/// Returned scaled by `c6` because the pair energy is **linear** in the combined coefficient —
/// `e = K(r)·c6` with `K` independent of it — so `∂E/∂c6i = E · (∂c6/∂c6i)/c6` and the caller
/// never needs `c6` or `K` separately. That linearity is why this needs no dual number: a factor
/// the energy is proportional to has a derivative the energy already contains.
///
/// With `A = (c6i² c6j² ni nj)^⅓`, `B = (c6i nj²)^⅓`, `C = (c6j ni²)^⅓` and `c6 = 2A/(B+C)`:
///
/// ```text
/// (∂c6/∂c6i)/c6 = [⅔ − ⅓·B/(B+C)] / c6i
/// ```
#[inline]
fn slater_kirkwood_logderiv(zi: u8, zj: u8, c6i: f64, c6j: f64) -> (f64, f64) {
    let (ni, nj) = (NEFF[zi as usize], NEFF[zj as usize]);
    let b = (c6i * nj * nj).cbrt();
    let c = (c6j * ni * ni).cbrt();
    let s = b + c;
    if s <= 0.0 || c6i <= 0.0 || c6j <= 0.0 {
        return (0.0, 0.0);
    }
    (
        (2.0 / 3.0 - b / (3.0 * s)) / c6i,
        (2.0 / 3.0 - c / (3.0 * s)) / c6j,
    )
}

/// Dispersion energy with the **continuous** `C6`, per unit cell.
///
/// See [`crate::scf::Pm7Options::smooth_dispersion`] for why this is opt-in.
pub fn dispersion_energy_smooth_cut(molecule: &Molecule, cutoff: f64) -> f64 {
    use rayon::prelude::*;
    let counts = smooth_counts(molecule);
    let c6: Vec<f64> = molecule
        .atoms
        .iter()
        .zip(&counts)
        .map(|(a, &n)| c6_atom_smooth(a.z, n).0)
        .collect();
    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    let terms: Vec<f64> = list
        .pairs
        .par_iter()
        .map(|p| {
            let (w, _) = crate::pbc::taper(p.r, r_on, cutoff);
            if w == 0.0 {
                return 0.0;
            }
            p.weight
                * w
                * pair_dispersion_with_c6::<f64>(
                    molecule.atoms[p.a].z,
                    molecule.atoms[p.b].z,
                    c6[p.a],
                    c6[p.b],
                    p.r,
                )
        })
        .collect();
    terms.iter().sum()
}

/// Gradient and virial of the smooth dispersion energy, **including** the coordination chain rule.
///
/// Two passes, which is what a coordination-dependent coefficient costs and why the smooth path is
/// not a two-line change:
///
/// 1. the ordinary pair loop, which also accumulates `∂E/∂cn_i` for every atom;
/// 2. a loop over the **counting** pairs, distributing that against `∂cn_i/∂R`.
///
/// Leaving the second pass out gives a continuous energy with a gradient that is not its
/// derivative — trading a rare discontinuity for a systematic force error everywhere a
/// coordination number is in transition, which is worse and much harder to notice.
/// `tests/dispersion_smooth.rs` compares against a finite difference for exactly that reason.
pub fn dispersion_gradient_smooth_cut(molecule: &Molecule, cutoff: f64) -> (Vec<Vec3>, Mat3) {
    let n = molecule.atoms.len();
    let counts = smooth_counts(molecule);
    let c6: Vec<(f64, f64)> = molecule
        .atoms
        .iter()
        .zip(&counts)
        .map(|(a, &cn)| c6_atom_smooth(a.z, cn))
        .collect();

    let list = crate::pbc::PairList::cached(molecule, cutoff);
    let r_on = crate::pbc::taper_onset(cutoff);
    let mut grad = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();
    // `∂E/∂cn_i`, accumulated over the pairs atom `i` takes part in.
    let mut dedcn = vec![0.0_f64; n];

    for p in &list.pairs {
        let (zi, zj) = (molecule.atoms[p.a].z, molecule.atoms[p.b].z);
        let (c6i, c6j) = (c6[p.a].0, c6[p.b].0);
        let e = pair_dispersion_with_c6::<Dual>(zi, zj, c6i, c6j, Dual::var(p.r, 0));
        let (w, dw) = crate::pbc::taper(p.r, r_on, cutoff);
        let dedr = dw * e.v + w * e.d[0];
        let dedd = p.d / p.r * (p.weight * dedr);
        grad[p.a] -= dedd;
        grad[p.b] += dedd;
        virial = virial.plus(&Mat3::outer(dedd, p.d));

        // The chain-rule half. `e` is linear in the combined `c6`, so `∂E/∂c6i = E · dlog`.
        if w != 0.0 {
            let (li, lj) = slater_kirkwood_logderiv(zi, zj, c6i, c6j);
            let scaled = p.weight * w * e.v;
            dedcn[p.a] += scaled * li * c6[p.a].1;
            dedcn[p.b] += scaled * lj * c6[p.b].1;
        }
    }

    // Second pass: distribute `∂E/∂cn` through `∂cn/∂R`. The counting pair list has its own,
    // shorter cutoff — a covalent contact is a few Angstrom — and must match `smooth_counts`
    // exactly, or the gradient is of a different coordination number than the energy used.
    let count_cutoff = 5.0 * crate::constants::ANGSTROM_TO_BOHR;
    let cut = |i: usize, j: usize| {
        1.3 * (crate::constants::covalent_radius_angstrom(molecule.atoms[i].z)
            + crate::constants::covalent_radius_angstrom(molecule.atoms[j].z))
            / PM7_A0
    };
    match molecule.cell {
        None => {
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = molecule.atoms[j].position - molecule.atoms[i].position;
                    let r = d.norm();
                    let (_, dw) = count_weight(r, cut(i, j));
                    if dw == 0.0 {
                        continue;
                    }
                    let f = d / r * ((dedcn[i] + dedcn[j]) * dw);
                    grad[i] -= f;
                    grad[j] += f;
                    virial = virial.plus(&Mat3::outer(f, d));
                }
            }
        }
        Some(_) => {
            for p in &crate::pbc::PairList::cached(molecule, count_cutoff).pairs {
                let (_, dw) = count_weight(p.r, cut(p.a, p.b));
                if dw == 0.0 {
                    continue;
                }
                let f = p.d / p.r * ((dedcn[p.a] + dedcn[p.b]) * dw * p.weight);
                grad[p.a] -= f;
                grad[p.b] += f;
                virial = virial.plus(&Mat3::outer(f, p.d));
            }
        }
    }
    (grad, virial)
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
