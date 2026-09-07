// SPDX-License-Identifier: GPL-3.0-or-later
//! MOPAC's molecular-mechanics corrections to the **heat of formation**.
//!
//! These are not part of the SCF and not part of the core–core repulsion. MOPAC adds them straight
//! to `atheat` in `compfg.F90:370-378`, under the comment "Add in any molecular-mechanics type
//! corrections here", so they change the heat of formation and the gradient while leaving every
//! orbital energy, every Mulliken charge and the whole energy partition untouched.
//!
//! That is exactly how they were found. The hundred-case oracle had `pm7-rs` and MOPAC disagreeing
//! on acetylene by 12.00000 kcal/mol, allene by 1.87, furan by 1.02 and silanol by 1.10, while
//! agreeing on **all eight orbital energies to five decimals and every Mulliken charge to six** —
//! and, once MOPAC was asked for its energy partition with `ENPART`, agreeing on the total energy
//! (`−237.4756 eV` against `−237.475790`) and on the nuclear–nuclear repulsion (`149.8327` against
//! `149.83265541`) as well. Same energy, different heat of formation: the difference had to be
//! something added to one and not the other.
//!
//! Two corrections apply to PM7:
//!
//! * [`c_triple_bond_energy`] — an empirical stabilization of acetylenic C–C bonds, switched on
//!   below 1.33 Å and full below 1.21 Å;
//! * [`si_o_h_energy`] — a bending penalty on Si–O–H, which PM7 alone carries.
//!
//! Two others in the same MOPAC block are deliberately **not** implemented, for different reasons:
//!
//! * `nsp2_correction` is PM6-only (`method_pm6 .and. N_3_present`), so PM7 never reaches it;
//! * the **MMOK amide correction** — `sum_dihed`, `htype·sin²(dihedral)` over every N–H–C=O
//!   linkage — is one MOPAC *does* apply by default to PM7 (`htype = 3.1595`, `moldat.F90:1517`;
//!   the check that would have demanded an explicit `MMOK`/`NOMM` keyword is commented out). It is
//!   off here **by choice**, not by oversight. It is an explicit molecular-mechanics term on a
//!   dihedral, MOPAC itself offers `NOMM` to switch it off, and a peptide energy that depends on a
//!   torsional fudge is not the semiempirical answer a caller of this crate is asking for.
//!
//!   The consequence is a **known divergence on amides**, recorded in `docs/fidelity.md`: a
//!   molecule with an N–H–C=O linkage will differ from a default MOPAC run by
//!   `3.1595 · Σ sin²(dihedral)` kcal/mol, and the two agree exactly when MOPAC is given `NOMM`.
//!   No case in the oracle set contains an amide, so the set stays exact either way.

use crate::math::{Mat3, Vec3};
use crate::system::Molecule;

/// Below this C–C separation (Å) the acetylenic correction is at full strength.
const YNE_MIN: f64 = 1.21;
/// Above this C–C separation (Å) it is off. MOPAC's comment: a C–C double bond is 1.34 Å.
const YNE_MAX: f64 = 1.33;
/// kcal/mol per acetylenic bond. MOPAC: "(The value 12 was determined empirically".
const YNE_SCALE: f64 = 12.0;
const YNE_P1: f64 = -5.0;
const YNE_P2: f64 = 25.0;

/// The per-bond weight, 1 below [`YNE_MIN`] and 0 at [`YNE_MAX`].
///
/// A quintic smoothstep plus a linear-in-`x` cubic tail, exactly as MOPAC writes it. Reproducing
/// the polynomial rather than substituting a smoother one matters: the measured discrepancies were
/// `12.00000`, `9.92044` and `2.32983` kcal/mol at 1.20, 1.25 and 1.30 Å, and this form gives
/// `12.00000`, `9.92044` and `2.32983`.
fn yne_weight(r: f64) -> f64 {
    if r < YNE_MIN {
        return 1.0;
    }
    if r >= YNE_MAX {
        return 0.0;
    }
    let x = (r - YNE_MIN) / (YNE_MAX - YNE_MIN);
    let (x3, x4, x5, x6) = (x * x * x, x.powi(4), x.powi(5), x.powi(6));
    let smooth = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
    let tail = x3 - 3.0 * x4 + 3.0 * x5 - x6;
    smooth + (YNE_P1 + YNE_P2 * x) * tail
}

/// Is this C–C pair close enough to be a bond for the acetylenic correction?
///
/// MOPAC walks its own connectivity table (`ibonds`) and then applies a weight that is zero beyond
/// 1.33 Å. Any two carbons within 1.33 Å are bonded under any criterion — a C–C double bond is
/// 1.34 — so the distance test alone selects the same set, without needing a bond perception pass
/// whose disagreements would be invisible.
fn is_carbon_pair(molecule: &Molecule, a: usize, b: usize) -> bool {
    molecule.atoms[a].z == 6 && molecule.atoms[b].z == 6
}

/// MOPAC's `C_triple_bond_C`, in **kcal/mol**.
pub fn c_triple_bond_energy(molecule: &Molecule) -> f64 {
    let n = molecule.atoms.len();
    let mut sum = 0.0;
    for a in 0..n {
        for b in (a + 1)..n {
            if !is_carbon_pair(molecule, a, b) {
                continue;
            }
            let r = (molecule.atoms[b].position - molecule.atoms[a].position).norm()
                * crate::constants::BOHR_TO_ANGSTROM;
            if r >= YNE_MAX {
                continue;
            }
            sum += yne_weight(r);
        }
    }
    sum * YNE_SCALE
}

/// The gradient of [`c_triple_bond_energy`] in kcal/mol/Bohr, with its virial.
///
/// Analytic rather than differenced, and in the same call as the virial because a periodic stress
/// needs both from one pass over the pairs.
pub fn c_triple_bond_gradient_and_virial(molecule: &Molecule) -> (Vec<Vec3>, Mat3) {
    let n = molecule.atoms.len();
    let mut grad = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();
    for a in 0..n {
        for b in (a + 1)..n {
            if !is_carbon_pair(molecule, a, b) {
                continue;
            }
            let d = molecule.atoms[b].position - molecule.atoms[a].position;
            let r_bohr = d.norm();
            let r = r_bohr * crate::constants::BOHR_TO_ANGSTROM;
            if r >= YNE_MAX || r <= YNE_MIN || r_bohr < 1.0e-12 {
                // Below `YNE_MIN` the weight is the constant 1 and its derivative is zero, which is
                // MOPAC's behaviour too — the correction is flat there, not merely large.
                continue;
            }
            let derivative =
                yne_weight_derivative(r) * YNE_SCALE * crate::constants::BOHR_TO_ANGSTROM;
            let force = d * (derivative / r_bohr);
            grad[a] -= force;
            grad[b] += force;
            virial = virial.plus(&Mat3::outer(force, d));
        }
    }
    (grad, virial)
}

/// `d(weight)/dr` in Å⁻¹, by hand because the piecewise form has no derivative at its ends anyway.
fn yne_weight_derivative(r: f64) -> f64 {
    if r <= YNE_MIN || r >= YNE_MAX {
        return 0.0;
    }
    let span = YNE_MAX - YNE_MIN;
    let x = (r - YNE_MIN) / span;
    let (x2, x3, x4, x5) = (x * x, x * x * x, x * x * x * x, x * x * x * x * x);
    let d_smooth = -30.0 * x2 + 60.0 * x3 - 30.0 * x4;
    let tail = x3 - 3.0 * x4 + 3.0 * x5 - x5 * x;
    let d_tail = 3.0 * x2 - 12.0 * x3 + 15.0 * x4 - 6.0 * x5;
    (d_smooth + YNE_P2 * tail + (YNE_P1 + YNE_P2 * x) * d_tail) / span
}

// --- Si-O-H ----------------------------------------------------------------------------------

/// The reference Si–O–H angle, in degrees. MOPAC's comment says 115 and its code says 125; the
/// code is what produces MOPAC's numbers, so the code is what is reproduced.
const SI_O_H_REFERENCE_DEGREES: f64 = 125.0;
/// kcal/mol per radian², MOPAC `Si_O_H_bond_correction`.
const SI_O_H_FORCE: f64 = 15.0;

/// Beyond these the Gaussians have switched the correction off completely.
///
/// MOPAC damps by `exp(-33 max(0, r_SiO^2 - 1.7^2))` and `exp(-68 max(0, r_OH^2 - 1.0^2))`. Its
/// comment says these reach 0.05 "when atoms are no longer considered connected", but the
/// arithmetic is far sharper than that: at `r_SiO = 2.02` the factor is `8e-18`. So a plain
/// distance test selects the same Si–O–H triples MOPAC's connectivity table would, and the cutoffs
/// below are generous enough that nothing switched on is excluded.
const SI_O_CUTOFF: f64 = 2.4;
const O_H_CUTOFF: f64 = 1.4;

/// Every `(Si, O, H)` triple the correction applies to, with the **nearest** Si and H on each O.
///
/// MOPAC keeps the last Si and last H it meets while walking `O`'s bond list, which is an ordering
/// artefact rather than a rule; nearest is the same choice for any Si–O–H that exists chemically
/// and is at least defined without reference to a bond-list order.
fn si_o_h_triples(molecule: &Molecule) -> Vec<(usize, usize, usize)> {
    let n = molecule.atoms.len();
    let mut out = Vec::new();
    for o in 0..n {
        if molecule.atoms[o].z != 8 {
            continue;
        }
        let nearest = |z: u8, cutoff: f64| -> Option<usize> {
            let mut best: Option<(f64, usize)> = None;
            for k in 0..n {
                if k == o || molecule.atoms[k].z != z {
                    continue;
                }
                let r = (molecule.atoms[k].position - molecule.atoms[o].position).norm()
                    * crate::constants::BOHR_TO_ANGSTROM;
                if r < cutoff && best.is_none_or(|(d, _)| r < d) {
                    best = Some((r, k));
                }
            }
            best.map(|(_, k)| k)
        };
        if let (Some(si), Some(h)) = (nearest(14, SI_O_CUTOFF), nearest(1, O_H_CUTOFF)) {
            out.push((si, o, h));
        }
    }
    out
}

/// One triple's contribution, in kcal/mol.
fn si_o_h_term(si: Vec3, o: Vec3, h: Vec3) -> f64 {
    let a0 = crate::constants::BOHR_TO_ANGSTROM;
    let r_si_o = (o - si).norm() * a0;
    let r_o_h = (o - h).norm() * a0;
    let damp = (-33.0 * (r_si_o * r_si_o - 1.7 * 1.7).max(0.0)).exp()
        * (-68.0 * (r_o_h * r_o_h - 1.0).max(0.0)).exp();
    let reference = SI_O_H_REFERENCE_DEGREES.to_radians();
    let angle = bond_angle(si, o, h);
    SI_O_H_FORCE * damp * (angle - reference) * (angle - reference)
}

/// The Si–O–H bond angle in radians.
fn bond_angle(a: Vec3, centre: Vec3, c: Vec3) -> f64 {
    let u = a - centre;
    let v = c - centre;
    let cosine = (u.dot(v) / (u.norm() * v.norm())).clamp(-1.0, 1.0);
    cosine.acos()
}

/// MOPAC's `Si_O_H_Correction`, in **kcal/mol**. PM7 only.
pub fn si_o_h_energy(molecule: &Molecule) -> f64 {
    si_o_h_triples(molecule)
        .into_iter()
        .map(|(si, o, h)| {
            si_o_h_term(
                molecule.atoms[si].position,
                molecule.atoms[o].position,
                molecule.atoms[h].position,
            )
        })
        .sum()
}

/// The gradient of [`si_o_h_energy`] in kcal/mol/Bohr, with its virial.
///
/// By central difference on the three atoms of each triple rather than by hand. The expression is a
/// product of a squared angle and two Gaussians in squared distances, whose analytic derivative is
/// long enough that a transcription error would be easy to make and hard to see — and the term is
/// small, local and rare, so five thousand extra evaluations of a three-atom function cost nothing
/// measurable. The step is chosen the way the crate's other finite differences are.
pub fn si_o_h_gradient_and_virial(molecule: &Molecule) -> (Vec<Vec3>, Mat3) {
    let n = molecule.atoms.len();
    let mut grad = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();
    let triples = si_o_h_triples(molecule);
    if triples.is_empty() {
        return (grad, virial);
    }
    const STEP: f64 = 1.0e-5; // Bohr
    let mut moved = molecule.clone();
    for &(si, o, h) in &triples {
        for atom in [si, o, h] {
            for axis in 0..3 {
                let original = moved.atoms[atom].position;
                let mut plus = original;
                let mut minus = original;
                match axis {
                    0 => {
                        plus.x += STEP;
                        minus.x -= STEP;
                    }
                    1 => {
                        plus.y += STEP;
                        minus.y -= STEP;
                    }
                    _ => {
                        plus.z += STEP;
                        minus.z -= STEP;
                    }
                }
                let evaluate = |p: Vec3| {
                    let mut positions = [
                        moved.atoms[si].position,
                        moved.atoms[o].position,
                        moved.atoms[h].position,
                    ];
                    if atom == si {
                        positions[0] = p;
                    } else if atom == o {
                        positions[1] = p;
                    } else {
                        positions[2] = p;
                    }
                    si_o_h_term(positions[0], positions[1], positions[2])
                };
                let derivative = (evaluate(plus) - evaluate(minus)) / (2.0 * STEP);
                match axis {
                    0 => grad[atom].x += derivative,
                    1 => grad[atom].y += derivative,
                    _ => grad[atom].z += derivative,
                }
                moved.atoms[atom].position = original;
            }
        }
    }
    // The virial of a term built only from the triple's internal geometry: `Σ_A r_A ⊗ ∂E/∂r_A`
    // relative to the oxygen, which makes it translation invariant as a virial has to be.
    for &(si, o, h) in &triples {
        for atom in [si, h] {
            let d = molecule.atoms[atom].position - molecule.atoms[o].position;
            virial = virial.plus(&Mat3::outer(grad[atom], d));
        }
    }
    (grad, virial)
}
