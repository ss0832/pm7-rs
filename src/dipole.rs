// SPDX-License-Identifier: GPL-3.0-or-later

//! The NDDO dipole operator, and the dipole moment built from it.
//!
//! In the ZDO approximation only two terms survive: the **point-charge** term, which is
//! independent of the parameterization, and the **one-centre hybridization** term from matrix
//! elements of the form `⟨ns|r|np⟩` and `⟨np|r|nd⟩`, which depends on the Slater exponents through
//! the charge separations `dd` and `ddp(5)`. This mirrors MOPAC's `src/properties/dipole.F90`.
//!
//! One operator serves three callers, which is the point of this module:
//!
//! * [`crate::field`] — an external field couples as `Σ_a f_a D_a`, so the field Hamiltonian *is*
//!   this operator contracted with the field;
//! * the dipole moment reported on [`crate::scf::Pm7Result`];
//! * `∂mu/∂R` for IR intensities, which contracts the same `D_a` against the CPHF response
//!   density.
//!
//! Building them separately is how the field and the dipole drift out of step, which — see
//! [`DipoleTerms`] — is a mistake MOPAC itself makes.

use crate::basis::Basis;
use crate::constants::AU_DIPOLE_TO_DEBYE;
use crate::data_tables::MASS;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// Which one-centre hybridization terms the dipole operator carries.
///
/// These are not interchangeable, and the difference is not academic: **MOPAC's external-field
/// operator and MOPAC's printed dipole disagree**. `hcore.F90:191-202` gives the field an s–p
/// term only, while `dipole.F90:118-148` gives the dipole an additional p–d term. So in MOPAC's
/// own convention `∂E/∂F` equals [`DipoleTerms::FieldConjugate`], not [`DipoleTerms::Full`], for
/// any molecule containing an atom with d orbitals.
///
/// `pm7-rs` reproduces that rather than quietly repairing it, and names the two operators so a
/// cross-check can compare like with like. See `docs/theory.md` (conventions C-1 and C-2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DipoleTerms {
    /// `Σ_A q_A R_A` alone. Diagnostic; no implementation uses it for physics.
    PointCharge,
    /// Point charge plus the s–p hybrid. **What MOPAC's `FIELD=` conjugates to.**
    FieldConjugate,
    /// Point charge plus the s–p and p–d hybrids. **MOPAC's printed dipole**, and the physical
    /// operator: this is what IR intensities and Born effective charges use.
    #[default]
    Full,
}

impl DipoleTerms {
    const fn has_sp(self) -> bool {
        !matches!(self, Self::PointCharge)
    }

    const fn has_pd(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Origin for the **point-charge** part of a dipole moment.
///
/// The hybrid parts are one-centre and origin-independent by construction, so this never touches
/// them. For a neutral molecule the choice cannot change the answer either, because
/// `Σ_A q_A (R_A − c) = Σ_A q_A R_A − c Σ_A q_A` and the second term vanishes.
///
/// It matters only for an ion, where the dipole is not a physical observable at all without
/// stating an origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DipoleOrigin {
    /// The input coordinate origin.
    ///
    /// For an ion this reports a number that depends on where the molecule happens to sit, which
    /// is why it is no longer the default. Kept because it is what `pm7-rs` ≤ 0.2.0 always did,
    /// and because it is the right choice when a caller has deliberately placed the origin.
    Coordinates,
    /// The centre of mass — **the default**, and what MOPAC does (`dipole.F90:86-103`).
    ///
    /// MOPAC recentres only outside a `FORCE`/`THERMO`/`IRC` run, because a moving origin would
    /// make dipole *derivatives* convention-dependent. `pm7-rs` reaches the same place through
    /// convention C-8 instead: `∂mu/∂R` ignores this setting entirely, so the reported dipole can
    /// be recentred without the IR intensities noticing.
    ///
    /// This changes nothing for a neutral molecule — [`dipole_origin`] short-circuits to the
    /// coordinate origin there, bit for bit.
    #[default]
    CentreOfMass,
    /// The centre of nuclear charge, `Σ_A Z_A R_A / Σ_A Z_A`.
    CentreOfCharge,
}

/// A dipole moment split into the terms it is made of, in **Debye**.
///
/// Mirrors MOPAC's `POINT-CHG. / HYBRID / SUM` print, with the hybrid split further so the p–d
/// term — the one that distinguishes [`DipoleTerms::Full`] from [`DipoleTerms::FieldConjugate`] —
/// is visible rather than buried.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DipoleBreakdown {
    /// `Σ_A q_A (R_A − origin)`.
    pub point_charge: Vec3,
    /// The one-centre s–p hybridization term. Origin-independent.
    pub sp_hybrid: Vec3,
    /// The one-centre p–d hybridization term. Zero unless some atom carries d orbitals.
    pub pd_hybrid: Vec3,
    /// The origin the point-charge part was taken about, in Bohr.
    pub origin: Vec3,
}

impl DipoleBreakdown {
    /// The full physical dipole — [`DipoleTerms::Full`].
    pub fn total(&self) -> Vec3 {
        self.point_charge + self.sp_hybrid + self.pd_hybrid
    }

    /// Point charge plus s–p only: the dipole MOPAC's `FIELD=` operator conjugates to, so that
    /// `∂E/∂F` equals **this** and not [`Self::total`]. See [`DipoleTerms`].
    pub fn field_conjugate(&self) -> Vec3 {
        self.point_charge + self.sp_hybrid
    }

    pub fn magnitude(&self) -> f64 {
        self.total().norm()
    }
}

/// MOPAC's `chargd` test (`dipole.F90:85`): below this the molecule counts as neutral and the
/// origin cannot matter.
const NEUTRAL_TOLERANCE: f64 = 0.5;

/// The origin to measure the point-charge dipole about.
///
/// Returns exactly `Vec3::zero()` for a neutral molecule whatever `choice` says. That is not an
/// optimization: subtracting a computed centre and re-summing would perturb the last bit of every
/// existing neutral-molecule dipole for no change in the mathematics, so the short-circuit is what
/// keeps published numbers reproducible under a setting that provably cannot affect them.
pub fn dipole_origin(
    molecule: &Molecule,
    params: &Pm7Parameters,
    charges: &[f64],
    choice: DipoleOrigin,
) -> Result<Vec3> {
    let net: f64 = charges.iter().sum();
    if net.abs() <= NEUTRAL_TOLERANCE {
        return Ok(Vec3::zero());
    }
    Ok(match choice {
        DipoleOrigin::Coordinates => Vec3::zero(),
        DipoleOrigin::CentreOfMass => {
            let mut centre = Vec3::zero();
            let mut total = 0.0;
            for atom in &molecule.atoms {
                let mass = MASS[atom.z as usize];
                centre += atom.position * mass;
                total += mass;
            }
            if total > 0.0 {
                centre * (1.0 / total)
            } else {
                Vec3::zero()
            }
        }
        DipoleOrigin::CentreOfCharge => {
            let mut centre = Vec3::zero();
            let mut total = 0.0;
            for atom in &molecule.atoms {
                let z = params.element(atom.z)?.core_charge;
                centre += atom.position * z;
                total += z;
            }
            if total > 0.0 {
                centre * (1.0 / total)
            } else {
                Vec3::zero()
            }
        }
    })
}

/// `1/sqrt(3)`, MOPAC's `xt` in the p–d block.
const INV_SQRT3: f64 = 0.577_350_269_189_625_8;

/// The **electronic** dipole operator in the AO basis, in e·Bohr, as three `nao × nao` matrices.
///
/// The electronic dipole is `mu_a^elec = Tr[P D_a]`, and the total dipole adds the nuclei:
///
/// ```text
/// mu_a = Σ_A Z_A (R_A − origin)_a  +  Tr[P D_a]
/// ```
///
/// **The factor of two matters.** `mu_sp = Σ_A (−2 dd_A) P_{s,p}` sums a *single* triangle
/// element, while `Tr[P D]` picks up both, so the stored entry is `−dd_A`, not `−2 dd_A`.
///
/// This is also the operator an external field couples to: `h^field = Σ_a f_a D_a` with
/// `origin = 0` and `terms = FieldConjugate` reproduces MOPAC's `hcore.F90:186-206` exactly. See
/// [`crate::field`].
pub fn dipole_operator(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    origin: Vec3,
    terms: DipoleTerms,
) -> Result<[Matrix; 3]> {
    let nao = basis.nao;
    let mut d = [
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
    ];

    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let r = atom.position - origin;

        // Point charge: the electron sits on its atom, so the position operator is diagonal.
        for axis in 0..3 {
            let value = -r.get(axis);
            for mu in 0..n {
                d[axis][(off + mu, off + mu)] = value;
            }
        }

        if terms.has_sp() && elem.has_p() {
            // ⟨s|r_k|p_k⟩ = −dd_A, symmetric in the two indices.
            for k in 0..3 {
                let value = -elem.dd;
                d[k][(off, off + 1 + k)] += value;
                d[k][(off + 1 + k, off)] += value;
            }
        }

        if terms.has_pd() && elem.has_d() {
            // MOPAC `dipole.F90:118-148`. Within-atom offsets are
            // `0 s, 1 px, 2 py, 3 pz, 4 d_x²−y², 5 d_xz, 6 d_z², 7 d_yz, 8 d_xy`, which is
            // MOPAC's own order (`overlap_d.rs:10`), so every index transcribes directly.
            let hyb = -elem.ddp_pd();
            const PD: [[(usize, usize, f64); 4]; 3] = [
                // x: p_z·d_xz + p_x·d_{x²−y²} + p_y·d_xy − (1/√3) p_x·d_z²
                [(5, 3, 1.0), (4, 1, 1.0), (8, 2, 1.0), (6, 1, -INV_SQRT3)],
                // y: p_z·d_yz − p_y·d_{x²−y²} + p_x·d_xy − (1/√3) p_y·d_z²
                [(7, 3, 1.0), (4, 2, -1.0), (8, 1, 1.0), (6, 2, -INV_SQRT3)],
                // z: p_x·d_xz + p_y·d_yz + (2/√3) p_z·d_z²
                [
                    (5, 1, 1.0),
                    (7, 2, 1.0),
                    (6, 3, 2.0 * INV_SQRT3),
                    (0, 0, 0.0),
                ],
            ];
            for (axis, entries) in PD.iter().enumerate() {
                for &(i, j, coefficient) in entries {
                    if coefficient == 0.0 {
                        continue;
                    }
                    let value = hyb * coefficient;
                    d[axis][(off + i, off + j)] += value;
                    d[axis][(off + j, off + i)] += value;
                }
            }
        }
    }
    Ok(d)
}

/// The dipole moment, split into its terms, in Debye.
///
/// `charges` are the Mulliken net charges `Z_A − P_AA` the caller has already formed.
///
/// A periodic system keeps its point-charge term **only along non-periodic directions**.
///
/// Along a lattice vector the point-charge sum is not an observable at all: shifting an atom by
/// one translation changes it, so the polarization needs a Berry-phase treatment this code does
/// not do. Along a direction with no lattice vector — a chain's transverse axes, a slab's normal —
/// the dipole per cell is perfectly well defined, and it is exactly what the transverse Born
/// charges and the finite-field cross-checks measure. Zeroing all three components treated a 1-D
/// chain like a crystal and threw away a real number; zeroing none of them would report a
/// meaningless one. The mask is per axis for that reason.
///
/// The hybrid terms are one-centre and need no such care.
pub fn breakdown(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    density: &Matrix,
    charges: &[f64],
    origin_choice: DipoleOrigin,
    periodic: bool,
) -> Result<DipoleBreakdown> {
    let origin = if periodic {
        Vec3::zero()
    } else {
        dipole_origin(molecule, params, charges, origin_choice)?
    };

    // Which Cartesian axes a lattice vector reaches along. `false` everywhere for a molecule.
    let mut constrained = [false; 3];
    if periodic {
        if let Some(cell) = molecule.cell {
            for lattice in cell.vectors() {
                for (axis, flag) in constrained.iter_mut().enumerate() {
                    if lattice.get(axis).abs() > 1.0e-12 {
                        *flag = true;
                    }
                }
            }
        }
    }

    let mut out = DipoleBreakdown {
        origin,
        ..Default::default()
    };

    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let c = (atom.position - origin) * charges[ia];
        let keep = |axis: usize, value: f64| if constrained[axis] { 0.0 } else { value };
        out.point_charge += Vec3::new(keep(0, c.x), keep(1, c.y), keep(2, c.z));
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];

        if elem.has_p() {
            let hyb = -2.0 * elem.dd;
            out.sp_hybrid += Vec3::new(
                hyb * density[(off, off + 1)],
                hyb * density[(off, off + 2)],
                hyb * density[(off, off + 3)],
            );
        }

        if elem.has_d() {
            // MOPAC `dipole.F90:118-148`; see `dipole_operator` for the index map.
            let p = |i: usize, j: usize| density[(off + i, off + j)];
            let hyb = -2.0 * elem.ddp_pd();
            let dx = p(5, 3) + p(4, 1) + p(8, 2) - INV_SQRT3 * p(6, 1);
            let dy = p(7, 3) - p(4, 2) + p(8, 1) - INV_SQRT3 * p(6, 2);
            let dz = p(5, 1) + p(7, 2) + 2.0 * INV_SQRT3 * p(6, 3);
            out.pd_hybrid += Vec3::new(hyb * dx, hyb * dy, hyb * dz);
        }
    }

    // e·Bohr → Debye, componentwise, after the sums so the conversion is applied once.
    out.point_charge = out.point_charge * AU_DIPOLE_TO_DEBYE;
    out.sp_hybrid = out.sp_hybrid * AU_DIPOLE_TO_DEBYE;
    out.pd_hybrid = out.pd_hybrid * AU_DIPOLE_TO_DEBYE;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::{run_pm7, Pm7Options};
    use crate::system::Atom;

    fn water() -> Molecule {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.96 * a, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.24 * a, 0.93 * a, 0.0),
            },
        ])
    }

    fn hydrogen_sulfide() -> Molecule {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Molecule::new(vec![
            Atom {
                z: 16,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.34 * a, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.35 * a, 1.29 * a, 0.0),
            },
        ])
    }

    /// `Tr[P D]` must reproduce the closed-form breakdown. This is what licenses using one
    /// operator for the field, the dipole and the IR intensities instead of three expressions.
    #[test]
    fn the_operator_trace_reproduces_the_closed_form_dipole() {
        let params = Pm7Parameters::standard().unwrap();
        for molecule in [water(), hydrogen_sulfide()] {
            let options = Pm7Options::default();
            let result = run_pm7(&molecule, &params, &options).unwrap();
            let basis = Basis::build(&molecule, &params).unwrap();
            let parts = breakdown(
                &molecule,
                &basis,
                &params,
                &result.density,
                &result.charges,
                DipoleOrigin::Coordinates,
                false,
            )
            .unwrap();

            let d = dipole_operator(&molecule, &basis, &params, Vec3::zero(), DipoleTerms::Full)
                .unwrap();
            for axis in 0..3 {
                // Nuclei + Tr[P D]; `charges` already carries `Z_A − P_AA`, so the nuclear term
                // is rebuilt here from the core charges directly.
                let mut nuclear = 0.0;
                for atom in &molecule.atoms {
                    nuclear +=
                        params.element(atom.z).unwrap().core_charge * atom.position.get(axis);
                }
                let electronic = result.density.frobenius_dot(&d[axis]);
                let via_operator = (nuclear + electronic) * AU_DIPOLE_TO_DEBYE;
                assert!(
                    (via_operator - parts.total().get(axis)).abs() < 1.0e-10,
                    "axis {axis}: operator {via_operator} vs closed form {}",
                    parts.total().get(axis)
                );
            }
        }
    }

    /// The p–d term is exactly the difference between the two operators, and it is zero for an
    /// sp-only molecule — the statement that makes convention C-2 checkable.
    #[test]
    fn the_pd_term_is_the_gap_between_the_two_operators() {
        let params = Pm7Parameters::standard().unwrap();
        let options = Pm7Options::default();

        let result = run_pm7(&water(), &params, &options).unwrap();
        let basis = Basis::build(&water(), &params).unwrap();
        let parts = breakdown(
            &water(),
            &basis,
            &params,
            &result.density,
            &result.charges,
            DipoleOrigin::Coordinates,
            false,
        )
        .unwrap();
        assert_eq!(parts.pd_hybrid, Vec3::zero(), "water has no d orbitals");
        assert_eq!(parts.total(), parts.field_conjugate());

        let result = run_pm7(&hydrogen_sulfide(), &params, &options).unwrap();
        let basis = Basis::build(&hydrogen_sulfide(), &params).unwrap();
        let parts = breakdown(
            &hydrogen_sulfide(),
            &basis,
            &params,
            &result.density,
            &result.charges,
            DipoleOrigin::Coordinates,
            false,
        )
        .unwrap();
        assert!(
            parts.pd_hybrid.norm() > 1.0e-6,
            "H2S must carry a p–d dipole term; got {:?}",
            parts.pd_hybrid
        );
        assert!((parts.total() - parts.field_conjugate() - parts.pd_hybrid).norm() < 1.0e-14);
    }

    /// A neutral molecule's dipole cannot depend on the origin, and the short-circuit in
    /// [`dipole_origin`] makes that hold **bit for bit** rather than to rounding.
    #[test]
    fn a_neutral_dipole_is_bit_identical_across_origins() {
        let params = Pm7Parameters::standard().unwrap();
        let result = run_pm7(&water(), &params, &Pm7Options::default()).unwrap();
        let basis = Basis::build(&water(), &params).unwrap();
        let of = |choice| {
            breakdown(
                &water(),
                &basis,
                &params,
                &result.density,
                &result.charges,
                choice,
                false,
            )
            .unwrap()
            .total()
        };
        let reference = of(DipoleOrigin::Coordinates);
        for choice in [DipoleOrigin::CentreOfMass, DipoleOrigin::CentreOfCharge] {
            let other = of(choice);
            assert_eq!(reference.x.to_bits(), other.x.to_bits(), "{choice:?}");
            assert_eq!(reference.y.to_bits(), other.y.to_bits(), "{choice:?}");
            assert_eq!(reference.z.to_bits(), other.z.to_bits(), "{choice:?}");
        }
    }
}
