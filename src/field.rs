// SPDX-License-Identifier: GPL-3.0-or-later

//! A uniform external electric field, in MOPAC's `FIELD=(x,y,z)` convention.
//!
//! Ported from MOPAC v23.2.5 `src/integrals/hcore.F90:143-208` (the one-electron terms and the
//! nuclear term) and `src/forces/dfield.F90` (the gradient).
//!
//! # What the field costs to implement, which is almost nothing
//!
//! Three facts make this small, and all three are properties of the NDDO model rather than of
//! this implementation:
//!
//! * The field Hamiltonian **is** the dipole operator contracted with the field,
//!   `h^field = Σ_a f_a D_a` ([`crate::dipole::dipole_operator`] at `origin = 0` with
//!   [`DipoleTerms::FieldConjugate`]). Nothing new has to be integrated.
//! * The gradient is **exactly** `∂E/∂R_A = q_A f` and needs no derivative integrals at all: the
//!   hybrid term `−dd_A f_k` has no coordinate dependence, because `dd` is a parameter.
//! * The **skeleton** second derivative is identically zero, since `E_field(R; P) = Σ_A q_A(P)
//!   (f·R_A)` is *linear* in `R` at fixed density. The entire field contribution to the Hessian
//!   therefore arrives through the CPHF's perturbed `∂h/∂R`, which gains one constant diagonal
//!   block per atom.
//!
//! # Sign
//!
//! MOPAC's vector is the potential gradient, not the physical field, so `E(F) = E_0 + F·mu` and
//! `F = −E_physical`. `pm7-rs` keeps MOPAC's sign because `FIELD=` is what the number means. See
//! `docs/theory.md`, convention C-1.

use crate::basis::Basis;
use crate::constants::PM7_A0;
use crate::dipole::{dipole_operator, DipoleTerms};
use crate::error::{Pm7Error, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// A uniform external electric field in **volts per Ångström**, MOPAC's `FIELD=` units and sign.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExternalField {
    pub volts_per_angstrom: [f64; 3],
}

impl ExternalField {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            volts_per_angstrom: [x, y, z],
        }
    }

    /// MOPAC's `fldon` test (`hcore.F90:149-150`): an exact zero, not a tolerance.
    ///
    /// Exactness is the point — it is what lets a caller pass `Some(ExternalField::default())`
    /// and get bit-identical numbers to passing `None`.
    pub fn is_zero(&self) -> bool {
        self.volts_per_angstrom.iter().all(|c| *c == 0.0)
    }

    pub fn is_finite(&self) -> bool {
        self.volts_per_angstrom.iter().all(|c| c.is_finite())
    }

    /// The field in the crate's internal system: eV per (e · Bohr).
    ///
    /// `f = F · a_0`. MOPAC works in Ångström and cancels an `a0/ev` against an `ev/a0`;
    /// `pm7-rs` works in Bohr, so the conversion collapses to this single factor and `f · R` is
    /// an energy in eV with `R` in Bohr.
    pub fn internal(&self) -> Vec3 {
        Vec3::new(
            self.volts_per_angstrom[0] * PM7_A0,
            self.volts_per_angstrom[1] * PM7_A0,
            self.volts_per_angstrom[2] * PM7_A0,
        )
    }

    /// The nuclear half of the field energy, `Σ_A Z_A (f·R_A)`, in eV.
    ///
    /// MOPAC's `enuclr -= fnuc*tore(nat(i))` (`hcore.F90:208`).
    pub fn core_energy(&self, molecule: &Molecule, params: &Pm7Parameters) -> Result<f64> {
        if self.is_zero() {
            return Ok(0.0);
        }
        let f = self.internal();
        let mut total = 0.0;
        for atom in &molecule.atoms {
            total += params.element(atom.z)?.core_charge * f.dot(atom.position);
        }
        Ok(total)
    }

    /// `∂E_field/∂R_A = q_A f`, in eV/Bohr, from the Mulliken net charges.
    ///
    /// Exact and closed form — MOPAC's `dfield.F90:51-55`. There is no derivative integral here
    /// because the only coordinate dependence in the field energy is the explicit `R_A`.
    pub fn gradient(&self, charges: &[f64]) -> Vec<Vec3> {
        let f = self.internal();
        charges.iter().map(|q| f * *q).collect()
    }

    /// The one-electron field term added to the core Hamiltonian, in eV.
    ///
    /// Equal to `Σ_a f_a D_a` with `D` the [`DipoleTerms::FieldConjugate`] dipole operator about
    /// the coordinate origin — which is what makes `E_field = f · mu_FieldConjugate` an identity
    /// rather than a coincidence.
    pub fn one_electron(
        &self,
        molecule: &Molecule,
        basis: &Basis,
        params: &Pm7Parameters,
    ) -> Result<Matrix> {
        let f = self.internal();
        let d = dipole_operator(
            molecule,
            basis,
            params,
            Vec3::zero(),
            DipoleTerms::FieldConjugate,
        )?;
        let mut h = Matrix::zeros(basis.nao, basis.nao);
        for axis in 0..3 {
            let scale = f.get(axis);
            if scale == 0.0 {
                continue;
            }
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    h[(i, j)] += scale * d[axis][(i, j)];
                }
            }
        }
        Ok(h)
    }

    /// `∂h_μν/∂R_{A,k} = −f_k δ_μν` for `μ` on atom `A`, and zero elsewhere.
    ///
    /// The whole field contribution to an analytic Hessian, because the skeleton term vanishes.
    /// The s–p hybrid part contributes nothing: `dd` is a parameter, so `−dd_A f_k` does not move
    /// when the atom does.
    pub fn derivative_h(&self, basis: &Basis, atom: usize, axis: usize) -> Matrix {
        let mut h = Matrix::zeros(basis.nao, basis.nao);
        let value = -self.internal().get(axis);
        if value != 0.0 {
            let off = basis.atom_offset[atom];
            for mu in 0..basis.atom_norb[atom] {
                h[(off + mu, off + mu)] = value;
            }
        }
        h
    }
}

/// Reject a field that a periodic calculation cannot represent.
///
/// A uniform field is **not** a periodic operator: the potential `−f·r` grows without bound under
/// lattice translation, so along a periodic direction there is no bounded operator to add and the
/// question has to be asked differently — through the commutator `[H, r]`, which *is* periodic
/// (see [`crate::dfpt`]).
///
/// Along a **non-periodic** direction there is no such problem: a 1-D chain's transverse axes and
/// a slab's surface normal are ordinary finite directions, and a field there is exactly as
/// well-defined as it is for a molecule. Refusing those too would have made the finite-field
/// cross-checks of the periodic response impossible to write, which is how this was noticed.
pub fn validate_for(molecule: &Molecule, field: &ExternalField) -> Result<()> {
    if !field.is_finite() {
        return Err(Pm7Error::InvalidInput(
            "the external field components must be finite".into(),
        ));
    }
    let Some(cell) = molecule.cell else {
        return Ok(());
    };
    if field.is_zero() {
        return Ok(());
    }
    // A component "along a periodic direction" means one with a non-zero projection on any
    // lattice vector. If `f · a_i = 0` for every lattice vector then `f · T = 0` for every lattice
    // translation, so `−f·r` is itself periodic and bounded — which is exactly the case a field
    // across a chain or normal to a slab falls into.
    let f = Vec3::new(
        field.volts_per_angstrom[0],
        field.volts_per_angstrom[1],
        field.volts_per_angstrom[2],
    );
    for (axis, lattice) in cell.vectors().iter().enumerate() {
        let projection = f.dot(*lattice) / lattice.norm();
        if projection.abs() > 1.0e-12 {
            return Err(Pm7Error::InvalidInput(format!(
                "a uniform electric field has a component of {projection:.3e} V/Angstrom along \
                 periodic lattice direction {axis}, where the potential -f.r is unbounded. Use a \
                 field along a non-periodic direction, or `dfpt::dynamical_matrix_dfpt`, which \
                 takes the field through the commutator [H, r] instead."
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::constants::ANGSTROM_TO_BOHR;
    use crate::system::Atom;

    fn chain() -> Molecule {
        let a = 3.0 * ANGSTROM_TO_BOHR;
        Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }])
        .with_cell(Cell::new(&[Vec3::new(a, 0.0, 0.0)]).unwrap())
    }

    /// The field operator and the term-by-term dipole formula are **the same quantity**.
    ///
    /// This is the piece the end-to-end checks kept confusing. The field energy at a fixed density
    /// is, in `e·Bohr` and before any unit conversion,
    ///
    /// ```text
    /// E_field / f  =  Tr[P D_a] + sum_A Z_A R_{A,a}  ==  sum_A q_A R_{A,a} + mu_sp,a + mu_pd,a
    /// ```
    ///
    /// — the left side assembled from the operator [`crate::dipole::dipole_operator`] that
    /// `one_electron` contracts, the right side from the independent per-atom sums in
    /// [`crate::dipole::breakdown`]. Nothing here is approximate and nothing depends on the SCF
    /// being converged, so any mismatch is a transcription error in one of the two.
    ///
    /// It is worth its own test because the two are written out separately, on purpose (one is a
    /// matrix, one is a sum over atoms), and a discrepancy between them shows up downstream as a
    /// wrong `dE/df` — which then looks like a bug in the *response*, where it is not.
    #[test]
    fn the_field_operator_and_the_dipole_formula_agree_term_by_term() {
        use crate::dipole::{breakdown, dipole_operator, DipoleOrigin, DipoleTerms};
        let a = ANGSTROM_TO_BOHR;
        // Water for s-p, hydrogen sulfide for s-p *and* p-d: the p-d block is where a
        // transcription slip is most likely and least visible.
        for molecule in [
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
            ]),
            Molecule::new(vec![
                Atom {
                    z: 16,
                    position: Vec3::new(0.11 * a, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(1.34 * a, 0.05 * a, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(-0.33 * a, 1.29 * a, 0.07 * a),
                },
            ]),
        ] {
            let params = crate::params::Pm7Parameters::method("pm7-".parse().unwrap()).unwrap();
            let options = crate::scf::Pm7Options {
                method: "pm7-".parse().unwrap(),
                e_tol: 1.0e-12,
                ..Default::default()
            };
            let scf = crate::scf::run_pm7(&molecule, &params, &options).unwrap();
            let basis = Basis::build(&molecule, &params).unwrap();

            for terms in [DipoleTerms::FieldConjugate, DipoleTerms::Full] {
                let d = dipole_operator(&molecule, &basis, &params, Vec3::zero(), terms).unwrap();
                let split = breakdown(
                    &molecule,
                    &basis,
                    &params,
                    &scf.density,
                    &scf.charges,
                    DipoleOrigin::Coordinates,
                    false,
                )
                .unwrap();
                let to_au = 1.0 / crate::constants::AU_DIPOLE_TO_DEBYE;
                for axis in 0..3 {
                    // Operator side: Tr[P D] plus the nuclear term.
                    let mut operator = 0.0;
                    for i in 0..basis.nao {
                        for j in 0..basis.nao {
                            operator += scf.density[(i, j)] * d[axis][(i, j)];
                        }
                    }
                    for atom in &molecule.atoms {
                        operator +=
                            params.element(atom.z).unwrap().core_charge * atom.position.get(axis);
                    }
                    // Formula side, in e*Bohr.
                    let mut formula =
                        split.point_charge.get(axis) * to_au + split.sp_hybrid.get(axis) * to_au;
                    if terms == DipoleTerms::Full {
                        formula += split.pd_hybrid.get(axis) * to_au;
                    }
                    assert!(
                        (operator - formula).abs() < 1.0e-10,
                        "{terms:?} axis {axis}: Tr[P D] + Z.R = {operator:.12} but the \
                         term-by-term dipole is {formula:.12}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_internal_field_converts_volts_per_angstrom_to_ev_per_bohr() {
        let f = ExternalField::new(1.0, 0.0, 0.0).internal();
        // One volt per Ångström acting over one Bohr is `a0` electron-volts.
        assert!((f.x - PM7_A0).abs() < 1.0e-15);
    }

    #[test]
    fn a_zero_field_is_exactly_zero() {
        assert!(ExternalField::default().is_zero());
        assert!(ExternalField::new(0.0, 0.0, 0.0).is_zero());
        assert!(!ExternalField::new(0.0, 1.0e-30, 0.0).is_zero());
    }

    /// A field along the chain is refused; one across it is fine. The transverse case is what the
    /// periodic Born-charge tests need, so this is a load-bearing distinction rather than a nicety.
    #[test]
    fn a_periodic_direction_refuses_a_field_and_a_free_direction_accepts_one() {
        let along = ExternalField::new(0.5, 0.0, 0.0);
        let error = validate_for(&chain(), &along).unwrap_err().to_string();
        assert!(error.contains("unbounded"), "{error}");
        assert!(error.contains("periodic lattice direction 0"), "{error}");

        let across = ExternalField::new(0.0, 0.5, 0.0);
        assert!(validate_for(&chain(), &across).is_ok());
        assert!(validate_for(&chain(), &ExternalField::default()).is_ok());
    }
}
