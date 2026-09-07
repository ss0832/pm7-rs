// SPDX-License-Identifier: GPL-3.0-or-later

//! The rigid-body subspace of a mass-weighted Hessian, removed by projection.
//!
//! A harmonic analysis is a statement about the curvature of the energy in the directions that
//! *change* the system. Overall translation and overall rotation do not: the energy is exactly
//! invariant along them, so they carry no vibrational information and belong out of the spectrum.
//!
//! # Why they do not simply come out at zero
//!
//! Through v0.2.2 this crate removed nothing, and the module header of [`crate::hessian`] asserted
//! that translations and rotations "appear as the ~6 (5 for linear molecules) near-zero modes".
//! Half of that is a theorem and half of it is false, and the difference is why this module exists.
//!
//! Translational invariance, differentiated twice, gives `sum_B H[Aa,Bb] = 0` at **any** geometry.
//! So `H t = 0` exactly, always, and the three translations really do come out at zero.
//!
//! Rotational invariance differentiated twice gives, for a generator `u` about axis `w`,
//!
//! ```text
//! u^T H u = sum_A g_A . d_A_perp        d_A_perp = (R_A - c) - w (w . (R_A - c))
//! ```
//!
//! which vanishes only when the gradient `g` does. At any geometry that is not a stationary point
//! the rotations have **real, non-zero curvature** -- and they are then indistinguishable, by size,
//! from a soft vibration. `pm7_rs_cli frequencies examples/water.xyz` (the experimental geometry,
//! not PM7's minimum) reported 71, 121 and 181 cm^-1 for its three rotations, sitting above nothing
//! and below the bend at 1409.
//!
//! That is why the rule here is symmetry, never magnitude. A threshold that called everything under
//! some cutoff "translation or rotation" would delete a genuine soft mode from an unrelaxed
//! structure and keep a rotation from a badly unrelaxed one -- wrong in both directions for the
//! same reason: size is not what makes a mode rigid.
//!
//! # No Gram-Schmidt
//!
//! The six generators are orthogonal *for free*, which is worth setting up deliberately rather than
//! discovering with a sequential orthogonalization that loses digits:
//!
//! * `<t_a, r_w> = w . (e_a x sum_A m_A d_A) = 0` **identically**, because `sum_A m_A d_A = 0` is
//!   the definition of the centre of mass.
//! * `<r_a, r_b> = a . I . b` with `I = sum_A m_A (|d_A|^2 1 - d_A (x) d_A)` the inertia tensor. Let
//!   `w` run over the **principal axes** of `I` and the rotations are mutually orthogonal too, with
//!   `||r_k||^2 = I_k`, the principal moment.
//!
//! So the whole construction is one 3x3 eigenproblem and a normalization.
//!
//! # Linearity, decided geometrically
//!
//! `I_1 = sum_A m_A |e_1 x d_A|^2` is zero **iff** every atom lies on the line through the centre of
//! mass along `e_1` -- at which point `r_1` is the zero vector and there is nothing to normalize or
//! project. So the test is [`LINEARITY_TOLERANCE`] on `I_1 / sum_A m_A |d_A|^2`, and its legitimacy
//! under "no magnitude rules" is not a technicality:
//!
//! * It reads **only masses and coordinates**. The Hessian, the SCF and the frequencies are never
//!   consulted, so it cannot mistake a soft vibration for a rotation -- it never sees a vibration.
//! * The ratio is dimensionless and invariant under uniform scaling, rigid motion and a global mass
//!   rescaling. It is the squared sine of the collinearity defect: a molecule bent by `d` gives
//!   `O(d^2)`, so `1e-10` fires only below about `1e-5` rad. Geometries are collinear by
//!   construction or visibly bent; nothing lands in between by accident.
//! * The number the decision was made on is **reported** -- [`RigidSubspace::principal_moments`] and
//!   [`RigidSubspace::inertia_scale`] are on the returned value.
//!
//! There is no threshold-free option here. Always removing six would delete a real vibration from
//! CO2. The choice is only ever *what* the threshold measures, and geometry is the honest answer.
//!
//! # Periodic cells: one rule, every dimension
//!
//! A rotation generator `u_A = w x (R_A - c)` is a zone-centre -- that is, cell-periodic --
//! displacement field **iff** `u_{A+T} = u_A` for every lattice vector `T`, i.e.
//!
//! ```text
//! w x T = 0   for every T in the lattice
//! ```
//!
//! * **Molecule** (no `T`): vacuously true for every `w`, so 3 rotations.
//! * **1-D chain**: `w` parallel to `a`, so exactly **1**, about the chain axis.
//! * **2-D slab**: `w` would have to be parallel to two independent vectors, so **0**. In particular
//!   rotation about the slab *normal* is not a zone-centre mode: `u_A = z x R_A` changes by
//!   `z x T != 0` under an in-plane translation, which makes it a finite-`q` field.
//! * **3-D**: **0**.
//!
//! This matters because [`crate::hessian::vibrational_modes_from`] is the periodic zone-centre path
//! as well as the molecular one. A projector that removed three rotations unconditionally would
//! have deleted three of diamond's optical phonons.
//!
//! MOPAC reaches the periodic half of this conclusion by the blunter route of `if (id /= 0)`
//! (`src/forces/frame.F90:153` -- zero the rotation generators whenever the system has any lattice
//! at all), which is right for a crystal and wrong for a chain. Its Eckart construction is otherwise
//! the same one (`frame.F90:74-98`), applied as a large level shift rather than a projection;
//! `freqcy.F90:154` calls it before every diagonalization, so MOPAC's *printed* frequencies have
//! always been Eckart-treated and this crate's were not.
//!
//! # What a molecule in a box does not get
//!
//! A molecule in a large 3-D cell has librations that go to zero as the box grows, and **no
//! projector may remove them**. They are not symmetries of the periodic Hamiltonian at finite
//! separation (`w x T != 0`); they are genuine soft modes of the system as posed. This is exactly
//! the case where a magnitude threshold would have been irresistible and would have been wrong. A
//! libration and an acoustic mode cannot be told apart by size, only by symmetry -- and only the
//! translations are a symmetry. Run the molecule without a cell if you want its rotations gone.

use crate::data_tables::MASS;
use crate::error::{Pm7Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::{Mat3, Vec3};
use crate::system::Molecule;

/// Which rigid-body subspace to remove before diagonalizing a mass-weighted Hessian.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    /// Every exact rigid-body symmetry the system actually has: three translations always, plus the
    /// rotations whose generator is cell-periodic (`w x T = 0`) and non-degenerate (`I_w > 0`).
    #[default]
    Rigid,
    /// Translations only -- the acoustic sum rule and nothing else.
    Translations,
    /// Remove nothing. The raw `3N` spectrum: what v0.2.2 printed, and what shows which numbers the
    /// projector took out.
    None,
}

impl Projection {
    /// The spelling accepted on both command lines and at the Python boundary.
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "rigid" => Ok(Projection::Rigid),
            "translations" | "translation" => Ok(Projection::Translations),
            "none" | "raw" => Ok(Projection::None),
            other => Err(Pm7Error::InvalidInput(format!(
                "unknown projection `{other}`: use rigid (the default: translations plus every \
                 rotation the lattice admits), translations, or none"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Projection::Rigid => "rigid",
            Projection::Translations => "translations",
            Projection::None => "none",
        }
    }
}

/// The dimensionless collinearity test, on `I_1 / sum_A m_A |R_A - R_com|^2`.
///
/// Reads masses and coordinates only -- see the module docs for why that is what makes it a
/// legitimate criterion rather than the magnitude rule this module exists to remove.
pub const LINEARITY_TOLERANCE: f64 = 1.0e-10;

/// A lattice vector is "along" a rotation axis to this relative tolerance. Compares two *geometric*
/// quantities -- `|w x T|` against `|T|` -- so like [`LINEARITY_TOLERANCE`] it never sees an energy.
const AXIS_TOLERANCE: f64 = 1.0e-9;

/// The orthonormal rigid-body generators of a system, and the geometry they were built from.
#[derive(Clone, Debug)]
pub struct RigidSubspace {
    /// Orthonormal mass-weighted generators as the **columns** of a `3N x k` matrix: the
    /// translations first, then the rotations.
    pub generators: Matrix,
    pub n_translations: usize,
    pub n_rotations: usize,
    /// Principal moments about the centre of mass, ascending, in amu*Bohr^2.
    pub principal_moments: [f64; 3],
    /// Principal axes as rows -- the frame that makes the rotation generators orthogonal without a
    /// Gram-Schmidt pass.
    pub principal_axes: Mat3,
    /// `sum_A m_A |R_A - R_com|^2`, the scale `principal_moments` is compared against.
    pub inertia_scale: f64,
    /// True when the geometry was found collinear.
    pub linear: bool,
    /// Which principal axes survived the lattice test `w x T = 0`, in the order they appear in
    /// `generators`. Empty for a 2-D or 3-D cell.
    pub rotation_axes: Vec<Vec3>,
}

impl RigidSubspace {
    /// Build the subspace `projection` asks for, from this molecule's geometry and masses.
    pub fn of(molecule: &Molecule, projection: Projection) -> Result<Self> {
        let nat = molecule.atoms.len();
        let ndof = 3 * nat;
        let mass = |a: usize| MASS[molecule.atoms[a].z as usize];

        // A zero total mass means an atom list with no mass-table entry at all, which is a broken
        // input rather than a physical case. `is_normal` also rejects a NaN, which would otherwise
        // sail through every comparison below and come out as a subspace of silent NaNs.
        let total: f64 = (0..nat).map(mass).sum();
        if nat > 0 && !(total.is_normal() && total.is_sign_positive()) {
            return Err(Pm7Error::InvalidInput(
                "the total mass is zero, so there is no centre of mass to build the rigid-body \
                 generators about"
                    .into(),
            ));
        }
        let mut com = Vec3::new(0.0, 0.0, 0.0);
        for a in 0..nat {
            com += molecule.atoms[a].position * mass(a);
        }
        if nat > 0 {
            com = com / total;
        }
        let d: Vec<Vec3> = (0..nat).map(|a| molecule.atoms[a].position - com).collect();

        // Inertia tensor about the centre of mass, and its principal frame.
        let mut inertia = Mat3::zero();
        let mut scale = 0.0;
        for a in 0..nat {
            let m = mass(a);
            let r = d[a];
            scale += m * r.norm2();
            for i in 0..3 {
                for j in 0..3 {
                    let delta = if i == j { r.norm2() } else { 0.0 };
                    inertia.add_assign_at(i, j, m * (delta - r.get(i) * r.get(j)));
                }
            }
        }
        let (moments, axes) = principal_frame(&inertia)?;

        // A single atom is a degenerate line with no extent at all: every moment is zero and there
        // is nothing to rotate. Decide it before the linearity test, which would otherwise call it
        // linear and try to build two rotations out of nothing.
        let monatomic = nat <= 1;
        let linear = !monatomic && moments[0] <= LINEARITY_TOLERANCE * scale;

        let n_translations = if projection == Projection::None { 0 } else { 3 };
        let mut rotation_axes: Vec<Vec3> = Vec::new();
        if projection == Projection::Rigid && !monatomic {
            for axis in admissible_axes(molecule, &axes) {
                // Degeneracy is per axis, not per molecule: `I_w = sum_A m_A |w x d_A|^2` is the
                // squared norm of the generator this axis would produce, so zero means the axis
                // moves nothing and there is nothing to normalize. For a bent molecule this skips
                // nothing; for a linear one it skips the molecular axis; for a chain whose atoms
                // sit on the lattice vector it skips the only candidate there was.
                let moment: f64 = (0..nat).map(|a| mass(a) * axis.cross(d[a]).norm2()).sum();
                if moment > LINEARITY_TOLERANCE * scale {
                    rotation_axes.push(axis);
                }
            }
        }
        let n_rotations = rotation_axes.len();
        let k = n_translations + n_rotations;

        let mut generators = Matrix::zeros(ndof, k);
        if n_translations == 3 {
            let inv = total.sqrt().recip();
            for a in 0..nat {
                let w = mass(a).sqrt() * inv;
                for alpha in 0..3 {
                    generators[(3 * a + alpha, alpha)] = w;
                }
            }
        }
        for (col, axis) in rotation_axes.iter().enumerate() {
            let column = n_translations + col;
            let mut norm2 = 0.0;
            for a in 0..nat {
                let sqrt_m = mass(a).sqrt();
                let v = axis.cross(d[a]) * sqrt_m;
                norm2 += v.norm2();
                for alpha in 0..3 {
                    generators[(3 * a + alpha, column)] = v.get(alpha);
                }
            }
            // `norm2` is the principal moment about this axis, which the linearity test has already
            // established is non-zero: a degenerate axis was skipped above.
            let inv = norm2.sqrt().recip();
            for row in 0..ndof {
                generators[(row, column)] *= inv;
            }
        }

        Ok(RigidSubspace {
            generators,
            n_translations,
            n_rotations,
            principal_moments: moments,
            principal_axes: Mat3::from_rows(axes[0], axes[1], axes[2]),
            inertia_scale: scale,
            linear,
            rotation_axes,
        })
    }

    /// How many directions are removed.
    pub fn dimension(&self) -> usize {
        self.n_translations + self.n_rotations
    }

    /// Orthonormal basis of the orthogonal complement, `3N x (3N - k)`, by Householder QR of
    /// `generators`.
    ///
    /// No rank decision and no threshold anywhere: the rank was fixed by [`RigidSubspace::of`] from
    /// the geometry. That is the reason not to do the obvious thing -- Gram-Schmidt the canonical
    /// basis against the generators and keep the columns whose residual norm is largest -- which
    /// would smuggle a magnitude rule back in at the last step.
    pub fn complement(&self) -> Matrix {
        let n = self.generators.rows;
        let k = self.dimension();
        if k == 0 {
            return Matrix::identity(n);
        }

        // Householder reflectors taking the generators onto the leading axes. `v[j]` is the j-th
        // reflector, stored full length with its leading zeros.
        let mut work = self.generators.clone();
        let mut reflectors: Vec<Vec<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut v = vec![0.0; n];
            let mut norm2 = 0.0;
            for i in j..n {
                v[i] = work[(i, j)];
                norm2 += v[i] * v[i];
            }
            let norm = norm2.sqrt();
            // The generators are orthonormal, so column `j` after `j` reflections still has unit
            // norm below the diagonal up to rounding; it cannot vanish.
            let alpha = if v[j] >= 0.0 { -norm } else { norm };
            v[j] -= alpha;
            let vnorm2: f64 = v[j..].iter().map(|x| x * x).sum();
            if vnorm2 > 0.0 {
                let inv = vnorm2.sqrt().recip();
                for x in v[j..].iter_mut() {
                    *x *= inv;
                }
            }
            // Apply to the remaining columns so the next reflector sees the deflated matrix.
            for c in (j + 1)..k {
                let dot: f64 = (j..n).map(|i| v[i] * work[(i, c)]).sum();
                for i in j..n {
                    work[(i, c)] -= 2.0 * dot * v[i];
                }
            }
            reflectors.push(v);
        }

        // Q = H_0 H_1 ... H_{k-1}; the complement is its trailing `n - k` columns.
        let mut out = Matrix::zeros(n, n - k);
        for (col, j) in (k..n).enumerate() {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            for v in reflectors.iter().rev() {
                let dot: f64 = (0..n).map(|i| v[i] * e[i]).sum();
                for i in 0..n {
                    e[i] -= 2.0 * dot * v[i];
                }
            }
            for (i, value) in e.into_iter().enumerate() {
                out[(i, col)] = value;
            }
        }
        out
    }

    /// `<v_i|A|v_i>` per generator, in `A`'s own units -- the curvature the projection removes.
    pub fn removed_curvature(&self, a: &Matrix) -> Vec<f64> {
        let n = self.generators.rows;
        (0..self.dimension())
            .map(|col| {
                let mut total = 0.0;
                for i in 0..n {
                    let vi = self.generators[(i, col)];
                    if vi == 0.0 {
                        continue;
                    }
                    let mut row = 0.0;
                    for j in 0..n {
                        row += a[(i, j)] * self.generators[(j, col)];
                    }
                    total += vi * row;
                }
                total
            })
            .collect()
    }
}

/// An orthonormal basis of the rotation axes the lattice admits: `{w : w x T = 0 for every lattice
/// vector T}`. See the module docs for why that is the condition.
///
/// The subspace comes from the **lattice**, and the basis of it is chosen for orthogonality:
///
/// * **Molecule** -- the whole of `R^3`, and the inertia tensor's `principal` axes are the basis
///   that makes the three generators mutually orthogonal for free (`<r_a, r_b> = a . I . b`).
/// * **1-D chain** -- the one-dimensional span of the lattice vector. The basis is that vector, and
///   it is generally *not* a principal axis; there is only one generator, so there is nothing for a
///   principal frame to orthogonalize and no reason to prefer one. Taking the nearest principal
///   axis instead would be wrong: a chain rotates about its own lattice vector or not at all.
/// * **2-D and 3-D** -- empty. Two independent lattice vectors leave no axis parallel to both, and
///   `Cell::new` rejects a degenerate pair, so this needs no tolerance.
fn admissible_axes(molecule: &Molecule, principal: &[Vec3; 3]) -> Vec<Vec3> {
    let Some(cell) = molecule.cell else {
        return principal.to_vec();
    };
    match cell.dim() {
        0 => principal.to_vec(),
        1 => {
            let axis = cell.vectors()[0].normalized();
            // Cheap self-check on the claim above rather than a bare `vec![axis]`: the lattice has
            // one vector, so the axis is admissible by construction, but a future `Cell` that
            // stored something else in slot 0 would silently produce a wrong generator.
            debug_assert!(axis.cross(cell.vectors()[0]).norm() <= AXIS_TOLERANCE * axis.norm());
            vec![axis]
        }
        _ => Vec::new(),
    }
}

/// Principal moments (ascending) and the corresponding axes of a symmetric 3x3.
fn principal_frame(inertia: &Mat3) -> Result<([f64; 3], [Vec3; 3])> {
    let mut m = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = inertia.get(i, j);
        }
    }
    let (values, vectors) = symmetric_eigen(&m)?;
    let moments = [values[0], values[1], values[2]];
    let axes = [
        Vec3::new(vectors[(0, 0)], vectors[(1, 0)], vectors[(2, 0)]),
        Vec3::new(vectors[(0, 1)], vectors[(1, 1)], vectors[(2, 1)]),
        Vec3::new(vectors[(0, 2)], vectors[(1, 2)], vectors[(2, 2)]),
    ];
    Ok((moments, axes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

    fn water() -> Molecule {
        Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
            0.0,
        )
        .unwrap()
    }

    fn co2() -> Molecule {
        Molecule::from_xyz_str(
            "3\nCO2\nC 0.0 0.0 0.0\nO 0.0 0.0 1.16\nO 0.0 0.0 -1.16\n",
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn a_bent_molecule_gives_three_translations_and_three_rotations() {
        let s = RigidSubspace::of(&water(), Projection::Rigid).unwrap();
        assert_eq!((s.n_translations, s.n_rotations), (3, 3));
        assert!(!s.linear);
    }

    #[test]
    fn a_linear_molecule_gives_two_rotations() {
        let s = RigidSubspace::of(&co2(), Projection::Rigid).unwrap();
        assert!(s.linear, "moments {:?}", s.principal_moments);
        assert_eq!((s.n_translations, s.n_rotations), (3, 2));
        // The degenerate moment is zero to rounding, not merely small: the atoms are collinear by
        // construction, so the only thing between it and exact zero is the eigensolver.
        assert!(
            s.principal_moments[0] < 1.0e-20 * s.inertia_scale.max(1.0),
            "{:?}",
            s.principal_moments
        );
    }

    #[test]
    fn a_nearly_linear_molecule_is_not_treated_as_linear() {
        // The same CO2 with the carbon pushed 0.01 A off the axis. The criterion has to notice a
        // bend that a frequency cutoff would not.
        let m = Molecule::from_xyz_str(
            "3\nbent CO2\nC 0.01 0.0 0.0\nO 0.0 0.0 1.16\nO 0.0 0.0 -1.16\n",
            0.0,
        )
        .unwrap();
        let s = RigidSubspace::of(&m, Projection::Rigid).unwrap();
        assert!(!s.linear, "moments {:?}", s.principal_moments);
        assert_eq!(s.n_rotations, 3);
    }

    #[test]
    fn a_single_atom_has_no_rigid_rotations() {
        let m = Molecule::from_xyz_str("1\nNe\nNe 0.0 0.0 0.0\n", 0.0).unwrap();
        let s = RigidSubspace::of(&m, Projection::Rigid).unwrap();
        assert_eq!((s.n_translations, s.n_rotations), (3, 0));
        // 3N - k = 0: a lone atom has no vibrations, and the complement is an empty basis rather
        // than a panic.
        assert_eq!(s.complement().cols, 0);
    }

    #[test]
    fn the_periodic_rotation_rule_is_the_lattice_one() {
        // The generator has to be cell-periodic, so the admissible axes are those parallel to every
        // lattice vector: all of them for a molecule, the axis for a chain, none for a sheet or a
        // crystal.
        let a = crate::constants::ANGSTROM_TO_BOHR;
        let base = "4\nCH2\nC 0.0 0.0 0.0\nH 0.0 0.9 0.6\nH 0.0 -0.9 0.6\nC 1.3 0.0 0.0\n";
        let molecule = Molecule::from_xyz_str(base, 0.0).unwrap();
        assert_eq!(
            RigidSubspace::of(&molecule, Projection::Rigid)
                .unwrap()
                .n_rotations,
            3
        );

        let chain = molecule
            .clone()
            .with_cell(Cell::new(&[Vec3::new(2.6 * a, 0.0, 0.0)]).unwrap());
        let s = RigidSubspace::of(&chain, Projection::Rigid).unwrap();
        assert_eq!(s.n_rotations, 1, "a chain turns about its own axis");
        assert!(
            s.rotation_axes[0].cross(Vec3::new(1.0, 0.0, 0.0)).norm() < 1.0e-8,
            "and that axis is the chain's: {:?}",
            s.rotation_axes[0]
        );

        let sheet = molecule.clone().with_cell(
            Cell::new(&[Vec3::new(2.6 * a, 0.0, 0.0), Vec3::new(0.0, 3.0 * a, 0.0)]).unwrap(),
        );
        assert_eq!(
            RigidSubspace::of(&sheet, Projection::Rigid)
                .unwrap()
                .n_rotations,
            0,
            "rotation about a slab normal is a finite-q field, not a zone-centre one"
        );

        let crystal = molecule.with_cell(
            Cell::new(&[
                Vec3::new(2.6 * a, 0.0, 0.0),
                Vec3::new(0.0, 3.0 * a, 0.0),
                Vec3::new(0.0, 0.0, 4.0 * a),
            ])
            .unwrap(),
        );
        assert_eq!(
            RigidSubspace::of(&crystal, Projection::Rigid)
                .unwrap()
                .n_rotations,
            0
        );
    }

    #[test]
    fn a_chain_whose_atoms_lie_on_its_axis_has_no_rotation_at_all() {
        // The admissible axis is also the degenerate one, so the lattice test and the linearity
        // test have to compose rather than fight.
        let a = crate::constants::ANGSTROM_TO_BOHR;
        let m = Molecule::from_xyz_str("2\nH2 chain\nH 0.0 0.0 0.0\nH 0.76 0.0 0.0\n", 0.0)
            .unwrap()
            .with_cell(Cell::new(&[Vec3::new(2.0 * a, 0.0, 0.0)]).unwrap());
        let s = RigidSubspace::of(&m, Projection::Rigid).unwrap();
        assert!(s.linear);
        assert_eq!(s.n_rotations, 0);
    }

    #[test]
    fn the_generators_are_orthonormal_without_a_gram_schmidt_pass() {
        // The claim in the module docs, tested rather than asserted: the centre-of-mass definition
        // makes translations orthogonal to rotations, and the principal frame makes the rotations
        // orthogonal to each other.
        for molecule in [water(), co2()] {
            let s = RigidSubspace::of(&molecule, Projection::Rigid).unwrap();
            let g = &s.generators;
            for a in 0..s.dimension() {
                for b in 0..s.dimension() {
                    let dot: f64 = (0..g.rows).map(|i| g[(i, a)] * g[(i, b)]).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1.0e-12,
                        "generator overlap ({a},{b}) = {dot}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_complement_is_orthonormal_and_orthogonal_to_the_generators() {
        for molecule in [water(), co2()] {
            let s = RigidSubspace::of(&molecule, Projection::Rigid).unwrap();
            let b = s.complement();
            assert_eq!(b.cols, 3 * molecule.atoms.len() - s.dimension());
            for p in 0..b.cols {
                for q in 0..b.cols {
                    let dot: f64 = (0..b.rows).map(|i| b[(i, p)] * b[(i, q)]).sum();
                    let want = if p == q { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1.0e-12, "BtB({p},{q}) = {dot}");
                }
                for g in 0..s.dimension() {
                    let dot: f64 = (0..b.rows).map(|i| b[(i, p)] * s.generators[(i, g)]).sum();
                    assert!(dot.abs() < 1.0e-12, "BtG({p},{g}) = {dot}");
                }
            }
        }
    }

    #[test]
    fn translations_only_and_none_do_what_they_say() {
        let m = water();
        let t = RigidSubspace::of(&m, Projection::Translations).unwrap();
        assert_eq!((t.n_translations, t.n_rotations), (3, 0));
        let n = RigidSubspace::of(&m, Projection::None).unwrap();
        assert_eq!(n.dimension(), 0);
        assert_eq!(n.complement().cols, 9);
    }

    #[test]
    fn the_projection_name_round_trips() {
        for p in [
            Projection::Rigid,
            Projection::Translations,
            Projection::None,
        ] {
            assert_eq!(Projection::parse(p.as_str()).unwrap(), p);
        }
        assert!(Projection::parse("eckart").is_err());
    }
}
