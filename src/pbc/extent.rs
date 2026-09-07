// SPDX-License-Identifier: GPL-3.0-or-later

//! `ε^∞` for a chain or a slab, where there is no volume to divide by.
//!
//! [`crate::dfpt::born_and_dielectric`] returns `polarizability`, the raw `∂μ_a/∂f_b` per cell,
//! in every dimensionality — and a `dielectric` tensor only in three, because `ε` needs a volume
//! and a slab's cell has an area. The missing ingredient is a **thickness** (slab) or a
//! **cross-section** (wire), and it is a *required argument* rather than something to infer: a
//! supercell says where the atoms are, not where the material stops. Doubling the vacuum must not
//! change `ε`, and if the code picked the cell height it would.
//!
//! # The conversion is a depolarization problem, not a division
//!
//! The naive step is `ε = 1 + 4πα/(measure · extent)`. That is right along directions where the
//! induced polarization creates **no** macroscopic field, and wrong along the others, because the
//! `α` this crate computes is the response to the *external* field: the induced charges interact
//! through the same Coulomb operator the SCF uses, so for a slab polarized along its normal the
//! depolarizing field is already inside `α`. Dividing and adding 1 would count the screening once
//! and the shape not at all.
//!
//! # The sign is MOPAC's, and it is why `ε` subtracts
//!
//! Everything here takes the polarizability **exactly as [`crate::dfpt::DfptFieldResult`] reports
//! it**: `∂μ_a/∂f_b` in MOPAC's `FIELD=` convention, where `f` is the *potential gradient* rather
//! than the physical field (convention C-1). In that convention the physical susceptibility is
//! `χ = −(∂μ/∂f)/volume`, which is why the three-dimensional formula reads
//! `ε = δ − (4π/Ω) ∂μ/∂f` with a **minus** (convention C-6). The negation happens once, here, so a
//! caller holding a `polarizability` cannot pair it with the wrong sign — and the test that the
//! three-dimensional case closes on [`crate::dfpt::born_and_dielectric`] is what holds that fixed.
//!
//! With `χ` the external-field susceptibility and `N` the depolarization factor of the assumed
//! body along a principal axis,
//!
//! ```text
//! ε = 1 + 4πχ / (1 − 4πNχ)
//! ```
//!
//! | body | in plane / along the axis | across |
//! |---|---|---|
//! | slab, thickness `d` | `N = 0` | `N = 1` (the normal) |
//! | wire, cross-section `S` | `N = 0` (along the wire) | `N = ½` (circular section) |
//! | crystal | `N = 0` | `N = 0` |
//!
//! The crystal row is not a special case bolted on: three-dimensional tin-foil summation removes
//! the macroscopic depolarizing field, so `α` there is already the response to the *internal*
//! field and `N = 0` is the correct entry. That the table closes on
//! [`crate::dfpt::DfptFieldResult::dielectric`] is the first thing the tests check.
//!
//! # What the convention cannot change
//!
//! Two combinations are **thickness-free**, and are reported alongside `ε` so a slab calculation
//! has something to quote that carries no convention at all:
//!
//! ```text
//! (ε_∥ − 1) d = 4π α_∥ / A          (1 − 1/ε_⊥) d = 4π α_⊥ / A
//! ```
//!
//! Both sides of each are free of `d`. They say the layer has a well-defined *sheet*
//! susceptibility in plane and a well-defined sheet **inverse** susceptibility out of plane, and
//! that everything else in `ε` is the convention. Half the first is the Rytova–Keldysh screening
//! length, which is what the monolayer literature reports for exactly this reason. Read the other
//! way round the two are capacitor stacking — parallel and series — which is an independent
//! derivation of the same pair.

use crate::error::{Pm7Error, Result};
use crate::math::{Mat3, Vec3};

/// How much space a low-dimensional cell's material is taken to occupy.
///
/// Required, never defaulted. The vacuum in a slab supercell is padding, not thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtentConvention {
    /// A slab (two periodic directions) assigned a **thickness** in Bohr, measured along the cell
    /// normal `â₁ × â₂`. The assigned volume per cell is `A · d`.
    SlabThickness(f64),
    /// A chain (one periodic direction) assigned a **cross-sectional area** in Bohr², transverse
    /// to the periodic axis. The assigned volume per cell is `L · S`.
    ///
    /// An area rather than a radius, so that it multiplies the cell measure into a volume the way
    /// a slab's thickness does; the section is taken to be **circular**, which is what fixes the
    /// transverse depolarization factor at `½`.
    WireCrossSection(f64),
}

impl ExtentConvention {
    /// The extent itself: Bohr for a slab, Bohr² for a wire.
    #[inline]
    pub fn value(&self) -> f64 {
        match self {
            Self::SlabThickness(d) => *d,
            Self::WireCrossSection(s) => *s,
        }
    }

    /// How many periodic directions this convention describes.
    #[inline]
    pub fn periodic_directions(&self) -> usize {
        match self {
            Self::SlabThickness(_) => 2,
            Self::WireCrossSection(_) => 1,
        }
    }

    fn units(&self) -> &'static str {
        match self {
            Self::SlabThickness(_) => "Bohr",
            Self::WireCrossSection(_) => "Bohr^2",
        }
    }

    /// Depolarization factors `(along the distinguished axis, in the plane transverse to it)`.
    ///
    /// The one line that is the entire difference between the two bodies.
    fn depolarization(&self) -> (f64, f64) {
        match self {
            Self::SlabThickness(_) => (1.0, 0.0),
            Self::WireCrossSection(_) => (0.0, 0.5),
        }
    }
}

/// An orthonormal frame with `e3` along `axis`.
fn frame(axis: Vec3) -> [Vec3; 3] {
    let e3 = axis * (1.0 / axis.norm());
    // Any vector not parallel to `e3`; picking the smallest component keeps the cross product
    // well conditioned.
    let seed = if e3.x.abs() <= e3.y.abs() && e3.x.abs() <= e3.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if e3.y.abs() <= e3.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let mut e1 = seed - e3 * e3.dot(seed);
    e1 = e1 * (1.0 / e1.norm());
    let e2 = e3.cross(e1);
    [e1, e2, e3]
}

fn quadratic(u: Vec3, m: &Mat3, v: Vec3) -> f64 {
    let (a, b) = ([u.x, u.y, u.z], [v.x, v.y, v.z]);
    let mut out = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            out += a[i] * m.get(i, j) * b[j];
        }
    }
    out
}

/// Eigenvalues and the rotation of a symmetric 2 × 2 block, as `(λ₁, λ₂, cos, sin)`.
fn eigen2(a: f64, b: f64, d: f64) -> (f64, f64, f64, f64) {
    if b.abs() < 1.0e-300 {
        return (a, d, 1.0, 0.0);
    }
    let theta = 0.5 * (2.0 * b).atan2(a - d);
    let (s, c) = theta.sin_cos();
    let l1 = a * c * c + 2.0 * b * s * c + d * s * s;
    let l2 = a * s * s - 2.0 * b * s * c + d * c * c;
    (l1, l2, c, s)
}

/// `ε = 1 + 4πχ / (1 − 4πNχ)` for one principal value.
fn scalar_epsilon(chi: f64, n: f64) -> Result<f64> {
    let denominator = 1.0 - 4.0 * std::f64::consts::PI * n * chi;
    if denominator.abs() < 1.0e-12 {
        return Err(Pm7Error::InvalidInput(format!(
            "the depolarization denominator 1 - 4πNχ is {denominator:.3e}, so the assigned extent \
             puts this cell at the polarization catastrophe: the assumed body would screen its own \
             field completely. The extent is too small for the polarizability, which usually means \
             it is smaller than the atoms it is meant to contain."
        )));
    }
    Ok(1.0 + 4.0 * std::f64::consts::PI * chi / denominator)
}

/// `ε^∞` from a raw polarizability, an assigned extent, and the depolarization of the assumed body.
///
/// `measure` is the cell's own periodic measure — an area for a slab, a length for a wire — so
/// `measure · extent` is the assigned volume per cell. `axis` is the slab normal or the wire axis;
/// only its direction is used.
pub fn epsilon_from_polarizability(
    alpha: &Mat3,
    axis: Vec3,
    measure: f64,
    extent: ExtentConvention,
) -> Result<Mat3> {
    let e = extent.value();
    // `is_finite` first: it is what rejects NaN, which every comparison below would let through.
    if !e.is_finite() || e <= 0.0 {
        return Err(Pm7Error::InvalidInput(format!(
            "the assigned extent must be a positive, finite number of {}; got {e}",
            extent.units()
        )));
    }
    if !measure.is_finite() || measure <= 0.0 {
        return Err(Pm7Error::InvalidInput(
            "the cell's periodic measure is not positive, so there is no volume to spread the \
             polarizability over"
                .into(),
        ));
    }
    if !axis.norm().is_finite() || axis.norm() < 1.0e-12 {
        return Err(Pm7Error::InvalidInput(
            "the distinguished axis is degenerate: a slab normal or a wire axis is needed".into(),
        ));
    }

    let volume = measure * e;
    let [e1, e2, e3] = frame(axis);
    let (n_axis, n_plane) = extent.depolarization();

    // The minus is convention C-6: `alpha` is the field derivative in MOPAC's convention, and
    // the physical susceptibility carries the opposite sign. See the module documentation.
    let chi = |u: Vec3, v: Vec3| -quadratic(u, alpha, v) / volume;
    let (c11, c12, c22, c33) = (chi(e1, e1), chi(e1, e2), chi(e2, e2), chi(e3, e3));

    // The 2 × 2 transverse block, diagonalized so the scalar law applies per principal value. For
    // `N = 0` it is `1 + 4πχ` and the rotation is a no-op, but doing it the same way in both cases
    // keeps one code path.
    let (l1, l2, cos, sin) = eigen2(c11, c12, c22);
    let (p1, p2) = (scalar_epsilon(l1, n_plane)?, scalar_epsilon(l2, n_plane)?);
    let p3 = scalar_epsilon(c33, n_axis)?;

    let v1 = e1 * cos + e2 * sin;
    let v2 = e1 * (-sin) + e2 * cos;
    let mut out = Mat3::zero();
    for (p, v) in [(p1, v1), (p2, v2), (p3, e3)] {
        let c = [v.x, v.y, v.z];
        for i in 0..3 {
            for j in 0..3 {
                out.set(i, j, out.get(i, j) + p * c[i] * c[j]);
            }
        }
    }
    Ok(out)
}

/// How much of `alpha` couples the distinguished axis to its complement, relative to the largest
/// diagonal entry.
///
/// [`epsilon_from_polarizability`] drops that coupling, because a depolarization factor is a
/// per-principal-axis quantity. Zero means the slab normal (or wire axis) **is** a principal axis
/// of the response and nothing was lost. Reported rather than asserted: it is a property of the
/// system, and a caller is entitled to know how much of an assumption they are buying.
pub fn extent_axis_mixing(alpha: &Mat3, axis: Vec3) -> f64 {
    if axis.norm() < 1.0e-12 {
        return f64::NAN;
    }
    let [e1, e2, e3] = frame(axis);
    let mut scale = 0.0_f64;
    for i in 0..3 {
        scale = scale.max(alpha.get(i, i).abs());
    }
    let coupling = quadratic(e1, alpha, e3)
        .abs()
        .max(quadratic(e2, alpha, e3).abs());
    if scale <= 0.0 {
        0.0
    } else {
        coupling / scale
    }
}

/// The two combinations of `ε` and the extent that the extent cannot change.
///
/// `(ε_∥ − 1) d` and `(1 − 1/ε_⊥) d` for a slab of area `A`, both equal to `4πα/A` with the
/// matching component of `α`. A slab calculation can quote these without choosing a thickness at
/// all; half the first is the Rytova–Keldysh screening length.
#[derive(Clone, Copy, Debug)]
pub struct SheetInvariants {
    /// `(ε_∥ − 1) d`, in Bohr. The in-plane value is the mean of the two transverse principal
    /// directions.
    pub parallel: f64,
    /// `(1 − 1/ε_⊥) d`, in Bohr.
    pub perpendicular: f64,
}

impl SheetInvariants {
    /// Straight from the polarizability, with no thickness anywhere in the arithmetic.
    pub fn of(alpha: &Mat3, axis: Vec3, area: f64) -> Self {
        let [e1, e2, e3] = frame(axis);
        let four_pi = 4.0 * std::f64::consts::PI;
        // Same C-6 minus as `epsilon_from_polarizability`.
        let plane = -0.5 * (quadratic(e1, alpha, e1) + quadratic(e2, alpha, e2));
        Self {
            parallel: four_pi * plane / area,
            perpendicular: -four_pi * quadratic(e3, alpha, e3) / area,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diagonal field derivative whose **physical** polarizability is (xx, yy, zz).
    ///
    /// The negation is convention C-1: these functions take the derivative MOPAC's FIELD=
    /// produces, and a normal material is positive there only after the sign flip. Writing the
    /// fixture this way keeps every expected value below in the natural 1 + 4πχ form.
    fn diagonal(xx: f64, yy: f64, zz: f64) -> Mat3 {
        let mut m = Mat3::zero();
        m.set(0, 0, -xx);
        m.set(1, 1, -yy);
        m.set(2, 2, -zz);
        m
    }

    /// With every depolarization factor zero the law is `1 + 4πχ`, which is what a 3-D cell uses.
    ///
    /// The table closing on the existing three-dimensional formula is the check that the
    /// low-dimensional conversion is a generalization rather than a second, unrelated rule.
    #[test]
    fn a_slab_in_plane_is_the_three_dimensional_law() {
        let alpha = diagonal(2.5, 2.5, 0.8);
        let area = 30.0;
        let thickness = 6.0;
        let eps = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            area,
            ExtentConvention::SlabThickness(thickness),
        )
        .unwrap();
        let expected = 1.0 + 4.0 * std::f64::consts::PI * 2.5 / (area * thickness);
        assert!(
            (eps.get(0, 0) - expected).abs() < 1.0e-12,
            "{}",
            eps.get(0, 0)
        );
        assert!((eps.get(1, 1) - expected).abs() < 1.0e-12);
    }

    /// Across a slab the law is the `N = 1` one, which screens rather than adds.
    #[test]
    fn across_a_slab_the_depolarizing_field_is_already_in_alpha() {
        let alpha = diagonal(2.5, 2.5, 0.8);
        let (area, thickness) = (30.0, 6.0);
        let eps = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            area,
            ExtentConvention::SlabThickness(thickness),
        )
        .unwrap();
        let chi = 0.8 / (area * thickness);
        let four_pi = 4.0 * std::f64::consts::PI;
        let expected = 1.0 + four_pi * chi / (1.0 - four_pi * chi);
        assert!((eps.get(2, 2) - expected).abs() < 1.0e-12);
        // And it is *larger* than the naive division would give, because the same `α` responded to
        // a field the depolarization had already reduced.
        assert!(eps.get(2, 2) > 1.0 + four_pi * chi);
    }

    /// The sheet invariants do not move when the thickness does.
    ///
    /// This is the whole point of reporting them: they are what a slab can say without choosing a
    /// convention, and the test is that choosing a different one changes nothing.
    #[test]
    fn the_sheet_invariants_are_thickness_free() {
        let alpha = diagonal(2.5, 2.5, 0.8);
        let area = 30.0;
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let reference = SheetInvariants::of(&alpha, axis, area);
        for thickness in [2.0_f64, 6.0, 18.0, 60.0] {
            let eps = epsilon_from_polarizability(
                &alpha,
                axis,
                area,
                ExtentConvention::SlabThickness(thickness),
            )
            .unwrap();
            let parallel = (eps.get(0, 0) - 1.0) * thickness;
            let perpendicular = (1.0 - 1.0 / eps.get(2, 2)) * thickness;
            assert!(
                (parallel - reference.parallel).abs() < 1.0e-10,
                "d = {thickness}: (eps_par - 1) d = {parallel}, expected {}",
                reference.parallel
            );
            assert!(
                (perpendicular - reference.perpendicular).abs() < 1.0e-10,
                "d = {thickness}: (1 - 1/eps_perp) d = {perpendicular}, expected {}",
                reference.perpendicular
            );
        }
    }

    /// A wire is the circular-cylinder law transverse and the free one along its axis.
    #[test]
    fn a_wire_uses_the_cylinder_factor_across_and_none_along() {
        let alpha = diagonal(4.0, 1.2, 1.2);
        let (length, section) = (5.0, 20.0);
        let eps = epsilon_from_polarizability(
            &alpha,
            Vec3::new(1.0, 0.0, 0.0),
            length,
            ExtentConvention::WireCrossSection(section),
        )
        .unwrap();
        let four_pi = 4.0 * std::f64::consts::PI;
        let volume = length * section;
        assert!((eps.get(0, 0) - (1.0 + four_pi * 4.0 / volume)).abs() < 1.0e-12);
        let chi = 1.2 / volume;
        let expected = 1.0 + four_pi * chi / (1.0 - 0.5 * four_pi * chi);
        assert!((eps.get(1, 1) - expected).abs() < 1.0e-12);
    }

    /// An extent small enough to drive `1 − 4πNχ` through zero is refused, not returned.
    #[test]
    fn the_polarization_catastrophe_is_an_error() {
        let alpha = diagonal(1.0, 1.0, 100.0);
        let error = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            ExtentConvention::SlabThickness(4.0 * std::f64::consts::PI * 100.0),
        );
        // A thickness that makes `4πNχ` exactly 1 is the singular point; walk onto it.
        let singular = epsilon_from_polarizability(
            &alpha,
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            ExtentConvention::SlabThickness(4.0 * std::f64::consts::PI * 100.0),
        );
        assert_eq!(error.is_err(), singular.is_err());
        assert!(
            singular.is_err(),
            "4πNχ = 1 should be refused, not returned as a huge epsilon"
        );
    }

    /// A non-positive extent is refused with a message naming the units.
    #[test]
    fn a_nonsense_extent_is_refused() {
        let alpha = diagonal(1.0, 1.0, 1.0);
        for extent in [
            ExtentConvention::SlabThickness(0.0),
            ExtentConvention::SlabThickness(-3.0),
            ExtentConvention::SlabThickness(f64::NAN),
            ExtentConvention::WireCrossSection(-1.0),
        ] {
            let out = epsilon_from_polarizability(&alpha, Vec3::new(0.0, 0.0, 1.0), 10.0, extent);
            assert!(out.is_err(), "{extent:?} was accepted");
        }
    }

    /// The mixing measure is zero when the axis is a principal direction and non-zero when it is
    /// not — otherwise it would report "nothing was lost" unconditionally.
    #[test]
    fn the_axis_mixing_measure_sees_off_axis_response() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        assert!(extent_axis_mixing(&diagonal(2.0, 2.0, 1.0), axis) < 1.0e-15);
        let mut tilted = diagonal(2.0, 2.0, 1.0);
        tilted.set(0, 2, -0.5);
        tilted.set(2, 0, -0.5);
        assert!((extent_axis_mixing(&tilted, axis) - 0.25).abs() < 1.0e-12);
    }
}
