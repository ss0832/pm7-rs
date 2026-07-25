// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic Slater-type diatomic overlap integrals (`diat.f`/`diat2.f`), ported from the
//! reference implementation. Supports valence shells with principal quantum number 1–3
//! (H through Ar), which covers the core organic + main-group PM7 elements. Returns the
//! 4×4 block `S[a_i][b_j]` in the global frame with orbital order `s, px, py, pz`.

use crate::dual::{Dual, Scalar};
use crate::dual2::Dual2;
use crate::error::Result;
use crate::integrals::rotation_to_x_g;
use crate::math::Vec3;
use crate::params::Pm7Element;

/// Overlap block between atom `i` (rows) and atom `j` (cols), both in the global frame.
pub fn diatom_overlap(
    ei: &Pm7Element,
    pos_i: Vec3,
    ej: &Pm7Element,
    pos_j: Vec3,
) -> Result<[[f64; 4]; 4]> {
    let d = pos_j - pos_i;
    let r = d.norm();
    if r < 1.0e-8 {
        let mut m = [[0.0; 4]; 4];
        for k in 0..4 {
            m[k][k] = 1.0;
        }
        return Ok(m);
    }
    let xij = d / r;
    // The kernel requires the first atom to have the higher principal quantum number.
    let (locals, dir, swap) = if ei.n >= ej.n {
        (overlap_locals(ei, ej, r)?, xij, false)
    } else {
        (overlap_locals(ej, ei, r)?, xij * -1.0, true)
    };
    let di = build_di_g::<f64>(locals, [dir.x, dir.y, dir.z]);
    Ok(if swap { transpose4(di) } else { di })
}

/// Dual-valued overlap block, seeded on the displacement `R_j − R_i`, giving the exact
/// derivative `∂S/∂R_j`. Both the radial and angular dependence differentiate in **closed form**
/// for all valence shells — `n ≤ 3` uses the analytic Slater kernel, `n ≥ 4` uses forward-mode
/// AD through the numerical Slater quadrature. No finite differences.
pub fn diatom_overlap_dual(
    ei: &Pm7Element,
    pos_i: Vec3,
    ej: &Pm7Element,
    pos_j: Vec3,
) -> Result<[[Dual; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    // a→b direction: +dvec/r if not swapped, else −dvec/r (j→i). Seed on the raw displacement.
    let dvec = [Dual::var(d.x, 0), Dual::var(d.y, 1), Dual::var(d.z, 2)];
    let r_dual = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv_r = r_dual.recip();
    let s = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * s,
        dvec[1] * inv_r * s,
        dvec[2] * inv_r * s,
    ];

    // Radial local scalars, differentiated in closed form for all n.
    let locals = overlap_locals::<Dual>(ea, eb, r_dual)?;

    let di = build_di_g::<Dual>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

/// Second-derivative overlap block: `[[Dual2;4];4]` seeded on the displacement `R_j − R_i`,
/// giving the exact `∂²S/∂R∂R` (value + gradient + 3×3 Hessian per orbital pair). The radial
/// *and* angular dependence are both differentiated in closed form (no finite differences), so
/// this feeds the fully analytic skeleton Hessian — for all valence shells (`n ≤ 3` analytic
/// kernel, `n ≥ 4` via second-order AD through the numerical Slater quadrature).
pub fn diatom_overlap_dual2(
    ei: &Pm7Element,
    pos_i: Vec3,
    ej: &Pm7Element,
    pos_j: Vec3,
) -> Result<[[Dual2; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    // Radial locals, exact first+second derivatives w.r.t. the displacement (via `r`).
    let locals = overlap_locals::<Dual2>(ea, eb, r)?;
    // a→b unit direction: +dvec/r if not swapped, else −dvec/r (j→i).
    let inv_r = r.recip();
    let sgn = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * sgn,
        dvec[1] * inv_r * sgn,
        dvec[2] * inv_r * sgn,
    ];

    let di = build_di_g::<Dual2>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

fn transpose4<T: Copy>(m: [[T; 4]; 4]) -> [[T; 4]; 4] {
    let mut out = m;
    for a in 0..4 {
        for b in 0..4 {
            out[a][b] = m[b][a];
        }
    }
    out
}

/// `ea` is the higher-(or equal-)quantum-number atom; `r` is the interatomic distance (Bohr).
/// Generic over the scalar type: instantiated at `f64` for the energy/overlap path, and at
/// [`Dual`]/[`crate::dual2::Dual2`] (seeding `r` on the displacement) for the exact first/second
/// radial derivatives. Products are ordered to keep the scalar on the left of any `f64` factor.
fn overlap_locals<S: Scalar>(ea: &Pm7Element, eb: &Pm7Element, r: S) -> Result<[S; 5]> {
    let jcall = match (ea.n, eb.n) {
        (1, 1) => 2,
        (2, 1) => 3,
        (2, 2) => 4,
        (3, 1) => 431,
        (3, 2) => 5,
        (3, 3) => 6,
        // n >= 4 (heavy PM7 elements) have no tabulated closed form here; use the general
        // numerical Slater overlap. Sentinel 0 selects it below.
        _ => 0,
    };

    let za = [ea.zeta_s, ea.zeta_p];
    let zb = [eb.zeta_s, eb.zeta_p];

    // A/B auxiliary integrals for each (zeta_a-shell, zeta_b-shell) pair.
    let a111 = aintgs(r * (0.5 * (za[0] + zb[0])));
    let b111 = bintgs(r * (0.5 * (za[0] - zb[0])));
    let a211 = aintgs(r * (0.5 * (za[1] + zb[0])));
    let b211 = bintgs(r * (0.5 * (za[1] - zb[0])));
    let a121 = aintgs(r * (0.5 * (za[0] + zb[1])));
    let b121 = bintgs(r * (0.5 * (za[0] - zb[1])));
    let a22 = aintgs(r * (0.5 * (za[1] + zb[1])));
    let b22 = bintgs(r * (0.5 * (za[1] - zb[1])));

    // `s111` is set by every (reachable) arm; the others default to 0 for arms that only
    // populate the sigma channels.
    let s111;
    let (mut s211, mut s121, mut s221, mut s222) =
        (S::cst(0.0), S::cst(0.0), S::cst(0.0), S::cst(0.0));
    let sq3 = 3.0_f64.sqrt();

    match jcall {
        0 => {
            // Heavy elements (n >= 4): the general numerical Slater overlap, now differentiated
            // analytically through the quadrature (r is carried as the scalar type S).
            let (a, b, c, d, e) = crate::overlap_numeric::slater_locals_numeric::<S>(
                ea.n, ea.zeta_s, ea.zeta_p, eb.n, eb.zeta_s, eb.zeta_p, r,
            );
            s111 = a;
            s211 = b;
            s121 = c;
            s221 = d;
            s222 = e;
        }
        2 => {
            s111 =
                (r * r * (za[0] * zb[0])).powf(1.5) * (a111[2] * b111[0] - b111[2] * a111[0]) / 4.0;
        }
        3 => {
            s111 = r.powi(4)
                * (zb[0].powf(1.5) * za[0].powf(2.5))
                * (a111[3] * b111[0] - b111[3] * a111[0] + a111[2] * b111[1] - b111[2] * a111[1])
                / (sq3 * 8.0);
            s211 = r.powi(4)
                * (zb[0].powf(1.5) * za[1].powf(2.5))
                * (a211[2] * b211[0] - b211[2] * a211[0] + a211[3] * b211[1] - b211[3] * a211[1])
                / 8.0;
        }
        4 => {
            s111 = r.powi(5)
                * (zb[0] * za[0]).powf(2.5)
                * (a111[4] * b111[0] + b111[4] * a111[0] - a111[2] * b111[2] * 2.0)
                / 48.0;
            s211 = r.powi(5)
                * (zb[0] * za[1]).powf(2.5)
                * (a211[3] * (b211[0] - b211[2]) - a211[1] * (b211[2] - b211[4])
                    + b211[3] * (a211[0] - a211[2])
                    - b211[1] * (a211[2] - a211[4]))
                / (16.0 * sq3);
            s121 = r.powi(5)
                * (zb[1] * za[0]).powf(2.5)
                * (a121[3] * (b121[0] - b121[2])
                    - a121[1] * (b121[2] - b121[4])
                    - b121[3] * (a121[0] - a121[2])
                    + b121[1] * (a121[2] - a121[4]))
                / (16.0 * sq3);
            let w = r.powi(5) * (zb[1] * za[1]).powf(2.5) / 16.0;
            s221 = -(w * (b22[2] * (a22[4] + a22[0]) - a22[2] * (b22[4] + b22[0])));
            s222 = w
                * (a22[4] * (b22[0] - b22[2]) - b22[4] * (a22[0] - a22[2]) - a22[2] * b22[0]
                    + b22[2] * a22[0])
                * 0.5;
        }
        431 => {
            s111 = r.powi(5)
                * (zb[0].powf(1.5) * za[0].powf(3.5))
                * (a111[4] * b111[0] + b111[1] * a111[3] * 2.0
                    - a111[1] * b111[3] * 2.0
                    - b111[4] * a111[0])
                / (10.0_f64.sqrt() * 24.0);
            s211 = r.powi(5)
                * (zb[0].powf(1.5) * za[1].powf(3.5))
                * (a211[3] * (b211[0] + b211[2]) - a211[1] * (b211[4] + b211[2])
                    + b211[1] * (a211[2] + a211[4])
                    - b211[3] * (a211[2] + a211[0]))
                / (8.0 * 30.0_f64.sqrt());
        }
        5 => {
            s111 = r.powi(6)
                * (zb[0].powf(2.5) * za[0].powf(3.5))
                * (a111[5] * b111[0] + b111[1] * a111[4]
                    - b111[2] * a111[3] * 2.0
                    - a111[2] * b111[3] * 2.0
                    + b111[4] * a111[1]
                    + b111[5] * a111[0])
                / (30.0_f64.sqrt() * 48.0);
            s211 = r.powi(6)
                * (zb[0].powf(2.5) * za[1].powf(3.5))
                * (a211[4] * b211[0] + b211[1] * a211[5]
                    - b211[3] * a211[3] * 2.0
                    - a211[2] * b211[2] * 2.0
                    + a211[1] * b211[5]
                    + a211[0] * b211[4])
                / (48.0 * 10.0_f64.sqrt());
            s121 = r.powi(6)
                * (zb[1].powf(2.5) * za[0].powf(3.5))
                * ((a121[4] * b121[0] - a121[5] * b121[1])
                    + (a121[3] * b121[1] - a121[4] * b121[2]) * 2.0
                    - (a121[1] * b121[3] - a121[2] * b121[4]) * 2.0
                    - (a121[0] * b121[4] - a121[1] * b121[5]))
                / (48.0 * 10.0_f64.sqrt());
            s221 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[3] * b22[0] - a22[5] * b22[2]) + (a22[2] * b22[1] - a22[4] * b22[3])
                    - (a22[1] * b22[2] - a22[3] * b22[4])
                    - (a22[0] * b22[3] - a22[2] * b22[5]))
                / (16.0 * 30.0_f64.sqrt());
            s222 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[5] - a22[3]) * (b22[0] - b22[2]) + (a22[4] - a22[2]) * (b22[1] - b22[3])
                    - (a22[3] - a22[1]) * (b22[2] - b22[4])
                    - (a22[2] - a22[0]) * (b22[3] - b22[5]))
                / (32.0 * 30.0_f64.sqrt());
        }
        6 => {
            s111 = r.powi(7)
                * (zb[0] * za[0]).powf(3.5)
                * (a111[6] * b111[0] - b111[2] * a111[4] * 3.0 + a111[2] * b111[4] * 3.0
                    - a111[0] * b111[6])
                / 1440.0;
            s211 = r.powi(7)
                * (zb[0] * za[1]).powf(3.5)
                * ((a211[5] * b211[0] + a211[6] * b211[1])
                    + (-a211[4] * b211[1] - a211[5] * b211[2])
                    - (a211[3] * b211[2] + a211[4] * b211[3]) * 2.0
                    - (-a211[2] * b211[3] - a211[3] * b211[4]) * 2.0
                    + (a211[1] * b211[4] + a211[2] * b211[5])
                    + (-a211[0] * b211[5] - a211[1] * b211[6]))
                / (480.0 * sq3);
            s121 = r.powi(7)
                * (zb[1] * za[0]).powf(3.5)
                * ((a121[5] * b121[0] - a121[6] * b121[1])
                    + (a121[4] * b121[1] - a121[5] * b121[2])
                    - (a121[3] * b121[2] - a121[4] * b121[3]) * 2.0
                    - (a121[2] * b121[3] - a121[3] * b121[4]) * 2.0
                    + (a121[1] * b121[4] - a121[2] * b121[5])
                    + (a121[0] * b121[5] - a121[1] * b121[6]))
                / (480.0 * sq3);
            s221 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[4] * b22[0] - a22[6] * b22[2]) - (a22[2] * b22[2] - a22[4] * b22[4]) * 2.0
                    + (a22[0] * b22[4] - a22[2] * b22[6]))
                / 480.0;
            s222 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[6] - a22[4]) * (b22[0] - b22[2])
                    - (a22[4] - a22[2]) * (b22[2] - b22[4]) * 2.0
                    + (a22[2] - a22[0]) * (b22[4] - b22[6]))
                / 960.0;
        }
        _ => unreachable!(),
    }

    Ok([s111, s211, s121, s221, s222])
}

/// Assemble the molecular-frame 4×4 overlap block from the local-frame quantities and the
/// a→b direction, generic over the scalar type (so the direction dependence differentiates
/// exactly). Shared by the analytic (n ≤ 3) and numerical (n ≥ 4) overlap paths.
fn build_di_g<S: Scalar>(s: [S; 5], dir: [S; 3]) -> [[S; 4]; 4] {
    let (s111, s211, s121, s221, s222) = (s[0], s[1], s[2], s[3], s[4]);
    // rot_ov = R(dir)^T ; its first column is the a→b direction cosines.
    let rr = rotation_to_x_g(dir[0], dir[1], dir[2]);
    let mut rot_ov = [[S::cst(0.0); 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            rot_ov[a][b] = rr[b][a];
        }
    }
    let mut di = [[S::cst(0.0); 4]; 4];
    di[0][0] = s111;
    for a in 0..3 {
        di[1 + a][0] = s211 * rot_ov[a][0];
        di[0][1 + a] = -s121 * rot_ov[a][0];
    }
    let mm = [-s221, s222, s222];
    for a in 0..3 {
        for b in 0..3 {
            let mut v = S::cst(0.0);
            for k in 0..3 {
                v = v + rot_ov[a][k] * mm[k] * rot_ov[b][k];
            }
            di[1 + a][1 + b] = v;
        }
    }
    di
}

/// Test-only: overlap block built with the *numeric* local overlaps (forces the general
/// path even for n ≤ 3), used to validate the numeric kernel against the analytic one.
#[cfg(test)]
pub(crate) fn diatom_overlap_forced_numeric(
    ei: &Pm7Element,
    pos_i: Vec3,
    ej: &Pm7Element,
    pos_j: Vec3,
) -> [[f64; 4]; 4] {
    let d = pos_j - pos_i;
    let r = d.norm();
    let dir = d / r;
    let (ea, eb, direction, swap) = if ei.n >= ej.n {
        (ei, ej, dir, false)
    } else {
        (ej, ei, dir * -1.0, true)
    };
    let (s111, s211, s121, s221, s222) = crate::overlap_numeric::slater_locals_numeric(
        ea.n, ea.zeta_s, ea.zeta_p, eb.n, eb.zeta_s, eb.zeta_p, r,
    );
    let di = build_di_g::<f64>(
        [s111, s211, s121, s221, s222],
        [direction.x, direction.y, direction.z],
    );
    if !swap {
        di
    } else {
        transpose4(di)
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::params::Pm7Parameters;

    #[test]
    fn numeric_reproduces_analytic_overlap() {
        // The general numerical overlap must match the MOPAC-validated analytic kernel for
        // n <= 3 pairs, across several orientations, for every AO of the block.
        let p = Pm7Parameters::standard().unwrap();
        let a0 = crate::constants::ANGSTROM_TO_BOHR;
        let cases = [(6u8, 1u8), (6, 6), (6, 8), (7, 8), (1, 8), (16, 1), (17, 6)];
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.5, 0.8),
            Vec3::new(-0.6, 0.2, -0.1),
        ];
        let mut max_delta = 0.0_f64;
        for (za, zb) in cases {
            let ea = p.element(za).unwrap();
            let eb = p.element(zb).unwrap();
            for d in dirs {
                let r_ang = 1.5;
                let pos_i = Vec3::zero();
                let pos_j = d.normalized() * (r_ang * a0);
                let ana = diatom_overlap(ea, pos_i, eb, pos_j).unwrap();
                let num = diatom_overlap_forced_numeric(ea, pos_i, eb, pos_j);
                for a in 0..4 {
                    for b in 0..4 {
                        max_delta = max_delta.max((ana[a][b] - num[a][b]).abs());
                    }
                }
            }
        }
        eprintln!("numeric-vs-analytic overlap max delta = {max_delta:.2e}");
        assert!(max_delta < 5e-5, "overlap mismatch {max_delta:.3e}");
    }
}

/// A auxiliary integrals `A_k(x) = ∫₁^∞ t^k e^{-xt} dt`, returned as `[A_0 … A_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
fn aintgs<S: Scalar>(x: S) -> [S; 13] {
    let mut a = [S::cst(0.0); 13];
    if x.val().abs() < 1.0e-12 {
        return a; // vanishing exponent (e.g. zeta_p of H); those channels are unused
    }
    let inv = x.recip();
    a[0] = (-x).exp() * inv;
    for i in 1..13 {
        a[i] = a[0] + a[i - 1] * inv * (i as f64);
    }
    a
}

/// B auxiliary integrals `B_k(x) = ∫₋₁^¹ t^k e^{-xt} dt`, returned as `[B_0 … B_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
fn bintgs<S: Scalar>(x: S) -> [S; 13] {
    let mut b = [S::cst(0.0); 13];
    let absx = x.val().abs();
    if absx > 0.5 {
        let inv = x.recip();
        let tx = x.exp() * inv;
        let tmx = (-x).exp() * inv * (-1.0);
        b[0] = tx + tmx;
        for i in 1..13 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            b[i] = tx * sign + tmx + b[i - 1] * inv * (i as f64);
        }
    } else if absx > 1.0e-6 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x4 * x2;
        // even index (b1,b3,...): power series in x²
        b[0] = x2 * (1.0 / 3.0) + x4 * (1.0 / 60.0) + x6 * (1.0 / 2520.0) + 2.0;
        b[2] = x2 * (1.0 / 5.0) + x4 * (1.0 / 84.0) + x6 * (1.0 / 3240.0) + 2.0 / 3.0;
        b[4] = x2 * (1.0 / 7.0) + x4 * (1.0 / 108.0) + x6 * (1.0 / 3960.0) + 2.0 / 5.0;
        b[6] = x2 * (1.0 / 9.0) + x4 * (1.0 / 132.0) + x6 * (1.0 / 4680.0) + 2.0 / 7.0;
        b[8] = x2 * (1.0 / 11.0) + x4 * (1.0 / 156.0) + x6 * (1.0 / 5400.0) + 2.0 / 9.0;
        b[10] = x2 * (1.0 / 13.0) + x4 * (1.0 / 180.0) + x6 * (1.0 / 6120.0) + 2.0 / 11.0;
        b[12] = x2 * (1.0 / 15.0) + x4 * (1.0 / 204.0) + x6 * (1.0 / 6840.0) + 2.0 / 13.0;
        // odd index (b2,b4,...): power series in x
        b[1] = x * (-2.0 / 3.0) - x3 * (1.0 / 15.0) - x5 * (1.0 / 420.0);
        b[3] = x * (-2.0 / 5.0) - x3 * (1.0 / 21.0) - x5 * (1.0 / 540.0);
        b[5] = x * (-2.0 / 7.0) - x3 * (1.0 / 27.0) - x5 * (1.0 / 660.0);
        b[7] = x * (-2.0 / 9.0) - x3 * (1.0 / 33.0) - x5 * (1.0 / 780.0);
        b[9] = x * (-2.0 / 11.0) - x3 * (1.0 / 39.0) - x5 * (1.0 / 900.0);
        b[11] = x * (-2.0 / 13.0) - x3 * (1.0 / 45.0) - x5 * (1.0 / 1020.0);
    } else {
        // x ≈ 0
        for (i, bi) in b.iter_mut().enumerate() {
            *bi = S::cst(if i % 2 == 0 {
                2.0 / (i as f64 + 1.0)
            } else {
                0.0
            });
        }
    }
    b
}
