// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO two-center two-electron integrals (Dewar–Sabelli–Klopman multipole model) and the
//! electron–core attraction integrals.
//!
//! The kernels are written generically over [`crate::dual::Scalar`]: instantiating them at
//! `f64` gives the (validated) energy path; instantiating at [`crate::dual::Dual`] gives the
//! exact derivatives with respect to an interatomic displacement, used by the fully analytic
//! gradient. Distances are in Bohr, energies in eV (`PM7_EV`); orbital order is `s,px,py,pz`.

use crate::constants::{PM7_A0, PM7_EV};
use crate::dual::{Dual, Scalar};
use crate::math::Vec3;
use crate::params::Pm7Element;

/// Lower-triangle pack index of an orbital pair `(a, b)` within a 4-orbital atom block.
#[inline]
pub fn pack(a: usize, b: usize) -> usize {
    let (h, l) = if a >= b { (a, b) } else { (b, a) };
    h * (h + 1) / 2 + l
}

/// PM7 "feathering" (`l_feather`, always on for PM7 — MOPAC `readmo.F90:787`).  As two atoms
/// separate, the NDDO two-center integrals smoothly become exact point-charge values, so
/// beyond `trunc_1 = 7 Å` a pair interacts as bare monopoles.  Port of `to_point`
/// (MOPAC `mndod.F90:3462`): for a distance `r` **in Bohr** returns `(cfrac, point)` where a
/// monopole (charge–charge) integral becomes `I*cfrac + point*(1 - cfrac)` and every
/// higher-multipole integral becomes `I*cfrac`.  `point = PM7_EV / r` is the point-charge
/// `(ss|ss)`.  The switch `1 - exp(-(r - 7)^2 * 0.22)` is C¹ at `r = 7 Å` (both value and
/// slope reach 0 there), so analytic gradients and Hessians stay smooth.
#[inline]
pub(crate) fn feather_to_point<S: Scalar>(r: S) -> (S, S) {
    let point = r.recip() * PM7_EV;
    let r_ang = r * PM7_A0;
    let cfrac = if r_ang.val() < 7.0 {
        S::cst(1.0) - (-(r_ang - 7.0).powi(2) * 0.22).exp()
    } else {
        S::cst(0.0)
    };
    (cfrac, point)
}

/// Feather the local-frame diatomic repulsion integrals `ri` in place: monopole
/// (charge–charge) integrals decay toward the point charge, all others toward zero.
/// `monopole` lists the indices of the charge–charge integrals (MOPAC `reppd.F90:802-823`).
#[inline]
pub(crate) fn feather_ri<S: Scalar>(ri: &mut [S], r: S, monopole: &[usize]) {
    let (cfrac, point) = feather_to_point(r);
    let add = point * (S::cst(1.0) - cfrac);
    for v in ri.iter_mut() {
        *v = *v * cfrac;
    }
    for &k in monopole {
        ri[k] = ri[k] + add;
    }
}

/// Charge–charge integrals in the 22-element `local_xx_g` set (MOPAC 1-based
/// {1,3,4,11,12,16,17,18,19,21}); the rest carry a net dipole/quadrupole and vanish at the
/// point-charge limit.
const XX_MONOPOLE: [usize; 10] = [0, 2, 3, 10, 11, 15, 16, 17, 18, 20];
/// Charge–charge integrals in the 4-element `local_xh_g` set (SS/SS, OO/SS, PP/SS).
const XH_MONOPOLE: [usize; 3] = [0, 2, 3];

/// Rotation matrix (rows) that rotates the unit vector `v` onto +x (`R·v = (1,0,0)`),
/// generic over the scalar type. Port of PySEQM `rotate_with_quaternion`.
pub fn rotation_to_x_g<S: Scalar>(vx: S, vy: S, vz: S) -> [[S; 3]; 3] {
    let mut qx = S::cst(0.0);
    let mut qy = vz;
    let mut qz = -vy;
    let mut qw = vx + 1.0;
    // Only the exact antipode (bond ∥ +x, `qw = 0`) is the degenerate 0/0; a threshold this
    // tight keeps the f64 *value* on the smooth normal branch arbitrarily close to the axis
    // (so the singular-pair FD fallback in `rotfix` sees a varying integral), while an exactly
    // aligned bond still takes the special case.  The near-axis Dual derivative is never used:
    // `rotfix::near_frame_singularity` (window ~1e-4) routes those pairs to the FD fallback.
    if qw.val().abs() < 1.0e-13 {
        qx = S::cst(0.0);
        qy = S::cst(0.0);
        qz = S::cst(1.0);
        qw = S::cst(0.0);
    }
    let norm = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    let inv = norm.recip();
    qx = qx * inv;
    qy = qy * inv;
    qz = qz * inv;
    qw = qw * inv;
    let _ = qx;
    [
        [
            (qy * qy + qz * qz) * (-2.0) + 1.0,
            qz * qw * (-2.0),
            qy * qw * 2.0,
        ],
        [qz * qw * 2.0, qz * qz * (-2.0) + 1.0, qy * qz * 2.0],
        [qy * qw * (-2.0), qy * qz * 2.0, qy * qy * (-2.0) + 1.0],
    ]
}

/// Rotation matrix for an `f64` unit vector (used by the overlap f64 path).
pub fn rotation_to_x(v: Vec3) -> [[f64; 3]; 3] {
    rotation_to_x_g(v.x, v.y, v.z)
}

/// Rotated two-electron integrals + electron–core attractions for one ordered atom pair,
/// generic over the scalar type.
pub struct PairTwoElecG<S: Scalar> {
    pub norb_i: usize,
    pub norb_j: usize,
    /// `w[pack_i(a,b)][pack_j(c,d)] = (a_i b_i | c_j d_j)` (eV).
    pub w: Vec<Vec<S>>,
    /// Electron–core attraction blocks, up to 9×9 for spd atoms (sp fills 4×4).
    pub e1b: [[S; 9]; 9],
    pub e2a: [[S; 9]; 9],
}

impl<S: Scalar> PairTwoElecG<S> {
    #[inline]
    pub fn two_e(&self, a: usize, b: usize, c: usize, d: usize) -> S {
        self.w[pack(a, b)][pack(c, d)]
    }
}

/// f64 alias used by the SCF/Fock.
pub type PairTwoElec = PairTwoElecG<f64>;

/// Compute the rotated two-electron integrals for the ordered pair (i, j) with `xij` the unit
/// vector from i to j and `r` the distance in Bohr (f64 energy path).
pub fn pair_two_electron(ei: &Pm7Element, ej: &Pm7Element, xij: Vec3, r: f64) -> PairTwoElec {
    pair_two_electron_g(ei, ej, [xij.x * r, xij.y * r, xij.z * r])
}

/// Generic driver: `dvec` is the interatomic displacement `R_j − R_i`. Seeding `dvec` with
/// [`Dual`] variables yields the integral derivatives.
pub fn pair_two_electron_g<S: Scalar>(
    ei: &Pm7Element,
    ej: &Pm7Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    // Any d-bearing atom routes through the full MNDO/d two-center kernel.
    if ei.n_orb == 9 || ej.n_orb == 9 {
        return crate::mndod_twocenter::pair_two_electron_d_g(ei, ej, dvec);
    }
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv = r.recip();
    let xij = [dvec[0] * inv, dvec[1] * inv, dvec[2] * inv];

    let heavy_i = ei.has_p();
    let heavy_j = ej.has_p();
    let mut e1b = [[S::cst(0.0); 9]; 9];
    let mut e2a = [[S::cst(0.0); 9]; 9];

    // Two-electron frame uses v = -xij (PySEQM convention).
    let rot = rotation_to_x_g(-xij[0], -xij[1], -xij[2]);
    let r0 = rot[0];
    let r1 = rot[1];
    let r2 = rot[2];

    if !heavy_i && !heavy_j {
        let aee = (ei.rho0 + ej.rho0).powi(2);
        let ee_nddo = (r * r + aee).sqrt().recip() * PM7_EV;
        let (cfrac, point) = feather_to_point(r);
        let ee = ee_nddo * cfrac + point * (S::cst(1.0) - cfrac);
        e1b[0][0] = ee * (-ej.core_charge);
        e2a[0][0] = ee * (-ei.core_charge);
        return PairTwoElecG {
            norb_i: 1,
            norb_j: 1,
            w: vec![vec![ee]],
            e1b,
            e2a,
        };
    }

    if heavy_i && !heavy_j {
        let ri = local_xh_g(ei, ej, r);
        let mut wxh = [S::cst(0.0); 10];
        build_wxh_g(&ri, &r0, &r1, &r2, &mut wxh);
        let mut w = vec![vec![S::cst(0.0)]; 10];
        for (p, wv) in wxh.iter().enumerate() {
            w[p][0] = *wv;
        }
        for a in 0..4 {
            for b in 0..4 {
                e1b[a][b] = wxh[pack(a, b)] * (-ej.core_charge);
            }
        }
        e2a[0][0] = wxh[0] * (-ei.core_charge);
        return PairTwoElecG {
            norb_i: 4,
            norb_j: 1,
            w,
            e1b,
            e2a,
        };
    }

    let ri = local_xx_g(ei, ej, r);
    let w100 = rotate_xx_g(&ri, &r0, &r1, &r2);
    let mut w = vec![vec![S::cst(0.0); 10]; 10];
    for a in 0..10 {
        for b in 0..10 {
            w[a][b] = w100[a * 10 + b];
        }
    }
    for a in 0..4 {
        for b in 0..4 {
            e1b[a][b] = w[pack(a, b)][0] * (-ej.core_charge);
            e2a[a][b] = w[0][pack(a, b)] * (-ei.core_charge);
        }
    }
    PairTwoElecG {
        norb_i: 4,
        norb_j: 4,
        w,
        e1b,
        e2a,
    }
}

fn local_xh_g<S: Scalar>(ei: &Pm7Element, ej: &Pm7Element, r: S) -> [S; 4] {
    let ev1 = PM7_EV / 2.0;
    let ev2 = PM7_EV / 4.0;
    let da = S::cst(ei.dd);
    let qa = S::cst(ei.qq * 2.0);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let ee = (r * r + aee).sqrt().recip() * PM7_EV;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let mut ri = [S::cst(0.0); 4];
    ri[0] = ee;
    ri[1] = ((r + da) * (r + da) + ade).sqrt().recip() * ev1
        - ((r - da) * (r - da) + ade).sqrt().recip() * ev1;
    ri[2] = ee
        + ((r + qa) * (r + qa) + aqe).sqrt().recip() * ev2
        + ((r - qa) * (r - qa) + aqe).sqrt().recip() * ev2
        - ev1dsqr6;
    ri[3] = ee + (r * r + qa * qa + aqe).sqrt().recip() * ev1 - ev1dsqr6;
    feather_ri(&mut ri, r, &XH_MONOPOLE);
    ri
}

fn local_xx_g<S: Scalar>(ei: &Pm7Element, ej: &Pm7Element, r: S) -> [S; 22] {
    let ev1 = PM7_EV / 2.0;
    let ev2 = PM7_EV / 4.0;
    let ev3 = PM7_EV / 8.0;
    let ev4 = PM7_EV / 16.0;
    let da = S::cst(ei.dd);
    let db = S::cst(ej.dd);
    let qa = S::cst(ei.qq * 2.0);
    let qb = S::cst(ej.qq * 2.0);
    let qa1 = S::cst(ei.qq);
    let qb1 = S::cst(ej.qq);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let aed = (ei.rho0 + ej.rho1).powi(2);
    let aeq = (ei.rho0 + ej.rho2).powi(2);
    let axx = (ei.rho1 + ej.rho1).powi(2);
    let adq = (ei.rho1 + ej.rho2).powi(2);
    let aqd = (ei.rho2 + ej.rho1).powi(2);
    let aqq = (ei.rho2 + ej.rho2).powi(2);

    // 1/sqrt(x + c) helper.
    let g = |x: S, c: f64| (x + c).sqrt().recip();
    let sq2 = |a: S, c: f64| (a * a + c).sqrt().recip(); // 1/sqrt(a^2 + c)

    let ee = (r * r + aee).sqrt().recip() * PM7_EV;
    let dze = sq2(r - da, ade) * ev1 - sq2(r + da, ade) * ev1;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let qzze = sq2(r - qa, aqe) * ev2 + sq2(r + qa, aqe) * ev2 - ev1dsqr6;
    let qxxe = g(r * r + qa * qa, aqe) * ev1 - ev1dsqr6;
    let edz = sq2(r + db, aed) * ev1 - sq2(r - db, aed) * ev1;
    let ev1dsqr12 = (r * r + aeq).sqrt().recip() * ev1;
    let eqzz = sq2(r - qb, aeq) * ev2 + sq2(r + qb, aeq) * ev2 - ev1dsqr12;
    let eqxx = g(r * r + qb * qb, aeq) * ev1 - ev1dsqr12;
    let ev2dsqr20 = sq2(r + da, adq) * ev2;
    let ev2dsqr22 = sq2(r - da, adq) * ev2;
    let ev2dsqr24 = sq2(r - db, aqd) * ev2;
    let ev2dsqr26 = sq2(r + db, aqd) * ev2;
    let ev2dsqr36 = (r * r + aqq).sqrt().recip() * ev2;
    let ev2dsqr39 = g(r * r + qa * qa, aqq) * ev2;
    let ev2dsqr40 = g(r * r + qb * qb, aqq) * ev2;
    let ev3dsqr42 = sq2(r - qb, aqq) * ev3;
    let ev3dsqr44 = sq2(r + qb, aqq) * ev3;
    let ev3dsqr46 = sq2(r + qa, aqq) * ev3;
    let ev3dsqr48 = sq2(r - qa, aqq) * ev3;

    let mut ri = [S::cst(0.0); 22];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = sq2(r + da - db, axx) * ev2 + sq2(r - da + db, axx) * ev2
        - sq2(r - da - db, axx) * ev2
        - sq2(r + da + db, axx) * ev2;
    ri[6] =
        g(r * r + (da - db) * (da - db), axx) * ev1 - g(r * r + (da + db) * (da + db), axx) * ev1;
    ri[7] = -edz + sq2(r + qa - db, aqd) * ev3 - sq2(r + qa + db, aqd) * ev3
        + sq2(r - qa - db, aqd) * ev3
        - sq2(r - qa + db, aqd) * ev3
        - ev2dsqr24
        + ev2dsqr26;
    ri[8] = -edz - ev2dsqr24 + g((r - db) * (r - db) + qa * qa, aqd) * ev2 + ev2dsqr26
        - g((r + db) * (r + db) + qa * qa, aqd) * ev2;
    ri[9] = g((qa1 - db) * (qa1 - db) + (r + qa1) * (r + qa1), aqd) * ev2
        - g((qa1 - db) * (qa1 - db) + (r - qa1) * (r - qa1), aqd) * ev2
        - g((qa1 + db) * (qa1 + db) + (r + qa1) * (r + qa1), aqd) * ev2
        + g((qa1 + db) * (qa1 + db) + (r - qa1) * (r - qa1), aqd) * ev2;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = -dze + sq2(r + da - qb, adq) * ev3 - sq2(r - da - qb, adq) * ev3
        + sq2(r + da + qb, adq) * ev3
        - sq2(r - da + qb, adq) * ev3
        + ev2dsqr22
        - ev2dsqr20;
    ri[13] = -dze - ev2dsqr20 + g((r + da) * (r + da) + qb * qb, adq) * ev2 + ev2dsqr22
        - g((r - da) * (r - da) + qb * qb, adq) * ev2;
    ri[14] = g((da - qb1) * (da - qb1) + (r - qb1) * (r - qb1), adq) * ev2
        - g((da - qb1) * (da - qb1) + (r + qb1) * (r + qb1), adq) * ev2
        - g((da + qb1) * (da + qb1) + (r - qb1) * (r - qb1), adq) * ev2
        + g((da + qb1) * (da + qb1) + (r + qb1) * (r + qb1), adq) * ev2;
    ri[15] = ee
        + eqzz
        + qzze
        + sq2(r + qa - qb, aqq) * ev4
        + sq2(r + qa + qb, aqq) * ev4
        + sq2(r - qa - qb, aqq) * ev4
        + sq2(r - qa + qb, aqq) * ev4
        - ev3dsqr48
        - ev3dsqr46
        - ev3dsqr42
        - ev3dsqr44
        + ev2dsqr36;
    ri[16] = ee
        + eqzz
        + qxxe
        + g((r - qb) * (r - qb) + qa * qa, aqq) * ev3
        + g((r + qb) * (r + qb) + qa * qa, aqq) * ev3
        - ev3dsqr42
        - ev3dsqr44
        - ev2dsqr39
        + ev2dsqr36;
    ri[17] = ee
        + eqxx
        + qzze
        + g((r + qa) * (r + qa) + qb * qb, aqq) * ev3
        + g((r - qa) * (r - qa) + qb * qb, aqq) * ev3
        - ev3dsqr46
        - ev3dsqr48
        - ev2dsqr40
        + ev2dsqr36;
    let qxxqxx = g(r * r + (qa - qb) * (qa - qb), aqq) * ev3
        + g(r * r + (qa + qb) * (qa + qb), aqq) * ev3
        - ev2dsqr39
        - ev2dsqr40
        + ev2dsqr36;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = g(
        (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
        aqq,
    ) * ev3
        - g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3;
    let qxxqyy = g(r * r + qa * qa + qb * qb, aqq) * ev2 - ev2dsqr39 - ev2dsqr40 + ev2dsqr36;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = (qxxqxx - qxxqyy) * 0.5;
    feather_ri(&mut ri, r, &XX_MONOPOLE);
    ri
}

fn build_wxh_g<S: Scalar>(ri: &[S; 4], r0: &[S; 3], r1: &[S; 3], r2: &[S; 3], wxh: &mut [S; 10]) {
    wxh[pack(0, 0)] = ri[0];
    for k in 0..3 {
        wxh[pack(k + 1, 0)] = ri[1] * r0[k];
    }
    for k in 0..3 {
        for l in 0..=k {
            let t0 = r0[k] * r0[l];
            let t1 = r1[k] * r1[l] + r2[k] * r2[l];
            wxh[pack(k + 1, l + 1)] = ri[2] * t0 + ri[3] * t1;
        }
    }
}

fn rotate_xx_g<S: Scalar>(ri: &[S; 22], r0: &[S; 3], r1: &[S; 3], r2: &[S; 3]) -> [S; 100] {
    let mut w = [S::cst(0.0); 100];
    let mut idx = 0usize;
    for kk in 0..4usize {
        for ll in 0..=kk {
            for mm in 0..4usize {
                for nn in 0..=mm {
                    let (k, l, m, n) = (
                        kk.wrapping_sub(1),
                        ll.wrapping_sub(1),
                        mm.wrapping_sub(1),
                        nn.wrapping_sub(1),
                    );
                    let val = if kk == 0 {
                        if mm == 0 {
                            ri[0]
                        } else if nn == 0 {
                            ri[4] * r0[m]
                        } else {
                            ri[10] * (r0[m] * r0[n]) + ri[11] * (r1[m] * r1[n] + r2[m] * r2[n])
                        }
                    } else if ll == 0 {
                        if mm == 0 {
                            ri[1] * r0[k]
                        } else if nn == 0 {
                            ri[5] * (r0[k] * r0[m]) + ri[6] * (r1[k] * r1[m] + r2[k] * r2[m])
                        } else {
                            let t0 = r0[k] * r0[m] * r0[n];
                            let t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k];
                            let mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n])
                                + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m]);
                            ri[12] * t0 + ri[13] * t1 + ri[14] * mix
                        }
                    } else if mm == 0 {
                        let t0 = r0[k] * r0[l];
                        let t1 = r1[k] * r1[l] + r2[k] * r2[l];
                        ri[2] * t0 + ri[3] * t1
                    } else if nn == 0 {
                        let t0 = r0[k] * r0[l] * r0[m];
                        let t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m];
                        let t2 = r1[l] * r1[m] + r2[l] * r2[m];
                        ri[7] * t0
                            + ri[8] * t1
                            + ri[9] * (r0[k] * t2 + r0[l] * (r1[k] * r1[m] + r2[k] * r2[m]))
                    } else {
                        let t0 = r0[k] * r0[l] * r0[m] * r0[n];
                        let mut v = ri[15] * t0;
                        v = v + ri[16] * ((r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n]);
                        v = v + ri[17] * ((r1[m] * r1[n] + r2[m] * r2[n]) * (r0[k] * r0[l]));
                        v = v + ri[18]
                            * (r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n]);
                        let mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n]);
                        let mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m]);
                        let val5 = r0[k] * (mix1 + mix2)
                            + r0[l]
                                * (r0[m] * (r1[k] * r1[n] + r2[k] * r2[n])
                                    + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m]));
                        v = v + ri[19] * val5;
                        v = v + ri[20]
                            * (r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n]);
                        let cross =
                            (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n]);
                        v = v + ri[21] * cross;
                        v
                    };
                    w[idx] = val;
                    idx += 1;
                }
            }
        }
    }
    w
}

/// Dual-valued two-electron integrals for a pair, seeded on the displacement `R_j − R_i`.
pub fn pair_two_electron_dual(ei: &Pm7Element, ej: &Pm7Element, dvec: Vec3) -> PairTwoElecG<Dual> {
    pair_two_electron_g(
        ei,
        ej,
        [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm7Parameters;

    #[test]
    fn rotation_is_orthonormal() {
        let v = Vec3::new(0.3, -0.5, 0.8).normalized();
        let r = rotation_to_x(v);
        for i in 0..3 {
            let ni: f64 = (0..3).map(|k| r[i][k] * r[i][k]).sum();
            assert!((ni - 1.0).abs() < 1e-10);
        }
        let rv0: f64 = (0..3).map(|k| r[0][k] * v.get(k)).sum();
        assert!((rv0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ssss_is_rotation_invariant() {
        let p = Pm7Parameters::standard().unwrap();
        let c = p.element(6).unwrap();
        let r = 2.6;
        let a = pair_two_electron(c, c, Vec3::new(1.0, 0.0, 0.0), r);
        let b = pair_two_electron(c, c, Vec3::new(0.3, -0.5, 0.8).normalized(), r);
        assert!((a.w[0][0] - b.w[0][0]).abs() < 1e-9);
        // The (ss|ss) integral is feathered toward the point charge (PM7 `l_feather`).
        let nddo = PM7_EV / (r * r + (2.0 * c.rho0).powi(2)).sqrt();
        let (cfrac, point) = feather_to_point(r);
        let expect = nddo * cfrac + point * (1.0 - cfrac);
        assert!((a.w[0][0] - expect).abs() < 1e-9);
    }

    #[test]
    fn dual_two_electron_matches_fd() {
        // The dual derivative of a two-electron integral must match a finite difference.
        let p = Pm7Parameters::standard().unwrap();
        let (c, o) = (p.element(6).unwrap(), p.element(8).unwrap());
        let d = Vec3::new(1.4, -0.9, 0.7);
        let dual = pair_two_electron_dual(c, o, d);
        let h = 1e-6;
        let mut max_delta = 0.0_f64;
        for axis in 0..3 {
            let mut dp = d;
            let mut dm = d;
            match axis {
                0 => {
                    dp.x += h;
                    dm.x -= h;
                }
                1 => {
                    dp.y += h;
                    dm.y -= h;
                }
                _ => {
                    dp.z += h;
                    dm.z -= h;
                }
            }
            let wp = pair_two_electron_g::<f64>(c, o, [dp.x, dp.y, dp.z]);
            let wm = pair_two_electron_g::<f64>(c, o, [dm.x, dm.y, dm.z]);
            for a in 0..10 {
                for b in 0..10 {
                    let fd = (wp.w[a][b] - wm.w[a][b]) / (2.0 * h);
                    max_delta = max_delta.max((dual.w[a][b].d[axis] - fd).abs());
                }
            }
        }
        eprintln!("dual 2e derivative max delta = {max_delta:.2e}");
        assert!(
            max_delta < 1e-6,
            "dual 2e derivative mismatch {max_delta:.3e}"
        );
    }
}
