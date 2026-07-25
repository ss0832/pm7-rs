// SPDX-License-Identifier: GPL-3.0-or-later

//! MNDO/d two-center two-electron and electron–core integrals for the PM7 `spd`
//! basis, generic over [`crate::dual::Scalar`].
//!
//! Faithful port of MOPAC v23.2.5 `src/integrals/{mndod,rotate}.F90`
//! (`rotmat`, `reppd`, `spcore`, `reppd2`, `rijkl`, `tx`, `rotatd`, `elenuc`).
//! All arrays use MOPAC's 1-based indexing (slot 0 unused) so the ported code
//! reads like the Fortran. Distances are in Bohr, energies in eV.
//!
//! The single geometric variable is the interatomic displacement; seeding it with
//! [`crate::dual::Dual`]/[`crate::dual2::Dual2`] yields the exact analytic gradient
//! and Hessian of the d-orbital integrals with no finite differences.
//!
//! Provenance: MOPAC, Apache-2.0 (c) 2021 Virginia Tech. See `THIRD_PARTY_NOTICES.md`.

use crate::constants::PM7_EV;
use crate::dual::Scalar;
use crate::integrals::PairTwoElecG;
use crate::mndod::charg;
use crate::mndod_tables::{CH, IND2, INDDD, INDDP, INDPP, ISYM};
use crate::params::Pm7Element;

// Orbital-pair classification of the 45 local charge distributions (MOPAC `met`).
// 1 = ss, 2 = sp, 3 = pp, 4 = sd, 5 = pd, 6 = dd. Index 1..=45.
const MET: [u8; 46] = [
    0, // slot 0 unused
    1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, 4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5,
    6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6,
];

// rep(1:34) = ri(IPOS) mapping (MOPAC reppd2). Index 1..=34.
const IPOS: [usize; 35] = [
    0, // slot 0 unused
    1, 5, 11, 12, 12, 2, 6, 13, 14, 14, 3, 8, 16, 18, 18, 7, 15, 10, 20, 4, 9, 17, 19, 21, 7, 15,
    10, 20, 22, 4, 9, 17, 21, 19,
];

// Sign convention applied to the 22 sp local integrals (MOPAC `nri`). Index 1..=22.
const NRI: [f64; 23] = [
    0.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];

/// MOPAC `indexd(i,j)` charge-distribution index (1..45), symmetric. 1-based i,j.
#[inline]
fn indexd(i: usize, j: usize) -> usize {
    let (i, j) = if i >= j { (i, j) } else { (j, i) };
    (i + 9 * (j - 1)) - (j * (j - 1)) / 2
}

/// MOPAC `indx(i,j)` lower-triangle pair index (1-based).
#[inline]
fn indx(i: usize, j: usize) -> usize {
    let (i, j) = if i >= j { (i, j) } else { (j, i) };
    (i * (i - 1)) / 2 + j
}

#[inline]
fn ch(i: usize, l: i32, m: i32) -> f64 {
    CH[((i - 1) * 3 + l as usize) * 5 + (m + 2) as usize]
}

/// Rotation tensors for the pair (1-based storage matching MOPAC).
struct Rot<S: Scalar> {
    sp: [[S; 4]; 4],       // sp[1..3][1..3] = p
    pp: [[[S; 4]; 4]; 7],  // pp[1..6][1..3][1..3]
    sd: [[S; 6]; 6],       // sd[1..5][1..5] = d
    dp: [[[S; 4]; 6]; 16], // dp[1..15][1..5][1..3]
    dd: [[[S; 6]; 6]; 16], // d_d[1..15][1..5][1..5]
}

/// MOPAC `rotmat`: build the p/d rotation tensors from the i→j displacement
/// `v = R_j − R_i` (Bohr). Returns the distance and the tensors.
fn rotmat<S: Scalar>(v: [S; 3], has_d: bool) -> (S, Rot<S>) {
    const SMALL: f64 = 1.0e-7;
    const PT5SQ3: f64 = 0.866_025_403_784_1;
    let (x11, x22, x33) = (v[0], v[1], v[2]);
    let b = x11 * x11 + x22 * x22;
    let r = (b + x33 * x33).sqrt();
    let sqb = b.sqrt();
    let mut sb = sqb * r.recip();
    let (ca, sa, cb);
    if sb.val() > SMALL {
        ca = x11 * sqb.recip();
        sa = x22 * sqb.recip();
        cb = x33 * r.recip();
    } else {
        sa = S::cst(0.0);
        sb = S::cst(0.0);
        if x33.val() < 0.0 {
            ca = S::cst(-1.0);
            cb = S::cst(-1.0);
        } else if x33.val() > 0.0 {
            ca = S::cst(1.0);
            cb = S::cst(1.0);
        } else {
            ca = S::cst(0.0);
            cb = S::cst(0.0);
        }
    }
    let mut p = [[S::cst(0.0); 4]; 4];
    p[1][1] = ca * sb;
    p[2][1] = ca * cb;
    p[3][1] = -sa;
    p[1][2] = sa * sb;
    p[2][2] = sa * cb;
    p[3][2] = ca;
    p[1][3] = cb;
    p[2][3] = -sb;
    p[3][3] = S::cst(0.0);

    let mut d = [[S::cst(0.0); 6]; 6];
    if has_d {
        let c2a = ca * ca * 2.0 - 1.0;
        let c2b = cb * cb * 2.0 - 1.0;
        let s2a = sa * ca * 2.0;
        let s2b = sb * cb * 2.0;
        d[1][1] = c2a * sb * sb * PT5SQ3;
        d[2][1] = c2a * s2b * 0.5;
        d[3][1] = -(s2a * sb);
        d[4][1] = c2a * (cb * cb + sb * sb * 0.5);
        d[5][1] = -(s2a * cb);
        d[1][2] = ca * s2b * PT5SQ3;
        d[2][2] = ca * c2b;
        d[3][2] = -(sa * cb);
        d[4][2] = ca * s2b * (-0.5);
        d[5][2] = sa * sb;
        d[1][3] = cb * cb - sb * sb * 0.5;
        d[2][3] = -(s2b * PT5SQ3);
        d[3][3] = S::cst(0.0);
        d[4][3] = sb * sb * PT5SQ3;
        d[5][3] = S::cst(0.0);
        d[1][4] = sa * s2b * PT5SQ3;
        d[2][4] = sa * c2b;
        d[3][4] = ca * cb;
        d[4][4] = sa * s2b * (-0.5);
        d[5][4] = -(ca * sb);
        d[1][5] = s2a * sb * sb * PT5SQ3;
        d[2][5] = s2a * s2b * 0.5;
        d[3][5] = c2a * sb;
        d[4][5] = s2a * (cb * cb + sb * sb * 0.5);
        d[5][5] = c2a * cb;
    }

    // S-P: sp = p.
    let sp = p;
    // P-P.
    let mut pp = [[[S::cst(0.0); 4]; 4]; 7];
    for k in 1..=3 {
        pp[1][k][k] = p[k][1] * p[k][1];
        pp[2][k][k] = p[k][2] * p[k][2];
        pp[3][k][k] = p[k][3] * p[k][3];
        pp[4][k][k] = p[k][1] * p[k][2];
        pp[5][k][k] = p[k][1] * p[k][3];
        pp[6][k][k] = p[k][2] * p[k][3];
        for j in 1..k {
            pp[1][k][j] = p[k][1] * p[j][1] * 2.0;
            pp[2][k][j] = p[k][2] * p[j][2] * 2.0;
            pp[3][k][j] = p[k][3] * p[j][3] * 2.0;
            pp[4][k][j] = p[k][1] * p[j][2] + p[k][2] * p[j][1];
            pp[5][k][j] = p[k][1] * p[j][3] + p[k][3] * p[j][1];
            pp[6][k][j] = p[k][2] * p[j][3] + p[k][3] * p[j][2];
        }
    }

    let mut sd = [[S::cst(0.0); 6]; 6];
    let mut dp = [[[S::cst(0.0); 4]; 6]; 16];
    let mut dd = [[[S::cst(0.0); 6]; 6]; 16];
    if has_d {
        // S-D.
        sd = d;
        // D-P.
        for k in 1..=5 {
            for col in 1..=3 {
                dp[1][k][col] = d[k][1] * p[col][1];
                dp[2][k][col] = d[k][1] * p[col][2];
                dp[3][k][col] = d[k][1] * p[col][3];
                dp[4][k][col] = d[k][2] * p[col][1];
                dp[5][k][col] = d[k][2] * p[col][2];
                dp[6][k][col] = d[k][2] * p[col][3];
                dp[7][k][col] = d[k][3] * p[col][1];
                dp[8][k][col] = d[k][3] * p[col][2];
                dp[9][k][col] = d[k][3] * p[col][3];
                dp[10][k][col] = d[k][4] * p[col][1];
                dp[11][k][col] = d[k][4] * p[col][2];
                dp[12][k][col] = d[k][4] * p[col][3];
                dp[13][k][col] = d[k][5] * p[col][1];
                dp[14][k][col] = d[k][5] * p[col][2];
                dp[15][k][col] = d[k][5] * p[col][3];
            }
        }
        // D-D.
        for k in 1..=5 {
            dd[1][k][k] = d[k][1] * d[k][1];
            dd[2][k][k] = d[k][2] * d[k][2];
            dd[3][k][k] = d[k][3] * d[k][3];
            dd[4][k][k] = d[k][4] * d[k][4];
            dd[5][k][k] = d[k][5] * d[k][5];
            dd[6][k][k] = d[k][1] * d[k][2];
            dd[7][k][k] = d[k][1] * d[k][3];
            dd[8][k][k] = d[k][2] * d[k][3];
            dd[9][k][k] = d[k][1] * d[k][4];
            dd[10][k][k] = d[k][2] * d[k][4];
            dd[11][k][k] = d[k][3] * d[k][4];
            dd[12][k][k] = d[k][1] * d[k][5];
            dd[13][k][k] = d[k][2] * d[k][5];
            dd[14][k][k] = d[k][3] * d[k][5];
            dd[15][k][k] = d[k][4] * d[k][5];
            for j in 1..k {
                dd[1][k][j] = d[k][1] * d[j][1] * 2.0;
                dd[2][k][j] = d[k][2] * d[j][2] * 2.0;
                dd[3][k][j] = d[k][3] * d[j][3] * 2.0;
                dd[4][k][j] = d[k][4] * d[j][4] * 2.0;
                dd[5][k][j] = d[k][5] * d[j][5] * 2.0;
                dd[6][k][j] = d[k][1] * d[j][2] + d[k][2] * d[j][1];
                dd[7][k][j] = d[k][1] * d[j][3] + d[k][3] * d[j][1];
                dd[8][k][j] = d[k][2] * d[j][3] + d[k][3] * d[j][2];
                dd[9][k][j] = d[k][1] * d[j][4] + d[k][4] * d[j][1];
                dd[10][k][j] = d[k][2] * d[j][4] + d[k][4] * d[j][2];
                dd[11][k][j] = d[k][3] * d[j][4] + d[k][4] * d[j][3];
                dd[12][k][j] = d[k][1] * d[j][5] + d[k][5] * d[j][1];
                dd[13][k][j] = d[k][2] * d[j][5] + d[k][5] * d[j][2];
                dd[14][k][j] = d[k][3] * d[j][5] + d[k][5] * d[j][3];
                dd[15][k][j] = d[k][4] * d[j][5] + d[k][5] * d[j][4];
            }
        }
    }
    (r, Rot { sp, pp, sd, dp, dd })
}

// Convenience: additive/charge-separation accessors on a d-capable element.
#[inline]
fn po(e: &Pm7Element, i: usize) -> f64 {
    e.dshell.as_ref().expect("dshell present").po[i]
}
#[inline]
fn ddp(e: &Pm7Element, i: usize) -> f64 {
    e.dshell.as_ref().expect("dshell present").ddp[i]
}

/// MOPAC `reppd`: the 22 sp local two-center integrals `ri(1..22)` (eV). Returns
/// the sign-corrected `ri`. Distances in Bohr.
fn reppd<S: Scalar>(ei: &Pm7Element, ej: &Pm7Element, r: S) -> [S; 23] {
    let ev = PM7_EV;
    let ev1 = ev / 2.0;
    let ev2 = ev1 / 2.0;
    let ev3 = ev2 / 2.0;
    let ev4 = ev3 / 2.0;
    let mut ri = [S::cst(0.0); 23];
    let si = ei.n_orb >= 4;
    let sj = ej.n_orb >= 4;
    let sq = |x: S, off: f64| (x * x + off).sqrt().recip();

    // aee for the two-electron term uses po(1); the core-core gab (po(9)) is
    // handled separately in repulsion.rs and not needed here.
    let aee = {
        let t = po(ei, 1) + po(ej, 1);
        t * t
    };
    if !si && !sj {
        ri[1] = sq(r, aee) * ev;
    } else if si && !sj {
        let da = ei.dd;
        let qa = ei.qq * 2.0;
        let ade = {
            let t = po(ei, 2) + po(ej, 1);
            t * t
        };
        let aqe = {
            let t = po(ei, 3) + po(ej, 1);
            t * t
        };
        let ee = sq(r, aee) * ev;
        ri[1] = ee;
        ri[2] = sq(r + da, ade) * ev1 - sq(r - da, ade) * ev1;
        ri[3] = ee + sq(r + qa, aqe) * ev2 + sq(r - qa, aqe) * ev2 - sq(r, aqe) * ev1;
        ri[4] = ee + (r * r + (qa * qa + aqe)).sqrt().recip() * ev1 - sq(r, aqe) * ev1;
    } else if !si && sj {
        let db = ej.dd;
        let qb = ej.qq * 2.0;
        let aed = {
            let t = po(ei, 1) + po(ej, 2);
            t * t
        };
        let aeq = {
            let t = po(ei, 1) + po(ej, 3);
            t * t
        };
        let ee = sq(r, aee) * ev;
        ri[1] = ee;
        ri[5] = sq(r - db, aed) * ev1 - sq(r + db, aed) * ev1;
        ri[11] = ee + sq(r - qb, aeq) * ev2 + sq(r + qb, aeq) * ev2 - sq(r, aeq) * ev1;
        ri[12] = ee + (r * r + (qb * qb + aeq)).sqrt().recip() * ev1 - sq(r, aeq) * ev1;
    } else {
        // heavy–heavy.
        let da = ei.dd;
        let db = ej.dd;
        let qa = ei.qq * 2.0;
        let qb = ej.qq * 2.0;
        let sqp = |a: f64, b: f64| {
            let t = a + b;
            t * t
        };
        let ade = sqp(po(ei, 2), po(ej, 1));
        let aqe = sqp(po(ei, 3), po(ej, 1));
        let aed = sqp(po(ei, 1), po(ej, 2));
        let aeq = sqp(po(ei, 1), po(ej, 3));
        let axx = sqp(po(ei, 2), po(ej, 2));
        let adq = sqp(po(ei, 2), po(ej, 3));
        let aqd = sqp(po(ei, 3), po(ej, 2));
        let aqq = sqp(po(ei, 3), po(ej, 3));
        // inv(shift, off) = 1/sqrt((r+shift)^2 + off) with shift/off constants.
        let g = |x: S, off: f64| (x * x + off).sqrt().recip();
        let ee = g(r, aee) * ev;
        let dze = -(sq(r + da, ade)) * ev1 + sq(r - da, ade) * ev1;
        let qzze = sq(r - qa, aqe) * ev2 + sq(r + qa, aqe) * ev2 - g(r, aqe) * ev1;
        let qxxe = (r * r + (qa * qa + aqe)).sqrt().recip() * ev1 - g(r, aqe) * ev1;
        let edz = -(sq(r - db, aed)) * ev1 + sq(r + db, aed) * ev1;
        let eqzz = sq(r - qb, aeq) * ev2 + sq(r + qb, aeq) * ev2 - g(r, aeq) * ev1;
        let eqxx = (r * r + (qb * qb + aeq)).sqrt().recip() * ev1 - g(r, aeq) * ev1;
        let dxdx = (r * r + ((da - db) * (da - db) + axx)).sqrt().recip() * ev1
            - (r * r + ((da + db) * (da + db) + axx)).sqrt().recip() * ev1;
        let dzdz = sq(r + da - db, axx) * ev2 + sq(r - da + db, axx) * ev2
            - sq(r - da - db, axx) * ev2
            - sq(r + da + db, axx) * ev2;
        // Terms using single-q (qa1,qb1).
        let qa1 = ei.qq;
        let qb1 = ej.qq;
        let dzqxx = sq(r + da, adq) * ev2
            - ((r + da) * (r + da) + (qb * qb + adq)).sqrt().recip() * ev2
            - sq(r - da, adq) * ev2
            + ((r - da) * (r - da) + (qb * qb + adq)).sqrt().recip() * ev2;
        let qxxdz = sq(r - db, aqd) * ev2
            - ((r - db) * (r - db) + (qa * qa + aqd)).sqrt().recip() * ev2
            - sq(r + db, aqd) * ev2
            + ((r + db) * (r + db) + (qa * qa + aqd)).sqrt().recip() * ev2;
        // Note: the two trailing ev2 terms use the plain (r∓da)²+adq / (r∓db)²+aqd
        // arguments (MOPAC sqr(20/22) and sqr(24/26)) — no extra quadrupole offset.
        let dzqzz = -(sq(r + da - qb, adq)) * ev3 + sq(r - da - qb, adq) * ev3
            - sq(r + da + qb, adq) * ev3
            + sq(r - da + qb, adq) * ev3
            - sq(r - da, adq) * ev2
            + sq(r + da, adq) * ev2;
        let qzzdz = -(sq(r + qa - db, aqd)) * ev3 + sq(r + qa + db, aqd) * ev3
            - sq(r - qa - db, aqd) * ev3
            + sq(r - qa + db, aqd) * ev3
            + sq(r - db, aqd) * ev2
            - sq(r + db, aqd) * ev2;
        // The qxx*qxx family uses the two-q separations (MOPAC args 36..53 are
        // precomputed with qa=2·qq before the single-q reassignment); only the
        // dxqxz/qxzdx/qxzqxz terms below use the single-q qa1/qb1.
        let qxxqxx = (r * r + ((qa - qb) * (qa - qb) + aqq)).sqrt().recip() * ev3
            + (r * r + ((qa + qb) * (qa + qb) + aqq)).sqrt().recip() * ev3
            - (r * r + (qa * qa + aqq)).sqrt().recip() * ev2
            - (r * r + (qb * qb + aqq)).sqrt().recip() * ev2
            + g(r, aqq) * ev2;
        let qxxqyy = (r * r + (qa * qa + qb * qb + aqq)).sqrt().recip() * ev2
            - (r * r + (qa * qa + aqq)).sqrt().recip() * ev2
            - (r * r + (qb * qb + aqq)).sqrt().recip() * ev2
            + g(r, aqq) * ev2;
        let qxxqzz = ((r - qb) * (r - qb) + (qa * qa + aqq)).sqrt().recip() * ev3
            + ((r + qb) * (r + qb) + (qa * qa + aqq)).sqrt().recip() * ev3
            - sq(r - qb, aqq) * ev3
            - sq(r + qb, aqq) * ev3
            - (r * r + (qa * qa + aqq)).sqrt().recip() * ev2
            + g(r, aqq) * ev2;
        let qzzqxx = ((r + qa) * (r + qa) + (qb * qb + aqq)).sqrt().recip() * ev3
            + ((r - qa) * (r - qa) + (qb * qb + aqq)).sqrt().recip() * ev3
            - sq(r + qa, aqq) * ev3
            - sq(r - qa, aqq) * ev3
            - (r * r + (qb * qb + aqq)).sqrt().recip() * ev2
            + g(r, aqq) * ev2;
        let qzzqzz = sq(r + qa - qb, aqq) * ev4
            + sq(r + qa + qb, aqq) * ev4
            + sq(r - qa - qb, aqq) * ev4
            + sq(r - qa + qb, aqq) * ev4
            - sq(r - qa, aqq) * ev3
            - sq(r + qa, aqq) * ev3
            - sq(r - qb, aqq) * ev3
            - sq(r + qb, aqq) * ev3
            + g(r, aqq) * ev2;
        // dxqxz / qxzdx / qxzqxz use qa1/qb1 (single q).
        let dxqxz = -((r - qb1) * (r - qb1) + ((da - qb1) * (da - qb1) + adq))
            .sqrt()
            .recip()
            * ev2
            + ((r + qb1) * (r + qb1) + ((da - qb1) * (da - qb1) + adq))
                .sqrt()
                .recip()
                * ev2
            + ((r - qb1) * (r - qb1) + ((da + qb1) * (da + qb1) + adq))
                .sqrt()
                .recip()
                * ev2
            - ((r + qb1) * (r + qb1) + ((da + qb1) * (da + qb1) + adq))
                .sqrt()
                .recip()
                * ev2;
        let qxzdx = -((r + qa1) * (r + qa1) + ((qa1 - db) * (qa1 - db) + aqd))
            .sqrt()
            .recip()
            * ev2
            + ((r - qa1) * (r - qa1) + ((qa1 - db) * (qa1 - db) + aqd))
                .sqrt()
                .recip()
                * ev2
            + ((r + qa1) * (r + qa1) + ((qa1 + db) * (qa1 + db) + aqd))
                .sqrt()
                .recip()
                * ev2
            - ((r - qa1) * (r - qa1) + ((qa1 + db) * (qa1 + db) + aqd))
                .sqrt()
                .recip()
                * ev2;
        let qxzqxz = ((r + qa1 - qb1) * (r + qa1 - qb1) + ((qa1 - qb1) * (qa1 - qb1) + aqq))
            .sqrt()
            .recip()
            * ev3
            - ((r + qa1 + qb1) * (r + qa1 + qb1) + ((qa1 - qb1) * (qa1 - qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            - ((r - qa1 - qb1) * (r - qa1 - qb1) + ((qa1 - qb1) * (qa1 - qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            + ((r - qa1 + qb1) * (r - qa1 + qb1) + ((qa1 - qb1) * (qa1 - qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            - ((r + qa1 - qb1) * (r + qa1 - qb1) + ((qa1 + qb1) * (qa1 + qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            + ((r + qa1 + qb1) * (r + qa1 + qb1) + ((qa1 + qb1) * (qa1 + qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            + ((r - qa1 - qb1) * (r - qa1 - qb1) + ((qa1 + qb1) * (qa1 + qb1) + aqq))
                .sqrt()
                .recip()
                * ev3
            - ((r - qa1 + qb1) * (r - qa1 + qb1) + ((qa1 + qb1) * (qa1 + qb1) + aqq))
                .sqrt()
                .recip()
                * ev3;

        ri[1] = ee;
        ri[2] = -dze;
        ri[3] = ee + qzze;
        ri[4] = ee + qxxe;
        ri[5] = -edz;
        ri[6] = dzdz;
        ri[7] = dxdx;
        ri[8] = -edz - qzzdz;
        ri[9] = -edz - qxxdz;
        ri[10] = -qxzdx;
        ri[11] = ee + eqzz;
        ri[12] = ee + eqxx;
        ri[13] = -dze - dzqzz;
        ri[14] = -dze - dzqxx;
        ri[15] = -dxqxz;
        ri[16] = ee + eqzz + qzze + qzzqzz;
        ri[17] = ee + eqzz + qxxe + qxxqzz;
        ri[18] = ee + eqxx + qzze + qzzqxx;
        ri[19] = ee + eqxx + qxxe + qxxqxx;
        ri[20] = qxzqxz;
        ri[21] = ee + eqxx + qxxe + qxxqyy;
        ri[22] = (qxxqxx - qxxqyy) * 0.5;
    }
    for k in 1..=22 {
        ri[k] = ri[k] * NRI[k];
    }
    // PM7 feathering (MOPAC `reppd.F90:800-824`, always on): the charge–charge integrals
    // decay to the point charge and the rest to zero as the atoms separate.  Applied
    // unconditionally over all 22 (as MOPAC does) so the built `rep`/`w` match bit-for-bit.
    // 1-based charge–charge indices {1,3,4,11,12,16,17,18,19,21}; NRI = +1 for all of them,
    // so feathering commutes with the sign multiplication above.
    crate::integrals::feather_ri(&mut ri, r, &RI_MONOPOLE);
    ri
}

/// Charge–charge integrals among the 22 sp local integrals (MOPAC 1-based indices).
const RI_MONOPOLE: [usize; 10] = [1, 3, 4, 11, 12, 16, 17, 18, 19, 21];
/// Charge–charge distributions among the 10 electron–core `core` integrals (MOPAC
/// `rotatd.F90:1160-1181` feathers exactly {1,3,4,7,9,10} toward the point charge).
const CORE_MONOPOLE: [usize; 6] = [1, 3, 4, 7, 9, 10];

/// MOPAC `spcore`: the sp core–electron attraction integrals `core[1..4][1..2]`
/// (eV). `core[..][1]` = electron on i, nucleus of j; `core[..][2]` = the reverse.
fn spcore<S: Scalar>(ei: &Pm7Element, ej: &Pm7Element, r: S) -> [[S; 3]; 11] {
    let ev = PM7_EV;
    let pxy = [0.0, 1.0, -0.5, -0.5, 0.5, 0.25, 0.25, 0.5];
    let mut core = [[S::cst(0.0); 3]; 11];
    let r2 = r * r;
    let aci = po(ei, 9);
    let acj = po(ej, 9);
    let ssj = {
        let t = acj + po(ei, 1);
        t * t
    };
    let ssi = {
        let t = aci + po(ej, 1);
        t * t
    };
    core[1][1] = -((r2 + ssj).sqrt().recip()) * (ei_tore(ej) * ev);
    core[1][2] = -((r2 + ssi).sqrt().recip()) * (ei_tore(ei) * ev);
    if ei.n_orb >= 4 || ej.n_orb >= 4 {
        if ei.n_orb >= 4 {
            let ppj = {
                let t = acj + po(ei, 7);
                t * t
            };
            let da = ddp(ei, 2);
            let qa = ddp(ei, 3) / 2.0_f64.sqrt();
            let twoqa = qa + qa;
            let adj = {
                let t = po(ei, 2) + acj;
                t * t
            };
            let aqj = {
                let t = po(ei, 3) + acj;
                t * t
            };
            let x1 = (r2 + ppj).sqrt().recip() * pxy[1];
            let x2 = (r2 + aqj).sqrt().recip() * pxy[2];
            let x3 = ((r + da) * (r + da) + adj).sqrt().recip() * pxy[3];
            let x4 = ((r - da) * (r - da) + adj).sqrt().recip() * pxy[4];
            let x5 = ((r - twoqa) * (r - twoqa) + aqj).sqrt().recip() * pxy[5];
            let x6 = ((r + twoqa) * (r + twoqa) + aqj).sqrt().recip() * pxy[6];
            let x7 = (r2 + (twoqa * twoqa + aqj)).sqrt().recip() * pxy[7];
            let aj2 = (x3 + x4) * ev;
            let aj3 = (x1 + x2 + x5 + x6) * ev;
            let aj4 = (x1 + x2 + x7) * ev;
            core[2][1] = -(aj2) * ei_tore(ej);
            core[3][1] = -(aj3) * ei_tore(ej);
            core[4][1] = -(aj4) * ei_tore(ej);
        }
        if ej.n_orb >= 4 {
            let ppi = {
                let t = aci + po(ej, 7);
                t * t
            };
            let db = ddp(ej, 2);
            let qb = ddp(ej, 3) / 2.0_f64.sqrt();
            let twoqb = qb + qb;
            let adi = {
                let t = po(ej, 2) + aci;
                t * t
            };
            let aqi = {
                let t = po(ej, 3) + aci;
                t * t
            };
            let x1 = (r2 + ppi).sqrt().recip() * pxy[1];
            let x2 = (r2 + aqi).sqrt().recip() * pxy[2];
            let x3 = ((r + db) * (r + db) + adi).sqrt().recip() * pxy[3];
            let x4 = ((r - db) * (r - db) + adi).sqrt().recip() * pxy[4];
            let x5 = ((r - twoqb) * (r - twoqb) + aqi).sqrt().recip() * pxy[5];
            let x6 = ((r + twoqb) * (r + twoqb) + aqi).sqrt().recip() * pxy[6];
            let x7 = (r2 + (twoqb * twoqb + aqi)).sqrt().recip() * pxy[7];
            let ai2 = -(x3 + x4) * ev;
            let ai3 = (x1 + x2 + x5 + x6) * ev;
            let ai4 = (x1 + x2 + x7) * ev;
            core[2][2] = -(ai2) * ei_tore(ei);
            core[3][2] = -(ai3) * ei_tore(ei);
            core[4][2] = -(ai4) * ei_tore(ei);
        }
    }
    core
}

#[inline]
fn ei_tore(e: &Pm7Element) -> f64 {
    e.core_charge
}

/// MOPAC `rijkl`: a single local two-center integral (eV-less) via the charg
/// multipole model. `ij`/`kl` are charge-distribution indices (1..45); `li..ll`
/// are the orbital angular momenta; `ic` selects the core additive term.
#[allow(clippy::too_many_arguments)]
fn rijkl<S: Scalar>(
    ei: &Pm7Element,
    ej: &Pm7Element,
    ij: usize,
    kl: usize,
    li: i32,
    lj: i32,
    lk: i32,
    ll: i32,
    ic: i32,
    r: S,
) -> S {
    let lij = indx((li + 1) as usize, (lj + 1) as usize);
    let lkl = indx((lk + 1) as usize, (ll + 1) as usize);
    let l1min = (li - lj).abs().min(2);
    let l1max = (li + lj).min(2);
    let l2min = (lk - ll).abs().min(2);
    let l2max = (lk + ll).min(2);
    let mut sum = S::cst(0.0);
    for l1 in l1min..=l1max {
        let (dij, pij);
        if l1 == 0 {
            dij = 0.0;
            pij = match lij {
                1 => {
                    if ic == 1 {
                        po(ei, 9)
                    } else {
                        po(ei, 1)
                    }
                }
                3 => po(ei, 7),
                6 => po(ei, 8),
                _ => po(ei, 1),
            };
        } else {
            dij = ddp(ei, lij);
            pij = po(ei, lij);
        }
        for l2 in l2min..=l2max {
            let (dkl, pkl);
            if l2 == 0 {
                dkl = 0.0;
                pkl = match lkl {
                    1 => {
                        if ic == 2 {
                            po(ej, 9)
                        } else {
                            po(ej, 1)
                        }
                    }
                    3 => po(ej, 7),
                    6 => po(ej, 8),
                    _ => po(ej, 1),
                };
            } else {
                dkl = ddp(ej, lkl);
                pkl = po(ej, lkl);
            }
            let add = (pij + pkl) * (pij + pkl);
            let lmin = l1.min(l2);
            for m in -lmin..=lmin {
                let ccc = ch(ij, l1, m) * ch(kl, l2, m);
                if ccc == 0.0 {
                    continue;
                }
                let mm = m.unsigned_abs() as u8;
                sum = sum + charg(r, l1 as u8, l2 as u8, mm, dij, dkl, add) * ccc;
            }
        }
    }
    sum
}

/// MOPAC `reppd2`: assemble the 491 local integrals `rep` and the d core terms.
fn reppd2<S: Scalar>(
    ei: &Pm7Element,
    ej: &Pm7Element,
    r: S,
    ri: &[S; 23],
    core: &mut [[S; 3]; 11],
) -> Vec<S> {
    let ev = PM7_EV;
    let dorbs_i = ei.n_orb == 9;
    let dorbs_j = ej.n_orb == 9;
    let lorb = [0i32, 0, 1, 1, 1, 2, 2, 2, 2, 2]; // lorb[1..9]
    let mut rep = vec![S::cst(0.0); 492];
    for k in 1..=34 {
        rep[k] = ri[IPOS[k]];
    }
    if dorbs_i || dorbs_j {
        // PM7 feathering of the local `d` two-electron integrals (MOPAC `reppd2.F90:927-934`):
        // a charge–charge (`coulomb`) distribution decays to the point charge, all others to 0.
        let (cfrac, point) = crate::integrals::feather_to_point(r);
        let point_add = point * (S::cst(1.0) - cfrac);
        let lasti = if dorbs_i {
            9
        } else if ei.n_orb == 1 {
            1
        } else {
            4
        };
        let lastk = if dorbs_j {
            9
        } else if ej.n_orb == 1 {
            1
        } else {
            4
        };
        for i in 1..=lasti {
            let li = lorb[i];
            for j in 1..=i {
                let lj = lorb[j];
                let ij = indexd(i, j);
                for k in 1..=lastk {
                    let lk = lorb[k];
                    for l in 1..=k {
                        let ll = lorb[l];
                        let kl = indexd(k, l);
                        let numb = IND2[(ij - 1) * 45 + (kl - 1)] as usize;
                        if numb <= 34 {
                            continue;
                        }
                        let nold = ISYM[numb];
                        if nold >= 35 {
                            // symmetry copy of an already-feathered integral
                            rep[numb] = rep[nold as usize];
                        } else if nold <= -35 {
                            rep[numb] = -rep[(-nold) as usize];
                        } else if nold == 0 {
                            let v = rijkl(ei, ej, ij, kl, li, lj, lk, ll, 0, r) * ev;
                            // coulomb = charge–charge distribution (i==j and k==l)
                            rep[numb] = if i == j && k == l {
                                v * cfrac + point_add
                            } else {
                                v * cfrac
                            };
                        }
                    }
                }
            }
        }
        for n in 5..=10 {
            core[n][1] = S::cst(0.0);
            core[n][2] = S::cst(0.0);
        }
        if dorbs_j {
            let tore_i = ei_tore(ei);
            let ij = 1;
            core[5][2] = -rijkl(ei, ej, ij, indexd(5, 1), 0, 0, 2, 0, 1, r) * (ev * tore_i);
            core[6][2] = -rijkl(ei, ej, ij, indexd(5, 2), 0, 0, 2, 1, 1, r) * (ev * tore_i);
            core[7][2] = -rijkl(ei, ej, ij, indexd(5, 5), 0, 0, 2, 2, 1, r) * (ev * tore_i);
            core[8][2] = -rijkl(ei, ej, ij, indexd(6, 3), 0, 0, 2, 1, 1, r) * (ev * tore_i);
            core[9][2] = -rijkl(ei, ej, ij, indexd(6, 6), 0, 0, 2, 2, 1, r) * (ev * tore_i);
            core[10][2] = -rijkl(ei, ej, ij, indexd(8, 8), 0, 0, 2, 2, 1, r) * (ev * tore_i);
        }
        if dorbs_i {
            let tore_j = ei_tore(ej);
            let ij = 1;
            core[5][1] = -rijkl(ei, ej, indexd(5, 1), ij, 2, 0, 0, 0, 2, r) * (ev * tore_j);
            core[6][1] = -rijkl(ei, ej, indexd(5, 2), ij, 2, 1, 0, 0, 2, r) * (ev * tore_j);
            core[7][1] = -rijkl(ei, ej, indexd(5, 5), ij, 2, 2, 0, 0, 2, r) * (ev * tore_j);
            core[8][1] = -rijkl(ei, ej, indexd(6, 3), ij, 2, 1, 0, 0, 2, r) * (ev * tore_j);
            core[9][1] = -rijkl(ei, ej, indexd(6, 6), ij, 2, 2, 0, 0, 2, r) * (ev * tore_j);
            core[10][1] = -rijkl(ei, ej, indexd(8, 8), ij, 2, 2, 0, 0, 2, r) * (ev * tore_j);
        }
    }
    rep
}

/// MOPAC `tx`: first rotation step, right (kl) index. Returns v[1..45][1..limkl].
fn tx<S: Scalar>(ii: usize, kk: usize, rep: &[S], rot: &Rot<S>) -> Vec<Vec<S>> {
    let limkl = indx(kk, kk);
    let mut v = vec![vec![S::cst(0.0); limkl + 1]; 46];
    for i1 in 1..=ii {
        for j1 in 1..=i1 {
            let ij = indexd(i1, j1);
            for k1 in 1..=kk {
                for l1 in 1..=k1 {
                    let kl = indexd(k1, l1);
                    let nd = IND2[(ij - 1) * 45 + (kl - 1)] as usize;
                    if nd == 0 {
                        continue;
                    }
                    let wrepp = rep[nd];
                    let ll = indx(k1, l1);
                    let mm = MET[ll];
                    match mm {
                        1 => v[ij][1] = wrepp,
                        2 => {
                            let k = k1 - 1;
                            v[ij][2] = v[ij][2] + rot.sp[k][1] * wrepp;
                            v[ij][4] = v[ij][4] + rot.sp[k][2] * wrepp;
                            v[ij][7] = v[ij][7] + rot.sp[k][3] * wrepp;
                        }
                        3 => {
                            let k = k1 - 1;
                            let l = l1 - 1;
                            v[ij][3] = v[ij][3] + rot.pp[1][k][l] * wrepp;
                            v[ij][6] = v[ij][6] + rot.pp[2][k][l] * wrepp;
                            v[ij][10] = v[ij][10] + rot.pp[3][k][l] * wrepp;
                            v[ij][5] = v[ij][5] + rot.pp[4][k][l] * wrepp;
                            v[ij][8] = v[ij][8] + rot.pp[5][k][l] * wrepp;
                            v[ij][9] = v[ij][9] + rot.pp[6][k][l] * wrepp;
                        }
                        4 => {
                            let k = k1 - 4;
                            v[ij][11] = v[ij][11] + rot.sd[k][1] * wrepp;
                            v[ij][16] = v[ij][16] + rot.sd[k][2] * wrepp;
                            v[ij][22] = v[ij][22] + rot.sd[k][3] * wrepp;
                            v[ij][29] = v[ij][29] + rot.sd[k][4] * wrepp;
                            v[ij][37] = v[ij][37] + rot.sd[k][5] * wrepp;
                        }
                        5 => {
                            let k = k1 - 4;
                            let l = l1 - 1;
                            let cols = [12, 13, 14, 17, 18, 19, 23, 24, 25, 30, 31, 32, 38, 39, 40];
                            for (t, &c) in cols.iter().enumerate() {
                                v[ij][c] = v[ij][c] + rot.dp[t + 1][k][l] * wrepp;
                            }
                        }
                        6 => {
                            let k = k1 - 4;
                            let l = l1 - 4;
                            let diag = [(15, 1), (21, 2), (28, 3), (36, 4), (45, 5)];
                            for &(c, t) in diag.iter() {
                                v[ij][c] = v[ij][c] + rot.dd[t][k][l] * wrepp;
                            }
                            let offd = [
                                (20, 6),
                                (26, 7),
                                (27, 8),
                                (33, 9),
                                (34, 10),
                                (35, 11),
                                (41, 12),
                                (42, 13),
                                (43, 14),
                                (44, 15),
                            ];
                            for &(c, t) in offd.iter() {
                                v[ij][c] = v[ij][c] + rot.dd[t][k][l] * wrepp;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    v
}

/// MOPAC `rotatd` second rotation step, left (ij) index → molecular `ww`
/// (flat, 1-based, `(indx(i,j)-1)*limkl + kl`).
fn rotate_left<S: Scalar>(
    ii: usize,
    kk: usize,
    v: &[Vec<S>],
    rot: &Rot<S>,
) -> (Vec<S>, usize, usize) {
    let limij = indx(ii, ii);
    let limkl = indx(kk, kk);
    let mut ww = vec![S::cst(0.0); limij * limkl + 1];
    let indw = |i: usize, j: usize, kl: usize| (indx(i, j) - 1) * limkl + kl;
    for i1 in 1..=ii {
        for j1 in 1..=i1 {
            let ij = indexd(i1, j1);
            let jj = indx(i1, j1);
            let mm = MET[jj];
            for k in 1..=kk {
                for l in 1..=k {
                    let kl = indx(k, l);
                    let wrepp = v[ij][kl];
                    match mm {
                        1 => {
                            let iw = indw(1, 1, kl);
                            ww[iw] = wrepp;
                        }
                        2 => {
                            for i in 1..=3 {
                                let iw = indw(i + 1, 1, kl);
                                ww[iw] = ww[iw] + rot.sp[i1 - 1][i] * wrepp;
                            }
                        }
                        3 => {
                            for i in 1..=3 {
                                let cc = rot.pp[i][i1 - 1][j1 - 1];
                                let iw = indw(i + 1, i + 1, kl);
                                ww[iw] = ww[iw] + cc * wrepp;
                                for j in 1..i {
                                    let cc = rot.pp[1 + i + j][i1 - 1][j1 - 1];
                                    let iw = indw(i + 1, j + 1, kl);
                                    ww[iw] = ww[iw] + cc * wrepp;
                                }
                            }
                        }
                        4 => {
                            for i in 1..=5 {
                                let iw = indw(i + 4, 1, kl);
                                ww[iw] = ww[iw] + rot.sd[i1 - 4][i] * wrepp;
                            }
                        }
                        5 => {
                            for i in 1..=5 {
                                for j in 1..=3 {
                                    let iw = indw(i + 4, j + 1, kl);
                                    let ij1 = 3 * (i - 1) + j;
                                    ww[iw] = ww[iw] + rot.dp[ij1][i1 - 4][j1 - 1] * wrepp;
                                }
                            }
                        }
                        6 => {
                            for i in 1..=5 {
                                let cc = rot.dd[i][i1 - 4][j1 - 4];
                                let iw = indw(i + 4, i + 4, kl);
                                ww[iw] = ww[iw] + cc * wrepp;
                                for j in 1..i {
                                    let ij1 = INDDD[(i - 1) * 5 + (j - 1)] as usize;
                                    let cc = rot.dd[ij1][i1 - 4][j1 - 4];
                                    let iw = indw(i + 4, j + 4, kl);
                                    ww[iw] = ww[iw] + cc * wrepp;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    (ww, limij, limkl)
}

/// PM7 d-imbalance correction (MOPAC `rotatd` `method_PM7` block): set the mean
/// of the two-center d Coulomb integrals equal to `<ss|ss>` (= ww[1]).
fn pm7_balance<S: Scalar>(ww: &mut [S], core: &mut [[S; 3]; 11], ei: &Pm7Element, ej: &Pm7Element) {
    let natorb_i = ei.n_orb;
    let natorb_j = ej.n_orb;
    if ei.iod > 0 {
        let k = match natorb_j {
            9 => 45,
            4 => 10,
            _ => 1,
        };
        if natorb_j > 1 {
            let mut sum = S::cst(0.0);
            for i in 5..=9 {
                let j = k * ((i * (i + 1)) / 2 - 1);
                sum = sum + ww[j + 3] + ww[j + 6] + ww[j + 10];
            }
            sum = ww[1] - sum * (1.0 / 15.0);
            for i in 5..=9 {
                let j = k * ((i * (i + 1)) / 2 - 1);
                for l in 2..=4 {
                    let idx = (l * (l + 1)) / 2 + j;
                    ww[idx] = ww[idx] + sum;
                }
            }
            let s = core[1][1] - (core[3][1] + core[4][1] * 2.0) * (1.0 / 3.0);
            core[3][1] = core[3][1] + s;
            core[4][1] = core[4][1] + s;
        }
        let mut sum = S::cst(0.0);
        for i in 5..=9 {
            sum = sum + ww[k * ((i * (i + 1)) / 2 - 1) + 1];
        }
        sum = ww[1] - sum * (1.0 / 5.0);
        for i in 5..=9 {
            let idx = k * ((i * (i + 1)) / 2 - 1) + 1;
            ww[idx] = ww[idx] + sum;
        }
        let s = core[1][1] - (core[7][1] + core[9][1] * 2.0 + core[10][1] * 2.0) * (1.0 / 5.0);
        core[7][1] = core[7][1] + s;
        core[9][1] = core[9][1] + s;
        core[10][1] = core[10][1] + s;
    }
    if ej.iod > 0 {
        if ei.iod > 0 {
            let mut sum = S::cst(0.0);
            for i in 5..=9 {
                let j = 45 * ((i * (i + 1)) / 2 - 1);
                sum = sum + ww[j + 15] + ww[j + 21] + ww[j + 28] + ww[j + 36] + ww[j + 45];
            }
            sum = ww[1] - sum * (1.0 / 25.0);
            for i in 5..=9 {
                let j = 45 * ((i * (i + 1)) / 2 - 1);
                for kk in 5..=9 {
                    let idx = (kk * (kk + 1)) / 2 + j;
                    ww[idx] = ww[idx] + sum;
                }
            }
        }
        if natorb_i > 1 {
            let mut sum = S::cst(0.0);
            for i in 2..=4 {
                let j = 45 * ((i * (i + 1)) / 2 - 1);
                sum = sum + ww[j + 15] + ww[j + 21] + ww[j + 28] + ww[j + 36] + ww[j + 45];
            }
            sum = ww[1] - sum * (1.0 / 15.0);
            for i in 2..=4 {
                let j = 45 * ((i * (i + 1)) / 2 - 1);
                for kk in 5..=9 {
                    let idx = (kk * (kk + 1)) / 2 + j;
                    ww[idx] = ww[idx] + sum;
                }
            }
            let s = core[1][2] - (core[3][2] + core[4][2] * 2.0) * (1.0 / 3.0);
            core[3][2] = core[3][2] + s;
            core[4][2] = core[4][2] + s;
        }
        let sum = ww[1] - (ww[15] + ww[21] + ww[28] + ww[36] + ww[45]) * (1.0 / 5.0);
        for kk in 5..=9 {
            let idx = (kk * (kk + 1)) / 2;
            ww[idx] = ww[idx] + sum;
        }
        let s = core[1][2] - (core[7][2] + core[9][2] * 2.0 + core[10][2] * 2.0) * (1.0 / 5.0);
        core[7][2] = core[7][2] + s;
        core[9][2] = core[9][2] + s;
        core[10][2] = core[10][2] + s;
    }
}

/// MOPAC `elenuc`: expand `core[10][n]` into the diagonal electron–core block for
/// one atom (n=1 → atom i / e1b, n=2 → atom j / e2a), using the rotation tensors.
fn elenuc_block<S: Scalar>(
    core: &[[S; 3]; 11],
    rot: &Rot<S>,
    n: usize,
    norb: usize,
) -> [[S; 9]; 9] {
    let mut blk = [[S::cst(0.0); 9]; 9];
    for i in 1..=norb {
        let ind1 = i - 1;
        for j in 1..=i {
            let ind2 = j - 1;
            let val;
            if ind1 == 0 {
                val = core[1][n]; // SS
            } else if ind1 < 4 {
                if ind2 == 0 {
                    val = rot.sp[1][ind1] * core[2][n]; // SP
                } else {
                    let ipp = INDPP[(ind1 - 1) * 3 + (ind2 - 1)] as usize; // PP
                    val = core[3][n] * rot.pp[ipp][1][1]
                        + core[4][n] * (rot.pp[ipp][2][2] + rot.pp[ipp][3][3]);
                }
            } else if ind2 == 0 {
                val = rot.sd[1][ind1 - 3] * core[5][n]; // SD
            } else if ind2 < 4 {
                let idp = INDDP[(ind1 - 4) * 3 + (ind2 - 1)] as usize; // PD
                val = core[6][n] * rot.dp[idp][1][1]
                    + core[8][n] * (rot.dp[idp][2][2] + rot.dp[idp][3][3]);
            } else {
                let idd = INDDD[(ind1 - 4) * 5 + (ind2 - 4)] as usize; // DD
                val = core[7][n] * rot.dd[idd][1][1]
                    + core[9][n] * (rot.dd[idd][2][2] + rot.dd[idd][3][3])
                    + core[10][n] * (rot.dd[idd][4][4] + rot.dd[idd][5][5]);
            }
            blk[i - 1][j - 1] = val;
            blk[j - 1][i - 1] = val;
        }
    }
    blk
}

/// Two-center two-electron + electron–core integrals for a d-involving ordered
/// pair (ei = first, ej = second), generic over the scalar type. `dvec = R_j − R_i`
/// (Bohr). Produces the same [`PairTwoElecG`] structure as the sp kernel, with `w`
/// sized to the atoms' orbital counts and 9×9 `e1b`/`e2a`.
pub fn pair_two_electron_d_g<S: Scalar>(
    ei: &Pm7Element,
    ej: &Pm7Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    let has_d = ei.n_orb == 9 || ej.n_orb == 9;
    let (r, rot) = rotmat(dvec, has_d);
    let ri = reppd(ei, ej, r);
    let mut core = spcore(ei, ej, r);
    let rep = reppd2(ei, ej, r, &ri, &mut core);
    // Feather the electron–core attractions toward the point-charge limit as the atoms
    // separate (MOPAC `rotatd.F90:1160-1181`).  point = -(ev/r)·tore(other-atom-core);
    // the charge–charge distributions {1,3,4,7,9,10} decay to it, the multipole ones to 0.
    // Applied over all ten unconditionally (as MOPAC does) for a bit-for-bit match.
    {
        let (cfrac, _) = crate::integrals::feather_to_point(r);
        let one_minus = S::cst(1.0) - cfrac;
        let point1 = r.recip() * (-PM7_EV * ei_tore(ej));
        let point2 = r.recip() * (-PM7_EV * ei_tore(ei));
        for k in 1..=10 {
            core[k][1] = core[k][1] * cfrac;
            core[k][2] = core[k][2] * cfrac;
        }
        for &k in &CORE_MONOPOLE {
            core[k][1] = core[k][1] + point1 * one_minus;
            core[k][2] = core[k][2] + point2 * one_minus;
        }
    }
    let ii = ei.n_orb;
    let kk = ej.n_orb;

    let v = tx(ii, kk, &rep, &rot);
    let (mut ww, limij, limkl) = rotate_left(ii, kk, &v, &rot);
    // PM7 always applies the d-balance correction (only active for iod > 0 atoms).
    pm7_balance(&mut ww, &mut core, ei, ej);

    // Molecular-frame w: w[pi][pj] with pi over atom-i pairs, pj over atom-j pairs.
    let mut w = vec![vec![S::cst(0.0); limkl]; limij];
    for pi in 0..limij {
        for pj in 0..limkl {
            w[pi][pj] = ww[pi * limkl + pj + 1];
        }
    }
    let e1b = elenuc_block(&core, &rot, 1, ii);
    let e2a = elenuc_block(&core, &rot, 2, kk);

    PairTwoElecG {
        norb_i: ii,
        norb_j: kk,
        w,
        e1b,
        e2a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::pair_two_electron_g;
    use crate::math::Vec3;
    use crate::params::Pm7Parameters;

    fn dvec(x: f64, y: f64, z: f64) -> [f64; 3] {
        [x, y, z]
    }

    // Isotropic (spherically symmetric) two-electron interaction energy
    // Σ_a Σ_c (a a | c c) over the four sp orbitals. This is invariant to the
    // internal p-orbital labeling/frame, so it must agree between the sp and
    // MNDO/d rotation schemes (and under rotation).
    fn iso_energy(te: &PairTwoElecG<f64>) -> f64 {
        let (ni, nj) = (te.norb_i, te.norb_j);
        let pk = |a: usize, b: usize| {
            let (h, l) = if a >= b { (a, b) } else { (b, a) };
            h * (h + 1) / 2 + l
        };
        let mut e = 0.0;
        for a in 0..ni {
            for c in 0..nj {
                e += te.w[pk(a, a)][pk(c, c)];
            }
        }
        e
    }

    #[test]
    fn dpath_reproduces_sp_path_physics() {
        // For an sp-only pair the general MNDO/d kernel must reproduce the same
        // physics as the validated sp kernel: the electron–core block matches
        // element-wise (same rotation), and rotation-invariant two-electron
        // contractions match (element-wise w may differ by internal p-labeling).
        let p = Pm7Parameters::standard().unwrap();
        for (zi, zj) in [(6u8, 6u8), (8, 1), (6, 8), (7, 1)] {
            let (ei, ej) = (p.element(zi).unwrap(), p.element(zj).unwrap());
            let d = dvec(1.3, -0.9, 0.7);
            let sp = pair_two_electron_g::<f64>(ei, ej, d);
            let dd = pair_two_electron_d_g::<f64>(ei, ej, d);
            let mut maxe = 0.0_f64;
            for a in 0..sp.norb_i {
                for b in 0..sp.norb_i {
                    maxe = maxe.max((sp.e1b[a][b] - dd.e1b[a][b]).abs());
                }
            }
            let iso_sp = iso_energy(&sp);
            let iso_dd = iso_energy(&dd);
            assert!(
                maxe < 1.0e-9,
                "Z{zi}-Z{zj}: electron-core mismatch {maxe:.2e}"
            );
            assert!(
                (iso_sp - iso_dd).abs() < 1.0e-9,
                "Z{zi}-Z{zj}: isotropic 2e energy sp={iso_sp:.6} d={iso_dd:.6}"
            );
        }
    }

    #[test]
    fn dpath_isotropic_energy_is_rotation_invariant() {
        // The MNDO/d two-electron block itself must be rotation invariant.
        let p = Pm7Parameters::standard().unwrap();
        let s = p.element(16).unwrap();
        let r = 3.1_f64;
        let a = pair_two_electron_d_g::<f64>(s, s, dvec(r, 0.0, 0.0));
        let v = Vec3::new(-0.2, 0.7, 0.5).normalized();
        let b = pair_two_electron_d_g::<f64>(s, s, dvec(v.x * r, v.y * r, v.z * r));
        assert!(
            (iso_energy(&a) - iso_energy(&b)).abs() < 1.0e-9,
            "d-atom isotropic 2e energy not rotation invariant: {} vs {}",
            iso_energy(&a),
            iso_energy(&b)
        );
    }

    #[test]
    fn dpath_ssss_is_rotation_invariant_for_d_atom() {
        // (ss|ss) for a d-bearing atom pair must be orientation-independent and
        // equal the closed-form monopole value.
        let p = Pm7Parameters::standard().unwrap();
        let s = p.element(16).unwrap(); // sulfur, carries d
        let r = 3.2_f64;
        let a = pair_two_electron_d_g::<f64>(s, s, dvec(r, 0.0, 0.0));
        let v = Vec3::new(0.3, -0.5, 0.8).normalized();
        let b = pair_two_electron_d_g::<f64>(s, s, dvec(v.x * r, v.y * r, v.z * r));
        // (ss|ss) is feathered toward the point charge (PM7 `l_feather`).
        let nddo = PM7_EV / (r * r + (2.0 * s.rho0).powi(2)).sqrt();
        let (cfrac, point) = crate::integrals::feather_to_point(r);
        let expect = nddo * cfrac + point * (1.0 - cfrac);
        assert!(
            (a.w[0][0] - expect).abs() < 1.0e-9,
            "ssss={} expect={}",
            a.w[0][0],
            expect
        );
        assert!(
            (a.w[0][0] - b.w[0][0]).abs() < 1.0e-9,
            "not rotation invariant"
        );
    }

    #[test]
    fn dpath_dual_gradient_matches_fd() {
        use crate::dual::Dual;
        let p = Pm7Parameters::standard().unwrap();
        let (s, h) = (p.element(16).unwrap(), p.element(1).unwrap());
        let d = Vec3::new(1.5, -0.6, 0.9);
        let dual = pair_two_electron_d_g::<Dual>(
            s,
            h,
            [Dual::var(d.x, 0), Dual::var(d.y, 1), Dual::var(d.z, 2)],
        );
        let step = 1.0e-6;
        let mut maxerr = 0.0_f64;
        for axis in 0..3 {
            let mut dp = [d.x, d.y, d.z];
            let mut dm = [d.x, d.y, d.z];
            dp[axis] += step;
            dm[axis] -= step;
            let wp = pair_two_electron_d_g::<f64>(s, h, dp);
            let wm = pair_two_electron_d_g::<f64>(s, h, dm);
            for a in 0..wp.w.len() {
                for b in 0..wp.w[a].len() {
                    let fd = (wp.w[a][b] - wm.w[a][b]) / (2.0 * step);
                    maxerr = maxerr.max((dual.w[a][b].d[axis] - fd).abs());
                }
            }
        }
        assert!(
            maxerr < 1.0e-6,
            "d-path dual gradient mismatch {maxerr:.2e}"
        );
    }
}
