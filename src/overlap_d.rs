// SPDX-License-Identifier: GPL-3.0-or-later

//! Diatomic overlap integrals for the PM7 `spd` basis, generic over
//! [`crate::dual::Scalar`] and consistent with the MNDO/d two-center rotation
//! (`crate::mndod_twocenter`) so that `H_core` mixes no frames.
//!
//! Faithful port of MOPAC v23.2.5 `src/integrals/{diat,coe,bfn}.F90` using the
//! general Slater-overlap radial integral `ss` (analytic A/B integrals) for every
//! shell pair. Orbital order matches the two-center kernel:
//! `1 s, 2 px, 3 py, 4 pz, 5 dx2-y2, 6 dxz, 7 dz2, 8 dyz, 9 dxy`.
//!
//! Provenance: MOPAC, Apache-2.0 (c) 2021 Virginia Tech. See `THIRD_PARTY_NOTICES.md`.

use crate::dual::Scalar;
use crate::mndod::{iii, iiid};
use crate::params::Pm7Element;

/// Factorials `FACT[n] = n!` up to 24!.
fn fact(n: usize) -> f64 {
    const F: [f64; 25] = {
        let mut f = [1.0; 25];
        let mut i = 1;
        while i < 25 {
            f[i] = f[i - 1] * i as f64;
            i += 1;
        }
        f
    };
    F[n]
}

/// Principal quantum number of shell `shell` (1 = s, 2 = p, 3 = d) of element `z`.
fn npq(z: u8, shell: usize) -> i32 {
    match shell {
        3 => iiid(z),
        _ => iii(z),
    }
}

/// MOPAC `bfn`: the "B" integrals `bf[0..=12]` for the overlap, generic over the
/// scalar type. `x = (ζa − ζb)·r/2`.
fn bfn<S: Scalar>(x: S) -> [S; 13] {
    let mut bf = [S::cst(0.0); 13];
    let absx = x.val().abs();
    if absx > 3.0 {
        let expx = x.exp();
        let expmx = expx.recip();
        let inv = x.recip();
        bf[0] = (expx - expmx) * inv;
        for i in 1..=12 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            bf[i] = (bf[i - 1] * (i as f64) + expx * sign - expmx) * inv;
        }
    } else if absx <= 1.0e-6 {
        // MOPAC bfn small-x limit: bf(i+1) = 2·mod(i+1,2)/(i+1).
        for i in 0..=12 {
            bf[i] = S::cst((2 * ((i + 1) % 2)) as f64 / (i as f64 + 1.0));
        }
    } else {
        let last = if absx > 2.0 {
            15
        } else if absx > 1.0 {
            12
        } else if absx > 0.5 {
            7
        } else {
            6
        };
        for i in 0..=12 {
            let mut y = S::cst(0.0);
            for m in 0..=last {
                let xf = if m != 0 { fact(m) } else { 1.0 };
                let coeff = (2 * ((m + i + 1) % 2)) as f64 / (xf * (m + i + 1) as f64);
                // (-x)^m
                let mut xm = S::cst(1.0);
                for _ in 0..m {
                    xm = xm * (-x);
                }
                y = y + xm * coeff;
            }
            bf[i] = y;
        }
    }
    bf
}

/// MOPAC `ss`: general Slater diatomic overlap between shell `la1`(=l+1) of atom a
/// and shell `lb1` of atom b, angular component `m1`(1 σ, 2 π, 3 δ). `na,nb` are
/// principal quantum numbers; `ua,ub` Slater exponents; `r` the distance (Bohr).
#[allow(clippy::too_many_arguments)]
fn ss<S: Scalar>(na: i32, nb: i32, la1: i32, lb1: i32, m1: i32, ua: f64, ub: f64, r: S) -> S {
    // aff(la, m, i) coefficients (Slater-overlap angular normalization).
    let aff = |la: i32, m: i32, i: i32| -> f64 {
        match (la, m, i) {
            (0, 0, 0) => 1.0,
            (1, 0, 0) => 1.0,
            (1, 1, 0) => 0.5_f64.sqrt(),
            (2, 0, 0) => 1.5,
            (2, 1, 0) => 1.5_f64.sqrt(),
            (2, 2, 0) => 0.375_f64.sqrt(),
            (2, 0, 2) => -0.5,
            _ => 0.0,
        }
    };
    // Binomials bi(i,j) = C(i,j).
    let bi = |i: i32, j: i32| -> f64 {
        if j < 0 || j > i {
            0.0
        } else {
            fact(i as usize) / (fact(j as usize) * fact((i - j) as usize))
        }
    };
    let m = m1 - 1;
    let lb = lb1 - 1;
    let la = la1 - 1;
    let p = r * ((ua + ub) * 0.5);
    let bb = r * ((ua - ub) * 0.5);
    // A integrals af(0..19).
    let mut af = [S::cst(0.0); 20];
    let quo = p.recip();
    af[0] = quo * (-p).exp();
    for n in 1..20 {
        af[n] = af[n - 1] * (n as f64) * quo + af[0];
    }
    let bf = bfn(bb);
    let mut sum = S::cst(0.0);
    let lam1 = la - m;
    let lbm1 = lb - m;
    let mut i = 0;
    while i <= lam1 {
        let ia = na + i - la;
        let ic = la - i - m;
        let mut j = 0;
        while j <= lbm1 {
            let ib = nb + j - lb;
            let id = lb - j - m;
            let mut sum1 = S::cst(0.0);
            let iab = ia + ib;
            for k1 in 0..=ia {
                for k2 in 0..=ib {
                    for k3 in 0..=ic {
                        for k4 in 0..=id {
                            for k5 in 0..=m {
                                let iaf = iab - k1 - k2 + k3 + k4 + 2 * k5;
                                for k6 in 0..=m {
                                    let ibf = k1 + k2 + k3 + k4 + 2 * k6;
                                    let sign = (1 - 2 * ((m + k2 + k4 + k5 + k6) % 2)) as f64;
                                    let coeff = bi(id, k4)
                                        * bi(ic, k3)
                                        * bi(ib, k2)
                                        * bi(ia, k1)
                                        * bi(m, k5)
                                        * bi(m, k6)
                                        * sign;
                                    sum1 = sum1 + af[iaf as usize] * bf[ibf as usize] * coeff;
                                }
                            }
                        }
                    }
                }
            }
            sum = sum + sum1 * (aff(la, m, i) * aff(lb, m, j));
            j += 2;
        }
        i += 2;
    }
    // r^(na+nb+1) * ua^na * ub^nb / 2 * sqrt(...).
    let mut rpow = S::cst(1.0);
    for _ in 0..(na + nb + 1) {
        rpow = rpow * r;
    }
    let norm = (ua * ub / (fact((na + na) as usize) * fact((nb + nb) as usize))
        * ((la + la + 1) * (lb + lb + 1)) as f64)
        .sqrt();
    sum * rpow * (ua.powi(na) * ub.powi(nb) / 2.0 * norm)
}

/// MOPAC `coe`: angular rotation coefficients `c[1..=75]` (flat `c(3,5,5)`,
/// column-major: `c(i,k,orb) = c[i + 3(k-1) + 15(orb-1)]`) and the distance.
fn coe<S: Scalar>(v: [S; 3], nij: usize) -> (S, [S; 76]) {
    const RT34: f64 = 0.866_025_403_784_44;
    const RT13: f64 = 0.577_350_269_189_63;
    let (x2, y2, z2) = (v[0], v[1], v[2]);
    let xy2 = x2 * x2 + y2 * y2;
    let r = (xy2 + z2 * z2).sqrt();
    let xy = xy2.sqrt();
    let (ca, cb, sa, sb);
    if xy.val() >= 1.0e-10 {
        ca = x2 * xy.recip();
        cb = z2 * r.recip();
        sa = y2 * xy.recip();
        sb = xy * r.recip();
    } else if z2.val() < 0.0 {
        ca = S::cst(-1.0);
        cb = S::cst(-1.0);
        sa = S::cst(0.0);
        sb = S::cst(0.0);
    } else if z2.val() == 0.0 {
        ca = S::cst(0.0);
        cb = S::cst(0.0);
        sa = S::cst(0.0);
        sb = S::cst(0.0);
    } else {
        ca = S::cst(1.0);
        cb = S::cst(1.0);
        sa = S::cst(0.0);
        sb = S::cst(0.0);
    }
    let mut c = [S::cst(0.0); 76];
    c[37] = S::cst(1.0);
    if nij >= 2 {
        c[56] = ca * cb;
        c[41] = ca * sb;
        c[26] = -sa;
        c[53] = -sb;
        c[38] = cb;
        c[23] = S::cst(0.0);
        c[50] = sa * cb;
        c[35] = sa * sb;
        c[20] = ca;
        if nij >= 5 {
            let c2a = ca * ca * 2.0 - 1.0;
            let c2b = cb * cb * 2.0 - 1.0;
            let s2a = sa * ca * 2.0;
            let s2b = sb * cb * 2.0;
            c[75] = c2a * cb * cb + c2a * sb * sb * 0.5;
            c[60] = c2a * s2b * 0.5;
            c[45] = c2a * sb * sb * RT34;
            c[30] = -(s2a * sb);
            c[15] = -(s2a * cb);
            c[72] = ca * s2b * (-0.5);
            c[57] = ca * c2b;
            c[42] = ca * s2b * RT34;
            c[27] = -(sa * cb);
            c[12] = sa * sb;
            c[69] = sb * sb * (RT13 * 1.5);
            c[54] = -(s2b * RT34);
            c[39] = cb * cb - sb * sb * 0.5;
            c[66] = sa * s2b * (-0.5);
            c[51] = sa * c2b;
            c[36] = sa * s2b * RT34;
            c[21] = ca * cb;
            c[6] = -(ca * sb);
            c[63] = s2a * cb * cb + s2a * sb * sb * 0.5;
            c[48] = s2a * s2b * 0.5;
            c[33] = s2a * sb * sb * RT34;
            c[18] = c2a * sb;
            c[3] = c2a * cb;
        }
    }
    (r, c)
}

// ival(i,k): orbital index for shell i (1 s,2 p,3 d) and M_L component k (1..5).
const IVAL: [[usize; 6]; 4] = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0], // s: only k=3 used (→ orbital 1)
    [0, 0, 3, 4, 2, 0], // p: k=2→py(3), k=3→pz(4), k=4→px(2)
    [0, 9, 8, 7, 6, 5], // d: k=1..5 → 9,8,7,6,5
];

/// Diatomic overlap `S(9×9)` in the molecular frame for ordered pair (ei, ej),
/// `dvec = R_j − R_i` (Bohr). Orbital order matches the two-center kernel.
pub fn diat_overlap<S: Scalar>(ei: &Pm7Element, ej: &Pm7Element, dvec: [S; 3]) -> [[S; 9]; 9] {
    let mut di = [[S::cst(0.0); 9]; 9];
    let norbi = ei.n_orb;
    let norbj = ej.n_orb;
    if norbi == 0 || norbj == 0 {
        return di;
    }
    let nij = norbi.max(norbj);
    let (r, c) = coe(dvec, nij);
    let cc = |i: usize, k: usize, orb: usize| c[i + 3 * (k - 1) + 15 * (orb - 1)];

    // Number of angular shells on each atom: 1 (s only), 2 (sp), 3 (spd).
    let ia = shells(norbi);
    let ib = shells(norbj);
    let exps_i = [ei.zeta_s, ei.zeta_p, ei.zeta_d.max(0.3)];
    let exps_j = [ej.zeta_s, ej.zeta_p, ej.zeta_d.max(0.3)];

    // Local Slater overlaps s(shell_a, shell_b, component) — σ,π,δ.
    let mut s = [[[S::cst(0.0); 4]; 4]; 4]; // s[i][j][k], 1-based
    let newk = (ia - 1).min(ib - 1);
    for i in 1..=ia {
        let pq1 = npq(ei.z, i);
        for j in 1..=ib {
            let pq2 = npq(ej.z, j);
            for k in 1..=(newk + 1) {
                if k > i || k > j {
                    continue;
                }
                let pi = pq1.max(i as i32);
                let pj = pq2.max(j as i32);
                s[i][j][k] = ss(
                    pi,
                    pj,
                    i as i32,
                    j as i32,
                    k as i32,
                    exps_i[i - 1],
                    exps_j[j - 1],
                    r,
                );
            }
        }
    }

    for i in 1..=ia {
        let kmin = 4 - i;
        let kmax = 2 + i;
        for j in 1..=ib {
            let (aa, bb): (f64, f64) = if j == 2 {
                (-1.0, 1.0)
            } else if j == 3 {
                (1.0, -1.0)
            } else {
                (1.0, 1.0)
            };
            let lmin = 4 - j;
            let lmax = 2 + j;
            for k in kmin..=kmax {
                for l in lmin..=lmax {
                    let ii = IVAL[i][k];
                    let jj = IVAL[j][l];
                    if ii == 0 || jj == 0 {
                        continue;
                    }
                    let s1 = s[i][j][1];
                    let s2 = s[i][j][2];
                    let s3 = s[i][j][3];
                    let val = s1 * (cc(i, k, 3) * cc(j, l, 3)) * aa
                        + s2 * (cc(i, k, 4) * cc(j, l, 4) + cc(i, k, 2) * cc(j, l, 2)) * bb
                        + s3 * (cc(i, k, 5) * cc(j, l, 5) + cc(i, k, 1) * cc(j, l, 1));
                    di[ii - 1][jj - 1] = val;
                }
            }
        }
    }
    di
}

#[inline]
fn shells(norb: usize) -> usize {
    match norb {
        1 => 1,
        4 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::overlap::diatom_overlap;
    use crate::params::Pm7Parameters;

    #[test]
    fn ss_overlap_matches_sp_analytic() {
        // The general Slater ss() s–s overlap must match the validated sp analytic
        // overlap (both in the s block) to high precision.
        let p = Pm7Parameters::standard().unwrap();
        let (c, o) = (p.element(6).unwrap(), p.element(8).unwrap());
        // Place O along +z from C so the sp analytic and MNDO/d frames agree on s.
        let pos_c = Vec3::new(0.0, 0.0, 0.0);
        let pos_o = Vec3::new(0.0, 0.0, 2.4);
        let sp = diatom_overlap(c, pos_c, o, pos_o).unwrap();
        let dd = diat_overlap::<f64>(c, o, [0.0, 0.0, 2.4]);
        // s-s overlap element.
        assert!(
            (sp[0][0] - dd[0][0]).abs() < 1.0e-6,
            "ss sp={} d={}",
            sp[0][0],
            dd[0][0]
        );
    }

    #[test]
    fn diat_overlap_matches_mopac_h2s() {
        // MOPAC v23.2.5 PM7 H2S overlap of S(9 AOs) with H1(s), geometry
        // S at origin, H1 at (0, 0.9705, 0.9430) Angstrom. Order s,px,py,pz,
        // dx2-y2,dxz,dz2,dyz,dxy.
        let p = Pm7Parameters::standard().unwrap();
        let (s, h) = (p.element(16).unwrap(), p.element(1).unwrap());
        let a0 = crate::constants::PM7_A0;
        let dvec = [0.0, 0.9705 / a0, 0.9430 / a0];
        let di = diat_overlap::<f64>(s, h, dvec);
        let mopac = [
            0.3884, 0.0000, 0.3473, 0.3375, -0.0459, 0.0000, 0.0236, 0.0893, 0.0000,
        ];
        let mine: Vec<f64> = (0..9).map(|i| di[i][0]).collect();
        eprintln!("MOPAC S-H1 overlap = {mopac:?}");
        eprintln!(
            "pm7-rs S-H1 overlap = {:?}",
            mine.iter()
                .map(|v| (v * 1e4).round() / 1e4)
                .collect::<Vec<_>>()
        );
        for i in 0..9 {
            assert!(
                (di[i][0] - mopac[i]).abs() < 5.0e-4,
                "AO {i}: pm7-rs {} vs MOPAC {}",
                di[i][0],
                mopac[i]
            );
        }
    }

    #[test]
    fn diat_vs_sp_overlap_oh() {
        // diat (MNDO/d) and the sp analytic overlap for O–H, element-wise.
        let p = Pm7Parameters::standard().unwrap();
        let (o, h) = (p.element(8).unwrap(), p.element(1).unwrap());
        let pos_o = Vec3::new(0.0, 0.0, 0.0);
        let pos_h = Vec3::new(0.9584, -0.24, 0.31);
        let sp = diatom_overlap(o, pos_o, h, pos_h).unwrap();
        let dd = diat_overlap::<f64>(o, h, [pos_h.x, pos_h.y, pos_h.z]);
        eprintln!("O-H overlap  (row=O orbital s,px,py,pz ; col 0 = H s)");
        for i in 0..4 {
            eprintln!(
                "  sp[{i}][0]={:+.5}  diat[{i}][0]={:+.5}",
                sp[i][0], dd[i][0]
            );
        }
        // H–H s-s overlap (the failing h_core[4][5] term).
        let pos_h2 = Vec3::new(-0.24, 0.9278, 0.0);
        let sp_hh = diatom_overlap(h, pos_h, h, pos_h2).unwrap();
        let dd_hh = diat_overlap::<f64>(
            h,
            h,
            [pos_h2.x - pos_h.x, pos_h2.y - pos_h.y, pos_h2.z - pos_h.z],
        );
        eprintln!("H-H s-s: sp={:+.5} diat={:+.5}", sp_hh[0][0], dd_hh[0][0]);
    }

    #[test]
    fn diat_overlap_dual_matches_fd() {
        use crate::dual::Dual;
        let p = Pm7Parameters::standard().unwrap();
        let (s, h) = (p.element(16).unwrap(), p.element(1).unwrap());
        let d = Vec3::new(1.6, -0.5, 0.9);
        let dual = diat_overlap::<Dual>(
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
            let wp = diat_overlap::<f64>(s, h, dp);
            let wm = diat_overlap::<f64>(s, h, dm);
            for a in 0..9 {
                for b in 0..9 {
                    let fd = (wp[a][b] - wm[a][b]) / (2.0 * step);
                    maxerr = maxerr.max((dual[a][b].d[axis] - fd).abs());
                }
            }
        }
        assert!(maxerr < 1.0e-6, "diat overlap dual mismatch {maxerr:.2e}");
    }
}
