// SPDX-License-Identifier: GPL-3.0-or-later

//! Robust derivative fallback for the two-center integrals of a bond that is (near-)exactly
//! aligned with the singular axis of its local-frame rotation.
//!
//! The sp two-electron frame [`crate::integrals::rotation_to_x_g`] is singular for a bond
//! along **+x** (`qw = vx + 1 → 0`), and the MNDO/d `rotmat`/`coe` rotations for a bond along
//! **±z** (`sin(polar) → 0`). At the antipode the rotation is genuinely direction-
//! discontinuous: its forward-mode (Dual/Dual2) derivative collapses to the constant special
//! case with zero slope, so the *perpendicular* component of the analytic gradient/Hessian is
//! lost. The integral **values** are correct everywhere (the f64 special case is a valid
//! rotation, which is why energies are unaffected), so for such a pair we take the
//! derivatives by central finite differences of the f64 integrals. This is exact to FD
//! precision and only ever triggers for a bond within a fraction of a degree of the singular
//! axis, so it has no measurable cost.

use crate::dual::Dual;
use crate::dual2::Dual2;
use crate::integrals::{pair_two_electron_g, PairTwoElecG};
use crate::math::Vec3;
use crate::params::Pm7Element;

/// Central-difference step for first derivatives (Bohr); near-optimal for f' rounding.
const H: f64 = 1.0e-5;
/// Larger step for the second-order Hessian stencil: a tight step makes the second
/// difference `f(+h) − 2f(0) + f(−h)` lose ~10 digits to cancellation, so `~2e-3` balances
/// rounding (∝ ε/h²) against truncation (∝ h²) for the curvature.
const H2: f64 = 2.0e-3;
/// Angular window (≈ `acos(1 - TOL)` ≈ 0.8°) around the singular axis.
const TOL: f64 = 1.0e-4;

/// True when the displacement `d` sits in the singular window of the local-frame rotation
/// used by this pair's integral path (sp path: bond ∥ +x; d path: bond ∥ ±z).
pub(crate) fn near_frame_singularity(d: Vec3, has_d: bool) -> bool {
    let n2 = d.x * d.x + d.y * d.y + d.z * d.z;
    if n2 <= 0.0 {
        return false;
    }
    let n = n2.sqrt();
    if has_d {
        (d.x * d.x + d.y * d.y).sqrt() / n < TOL
    } else {
        d.x / n > 1.0 - TOL
    }
}

/// f64 overlap block (9×9; sp fills the leading 4×4) for a displacement `d`.
fn s_f64(ea: &Pm7Element, eb: &Pm7Element, d: Vec3, has_d: bool) -> [[f64; 9]; 9] {
    if has_d {
        crate::overlap_d::diat_overlap::<f64>(ea, eb, [d.x, d.y, d.z])
    } else {
        let mut s9 = [[0.0f64; 9]; 9];
        if let Ok(s4) = crate::overlap::diatom_overlap(ea, Vec3::zero(), eb, d) {
            for i in 0..4 {
                for j in 0..4 {
                    s9[i][j] = s4[i][j];
                }
            }
        }
        s9
    }
}

const AXES: [Vec3; 3] = [
    Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    },
    Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    },
    Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    },
];

/// Gradient (Dual) two-center integrals + overlap for a singular pair, by central FD of the
/// f64 integrals. Derivatives are w.r.t. the displacement `d = R_j − R_i` (matching the
/// analytic path), so downstream `∂/∂R_j` accumulation is unchanged.
pub(crate) fn pair_dual_fd(
    ea: &Pm7Element,
    eb: &Pm7Element,
    d: Vec3,
    has_d: bool,
) -> (PairTwoElecG<Dual>, [[Dual; 9]; 9]) {
    let te0 = pair_two_electron_g::<f64>(ea, eb, [d.x, d.y, d.z]);
    let s0 = s_f64(ea, eb, d, has_d);
    let (mut tp, mut tm, mut sp, mut sm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for a in AXES {
        let (dp, dm) = (d + a * H, d - a * H);
        tp.push(pair_two_electron_g::<f64>(ea, eb, [dp.x, dp.y, dp.z]));
        tm.push(pair_two_electron_g::<f64>(ea, eb, [dm.x, dm.y, dm.z]));
        sp.push(s_f64(ea, eb, dp, has_d));
        sm.push(s_f64(ea, eb, dm, has_d));
    }
    let du = |v: f64, p: [f64; 3], m: [f64; 3]| Dual {
        v,
        d: [
            (p[0] - m[0]) / (2.0 * H),
            (p[1] - m[1]) / (2.0 * H),
            (p[2] - m[2]) / (2.0 * H),
        ],
    };
    let w: Vec<Vec<Dual>> = te0
        .w
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, &b)| {
                    du(
                        b,
                        [tp[0].w[i][j], tp[1].w[i][j], tp[2].w[i][j]],
                        [tm[0].w[i][j], tm[1].w[i][j], tm[2].w[i][j]],
                    )
                })
                .collect()
        })
        .collect();
    let mut e1b = [[Dual::constant(0.0); 9]; 9];
    let mut e2a = [[Dual::constant(0.0); 9]; 9];
    let mut s = [[Dual::constant(0.0); 9]; 9];
    for a in 0..9 {
        for b in 0..9 {
            e1b[a][b] = du(
                te0.e1b[a][b],
                [tp[0].e1b[a][b], tp[1].e1b[a][b], tp[2].e1b[a][b]],
                [tm[0].e1b[a][b], tm[1].e1b[a][b], tm[2].e1b[a][b]],
            );
            e2a[a][b] = du(
                te0.e2a[a][b],
                [tp[0].e2a[a][b], tp[1].e2a[a][b], tp[2].e2a[a][b]],
                [tm[0].e2a[a][b], tm[1].e2a[a][b], tm[2].e2a[a][b]],
            );
            s[a][b] = du(
                s0[a][b],
                [sp[0][a][b], sp[1][a][b], sp[2][a][b]],
                [sm[0][a][b], sm[1][a][b], sm[2][a][b]],
            );
        }
    }
    (
        PairTwoElecG {
            norb_i: te0.norb_i,
            norb_j: te0.norb_j,
            w,
            e1b,
            e2a,
        },
        s,
    )
}

/// Build a `Dual2` from the 19-point second-order FD stencil (step `hs`) for one scalar element.
#[inline]
fn mk_d2(
    base: f64,
    p: [f64; 3],
    m: [f64; 3],
    pp: [f64; 3],
    pm: [f64; 3],
    mp: [f64; 3],
    mm: [f64; 3],
    hs: f64,
) -> Dual2 {
    let g = [
        (p[0] - m[0]) / (2.0 * hs),
        (p[1] - m[1]) / (2.0 * hs),
        (p[2] - m[2]) / (2.0 * hs),
    ];
    let mut h = [[0.0f64; 3]; 3];
    for k in 0..3 {
        h[k][k] = (p[k] - 2.0 * base + m[k]) / (hs * hs);
    }
    // Off-diagonals: pair index 0=(0,1), 1=(0,2), 2=(1,2).
    for (idx, &(i, j)) in [(0usize, 1usize), (0, 2), (1, 2)].iter().enumerate() {
        let hij = (pp[idx] - pm[idx] - mp[idx] + mm[idx]) / (4.0 * hs * hs);
        h[i][j] = hij;
        h[j][i] = hij;
    }
    Dual2 { v: base, g, h }
}

/// Hessian (Dual2) two-center integrals + overlap for a singular pair, by second-order FD.
pub(crate) fn pair_dual2_fd(
    ea: &Pm7Element,
    eb: &Pm7Element,
    d: Vec3,
    has_d: bool,
) -> (PairTwoElecG<Dual2>, [[Dual2; 9]; 9]) {
    let te = |dd: Vec3| pair_two_electron_g::<f64>(ea, eb, [dd.x, dd.y, dd.z]);
    let sv = |dd: Vec3| s_f64(ea, eb, dd, has_d);
    let te0 = te(d);
    let s0 = sv(d);
    // Per-axis ± and per-pair ±± stencil points (larger step for curvature precision).
    let (mut tp, mut tm, mut sp, mut sm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for a in AXES {
        tp.push(te(d + a * H2));
        tm.push(te(d - a * H2));
        sp.push(sv(d + a * H2));
        sm.push(sv(d - a * H2));
    }
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let (mut tpp, mut tpm, mut tmp, mut tmm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut spp, mut spm, mut smp, mut smm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &(i, j) in pairs.iter() {
        let (di, dj) = (AXES[i] * H2, AXES[j] * H2);
        tpp.push(te(d + di + dj));
        tpm.push(te(d + di - dj));
        tmp.push(te(d - di + dj));
        tmm.push(te(d - di - dj));
        spp.push(sv(d + di + dj));
        spm.push(sv(d + di - dj));
        smp.push(sv(d - di + dj));
        smm.push(sv(d - di - dj));
    }
    let cw = |get: &dyn Fn(&PairTwoElecG<f64>) -> f64,
              gets: &dyn Fn(&[[f64; 9]; 9]) -> f64,
              base_te: &PairTwoElecG<f64>,
              base_s: &[[f64; 9]; 9],
              want_s: bool|
     -> Dual2 {
        if want_s {
            mk_d2(
                gets(base_s),
                [gets(&sp[0]), gets(&sp[1]), gets(&sp[2])],
                [gets(&sm[0]), gets(&sm[1]), gets(&sm[2])],
                [gets(&spp[0]), gets(&spp[1]), gets(&spp[2])],
                [gets(&spm[0]), gets(&spm[1]), gets(&spm[2])],
                [gets(&smp[0]), gets(&smp[1]), gets(&smp[2])],
                [gets(&smm[0]), gets(&smm[1]), gets(&smm[2])],
                H2,
            )
        } else {
            mk_d2(
                get(base_te),
                [get(&tp[0]), get(&tp[1]), get(&tp[2])],
                [get(&tm[0]), get(&tm[1]), get(&tm[2])],
                [get(&tpp[0]), get(&tpp[1]), get(&tpp[2])],
                [get(&tpm[0]), get(&tpm[1]), get(&tpm[2])],
                [get(&tmp[0]), get(&tmp[1]), get(&tmp[2])],
                [get(&tmm[0]), get(&tmm[1]), get(&tmm[2])],
                H2,
            )
        }
    };
    let dummy_s = [[0.0f64; 9]; 9];
    let w: Vec<Vec<Dual2>> = (0..te0.w.len())
        .map(|i| {
            (0..te0.w[i].len())
                .map(|j| cw(&|t| t.w[i][j], &|_| 0.0, &te0, &dummy_s, false))
                .collect()
        })
        .collect();
    let mut e1b = [[Dual2::constant(0.0); 9]; 9];
    let mut e2a = [[Dual2::constant(0.0); 9]; 9];
    let mut s = [[Dual2::constant(0.0); 9]; 9];
    for a in 0..9 {
        for b in 0..9 {
            e1b[a][b] = cw(&|t| t.e1b[a][b], &|_| 0.0, &te0, &dummy_s, false);
            e2a[a][b] = cw(&|t| t.e2a[a][b], &|_| 0.0, &te0, &dummy_s, false);
            s[a][b] = cw(&|_| 0.0, &|ss| ss[a][b], &te0, &s0, true);
        }
    }
    (
        PairTwoElecG {
            norb_i: te0.norb_i,
            norb_j: te0.norb_j,
            w,
            e1b,
            e2a,
        },
        s,
    )
}
