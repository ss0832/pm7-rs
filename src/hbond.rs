// SPDX-License-Identifier: GPL-3.0-or-later

//! PM7 post-SCF hydrogen-bond correction ("EH+"), ported faithfully from MOPAC v23.2.5:
//! `src/corrections/Hydrogen_bond_corrections.F90` (topology: `all_h_bonds`,
//! `find_XH_bonds`, `find_H__Y_bonds`, `setup_DH_Plus`), `H_bond_correction_bits.F90`
//! (`bonding`, `connected`, `distance`) and `H_bond_correction_EH_plus.F90` (the energy
//! `EH_plus`), together with the geometry primitives `bangle`/`dihed`/`dang`
//! (`src/geometry/*.F90`).
//!
//! The PM7 correction is a purely geometric many-atom term: for every donor(D)–H···(A)acceptor
//! arrangement of N/O atoms it evaluates a product of the D–H–A angle, two D-/A-side angles
//! and two torsions, with distance dampings, scaled by element-dependent constants. It carries
//! **no dependence on the SCF charges** (unlike the PM6-DH2 method). Energies are in kcal/mol.
//!
//! Units: MOPAC evaluates this block with Cartesian coordinates in **Ångström**; we convert the
//! internal Bohr coordinates on entry. Constants (`ev`, `a0`, `fpc_9`, `pi`) are the MOPAC
//! `funcon_C` CODATA values, matching [`crate::constants`], so the energy reproduces MOPAC.
//!
//! Gradient: MOPAC differentiates this term by one-sided finite difference. We evaluate the same
//! fixed-topology energy with first-order three-variable AD, seeding one participating atom at a
//! time, so the reported x/y/z derivatives are analytic derivatives of the reported energy.
//!
//! Hessian: fully **analytic**. Each bond's EH+ energy is re-expressed generically over the
//! internal `HbScalar` trait (`eh_plus_g`) and instantiated at `Dual2N<27>` forward-mode second-order
//! AD, yielding the exact local ≤27×27 second-derivative block in one pass (no finite differences).
//! The generic path mirrors the `f64` energy operation-for-operation — a regression test pins
//! `eh_plus_g::<f64>` to `eh_plus` — so the analytic Hessian differentiates exactly the reported
//! energy. See [`add_hbond_hessian`].
//!
//! Provenance: MOPAC, Apache-2.0, © 2021 Virginia Polytechnic Institute and State University.
//! Method reference: M. Korth, *J. Chem. Theory Comput.* **6**, 3808 (2010); J. J. P. Stewart,
//! *J. Mol. Model.* **19**, 1 (2013).

use crate::constants::{EV_TO_KCAL, KCAL_TO_EV, PM7_A0, PM7_EV};
use crate::dual2n::{Dual2N, HbScalar};
use crate::math::Vec3;
use crate::system::Molecule;

const PI: f64 = std::f64::consts::PI;

/// MOPAC's verbatim 2π literal (`dihed.F90`/`dang.F90` use `6.28318530717959`), kept as-is
/// rather than `std::f64::consts::TAU` so the dihedral wrap matches MOPAC bit-for-bit.
#[allow(clippy::approx_constant)]
const MOPAC_TWO_PI: f64 = 6.283_185_307_179_59;

// EH_plus distance cut-offs (Ångström), from H_bond_correction_EH_plus.F90.
const SHORTCUT: f64 = 2.4;
const LONGCUT: f64 = 7.0;
const COVCUT: f64 = 1.2;

/// MOPAC-convention degrees→radians used for the target angles: `pi/(180/deg)`.
#[inline]
fn deg(d: f64) -> f64 {
    PI / (180.0 / d)
}

/// One perceived hydrogen bond, holding the 9 atom slots (0-based atom indices) of MOPAC's
/// `hblist(:,1..9)` plus the neighbour counts `nrbondsa`/`nrbondsb` and the 1-3/1-4 disable flag
/// (`hblist(:,10) == -666`).
#[derive(Clone, Debug)]
struct HBond {
    /// slots 1..9 (index 0 → slot 1, index 8 → slot 9 = the bridging hydrogen).
    s: [usize; 9],
    nrbondsa: usize,
    nrbondsb: usize,
    disabled: bool,
}

// ---------------------------------------------------------------------------
// Geometry primitives (literal ports; coordinates in Ångström, 0-based indices)
// ---------------------------------------------------------------------------

#[inline]
fn distance(c: &[Vec3], a: usize, b: usize) -> f64 {
    (c[a] - c[b]).norm()
}

/// Angle (radians) at vertex `j` of `i-j-k`. Port of `bangle.F90`.
fn bangle(c: &[Vec3], i: usize, j: usize, k: usize) -> f64 {
    let d2ij = (c[i] - c[j]).norm2();
    let d2jk = (c[j] - c[k]).norm2();
    let d2ik = (c[i] - c[k]).norm2();
    let xy = (d2ij * d2jk).sqrt();
    if xy < 1.0e-20 {
        return 0.0;
    }
    let temp = (0.5 * (d2ij + d2jk - d2ik) / xy).clamp(-1.0, 1.0);
    temp.acos()
}

/// Angle between `(a1,a2)`,`(0,0)`,`(b1,b2)`. Port of `dang.F90` (returns the value in
/// `(-2π, 0]`, matching the sign convention consumed by `dihed`).
fn dang(mut a1: f64, mut a2: f64, mut b1: f64, mut b2: f64) -> f64 {
    let zero = 1.0e-6;
    if (a1.abs() >= zero || a2.abs() >= zero) && (b1.abs() >= zero || b2.abs() >= zero) {
        let anorm = 1.0 / (a1 * a1 + a2 * a2).sqrt();
        let bnorm = 1.0 / (b1 * b1 + b2 * b2).sqrt();
        a1 *= anorm;
        a2 *= anorm;
        b1 *= bnorm;
        b2 *= bnorm;
        let sinth = a1 * b2 - a2 * b1;
        let costh = (a1 * b1 + a2 * b2).clamp(-1.0, 1.0);
        let mut rcos = costh.acos();
        if rcos.abs() >= 4.0e-5 {
            if sinth > 0.0 {
                rcos = MOPAC_TWO_PI - rcos;
            }
            return -rcos;
        }
    }
    0.0
}

/// Dihedral angle (radians, in `[0, 2π)`) of `i-j-k-l`. Port of `dihed.F90`.
fn dihed(c: &[Vec3], i: usize, j: usize, k: usize, l: usize) -> f64 {
    let xi1 = c[i].x - c[k].x;
    let xj1 = c[j].x - c[k].x;
    let xl1 = c[l].x - c[k].x;
    let yi1 = c[i].y - c[k].y;
    let yj1 = c[j].y - c[k].y;
    let yl1 = c[l].y - c[k].y;
    let zi1 = c[i].z - c[k].z;
    let zj1 = c[j].z - c[k].z;
    let zl1 = c[l].z - c[k].z;
    let dist = (xj1 * xj1 + yj1 * yj1 + zj1 * zj1).sqrt();
    let cosa = if dist > 0.0 {
        (zj1 / dist).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let ddd = 1.0 - cosa * cosa;

    // Rotate KJ onto the z-axis. The `yxdist <= 1e-6` (or `ddd <= 0`) case is MOPAC's
    // degenerate branch (`go to 10`): keep the raw components with sinth = 0.
    let (xi2, xl2, yi2, yl2, costh, sinth);
    let yxdist = if ddd > 0.0 { dist * ddd.sqrt() } else { 0.0 };
    if ddd > 0.0 && yxdist > 1.0e-6 {
        let cosph = yj1 / yxdist;
        let sinph = xj1 / yxdist;
        xi2 = xi1 * cosph - yi1 * sinph;
        xl2 = xl1 * cosph - yl1 * sinph;
        yi2 = xi1 * sinph + yi1 * cosph;
        let yj2 = xj1 * sinph + yj1 * cosph;
        yl2 = xl1 * sinph + yl1 * cosph;
        costh = cosa;
        sinth = yj2 / dist;
    } else {
        xi2 = xi1;
        xl2 = xl1;
        yi2 = yi1;
        yl2 = yl1;
        costh = cosa;
        sinth = 0.0;
    }
    let yi3 = yi2 * costh - zi1 * sinth;
    let yl3 = yl2 * costh - zl1 * sinth;
    let mut angle = dang(xl2, yl3, xi2, yi3);
    if angle < 0.0 {
        angle += 2.0 * PI;
    }
    if angle >= MOPAC_TWO_PI {
        angle = 0.0;
    }
    angle
}

// ---------------------------------------------------------------------------
// Topology perception
// ---------------------------------------------------------------------------

/// MOPAC `covrad` table (Å) for Z=1..94, already scaled by 4/3 (the `first`-time scaling in
/// `Hydrogen_bond_corrections.F90`). Used by `bonding` in `setup_DH_Plus` only.
fn covrad_scaled(z: u8) -> f64 {
    const COVRAD: [f64; 94] = [
        0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04, 1.10,
        1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09,
        1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13,
        1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57,
        1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37, 1.31, 1.23, 1.18,
        1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58,
        1.52, 1.53, 1.54, 1.55,
    ];
    let idx = z as usize;
    if idx == 0 || idx > COVRAD.len() {
        return 1.5 * 4.0 / 3.0;
    }
    COVRAD[idx - 1] * 4.0 / 3.0
}

/// `bonding(x,y)` = scaled covrad(x) + scaled covrad(y).
#[inline]
fn bonding(nat: &[u8], x: usize, y: usize) -> f64 {
    covrad_scaled(nat[x]) + covrad_scaled(nat[y])
}

/// Build the list of hydrogen bonds for the current geometry (topology only; indices are 0-based).
/// Reproduces `all_h_bonds` + `setup_DH_Plus`.
fn build_hbonds(coords: &[Vec3], nat: &[u8]) -> Vec<HBond> {
    let numat = coords.len();
    // Any N or O acceptors at all?  (max_h_bonds gate)
    if !nat.iter().any(|&z| z == 7 || z == 8) {
        return Vec::new();
    }

    // find_XH_bonds (PM7: RAH = 1.4 Å, acceptors = N/O).
    let rah2 = 1.4 * 1.4;
    let mut acceptors: Vec<usize> = Vec::new();
    let mut bonding_h: Vec<usize> = Vec::new();
    let mut used = vec![false; numat];
    for i in 0..numat {
        if nat[i] == 7 || nat[i] == 8 {
            acceptors.push(i);
            for j in 0..numat {
                if nat[j] == 1 && !used[j] && (coords[i] - coords[j]).norm2() < rah2 {
                    bonding_h.push(j);
                    used[j] = true;
                }
            }
        }
    }

    // find_H__Y_bonds (PM7: RAH = 1.4, cutoff = 7.0).
    let cutoff2 = 7.0 * 7.0;
    // pairs: (hblist1 = heavy atom bonded to H, hblist2 = H, hblist3 = distant acceptor)
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    for &i in &acceptors {
        for &j in &bonding_h {
            if (coords[i] - coords[j]).norm2() >= rah2 {
                continue;
            }
            for &k in &acceptors {
                if k == i {
                    continue;
                }
                if (coords[k] - coords[i]).norm2() >= cutoff2 {
                    continue;
                }
                if bangle(coords, k, j, i) <= PI * 0.5 {
                    continue;
                }
                // dedup: skip if (k,j,i) or (i,j,k) already present.
                let dup = pairs
                    .iter()
                    .any(|&(p1, p2, p3)| p2 == j && ((p1 == k && p3 == i) || (p1 == i && p3 == k)));
                if dup {
                    continue;
                }
                pairs.push((i, j, k));
            }
        }
    }

    // setup_DH_Plus: for each pair, find neighbours of atom1 and atom5, reference atoms, flags.
    let mut hbonds = Vec::with_capacity(pairs.len());
    for &(a1, h, a5) in &pairs {
        let mut s = [0usize; 9];
        s[0] = a1;
        s[4] = a5;
        s[8] = h;

        let na = neighbours_capped(coords, nat, a1);
        let nb = neighbours_capped(coords, nat, a5);
        let nrbondsa = na.len();
        let nrbondsb = nb.len();

        // 1-3 / 1-4 detection.
        let mut disabled = false;
        for &j in &na {
            if j == a5 {
                disabled = true;
            }
            for &kk in &nb {
                if j == kk && kk != h {
                    disabled = true;
                }
            }
        }
        if disabled {
            hbonds.push(HBond {
                s,
                nrbondsa,
                nrbondsb,
                disabled: true,
            });
            continue;
        }

        // hbs1: reference atoms for slot1 (fill slots 2,3,4).
        assign_refs(coords, nat, &na, a1, h, &mut s, 1);
        // hbs2: reference atoms for slot5 (fill slots 6,7,8).
        assign_refs(coords, nat, &nb, a5, h, &mut s, 5);

        hbonds.push(HBond {
            s,
            nrbondsa,
            nrbondsb,
            disabled: false,
        });
    }
    hbonds
}

/// Covalent neighbours of `atom`, capped at 4 (dropping the longest when a 5th appears), per
/// `setup_DH_Plus`.
fn neighbours_capped(coords: &[Vec3], nat: &[u8], atom: usize) -> Vec<usize> {
    let numat = coords.len();
    let mut list: Vec<usize> = Vec::new();
    for j in 0..numat {
        if j != atom && distance(coords, j, atom) < bonding(nat, j, atom) {
            list.push(j);
            if list.len() == 5 {
                // drop the longest bond.
                let mut worst = 0usize;
                let mut worst_d = -1.0;
                for (idx, &b) in list.iter().enumerate() {
                    let d = distance(coords, atom, b);
                    if d > worst_d {
                        worst_d = d;
                        worst = idx;
                    }
                }
                list.remove(worst);
            }
        }
    }
    list
}

/// Fill the three reference slots for one side, given the heavy atom `x`, hydrogen `h`, and the
/// heavy atom's covalent neighbours `nb`. `base` = 1 fills slots 2,3,4; `base` = 5 fills 6,7,8.
/// Port of the `hbs1`/`hbs2` blocks of `setup_DH_Plus`.
fn assign_refs(
    coords: &[Vec3],
    nat: &[u8],
    nb: &[usize],
    x: usize,
    h: usize,
    s: &mut [usize; 9],
    base: usize,
) {
    let (i2, i3, i4) = (base, base + 1, base + 2); // slot indices (0-based)
    let dist_h = |a: usize| distance(coords, a, h);
    let n = nb.len();
    if n == 3 || n == 4 {
        // three farthest-from-H neighbours, in descending order.
        let mut order = nb.to_vec();
        order.sort_by(|&a, &b| dist_h(b).partial_cmp(&dist_h(a)).unwrap());
        s[i2] = order[0];
        s[i3] = order[1];
        s[i4] = order[2];
    } else if n == 2 {
        let (far, other) = if dist_h(nb[0]) >= dist_h(nb[1]) {
            (nb[0], nb[1])
        } else {
            (nb[1], nb[0])
        };
        s[i2] = far;
        s[i3] = other;
        s[i4] = if distance(coords, x, h) < bonding(nat, x, h) {
            h
        } else {
            x
        };
    } else if n == 1 {
        s[i2] = nb[0];
        // neighbours of the first bonded atom, farthest from H.
        let nc = neighbours_all(coords, nat, nb[0]);
        let mut best = nb[0];
        let mut best_d = -1.0;
        for &k in &nc {
            let d = dist_h(k);
            if d > best_d {
                best_d = d;
                best = k;
            }
        }
        s[i3] = best;
        s[i4] = if distance(coords, x, h) < bonding(nat, x, h) {
            h
        } else {
            x
        };
    } else {
        // zero neighbours.
        let fill = if distance(coords, x, h) < bonding(nat, x, h) {
            h
        } else {
            x
        };
        s[i2] = fill;
        s[i3] = fill;
        s[i4] = fill;
    }
}

/// All covalent neighbours of `atom` (uncapped), used for the 1-neighbour reference lookup.
fn neighbours_all(coords: &[Vec3], nat: &[u8], atom: usize) -> Vec<usize> {
    let numat = coords.len();
    let mut list = Vec::new();
    for k in 0..numat {
        if k != atom && distance(coords, k, atom) < bonding(nat, k, atom) {
            list.push(k);
        }
    }
    list
}

// ---------------------------------------------------------------------------
// EH_plus energy (PM7 branch)
// ---------------------------------------------------------------------------

/// One hydrogen bond's EH+ energy (kcal/mol). Literal port of the `method_PM7` branch of
/// `EH_plus` in `H_bond_correction_EH_plus.F90`.
fn eh_plus(coords: &[Vec3], hb: &HBond, nat: &[u8]) -> f64 {
    if hb.disabled {
        return 0.0;
    }
    let a0 = PM7_A0;
    let scale_nsp3 = -0.171271 * a0 * a0;
    let scale_osp3 = -0.098822 * a0 * a0;
    let scale_nsp2 = -0.171271 * a0 * a0;
    let scale_osp2 = scale_osp3;
    let hartree2kcal = PM7_EV * EV_TO_KCAL;

    let s1 = hb.s[0];
    let s2 = hb.s[1];
    let s3 = hb.s[2];
    let s4 = hb.s[3];
    let s5 = hb.s[4];
    let s6 = hb.s[5];
    let s7 = hb.s[6];
    let s8 = hb.s[7];
    let h = hb.s[8];

    // first angle: cos(pi - angle(1,9,5))
    let angle_cos = -bangle(coords, s1, h, s5).cos();
    if angle_cos <= 0.0 {
        return 0.0;
    }

    // ---- side A (slot 1) target angles ----
    let (angle2_cos, torsion_cos) = match side_terms(coords, nat, hb.nrbondsa, s1, s2, s3, s4, h) {
        Some(v) => v,
        None => return 0.0,
    };

    // ---- side B (slot 5) target angles ----
    let (angle2_cos_new, torsion_cos_new_signed) =
        match side_terms(coords, nat, hb.nrbondsb, s5, s6, s7, s8, h) {
            Some(v) => v,
            None => return 0.0,
        };
    // Side B does not early-return on a negative torsion cosine; it uses |cos|.
    let torsion_cos_new = torsion_cos_new_signed.abs();

    // ---- scale constants ----
    let scale_a = if nat[s1] == 7 {
        if hb.nrbondsa >= 3 {
            scale_nsp3
        } else {
            scale_nsp2
        }
    } else if hb.nrbondsa >= 2 {
        scale_osp3
    } else {
        scale_osp2
    };
    let scale_b = if nat[s5] == 7 {
        if hb.nrbondsb >= 3 {
            scale_nsp3
        } else {
            scale_nsp2
        }
    } else if hb.nrbondsb >= 2 {
        scale_osp3
    } else {
        scale_osp2
    };
    let scale_c = (scale_a + scale_b) / 2.0;

    // ---- damping ----
    let ha_dist = distance(coords, h, s1);
    let hb_dist = distance(coords, h, s5);
    let xc_min = ha_dist.min(hb_dist);
    let xy_gap = ha_dist.max(hb_dist) - xc_min;
    let mut damping = if xy_gap > 0.5 {
        1.0 - 1.0 / (1.0 + (-60.0 * (xc_min / COVCUT - 1.0)).exp())
    } else {
        1.0
    };
    let oo_dist = distance(coords, s1, s5);
    damping /= 1.0 + (-100.0 * (oo_dist / SHORTCUT - 1.0)).exp();
    damping *= 1.0 - 1.0 / (1.0 + (-10.0 * (oo_dist / LONGCUT - 1.0)).exp());

    let xy_dist = distance(coords, s1, s5);
    let inner = 1.0 - angle2_cos * torsion_cos * angle2_cos_new * torsion_cos_new;
    let mut e = scale_c / xy_dist.powf(2.0)
        * angle_cos.powi(2)
        * (1.0 - inner.powi(2))
        * hartree2kcal
        * damping;

    // extra O–O short-range stabilization.
    if nat[s1] == 8 && nat[s5] == 8 {
        let short = -2.5 * (-80.0 * (xy_dist - 2.67).max(0.0).powi(2)).exp() * angle_cos.powi(4);
        e += short;
    }
    e
}

/// Compute `(angle2_cos, torsion_cos)` for one side (donor or acceptor) of the H-bond.
/// `heavy` = the O/N atom (slot 1 or 5), `r1,r2,r3` its three reference atoms (slots 2,3,4 or
/// 6,7,8), `h` the bridging hydrogen. Returns `None` when a hard cut-off (`return`) is hit.
/// Faithful port of the two structurally identical halves of `EH_plus`.
#[allow(clippy::too_many_arguments)]
fn side_terms(
    coords: &[Vec3],
    nat: &[u8],
    nrbonds: usize,
    heavy: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    h: usize,
) -> Option<(f64, f64)> {
    let mut angle2_shift = 0.0;
    let mut angle2_shift_2 = 0.0;
    let mut torsion_shift = 0.0;
    let mut torsion_check_set = false;
    let mut torsion_check_set2 = false;

    if nat[heavy] == 8 {
        if nrbonds == 1 {
            // >C=O ··· H-X
            angle2_shift = PI;
            angle2_shift_2 = deg(120.0);
            torsion_shift = 0.0;
            torsion_check_set2 = true;
        } else {
            // >O ··· H-X
            angle2_shift = deg(109.48);
            angle2_shift_2 = angle2_shift;
            torsion_shift = deg(54.74);
        }
    } else if nat[heavy] == 7 {
        if nrbonds == 2 {
            // >N ··· H-X
            angle2_shift = deg(120.0);
            angle2_shift_2 = angle2_shift;
            torsion_shift = 0.0;
        } else {
            // >N- ··· H-X  (NR3)
            angle2_shift = deg(109.48);
            angle2_shift_2 = angle2_shift;
            torsion_shift = deg(54.74);
            torsion_check_set = true;
        }
    }

    // extrapolation between tetragonal and planar NR3 group.
    let mut torsion_check = 0.0f64;
    if torsion_check_set {
        let mut tc = dihed(coords, r2, r1, heavy, r3); // torsion(3,2,1,4)
        if tc <= -PI {
            tc += 2.0 * PI;
        }
        if tc > PI {
            tc -= 2.0 * PI;
        }
        tc = if tc < 0.0 { -PI - tc } else { PI - tc };
        let tc_bac = tc; // save sign
        let mut tcp = if tc < 0.0 { -tc } else { tc };
        tcp *= 180.0 / PI;
        torsion_shift += deg((54.74 - tcp) / 54.74 * 35.26);
        angle2_shift -= deg((54.74 - tcp) / 54.74 * 19.48);
        angle2_shift_2 = angle2_shift;
        torsion_check = tc_bac; // restore sign
    }

    // second angle.
    let angle2 = bangle(coords, r1, heavy, h);
    let mut angle2_cos = (angle2_shift - angle2).cos();
    let angle2_cos_2 = (angle2_shift_2 - angle2).cos();
    if angle2_cos_2 > angle2_cos {
        angle2_cos = angle2_cos_2;
    }
    if angle2_cos <= 0.0 {
        return None;
    }

    // torsion angle.
    let mut torsion_correct = dihed(coords, r2, r1, heavy, h); // torsion(3,2,1,9)
    if torsion_correct <= -PI {
        torsion_correct += 2.0 * PI;
    }
    if torsion_correct > PI {
        torsion_correct -= 2.0 * PI;
    }
    if !torsion_check_set2 || (torsion_correct * 180.0 / PI).abs() > 90.0 {
        torsion_correct = if torsion_correct < 0.0 {
            -PI - torsion_correct
        } else {
            PI - torsion_correct
        };
    }

    let mut torsion_cos;
    if torsion_check < 0.0 {
        let mut tv = torsion_shift - torsion_correct;
        tv = wrap_pi(tv);
        torsion_cos = tv.cos();
    } else if torsion_check > 0.0 {
        let mut tv = -torsion_shift - torsion_correct;
        tv = wrap_pi(tv);
        torsion_cos = tv.cos();
    } else {
        let tv = wrap_pi(torsion_shift - torsion_correct);
        let tv2 = wrap_pi(-torsion_shift - torsion_correct);
        let c1 = tv.cos();
        let c2 = tv2.cos();
        torsion_cos = if c2 > c1 { c2 } else { c1 };
    }
    // `distance(9,1) > distance(9,2)` on the donor side; here heavy=slot1/5, r1=slot2/6.
    if distance(coords, h, heavy) > distance(coords, h, r1) && torsion_check_set2 {
        torsion_cos = 0.0;
    }
    if r2 == r3 || h == r3 {
        torsion_cos = 1.0;
    }
    Some((angle2_cos, torsion_cos))
}

/// Wrap an angle into `(-π, π]` the way the EH_plus torsion code does.
#[inline]
fn wrap_pi(mut v: f64) -> f64 {
    if v <= -PI {
        v += 2.0 * PI;
    }
    if v > PI {
        v -= 2.0 * PI;
    }
    v
}

// ---------------------------------------------------------------------------
// Generic (HbScalar) EH_plus energy — the analytic-Hessian twin of the f64 path
// ---------------------------------------------------------------------------
//
// These mirror the `f64`/`Vec3` energy above operation-for-operation over the [`HbScalar`]
// abstraction, so `eh_plus_g::<f64>` reproduces `eh_plus` to rounding, while
// `eh_plus_g::<Dual2N<27>>` yields the exact value + 27-gradient + 27×27 Hessian of one bond's
// energy in a single pass (no finite differences). The f64 path is left untouched — zero risk to
// the bit-faithful energy — and a regression test pins the two together.
//
// Coordinates are indexed the same way (Ångström); the branchy angle/torsion decisions key off
// `.val()`, exactly the smooth-branch selection the f64 port makes.

#[inline]
fn gsub<S: HbScalar>(c: &[[S; 3]], a: usize, b: usize) -> [S; 3] {
    [c[a][0] - c[b][0], c[a][1] - c[b][1], c[a][2] - c[b][2]]
}
#[inline]
fn gnorm2<S: HbScalar>(d: &[S; 3]) -> S {
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn distance_g<S: HbScalar>(c: &[[S; 3]], a: usize, b: usize) -> S {
    gnorm2(&gsub(c, a, b)).sqrt()
}

/// Generic twin of [`bangle`].
fn bangle_g<S: HbScalar>(c: &[[S; 3]], i: usize, j: usize, k: usize) -> S {
    let d2ij = gnorm2(&gsub(c, i, j));
    let d2jk = gnorm2(&gsub(c, j, k));
    let d2ik = gnorm2(&gsub(c, i, k));
    let xy = (d2ij * d2jk).sqrt();
    if xy.val() < 1.0e-20 {
        return S::cst(0.0);
    }
    let temp = ((d2ij + d2jk - d2ik) * 0.5 / xy).clamp2(-1.0, 1.0);
    temp.acos()
}

/// Generic twin of [`dang`].
fn dang_g<S: HbScalar>(mut a1: S, mut a2: S, mut b1: S, mut b2: S) -> S {
    let zero = 1.0e-6;
    if (a1.val().abs() >= zero || a2.val().abs() >= zero)
        && (b1.val().abs() >= zero || b2.val().abs() >= zero)
    {
        let anorm = (a1 * a1 + a2 * a2).sqrt().recip();
        let bnorm = (b1 * b1 + b2 * b2).sqrt().recip();
        a1 = a1 * anorm;
        a2 = a2 * anorm;
        b1 = b1 * bnorm;
        b2 = b2 * bnorm;
        let sinth = a1 * b2 - a2 * b1;
        let costh = (a1 * b1 + a2 * b2).clamp2(-1.0, 1.0);
        let rcos = costh.acos();
        if rcos.val().abs() >= 4.0e-5 {
            if sinth.val() > 0.0 {
                return rcos - MOPAC_TWO_PI; // -(2π − rcos)
            }
            return -rcos;
        }
    }
    S::cst(0.0)
}

/// Generic twin of [`dihed`].
fn dihed_g<S: HbScalar>(c: &[[S; 3]], i: usize, j: usize, k: usize, l: usize) -> S {
    let xi1 = c[i][0] - c[k][0];
    let xj1 = c[j][0] - c[k][0];
    let xl1 = c[l][0] - c[k][0];
    let yi1 = c[i][1] - c[k][1];
    let yj1 = c[j][1] - c[k][1];
    let yl1 = c[l][1] - c[k][1];
    let zi1 = c[i][2] - c[k][2];
    let zj1 = c[j][2] - c[k][2];
    let zl1 = c[l][2] - c[k][2];
    let dist = (xj1 * xj1 + yj1 * yj1 + zj1 * zj1).sqrt();
    let cosa = if dist.val() > 0.0 {
        (zj1 / dist).clamp2(-1.0, 1.0)
    } else {
        S::cst(0.0)
    };
    let ddd = -(cosa * cosa) + 1.0;

    let yxdist = if ddd.val() > 0.0 {
        dist * ddd.sqrt()
    } else {
        S::cst(0.0)
    };
    let (xi2, xl2, yi2, yl2, costh, sinth);
    if ddd.val() > 0.0 && yxdist.val() > 1.0e-6 {
        let cosph = yj1 / yxdist;
        let sinph = xj1 / yxdist;
        xi2 = xi1 * cosph - yi1 * sinph;
        xl2 = xl1 * cosph - yl1 * sinph;
        yi2 = xi1 * sinph + yi1 * cosph;
        let yj2 = xj1 * sinph + yj1 * cosph;
        yl2 = xl1 * sinph + yl1 * cosph;
        costh = cosa;
        sinth = yj2 / dist;
    } else {
        xi2 = xi1;
        xl2 = xl1;
        yi2 = yi1;
        yl2 = yl1;
        costh = cosa;
        sinth = S::cst(0.0);
    }
    let yi3 = yi2 * costh - zi1 * sinth;
    let yl3 = yl2 * costh - zl1 * sinth;
    let mut angle = dang_g(xl2, yl3, xi2, yi3);
    if angle.val() < 0.0 {
        angle = angle + 2.0 * PI;
    }
    if angle.val() >= MOPAC_TWO_PI {
        angle = S::cst(0.0);
    }
    angle
}

/// Generic twin of [`wrap_pi`].
#[inline]
fn wrap_pi_g<S: HbScalar>(mut v: S) -> S {
    if v.val() <= -PI {
        v = v + 2.0 * PI;
    }
    if v.val() > PI {
        v = v - 2.0 * PI;
    }
    v
}

/// Generic twin of [`side_terms`]. The `angle2_shift`/`torsion_shift` targets are kept as `S`
/// (not `f64`) because the NR3 branch feeds a geometry-dependent (`dihed_g`) correction into them.
#[allow(clippy::too_many_arguments)]
fn side_terms_g<S: HbScalar>(
    coords: &[[S; 3]],
    nat: &[u8],
    nrbonds: usize,
    heavy: usize,
    r1: usize,
    r2: usize,
    r3: usize,
    h: usize,
) -> Option<(S, S)> {
    let mut angle2_shift = S::cst(0.0);
    let mut angle2_shift_2 = S::cst(0.0);
    let mut torsion_shift = S::cst(0.0);
    let mut torsion_check_set = false;
    let mut torsion_check_set2 = false;

    if nat[heavy] == 8 {
        if nrbonds == 1 {
            angle2_shift = S::cst(PI);
            angle2_shift_2 = S::cst(deg(120.0));
            torsion_shift = S::cst(0.0);
            torsion_check_set2 = true;
        } else {
            angle2_shift = S::cst(deg(109.48));
            angle2_shift_2 = angle2_shift;
            torsion_shift = S::cst(deg(54.74));
        }
    } else if nat[heavy] == 7 {
        if nrbonds == 2 {
            angle2_shift = S::cst(deg(120.0));
            angle2_shift_2 = angle2_shift;
            torsion_shift = S::cst(0.0);
        } else {
            angle2_shift = S::cst(deg(109.48));
            angle2_shift_2 = angle2_shift;
            torsion_shift = S::cst(deg(54.74));
            torsion_check_set = true;
        }
    }

    let mut torsion_check = S::cst(0.0);
    if torsion_check_set {
        let mut tc = dihed_g(coords, r2, r1, heavy, r3);
        if tc.val() <= -PI {
            tc = tc + 2.0 * PI;
        }
        if tc.val() > PI {
            tc = tc - 2.0 * PI;
        }
        // tc = if tc < 0 { -PI - tc } else { PI - tc }
        tc = if tc.val() < 0.0 { -tc - PI } else { -tc + PI };
        let tc_bac = tc;
        let mut tcp = if tc.val() < 0.0 { -tc } else { tc };
        tcp = tcp * (180.0 / PI);
        // torsion_shift += deg((54.74 - tcp)/54.74*35.26);  deg(x) = x·π/180
        let d_shift = (-tcp + 54.74) / 54.74 * 35.26 * (PI / 180.0);
        torsion_shift = torsion_shift + d_shift;
        let a_shift = (-tcp + 54.74) / 54.74 * 19.48 * (PI / 180.0);
        angle2_shift = angle2_shift - a_shift;
        angle2_shift_2 = angle2_shift;
        torsion_check = tc_bac;
    }

    let angle2 = bangle_g(coords, r1, heavy, h);
    let mut angle2_cos = (angle2_shift - angle2).cos();
    let angle2_cos_2 = (angle2_shift_2 - angle2).cos();
    if angle2_cos_2.val() > angle2_cos.val() {
        angle2_cos = angle2_cos_2;
    }
    if angle2_cos.val() <= 0.0 {
        return None;
    }

    let mut torsion_correct = dihed_g(coords, r2, r1, heavy, h);
    if torsion_correct.val() <= -PI {
        torsion_correct = torsion_correct + 2.0 * PI;
    }
    if torsion_correct.val() > PI {
        torsion_correct = torsion_correct - 2.0 * PI;
    }
    if !torsion_check_set2 || (torsion_correct.val() * 180.0 / PI).abs() > 90.0 {
        torsion_correct = if torsion_correct.val() < 0.0 {
            -torsion_correct - PI
        } else {
            -torsion_correct + PI
        };
    }

    let mut torsion_cos;
    if torsion_check.val() < 0.0 {
        let tv = wrap_pi_g(torsion_shift - torsion_correct);
        torsion_cos = tv.cos();
    } else if torsion_check.val() > 0.0 {
        let tv = wrap_pi_g(-torsion_shift - torsion_correct);
        torsion_cos = tv.cos();
    } else {
        let tv = wrap_pi_g(torsion_shift - torsion_correct);
        let tv2 = wrap_pi_g(-torsion_shift - torsion_correct);
        let c1 = tv.cos();
        let c2 = tv2.cos();
        torsion_cos = if c2.val() > c1.val() { c2 } else { c1 };
    }
    if distance_g(coords, h, heavy).val() > distance_g(coords, h, r1).val() && torsion_check_set2 {
        torsion_cos = S::cst(0.0);
    }
    if r2 == r3 || h == r3 {
        torsion_cos = S::cst(1.0);
    }
    Some((angle2_cos, torsion_cos))
}

/// Generic twin of [`eh_plus`]: one hydrogen bond's EH+ energy (kcal/mol) over any [`HbScalar`].
fn eh_plus_g<S: HbScalar>(coords: &[[S; 3]], hb: &HBond, nat: &[u8]) -> S {
    if hb.disabled {
        return S::cst(0.0);
    }
    let a0 = PM7_A0;
    let scale_nsp3 = -0.171271 * a0 * a0;
    let scale_osp3 = -0.098822 * a0 * a0;
    let scale_nsp2 = -0.171271 * a0 * a0;
    let scale_osp2 = scale_osp3;
    let hartree2kcal = PM7_EV * EV_TO_KCAL;

    let s1 = hb.s[0];
    let s2 = hb.s[1];
    let s3 = hb.s[2];
    let s4 = hb.s[3];
    let s5 = hb.s[4];
    let s6 = hb.s[5];
    let s7 = hb.s[6];
    let s8 = hb.s[7];
    let h = hb.s[8];

    let angle_cos = -bangle_g(coords, s1, h, s5).cos();
    if angle_cos.val() <= 0.0 {
        return S::cst(0.0);
    }

    let (angle2_cos, torsion_cos) = match side_terms_g(coords, nat, hb.nrbondsa, s1, s2, s3, s4, h)
    {
        Some(v) => v,
        None => return S::cst(0.0),
    };
    let (angle2_cos_new, torsion_cos_new_signed) =
        match side_terms_g(coords, nat, hb.nrbondsb, s5, s6, s7, s8, h) {
            Some(v) => v,
            None => return S::cst(0.0),
        };
    let torsion_cos_new = torsion_cos_new_signed.abs();

    let scale_a = if nat[s1] == 7 {
        if hb.nrbondsa >= 3 {
            scale_nsp3
        } else {
            scale_nsp2
        }
    } else if hb.nrbondsa >= 2 {
        scale_osp3
    } else {
        scale_osp2
    };
    let scale_b = if nat[s5] == 7 {
        if hb.nrbondsb >= 3 {
            scale_nsp3
        } else {
            scale_nsp2
        }
    } else if hb.nrbondsb >= 2 {
        scale_osp3
    } else {
        scale_osp2
    };
    let scale_c = (scale_a + scale_b) / 2.0;

    let ha_dist = distance_g(coords, h, s1);
    let hb_dist = distance_g(coords, h, s5);
    let xc_min = ha_dist.min(hb_dist);
    let xy_gap = ha_dist.max(hb_dist) - xc_min;
    let mut damping = if xy_gap.val() > 0.5 {
        let ex = ((xc_min / COVCUT - 1.0) * -60.0).exp();
        -(ex + 1.0).recip() + 1.0
    } else {
        S::cst(1.0)
    };
    let oo_dist = distance_g(coords, s1, s5);
    let denom = ((oo_dist / SHORTCUT - 1.0) * -100.0).exp() + 1.0;
    damping = damping / denom;
    let ex2 = ((oo_dist / LONGCUT - 1.0) * -10.0).exp();
    damping = damping * (-(ex2 + 1.0).recip() + 1.0);

    let xy_dist = distance_g(coords, s1, s5);
    let prod = angle2_cos * torsion_cos * angle2_cos_new * torsion_cos_new;
    let inner = -prod + 1.0;
    let mut e = xy_dist.powf(2.0).recip()
        * scale_c
        * angle_cos.powi(2)
        * (-inner.powi(2) + 1.0)
        * hartree2kcal
        * damping;

    if nat[s1] == 8 && nat[s5] == 8 {
        let g = (xy_dist - 2.67).max(S::cst(0.0));
        let short = (g.powi(2) * -80.0).exp() * angle_cos.powi(4) * -2.5;
        e = e + short;
    }
    e
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Molecule coordinates converted to Ångström (the unit MOPAC's H-bond block uses).
fn coords_angstrom(molecule: &Molecule) -> Vec<Vec3> {
    molecule.atoms.iter().map(|a| a.position * PM7_A0).collect()
}

fn atomic_numbers(molecule: &Molecule) -> Vec<u8> {
    molecule.atoms.iter().map(|a| a.z).collect()
}

/// Total PM7 hydrogen-bond correction energy (kcal/mol).
pub fn hydrogen_bond_energy(molecule: &Molecule) -> f64 {
    let coords = coords_angstrom(molecule);
    let nat = atomic_numbers(molecule);
    let hbonds = build_hbonds(&coords, &nat);
    hbonds.iter().map(|hb| eh_plus(&coords, hb, &nat)).sum()
}

/// Sum EH+ over a fixed H-bond list at the given (Ångström) coordinates.
/// For each atom, the indices of the (non-disabled) hydrogen bonds it participates in. Used to
/// restrict the finite-difference gradient/Hessian to the H-bonds actually affected by moving an
/// atom, turning the naive O(N_atoms · N_hbonds) sweep into O(Σ hbond sizes).
fn atom_to_hbonds(hbonds: &[HBond], n: usize) -> Vec<Vec<usize>> {
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (hi, hb) in hbonds.iter().enumerate() {
        if hb.disabled {
            continue;
        }
        let mut seen: [usize; 9] = hb.s;
        seen.sort_unstable();
        let mut last = usize::MAX;
        for &a in &seen {
            if a != last {
                map[a].push(hi);
                last = a;
            }
        }
    }
    map
}

/// PM7 hydrogen-bond correction gradient, in **kcal/mol per Bohr** (the caller multiplies by
/// `KCAL_TO_EV` to reach eV/Bohr, matching the dispersion path). Central finite difference of the
/// energy at fixed H-bond topology.
#[allow(dead_code)]
fn hydrogen_bond_gradient_fd(molecule: &Molecule) -> Vec<Vec3> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let coords0 = coords_angstrom(molecule);
    let nat = atomic_numbers(molecule);
    let hbonds = build_hbonds(&coords0, &nat);
    if hbonds.is_empty() {
        return vec![Vec3::zero(); n];
    }
    let atom_hb = atom_to_hbonds(&hbonds, n);
    let delta = 1.0e-4; // Å
                        // Each atom's 3 gradient components are independent; only the H-bonds involving that atom
                        // depend on it, so the perturbed sum runs over `atom_hb[a]` alone. Parallel over atoms, each
                        // task holding its own perturbed coordinate copy (race-free).
    (0..n)
        .into_par_iter()
        .map(|a| {
            let mut g = Vec3::zero();
            if atom_hb[a].is_empty() {
                return g;
            }
            let mut coords = coords0.clone();
            for c in 0..3 {
                let orig = component(coords[a], c);
                set_component(&mut coords[a], c, orig + delta);
                let ep: f64 = atom_hb[a]
                    .iter()
                    .map(|&hi| eh_plus(&coords, &hbonds[hi], &nat))
                    .sum();
                set_component(&mut coords[a], c, orig - delta);
                let em: f64 = atom_hb[a]
                    .iter()
                    .map(|&hi| eh_plus(&coords, &hbonds[hi], &nat))
                    .sum();
                set_component(&mut coords[a], c, orig);
                set_component(&mut g, c, (ep - em) / (2.0 * delta) * PM7_A0);
            }
            g
        })
        .collect()
}

/// Analytic PM7 hydrogen-bond correction gradient in kcal/mol per Bohr. The generic EH+
/// expression is evaluated once per participating atom with x/y/z seeded as first-order duals.
pub fn hydrogen_bond_gradient(molecule: &Molecule) -> Vec<Vec3> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let coords0 = coords_angstrom(molecule);
    let nat = atomic_numbers(molecule);
    let hbonds = build_hbonds(&coords0, &nat);
    let contributions: Vec<(usize, Vec3)> = hbonds
        .par_iter()
        .filter(|hb| !hb.disabled)
        .flat_map_iter(|hb| hbond_gradient_block(hb, &coords0, &nat))
        .collect();
    let mut gradient = vec![Vec3::zero(); n];
    for (atom, value) in contributions {
        gradient[atom] += value;
    }
    gradient
}

fn hbond_gradient_block(hb: &HBond, coords0: &[Vec3], nat: &[u8]) -> Vec<(usize, Vec3)> {
    use crate::dual::Dual;
    let mut atoms: Vec<usize> = hb.s.to_vec();
    atoms.sort_unstable();
    atoms.dedup();
    let local_coords: Vec<Vec3> = atoms.iter().map(|&index| coords0[index]).collect();
    let local_nat: Vec<u8> = atoms.iter().map(|&index| nat[index]).collect();
    let mut local_bond = hb.clone();
    for slot in &mut local_bond.s {
        *slot = atoms
            .binary_search(slot)
            .expect("bond atom present in its own list");
    }

    let mut output = Vec::with_capacity(atoms.len());
    for (seed, &global_atom) in atoms.iter().enumerate() {
        let mut coordinates: Vec<[Dual; 3]> = local_coords
            .iter()
            .map(|p| {
                [
                    Dual::constant(p.x),
                    Dual::constant(p.y),
                    Dual::constant(p.z),
                ]
            })
            .collect();
        coordinates[seed] = [
            Dual::var(local_coords[seed].x, 0),
            Dual::var(local_coords[seed].y, 1),
            Dual::var(local_coords[seed].z, 2),
        ];
        let energy = eh_plus_g::<Dual>(&coordinates, &local_bond, &local_nat);
        output.push((
            global_atom,
            Vec3::new(energy.d[0], energy.d[1], energy.d[2]) * PM7_A0,
        ));
    }
    output
}

/// PM7 hydrogen-bond contribution to the Hessian, in **eV/Bohr²**, scattered into `hess`.
/// No-op when the geometry has no hydrogen bonds (e.g. every isolated-molecule test case).
///
/// Each H-bond's EH+ energy is a purely geometric function of its ≤ 9 atoms, so its Hessian is a
/// small (≤ 27×27) local block and the total is the sum over bonds. Every block is evaluated
/// **analytically** in a single pass with `Dual2N<27>` forward-mode second-order AD — no finite
/// differences — by seeding only that bond's atoms and scattering the local block into `hess`.
///
/// OOM / scaling: the working set is one `Dual2N<27>` expression tree per bond (a few hundred KB
/// of stack, *independent of system size*), and the only heap growth is the `(row, col, value)`
/// triples (≤ 27² per bond). Nothing is ever materialised at O(N_atoms²), so a large molecule with
/// a handful of H-bonds costs the same per bond as a small one. Work is parallel over bonds.
pub fn add_hbond_hessian(molecule: &Molecule, hess: &mut crate::linalg::Matrix) {
    use rayon::prelude::*;
    let coords0 = coords_angstrom(molecule);
    let nat = atomic_numbers(molecule);
    let hbonds = build_hbonds(&coords0, &nat);
    if hbonds.is_empty() {
        return;
    }
    // d²E/dÅ² (kcal/mol) → eV/Bohr²: ×PM7_A0² (Å²/Bohr²) then ×KCAL_TO_EV.
    let unit = PM7_A0 * PM7_A0 * KCAL_TO_EV;
    // The per-bond `Dual2N<27>` expression trees are large (~6 KB/node) and nest deeply, so run
    // them on a dedicated pool whose workers have an oversized stack (reserved address space is
    // committed lazily — no real memory cost) rather than the 1–2 MB default rayon/test stacks.
    let contribs: Vec<(usize, usize, f64)> = hbond_pool().install(|| {
        hbonds
            .par_iter()
            .filter(|hb| !hb.disabled)
            .flat_map_iter(|hb| hbond_hessian_block(hb, &coords0, &nat, unit))
            .collect()
    });
    for (row, col, v) in contribs {
        hess[(row, col)] += v;
    }
}

/// Process-wide rayon pool with a large stack, used only for the analytic H-bond Hessian blocks.
fn hbond_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads().clamp(1, 4))
            .stack_size(32 * 1024 * 1024)
            .thread_name(|i| format!("pm7-hbond-hess-{i}"))
            .build()
            .expect("build H-bond Hessian rayon pool")
    })
}

/// Exact local Hessian of one H-bond's EH+ energy via [`Dual2N<27>`], returned as global
/// `(row, col, value)` triples already scaled to eV/Bohr² by `unit`.
fn hbond_hessian_block(
    hb: &HBond,
    coords0: &[Vec3],
    nat: &[u8],
    unit: f64,
) -> Vec<(usize, usize, f64)> {
    // Unique atoms of this bond (≤ 9). 3·m ≤ 27 = N, so `Dual2N<27>` always has room.
    let mut atoms: Vec<usize> = hb.s.to_vec();
    atoms.sort_unstable();
    atoms.dedup();
    let m = atoms.len();

    // Local coordinates (length m): every Cartesian seeded as an independent variable, plus a
    // local atomic-number table and a slot→local-index remap of the bond.
    let mut lc: Vec<[Dual2N<27>; 3]> = Vec::with_capacity(m);
    let mut lnat: Vec<u8> = Vec::with_capacity(m);
    for (li, &g) in atoms.iter().enumerate() {
        let p = coords0[g];
        lc.push([
            Dual2N::<27>::var(p.x, 3 * li),
            Dual2N::<27>::var(p.y, 3 * li + 1),
            Dual2N::<27>::var(p.z, 3 * li + 2),
        ]);
        lnat.push(nat[g]);
    }
    let mut lhb = hb.clone();
    for s in lhb.s.iter_mut() {
        *s = atoms
            .binary_search(s)
            .expect("bond atom present in its own list");
    }

    let e = eh_plus_g::<Dual2N<27>>(&lc, &lhb, &lnat);

    let mut out = Vec::with_capacity(9 * m * m);
    for (li, &gi) in atoms.iter().enumerate() {
        for ci in 0..3 {
            let a = 3 * li + ci;
            let row = 3 * gi + ci;
            for (lj, &gj) in atoms.iter().enumerate() {
                for cj in 0..3 {
                    let hval = e.h[a][3 * lj + cj];
                    if hval != 0.0 {
                        out.push((row, 3 * gj + cj, hval * unit));
                    }
                }
            }
        }
    }
    out
}

#[inline]
fn component(v: Vec3, i: usize) -> f64 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[inline]
fn set_component(v: &mut Vec3, i: usize, val: f64) {
    match i {
        0 => v.x = val,
        1 => v.y = val,
        _ => v.z = val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asymmetric water dimer in **Ångström** (the unit the EH+ block works in). Asymmetric so
    /// `setup_DH_Plus`'s distance-sorted reference slots don't sit on a swap-tie under FD.
    fn water_dimer_ang() -> (Vec<Vec3>, Vec<u8>) {
        let coords = vec![
            Vec3::new(-1.551007, -0.114520, 0.000000),
            Vec3::new(-1.934259, 0.762503, 0.000000),
            Vec3::new(-0.599677, 0.040712, 0.000000),
            Vec3::new(1.350625, 0.111469, 0.000000),
            Vec3::new(1.680398, -0.373741, -0.758561),
            Vec3::new(1.980398, -0.073741, 0.858561),
        ];
        (coords, vec![8, 1, 1, 8, 1, 1])
    }

    /// The generic energy at `f64` must reproduce the validated `Vec3` energy for every perceived
    /// bond (so its `Dual2N` twin differentiates *the same* function). This is the guard that keeps
    /// the analytic-Hessian port honest against the bit-faithful energy path.
    #[test]
    fn eh_plus_generic_reproduces_scalar() {
        let (coords, nat) = water_dimer_ang();
        let hbonds = build_hbonds(&coords, &nat);
        assert!(
            !hbonds.is_empty(),
            "water dimer must perceive at least one H-bond"
        );
        let coords_arr: Vec<[f64; 3]> = coords.iter().map(|v| [v.x, v.y, v.z]).collect();
        let mut active = 0;
        for hb in &hbonds {
            let e_ref = eh_plus(&coords, hb, &nat);
            let e_gen = eh_plus_g::<f64>(&coords_arr, hb, &nat);
            assert!(
                (e_ref - e_gen).abs() < 1.0e-9,
                "eh_plus_g::<f64> {e_gen} vs eh_plus {e_ref}"
            );
            if e_ref.abs() > 1.0e-6 {
                active += 1;
            }
        }
        assert!(active >= 1, "expected an attractive (non-zero) H-bond term");
    }

    /// The analytic `Dual2N<27>` local Hessian of a single bond must match an independent
    /// finite-difference of the `f64` energy (gradient *and* second derivatives).
    #[test]
    fn eh_plus_analytic_block_matches_fd() {
        let (coords, nat) = water_dimer_ang();
        let hbonds = build_hbonds(&coords, &nat);
        // Pick the bond that actually carries energy.
        let hb = hbonds
            .iter()
            .find(|hb| !hb.disabled && eh_plus(&coords, hb, &nat).abs() > 1.0e-6)
            .expect("an active H-bond");

        let mut atoms: Vec<usize> = hb.s.to_vec();
        atoms.sort_unstable();
        atoms.dedup();
        let m = atoms.len();
        let dim = 3 * m;

        // Local f64 coords + remapped bond for the FD reference.
        let lcf0: Vec<Vec3> = atoms.iter().map(|&g| coords[g]).collect();
        let lnat: Vec<u8> = atoms.iter().map(|&g| nat[g]).collect();
        let mut lhb = hb.clone();
        for s in lhb.s.iter_mut() {
            *s = atoms.binary_search(s).unwrap();
        }

        // Analytic: seed every local Cartesian and evaluate once (on a big-stack thread, matching
        // the production `hbond_pool`, since a `Dual2N<27>` tree overflows the default test stack).
        let mut lc: Vec<[Dual2N<27>; 3]> = Vec::with_capacity(m);
        for (li, p) in lcf0.iter().enumerate() {
            lc.push([
                Dual2N::<27>::var(p.x, 3 * li),
                Dual2N::<27>::var(p.y, 3 * li + 1),
                Dual2N::<27>::var(p.z, 3 * li + 2),
            ]);
        }
        let (lc_t, lhb_t, lnat_t) = (lc, lhb.clone(), lnat.clone());
        let (e_v, e_g, e_h) = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let e = eh_plus_g::<Dual2N<27>>(&lc_t, &lhb_t, &lnat_t);
                (e.v, e.g, e.h)
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(
            (e_v - eh_plus(&lcf0, &lhb, &lnat)).abs() < 1.0e-9,
            "analytic value drift"
        );

        // FD reference: perturb the f64 energy at two (possibly equal) local Cartesians.
        let h = 1.0e-4;
        let eval = |a: usize, sa: f64, b: usize, sb: f64| -> f64 {
            let mut c = lcf0.clone();
            let ca = component(c[a / 3], a % 3);
            set_component(&mut c[a / 3], a % 3, ca + sa * h);
            let cb = component(c[b / 3], b % 3);
            set_component(&mut c[b / 3], b % 3, cb + sb * h);
            eh_plus(&c, &lhb, &lnat)
        };

        // Gradient check.
        let mut gmax = 0.0f64;
        for a in 0..dim {
            let fd = (eval(a, 1.0, a, 0.0) - eval(a, -1.0, a, 0.0)) / (2.0 * h);
            gmax = gmax.max((e_g[a] - fd).abs());
        }
        assert!(gmax < 1.0e-4, "analytic H-bond gradient vs FD {gmax:.3e}");

        // Hessian check (mixed central difference, valid on and off diagonal).
        let mut hmax = 0.0f64;
        for a in 0..dim {
            for b in 0..dim {
                let fd = (eval(a, 1.0, b, 1.0) - eval(a, 1.0, b, -1.0) - eval(a, -1.0, b, 1.0)
                    + eval(a, -1.0, b, -1.0))
                    / (4.0 * h * h);
                hmax = hmax.max((e_h[a][b] - fd).abs());
            }
        }
        assert!(
            hmax < 1.0e-2,
            "analytic H-bond Hessian vs FD {hmax:.3e} kcal/Å²"
        );
    }
}
