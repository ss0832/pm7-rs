// SPDX-License-Identifier: GPL-3.0-or-later

//! MNDO/d one-center integral machinery for the PM7 `spd` basis.
//!
//! This is a faithful port of the geometry-independent parts of MOPAC's
//! `src/integrals/mndod.F90` (`fbx`, `rsc`, `scprm`, `inighd`, `aijm`, `ddpo`,
//! `poij`, `eiscor`): the 52 one-center two-electron integrals `repd`, the
//! Klopman–Ohno additive terms `po`, the multipole charge separations `ddp`, and
//! the `aij` multipole coefficients (Thiel–Voityuk *Theor. Chim. Acta* Eq. 7).
//! Everything here is computed once per element at parameter-load time in `f64`;
//! the distance-dependent two-center kernels (which must be generic over
//! [`crate::dual::Scalar`]) consume these values.
//!
//! Provenance: MOPAC v23.2.5, Apache-2.0 (© 2021 Virginia Tech). See
//! `THIRD_PARTY_NOTICES.md`. Reference: W. Thiel, A. A. Voityuk,
//! *Theor. Chim. Acta* **81**, 391 (1992); *J. Phys. Chem.* **100**, 616 (1996).

use crate::constants::PM7_EV;
use crate::dual::Scalar;

/// Interaction between two point-charge multipole configurations (MOPAC `charg`),
/// generic over the scalar type so the two-center d-integral derivatives follow
/// automatically. `r` is the interatomic distance (Bohr); `da`/`db` are the
/// multipole charge separations and `add` the Klopman additive term (both
/// geometry-independent constants). `l1`/`l2` are multipole orders (0,1,2) and
/// `m` the shared azimuthal component.
///
/// Reference: Thiel–Voityuk multipole model; matches the sp Dewar–Sabelli–Klopman
/// expansion in [`crate::integrals`] for the sp subset.
pub fn charg<S: Scalar>(r: S, l1: u8, l2: u8, m: u8, da: f64, db: f64, add: f64) -> S {
    // 1/sqrt((r + shift)^2 + off), with `off` a runtime f64 constant.
    let inv = |x: S, off: f64| (x * x + off).sqrt().recip();
    match (l1, l2, m) {
        // Q - Q
        (0, 0, _) => (r * r + add).sqrt().recip(),
        // Z - Q
        (1, 0, _) => (inv(r - da, add) - inv(r + da, add)) * 0.5,
        // Q - Z
        (0, 1, _) => (inv(r + db, add) - inv(r - db, add)) * 0.5,
        // Z - Z
        (1, 1, 0) => {
            (inv(r + da - db, add) + inv(r - da + db, add)
                - inv(r - da - db, add)
                - inv(r + da + db, add))
                * 0.25
        }
        // X - X
        (1, 1, _) => {
            ((r * r + (da - db) * (da - db) + add).sqrt().recip()
                - (r * r + (da + db) * (da + db) + add).sqrt().recip())
                * 0.5
        }
        // Q - ZZ
        (0, 2, _) => {
            (inv(r - db, add) - (r * r + (db * db + add)).sqrt().recip() * 2.0 + inv(r + db, add))
                * 0.25
        }
        // ZZ - Q
        (2, 0, _) => {
            (inv(r - da, add) - (r * r + (da * da + add)).sqrt().recip() * 2.0 + inv(r + da, add))
                * 0.25
        }
        // Z - ZZ
        (1, 2, 0) => {
            (inv(r - da - db, add) - ((r - da) * (r - da) + (db * db + add)).sqrt().recip() * 2.0
                + inv(r + db - da, add)
                - inv(r - db + da, add)
                + ((r + da) * (r + da) + (db * db + add)).sqrt().recip() * 2.0
                - inv(r + da + db, add))
                * 0.125
        }
        // ZZ - Z
        (2, 1, 0) => {
            (-inv(r - da - db, add) + ((r - db) * (r - db) + (da * da + add)).sqrt().recip() * 2.0
                - inv(r + da - db, add)
                + inv(r - da + db, add)
                - ((r + db) * (r + db) + (da * da + add)).sqrt().recip() * 2.0
                + inv(r + da + db, add))
                * 0.125
        }
        // ZZ - ZZ
        (2, 2, 0) => {
            let zzzz = inv(r - da - db, add)
                + inv(r + da + db, add)
                + inv(r - da + db, add)
                + inv(r + da - db, add)
                - ((r - da) * (r - da) + (db * db + add)).sqrt().recip() * 2.0
                - ((r - db) * (r - db) + (da * da + add)).sqrt().recip() * 2.0
                - ((r + da) * (r + da) + (db * db + add)).sqrt().recip() * 2.0
                - ((r + db) * (r + db) + (da * da + add)).sqrt().recip() * 2.0
                + (r * r + ((da - db) * (da - db) + add)).sqrt().recip() * 2.0
                + (r * r + ((da + db) * (da + db) + add)).sqrt().recip() * 2.0;
            let xyxy = (r * r + ((da - db) * (da - db) + add)).sqrt().recip() * 4.0
                + (r * r + ((da + db) * (da + db) + add)).sqrt().recip() * 4.0
                - (r * r + (da * da + db * db + add)).sqrt().recip() * 8.0;
            zzzz * (1.0 / 16.0) - xyxy * (1.0 / 64.0)
        }
        // X - ZX
        (1, 2, _) => {
            let ab = db / 2.0_f64.sqrt();
            (((r - ab) * (r - ab) + ((da - ab) * (da - ab) + add))
                .sqrt()
                .recip()
                * (-2.0)
                + ((r + ab) * (r + ab) + ((da - ab) * (da - ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                + ((r - ab) * (r - ab) + ((da + ab) * (da + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                - ((r + ab) * (r + ab) + ((da + ab) * (da + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0)
                * 0.125
        }
        // ZX - X
        (2, 1, _) => {
            let aa = da / 2.0_f64.sqrt();
            (((r + aa) * (r + aa) + ((aa - db) * (aa - db) + add))
                .sqrt()
                .recip()
                * (-2.0)
                + ((r - aa) * (r - aa) + ((aa - db) * (aa - db) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                + ((r + aa) * (r + aa) + ((aa + db) * (aa + db) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                - ((r - aa) * (r - aa) + ((aa + db) * (aa + db) + add))
                    .sqrt()
                    .recip()
                    * 2.0)
                * 0.125
        }
        // ZX - ZX
        (2, 2, 1) => {
            let aa = da / 2.0_f64.sqrt();
            let ab = db / 2.0_f64.sqrt();
            (((r + aa - ab) * (r + aa - ab) + ((aa - ab) * (aa - ab) + add))
                .sqrt()
                .recip()
                * 2.0
                - ((r + aa + ab) * (r + aa + ab) + ((aa - ab) * (aa - ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                - ((r - aa - ab) * (r - aa - ab) + ((aa - ab) * (aa - ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                + ((r - aa + ab) * (r - aa + ab) + ((aa - ab) * (aa - ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                - ((r + aa - ab) * (r + aa - ab) + ((aa + ab) * (aa + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                + ((r + aa + ab) * (r + aa + ab) + ((aa + ab) * (aa + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                + ((r - aa - ab) * (r - aa - ab) + ((aa + ab) * (aa + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0
                - ((r - aa + ab) * (r - aa + ab) + ((aa + ab) * (aa + ab) + add))
                    .sqrt()
                    .recip()
                    * 2.0)
                * (1.0 / 16.0)
        }
        // XX - XX
        (2, 2, _) => {
            let xyxy = (r * r + ((da - db) * (da - db) + add)).sqrt().recip() * 4.0
                + (r * r + ((da + db) * (da + db) + add)).sqrt().recip() * 4.0
                - (r * r + (da * da + db * db + add)).sqrt().recip() * 8.0;
            xyxy * (1.0 / 16.0)
        }
        _ => S::cst(0.0),
    }
}

/// One-center MNDO/d parameters used by the two-center integral kernel.
///
/// Fortran arrays are 1-based; we keep the same indexing (slot 0 is unused) so
/// the ported formulas read identically to `mndod.F90`.
#[derive(Clone, Debug)]
pub struct DShell {
    /// One-center two-electron integrals, `repd(1..=52)` (eV).
    pub repd: [f64; 53],
    /// Klopman–Ohno additive terms (Bohr): 1 SS, 2 SP, 3 PP(quad), 4 SD, 5 PD,
    /// 6 DD(quad), 7 PP(mono), 8 DD(mono), 9 core.
    pub po: [f64; 10],
    /// Multipole charge separations (Bohr): 2 SP, 3 PP, 4 SD, 5 PD, 6 DD.
    pub ddp: [f64; 7],
    /// Multipole coefficients `aij(2..=6)` (Thiel–Voityuk Eq. 7).
    pub aij: [f64; 7],
    /// Additive one-center energy of the partly filled d shell (eV), added to `e_isol`.
    pub eiscor: f64,
    /// True when the element carries valence d orbitals (`dorbs`).
    pub has_d: bool,
    /// One-center two-electron integral matrix `(μν|λσ) = onecenter[(ij-1)*45 + (kl-1)]`
    /// indexed by lower-triangle pair indices `ij = indx(μ,ν)`, `kl = indx(λ,σ)` (eV).
    /// Populated only for d-bearing elements (MOPAC `wstore`); empty otherwise.
    pub onecenter: Vec<f64>,
}

/// Principal quantum number of the valence s/p shell (MOPAC `iii`).
pub fn iii(z: u8) -> i32 {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 0,
    }
}

/// Principal quantum number of the valence d shell (MOPAC `iiid`).
pub fn iiid(z: u8) -> i32 {
    match z {
        1..=30 => 3,
        31..=48 => 4,
        49..=80 => 5,
        81..=86 => 6,
        _ => 0,
    }
}

/// MOPAC `main_group`: false only for the transition metals (true valence d).
pub fn main_group(z: u8) -> bool {
    !matches!(z, 21..=29 | 39..=47 | 57..=79)
}

/// Factorials `fx(i) = (i-1)!` and binomials `b(i,j) = C(i-1, j-1)` (MOPAC `fbx`).
struct FbTables {
    fx: [f64; 31],
    b: [[f64; 31]; 31],
}

impl FbTables {
    fn new() -> Self {
        let mut fx = [0.0; 31];
        fx[1] = 1.0;
        for i in 2..=30 {
            fx[i] = fx[i - 1] * (i as f64 - 1.0);
        }
        let mut b = [[0.0; 31]; 31];
        for row in b.iter_mut() {
            row[1] = 1.0;
        }
        for i in 2..=30 {
            for j in 2..=i {
                b[i][j] = b[i - 1][j - 1] + b[i - 1][j];
            }
        }
        Self { fx, b }
    }

    /// Radial part of a one-center two-electron integral (MOPAC `rsc`).
    /// `k` is the Slater–Condon multipole order (0..=4).
    #[allow(clippy::too_many_arguments)]
    fn rsc(
        &self,
        k: usize,
        na: i32,
        ea: f64,
        nb: i32,
        eb: f64,
        nc: i32,
        ec: f64,
        nd: i32,
        ed: f64,
    ) -> f64 {
        let fx = &self.fx;
        let b = &self.b;
        let aea = ea.ln();
        let aeb = eb.ln();
        let aec = ec.ln();
        let aed = ed.ln();
        let nab = (na + nb) as usize;
        let ncd = (nc + nd) as usize;
        let ecd = ec + ed;
        let eab = ea + eb;
        let e = ecd + eab;
        let n = nab + ncd;
        let ae = e.ln();
        let a2 = 2.0_f64.ln();
        let acd = ecd.ln();
        let aab = eab.ln();
        let ff = fx[n]
            / (fx[(2 * na + 1) as usize]
                * fx[(2 * nb + 1) as usize]
                * fx[(2 * nc + 1) as usize]
                * fx[(2 * nd + 1) as usize])
                .sqrt();
        let c = PM7_EV
            * ff
            * (na as f64 * aea
                + nb as f64 * aeb
                + nc as f64 * aec
                + nd as f64 * aed
                + 0.5 * (aea + aeb + aec + aed)
                + a2 * (n as f64 + 2.0)
                - ae * n as f64)
                .exp();
        let mut s0 = 1.0 / e;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let m = ncd - k;
        for i in 1..=m {
            s0 *= e / ecd;
            s1 += s0 * (b[ncd - k][i] - b[ncd + k + 1][i]) / b[n][i];
        }
        let m1 = m + 1;
        let m2 = ncd + k + 1;
        for i in m1..=m2 {
            s0 *= e / ecd;
            s2 += s0 * b[m2][i] / b[n][i];
        }
        let s3 = (ae * n as f64 - acd * m2 as f64 - aab * (nab - k) as f64).exp() / b[n][m2];
        c * (s1 - s2 + s3)
    }
}

/// The twelve radial Slater–Condon parameters (MOPAC `scprm`).
struct RadialSc {
    r066: f64,
    r266: f64,
    r466: f64,
    r016: f64,
    r244: f64,
    r036: f64,
    r236: f64,
    r155: f64,
    r355: f64,
    r125: f64,
    r234: f64,
    r246: f64,
}

fn scprm(fb: &FbTables, z: u8, zsn: f64, zpn: f64, zdn: f64) -> RadialSc {
    let ns = iii(z);
    let nd = iiid(z);
    let (es, ep, ed) = (zsn, zpn, zdn);
    RadialSc {
        r016: fb.rsc(0, ns, es, ns, es, nd, ed, nd, ed),
        r036: fb.rsc(0, ns, ep, ns, ep, nd, ed, nd, ed),
        r066: fb.rsc(0, nd, ed, nd, ed, nd, ed, nd, ed),
        r155: fb.rsc(1, ns, ep, nd, ed, ns, ep, nd, ed),
        r125: fb.rsc(1, ns, es, ns, ep, ns, ep, nd, ed),
        r244: fb.rsc(2, ns, es, nd, ed, ns, es, nd, ed),
        r236: fb.rsc(2, ns, ep, ns, ep, nd, ed, nd, ed),
        r266: fb.rsc(2, nd, ed, nd, ed, nd, ed, nd, ed),
        r234: fb.rsc(2, ns, ep, ns, ep, ns, es, nd, ed),
        r246: fb.rsc(2, ns, es, nd, ed, nd, ed, nd, ed),
        r355: fb.rsc(3, ns, ep, nd, ed, ns, ep, nd, ed),
        r466: fb.rsc(4, nd, ed, nd, ed, nd, ed, nd, ed),
    }
}

/// Assemble `repd(1..=52)` and the d-shell atomic-energy correction (MOPAC
/// `inighd` + `eiscor`). `f0sd`/`g2sd` override the computed `r016`/`r244`.
fn inighd(
    fb: &FbTables,
    z: u8,
    zsn: f64,
    zpn: f64,
    zdn: f64,
    f0sd: f64,
    g2sd: f64,
) -> ([f64; 53], f64) {
    let s3 = 1.732_050_8_f64;
    let s5 = 2.236_067_97_f64;
    let s15 = 3.872_983_34_f64;
    let mut sc = scprm(fb, z, zsn, zpn, zdn);
    if f0sd > 0.001 {
        sc.r016 = f0sd;
    }
    if g2sd > 0.001 {
        sc.r244 = g2sd;
    }
    let eiscor = eiscor(z, sc.r016, sc.r066, sc.r244, sc.r266, sc.r466);
    let RadialSc {
        r066,
        r266,
        r466,
        r016,
        r244,
        r036,
        r236,
        r155,
        r355,
        r125,
        r234,
        r246,
    } = sc;
    let mut repd = [0.0_f64; 53];
    repd[1] = r016;
    repd[2] = 2.0 / (3.0 * s5) * r125;
    repd[3] = 1.0 / s15 * r125;
    repd[4] = 2.0 / (5.0 * s5) * r234;
    repd[5] = r036 + 4.0 / 35.0 * r236;
    repd[6] = r036 + 2.0 / 35.0 * r236;
    repd[7] = r036 - 4.0 / 35.0 * r236;
    repd[8] = -1.0 / (3.0 * s5) * r125;
    repd[9] = (3.0_f64 / 125.0).sqrt() * r234;
    repd[10] = s3 / 35.0 * r236;
    repd[11] = 3.0 / 35.0 * r236;
    repd[12] = -1.0 / (5.0 * s5) * r234;
    repd[13] = r036 - 2.0 / 35.0 * r236;
    repd[14] = -2.0 * s3 / 35.0 * r236;
    repd[15] = -repd[3];
    repd[16] = -repd[11];
    repd[17] = -repd[9];
    repd[18] = -repd[14];
    repd[19] = 1.0 / 5.0 * r244;
    repd[20] = 2.0 / (7.0 * s5) * r246;
    repd[21] = repd[20] / 2.0;
    repd[22] = -repd[20];
    repd[23] = 4.0 / 15.0 * r155 + 27.0 / 245.0 * r355;
    repd[24] = 2.0 * s3 / 15.0 * r155 - 9.0 * s3 / 245.0 * r355;
    repd[25] = 1.0 / 15.0 * r155 + 18.0 / 245.0 * r355;
    repd[26] = -s3 / 15.0 * r155 + 12.0 * s3 / 245.0 * r355;
    repd[27] = -s3 / 15.0 * r155 - 3.0 * s3 / 245.0 * r355;
    repd[28] = -repd[27];
    repd[29] = r066 + 4.0 / 49.0 * r266 + 4.0 / 49.0 * r466;
    repd[30] = r066 + 2.0 / 49.0 * r266 - 24.0 / 441.0 * r466;
    repd[31] = r066 - 4.0 / 49.0 * r266 + 6.0 / 441.0 * r466;
    repd[32] = (3.0_f64 / 245.0).sqrt() * r246;
    repd[33] = 1.0 / 5.0 * r155 + 24.0 / 245.0 * r355;
    repd[34] = 1.0 / 5.0 * r155 - 6.0 / 245.0 * r355;
    repd[35] = 3.0 / 49.0 * r355;
    repd[36] = 1.0 / 49.0 * r266 + 30.0 / 441.0 * r466;
    repd[37] = s3 / 49.0 * r266 - 5.0 * s3 / 441.0 * r466;
    repd[38] = r066 - 2.0 / 49.0 * r266 - 4.0 / 441.0 * r466;
    repd[39] = -2.0 * s3 / 49.0 * r266 + 10.0 * s3 / 441.0 * r466;
    repd[40] = -repd[32];
    repd[41] = -repd[34];
    repd[42] = -repd[35];
    repd[43] = -repd[37];
    repd[44] = 3.0 / 49.0 * r266 + 20.0 / 441.0 * r466;
    repd[45] = -repd[39];
    repd[46] = 1.0 / 5.0 * r155 - 3.0 / 35.0 * r355;
    repd[47] = -repd[46];
    repd[48] = 4.0 / 49.0 * r266 + 15.0 / 441.0 * r466;
    repd[49] = 3.0 / 49.0 * r266 - 5.0 / 147.0 * r466;
    repd[50] = -repd[49];
    repd[51] = r066 + 4.0 / 49.0 * r266 - 34.0 / 441.0 * r466;
    repd[52] = 35.0 / 441.0 * r466;
    (repd, eiscor)
}

/// Partly-filled d-shell one-center atomic-energy correction (MOPAC `eiscor`).
fn eiscor(z: u8, r016: f64, r066: f64, r244: f64, r266: f64, r466: f64) -> f64 {
    // (ir016, ir066, ir244, ir266, ir466) reference-configuration occupations.
    let (a016, a066, a244, a266, a466): (i32, i32, i32, i32, i32) = match z {
        21 => (2, 0, 1, 0, 0),
        22 => (4, 1, 2, 8, 1),
        23 => (6, 3, 3, 15, 8),
        24 => (5, 10, 5, 35, 35),
        25 => (10, 10, 5, 35, 35),
        26 => (12, 15, 6, 35, 35),
        27 => (14, 21, 7, 43, 36),
        28 => (16, 28, 8, 50, 43),
        29 => (10, 45, 5, 70, 70),
        39 => (2, 0, 1, 0, 0),
        40 => (4, 1, 2, 8, 1),
        41 => (4, 6, 4, 21, 21),
        42 => (5, 10, 5, 35, 35),
        43 => (10, 10, 5, 35, 35),
        44 => (7, 21, 5, 43, 36),
        45 => (8, 28, 5, 50, 43),
        46 => (0, 45, 0, 70, 70),
        47 => (10, 45, 5, 70, 70),
        57 | 71 => (2, 0, 1, 0, 0),
        72 => (4, 1, 2, 8, 1),
        73 => (6, 3, 3, 15, 8),
        74 => (5, 10, 5, 35, 35),
        75 => (10, 10, 5, 35, 35),
        76 => (12, 15, 6, 35, 35),
        77 => (14, 21, 7, 43, 36),
        78 => (9, 36, 5, 56, 56),
        79 => (10, 45, 5, 70, 70),
        _ => (0, 0, 0, 0, 0),
    };
    a016 as f64 * r016 + a066 as f64 * r066
        - a244 as f64 * r244 / 5.0
        - a266 as f64 * r266 / 49.0
        - a466 as f64 * r466 / 49.0
}

/// Multipole coefficient `aijl` (Thiel–Voityuk Eq. 7).
fn aijl(fb: &FbTables, z1: f64, z2: f64, n1: i32, n2: i32, l: i32) -> f64 {
    let zz = z1 + z2 + 1.0e-20;
    fb.fx[(n1 + n2 + l + 1) as usize]
        / (fb.fx[(2 * n1 + 1) as usize] * fb.fx[(2 * n2 + 1) as usize]).sqrt()
        * (2.0 * z1 / zz).powi(n1)
        * (2.0 * z1 / zz).sqrt()
        * (2.0 * z2 / zz).powi(n2)
        * (2.0 * z2 / zz).sqrt()
        * 2.0_f64.powi(l)
        / zz.powi(l)
}

/// `aij(2..=6)` for an element (MOPAC `aijm`). Uses the valence Slater exponents.
fn aijm(fb: &FbTables, z: u8, zs: f64, zp: f64, zd: f64, has_d: bool) -> [f64; 7] {
    let mut aij = [0.0_f64; 7];
    if z < 3 {
        return aij;
    }
    if zs * zp < 0.01 {
        return aij;
    }
    let nsp = iii(z);
    aij[2] = aijl(fb, zs, zp, nsp, nsp, 1);
    aij[3] = aijl(fb, zp, zp, nsp, nsp, 2);
    if has_d {
        let nd = iiid(z);
        aij[4] = aijl(fb, zs, zd, nsp, nd, 2);
        aij[5] = aijl(fb, zp, zd, nsp, nd, 1);
        aij[6] = aijl(fb, zd, zd, nd, nd, 2);
    }
    aij
}

/// Klopman–Ohno additive-term root finder (MOPAC `poij`, golden-section search).
fn poij(l: i32, d: f64, fg: f64) -> f64 {
    const EPS: f64 = 1.0e-8;
    const G1: f64 = 0.382;
    const G2: f64 = 0.618;
    if l == 0 {
        return 0.5 * PM7_EV / fg;
    }
    let dsq = d * d;
    let ev4 = PM7_EV * 0.25;
    let ev8 = PM7_EV / 8.0;
    let mut a1 = 0.1;
    let mut a2 = 5.0;
    let (mut f1, mut f2) = (0.0, 0.0);
    for _ in 0..100 {
        let delta = a2 - a1;
        if delta < EPS {
            break;
        }
        let y1 = a1 + delta * G1;
        let y2 = a1 + delta * G2;
        if l == 1 {
            f1 = (ev4 * (1.0 / y1 - 1.0 / (y1 * y1 + dsq).sqrt()) - fg).powi(2);
            f2 = (ev4 * (1.0 / y2 - 1.0 / (y2 * y2 + dsq).sqrt()) - fg).powi(2);
        } else {
            f1 = (ev8
                * (1.0 / y1 - 2.0 / (y1 * y1 + dsq * 0.5).sqrt() + 1.0 / (y1 * y1 + dsq).sqrt())
                - fg)
                .powi(2);
            f2 = (ev8
                * (1.0 / y2 - 2.0 / (y2 * y2 + dsq * 0.5).sqrt() + 1.0 / (y2 * y2 + dsq).sqrt())
                - fg)
                .powi(2);
        }
        if f1 < f2 {
            a2 = y2;
        } else {
            a1 = y1;
        }
    }
    if f1 >= f2 {
        a2
    } else {
        a1
    }
}

/// Inputs required to build the one-center d machinery for one element.
pub struct DShellInput {
    pub z: u8,
    pub zsn: f64,
    pub zpn: f64,
    pub zdn: f64,
    pub zeta_s: f64,
    pub zeta_p: f64,
    pub zeta_d: f64,
    pub f0sd: f64,
    pub g2sd: f64,
    pub g_ss: f64,
    pub g_sp: f64,
    pub h_sp: f64,
    pub g_pp: f64,
    pub g_p2: f64,
    pub poc: f64,
    /// Validated sp additive terms (Bohr) reused for main-group `po(1..3)`.
    pub rho0: f64,
    pub rho1: f64,
    pub rho2: f64,
    /// sp charge separations (Bohr) reused for main-group `ddp(2..3)`.
    pub dd: f64,
    pub qq: f64,
}

/// Derive the one-center MNDO/d parameters for a single element (MOPAC `inid`
/// applied to one atom: `aijm` → `inighd` → `ddpo` → main-group override).
pub fn derive_dshell(input: &DShellInput) -> DShell {
    let fb = FbTables::new();
    let has_d = input.zeta_d > 1.0e-20;
    let aij = aijm(
        &fb,
        input.z,
        input.zeta_s,
        input.zeta_p,
        input.zeta_d,
        has_d,
    );

    let (repd, eiscor_energy) = if has_d && input.zdn > 1.0e-4 {
        inighd(
            &fb, input.z, input.zsn, input.zpn, input.zdn, input.f0sd, input.g2sd,
        )
    } else {
        ([0.0_f64; 53], 0.0)
    };

    let mut po = [0.0_f64; 10];
    let mut ddp = [0.0_f64; 7];

    // MOPAC ddpo: SS monopole.
    if input.g_ss > 0.1 {
        po[1] = poij(0, 1.0, input.g_ss);
    }
    if input.z >= 3 {
        // SP dipole.
        let d = aij[2] / 12.0_f64.sqrt();
        ddp[2] = d;
        po[2] = poij(1, d, input.h_sp);
        // PP monopole/quadrupole.
        po[7] = po[1];
        let d = (aij[3] * 0.1).sqrt();
        ddp[3] = d;
        po[3] = poij(2, d, 0.5 * (input.g_pp - input.g_p2));
        if has_d {
            // SD quadrupole.
            let da = (1.0_f64 / 60.0).sqrt();
            let d = (aij[4] * da).sqrt();
            ddp[4] = d;
            po[4] = poij(2, d, repd[19]);
            // PD dipole.
            let d = aij[5] / 20.0_f64.sqrt();
            ddp[5] = d;
            po[5] = poij(1, d, repd[23] - 1.8 * repd[35]);
            // DD monopole.
            let fg = 0.2 * (repd[29] + 2.0 * repd[30] + 2.0 * repd[31]);
            po[8] = if fg > 1.0e-5 { poij(0, 1.0, fg) } else { 1.0e5 };
            // DD quadrupole.
            let d = (aij[6] / 14.0).sqrt();
            ddp[6] = d;
            po[6] = poij(2, d, repd[44] - (20.0 / 35.0) * repd[52]);
        }
    }

    // MOPAC inid: reuse the validated sp additive terms for main-group elements
    // (and for every sp-only element). Transition metals keep the ddpo values.
    if main_group(input.z) || !has_d {
        po[1] = input.rho0;
        if input.rho1 > 1.0e-5 {
            po[2] = input.rho1;
        }
        if input.rho2 > 1.0e-5 {
            po[3] = input.rho2;
        }
        po[7] = po[1];
        ddp[2] = input.dd;
        ddp[3] = input.qq * 2.0_f64.sqrt();
    }
    // Core additive term.
    po[9] = if input.poc > 1.0e-5 { input.poc } else { po[1] };

    let onecenter = if has_d {
        build_onecenter(input, &repd)
    } else {
        Vec::new()
    };

    DShell {
        repd,
        po,
        ddp,
        aij,
        eiscor: eiscor_energy,
        has_d,
        onecenter,
    }
}

/// Build the 45×45 one-center two-electron integral matrix for a d element
/// (MOPAC `wstore`): sp block from gss/gsp/gpp/gp2/hsp, d block from `repd`
/// via the INTIJ/INTKL/INTREP maps. Flat, 0-based: `w[(ij-1)*45 + (kl-1)]`.
fn build_onecenter(input: &DShellInput, repd: &[f64; 53]) -> Vec<f64> {
    use crate::mndod_tables::{INTIJ, INTKL, INTREP};
    let mut w = vec![0.0_f64; 45 * 45];
    let set = |w: &mut [f64], ij: usize, kl: usize, v: f64| {
        w[(ij - 1) * 45 + (kl - 1)] = v;
    };
    let (gss, gsp, gpp, gp2, hsp) = (input.g_ss, input.g_sp, input.g_pp, input.g_p2, input.h_sp);
    set(&mut w, 1, 1, gss);
    // sp block: ip=1, ipx=3, ipy=6, ipz=10 (indx of the diagonal p pairs).
    for &ip_diag in &[3usize, 6, 10] {
        set(&mut w, ip_diag, 1, gsp);
        set(&mut w, 1, ip_diag, gsp);
        set(&mut w, ip_diag, ip_diag, gpp);
    }
    for &(a, b) in &[(6, 3), (10, 3), (10, 6), (3, 6), (3, 10), (6, 10)] {
        set(&mut w, a, b, gp2);
    }
    for &p in &[2usize, 4, 7] {
        set(&mut w, p, p, hsp);
    }
    for &p in &[5usize, 8, 9] {
        set(&mut w, p, p, 0.5 * (gpp - gp2));
    }
    // d block.
    for i in 0..243 {
        let ij = INTIJ[i] as usize;
        let kl = INTKL[i] as usize;
        let int = INTREP[i] as usize;
        set(&mut w, ij, kl, repd[int]);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorials_and_binomials() {
        let fb = FbTables::new();
        assert_eq!(fb.fx[1], 1.0); // 0!
        assert_eq!(fb.fx[2], 1.0); // 1!
        assert_eq!(fb.fx[5], 24.0); // 4!
        assert_eq!(fb.b[5][2], 4.0); // C(4,1)
        assert_eq!(fb.b[5][3], 6.0); // C(4,2)
    }

    #[test]
    fn main_group_matches_mopac() {
        assert!(main_group(16)); // sulfur: main-group with d polarization
        assert!(main_group(30)); // zinc: main-group
        assert!(!main_group(26)); // iron: transition metal
        assert!(!main_group(46)); // palladium
        assert!(main_group(35)); // bromine
    }

    #[test]
    fn one_center_dshell_main_group() {
        // Sulfur is a main-group hypervalent element: it carries d polarization
        // functions but has f0sd = g2sd = 0, so repd is computed via rsc.
        let params = crate::params::Pm7Parameters::standard().unwrap();
        let s = params.element(16).unwrap();
        assert_eq!(s.n_orb, 9, "PM7 sulfur must carry d orbitals");
        assert!(main_group(16));
        let dshell = s.dshell.as_ref().expect("d element must have a DShell");
        // Main-group po(1..3) reproduce the validated sp additive terms.
        assert!((dshell.po[1] - s.rho0).abs() < 1.0e-12);
        assert!((dshell.po[2] - s.rho1).abs() < 1.0e-12);
        assert!((dshell.po[3] - s.rho2).abs() < 1.0e-12);
        assert!((dshell.ddp[2] - s.dd).abs() < 1.0e-12);
        // The one-center d integrals from rsc must be finite and O(eV).
        assert!(dshell.repd[1..=52].iter().all(|v| v.is_finite()));
        assert!(
            dshell.repd[1] > 1.0 && dshell.repd[1] < 50.0,
            "r016={}",
            dshell.repd[1]
        );
        assert!(dshell.po[1..=9].iter().all(|v| v.is_finite() && *v >= 0.0));
        // Main-group elements have no partly-filled-d atomic-energy correction.
        assert_eq!(dshell.eiscor, 0.0);
    }

    #[test]
    fn charg_matches_sp_multipole_closed_forms() {
        // charg must reproduce the Dewar–Sabelli–Klopman monopole/dipole terms
        // used by the (validated) sp two-center kernel in integrals::local_xx_g.
        let (r, da, db, add) = (2.7_f64, 0.45_f64, 0.38_f64, 1.9_f64);
        // Q-Q.
        let qq = charg::<f64>(r, 0, 0, 0, da, db, add);
        assert!((qq - 1.0 / (r * r + add).sqrt()).abs() < 1.0e-13);
        // Z-Q (dipole on A). Matches ev1*(1/sqrt((r-da)^2+add) - 1/sqrt((r+da)^2+add))/ev1.
        let zq = charg::<f64>(r, 1, 0, 0, da, db, add);
        let ref_zq = 0.5
            * (1.0 / ((r - da) * (r - da) + add).sqrt() - 1.0 / ((r + da) * (r + da) + add).sqrt());
        assert!((zq - ref_zq).abs() < 1.0e-13);
        // X-X (perpendicular dipole–dipole).
        let xx = charg::<f64>(r, 1, 1, 1, da, db, add);
        let ref_xx = 0.5
            * (1.0 / (r * r + (da - db) * (da - db) + add).sqrt()
                - 1.0 / (r * r + (da + db) * (da + db) + add).sqrt());
        assert!((xx - ref_xx).abs() < 1.0e-13);
    }

    #[test]
    fn charg_dual_derivative_matches_fd() {
        // The generic Scalar path must differentiate exactly — this is what makes
        // the d-orbital analytic gradient/Hessian free.
        use crate::dual::Dual;
        let (da, db, add) = (0.5_f64, 0.4_f64, 1.6_f64);
        let r0 = 2.3_f64;
        let h = 1.0e-6;
        for &(l1, l2, m) in &[
            (0u8, 0u8, 0u8),
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (2, 0, 0),
            (1, 2, 0),
            (2, 1, 0),
            (2, 2, 0),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 1),
            (2, 2, 2),
        ] {
            let d = charg::<Dual>(Dual::var(r0, 0), l1, l2, m, da, db, add);
            let fp = charg::<f64>(r0 + h, l1, l2, m, da, db, add);
            let fm = charg::<f64>(r0 - h, l1, l2, m, da, db, add);
            let fd = (fp - fm) / (2.0 * h);
            assert!(
                (d.d[0] - fd).abs() < 1.0e-6,
                "charg d/dr mismatch for ({l1},{l2},{m}): dual={} fd={}",
                d.d[0],
                fd
            );
        }
    }

    #[test]
    fn onecenter_sp_block_matches_oc_two_electron() {
        // The sp part of the d element's 45×45 one-center matrix must reproduce the
        // validated sp one-center integrals (fock::oc_two_electron) for all sp pairs.
        let params = crate::params::Pm7Parameters::standard().unwrap();
        let s = params.element(16).unwrap();
        let d = s.dshell.as_ref().unwrap();
        let pk = |a: usize, b: usize| {
            let (h, l) = if a >= b { (a, b) } else { (b, a) };
            h * (h + 1) / 2 + l
        };
        let (gss, gsp, gpp, gp2, hsp) = (s.g_ss, s.g_sp, s.g_pp, s.g_p2, s.h_sp);
        let mut maxerr = 0.0_f64;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for e in 0..4 {
                        let oc = crate::fock::oc_two_electron(a, b, c, e, gss, gsp, gpp, gp2, hsp);
                        let got = d.onecenter[pk(a, b) * 45 + pk(c, e)];
                        maxerr = maxerr.max((oc - got).abs());
                    }
                }
            }
        }
        assert!(maxerr < 1.0e-9, "onecenter sp block mismatch {maxerr:.3e}");
    }

    #[test]
    fn one_center_dshell_transition_metal() {
        // Iron supplies f0sd/g2sd explicitly, so repd(1) = f0sd, repd(19) = g2sd/5.
        let params = crate::params::Pm7Parameters::standard().unwrap();
        let fe = params.element(26).unwrap();
        assert_eq!(fe.n_orb, 9);
        assert!(!main_group(26));
        let dshell = fe.dshell.as_ref().expect("Fe must have a DShell");
        assert!(
            (dshell.repd[1] - fe.f0sd).abs() < 1.0e-9,
            "repd1={}",
            dshell.repd[1]
        );
        assert!((dshell.repd[19] - fe.g2sd / 5.0).abs() < 1.0e-9);
        // Transition-metal po(1) is still the SS monopole additive term.
        assert!((dshell.po[1] - fe.rho0).abs() < 1.0e-9);
        // The partly-filled d shell contributes a nonzero isolated-atom correction.
        assert!(dshell.eiscor.abs() > 1.0);
        assert!(dshell.repd[1..=52].iter().all(|v| v.is_finite()));
    }
}
