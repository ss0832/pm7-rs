// SPDX-License-Identifier: GPL-3.0-or-later

//! PM7 element, pair, and derived NDDO multipole parameters.
//!
//! The embedded CSV tables are generated directly from MOPAC v23.2.5.  This
//! module performs only unit-preserving parsing and the `calpar.F90` derived
//! quantities needed by the shared NDDO kernel; it never substitutes fitted
//! values or silently falls back to another semiempirical model.

use crate::constants::{KCAL_TO_EV, PM7_EV};
use crate::data_tables;
use crate::error::{Pm7Error, Result};
use crate::method::Pm7Method;
use std::collections::HashMap;

/// Per-element PM7 parameters.  Energies are eV; exponents and core-core
/// parameters retain MOPAC's native units.
#[derive(Clone, Debug)]
pub struct Pm7Element {
    pub z: u8,
    pub n: u8,
    pub ios: u8,
    pub iop: u8,
    pub iod: u8,
    pub u_ss: f64,
    pub u_pp: f64,
    pub u_dd: f64,
    pub zeta_s: f64,
    pub zeta_p: f64,
    pub zeta_d: f64,
    pub beta_s: f64,
    pub beta_p: f64,
    pub beta_d: f64,
    pub g_ss: f64,
    pub g_sp: f64,
    pub g_pp: f64,
    pub g_p2: f64,
    pub h_sp: f64,
    pub zsn: f64,
    pub zpn: f64,
    pub zdn: f64,
    pub f0sd: f64,
    pub g2sd: f64,
    pub alpha: f64,
    pub poc: f64,
    pub polvol: f64,
    /// MOPAC `guess1/guess2/guess3` core Gaussian triples `(K, L, M)`.
    pub gauss: Vec<(f64, f64, f64)>,
    /// Number of valence AOs: 0 (Sparkle), 1 (s), 4 (sp), or 9 (spd).
    pub n_orb: usize,
    pub core_charge: f64,
    pub eheat_ev: f64,
    pub e_isol: f64,
    /// Derived NDDO charge separations (Bohr) and Klopman-Ohno addends.
    pub dd: f64,
    pub qq: f64,
    pub rho0: f64,
    pub rho1: f64,
    pub rho2: f64,
    pub is_sparkle: bool,
    /// One-center MNDO/d parameters, present only for d-bearing elements (`n_orb == 9`).
    pub dshell: Option<crate::mndod::DShell>,
}

impl Pm7Element {
    pub fn has_p(&self) -> bool {
        self.n_orb >= 4
    }

    pub fn has_d(&self) -> bool {
        self.n_orb == 9
    }

    /// MOPAC `po(9)`: the Klopman–Ohno monopole radius (Bohr) used for the **core**
    /// (core–core repulsion and electron–core attraction). PM7 overrides the default
    /// `0.5/am = ev/(2·gss) = rho0` with the element's `poc` (MOPAC `pocord`) when
    /// defined; the electron–electron two-centre integrals keep using `rho0`.
    /// See MOPAC `mndod.F90:46-54, 503-505` and `calpar.F90`.
    #[inline]
    pub fn po9(&self) -> f64 {
        if self.poc > 1.0e-5 {
            self.poc
        } else {
            self.rho0
        }
    }
}

/// PM7 pair-specific core-core scaling parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pm7Pair {
    pub alpb: f64,
    pub xfac: f64,
}

#[derive(Clone, Debug)]
pub struct Pm7Parameters {
    pub method: Pm7Method,
    pub elements: HashMap<u8, Pm7Element>,
    pub pairs: HashMap<(u8, u8), Pm7Pair>,
    /// MOPAC's one-based global PM7 correction table; index 0 maps to Fortran 1.
    pub vpar: [f64; 60],
}

impl Pm7Parameters {
    /// Load the standard PM7 model from the embedded MOPAC-derived tables.
    pub fn standard() -> Result<Self> {
        Self::method(Pm7Method::Pm7)
    }

    /// Load a PM7 family method while preserving the common NDDO API.
    pub fn method(method: Pm7Method) -> Result<Self> {
        let (elements, pairs, vpar) = match method {
            Pm7Method::Pm7Ts => (
                data_tables::PM7_TS_ELEMENTS_CSV,
                data_tables::PM7_TS_PAIRS_CSV,
                data_tables::PM7_TS_VPAR_CSV,
            ),
            _ => (
                data_tables::PM7_ELEMENTS_CSV,
                data_tables::PM7_PAIRS_CSV,
                data_tables::PM7_VPAR_CSV,
            ),
        };
        let mut parsed = Self::from_tables(elements, pairs, vpar, method)?;
        if method.uses_sparkles() {
            parsed.install_sparkles(data_tables::PM7_SPARKLES_CSV)?;
        }
        Ok(parsed)
    }

    /// Parse an element-only table for focused unit tests or downstream tools.
    /// Pair and global parameters default to empty/zero and are not suitable for SCF.
    pub fn from_csv(text: &str) -> Result<Self> {
        Self::from_tables(
            text,
            "z_hi,z_lo,alpb,xfac\n",
            "index,value\n",
            Pm7Method::Pm7,
        )
    }

    pub fn element(&self, z: u8) -> Result<&Pm7Element> {
        self.elements.get(&z).ok_or(Pm7Error::MissingElement(z))
    }

    /// Return an explicit pair entry or MOPAC's documented diagonal-average
    /// fallback for a missing PM7 pair.
    pub fn pair(&self, zi: u8, zj: u8) -> Pm7Pair {
        let key = ordered_pair(zi, zj);
        if let Some(pair) = self.pairs.get(&key) {
            return *pair;
        }
        let ii = self
            .pairs
            .get(&ordered_pair(zi, zi))
            .copied()
            .unwrap_or(Pm7Pair {
                alpb: 0.0,
                xfac: 0.0,
            });
        let jj = self
            .pairs
            .get(&ordered_pair(zj, zj))
            .copied()
            .unwrap_or(Pm7Pair {
                alpb: 0.0,
                xfac: 0.0,
            });
        Pm7Pair {
            alpb: 0.5 * (ii.alpb + jj.alpb),
            xfac: 0.5 * (ii.xfac + jj.xfac),
        }
    }

    /// Access the one-based MOPAC global correction parameter `v_par(index)`.
    pub fn vpar(&self, index: usize) -> f64 {
        self.vpar
            .get(index.saturating_sub(1))
            .copied()
            .unwrap_or(0.0)
    }

    fn from_tables(
        elements_text: &str,
        pairs_text: &str,
        vpar_text: &str,
        method: Pm7Method,
    ) -> Result<Self> {
        let mut elements = HashMap::new();
        let (header, rows) = csv_rows(elements_text)?;
        let col = |name: &str| column(&header, name);
        for row in rows {
            let z = get_u8(&row, col("z")?, "z")?;
            let zeta_s = get_f64(&row, col("zeta_s")?, "zeta_s")?;
            let beta_s = get_f64(&row, col("beta_s")?, "beta_s")?;
            if zeta_s == 0.0 && beta_s == 0.0 {
                continue;
            }
            let value_of = |name: &str| -> Result<f64> { get_f64(&row, col(name)?, name) };
            let ios = get_u8(&row, col("ios")?, "ios")?;
            let iop = get_u8(&row, col("iop")?, "iop")?;
            let iod = get_u8(&row, col("iod")?, "iod")?;
            let zeta_p = value_of("zeta_p")?;
            let zeta_d = value_of("zeta_d")?;
            let n_orb = if zeta_d > 1.0e-20 {
                9
            } else if zeta_p > 1.0e-20 {
                4
            } else {
                1
            };
            let g_ss = value_of("g_ss")?;
            let g_pp = value_of("g_pp")?;
            let g_p2 = value_of("g_p2")?;
            let h_sp = value_of("h_sp")?;
            let n = get_u8(&row, col("n")?, "n")?;
            let (dd, qq, rho1, rho2) = if n_orb >= 4 {
                let dd = dd_charge_sep(n, zeta_s, zeta_p.max(0.3));
                let qq = qq_charge_sep(n, zeta_p.max(0.3));
                let rho1 = additive_rho1(h_sp.max(1.0e-7), dd);
                let rho2 = additive_rho2((0.5 * (g_pp - g_p2)).max(0.1), qq);
                (dd, qq, rho1, rho2)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            let mut gauss = Vec::new();
            for slot in 1..=4 {
                let k = value_of(&format!("g1_{slot}"))?;
                let l = value_of(&format!("g2_{slot}"))?;
                let m = value_of(&format!("g3_{slot}"))?;
                if k != 0.0 || l != 0.0 || m != 0.0 {
                    gauss.push((k, l, m));
                }
            }
            let gssc = f64::from(ios.saturating_sub(1));
            let p = f64::from(iop);
            let l = p.min(6.0 - p);
            let gspc = f64::from(ios) * p;
            let gp2c = p * (p - 1.0) / 2.0 + 0.5 * l * (l - 1.0) / 2.0;
            let gppc = -0.5 * l * (l - 1.0) / 2.0;
            let hspc = -p * f64::from(ios) * 0.5;
            let u_ss = value_of("u_ss")?;
            let u_pp = value_of("u_pp")?;
            let u_dd = value_of("u_dd")?;
            let mut e_isol = u_ss * f64::from(ios)
                + u_pp * p
                + u_dd * f64::from(iod)
                + g_ss * gssc
                + g_pp * gppc
                + value_of("g_sp")? * gspc
                + g_p2 * gp2c
                + h_sp * hspc;
            let g_sp = value_of("g_sp")?;
            let beta_p = value_of("beta_p")?;
            let beta_d = value_of("beta_d")?;
            let zsn = value_of("zsn")?;
            let zpn = value_of("zpn")?;
            let zdn = value_of("zdn")?;
            let f0sd = value_of("f0sd")?;
            let g2sd = value_of("g2sd")?;
            let poc = value_of("poc")?;
            let rho0 = if g_ss != 0.0 {
                PM7_EV / (2.0 * g_ss)
            } else {
                0.0
            };
            // One-center MNDO/d parameters. Computed for every element so the
            // MNDO/d two-center kernel has the `po`/`ddp` additive terms of both
            // atoms in any pair (sp-only atoms take the main-group override path).
            // The d-shell atomic-energy correction (`eiscor`) folds into e_isol.
            let dshell = {
                let d = crate::mndod::derive_dshell(&crate::mndod::DShellInput {
                    z,
                    zsn,
                    zpn,
                    zdn,
                    zeta_s,
                    zeta_p,
                    zeta_d,
                    f0sd,
                    g2sd,
                    g_ss,
                    g_sp,
                    h_sp,
                    g_pp,
                    g_p2,
                    poc,
                    rho0,
                    rho1,
                    rho2,
                    dd,
                    qq,
                });
                e_isol += d.eiscor;
                Some(d)
            };
            elements.insert(
                z,
                Pm7Element {
                    z,
                    n,
                    ios,
                    iop,
                    iod,
                    u_ss,
                    u_pp,
                    u_dd,
                    zeta_s,
                    zeta_p,
                    zeta_d,
                    beta_s,
                    beta_p,
                    beta_d,
                    g_ss,
                    g_sp,
                    g_pp,
                    g_p2,
                    h_sp,
                    zsn,
                    zpn,
                    zdn,
                    f0sd,
                    g2sd,
                    alpha: value_of("alpha")?,
                    poc,
                    polvol: value_of("polvol")?,
                    gauss,
                    n_orb,
                    core_charge: value_of("core_charge")?,
                    eheat_ev: value_of("eheat_kcal")? * KCAL_TO_EV,
                    e_isol,
                    dd,
                    qq,
                    rho0,
                    rho1,
                    rho2,
                    is_sparkle: false,
                    dshell,
                },
            );
        }
        if elements.is_empty() {
            return Err(Pm7Error::InvalidInput(
                "no PM7 elements parsed from parameter table".into(),
            ));
        }

        let mut pairs = HashMap::new();
        let (pair_header, pair_rows) = csv_rows(pairs_text)?;
        if !pair_header.is_empty() {
            let hi = column(&pair_header, "z_hi")?;
            let lo = column(&pair_header, "z_lo")?;
            let alpb = column(&pair_header, "alpb")?;
            let xfac = column(&pair_header, "xfac")?;
            for row in pair_rows {
                let zi = get_u8(&row, hi, "z_hi")?;
                let zj = get_u8(&row, lo, "z_lo")?;
                pairs.insert(
                    ordered_pair(zi, zj),
                    Pm7Pair {
                        alpb: get_f64(&row, alpb, "alpb")?,
                        xfac: get_f64(&row, xfac, "xfac")?,
                    },
                );
            }
        }

        let mut vpar = [0.0; 60];
        let (vpar_header, vpar_rows) = csv_rows(vpar_text)?;
        if !vpar_header.is_empty() {
            let index = column(&vpar_header, "index")?;
            let value = column(&vpar_header, "value")?;
            for row in vpar_rows {
                let idx = get_u8(&row, index, "index")? as usize;
                if (1..=60).contains(&idx) {
                    vpar[idx - 1] = get_f64(&row, value, "value")?;
                }
            }
        }
        Ok(Self {
            method,
            elements,
            pairs,
            vpar,
        })
    }

    fn install_sparkles(&mut self, text: &str) -> Result<()> {
        // MOPAC `eheat_sparkles` (kcal/mol) — the Ln(III) reference heats used for
        // the sparkle heat of formation (parameters_C.F90).
        const EHEAT_SPARKLE: [(u8, f64); 15] = [
            (57, 928.9),
            (58, 944.7),
            (59, 952.9),
            (60, 962.8),
            (61, 976.9),
            (62, 974.4),
            (63, 1006.6),
            (64, 991.37),
            (65, 999.0),
            (66, 1001.3),
            (67, 1009.6),
            (68, 1016.15),
            (69, 1022.06),
            (70, 1039.03),
            (71, 1031.2),
        ];
        let (header, rows) = csv_rows(text)?;
        for row in rows {
            let z = get_u8(&row, column(&header, "z")?, "z")?;
            if !(58..=70).contains(&z) {
                continue;
            }
            let eheat_kcal = EHEAT_SPARKLE
                .iter()
                .find(|(zz, _)| *zz == z)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            let alpha = get_f64(&row, column(&header, "alpha")?, "alpha")?;
            let g_ss = get_f64(&row, column(&header, "g_ss")?, "g_ss")?;
            let mut gauss = Vec::new();
            for slot in 1..=2 {
                gauss.push((
                    get_f64(&row, column(&header, &format!("g1_{slot}"))?, "g1")?,
                    get_f64(&row, column(&header, &format!("g2_{slot}"))?, "g2")?,
                    get_f64(&row, column(&header, &format!("g3_{slot}"))?, "g3")?,
                ));
            }
            self.elements.insert(
                z,
                Pm7Element {
                    z,
                    n: 6,
                    ios: 0,
                    iop: 0,
                    iod: 0,
                    n_orb: 0,
                    core_charge: 3.0,
                    u_ss: 0.0,
                    u_pp: 0.0,
                    u_dd: 0.0,
                    zeta_s: 0.0,
                    zeta_p: 0.0,
                    zeta_d: 0.0,
                    beta_s: 0.0,
                    beta_p: 0.0,
                    beta_d: 0.0,
                    g_ss,
                    g_sp: 0.0,
                    g_pp: 0.0,
                    g_p2: 0.0,
                    h_sp: 0.0,
                    zsn: 0.0,
                    zpn: 0.0,
                    zdn: 0.0,
                    f0sd: 0.0,
                    g2sd: 0.0,
                    alpha,
                    poc: 0.0,
                    polvol: 0.0,
                    gauss,
                    rho0: PM7_EV / (2.0 * g_ss),
                    dd: 0.0,
                    qq: 0.0,
                    rho1: 0.0,
                    rho2: 0.0,
                    e_isol: 0.0,
                    eheat_ev: eheat_kcal * KCAL_TO_EV,
                    is_sparkle: true,
                    dshell: None,
                },
            );
        }
        // MOPAC `switch.F90` (PM7 branch, lines 472-475): after the sparkle
        // parameters are installed, the whole `alpb`/`xfac` rows *and* columns for
        // the sparkle range are zeroed, so a lanthanide never carries the diatomic
        // core-core scaling it would have as a full PM7 atom.  We drop those pair
        // entries (rather than store zeros) so `pair` re-completes them as
        // `0.5*(0 + X-X)` — exactly what MOPAC ccrep does with a zeroed pair.
        // Otherwise the real Gd-F (64,9) / Gd-Gd (64,64) entries survive and make
        // GdF3 core-core ~0.14 eV/bond too repulsive (a Gd-specific +0.57 eV / +13
        // kcal/mol error absent from the other lanthanides, which lack such entries).
        self.pairs
            .retain(|&(a, b), _| !(58..=70).contains(&a) && !(58..=70).contains(&b));
        Ok(())
    }
}

fn ordered_pair(zi: u8, zj: u8) -> (u8, u8) {
    (zi.max(zj), zi.min(zj))
}

fn csv_rows(text: &str) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut lines = text.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let Some(header) = lines.next() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let header = header
        .split(',')
        .map(|field| field.trim().to_string())
        .collect();
    let rows = lines
        .map(|line| {
            line.split(',')
                .map(|field| field.trim().to_string())
                .collect()
        })
        .collect();
    Ok((header, rows))
}

fn column(header: &[String], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|field| field == name)
        .ok_or_else(|| Pm7Error::MissingParameter(name.into()))
}

fn get_f64(row: &[String], index: usize, name: &str) -> Result<f64> {
    row.get(index)
        .ok_or_else(|| Pm7Error::MissingParameter(name.into()))?
        .parse::<f64>()
        .map_err(|_| Pm7Error::InvalidInput(format!("invalid {name} value")))
}

fn get_u8(row: &[String], index: usize, name: &str) -> Result<u8> {
    row.get(index)
        .ok_or_else(|| Pm7Error::MissingParameter(name.into()))?
        .parse::<u8>()
        .map_err(|_| Pm7Error::InvalidInput(format!("invalid {name} value")))
}

/// MOPAC `ddpo.F90`: s-p dipole charge separation in Bohr.
pub fn dd_charge_sep(n: u8, zs: f64, zp: f64) -> f64 {
    let n = f64::from(n);
    (2.0 * n + 1.0) * (4.0 * zs * zp).powf(n + 0.5) / (zs + zp).powf(2.0 * n + 2.0) / 3.0_f64.sqrt()
}

/// MOPAC `ddpo.F90`: p-p quadrupole charge separation in Bohr.
pub fn qq_charge_sep(n: u8, zp: f64) -> f64 {
    let n = f64::from(n);
    ((4.0 * n * n + 6.0 * n + 2.0) / 20.0).sqrt() / zp
}

/// MOPAC `calpar.F90` secant for an additive (dipole/quadrupole) Klopman term. MOPAC runs a
/// **fixed 5-step** secant (`jmax = 5`) from `guess`, exits early only when the two function
/// values coincide to 1e-25, and takes the **last iterate** — it does NOT iterate to convergence.
/// Replicating the exact step count is essential: for a target integral with no real root (e.g.
/// boron's negative `hpp`, where `hpp = ½(gpp − gp2) < 0`), extra steps drift the diverging secant
/// to a different value, which is precisely the derived `rho2` MOPAC uses. `f` solves `f(x) = target`.
fn calpar_secant(target_au: f64, guess: f64, f: impl Fn(f64) -> f64) -> f64 {
    let mut d1 = guess;
    let mut d2 = guess + 0.04;
    for _ in 0..5 {
        let (f1, f2) = (f(d1), f(d2));
        if (f2 - f1).abs() < 1.0e-25 {
            break;
        }
        let d3 = d1 + (d2 - d1) * (target_au - f1) / (f2 - f1);
        d1 = d2;
        d2 = d3;
    }
    d2
}

pub fn additive_rho1(hsp_ev: f64, dd: f64) -> f64 {
    let hsp = hsp_ev / PM7_EV;
    let f = |d: f64| 0.5 * d - 0.5 / (4.0 * dd * dd + 1.0 / (d * d)).sqrt();
    let guess = (hsp.abs() / (dd * dd)).powf(1.0 / 3.0) * hsp.signum();
    0.5 / calpar_secant(hsp, guess, f)
}

pub fn additive_rho2(hpp_ev: f64, qq: f64) -> f64 {
    let hpp = hpp_ev / PM7_EV;
    let f = |q: f64| {
        0.25 * q - 0.5 / (4.0 * qq * qq + 1.0 / (q * q)).sqrt()
            + 0.25 / (8.0 * qq * qq + 1.0 / (q * q)).sqrt()
    };
    let guess = (hpp.abs() / (3.0 * qq.powi(4))).powf(0.2) * hpp.signum();
    0.5 / calpar_secant(hpp, guess, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_table_matches_mopac_sp_references() {
        let parameters = Pm7Parameters::standard().unwrap();
        let carbon = parameters.element(6).unwrap();
        assert!((carbon.u_ss + 51.372620).abs() < 1.0e-9);
        assert!((carbon.g_ss - 12.347323).abs() < 1.0e-9);
        assert_eq!(carbon.n_orb, 4);
        assert_eq!(carbon.gauss.len(), 1);
        assert!((parameters.pair(6, 1).alpb - 1.038716).abs() < 1.0e-9);
        assert!((parameters.pair(6, 1).xfac - 0.204582).abs() < 1.0e-9);
        assert!((parameters.vpar(1) - 8.947612).abs() < 1.0e-9);
    }

    #[test]
    fn method_tables_are_distinct_and_sparkles_are_explicit() {
        let standard = Pm7Parameters::standard().unwrap();
        let ts = Pm7Parameters::method(Pm7Method::Pm7Ts).unwrap();
        assert!((standard.element(1).unwrap().u_ss - ts.element(1).unwrap().u_ss).abs() > 1.0e-4);
        let sparkle = Pm7Parameters::method(Pm7Method::Pm7Sparkle).unwrap();
        let europium = sparkle.element(63).unwrap();
        assert!(europium.is_sparkle);
        assert_eq!(europium.n_orb, 0);
        assert_eq!(europium.core_charge, 3.0);
        assert_eq!(europium.gauss.len(), 2);
    }
}
