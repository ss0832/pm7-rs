// SPDX-License-Identifier: GPL-3.0-or-later

//! Wavefunction output in **Molden** format.
//!
//! PM7's valence basis is Slater-type: one `s` shell, a `p` set for everything past helium, and a
//! `d` set for the heavier main-group and transition elements, with exponents `zeta_s`, `zeta_p`,
//! `zeta_d` straight from the parameter table and principal quantum numbers from
//! [`crate::mndod::iii`] / [`crate::mndod::iiid`].
//!
//! # The caveat that matters, stated plainly
//!
//! **NDDO *assumes* an orthonormal AO basis.** Its working equations are `F C = C eps` with no
//! overlap matrix anywhere, so the coefficients in `[MO]` live in an implicitly orthogonalized
//! (Löwdin-like) basis, while the basis section describes the *un*-orthogonalized Slater functions
//! a viewer will actually draw. The two differ by `S^{-1/2}`, and `S` is not the identity for real
//! Slater functions at bonding distances.
//!
//! So a rendered orbital is a faithful picture of an approximation, not of an exact wavefunction:
//! shapes, nodal structure and symmetry are right, detailed amplitudes in the bonding region are
//! not. This is the same compromise MOPAC's own Molden output makes, and it is inherent to writing
//! an NDDO wavefunction in a format that presumes a real basis. It is written into the file as
//! well as here, so it travels with the data.
//!
//! # Two basis sections, and why both exist
//!
//! [`MoldenBasis::Sto`] writes `[STO]`, whose primitive is
//!
//! ```text
//! norm * x^kx y^ky z^kz r^kr e^{-alfa r}
//! ```
//!
//! and which therefore represents the s and p shells **exactly**:
//!
//! ```text
//! n s   ->  kx=ky=kz=0, kr=n-1        (r^{n-1} e^{-zr})
//! n p_i ->  k_i=1, others 0, kr=n-2   (x r^{n-2} e^{-zr} = r^{n-1} e^{-zr} * x/r)
//! ```
//!
//! It **cannot** represent the d shell. `xy`, `xz` and `yz` are single monomials, but `x²-y²` and
//! `2z²-x²-y²` are not, and `[STO]` has no contraction mechanism to build them — one line is one
//! basis function. So `Sto` is refused for a molecule carrying d functions rather than writing
//! three of the five and silently dropping two.
//!
//! [`MoldenBasis::StoNg`] writes `[GTO]`, a least-squares Gaussian expansion of each Slater
//! function. Every common viewer reads it, and with `[5D]` it handles the d shell properly. It is
//! a *rendering* basis and the distinction matters:
//!
//! * the MO **coefficients are unchanged**, so orbital shapes, nodal structure and densities are
//!   as right as the orthogonality caveat above allows;
//! * anything *integrated* from the file — an overlap, a population, a multipole — is an STO-nG
//!   quantity, not a PM7 one.
//!
//! # Why the expansion is fitted rather than tabulated
//!
//! The usual source is Stewart, *J. Chem. Phys.* **52**, 431 (1970), which tabulates STO-nG for
//! 1s–5s, 2p–5p and 3d–5d. Two things argue against transcribing it:
//!
//! 1. PM7 reaches Bi, so it needs **6s and 6p**, for which no published STO-nG expansion exists.
//!    A table would have to be patched with an ad-hoc rule exactly where it runs out.
//! 2. A long numerical table copied by hand is a correctness risk that no test here could
//!    distinguish from a transcription slip, because the table would be its own reference.
//!
//! The fit is even-tempered — `alpha_i = alpha_0 beta^i` with `(alpha_0, beta)` optimized — and it
//! **measures its own quality**: [`FitQuality`] carries the normalized overlap `<STO|STO-nG>` and
//! the residual, both written into `[Title]`. One mechanism covers every shell, the file states how
//! good it is, and a test asserts a bound per element rather than trusting a table. Stewart's
//! optimal exponents are slightly better at equal `n`; for drawing an orbital the difference is
//! invisible, and the header says by how much.
//!
//! # Units
//!
//! `[STO]` is documented by Molden as Ångström, so that path writes `[Atoms] Angs` with `alfa` in
//! Å⁻¹ and `norm` in Å^{-3/2}. `[GTO]` exponents are universally atomic units, so that path writes
//! `[Atoms] AU` with exponents in Bohr⁻². Each file is internally consistent; mixing the two would
//! be silently wrong rather than rejected. Orbital energies are Hartree either way, converted from
//! the crate's eV with **its own** constant, not CODATA's — see [`crate::constants`].

use crate::basis::Basis;
use crate::constants::{BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::error::{Pm7Error, Result};
use crate::linalg::Matrix;
use crate::params::Pm7Parameters;
use crate::scf::Pm7Result;
use crate::system::{z_to_symbol, Molecule};
use std::fmt::Write as _;

/// Which basis representation the file carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoldenBasis {
    /// `[STO]` — the exact Slater basis PM7 uses. Faithful, but s/p only, and few readers support
    /// it.
    Sto,
    /// `[GTO]` — an `n`-Gaussian least-squares expansion. Widely readable, handles d functions; a
    /// rendering basis, see the module docs.
    StoNg { n: usize },
}

impl Default for MoldenBasis {
    fn default() -> Self {
        Self::StoNg { n: 6 }
    }
}

/// How well an `n`-Gaussian expansion reproduces its Slater function.
#[derive(Clone, Copy, Debug)]
pub struct FitQuality {
    /// `<STO|STO-nG>` with both sides normalized. 1 is exact.
    pub overlap: f64,
    /// `||R - P R||` in the radial `L²` norm, with `R` normalized.
    pub residual: f64,
}

/// Options for [`to_molden`].
#[derive(Clone, Debug, Default)]
pub struct MoldenOptions {
    pub basis: MoldenBasis,
    /// Extra text for `[Title]`. The method, basis and fit quality are always written.
    pub comment: Option<String>,
}

/// Molden's spherical-d order is `z², xz, yz, x²−y², xy`; the internal order (MOPAC's, see
/// `overlap_d.rs`) is `x²−y², xz, z², yz, xy`. The map is a pure permutation, pinned by test.
pub const D_PERMUTATION: [usize; 5] = [2, 1, 3, 0, 4];

// ---------------------------------------------------------------------------------------------
// Radial integrals
// ---------------------------------------------------------------------------------------------

/// `Gamma(l + 3/2)`, the only gamma values a radial Gaussian overlap needs.
fn gamma_half(l: usize) -> f64 {
    let mut value = std::f64::consts::PI.sqrt() / 2.0;
    for k in 0..l {
        value *= k as f64 + 1.5;
    }
    value
}

/// Norm of the radial Gaussian `r^l e^{-a r²}` under `int f² r² dr`.
fn gaussian_norm(alpha: f64, l: usize) -> f64 {
    (2.0 * (2.0 * alpha).powf(l as f64 + 1.5) / gamma_half(l)).sqrt()
}

/// Norm of the radial Slater `r^{n-1} e^{-z r}` under `int f² r² dr`.
fn slater_radial_norm(zeta: f64, n: usize) -> f64 {
    let mut factorial = 1.0_f64;
    for k in 2..=(2 * n) {
        factorial *= k as f64;
    }
    (2.0 * zeta).powf(n as f64 + 0.5) / factorial.sqrt()
}

/// A cached 192-point Gauss-Legendre rule on `[-1, 1]`.
fn quadrature_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    RULE.get_or_init(|| crate::special::gauss_legendre(192))
}

/// `int_0^inf r^m e^{-a r² - z r} dr`, by Gauss–Legendre quadrature.
///
/// # Why not the closed form
///
/// There is one. `I_0 = (1/2) sqrt(pi/a) erfcx(z / 2 sqrt(a))`, and integrating
/// `d/dr e^{-a r² - z r}` by parts gives the upward recursion
///
/// ```text
/// I_m = ((m - 1) I_{m-2} - z I_{m-1}) / (2 a)
/// ```
///
/// It is **numerically unstable in that direction**, and this fit is exactly where it breaks: the
/// numerator is a difference of two comparable large numbers and the divisor `2a` is small, so for
/// the smallest trial exponents each step amplifies rounding by hundreds. At `m = 4`, `a = 8.1e-4`,
/// `z = 0.9` it has already lost six digits against quadrature. That was measured, not assumed —
/// the test below is what found it, and it is kept pointed at this function.
///
/// The integrand is smooth and analytic, so a Gauss–Legendre rule converges spectrally and 256
/// nodes reach machine precision. The range is cut where **either** decay has killed the
/// integrand, since for large `a` the Gaussian bites long before the exponential does.
fn slater_gaussian_moment(m: usize, alpha: f64, zeta: f64) -> f64 {
    let reach = m as f64 + 40.0;
    let r_max = (reach / zeta).min((reach / alpha).sqrt());
    let (nodes, weights) = quadrature_rule();
    let half = 0.5 * r_max;
    let mut acc = 0.0;
    for (x, w) in nodes.iter().zip(weights) {
        let r = half * (x + 1.0);
        acc += w * r.powi(m as i32) * (-alpha * r * r - zeta * r).exp();
    }
    acc * half
}

/// Solve a small symmetric positive-definite system by Cholesky. `None` if not positive definite,
/// which here means the trial exponents collapsed together and the trial should be rejected.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1.0e-14 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Some(x)
}

/// One fitted shell: contraction coefficients over normalized primitives, plus the fit quality.
#[derive(Clone, Debug)]
pub struct Contraction {
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub quality: FitQuality,
}

/// Least-squares expand `r^{n-1} e^{-zeta r}` in `count` normalized Gaussians of angular momentum
/// `l`, with an even-tempered exponent sequence whose two parameters are optimized.
///
/// For a *given* exponent set the coefficients are the exact projection — `S c = v` with `S` the
/// analytic Gaussian overlap matrix and `v` the analytic Gaussian–Slater overlaps — so only
/// `(alpha_0, beta)` need searching. Because the projection is exact, `c . v` is `||P R||²` and
/// both reported numbers fall out of it with no extra work and no estimate.
pub fn fit_shell(zeta: f64, n: usize, l: usize, count: usize) -> Result<Contraction> {
    if count == 0 {
        return Err(Pm7Error::InvalidInput(
            "an STO-nG expansion needs at least one Gaussian".into(),
        ));
    }
    if !(zeta.is_finite() && zeta > 0.0) {
        return Err(Pm7Error::InvalidInput(format!(
            "a Slater exponent must be positive and finite, got {zeta}"
        )));
    }
    if n == 0 || n <= l {
        return Err(Pm7Error::InvalidInput(format!(
            "a shell with principal quantum number {n} has no l = {l} subshell"
        )));
    }
    let ns = slater_radial_norm(zeta, n);
    let moment = n + l + 1;

    let evaluate = |alpha0: f64, beta: f64| -> Option<(f64, Vec<f64>, Vec<f64>)> {
        let exponents: Vec<f64> = (0..count).map(|i| alpha0 * beta.powi(i as i32)).collect();
        if exponents.iter().any(|a| !a.is_finite() || *a <= 0.0) {
            return None;
        }
        let norms: Vec<f64> = exponents.iter().map(|&a| gaussian_norm(a, l)).collect();
        let mut s = vec![vec![0.0_f64; count]; count];
        for i in 0..count {
            for j in 0..count {
                let a = exponents[i] + exponents[j];
                s[i][j] = norms[i] * norms[j] * gamma_half(l) / (2.0 * a.powf(l as f64 + 1.5));
            }
        }
        let v: Vec<f64> = (0..count)
            .map(|i| norms[i] * ns * slater_gaussian_moment(moment, exponents[i], zeta))
            .collect();
        if v.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let c = cholesky_solve(&s, &v)?;
        let projected: f64 = c.iter().zip(&v).map(|(a, b)| a * b).sum();
        if !projected.is_finite() || projected <= 0.0 {
            return None;
        }
        Some((projected, c, exponents))
    };

    // Coarse-to-fine search over `(log10(alpha_0 / zeta²), beta)`. The natural scale of the
    // exponents is `zeta²`, which is what lets one fixed window serve every element.
    let (mut lo_a, mut hi_a) = (-3.0_f64, 1.5_f64);
    let (mut lo_b, mut hi_b) = (1.5_f64, 9.0_f64);
    let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
    // Wide first pass, then narrow ones: the score is smooth in both parameters, so once the
    // window has shrunk there is nothing for a fine grid to find. A uniform 21x21 for five rounds
    // costs three times as much and lands on the same exponents.
    for &steps in &[20usize, 10, 8, 8, 8] {
        let mut round_best: Option<(f64, f64, f64)> = None;
        for ia in 0..=steps {
            let a = lo_a + (hi_a - lo_a) * ia as f64 / steps as f64;
            for ib in 0..=steps {
                let b = lo_b + (hi_b - lo_b) * ib as f64 / steps as f64;
                let Some((score, c, e)) = evaluate(zeta * zeta * 10.0_f64.powf(a), b) else {
                    continue;
                };
                if round_best.map(|(s, _, _)| score > s).unwrap_or(true) {
                    round_best = Some((score, a, b));
                }
                if best.as_ref().map(|(s, _, _)| score > *s).unwrap_or(true) {
                    best = Some((score, c, e));
                }
            }
        }
        let Some((_, a, b)) = round_best else { break };
        let (span_a, span_b) = ((hi_a - lo_a) / 8.0, (hi_b - lo_b) / 8.0);
        lo_a = a - span_a;
        hi_a = a + span_a;
        lo_b = (b - span_b).max(1.02);
        hi_b = b + span_b;
    }

    let (projected, coefficients, exponents) = best.ok_or_else(|| {
        Pm7Error::LinearAlgebra(format!(
            "no {count}-Gaussian expansion of the n={n} l={l} zeta={zeta} Slater function converged"
        ))
    })?;
    // `projected = ||P R||²` and `R` is normalized, so these are exact rather than estimates.
    let overlap = projected.clamp(0.0, 1.0).sqrt();
    let residual = (1.0 - projected).max(0.0).sqrt();
    Ok(Contraction {
        exponents,
        coefficients,
        quality: FitQuality { overlap, residual },
    })
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

/// `(2z)^{n+1/2} sqrt(c / (4 pi (2n)!))` — the factor normalizing a Slater primitive in Molden's
/// Cartesian `r^{kr} e^{-zr}` form, with `c = 1` for `s` and `3` for `p`.
///
/// Derived rather than tabulated so it stays right for any `n` the parameter set uses, and checked
/// against a numerical volume integral in the tests rather than against the same algebra.
fn molden_sto_norm(n: usize, zeta: f64, angular: u32) -> f64 {
    let factorial = |m: usize| -> f64 { (1..=m).map(|k| k as f64).product::<f64>().max(1.0) };
    let c = if angular == 0 { 1.0 } else { 3.0 };
    (2.0 * zeta).powf(n as f64 + 0.5) * (c / (4.0 * std::f64::consts::PI * factorial(2 * n))).sqrt()
}

/// The shells one atom carries, as `(label, l, zeta, principal quantum number)`.
fn shells(z: u8, params: &Pm7Parameters) -> Result<Vec<(&'static str, usize, f64, usize)>> {
    let elem = params.element(z)?;
    if elem.n_orb == 0 {
        return Ok(Vec::new());
    }
    let n_sp = crate::mndod::iii(z).max(1) as usize;
    let mut out = vec![("s", 0usize, elem.zeta_s, n_sp)];
    if elem.has_p() {
        // `n > l` or there is no such orbital: `r^{n-1}` with `n = 1` and `l = 1` is `x/r`, which
        // is singular at the origin and is not a 1p function, because there is no 1p function.
        // PM7 gives **helium** a p shell (`zeta_p = 3.657`, `beta_p = -37.04`) as a polarization
        // function while `iii(2) = 1`, so this is a real case and not a defensive `max`.
        out.push(("p", 1, elem.zeta_p, n_sp.max(2)));
    }
    if elem.has_d() {
        out.push(("d", 2, elem.zeta_d, (crate::mndod::iiid(z) as usize).max(3)));
    }
    Ok(out)
}

/// Reorder one MO's coefficients from the internal AO order into Molden's.
///
/// Only the d block moves: `s` is one function, Molden's `p` order `x, y, z` is already the
/// internal one, and the d permutation is [`D_PERMUTATION`].
fn molden_order(column: &[f64], basis: &Basis, molecule: &Molecule) -> Vec<f64> {
    let mut out = vec![0.0; column.len()];
    for atom in 0..molecule.atoms.len() {
        let off = basis.atom_offset[atom];
        let norb = basis.atom_norb[atom];
        let sp = norb.min(4);
        out[off..off + sp].copy_from_slice(&column[off..off + sp]);
        if norb == 9 {
            for (slot, &internal) in D_PERMUTATION.iter().enumerate() {
                out[off + 4 + slot] = column[off + 4 + internal];
            }
        }
    }
    out
}

/// Render a converged SCF result as a Molden-format string.
///
/// Both spin channels are written for an unrestricted result; a restricted one gets a single block
/// with occupation 2.
pub fn to_molden(
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf: &Pm7Result,
    options: &MoldenOptions,
) -> Result<String> {
    if molecule.cell.is_some() {
        return Err(Pm7Error::InvalidInput(
            "Molden has no representation for a periodic system: its [MO] section is a list of \
             molecular orbitals, not Bloch states. Write the molecule, or a supercell of it, \
             without a cell."
                .into(),
        ));
    }
    let basis = Basis::build(molecule, params)?;
    if scf.mo_coeff.rows != basis.nao {
        return Err(Pm7Error::InvalidInput(format!(
            "the orbital coefficients are {}x{} but the basis has {} functions; this result does \
             not belong to this molecule",
            scf.mo_coeff.rows, scf.mo_coeff.cols, basis.nao
        )));
    }
    let has_d = basis.aos.iter().any(|ao| ao.orb >= 4);
    if options.basis == MoldenBasis::Sto && has_d {
        return Err(Pm7Error::InvalidInput(
            "[STO] cannot represent a d shell: its primitive is a single Cartesian monomial \
             `x^kx y^ky z^kz r^kr e^{-ar}`, and `x2-y2` and `2z2-x2-y2` are not monomials, so two \
             of the five d functions have no line to be written on. Use `MoldenBasis::StoNg`, \
             whose [GTO] section carries [5D] and represents all five."
                .into(),
        ));
    }

    let mut out = String::with_capacity(1024 + 24 * basis.nao * basis.nao);
    out.push_str("[Molden Format]\n");

    // --- fit every distinct shell once, so the title can state the worst case ------------------
    let mut worst = FitQuality {
        overlap: 1.0,
        residual: 0.0,
    };
    let mut fits: Vec<(u8, Vec<(&'static str, Contraction)>)> = Vec::new();
    if let MoldenBasis::StoNg { n } = options.basis {
        let mut seen: Vec<u8> = molecule.atoms.iter().map(|a| a.z).collect();
        seen.sort_unstable();
        seen.dedup();
        for z in seen {
            let mut per_atom = Vec::new();
            for (label, l, zeta, principal) in shells(z, params)? {
                let fit = fit_shell(zeta, principal, l, n)?;
                if fit.quality.overlap < worst.overlap {
                    worst = fit.quality;
                }
                per_atom.push((label, fit));
            }
            fits.push((z, per_atom));
        }
    }

    // --- title ---------------------------------------------------------------------------------
    out.push_str("[Title]\n");
    writeln!(
        out,
        " pm7-rs {} wavefunction ({:?} parameterization), orbitals from {:?}",
        env!("CARGO_PKG_VERSION"),
        params.method,
        scf.orbital_source
    )
    .ok();
    // Written without square brackets on purpose: a bracketed keyword inside the title block is
    // exactly what a parser scanning for section headers would trip over.
    out.push_str(
        " NOTE: NDDO assumes an orthonormal AO basis, so these MO coefficients are in an\n\
         \x20implicitly orthogonalized basis while the functions listed below are the raw,\n\
         \x20non-orthogonal ones. Orbital shapes, nodes and symmetry are faithful; amplitudes in\n\
         \x20the bonding region are approximate. See the pm7-rs `molden` module documentation.\n",
    );
    match options.basis {
        MoldenBasis::Sto => out.push_str(
            " basis: exact single-zeta Slater ([STO], Angstrom) -- the basis PM7 actually uses.\n",
        ),
        MoldenBasis::StoNg { n } => {
            writeln!(
                out,
                " basis: STO-{n}G least-squares rendering basis ([GTO], atomic units); worst shell\n\
                 \x20overlap <STO|STO-nG> = {:.8}, radial residual {:.2e}. MO coefficients are\n\
                 \x20PM7's and unchanged; anything integrated from this file is an STO-{n}G quantity.",
                worst.overlap, worst.residual
            )
            .ok();
        }
    }
    if let Some(comment) = &options.comment {
        writeln!(out, " {comment}").ok();
    }

    // --- geometry ------------------------------------------------------------------------------
    // The units follow the basis section, so no file ever mixes the two.
    let angstrom = options.basis == MoldenBasis::Sto;
    writeln!(out, "[Atoms] {}", if angstrom { "Angs" } else { "AU" }).ok();
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let p = if angstrom {
            atom.position * BOHR_TO_ANGSTROM
        } else {
            atom.position
        };
        writeln!(
            out,
            " {:<2} {:5} {:5} {:18.10} {:18.10} {:18.10}",
            z_to_symbol(atom.z).unwrap_or("X"),
            i + 1,
            atom.z,
            p.x,
            p.y,
            p.z
        )
        .ok();
    }

    // --- basis ---------------------------------------------------------------------------------
    match options.basis {
        MoldenBasis::Sto => {
            out.push_str("[STO]\n");
            for ao in &basis.aos {
                let elem = params.element(ao.z)?;
                // Same `n > l` rule as `shells`: helium's p shell is a real case.
                let n =
                    (crate::mndod::iii(ao.z).max(1) as usize).max(if ao.orb == 0 { 1 } else { 2 });
                // zeta is Bohr^-1 inside the crate; this section is Angstrom.
                let zeta = if ao.orb == 0 {
                    elem.zeta_s
                } else {
                    elem.zeta_p
                } / BOHR_TO_ANGSTROM;
                let (kx, ky, kz, kr) = match ao.orb {
                    0 => (0, 0, 0, n as i32 - 1),
                    1 => (1, 0, 0, n as i32 - 2),
                    2 => (0, 1, 0, n as i32 - 2),
                    _ => (0, 0, 1, n as i32 - 2),
                };
                let angular = u32::from(ao.orb != 0);
                writeln!(
                    out,
                    " {:5} {:3} {:3} {:3} {:3} {:18.10} {:18.10}",
                    ao.atom + 1,
                    kx,
                    ky,
                    kz,
                    kr,
                    zeta,
                    molden_sto_norm(n, zeta, angular)
                )
                .ok();
            }
        }
        MoldenBasis::StoNg { .. } => {
            out.push_str("[GTO]\n");
            for (i, atom) in molecule.atoms.iter().enumerate() {
                writeln!(out, " {:>4} 0", i + 1).ok();
                let per_atom = fits
                    .iter()
                    .find(|(z, _)| *z == atom.z)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                for (label, fit) in per_atom {
                    writeln!(out, " {label:<2} {:>3} 1.00", fit.exponents.len()).ok();
                    for (alpha, coefficient) in fit.exponents.iter().zip(&fit.coefficients) {
                        writeln!(out, "{alpha:>22.10e} {coefficient:>22.10e}").ok();
                    }
                }
                // A blank line closes every atom -- **including a sparkle**, whose shell list is
                // empty. Omitting the block for a sparkle would shift every later atom's basis by
                // one and silently mis-assign the whole [MO] section.
                out.push('\n');
            }
        }
    }

    // Mandatory the moment any d function exists: without it a reader assumes six Cartesian d
    // functions, reads five, and mis-maps every coefficient from that atom onward.
    if has_d {
        out.push_str("[5D]\n");
    }

    // --- orbitals ------------------------------------------------------------------------------
    out.push_str("[MO]\n");
    write_channel(
        &mut out,
        &scf.mo_coeff,
        &scf.mo_energies,
        &scf.occupations(),
        "Alpha",
        &basis,
        molecule,
    );
    if let (Some(energies), Some(coefficients), Some(occ)) = (
        scf.mo_energies_beta.as_ref(),
        scf.mo_coeff_beta.as_ref(),
        scf.occupations_beta(),
    ) {
        write_channel(
            &mut out,
            coefficients,
            energies,
            &occ,
            "Beta",
            &basis,
            molecule,
        );
    }
    Ok(out)
}

fn write_channel(
    out: &mut String,
    coefficients: &Matrix,
    energies: &[f64],
    occupations: &[f64],
    spin: &str,
    basis: &Basis,
    molecule: &Molecule,
) {
    let n = energies.len().min(coefficients.cols);
    for k in 0..n {
        let column: Vec<f64> = (0..coefficients.rows)
            .map(|row| coefficients[(row, k)])
            .collect();
        let reordered = molden_order(&column, basis, molecule);
        // No symmetry perception here, so every orbital is labelled `a`. A label is required by
        // the format; inventing an irreducible representation would be worse than declining to.
        writeln!(out, " Sym= {}a", k + 1).ok();
        writeln!(out, " Ene= {:18.10}", energies[k] * EV_TO_HARTREE).ok();
        writeln!(out, " Spin= {spin}").ok();
        writeln!(
            out,
            " Occup= {:12.6}",
            occupations.get(k).copied().unwrap_or(0.0)
        )
        .ok();
        for (mu, value) in reordered.iter().enumerate() {
            writeln!(out, " {:5} {:18.10}", mu + 1, value).ok();
        }
    }
}

/// Run [`to_molden`] and write the result to `path`.
pub fn write_molden(
    path: impl AsRef<std::path::Path>,
    molecule: &Molecule,
    params: &Pm7Parameters,
    scf: &Pm7Result,
    options: &MoldenOptions,
) -> Result<()> {
    std::fs::write(path, to_molden(molecule, params, scf, options)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::scf::{run_pm7, Pm7Options};
    use crate::system::Atom;

    fn water() -> Molecule {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.96 * a, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.24 * a, 0.93 * a, 0.0),
            },
        ])
    }

    fn hydrogen_sulfide() -> Molecule {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Molecule::new(vec![
            Atom {
                z: 16,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.34 * a, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.33 * a, 1.29 * a, 0.0),
            },
        ])
    }

    fn options() -> Pm7Options {
        Pm7Options {
            method: "pm7".parse().unwrap(),
            e_tol: 1.0e-12,
            ..Default::default()
        }
    }

    /// The Gauss-Legendre radial moment against a completely different rule.
    ///
    /// The recursion divides by `2a` at every step, so it could in principle amplify rounding for
    /// the smallest exponents; this says it does not, over the whole range the fit searches. It is
    /// the piece the entire `[GTO]` section rests on — a wrong moment would give a confidently
    /// reported, wrong overlap in `[Title]`.
    #[test]
    fn the_radial_moment_matches_an_independent_quadrature() {
        for &zeta in &[0.9_f64, 2.0, 5.5] {
            for &scale in &[1.0e-3_f64, 1.0e-2, 0.1, 1.0, 10.0] {
                let alpha = zeta * zeta * scale;
                for m in 0..=9usize {
                    let exact = slater_gaussian_moment(m, alpha, zeta);
                    // Simpson on a grid long enough that the tail is below 1e-18.
                    let r_max = (m as f64 + 40.0) / zeta;
                    let steps = 200_000;
                    let h = r_max / steps as f64;
                    let mut acc = 0.0;
                    for i in 0..=steps {
                        let r = i as f64 * h;
                        let w = if i == 0 || i == steps {
                            1.0
                        } else if i % 2 == 1 {
                            4.0
                        } else {
                            2.0
                        };
                        acc += w * r.powi(m as i32) * (-alpha * r * r - zeta * r).exp();
                    }
                    acc *= h / 3.0;
                    let tolerance = 1.0e-8 * exact.abs().max(1.0e-12);
                    assert!(
                        (exact - acc).abs() < tolerance,
                        "m={m} alpha={alpha} zeta={zeta}: closed form {exact:e} vs quadrature {acc:e}"
                    );
                }
            }
        }
    }

    /// `norm` on an `[STO]` line must actually normalize the primitive it is attached to. Checked
    /// by integrating `|chi|²` numerically rather than by re-deriving the same closed form, so an
    /// algebra slip cannot agree with itself.
    #[test]
    fn the_stated_sto_normalization_integrates_to_one() {
        for (n, zeta, angular) in [(1usize, 1.3, 0u32), (2, 2.7, 0), (2, 2.0, 1), (3, 1.8, 1)] {
            let norm = molden_sto_norm(n, zeta, angular);
            let kr = if angular == 0 {
                n as i32 - 1
            } else {
                n as i32 - 2
            };
            // For s: 4 pi int |N r^kr e^{-zr}|² r² dr.
            // For p: chi = N x r^kr e^{-zr}, and <x²> over the sphere contributes 4 pi / 3 r².
            let angular_factor = if angular == 0 {
                4.0 * std::f64::consts::PI
            } else {
                4.0 * std::f64::consts::PI / 3.0
            };
            let extra = if angular == 0 { 0 } else { 2 };
            let steps = 200_000;
            let r_max = 60.0 / zeta;
            let h = r_max / steps as f64;
            let mut acc = 0.0;
            for i in 0..=steps {
                let r = i as f64 * h;
                let w = if i == 0 || i == steps {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                acc += w * r.powi(2 * kr + 2 + extra) * (-2.0 * zeta * r).exp();
            }
            acc *= h / 3.0 * angular_factor * norm * norm;
            assert!(
                (acc - 1.0).abs() < 1.0e-8,
                "n={n} zeta={zeta} angular={angular}: |chi|² integrates to {acc}"
            );
        }
    }

    /// The fit is a projection, so its two reported numbers satisfy `overlap² + residual² = 1`
    /// exactly. This is what makes `[Title]`'s claim checkable rather than decorative.
    #[test]
    fn the_reported_fit_quality_is_self_consistent() {
        for (zeta, n, l) in [(1.3, 1, 0), (2.0, 2, 1), (1.8, 3, 2), (2.5, 6, 1)] {
            let q = fit_shell(zeta, n, l, 6).unwrap().quality;
            assert!(
                (q.overlap * q.overlap + q.residual * q.residual - 1.0).abs() < 1.0e-10,
                "overlap {} and residual {} are not a projection pair",
                q.overlap,
                q.residual
            );
        }
    }

    /// More Gaussians cannot fit worse.
    #[test]
    fn more_gaussians_fit_better() {
        let mut previous = 0.0;
        for count in 1..=6 {
            let fit = fit_shell(1.8, 2, 0, count).unwrap();
            assert!(
                fit.quality.overlap > previous - 1.0e-6,
                "STO-{count}G overlap {} is worse than the previous {previous}",
                fit.quality.overlap
            );
            previous = fit.quality.overlap;
        }
        assert!(
            previous > 0.999,
            "STO-6G should be excellent, got {previous}"
        );
    }

    /// Every element PM7 parameterizes, every shell, at the default `n = 6`.
    ///
    /// The bound is what a tabulated expansion could not offer: it covers 6s and 6p, for which no
    /// published STO-nG exists, on the same footing as everything else.
    #[test]
    fn every_element_fits_to_a_stated_bound() {
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let mut worst = (1.0_f64, 0_u8, "");
        for z in 1..=86_u8 {
            let Ok(element) = params.element(z) else {
                continue;
            };
            if element.n_orb == 0 {
                continue;
            }
            for (label, l, zeta, n) in shells(z, &params).unwrap() {
                if zeta <= 0.0 {
                    continue;
                }
                let fit = fit_shell(zeta, n, l, 6).unwrap();
                if fit.quality.overlap < worst.0 {
                    worst = (fit.quality.overlap, z, label);
                }
            }
        }
        assert!(
            worst.0 > 0.995,
            "worst STO-6G overlap is {} on Z={} {} shell",
            worst.0,
            worst.1,
            worst.2
        );
    }

    /// The d permutation is a bijection, and it is the one Molden documents.
    #[test]
    fn the_d_permutation_is_a_bijection_onto_moldens_order() {
        let mut seen = D_PERMUTATION.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4], "not a permutation");
        let internal = ["x2-y2", "xz", "z2", "yz", "xy"];
        let molden: Vec<&str> = D_PERMUTATION.iter().map(|&i| internal[i]).collect();
        assert_eq!(molden, vec!["z2", "xz", "yz", "x2-y2", "xy"]);
    }

    /// The file has the sections a reader needs, and the caveat travels with it.
    #[test]
    fn a_water_file_has_the_required_sections() {
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let scf = run_pm7(&water(), &params, &options()).unwrap();
        let text = to_molden(&water(), &params, &scf, &MoldenOptions::default()).unwrap();
        assert!(text.starts_with("[Molden Format]"));
        for section in ["[Title]", "[Atoms] AU", "[GTO]", "[MO]"] {
            assert!(text.contains(section), "missing {section}");
        }
        assert!(!text.contains("[5D]"), "water has no d functions");
        assert_eq!(text.matches(" Ene= ").count(), 6);
        assert_eq!(text.matches(" Occup= ").count(), 6);
        assert!(
            text.contains("<STO|STO-nG> ="),
            "no fit quality in the title"
        );
        assert!(
            text.contains("orthonormal AO basis"),
            "the orthogonality caveat must travel with the file"
        );
    }

    /// `[STO]` is exact for s/p, and says so in Angstrom.
    #[test]
    fn an_sto_file_is_written_in_angstrom() {
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let scf = run_pm7(&water(), &params, &options()).unwrap();
        let text = to_molden(
            &water(),
            &params,
            &scf,
            &MoldenOptions {
                basis: MoldenBasis::Sto,
                comment: None,
            },
        )
        .unwrap();
        assert!(text.contains("[Atoms] Angs"), "STO files are Angstrom");
        assert!(text.contains("[STO]"));
        assert!(!text.contains("[GTO]"));
        // One line per AO: water has 6.
        let sto = text.split("[STO]\n").nth(1).unwrap();
        let lines = sto.lines().take_while(|l| !l.starts_with('[')).count();
        assert_eq!(lines, 6, "one [STO] line per atomic orbital");
    }

    /// `[STO]` is refused for a d-bearing molecule instead of dropping two of the five functions.
    #[test]
    fn sto_refuses_d_functions() {
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let scf = run_pm7(&hydrogen_sulfide(), &params, &options()).unwrap();
        let error = to_molden(
            &hydrogen_sulfide(),
            &params,
            &scf,
            &MoldenOptions {
                basis: MoldenBasis::Sto,
                comment: None,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("d shell"), "{error}");
        assert!(error.contains("StoNg"), "the error should say what to use");
    }

    /// A d-bearing molecule gets `[5D]`, and its coefficients are permuted rather than copied.
    #[test]
    fn a_d_molecule_declares_five_d_and_permutes() {
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let molecule = hydrogen_sulfide();
        let scf = run_pm7(&molecule, &params, &options()).unwrap();
        let text = to_molden(&molecule, &params, &scf, &MoldenOptions::default()).unwrap();
        assert!(text.contains("[5D]"), "a d basis must declare [5D]");

        let basis = Basis::build(&molecule, &params).unwrap();
        let column: Vec<f64> = (0..basis.nao).map(|row| scf.mo_coeff[(row, 0)]).collect();
        let reordered = molden_order(&column, &basis, &molecule);
        // `reordered[4 + slot] == column[4 + D_PERMUTATION[slot]]`, sulfur being atom 0.
        assert_eq!(reordered[4], column[6], "Molden slot 0 is z2");
        assert_eq!(reordered[5], column[5], "slot 1 is xz, a fixed point");
        assert_eq!(reordered[6], column[7], "slot 2 is yz");
        assert_eq!(reordered[7], column[4], "slot 3 is x2-y2");
        assert_eq!(reordered[8], column[8], "slot 4 is xy, a fixed point");
        for i in 0..4 {
            assert_eq!(reordered[i], column[i], "s and p are untouched");
        }
    }

    /// A periodic system is refused rather than written as though it were a molecule.
    #[test]
    fn a_periodic_system_is_refused() {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        let chain =
            water().with_cell(crate::cell::Cell::new(&[Vec3::new(8.0 * a, 0.0, 0.0)]).unwrap());
        let params = Pm7Parameters::method("pm7".parse().unwrap()).unwrap();
        let scf = run_pm7(&chain, &params, &options()).unwrap();
        let error = to_molden(&chain, &params, &scf, &MoldenOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("periodic"), "{error}");
    }
}
