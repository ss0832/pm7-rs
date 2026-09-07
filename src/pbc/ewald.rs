// SPDX-License-Identifier: GPL-3.0-or-later

//! Ewald summation of the periodic monopole Coulomb interaction, in 1, 2, and 3 dimensions,
//! for neutral **and** charged cells.
//!
//! # What this computes
//!
//! ```text
//! E = ½ Σ'_{A,B,T} q_A q_B / |R_B + T − R_A|          (in atomic-charge units; eV via PM7_EV)
//! V_A = ∂E/∂q_A = Σ'_{B,T} q_B / |R_B + T − R_A|
//! ```
//!
//! The primed sum excludes the `A = B, T = 0` self term. This is the *only* long-range object
//! the periodic NDDO energy needs, because PM7's feathering makes every two-centre integral
//! exactly the point-charge value beyond 7 Å ([`crate::pbc`]); everything shorter ranged is a
//! compact-support real-space correction handled elsewhere.
//!
//! # The split
//!
//! `1/r = erfc(αr)/r + erf(αr)/r`. The first term is short ranged and summed in real space; the
//! second is smooth and summed in reciprocal space. The total is independent of `α`, which is
//! the strongest available correctness test and is exercised directly by
//! `energy_is_independent_of_the_splitting_parameter`.
//!
//! # Dimensionality
//!
//! Each dimension needs its own reciprocal-space term, because the Fourier transform is taken
//! only over the periodic directions:
//!
//! | dim | reciprocal-space kernel |
//! |---|---|
//! | 3 | `(4π/V) Σ_{G≠0} e^{−G²/4α²}/G² cos(G·r)` |
//! | 2 | Parry: `(π/A) Σ_{G≠0} cos(G·r_∥)/G · [e^{Gz} erfc(αz + G/2α) + e^{−Gz} erfc(−αz + G/2α)]` plus a `G = 0` term |
//! | 1 | `(2/L) Σ_{G≠0} cos(G x) · ∫₀^{α²} (dt/t) e^{−ρ²t − G²/4t}` plus a `G = 0` term |
//!
//! The 1-D kernel is derived in one line from `erf(αr)/r = (2/√π)∫₀^α e^{−s²r²} ds` followed by
//! a Fourier transform along the periodic axis; it is a smooth integral over a **finite** range
//! whose integrand vanishes exponentially at both ends, so Gauss–Legendre converges immediately.
//!
//! # Charged cells
//!
//! For `q_tot = Σ_A q_A ≠ 0` the `G = 0` reciprocal term diverges in every dimension. Removing it
//! is exactly the statement that a uniform neutralizing background has been added, and it leaves
//! a finite, `α`-dependent constant that must be kept for the total to stay `α`-independent:
//!
//! * 3-D: `E_bg = −π q_tot² / (2 α² V)`
//! * 2-D: the `G = 0` Parry term already handles a charged sheet; what is dropped is the
//!   divergent constant, leaving `E_bg = −√π q_tot² / (2 α A)` from the `z`-independent part.
//! * 1-D: `E_bg = (q_tot² / 2L) · (ln(α²·ℓ²) + γ)`-type constant, absorbed by defining the
//!   background on the same length scale; see [`EwaldPotential::background_energy`].
//!
//! The background is *uniform*, so it exerts **no force on any atom**, but it does depend on the
//! cell measure and therefore **contributes an isotropic term to the stress**. Omitting that is
//! the classic silent bug in charged-cell stresses, so it is computed explicitly here and
//! checked by `charged_cell_stress_matches_finite_difference`.

use crate::cell::Cell;
use crate::constants::PM7_EV;
use crate::error::Result;
use crate::math::{Mat3, Vec3};
use crate::pbc::{BackgroundCharge, PbcOptions};
use crate::special::{erfc, erfcx, exp_integral_e1, gauss_legendre};

const SQRT_PI: f64 = 1.772_453_850_905_516;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Reference length² for the 1-D `G = 0` regularization, in Bohr².
///
/// The 1-D lattice sum's `G = 0` term diverges logarithmically, and what is dropped is
/// `q_tot² · ln ε` — a constant with units, so *some* length has to set the scale. It must not
/// be the cell length: `ln L²` would then differ between a cell and its supercell, and a
/// `n×1×1` k mesh would disagree with the `n`-fold supercell it is equivalent to by
/// `2 q_tot² ln(n)/L`. A fixed reference keeps that identity exact.
///
/// For a **neutral** cell the choice is invisible: the term multiplies `Σ_{i,j} q_i q_j = q_tot²`
/// and vanishes. For a charged wire the absolute energy is convention-dependent no matter what,
/// and this states the convention instead of hiding it in the cell size.
const WIRE_REFERENCE_LENGTH_SQUARED: f64 = 1.0;

/// Real- and reciprocal-space cutoffs and the splitting parameter for one cell.
#[derive(Clone, Debug)]
pub struct EwaldParameters {
    /// Splitting parameter, Bohr⁻¹.
    pub alpha: f64,
    /// Real-space cutoff, Bohr.
    pub r_cut: f64,
    /// Reciprocal-space cutoff, Bohr⁻¹ (on `|G|`).
    pub g_cut: f64,
    /// Real-space lattice translations within `r_cut` (includes `[0,0,0]`).
    pub real_images: Vec<[i32; 3]>,
    /// Reciprocal-lattice vectors within `g_cut`, excluding `G = 0`.
    pub g_vectors: Vec<Vec3>,
}

impl EwaldParameters {
    /// Choose `α` and the two cutoffs for `cell` and `n_atoms` at the requested accuracy.
    ///
    /// The standard balance `α = (N π³ / V²)^{1/6}` equalizes the real- and reciprocal-space
    /// work in 3-D; for lower dimensions the same expression is used with the cell measure
    /// raised to the matching power, which is heuristic — and harmless, because the *result* is
    /// α-independent. Only the cost depends on this choice.
    pub fn new(cell: &Cell, n_atoms: usize, accuracy: f64, alpha: Option<f64>) -> Self {
        let dim = cell.dim().max(1);
        let measure = cell.measure().max(1.0e-12);
        let n = n_atoms.max(1) as f64;
        let alpha = alpha.unwrap_or_else(|| {
            let density = n / measure;
            // α ~ √π (density)^{1/dim} balances the two sums; the √π keeps 3-D close to the
            // textbook (Nπ³/V²)^{1/6}.
            SQRT_PI * density.powf(1.0 / dim as f64)
        });
        // erfc(α r_cut) ≈ accuracy  →  r_cut ≈ sqrt(−ln(accuracy)) / α, with the standard
        // logarithmic correction folded in by solving iteratively.
        let s = (-accuracy.ln()).max(1.0).sqrt();
        let mut r_cut = s / alpha;
        for _ in 0..40 {
            // Refine against the true tail erfc(αr)/r ≈ accuracy.
            let f = erfc(alpha * r_cut) / r_cut;
            if f <= accuracy {
                break;
            }
            r_cut *= 1.1;
        }
        // exp(−G²/4α²) ≈ accuracy → G_cut = 2α sqrt(−ln accuracy).
        let g_cut = 2.0 * alpha * s;

        let real_images = cell.image_indices(r_cut, cell_diameter(cell));
        let g_vectors = reciprocal_vectors(cell, g_cut);
        Self {
            alpha,
            r_cut,
            g_cut,
            real_images,
            g_vectors,
        }
    }
}

/// Reciprocal-lattice vectors (with `2π`) up to `g_cut`, excluding `G = 0`, in a fixed order.
///
/// Only the periodic directions contribute: a 2-D cell has a 2-D reciprocal lattice, so the sum
/// is over a plane of `G` vectors, not a 3-D shell.
fn reciprocal_vectors(cell: &Cell, g_cut: f64) -> Vec<Vec3> {
    let dim = cell.dim();
    if dim == 0 {
        return Vec::new();
    }
    let b = cell.reciprocal_2pi();
    let mut n = [0i32; 3];
    for (k, slot) in n.iter_mut().enumerate().take(dim) {
        *slot = (g_cut / b[k].norm()).ceil() as i32;
    }
    let g2 = g_cut * g_cut;
    let mut out = Vec::new();
    for i in -n[0]..=n[0] {
        for j in -n[1]..=n[1] {
            for k in -n[2]..=n[2] {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                let g = b[0] * i as f64 + b[1] * j as f64 + b[2] * k as f64;
                if g.norm2() <= g2 {
                    out.push(g);
                }
            }
        }
    }
    out
}

fn cell_diameter(cell: &Cell) -> f64 {
    let v = cell.vectors();
    let dim = v.len();
    let mut worst = 0.0_f64;
    for mask in 0..(1usize << dim) {
        let mut corner = Vec3::zero();
        for (k, a) in v.iter().enumerate() {
            if mask & (1 << k) != 0 {
                corner += *a;
            }
        }
        worst = worst.max(corner.norm());
    }
    worst
}

/// The periodic monopole electrostatics of one geometry.
///
/// Energies are in eV and charges in units of `e`, so every returned quantity already carries
/// PM7's `PM7_EV` conversion (`e²/Bohr → eV`).
#[derive(Clone, Debug)]
pub struct EwaldPotential {
    /// `V_A = ∂E/∂q_A` in eV per unit charge — the quantity the Fock diagonal needs.
    pub potential: Vec<f64>,
    /// Total monopole energy in eV, including the neutralizing background.
    pub energy: f64,
    /// The background (charged-cell) part of `energy`, reported separately because it is a
    /// convention-dependent constant rather than a physical interaction.
    pub background_energy: f64,
    /// `∂E/∂R_A` in eV/Bohr.
    pub gradient: Vec<Vec3>,
    /// `∂E/∂ε` in eV (the *unnormalized* virial; divide by the cell measure for a stress).
    pub virial: Mat3,
    /// Total cell charge that produced this potential.
    pub total_charge: f64,
}

/// Compute the periodic monopole energy, per-atom potential, forces, and virial.
///
/// `charges[A] = q_A` is the net charge on atom `A` (`Z_A − P_A` in NDDO).
pub fn ewald(
    cell: &Cell,
    positions: &[Vec3],
    charges: &[f64],
    params: &EwaldParameters,
    options: &PbcOptions,
) -> Result<EwaldPotential> {
    let q_tot: f64 = charges.iter().sum();
    if options.background == BackgroundCharge::Forbid && q_tot.abs() > 1.0e-8 {
        return Err(crate::error::Pm7Error::InvalidInput(format!(
            "cell carries a net charge of {q_tot:+.6} e and BackgroundCharge::Forbid was set; \
             use BackgroundCharge::Jellium to add a uniform neutralizing background"
        )));
    }
    let mut acc = Accumulator::new(positions.len());
    real_space(cell, positions, charges, params, &mut acc);
    match cell.dim() {
        3 => reciprocal_3d(cell, positions, charges, params, &mut acc),
        2 => reciprocal_2d(cell, positions, charges, params, &mut acc),
        1 => reciprocal_1d(cell, positions, charges, params, &mut acc),
        _ => {}
    }
    self_term(charges, params, &mut acc);
    let background_energy = background(cell, charges, params, &mut acc);

    // Strain only acts within the periodic subspace: a 2-D slab has no lattice vector normal to
    // it, so there is no stress component conjugate to that direction and the out-of-plane parts
    // of the accumulated virial are not physical. Projecting here — once, at the end — is both
    // correct and cheaper than threading the projector through every term.
    let p = periodic_projector(cell);
    let virial = p.mul_mat(&acc.virial).mul_mat(&p);

    Ok(EwaldPotential {
        potential: acc.potential.iter().map(|v| v * PM7_EV).collect(),
        energy: acc.energy * PM7_EV,
        background_energy: background_energy * PM7_EV,
        gradient: acc.gradient.iter().map(|g| *g * PM7_EV).collect(),
        virial: virial.scaled(PM7_EV),
        total_charge: q_tot,
    })
}

struct Accumulator {
    energy: f64,
    potential: Vec<f64>,
    gradient: Vec<Vec3>,
    virial: Mat3,
}

impl Accumulator {
    fn new(n: usize) -> Self {
        Self {
            energy: 0.0,
            potential: vec![0.0; n],
            gradient: vec![Vec3::zero(); n],
            virial: Mat3::zero(),
        }
    }

    /// Accumulate a term that depends on the atoms only through the displacement `d`.
    ///
    /// `value` is the per-unit-charge energy, `dedd` the derivative of the **already
    /// charge-weighted** energy with respect to `d`. Then `∂E/∂R_b = dedd`,
    /// `∂E/∂R_a = −dedd`, and the strain derivative of such a term is `(∂E/∂d) ⊗ d`.
    /// Routing every pair term through one place is what keeps the gradient and the stress
    /// from drifting apart as terms are added.
    #[inline]
    fn add_pair_term(
        &mut self,
        a: usize,
        b: usize,
        qa: f64,
        qb: f64,
        scale: f64,
        value: f64,
        d: Vec3,
        dedd: Vec3,
    ) {
        self.energy += qa * qb * scale * value;
        self.potential[a] += qb * scale * value;
        self.potential[b] += qa * scale * value;
        self.gradient[a] -= dedd;
        self.gradient[b] += dedd;
        self.virial = self.virial.plus(&Mat3::outer(dedd, d));
    }

    /// Add an explicit strain derivative that does not come from a displacement (a `1/V`,
    /// `1/A`, `1/L`, or `1/G` prefactor).
    #[inline]
    fn add_virial(&mut self, m: &Mat3) {
        self.virial = self.virial.plus(m);
    }
}

/// Real-space part: `Σ' q_A q_B erfc(α r)/r`.
fn real_space(
    cell: &Cell,
    positions: &[Vec3],
    charges: &[f64],
    params: &EwaldParameters,
    acc: &mut Accumulator,
) {
    let n = positions.len();
    let alpha = params.alpha;
    let r_cut = params.r_cut;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / SQRT_PI;
    for a in 0..n {
        for b in a..n {
            for &t in &params.real_images {
                if a == b && t == [0, 0, 0] {
                    continue;
                }
                // Keep each unordered pair once: for a == b, half the translations.
                if a == b && !positive_translation(t) {
                    continue;
                }
                let d = positions[b] + cell.translation(t) - positions[a];
                let r = d.norm();
                if r > r_cut || r < 1.0e-12 {
                    continue;
                }
                let e = erfc(alpha * r) / r;
                // d/dr [erfc(αr)/r] = −erfc(αr)/r² − (2α/√π) e^{−α²r²}/r
                let dedr = -(e + two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp()) / r;
                let (qa, qb) = (charges[a], charges[b]);
                let dedd = d * (qa * qb * dedr / r);
                acc.add_pair_term(a, b, qa, qb, 1.0, e, d, dedd);
            }
        }
    }
}

/// 3-D reciprocal space: `(2π/V) Σ_{G≠0} e^{−G²/4α²}/G² · |S(G)|²` with `S(G) = Σ_A q_A e^{iG·R_A}`.
fn reciprocal_3d(
    cell: &Cell,
    positions: &[Vec3],
    charges: &[f64],
    params: &EwaldParameters,
    acc: &mut Accumulator,
) {
    let v = cell.measure();
    let inv_4a2 = 1.0 / (4.0 * params.alpha * params.alpha);
    let n = positions.len();
    for &g in &params.g_vectors {
        let g2 = g.norm2();
        let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
        // Structure factor.
        let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
        let mut phase = vec![0.0_f64; n];
        for (i, (p, q)) in positions.iter().zip(charges).enumerate() {
            let ph = g.dot(*p);
            phase[i] = ph;
            sre += q * ph.cos();
            sim += q * ph.sin();
        }
        let s2 = sre * sre + sim * sim;
        acc.energy += pref * s2;
        for i in 0..n {
            let (c, s) = (phase[i].cos(), phase[i].sin());
            // ∂E/∂q_i = 2·pref·(S_re cos + S_im sin)
            acc.potential[i] += 2.0 * pref * (sre * c + sim * s);
            // ∂E/∂R_i = 2·pref·q_i·(S_re·(−sin) + S_im·cos)·G
            let f = 2.0 * pref * charges[i] * (sim * c - sre * s);
            acc.gradient[i] += g * f;
        }
        // Strain derivative. Under `R → (1+ε)R` the reciprocal vectors transform as
        // `G → (1 − εᵀ)G`, so `∂G²/∂ε_ab = −2 G_a G_b`, while `G·R` is strain-invariant
        // (`|S|²` does not change). Collecting the `1/V`, `e^{−G²/4α²}`, and `1/G²` factors:
        //   ∂/∂ε_ab [pref·|S|²] = pref·|S|² · ( 2(1/G² + 1/4α²) G_a G_b − δ_ab )
        let f = pref * s2;
        let c = 2.0 * (1.0 / g2 + inv_4a2);
        let mut m = Mat3::zero();
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                m.set(a, b, f * (c * g.get(a) * g.get(b) - delta));
            }
        }
        acc.add_virial(&m);
    }
}

/// 2-D (Parry) reciprocal space for a slab periodic in the plane of `cell`.
///
/// ```text
/// E_recip = (π/2A) Σ_{G≠0} Σ_{A,B} q_A q_B cos(G·r_∥)/G ·
///               [ e^{ G z} erfc(αz + G/2α) + e^{−G z} erfc(−αz + G/2α) ]
///         − (√π/A)  Σ_{A,B} q_A q_B [ z erf(αz) + e^{−α²z²}/(α√π) ] · (√π/2)
/// ```
///
/// with `z = z_B − z_A` measured along the slab normal. The exponentials are evaluated through
/// [`erfcx`] so that `e^{Gz}` never overflows: the combination reduces to
/// `erfcx(αz + G/2α) · exp(−(αz − G/2α)²)`, whose exponent is never positive.
fn reciprocal_2d(
    cell: &Cell,
    positions: &[Vec3],
    charges: &[f64],
    params: &EwaldParameters,
    acc: &mut Accumulator,
) {
    let area = cell.measure();
    let normal = cell.completed_vectors()[2].normalized();
    let alpha = params.alpha;
    let n = positions.len();

    // G ≠ 0 terms.
    for &g in &params.g_vectors {
        let gn = g.norm();
        let pref = std::f64::consts::PI / (2.0 * area * gn);
        for a in 0..n {
            for b in a..n {
                let scale = if a == b { 1.0 } else { 2.0 };
                let d = positions[b] - positions[a];
                let z = normal.dot(d);
                let (val, dval_dz, dval_dpar, dval_dg) = parry_kernel(gn, alpha, z, g.dot(d));
                let qq = charges[a] * charges[b] * scale;
                // ∂/∂d = (∂/∂z) n̂ + (∂/∂(G·d)) G
                let dedd = normal * (pref * qq * dval_dz) + g * (pref * qq * dval_dpar);
                acc.add_pair_term(a, b, charges[a], charges[b], scale, pref * val, d, dedd);
                // Strain dependence beyond the displacement:
                //   ∂(1/A)/∂ε_ab = −(1/A) P_ab   (P = in-plane projector)
                //   ∂G/∂ε_ab     = −G_a G_b / G
                //   ∂(G·d)/∂ε_ab = −G_a d_b      (from G; the +G_a d_b from d is already in
                //                                 `dedd`, and the two cancel exactly because
                //                                 `G·d` is strain-invariant)
                // The last line is easy to miss: `dval_dg` is taken at *fixed* `G·d`, so without
                // it the `G·d` path would be counted once instead of zero times.
                let e_term = pref * qq * val;
                let dg_term = pref * qq * dval_dg;
                let dpar_term = pref * qq * dval_dpar;
                let mut m = Mat3::zero();
                for i in 0..3 {
                    for j in 0..3 {
                        let p_ij = if i == j { 1.0 } else { 0.0 } - normal.get(i) * normal.get(j);
                        let ghat = g.get(i) * g.get(j) / gn;
                        m.set(
                            i,
                            j,
                            e_term * (ghat / gn - p_ij)
                                - dg_term * ghat
                                - dpar_term * g.get(i) * d.get(j),
                        );
                    }
                }
                acc.add_virial(&m);
            }
        }
    }

    // G = 0 term: −(π/A) Σ q_A q_B [ z erf(αz) + e^{−α²z²}/(α√π) ].
    for a in 0..n {
        for b in a..n {
            let scale = if a == b { 1.0 } else { 2.0 };
            let d = positions[b] - positions[a];
            let z = normal.dot(d);
            let az = alpha * z;
            let erf_az = 1.0 - erfc(az);
            let val = -(std::f64::consts::PI / area)
                * (z * erf_az + (-az * az).exp() / (alpha * SQRT_PI));
            // d/dz [z erf(αz) + e^{−α²z²}/(α√π)] = erf(αz)
            let dval_dz = -(std::f64::consts::PI / area) * erf_az;
            let qq = charges[a] * charges[b] * scale;
            let dedd = normal * (qq * dval_dz);
            acc.add_pair_term(a, b, charges[a], charges[b], scale, val, d, dedd);
            // 1/A prefactor: ∂(1/A)/∂ε_ab = −(1/A) P_ab.
            let mut m = Mat3::zero();
            for i in 0..3 {
                for j in 0..3 {
                    let p_ij = if i == j { 1.0 } else { 0.0 } - normal.get(i) * normal.get(j);
                    m.set(i, j, -qq * val * p_ij);
                }
            }
            acc.add_virial(&m);
        }
    }
}

/// The Parry `G ≠ 0` kernel and its derivatives.
///
/// Returns `(value, ∂/∂z, ∂/∂(G·d), ∂/∂G)` for
/// `value = cos(G·d) · [ e^{Gz} erfc(αz + G/2α) + e^{−Gz} erfc(−αz + G/2α) ]`,
/// where the `∂/∂G` is at fixed `z` and fixed `G·d` (the explicit `G` in the erfc arguments and
/// in the exponentials); the strain dependence of `G·d` is zero and is not included here.
///
/// With `u = αz + G/2α` and `v = −αz + G/2α`, expanding the squares gives
/// `Gz − u² = −Gz − v² = −(αz)² − (G/2α)²`. Both products therefore carry the **same**
/// exponential factor `w = exp(−α²z² − G²/4α²)`, which is never positive, so writing them as
/// `erfcx(u)·w` and `erfcx(v)·w` keeps everything in range no matter how large `Gz` is.
fn parry_kernel(g: f64, alpha: f64, z: f64, gdotd: f64) -> (f64, f64, f64, f64) {
    let c = gdotd.cos();
    let s = gdotd.sin();
    let az = alpha * z;
    let g2a = g / (2.0 * alpha);
    let u = az + g2a;
    let v = -az + g2a;
    let w = (-az * az - g2a * g2a).exp();
    let plus = erfcx(u) * w; // e^{Gz} erfc(u)
    let minus = erfcx(v) * w; // e^{−Gz} erfc(v)
    let bracket = plus + minus;
    // ∂/∂z. d/dz[e^{Gz}erfc(u)] = G·plus − (2α/√π)·e^{Gz−u²} = G·plus − (2α/√π)·w
    //       d/dz[e^{−Gz}erfc(v)] = −G·minus + (2α/√π)·e^{−Gz−v²} = −G·minus + (2α/√π)·w
    // The two (2α/√π)·w terms cancel exactly, which is why the z-derivative is this simple.
    let dz = g * (plus - minus);
    // ∂/∂G at fixed z. d/dG[e^{Gz}erfc(u)]  =  z·plus  − (1/(α√π))·w
    //                  d/dG[e^{−Gz}erfc(v)] = −z·minus − (1/(α√π))·w
    let dg = z * (plus - minus) - 2.0 * w / (alpha * SQRT_PI);
    (c * bracket, c * dz, -s * bracket, c * dg)
}

/// 1-D reciprocal space for a wire periodic along `cell`'s single lattice vector.
///
/// ```text
/// E_recip = (1/L) Σ_{G≠0} Σ_{A,B} q_A q_B cos(G x) · Î(G, ρ)
/// Î(G, ρ) = ∫₀^{α²} (dt/t) exp(−ρ² t − G²/4t)
/// ```
///
/// plus the `G = 0` remainder `(1/L) Σ q_A q_B [ −γ − 2 ln ρ − E₁(α²ρ²) ]`, whose divergent
/// `−ln ε` piece is what the neutralizing background removes. `x` is the coordinate along the
/// wire and `ρ` the perpendicular distance.
fn reciprocal_1d(
    cell: &Cell,
    positions: &[Vec3],
    charges: &[f64],
    params: &EwaldParameters,
    acc: &mut Accumulator,
) {
    let axis = cell.vectors()[0].normalized();
    let length = cell.measure();
    let alpha = params.alpha;
    let n = positions.len();
    let quad = gauss_legendre(64);
    let axial = Mat3::outer(axis, axis);
    // The Ewald energy is `½ Σ_{i,j} q_i q_j (…)`, and the loops below run the *full* double sum
    // through `scale`, so the ½ belongs in this prefactor.
    let pref = 0.5 / length;

    for &g in &params.g_vectors {
        let gn = g.norm();
        for a in 0..n {
            for b in a..n {
                let scale = if a == b { 1.0 } else { 2.0 };
                let d = positions[b] - positions[a];
                let x = axis.dot(d);
                let perp = d - axis * x;
                let rho = perp.norm();
                let (i_val, di_drho2, di_dg) = wire_kernel(gn, alpha, rho, &quad);
                let c = (gn * x).cos();
                let s = (gn * x).sin();
                let val = c * i_val * pref;
                let qq = charges[a] * charges[b] * scale;
                // ∂/∂d = (∂/∂x)·â + (∂/∂ρ²)·2·perp
                let dval_dx = -gn * s * i_val * pref;
                let dedd = axis * (qq * dval_dx) + perp * (2.0 * qq * c * di_drho2 * pref);
                acc.add_pair_term(a, b, charges[a], charges[b], scale, val, d, dedd);
                // Strain along the axis changes both `1/L` and `G`:
                //   ∂(1/L)/∂ε_ab = −(1/L) â_a â_b,  ∂G/∂ε_ab = −G â_a â_b.
                let d_dg = qq * pref * (c * di_dg - x * s * i_val); // ∂E_term/∂G
                acc.add_virial(&axial.scaled(-qq * val - gn * d_dg));
            }
        }
    }

    // G = 0 remainder. The full `G = 0` integral carries a `−ln ε` divergence that is
    // independent of both ρ and α; it multiplies `Σ_{i,j} q_i q_j = q_tot²`, so it vanishes
    // identically for a neutral cell and is a pure (infinite) constant for a charged one. It is
    // dropped here with the **cell length as the reference scale**, i.e. the retained kernel is
    // `−γ − ln(ρ²/L²) − E₁(α²ρ²)`. That choice is dimensionally consistent and, crucially,
    // α-independent, so the total energy stays α-independent for a charged wire too. What it
    // cannot do is make the absolute energy of a charged wire meaningful — no convention can.
    let a2 = alpha * alpha;
    let l2 = WIRE_REFERENCE_LENGTH_SQUARED;
    for a in 0..n {
        for b in a..n {
            let scale = if a == b { 1.0 } else { 2.0 };
            let d = positions[b] - positions[a];
            let x = axis.dot(d);
            let perp = d - axis * x;
            let rho2 = perp.norm2();
            let small = rho2 * a2 < 1.0e-12;
            // Continuous at ρ → 0, where E₁(z) → −γ − ln z makes the limit ln(α²L²).
            let kernel = if small {
                (a2 * l2).ln()
            } else {
                -EULER_GAMMA - (rho2 / l2).ln() - exp_integral_e1(a2 * rho2)
            };
            let val = kernel * pref;
            // d/d(ρ²)[−ln ρ² − E₁(α²ρ²)] = (−1 + e^{−α²ρ²})/ρ²  (using E₁'(z) = −e^{−z}/z)
            let dval_drho2 = pref
                * if small {
                    -a2 / 2.0
                } else {
                    (-1.0 + (-a2 * rho2).exp()) / rho2
                };
            let qq = charges[a] * charges[b] * scale;
            let dedd = perp * (2.0 * qq * dval_drho2);
            acc.add_pair_term(a, b, charges[a], charges[b], scale, val, d, dedd);
            // Only the `1/L` prefactor depends on the cell — the kernel's reference length is a
            // constant, not `L` — so `∂/∂ε [pref·kernel] = −pref·kernel â ⊗ â`.
            acc.add_virial(&axial.scaled(-qq * val));
        }
    }
}

/// `Î(G, ρ) = ∫₀^{α²} (dt/t) exp(−ρ²t − G²/4t)` with its `∂/∂ρ²` and `∂/∂G`.
///
/// Substituting `t = e^{u}` turns `dt/t` into `du` and the integrand into
/// `exp(−ρ² e^u − (G²/4) e^{−u})`, which decays **doubly exponentially** at both ends of the
/// `u` range. The lower limit is therefore not a truncation: it is placed where the
/// `e^{−(G²/4)e^{−u}}` factor has already underflowed, so extending it further adds nothing a
/// double can represent. A 64-point Gauss–Legendre rule on that interval is converged to
/// machine precision, which `wire_kernel_matches_adaptive_quadrature` checks against an
/// independent refinement.
fn wire_kernel(g: f64, alpha: f64, rho: f64, quad: &(Vec<f64>, Vec<f64>)) -> (f64, f64, f64) {
    let a2 = alpha * alpha;
    let rho2 = rho * rho;
    let g2_over_4 = g * g / 4.0;
    if g2_over_4 <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    // exp(−(G²/4)/t) underflows below t = (G²/4)/700.
    let t_min = g2_over_4 / 700.0;
    if t_min >= a2 {
        return (0.0, 0.0, 0.0);
    }
    let ln_lo = t_min.ln();
    let ln_hi = a2.ln();
    let half = 0.5 * (ln_hi - ln_lo);
    let mid = 0.5 * (ln_hi + ln_lo);
    let (nodes, weights) = quad;
    let (mut value, mut d_rho2, mut d_g) = (0.0, 0.0, 0.0);
    for (node, w) in nodes.iter().zip(weights) {
        let t = (mid + half * node).exp();
        let f = (-rho2 * t - g2_over_4 / t).exp();
        value += w * f;
        d_rho2 += w * f * (-t);
        d_g += w * f * (-0.5 * g / t);
    }
    (value * half, d_rho2 * half, d_g * half)
}

/// `∂²Î/∂(ρ²)²` of the 1-D reciprocal kernel, on the same quadrature rule as [`wire_kernel`].
///
/// `Î = ∫ (dt/t) e^{−ρ²t − G²/4t}`, so differentiating twice in `ρ²` brings down `t²`. Kept
/// separate from `wire_kernel` because only the Hessian needs it and the first-derivative path is
/// on every gradient's critical path.
fn wire_kernel_second(g: f64, alpha: f64, rho: f64, quad: &(Vec<f64>, Vec<f64>)) -> f64 {
    let a2 = alpha * alpha;
    let rho2 = rho * rho;
    let g2_over_4 = g * g / 4.0;
    if g2_over_4 <= 0.0 {
        return 0.0;
    }
    let t_min = g2_over_4 / 700.0;
    if t_min >= a2 {
        return 0.0;
    }
    let ln_lo = t_min.ln();
    let ln_hi = a2.ln();
    let half = 0.5 * (ln_hi - ln_lo);
    let mid = 0.5 * (ln_hi + ln_lo);
    let (nodes, weights) = quad;
    let mut acc = 0.0;
    for (node, w) in nodes.iter().zip(weights) {
        let t = (mid + half * node).exp();
        acc += w * (-rho2 * t - g2_over_4 / t).exp() * t * t;
    }
    acc * half
}

/// The `z`-dependent part of the Parry 2-D kernel and its first two `z` derivatives, with the
/// in-plane phase factored out.
///
/// [`parry_kernel`] bakes `cos(G·d)` in, which is fine while the sum is real. A phased sum at
/// wavevector `q` needs `e^{-iK·d}` instead, so the two have to come apart.
fn parry_bracket(g: f64, alpha: f64, z: f64) -> (f64, f64, f64) {
    let az = alpha * z;
    let g2a = g / (2.0 * alpha);
    let w = (-az * az - g2a * g2a).exp();
    let plus = erfcx(az + g2a) * w;
    let minus = erfcx(-az + g2a) * w;
    let bracket = plus + minus;
    (
        bracket,
        g * (plus - minus),
        g * (g * bracket - 4.0 * alpha * w / SQRT_PI),
    )
}

/// Second derivatives of the Parry 2-D kernel: `(∂²/∂z², ∂²/∂z∂(G·d), ∂²/∂(G·d)²)`.
///
/// The `z` derivatives use `d/dz[e^{Gz}erfc(u)] = G·plus − (2α/√π)w` and its mirror, exactly as
/// [`parry_kernel`] does; differentiating once more leaves a single uncancelled Gaussian where
/// the first derivative had none.
fn parry_kernel_second(g: f64, alpha: f64, z: f64, gdotd: f64) -> (f64, f64, f64) {
    let c = gdotd.cos();
    let s = gdotd.sin();
    let az = alpha * z;
    let g2a = g / (2.0 * alpha);
    let w = (-az * az - g2a * g2a).exp();
    let plus = erfcx(az + g2a) * w;
    let minus = erfcx(-az + g2a) * w;
    let bracket = plus + minus;
    let dz = g * (plus - minus);
    let dzz = g * (g * bracket - 4.0 * alpha * w / SQRT_PI);
    (c * dzz, -s * dz, -c * bracket)
}

/// Second derivatives of `E = ½ Σ_{A,B} c_AB M_AB` with respect to the atomic positions, as a
/// `3N × 3N` matrix in eV/Bohr².
///
/// Every term depends on the atoms only through a pair displacement `d_AB`, so each contributes a
/// 3×3 block `∂²/∂d∂d` that scatters `+h` onto the two diagonal blocks and `−h` onto the two
/// off-diagonal ones — the same pattern the molecular Hessian uses. A self pair (`A = B`, any
/// image) has a displacement independent of the atom's position, and the four scattered terms
/// cancel to nothing, which is exactly right.
///
/// The enumeration mirrors [`ewald_pair_matrix`] term for term so the Hessian differentiates the
/// energy that function returns rather than a nearby one.
pub fn ewald_pair_hessian(
    cell: &Cell,
    positions: &[Vec3],
    coefficients: &[Vec<f64>],
    params: &EwaldParameters,
) -> crate::linalg::Matrix {
    ewald_pair_hessian_with(cell, positions, coefficients, params, true)
}

/// [`ewald_pair_hessian`], with the reciprocal sum optionally left out.
///
/// The companion of [`ewald_pair_matrix_with`], for the same caller: a Born-von Karman class
/// loop that has already done every class's reciprocal term in one pass over `G` still needs
/// the real-space sum, whose distances change with the translation. See
/// [`ewald_reciprocal_hessian_bvk`].
pub fn ewald_pair_hessian_with(
    cell: &Cell,
    positions: &[Vec3],
    coefficients: &[Vec<f64>],
    params: &EwaldParameters,
    reciprocal: bool,
) -> crate::linalg::Matrix {
    let n = positions.len();
    let alpha = params.alpha;
    let coef = |a: usize, b: usize| coefficients[a][b];
    let mut h = crate::linalg::Matrix::zeros(3 * n, 3 * n);
    let mut scatter = |a: usize, b: usize, block: &Mat3| {
        if a == b {
            return;
        }
        for i in 0..3 {
            for j in 0..3 {
                let v = block.get(i, j);
                h[(3 * a + i, 3 * a + j)] += v;
                h[(3 * b + i, 3 * b + j)] += v;
                h[(3 * a + i, 3 * b + j)] -= v;
                h[(3 * b + i, 3 * a + j)] -= v;
            }
        }
    };

    // --- real space -------------------------------------------------------------------
    // f(r) = erfc(αr)/r, with f' = −(f + E)/r and E = (2α/√π)e^{−α²r²}, so
    // f'' = −2f'/r + 2α²E. The Hessian in `d` follows from the radial decomposition.
    let two_alpha_over_sqrt_pi = 2.0 * alpha / SQRT_PI;
    for a in 0..n {
        for b in a..n {
            let c = coef(a, b);
            if c == 0.0 || a == b {
                continue;
            }
            for &t in &params.real_images {
                let d = positions[b] + cell.translation(t) - positions[a];
                let r = d.norm();
                if r > params.r_cut || r < 1.0e-12 {
                    continue;
                }
                let e = erfc(alpha * r) / r;
                let gauss = two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp();
                let dedr = -(e + gauss) / r;
                let d2edr2 = -2.0 * dedr / r + 2.0 * alpha * alpha * gauss;
                let mut block = Mat3::zero();
                for i in 0..3 {
                    for j in 0..3 {
                        let radial = d.get(i) * d.get(j) / (r * r);
                        let delta = if i == j { 1.0 } else { 0.0 };
                        block.set(i, j, c * (d2edr2 * radial + dedr * (delta - radial) / r));
                    }
                }
                scatter(a, b, &block);
            }
        }
    }

    // --- reciprocal space -------------------------------------------------------------
    //
    // Emptying the `G` set rather than branching around each kernel, as in [`ewald_pair_matrix_with`].
    let g_vectors: &[Vec3] = if reciprocal { &params.g_vectors } else { &[] };
    match cell.dim() {
        3 => {
            let v = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &g in g_vectors {
                let g2 = g.norm2();
                let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
                for a in 0..n {
                    for b in 0..n {
                        let c = coef(a, b);
                        if c == 0.0 || a == b {
                            continue;
                        }
                        let d = positions[b] - positions[a];
                        let factor = -pref * c * g.dot(d).cos();
                        let mut block = Mat3::zero();
                        for i in 0..3 {
                            for j in 0..3 {
                                block.set(i, j, factor * g.get(i) * g.get(j));
                            }
                        }
                        scatter(a, b, &block);
                    }
                }
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &g in g_vectors {
                let gn = g.norm();
                let pref = std::f64::consts::PI / (2.0 * area * gn);
                for a in 0..n {
                    for b in a..n {
                        if coef(a, b) == 0.0 || a == b {
                            continue;
                        }
                        let d = positions[b] - positions[a];
                        let z = normal.dot(d);
                        let (dzz, dzp, dpp) = parry_kernel_second(gn, alpha, z, g.dot(d));
                        let c = coef(a, b) * 2.0;
                        let s = pref * c;
                        let mut block = Mat3::zero();
                        for i in 0..3 {
                            for j in 0..3 {
                                let nn = normal.get(i) * normal.get(j);
                                let ng = normal.get(i) * g.get(j) + g.get(i) * normal.get(j);
                                let gg = g.get(i) * g.get(j);
                                block.set(i, j, s * (dzz * nn + dzp * ng + dpp * gg));
                            }
                        }
                        scatter(a, b, &block);
                    }
                }
            }
            // The G = 0 sheet term: −(π/A)[z erf(αz) + e^{−α²z²}/(α√π)], whose second z
            // derivative is −(π/A)(2α/√π)e^{−α²z²}.
            for a in 0..n {
                for b in a..n {
                    if coef(a, b) == 0.0 || a == b {
                        continue;
                    }
                    let d = positions[b] - positions[a];
                    let az = alpha * normal.dot(d);
                    let dzz =
                        -(std::f64::consts::PI / area) * two_alpha_over_sqrt_pi * (-az * az).exp();
                    let c = coef(a, b) * 2.0;
                    let mut block = Mat3::zero();
                    for i in 0..3 {
                        for j in 0..3 {
                            block.set(i, j, c * dzz * normal.get(i) * normal.get(j));
                        }
                    }
                    scatter(a, b, &block);
                }
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let pref = 0.5 / length;
            let a2 = alpha * alpha;
            // With x = â·d and ρ² = |d − â x|², the chain rule gives
            //   ∂²/∂d∂d = â⊗â ∂ₓₓ + 2(â⊗ρ + ρ⊗â) ∂ₓ∂_{ρ²} + 4 ρ⊗ρ ∂_{ρ²ρ²} + 2 P⊥ ∂_{ρ²}.
            let block_of = |perp: Vec3, dxx: f64, dxr: f64, drr: f64, dr: f64| -> Mat3 {
                let mut m = Mat3::zero();
                for i in 0..3 {
                    for j in 0..3 {
                        let aa = axis.get(i) * axis.get(j);
                        let ap = axis.get(i) * perp.get(j) + perp.get(i) * axis.get(j);
                        let pp = perp.get(i) * perp.get(j);
                        let delta = if i == j { 1.0 } else { 0.0 };
                        m.set(
                            i,
                            j,
                            dxx * aa + 2.0 * dxr * ap + 4.0 * drr * pp + 2.0 * dr * (delta - aa),
                        );
                    }
                }
                m
            };
            for &g in g_vectors {
                let gn = g.norm();
                for a in 0..n {
                    for b in a..n {
                        if coef(a, b) == 0.0 || a == b {
                            continue;
                        }
                        let d = positions[b] - positions[a];
                        let x = axis.dot(d);
                        let perp = d - axis * x;
                        let (i_val, di_drho2, _) = wire_kernel(gn, alpha, perp.norm(), &quad);
                        let d2i = wire_kernel_second(gn, alpha, perp.norm(), &quad);
                        let (cs, sn) = ((gn * x).cos(), (gn * x).sin());
                        let c = coef(a, b) * 2.0 * pref;
                        let block = block_of(
                            perp,
                            -c * gn * gn * cs * i_val,
                            -c * gn * sn * di_drho2,
                            c * cs * d2i,
                            c * cs * di_drho2,
                        );
                        scatter(a, b, &block);
                    }
                }
            }
            // The G = 0 wire term: kernel(u) = −γ − ln(u/l²) − E₁(α²u) with u = ρ², whose first
            // derivative is (−1 + e^{−α²u})/u and second is [1 − e^{−α²u}(1 + α²u)]/u².
            for a in 0..n {
                for b in a..n {
                    if coef(a, b) == 0.0 || a == b {
                        continue;
                    }
                    let d = positions[b] - positions[a];
                    let x = axis.dot(d);
                    let perp = d - axis * x;
                    let u = perp.norm2();
                    let (d1, d2) = if u * a2 < 1.0e-12 {
                        // Series limits: the log and E₁ divergences cancel, leaving
                        // −α² and +α⁴/2.
                        (-a2, 0.5 * a2 * a2)
                    } else {
                        let ex = (-a2 * u).exp();
                        ((-1.0 + ex) / u, (1.0 - ex * (1.0 + a2 * u)) / (u * u))
                    };
                    let c = coef(a, b) * 2.0 * pref;
                    let block = block_of(perp, 0.0, 0.0, c * d2, c * d1);
                    scatter(a, b, &block);
                }
            }
        }
        _ => {}
    }

    for v in h.as_mut_slice() {
        *v *= PM7_EV;
    }
    h
}

/// Self-interaction removal: `−α/√π Σ_A q_A²`.
fn self_term(charges: &[f64], params: &EwaldParameters, acc: &mut Accumulator) {
    let c = params.alpha / SQRT_PI;
    for (i, q) in charges.iter().enumerate() {
        acc.energy -= c * q * q;
        acc.potential[i] -= 2.0 * c * q;
    }
    // No position or strain dependence.
}

/// Uniform neutralizing background for a charged **3-D** cell.
///
/// Only 3-D needs one. In 2-D the Parry `G = 0` term is finite for any net charge — it *is* the
/// linearly growing potential of a charged sheet, which is the physically correct answer for a
/// slab in vacuum. In 1-D the divergence is logarithmic and is removed inside
/// [`reciprocal_1d`] by fixing the reference length to the cell length, which keeps the result
/// α-independent.
///
/// The background is uniform, so it exerts **no force on any atom**, but it scales as `1/V` and
/// therefore contributes an isotropic term to the virial. Leaving that out is the classic
/// silent bug in charged-cell stresses; `charged_cell_stress_matches_finite_difference` exists
/// to catch it.
fn background(
    cell: &Cell,
    charges: &[f64],
    params: &EwaldParameters,
    acc: &mut Accumulator,
) -> f64 {
    let q: f64 = charges.iter().sum();
    if q.abs() < 1.0e-14 || cell.dim() != 3 {
        return 0.0;
    }
    let volume = cell.measure();
    let alpha = params.alpha;
    let e_bg = -std::f64::consts::PI * q * q / (2.0 * alpha * alpha * volume);
    acc.energy += e_bg;
    // E_bg ∝ q², so ∂E_bg/∂q_A = 2 E_bg / q for every atom.
    let dv = 2.0 * e_bg / q;
    for p in acc.potential.iter_mut() {
        *p += dv;
    }
    // ∂(1/V)/∂ε_ab = −(1/V) δ_ab.
    acc.add_virial(&Mat3::identity().scaled(-e_bg));
    e_bg
}

/// Projector onto the periodic subspace: `Σ_{k<dim} â_k ⊗ â_k` for an orthonormalized basis of
/// the periodic directions. Used to restrict `δ_ab` to the directions that actually strain.
fn periodic_projector(cell: &Cell) -> Mat3 {
    let dim = cell.dim();
    let mut basis: Vec<Vec3> = Vec::with_capacity(dim);
    for v in cell.vectors() {
        // Gram-Schmidt against what we already have.
        let mut u = *v;
        for e in &basis {
            u -= *e * e.dot(u);
        }
        let n = u.norm();
        if n > 1.0e-12 {
            basis.push(u / n);
        }
    }
    let mut p = Mat3::zero();
    for e in &basis {
        p = p.plus(&Mat3::outer(*e, *e));
    }
    p
}

/// Whether `t` is in the "positive" half of translation space (lexicographic sign).
#[inline]
fn positive_translation(t: [i32; 3]) -> bool {
    for &c in &t {
        if c > 0 {
            return true;
        }
        if c < 0 {
            return false;
        }
    }
    false
}

/// The same lattice sums as [`ewald`], but weighted by a general symmetric **pair matrix**
/// instead of the outer product of charges:
///
/// ```text
/// E = ½ Σ_{A,B} c_{AB} M_{AB}
/// ```
///
/// This is what the periodic **exchange** needs. Exchange contracts as
/// `−P^σ(μ_A, λ_B) · Σ'_T v(r_AB + T)`, i.e. against a coefficient
/// `c_AB = −Σ_{μλ} P^σ(μ_A, λ_B) P(μ_A, λ_B)`-shaped matrix that does not factorize into a
/// product of per-atom numbers. Everything else is identical: the same real-space `erfc` sum,
/// the same reciprocal kernels, the same self and background terms.
///
/// The one thing lost is the structure-factor trick that makes the 3-D reciprocal sum `O(N)` per
/// `G`; a general coefficient matrix makes it `O(N²)`. That is why the charge path is kept
/// separate rather than being expressed through this one.
///
/// `pair_matrix_reproduces_the_charge_path` checks the two against each other with
/// `c_AB = q_A q_B`, so the duplication cannot drift.
pub fn ewald_pair_matrix(
    cell: &Cell,
    positions: &[Vec3],
    coefficients: &[Vec<f64>],
    params: &EwaldParameters,
) -> EwaldPotential {
    ewald_pair_matrix_with(cell, positions, coefficients, params, true)
}

/// [`ewald_pair_matrix`], with the reciprocal sum optionally left out.
///
/// A caller that evaluates the *same* lattice sum for many coefficient matrices which differ only
/// by a lattice translation — which is exactly what the Born–von Kármán residue classes of the
/// long-range exchange are — can do every reciprocal term for every class in one pass over `G`
/// (see [`ewald_reciprocal_bvk`]). It then needs this function for everything else: the real-space
/// sum, whose distances genuinely change with the translation, and the self and background terms.
///
/// `reciprocal: true` is the ordinary entry point and is what [`ewald_pair_matrix`] calls, so the
/// single-shot behaviour is unchanged.
pub fn ewald_pair_matrix_with(
    cell: &Cell,
    positions: &[Vec3],
    coefficients: &[Vec<f64>],
    params: &EwaldParameters,
    reciprocal: bool,
) -> EwaldPotential {
    let n = positions.len();
    let alpha = params.alpha;
    let mut acc = Accumulator::new(n);
    let coef = |a: usize, b: usize| coefficients[a][b];

    // --- real space -------------------------------------------------------------------
    let two_alpha_over_sqrt_pi = 2.0 * alpha / SQRT_PI;
    for a in 0..n {
        for b in a..n {
            // A zero coefficient contributes nothing anywhere, and skipping it early is what
            // makes the shifted-pair construction in
            // `gradient::long_range_exchange_derivatives` affordable: it doubles the atom list
            // and leaves three quarters of the coefficient matrix empty.
            if coef(a, b) == 0.0 {
                continue;
            }
            for &t in &params.real_images {
                if a == b && (t == [0, 0, 0] || !positive_translation(t)) {
                    continue;
                }
                let d = positions[b] + cell.translation(t) - positions[a];
                let r = d.norm();
                if r > params.r_cut || r < 1.0e-12 {
                    continue;
                }
                let e = erfc(alpha * r) / r;
                let dedr = -(e + two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp()) / r;
                // The `½ Σ_{A,B}` double sum visits each unordered pair twice; the enumeration
                // here visits it once, hence `coef(a,b)` with no extra ½ for a ≠ b, and the same
                // for a self pair (which stands for both ±T).
                let c = coef(a, b);
                let dedd = d * (c * dedr / r);
                acc.add_pair_term(a, b, 1.0, 1.0, c, e, d, dedd);
            }
        }
    }

    // --- reciprocal space -------------------------------------------------------------
    //
    // Emptying the `G` set rather than branching around each of the three kernels: the 2-D and 1-D
    // arms carry a second, `G`-free correction loop after the reciprocal one, and that loop is
    // *not* part of what a Born–von Kármán caller does elsewhere. Skipping the sum by giving it
    // nothing to sum over keeps that distinction impossible to get wrong.
    let g_vectors: &[Vec3] = if reciprocal { &params.g_vectors } else { &[] };
    match cell.dim() {
        3 => {
            let v = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &g in g_vectors {
                let g2 = g.norm2();
                let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
                // Σ_{A,B} c_AB cos(G·d_AB). Without a rank-1 coefficient this cannot be folded
                // into |S(G)|², so it is an explicit double loop.
                let mut sum = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        let c = coef(a, b);
                        if c == 0.0 {
                            continue;
                        }
                        let d = positions[b] - positions[a];
                        let phase = g.dot(d);
                        sum += c * phase.cos();
                        // ∂/∂R_b of cos(G·d) is −sin(G·d) G; the a-derivative is its negative.
                        let f = g * (-pref * c * phase.sin());
                        acc.gradient[b] += f;
                        acc.gradient[a] -= f;
                    }
                }
                acc.energy += pref * sum;
                // Same strain algebra as the charge path: G·d is strain-invariant, so only the
                // 1/V, e^{−G²/4α²}, and 1/G² factors contribute.
                let e_term = pref * sum;
                let cc = 2.0 * (1.0 / g2 + inv_4a2);
                let mut m = Mat3::zero();
                for i in 0..3 {
                    for j in 0..3 {
                        let delta = if i == j { 1.0 } else { 0.0 };
                        m.set(i, j, e_term * (cc * g.get(i) * g.get(j) - delta));
                    }
                }
                acc.add_virial(&m);
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &g in g_vectors {
                let gn = g.norm();
                let pref = std::f64::consts::PI / (2.0 * area * gn);
                for a in 0..n {
                    for b in a..n {
                        if coef(a, b) == 0.0 {
                            continue;
                        }
                        let scale = if a == b { 1.0 } else { 2.0 };
                        let d = positions[b] - positions[a];
                        let z = normal.dot(d);
                        let (val, dval_dz, dval_dpar, dval_dg) =
                            parry_kernel(gn, alpha, z, g.dot(d));
                        let c = coef(a, b) * scale;
                        let dedd = normal * (pref * c * dval_dz) + g * (pref * c * dval_dpar);
                        acc.add_pair_term(a, b, 1.0, 1.0, c, pref * val, d, dedd);
                        let e_term = pref * c * val;
                        let dg_term = pref * c * dval_dg;
                        let dpar_term = pref * c * dval_dpar;
                        let mut m = Mat3::zero();
                        for i in 0..3 {
                            for j in 0..3 {
                                let p_ij =
                                    if i == j { 1.0 } else { 0.0 } - normal.get(i) * normal.get(j);
                                let ghat = g.get(i) * g.get(j) / gn;
                                m.set(
                                    i,
                                    j,
                                    e_term * (ghat / gn - p_ij)
                                        - dg_term * ghat
                                        - dpar_term * g.get(i) * d.get(j),
                                );
                            }
                        }
                        acc.add_virial(&m);
                    }
                }
            }
            for a in 0..n {
                for b in a..n {
                    let scale = if a == b { 1.0 } else { 2.0 };
                    let d = positions[b] - positions[a];
                    let z = normal.dot(d);
                    let az = alpha * z;
                    let erf_az = 1.0 - erfc(az);
                    let val = -(std::f64::consts::PI / area)
                        * (z * erf_az + (-az * az).exp() / (alpha * SQRT_PI));
                    let dval_dz = -(std::f64::consts::PI / area) * erf_az;
                    let c = coef(a, b) * scale;
                    let dedd = normal * (c * dval_dz);
                    acc.add_pair_term(a, b, 1.0, 1.0, c, val, d, dedd);
                    let mut m = Mat3::zero();
                    for i in 0..3 {
                        for j in 0..3 {
                            let p_ij =
                                if i == j { 1.0 } else { 0.0 } - normal.get(i) * normal.get(j);
                            m.set(i, j, -c * val * p_ij);
                        }
                    }
                    acc.add_virial(&m);
                }
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let axial = Mat3::outer(axis, axis);
            let pref = 0.5 / length;
            let a2 = alpha * alpha;
            let l2 = WIRE_REFERENCE_LENGTH_SQUARED;
            for &g in g_vectors {
                let gn = g.norm();
                for a in 0..n {
                    for b in a..n {
                        if coef(a, b) == 0.0 {
                            continue;
                        }
                        let scale = if a == b { 1.0 } else { 2.0 };
                        let d = positions[b] - positions[a];
                        let x = axis.dot(d);
                        let perp = d - axis * x;
                        let (i_val, di_drho2, di_dg) = wire_kernel(gn, alpha, perp.norm(), &quad);
                        let cs = (gn * x).cos();
                        let sn = (gn * x).sin();
                        let val = cs * i_val * pref;
                        let c = coef(a, b) * scale;
                        let dedd = axis * (c * (-gn * sn * i_val * pref))
                            + perp * (2.0 * c * cs * di_drho2 * pref);
                        acc.add_pair_term(a, b, 1.0, 1.0, c, val, d, dedd);
                        let d_dg = c * pref * (cs * di_dg - x * sn * i_val);
                        acc.add_virial(&axial.scaled(-c * val - gn * d_dg));
                    }
                }
            }
            for a in 0..n {
                for b in a..n {
                    if coef(a, b) == 0.0 {
                        continue;
                    }
                    let scale = if a == b { 1.0 } else { 2.0 };
                    let d = positions[b] - positions[a];
                    let x = axis.dot(d);
                    let perp = d - axis * x;
                    let rho2 = perp.norm2();
                    let small = rho2 * a2 < 1.0e-12;
                    let kernel = if small {
                        (a2 * l2).ln()
                    } else {
                        -EULER_GAMMA - (rho2 / l2).ln() - exp_integral_e1(a2 * rho2)
                    };
                    let val = kernel * pref;
                    // d/du of `−ln(u/l²) − E₁(α²u)` is `(−1 + e^{−α²u})/u`, whose `u → 0` limit is
                    // `−α²` (the log and E₁ singularities cancel). The gradient multiplies this by
                    // the perpendicular offset, which vanishes with `u`, so the limit is invisible
                    // here — but `ewald_pair_hessian` picks it up undamped through the `2 P⊥ ∂_u`
                    // term, and a wire has every on-axis pair sitting exactly at `u = 0`.
                    let dval_drho2 = pref
                        * if small {
                            -a2
                        } else {
                            (-1.0 + (-a2 * rho2).exp()) / rho2
                        };
                    let c = coef(a, b) * scale;
                    let dedd = perp * (2.0 * c * dval_drho2);
                    acc.add_pair_term(a, b, 1.0, 1.0, c, val, d, dedd);
                    acc.add_virial(&axial.scaled(-c * val));
                }
            }
        }
        _ => {}
    }

    // --- self and background ------------------------------------------------------------
    let self_c = alpha / SQRT_PI;
    for a in 0..n {
        acc.energy -= self_c * coef(a, a);
    }
    let total: f64 = (0..n)
        .flat_map(|a| (0..n).map(move |b| (a, b)))
        .map(|(a, b)| coef(a, b))
        .sum();
    let background_energy = if cell.dim() == 3 && total.abs() > 1.0e-14 {
        let e_bg = -std::f64::consts::PI * total / (2.0 * alpha * alpha * cell.measure());
        acc.energy += e_bg;
        acc.add_virial(&Mat3::identity().scaled(-e_bg));
        e_bg
    } else {
        0.0
    };

    let p = periodic_projector(cell);
    let virial = p.mul_mat(&acc.virial).mul_mat(&p);
    EwaldPotential {
        potential: vec![0.0; n], // not meaningful for a general coefficient matrix
        energy: acc.energy * PM7_EV,
        background_energy: background_energy * PM7_EV,
        gradient: acc.gradient.iter().map(|g| *g * PM7_EV).collect(),
        virial: virial.scaled(PM7_EV),
        total_charge: 0.0,
    }
}

/// The translation structure factors `S_AB(m) = Σ_t c_AB(t) e^{2πi Σ_j m_j t_j / n_j}`.
///
/// Returned as separate real and imaginary arrays laid out `[class][a * n + b]`, matching the
/// class enumeration of [`crate::hamiltonian::bvk_representatives`]. That enumeration folds each
/// index into the symmetric range, but it visits them in plain `(i, j, k)` order and the phase is
/// periodic in each index with exactly that period, so the raw enumeration index is the right one
/// to transform over.
///
/// The transform separates into one pass per axis, `O(N² C (n₁+n₂+n₃))` rather than the `O(N² C²)`
/// a direct evaluation would cost — which matters, because an `O(C²)` step here would simply move
/// the problem this whole construction exists to remove.
fn bvk_structure_factors(
    divisions: [usize; 3],
    n: usize,
    coefficients: &[Vec<Vec<f64>>],
) -> (Vec<f64>, Vec<f64>) {
    let classes = divisions[0] * divisions[1] * divisions[2];
    debug_assert_eq!(coefficients.len(), classes);
    let stride = n * n;
    let mut re: Vec<f64> = vec![0.0; classes * stride];
    let mut im: Vec<f64> = vec![0.0; classes * stride];
    for (t, matrix) in coefficients.iter().enumerate() {
        let dst = &mut re[t * stride..(t + 1) * stride];
        for (a, row) in matrix.iter().enumerate() {
            dst[a * n..a * n + n].copy_from_slice(&row[..n]);
        }
    }

    let [n0, n1, n2] = divisions;
    let at = |i: usize, j: usize, k: usize| ((i * n1) + j) * n2 + k;
    let mut work_re = vec![0.0; classes * stride];
    let mut work_im = vec![0.0; classes * stride];

    // `index` maps a position along the axis being transformed, plus the two spectator indices,
    // to a class index; `outer` and `inner` are the spectators' extents.
    let pass = |len: usize,
                index: &dyn Fn(usize, usize, usize) -> usize,
                outer: usize,
                inner: usize,
                re: &mut Vec<f64>,
                im: &mut Vec<f64>,
                work_re: &mut [f64],
                work_im: &mut [f64]| {
        if len == 1 {
            return;
        }
        let table: Vec<(f64, f64)> = (0..len)
            .map(|p| {
                let phase = std::f64::consts::TAU * p as f64 / len as f64;
                (phase.cos(), phase.sin())
            })
            .collect();
        for o in 0..outer {
            for i in 0..inner {
                for m in 0..len {
                    let dst = index(m, o, i) * stride;
                    work_re[dst..dst + stride].fill(0.0);
                    work_im[dst..dst + stride].fill(0.0);
                    for u in 0..len {
                        let (c, s) = table[(m * u) % len];
                        let src = index(u, o, i) * stride;
                        for e in 0..stride {
                            let (xr, xi) = (re[src + e], im[src + e]);
                            work_re[dst + e] += xr * c - xi * s;
                            work_im[dst + e] += xr * s + xi * c;
                        }
                    }
                }
            }
        }
        re.copy_from_slice(work_re);
        im.copy_from_slice(work_im);
    };

    pass(
        n0,
        &|m, j, k| at(m, j, k),
        n1,
        n2,
        &mut re,
        &mut im,
        &mut work_re,
        &mut work_im,
    );
    pass(
        n1,
        &|m, i, k| at(i, m, k),
        n0,
        n2,
        &mut re,
        &mut im,
        &mut work_re,
        &mut work_im,
    );
    pass(
        n2,
        &|m, i, j| at(i, j, m),
        n0,
        n1,
        &mut re,
        &mut im,
        &mut work_re,
        &mut work_im,
    );
    (re, im)
}

/// Which structure-factor entry a supercell reciprocal vector reads.
///
/// `G·a_j = 2π m_j / n_j` exactly for a supercell reciprocal vector and a primitive lattice
/// vector `a_j`, so rounding recovers the integer. A `G` that is not one would land between grid
/// points; that is a caller error rather than something to interpolate.
fn bvk_slot(g: Vec3, vectors: &[Vec3], divisions: [usize; 3]) -> usize {
    let tau = std::f64::consts::TAU;
    let mut slot = 0usize;
    for (axis, &len) in divisions.iter().enumerate() {
        let m = (g.dot(vectors[axis]) * len as f64 / tau).round() as i64;
        slot = slot * len + m.rem_euclid(len as i64) as usize;
    }
    slot
}

/// The 3-D reciprocal **second** derivative for every Born–von Kármán residue class at once.
///
/// [`ewald_reciprocal_bvk`] for the force-constant matrix. The reciprocal Hessian reaches the
/// separation through `cos(G·d)` alone, so the same translation structure factor collapses the
/// class loop — see that function for the derivation and for why this is three-dimensional only.
///
/// The result is already folded onto the `n` home atoms. A pair and its own translated image
/// (`a == b`, separation `T_t`) cancels to zero under that fold, which is the same thing the
/// `a == b` guard in the single-shot scatter does, and for the same reason: an atom and its images
/// move together, so their relative separation is not a degree of freedom.
pub fn ewald_reciprocal_hessian_bvk(
    cell: &Cell,
    super_cell: &Cell,
    divisions: [usize; 3],
    positions: &[Vec3],
    coefficients: &[Vec<Vec<f64>>],
    params: &EwaldParameters,
) -> crate::linalg::Matrix {
    let n = positions.len();
    debug_assert_eq!(super_cell.dim(), 3);
    let (re, im) = bvk_structure_factors(divisions, n, coefficients);
    let stride = n * n;

    let alpha = params.alpha;
    let v = super_cell.measure();
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let vectors = cell.vectors();
    let tau = std::f64::consts::TAU;
    let mut h = crate::linalg::Matrix::zeros(3 * n, 3 * n);

    for &g in &params.g_vectors {
        let g2 = g.norm2();
        let pref = tau / v * (-g2 * inv_4a2).exp() / g2;
        let slot = bvk_slot(g, vectors, divisions);
        let s_re = &re[slot * stride..slot * stride + stride];
        let s_im = &im[slot * stride..slot * stride + stride];

        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let e = a * n + b;
                let (sr, si) = (s_re[e], s_im[e]);
                if sr == 0.0 && si == 0.0 {
                    continue;
                }
                let phase = g.dot(positions[b] - positions[a]);
                let factor = -pref * (sr * phase.cos() - si * phase.sin());
                for i in 0..3 {
                    for j in 0..3 {
                        let value = factor * g.get(i) * g.get(j);
                        h[(3 * a + i, 3 * a + j)] += value;
                        h[(3 * b + i, 3 * b + j)] += value;
                        h[(3 * a + i, 3 * b + j)] -= value;
                        h[(3 * b + i, 3 * a + j)] -= value;
                    }
                }
            }
        }
    }

    for x in h.as_mut_slice() {
        *x *= PM7_EV;
    }
    h
}

/// The 3-D reciprocal sum for **every Born–von Kármán residue class at once**.
///
/// The long-range exchange derivative evaluates the same supercell lattice sum once per residue
/// class `t`, with the pair separation shifted by that class's lattice translation `T_t`:
///
/// ```text
/// E = Σ_t Σ_{A,B} c_AB(t) · M(R_B − R_A + T_t)
/// ```
///
/// Done class by class that costs `O(C · |G| · N²)`, and `|G| ∝ C` because the reciprocal lattice
/// of the supercell is `C` times denser than the primitive one — so the whole thing grows as the
/// **square** of the k mesh. Measured on diamond, the local exponent in `C` climbs from 1.18 at
/// `C = 27` to 1.93 at `C = 343`, where the derivative already costs ten times its own SCF.
///
/// The reciprocal kernels reach the separation only through `cos(G·d)` and `sin(G·d)`, and a
/// lattice translation shifts that argument without touching anything else:
///
/// ```text
/// Σ_t c_AB(t) cos(G·d + G·T_t) = Re S_AB(G) · cos(G·d) − Im S_AB(G) · sin(G·d)
/// Σ_t c_AB(t) sin(G·d + G·T_t) = Re S_AB(G) · sin(G·d) + Im S_AB(G) · cos(G·d)
/// ```
///
/// with the **translation structure factor**
///
/// ```text
/// S_AB(G) = Σ_t c_AB(t) e^{i G·T_t}
/// ```
///
/// which turns the class loop into a rotation of the `(cos, sin)` pair the kernel already
/// computes. `G·T_t = 2π Σ_j m_j t_j / n_j` for a supercell reciprocal vector, so `S` depends on
/// `G` only through `m mod n` and takes **`C` distinct values however many `G` there are**. Those
/// `C` values are a 3-D DFT of `c_AB(t)`, and it separates into three one-dimensional passes:
/// `O(N² · C · (n₁+n₂+n₃))` rather than the `O(N² C²)` a direct evaluation would cost.
///
/// What remains is one pass over `G` at `O(|G| · N²)`, linear in `C`.
///
/// # Note on the earlier rejected structure factor
///
/// `docs/performance.md` records an attempt to fold this sum into `|S(G)|²` the way the *charge*
/// path does, and why it failed: a general coefficient matrix has no rank-1 factorization over
/// atoms, so the `N²` pair loop cannot be removed. That is still true and this does not attempt
/// it. The factorization here is over **lattice translations**, a different index entirely, and
/// the phases it introduces are rational multiples of `2π` — bounded, exactly periodic, and free
/// of the large-argument trigonometric range reduction that made the atom-pair version lossy.
///
/// # Scope
///
/// 3-D only. The order reduction is in `C = n₁n₂n₃`, and only a three-dimensional mesh makes `C`
/// large enough for the square to hurt: a wire with a 20-point mesh has `C = 20`. The 1-D and 2-D
/// virial kernels also carry the separation *explicitly* (`− dpar·G_i·d_j` in the Parry kernel),
/// which shifts with `T_t` and would need three more transforms — real work for a case that does
/// not have the problem. Callers keep the class loop for those dimensionalities.
///
/// `cell` is the **primitive** cell (whose vectors define `T_t`), `super_cell` the Born–von Kármán
/// supercell the sum is periodic in. `coefficients[i]` is the pair matrix of the class
/// `classes[i]`, in the order [`crate::hamiltonian::bvk_representatives`] returns.
pub fn ewald_reciprocal_bvk(
    cell: &Cell,
    super_cell: &Cell,
    divisions: [usize; 3],
    positions: &[Vec3],
    coefficients: &[Vec<Vec<f64>>],
    params: &EwaldParameters,
) -> (f64, Vec<Vec3>, Mat3) {
    let n = positions.len();
    let classes = divisions[0] * divisions[1] * divisions[2];
    debug_assert_eq!(coefficients.len(), classes);
    debug_assert_eq!(super_cell.dim(), 3);

    // `S[m][a][b] = Σ_t c_AB(t) e^{2πi (m1t1/n1 + m2t2/n2 + m3t3/n3)}`; see
    // [`bvk_structure_factors`] for why the class enumeration index is the one to transform over.
    let stride = n * n;
    let (re, im) = bvk_structure_factors(divisions, n, coefficients);

    // --- one pass over G --------------------------------------------------------------
    let alpha = params.alpha;
    let v = super_cell.measure();
    let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
    let vectors = cell.vectors();
    let tau = std::f64::consts::TAU;

    let mut energy = 0.0_f64;
    let mut gradient = vec![Vec3::zero(); n];
    let mut virial = Mat3::zero();

    // `d[a][b] = R_b − R_a`, hoisted: the inner loop reads it once per G.
    let separations: Vec<Vec3> = (0..n)
        .flat_map(|a| (0..n).map(move |b| (a, b)))
        .map(|(a, b)| positions[b] - positions[a])
        .collect();

    for &g in &params.g_vectors {
        let g2 = g.norm2();
        let pref = tau / v * (-g2 * inv_4a2).exp() / g2;

        // Which of the `C` transform entries this `G` reads. `G·a_j = 2π m_j / n_j` exactly, so
        // rounding recovers an integer; a `G` that is not a supercell reciprocal vector would
        // land between grid points and is a caller error, not something to interpolate.
        let mut slot = 0usize;
        for (axis, &len) in divisions.iter().enumerate() {
            let m = (g.dot(vectors[axis]) * len as f64 / tau).round() as i64;
            slot = slot * len + m.rem_euclid(len as i64) as usize;
        }
        let s_re = &re[slot * stride..slot * stride + stride];
        let s_im = &im[slot * stride..slot * stride + stride];

        let mut sum = 0.0;
        for a in 0..n {
            for b in 0..n {
                let e = a * n + b;
                let (sr, si) = (s_re[e], s_im[e]);
                if sr == 0.0 && si == 0.0 {
                    continue;
                }
                let phase = g.dot(separations[e]);
                let (c, s) = (phase.cos(), phase.sin());
                // The rotation: what the whole class loop would have summed, in one term.
                sum += sr * c - si * s;
                let f = g * (-pref * (sr * s + si * c));
                // A self pair (`a == b`, separation `T_t`) lands both of these on the same atom
                // and cancels, which is the same net zero the class loop produced by folding the
                // displaced copy's force back onto its home atom.
                gradient[b] += f;
                gradient[a] -= f;
            }
        }

        energy += pref * sum;
        let e_term = pref * sum;
        let cc = 2.0 * (1.0 / g2 + inv_4a2);
        let mut m = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                m.set(i, j, e_term * (cc * g.get(i) * g.get(j) - delta));
            }
        }
        virial = virial.plus(&m);
    }

    let p = periodic_projector(super_cell);
    (
        energy * PM7_EV,
        gradient.iter().map(|x| *x * PM7_EV).collect(),
        p.mul_mat(&virial).mul_mat(&p).scaled(PM7_EV),
    )
}

/// The periodic Coulomb interaction **matrix** `M`, in eV per unit charge squared, such that
///
/// ```text
/// E = ½ Σ_{A,B} q_A M_{AB} q_B        and        V_A = ∂E/∂q_A = (M q)_A
/// ```
///
/// The atomic charges change every SCF iteration but the geometry does not, so building `M`
/// once and contracting it per iteration turns the Ewald cost from "a lattice sum per iteration"
/// into "one matrix-vector product per iteration". `M` is symmetric and includes the self term
/// and, for a charged 3-D cell, the neutralizing background — which is why `M_AB` carries a
/// constant shift when the background is active.
///
/// `ewald_matrix_reproduces_the_direct_energy` checks this against [`ewald`] rather than
/// trusting the duplication.
pub fn ewald_matrix(cell: &Cell, positions: &[Vec3], params: &EwaldParameters) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let n = positions.len();
    let mut m = vec![vec![0.0_f64; n]; n];
    let alpha = params.alpha;

    // Real space: erfc(αr)/r over the same pair enumeration `real_space` uses.
    for a in 0..n {
        for b in a..n {
            for &t in &params.real_images {
                if a == b && (t == [0, 0, 0] || !positive_translation(t)) {
                    continue;
                }
                let d = positions[b] + cell.translation(t) - positions[a];
                let r = d.norm();
                if r > params.r_cut || r < 1.0e-12 {
                    continue;
                }
                let e = erfc(alpha * r) / r;
                m[a][b] += e;
                if a != b {
                    m[b][a] += e;
                } else {
                    // A self-image pair contributes `q_A² e`, i.e. `2 · ½ q_A M_AA q_A`.
                    m[a][a] += e;
                }
            }
        }
    }

    match cell.dim() {
        3 => {
            let v = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &g in &params.g_vectors {
                let g2 = g.norm2();
                let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
                // `E_rec = pref Σ_{A,B} q_A q_B cos(G·d)` is the *full* double sum, while
                // `E = ½ Σ_{A,B} q_A M_AB q_B` carries a ½ — hence the factor 2 here and in
                // every other reciprocal block below.
                //
                // Parallel by row: each row is written by exactly one thread and the `G` loop
                // stays serial, so the arithmetic is element-for-element what the scalar version
                // did and the answer does not depend on the thread count.
                //
                // # The factorization that is deliberately not taken
                //
                // `cos(G·(R_b − R_a)) = cos(G·R_a)cos(G·R_b) + sin(G·R_a)sin(G·R_b)` separates the
                // structure factor and would make this **N** transcendentals per `G` instead of
                // `N²`, leaving two multiply-adds in the inner loop. It was implemented, measured
                // and removed: it is faster and **less accurate**. `G·R_a` grows with `|G|` and
                // with how far the atom sits from the origin, so each `sin`/`cos` pays a
                // large-argument reduction, while `G·(R_b − R_a)` is bounded by `|G|` times the
                // cell diameter wherever the cell is placed. The lost digits are small, but they
                // are in the matrix every periodic SCF iterates against, and they were enough to
                // stop a rattled Γ-point cell converging inside its iteration budget —
                // `tests/scf_convergence.rs` caught it. Fewer transcendentals is not worth an SCF
                // that does not converge.
                m.par_iter_mut().enumerate().for_each(|(a, row)| {
                    let pa = positions[a];
                    for (b, entry) in row.iter_mut().enumerate() {
                        *entry += 2.0 * pref * g.dot(positions[b] - pa).cos();
                    }
                });
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &g in &params.g_vectors {
                let gn = g.norm();
                let pref = std::f64::consts::PI / (2.0 * area * gn);
                // Parry's slab kernel does not separate the way the 3-D cosine does — it mixes the
                // in-plane and out-of-plane displacements through an `erfc` — so the count stays
                // `N²`. The rows are still independent, and each is written by one thread, so the
                // parallelization is exact rather than a reduction over an unspecified order.
                m.par_iter_mut().enumerate().for_each(|(a, row)| {
                    for (b, entry) in row.iter_mut().enumerate() {
                        let d = positions[b] - positions[a];
                        let (val, _, _, _) = parry_kernel(gn, alpha, normal.dot(d), g.dot(d));
                        *entry += 2.0 * pref * val;
                    }
                });
            }
            for a in 0..n {
                for b in 0..n {
                    let z = normal.dot(positions[b] - positions[a]);
                    let az = alpha * z;
                    m[a][b] -= 2.0
                        * (std::f64::consts::PI / area)
                        * (z * (1.0 - erfc(az)) + (-az * az).exp() / (alpha * SQRT_PI));
                }
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let pref = 0.5 / length;
            let a2 = alpha * alpha;
            let l2 = WIRE_REFERENCE_LENGTH_SQUARED;
            for &g in &params.g_vectors {
                let gn = g.norm();
                for a in 0..n {
                    for b in 0..n {
                        let d = positions[b] - positions[a];
                        let x = axis.dot(d);
                        let rho = (d - axis * x).norm();
                        let (i_val, _, _) = wire_kernel(gn, alpha, rho, &quad);
                        m[a][b] += 2.0 * pref * (gn * x).cos() * i_val;
                    }
                }
            }
            for a in 0..n {
                for b in 0..n {
                    let d = positions[b] - positions[a];
                    let x = axis.dot(d);
                    let rho2 = (d - axis * x).norm2();
                    let kernel = if rho2 * a2 < 1.0e-12 {
                        (a2 * l2).ln()
                    } else {
                        -EULER_GAMMA - (rho2 / l2).ln() - exp_integral_e1(a2 * rho2)
                    };
                    m[a][b] += 2.0 * pref * kernel;
                }
            }
        }
        _ => {}
    }

    // Self term: −α/√π Σ q² means M_AA gets −2α/√π.
    let c = 2.0 * alpha / SQRT_PI;
    for (a, row) in m.iter_mut().enumerate() {
        row[a] -= c;
    }

    // Neutralizing background (3-D only): E_bg = −π q_tot²/(2α²V) = ½ Σ_AB q_A (−π/(α²V)) q_B.
    if cell.dim() == 3 {
        let shift = -std::f64::consts::PI / (alpha * alpha * cell.measure());
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v += shift;
            }
        }
    }

    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v *= PM7_EV;
        }
    }
    m
}

/// The phased lattice sum and its first two derivatives with respect to the displacement.
///
/// ```text
/// Φ_q(d) = Σ'_T e^{i q·T} / |d + T|
/// ```
///
/// This is what a phonon at wavevector `q` needs from the long-range Coulomb: the ordinary lattice
/// sum weights every image equally, and a `q ≠ 0` displacement pattern does not.
///
/// Everything is complex, stored as `[re, im]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhasedKernel {
    pub value: [f64; 2],
    /// `∂Φ_q/∂d`.
    pub gradient: [[f64; 2]; 3],
    /// `∂²Φ_q/∂d∂d`.
    pub hessian: [[[f64; 2]; 3]; 3],
}

/// Reciprocal vectors shifted by `q`, selected on `|G + q| ≤ g_cut`.
///
/// Selecting on `|G|` instead — reusing the stored `g_vectors` and adding `q` afterwards — would
/// make the truncation window move with `q`, and `Φ_q` would then stop being periodic in `q` under
/// a reciprocal-lattice vector. It is periodic by construction, so a numerical answer that is not
/// says the sum is being taken over the wrong set.
///
/// At `q = 0` this reproduces the stored set exactly, `G = 0` excluded; away from it, `K = q` is an
/// ordinary member and comes in on its own.
fn shifted_reciprocal(cell: &Cell, q: Vec3, g_cut: f64) -> Vec<Vec3> {
    let dim = cell.dim();
    if dim == 0 {
        return Vec::new();
    }
    let b = cell.reciprocal_2pi();
    let reach = g_cut + q.norm();
    let mut n = [0i32; 3];
    for (k, slot) in n.iter_mut().enumerate().take(dim) {
        *slot = (reach / b[k].norm()).ceil() as i32 + 1;
    }
    let g2 = g_cut * g_cut;
    let mut out = Vec::new();
    for i in -n[0]..=n[0] {
        for j in -n[1]..=n[1] {
            for k in -n[2]..=n[2] {
                let v = b[0] * i as f64 + b[1] * j as f64 + b[2] * k as f64 + q;
                let norm2 = v.norm2();
                if norm2 <= g2 && norm2 > 1.0e-20 {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// `true` when `q` is a reciprocal lattice vector of `cell`, `q = 0` included.
///
/// These are exactly the wavevectors at which every phase `e^{iq·T}` is 1, so `Φ_q` has to reduce
/// to the unphased sum — the zero wavevector is then a member of the reciprocal set and carries the
/// special treatment (the 2-D sheet term, the 1-D log term, the 3-D neutralizing background) that Γ
/// does.
///
/// Testing `q == 0` instead is the silent version of this test. [`shifted_reciprocal`] drops
/// `|G + q| ≈ 0` from the sum, so if nothing puts that term back the answer is short by it — and
/// `Φ_q` stops being periodic in `q`, which is the property the whole construction rests on. It is
/// not a corner case either: the long-range exchange sums a **supercell** lattice at a `q`
/// commensurate with the k mesh, and every such `q` is a supercell reciprocal lattice vector.
fn is_reciprocal_lattice_vector(cell: &Cell, q: Vec3) -> bool {
    let dim = cell.dim();
    if dim == 0 {
        return true;
    }
    let b = cell.reciprocal_2pi();
    let vectors = cell.vectors();
    let mut g = Vec3::zero();
    for k in 0..dim {
        g += b[k] * (q.dot(vectors[k]) / std::f64::consts::TAU).round();
    }
    // Compare against the shortest reciprocal vector so the test is scale free, and so a `q` that
    // merely sits *near* a lattice vector is correctly rejected: `Φ_q` really is discontinuous
    // there, and rounding it into place would hide physics rather than noise.
    let scale = (0..dim).map(|k| b[k].norm()).fold(f64::INFINITY, f64::min);
    (q - g).norm() < 1.0e-9 * scale
}

#[inline]
fn cadd(acc: &mut [f64; 2], scale: [f64; 2], real: f64) {
    acc[0] += scale[0] * real;
    acc[1] += scale[1] * real;
}

/// `Σ'_T e^{iq·T} v(d+T)` and its derivatives, at each displacement, in eV.
///
/// `q` is a **Cartesian** wavevector lying in the periodic subspace. At `q = 0` this reproduces
/// [`ewald_potentials_at`] exactly, background and self term included — a fact
/// `phased_kernel_reduces_to_the_unphased_sum_at_gamma` checks rather than assumes.
///
/// Away from `q = 0` the reciprocal sum is *simpler*, not harder: `|G + q|` never vanishes, so
/// there is no divergent `G = 0` term, no neutralizing background, and no special case. The price
/// is that `|G + q| ≠ |−G + q|`, so the sum has to run over the full set of reciprocal vectors and
/// cannot be folded onto cosines.
pub fn ewald_phased(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParameters,
) -> Vec<PhasedKernel> {
    let alpha = params.alpha;
    // Not `q == 0`: every `q` in the reciprocal lattice phases the sum by 1 and needs the same
    // treatment. See [`is_reciprocal_lattice_vector`].
    let gamma = is_reciprocal_lattice_vector(cell, q);
    let mut out = vec![PhasedKernel::default(); displacements.len()];
    let two_alpha_over_sqrt_pi = 2.0 * alpha / SQRT_PI;

    // --- real space -------------------------------------------------------------------
    for (i, d) in displacements.iter().enumerate() {
        let slot = &mut out[i];
        for &t in &params.real_images {
            let shift = cell.translation(t);
            let v = *d + shift;
            let r = v.norm();
            if r < 1.0e-12 || r > params.r_cut {
                continue;
            }
            // At a reciprocal lattice vector the phase is 1 exactly; taking the cosine of
            // `2πn` instead leaves an imaginary residue of order 1e-16 per image, and the
            // reduction to the unphased sum then holds only to within that.
            let e = if gamma {
                [1.0, 0.0]
            } else {
                let phase = q.dot(shift);
                [phase.cos(), phase.sin()]
            };
            let f = erfc(alpha * r) / r;
            let gauss = two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp();
            let df = -(f + gauss) / r;
            let d2f = -2.0 * df / r + 2.0 * alpha * alpha * gauss;
            cadd(&mut slot.value, e, f);
            for a in 0..3 {
                cadd(&mut slot.gradient[a], e, df * v.get(a) / r);
                for b in 0..3 {
                    let radial = v.get(a) * v.get(b) / (r * r);
                    let delta = if a == b { 1.0 } else { 0.0 };
                    cadd(
                        &mut slot.hessian[a][b],
                        e,
                        d2f * radial + df * (delta - radial) / r,
                    );
                }
            }
        }
    }

    // --- reciprocal space -------------------------------------------------------------
    let wavevectors = shifted_reciprocal(cell, q, params.g_cut);

    match cell.dim() {
        3 => {
            let volume = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &k in &wavevectors {
                let k2 = k.norm2();
                if k2 < 1.0e-20 {
                    continue;
                }
                let pref = 2.0 * std::f64::consts::TAU / volume * (-k2 * inv_4a2).exp() / k2;
                for (i, d) in displacements.iter().enumerate() {
                    let phase = k.dot(*d);
                    let e = [phase.cos(), -phase.sin()];
                    let slot = &mut out[i];
                    cadd(&mut slot.value, e, pref);
                    // ∂/∂d of e^{-iK·d} is −iK e^{-iK·d}; multiplying [re, im] by −i swaps them.
                    for a in 0..3 {
                        let ka = k.get(a);
                        let g = [e[1] * ka, -e[0] * ka];
                        cadd(&mut slot.gradient[a], g, pref);
                        for b in 0..3 {
                            cadd(&mut slot.hessian[a][b], e, -pref * ka * k.get(b));
                        }
                    }
                }
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &k in &wavevectors {
                let kn = k.norm();
                if kn < 1.0e-12 {
                    continue;
                }
                let pref = std::f64::consts::PI / (area * kn);
                for (i, d) in displacements.iter().enumerate() {
                    let z = normal.dot(*d);
                    let (bracket, dz, dzz) = parry_bracket(kn, alpha, z);
                    let phase = k.dot(*d);
                    let e = [phase.cos(), -phase.sin()];
                    let de = [e[1], -e[0]]; // the −i that ∂/∂(K·d) brings down
                    let slot = &mut out[i];
                    cadd(&mut slot.value, e, pref * bracket);
                    for a in 0..3 {
                        let (na, ka) = (normal.get(a), k.get(a));
                        cadd(&mut slot.gradient[a], e, pref * dz * na);
                        cadd(&mut slot.gradient[a], de, pref * bracket * ka);
                        for b in 0..3 {
                            let (nb, kb) = (normal.get(b), k.get(b));
                            cadd(&mut slot.hessian[a][b], e, pref * dzz * na * nb);
                            cadd(&mut slot.hessian[a][b], de, pref * dz * (na * kb + ka * nb));
                            cadd(&mut slot.hessian[a][b], e, -pref * bracket * ka * kb);
                        }
                    }
                }
            }
            if gamma {
                // The `G = 0` sheet term, which only exists when the in-plane wavevector is zero.
                for (i, d) in displacements.iter().enumerate() {
                    let z = normal.dot(*d);
                    let az = alpha * z;
                    let c = std::f64::consts::PI / area;
                    let val = -c * (z * (1.0 - erfc(az)) + (-az * az).exp() / (alpha * SQRT_PI));
                    let dz = -c * (1.0 - erfc(az));
                    let dzz = -c * two_alpha_over_sqrt_pi * (-az * az).exp();
                    let slot = &mut out[i];
                    cadd(&mut slot.value, [2.0, 0.0], val);
                    for a in 0..3 {
                        cadd(&mut slot.gradient[a], [2.0, 0.0], dz * normal.get(a));
                        for b in 0..3 {
                            cadd(
                                &mut slot.hessian[a][b],
                                [2.0, 0.0],
                                dzz * normal.get(a) * normal.get(b),
                            );
                        }
                    }
                }
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let pref = 1.0 / length;
            let a2 = alpha * alpha;
            let l2 = WIRE_REFERENCE_LENGTH_SQUARED;
            for &k in &wavevectors {
                let kn = k.norm();
                if kn < 1.0e-12 {
                    continue;
                }
                let along = axis.dot(k);
                for (i, d) in displacements.iter().enumerate() {
                    let x = axis.dot(*d);
                    let perp = *d - axis * x;
                    let rho = perp.norm();
                    let (i_val, di, _) = wire_kernel(kn, alpha, rho, &quad);
                    let d2i = wire_kernel_second(kn, alpha, rho, &quad);
                    let phase = along * x;
                    let e = [phase.cos(), -phase.sin()];
                    let de = [e[1], -e[0]];
                    let slot = &mut out[i];
                    cadd(&mut slot.value, e, pref * i_val);
                    for a in 0..3 {
                        let (aa, pa) = (axis.get(a), perp.get(a));
                        cadd(&mut slot.gradient[a], de, pref * i_val * along * aa);
                        cadd(&mut slot.gradient[a], e, pref * 2.0 * di * pa);
                        for b in 0..3 {
                            let (ab, pb) = (axis.get(b), perp.get(b));
                            let delta = if a == b { 1.0 } else { 0.0 };
                            cadd(
                                &mut slot.hessian[a][b],
                                e,
                                -pref * i_val * along * along * aa * ab,
                            );
                            cadd(
                                &mut slot.hessian[a][b],
                                de,
                                pref * 2.0 * di * along * (aa * pb + pa * ab),
                            );
                            cadd(&mut slot.hessian[a][b], e, pref * 4.0 * d2i * pa * pb);
                            cadd(
                                &mut slot.hessian[a][b],
                                e,
                                pref * 2.0 * di * (delta - aa * ab),
                            );
                        }
                    }
                }
            }
            if gamma {
                // The `G = 0` log term, which the phased sum turns into an ordinary `K = q` one.
                for (i, d) in displacements.iter().enumerate() {
                    let x = axis.dot(*d);
                    let perp = *d - axis * x;
                    let u = perp.norm2();
                    let small = u * a2 < 1.0e-12;
                    let kernel = if small {
                        (a2 * l2).ln()
                    } else {
                        -EULER_GAMMA - (u / l2).ln() - exp_integral_e1(a2 * u)
                    };
                    let (d1, d2) = if small {
                        (-a2, 0.5 * a2 * a2)
                    } else {
                        let ex = (-a2 * u).exp();
                        ((-1.0 + ex) / u, (1.0 - ex * (1.0 + a2 * u)) / (u * u))
                    };
                    let slot = &mut out[i];
                    cadd(&mut slot.value, [1.0, 0.0], pref * kernel);
                    for a in 0..3 {
                        let (aa, pa) = (axis.get(a), perp.get(a));
                        cadd(&mut slot.gradient[a], [1.0, 0.0], pref * 2.0 * d1 * pa);
                        for b in 0..3 {
                            let (ab, pb) = (axis.get(b), perp.get(b));
                            let delta = if a == b { 1.0 } else { 0.0 };
                            cadd(
                                &mut slot.hessian[a][b],
                                [1.0, 0.0],
                                pref * (4.0 * d2 * pa * pb + 2.0 * d1 * (delta - aa * ab)),
                            );
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // --- self term and background -------------------------------------------------------
    // The reciprocal sum put back the `T = 0` term at zero displacement that real space left out.
    // The neutralizing background exists only at Γ, where `G = 0` was dropped.
    let self_c = two_alpha_over_sqrt_pi;
    let background = if gamma && cell.dim() == 3 {
        -std::f64::consts::PI / (alpha * alpha * cell.measure())
    } else {
        0.0
    };
    for (i, d) in displacements.iter().enumerate() {
        let slot = &mut out[i];
        if d.norm2() < 1.0e-20 {
            slot.value[0] -= self_c;
        }
        slot.value[0] += background;
        slot.value[0] *= PM7_EV;
        slot.value[1] *= PM7_EV;
        for a in 0..3 {
            slot.gradient[a][0] *= PM7_EV;
            slot.gradient[a][1] *= PM7_EV;
            for b in 0..3 {
                slot.hessian[a][b][0] *= PM7_EV;
                slot.hessian[a][b][1] *= PM7_EV;
            }
        }
    }
    out
}

/// The field `F_AB = ∂M_AB/∂d_AB` of the Ewald interaction matrix, in eV/Bohr per unit charge².
///
/// `M_AB` depends on the atoms only through `d_AB = R_B − R_A`, so this one array carries every
/// position derivative of `M`:
///
/// ```text
/// ∂M_AB/∂R_C = (δ_CB − δ_CA) F_AB
/// ```
///
/// It is what the derivative **Fock** matrix needs — the Ewald potential `−V_A = −Σ_B M_AB q_B`
/// sits on the diagonal of `F`, and the regularized exchange `−P_{μλ}(M_AB − M_self)` off it, so
/// both move when an atom moves. `M` is an even function of `d`, hence `F` is odd and
/// `F_BA = −F_AB`; a self pair has a position-independent displacement and gets zero.
///
/// `field_matrix_matches_finite_differences` checks it against `ewald_matrix` rather than
/// trusting the duplicated kernels.
pub fn ewald_field_matrix(
    cell: &Cell,
    positions: &[Vec3],
    params: &EwaldParameters,
) -> Vec<Vec<Vec3>> {
    let n = positions.len();
    let mut f = vec![vec![Vec3::zero(); n]; n];
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / SQRT_PI;

    // Real space, over the same enumeration `ewald_matrix` uses. Only `a < b` matters: a self
    // pair's displacement is a lattice vector and does not move with the atom.
    for a in 0..n {
        for b in (a + 1)..n {
            let mut acc = Vec3::zero();
            for &t in &params.real_images {
                let d = positions[b] + cell.translation(t) - positions[a];
                let r = d.norm();
                if r > params.r_cut || r < 1.0e-12 {
                    continue;
                }
                let e = erfc(alpha * r) / r;
                let dedr = -(e + two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp()) / r;
                acc += d * (dedr / r);
            }
            f[a][b] += acc;
        }
    }

    match cell.dim() {
        3 => {
            let v = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &g in &params.g_vectors {
                let g2 = g.norm2();
                let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
                for a in 0..n {
                    for b in (a + 1)..n {
                        let phase = g.dot(positions[b] - positions[a]);
                        f[a][b] += g * (-2.0 * pref * phase.sin());
                    }
                }
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &g in &params.g_vectors {
                let gn = g.norm();
                let pref = std::f64::consts::PI / (2.0 * area * gn);
                for a in 0..n {
                    for b in (a + 1)..n {
                        let d = positions[b] - positions[a];
                        let (_, dz, dpar, _) = parry_kernel(gn, alpha, normal.dot(d), g.dot(d));
                        f[a][b] += (normal * dz + g * dpar) * (2.0 * pref);
                    }
                }
            }
            for a in 0..n {
                for b in (a + 1)..n {
                    let z = normal.dot(positions[b] - positions[a]);
                    let erf_az = 1.0 - erfc(alpha * z);
                    f[a][b] += normal * (-2.0 * (std::f64::consts::PI / area) * erf_az);
                }
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let pref = 0.5 / length;
            let a2 = alpha * alpha;
            for &g in &params.g_vectors {
                let gn = g.norm();
                for a in 0..n {
                    for b in (a + 1)..n {
                        let d = positions[b] - positions[a];
                        let x = axis.dot(d);
                        let perp = d - axis * x;
                        let (i_val, di_drho2, _) = wire_kernel(gn, alpha, perp.norm(), &quad);
                        let (cs, sn) = ((gn * x).cos(), (gn * x).sin());
                        f[a][b] += (axis * (-gn * sn * i_val) + perp * (2.0 * cs * di_drho2))
                            * (2.0 * pref);
                    }
                }
            }
            for a in 0..n {
                for b in (a + 1)..n {
                    let d = positions[b] - positions[a];
                    let perp = d - axis * axis.dot(d);
                    let u = perp.norm2();
                    let d1 = if u * a2 < 1.0e-12 {
                        -a2
                    } else {
                        (-1.0 + (-a2 * u).exp()) / u
                    };
                    f[a][b] += perp * (4.0 * pref * d1);
                }
            }
        }
        _ => {}
    }

    // The self term and the background are position-independent, so they add nothing. Fill the
    // lower triangle from `F_BA = −F_AB`.
    for a in 0..n {
        for b in (a + 1)..n {
            f[a][b] = f[a][b] * PM7_EV;
            f[b][a] = f[a][b] * -1.0;
        }
    }
    f
}

/// The regularized lattice sum `Σ'_T 1/|d + T|` at each of the given displacements, in eV.
///
/// This is the same quantity [`ewald_matrix`] holds for the displacements between a set of
/// atoms, but evaluated at arbitrary displacements — which is what the **Born–von Kármán
/// consistent** exchange needs. A `n₁×n₂×n₃` k mesh is the Γ point of the corresponding
/// supercell, so the exchange lattice sum splits into one sum per residue class of the
/// superlattice, and each of those is this function evaluated on the *superlattice* at the
/// displacement `R_AB + T`.
///
/// `d = 0` gives the self-potential (the `T = 0` term excluded), which is the Madelung constant
/// of the lattice and is what the divergence correction subtracts.
pub fn ewald_potentials_at(
    cell: &Cell,
    displacements: &[Vec3],
    params: &EwaldParameters,
) -> Vec<f64> {
    let alpha = params.alpha;
    let mut out = vec![0.0_f64; displacements.len()];

    // Real space.
    for (i, d) in displacements.iter().enumerate() {
        let self_pair = d.norm2() < 1.0e-20;
        for &t in &params.real_images {
            let v = *d + cell.translation(t);
            let r = v.norm();
            // The `T = 0` term of a zero displacement is the self-interaction, excluded by
            // definition; every other term is kept.
            if r < 1.0e-12 || r > params.r_cut {
                debug_assert!(r >= 1.0e-12 || self_pair);
                continue;
            }
            out[i] += erfc(alpha * r) / r;
        }
    }

    // Reciprocal space.
    match cell.dim() {
        3 => {
            let v = cell.measure();
            let inv_4a2 = 1.0 / (4.0 * alpha * alpha);
            for &g in &params.g_vectors {
                let g2 = g.norm2();
                let pref = std::f64::consts::TAU / v * (-g2 * inv_4a2).exp() / g2;
                for (i, d) in displacements.iter().enumerate() {
                    out[i] += 2.0 * pref * g.dot(*d).cos();
                }
            }
        }
        2 => {
            let area = cell.measure();
            let normal = cell.completed_vectors()[2].normalized();
            for &g in &params.g_vectors {
                let gn = g.norm();
                let pref = std::f64::consts::PI / (2.0 * area * gn);
                for (i, d) in displacements.iter().enumerate() {
                    let (val, _, _, _) = parry_kernel(gn, alpha, normal.dot(*d), g.dot(*d));
                    out[i] += 2.0 * pref * val;
                }
            }
            for (i, d) in displacements.iter().enumerate() {
                let z = normal.dot(*d);
                let az = alpha * z;
                out[i] -= 2.0
                    * (std::f64::consts::PI / area)
                    * (z * (1.0 - erfc(az)) + (-az * az).exp() / (alpha * SQRT_PI));
            }
        }
        1 => {
            let axis = cell.vectors()[0].normalized();
            let length = cell.measure();
            let quad = gauss_legendre(64);
            let pref = 0.5 / length;
            let a2 = alpha * alpha;
            let l2 = WIRE_REFERENCE_LENGTH_SQUARED;
            for &g in &params.g_vectors {
                let gn = g.norm();
                for (i, d) in displacements.iter().enumerate() {
                    let x = axis.dot(*d);
                    let rho = (*d - axis * x).norm();
                    let (i_val, _, _) = wire_kernel(gn, alpha, rho, &quad);
                    out[i] += 2.0 * pref * (gn * x).cos() * i_val;
                }
            }
            for (i, d) in displacements.iter().enumerate() {
                let x = axis.dot(*d);
                let rho2 = (*d - axis * x).norm2();
                let kernel = if rho2 * a2 < 1.0e-12 {
                    (a2 * l2).ln()
                } else {
                    -EULER_GAMMA - (rho2 / l2).ln() - exp_integral_e1(a2 * rho2)
                };
                out[i] += 2.0 * pref * kernel;
            }
        }
        _ => {}
    }

    // Self term (only for the zero displacement) and the neutralizing background.
    let c = 2.0 * alpha / SQRT_PI;
    let shift = if cell.dim() == 3 {
        -std::f64::consts::PI / (alpha * alpha * cell.measure())
    } else {
        0.0
    };
    for (i, d) in displacements.iter().enumerate() {
        if d.norm2() < 1.0e-20 {
            out[i] -= c;
        }
        out[i] += shift;
        out[i] *= PM7_EV;
    }
    out
}

/// Makov–Payne finite-size correction for a charged cell, in eV.
///
/// This estimates how much the periodic model's energy differs from an isolated charged system.
/// It is reported as a diagnostic and never added to the energy: adding it would make the energy
/// inconsistent with its own gradient and stress, and its validity depends on assumptions
/// (cubic-ish cell, localized charge) that the code cannot verify.
pub fn makov_payne(cell: &Cell, charges: &[f64]) -> Option<f64> {
    let q: f64 = charges.iter().sum();
    if q.abs() < 1.0e-14 || cell.dim() != 3 {
        return None;
    }
    let v = cell.volume()?;
    // The Madelung constant of a simple-cubic lattice of point charges in a jellium background.
    const MADELUNG_SC: f64 = 2.837_297_479;
    let l = v.cbrt();
    Some(MADELUNG_SC * q * q / (2.0 * l) * PM7_EV)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::PbcOptions;

    fn opts() -> PbcOptions {
        PbcOptions::default()
    }

    fn run(cell: &Cell, pos: &[Vec3], q: &[f64], alpha: Option<f64>) -> EwaldPotential {
        let p = EwaldParameters::new(cell, pos.len(), 1.0e-12, alpha);
        ewald(cell, pos, q, &p, &opts()).unwrap()
    }

    #[test]
    fn energy_is_independent_of_the_splitting_parameter() {
        // The single sharpest test of an Ewald implementation: the split is arbitrary, so any
        // alpha must give the same total. A wrong self term, background, or G=0 term breaks it.
        let cases: Vec<(Cell, Vec<Vec3>, Vec<f64>)> = vec![
            (
                Cell::cubic(9.0).unwrap(),
                vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.5, 4.5, 4.5)],
                vec![1.0, -1.0],
            ),
            (
                Cell::new(&[
                    Vec3::new(8.0, 0.0, 0.0),
                    Vec3::new(1.5, 7.5, 0.0),
                    Vec3::new(0.4, -0.8, 9.3),
                ])
                .unwrap(),
                vec![
                    Vec3::new(0.3, 0.2, 0.1),
                    Vec3::new(3.1, 2.4, 4.0),
                    Vec3::new(6.0, 5.5, 7.2),
                ],
                vec![0.7, -0.3, -0.4],
            ),
        ];
        for (cell, pos, q) in cases {
            let mut ref_e = None;
            for alpha in [0.15, 0.25, 0.35, 0.5, 0.7] {
                let out = run(&cell, &pos, &q, Some(alpha));
                match ref_e {
                    None => ref_e = Some(out.energy),
                    Some(e0) => assert!(
                        (out.energy - e0).abs() < 1e-7 * e0.abs().max(1.0),
                        "alpha={alpha}: E={} vs {e0}",
                        out.energy
                    ),
                }
            }
        }
    }

    #[test]
    fn charged_cell_energy_is_also_alpha_independent() {
        // With a net charge the G = 0 term is dropped and the background constant put back;
        // if the constant is wrong, alpha-independence fails immediately.
        let cell = Cell::cubic(10.0).unwrap();
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 5.0, 5.0)];
        let q = vec![1.0, 0.5]; // net +1.5
        let mut ref_e = None;
        for alpha in [0.15, 0.22, 0.3, 0.45, 0.6] {
            let out = run(&cell, &pos, &q, Some(alpha));
            assert!(
                out.background_energy != 0.0,
                "no background for a charged cell"
            );
            match ref_e {
                None => ref_e = Some(out.energy),
                Some(e0) => assert!(
                    (out.energy - e0).abs() < 1e-7 * e0.abs().max(1.0),
                    "charged, alpha={alpha}: E={} vs {e0}",
                    out.energy
                ),
            }
        }
    }

    #[test]
    fn madelung_constant_of_rocksalt_is_reproduced() {
        // NaCl: E = −M q²/a per ion pair with M = 1.747_564_594_633_2 (Madelung, in units of
        // e²/a with a the nearest-neighbour distance). This is an absolute, independent check
        // that the real, reciprocal, self, and (absent) background terms combine correctly.
        let a = 4.0_f64; // conventional cubic cell edge, Bohr
        let cell = Cell::cubic(a).unwrap();
        let h = a / 2.0;
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(h, h, 0.0),
            Vec3::new(h, 0.0, h),
            Vec3::new(0.0, h, h),
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(0.0, h, 0.0),
            Vec3::new(0.0, 0.0, h),
            Vec3::new(h, h, h),
        ];
        let q = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let out = run(&cell, &pos, &q, None);
        // 4 ion pairs per conventional cell; nearest-neighbour distance is a/2.
        let per_pair = out.energy / 4.0 / PM7_EV; // back to e²/Bohr
        let madelung = -per_pair * h;
        assert!(
            (madelung - 1.747_564_594_633_2).abs() < 1e-9,
            "Madelung constant = {madelung:.12}"
        );
    }

    #[test]
    fn potential_is_the_charge_derivative_of_the_energy() {
        let cell = Cell::new(&[
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(0.9, 6.4, 0.0),
            Vec3::new(0.0, 0.5, 8.1),
        ])
        .unwrap();
        let pos = vec![
            Vec3::new(0.4, 0.3, 0.2),
            Vec3::new(3.0, 1.9, 4.4),
            Vec3::new(5.5, 4.8, 6.0),
        ];
        let q = vec![0.8, -0.5, 0.2];
        let out = run(&cell, &pos, &q, Some(0.3));
        let h = 1e-6;
        for i in 0..q.len() {
            let mut qp = q.clone();
            let mut qm = q.clone();
            qp[i] += h;
            qm[i] -= h;
            let ep = run(&cell, &pos, &qp, Some(0.3)).energy;
            let em = run(&cell, &pos, &qm, Some(0.3)).energy;
            let fd = (ep - em) / (2.0 * h);
            assert!(
                (out.potential[i] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                "atom {i}: V = {} vs dE/dq = {fd}",
                out.potential[i]
            );
        }
    }

    #[test]
    fn gradient_matches_finite_difference_in_every_dimension() {
        let cases: Vec<(Cell, Vec<Vec3>, Vec<f64>)> = vec![
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![Vec3::new(0.3, 0.4, 0.0), Vec3::new(2.1, -1.2, 0.6)],
                vec![0.6, -0.6],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(1.0, 5.5, 0.0)]).unwrap(),
                vec![Vec3::new(0.2, 0.1, 0.5), Vec3::new(2.7, 3.0, -0.9)],
                vec![0.5, -0.5],
            ),
            (
                Cell::cubic(8.0).unwrap(),
                vec![Vec3::new(0.1, 0.2, 0.3), Vec3::new(4.0, 3.5, 4.5)],
                vec![0.9, -0.9],
            ),
        ];
        for (cell, pos, q) in cases {
            let dim = cell.dim();
            let out = run(&cell, &pos, &q, Some(0.3));
            let h = 1e-6;
            for i in 0..pos.len() {
                for ax in 0..3 {
                    let mut pp = pos.clone();
                    let mut pm = pos.clone();
                    match ax {
                        0 => {
                            pp[i].x += h;
                            pm[i].x -= h;
                        }
                        1 => {
                            pp[i].y += h;
                            pm[i].y -= h;
                        }
                        _ => {
                            pp[i].z += h;
                            pm[i].z -= h;
                        }
                    }
                    let ep = run(&cell, &pp, &q, Some(0.3)).energy;
                    let em = run(&cell, &pm, &q, Some(0.3)).energy;
                    let fd = (ep - em) / (2.0 * h);
                    let got = out.gradient[i].get(ax);
                    assert!(
                        (got - fd).abs() < 1e-5 * fd.abs().max(1.0),
                        "dim {dim} atom {i} axis {ax}: {got} vs {fd}"
                    );
                }
            }
        }
    }

    #[test]
    fn forces_sum_to_zero() {
        let cell = Cell::cubic(9.0).unwrap();
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 1.0, 2.0),
            Vec3::new(2.0, 6.0, 7.0),
        ];
        for q in [vec![1.0, -1.0, 0.0], vec![1.0, 0.5, 0.25]] {
            let out = run(&cell, &pos, &q, Some(0.3));
            let sum = out.gradient.iter().fold(Vec3::zero(), |acc, g| acc + *g);
            assert!(
                sum.norm() < 1e-9,
                "net force {sum:?} for charges {q:?} (charged cell must still be force-free)"
            );
        }
    }

    #[test]
    fn one_dimensional_sum_matches_a_direct_neutral_group_summation() {
        // A neutral 1-D cell's lattice sum converges absolutely (dipole-dipole ~ 1/n³), so it
        // can be summed directly to high accuracy and compared against the Ewald result.
        let l = 5.0_f64;
        let cell = Cell::new(&[Vec3::new(l, 0.0, 0.0)]).unwrap();
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.2, 0.8, 0.0)];
        let q = vec![1.0, -1.0];
        let ew = run(&cell, &pos, &q, Some(0.35)).energy / PM7_EV;

        // Direct: ½ Σ'_{A,B,n} q_A q_B / |r_AB + nL|, truncated at large |n|. Because the cell is
        // neutral the tail decays as 1/n³ and 200_000 cells is far past 1e-10.
        let n_max = 200_000i64;
        let mut direct = 0.0_f64;
        for a in 0..2 {
            for b in 0..2 {
                for n in -n_max..=n_max {
                    if a == b && n == 0 {
                        continue;
                    }
                    let d = pos[b] - pos[a] + Vec3::new(n as f64 * l, 0.0, 0.0);
                    direct += 0.5 * q[a] * q[b] / d.norm();
                }
            }
        }
        assert!(
            (ew - direct).abs() < 1e-6 * direct.abs().max(1.0),
            "1-D Ewald {ew:.10} vs direct sum {direct:.10}"
        );
    }

    #[test]
    fn two_dimensional_sum_matches_a_direct_neutral_group_summation() {
        // Same idea in 2-D: a neutral cell's tail is 1/n³ against a 2-D shell count ~ n, so the
        // direct sum converges as 1/N. Extrapolate two cutoffs to remove the leading 1/N error.
        let a = 6.0_f64;
        let cell = Cell::new(&[Vec3::new(a, 0.0, 0.0), Vec3::new(0.0, a, 0.0)]).unwrap();
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.5, 1.0, 0.7)];
        let q = vec![1.0, -1.0];
        let ew = run(&cell, &pos, &q, Some(0.3)).energy / PM7_EV;

        let direct = |n_max: i64| {
            let mut s = 0.0_f64;
            for ia in 0..2 {
                for ib in 0..2 {
                    for i in -n_max..=n_max {
                        for j in -n_max..=n_max {
                            if ia == ib && i == 0 && j == 0 {
                                continue;
                            }
                            let d = pos[ib] - pos[ia] + Vec3::new(i as f64 * a, j as f64 * a, 0.0);
                            s += 0.5 * q[ia] * q[ib] / d.norm();
                        }
                    }
                }
            }
            s
        };
        // Richardson: S(N) ≈ S∞ + c/N  →  S∞ ≈ 2 S(2N) − S(N).
        let s1 = direct(400);
        let s2 = direct(800);
        let extrapolated = 2.0 * s2 - s1;
        assert!(
            (ew - extrapolated).abs() < 1e-5 * extrapolated.abs().max(1.0),
            "2-D Ewald {ew:.10} vs extrapolated direct sum {extrapolated:.10} (S400={s1:.10}, S800={s2:.10})"
        );
    }

    #[test]
    fn wire_kernel_matches_adaptive_quadrature() {
        // Independent check of the 1-D reciprocal kernel and its two derivatives: refine the
        // rule and compare, then finite-difference in rho^2 and G.
        let coarse = gauss_legendre(64);
        let fine = gauss_legendre(400);
        for &(g, alpha, rho) in &[
            (0.5_f64, 0.3_f64, 0.0_f64),
            (0.5, 0.3, 1.7),
            (2.0, 0.4, 0.4),
            (5.0, 0.6, 2.5),
        ] {
            let (c, dr, dg) = wire_kernel(g, alpha, rho, &coarse);
            let (f, _, _) = wire_kernel(g, alpha, rho, &fine);
            assert!(
                (c - f).abs() < 1e-12 * f.abs().max(1.0),
                "64-pt {c:.12e} vs 400-pt {f:.12e} at G={g} rho={rho}"
            );
            let h = 1e-6;
            // A central difference in rho^2 needs rho^2 - h >= 0; at rho = 0 it would be
            // one-sided and read half the true slope, so only the value is checked there.
            if rho * rho > 10.0 * h {
                let up = wire_kernel(g, alpha, (rho * rho + h).sqrt(), &fine).0;
                let dn = wire_kernel(g, alpha, (rho * rho - h).sqrt(), &fine).0;
                let fd = (up - dn) / (2.0 * h);
                assert!(
                    (dr - fd).abs() < 1e-5 * fd.abs().max(1e-3),
                    "d/d(rho^2): {dr} vs {fd} at G={g} rho={rho}"
                );
            }
            let gp = wire_kernel(g + h, alpha, rho, &fine).0;
            let gm = wire_kernel(g - h, alpha, rho, &fine).0;
            let fdg = (gp - gm) / (2.0 * h);
            assert!(
                (dg - fdg).abs() < 1e-5 * fdg.abs().max(1e-3),
                "d/dG: {dg} vs {fdg} at G={g} rho={rho}"
            );
        }
    }

    /// Stress from the virial, `σ = (1/measure) ∂E/∂ε`, compared with a finite difference of
    /// the energy under an applied strain. Atoms are carried along by the strain, exactly as in
    /// the analytic derivation.
    fn check_stress_against_fd(cell: &Cell, pos: &[Vec3], q: &[f64], tag: &str) {
        let alpha = Some(0.3);
        let out = run(cell, pos, q, alpha);
        let measure = cell.measure();
        let p = periodic_projector(cell);
        let h = 1e-6;
        for i in 0..3 {
            for j in 0..3 {
                // Only strain components inside the periodic subspace are meaningful.
                if p.get(i, i).abs() < 1e-12 || p.get(j, j).abs() < 1e-12 {
                    continue;
                }
                let mut eps = Mat3::zero();
                // Symmetric strain, so the finite difference matches the symmetrized virial.
                eps.set(i, j, 0.5 * h);
                eps.set(j, i, eps.get(j, i) + 0.5 * h);
                let strain = |s: f64| -> f64 {
                    let e = eps.scaled(s);
                    let c2 = cell.strained(&e).unwrap();
                    let p2: Vec<Vec3> = pos.iter().map(|r| *r + e.mul_vec(*r)).collect();
                    run(&c2, &p2, q, alpha).energy
                };
                let fd = (strain(1.0) - strain(-1.0)) / (2.0 * h);
                let got = out.virial.symmetrized().get(i, j);
                assert!(
                    (got - fd).abs() < 2e-4 * fd.abs().max(measure.abs().max(1.0) * 1e-4),
                    "{tag} virial[{i}][{j}] = {got:.9} vs finite difference {fd:.9}"
                );
            }
        }
    }

    #[test]
    fn stress_matches_finite_difference_in_every_dimension() {
        check_stress_against_fd(
            &Cell::cubic(8.0).unwrap(),
            &[Vec3::new(0.3, 0.2, 0.1), Vec3::new(4.1, 3.2, 4.6)],
            &[1.0, -1.0],
            "3-D neutral",
        );
        check_stress_against_fd(
            &Cell::new(&[
                Vec3::new(7.0, 0.0, 0.0),
                Vec3::new(1.1, 6.2, 0.0),
                Vec3::new(0.3, 0.5, 8.0),
            ])
            .unwrap(),
            &[
                Vec3::new(0.4, 0.1, 0.9),
                Vec3::new(2.9, 3.3, 4.1),
                Vec3::new(5.2, 1.8, 6.7),
            ],
            &[0.6, -0.9, 0.3],
            "3-D triclinic",
        );
        check_stress_against_fd(
            &Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.8, 5.4, 0.0)]).unwrap(),
            &[Vec3::new(0.2, 0.3, 0.6), Vec3::new(2.5, 2.8, -0.8)],
            &[0.7, -0.7],
            "2-D slab",
        );
        check_stress_against_fd(
            &Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
            &[Vec3::new(0.1, 0.5, 0.0), Vec3::new(2.2, -1.1, 0.7)],
            &[0.5, -0.5],
            "1-D wire",
        );
    }

    #[test]
    fn charged_cell_stress_matches_finite_difference() {
        // The neutralizing background scales as 1/V and so contributes an isotropic stress.
        // Dropping it leaves the forces right and the stress quietly wrong, which is exactly
        // the failure mode this test exists for.
        check_stress_against_fd(
            &Cell::cubic(9.0).unwrap(),
            &[Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.5, 4.0, 3.5)],
            &[1.0, 0.5],
            "3-D charged",
        );
        check_stress_against_fd(
            &Cell::new(&[Vec3::new(6.5, 0.0, 0.0), Vec3::new(0.0, 6.0, 0.0)]).unwrap(),
            &[Vec3::new(0.3, 0.2, 0.4), Vec3::new(3.0, 2.5, -0.6)],
            &[0.8, 0.2],
            "2-D charged",
        );
        check_stress_against_fd(
            &Cell::new(&[Vec3::new(5.5, 0.0, 0.0)]).unwrap(),
            &[Vec3::new(0.2, 0.6, 0.0), Vec3::new(2.4, -0.9, 0.5)],
            &[0.7, 0.1],
            "1-D charged",
        );
    }

    #[test]
    fn charged_low_dimensional_cells_are_alpha_independent_too() {
        for (cell, pos) in [
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.0, 5.5, 0.0)]).unwrap(),
                vec![Vec3::new(0.2, 0.1, 0.5), Vec3::new(2.8, 2.4, -0.7)],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![Vec3::new(0.1, 0.4, 0.0), Vec3::new(2.0, -1.0, 0.6)],
            ),
        ] {
            let q = vec![0.9, 0.3]; // net +1.2
            let mut ref_e = None;
            for alpha in [0.2, 0.3, 0.45, 0.6] {
                let e = run(&cell, &pos, &q, Some(alpha)).energy;
                match ref_e {
                    None => ref_e = Some(e),
                    Some(e0) => assert!(
                        (e - e0).abs() < 1e-6 * e0.abs().max(1.0),
                        "dim {} charged, alpha={alpha}: {e} vs {e0}",
                        cell.dim()
                    ),
                }
            }
        }
    }

    #[test]
    fn virial_lives_only_in_the_periodic_subspace() {
        // A 2-D slab has no lattice vector along its normal, so no stress component can be
        // conjugate to it. A 1-D wire has only the axial component.
        let slab = Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.0, 5.0, 0.0)]).unwrap();
        let out = run(
            &slab,
            &[Vec3::new(0.2, 0.3, 0.7), Vec3::new(2.6, 2.1, -0.5)],
            &[0.6, -0.6],
            Some(0.3),
        );
        for i in 0..3 {
            assert!(out.virial.get(i, 2).abs() < 1e-10, "slab virial has z rows");
            assert!(out.virial.get(2, i).abs() < 1e-10, "slab virial has z cols");
        }

        let wire = Cell::new(&[Vec3::new(0.0, 4.0, 0.0)]).unwrap();
        let out = run(
            &wire,
            &[Vec3::new(0.5, 0.1, 0.2), Vec3::new(-0.7, 1.9, 0.9)],
            &[0.4, -0.4],
            Some(0.35),
        );
        for i in 0..3 {
            for j in 0..3 {
                if i == 1 && j == 1 {
                    continue;
                }
                assert!(
                    out.virial.get(i, j).abs() < 1e-10,
                    "wire virial[{i}][{j}] = {} should vanish off the axis",
                    out.virial.get(i, j)
                );
            }
        }
    }

    #[test]
    fn ewald_matrix_reproduces_the_direct_energy_and_potential() {
        // The matrix form is a second implementation of the same sums, kept because the SCF
        // needs `V = M q` every iteration. Cross-checking it against `ewald` is what makes the
        // duplication safe.
        let cases: Vec<(Cell, Vec<Vec3>, Vec<f64>)> = vec![
            (
                Cell::cubic(8.0).unwrap(),
                vec![
                    Vec3::new(0.3, 0.2, 0.1),
                    Vec3::new(4.1, 3.2, 4.6),
                    Vec3::new(1.5, 6.0, 2.2),
                ],
                vec![0.8, -0.5, -0.3],
            ),
            (
                // Charged 3-D: exercises the background shift in the matrix.
                Cell::cubic(9.0).unwrap(),
                vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.5, 4.0, 3.5)],
                vec![1.0, 0.5],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
                vec![Vec3::new(0.2, 0.3, 0.6), Vec3::new(2.5, 2.8, -0.8)],
                vec![0.7, -0.7],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![Vec3::new(0.1, 0.5, 0.0), Vec3::new(2.2, -1.1, 0.7)],
                vec![0.6, -0.6],
            ),
        ];
        for (cell, pos, q) in cases {
            let params = EwaldParameters::new(&cell, pos.len(), 1e-12, Some(0.3));
            let direct = ewald(&cell, &pos, &q, &params, &opts()).unwrap();
            let m = ewald_matrix(&cell, &pos, &params);
            let n = pos.len();
            let energy: f64 = (0..n)
                .flat_map(|a| (0..n).map(move |b| (a, b)))
                .map(|(a, b)| 0.5 * q[a] * m[a][b] * q[b])
                .sum();
            assert!(
                (energy - direct.energy).abs() < 1e-9 * direct.energy.abs().max(1.0),
                "dim {}: matrix energy {energy} vs direct {}",
                cell.dim(),
                direct.energy
            );
            for a in 0..n {
                let v: f64 = (0..n).map(|b| m[a][b] * q[b]).sum();
                assert!(
                    (v - direct.potential[a]).abs() < 1e-9 * direct.potential[a].abs().max(1.0),
                    "dim {} atom {a}: matrix V {v} vs direct {}",
                    cell.dim(),
                    direct.potential[a]
                );
            }
            // Symmetry is not decorative here: an asymmetric M would make the Fock
            // contribution inconsistent with the energy.
            for a in 0..n {
                for b in 0..n {
                    assert!(
                        (m[a][b] - m[b][a]).abs() < 1e-9 * m[a][b].abs().max(1.0),
                        "M is not symmetric at ({a},{b})"
                    );
                }
            }
        }
    }

    #[test]
    fn potentials_at_displacements_agree_with_the_matrix() {
        // `ewald_potentials_at` is a third implementation of the same lattice sum, needed because
        // the Born–von Kármán exchange evaluates it at displacements that are not between any two
        // atoms. Pinning it against `ewald_matrix` keeps all three consistent.
        for cell in [
            Cell::cubic(8.0).unwrap(),
            Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
            Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
        ] {
            let pos = vec![
                Vec3::new(0.2, 0.3, 0.6),
                Vec3::new(2.5, 1.8, -0.8),
                Vec3::new(1.1, 0.4, 1.9),
            ];
            let params = EwaldParameters::new(&cell, pos.len(), 1e-12, Some(0.3));
            let m = ewald_matrix(&cell, &pos, &params);
            let displacements: Vec<Vec3> = (0..pos.len())
                .flat_map(|a| (0..pos.len()).map(move |b| (a, b)))
                .map(|(a, b)| pos[b] - pos[a])
                .collect();
            let phi = ewald_potentials_at(&cell, &displacements, &params);
            for a in 0..pos.len() {
                for b in 0..pos.len() {
                    let got = phi[a * pos.len() + b];
                    let want = m[a][b];
                    assert!(
                        (got - want).abs() < 1e-8 * want.abs().max(1.0),
                        "dim {} ({a},{b}): {got} vs {want}",
                        cell.dim()
                    );
                }
            }
        }
    }

    #[test]
    fn splitting_a_lattice_sum_over_supercell_residue_classes_reproduces_it() {
        // The identity the Born–von Kármán exchange rests on: summing the *superlattice*
        // potential over the residue classes of a cell inside it reproduces the cell's own
        // lattice sum. If this failed, a k mesh could not agree with its supercell.
        let cell = Cell::new(&[Vec3::new(4.0, 0.0, 0.0)]).unwrap();
        let n = 3usize;
        let (super_cell, _) = cell.supercell([n, 1, 1]).unwrap();
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.3, 0.7, 0.0)];
        let prim = EwaldParameters::new(&cell, pos.len(), 1e-12, Some(0.25));
        let sup = EwaldParameters::new(&super_cell, pos.len() * n, 1e-12, Some(0.25));
        let m_prim = ewald_matrix(&cell, &pos, &prim);
        for (a, b) in [(0usize, 0usize), (0, 1), (1, 1)] {
            let d0 = pos[b] - pos[a];
            let displacements: Vec<Vec3> = (0..n)
                .map(|t| d0 + cell.translation([t as i32, 0, 0]))
                .collect();
            let phi = ewald_potentials_at(&super_cell, &displacements, &sup);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - m_prim[a][b]).abs() < 1e-7 * m_prim[a][b].abs().max(1.0),
                "({a},{b}): residue-class sum {sum} vs primitive lattice sum {}",
                m_prim[a][b]
            );
        }
    }

    /// The folded reciprocal pass reproduces the class-by-class one it replaces.
    ///
    /// This is the whole justification for [`ewald_reciprocal_bvk`]: it is an `O(C)` rewrite of an
    /// `O(C²)` loop and must be worth exactly the same number. The comparison is made against the
    /// class loop *as the gradient path actually runs it*, doubled atom list and all, and the
    /// reciprocal part is isolated by differencing the two settings of the `reciprocal` flag —
    /// so the reference is the real code path, not a second transcription of the same formula.
    ///
    /// The mesh is deliberately `3×2×4`. A cubic mesh cannot see a transposed axis in the
    /// separable transform or in the `G → m mod n` lookup: every permutation of the indices gives
    /// the same answer, and the test passes while the code is wrong for every cell anyone would
    /// actually use. The lattice is triclinic for the same reason.
    #[test]
    fn the_folded_reciprocal_sum_reproduces_the_class_loop() {
        let cell = Cell::new(&[
            Vec3::new(5.4, 0.0, 0.0),
            Vec3::new(0.7, 4.9, 0.0),
            Vec3::new(-0.4, 0.6, 6.1),
        ])
        .unwrap();
        let divisions = [3usize, 2, 4];
        let classes = crate::hamiltonian::bvk_representatives(divisions);
        let (super_cell, _) = cell.supercell(divisions).unwrap();
        let positions = vec![
            Vec3::new(0.2, 0.3, 0.1),
            Vec3::new(2.6, 1.1, 3.0),
            Vec3::new(1.4, 3.7, 5.2),
        ];
        let n = positions.len();
        let params = EwaldParameters::new(&super_cell, n * classes.len(), 1e-12, Some(0.25));

        // Coefficients that are neither symmetric in the atom pair nor symmetric under `t → −t`:
        // a symmetric set makes every structure factor real and would let an imaginary-part sign
        // error through untouched.
        let coefficient = |t: usize, a: usize, b: usize| {
            let s = (t * 7 + a * 3 + b) as f64;
            0.31 * (s * 0.7).sin() - 0.12 * (s * 1.9).cos() + 0.05 * (a as f64 - b as f64)
        };
        let per_class: Vec<Vec<Vec<f64>>> = (0..classes.len())
            .map(|t| {
                (0..n)
                    .map(|a| (0..n).map(|b| coefficient(t, a, b)).collect())
                    .collect()
            })
            .collect();

        // --- reference: the class loop, reciprocal part only -----------------------------
        let mut ref_energy = 0.0;
        let mut ref_gradient = vec![Vec3::zero(); n];
        let mut ref_virial = Mat3::zero();
        for (index, t) in classes.iter().enumerate() {
            let c = &per_class[index];
            let (positions_t, c_t): (Vec<Vec3>, Vec<Vec<f64>>) = if *t == [0, 0, 0] {
                (positions.clone(), c.clone())
            } else {
                let shift = cell.translation(*t);
                let mut doubled = positions.clone();
                doubled.extend(positions.iter().map(|p| *p + shift));
                let mut c2 = vec![vec![0.0; 2 * n]; 2 * n];
                for (a, row) in c.iter().enumerate() {
                    for (b, v) in row.iter().enumerate() {
                        c2[a][n + b] = 0.5 * v;
                        c2[n + b][a] = 0.5 * v;
                    }
                }
                (doubled, c2)
            };
            let with = ewald_pair_matrix_with(&super_cell, &positions_t, &c_t, &params, true);
            let without = ewald_pair_matrix_with(&super_cell, &positions_t, &c_t, &params, false);
            ref_energy += with.energy - without.energy;
            for i in 0..n {
                let mut d = with.gradient[i] - without.gradient[i];
                if *t != [0, 0, 0] {
                    d += with.gradient[n + i] - without.gradient[n + i];
                }
                ref_gradient[i] += d;
            }
            ref_virial = ref_virial.plus(&with.virial.plus(&without.virial.scaled(-1.0)));
        }

        // --- the folded pass -------------------------------------------------------------
        let (energy, gradient, virial) = ewald_reciprocal_bvk(
            &cell,
            &super_cell,
            divisions,
            &positions,
            &per_class,
            &params,
        );

        let scale = ref_energy.abs().max(1.0);
        assert!(
            (energy - ref_energy).abs() < 1e-10 * scale,
            "energy: folded {energy:.12e} vs class loop {ref_energy:.12e}"
        );
        for i in 0..n {
            let d = (gradient[i] - ref_gradient[i]).norm();
            assert!(
                d < 1e-10 * ref_gradient[i].norm().max(1.0),
                "atom {i}: gradient differs by {d:.3e} ({:?} vs {:?})",
                gradient[i],
                ref_gradient[i]
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                let d = (virial.get(i, j) - ref_virial.get(i, j)).abs();
                assert!(
                    d < 1e-10 * ref_virial.get(i, j).abs().max(1.0),
                    "virial ({i},{j}) differs by {d:.3e}: {} vs {}",
                    virial.get(i, j),
                    ref_virial.get(i, j)
                );
            }
        }

        // A single class is the degenerate case the Gamma-point path takes, and it must come out
        // of the same code rather than a special case somewhere upstream.
        let single: Vec<Vec<Vec<f64>>> = vec![per_class[0].clone()];
        let (super_one, _) = cell.supercell([1, 1, 1]).unwrap();
        let params_one = EwaldParameters::new(&super_one, n, 1e-12, Some(0.25));
        let full = ewald_pair_matrix_with(&super_one, &positions, &single[0], &params_one, true);
        let bare = ewald_pair_matrix_with(&super_one, &positions, &single[0], &params_one, false);
        let (e1, g1, _) = ewald_reciprocal_bvk(
            &cell,
            &super_one,
            [1, 1, 1],
            &positions,
            &single,
            &params_one,
        );
        let want = full.energy - bare.energy;
        assert!(
            (e1 - want).abs() < 1e-10 * want.abs().max(1.0),
            "one class: {e1:.12e} vs {want:.12e}"
        );
        for i in 0..n {
            let want = full.gradient[i] - bare.gradient[i];
            assert!(
                (g1[i] - want).norm() < 1e-10 * want.norm().max(1.0),
                "one class atom {i}: {:?} vs {:?}",
                g1[i],
                want
            );
        }
    }

    #[test]
    fn pair_matrix_reproduces_the_charge_path() {
        // With c_AB = q_A q_B the general coefficient path must reproduce the charge path
        // exactly — energy, forces, and virial — in every dimension and for a charged cell.
        let cases: Vec<(Cell, Vec<Vec3>, Vec<f64>)> = vec![
            (
                Cell::cubic(8.0).unwrap(),
                vec![
                    Vec3::new(0.3, 0.2, 0.1),
                    Vec3::new(4.1, 3.2, 4.6),
                    Vec3::new(1.5, 6.0, 2.2),
                ],
                vec![0.8, -0.5, -0.3],
            ),
            (
                Cell::cubic(9.0).unwrap(),
                vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.5, 4.0, 3.5)],
                vec![1.0, 0.5],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
                vec![Vec3::new(0.2, 0.3, 0.6), Vec3::new(2.5, 2.8, -0.8)],
                vec![0.7, -0.7],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![Vec3::new(0.1, 0.5, 0.0), Vec3::new(2.2, -1.1, 0.7)],
                vec![0.6, -0.6],
            ),
        ];
        for (cell, pos, q) in cases {
            let n = pos.len();
            let params = EwaldParameters::new(&cell, n, 1e-12, Some(0.3));
            let reference = ewald(&cell, &pos, &q, &params, &opts()).unwrap();
            let c: Vec<Vec<f64>> = (0..n)
                .map(|a| (0..n).map(|b| q[a] * q[b]).collect())
                .collect();
            let got = ewald_pair_matrix(&cell, &pos, &c, &params);
            assert!(
                (got.energy - reference.energy).abs() < 1e-9 * reference.energy.abs().max(1.0),
                "dim {}: {} vs {}",
                cell.dim(),
                got.energy,
                reference.energy
            );
            for i in 0..n {
                let d = (got.gradient[i] - reference.gradient[i]).norm();
                assert!(
                    d < 1e-9 * reference.gradient[i].norm().max(1.0),
                    "dim {} atom {i}: gradient differs by {d:.3e}",
                    cell.dim()
                );
            }
            let dv = got.virial.plus(&reference.virial.scaled(-1.0)).max_abs();
            assert!(
                dv < 1e-8 * reference.virial.max_abs().max(1.0),
                "dim {}: virial differs by {dv:.3e}",
                cell.dim()
            );
        }
    }

    fn phased_cases() -> Vec<(Cell, Vec<Vec3>)> {
        vec![
            (
                Cell::new(&[
                    Vec3::new(7.0, 0.0, 0.0),
                    Vec3::new(1.0, 6.3, 0.0),
                    Vec3::new(0.4, 0.7, 6.9),
                ])
                .unwrap(),
                vec![
                    Vec3::zero(),
                    Vec3::new(2.3, 1.2, 0.6),
                    Vec3::new(-1.1, 3.0, 2.2),
                ],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
                vec![
                    Vec3::zero(),
                    Vec3::new(2.5, 1.8, -0.8),
                    Vec3::new(1.2, 0.3, 1.4),
                ],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![
                    Vec3::zero(),
                    Vec3::new(2.1, 0.0, 0.0),
                    Vec3::new(1.3, -1.1, 0.7),
                ],
            ),
        ]
    }

    #[test]
    fn phased_kernel_reduces_to_the_unphased_sum_at_gamma() {
        // `q = 0` has to give back exactly what `ewald_potentials_at` computes — self term,
        // background, the `G = 0` special cases in 2-D and 1-D, all of it. The phased path takes a
        // different branch through every one of those, so this is the check that it did not
        // quietly redefine the quantity.
        for (cell, displacements) in phased_cases() {
            let params = EwaldParameters::new(&cell, displacements.len(), 1e-12, Some(0.3));
            let reference = ewald_potentials_at(&cell, &displacements, &params);
            let phased = ewald_phased(&cell, &displacements, Vec3::zero(), &params);
            for (i, k) in phased.iter().enumerate() {
                assert!(
                    (k.value[0] - reference[i]).abs() < 1e-9 * reference[i].abs().max(1.0),
                    "dim {}: displacement {i}: phased {:.9} vs unphased {:.9}",
                    cell.dim(),
                    k.value[0],
                    reference[i]
                );
                assert!(
                    k.value[1].abs() < 1e-12,
                    "dim {}: the Γ sum picked up an imaginary part {:.3e}",
                    cell.dim(),
                    k.value[1]
                );
            }
        }
    }

    /// A `q` that **is** a reciprocal lattice vector must give exactly the `q = 0` answer.
    ///
    /// Every phase `e^{iq·T}` is then 1, so this is the same sum written differently — value,
    /// gradient and Hessian alike. The reason it needs its own test is that the zero wavevector is
    /// dropped from the reciprocal set by construction, and the special terms that stand in for it
    /// (the 2-D sheet term, the 1-D log term, the 3-D background) used to be keyed off `q == 0`
    /// rather than off `q ∈ reciprocal lattice`. That is silent: the sum stays finite and merely
    /// comes out short, most visibly in the 1-D transverse Hessian.
    ///
    /// It is the case that matters in practice, not a curiosity. The long-range exchange sums a
    /// **supercell** lattice at a `q` commensurate with the k mesh, and every such `q` is a
    /// supercell reciprocal lattice vector.
    #[test]
    fn the_phased_sum_at_a_reciprocal_lattice_vector_is_the_gamma_sum() {
        for (cell, displacements) in phased_cases() {
            let params = EwaldParameters::new(&cell, displacements.len(), 1e-12, Some(0.3));
            let reference = ewald_phased(&cell, &displacements, Vec3::zero(), &params);
            let b = cell.reciprocal_2pi();
            for k in 0..cell.dim() {
                for multiple in [1.0_f64, -1.0, 2.0] {
                    let q = b[k] * multiple;
                    let got = ewald_phased(&cell, &displacements, q, &params);
                    for (i, (a, r)) in got.iter().zip(&reference).enumerate() {
                        let scale = r.value[0].abs().max(1.0);
                        assert!(
                            (a.value[0] - r.value[0]).abs() < 1e-9 * scale
                                && a.value[1].abs() < 1e-9 * scale,
                            "dim {}, q = {multiple} b{k}, displacement {i}: value \
                             ({:.9}, {:.9}) vs Γ {:.9}",
                            cell.dim(),
                            a.value[0],
                            a.value[1],
                            r.value[0]
                        );
                        for x in 0..3 {
                            let s = r.gradient[x][0].abs().max(1.0);
                            assert!(
                                (a.gradient[x][0] - r.gradient[x][0]).abs() < 1e-9 * s
                                    && a.gradient[x][1].abs() < 1e-9 * s,
                                "dim {}, q = {multiple} b{k}, displacement {i}, axis {x}: \
                                 gradient ({:.9}, {:.9}) vs Γ {:.9}",
                                cell.dim(),
                                a.gradient[x][0],
                                a.gradient[x][1],
                                r.gradient[x][0]
                            );
                            for y in 0..3 {
                                let s = r.hessian[x][y][0].abs().max(1.0);
                                assert!(
                                    (a.hessian[x][y][0] - r.hessian[x][y][0]).abs() < 1e-9 * s
                                        && a.hessian[x][y][1].abs() < 1e-9 * s,
                                    "dim {}, q = {multiple} b{k}, displacement {i}, ({x},{y}): \
                                     hessian ({:.9}, {:.9}) vs Γ {:.9}",
                                    cell.dim(),
                                    a.hessian[x][y][0],
                                    a.hessian[x][y][1],
                                    r.hessian[x][y][0]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn phased_kernel_derivatives_match_finite_differences() {
        // Both derivatives, real and imaginary parts, at a q that is not a special point.
        for (cell, displacements) in phased_cases() {
            let params = EwaldParameters::new(&cell, displacements.len(), 1e-12, Some(0.3));
            let b = cell.reciprocal_2pi();
            let mut q = Vec3::zero();
            for (k, frac) in [0.23_f64, -0.31, 0.17].iter().enumerate().take(cell.dim()) {
                q += b[k] * *frac;
            }
            let base = ewald_phased(&cell, &displacements, q, &params);
            let h = 1.0e-5;
            for (i, d) in displacements.iter().enumerate() {
                // The zero displacement *is* the self potential, whose real-space `T = 0` term is
                // excluded by definition. Nudging it switches that term on, so a finite difference
                // there measures the switch and not a derivative.
                if d.norm2() < 1.0e-12 {
                    continue;
                }
                for axis in 0..3 {
                    let shifted = |s: f64| -> PhasedKernel {
                        let mut moved = *d;
                        match axis {
                            0 => moved.x += s * h,
                            1 => moved.y += s * h,
                            _ => moved.z += s * h,
                        }
                        ewald_phased(&cell, &[moved], q, &params)[0]
                    };
                    let (up, down) = (shifted(1.0), shifted(-1.0));
                    for part in 0..2 {
                        let fd = (up.value[part] - down.value[part]) / (2.0 * h);
                        let got = base[i].gradient[axis][part];
                        assert!(
                            (got - fd).abs() < 1e-4 * fd.abs().max(1.0),
                            "dim {}: d{i} grad[{axis}][{part}] = {got:.9} vs {fd:.9}",
                            cell.dim()
                        );
                        for other in 0..3 {
                            let fd2 =
                                (up.gradient[other][part] - down.gradient[other][part]) / (2.0 * h);
                            let got2 = base[i].hessian[axis][other][part];
                            assert!(
                                (got2 - fd2).abs() < 1e-4 * fd2.abs().max(1.0),
                                "dim {}: d{i} hess[{axis}][{other}][{part}] = {got2:.9} vs {fd2:.9}",
                                cell.dim()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_phased_sum_is_periodic_and_conjugate_symmetric_in_q() {
        // `Φ_{q+G}(d) = Φ_q(d)` for a reciprocal-lattice `G`, and `Φ_{-q} = Φ_q^*` because the
        // real-space sum is real. Both follow from the construction, which is exactly why they
        // catch a sign slip in the reciprocal branch.
        for (cell, displacements) in phased_cases() {
            let params = EwaldParameters::new(&cell, displacements.len(), 1e-12, Some(0.3));
            let b = cell.reciprocal_2pi();
            let q = b[0] * 0.29;
            let base = ewald_phased(&cell, &displacements, q, &params);
            let shifted = ewald_phased(&cell, &displacements, q + b[0], &params);
            let negated = ewald_phased(&cell, &displacements, q * -1.0, &params);
            for i in 0..displacements.len() {
                for part in 0..2 {
                    let scale = base[i].value[part].abs().max(1.0);
                    assert!(
                        (base[i].value[part] - shifted[i].value[part]).abs() < 1e-7 * scale,
                        "dim {}: Φ_q and Φ_{{q+G}} differ at displacement {i}",
                        cell.dim()
                    );
                }
                assert!(
                    (base[i].value[0] - negated[i].value[0]).abs()
                        < 1e-9 * base[i].value[0].abs().max(1.0)
                        && (base[i].value[1] + negated[i].value[1]).abs()
                            < 1e-9 * base[i].value[1].abs().max(1.0),
                    "dim {}: Φ_{{-q}} is not the conjugate of Φ_q at displacement {i}",
                    cell.dim()
                );
            }
        }
    }

    #[test]
    fn field_matrix_matches_finite_differences() {
        // `F_AB = ∂M_AB/∂d_AB`, checked entry by entry against a central difference of
        // `ewald_matrix` in each dimensionality. The 1-D case again includes an on-axis pair.
        let cases: Vec<(Cell, Vec<Vec3>)> = vec![
            (
                Cell::new(&[
                    Vec3::new(7.0, 0.0, 0.0),
                    Vec3::new(1.0, 6.3, 0.0),
                    Vec3::new(0.4, 0.7, 6.9),
                ])
                .unwrap(),
                vec![
                    Vec3::new(0.3, 0.2, 0.1),
                    Vec3::new(3.1, 2.2, 3.6),
                    Vec3::new(1.5, 4.0, 1.2),
                ],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
                vec![
                    Vec3::new(0.2, 0.3, 0.6),
                    Vec3::new(2.5, 2.8, -0.8),
                    Vec3::new(3.4, 1.1, 0.35),
                ],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![
                    Vec3::new(0.1, 0.0, 0.0),
                    Vec3::new(2.2, 0.0, 0.0),
                    Vec3::new(1.3, -1.1, 0.7),
                ],
            ),
        ];
        for (cell, pos) in cases {
            let n = pos.len();
            let params = EwaldParameters::new(&cell, n, 1e-12, Some(0.3));
            let field = ewald_field_matrix(&cell, &pos, &params);
            let h = 1.0e-6;
            for a in 0..n {
                for b in 0..n {
                    for axis in 0..3 {
                        // Move atom b, which shifts d_ab by +h e_axis.
                        let shifted = |s: f64| -> f64 {
                            let mut p = pos.clone();
                            let mut d = Vec3::zero();
                            match axis {
                                0 => d.x = s * h,
                                1 => d.y = s * h,
                                _ => d.z = s * h,
                            }
                            p[b] += d;
                            ewald_matrix(&cell, &p, &params)[a][b]
                        };
                        if a == b {
                            assert_eq!(field[a][b].get(axis), 0.0);
                            continue;
                        }
                        let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
                        let got = field[a][b].get(axis);
                        assert!(
                            (got - fd).abs() < 1e-5 * fd.abs().max(1.0),
                            "dim {}: F[{a}][{b}][{axis}] = {got:.9} vs finite difference {fd:.9}",
                            cell.dim()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pair_hessian_matches_finite_differences_of_the_analytic_gradient() {
        // One case per dimensionality, with a coefficient matrix that is deliberately *not*
        // rank-1 and not symmetric-by-accident, so every branch of the kernel is exercised.
        // The 1-D case puts two atoms exactly on the wire axis: the perpendicular offset is zero
        // there, which is invisible to the gradient (it multiplies by that offset) but reaches
        // the Hessian undamped through the transverse term.
        let cases: Vec<(Cell, Vec<Vec3>)> = vec![
            (
                Cell::new(&[
                    Vec3::new(7.0, 0.0, 0.0),
                    Vec3::new(1.0, 6.3, 0.0),
                    Vec3::new(0.4, 0.7, 6.9),
                ])
                .unwrap(),
                vec![
                    Vec3::new(0.3, 0.2, 0.1),
                    Vec3::new(3.1, 2.2, 3.6),
                    Vec3::new(1.5, 4.0, 1.2),
                ],
            ),
            (
                Cell::new(&[Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.9, 5.4, 0.0)]).unwrap(),
                vec![
                    Vec3::new(0.2, 0.3, 0.6),
                    Vec3::new(2.5, 2.8, -0.8),
                    Vec3::new(3.4, 1.1, 0.35),
                ],
            ),
            (
                Cell::new(&[Vec3::new(5.0, 0.0, 0.0)]).unwrap(),
                vec![
                    Vec3::new(0.1, 0.0, 0.0),
                    Vec3::new(2.2, 0.0, 0.0),
                    Vec3::new(1.3, -1.1, 0.7),
                ],
            ),
        ];
        for (cell, pos) in cases {
            let n = pos.len();
            let params = EwaldParameters::new(&cell, n, 1e-12, Some(0.3));
            let c: Vec<Vec<f64>> = (0..n)
                .map(|a| {
                    (0..n)
                        .map(|b| 0.4 + 0.3 * (a as f64) - 0.2 * (b as f64) + 0.1 * (a * b) as f64)
                        .collect()
                })
                .collect();
            // Symmetrize: the routines assume `c` is symmetric, which is legitimate because `M`
            // is, but the test must feed them what they assume.
            let c: Vec<Vec<f64>> = (0..n)
                .map(|a| (0..n).map(|b| 0.5 * (c[a][b] + c[b][a])).collect())
                .collect();
            let hess = ewald_pair_hessian(&cell, &pos, &c, &params);
            let h = 1.0e-5;
            for atom in 0..n {
                for axis in 0..3 {
                    let shifted = |s: f64| -> Vec<Vec3> {
                        let mut p = pos.clone();
                        let mut d = Vec3::zero();
                        match axis {
                            0 => d.x = s * h,
                            1 => d.y = s * h,
                            _ => d.z = s * h,
                        }
                        p[atom] += d;
                        ewald_pair_matrix(&cell, &p, &c, &params).gradient
                    };
                    let (up, down) = (shifted(1.0), shifted(-1.0));
                    for other in 0..n {
                        for beta in 0..3 {
                            let fd = (up[other].get(beta) - down[other].get(beta)) / (2.0 * h);
                            let got = hess[(3 * atom + axis, 3 * other + beta)];
                            assert!(
                                (got - fd).abs() < 1e-5 * fd.abs().max(1.0),
                                "dim {}: H[{atom},{axis}][{other},{beta}] = {got:.9} vs finite \
                                 difference {fd:.9}",
                                cell.dim()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pair_matrix_gradient_and_stress_match_finite_differences() {
        // A coefficient matrix that is *not* rank-1, which is the case the exchange actually
        // needs and the one a q_A q_B cross-check cannot exercise.
        let cell = Cell::new(&[
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(1.0, 6.3, 0.0),
            Vec3::new(0.2, 0.4, 7.7),
        ])
        .unwrap();
        let pos = vec![
            Vec3::new(0.3, 0.1, 0.5),
            Vec3::new(3.0, 2.6, 3.9),
            Vec3::new(5.4, 1.2, 6.1),
        ];
        // Symmetric, non-factorizable.
        let c = vec![
            vec![0.7, -0.4, 0.15],
            vec![-0.4, 1.1, -0.25],
            vec![0.15, -0.25, 0.5],
        ];
        let params = EwaldParameters::new(&cell, pos.len(), 1e-12, Some(0.3));
        let out = ewald_pair_matrix(&cell, &pos, &c, &params);
        let h = 1e-6;
        for i in 0..pos.len() {
            for ax in 0..3 {
                let shift = |s: f64| {
                    let mut p = pos.clone();
                    match ax {
                        0 => p[i].x += s * h,
                        1 => p[i].y += s * h,
                        _ => p[i].z += s * h,
                    }
                    ewald_pair_matrix(&cell, &p, &c, &params).energy
                };
                let fd = (shift(1.0) - shift(-1.0)) / (2.0 * h);
                let got = out.gradient[i].get(ax);
                assert!(
                    (got - fd).abs() < 1e-5 * fd.abs().max(1.0),
                    "atom {i} axis {ax}: {got} vs {fd}"
                );
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let mut eps = Mat3::zero();
                eps.set(i, j, eps.get(i, j) + 0.5 * h);
                eps.set(j, i, eps.get(j, i) + 0.5 * h);
                let strain = |s: f64| {
                    let e = eps.scaled(s);
                    let c2 = cell.strained(&e).unwrap();
                    let p2: Vec<Vec3> = pos.iter().map(|r| *r + e.mul_vec(*r)).collect();
                    // The reciprocal lattice moves with the cell, so the parameters must be
                    // rebuilt; reusing the unstrained ones would freeze the G vectors and make
                    // the finite difference measure a different function than the analytic
                    // derivative describes.
                    let p = EwaldParameters::new(&c2, p2.len(), 1e-12, Some(0.3));
                    ewald_pair_matrix(&c2, &p2, &c, &p).energy
                };
                let fd = (strain(1.0) - strain(-1.0)) / (2.0 * h);
                let got = out.virial.symmetrized().get(i, j);
                assert!(
                    (got - fd).abs() < 2e-4 * fd.abs().max(1.0),
                    "virial[{i}][{j}] = {got:.9} vs {fd:.9}"
                );
            }
        }
    }

    #[test]
    fn a_forbidden_background_rejects_a_charged_cell() {
        let cell = Cell::cubic(8.0).unwrap();
        let pos = vec![Vec3::new(0.0, 0.0, 0.0)];
        let q = vec![1.0];
        let p = EwaldParameters::new(&cell, 1, 1e-10, None);
        let forbid = PbcOptions {
            background: BackgroundCharge::Forbid,
            ..PbcOptions::default()
        };
        assert!(ewald(&cell, &pos, &q, &p, &forbid).is_err());
        let jellium = PbcOptions {
            background: BackgroundCharge::Jellium,
            ..PbcOptions::default()
        };
        assert!(ewald(&cell, &pos, &q, &p, &jellium).is_ok());
        // A neutral cell is fine either way.
        assert!(ewald(
            &cell,
            &pos,
            &[0.0],
            &p,
            &PbcOptions {
                background: BackgroundCharge::Forbid,
                ..PbcOptions::default()
            }
        )
        .is_ok());
    }
}
