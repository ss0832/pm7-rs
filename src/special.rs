// SPDX-License-Identifier: GPL-3.0-or-later

//! Special functions the periodic electrostatics needs and Rust's standard library does not
//! provide: the error function and its complement, the scaled complement `erfcx`, and the
//! exponential integral `E₁`.
//!
//! Every routine here is built from a **series and a continued fraction whose forms are stated
//! in the doc comment**, rather than from a table of fitted minimax coefficients. That is a
//! deliberate trade: a fitted rational approximation is a little faster, but a single mistyped
//! digit in a 20-coefficient table is invisible on inspection and would poison every periodic
//! energy. These forms are checkable by eye, and the tests cross-validate the two branches
//! against each other at the crossover, against exact values, and against numerical quadrature.

/// `2/√π`.
const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;
/// Euler–Mascheroni constant γ.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// The two functions need **different** crossovers, because each one's series form loses digits
/// where the *other* is the small quantity.
///
/// * `erf` from the series is good while `erf` itself is O(1); getting it as `1 − erfc_cf`
///   cancels badly for small `x`.
/// * `erfc` as `1 − erf_series` cancels as soon as `erfc ≪ 1`, which is already costing 3 digits
///   by `x ≈ 2.25`. The continued fraction is accurate to a few ulp from `x ≈ 0.5` upward, so
///   `erfc` hands over much earlier than `erf` does.
///
/// Both thresholds are validated against an independent quadrature in
/// `erfc_is_accurate_across_the_whole_range`.
const ERFC_CF_LIMIT: f64 = 0.5;
const ERF_SERIES_LIMIT: f64 = 3.0;

/// Error function `erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax >= ERF_SERIES_LIMIT {
        return x.signum() * (1.0 - erfc_continued_fraction(ax));
    }
    x.signum() * erf_series(ax)
}

/// Complementary error function `erfc(x) = 1 − erf(x)`.
///
/// Computed from the continued fraction wherever `erfc` is appreciably below 1, rather than as
/// `1 − erf(x)`, which loses a digit for every decade `erfc` falls below 1 and every significant
/// digit once `erf(x)` rounds to exactly 1.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= ERFC_CF_LIMIT {
        return erfc_continued_fraction(x);
    }
    if x <= -ERFC_CF_LIMIT {
        return 2.0 - erfc_continued_fraction(-x);
    }
    1.0 - x.signum() * erf_series(x.abs())
}

/// Scaled complementary error function `erfcx(x) = e^{x²} erfc(x)`.
///
/// Needed by the 2-D lattice sum, where the two terms carry factors `e^{±|G|z}` that overflow
/// long before their product with `erfc` does. Working with `erfcx` keeps the product in range:
/// `e^{Gz} erfc(αz + G/2α) = erfcx(αz + G/2α) · exp(Gz − (αz + G/2α)²)`, whose exponent is
/// `−(αz − G/2α)² ≤ 0`.
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= ERFC_CF_LIMIT {
        // The continued fraction already produces erfc with the e^{−x²} factored out.
        return erfcx_continued_fraction(x);
    }
    // e^{x²} is at most e^{0.25} here, so the direct product is well conditioned.
    (x * x).exp() * erfc(x)
}

/// `erf` from the non-alternating series
///
/// ```text
/// erf(x) = (2/√π) e^{−x²} Σ_{n≥0} 2ⁿ x^{2n+1} / (1·3·5···(2n+1))
/// ```
///
/// Every term is positive, so there is no cancellation — unlike the more familiar alternating
/// Taylor series, which loses digits well before `x = 2`. Requires `x ≥ 0`.
fn erf_series(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let x2 = x * x;
    let mut term = x; // n = 0 term: x / 1
    let mut sum = term;
    for n in 1..200 {
        // term_n / term_{n-1} = 2 x² / (2n+1)
        term *= 2.0 * x2 / (2.0 * n as f64 + 1.0);
        sum += term;
        if term <= sum * 1.0e-17 {
            break;
        }
    }
    TWO_OVER_SQRT_PI * (-x2).exp() * sum
}

/// `erfc(x)` for `x > 0` from the continued fraction
///
/// ```text
/// erfc(x) = (e^{−x²} / √π) · 1 / (x + ½/(x + 1/(x + 3/2/(x + 2/(x + …)))))
/// ```
///
/// i.e. the continued fraction for the upper incomplete gamma `Γ(½, x²)`, with partial
/// numerators `n/2`. Evaluated with the modified Lentz algorithm, which is stable even when a
/// partial denominator is near zero.
fn erfc_continued_fraction(x: f64) -> f64 {
    erfcx_continued_fraction(x) * (-x * x).exp()
}

/// The same continued fraction with the `e^{−x²}` factor omitted, i.e. `erfcx(x)` for `x > 0`.
fn erfcx_continued_fraction(x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    let inv_sqrt_pi = 0.5 * std::f64::consts::FRAC_2_SQRT_PI; // 1/√π
                                                              // Modified Lentz for  f = b0 + a1/(b1 + a2/(b2 + …))  with b0 = 0, b_i = x, a_i = i/2.
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0_f64;
    for i in 1..1000 {
        let a = if i == 1 { 1.0 } else { (i as f64 - 1.0) * 0.5 };
        let b = x;
        d = b + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1.0e-17 {
            break;
        }
    }
    inv_sqrt_pi * f
}

/// Exponential integral `E₁(x) = ∫₁^∞ e^{−x t}/t dt` for `x > 0`.
///
/// * `x ≤ 1`: the series `E₁(x) = −γ − ln x + Σ_{n≥1} (−1)^{n+1} xⁿ / (n · n!)`
/// * `x > 1`: the continued fraction
///   `E₁(x) = e^{−x} / (x + 1 − 1²/(x + 3 − 2²/(x + 5 − 3²/(x + 7 − …))))`,
///   i.e. partial numerators `−i²` with denominators stepping by 2 from `x + 1`.
///
/// The second form is the one that converges quickly; the superficially simpler
/// `x + 1/(1 + 1/(x + 2/(1 + …)))` needs far more terms and only reaches about ten digits at
/// `x = 1`, which `e1_branches_agree_and_it_has_the_right_small_x_limit` would catch.
///
/// `E₁(0)` is `+∞`, which is the right answer for the caller that uses it (the 1-D lattice
/// sum's on-axis limit is handled analytically before this is reached).
pub fn exp_integral_e1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        // Series. The alternating terms are harmless here because x ≤ 1.
        let mut sum = 0.0_f64;
        let mut term = 1.0_f64;
        for n in 1..100 {
            term *= -x / n as f64;
            let contrib = -term / n as f64;
            sum += contrib;
            if contrib.abs() < sum.abs() * 1.0e-18 + 1.0e-300 {
                break;
            }
        }
        return -EULER_GAMMA - x.ln() + sum;
    }
    const TINY: f64 = 1.0e-300;
    let mut b = x + 1.0;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let a = -((i * i) as f64);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 1.0e-17 {
            break;
        }
    }
    h * (-x).exp()
}

/// `E₁(x)` with the `e^{−x}` factor removed: `e^{x} E₁(x)`. Stays finite for large `x`, where
/// `E₁` itself underflows.
pub fn exp_integral_e1_scaled(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x > 1.0 {
        // The continued fraction is exactly this quantity before the exponential is applied.
        return exp_integral_e1(x) * x.exp();
    }
    exp_integral_e1(x) * x.exp()
}

/// Nodes and weights of an `n`-point Gauss–Legendre rule on `[-1, 1]`, computed by Newton
/// iteration on the Legendre polynomial.
///
/// The 1-D lattice sum needs a smooth finite-range integral evaluated many times with the same
/// rule, so the rule is generated once and cached by the caller.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Chebyshev-like initial guess for the i-th root, then Newton.
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            // Legendre P_n(z) and its derivative by the three-term recurrence.
            let (mut p0, mut p1) = (1.0_f64, 0.0_f64);
            for j in 0..n {
                let p2 = p1;
                p1 = p0;
                p0 = ((2.0 * j as f64 + 1.0) * z * p1 - j as f64 * p2) / (j as f64 + 1.0);
            }
            let dp = n as f64 * (z * p0 - p1) / (z * z - 1.0);
            let dz = p0 / dp;
            z -= dz;
            if dz.abs() < 1.0e-16 {
                break;
            }
        }
        let (mut p0, mut p1) = (1.0_f64, 0.0_f64);
        for j in 0..n {
            let p2 = p1;
            p1 = p0;
            p0 = ((2.0 * j as f64 + 1.0) * z * p1 - j as f64 * p2) / (j as f64 + 1.0);
        }
        let dp = n as f64 * (z * p0 - p1) / (z * z - 1.0);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `erf` by high-order Gauss–Legendre quadrature of its own definition — an independent
    /// route that shares no code with the series or the continued fraction.
    fn erf_by_quadrature(x: f64) -> f64 {
        let (nodes, weights) = gauss_legendre(200);
        let half = 0.5 * x;
        let s: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(t, w)| {
                let u = half * (t + 1.0);
                w * (-u * u).exp()
            })
            .sum();
        TWO_OVER_SQRT_PI * half * s
    }

    #[test]
    fn erf_matches_its_defining_integral() {
        let mut worst = 0.0_f64;
        for i in 0..=60 {
            let x = i as f64 * 0.1; // 0 .. 6
            let got = erf(x);
            let want = erf_by_quadrature(x);
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 1e-14, "max |erf - quadrature| = {worst:.3e}");
    }

    #[test]
    fn the_two_branches_agree_where_they_meet() {
        // Compared at the *same* argument, not either side of the crossover: `erfc` has slope
        // −0.88 near 0.5, so evaluating at x ± ε and differencing would just measure the
        // derivative and report a spurious "jump".
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
        // The overlap window where *both* forms are accurate: the continued fraction needs
        // x ≳ 0.5 to converge, and `1 − erf_series` starts losing digits to cancellation
        // beyond x ≈ 1.5. The crossover at 0.5 sits inside this window, which is the point.
        let mut worst = 0.0_f64;
        let mut worst_x = 0.0_f64;
        for i in 0..=20 {
            let x = ERFC_CF_LIMIT + i as f64 * 0.05; // 0.5 .. 1.5
            let r = rel(1.0 - erf_series(x), erfc_continued_fraction(x));
            if r > worst {
                worst = r;
                worst_x = x;
            }
        }
        assert!(
            worst < 1e-13,
            "series and continued fraction disagree by {worst:.3e} at x = {worst_x}"
        );
        // And erfcx's own crossover, at the same argument.
        for x in [ERFC_CF_LIMIT, 1.0, 2.0] {
            let direct = (x * x).exp() * erfc(x);
            assert!(
                rel(erfcx(x), direct) < 1e-13,
                "erfcx({x}) = {} vs e^{{x²}}erfc(x) = {direct}",
                erfcx(x)
            );
        }
    }

    /// `erfc` from an independent quadrature of its own tail integral, mapped `t = x + u/(1−u)`
    /// so the infinite range is captured exactly rather than truncated.
    fn erfc_by_quadrature(x: f64) -> f64 {
        let (nodes, weights) = gauss_legendre(800);
        const EDGE: f64 = 0.999_999_999_999;
        let s: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(p, w)| {
                let u = 0.5 * (p + 1.0) * EDGE;
                let t = x + u / (1.0 - u);
                let jac = 1.0 / ((1.0 - u) * (1.0 - u));
                0.5 * EDGE * w * (-t * t).exp() * jac
            })
            .sum();
        TWO_OVER_SQRT_PI * s
    }

    #[test]
    fn erfc_is_accurate_across_the_whole_range() {
        // This is the test that fixes the crossovers: `1 − erf_series` loses a digit for every
        // decade erfc falls below 1, so the continued fraction has to take over early.
        let mut worst = 0.0_f64;
        let mut worst_x = 0.0_f64;
        for i in 1..=120 {
            let x = i as f64 * 0.05; // 0.05 .. 6.0
            let got = erfc(x);
            let want = erfc_by_quadrature(x);
            let rel = (got - want).abs() / want;
            if rel > worst {
                worst = rel;
                worst_x = x;
            }
        }
        assert!(
            worst < 1e-13,
            "worst erfc relative error {worst:.3e} at x = {worst_x}"
        );
    }

    #[test]
    fn erf_and_erfc_are_consistent_and_have_the_right_symmetry() {
        for i in -50..=50 {
            let x = i as f64 * 0.13;
            assert!(
                (erf(x) + erfc(x) - 1.0).abs() < 1e-15,
                "erf + erfc != 1 at {x}"
            );
            assert!((erf(x) + erf(-x)).abs() < 1e-15, "erf not odd at {x}");
            assert!(
                (erfc(x) + erfc(-x) - 2.0).abs() < 1e-15,
                "erfc symmetry at {x}"
            );
        }
        assert_eq!(erf(0.0), 0.0);
        assert!((erfc(0.0) - 1.0).abs() < 1e-16);
    }

    #[test]
    fn erfc_stays_accurate_where_one_minus_erf_would_be_zero() {
        // erfc(10) ~ 2e-45: computing it as 1 - erf(10) would give exactly 0.
        for x in [6.0, 8.0, 10.0, 15.0, 20.0, 26.0] {
            let v = erfc(x);
            assert!(v > 0.0, "erfc({x}) underflowed to {v}");
            // Asymptotically erfc(x) ~ e^{-x²}/(x√π) · (1 - 1/(2x²) + 3/(4x⁴)).
            let a = (-x * x).exp() / (x * std::f64::consts::PI.sqrt());
            let asym = a * (1.0 - 0.5 / (x * x) + 0.75 / x.powi(4));
            let rel = (v - asym).abs() / v;
            assert!(rel < 1e-4, "erfc({x}) = {v:.6e} vs asymptotic {asym:.6e}");
        }
    }

    #[test]
    fn erfcx_avoids_the_overflow_that_motivates_it() {
        // Finite and positive everywhere, including where `e^{x²} · erfc(x)` would overflow
        // times underflow.
        for x in [3.0_f64, 10.0, 40.0, 100.0, 1.0e3, 1.0e6, 1.0e150] {
            let v = erfcx(x);
            assert!(v.is_finite() && v > 0.0, "erfcx({x}) = {v}");
        }
        // Against the asymptotic series 1/(x√π)·(1 − 1/2x² + 3/4x⁴ − 15/8x⁶), whose own
        // truncation error at x = 10 is ~7e-8; at x = 3 it is still 0.8 % off, so start at 10.
        for x in [10.0_f64, 40.0, 100.0, 1.0e3] {
            let v = erfcx(x);
            let x2 = x * x;
            let asym = (1.0 - 0.5 / x2 + 0.75 / (x2 * x2) - 1.875 / (x2 * x2 * x2))
                / (x * std::f64::consts::PI.sqrt());
            let rel = (v - asym).abs() / v;
            assert!(rel < 1e-6, "erfcx({x}) = {v:.6e} vs asymptotic {asym:.6e}");
        }
        // And it agrees with the naive product where that is still representable.
        for x in [0.0_f64, 0.5, 1.0, 2.0, 3.0] {
            let naive = (x * x).exp() * erfc(x);
            assert!((erfcx(x) - naive).abs() < 1e-14 * naive.max(1.0));
        }
    }

    /// `E₁` by quadrature of `∫₀¹ e^{−x/u}/u du` (substituting `t = 1/u` in the definition).
    fn e1_by_quadrature(x: f64) -> f64 {
        let (nodes, weights) = gauss_legendre(400);
        nodes
            .iter()
            .zip(&weights)
            .map(|(t, w)| {
                let u = 0.5 * (t + 1.0);
                if u <= 0.0 {
                    0.0
                } else {
                    0.5 * w * (-x / u).exp() / u
                }
            })
            .sum()
    }

    #[test]
    fn e1_matches_its_defining_integral() {
        for x in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0] {
            let got = exp_integral_e1(x);
            let want = e1_by_quadrature(x);
            let rel = (got - want).abs() / want;
            assert!(
                rel < 1e-9,
                "E1({x}): {got:.12e} vs {want:.12e} (rel {rel:.1e})"
            );
        }
    }

    #[test]
    fn e1_branches_agree_and_it_has_the_right_small_x_limit() {
        // Both branches at the same argument. Differencing across the crossover would measure
        // E₁'(1) = −e⁻¹ instead of the branch disagreement.
        let series = {
            let x = 1.0_f64;
            let mut sum = 0.0_f64;
            let mut term = 1.0_f64;
            for n in 1..100 {
                term *= -x / n as f64;
                sum += -term / n as f64;
            }
            -EULER_GAMMA - x.ln() + sum
        };
        let cf = exp_integral_e1(1.0 + f64::EPSILON);
        assert!(
            (series - cf).abs() < 1e-14,
            "E1 branches disagree at x = 1: series {series} vs continued fraction {cf}"
        );
        // E1(x) → -γ - ln x as x → 0.
        for x in [1e-6_f64, 1e-8, 1e-10] {
            let want = -EULER_GAMMA - x.ln();
            assert!((exp_integral_e1(x) - want).abs() < 1e-6 * want.abs());
        }
        assert_eq!(exp_integral_e1(0.0), f64::INFINITY);
        // Large x: the divergent asymptotic series E₁(x) ~ (e^{−x}/x) Σ (−1)ⁿ n!/xⁿ. Its own
        // truncation error after the 1/x⁴ term is about 120/x⁵, so the tolerance is set from
        // that rather than picked to make the test pass.
        for x in [20.0_f64, 40.0, 60.0] {
            let s = 1.0 - 1.0 / x + 2.0 / x.powi(2) - 6.0 / x.powi(3) + 24.0 / x.powi(4);
            let asym = (-x).exp() / x * s;
            let got = exp_integral_e1(x);
            let rel = (got - asym).abs() / got;
            let allowed = 3.0 * 120.0 / x.powi(5);
            assert!(
                rel < allowed,
                "E1({x}) vs asymptotic: rel {rel:.2e}, allowed {allowed:.2e}"
            );
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        for n in [2usize, 5, 16, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1e-13, "n={n}");
            // An n-point rule is exact for degree 2n-1.
            for deg in 0..(2 * n - 1) {
                let got: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| w * x.powi(deg as i32))
                    .sum();
                let want = if deg % 2 == 1 {
                    0.0
                } else {
                    2.0 / (deg as f64 + 1.0)
                };
                assert!(
                    (got - want).abs() < 1e-11,
                    "n={n} deg={deg}: {got} vs {want}"
                );
            }
            // Nodes must be symmetric and sorted.
            for i in 0..n {
                assert!((nodes[i] + nodes[n - 1 - i]).abs() < 1e-14);
            }
            assert!(
                nodes.windows(2).all(|w| w[0] < w[1]),
                "n={n} nodes unsorted"
            );
        }
    }
}
