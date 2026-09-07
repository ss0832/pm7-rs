// SPDX-License-Identifier: GPL-3.0-or-later

//! MOPAC's solid-state distance truncation (`trunk` / `derp`), used only by
//! [`crate::pbc::PbcMode::MopacCluster`].
//!
//! MOPAC does not Ewald-sum. Its periodic Coulomb sum is a truncated real-space image sum, made
//! convergent by **compressing distances**: beyond `clower` the effective distance grows more
//! and more slowly, and above `cutofp` every interaction is frozen at the same effective
//! distance, so the `1/r` tail becomes a finite constant times the (zero, for a neutral cell)
//! total charge. That is an approximation with no small parameter, which is exactly why it is
//! not the default here — but reproducing it bit-for-bit is the only way to compare against
//! MOPAC's published PM7 solid-state heats of formation.
//!
//! The compression function is `trunk` (MOPAC `solrot.F90:193-244`), differentiated by `derp`
//! (`dcart.F90:459-501`). It is a plain **quadratic** between `clower` and `cupper`, fixed by
//! three conditions:
//!
//! * `r_eff(r) = r` and slope 1 at `r = clower`,
//! * slope 0 at `r = cupper` (which MOPAC sets equal to `cutofp`),
//! * constant above `cupper`.
//!
//! Note that the saturated value is **not** `cutofp`, despite MOPAC's own comment saying
//! "Above CUTOFP R=CUTOFP". Working the quadratic out gives `r_eff(∞) = (cutofp + clower)/2`,
//! i.e. 21.5 Å for the defaults — a factor that matters, because it is what the `1/r` tail of
//! every truncated interaction saturates to.

use crate::constants::ANGSTROM_TO_BOHR;

/// MOPAC's `clower`: below this the distance is untouched. Default 13 Å (`molkst_C.F90:217`).
pub const CLOWER_ANGSTROM: f64 = 13.0;
/// MOPAC's `cutofp`: the saturated effective distance. Default 30 Å for polymers, layers, and
/// solids (`molkst_C.F90:215`).
pub const CUTOFP_ANGSTROM: f64 = 30.0;

pub const CLOWER_BOHR: f64 = CLOWER_ANGSTROM * ANGSTROM_TO_BOHR;
pub const CUTOFP_BOHR: f64 = CUTOFP_ANGSTROM * ANGSTROM_TO_BOHR;

/// Parameters of the truncation function, in Ångström (MOPAC works in Ångström here).
#[derive(Clone, Copy, Debug)]
pub struct Truncation {
    pub clower: f64,
    pub cupper: f64,
    pub cutofp: f64,
}

impl Default for Truncation {
    fn default() -> Self {
        Self {
            clower: CLOWER_ANGSTROM,
            cupper: CUTOFP_ANGSTROM,
            cutofp: CUTOFP_ANGSTROM,
        }
    }
}

impl Truncation {
    /// MOPAC's `trunk` coefficients (`solrot.F90:206-218`), transcribed.
    ///
    /// With `b = clower/cutofp` and `range = cupper/cutofp − b`:
    /// `c = −½ b² cutofp / range`, `c_r = 1 + b/range`, `c_r2 = −1/(2 cutofp range)`.
    fn coefficients(&self) -> (f64, f64, f64) {
        let bound1 = self.clower / self.cutofp;
        let bound2 = self.cupper / self.cutofp;
        let range = bound2 - bound1;
        let c = -0.5 * bound1 * bound1 * self.cutofp / range;
        let cr = 1.0 + bound1 / range;
        let cr2 = -1.0 / (self.cutofp * 2.0 * range);
        (c, cr, cr2)
    }

    /// The constant that `r_eff` saturates to above `cupper` (MOPAC's `clim`).
    ///
    /// Working the quadratic out gives `(cutofp + clower)/2` when `cupper = cutofp`, which is
    /// 21.5 Å for MOPAC's defaults — not `cutofp` itself.
    pub fn saturated(&self) -> f64 {
        let (c, cr, cr2) = self.coefficients();
        c + cr * self.cupper + cr2 * self.cupper * self.cupper
    }

    /// The effective distance `r_eff(r)` and its derivative `dr_eff/dr`, both in Ångström.
    pub fn effective(&self, r: f64) -> (f64, f64) {
        if r <= self.clower {
            return (r, 1.0);
        }
        if r > self.cupper {
            return (self.saturated(), 0.0);
        }
        let (c, cr, cr2) = self.coefficients();
        (c + cr * r + cr2 * r * r, cr + 2.0 * cr2 * r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_below_clower_and_constant_above_cupper() {
        let t = Truncation::default();
        for r in [0.5_f64, 3.0, 12.999] {
            let (v, d) = t.effective(r);
            assert!((v - r).abs() < 1e-14, "r_eff({r}) = {v}");
            assert!((d - 1.0).abs() < 1e-14);
        }
        for r in [30.001_f64, 45.0, 1000.0] {
            let (v, d) = t.effective(r);
            assert!((v - t.saturated()).abs() < 1e-14, "r_eff({r}) = {v}");
            assert_eq!(d, 0.0);
        }
        // The saturated value is (cutofp + clower)/2 = 21.5 Å for MOPAC's defaults, *not*
        // cutofp. Pinning it here so the discrepancy with MOPAC's own comment stays visible.
        assert!(
            (t.saturated() - 21.5).abs() < 1e-12,
            "saturated r_eff = {}",
            t.saturated()
        );
    }

    #[test]
    fn the_compression_is_c1_at_both_ends() {
        let t = Truncation::default();
        let eps = 1e-7;
        for edge in [t.clower, t.cupper] {
            let (lo, dlo) = t.effective(edge - eps);
            let (hi, dhi) = t.effective(edge + eps);
            assert!(
                (lo - hi).abs() < 1e-6,
                "value jumps at {edge}: {lo} vs {hi}"
            );
            assert!(
                (dlo - dhi).abs() < 1e-5,
                "slope jumps at {edge}: {dlo} vs {dhi}"
            );
        }
    }

    #[test]
    fn the_analytic_derivative_matches_a_finite_difference() {
        let t = Truncation::default();
        let h = 1e-6;
        for r in [14.0_f64, 18.0, 22.0, 27.0, 29.5] {
            let (_, d) = t.effective(r);
            let fd = (t.effective(r + h).0 - t.effective(r - h).0) / (2.0 * h);
            assert!((d - fd).abs() < 1e-6, "r={r}: {d} vs {fd}");
        }
    }

    #[test]
    fn the_compression_is_monotone_and_bounded() {
        let t = Truncation::default();
        let mut prev = 0.0_f64;
        let mut r = 0.0_f64;
        while r < 40.0 {
            let (v, d) = t.effective(r);
            assert!(v >= prev - 1e-12, "not monotone at r={r}");
            assert!(
                v <= t.saturated() + 1e-12,
                "exceeds the saturated value at r={r}: {v}"
            );
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&d),
                "slope {d} out of range at r={r}"
            );
            prev = v;
            r += 0.01;
        }
    }
}
