// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic boundary conditions for the PM7 family: 1-D polymers, 2-D layers, and 3-D crystals,
//! at the Γ point or on a k-point mesh, for neutral **and** charged cells.
//!
//! Two electrostatic treatments are available, and they answer different questions.
//!
//! # [`PbcMode::Ewald`] — the default, physically rigorous path
//!
//! PM7 always runs with MOPAC's `l_feather` switch on, which means every two-centre integral
//! becomes **exactly** the point-charge value `q_A q_B / r` beyond 7 Å (see
//! [`crate::integrals::feather_to_point`]). That is not an approximation we impose — it is part
//! of the published model. It lets the lattice sum be split with no arbitrary truncation:
//!
//! ```text
//! I_{μν,λσ}(R) = [ I_{μν,λσ}(R) − δ_{μν} δ_{λσ} v_point(R) ] + δ_{μν} δ_{λσ} v_point(R)
//!                 └─ identically zero for R > 7 Å: compact support ─┘   └─ 1/r: Ewald ─┘
//! ```
//!
//! The bracket is summed in real space over a short neighbour list; the monopole part is a
//! `1/r` lattice sum over the net atomic charges `q_A = Z_A − P_A` and is evaluated by Ewald
//! summation, which is absolutely convergent and independent of the splitting parameter.
//! Exchange is *not* a Coulomb series — it decays with the density matrix — so it is cut by an
//! explicit, convergence-testable real-space cutoff instead.
//!
//! A **charged cell** (`Σ_A q_A ≠ 0`) is supported: the divergent `G = 0` term is removed and a
//! uniform neutralizing background is added. See [`ewald`] for the background term and its
//! stress contribution.
//!
//! # [`PbcMode::MopacCluster`] — the compatibility gate
//!
//! MOPAC's own solid-state path is a *truncated* real-space image sum with two asymmetries that
//! have to be reproduced exactly to match its published PM7 solid-state heats of formation:
//! the Coulomb integrals are summed over **all** images while the exchange integrals come from
//! the **nearest image only** (`solrot.F90:93-102`), and beyond `clower` the integrals become
//! bare point charges with distances compressed toward `cutofp` by `trunk`. This mode exists to
//! validate against MOPAC, not to be the default: its absolute energy depends on the truncation.

pub mod berry;
pub mod ewald;
pub mod extent;
pub mod finite_field;
pub mod images;
pub mod kpoints;
pub mod trunc;

use crate::error::{Pm7Error, Result};

pub use ewald::{EwaldParameters, EwaldPotential};
pub use extent::{
    epsilon_from_polarizability, extent_axis_mixing, ExtentConvention, SheetInvariants,
};
pub use images::{ImagePair, PairList};
pub use kpoints::{KMesh, KPoint, KPointSet};

/// Distance (Bohr) beyond which PM7's feathering has made every two-centre integral exactly a
/// point charge. `7 Å` in MOPAC's `to_point` / `nddo_to_point`.
pub const FEATHER_RANGE_BOHR: f64 = 7.0 * crate::constants::ANGSTROM_TO_BOHR;

/// Fraction of the correction cutoff over which the pairwise post-SCF terms are tapered to zero.
///
/// A hard cutoff makes the energy jump whenever a pair crosses it, which is invisible in a
/// single-point energy and fatal everywhere else: the jump shows up as a `1/h` blow-up in any
/// finite difference, and as a discontinuous force in an optimization or an MD run. The taper is
/// small (the tapered region contributes ~1e-7 eV) but it is the difference between a smooth
/// potential-energy surface and one with steps in it.
pub const CORRECTION_TAPER_FRACTION: f64 = 0.15;

/// A C² smooth taper: `1` below `r_on`, `0` above `r_cut`, and the quintic smootherstep
/// `1 − t³(6t² − 15t + 10)` in between. Returns `(w, dw/dr)`.
///
/// Both the value and the first two derivatives are continuous at each end, so a tapered pair
/// term keeps an analytic gradient *and* an analytic Hessian.
#[inline]
pub fn taper(r: f64, r_on: f64, r_cut: f64) -> (f64, f64) {
    if r <= r_on {
        return (1.0, 0.0);
    }
    if r >= r_cut {
        return (0.0, 0.0);
    }
    let span = r_cut - r_on;
    let t = (r - r_on) / span;
    let s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let ds = 30.0 * t * t * (t * (t - 2.0) + 1.0) / span;
    (1.0 - s, -ds)
}

/// [`taper`] as a generic scalar, so second-order dual arithmetic gets `w`, `w'` and `w''` from
/// one expression instead of three hand-differentiated ones.
///
/// The branch is taken on the *value*, which is what a piecewise-C² function requires: the
/// smootherstep and its first two derivatives all vanish at both ends, so the pieces join without
/// a step in anything the Hessian can see.
pub fn taper_scalar<S: crate::dual::Scalar>(r: S, r_on: f64, r_cut: f64) -> S {
    if r.val() <= r_on {
        return S::cst(1.0);
    }
    if r.val() >= r_cut {
        return S::cst(0.0);
    }
    let span = r_cut - r_on;
    let t = (r - r_on) / span;
    let s = t * t * t * ((t * (t * 6.0 - 15.0)) + 10.0);
    S::cst(1.0) - s
}

/// Where the taper starts, for a given cutoff. Infinite cutoffs (molecules) never taper.
#[inline]
pub fn taper_onset(cutoff: f64) -> f64 {
    if cutoff.is_finite() {
        cutoff * (1.0 - CORRECTION_TAPER_FRACTION)
    } else {
        f64::INFINITY
    }
}

/// Which periodic electrostatics to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PbcMode {
    /// Ewald-summed monopoles plus a compact-support short-range remainder. Absolutely
    /// convergent, splitting-parameter independent, and defined for charged cells.
    #[default]
    Ewald,
    /// MOPAC's truncated image sum (`hcore.F90` `id /= 0`, `solrot.F90`, `trunk`). Reproduces
    /// MOPAC's solid-state numbers; Γ point only.
    MopacCluster,
}

impl std::fmt::Display for PbcMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PbcMode::Ewald => "ewald",
            PbcMode::MopacCluster => "mopac",
        })
    }
}

impl std::str::FromStr for PbcMode {
    type Err = Pm7Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ewald" | "" => Ok(PbcMode::Ewald),
            "mopac" | "cluster" | "mopac-cluster" => Ok(PbcMode::MopacCluster),
            other => Err(Pm7Error::InvalidInput(format!(
                "unknown PBC mode `{other}` (expected ewald or mopac)"
            ))),
        }
    }
}

/// Occupation broadening for metallic systems.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Smearing {
    /// Integer aufbau filling. Correct for a system with a gap; fails to converge for a metal.
    #[default]
    None,
    /// Fermi–Dirac at an electronic temperature. `width` is `k_B T` in eV.
    FermiDirac { width_ev: f64 },
    /// Gaussian broadening. `width` in eV.
    Gaussian { width_ev: f64 },
    /// Methfessel–Paxton of the given order. `width` in eV.
    MethfesselPaxton { width_ev: f64, order: usize },
}

impl Smearing {
    pub fn width_ev(self) -> f64 {
        match self {
            Smearing::None => 0.0,
            Smearing::FermiDirac { width_ev }
            | Smearing::Gaussian { width_ev }
            | Smearing::MethfesselPaxton { width_ev, .. } => width_ev,
        }
    }

    fn validate(self) -> Result<()> {
        if let Smearing::MethfesselPaxton { order, .. } = self {
            if order > 4 {
                return Err(Pm7Error::InvalidInput(
                    "Methfessel-Paxton order above 4 is numerically useless".into(),
                ));
            }
        }
        let w = self.width_ev();
        if !matches!(self, Smearing::None) && (!w.is_finite() || w <= 0.0) {
            return Err(Pm7Error::InvalidInput(
                "smearing width must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// How the charged-cell divergence is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackgroundCharge {
    /// Uniform neutralizing background (jellium): drop the `G = 0` term and add the
    /// corresponding constant. The standard, and the only choice that keeps the energy finite
    /// and splitting-parameter independent.
    #[default]
    Jellium,
    /// Refuse to run a charged cell. Useful when a charged cell would be a modelling mistake
    /// (the absolute energy is background-convention dependent) and the caller wants to be told.
    Forbid,
}

/// Periodic-calculation settings.
///
/// Every cutoff is in **Bohr** and is a public, convergence-testable parameter: the defaults are
/// chosen to be safe, not to be a hidden approximation. `docs/pbc.md` records the convergence
/// data behind each one.
#[derive(Clone, Debug, PartialEq)]
pub struct PbcOptions {
    pub mode: PbcMode,
    /// k-point sampling. `KMesh::Gamma` is the default.
    pub kmesh: KMesh,
    pub smearing: Smearing,
    /// Real-space cutoff for the short-range (non-point-charge) part of the two-centre
    /// integrals. Must be at least [`FEATHER_RANGE_BOHR`], beyond which the remainder is exactly
    /// zero, so the default carries a margin rather than an approximation.
    pub short_range_cutoff: f64,
    /// Real-space cutoff for the exchange contribution, which decays with the density matrix
    /// rather than as `1/r`. This one *is* an approximation and must be converged.
    pub exchange_cutoff: f64,
    /// Real-space cutoff for the pairwise post-SCF corrections (dispersion, PM7-HH). The
    /// dispersion `−C6/R⁶` converges absolutely, so this is a truncation error that shrinks
    /// as `R⁻³`.
    pub correction_cutoff: f64,
    /// Ewald splitting parameter `α` (Bohr⁻¹). `None` picks a near-optimal value from the cell
    /// size and atom count. Explicit values exist so a test can prove α-independence.
    pub ewald_alpha: Option<f64>,
    /// Ewald real-space and reciprocal-space accuracy target (relative). Sets the two cutoffs.
    pub ewald_accuracy: f64,
    pub background: BackgroundCharge,
    /// Report Makov–Payne finite-size corrections for a charged cell as a diagnostic. They are
    /// never added to the energy: they estimate the error of the periodic model, and adding
    /// them silently would make the energy inconsistent with its own gradient and stress.
    pub report_makov_payne: bool,
}

impl Default for PbcOptions {
    fn default() -> Self {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Self {
            mode: PbcMode::Ewald,
            kmesh: KMesh::Gamma,
            smearing: Smearing::None,
            // 9 Å: 7 Å is where the remainder is identically zero, plus margin.
            short_range_cutoff: 9.0 * a,
            exchange_cutoff: 15.0 * a,
            correction_cutoff: 30.0 * a,
            ewald_alpha: None,
            ewald_accuracy: 1.0e-10,
            background: BackgroundCharge::Jellium,
            report_makov_payne: true,
        }
    }
}

impl PbcOptions {
    /// Γ-point Ewald defaults.
    pub fn gamma() -> Self {
        Self::default()
    }

    /// Monkhorst–Pack `n1 × n2 × n3` sampling with otherwise default settings.
    pub fn monkhorst_pack(n: [usize; 3]) -> Self {
        Self {
            kmesh: KMesh::MonkhorstPack {
                n,
                shift: [0.0; 3],
                gamma_centred: true,
            },
            ..Self::default()
        }
    }

    /// MOPAC-compatibility defaults (Γ point, MOPAC's truncated image sum).
    pub fn mopac_compatible() -> Self {
        Self {
            mode: PbcMode::MopacCluster,
            kmesh: KMesh::Gamma,
            short_range_cutoff: trunc::CUTOFP_BOHR,
            exchange_cutoff: trunc::CUTOFP_BOHR,
            correction_cutoff: trunc::CUTOFP_BOHR,
            ..Self::default()
        }
    }

    pub fn with_smearing(mut self, smearing: Smearing) -> Self {
        self.smearing = smearing;
        self
    }

    /// The largest cutoff any term uses — the radius the shared neighbour list must cover.
    pub fn max_cutoff(&self) -> f64 {
        self.short_range_cutoff
            .max(self.exchange_cutoff)
            .max(self.correction_cutoff)
    }

    pub fn validate(&self) -> Result<()> {
        let cutoffs = [
            ("short_range_cutoff", self.short_range_cutoff),
            ("exchange_cutoff", self.exchange_cutoff),
            ("correction_cutoff", self.correction_cutoff),
        ];
        for (name, v) in cutoffs {
            if !v.is_finite() || v <= 0.0 {
                return Err(Pm7Error::InvalidInput(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        if self.mode == PbcMode::Ewald && self.short_range_cutoff < FEATHER_RANGE_BOHR {
            return Err(Pm7Error::InvalidInput(format!(
                "short_range_cutoff ({:.2} Bohr) is below the {:.2} Bohr feather range, where the \
                 short-range remainder is still non-zero; the Ewald split would drop a real term",
                self.short_range_cutoff, FEATHER_RANGE_BOHR
            )));
        }
        if let Some(alpha) = self.ewald_alpha {
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(Pm7Error::InvalidInput(
                    "ewald_alpha must be finite and positive".into(),
                ));
            }
        }
        if !self.ewald_accuracy.is_finite() || self.ewald_accuracy <= 0.0 {
            return Err(Pm7Error::InvalidInput(
                "ewald_accuracy must be finite and positive".into(),
            ));
        }
        self.smearing.validate()?;
        self.kmesh.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_cover_the_feather_range() {
        let o = PbcOptions::default();
        o.validate().unwrap();
        assert!(o.short_range_cutoff >= FEATHER_RANGE_BOHR);
        assert_eq!(o.mode, PbcMode::Ewald);
    }

    #[test]
    fn a_short_range_cutoff_inside_the_feather_range_is_rejected() {
        let o = PbcOptions {
            short_range_cutoff: FEATHER_RANGE_BOHR * 0.5,
            ..PbcOptions::default()
        };
        assert!(o.validate().is_err());
    }

    #[test]
    fn modes_and_smearing_round_trip_through_strings() {
        use std::str::FromStr;
        assert_eq!(PbcMode::from_str("ewald").unwrap(), PbcMode::Ewald);
        assert_eq!(PbcMode::from_str("MOPAC").unwrap(), PbcMode::MopacCluster);
        assert!(PbcMode::from_str("nope").is_err());
        assert_eq!(PbcMode::Ewald.to_string(), "ewald");

        assert!(Smearing::FermiDirac { width_ev: -1.0 }.validate().is_err());
        assert!(Smearing::FermiDirac { width_ev: 0.1 }.validate().is_ok());
        assert!(Smearing::None.validate().is_ok());
    }
}
