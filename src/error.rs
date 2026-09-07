// SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;

pub type Result<T> = std::result::Result<T, Pm7Error>;

/// Errors raised across the PM7 pipeline.
///
/// Parallels `gfn1-rs`'s `Gfn1Error`, renamed and trimmed for the molecular NDDO
/// method: there is no global-parameter table and no periodic cell, and SCF
/// non-convergence is reported on the density residual rather than a shell-charge rms.
#[derive(Debug)]
pub enum Pm7Error {
    Io(std::io::Error),
    Parse {
        line: usize,
        message: String,
    },
    InvalidInput(String),
    /// No PM7 parameter block exists for this atomic number.
    MissingElement(u8),
    /// A named per-element or derived parameter is absent.
    MissingParameter(String),
    /// A deliberately staged capability has not yet reached its implementation milestone.
    UnsupportedFeature(String),
    LinearAlgebra(String),
    /// The SCF loop hit `max_scf` without reaching the density/energy tolerance.
    ScfNotConverged {
        iterations: usize,
        error: f64,
    },
    /// A linear-response solve failed to converge, or produced a result that violates an
    /// invariant it is required to satisfy.
    ///
    /// Separate from [`Self::ScfNotConverged`] because the failure mode is different: the response
    /// is a *linear* fixed point, so it does not wander near an answer, it diverges geometrically
    /// and returns a number many orders of magnitude too large. It has to be refused rather than
    /// reported with a flag.
    ResponseFailed(String),
    /// The estimated peak memory for this system exceeds the configured/available budget.
    /// Raised *before* the large allocations, so the process fails cleanly instead of being
    /// OOM-killed mid-run.
    InsufficientMemory {
        needed_mb: u64,
        budget_mb: u64,
        what: &'static str,
    },
}

impl fmt::Display for Pm7Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement(z) => write!(f, "missing PM7 parameter block for Z={z}"),
            Self::MissingParameter(key) => write!(f, "missing PM7 parameter `{key}`"),
            Self::UnsupportedFeature(message) => write!(f, "unsupported PM7 feature: {message}"),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::ScfNotConverged { iterations, error } => write!(
                f,
                "PM7 SCF did not converge after {iterations} iterations (error={error:.3e})"
            ),
            Self::ResponseFailed(msg) => write!(f, "linear response failed: {msg}"),
            Self::InsufficientMemory {
                needed_mb,
                budget_mb,
                what,
            } => write!(
                f,
                "estimated {what} memory ~{needed_mb} MB exceeds the {budget_mb} MB budget; \
                 raise it via Pm7Options::max_memory_mb or the PM7_MEM_BUDGET_MB env var, \
                 or reduce the system size"
            ),
        }
    }
}

impl std::error::Error for Pm7Error {}

impl From<std::io::Error> for Pm7Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
