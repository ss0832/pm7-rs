// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed-size indexed loops and compact tuples mirror published NDDO equations. Iterator
// rewrites or wrapper structs obscure orbital indices without improving safety or performance.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Several literals are copied from MOPAC tables and must retain the exact f64 rounding.
#![allow(clippy::excessive_precision)]
//! # pm7-rs
//!
//! A Rust-native implementation of the **PM7** semiempirical NDDO method and its
//! PM7-TS, PM7-minus, PM7-HH, and Sparkle/PM7 methods.  The parameter tables are
//! generated from a pinned MOPAC v23.2.5 checkout; the common SCF, derivative, and
//! optimization APIs are shared by every method.
//!
//! Reference: J. J. P. Stewart, *J. Mol. Model.* **19**, 1 (2013).

pub mod basis;
pub mod constants;
pub mod data_tables;
pub mod dispersion;
pub mod dual;
pub mod dual2;
pub(crate) mod dual2n;
pub mod error;
pub mod fock;
pub mod gradient;
pub mod hamiltonian;
pub mod hbond;
pub mod hessian;
pub mod hh_rep;
pub mod integrals;
pub mod linalg;
pub mod math;
pub mod memory;
pub mod mndod;
pub mod mndod_tables;
pub mod mndod_twocenter;
pub mod optimizer;
pub mod overlap;
pub mod overlap_d;
pub mod overlap_numeric;
pub mod params;
pub mod repulsion;
pub(crate) mod rotfix;
pub mod scf;
pub mod system;
pub mod method;

#[cfg(feature = "python")]
pub mod python;

pub use error::{Pm7Error, Result};
pub use gradient::{
    analytic_gradient, closed_form_gradient, electronic_gradient_fixed_density,
    energy_at_fixed_density, fixed_density_gradient, numerical_gradient, GradientResult,
};
pub use hessian::{analytic_hessian, numerical_hessian, vibrational_analysis, VibrationalModes};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use optimizer::{optimize, OptOptions, OptResult};
pub use params::{Pm7Element, Pm7Pair, Pm7Parameters};
pub use scf::{run_pm7, Pm7Calculator, Pm7Options, Pm7Result, ScfAccelerator, ScfReference};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};
pub use method::Pm7Method;
