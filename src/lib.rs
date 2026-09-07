// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed-size indexed loops and compact tuples mirror published NDDO equations. Iterator
// rewrites or wrapper structs obscure orbital indices without improving safety or performance.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Several literals are copied from MOPAC tables and must retain the exact f64 rounding.
#![allow(clippy::excessive_precision)]
// The crate has exactly one `unsafe` block: the Windows `GlobalMemoryStatusEx` call in
// `memory::available_memory_bytes`, which carries an `#[allow(unsafe_code)]` and a `// SAFETY:`
// note. Denying it here makes "there is only one" an invariant the compiler enforces rather than
// a claim that quietly stops being true.
#![deny(unsafe_code)]
//! # pm7-rs
//!
//! A Rust-native implementation of the **PM7** semiempirical NDDO method and its
//! PM7-TS, PM7-minus, PM7-HH, and Sparkle/PM7 methods.  The parameter tables are
//! generated from a pinned MOPAC v23.2.5 checkout; the common SCF, derivative, and
//! optimization APIs are shared by every method.
//!
//! Reference: J. J. P. Stewart, *J. Mol. Model.* **19**, 1 (2013).
//!
//! ## Periodic systems
//!
//! Attaching a [`Cell`] to a [`Molecule`] turns it into a 1-D polymer, 2-D layer, or 3-D crystal,
//! and every entry point — [`run_pm7`], [`closed_form_gradient`], [`analytic_hessian`],
//! [`optimize`] — dispatches on it. Neutral and charged cells are both supported. See
//! `docs/pbc.md` for the formalism and [`pbc::PbcOptions`] for the convergence parameters.

pub mod basis;
pub mod cell;
pub mod cmatrix;
pub mod constants;
pub mod dandc;
pub mod data_tables;
pub mod dfpt;
pub mod dipole;
pub mod dispersion;
pub mod dual;
pub mod dual2;
pub(crate) mod dual2n;
pub mod error;
pub mod field;
pub mod fock;
pub mod gradient;
pub mod hamiltonian;
pub mod hbond;
pub mod hessian;
pub mod hessian_pbc;
pub mod hh_rep;
pub mod integrals;
pub mod ir;
pub mod linalg;
pub mod math;
pub mod memory;
pub mod method;
pub mod mm_corrections;
pub mod mndod;
pub mod mndod_tables;
pub mod mndod_twocenter;
pub mod molden;
pub mod optimizer;
pub mod overlap;
pub mod overlap_d;
pub mod overlap_numeric;
pub mod params;
pub mod pbc;
pub mod profile;
pub mod projection;
pub mod repulsion;
pub(crate) mod rotfix;
pub mod scf;
pub mod scf_pbc;
pub mod spatial;
pub mod special;
pub mod stability;
pub mod stress;
pub mod system;

#[cfg(feature = "python")]
pub mod python;

pub use cell::{AxisRotation, Cell, Periodicity};
pub use dandc::{dandc_derivatives, run_dandc, DandcDerivatives, DandcOptions, DandcResult};
pub use dfpt::{
    born_and_dielectric, dielectric_origin_sensitivity, dielectric_with_extent,
    dynamical_matrix_dfpt, polarizability, static_dielectric_tensor, DfptFieldResult, DfptOptions,
    DfptResult, ExtentDielectric, LongRange, NonAnalytic, StaticDielectric,
};
pub use dipole::{DipoleBreakdown, DipoleOrigin, DipoleTerms};
pub use error::{Pm7Error, Result};
pub use field::ExternalField;
pub use gradient::{
    analytic_gradient, closed_form_gradient, electronic_gradient_fixed_density,
    energy_at_fixed_density, fixed_density_gradient, numerical_gradient, GradientResult,
};
pub use hessian::{
    analytic_hessian, analytic_hessian_with, numerical_hessian, vibrational_analysis,
    vibrational_analysis_projected, vibrational_modes_from, vibrational_modes_projected,
    HessianRequest, HessianResult, OrbitalResponse, VibrationalModes,
};
pub use hessian_pbc::{analytic_hessian_periodic, force_constants, ForceConstants, PhononModes};
pub use ir::{dipole_derivatives, ir_spectrum, ir_spectrum_projected, IrSpectrum};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use method::Pm7Method;
pub use molden::{fit_shell, to_molden, write_molden, MoldenBasis, MoldenOptions};
pub use optimizer::{optimize, OptOptions, OptResult};
pub use params::{Pm7Element, Pm7Pair, Pm7Parameters};
pub use pbc::berry::{berry_polarization, BerryPolarization};
pub use pbc::finite_field::{run_finite_field, FiniteFieldOptions, FiniteFieldResult};
pub use pbc::{
    epsilon_from_polarizability, BackgroundCharge, ExtentConvention, KMesh, PbcMode, PbcOptions,
    SheetInvariants, Smearing,
};
pub use projection::{Projection, RigidSubspace, LINEARITY_TOLERANCE};
pub use scf::{run_pm7, Pm7Calculator, Pm7Options, Pm7Result, ScfAccelerator, ScfReference};
pub use scf_pbc::{band_structure, BandStructure};
pub use stress::{analytic_stress, StressResult};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};
