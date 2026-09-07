// SPDX-License-Identifier: GPL-3.0-or-later
//! Python bindings for the PM7-family native API.

use crate::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::gradient::closed_form_gradient;
use crate::method::Pm7Method;
use crate::optimizer::{optimize as optimize_geometry, OptOptions};
use crate::params::Pm7Parameters;
use crate::scf::{run_pm7, Pm7Options, ScfReference};
use crate::system::{Atom, Molecule};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::str::FromStr;

/// eV/Bohr² → eV/Å² (Hessian, second derivative in length).
const EV_PER_BOHR2_TO_EV_PER_ANGSTROM2: f64 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;

fn to_py_err(error: crate::error::Pm7Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The periodic keywords every entry point accepts, grouped so they travel together.
///
/// All are optional and all default to the molecular behaviour, so an existing call that passes
/// none of them behaves exactly as it did before periodic support existed.
#[derive(Clone, Debug, Default)]
pub struct PeriodicArgs {
    /// Lattice vectors as three rows in Ångström, or `None` for a molecule.
    pub cell: Option<Vec<Vec<f64>>>,
    /// Which directions are periodic. The periodic ones must come first.
    pub pbc: Option<Vec<bool>>,
    /// Monkhorst–Pack divisions, or `None` for the Γ point.
    pub kpoints: Option<Vec<usize>>,
    pub kpoint_shift: Option<Vec<f64>>,
    /// `("fermi" | "gauss" | "mp", width_ev, order)`.
    pub smearing: Option<(String, f64, usize)>,
    /// `"ewald"` (default) or `"mopac"`.
    pub pbc_mode: Option<String>,
    /// SCF density-convergence threshold. `None` keeps the default.
    pub scf_tolerance: Option<f64>,
    /// Maximum SCF iterations. `None` keeps the default.
    pub max_scf: Option<usize>,
    /// Operator applications one CPHF orbital-response solve may spend. `None` keeps the default
    /// (100). A private constant through 0.2.2, which left an ill-conditioned cell with no remedy
    /// but a different entry point.
    pub cphf_max_iterations: Option<usize>,
    /// `"off"` (default), `"check"` or `"follow"`: whether to ask if the converged SCF solution is
    /// a minimum, and whether to escape it if not. See [`crate::stability`].
    pub stability: Option<String>,
    /// `(inner, outer)` in **Bohr** for the analytic Hessian's smooth long-range-exchange cutoff.
    /// `None` (the default) keeps the Hessian bit-identical. Rust-CLI-only through 0.2.2, so a
    /// `pip install` could not reach it.
    pub exchange_cutoff: Option<(f64, f64)>,
    /// Whether to use DIIS acceleration. `None` keeps the default (on). Rust-CLI-only through
    /// 0.2.2, and the first thing to turn off when an SCF oscillates.
    pub use_diis: Option<bool>,
    /// A uniform external electric field `[x, y, z]` in **volts/Ångström**, in MOPAC's `FIELD=`
    /// convention and sign. See `docs/theory.md`, convention C-1.
    pub field: Option<Vec<f64>>,
    /// `"coordinates"`, `"com"`/`"centre-of-mass"`, or `"charge"`/`"centre-of-charge"`. Only
    /// affects a charged molecule's reported dipole.
    pub dipole_origin: Option<String>,
}

fn build_field(args: &PeriodicArgs) -> PyResult<Option<crate::field::ExternalField>> {
    let Some(components) = &args.field else {
        return Ok(None);
    };
    if components.len() != 3 {
        return Err(PyValueError::new_err(
            "field must be three components [x, y, z] in volts/Angstrom",
        ));
    }
    let field = crate::field::ExternalField::new(components[0], components[1], components[2]);
    if !field.is_finite() {
        return Err(PyValueError::new_err("field components must be finite"));
    }
    Ok(Some(field))
}

fn build_dipole_origin(args: &PeriodicArgs) -> PyResult<crate::dipole::DipoleOrigin> {
    use crate::dipole::DipoleOrigin;
    let Some(name) = &args.dipole_origin else {
        return Ok(DipoleOrigin::default());
    };
    match name.trim().to_ascii_lowercase().as_str() {
        "coordinates" | "origin" => Ok(DipoleOrigin::Coordinates),
        "com" | "centre-of-mass" | "center-of-mass" | "mass" => Ok(DipoleOrigin::CentreOfMass),
        "charge" | "centre-of-charge" | "center-of-charge" => Ok(DipoleOrigin::CentreOfCharge),
        other => Err(PyValueError::new_err(format!(
            "unknown dipole_origin {other:?}; expected one of \
             \"coordinates\", \"com\", \"charge\""
        ))),
    }
}

/// How `pbc=` reorders the lattice vectors.
///
/// `pbc=(True, False, True)` — an ASE slab built along *y* — used to be refused on every surface
/// with "the periodic directions must be the leading ones". It is now reordered instead, and this
/// is the reordering. Anything the caller indexes by lattice vector goes through the same one.
fn axis_rotation(args: &PeriodicArgs) -> PyResult<crate::cell::AxisRotation> {
    let Some(flags) = &args.pbc else {
        return Ok(crate::cell::AxisRotation::IDENTITY);
    };
    if flags.len() != 3 {
        return Err(PyValueError::new_err("pbc must have three entries"));
    }
    Ok(crate::cell::AxisRotation::for_flags([
        flags[0], flags[1], flags[2],
    ]))
}

fn build_cell(args: &PeriodicArgs) -> PyResult<Option<crate::cell::Cell>> {
    let Some(rows) = &args.cell else {
        if args.kpoints.is_some() {
            return Err(PyValueError::new_err(
                "kpoints was given without a cell; k-point sampling needs a periodic lattice",
            ));
        }
        return Ok(None);
    };
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(
            "cell must be a 3x3 array of lattice vectors in Angstrom (one row per vector)",
        ));
    }
    let pbc = match &args.pbc {
        None => [true; 3],
        Some(flags) => {
            if flags.len() != 3 {
                return Err(PyValueError::new_err("pbc must have three entries"));
            }
            [flags[0], flags[1], flags[2]]
        }
    };
    let as_rows = [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ];
    Ok(crate::cell::Cell::from_angstrom_rows_pbc(&as_rows, pbc)
        .map_err(to_py_err)?
        .0)
}

fn build_pbc_options(args: &PeriodicArgs) -> PyResult<crate::pbc::PbcOptions> {
    let mut o = crate::pbc::PbcOptions::default();
    if let Some(mode) = &args.pbc_mode {
        o.mode = crate::pbc::PbcMode::from_str(mode).map_err(to_py_err)?;
    }
    if let Some(n) = &args.kpoints {
        if n.len() != 3 {
            return Err(PyValueError::new_err(
                "kpoints must be three Monkhorst-Pack divisions, e.g. (4, 4, 4)",
            ));
        }
        let shift = match &args.kpoint_shift {
            None => [0.0; 3],
            Some(s) if s.len() == 3 => [s[0], s[1], s[2]],
            Some(_) => {
                return Err(PyValueError::new_err(
                    "kpoint_shift must have three entries",
                ))
            }
        };
        // `kpoints` and `kpoint_shift` are per-lattice-vector, given in the caller's axis order.
        // When `pbc=` reorders the lattice vectors they have to move with them, or a slab along
        // *y* would be sampled densely along the direction it does not repeat in.
        let rotation = axis_rotation(args)?;
        o.kmesh = crate::pbc::KMesh::MonkhorstPack {
            n: rotation.apply([n[0], n[1], n[2]]),
            shift: rotation.apply(shift),
            gamma_centred: true,
        };
    }
    if let Some((kind, width, order)) = &args.smearing {
        o.smearing = match kind.to_ascii_lowercase().as_str() {
            "none" => crate::pbc::Smearing::None,
            "fermi" | "fermi-dirac" | "fd" => crate::pbc::Smearing::FermiDirac { width_ev: *width },
            "gauss" | "gaussian" => crate::pbc::Smearing::Gaussian { width_ev: *width },
            "mp" | "methfessel-paxton" => crate::pbc::Smearing::MethfesselPaxton {
                width_ev: *width,
                order: *order,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown smearing `{other}` (expected none, fermi, gauss, or mp)"
                )))
            }
        };
    }
    o.validate().map_err(to_py_err)?;
    Ok(o)
}

fn molecule(
    numbers: &[u8],
    positions: &[Vec<f64>],
    charge: f64,
    multiplicity: usize,
    periodic: &PeriodicArgs,
) -> PyResult<Molecule> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err(
            "numbers and positions length mismatch",
        ));
    }
    let mut atoms = Vec::with_capacity(numbers.len());
    for (&z, position) in numbers.iter().zip(positions) {
        if position.len() != 3 {
            return Err(PyValueError::new_err(
                "each position must have three components",
            ));
        }
        atoms.push(Atom {
            z,
            position: crate::math::Vec3::new(position[0], position[1], position[2])
                * ANGSTROM_TO_BOHR,
        });
    }
    Ok(Molecule::new(atoms)
        .with_charge(charge)
        .with_multiplicity(multiplicity)
        .with_optional_cell(build_cell(periodic)?))
}

fn model(
    method: &str,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    periodic: &PeriodicArgs,
) -> PyResult<(Pm7Parameters, Pm7Options)> {
    let method = method.parse::<Pm7Method>().map_err(to_py_err)?;
    let reference = ScfReference::from_str(reference).map_err(to_py_err)?;
    let parameters = Pm7Parameters::method(method).map_err(to_py_err)?;
    let pbc = if periodic.cell.is_some() {
        Some(build_pbc_options(periodic)?)
    } else {
        None
    };
    let mut options = Pm7Options {
        method,
        charge,
        multiplicity,
        reference,
        pbc,
        field: build_field(periodic)?,
        dipole_origin: build_dipole_origin(periodic)?,
        ..Pm7Options::default()
    };
    // MD and geometry optimization need these: a trajectory has to converge at *every* step, and
    // the tolerance a single point wants is often tighter than a force evaluation needs.
    if let Some(tol) = periodic.scf_tolerance {
        if !tol.is_finite() || tol <= 0.0 {
            return Err(PyValueError::new_err(
                "scf_tolerance must be finite and positive",
            ));
        }
        options.p_tol = tol;
        options.e_tol = tol * 1.0e-2;
    }
    if let Some(n) = periodic.max_scf {
        if n == 0 {
            return Err(PyValueError::new_err("max_scf must be at least 1"));
        }
        options.max_scf = n;
    }
    if let Some(n) = periodic.cphf_max_iterations {
        if n == 0 {
            return Err(PyValueError::new_err(
                "cphf_max_iterations must be at least 1: the orbital response is solved \
                 iteratively, and a budget of zero asks for no iterations at all rather than for \
                 a cheap answer",
            ));
        }
        options.cphf_max_iterations = n;
    }
    if let Some((inner, outer)) = periodic.exchange_cutoff {
        if !inner.is_finite() || !outer.is_finite() || inner < 0.0 || outer <= inner {
            return Err(PyValueError::new_err(
                "exchange_cutoff must be (inner, outer) in Bohr with 0 <= inner < outer",
            ));
        }
        options.exchange_cutoff = Some((inner, outer));
    }
    if let Some(on) = periodic.use_diis {
        options.use_diis = on;
    }
    if let Some(name) = &periodic.stability {
        options.stability = match name.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => crate::stability::ScfStability::Off,
            "check" => crate::stability::ScfStability::Check,
            "follow" => crate::stability::ScfStability::Follow,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown stability {other:?}; expected \"off\", \"check\" or \"follow\""
                )))
            }
        };
    }
    Ok((parameters, options))
}

/// Is the converged SCF solution a **minimum**, or only a stationary point?
///
/// Runs the SCF and then measures the curvature of the energy with respect to orbital rotations,
/// returning `lowest_ev` for the singlet (spin-preserving) channel and `lowest_triplet_ev` for the
/// spin-breaking one — negative means a saddle. `unstable` is true when either is.
///
/// The two channels ask different questions and a closed shell can pass the first and fail the
/// second: stretched H2 is a minimum among closed-shell solutions and 121 kcal/mol above the right
/// answer, which only the triplet channel sees. `None` for `lowest_triplet_ev` means the solution
/// was already unrestricted, where there is no further spin symmetry left to break.
///
/// It measures **the solution the options it was given produce**, which is the plain SCF solution
/// by default. Passing `stability="follow"` here is not an error and not a no-op: the SCF escapes
/// first and what comes back describes the *escaped* solution, so a followed run correctly reports
/// `unstable: false`. To act on an instability, that argument belongs on the entry point whose
/// answer you actually want — `single_point`, `optimize`, and the rest all take it.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
#[allow(clippy::too_many_arguments)]
fn scf_stability(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let result = run_pm7(&molecule, &parameters, &options).map_err(to_py_err)?;
    let found =
        crate::stability::check(&molecule, &parameters, &options, &result).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("energy_ev", result.total_ev)?;
    dict.set_item("heat_of_formation_kcal", result.heat_of_formation_kcal)?;
    dict.set_item("unrestricted", result.unrestricted)?;
    dict.set_item("spin_squared", result.spin_squared())?;
    match found {
        Some(s) => {
            dict.set_item("analysed", true)?;
            dict.set_item("lowest_ev", s.lowest_ev)?;
            dict.set_item("lowest_triplet_ev", s.lowest_triplet_ev)?;
            dict.set_item("unstable", s.unstable)?;
        }
        None => {
            // A periodic cell, an unconverged solve, or nothing to rotate into. Reported as a
            // flag rather than raised: asking is always allowed, and "not applicable" is an answer.
            dict.set_item("analysed", false)?;
            dict.set_item("lowest_ev", py.None())?;
            dict.set_item("lowest_triplet_ev", py.None())?;
            dict.set_item("unstable", py.None())?;
        }
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let result = run_pm7(&molecule, &parameters, &options).map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("method", method)?;
    // Echo the accepted inputs so callers can confirm they were received.
    dict.set_item("charge", charge)?;
    dict.set_item("multiplicity", multiplicity)?;
    dict.set_item("reference", options.reference.to_string())?;
    dict.set_item("unrestricted", result.unrestricted)?;
    // MOPAC's `(S**2)`, and `None` for a restricted run rather than the exact value: a caller
    // testing `if result["spin_squared"]` should be asking about *contamination*, which a
    // restricted solution cannot have, and filling in `S(S+1)` there would make the key look
    // meaningful where it is a tautology.
    dict.set_item("spin_squared", result.spin_squared())?;
    dict.set_item("energy_hartree", result.total_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", result.total_ev)?;
    // The Mermin *electronic* free energy E - TS, equal to energy_ev without smearing. Not a
    // thermochemical Gibbs energy: no vibrational partition function enters it. See
    // Pm7Result::free_energy_ev.
    dict.set_item("free_energy_ev", result.free_energy_ev())?;
    dict.set_item("heat_of_formation_kcal", result.heat_of_formation_kcal)?;
    dict.set_item("electronic_ev", result.electronic_ev)?;
    dict.set_item("core_ev", result.core_ev)?;
    dict.set_item("charges", result.charges.clone())?;
    dict.set_item(
        "dipole_debye",
        [
            result.dipole_debye.x,
            result.dipole_debye.y,
            result.dipole_debye.z,
        ],
    )?;
    dict.set_item(
        "dipole_point_charge_debye",
        vec3(result.dipole.point_charge),
    )?;
    dict.set_item("dipole_sp_hybrid_debye", vec3(result.dipole.sp_hybrid))?;
    dict.set_item("dipole_pd_hybrid_debye", vec3(result.dipole.pd_hybrid))?;
    dict.set_item("dipole_origin_bohr", vec3(result.dipole.origin))?;
    dict.set_item("homo_ev", result.homo_ev)?;
    dict.set_item("lumo_ev", result.lumo_ev)?;
    dict.set_item("homo_ev_beta", result.homo_ev_beta)?;
    dict.set_item("lumo_ev_beta", result.lumo_ev_beta)?;
    dict.set_item("gap_ev", result.gap_ev())?;
    dict.set_item("orbital_source", result.orbital_source.as_str())?;
    dict.set_item("mo_energies_ev", result.mo_energies.clone())?;
    dict.set_item("n_occ", result.n_occ)?;
    dict.set_item("field_ev", result.field_ev)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    add_periodic_keys(&dict, &molecule, &parameters, &options, &result)?;
    Ok(dict.into())
}

/// A `Vec3` as a plain three-element list, the shape every vector key in this module uses.
fn vec3(v: crate::math::Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Flatten a `Matrix` into a list of rows.
/// A `3x3` tensor as nested lists.
fn mat3_rows(m: &crate::math::Mat3) -> Vec<Vec<f64>> {
    (0..3)
        .map(|i| (0..3).map(|j| m.get(i, j)).collect())
        .collect()
}

/// A complex matrix as `{"real": [[..]], "imag": [[..]]}`-shaped pair of nested lists.
///
/// Two real matrices rather than a list of 2-tuples: the consumer is numpy, which builds a complex
/// array from a real and an imaginary part in one step and would otherwise have to unpack `n²`
/// tuples.
fn complex_rows(m: &crate::cmatrix::CMatrix) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let real = (0..m.n)
        .map(|i| (0..m.n).map(|j| m.get(i, j).0).collect())
        .collect();
    let imag = (0..m.n)
        .map(|i| (0..m.n).map(|j| m.get(i, j).1).collect())
        .collect();
    (real, imag)
}

/// Is this q the zone centre, where the non-analytic term belongs?
fn is_zone_centre(q: &[f64]) -> bool {
    q.iter().all(|c| c.abs() <= 1.0e-12)
}

/// The LO–TO material constants for a cell, or an error explaining why there are none.
///
/// Both phonon routes need the same thing — Born charges and `ε^∞` from a field response — and
/// neither could reach it from Python before v0.2.2. `DfptResult::frequencies_cm_lo_to` and
/// `ForceConstants::frequencies_cm_lo_to` existed, were documented in `docs/properties.md` as the
/// way to use the splitting, and had **no callers anywhere in the repository**: the bindings
/// handed back the raw `D^NA` matrix and left the caller to add it to the force constants,
/// mass-weight and re-diagonalize by hand.
fn non_analytic_term(
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &crate::scf::Pm7Options,
    dfpt_options: &crate::dfpt::DfptOptions,
) -> PyResult<crate::dfpt::NonAnalytic> {
    let field = crate::dfpt::born_and_dielectric(molecule, parameters, options, dfpt_options)
        .map_err(to_py_err)?;
    field.non_analytic().map_err(to_py_err)
}

/// A Cartesian LO–TO direction from the Python argument, validated.
fn lo_to_direction_of(direction: &[f64]) -> PyResult<[f64; 3]> {
    if direction.len() != 3 {
        return Err(PyValueError::new_err(
            "lo_to_direction needs three Cartesian components",
        ));
    }
    Ok([direction[0], direction[1], direction[2]])
}

/// Perturbation-solver settings, with the crate's defaults for anything not given.
fn dfpt_settings(
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
    mixing: Option<f64>,
) -> crate::dfpt::DfptOptions {
    let mut out = crate::dfpt::DfptOptions::default();
    if let Some(t) = tolerance {
        out.tolerance = t;
    }
    if let Some(n) = max_iterations {
        out.max_iterations = n;
    }
    if let Some(m) = mixing {
        out.mixing = m;
    }
    out
}

fn rows(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Per-AO labels, so a coefficient matrix is interpretable without rebuilding the basis.
fn ao_labels(
    molecule: &Molecule,
    basis: &crate::basis::Basis,
) -> (Vec<String>, Vec<usize>, Vec<String>) {
    const SHELL: [&str; 9] = ["s", "px", "py", "pz", "dx2-y2", "dxz", "dz2", "dyz", "dxy"];
    let mut labels = Vec::with_capacity(basis.nao);
    let mut atom_index = Vec::with_capacity(basis.nao);
    let mut shells = Vec::with_capacity(basis.nao);
    for ao in &basis.aos {
        let symbol = crate::system::z_to_symbol(ao.z).unwrap_or("X");
        let shell = SHELL[ao.orb as usize % 9];
        labels.push(format!("{} {} {}", ao.atom + 1, symbol, shell));
        atom_index.push(ao.atom);
        shells.push(shell.to_string());
    }
    let _ = molecule;
    (labels, atom_index, shells)
}

/// Add the keys that only exist for a periodic system. A molecule gets none of them, so a
/// caller can test `"stress_ev_per_angstrom3" in result` to know which kind of system it ran.
fn add_periodic_keys(
    dict: &Bound<'_, PyDict>,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
    result: &crate::scf::Pm7Result,
) -> PyResult<()> {
    let Some(cell) = molecule.cell else {
        return Ok(());
    };
    dict.set_item("periodicity", cell.dim())?;
    dict.set_item("cell_angstrom", cell.angstrom_rows())?;
    dict.set_item("n_kpoints", result.n_kpoints)?;
    dict.set_item("fermi_ev", result.fermi_ev)?;
    dict.set_item("entropy_ev", result.entropy_ev)?;
    dict.set_item("ewald_ev", result.ewald_ev)?;
    dict.set_item("background_ev", result.background_ev)?;
    dict.set_item("makov_payne_ev", result.makov_payne_ev)?;
    if let Some(v) = cell.volume() {
        dict.set_item(
            "volume_angstrom3",
            v * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM,
        )?;
    }
    if let Some(bands) = &result.band_energies {
        dict.set_item("band_energies_ev", bands.clone())?;
    }
    let stress =
        crate::stress::analytic_stress(molecule, parameters, options, result).map_err(to_py_err)?;
    // eV/Bohr^dim → eV/Å^dim. The cell measure is a length, an area, or a volume depending on
    // the periodicity, so the conversion power follows the dimension.
    let scale = ANGSTROM_TO_BOHR.powi(cell.dim() as i32);
    let voigt: Vec<f64> = stress.voigt().iter().map(|v| v * scale).collect();
    let full: Vec<Vec<f64>> = (0..3)
        .map(|i| (0..3).map(|j| stress.stress.get(i, j) * scale).collect())
        .collect();
    dict.set_item("stress_voigt", voigt)?;
    dict.set_item("stress", full)?;
    if let Some(p) = stress.pressure_gpa(molecule) {
        dict.set_item("pressure_gpa", p)?;
    }
    Ok(())
}

// Shared driver for gradient/forces: the `sign` is +1 for the energy gradient
// (dE/dx) and −1 for the force (−dE/dx). Returns (au [Hartree/Bohr], eV/Å, scf hof).
fn gradient_vectors(
    numbers: &[u8],
    positions: &[Vec<f64>],
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    periodic: &PeriodicArgs,
    sign: f64,
) -> PyResult<(f64, Vec<[f64; 3]>, Vec<[f64; 3]>, f64, f64)> {
    let molecule = molecule(numbers, positions, charge, multiplicity, periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, periodic)?;
    let result = closed_form_gradient(&molecule, &parameters, &options).map_err(to_py_err)?;
    let au: Vec<[f64; 3]> = result
        .gradient
        .iter()
        .map(|v| {
            [
                sign * v.x * EV_TO_HARTREE,
                sign * v.y * EV_TO_HARTREE,
                sign * v.z * EV_TO_HARTREE,
            ]
        })
        .collect();
    let ev_ang: Vec<[f64; 3]> = result
        .gradient
        .iter()
        .map(|v| {
            [
                sign * v.x * ANGSTROM_TO_BOHR,
                sign * v.y * ANGSTROM_TO_BOHR,
                sign * v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    Ok((
        result.energy_ev,
        au,
        ev_ang,
        result.scf.heat_of_formation_kcal,
        result.scf.free_energy_ev(),
    ))
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let (energy_ev, au, ev_ang, hof, free_energy_ev) = gradient_vectors(
        &numbers,
        &positions,
        charge,
        multiplicity,
        method,
        reference,
        &periodic,
        1.0,
    )?;
    let dict = PyDict::new(py);
    dict.set_item("energy_hartree", energy_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", energy_ev)?;
    dict.set_item("free_energy_ev", free_energy_ev)?;
    dict.set_item("heat_of_formation_kcal", hof)?;
    dict.set_item("gradient_hartree_per_bohr", au)?;
    dict.set_item("gradient_ev_per_angstrom", ev_ang)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
fn forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    // Force = −∂E/∂x.
    let (energy_ev, au, ev_ang, hof, free_energy_ev) = gradient_vectors(
        &numbers,
        &positions,
        charge,
        multiplicity,
        method,
        reference,
        &periodic,
        -1.0,
    )?;
    let dict = PyDict::new(py);
    dict.set_item("energy_hartree", energy_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", energy_ev)?;
    dict.set_item("free_energy_ev", free_energy_ev)?;
    dict.set_item("heat_of_formation_kcal", hof)?;
    dict.set_item("forces_hartree_per_bohr", au)?;
    dict.set_item("forces_ev_per_angstrom", ev_ang)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None, relax_cell=false, gtol=None, stress_tol=None, max_iter=None, stability_every=None))]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    relax_cell: bool,
    gtol: Option<f64>,
    stress_tol: Option<f64>,
    max_iter: Option<usize>,
    stability_every: Option<usize>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    // None of `OptOptions` was reachable from Python through 0.2.2 -- `gtol`, `max_iter` and the
    // rest were `OptOptions::default()` at both call sites, so a run that needed tighter
    // convergence had no way to ask.
    let defaults = OptOptions::default();
    let opt = OptOptions {
        relax_cell,
        gtol: gtol.unwrap_or(defaults.gtol),
        stress_tol: stress_tol.unwrap_or(defaults.stress_tol),
        max_iter: max_iter.unwrap_or(defaults.max_iter),
        // A geometry step changes the orbitals, so a solution that was a minimum at the start can
        // stop being one on the way -- and the optimizer would then converge on a saddle of the
        // wrong surface without saying so. `0` (the default) never checks; `n` checks every `n`th
        // step and switches the run to `Follow` for good once an instability is found.
        stability_every: stability_every.unwrap_or(defaults.stability_every),
        ..defaults
    };
    let result = optimize_geometry(&molecule, &parameters, &options, &opt).map_err(to_py_err)?;
    let coordinates: Vec<[f64; 3]> = result
        .molecule
        .atoms
        .iter()
        .map(|atom| {
            let position = atom.position * BOHR_TO_ANGSTROM;
            [position.x, position.y, position.z]
        })
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("positions_angstrom", coordinates)?;
    dict.set_item("energy_hartree", result.energy_ev * EV_TO_HARTREE)?;
    dict.set_item("heat_of_formation_kcal", result.heat_of_formation_kcal)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("iterations", result.iterations)?;
    // Populated all along and dropped here, so a caller could not see the path the optimizer took
    // -- not even to plot the energy against the step, which is the first thing anyone does with a
    // stalled optimization.
    let trajectory = PyList::empty(py);
    for step in &result.trajectory {
        let entry = PyDict::new(py);
        entry.set_item("energy_ev", step.energy_ev)?;
        entry.set_item("heat_of_formation_kcal", step.heat_of_formation_kcal)?;
        entry.set_item("max_gradient_ev_per_bohr", step.max_gradient)?;
        entry.set_item("max_stress", step.max_stress)?;
        entry.set_item(
            "positions_angstrom",
            step.positions
                .iter()
                .map(|p| {
                    vec![
                        p.x * BOHR_TO_ANGSTROM,
                        p.y * BOHR_TO_ANGSTROM,
                        p.z * BOHR_TO_ANGSTROM,
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        if let Some(cell) = step.cell {
            entry.set_item("cell_angstrom", cell.angstrom_rows())?;
        }
        trajectory.append(entry)?;
    }
    dict.set_item("trajectory", trajectory)?;
    // Without the cell a periodic optimization cannot be read back: the coordinates alone do not
    // say what they are periodic in. Reported in the caller's axis order, so a `pbc=` that
    // reordered the lattice vectors does not leak into the answer.
    if let Some(cell) = result.molecule.cell {
        // `completed_vectors` fills the open directions with unit normals, which is what extended
        // XYZ wants: a full 3x3 `Lattice` plus a `pbc` that says which rows are real.
        let full = cell.completed_vectors().map(|v| {
            [
                v.x * BOHR_TO_ANGSTROM,
                v.y * BOHR_TO_ANGSTROM,
                v.z * BOHR_TO_ANGSTROM,
            ]
        });
        let rotation = axis_rotation(&periodic)?;
        dict.set_item("cell_angstrom", rotation.undo(full).map(|r| r.to_vec()))?;
        dict.set_item("pbc", rotation.undo(cell.periodicity().flags()).to_vec())?;
        dict.set_item("periodicity", cell.dim())?;
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None, projection=None))]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    projection: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let result = crate::hessian::vibrational_analysis_projected(
        &molecule,
        &parameters,
        &options,
        1.0e-3,
        parse_projection(&projection)?,
    )
    .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("frequencies_cm", result.frequencies_cm)?;
    dict.set_item("eigenvalues", result.eigenvalues)?;
    Ok(dict.into())
}

/// `projection=` on the vibrational entry points: what to remove before diagonalizing.
///
/// `None` is [`Projection::Rigid`], which is what a harmonic spectrum means. `"none"` returns the
/// raw `3N` set — the opt-out, for looking at what the projector took out rather than for
/// spectroscopy.
fn parse_projection(name: &Option<String>) -> PyResult<crate::projection::Projection> {
    match name {
        None => Ok(crate::projection::Projection::default()),
        Some(text) => crate::projection::Projection::parse(text).map_err(to_py_err),
    }
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    // Analytic Cartesian Hessian in eV/Bohr² (3N × 3N).
    let h = crate::hessian::analytic_hessian(&molecule, &parameters, &options, 1.0e-3)
        .map_err(to_py_err)?;
    let n = h.rows;
    let mut au = Vec::with_capacity(n); // Hartree/Bohr²
    let mut ev_ang = Vec::with_capacity(n); // eV/Å²
    for i in 0..n {
        let mut row_au = Vec::with_capacity(n);
        let mut row_ev = Vec::with_capacity(n);
        for j in 0..n {
            let v = h[(i, j)];
            row_au.push(v * EV_TO_HARTREE);
            row_ev.push(v * EV_PER_BOHR2_TO_EV_PER_ANGSTROM2);
        }
        au.push(row_au);
        ev_ang.push(row_ev);
    }
    let dict = PyDict::new(py);
    dict.set_item("hessian_hartree_per_bohr2", au)?;
    dict.set_item("hessian_ev_per_angstrom2", ev_ang)?;
    Ok(dict.into())
}

/// Energy, forces, and the analytic stress tensor of a periodic system in one call.
///
/// ASE asks for energy, forces, and stress together on every step of a variable-cell relaxation
/// or an NPT run, and they share one SCF — so computing them separately would triple the cost of
/// exactly the workflow that needs them most.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
#[allow(clippy::too_many_arguments)]
fn stress(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "stress is only defined for a periodic system; pass a `cell`",
        ));
    }
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let result = closed_form_gradient(&molecule, &parameters, &options).map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("energy_ev", result.energy_ev)?;
    dict.set_item("energy_hartree", result.energy_ev * EV_TO_HARTREE)?;
    dict.set_item("free_energy_ev", result.scf.free_energy_ev())?;
    dict.set_item("heat_of_formation_kcal", result.scf.heat_of_formation_kcal)?;
    let forces_ev_ang: Vec<[f64; 3]> = result
        .gradient
        .iter()
        .map(|v| {
            [
                -v.x * ANGSTROM_TO_BOHR,
                -v.y * ANGSTROM_TO_BOHR,
                -v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    dict.set_item("forces_ev_per_angstrom", forces_ev_ang)?;
    add_periodic_keys(&dict, &molecule, &parameters, &options, &result.scf)?;
    Ok(dict.into())
}

/// Phonon dispersion of a periodic system.
///
/// Force constants come from the analytic Hessian of the `supercell` repeat, which resolves
/// `Φ(0A, TB)` for every translation inside it — so the frequencies are exact at every `q`
/// commensurate with that supercell and Fourier-interpolated between them. `supercell=(1,1,1)`
/// gives the zone centre and nothing else.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, qpoints, lo_to_direction=None, charge=0.0, multiplicity=1, method="pm7", reference="auto", supercell=None, pbc=None, acoustic_sum_rule=true, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None))]
#[allow(clippy::too_many_arguments)]
fn phonons(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    qpoints: Vec<Vec<f64>>,
    lo_to_direction: Option<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    supercell: Option<Vec<usize>>,
    pbc: Option<Vec<bool>>,
    acoustic_sum_rule: bool,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
) -> PyResult<PyObject> {
    // `smearing` and `pbc_mode` reach the SCF the force constants differentiate; they used to be
    // dropped by a `..Default::default()` and are threaded through now.
    //
    // A **k mesh** is refused instead. `force_constants` builds the `supercell` repeat and solves
    // it at its Γ point, because Γ of an `n1 x n2 x n3` supercell *is* the `n1 x n2 x n3` mesh of
    // the cell — the identity `tests/pbc_equivalence.rs` pins. So `kpoints` cannot change this
    // calculation, and accepting it would be the same silent no-op as dropping it: the caller
    // would get an answer, believe the mesh had applied, and never learn otherwise. The sampling
    // knob here is `supercell`.
    if kpoints.is_some() || kpoint_shift.is_some() {
        return Err(PyValueError::new_err(
            "phonons take their Brillouin-zone sampling from `supercell`, not `kpoints`: the \
             force constants come from the supercell's Gamma point, which is exactly the \
             n1 x n2 x n3 mesh of the cell. Pass supercell=(n1, n2, n3) instead, or use `dfpt`, \
             which solves the response at each q directly and does take a k mesh.",
        ));
    }
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
        ..PeriodicArgs::default()
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "phonons need a periodic system; pass a non-degenerate `cell`",
        ));
    }
    // Per-lattice-vector, so it follows any reordering `pbc=` applied — as do the fractional `q`
    // components below.
    let rotation = axis_rotation(&periodic)?;
    let repeat = rotation.apply(match &supercell {
        None => [1, 1, 1],
        Some(n) if n.len() == 3 => [n[0], n[1], n[2]],
        Some(_) => {
            return Err(PyValueError::new_err(
                "supercell must have three entries, e.g. (2, 2, 2)",
            ))
        }
    });
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let mut fc = crate::hessian_pbc::force_constants(&molecule, &parameters, &options, repeat)
        .map_err(to_py_err)?;
    if acoustic_sum_rule {
        fc.enforce_acoustic_sum_rule();
    }
    let mut frequencies = Vec::with_capacity(qpoints.len());
    let mut eigenvectors = Vec::with_capacity(qpoints.len());
    let mut cartesian = Vec::with_capacity(qpoints.len());
    for q in &qpoints {
        if q.len() != 3 {
            return Err(PyValueError::new_err(
                "each q point needs three fractional components",
            ));
        }
        let modes = fc
            .modes(rotation.apply([q[0], q[1], q[2]]))
            .map_err(to_py_err)?;
        frequencies.push(modes.frequencies_cm);
        eigenvectors.push(complex_rows(&modes.eigenvectors));
        cartesian.push(complex_rows(&modes.cartesian_modes));
    }

    // The LO-TO split, when a direction is given. It is opt-in and the direction is required,
    // because the q -> 0 limit of the macroscopic field is direction dependent -- there is no
    // such thing as "the" correction at the zone centre, so a silently chosen direction would be
    // a wrong answer rather than a default. Away from the zone centre the term does not belong at
    // all (the macroscopic field is already inside Phi(q) there), so those entries are None
    // rather than a number the caller might read as a correction.
    let lo_to = match &lo_to_direction {
        None => None,
        Some(direction) => {
            let q_hat = lo_to_direction_of(direction)?;
            let na = non_analytic_term(
                &molecule,
                &parameters,
                &options,
                &crate::dfpt::DfptOptions::default(),
            )?;
            let mut split: Vec<Option<Vec<f64>>> = Vec::with_capacity(qpoints.len());
            for q in &qpoints {
                split.push(if is_zone_centre(q) {
                    Some(fc.frequencies_cm_lo_to(&na, q_hat).map_err(to_py_err)?)
                } else {
                    None
                });
            }
            Some(split)
        }
    };

    let dict = PyDict::new(py);
    dict.set_item("qpoints", qpoints)?;
    dict.set_item("frequencies_cm", frequencies)?;
    // The polarization vectors, one entry per q, each a `(real, imag)` pair of `3N x 3N` row
    // lists with one mode per **column** — the same complex convention `dfpt` uses for its force
    // constants, and the same column-per-mode layout the molecular `vibrations` uses.
    //
    // Through 0.2.3 this route returned frequencies and nothing else, though the eigenvectors were
    // computed and discarded on every call. That made a whole class of question unanswerable from
    // Python: which atoms a soft branch moves, how to displace a structure along a mode, whether
    // two branches at the same frequency are the degenerate pair the space group requires.
    dict.set_item("modes", eigenvectors)?;
    dict.set_item("cartesian_modes", cartesian)?;
    dict.set_item("frequencies_cm_lo_to", lo_to)?;
    dict.set_item(
        "lo_to_direction",
        lo_to_direction.clone().map(|d| vec![d[0], d[1], d[2]]),
    )?;
    dict.set_item("acoustic_residual_ev_per_bohr2", fc.acoustic_residual())?;
    // Reported back in the caller's axis order: they asked for `supercell=(2, 1, 2)` and should
    // read `[2, 1, 2]`, not the internally reordered triple.
    dict.set_item(
        "translations",
        fc.translations
            .iter()
            .map(|t| rotation.undo(*t).to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("supercell", rotation.undo(repeat).to_vec())?;
    Ok(dict.into())
}

/// Band energies along a path in the Brillouin zone.
///
/// This is **not** an SCF along the path. A band path runs down high-symmetry lines, which is the
/// wrong set of points to build a density from — it would weight those lines as though they were
/// the whole zone. The density and the Fermi level come from the `kpoints` mesh; the path only
/// asks the converged Hamiltonian what its eigenvalues are elsewhere.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, kpath, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
#[allow(clippy::too_many_arguments)]
fn band_structure(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    kpath: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "a band structure needs a periodic system; pass a non-degenerate `cell`",
        ));
    }
    let mut path = Vec::with_capacity(kpath.len());
    for k in &kpath {
        if k.len() != 3 {
            return Err(PyValueError::new_err(
                "each k point needs three fractional components",
            ));
        }
        path.push([k[0], k[1], k[2]]);
    }
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let bands = crate::scf_pbc::band_structure(&molecule, &parameters, &options, &path)
        .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("kpath", kpath)?;
    dict.set_item("energies_ev", bands.energies)?;
    if let Some(beta) = bands.energies_beta {
        dict.set_item("energies_beta_ev", beta)?;
    }
    dict.set_item("fermi_ev", bands.fermi_ev)?;
    Ok(dict.into())
}

/// Divide-and-conquer single point: energy, forces, and stress for a system too large for the
/// cubic-scaling SCF.
///
/// `buffer` is the accuracy knob, in **Ångström**. Everything outside a subsystem's buffer reaches
/// it only as a monopole, so widening it walks the answer onto the exact SCF — at a cost that
/// grows with the buffer cubed. Below roughly 300 atoms the ordinary path is both exact and
/// faster; see `docs/divide_and_conquer.md` for the measured crossover and the accuracy table.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", buffer=8.0, core_size=12, cell=None, pbc=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
#[allow(clippy::too_many_arguments)]
fn divide_and_conquer(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    buffer: f64,
    core_size: usize,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
        ..PeriodicArgs::default()
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let dandc = crate::dandc::DandcOptions {
        buffer: buffer * ANGSTROM_TO_BOHR,
        core_size,
        max_scf: max_scf.unwrap_or(200),
        p_tol: scf_tolerance.unwrap_or(1.0e-6),
        ..crate::dandc::DandcOptions::default()
    };
    let result =
        crate::dandc::run_dandc(&molecule, &parameters, &options, &dandc).map_err(to_py_err)?;
    if !result.converged {
        return Err(PyValueError::new_err(format!(
            "the divide-and-conquer SCF did not converge after {} iterations (error={:.3e})",
            result.iterations, result.density_error
        )));
    }
    let derivatives = crate::dandc::dandc_derivatives(&molecule, &parameters, &options, &result)
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("energy_ev", derivatives.energy_ev)?;
    dict.set_item("energy_hartree", derivatives.energy_ev * EV_TO_HARTREE)?;
    let forces: Vec<[f64; 3]> = derivatives
        .gradient
        .iter()
        .map(|v| {
            [
                -v.x * ANGSTROM_TO_BOHR,
                -v.y * ANGSTROM_TO_BOHR,
                -v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    dict.set_item("forces_ev_per_angstrom", forces)?;
    if let Some(stress) = derivatives.stress {
        let per_a3 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
        let voigt = stress.to_voigt();
        dict.set_item(
            "stress_voigt",
            voigt.iter().map(|v| v * per_a3).collect::<Vec<_>>(),
        )?;
    }
    dict.set_item("fermi_ev", result.fermi_ev)?;
    dict.set_item("subsystems", result.subsystems)?;
    dict.set_item("largest_subsystem", result.largest_subsystem)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("unrestricted", result.unrestricted)?;
    dict.set_item("stored_density_elements", result.density.stored_elements())?;
    Ok(dict.into())
}

/// Orbital energies and coefficients, for both spin channels.
///
/// Kept out of `single_point` because the coefficient matrix is `nao x nao`: putting it on every
/// single point would make every step of an MD run marshal a matrix nobody asked for.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None))]
#[allow(clippy::too_many_arguments)]
fn orbitals(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let result = run_pm7(&molecule, &parameters, &options).map_err(to_py_err)?;
    let basis = crate::basis::Basis::build(&molecule, &parameters).map_err(to_py_err)?;
    let (labels, atom_index, shells) = ao_labels(&molecule, &basis);

    let dict = PyDict::new(py);
    dict.set_item("mo_energies_ev", result.mo_energies.clone())?;
    dict.set_item(
        "mo_energies_hartree",
        result
            .mo_energies
            .iter()
            .map(|e| e * EV_TO_HARTREE)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("mo_coefficients", rows(&result.mo_coeff))?;
    dict.set_item("occupations", result.occupations())?;
    dict.set_item("n_occ", result.n_occ)?;
    dict.set_item("homo_ev", result.homo_ev)?;
    dict.set_item("lumo_ev", result.lumo_ev)?;
    dict.set_item("gap_ev", result.gap_ev())?;
    dict.set_item("orbital_source", result.orbital_source.as_str())?;
    dict.set_item("ao_labels", labels)?;
    dict.set_item("ao_atom_index", atom_index)?;
    dict.set_item("ao_shell", shells)?;
    dict.set_item("unrestricted", result.unrestricted)?;
    dict.set_item("spin_squared", result.spin_squared())?;
    if let Some(eps) = &result.mo_energies_beta {
        dict.set_item("mo_energies_beta_ev", eps.clone())?;
        dict.set_item(
            "mo_energies_beta_hartree",
            eps.iter().map(|e| e * EV_TO_HARTREE).collect::<Vec<_>>(),
        )?;
    }
    if let Some(c) = &result.mo_coeff_beta {
        dict.set_item("mo_coefficients_beta", rows(c))?;
    }
    dict.set_item("occupations_beta", result.occupations_beta())?;
    dict.set_item("n_occ_beta", result.n_occ_beta)?;
    dict.set_item("homo_ev_beta", result.homo_ev_beta)?;
    dict.set_item("lumo_ev_beta", result.lumo_ev_beta)?;
    // `occupations` is an aufbau count over the Γ states of the converged Hamiltonian (see
    // `Pm7Result::occupations`), which is the filling only when a gap straddles the Fermi level.
    // This function accepts `smearing` and a k mesh, so it can be asked for a case where that is
    // not true, and until 0.2.3 it returned the count with nothing beside it to say so: every
    // other periodic surface set `fermi_ev` and this one did not. `entropy_ev` is the tell --
    // non-zero means some state is fractionally occupied and the count is a band count.
    dict.set_item("fermi_ev", result.fermi_ev)?;
    dict.set_item("entropy_ev", result.entropy_ev)?;
    dict.set_item("n_kpoints", result.n_kpoints)?;
    Ok(dict.into())
}

/// The dynamical matrix at one or more `q` points by density-functional perturbation theory.
///
/// The alternative to `phonons`, which builds a supercell. DFPT needs no supercell at all: it
/// solves the response directly at each `q`, so the cost is one linear-response solve per `q`
/// rather than one SCF over `n1*n2*n3` cells, and no commensurability condition applies — any `q`
/// is reachable, not just those a supercell happens to fold onto.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, qpoints, lo_to_direction=None, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field=None, dipole_origin=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None, long_range=None, keep_response=false))]
#[allow(clippy::too_many_arguments)]
fn dfpt(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    qpoints: Vec<Vec<f64>>,
    lo_to_direction: Option<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
    long_range: Option<String>,
    keep_response: bool,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "DFPT needs a periodic system; pass a non-degenerate `cell`",
        ));
    }
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let mut dfpt_options = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    dfpt_options.long_range = parse_long_range(long_range.as_deref())?;
    dfpt_options.keep_response = keep_response;

    // The LO-TO material constants, if the caller wants the split. Solved once for the whole q
    // list, not per q: Z* and eps^inf are properties of the cell.
    //
    // With dfpt the term belongs **only** at the zone centre. Away from it the macroscopic
    // field is already inside Phi(q) through the phased Ewald sum, so adding D^NA as well
    // would count it twice -- which is what DfptResult::force_constants_with_lo_to refuses, and
    // why the entries for a non-zero q are None rather than an unsplit copy.
    let na = match &lo_to_direction {
        None => None,
        Some(direction) => {
            let q_hat = lo_to_direction_of(direction)?;
            Some((
                non_analytic_term(&molecule, &parameters, &options, &dfpt_options)?,
                q_hat,
            ))
        }
    };

    let mut frequencies = Vec::with_capacity(qpoints.len());
    let mut eigenvectors = Vec::with_capacity(qpoints.len());
    let mut cartesian = Vec::with_capacity(qpoints.len());
    let mut lo_to: Vec<Option<Vec<f64>>> = Vec::with_capacity(qpoints.len());
    let mut force_constants = Vec::with_capacity(qpoints.len());
    let mut iterations = 0usize;
    let mut converged = true;
    let mut residual = 0.0_f64;
    let mut hermiticity = 0.0_f64;
    for q in &qpoints {
        if q.len() != 3 {
            return Err(PyValueError::new_err(
                "each q point needs three fractional components",
            ));
        }
        let result = crate::dfpt::dynamical_matrix_dfpt(
            &molecule,
            &parameters,
            &options,
            // Fractional, so it is indexed by lattice vector and follows any `pbc=` reordering.
            axis_rotation(&periodic)?.apply([q[0], q[1], q[2]]),
            &dfpt_options,
        )
        .map_err(to_py_err)?;
        let modes = result.modes().map_err(to_py_err)?;
        frequencies.push(modes.frequencies_cm);
        eigenvectors.push(complex_rows(&modes.eigenvectors));
        cartesian.push(complex_rows(&modes.cartesian_modes));
        lo_to.push(match &na {
            Some((term, q_hat)) if is_zone_centre(q) => Some(
                result
                    .frequencies_cm_lo_to(term, *q_hat)
                    .map_err(to_py_err)?,
            ),
            _ => None,
        });
        force_constants.push(complex_rows(&result.force_constants));
        iterations = iterations.max(result.iterations);
        converged &= result.converged;
        residual = residual.max(result.residual);
        hermiticity = hermiticity.max(result.hermiticity);
    }
    let dict = PyDict::new(py);
    dict.set_item("qpoints", qpoints)?;
    dict.set_item("frequencies_cm", frequencies)?;
    // Same shape and same convention as `phonons`, deliberately: the two routes compute the same
    // object by different means, and a caller comparing them should not also have to translate
    // between two layouts.
    dict.set_item("modes", eigenvectors)?;
    dict.set_item("cartesian_modes", cartesian)?;
    dict.set_item(
        "frequencies_cm_lo_to",
        lo_to_direction.as_ref().map(|_| lo_to),
    )?;
    dict.set_item(
        "lo_to_direction",
        lo_to_direction.map(|d| vec![d[0], d[1], d[2]]),
    )?;
    dict.set_item("force_constants_ev_per_bohr2", force_constants)?;
    dict.set_item("iterations", iterations)?;
    dict.set_item("converged", converged)?;
    dict.set_item("residual", residual)?;
    dict.set_item("hermiticity", hermiticity)?;
    Ok(dict.into())
}

/// The cell's electronic polarizability on its own, in every dimensionality.
///
/// `alpha_ab = d(mu_a)/d(f_b)`, the same tensor `born_charges` reports, without computing the
/// Born charges to get it. It exists separately because `alpha` is defined for a chain and a slab
/// where `eps_inf` is not — that conversion needs someone to say where the material stops, which
/// is what `dielectric_with_extent` is for.
///
/// In the MOPAC `FIELD=` sign convention (C-1), so it carries the opposite sign to the physical
/// polarizability.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None, long_range=None))]
#[allow(clippy::too_many_arguments)]
fn polarizability(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
    long_range: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let mut settings = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    settings.long_range = parse_long_range(long_range.as_deref())?;
    let out = crate::dfpt::polarizability(&molecule, &parameters, &options, &settings)
        .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("polarizability", mat3_rows(&out))?;
    Ok(dict.into())
}

/// How much the polarizability moves when the whole cell is translated.
///
/// The position operator a field perturbation is built on is not a well-defined periodic operator.
/// That the *response* is nevertheless well defined is an argument; this measures it. A value near
/// machine precision says the argument holds for this system, a large one says the number being
/// reported is a statement about where the origin was put.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, offset, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None))]
#[allow(clippy::too_many_arguments)]
fn dielectric_origin_sensitivity(
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    offset: Vec<f64>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
) -> PyResult<f64> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let settings = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    crate::dfpt::dielectric_origin_sensitivity(
        &molecule,
        &parameters,
        &options,
        &settings,
        vec3_arg(&offset, "offset")?,
    )
    .map_err(to_py_err)
}

/// `eps^0`, the static dielectric tensor: the clamped-ion `eps^inf` plus the ionic term.
///
/// `skipped_modes` counts modes left out of the `1/omega^2` sum for sitting below the soft-mode
/// floor. **Three is the expected count** — the acoustic branch. More than three means the
/// geometry is not a minimum, and the ionic term is then missing whatever those modes would have
/// contributed, which for a soft mode is most of it.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None))]
#[allow(clippy::too_many_arguments)]
fn static_dielectric(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let settings = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    let out = crate::dfpt::static_dielectric_tensor(&molecule, &parameters, &options, &settings)
        .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("dielectric", mat3_rows(&out.epsilon))?;
    dict.set_item("electronic", mat3_rows(&out.electronic))?;
    dict.set_item("ionic", mat3_rows(&out.ionic))?;
    dict.set_item("skipped_modes", out.skipped_modes)?;
    dict.set_item("softest_kept", out.softest_kept)?;
    // `soft_mode_floor` was here, reporting the constant a caller was expected to compare against.
    // There is no floor any more: the acoustic modes are identified by their overlap with the
    // uniform translations, so `skipped_modes` is three by construction and a geometry that is not
    // a minimum shows up in `soft_optical_modes` instead of being conflated with it.
    dict.set_item("soft_optical_modes", out.soft_optical_modes)?;
    Ok(dict.into())
}

/// Berry-phase electronic polarization, King-Smith and Vanderbilt.
///
/// Defined **modulo** `quantum`: a different branch of the logarithm assigns the electrons to a
/// different unit cell, which is an equally valid choice. Two polarizations should be compared by
/// reducing their difference onto the nearest branch, not by subtracting `polarization` directly —
/// a finite displacement commonly crosses a branch and the raw difference is then off by exactly
/// one quantum.
///
/// `strings` is the convergence parameter and the answer has to stop moving with it.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, strings=16, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None))]
#[allow(clippy::too_many_arguments)]
fn berry_polarization(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    strings: usize,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let out = crate::pbc::berry::berry_polarization(&molecule, &parameters, &options, strings)
        .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("electronic", vec3_list(out.electronic))?;
    dict.set_item("ionic", vec3_list(out.ionic))?;
    dict.set_item("polarization", vec3_list(out.total))?;
    dict.set_item("phase", out.phase.to_vec())?;
    dict.set_item(
        "quantum",
        out.quantum
            .iter()
            .map(|q| vec3_list(*q))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("string_length", out.string_length)?;
    Ok(dict.into())
}

/// A finite electric field applied **along** a periodic direction.
///
/// A field orthogonal to every lattice vector is an ordinary calculation and goes through the
/// `field=` keyword; along a periodic direction the `E.R` potential is unbounded and there is no
/// ground state to find. This minimizes the Nunes-Gonze electric enthalpy `E - Omega E.P` with `P`
/// the Berry-phase polarization, which is bounded.
///
/// `divisions` is the k mesh **and** the string length, so it is the convergence parameter for the
/// polarization as well as for the Brillouin-zone integral. An axis the field touches needs at
/// least three.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, divisions, field, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, field_tolerance=None, field_max_iterations=None, field_mixing=None))]
#[allow(clippy::too_many_arguments)]
fn finite_field(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    divisions: Vec<usize>,
    field: Vec<f64>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    field_tolerance: Option<f64>,
    field_max_iterations: Option<usize>,
    field_mixing: Option<f64>,
) -> PyResult<PyObject> {
    if divisions.len() != 3 {
        return Err(PyValueError::new_err("divisions must have three entries"));
    }
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints: None,
        kpoint_shift: None,
        smearing: None,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let mut ff = crate::pbc::finite_field::FiniteFieldOptions::default();
    if let Some(t) = field_tolerance {
        ff.tol = t;
    }
    if let Some(n) = field_max_iterations {
        ff.max_iterations = n;
    }
    if let Some(m) = field_mixing {
        ff.mixing = m;
    }
    let out = crate::pbc::finite_field::run_finite_field(
        &molecule,
        &parameters,
        &options,
        [divisions[0], divisions[1], divisions[2]],
        vec3_arg(&field, "field")?,
        &ff,
    )
    .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("energy_ev", out.energy_ev)?;
    dict.set_item("enthalpy_ev", out.enthalpy_ev)?;
    dict.set_item("field", vec3_list(out.field))?;
    dict.set_item("phase", out.phase.to_vec())?;
    dict.set_item(
        "electronic_polarization",
        vec3_list(out.electronic_polarization),
    )?;
    dict.set_item("ionic_polarization", vec3_list(out.ionic_polarization))?;
    dict.set_item("polarization", vec3_list(out.polarization))?;
    dict.set_item("iterations", out.iterations)?;
    dict.set_item("converged", out.converged)?;
    dict.set_item("resolved", out.resolved.to_vec())?;
    Ok(dict.into())
}

/// A three-element Python list as a `Vec3`, refusing any other length by name.
fn vec3_arg(v: &[f64], what: &str) -> PyResult<crate::math::Vec3> {
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have three components, got {}",
            v.len()
        )));
    }
    Ok(crate::math::Vec3::new(v[0], v[1], v[2]))
}

/// A `Vec3` as a three-element list, for the result dicts.
fn vec3_list(v: crate::math::Vec3) -> Vec<f64> {
    vec![v.x, v.y, v.z]
}

/// Parse the `long_range` keyword, naming the accepted values rather than falling back silently.
fn parse_long_range(value: Option<&str>) -> PyResult<crate::dfpt::LongRange> {
    match value {
        None | Some("auto") => Ok(crate::dfpt::LongRange::Auto),
        Some("require") => Ok(crate::dfpt::LongRange::Require),
        Some("off") => Ok(crate::dfpt::LongRange::Off),
        Some(other) => Err(PyValueError::new_err(format!(
            "long_range must be 'auto', 'require' or 'off', got {other:?}"
        ))),
    }
}

/// `eps_inf` for a chain or a slab, where the cell has no volume of its own.
///
/// `slab_thickness` (Bohr) or `wire_cross_section` (Bohr^2) -- exactly one, and required. A
/// supercell says where the atoms are, not where the material stops, so there is nothing to
/// infer: doubling the vacuum must not change `eps`. The conversion carries the depolarization
/// factor of the assumed body rather than being a division; see `docs/pbc.md`.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, slab_thickness=None, wire_cross_section=None, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None))]
#[allow(clippy::too_many_arguments)]
fn dielectric_with_extent(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    slab_thickness: Option<f64>,
    wire_cross_section: Option<f64>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
) -> PyResult<PyObject> {
    let extent =
        match (slab_thickness, wire_cross_section) {
            (Some(d), None) => crate::pbc::ExtentConvention::SlabThickness(d),
            (None, Some(s)) => crate::pbc::ExtentConvention::WireCrossSection(s),
            (None, None) => {
                return Err(PyValueError::new_err(
                    "an extent is required: pass slab_thickness (Bohr) for a 2-D cell or \
                 wire_cross_section (Bohr^2) for a 1-D one. There is no default, because a \
                 supercell says where the atoms are and not where the material stops.",
                ))
            }
            (Some(_), Some(_)) => return Err(PyValueError::new_err(
                "pass exactly one of slab_thickness and wire_cross_section: a cell is a slab or a \
                 wire, not both",
            )),
        };
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "an assigned extent needs a periodic system; pass a non-degenerate `cell`",
        ));
    }
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let settings = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    let out =
        crate::dfpt::dielectric_with_extent(&molecule, &parameters, &options, &settings, extent)
            .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("dielectric", mat3_rows(&out.dielectric))?;
    dict.set_item("polarizability", mat3_rows(&out.polarizability))?;
    dict.set_item("axis", vec![out.axis.x, out.axis.y, out.axis.z])?;
    dict.set_item("measure_bohr", out.measure)?;
    dict.set_item("extent", out.extent.value())?;
    // Which convention that number is in. Without this the dict carries a bare `3.33` and no way
    // to tell a thickness in Bohr from a cross-section in Bohr² — the units differ, so do the
    // depolarization factors, and a reader cannot recover either from the value alone.
    dict.set_item(
        "extent_convention",
        match out.extent {
            crate::pbc::ExtentConvention::SlabThickness(_) => "slab_thickness",
            crate::pbc::ExtentConvention::WireCrossSection(_) => "wire_cross_section",
        },
    )?;
    dict.set_item(
        "extent_unit",
        if out.extent.periodic_directions() == 2 {
            "bohr"
        } else {
            "bohr2"
        },
    )?;
    dict.set_item("sheet_parallel_bohr", out.invariants.parallel)?;
    dict.set_item("sheet_perpendicular_bohr", out.invariants.perpendicular)?;
    dict.set_item("axis_mixing", out.axis_mixing)?;
    Ok(dict.into())
}

/// Born effective charges, the electronic dielectric tensor, and the LO-TO data built from them.
///
/// A homogeneous field is not a periodic operator, so this goes through the commutator [H, r]
/// (see docs/theory.md, convention C-3). Z* is quantitative in PM7; eps_inf is
/// systematically low because the minimal valence basis has no polarization functions -- see
/// docs/fidelity.md, which carries the measured numbers.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, charge=0.0, multiplicity=1, method="pm7", reference="auto", pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, dfpt_tolerance=None, dfpt_max_iterations=None, dfpt_mixing=None, lo_to_direction=None, long_range=None, keep_response=false))]
#[allow(clippy::too_many_arguments)]
fn born_charges(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    dfpt_tolerance: Option<f64>,
    dfpt_max_iterations: Option<usize>,
    dfpt_mixing: Option<f64>,
    lo_to_direction: Option<Vec<f64>>,
    long_range: Option<String>,
    keep_response: bool,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell: Some(cell),
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field: None,
        dipole_origin: None,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    if molecule.cell.is_none() {
        return Err(PyValueError::new_err(
            "Born charges need a periodic system; pass a non-degenerate `cell`",
        ));
    }
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let mut dfpt_options = dfpt_settings(dfpt_tolerance, dfpt_max_iterations, dfpt_mixing);
    dfpt_options.long_range = parse_long_range(long_range.as_deref())?;
    dfpt_options.keep_response = keep_response;
    let result = crate::dfpt::born_and_dielectric(&molecule, &parameters, &options, &dfpt_options)
        .map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item(
        "born_charges",
        result.born.iter().map(mat3_rows).collect::<Vec<_>>(),
    )?;
    dict.set_item("dielectric", mat3_rows(&result.dielectric))?;
    dict.set_item("polarizability", mat3_rows(&result.polarizability))?;
    dict.set_item("acoustic_residual", result.acoustic_residual())?;
    dict.set_item("volume_bohr3", result.volume_bohr3)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("residual", result.residual)?;
    // LO–TO is 3-D only and needs a direction, so it is present only when both hold. Reporting
    // `None` rather than raising keeps the common 1-D/2-D case from needing a try/except.
    match (result.non_analytic(), lo_to_direction) {
        (Ok(na), Some(q)) if q.len() == 3 => {
            let matrix = na.matrix([q[0], q[1], q[2]]).map_err(to_py_err)?;
            dict.set_item("lo_to_direction", vec![q[0], q[1], q[2]])?;
            dict.set_item("lo_to_force_constants_ev_per_bohr2", complex_rows(&matrix))?;
        }
        (_, Some(q)) if q.len() != 3 => {
            return Err(PyValueError::new_err(
                "lo_to_direction needs three Cartesian components",
            ))
        }
        _ => {
            dict.set_item("lo_to_force_constants_ev_per_bohr2", py.None())?;
        }
    }
    Ok(dict.into())
}

/// The converged wavefunction as a Molden-format string.
///
/// Returns the text rather than writing a file, so the caller decides where it goes. See the Rust
/// `molden` module for the orthogonality caveat that travels inside the file: NDDO assumes an
/// orthonormal AO basis, so the coefficients are in an implicitly orthogonalized basis while the
/// listed functions are the raw non-orthogonal ones.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", basis="sto-6g", comment=None, field=None, dipole_origin=None, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None))]
#[allow(clippy::too_many_arguments)]
fn molden(
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    basis: &str,
    comment: Option<String>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
) -> PyResult<String> {
    // Molden is molecules only — its `[MO]` section is a list of molecular orbitals, not Bloch
    // states — so this deliberately takes no periodic keywords.
    let periodic = PeriodicArgs {
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
        ..PeriodicArgs::default()
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;
    let lowered = basis.to_ascii_lowercase();
    let chosen = match lowered.as_str() {
        "sto" | "slater" => crate::molden::MoldenBasis::Sto,
        other => {
            let n = other
                .strip_prefix("sto-")
                .and_then(|rest| rest.strip_suffix('g'))
                .and_then(|digits| digits.parse::<usize>().ok())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "unknown basis {basis:?}; use \"sto\" for the exact Slater basis or \
                         \"sto-nG\" (e.g. \"sto-6g\") for a Gaussian rendering basis"
                    ))
                })?;
            if n == 0 || n > 12 {
                return Err(PyValueError::new_err(
                    "an STO-nG expansion needs between 1 and 12 Gaussians",
                ));
            }
            crate::molden::MoldenBasis::StoNg { n }
        }
    };
    let result = run_pm7(&molecule, &parameters, &options).map_err(to_py_err)?;
    crate::molden::to_molden(
        &molecule,
        &parameters,
        &result,
        &crate::molden::MoldenOptions {
            basis: chosen,
            comment,
        },
    )
    .map_err(to_py_err)
}

/// Everything that comes out of one Hessian: frequencies, modes, IR intensities, and — on
/// request — the CPHF orbital response.
///
/// One entry point rather than several because they all come from the **same** CPHF solve. Asking
/// for frequencies and then a spectrum through separate calls would pay for that solve twice, and
/// the existing `frequencies`/`hessian` functions stay exactly as they are for callers who only
/// want the cheap thing.
///
/// `orbital_response` is off by default. It is not slow — the Hessian solves for it either way —
/// but it is `3N x n_vir x n_occ` floats to hand across the language boundary, so it is opt-in.
/// It comes back as a **flat list plus a shape** rather than a nested list, because a nested one
/// would materialize that many Python float objects.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto", cell=None, pbc=None, kpoints=None, kpoint_shift=None, smearing=None, pbc_mode=None, field=None, dipole_origin=None, hessian=true, frequencies=true, modes=false, ir=false, orbital_response=false, step=1.0e-3, scf_tolerance=None, max_scf=None, cphf_max_iterations=None, stability=None, exchange_cutoff=None, use_diis=None, projection=None))]
#[allow(clippy::too_many_arguments)]
fn vibrations(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    kpoints: Option<Vec<usize>>,
    kpoint_shift: Option<Vec<f64>>,
    smearing: Option<(String, f64, usize)>,
    pbc_mode: Option<String>,
    field: Option<Vec<f64>>,
    dipole_origin: Option<String>,
    hessian: bool,
    frequencies: bool,
    modes: bool,
    ir: bool,
    orbital_response: bool,
    step: f64,
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    cphf_max_iterations: Option<usize>,
    stability: Option<String>,
    exchange_cutoff: Option<(f64, f64)>,
    use_diis: Option<bool>,
    projection: Option<String>,
) -> PyResult<PyObject> {
    let periodic = PeriodicArgs {
        cell,
        pbc,
        kpoints,
        kpoint_shift,
        smearing,
        pbc_mode,
        scf_tolerance,
        max_scf,
        cphf_max_iterations,
        stability,
        exchange_cutoff,
        use_diis,
        field,
        dipole_origin,
    };
    let molecule = molecule(&numbers, &positions, charge, multiplicity, &periodic)?;
    let (parameters, options) = model(method, charge, multiplicity, reference, &periodic)?;

    let projection = parse_projection(&projection)?;
    let dict = PyDict::new(py);
    // The IR path is a strict superset of the others, so take it whenever anything needs the
    // dipole derivatives and fall back to the plain Hessian otherwise.
    if ir {
        let spectrum =
            crate::ir::ir_spectrum_projected(&molecule, &parameters, &options, step, projection)
                .map_err(to_py_err)?;
        if hessian {
            dict.set_item("hessian_ev_per_angstrom2", scaled_rows(&spectrum.hessian))?;
            dict.set_item("hessian_hartree_per_bohr2", hartree_rows(&spectrum.hessian))?;
        }
        if frequencies {
            dict.set_item("frequencies_cm", spectrum.frequencies_cm.clone())?;
        }
        if modes {
            dict.set_item("modes", rows(&spectrum.modes))?;
            dict.set_item("cartesian_modes", rows(&spectrum.cartesian_modes))?;
        }
        dict.set_item("ir_intensities_km_per_mol", spectrum.intensities_km_per_mol)?;
        dict.set_item("dipole_derivatives_e", rows(&spectrum.dipole_derivatives))?;
        dict.set_item(
            "dipole_derivatives_debye_per_angstrom",
            spectrum
                .dipole_derivatives
                .as_slice()
                .chunks(spectrum.dipole_derivatives.cols)
                .map(|row| {
                    row.iter()
                        .map(|v| v * crate::constants::E_IN_DEBYE_PER_ANGSTROM)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "mode_dipole_derivatives",
            rows(&spectrum.mode_dipole_derivatives),
        )?;
        dict.set_item("mopac_dipt", spectrum.mopac_dipt)?;
        dict.set_item("mopac_trdip", rows(&spectrum.mopac_trdip))?;
    } else {
        let request = crate::hessian::HessianRequest {
            response: orbital_response,
        };
        let solved =
            crate::hessian::analytic_hessian_with(&molecule, &parameters, &options, step, &request)
                .map_err(to_py_err)?;
        if hessian {
            dict.set_item("hessian_ev_per_angstrom2", scaled_rows(&solved.hessian))?;
            dict.set_item("hessian_hartree_per_bohr2", hartree_rows(&solved.hessian))?;
        }
        if frequencies || modes {
            let out = crate::hessian::vibrational_modes_projected(
                &molecule,
                solved.hessian.clone(),
                projection,
            )
            .map_err(to_py_err)?;
            if frequencies {
                dict.set_item("frequencies_cm", out.frequencies_cm)?;
            }
            if modes {
                dict.set_item("modes", rows(&out.modes))?;
                dict.set_item("cartesian_modes", rows(&out.cartesian_modes))?;
            }
        }
        if let Some(response) = &solved.response {
            add_response(&dict, response)?;
        }
    }

    // The response is only reachable through the non-IR branch above, so ask for it explicitly
    // when both were requested.
    if orbital_response && !dict.contains("orbital_response")? {
        let request = crate::hessian::HessianRequest { response: true };
        let solved =
            crate::hessian::analytic_hessian_with(&molecule, &parameters, &options, step, &request)
                .map_err(to_py_err)?;
        if let Some(response) = &solved.response {
            add_response(&dict, response)?;
        }
    }
    Ok(dict.into())
}

/// The CPHF response as a flat list plus its shape. See [`vibrations`] for why it is not nested.
fn add_response(
    dict: &Bound<'_, PyDict>,
    response: &crate::hessian::OrbitalResponse,
) -> PyResult<()> {
    let flatten = |blocks: &[crate::linalg::Matrix]| -> (Vec<f64>, Vec<usize>) {
        let (r, c) = blocks.first().map(|m| (m.rows, m.cols)).unwrap_or((0, 0));
        let mut flat = Vec::with_capacity(blocks.len() * r * c);
        for block in blocks {
            flat.extend_from_slice(block.as_slice());
        }
        (flat, vec![blocks.len(), r, c])
    };
    let (flat, shape) = flatten(&response.u);
    dict.set_item("orbital_response", flat)?;
    dict.set_item("orbital_response_shape", shape)?;
    if let Some(beta) = &response.u_beta {
        let (flat, shape) = flatten(beta);
        dict.set_item("orbital_response_beta", flat)?;
        dict.set_item("orbital_response_beta_shape", shape)?;
    }
    Ok(())
}

fn scaled_rows(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|i| {
            (0..m.cols)
                .map(|j| m[(i, j)] * EV_PER_BOHR2_TO_EV_PER_ANGSTROM2)
                .collect()
        })
        .collect()
}

fn hartree_rows(m: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m[(i, j)] * EV_TO_HARTREE).collect())
        .collect()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(scf_stability, module)?)?;
    module.add_function(wrap_pyfunction!(orbitals, module)?)?;
    module.add_function(wrap_pyfunction!(vibrations, module)?)?;
    module.add_function(wrap_pyfunction!(single_point, module)?)?;
    module.add_function(wrap_pyfunction!(gradient, module)?)?;
    module.add_function(wrap_pyfunction!(forces, module)?)?;
    module.add_function(wrap_pyfunction!(stress, module)?)?;
    module.add_function(wrap_pyfunction!(optimize, module)?)?;
    module.add_function(wrap_pyfunction!(frequencies, module)?)?;
    module.add_function(wrap_pyfunction!(hessian, module)?)?;
    module.add_function(wrap_pyfunction!(phonons, module)?)?;
    module.add_function(wrap_pyfunction!(dfpt, module)?)?;
    module.add_function(wrap_pyfunction!(born_charges, module)?)?;
    module.add_function(wrap_pyfunction!(dielectric_with_extent, module)?)?;
    module.add_function(wrap_pyfunction!(polarizability, module)?)?;
    module.add_function(wrap_pyfunction!(dielectric_origin_sensitivity, module)?)?;
    module.add_function(wrap_pyfunction!(static_dielectric, module)?)?;
    module.add_function(wrap_pyfunction!(berry_polarization, module)?)?;
    module.add_function(wrap_pyfunction!(finite_field, module)?)?;
    module.add_function(wrap_pyfunction!(molden, module)?)?;
    module.add_function(wrap_pyfunction!(band_structure, module)?)?;
    module.add_function(wrap_pyfunction!(divide_and_conquer, module)?)?;
    Ok(())
}
