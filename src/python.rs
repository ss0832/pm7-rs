// SPDX-License-Identifier: GPL-3.0-or-later
//! Python bindings for the PM7-family native API.

use crate::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::gradient::closed_form_gradient;
use crate::optimizer::{optimize as optimize_geometry, OptOptions};
use crate::params::Pm7Parameters;
use crate::scf::{run_pm7, Pm7Options, ScfReference};
use crate::system::{Atom, Molecule};
use crate::method::Pm7Method;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::str::FromStr;

/// eV/Bohr² → eV/Å² (Hessian, second derivative in length).
const EV_PER_BOHR2_TO_EV_PER_ANGSTROM2: f64 = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;

fn to_py_err(error: crate::error::Pm7Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn molecule(
    numbers: &[u8],
    positions: &[Vec<f64>],
    charge: f64,
    multiplicity: usize,
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
    Ok(Molecule {
        atoms,
        charge,
        multiplicity: multiplicity.max(1),
    })
}

fn model(
    method: &str,
    charge: f64,
    multiplicity: usize,
    reference: &str,
) -> PyResult<(Pm7Parameters, Pm7Options)> {
    let method = method.parse::<Pm7Method>().map_err(to_py_err)?;
    let reference = ScfReference::from_str(reference).map_err(to_py_err)?;
    let parameters = Pm7Parameters::method(method).map_err(to_py_err)?;
    Ok((
        parameters,
        Pm7Options {
            method,
            charge,
            multiplicity,
            reference,
            ..Pm7Options::default()
        },
    ))
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    let molecule = molecule(&numbers, &positions, charge, multiplicity)?;
    let (parameters, options) = model(method, charge, multiplicity, reference)?;
    let result = run_pm7(&molecule, &parameters, &options).map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("method", method)?;
    // Echo the accepted inputs so callers can confirm they were received.
    dict.set_item("charge", charge)?;
    dict.set_item("multiplicity", multiplicity)?;
    dict.set_item("reference", options.reference.to_string())?;
    dict.set_item("unrestricted", result.unrestricted)?;
    dict.set_item("energy_hartree", result.total_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", result.total_ev)?;
    dict.set_item("heat_of_formation_kcal", result.heat_of_formation_kcal)?;
    dict.set_item("electronic_ev", result.electronic_ev)?;
    dict.set_item("core_ev", result.core_ev)?;
    dict.set_item("charges", result.charges)?;
    dict.set_item(
        "dipole_debye",
        [
            result.dipole_debye.x,
            result.dipole_debye.y,
            result.dipole_debye.z,
        ],
    )?;
    dict.set_item("homo_ev", result.homo_ev)?;
    dict.set_item("lumo_ev", result.lumo_ev)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
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
    sign: f64,
) -> PyResult<(f64, Vec<[f64; 3]>, Vec<[f64; 3]>, f64)> {
    let molecule = molecule(numbers, positions, charge, multiplicity)?;
    let (parameters, options) = model(method, charge, multiplicity, reference)?;
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
    ))
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    let (energy_ev, au, ev_ang, hof) = gradient_vectors(
        &numbers,
        &positions,
        charge,
        multiplicity,
        method,
        reference,
        1.0,
    )?;
    let dict = PyDict::new(py);
    dict.set_item("energy_hartree", energy_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", energy_ev)?;
    dict.set_item("heat_of_formation_kcal", hof)?;
    dict.set_item("gradient_hartree_per_bohr", au)?;
    dict.set_item("gradient_ev_per_angstrom", ev_ang)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    // Force = −∂E/∂x.
    let (energy_ev, au, ev_ang, hof) = gradient_vectors(
        &numbers,
        &positions,
        charge,
        multiplicity,
        method,
        reference,
        -1.0,
    )?;
    let dict = PyDict::new(py);
    dict.set_item("energy_hartree", energy_ev * EV_TO_HARTREE)?;
    dict.set_item("energy_ev", energy_ev)?;
    dict.set_item("heat_of_formation_kcal", hof)?;
    dict.set_item("forces_hartree_per_bohr", au)?;
    dict.set_item("forces_ev_per_angstrom", ev_ang)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    let molecule = molecule(&numbers, &positions, charge, multiplicity)?;
    let (parameters, options) = model(method, charge, multiplicity, reference)?;
    let result = optimize_geometry(&molecule, &parameters, &options, &OptOptions::default())
        .map_err(to_py_err)?;
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
    dict.set_item("energy_hartree", result.scf.total_ev * EV_TO_HARTREE)?;
    dict.set_item("heat_of_formation_kcal", result.scf.heat_of_formation_kcal)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    let molecule = molecule(&numbers, &positions, charge, multiplicity)?;
    let (parameters, options) = model(method, charge, multiplicity, reference)?;
    let result = crate::hessian::vibrational_analysis(&molecule, &parameters, &options, 1.0e-3)
        .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("frequencies_cm", result.frequencies_cm)?;
    dict.set_item("eigenvalues", result.eigenvalues)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, method="pm7", reference="auto"))]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    method: &str,
    reference: &str,
) -> PyResult<PyObject> {
    let molecule = molecule(&numbers, &positions, charge, multiplicity)?;
    let (parameters, options) = model(method, charge, multiplicity, reference)?;
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(single_point, module)?)?;
    module.add_function(wrap_pyfunction!(gradient, module)?)?;
    module.add_function(wrap_pyfunction!(forces, module)?)?;
    module.add_function(wrap_pyfunction!(optimize, module)?)?;
    module.add_function(wrap_pyfunction!(frequencies, module)?)?;
    module.add_function(wrap_pyfunction!(hessian, module)?)?;
    Ok(())
}
