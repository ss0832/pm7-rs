// SPDX-License-Identifier: GPL-3.0-or-later

//! Molecular geometry and XYZ I/O, ported from `gfn1-rs`'s `system.rs` with the
//! periodic-cell machinery removed (PM7 is a molecular method).
//!
//! Positions are stored in **Bohr** internally. `from_xyz_*` reads Ångström by
//! default (the XYZ convention) and converts on input.

use crate::constants::ANGSTROM_TO_BOHR;
use crate::error::{Pm7Error, Result};
use crate::math::Vec3;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    /// Position in Bohr.
    pub position: Vec3,
}

#[derive(Clone, Debug)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// Total molecular charge (electrons removed = positive).
    pub charge: f64,
    /// Spin multiplicity (2S+1). 1 = closed-shell singlet.
    pub multiplicity: usize,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0.0,
            multiplicity: 1,
        }
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_multiplicity(mut self, multiplicity: usize) -> Self {
        self.multiplicity = multiplicity.max(1);
        self
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Sum of atomic numbers (used to derive the electron count).
    pub fn total_nuclear_charge(&self) -> u32 {
        self.atoms.iter().map(|a| a.z as u32).sum()
    }

    pub fn from_xyz_file(path: impl AsRef<Path>, charge: f64) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?, charge)
    }

    /// Parse a standard XYZ block. Coordinates are Ångström and converted to Bohr.
    pub fn from_xyz_str(text: &str, charge: f64) -> Result<Self> {
        let mut lines = text.lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| Pm7Error::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            Pm7Error::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let _comment = lines.next().unwrap_or_default();
        let mut atoms = Vec::with_capacity(natoms);
        for idx in 0..natoms {
            let line_no = idx + 3;
            let line = lines
                .next()
                .ok_or_else(|| Pm7Error::InvalidInput(format!("XYZ ended before atom {idx}")))?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 4 {
                return Err(Pm7Error::InvalidInput(format!(
                    "XYZ line {line_no} has fewer than 4 fields"
                )));
            }
            let z = symbol_to_z(parts[0]).ok_or_else(|| {
                Pm7Error::InvalidInput(format!("unknown element on line {line_no}: {}", parts[0]))
            })?;
            let position = Vec3::new(
                parse_f64(parts[1], line_no)?,
                parse_f64(parts[2], line_no)?,
                parse_f64(parts[3], line_no)?,
            ) * ANGSTROM_TO_BOHR;
            atoms.push(Atom { z, position });
        }
        Ok(Self {
            atoms,
            charge,
            multiplicity: 1,
        })
    }
}

fn parse_f64(token: &str, line: usize) -> Result<f64> {
    token.parse::<f64>().map_err(|_| Pm7Error::Parse {
        line,
        message: format!("invalid floating point value: {token}"),
    })
}

pub fn symbol_to_z(sym: &str) -> Option<u8> {
    if let Ok(z) = sym.parse::<u8>() {
        if (1..=86).contains(&z) {
            return Some(z);
        }
    }
    let s = normalize_symbol(sym);
    ELEMENTS
        .iter()
        .position(|&x| x == s.as_str())
        .map(|i| i as u8)
}

pub fn z_to_symbol(z: u8) -> Option<&'static str> {
    ELEMENTS.get(z as usize).copied().filter(|s| !s.is_empty())
}

fn normalize_symbol(sym: &str) -> String {
    let mut chars = sym.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.push(first.to_ascii_uppercase());
            for c in chars {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        None => String::new(),
    }
}

pub const ELEMENTS: [&str; 87] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
];
