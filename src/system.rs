// SPDX-License-Identifier: GPL-3.0-or-later

//! Molecular and periodic geometry, plus XYZ I/O.
//!
//! Positions are stored in **Bohr** internally. `from_xyz_*` reads Ångström by
//! default (the XYZ convention) and converts on input.
//!
//! A [`Molecule`] carries an optional [`Cell`]. `cell: None` is a molecule and takes exactly the
//! v0.1.x code paths; `cell: Some(..)` makes the same structure a 1-D polymer, 2-D layer, or 3-D
//! crystal. Keeping one type (rather than a separate `Crystal`) is what lets `run_pm7`,
//! `closed_form_gradient`, `analytic_hessian`, and `optimize` stay single entry points.

use crate::cell::{AxisRotation, Cell, Periodicity};
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
    /// Total charge of the system — of the molecule, or **of the unit cell** for a periodic
    /// system (electrons removed = positive). A charged periodic cell is supported: the
    /// Ewald path adds a uniform neutralizing background, so the energy stays finite and
    /// α-independent. See `docs/pbc.md`.
    pub charge: f64,
    /// Spin multiplicity (2S+1). 1 = closed-shell singlet.
    pub multiplicity: usize,
    /// Periodic lattice, or `None` for a molecule.
    pub cell: Option<Cell>,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0.0,
            multiplicity: 1,
            cell: None,
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

    /// Attach a periodic lattice.
    pub fn with_cell(mut self, cell: Cell) -> Self {
        self.cell = Some(cell);
        self
    }

    /// Attach a lattice only if one is given (convenience for plumbing `Option<Cell>` through).
    pub fn with_optional_cell(mut self, cell: Option<Cell>) -> Self {
        self.cell = cell;
        self
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// `true` when the system has at least one periodic direction.
    pub fn is_periodic(&self) -> bool {
        self.cell.is_some()
    }

    /// Number of periodic directions (0 for a molecule).
    pub fn periodicity(&self) -> Periodicity {
        self.cell.map(|c| c.periodicity()).unwrap_or_default()
    }

    /// Sum of atomic numbers (used to derive the electron count).
    pub fn total_nuclear_charge(&self) -> u32 {
        self.atoms.iter().map(|a| a.z as u32).sum()
    }

    /// Wrap every atom into the primitive cell. A no-op for a molecule.
    ///
    /// Energies are invariant under this, so it is only a presentation/robustness helper — it
    /// keeps a long MD trajectory's coordinates from drifting far outside the cell, which would
    /// make the minimum-image search do more work than necessary.
    pub fn wrapped(&self) -> Self {
        let mut out = self.clone();
        if let Some(cell) = self.cell {
            for atom in &mut out.atoms {
                atom.position = cell.wrap(atom.position);
            }
        }
        out
    }

    pub fn from_xyz_file(path: impl AsRef<Path>, charge: f64) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?, charge)
    }

    /// [`Self::from_xyz_file`], also returning how the lattice vectors were reordered.
    ///
    /// A file carrying `pbc="T F T"` has its lattice vectors rotated so the periodic ones lead
    /// (see [`AxisRotation`]). Anything the caller indexes by lattice vector — `--kpoints`,
    /// `--supercell`, a fractional `q` — was written in the file's axis order and has to be put
    /// through the same rotation, so the rotation has to be reachable.
    pub fn from_xyz_file_with_axes(
        path: impl AsRef<Path>,
        charge: f64,
    ) -> Result<(Self, AxisRotation)> {
        Self::from_xyz_str_with_axes(&fs::read_to_string(path)?, charge)
    }

    /// [`Self::from_xyz_str`], also returning how the lattice vectors were reordered.
    pub fn from_xyz_str_with_axes(text: &str, charge: f64) -> Result<(Self, AxisRotation)> {
        Self::parse_xyz(text, charge)
    }

    /// Parse an XYZ block. Coordinates are Ångström and converted to Bohr.
    ///
    /// The comment line is parsed for the **extended XYZ** `Lattice="..."` key, so a periodic
    /// structure written by ASE (`atoms.write("x.xyz")`) round-trips without a separate format.
    /// A plain XYZ file has no such key and yields `cell: None`, exactly as before.
    ///
    /// `Lattice` holds 9 numbers in Ångström, row-major: `a_x a_y a_z b_x b_y b_z c_x c_y c_z`.
    /// A companion `pbc="T T F"` key selects which of those rows are actually periodic; without
    /// it, all three are (the extended-XYZ default for a 3×3 lattice).
    ///
    /// The periodic directions need not be the leading ones: `pbc="T F T"` is accepted, and the
    /// lattice vectors are cyclically reordered so they are (see [`AxisRotation`]). Use
    /// [`Self::from_xyz_str_with_axes`] when you also need to know what the reordering was.
    pub fn from_xyz_str(text: &str, charge: f64) -> Result<Self> {
        Ok(Self::parse_xyz(text, charge)?.0)
    }

    fn parse_xyz(text: &str, charge: f64) -> Result<(Self, AxisRotation)> {
        // A UTF-8 byte-order mark is invisible in every editor and turns the first line into
        // something `parse::<usize>()` rejects, so the file reports "invalid XYZ atom count: 2"
        // where the 2 is plainly a 2. Windows tooling writes one by default -- PowerShell's
        // `Set-Content -Encoding UTF8` and Notepad's "UTF-8" both do -- so this is the ordinary
        // way for a user on this platform to produce a file, not an exotic one.
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut lines = text.lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| Pm7Error::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            Pm7Error::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let comment = lines.next().unwrap_or_default();
        let (cell, rotation) = parse_extxyz_cell(comment)?;
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
        Ok((
            Self {
                atoms,
                charge,
                multiplicity: 1,
                cell,
            },
            rotation,
        ))
    }

    /// Render as an extended-XYZ block (Ångström), including `Lattice`/`pbc` when periodic.
    pub fn to_xyz_string(&self, title: &str) -> String {
        let a0 = crate::constants::BOHR_TO_ANGSTROM;
        let mut comment = title.replace(['"', '\n'], " ");
        if let Some(cell) = self.cell {
            // Extended XYZ always writes a full 3×3 lattice; the open directions are written as
            // the unit normals from the completion basis and marked non-periodic via `pbc`.
            let b = cell.completed_vectors();
            let n: Vec<String> = b
                .iter()
                .flat_map(|v| [v.x * a0, v.y * a0, v.z * a0])
                .map(|x| format!("{x:.12}"))
                .collect();
            let flags = cell.periodicity().flags();
            let pbc = flags
                .iter()
                .map(|p| if *p { "T" } else { "F" })
                .collect::<Vec<_>>()
                .join(" ");
            comment = format!(
                "Lattice=\"{}\" Properties=species:S:1:pos:R:3 pbc=\"{pbc}\" {comment}",
                n.join(" ")
            );
        }
        let mut text = format!("{}\n{comment}\n", self.atoms.len());
        for atom in &self.atoms {
            text.push_str(&format!(
                "{} {:.12} {:.12} {:.12}\n",
                z_to_symbol(atom.z).unwrap_or("X"),
                atom.position.x * a0,
                atom.position.y * a0,
                atom.position.z * a0
            ));
        }
        text
    }
}

/// Extract `Lattice="..."` (and optional `pbc="..."`) from an extended-XYZ comment line.
fn parse_extxyz_cell(comment: &str) -> Result<(Option<Cell>, AxisRotation)> {
    let Some(lattice) = quoted_value(comment, "Lattice") else {
        return Ok((None, AxisRotation::IDENTITY));
    };
    let nums: Vec<f64> = lattice
        .split_whitespace()
        .map(|t| {
            t.parse::<f64>()
                .map_err(|_| Pm7Error::InvalidInput(format!("invalid Lattice entry `{t}`")))
        })
        .collect::<Result<Vec<_>>>()?;
    if nums.len() != 9 {
        return Err(Pm7Error::InvalidInput(format!(
            "extended-XYZ Lattice needs 9 numbers, got {}",
            nums.len()
        )));
    }
    let rows = [
        [nums[0], nums[1], nums[2]],
        [nums[3], nums[4], nums[5]],
        [nums[6], nums[7], nums[8]],
    ];
    let pbc = match quoted_value(comment, "pbc") {
        Some(p) => {
            let flags: Vec<bool> = p
                .split_whitespace()
                .map(|t| matches!(t, "T" | "t" | "True" | "true" | "1"))
                .collect();
            if flags.len() != 3 {
                return Err(Pm7Error::InvalidInput(format!(
                    "extended-XYZ pbc needs 3 flags, got {}",
                    flags.len()
                )));
            }
            [flags[0], flags[1], flags[2]]
        }
        None => [true; 3],
    };
    // `pbc="T F T"` used to be an error here. It is an ordinary thing for ASE to write -- a slab
    // built along y, say -- so it is now reordered rather than refused.
    Cell::from_angstrom_rows_pbc(&rows, pbc)
}

/// Value of `key="..."` (or a bare `key=value`) in an extended-XYZ comment line.
fn quoted_value(text: &str, key: &str) -> Option<String> {
    let mut rest = text;
    loop {
        let idx = rest.find(key)?;
        let before_ok = idx == 0
            || rest[..idx]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace());
        let after = &rest[idx + key.len()..];
        if before_ok {
            if let Some(after) = after.strip_prefix('=') {
                return Some(match after.strip_prefix('"') {
                    Some(q) => q.split('"').next().unwrap_or_default().to_string(),
                    None => after
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }
        rest = &rest[idx + key.len()..];
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
        // The whole table, not a hard-coded 86: an atomic number this accepts and `ELEMENTS`
        // cannot name would come back from `z_to_symbol` as `None` and round-trip to nothing.
        if z >= 1 && (z as usize) < ELEMENTS.len() {
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

/// Element symbols by atomic number, index 0 unused.
///
/// **Through radon was not far enough.** `src/data/pm7_elements.csv` carries parameters for thorium
/// (90), californium (98) and nobelium (102) — the actinide sparkles — and this table stopped at
/// 86, so `Th 0.0 0.0 0.0` in an XYZ file was rejected as an unknown element by a program that had
/// its parameters loaded. Three parameterized elements were unreachable from every file-based entry
/// point, and nothing noticed until the oracle's coverage check required a case for each of them.
pub const ELEMENTS: [&str; 104] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
];
