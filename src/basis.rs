// SPDX-License-Identifier: GPL-3.0-or-later

//! Minimal valence AO basis: one `s` shell for H/He, an `s`+`p` set (s, px, py, pz)
//! for every heavier PM7 element. Slater exponents come from the parameters; overlaps
//! are evaluated analytically (see [`crate::overlap`]).

use crate::error::Result;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// Angular-momentum / Cartesian label of an AO within its atom's block.
/// 0 = s, 1 = px, 2 = py, 3 = pz.
#[derive(Clone, Copy, Debug)]
pub struct AoInfo {
    pub atom: usize,
    pub z: u8,
    pub orb: u8,
}

impl AoInfo {
    pub fn is_s(&self) -> bool {
        self.orb == 0
    }
}

#[derive(Clone, Debug)]
pub struct Basis {
    pub aos: Vec<AoInfo>,
    /// First AO index of each atom.
    pub atom_offset: Vec<usize>,
    /// Number of AOs on each atom (1 or 4).
    pub atom_norb: Vec<usize>,
    pub nao: usize,
}

impl Basis {
    pub fn build(molecule: &Molecule, params: &Pm7Parameters) -> Result<Self> {
        let mut aos = Vec::new();
        let mut atom_offset = Vec::with_capacity(molecule.atoms.len());
        let mut atom_norb = Vec::with_capacity(molecule.atoms.len());
        for (ia, atom) in molecule.atoms.iter().enumerate() {
            let elem = params.element(atom.z)?;
            // Sparkles (lanthanide +3 cores) contribute no atomic orbitals; they enter
            // only through the electron–core attraction and core–core repulsion.
            atom_offset.push(aos.len());
            atom_norb.push(elem.n_orb);
            for orb in 0..elem.n_orb as u8 {
                aos.push(AoInfo {
                    atom: ia,
                    z: atom.z,
                    orb,
                });
            }
        }
        let nao = aos.len();
        Ok(Self {
            aos,
            atom_offset,
            atom_norb,
            nao,
        })
    }
}
