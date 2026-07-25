// SPDX-License-Identifier: GPL-3.0-or-later

//! Physical constants and unit conversions.
//!
//! Units policy: the semiempirical block is computed in **eV with distances in Bohr**.
//! MOPAC v23.2.5 (our oracle) uses the **2018 CODATA** fundamental constants by default
//! (the legacy MOPAC7 values apply only under the `OLDFPC` keyword), so we adopt the same
//! CODATA values here to reproduce MOPAC bit-for-bit: `ev = 27.211386245988`,
//! `a0 = 0.529177210903 Å`, `1 eV = 23.060547830619029 kcal/mol`. The native API then
//! reports energies in Hartree (= eV / `PM7_EV`) and coordinates in Bohr; the ASE layer
//! reports eV / Å (PM7 energies are natively in eV, so that boundary is exact).

/// eV per Hartree (2018 CODATA, MOPAC v23.2.5 `fpcref(1,4)`).
pub const PM7_EV: f64 = 27.211_386_245_988;
/// Bohr radius in Ångström (2018 CODATA, MOPAC v23.2.5 `fpcref(1,3)`).
pub const PM7_A0: f64 = 0.529_177_210_903;

pub const HARTREE_TO_EV: f64 = PM7_EV;
pub const EV_TO_HARTREE: f64 = 1.0 / PM7_EV;

pub const ANGSTROM_TO_BOHR: f64 = 1.0 / PM7_A0;
pub const BOHR_TO_ANGSTROM: f64 = PM7_A0;

/// 1 eV in kcal/mol (2018 CODATA, MOPAC v23.2.5 `fpcref(1,9)`); heats of formation
/// are reported in kcal/mol.
pub const EV_TO_KCAL: f64 = 23.060_547_830_619_03;
pub const KCAL_TO_EV: f64 = 1.0 / EV_TO_KCAL;

/// Force conversion for the ASE boundary: Hartree/Bohr → eV/Å. (PM7 forces are natively
/// eV/Å; this is only used if a caller works from atomic-unit gradients.)
pub const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = HARTREE_TO_EV / BOHR_TO_ANGSTROM;

/// Atomic-unit dipole (e·a0) to Debye.
pub const AU_DIPOLE_TO_DEBYE: f64 = 2.541_746_473;

/// Cordero/Pyykkö-style covalent radii (Å), used only for geometric bond perception
/// in geometry utilities — generic element data, not PM7 model parameters.
/// Index by atomic number; unknown Z falls back to 1.5 Å.
pub fn covalent_radius_angstrom(z: u8) -> f64 {
    const RAD_A: [f64; 87] = [
        0.0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
        1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
        1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,
        1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
        2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62,
        1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    ];
    RAD_A.get(z as usize).copied().unwrap_or(1.5)
}
