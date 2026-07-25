// SPDX-License-Identifier: GPL-3.0-or-later

//! Embedded PM7 data tables generated from the pinned MOPAC v23.2.5 source.
//!
//! Regenerate these files with `python tools/extract_params/extract_pm7.py` after
//! placing the matching MOPAC source tree at `.mopac-source/`.  The generator is
//! deliberately checked in so each numeric value has a direct, reproducible origin.

pub const PM7_ELEMENTS_CSV: &str = include_str!("data/pm7_elements.csv");
pub const PM7_PAIRS_CSV: &str = include_str!("data/pm7_pairs.csv");
pub const PM7_VPAR_CSV: &str = include_str!("data/pm7_vpar.csv");
pub const PM7_TS_ELEMENTS_CSV: &str = include_str!("data/pm7ts_elements.csv");
pub const PM7_TS_PAIRS_CSV: &str = include_str!("data/pm7ts_pairs.csv");
pub const PM7_TS_VPAR_CSV: &str = include_str!("data/pm7ts_vpar.csv");
pub const PM7_SPARKLES_CSV: &str = include_str!("data/pm7_sparkles.csv");

/// Atomic masses (u), indexed by atomic number.  Values through Rn are used by
/// vibrational analysis; unlisted synthetic elements intentionally retain 0.0.
pub const MASS: [f64; 87] = [
    0.0, 1.0079, 4.0026, 6.94, 9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.9984, 20.179, 22.98977,
    24.305, 26.98154, 28.0855, 30.97376, 32.06, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867,
    50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.63, 74.922, 78.971,
    79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 97.0, 101.07, 102.91, 106.42,
    107.87, 112.41, 114.82, 118.71, 121.76, 127.6, 126.9, 131.29, 132.91, 137.33, 138.91, 140.12,
    140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.5, 164.93, 167.26, 168.93, 173.05,
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2,
    208.98, 209.0, 210.0, 222.0,
];
