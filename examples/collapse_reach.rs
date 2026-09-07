// SPDX-License-Identifier: GPL-3.0-or-later
//! How much of a molecule's pair work is past the 7 A point-charge range?
//!
//! `cargo run --release --example collapse_reach`
//!
//! Past the feather range a two-centre block is exactly its monopole, so the periodic path builds
//! it in O(1) instead of evaluating `b^2 x b^2` integrals. Molecules do not take that branch, and
//! extending it to them means adding the same collapse to three more sites in the Hessian path
//! -- energy, gradient and second derivative must come from the same Hamiltonian or the
//! derivatives are of something else.
//!
//! Whether that is worth three new implementations depends on how many pairs are actually beyond
//! the range and on what the pair loop costs, which is what this prints.

use pm7_rs::{Molecule, Pm7Options, Pm7Parameters};

fn main() {
    let params = Pm7Parameters::standard().unwrap();
    let feather = pm7_rs::pbc::FEATHER_RANGE_BOHR;
    println!(
        "feather range: {:.3} Bohr ({:.2} A)",
        feather,
        feather * 0.529_177_210_903
    );
    println!(
        "\n{:>18} {:>7} {:>9} {:>9} {:>9} {:>12}",
        "molecule", "atoms", "pairs", "past 7A", "share", "orb work saved"
    );
    for path in [
        "examples/water.xyz",
        "examples/ethanol.xyz",
        "examples/bench102.xyz",
    ] {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let molecule = Molecule::from_xyz_str(&text, 0.0).unwrap();
        let n = molecule.atoms.len();
        let mut pairs = 0usize;
        let mut far = 0usize;
        // Orbital work is `n_i^2 * n_j^2` for an evaluated block and O(1) for a collapsed one.
        let mut work_all = 0.0_f64;
        let mut work_near = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = (molecule.atoms[j].position - molecule.atoms[i].position).norm();
                let bi = params
                    .element(molecule.atoms[i].z)
                    .map(|e| e.n_orb)
                    .unwrap_or(0) as f64;
                let bj = params
                    .element(molecule.atoms[j].z)
                    .map(|e| e.n_orb)
                    .unwrap_or(0) as f64;
                let w = bi * bi * bj * bj;
                pairs += 1;
                work_all += w;
                if r > feather {
                    far += 1;
                } else {
                    work_near += w;
                }
            }
        }
        println!(
            "{:>18} {n:>7} {pairs:>9} {far:>9} {:>8.1}% {:>11.1}%",
            path.trim_start_matches("examples/"),
            100.0 * far as f64 / pairs.max(1) as f64,
            100.0 * (1.0 - work_near / work_all.max(1.0))
        );
    }

    // And what the pair loop costs, so the share above can be read against something.
    if std::env::var_os("PM7_PROFILE").is_some() {
        let text = std::fs::read_to_string("examples/bench102.xyz").unwrap();
        let molecule = Molecule::from_xyz_str(&text, 0.0).unwrap();
        let _ = pm7_rs::analytic_hessian(&molecule, &params, &Pm7Options::default(), 1e-4).unwrap();
        println!();
        pm7_rs::profile::report_and_reset();
    } else {
        println!("\n(set PM7_PROFILE=1 to see what share of a Hessian the pair loops are)");
    }
}
