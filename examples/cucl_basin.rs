// SPDX-License-Identifier: GPL-3.0-or-later
//! Can anything short of a stability analysis find CuCl's lower SCF solution?
//!
//! `cargo run --release --example cucl_basin`
//!
//! The 176-case oracle has `pm7-rs` converging CuCl to 13.39233 kcal/mol and MOPAC to 8.21456 —
//! 5.18 kcal/mol lower, with different Mulliken charges, so it is a different SCF solution rather
//! than a different integral. AgCl behaves the same way. Neither `--reference uhf`, `--no-diis`
//! nor a `1e-11` tolerance moves ours.
//!
//! Before reaching for orbital-Hessian stability analysis — which is a real feature with a real
//! cost — it is worth asking whether the lower basin is reachable by any of the knobs that already
//! exist. A level shift changes which solution the iteration falls into without changing what a
//! solution *is*; an unrestricted start from a broken-symmetry guess does the same. If one of them
//! lands on 8.21, the answer is a better default, not a new analysis.

use pm7_rs::math::Vec3;
use pm7_rs::{run_pm7, Atom, Molecule, Pm7Method, Pm7Options, Pm7Parameters, ScfReference};

fn main() {
    let params = Pm7Parameters::method(Pm7Method::Pm7).expect("PM7 parameters");
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;

    for (label, z_metal, bond, mopac) in
        [("CuCl", 29u8, 2.05, 8.21456), ("AgCl", 47, 2.28, 18.93322)]
    {
        let molecule = Molecule::new(vec![
            Atom {
                z: z_metal,
                position: Vec3::zero(),
            },
            Atom {
                z: 17,
                position: Vec3::new(bond * a0, 0.0, 0.0),
            },
        ]);
        println!("=== {label}: MOPAC finds {mopac:.5} kcal/mol ===");

        let mut best = f64::INFINITY;
        for shift in [0.0, 1.0, 3.0, 5.0, 10.0, 20.0] {
            for reference in [ScfReference::Auto, ScfReference::Unrestricted] {
                for accelerator in [
                    pm7_rs::scf::ScfAccelerator::AdiisCdiis,
                    pm7_rs::scf::ScfAccelerator::Cdiis,
                    pm7_rs::scf::ScfAccelerator::None,
                ] {
                    let options = Pm7Options {
                        level_shift: shift,
                        reference,
                        accelerator,
                        p_tol: 1.0e-10,
                        e_tol: 1.0e-11,
                        max_scf: 500,
                        ..Pm7Options::default()
                    };
                    match run_pm7(&molecule, &params, &options) {
                        Ok(result) if result.converged => {
                            let hof = result.heat_of_formation_kcal;
                            if hof < best - 1.0e-6 {
                                best = hof;
                                println!(
                                    "  new lowest {hof:12.5}  (shift {shift:4.1}, {reference:?}, \
                                     {accelerator:?})"
                                );
                            }
                        }
                        Ok(_) => {}
                        Err(_) => {}
                    }
                }
            }
        }
        println!(
            "  best found {best:.5}, MOPAC {mopac:.5}, gap {:.5}\n",
            best - mopac
        );
    }
}
