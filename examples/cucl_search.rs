// SPDX-License-Identifier: GPL-3.0-or-later
//! Is CuCl's lower solution reachable from a perturbed starting density?
//!
//! `cargo run --release --example cucl_search`
//!
//! The stability analysis answered the question it was built for and ruled itself out as the fix:
//! CuCl's converged solution has a lowest orbital-Hessian eigenvalue of **+6.05 eV**, so it is a
//! genuine local minimum, not a saddle. MOPAC's solution, 5.18 kcal/mol lower, is a *different*
//! minimum. Both codes start from the same place — `sad_density` is MOPAC's own diagonal guess
//! (`moldat.F90:731-790`) — so what differs is the path the iteration takes, and no analysis of the
//! endpoint can find a basin the path never enters.
//!
//! What can is a **search**: perturb the starting density and see whether any perturbation lands
//! somewhere lower. This asks that question before any such thing is built, because a multi-start
//! that never finds anything is a cost with no benefit.

use pm7_rs::linalg::Matrix;
use pm7_rs::math::Vec3;
use pm7_rs::{run_pm7, Atom, Molecule, Pm7Method, Pm7Options, Pm7Parameters};

fn main() {
    let params = Pm7Parameters::method(Pm7Method::Pm7).expect("PM7 parameters");
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;

    for (label, z, bond, mopac) in [("CuCl", 29u8, 2.05, 8.21456), ("AgCl", 47, 2.28, 18.93322)] {
        let molecule = Molecule::new(vec![
            Atom {
                z,
                position: Vec3::zero(),
            },
            Atom {
                z: 17,
                position: Vec3::new(bond * a0, 0.0, 0.0),
            },
        ]);
        let base = Pm7Options {
            p_tol: 1.0e-10,
            e_tol: 1.0e-11,
            max_scf: 500,
            ..Pm7Options::default()
        };
        let plain = run_pm7(&molecule, &params, &base).expect("scf");
        let basis = pm7_rs::basis::Basis::build(&molecule, &params).expect("basis");
        let nao = basis.nao;
        let start = plain.density.clone();

        println!(
            "=== {label}: default {:.5}, MOPAC {mopac:.5} ===",
            plain.heat_of_formation_kcal
        );
        let mut best = plain.heat_of_formation_kcal;
        let mut found = 0usize;
        // A deterministic sweep rather than a random one: the same perturbations every run, so a
        // result here is reproducible and a failure is a fact rather than a seed.
        for trial in 0..48 {
            let mut guess = start.clone();
            let scale = 0.05 + 0.05 * (trial % 8) as f64;
            for mu in 0..nao {
                for nu in 0..nao {
                    let wobble = (((mu * 13 + nu * 7 + trial * 29) % 17) as f64 - 8.0) / 8.0;
                    guess[(mu, nu)] += scale * wobble * if mu == nu { 1.0 } else { 0.3 };
                }
            }
            // Symmetrize: a density matrix is symmetric, and an asymmetric guess is not a density.
            let mut symmetric = Matrix::zeros(nao, nao);
            for mu in 0..nao {
                for nu in 0..nao {
                    symmetric[(mu, nu)] = 0.5 * (guess[(mu, nu)] + guess[(nu, mu)]);
                }
            }
            let options = Pm7Options {
                initial_density: Some(symmetric),
                ..base.clone()
            };
            if let Ok(result) = run_pm7(&molecule, &params, &options) {
                if result.converged && result.heat_of_formation_kcal < best - 1.0e-6 {
                    best = result.heat_of_formation_kcal;
                    found += 1;
                    println!(
                        "  trial {trial:2}: {:.5} kcal/mol",
                        result.heat_of_formation_kcal
                    );
                }
            }
        }
        println!(
            "  best {best:.5} after 48 perturbed starts ({found} improvement(s)); \
             {:.5} from MOPAC\n",
            best - mopac
        );
    }
}
