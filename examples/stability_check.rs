// SPDX-License-Identifier: GPL-3.0-or-later
//! Does the stability analysis find the instabilities it is supposed to, and leave the rest alone?
//!
//! `cargo run --release --example stability_check`
//!
//! Three claims, and the first is the one that validates the machinery against something already
//! known rather than against this crate's own output.
//!
//! **Stretched H₂ is triplet-unstable, and every quantum chemistry text says so.** At equilibrium
//! the closed-shell solution is the right one; pulled apart, RHF forces the two electrons to share
//! a delocalized orbital and the energy sits above the correct dissociation, with the instability
//! showing up as a negative eigenvalue of the *spin-flip* orbital Hessian. An analysis that misses
//! it is not an analysis. This one finds it, and following it drops the energy.
//!
//! **Ordinary closed shells are left exactly alone.** Water, methane and benzene are minima in both
//! channels, so nothing moves — which is the property that makes the option safe to switch on.
//!
//! **CuCl and AgCl are not instabilities.** Both channels come back positive, which is what put the
//! 5.18 and 3.82 kcal/mol gaps against MOPAC into `docs/fidelity.md` as a value residual rather
//! than a basin problem.

use pm7_rs::math::Vec3;
use pm7_rs::stability::ScfStability;
use pm7_rs::{run_pm7, Atom, Molecule, Pm7Method, Pm7Options, Pm7Parameters};

fn options(stability: ScfStability) -> Pm7Options {
    Pm7Options {
        stability,
        p_tol: 1.0e-10,
        e_tol: 1.0e-11,
        max_scf: 400,
        ..Pm7Options::default()
    }
}

fn main() {
    let params = Pm7Parameters::method(Pm7Method::Pm7).expect("PM7 parameters");
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let at = |z: u8, x: f64, y: f64, z_: f64| Atom {
        z,
        position: Vec3::new(x * a0, y * a0, z_ * a0),
    };
    let diatomic =
        |z: u8, w: u8, r: f64| Molecule::new(vec![at(z, 0.0, 0.0, 0.0), at(w, r, 0.0, 0.0)]);

    let cases: Vec<(String, Molecule)> = vec![
        ("H2 at 0.74 A".into(), diatomic(1, 1, 0.74)),
        ("H2 at 1.50 A".into(), diatomic(1, 1, 1.50)),
        ("H2 at 2.50 A".into(), diatomic(1, 1, 2.50)),
        ("H2 at 4.00 A".into(), diatomic(1, 1, 4.00)),
        ("CuCl".into(), diatomic(29, 17, 2.05)),
        ("AgCl".into(), diatomic(47, 17, 2.28)),
        (
            "water".into(),
            Molecule::new(vec![
                at(8, 0.0, 0.0, 0.0),
                at(1, 0.96, 0.0, 0.0),
                at(1, -0.24, 0.93, 0.0),
            ]),
        ),
        (
            "methane".into(),
            Molecule::new(vec![
                at(6, 0.0, 0.0, 0.0),
                at(1, 0.6293, 0.6293, 0.6293),
                at(1, 0.6293, -0.6293, -0.6293),
                at(1, -0.6293, 0.6293, -0.6293),
                at(1, -0.6293, -0.6293, 0.6293),
            ]),
        ),
    ];

    println!(
        "{:>13}  {:>11}  {:>11}  {:>12}  {:>12}  verdict",
        "case", "singlet eV", "triplet eV", "default", "with Follow"
    );
    for (label, molecule) in cases {
        let plain = run_pm7(&molecule, &params, &options(ScfStability::Off)).expect("scf");
        let followed =
            run_pm7(&molecule, &params, &options(ScfStability::Follow)).expect("scf + stability");
        let s = followed.stability.expect("the analysis ran");
        let moved = plain.heat_of_formation_kcal - followed.heat_of_formation_kcal;
        let verdict = if !s.unstable {
            format!(
                "stable both channels{}",
                if moved.abs() < 1e-9 {
                    ""
                } else {
                    " ** MOVED **"
                }
            )
        } else if moved > 1.0e-6 {
            format!("unstable -> escaped, {moved:.4} kcal/mol lower")
        } else {
            "unstable but nothing lower found".to_string()
        };
        println!(
            "{label:>13}  {:>11.4}  {:>11}  {:>12.5}  {:>12.5}  {verdict}",
            s.lowest_ev,
            s.lowest_triplet_ev
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".into()),
            plain.heat_of_formation_kcal,
            followed.heat_of_formation_kcal,
        );
    }
}
