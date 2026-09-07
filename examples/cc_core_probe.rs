// SPDX-License-Identifier: GPL-3.0-or-later
//! The C–C core–core repulsion, term by term, against MOPAC's `ccrep`.
//!
//! `cargo run --release --example cc_core_probe`
//!
//! The 100-case oracle found `pm7-rs` and MOPAC disagreeing on the heat of formation of every
//! molecule with a short C–C bond — acetylene by 12.00000 kcal/mol, allene by 1.87, furan by 1.02 —
//! while agreeing on **every orbital energy to five decimals and every Mulliken charge to six**.
//! That combination localizes it precisely: the SCF is identical, so the difference is in a term
//! added after it, and the only such term that depends on geometry is the core–core repulsion.
//!
//! A bare `C2` reproduces the discrepancy exactly (identical deltas to the acetylene scan), which
//! narrows it to one atom pair and one distance. This prints what `pair_core_energy` returns
//! against the pieces MOPAC's `ccrep.F90` assembles, so the disagreeing term names itself.

use pm7_rs::math::Vec3;
use pm7_rs::{Pm7Method, Pm7Parameters};

fn main() {
    let params = Pm7Parameters::method(Pm7Method::Pm7).expect("PM7 parameters");
    let a0 = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let ev_per_kcal = 1.0 / 23.060547830619;

    // MOPAC's own assembly, transcribed from `ccrep.F90` for a defined PM7 pair.
    let pair = params.pair(6, 6);
    let element = params.element(6).expect("carbon");
    let mopac_core = |r_ang: f64| -> f64 {
        let alpb = if pair.alpb < 1.0e-6 { 1.2 } else { pair.alpb };
        // `scale = 1 + 2 fff exp(-abond (r + 0.0003 r^6))`, then the (6,6) special case.
        let mut scale = 1.0 + 2.0 * pair.xfac * (-alpb * (r_ang + 0.0003 * r_ang.powi(6))).exp();
        scale += params.vpar(1) * (-params.vpar(2) * r_ang).exp();
        scale
    };

    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>12}",
        "r (A)", "pm7-rs (eV)", "scale(MOPAC)", "implied scale", "ratio"
    );
    for r_ang in [1.10, 1.20, 1.25, 1.30, 1.32, 1.34, 1.40, 1.55] {
        let r = r_ang * a0;
        let ours = pm7_rs::repulsion::pair_core_energy(
            6,
            6,
            Vec3::zero(),
            Vec3::new(r, 0.0, 0.0),
            &params,
        )
        .expect("carbon core-core");
        // The bare monopole, so the printed scale is comparable with MOPAC's.
        let zz = element.core_charge * element.core_charge;
        let rho = element.po9() * 2.0;
        let gab = pm7_rs::constants::HARTREE_TO_EV / (r * r + rho * rho).sqrt();
        let enuc = gab * zz;
        let implied = ours / enuc;
        println!(
            "{r_ang:6.2}  {ours:14.8}  {:14.8}  {implied:14.8}  {:12.8}",
            mopac_core(r_ang),
            implied / mopac_core(r_ang),
        );
    }
    println!(
        "\n(the 12 kcal/mol the oracle sees is {:.8} eV)",
        12.0 * ev_per_kcal
    );
}
