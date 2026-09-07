// SPDX-License-Identifier: GPL-3.0-or-later
//! How the EH+ derivatives behave as each of its angles approaches a singular configuration.
//! Reports; asserts nothing.
//!
//! `cargo test --release --test hbond_singularity -- --nocapture --ignored`
//!
//! Two different angles can go singular and they are not the same problem:
//!
//! * **D–H···A → 180°** is the *ideal* hydrogen bond. `cos θ` there is a smooth maximum, so the
//!   derivative is a well-behaved zero and nothing should degrade.
//! * **R–X···H → 180°** enters the correction as `cos(shift − θ)` with `shift = 109.48°`, and
//!   `sin(shift) ≠ 0` means the answer depends on `sin θ`, which has a genuine `|·|` corner. The
//!   derivative is bounded but direction-discontinuous — a property of the model. The question a
//!   measurement can settle is whether the *implementation* adds numerical damage on top of it.

use pm7_rs::math::Vec3;
use pm7_rs::Molecule;

fn build(atoms: &[(u8, Vec3)]) -> Molecule {
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    Molecule::new(
        atoms
            .iter()
            .map(|(z, p)| pm7_rs::Atom {
                z: *z,
                position: *p * a,
            })
            .collect(),
    )
}

/// `tilt` bends the donor O–H off the O···O axis: at zero, D–H···A is exactly 180°.
fn straight_donor(tilt: f64) -> Molecule {
    let (oo, oh) = (2.86, 0.96);
    build(&[
        (8, Vec3::new(0.0, 0.0, 0.0)),
        (1, Vec3::new(oh * tilt.cos(), oh * tilt.sin(), 0.0)),
        (1, Vec3::new(-0.24, -0.93, 0.0)),
        (8, Vec3::new(oo, 0.0, 0.0)),
        (1, Vec3::new(oo + 0.34, 0.90, 0.0)),
        (1, Vec3::new(oo + 0.34, -0.45, 0.78)),
    ])
}

/// `tilt` rotates one acceptor O–H: at zero it is exactly anti-parallel to O···H, so the
/// acceptor-side `R–X···H` angle is 180°.
fn straight_acceptor(tilt: f64) -> Molecule {
    let (oo, oh) = (2.86, 0.96);
    build(&[
        (8, Vec3::new(0.0, 0.0, 0.0)),
        (1, Vec3::new(oh, 0.0, 0.0)),
        (1, Vec3::new(-0.24, -0.93, 0.0)),
        (8, Vec3::new(oo, 0.0, 0.0)),
        (1, Vec3::new(oo + oh * tilt.cos(), oh * tilt.sin(), 0.0)),
        (1, Vec3::new(oo - 0.30, 0.0, 0.91)),
    ])
}

fn sweep(label: &str, geometry: impl Fn(f64) -> Molecule) {
    println!("\n=== {label} ===");
    println!(" tilt   E (kcal)      worst |analytic - FD|   relative   FD scale");
    for exponent in 1..=10 {
        let tilt = 10f64.powi(-exponent);
        let molecule = geometry(tilt);
        let energy = pm7_rs::hbond::hydrogen_bond_energy(&molecule);
        let analytic = pm7_rs::hbond::hydrogen_bond_gradient(&molecule);
        let h = 1.0e-5 * pm7_rs::constants::ANGSTROM_TO_BOHR;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut loudest = (0usize, 0usize);
        #[allow(clippy::needless_range_loop)] // the index names the atom in the report below
        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let shifted = |s: f64| {
                    let mut m = molecule.clone();
                    match axis {
                        0 => m.atoms[atom].position.x += s * h,
                        1 => m.atoms[atom].position.y += s * h,
                        _ => m.atoms[atom].position.z += s * h,
                    }
                    pm7_rs::hbond::hydrogen_bond_energy(&m)
                };
                let fd = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
                let mine = analytic[atom].get(axis);
                if fd.abs() > scale {
                    scale = fd.abs();
                    loudest = (atom, axis);
                }
                worst = worst.max((mine - fd).abs());
            }
        }
        println!(
            "1e-{exponent:<2} {energy:12.6}  {worst:20.6e}  {:9.2e}  {scale:9.3e}   atom {} {}",
            worst / scale.max(1.0e-30),
            loudest.0,
            ["x", "y", "z"][loudest.1]
        );
    }
}

#[test]
#[ignore = "diagnostic"]
fn eh_plus_derivative_error_near_each_singular_angle() {
    sweep(
        "D-H...A straight (cos theta: smooth maximum)",
        straight_donor,
    );
    sweep(
        "R-X...H straight (sin theta: genuine corner)",
        straight_acceptor,
    );
}
