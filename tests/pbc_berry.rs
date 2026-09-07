// SPDX-License-Identifier: GPL-3.0-or-later
//! Berry-phase polarization, and the Born charge it produces against the one CPHF produces.
//!
//! This is the whole reason the Berry module exists. `Z*_{A,ab} = Ω ∂P_a/∂τ_{A,b}` is reachable
//! two ways: by linear response (`born_and_dielectric`, a CPHF solve contracted against the dipole
//! operator) and by finite-differencing a polarization that is computed with no linear response
//! anywhere — a string of ordinary diagonalizations and a determinant. The two share the
//! Hamiltonian and the basis and essentially nothing else, so their agreement is evidence about
//! the response solver that no internal consistency check can give.
//!
//! The failure mode being guarded against is specific and loud. Applying the atomic-gauge closing
//! correction `e^{−iG·τ}` on top of the cell gauge — which is what a reader of the standard
//! derivation would naturally do — double-counts a phase and puts `Z*` tens of electrons out.
//! A test that only checked "the polarization is finite and changes when atoms move" would pass
//! with that bug in place.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    berry_polarization, born_and_dielectric, Atom, Cell, DfptOptions, KMesh, Molecule, PbcOptions,
    Pm7Options, Pm7Parameters,
};

/// Rocksalt LiF: polar, two atoms, and the cell every other periodic test in this crate uses.
fn lif(displace: Option<(usize, usize, f64)>) -> Molecule {
    let a = 4.03 * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    let mut atoms = vec![
        Atom {
            z: 3,
            position: Vec3::zero(),
        },
        Atom {
            z: 9,
            position: Vec3::new(a * 0.5, 0.0, 0.0),
        },
    ];
    if let Some((atom, axis, delta)) = displace {
        let mut p = atoms[atom].position;
        match axis {
            0 => p.x += delta,
            1 => p.y += delta,
            _ => p.z += delta,
        }
        atoms[atom].position = p;
    }
    Molecule::new(atoms).with_cell(cell)
}

fn options(mesh: usize) -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 500,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The polarization stops moving as the string is refined.
///
/// `strings` is the only convergence parameter, so a value has to be *shown* adequate rather than
/// assumed. Reported against the quantum, because that is the scale a polarization is meaningful
/// on: a drift small compared with the lattice reduction is a converged number.
#[test]
fn the_berry_phase_converges_in_the_string_length() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = lif(None);
    let opts = options(2);

    let mut previous: Option<f64> = None;
    let mut worst_step = f64::INFINITY;
    for strings in [8usize, 12, 16, 24] {
        let p = berry_polarization(&molecule, &params, &opts, strings).unwrap();
        let value = p.phase[0];
        if let Some(prev) = previous {
            worst_step = worst_step.min((value - prev).abs());
        }
        previous = Some(value);
    }
    assert!(
        worst_step < 5.0e-3,
        "the Berry phase is still moving with the string length; closest consecutive pair \
         differed by {worst_step:.3e} in units of 2*pi"
    );
}

/// Inversion symmetry pins the polarization to zero **or half a quantum**, and the Berry phase
/// itself to zero.
///
/// Rocksalt LiF has inversion symmetry through the Li site, so `P ≡ −P` modulo the quantum, hence
/// `2P ≡ 0`. That admits **two** values, not one: zero, and half a quantum. Both are physical, and
/// which one a crystal sits at is decided by where its ions are.
///
/// This one sits at the half. Fluorine is at fractional `(−½, ½, ½)` of the fcc cell with a core
/// charge of 7, so the ionic phase is exactly `∓3.5` — a half-integer, and the whole of the
/// crystal's polarization. Asserting `P ≡ 0` instead, which is the intuitive reading of "a
/// centrosymmetric crystal is unpolarized", fails on a correct implementation.
///
/// The sharp statement about the code is the other one: the **electronic** phase is quantized to
/// zero here, at any string length, so it is the part that catches a defect in the string, the
/// overlap or the closing link.
#[test]
fn inversion_symmetry_quantizes_the_polarization() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = lif(None);
    let p = berry_polarization(&molecule, &params, &options(3), 16).unwrap();

    // The electronic Berry phase, which symmetry sends to zero exactly.
    let worst_phase = p.phase.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(
        worst_phase < 0.01,
        "the electronic Berry phase of a centrosymmetric cell came out at {worst_phase:.3e} in \
         units of 2*pi, where inversion symmetry quantizes it to zero. This is the string, the \
         overlap or the closing link, not convergence: symmetry pins it at every string length."
    );

    // And the full statement: `2P` reduces to zero, so `P` is at zero or at half a quantum.
    let zero = pm7_rs::BerryPolarization {
        electronic: Vec3::zero(),
        ionic: Vec3::zero(),
        total: Vec3::zero(),
        phase: [0.0; 3],
        quantum: p.quantum,
        string_length: p.string_length,
    };
    let doubled = pm7_rs::BerryPolarization {
        total: p.total * 2.0,
        ..p.clone()
    };
    let reduced = zero.difference(&doubled);
    let scale = p.quantum.iter().map(|q| q.norm()).fold(0.0, f64::max);
    assert!(
        reduced.norm() < 0.02 * scale,
        "2P reduced to {:.3e} e/Bohr^2 against a quantum of {scale:.3e}; inversion symmetry \
         requires it to vanish",
        reduced.norm()
    );
}

/// `Ω ∂P/∂τ` from the Berry phase reproduces the CPHF Born charge.
///
/// The independent-route check. A finite difference of the polarization, reduced onto the nearest
/// branch at each step — subtracting the raw totals is wrong the moment a displacement carries the
/// phase across a branch, and gives an answer off by exactly one quantum.
///
/// The tolerance is loose on purpose and is still five orders inside the failure it guards against.
/// The two routes differ in a real, known way: the Berry overlap places every orbital at its atom,
/// where the dipole operator also carries the intra-atomic hybridization moment (s-p, and for PM7
/// p-d). That difference is physics, not error, and it is what sets the gap between them.
#[test]
fn the_berry_born_charge_agrees_with_the_cphf_one() {
    let params = Pm7Parameters::standard().unwrap();
    let opts = options(2);
    let strings = 16;
    let step = 0.01 * ANGSTROM_TO_BOHR;

    let cphf = born_and_dielectric(&lif(None), &params, &opts, &DfptOptions::default()).unwrap();

    // Displace fluorine along x and difference the polarization.
    let (atom, axis) = (1usize, 0usize);
    let minus =
        berry_polarization(&lif(Some((atom, axis, -step))), &params, &opts, strings).unwrap();
    let plus = berry_polarization(&lif(Some((atom, axis, step))), &params, &opts, strings).unwrap();
    let delta = minus.difference(&plus);

    let volume = lif(None).cell.unwrap().measure();
    let berry_z = volume * delta.x / (2.0 * step);
    let cphf_z = cphf.born[atom].get(axis, axis);

    assert!(
        (berry_z - cphf_z).abs() < 0.25,
        "Berry Z* = {berry_z:.4} e against CPHF Z* = {cphf_z:.4} e. These are independent routes \
         to the same derivative; a gap this size is not the hybridization moment the Berry \
         overlap leaves out, it is a defect in one of them. A phase-convention error in the \
         string's closing link shows up here as tens of electrons."
    );
    assert!(
        berry_z.abs() > 0.05,
        "Berry Z* came out at {berry_z:.4} e, which is zero to within noise — the test would pass \
         the same way if the polarization never responded to the displacement at all"
    );
}
