// SPDX-License-Identifier: GPL-3.0-or-later
//! Orbital energies and coefficients.
//!
//! The k-mesh test here is a **regression** test for a real defect. Before v0.2.1 a k-point run
//! reported `mo_energies` taken from `band_energies[0]` — the first *expanded* k point, which on
//! a shifted mesh is not Γ at all — while `mo_coeff` was carried over from the pre-SCF Γ solve,
//! whose Fock was built from the starting density. The two were eigen-data of different matrices,
//! only one of which had converged, and nothing in the API said so.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::scf::OrbitalSource;
use pm7_rs::{
    band_structure, run_pm7, Atom, Cell, KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters,
    ScfReference,
};

fn at(z: u8, x: f64, y: f64, z_: f64) -> Atom {
    Atom {
        z,
        position: Vec3::new(
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z_ * ANGSTROM_TO_BOHR,
        ),
    }
}

fn water() -> Molecule {
    Molecule::new(vec![
        at(8, 0.0, 0.0, 0.0),
        at(1, 0.96, 0.0, 0.0),
        at(1, -0.24, 0.93, 0.0),
    ])
}

fn methyl() -> Molecule {
    Molecule::new(vec![
        at(6, 0.0, 0.0, 0.0),
        at(1, 1.08, 0.0, 0.0),
        at(1, -0.54, 0.94, 0.0),
        at(1, -0.54, -0.94, 0.0),
    ])
}

/// A polar chain, so a shifted mesh genuinely differs from a Γ-centred one.
fn hf_chain() -> Molecule {
    let a = 2.80 * ANGSTROM_TO_BOHR;
    Molecule::new(vec![at(9, 0.0, 0.0, 0.0), at(1, 0.93, 0.0, 0.0)])
        .with_cell(Cell::new(&[Vec3::new(a, 0.0, 0.0)]).unwrap())
}

fn tight() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..Default::default()
    }
}

/// MOPAC v23.2.5, `PM7 1SCF PRECISE`, on the water geometry above.
const MOPAC_WATER_HOMO: f64 = -12.082;
const MOPAC_WATER_LUMO: f64 = 4.025;
/// MOPAC's `ALPHA SOMO LUMO` / `BETA SOMO LUMO` for the methyl radical.
const MOPAC_METHYL_ALPHA: (f64, f64) = (-9.897, 6.053);
const MOPAC_METHYL_BETA: (f64, f64) = (-14.026, 0.613);

#[test]
fn the_frontier_orbitals_match_mopac() {
    let params = Pm7Parameters::standard().unwrap();
    let out = run_pm7(&water(), &params, &tight()).unwrap();
    assert!((out.homo_ev.unwrap() - MOPAC_WATER_HOMO).abs() < 2.0e-3);
    assert!((out.lumo_ev.unwrap() - MOPAC_WATER_LUMO).abs() < 2.0e-3);
    assert_eq!(out.orbital_source, OrbitalSource::Molecular);
}

#[test]
fn both_spin_channels_are_reported_and_match_mopac() {
    let params = Pm7Parameters::standard().unwrap();
    let mut options = tight();
    options.multiplicity = 2;
    options.reference = ScfReference::Unrestricted;
    let out = run_pm7(&methyl(), &params, &options).unwrap();

    assert!((out.homo_ev.unwrap() - MOPAC_METHYL_ALPHA.0).abs() < 2.0e-3);
    assert!((out.lumo_ev.unwrap() - MOPAC_METHYL_ALPHA.1).abs() < 2.0e-3);
    assert!((out.homo_ev_beta.unwrap() - MOPAC_METHYL_BETA.0).abs() < 2.0e-3);
    assert!((out.lumo_ev_beta.unwrap() - MOPAC_METHYL_BETA.1).abs() < 2.0e-3);

    // The physical frontier gap crosses the channels: here the beta LUMO lies below the alpha
    // one, so an alpha-only gap would be the wrong number.
    let gap = out.gap_ev().unwrap();
    let expected = MOPAC_METHYL_BETA.1 - MOPAC_METHYL_ALPHA.0;
    assert!((gap - expected).abs() < 4.0e-3, "gap {gap} vs {expected}");

    assert_eq!(out.occupations().iter().sum::<f64>(), 4.0);
    assert_eq!(out.occupations_beta().unwrap().iter().sum::<f64>(), 3.0);
}

/// The MO coefficients are orthonormal — exactly, because the NDDO working basis is orthonormal
/// and the Fock problem is an ordinary symmetric eigenproblem, not a generalized one.
#[test]
fn the_coefficients_are_orthonormal() {
    let params = Pm7Parameters::standard().unwrap();
    let out = run_pm7(&water(), &params, &tight()).unwrap();
    let c = &out.mo_coeff;
    let n = c.rows;
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|mu| c[(mu, i)] * c[(mu, j)]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1.0e-10,
                "C^T C [{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

/// The density is `2 Σ_occ c c^T`, which is what says the reported coefficients are the ones the
/// reported density was actually built from.
#[test]
fn the_density_is_rebuilt_from_the_coefficients() {
    let params = Pm7Parameters::standard().unwrap();
    let out = run_pm7(&water(), &params, &tight()).unwrap();
    let (c, n) = (&out.mo_coeff, out.mo_coeff.rows);
    for mu in 0..n {
        for nu in 0..n {
            let rebuilt: f64 = (0..out.n_occ).map(|i| 2.0 * c[(mu, i)] * c[(nu, i)]).sum();
            assert!(
                (rebuilt - out.density[(mu, nu)]).abs() < 1.0e-9,
                "P[{mu},{nu}]: {rebuilt} vs {}",
                out.density[(mu, nu)]
            );
        }
    }
}

/// **The regression test.** On a k mesh the reported orbitals must be a genuine eigenpair of the
/// converged Hamiltonian at Γ — which is exactly what `band_structure` evaluated at Γ returns.
///
/// The mesh is deliberately **shifted**, because that is the case the old code got wrong in two
/// ways at once: `band_energies[0]` was the first expanded k point rather than Γ, and the
/// coefficients came from a Fock built out of the starting density.
#[test]
fn kmesh_orbitals_are_a_real_eigenpair_at_gamma() {
    let params = Pm7Parameters::standard().unwrap();
    for (label, kmesh) in [
        ("gamma-centred", KMesh::grid(4, 1, 1)),
        (
            "half-shifted",
            KMesh::MonkhorstPack {
                n: [4, 1, 1],
                shift: [0.5, 0.0, 0.0],
                gamma_centred: true,
            },
        ),
        (
            "original Monkhorst-Pack",
            KMesh::MonkhorstPack {
                n: [4, 1, 1],
                shift: [0.0; 3],
                gamma_centred: false,
            },
        ),
    ] {
        let mut options = tight();
        options.pbc = Some(PbcOptions {
            kmesh: kmesh.clone(),
            ..Default::default()
        });
        let out = run_pm7(&hf_chain(), &params, &options).unwrap();
        assert_eq!(out.orbital_source, OrbitalSource::KMeshGamma, "{label}");

        let bands = band_structure(&hf_chain(), &params, &options, &[[0.0, 0.0, 0.0]]).unwrap();
        let reference = &bands.energies[0];
        assert_eq!(out.mo_energies.len(), reference.len(), "{label}");
        for (i, (ours, theirs)) in out.mo_energies.iter().zip(reference).enumerate() {
            assert!(
                (ours - theirs).abs() < 1.0e-8,
                "{label} band {i}: reported {ours} but the converged Fock at Gamma gives {theirs}"
            );
        }

        // And the coefficients belong to those energies: orthonormal, and the right count.
        let c = &out.mo_coeff;
        assert_eq!(c.rows, out.mo_energies.len(), "{label}");
        for i in 0..c.rows {
            let norm: f64 = (0..c.rows).map(|mu| c[(mu, i)] * c[(mu, i)]).sum();
            assert!(
                (norm - 1.0).abs() < 1.0e-10,
                "{label} column {i} norm {norm}"
            );
        }
    }
}

/// A k-point shift the machinery cannot represent is refused, and the message says which shifts
/// work.
///
/// The mesh `{(i+s)/n}` is closed under `k → −k` only when `2s` is an integer. Anything else made
/// the SCF stall at a small non-zero residual — a failure mode that reads as a convergence
/// problem and is really an unrepresentable request.
#[test]
fn an_unsupported_kpoint_shift_is_refused_rather_than_stalling() {
    let params = Pm7Parameters::standard().unwrap();
    for bad in [0.25, 0.125, 0.3] {
        let mut options = tight();
        options.pbc = Some(PbcOptions {
            kmesh: KMesh::MonkhorstPack {
                n: [4, 1, 1],
                shift: [bad, 0.0, 0.0],
                gamma_centred: true,
            },
            ..Default::default()
        });
        let message = run_pm7(&hf_chain(), &params, &options)
            .unwrap_err()
            .to_string();
        assert!(message.contains("0.5"), "{bad}: {message}");
        assert!(message.contains("k -> -k"), "{bad}: {message}");
    }
    // The two supported shifts still work.
    for good in [0.0, 0.5] {
        let mut options = tight();
        options.pbc = Some(PbcOptions {
            kmesh: KMesh::MonkhorstPack {
                n: [4, 1, 1],
                shift: [good, 0.0, 0.0],
                gamma_centred: true,
            },
            ..Default::default()
        });
        assert!(run_pm7(&hf_chain(), &params, &options).is_ok(), "{good}");
    }
}

/// A Γ-only periodic run says so, and is not confused with a mesh run.
#[test]
fn a_gamma_only_run_is_labelled_gamma() {
    let params = Pm7Parameters::standard().unwrap();
    let mut options = tight();
    options.pbc = Some(PbcOptions {
        kmesh: KMesh::Gamma,
        ..Default::default()
    });
    let out = run_pm7(&hf_chain(), &params, &options).unwrap();
    assert_eq!(out.orbital_source, OrbitalSource::Gamma);
}
