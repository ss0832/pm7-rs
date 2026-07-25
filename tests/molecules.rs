// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests: PM7 heats of formation / charges for small molecules against
//! published MOPAC PM7 references (baked in as constants).

use pm7_rs::optimizer::{optimize, OptOptions};
use pm7_rs::scf::{run_pm7, Pm7Options};
use pm7_rs::{Molecule, Pm7Parameters, Pm7Method};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().unwrap()
}

#[test]
fn europium_trifluoride_sparkle_matches_mopac() {
    // EuF3: Eu is a Sparkle/PM7 lanthanide (+3 point core, 0 AOs), the F's are F⁻ (fully ionic).
    // The Sparkle has no orbitals, so `build_fock` must not iterate the two-electron J/K over its
    // fictitious integral-block s orbital — doing so aliases the next atom's density and inflates
    // the energy by ~300 eV (ΔHf was +6934). Guards that regression.
    let mol = Molecule::from_xyz_str(
        "4\nEuF3\nEu 0.0 0.0 0.0\nF 2.1 0.0 0.0\nF -1.05 1.818 0.0\nF -1.05 -1.818 0.0\n",
        0.0,
    )
    .unwrap();
    let p = Pm7Parameters::method(Pm7Method::Pm7Sparkle).unwrap();
    let o = Pm7Options {
        method: Pm7Method::Pm7Sparkle,
        ..Pm7Options::default()
    };
    let r = run_pm7(&mol, &p, &o).unwrap();
    assert!(r.converged);
    // MOPAC PM7 SPARKLE ΔHf(EuF3) = −13.02 kcal/mol at this geometry.
    assert!(
        (r.heat_of_formation_kcal + 13.02).abs() < 2.0,
        "EuF3 ΔHf {} kcal/mol (expected ≈ −13; sparkle two-electron regression?)",
        r.heat_of_formation_kcal
    );
    // Fully ionic: Eu³⁺, F⁻.
    assert!(
        (r.charges[0] - 3.0).abs() < 0.05,
        "Eu charge {}",
        r.charges[0]
    );

    // The Sparkle analytic gradient and Hessian must also be correct (the 0-AO atom's +3 core
    // still exerts forces via the electron-core term). Check against finite differences.
    use pm7_rs::gradient::{closed_form_gradient, numerical_gradient};
    use pm7_rs::hessian::{analytic_hessian, numerical_hessian};
    let ga = closed_form_gradient(&mol, &p, &o).unwrap();
    let gn = numerical_gradient(&mol, &p, &o, 1.0e-4).unwrap();
    let mut gmax = 0.0_f64;
    for (a, b) in ga.gradient.iter().zip(&gn.gradient) {
        for k in 0..3 {
            gmax = gmax.max((a.get(k) - b.get(k)).abs());
        }
    }
    assert!(gmax < 1.0e-5, "sparkle gradient vs FD {gmax:.3e}");
    let ha = analytic_hessian(&mol, &p, &o, 1.0e-3).unwrap();
    let hn = numerical_hessian(&mol, &p, &o, 1.0e-3).unwrap();
    let n = 3 * mol.atoms.len();
    let mut hmax = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            hmax = hmax.max((ha[(i, j)] - hn[(i, j)]).abs());
        }
    }
    assert!(hmax < 5.0e-4, "sparkle Hessian vs FD {hmax:.3e}");
}

#[test]
fn gadolinium_trifluoride_uses_zeroed_sparkle_pairs() {
    // Gd (Z=64) is the one lanthanide that carries full PM7 *atomic* alpb/xfac pair
    // entries (Gd-F 64,9 and Gd-Gd 64,64).  MOPAC `switch.F90:472-475` zeroes the
    // entire alpb/xfac rows and columns for the sparkle range once the sparkle
    // parameters are installed, so a Sparkle/PM7 Gd must NOT reuse that diatomic
    // core-core scaling.  With the stray entries left in place GdF3 came out +13
    // kcal/mol too high (a Gd-only outlier); zeroing them (params.rs `install_sparkles`
    // pair retain) puts it on the same footing as every other LnF3.  Guards that
    // regression together with the core-core feathering (see `*_feathering_*` below).
    let mol = Molecule::from_xyz_str(
        "4\nGdF3\nGd 0.0 0.0 0.0\nF 2.1 0.0 0.0\nF -1.05 1.818 0.0\nF -1.05 -1.818 0.0\n",
        0.0,
    )
    .unwrap();
    let p = Pm7Parameters::method(Pm7Method::Pm7Sparkle).unwrap();
    let o = Pm7Options {
        method: Pm7Method::Pm7Sparkle,
        ..Pm7Options::default()
    };
    let r = run_pm7(&mol, &p, &o).unwrap();
    assert!(r.converged);
    // MOPAC PM7 SPARKLE ΔHf(GdF3) = −27.30622 kcal/mol at this geometry; pm7-rs now
    // matches to the MOPAC print floor (~1e-5 kcal).  A tolerance of 0.02 kcal both
    // confirms the bit-level match and catches the ~+13 kcal stray-pair regression.
    assert!(
        (r.heat_of_formation_kcal + 27.30622).abs() < 0.02,
        "GdF3 ΔHf {} kcal/mol (expected −27.30622; stray Gd-F/Gd-Gd pair regression?)",
        r.heat_of_formation_kcal
    );
    assert!(
        (r.charges[0] - 3.0).abs() < 0.05,
        "Gd charge {}",
        r.charges[0]
    );
    // The stray-pair removal must be lanthanide-local: Eu carries no such entry, so
    // EuF3 must stay put at MOPAC's −13.01995.
    let eu = Molecule::from_xyz_str(
        "4\nEuF3\nEu 0.0 0.0 0.0\nF 2.1 0.0 0.0\nF -1.05 1.818 0.0\nF -1.05 -1.818 0.0\n",
        0.0,
    )
    .unwrap();
    let reu = run_pm7(&eu, &p, &o).unwrap();
    assert!(
        (reu.heat_of_formation_kcal + 13.01995).abs() < 0.02,
        "EuF3 ΔHf {} perturbed by the Gd-pair fix (should stay ≈ −13.02)",
        reu.heat_of_formation_kcal
    );
}

#[test]
fn axis_aligned_bond_derivatives_match_numerical() {
    // A bond exactly on the local-frame rotation's singular axis (sp two-electron frame: +x;
    // MNDO/d rotmat/coe: ±z) used to lose the perpendicular gradient/Hessian component (the
    // rotation is direction-discontinuous at the antipode, so its forward-mode derivative
    // collapsed). `rotfix` finite-differences the correct-value f64 integrals for such pairs.
    // Guards that the analytic gradient/Hessian match a clean-step numerical reference for
    // geometries with an on-axis bond — the same accuracy as off-axis.
    use pm7_rs::gradient::{closed_form_gradient, numerical_gradient};
    use pm7_rs::hessian::{analytic_hessian, numerical_hessian};
    let p = params();
    let o = Pm7Options::default();
    // (name, xyz): water with O–H1 exactly on +x (sp path); H2S with an S–H exactly on +z (d path).
    for (name, xyz) in [
        (
            "water+x",
            "3\nw\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        ),
        (
            "h2s+z",
            "3\nh2s\nS 0.0 0.0 0.0\nH 0.0 0.0 1.34\nH 0.0 1.20 -0.60\n",
        ),
    ] {
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let ga = closed_form_gradient(&mol, &p, &o).unwrap();
        let gn = numerical_gradient(&mol, &p, &o, 4.0e-3).unwrap();
        let mut gmax = 0.0f64;
        for (a, b) in ga.gradient.iter().zip(&gn.gradient) {
            for k in 0..3 {
                gmax = gmax.max((a.get(k) - b.get(k)).abs());
            }
        }
        assert!(
            gmax < 5.0e-4,
            "{name} axis-aligned gradient vs numeric {gmax:.2e}"
        );
        let ha = analytic_hessian(&mol, &p, &o, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &p, &o, 3.0e-3).unwrap();
        let n = 3 * mol.atoms.len();
        let mut hmax = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                hmax = hmax.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        assert!(
            hmax < 5.0e-3,
            "{name} axis-aligned Hessian vs numeric {hmax:.2e}"
        );
    }
}

#[test]
fn correction_terms_have_analytic_hessian() {
    // The post-SCF correction Hessian (PM6-DH dispersion + PM7-HH H–H repulsion) is fully
    // analytic (Dual2, per pair) — see `gradient::add_correction_hessian`. Verify it against a
    // full finite-difference Hessian and confirm the correction actually contributes.
    use pm7_rs::hessian::{analytic_hessian, numerical_hessian};
    let dimer = "10\nd\nC 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\nC 0 0 3.6\nH 0.629 0.629 4.229\nH -0.629 -0.629 4.229\nH -0.629 0.629 2.971\nH 0.629 -0.629 2.971\n";
    let mol = Molecule::from_xyz_str(dimer, 0.0).unwrap();
    let n = 3 * mol.atoms.len();
    let max_diff = |a: &pm7_rs::Matrix, b: &pm7_rs::Matrix| {
        let mut m = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                m = m.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        m
    };
    let hess = |v: Pm7Method| {
        let p = Pm7Parameters::method(v).unwrap();
        let o = Pm7Options {
            method: v,
            ..Pm7Options::default()
        };
        analytic_hessian(&mol, &p, &o, 1.0e-3).unwrap()
    };
    // PM7-HH: the analytic Hessian (SCF + dispersion + H–H repulsion) matches a full FD Hessian.
    let o_hh = Pm7Options {
        method: Pm7Method::Pm7Hh,
        ..Pm7Options::default()
    };
    let hh_a = hess(Pm7Method::Pm7Hh);
    let hh_n = numerical_hessian(
        &mol,
        &Pm7Parameters::method(Pm7Method::Pm7Hh).unwrap(),
        &o_hh,
        2.0e-3,
    )
    .unwrap();
    assert!(
        max_diff(&hh_a, &hh_n) < 5.0e-3,
        "PM7-HH analytic Hessian vs numeric {:.2e}",
        max_diff(&hh_a, &hh_n)
    );
    // The correction is non-trivial: PM7-HH and PM7-minus Hessians differ.
    assert!(
        max_diff(&hh_a, &hess(Pm7Method::Pm7Minus)) > 1.0e-2,
        "correction Hessian is missing (PM7-HH == PM7-minus)"
    );
}

#[test]
fn pm7_methods_match_mopac() {
    // Each PM7 family method against MOPAC (`PM7-TS` / `PM7-` / `PM7-HH` keywords, 1SCF PRECISE).
    let ch4 = "5\nm\nC 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n";
    let h2o = "3\nw\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
    let dimer = "10\nd\nC 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\nC 0 0 3.6\nH 0.629 0.629 4.229\nH -0.629 -0.629 4.229\nH -0.629 0.629 2.971\nH 0.629 -0.629 2.971\n";
    for (v, xyz, mopac) in [
        (Pm7Method::Pm7Ts, ch4, -6.85849),
        (Pm7Method::Pm7Minus, h2o, -57.78417),
        (Pm7Method::Pm7Hh, dimer, -7.03415),
    ] {
        let p = Pm7Parameters::method(v).unwrap();
        let o = Pm7Options {
            method: v,
            ..Pm7Options::default()
        };
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let r = run_pm7(&mol, &p, &o).unwrap();
        assert!(
            (r.heat_of_formation_kcal - mopac).abs() < 1.0e-3,
            "{v} ΔHf {} vs MOPAC {mopac}",
            r.heat_of_formation_kcal
        );
    }
}

#[test]
fn scf_bistable_diatomics_select_mopac_basin() {
    // BF/AsF/AlN are ionic/covalent SCF-bistable: the SCF has two aufbau-valid solutions and the
    // initial guess selects the basin. Using MOPAC's exact diagonal guess (`sad_density`,
    // moldat.F90:731) makes pm7-rs converge to the SAME stationary point as MOPAC. Guards that the
    // guess keeps selecting MOPAC's basin. MOPAC PM7 `1SCF PRECISE` at these geometries.
    let p = params();
    for (name, xyz, mopac) in [
        ("BF", "2\nbf\nB 0 0 0\nF 1.263 0 0\n", 6.00850),
        ("AsF", "2\nasf\nAs 0 0 0\nF 1.74 0 0\n", 51.59464),
        ("AlN", "2\naln\nAl 0 0 0\nN 1.79 0 0\n", 179.98769),
    ] {
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let r = run_pm7(&mol, &p, &Pm7Options::default()).unwrap();
        assert!(
            (r.heat_of_formation_kcal - mopac).abs() < 1.0e-3,
            "{name} ΔHf {} vs MOPAC {mopac} (SCF selected the wrong basin — guess regression?)",
            r.heat_of_formation_kcal
        );
    }
}

#[test]
fn pm7_feathering_matches_mopac_bitwise() {
    // PM7 always feathers the two-center NDDO integrals toward the exact point charge
    // as atoms separate (MOPAC `l_feather`, readmo.F90:787).  With it, pm7-rs reproduces
    // MOPAC PM7 ΔHf to the printed 1e-5 kcal/mol across sp, d, and transition-metal
    // species — the residual that was ~0.02 kcal (and up to ~0.6 kcal for some) is gone.
    // References: MOPAC v23.2.5 `PM7 1SCF PRECISE` at these exact geometries.
    let p = Pm7Parameters::standard().unwrap();
    let cases: [(&str, &str, f64); 4] = [
        ("water", "3\nw\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n", -57.78934),
        ("methane", "5\nm\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n", -14.37980),
        ("h2s", "3\nh2s\nS 0.0 0.0 0.0\nH 0.0 0.97 0.94\nH 0.0 -0.97 0.94\n", -3.10254),
        ("so2", "3\nso2\nS 0.0 0.0 0.0\nO 1.43 0.79 0.0\nO -1.43 0.79 0.0\n", -22.93490),
    ];
    for (name, xyz, mopac) in cases {
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let r = run_pm7(&mol, &p, &Pm7Options::default()).unwrap();
        assert!(
            (r.heat_of_formation_kcal - mopac).abs() < 1.0e-3,
            "{name} ΔHf {} vs MOPAC {mopac} (feathering regression?)",
            r.heat_of_formation_kcal
        );
    }
}

#[test]
fn water_single_point_matches_mopac() {
    let mol = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    let r = run_pm7(&mol, &params(), &Pm7Options::default()).unwrap();
    assert!(r.converged);
    // MOPAC PM7 ΔHf(H2O) ≈ −59.24 kcal/mol (near, not at, the minimum here).
    assert!((r.heat_of_formation_kcal + 57.78417).abs() < 0.1);
    assert!(r.dipole_magnitude > 1.0);
}

#[test]
fn water_optimizes_to_pm7_minimum() {
    let mol = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n",
        0.0,
    )
    .unwrap();
    let res = optimize(
        &mol,
        &params(),
        &Pm7Options::default(),
        &OptOptions::default(),
    )
    .unwrap();
    assert!(res.converged);
    assert!((res.scf.heat_of_formation_kcal + 57.8).abs() < 0.5);
}

#[test]
fn d_shell_bromine_runs() {
    // HBr contains Br, a d-orbital PM7 element, exercising the MNDO/d path. The SCF
    // must converge and give a sensible polarization (Br slightly negative).
    let mol = Molecule::from_xyz_str("2\nHBr\nBr 0.0 0.0 0.0\nH 1.414 0.0 0.0\n", 0.0).unwrap();
    let r = run_pm7(&mol, &params(), &Pm7Options::default()).unwrap();
    assert!(r.converged);
    let qsum: f64 = r.charges.iter().sum();
    assert!(qsum.abs() < 1e-6);
    assert!(r.charges[0] < 0.0, "Br charge {}", r.charges[0]);
}

#[test]
fn formaldehyde_carbonyl_polarization() {
    // H2C=O: O should be markedly negative, C positive (carbonyl).
    let mol = Molecule::from_xyz_str(
        "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n",
        0.0,
    )
    .unwrap();
    let r = run_pm7(&mol, &params(), &Pm7Options::default()).unwrap();
    assert!(r.converged);
    assert!(r.charges[1] < -0.2, "carbonyl O charge {}", r.charges[1]);
    assert!(r.charges[0] > 0.1, "carbonyl C charge {}", r.charges[0]);
    let qsum: f64 = r.charges.iter().sum();
    assert!(qsum.abs() < 1e-6);
}

#[test]
fn scandium_fluoride_core_core_uses_poc() {
    // ScF3 (Sc³⁺, d⁰ closed shell). Sc has a PM7 `poc` parameter, so the core–core
    // repulsion MUST use the core Klopman radius po(9) = poc, not rho0. With rho0 the
    // ΔHf is catastrophically wrong (~ −1228 kcal/mol vs MOPAC −242.47); with poc it is
    // within ~0.4 kcal/mol. Guards against a regression of the po(9) core-core fix.
    let mol = Molecule::from_xyz_str(
        "4\nScF3\nSc 0.0 0.0 0.0\nF 1.91 0.0 0.0\nF -0.955 1.654 0.0\nF -0.955 -1.654 0.0\n",
        0.0,
    )
    .unwrap();
    let r = run_pm7(&mol, &params(), &Pm7Options::default()).unwrap();
    assert!(r.converged);
    // MOPAC PM7 ΔHf(ScF3) = −242.47135 kcal/mol at this geometry.
    assert!(
        (r.heat_of_formation_kcal + 242.47135).abs() < 1.0,
        "ScF3 ΔHf {} kcal/mol (expected ≈ −242.47; core-core po(9)/poc regression?)",
        r.heat_of_formation_kcal
    );
}

#[test]
fn water_dimer_hydrogen_bond_matches_mopac() {
    // Cs water dimer. The PM7 dispersion + H-bond correction is bit-exact vs MOPAC:
    // total ΔHf −120.4695 kcal/mol (MOPAC). The SCF residual leaves ~0.011 kcal/mol.
    let mol = Molecule::from_xyz_str(
        "6\nwater dimer\n\
         O -1.551007 -0.114520 0.000000\n\
         H -1.934259  0.762503 0.000000\n\
         H -0.599677  0.040712 0.000000\n\
         O  1.350625  0.111469 0.000000\n\
         H  1.680398 -0.373741 -0.758561\n\
         H  1.680398 -0.373741  0.758561\n",
        0.0,
    )
    .unwrap();
    let r = run_pm7(&mol, &params(), &Pm7Options::default()).unwrap();
    assert!(r.converged);
    assert!(
        (r.heat_of_formation_kcal + 120.4695).abs() < 0.1,
        "water dimer ΔHf {} kcal/mol (expected ≈ −120.47)",
        r.heat_of_formation_kcal
    );
    // The H-bond term itself must be active and attractive (~ −4.6 kcal/mol here).
    let ehb = pm7_rs::hbond::hydrogen_bond_energy(&mol);
    assert!(
        ehb < -1.0,
        "H-bond energy {ehb} kcal/mol (expected clearly negative)"
    );
}

#[test]
fn exchange_cutoff_is_bit_identical_beyond_all_pairs_and_close_when_finite() {
    use pm7_rs::hessian::analytic_hessian;
    // Two water molecules a few Å apart (has short + medium interatomic distances).
    let mol = Molecule::from_xyz_str(
        "6\nwater dimer\n\
         O 0.000000 0.000000 0.000000\n\
         H 0.958400 0.000000 0.000000\n\
         H -0.240000 0.927800 0.000000\n\
         O 2.900000 0.000000 0.000000\n\
         H 3.230000 0.900000 0.000000\n\
         H 3.230000 -0.900000 0.000000\n",
        0.0,
    )
    .unwrap();
    let p = params();
    let base = analytic_hessian(&mol, &p, &Pm7Options::default(), 1.0e-3).unwrap();
    let n = 3 * mol.atoms.len();

    // (a) A cutoff beyond every interatomic distance (switch ≡ 1) is BIT-IDENTICAL to no cutoff.
    let far = Pm7Options {
        exchange_cutoff: Some((100.0, 101.0)),
        ..Pm7Options::default()
    };
    let h_far = analytic_hessian(&mol, &p, &far, 1.0e-3).unwrap();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                base[(i, j)],
                h_far[(i, j)],
                "exchange cutoff beyond all pairs must be bit-identical at ({i},{j})"
            );
        }
    }

    // (b) A finite cutoff drops only long-range exchange → Hessian stays close to exact.
    let finite = Pm7Options {
        exchange_cutoff: Some((6.0, 12.0)),
        ..Pm7Options::default()
    };
    let h_cut = analytic_hessian(&mol, &p, &finite, 1.0e-3).unwrap();
    let mut maxdiff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            maxdiff = maxdiff.max((base[(i, j)] - h_cut[(i, j)]).abs());
        }
    }
    assert!(
        maxdiff < 0.05,
        "finite exchange cutoff perturbs the Hessian by {maxdiff:.4e} eV/Bohr² (expected small)"
    );
}

#[test]
fn hbond_derivatives_are_self_consistent() {
    use pm7_rs::gradient::{analytic_gradient, closed_form_gradient, numerical_gradient};
    use pm7_rs::hessian::{analytic_hessian, numerical_hessian};
    // Asymmetric dimer: the acceptor's two H's sit at clearly different distances from the
    // bridging donor H, so setup_DH_Plus's distance-sorted reference slots are stable under a
    // finite-difference perturbation (a *symmetric* dimer sits exactly on the slot-swap tie,
    // which pollutes the numerical FD reference but not the fixed-topology analytic Hessian).
    let mol = Molecule::from_xyz_str(
        "6\nwater dimer\n\
         O -1.551007 -0.114520 0.000000\n\
         H -1.934259  0.762503 0.000000\n\
         H -0.599677  0.040712 0.000000\n\
         O  1.350625  0.111469 0.000000\n\
         H  1.680398 -0.373741 -0.758561\n\
         H  1.980398 -0.073741  0.858561\n",
        0.0,
    )
    .unwrap();
    let p = params();
    let o = Pm7Options::default();
    // Both AD and fixed-density validation gradients include every post-SCF correction.
    let ga = closed_form_gradient(&mol, &p, &o).unwrap();
    let gv = analytic_gradient(&mol, &p, &o, 1.0e-4).unwrap();
    let gn = numerical_gradient(&mol, &p, &o, 1.0e-4).unwrap();
    let mut gmax = 0.0_f64;
    for (a, b) in ga.gradient.iter().zip(&gn.gradient) {
        for k in 0..3 {
            gmax = gmax.max((a.get(k) - b.get(k)).abs());
        }
    }
    assert!(gmax < 1.0e-4, "H-bond gradient vs FD {gmax:.3e}");
    let mut vmax = 0.0_f64;
    for (a, b) in gv.gradient.iter().zip(&gn.gradient) {
        for k in 0..3 {
            vmax = vmax.max((a.get(k) - b.get(k)).abs());
        }
    }
    assert!(vmax < 1.0e-4, "validation gradient vs FD {vmax:.3e}");
    // Analytic H-bond Hessian vs a full-SCF FD Hessian.
    let ha = analytic_hessian(&mol, &p, &o, 1.0e-3).unwrap();
    let hn = numerical_hessian(&mol, &p, &o, 1.0e-3).unwrap();
    let n = 3 * mol.atoms.len();
    let mut hmax = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            hmax = hmax.max((ha[(i, j)] - hn[(i, j)]).abs());
        }
    }
    assert!(hmax < 5.0e-4, "H-bond Hessian vs FD {hmax:.3e}");
}
