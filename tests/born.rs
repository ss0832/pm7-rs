// SPDX-License-Identifier: GPL-3.0-or-later
//! Born effective charges, the electronic dielectric tensor, and LO–TO splitting.
//!
//! A homogeneous electric field is not a periodic operator, so this whole path goes through the
//! commutator `[H, r]` (convention C-3). Nothing about that is self-evidently right, and the
//! quantities it produces have no MOPAC reference — MOPAC has no periodic `FIELD=`. So the checks
//! here are **identities and independent references**, not tolerances against a table:
//!
//! 1. the **acoustic sum rule** `Σ_A Z*_A = q_tot`, which is what catches a missing *explicit*
//!    term in `Z*` — the `q_A δ_ab` piece the two contraction orders agree on and therefore
//!    cannot detect between them;
//! 2. a **finite field along a non-periodic direction**, where `−f·r` is bounded and the ordinary
//!    external-field machinery applies. That route shares no code with the commutator, so
//!    agreement is meaningful;
//! 3. a **commensurate supercell**, since `Z*` per atom and `ε^∞` are intensive;
//! 4. an **external reference**: LiF, where a nearly complete charge transfer puts `Z*` at `±1`.
//!
//! Note what item 2 has to be applied to. A finite field validates `d/df d/dR` (the Born charge)
//! and `d/df d/df` (the dielectric) **separately**, because they use the field response on one
//! side and on both. Only the second catches an error in the field's *bare* term, and that is
//! exactly the bug that survived every check in items 1, 3 and 4 while `eps^inf` came back as
//! the identity.
//!
//! And on tolerances: the polarizability is checked against a **first** difference of the dipole,
//! not a second difference of the energy. At a field small enough to stay linear the latter is a
//! few times `10^-9` eV out of `-479`, which is SCF noise, and it produced a "reference" that
//! moved by a factor of three when the density tolerance changed.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::dfpt::{born_and_dielectric, DfptOptions};
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{
    closed_form_gradient, Atom, Cell, ExternalField, Molecule, Pm7Options, Pm7Parameters,
};

fn params() -> Pm7Parameters {
    Pm7Parameters::method("pm7-".parse().unwrap()).unwrap()
}

fn options(mesh: KMesh) -> Pm7Options {
    Pm7Options {
        method: "pm7-".parse().unwrap(),
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 800,
        pbc: Some(PbcOptions {
            kmesh: mesh,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// A polar 1-D chain: strongly ionic, so `Z*` is unambiguous, and it has two free directions in
/// which a finite field is legitimate.
fn hf_chain() -> Molecule {
    let a = ANGSTROM_TO_BOHR;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.93 * a, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(2.80 * a, 0.0, 0.0)]).unwrap())
}

fn diamond() -> Molecule {
    let a = 3.567 * ANGSTROM_TO_BOHR;
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(
        Cell::new(&[
            Vec3::new(0.0, a * 0.5, a * 0.5),
            Vec3::new(a * 0.5, 0.0, a * 0.5),
            Vec3::new(a * 0.5, a * 0.5, 0.0),
        ])
        .unwrap(),
    )
}

/// **The load-bearing test.** A neutral cell's Born charges must sum to zero.
///
/// This is what would catch using the *core* charge `Z_A` in the explicit term instead of the net
/// Mulliken charge `q_A = Z_A − n_A`. With `Z_A` a two-atom ionic cell would report `Z* = +Z` on
/// **both** sites and the sum would be wildly non-zero — and neither the "both contraction orders
/// agree" check nor a convergence check would notice, because the wrong term is common to both.
#[test]
fn the_born_charges_obey_the_acoustic_sum_rule() {
    let out = born_and_dielectric(
        &hf_chain(),
        &params(),
        &options(KMesh::grid(4, 1, 1)),
        &DfptOptions::default(),
    )
    .expect("Born charges");
    assert!(out.converged, "the field response did not converge");
    assert!(
        out.acoustic_residual() < 1.0e-8,
        "acoustic sum rule residual {:.3e}",
        out.acoustic_residual()
    );

    // Two atoms, one ionic bond: the charges must be equal and opposite along the chain, and
    // large enough that a sign error would be obvious.
    let (f, h) = (&out.born[0], &out.born[1]);
    assert!(
        (f.get(0, 0) + h.get(0, 0)).abs() < 1.0e-8,
        "Z*_F = {} and Z*_H = {} should be opposite",
        f.get(0, 0),
        h.get(0, 0)
    );
    assert!(
        f.get(0, 0).abs() > 0.2,
        "an HF chain should carry a substantial Born charge, got {}",
        f.get(0, 0)
    );
    // Fluorine is the anion.
    assert!(
        f.get(0, 0) < 0.0,
        "Z*_F should be negative: {}",
        f.get(0, 0)
    );
}

/// The **transverse** Born charge against a finite field.
///
/// A field across the chain is bounded — there is no lattice translation in `y` — so the ordinary
/// `ExternalField` path applies and gives an independent value of `∂²E/∂f_y ∂R_{A,y}` by finite
/// differences of the force. The two routes share no code: one goes through the commutator and
/// the response, the other through a real term in the core Hamiltonian.
#[test]
fn a_transverse_born_charge_matches_a_finite_field() {
    let molecule = hf_chain();
    let base = options(KMesh::grid(4, 1, 1));
    let out = born_and_dielectric(&molecule, &params(), &base, &DfptOptions::default())
        .expect("Born charges");

    let h = 2.0e-3; // V/Angstrom, across the chain
    let mut plus = base.clone();
    let mut minus = base.clone();
    plus.field = Some(ExternalField::new(0.0, h, 0.0));
    minus.field = Some(ExternalField::new(0.0, -h, 0.0));
    let gp = closed_form_gradient(&molecule, &params(), &plus).expect("gradient");
    let gm = closed_form_gradient(&molecule, &params(), &minus).expect("gradient");

    for atom in 0..molecule.atoms.len() {
        // d(dE/dR_y)/df_y in eV/Bohr per (V/Angstrom); Z* is in units of e, and one e couples to
        // a V/Angstrom field over a Bohr as `a0` eV.
        let cross = (gp.gradient[atom].y - gm.gradient[atom].y) / (2.0 * h);
        let expected = out.born[atom].get(1, 1) * pm7_rs::constants::PM7_A0;
        assert!(
            (cross - expected).abs() < 5.0e-4,
            "atom {atom}: finite-field cross derivative {cross} vs Z*_yy {expected}"
        );
    }
}

/// The **field–field** response against a finite field, across the chain.
///
/// The companion to the transverse Born-charge test, and just as necessary: that one validates
/// `d/df d/dR`, this one validates `d/df d/df`, and they exercise different halves of the solver.
/// Without it `eps^inf` has no independent reference anywhere — MOPAC has no periodic `FIELD=`,
/// and a 3-D cell has no free direction to put a finite field along.
///
/// Compared against the **raw** `d(mu)/d(f)`, not `eps`: a 1-D cell has no volume, so `eps` is not
/// defined for it, and for a slab the out-of-plane dielectric constant is depolarization-dependent
/// and is not what this is testing. `eps` is this tensor times `-4 pi / Omega` (convention C-6),
/// so validating the tensor validates `eps` up to a documented conversion.
#[test]
fn the_transverse_polarizability_matches_a_finite_field() {
    let molecule = hf_chain();
    let base = options(KMesh::grid(4, 1, 1));
    let out = born_and_dielectric(&molecule, &params(), &base, &DfptOptions::default())
        .expect("field response");

    // A **first** difference of the dipole, not a second difference of the energy.
    //
    // Both measure `dmu/df`, but the energy route is hopeless here: at a field small enough to
    // stay linear, `E(h) - 2E(0) + E(-h)` is a few times `10^-9` eV out of a total near `-479`,
    // which is the size of the SCF's own convergence noise. It duly produced a "reference" that
    // moved by a factor of three when the density tolerance changed. `mu` is computed directly
    // rather than as a difference of large numbers, so its central difference is exact to the
    // step's own truncation error and stable over five orders of magnitude of step.
    let h = 1.0e-2;
    let mu_y_e_bohr = |fy: f64| {
        let mut o = base.clone();
        o.field = Some(ExternalField::new(0.0, fy, 0.0));
        pm7_rs::run_pm7(&molecule, &params(), &o)
            .expect("scf")
            .dipole
            .total()
            .y
            / pm7_rs::constants::AU_DIPOLE_TO_DEBYE
    };
    // `f_internal = F[V/Angstrom] * a0`, so the derivative in internal units divides by that.
    let found = (mu_y_e_bohr(h) - mu_y_e_bohr(-h)) / (2.0 * h * pm7_rs::constants::PM7_A0);
    let expected = out.polarizability.get(1, 1);
    assert!(
        (found - expected).abs() < 1.0e-6 * expected.abs(),
        "finite-field dmu_y/df_y = {found:.12} but DFPT gives {expected:.12}"
    );
    // Not two zeros agreeing: a real system polarizes. Under convention C-1 `E = E_0 + f.mu`, so
    // second-order perturbation theory lowering the energy makes this derivative negative.
    assert!(
        expected < -1.0e-6,
        "a transverse polarizability should be substantial and negative under C-1, got {expected:.3e}"
    );
}

/// `ε^∞` for a cubic crystal: symmetric, isotropic, and **strictly** above 1.
///
/// The `> 1` is the part that earns its keep. An earlier version of this test asked only for
/// `>= 1` and for symmetry and isotropy — all three of which `eps = 1` satisfies perfectly. It
/// passed while the field response was identically zero, which is exactly the failure it existed
/// to catch. A semiconductor polarizes; a test that cannot tell that from a solver returning
/// nothing is not a test.
///
/// An **odd** mesh: a Γ-centred even mesh does not sample the cubic star symmetrically, so it
/// leaves an off-diagonal of a few times `10^-3` that has nothing to do with the physics and
/// shrinks with refinement (`0.21` at 2³, `3.6e-3` at 3³, `8e-5` at 5³).
#[test]
fn the_dielectric_tensor_is_physical() {
    let out = born_and_dielectric(
        &diamond(),
        &params(),
        &options(KMesh::grid(3, 3, 3)),
        &DfptOptions::default(),
    )
    .expect("dielectric tensor");
    assert!(out.converged);
    let e = &out.dielectric;
    for a in 0..3 {
        for b in 0..3 {
            assert!(
                (e.get(a, b) - e.get(b, a)).abs() < 1.0e-8,
                "eps should be symmetric: ({a},{b}) = {} vs {}",
                e.get(a, b),
                e.get(b, a)
            );
        }
        assert!(
            e.get(a, a) > 1.0 + 1.0e-3,
            "diamond is a polarizable semiconductor, so eps_{a}{a} must exceed 1, got {}",
            e.get(a, a)
        );
    }
    // Cubic: isotropic diagonal, and off-diagonals small next to the response itself.
    assert!(
        (e.get(0, 0) - e.get(1, 1)).abs() < 1.0e-6 && (e.get(1, 1) - e.get(2, 2)).abs() < 1.0e-6,
        "a cubic crystal has an isotropic eps: {:?}",
        (e.get(0, 0), e.get(1, 1), e.get(2, 2))
    );
    let response = e.get(0, 0) - 1.0;
    assert!(
        e.get(0, 1).abs() < 0.05 * response,
        "off-diagonal {} is not small next to the response {response}",
        e.get(0, 1)
    );
    // Diamond is non-polar: `Z*` vanishes by site symmetry. Worth pinning next to a *non-zero*
    // dielectric response, because it says the zero is symmetry and not a dead solver.
    assert!(
        out.born.iter().all(|z| z.get(0, 0).abs() < 1.0e-6),
        "diamond is non-polar; Z* should vanish, got {}",
        out.born[0].get(0, 0)
    );
}

/// A fully ionic crystal has `Z* = ±1`, and LiF is the textbook case.
///
/// The one place in this file with an **external** reference rather than an internal identity:
/// LiF's measured Born charge is close to `+1.04` on Li, because the bond is ionic enough that a
/// displaced Li carries very nearly its whole formal charge. Getting `1.03` is a real statement
/// that the magnitude — not just the sum rule, not just the sign — is right.
///
/// `eps^inf` is *not* asserted against experiment: PM7's minimal valence basis has no polarization
/// functions, so it recovers about `1.01` against a measured `1.92`, and diamond about `1.12`
/// against `5.7`. That is a property of the model, recorded in `docs/fidelity.md`, not something
/// this code can fix — so the test pins the ratio it can defend and documents the one it cannot.
#[test]
fn an_ionic_crystal_has_unit_born_charges() {
    let a = 4.03 * ANGSTROM_TO_BOHR;
    let lif = Molecule::new(vec![
        Atom {
            z: 3,
            position: Vec3::zero(),
        },
        Atom {
            z: 9,
            position: Vec3::new(a * 0.5, 0.0, 0.0),
        },
    ])
    .with_cell(
        Cell::new(&[
            Vec3::new(0.0, a * 0.5, a * 0.5),
            Vec3::new(a * 0.5, 0.0, a * 0.5),
            Vec3::new(a * 0.5, a * 0.5, 0.0),
        ])
        .unwrap(),
    );
    let out = born_and_dielectric(
        &lif,
        &params(),
        &options(KMesh::grid(3, 3, 3)),
        &DfptOptions::default(),
    )
    .expect("LiF");
    assert!(out.converged);
    let li = out.born[0].get(0, 0);
    let f = out.born[1].get(0, 0);
    assert!(
        (li - 1.03).abs() < 0.10,
        "Li in LiF should carry a Born charge near +1, got {li}"
    );
    assert!(
        (li + f).abs() < 1.0e-8,
        "Z*(Li) = {li} and Z*(F) = {f} must be equal and opposite in a two-atom neutral cell"
    );
    assert!(out.acoustic_residual() < 1.0e-8);
    // Cubic and isotropic, so a single number describes it.
    for axis in 1..3 {
        assert!(
            (out.born[0].get(axis, axis) - li).abs() < 1.0e-6,
            "rocksalt Z* is isotropic: {} vs {li}",
            out.born[0].get(axis, axis)
        );
    }
}

/// `Z*` per atom is intensive: a doubled cell with a halved mesh gives the same charges.
///
/// `prim 2n` and `dbl n` sample exactly the same set of primitive k points — the SCF energies per
/// formula unit agree to `10^-9`, which is how that is known here rather than assumed. So the
/// Born charges must agree too, and this is the only independent reference a 3-D cell can have:
/// a finite field needs a free direction and 3-D has none.
///
/// **Measured caveat, deliberately encoded rather than hidden.** The agreement is exact to about
/// `10^-6` for `dbl n` with `n >= 3`, and fails by ~5 % at the two coarsest meshes (`dbl 1`,
/// `dbl 2`), where the primitive cell is itself far from converged (`Z*_xx` moves from `-0.665`
/// at `prim 2` to `-0.6507` by `prim 16`). The **transverse** component agrees to `10^-9` at
/// *every* mesh including the coarsest, so this is specific to the periodic direction, where the
/// folded cell must recover from degenerate band pairs at one k what the primitive cell reads off
/// distinct k points. Both facts are asserted below; the coarse-mesh longitudinal disagreement is
/// a known limitation recorded in `docs/scope.md`, not a passing test dressed up as agreement.
#[test]
fn the_born_charges_are_intensive() {
    let a = 2.80 * ANGSTROM_TO_BOHR;
    let primitive = hf_chain();
    let doubled = Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.93 * ANGSTROM_TO_BOHR, 0.0, 0.0),
        },
        Atom {
            z: 9,
            position: Vec3::new(a, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(a + 0.93 * ANGSTROM_TO_BOHR, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(2.0 * a, 0.0, 0.0)]).unwrap());

    let solve = |m: &Molecule, n: usize| {
        born_and_dielectric(
            m,
            &params(),
            &options(KMesh::grid(n, 1, 1)),
            &DfptOptions::default(),
        )
        .expect("field response")
    };

    // The premise: the two representations are the same calculation. If this drifts, nothing
    // below means anything, so it is checked rather than assumed.
    let e_prim = pm7_rs::run_pm7(&primitive, &params(), &options(KMesh::grid(6, 1, 1)))
        .expect("scf")
        .total_ev;
    let e_dbl = pm7_rs::run_pm7(&doubled, &params(), &options(KMesh::grid(3, 1, 1)))
        .expect("scf")
        .total_ev;
    assert!(
        (e_dbl / 2.0 - e_prim).abs() < 1.0e-7,
        "the doubled cell is not the same system: {} vs {e_prim} per formula unit",
        e_dbl / 2.0
    );

    // Longitudinal, at a mesh where both cells are converged.
    let one = solve(&primitive, 6);
    let two = solve(&doubled, 3);
    for atom in 0..4 {
        let reference = one.born[atom % 2].get(0, 0);
        let found = two.born[atom].get(0, 0);
        assert!(
            (found - reference).abs() < 1.0e-5,
            "atom {atom}: supercell Z*_xx {found} vs primitive {reference}"
        );
    }

    // Transverse, at the coarsest mesh there is. This one holds where the longitudinal does not,
    // which is the observation that localises the coarse-mesh limitation to the periodic axis.
    let one = solve(&primitive, 4);
    let two = solve(&doubled, 2);
    for atom in 0..4 {
        let reference = one.born[atom % 2].get(1, 1);
        let found = two.born[atom].get(1, 1);
        assert!(
            (found - reference).abs() < 1.0e-8,
            "atom {atom}: supercell Z*_yy {found} vs primitive {reference}"
        );
    }
}

/// The LO–TO term is 3-D only, needs a direction, and is positive semi-definite along it.
#[test]
fn the_lo_to_term_needs_three_dimensions_and_a_direction() {
    let chain = born_and_dielectric(
        &hf_chain(),
        &params(),
        &options(KMesh::grid(4, 1, 1)),
        &DfptOptions::default(),
    )
    .expect("chain");
    // A 1-D cell has no volume, so the non-analytic term is refused rather than invented.
    let message = chain.non_analytic().unwrap_err().to_string();
    assert!(message.contains("3-D"), "{message}");

    let crystal = born_and_dielectric(
        &diamond(),
        &params(),
        &options(KMesh::grid(2, 2, 2)),
        &DfptOptions::default(),
    )
    .expect("diamond");
    let na = crystal.non_analytic().expect("a 3-D cell has a LO-TO term");
    assert!(
        na.matrix([0.0, 0.0, 0.0]).is_err(),
        "q_hat must be non-zero"
    );

    let d = na.matrix([1.0, 0.0, 0.0]).expect("LO-TO matrix");
    // `D^NA` is an outer product of real vectors, so it is real, symmetric and positive
    // semi-definite — its diagonal cannot be negative.
    for i in 0..d.n {
        let (re, im) = d.get(i, i);
        assert!(im.abs() < 1.0e-12, "D^NA should be real, got {im}");
        assert!(re >= -1.0e-12, "D^NA diagonal {re} should be non-negative");
        for j in 0..d.n {
            let (a, _) = d.get(i, j);
            let (b, _) = d.get(j, i);
            assert!((a - b).abs() < 1.0e-12, "D^NA should be symmetric");
        }
    }
    // Diamond is non-polar: its Born charges vanish by symmetry, so the LO-TO term does too.
    // That is the physically right answer and worth pinning, because a non-zero one here would
    // mean the charges are wrong.
    assert!(
        crystal.born.iter().all(|z| z.get(0, 0).abs() < 1.0e-2),
        "diamond is non-polar; Z* should vanish, got {:?}",
        crystal.born[0].get(0, 0)
    );
}
