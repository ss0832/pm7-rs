// SPDX-License-Identifier: GPL-3.0-or-later
//! Perturbation theory at arbitrary wavevector, against the two things it must reproduce.
//!
//! There are exactly two independent references, and between them they pin every phase convention
//! and every factor in the response:
//!
//! * at `q = 0` the answer must equal the zone-centre analytic Hessian, which is computed by a
//!   completely different route (a real second derivative plus a real CPHF);
//! * at a `q` commensurate with an `n₁×n₂×n₃` supercell it must equal what that supercell's force
//!   constants give, which involves no perturbation theory at all.
//!
//! Anything that satisfies both is not plausibly wrong.

use pm7_rs::cell::Cell;
use pm7_rs::dfpt::{dynamical_matrix_dfpt, DfptOptions};
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{analytic_hessian, force_constants, Molecule, Pm7Options, Pm7Parameters};

fn params(method: &str) -> Pm7Parameters {
    Pm7Parameters::method(method.parse().unwrap()).unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

fn options(method: &str, mesh: KMesh) -> Pm7Options {
    Pm7Options {
        method: method.parse().unwrap(),
        e_tol: 1.0e-12,
        // Tight, because the references are finite differences; and a generous iteration budget,
        // because a displaced 1-D chain on a k mesh is one of the slower SCF cases.
        p_tol: 1.0e-8,
        max_scf: 800,
        pbc: Some(PbcOptions {
            kmesh: mesh,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

/// A 1-D chain of well-separated H₂ molecules.
///
/// The reference for a k-mesh test is always a finite difference, so the system has to be one
/// whose SCF solution is a *continuous function of the geometry*. A wide-gap closed-shell
/// insulator is; a small-gap chain is not, and its finite differences are then noise rather than
/// a reference. This one has a 21 eV gap that barely moves with the mesh.
fn h2_chain() -> Molecule {
    let a = a0();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.76 * a, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(3.2 * a, 0.0, 0.0)]).unwrap())
}

/// A polar 1-D chain: hydrogen fluoride molecules head to tail.
///
/// The homonuclear chain leaves the monopole part of the response almost untouched, so it cannot
/// see a mistake in the phased Ewald term. A chain with a real dipole per cell can.
fn hf_chain() -> Molecule {
    let a = a0();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 9,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.95 * a, 0.0, 0.0),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(3.4 * a, 0.0, 0.0)]).unwrap())
}

/// A 1-D CH2 chain — the cheapest periodic system with more than one atom per cell.
fn ch2_chain() -> Molecule {
    let a = a0();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, 0.89 * a),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 0.63 * a, -0.89 * a),
        },
    ])
    .with_cell(Cell::new(&[Vec3::new(2.55 * a, 0.0, 0.0)]).unwrap())
}

/// Well-separated methyl radicals: seven electrons per cell, a doublet, and far enough apart that
/// the bands stay flat and the gap stays open.
///
/// A half-filled band would be a metal, which the response refuses rather than answers.
fn methyl_chain() -> Molecule {
    let a = a0();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, 1.08 * a, 0.0),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, -0.54 * a, 0.935 * a),
        },
        pm7_rs::Atom {
            z: 1,
            position: Vec3::new(0.0, -0.54 * a, -0.935 * a),
        },
    ])
    .with_multiplicity(2)
    .with_cell(Cell::new(&[Vec3::new(7.0 * a, 0.0, 0.0)]).unwrap())
}
fn worst_difference(a: &pm7_rs::linalg::Matrix, b: &pm7_rs::cmatrix::CMatrix) -> (f64, f64) {
    let mut real = 0.0_f64;
    let mut imaginary = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let (re, im) = b.get(i, j);
            real = real.max((a[(i, j)] - re).abs());
            imaginary = imaginary.max(im.abs());
        }
    }
    (real, imaginary)
}

#[test]
fn at_the_zone_centre_it_reproduces_the_analytic_hessian() {
    let molecule = ch2_chain();
    let opts = options("pm7-", KMesh::grid(1, 1, 1));
    let reference =
        analytic_hessian(&molecule, &params("pm7-"), &opts, 1.0e-4).expect("analytic Hessian");
    let out = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7-"),
        &opts,
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("DFPT");
    assert!(out.converged, "the response did not converge");
    let (real, imaginary) = worst_difference(&reference, &out.force_constants);
    assert!(
        imaginary < 1.0e-8,
        "D(0) should be real but has an imaginary part of {imaginary:.3e}"
    );
    assert!(
        real < 2.0e-4,
        "D(0) differs from the analytic Hessian by {real:.3e} eV/Bohr²"
    );
}

#[test]
fn a_multi_point_mesh_still_reproduces_the_zone_centre_hessian() {
    // Separates "the q phases are wrong" from "handling more than one k point is wrong". At
    // `q = 0` every phase in the perturbation is 1, so this exercises only the second.
    for (name, molecule) in [("H2", h2_chain()), ("HF", hf_chain())] {
        for n in [2usize, 4] {
            let opts = options("pm7-", KMesh::grid(n, 1, 1));
            // `analytic_hessian` samples Γ only, so the reference here is the numerical Hessian,
            // which differentiates the k-point-correct analytic gradient.
            let reference = pm7_rs::numerical_hessian(&molecule, &params("pm7-"), &opts, 1.0e-3)
                .expect("numerical Hessian");
            let out = dynamical_matrix_dfpt(
                &molecule,
                &params("pm7-"),
                &opts,
                [0.0, 0.0, 0.0],
                &DfptOptions::default(),
            )
            .expect("DFPT");
            assert!(
                out.converged,
                "{name} {n}×1×1: the response did not converge"
            );
            let (real, imaginary) = worst_difference(&reference, &out.force_constants);
            let scale = (0..reference.rows)
                .flat_map(|i| (0..reference.cols).map(move |j| (i, j)))
                .fold(1.0_f64, |m, (i, j)| m.max(reference[(i, j)].abs()));
            assert!(
                imaginary < 1.0e-8,
                "{name} {n}×1×1: D(0) picked up {imaginary:.3e} imaginary"
            );
            assert!(
                real < 1.0e-3 * scale,
                "{name}, {n}×1×1 mesh: D(0) differs from the numerical Hessian by \
                 {real:.3e} eV/Bohr² (values up to {scale:.3e})"
            );
        }
    }
}

#[test]
fn at_a_commensurate_wavevector_it_reproduces_the_supercell() {
    // `q = (½, 0, 0)` is commensurate with a 2×1×1 supercell, so the supercell force constants
    // give the exact answer there with no interpolation. This is the test that the `q` phases are
    // right: at `q = 0` they are all 1 and cannot be wrong.
    for (name, molecule) in [("H2", h2_chain()), ("HF", hf_chain())] {
        let opts = options("pm7-", KMesh::grid(2, 1, 1));
        let fc = force_constants(&molecule, &params("pm7-"), &opts, [2, 1, 1])
            .expect("supercell force constants");
        let reference = fc.dynamical_matrix([0.5, 0.0, 0.0]);

        let out = dynamical_matrix_dfpt(
            &molecule,
            &params("pm7-"),
            &opts,
            [0.5, 0.0, 0.0],
            &DfptOptions::default(),
        )
        .expect("DFPT");
        assert!(out.converged, "{name}: the response did not converge");
        let mine = out.dynamical_matrix();
        let mut worst = 0.0_f64;
        let mut scale = 1.0_f64;
        for i in 0..mine.n {
            for j in 0..mine.n {
                let (ar, ai) = reference.get(i, j);
                let (br, bi) = mine.get(i, j);
                scale = scale.max(ar.abs()).max(ai.abs());
                worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            }
        }
        assert!(
            worst < 1.0e-3 * scale,
            "{name}: D(½,0,0) differs from the 2×1×1 supercell by {worst:.3e} \
             eV/(Å²·amu) (values up to {scale:.3e})"
        );
    }
}

/// The same check at a `q` whose phases are **complex**, which is where it bites.
///
/// `q = 0` makes every phase 1 and `q = ½` makes every phase `±1`; both are real, and both were
/// the only wavevectors this suite tested. A whole class of defect lives in the imaginary part and
/// is invisible to them — the exchange response used to build its `F(T)` and `F(−T)` blocks from
/// the same `Δp(T)`, which is right only when `Δp` is real. Every general `q` came back with a
/// non-Hermitian `D(q)`, and the assembly then symmetrized it on the way out, so the error was
/// laundered into a plausible matrix instead of surfacing.
///
/// `q = ⅓` and `q = ¼` are commensurate with a 3× and 4× supercell, so the supercell answer is
/// exact there and the comparison needs no interpolation.
#[test]
fn at_a_complex_phase_wavevector_it_reproduces_the_supercell() {
    for (name, molecule) in [("H2", h2_chain()), ("HF", hf_chain())] {
        for (repeat, q) in [(3usize, 1.0 / 3.0), (4, 0.25)] {
            let opts = options("pm7-", KMesh::grid(repeat, 1, 1));
            let fc = force_constants(&molecule, &params("pm7-"), &opts, [repeat, 1, 1])
                .expect("supercell force constants");
            let reference = fc.dynamical_matrix([q, 0.0, 0.0]);

            let out = dynamical_matrix_dfpt(
                &molecule,
                &params("pm7-"),
                &opts,
                [q, 0.0, 0.0],
                &DfptOptions::default(),
            )
            .expect("DFPT");
            assert!(
                out.converged,
                "{name}: the response did not converge at q={q}"
            );
            let mine = out.dynamical_matrix();
            let (mut worst, mut scale, mut imaginary) = (0.0_f64, 1.0_f64, 0.0_f64);
            for i in 0..mine.n {
                for j in 0..mine.n {
                    let (ar, ai) = reference.get(i, j);
                    let (br, bi) = mine.get(i, j);
                    scale = scale.max(ar.abs()).max(ai.abs());
                    worst = worst.max((ar - br).abs()).max((ai - bi).abs());
                    imaginary = imaginary.max(ai.abs());
                }
            }
            // The premise: these phases really are complex. If the reference happened to be real
            // this test would silently be a duplicate of the `q = ½` one.
            assert!(
                imaginary > 1.0e-3,
                "{name} at q={q}: the reference has no imaginary part ({imaginary:.3e}), so this \
                 is not testing a complex phase at all"
            );
            assert!(
                worst < 1.0e-3 * scale,
                "{name}: D({q},0,0) differs from the {repeat}x1x1 supercell by {worst:.3e} \
                 eV/(Å²·amu) (values up to {scale:.3e})"
            );
        }
    }
}

/// The post-SCF corrections belong in D(q), and were missing from it entirely.
#[test]
fn the_post_scf_corrections_are_in_the_dynamical_matrix() {
    // `"pm7"`, not `"pm7-"`. Every other test in this file switches the corrections off, which is
    // precisely why their absence from D(q) went unnoticed: dispersion, PM7-HH and EH+ are
    // classical position-only terms that never enter the electronic response, so omitting them
    // leaves a matrix that converges and looks entirely reasonable.
    let molecule = ch2_chain();
    let opts = options("pm7", KMesh::grid(1, 1, 1));
    let reference =
        analytic_hessian(&molecule, &params("pm7"), &opts, 1.0e-4).expect("analytic Hessian");
    let out = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7"),
        &opts,
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("DFPT");
    assert!(out.converged, "the response did not converge");
    let (real, imaginary) = worst_difference(&reference, &out.force_constants);
    assert!(imaginary < 1.0e-8, "D(0) imaginary part {imaginary:.3e}");
    assert!(
        real < 2.0e-4,
        "D(0) with corrections differs from the analytic Hessian by {real:.3e} eV/Bohr^2"
    );

    // And the corrections are actually large enough that dropping them would have shown up here:
    // compare against the same molecule with them switched off.
    let bare = options("pm7-", KMesh::grid(1, 1, 1));
    let without =
        analytic_hessian(&molecule, &params("pm7-"), &bare, 1.0e-4).expect("analytic Hessian");
    let mut gap = 0.0f64;
    for i in 0..reference.rows {
        for j in 0..reference.cols {
            gap = gap.max((reference[(i, j)] - without[(i, j)]).abs());
        }
    }
    assert!(
        gap > 1.0e-3,
        "the corrections should change the Hessian appreciably; largest change {gap:.3e}"
    );
}

/// A shifted mesh must run the response on the mesh the SCF converged on.
///
/// `full_mesh` used to rebuild a Gamma-centred unshifted grid from `divisions()` alone, so this
/// combination silently perturbed a different k set from the one the density came from.
#[test]
fn a_shifted_mesh_uses_the_mesh_the_scf_used() {
    let molecule = hf_chain();
    let shifted = KMesh::MonkhorstPack {
        n: [2, 1, 1],
        shift: [0.5, 0.0, 0.0],
        gamma_centred: true,
    };
    let opts = options("pm7-", shifted);
    let out = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7-"),
        &opts,
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("DFPT on a shifted mesh");
    assert!(out.converged, "the response did not converge");

    // At q = 0 the answer must still be real and must still satisfy the acoustic sum rule: a
    // translation of the whole chain costs nothing whatever mesh samples it.
    let n = out.force_constants.n;
    let mut worst_row = 0.0f64;
    for i in 0..n {
        for axis in 0..3 {
            let mut re_sum = 0.0;
            let mut im_sum = 0.0;
            for atom in 0..n / 3 {
                let (re, im) = out.force_constants.get(i, 3 * atom + axis);
                re_sum += re;
                im_sum += im;
            }
            worst_row = worst_row.max(re_sum.abs()).max(im_sum.abs());
        }
    }
    assert!(
        worst_row < 5.0e-4,
        "acoustic sum rule violated on a shifted mesh by {worst_row:.3e} eV/Bohr^2"
    );
}

/// `KMesh::Gamma` is the same sampling as `grid(1, 1, 1)` and is now promoted to it.
#[test]
fn a_gamma_kmesh_is_accepted_rather_than_refused() {
    let molecule = ch2_chain();
    let out = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7-"),
        &options("pm7-", KMesh::Gamma),
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("KMesh::Gamma should be promoted to grid(1, 1, 1), not refused");
    assert!(out.converged);

    let explicit = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7-"),
        &options("pm7-", KMesh::grid(1, 1, 1)),
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("DFPT");
    let n = out.force_constants.n;
    for i in 0..n {
        for j in 0..n {
            let (a, _) = out.force_constants.get(i, j);
            let (b, _) = explicit.force_constants.get(i, j);
            assert!((a - b).abs() < 1.0e-10, "({i},{j}): {a} vs {b}");
        }
    }
}

/// Forcing UHF on a **closed shell** must reproduce the RHF answer exactly.
///
/// The sharpest test of the spin-resolved response there is, because it has a reference that is
/// already validated rather than merely plausible. A closed-shell UHF solution has
/// `P_alpha = P_beta = P/2` and two identical Focks, so every piece of the two-channel path — the
/// per-spin band sets, the `(total, same-spin)` kernel split, the summed response — has to
/// collapse onto the one-channel path it replaced. Any factor of two, any channel double-counted,
/// any exchange term reading the total density instead of its own, shows up here immediately.
#[test]
fn a_forced_uhf_closed_shell_reproduces_the_rhf_response() {
    for (name, molecule) in [("H2", h2_chain()), ("HF", hf_chain())] {
        for q in [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]] {
            let mut restricted = options("pm7-", KMesh::grid(4, 1, 1));
            restricted.reference = pm7_rs::ScfReference::Restricted;
            let mut unrestricted = options("pm7-", KMesh::grid(4, 1, 1));
            unrestricted.reference = pm7_rs::ScfReference::Unrestricted;

            let a = dynamical_matrix_dfpt(
                &molecule,
                &params("pm7-"),
                &restricted,
                q,
                &DfptOptions::default(),
            )
            .expect("RHF DFPT");
            let b = dynamical_matrix_dfpt(
                &molecule,
                &params("pm7-"),
                &unrestricted,
                q,
                &DfptOptions::default(),
            )
            .expect("forced-UHF DFPT");

            let (mut worst, mut scale) = (0.0_f64, 1.0_f64);
            for i in 0..a.force_constants.n {
                for j in 0..a.force_constants.n {
                    let (ar, ai) = a.force_constants.get(i, j);
                    let (br, bi) = b.force_constants.get(i, j);
                    scale = scale.max(ar.abs()).max(ai.abs());
                    worst = worst.max((ar - br).abs()).max((ai - bi).abs());
                }
            }
            assert!(
                worst < 1.0e-6 * scale,
                "{name} at q={q:?}: forced UHF differs from RHF by {worst:.3e} \
                 (values up to {scale:.3e})"
            );
        }
    }
}

/// A genuinely open-shell periodic system, against the numerical Hessian.
///
/// `analytic_hessian_periodic` refuses an unrestricted cell, so the supercell route offers no
/// reference here; `numerical_hessian` does, by differentiating the analytic gradient, which is
/// already unrestricted. That is a completely independent route — no perturbation theory in it at
/// all — so agreement pins the open-shell response rather than merely its self-consistency.
#[test]
fn an_open_shell_chain_reproduces_the_numerical_hessian() {
    let molecule = methyl_chain();

    let mut opts = options("pm7-", KMesh::grid(2, 1, 1));
    opts.multiplicity = 2;
    let reference = pm7_rs::numerical_hessian(&molecule, &params("pm7-"), &opts, 1.0e-3)
        .expect("numerical Hessian");
    let out = dynamical_matrix_dfpt(
        &molecule,
        &params("pm7-"),
        &opts,
        [0.0, 0.0, 0.0],
        &DfptOptions::default(),
    )
    .expect("open-shell DFPT");
    assert!(out.converged, "the open-shell response did not converge");

    let (real, imaginary) = worst_difference(&reference, &out.force_constants);
    let scale = (0..reference.rows)
        .flat_map(|i| (0..reference.cols).map(move |j| (i, j)))
        .fold(1.0_f64, |m, (i, j)| m.max(reference[(i, j)].abs()));
    assert!(
        imaginary < 1.0e-8,
        "D(0) must be real, got {imaginary:.3e} imaginary"
    );
    assert!(
        real < 2.0e-3 * scale,
        "open-shell D(0) differs from the numerical Hessian by {real:.3e} eV/Bohr^2 \
         (values up to {scale:.3e})"
    );
}

/// An open-shell cell at a general `q` still produces a Hermitian `D(q)`.
///
/// The assembly refuses a non-Hermitian result, so this passing at all is the assertion — and it
/// covers the two-channel path at complex phases, where the RHF version of this defect lived.
#[test]
fn an_open_shell_chain_is_hermitian_at_a_general_q() {
    let molecule = methyl_chain();
    let mut opts = options("pm7-", KMesh::grid(3, 1, 1));
    opts.multiplicity = 2;
    for q in [[0.25, 0.0, 0.0], [0.3, 0.0, 0.0]] {
        let out = dynamical_matrix_dfpt(
            &molecule,
            &params("pm7-"),
            &opts,
            q,
            &DfptOptions::default(),
        )
        .unwrap_or_else(|e| panic!("open-shell DFPT at q={q:?}: {e}"));
        assert!(out.converged, "q={q:?} did not converge");
    }
}

/// Forcing UHF on a **closed shell** must reproduce the restricted field response exactly.
///
/// The sharpest check available for the unrestricted path, and the one that would catch every way
/// of getting it half right: a factor of two, a channel counted twice, one commutator serving both
/// spins, or a contraction done in the wrong band basis. With `α = β` the two channels are
/// identical, so the answer has to be the restricted one — which is an already-validated
/// reference rather than a second calculation of the same kind.
///
/// v0.2.1 refused this call rather than approximating it; the refusal named the three things that
/// had to be per spin, and they are the three things the implementation now does.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_field_response() {
    let molecule = hf_chain();
    let restricted = pm7_rs::dfpt::born_and_dielectric(
        &molecule,
        &params("pm7-"),
        &options("pm7-", KMesh::grid(2, 1, 1)),
        &DfptOptions::default(),
    )
    .expect("restricted field response");

    let mut opts = options("pm7-", KMesh::grid(2, 1, 1));
    opts.reference = pm7_rs::ScfReference::Unrestricted;
    let forced = pm7_rs::dfpt::born_and_dielectric(
        &molecule,
        &params("pm7-"),
        &opts,
        &DfptOptions::default(),
    )
    .expect("an unrestricted cell is supported now, not refused");

    let mut worst_born = 0.0_f64;
    let mut scale = 0.0_f64;
    for (r, u) in restricted.born.iter().zip(&forced.born) {
        for a in 0..3 {
            for b in 0..3 {
                worst_born = worst_born.max((r.get(a, b) - u.get(a, b)).abs());
                scale = scale.max(r.get(a, b).abs());
            }
        }
    }
    assert!(scale > 0.1, "Born charges are trivially small: {scale:.3e}");
    assert!(
        worst_born < 1.0e-8,
        "forced UHF moved a Born charge by {worst_born:.3e} against a scale of {scale:.3e}"
    );

    let mut worst_alpha = 0.0_f64;
    let mut alpha_scale = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            let (r, u) = (
                restricted.polarizability.get(a, b),
                forced.polarizability.get(a, b),
            );
            worst_alpha = worst_alpha.max((r - u).abs());
            alpha_scale = alpha_scale.max(r.abs());
        }
    }
    assert!(
        alpha_scale > 1.0e-3,
        "the polarizability is trivially small: {alpha_scale:.3e}"
    );
    assert!(
        worst_alpha < 1.0e-8 * alpha_scale.max(1.0),
        "forced UHF moved the polarizability by {worst_alpha:.3e} against {alpha_scale:.3e}"
    );
}

/// A genuine open shell gives a field response that satisfies the sum rule and is *not* the
/// closed-shell one.
///
/// The second half matters: an implementation that quietly averaged the two channels, or built
/// one commutator from the total density, would pass every symmetry and sum-rule check while
/// returning the restricted answer for a doublet.
#[test]
fn an_open_shell_cell_has_its_own_field_response() {
    let molecule = methyl_chain();
    let mut opts = options("pm7-", KMesh::grid(2, 1, 1));
    opts.multiplicity = 2;
    opts.reference = pm7_rs::ScfReference::Unrestricted;
    let open = pm7_rs::dfpt::born_and_dielectric(
        &molecule,
        &params("pm7-"),
        &opts,
        &DfptOptions::default(),
    )
    .expect("open-shell field response");

    assert!(
        open.converged,
        "the open-shell field response did not converge"
    );
    // `Σ_A Z*_A = q_tot`, which is zero for this neutral cell. The explicit Mulliken term and the
    // electronic response have to cancel across the whole cell for this to hold.
    assert!(
        open.acoustic_residual() < 1.0e-8,
        "Born charges do not sum to zero: {:.3e}",
        open.acoustic_residual()
    );

    let mut scale = 0.0_f64;
    for z in &open.born {
        for a in 0..3 {
            for b in 0..3 {
                scale = scale.max(z.get(a, b).abs());
            }
        }
    }
    assert!(
        scale > 0.01,
        "Born charges are trivially small: {scale:.3e}"
    );

    // The same cell forced closed-shell is a different physical system, and must give a different
    // answer. If it does not, the spin resolution is not reaching the response.
    let mut closed = options("pm7-", KMesh::grid(2, 1, 1));
    closed.reference = pm7_rs::ScfReference::Restricted;
    if let Ok(rhf) = pm7_rs::dfpt::born_and_dielectric(
        &molecule,
        &params("pm7-"),
        &closed,
        &DfptOptions::default(),
    ) {
        let mut difference = 0.0_f64;
        for (o, r) in open.born.iter().zip(&rhf.born) {
            for a in 0..3 {
                for b in 0..3 {
                    difference = difference.max((o.get(a, b) - r.get(a, b)).abs());
                }
            }
        }
        assert!(
            difference > 1.0e-4,
            "the doublet's Born charges are the closed-shell ones to {difference:.3e}; the spin \
             resolution is not reaching the field response"
        );
    }
}

/// **A degeneracy the sampling mesh does not resolve, and neither tolerance owns it.**
///
/// Along `[1, 0, 0]` in diamond the two transverse acoustic branches are degenerate by symmetry,
/// so a splitting between them is numerical and worth attributing. The release plan recorded one —
/// `377.2476` against `377.3460 cm^-1` at `q = (0.25, 0, 0)` — and named two suspects: the CG
/// tolerance, or the Hermiticity threshold in `dfpt.rs`.
///
/// It is neither. Measured:
///
/// | mesh | TA pair (cm^-1) | split |
/// |---|---|---|
/// | 2x2x2 | 377.2476 / 377.3460 | **0.098** |
/// | 3x3x3 | 396.5690 / 396.5690 | 0.0000 |
/// | 4x4x4 | 399.2947 / 399.2947 | 0.0000 |
/// | 5x5x5 | 399.6508 / 399.6509 | 0.0000 |
/// | 6x6x6 | 399.8903 / 399.8913 | 0.0009 |
///
/// and tightening the tolerances moves it not at all — exactly degenerate at a 4x4x4 mesh across
/// `dfpt_tolerance` from `1e-8` to `1e-12` and `scf_tolerance` from `1e-7` to `1e-11`. So the
/// splitting is **under-sampling**: a 2x2x2 mesh does not resolve the point group that makes the
/// two branches degenerate, and the `k` and `k + q` sets it offers at `q = (0.25, 0, 0)` are not
/// related by the symmetry the degeneracy rests on. It is a property of the mesh, not a defect in
/// the solver, and it is the same mesh that puts the frequency 22 cm^-1 below the converged value.
///
/// This test pins the resolution rather than the number: refine the mesh and the splitting has to
/// go, whatever it happens to be at the coarse end.
#[test]
fn the_transverse_acoustic_degeneracy_is_a_mesh_artefact() {
    let a = 3.567 * a0();
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    let diamond = Molecule::new(vec![
        pm7_rs::Atom {
            z: 6,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell);

    let params = params("pm7");
    let settings = DfptOptions {
        tolerance: 1.0e-10,
        max_iterations: 400,
        ..DfptOptions::default()
    };

    let split = |mesh: usize| -> f64 {
        let out = dynamical_matrix_dfpt(
            &diamond,
            &params,
            &options("pm7", KMesh::grid(mesh, mesh, mesh)),
            [0.25, 0.0, 0.0],
            &settings,
        )
        .expect("dfpt at q = (0.25, 0, 0)");
        let mut f = out.frequencies_cm().expect("frequencies");
        f.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // Six modes; the transverse acoustic pair is the two lowest above the numerical zero.
        let ta: Vec<f64> = f.into_iter().filter(|v| *v > 1.0).take(2).collect();
        (ta[1] - ta[0]).abs()
    };

    let coarse = split(2);
    let refined = split(4);
    assert!(
        coarse > 0.05,
        "the 2x2x2 artefact should still be there ({coarse} cm^-1); if it has gone, this test no \
         longer measures anything and the doc comment above is stale"
    );
    assert!(
        refined < 1.0e-3,
        "refining the mesh must close the splitting, and it did not: {coarse} at 2x2x2 against \
         {refined} at 4x4x4"
    );
}

/// Zinc blende ZnS in its fcc primitive cell — a cell whose PM7 gap depends on the mesh.
///
/// It is the fixture for the partial-occupation gate precisely because of that dependence: the
/// same structure is spuriously metallic at `3×3×3` and gapped at `4×4×4`, so one geometry
/// exercises both sides of the gate and nothing else about the two runs differs.
fn zinc_blende_zns() -> Molecule {
    let a = 5.41 * a0();
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 30,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 16,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell)
}

/// A `q = 0` response of a genuinely partially occupied cell is refused; everything else is not.
///
/// The response carries no integer-occupation assumption — band pairs are weighted by `Δf/Δε`,
/// which is the metallic form — so this is not a gate on metals. It is a gate on the one place the
/// implementation is **incomplete**: a uniform perturbation moves the Fermi level, and the
/// intraband term that goes with holding the electron count fixed is not implemented, so a zone
/// centre computed for a partially occupied cell is wrong by an amount the result does not show.
///
/// Three claims, and each fails differently if the gate is written wrong:
///
/// * the **gapless** mesh is refused at `q = 0` — a gate that tested the entropy instead, or that
///   ran only when smearing was absent, would let this through;
/// * the same cell, mesh and width at **`q ≠ 0`** is *not* refused — the missing term vanishes
///   there by symmetry, and gating on occupations alone would wrongly stop it;
/// * the **gapped** mesh is not refused although it is smeared — smearing applied to a cell with a
///   gap is a change of path and not of answer, and a gate on "is there smearing" would stop it.
#[test]
fn a_partially_occupied_zone_centre_is_refused_and_nothing_else_is() {
    let molecule = zinc_blende_zns();
    let smeared = |mesh: usize| -> Pm7Options {
        let mut opts = options("pm7", KMesh::grid(mesh, mesh, mesh));
        opts.pbc = Some(PbcOptions {
            kmesh: KMesh::grid(mesh, mesh, mesh),
            smearing: pm7_rs::pbc::Smearing::FermiDirac { width_ev: 0.10 },
            ..PbcOptions::default()
        });
        opts
    };
    let solve = |opts: &Pm7Options, q: [f64; 3]| {
        dynamical_matrix_dfpt(&molecule, &params("pm7"), opts, q, &DfptOptions::default())
    };

    // 3x3x3: PM7 leaves this cell gapless, so a third of a state sits on either side of E_F.
    let error = solve(&smeared(3), [0.0; 3])
        .expect_err("a gapless zone centre must be refused, not answered");
    let message = error.to_string();
    assert!(
        message.contains("partially occupied") && message.contains("q != 0"),
        "the refusal must say what is wrong and what to do instead: {message}"
    );

    // The same everything, away from the zone centre, where the missing term vanishes.
    solve(&smeared(3), [1.0 / 3.0, 0.0, 0.0])
        .expect("q != 0 is complete for a partially occupied cell and must not be refused");

    // 4x4x4: the same structure, now with a 2.6 eV gap, so the smearing leaves the occupations
    // integral and the zone centre is a zone centre like any other.
    solve(&smeared(4), [0.0; 3])
        .expect("smearing that leaves the occupations integral must not be refused");
}
