// SPDX-License-Identifier: GPL-3.0-or-later
//! Periodic force constants: the analytic Hessian against finite differences of the analytic
//! gradient, and the physical invariants a set of force constants has to satisfy.
//!
//! The reference is `numerical_hessian`, which differentiates the periodic analytic gradient —
//! itself pinned against full-SCF finite differences in `tests/stress.rs`. So the chain is
//! energy → gradient → Hessian with an independent check at each link, rather than one analytic
//! expression validated against another written from the same notes.

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::PbcOptions;
use pm7_rs::{analytic_hessian, numerical_hessian, run_pm7, Molecule, Pm7Options, Pm7Parameters};

fn params(method: &str) -> Pm7Parameters {
    Pm7Parameters::method(method.parse().unwrap()).unwrap()
}

fn a0() -> f64 {
    pm7_rs::constants::ANGSTROM_TO_BOHR
}

fn options(method: &str) -> Pm7Options {
    Pm7Options {
        method: method.parse().unwrap(),
        // The reference is a finite difference of the gradient, so the SCF has to be converged
        // well past the difference itself or the comparison measures SCF noise.
        e_tol: 1.0e-12,
        p_tol: 1.0e-10,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    }
}

/// A 1-D polyethylene-like chain: CH2 repeating along x.
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

/// A 2-D boron-nitride-like sheet.
fn bn_sheet() -> Molecule {
    let a = 2.5 * a0();
    let cell = Cell::new(&[
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(a * 0.5, a * 0.866_025_403_784_44, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        pm7_rs::Atom {
            z: 5,
            position: Vec3::zero(),
        },
        pm7_rs::Atom {
            z: 7,
            position: Vec3::new(a * 0.5, a * 0.288_675_134_594_81, 0.0),
        },
    ])
    .with_cell(cell)
}

/// A 3-D cell with two water molecules — polar, so the Ewald terms carry real weight.
fn water_crystal() -> Molecule {
    let a = 5.6 * a0();
    let cell = Cell::cubic(a).unwrap();
    let mut atoms = Vec::new();
    for (shift, flip) in [
        (Vec3::zero(), 1.0),
        (Vec3::new(a * 0.5, a * 0.5, a * 0.5), -1.0),
    ] {
        atoms.push(pm7_rs::Atom {
            z: 8,
            position: shift,
        });
        atoms.push(pm7_rs::Atom {
            z: 1,
            position: shift + Vec3::new(0.9584 * a0() * flip, 0.0, 0.0),
        });
        atoms.push(pm7_rs::Atom {
            z: 1,
            position: shift + Vec3::new(-0.2400 * a0() * flip, 0.9278 * a0(), 0.0),
        });
    }
    Molecule::new(atoms).with_cell(cell)
}

fn check(molecule: &Molecule, label: &str, method: &str, tolerance: f64) {
    let opts = options(method);
    let analytic =
        analytic_hessian(molecule, &params(method), &opts, 1.0e-4).expect("analytic Hessian");
    let numeric =
        numerical_hessian(molecule, &params(method), &opts, 2.0e-4).expect("numerical Hessian");
    let n = analytic.rows;
    let mut worst = 0.0_f64;
    let mut where_worst = (0, 0);
    for i in 0..n {
        for j in 0..n {
            let d = (analytic[(i, j)] - numeric[(i, j)]).abs();
            if d > worst {
                worst = d;
                where_worst = (i, j);
            }
        }
    }
    assert!(
        worst < tolerance,
        "{label} ({method}): analytic and numerical Hessians differ by {worst:.3e} eV/Bohr² at \
         [{}][{}] (analytic {:.9}, numerical {:.9}, tolerance {tolerance:.1e})",
        where_worst.0,
        where_worst.1,
        analytic[where_worst],
        numeric[where_worst],
    );
}

#[test]
fn periodic_analytic_hessian_matches_finite_differences_in_one_dimension() {
    // PM7-minus first: it isolates the electronic + core + Ewald machinery from the post-SCF
    // corrections, so a failure here points at the SCF-side terms and not at dispersion.
    check(&ch2_chain(), "1-D CH2 chain", "pm7-", 2.0e-3);
}

#[test]
fn periodic_analytic_hessian_matches_finite_differences_in_two_dimensions() {
    check(&bn_sheet(), "2-D BN sheet", "pm7-", 2.0e-3);
}

#[test]
fn periodic_analytic_hessian_matches_finite_differences_in_three_dimensions() {
    check(&water_crystal(), "3-D water crystal", "pm7-", 3.0e-3);
}

#[test]
fn the_dispersion_and_hh_corrections_are_periodic_in_the_hessian_too() {
    // A hydrocarbon chain has no hydrogen bonds, so this isolates the pair corrections —
    // dispersion and the PM7-HH repulsion — including the second derivative of their taper.
    check(&ch2_chain(), "1-D CH2 chain", "pm7", 2.0e-3);
}

/// A hydrogen-bonded water chain: O···O ≈ 2.8 Å along **x**, each molecule donating to the next.
///
/// Deliberately *not* symmetric. The EH+ correction is built from bond angles and dihedrals, and
/// those are non-smooth where three atoms go collinear — a perfectly symmetric arrangement can sit
/// exactly on such a point, where the analytic gradient of the correction diverges even though its
/// energy stays finite. That is a property of the functional form (MOPAC's included), not of this
/// implementation, so the test uses a geometry a real hydrogen-bonded solid would have.
fn water_wire() -> Molecule {
    let a = a0();
    let cell = Cell::new(&[Vec3::new(5.52 * a, 0.0, 0.0)]).unwrap();
    let coords = [
        (8, [0.00, 0.00, 0.00]),
        (1, [0.90, 0.30, 0.05]), // donor, pointing at the next oxygen
        (1, [-0.30, 0.90, 0.06]),
        (8, [2.75, 0.60, 0.10]),
        (1, [3.69, 0.40, 0.07]), // donor, pointing at the next cell's oxygen
        (1, [2.45, 1.50, 0.30]),
    ];
    Molecule::new(
        coords
            .iter()
            .map(|(z, r)| pm7_rs::Atom {
                z: *z,
                position: Vec3::new(r[0] * a, r[1] * a, r[2] * a),
            })
            .collect(),
    )
    .with_cell(cell)
}

#[test]
fn the_hydrogen_bond_correction_is_periodic_in_the_hessian_too() {
    check(&water_wire(), "hydrogen-bonded water wire", "pm7", 4.0e-3);
}

#[test]
fn the_acoustic_sum_rule_holds() {
    // Translating the whole crystal costs nothing, so `Σ_B Φ(0A, TB) = 0` exactly. It follows
    // from every term depending on displacement *differences*, which makes it a sharp check that
    // no term was scattered onto the wrong atom.
    for (molecule, label) in [
        (ch2_chain(), "1-D CH2 chain"),
        (bn_sheet(), "2-D BN sheet"),
        (water_crystal(), "3-D water crystal"),
    ] {
        let opts = options("pm7");
        let h = analytic_hessian(&molecule, &params("pm7"), &opts, 1.0e-4).expect("Hessian");
        let nat = molecule.atoms.len();
        let mut worst = 0.0_f64;
        for a in 0..nat {
            for i in 0..3 {
                for j in 0..3 {
                    let sum: f64 = (0..nat).map(|b| h[(3 * a + i, 3 * b + j)]).sum();
                    worst = worst.max(sum.abs());
                }
            }
        }
        assert!(
            worst < 1.0e-6,
            "{label}: acoustic sum rule violated by {worst:.3e} eV/Bohr²"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Phonon dispersion.
// ---------------------------------------------------------------------------------------------

/// Sorted frequencies, for comparing two spectra that need not come out in the same order.
fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn the_supercell_spectrum_is_the_union_of_the_commensurate_q_points() {
    // The identity that makes Fourier-interpolated phonons meaningful: the Γ spectrum of an
    // `n`-fold supercell is exactly the union of `D(q)` over the `n` commensurate wavevectors.
    // It is a strong test of the whole chain — the supercell Hessian, the way its blocks are
    // sliced by translation, and the phase convention in `D(q)` all have to agree.
    let molecule = ch2_chain();
    let opts = options("pm7");
    let fc = pm7_rs::force_constants(&molecule, &params("pm7"), &opts, [2, 1, 1])
        .expect("force constants");

    let mut from_q = Vec::new();
    for q in [0.0, 0.5] {
        from_q.extend(fc.frequencies_cm([q, 0.0, 0.0]).expect("frequencies"));
    }

    // The same supercell, diagonalized directly as a Γ-point Hessian.
    let doubled = {
        let cell = molecule.cell.unwrap();
        let (super_cell, shifts) = cell.supercell([2, 1, 1]).unwrap();
        let mut atoms = Vec::new();
        for t in &shifts {
            let shift = cell.translation(*t);
            for atom in &molecule.atoms {
                atoms.push(pm7_rs::Atom {
                    z: atom.z,
                    position: atom.position + shift,
                });
            }
        }
        Molecule::new(atoms).with_cell(super_cell)
    };
    let direct = pm7_rs::force_constants(&doubled, &params("pm7"), &opts, [1, 1, 1])
        .expect("supercell force constants");
    let from_supercell = direct.frequencies_cm([0.0, 0.0, 0.0]).expect("frequencies");

    let (a, b) = (sorted(from_q), sorted(from_supercell));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(
            (x - y).abs() < 1.0,
            "commensurate-q spectrum {a:?} does not match the supercell spectrum {b:?}"
        );
    }
}

#[test]
fn the_acoustic_branches_vanish_at_the_zone_centre() {
    // One acoustic mode per periodic direction goes to zero at q = 0, because translating the
    // crystal along a lattice vector costs nothing. A 1-D chain has one; the other two "acoustic"
    // modes of a wire are transverse and are *not* required to vanish — a chain resists bending.
    let fc = pm7_rs::force_constants(&ch2_chain(), &params("pm7"), &options("pm7"), [1, 1, 1])
        .expect("force constants");
    let freqs = sorted(fc.frequencies_cm([0.0, 0.0, 0.0]).expect("frequencies"));
    assert!(
        freqs[0].abs() < 5.0,
        "the acoustic branch is at {:.3} cm⁻¹ instead of zero; the full spectrum is {freqs:?}",
        freqs[0]
    );
    // And nothing else is spuriously soft: a CH2 chain's real modes are hundreds of cm⁻¹.
    assert!(freqs[3] > 100.0, "too many near-zero modes: {freqs:?}");
}

#[test]
fn the_dispersion_is_periodic_in_q_and_even() {
    // `D(q)` is built from a real-space sum with `e^{2πi q·T}`, so it must repeat with period 1
    // in fractional q and satisfy `ω(−q) = ω(q)` by time reversal. Both are properties of the
    // construction rather than of the model, which is what makes them worth pinning: they catch a
    // wrong phase sign or a translation list that is not closed under negation.
    let fc = pm7_rs::force_constants(&ch2_chain(), &params("pm7"), &options("pm7"), [3, 1, 1])
        .expect("force constants");
    for q in [0.13, 0.37] {
        let base = sorted(fc.frequencies_cm([q, 0.0, 0.0]).unwrap());
        for other in [[q + 1.0, 0.0, 0.0], [-q, 0.0, 0.0]] {
            let got = sorted(fc.frequencies_cm(other).unwrap());
            for (x, y) in base.iter().zip(&got) {
                assert!(
                    (x - y).abs() < 1.0e-6,
                    "ω(q) at {q} is not reproduced at {other:?}: {base:?} vs {got:?}"
                );
            }
        }
    }
}

#[test]
fn the_acoustic_sum_rule_survives_the_supercell_slicing() {
    // After the `Φ(−T) = Φ(T)ᵀ` symmetrization the residual measures two things at once: the sum
    // rule itself, and how far the supercell Hessian is from being exactly translationally
    // invariant. The second is set by the image cutoffs, so it is small but not machine-zero —
    // the bound is what a converged calculation should manage, four orders below a real force
    // constant.
    let mut fc = pm7_rs::force_constants(&ch2_chain(), &params("pm7"), &options("pm7"), [2, 1, 1])
        .expect("force constants");
    let raw = fc.acoustic_residual();
    assert!(
        raw < 5.0e-3,
        "acoustic sum rule violated by {raw:.3e} eV/Bohr² after slicing the supercell Hessian"
    );
    // And the projection has to actually remove it.
    fc.enforce_acoustic_sum_rule();
    let projected = fc.acoustic_residual();
    assert!(
        projected < 1.0e-12,
        "enforcing the acoustic sum rule left {projected:.3e} eV/Bohr² behind"
    );
    // The zone-centre acoustic branch is then at the numerical floor. It cannot be *exactly*
    // zero in cm⁻¹: a frequency is `√λ`, so an eigenvalue at machine precision relative to the
    // 11 800 cm⁻¹ top of this spectrum still shows up in the fourth decimal place.
    let freqs = sorted(fc.frequencies_cm([0.0, 0.0, 0.0]).unwrap());
    assert!(
        freqs[0].abs() < 1.0e-3,
        "after projection the acoustic branch is still at {:.3e} cm⁻¹",
        freqs[0]
    );
}

#[test]
fn diamond_has_a_triply_degenerate_raman_mode_near_the_measured_frequency() {
    // A physical check rather than an internal-consistency one. Diamond's zone-centre optical
    // mode is the textbook Raman line at 1332 cm⁻¹, and it is triply degenerate by cubic
    // symmetry. Nothing in this crate was fitted to it, so agreement says the force constants,
    // the mass weighting, the unit conversion and the dynamical matrix are all right at once —
    // in a way that no comparison against another expression of the same code can.
    let a = 3.567 * a0();
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    let molecule = Molecule::new(vec![
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

    let mut fc = pm7_rs::force_constants(&molecule, &params("pm7"), &options("pm7"), [2, 2, 2])
        .expect("force constants");
    fc.enforce_acoustic_sum_rule();
    let freqs = sorted(fc.frequencies_cm([0.0, 0.0, 0.0]).expect("frequencies"));

    // Three acoustic modes at zero, then three degenerate optical ones.
    for (index, f) in freqs[..3].iter().enumerate() {
        assert!(
            f.abs() < 1.0e-3,
            "acoustic mode {index} is at {f:.4} cm⁻¹, not zero: {freqs:?}"
        );
    }
    let optical = &freqs[3..];
    let spread = optical[2] - optical[0];
    assert!(
        spread < 1.0,
        "the optical branch should be triply degenerate but spans {spread:.3} cm⁻¹: {freqs:?}"
    );
    // PM7 puts it at ~1318 cm⁻¹; experiment says 1332. The window is wide enough to be a check on
    // the machinery rather than on PM7's parameterization, and narrow enough to catch a factor of
    // √2 in the mass weighting or a wrong unit.
    assert!(
        (1150.0..1500.0).contains(&optical[0]),
        "the Raman mode came out at {:.1} cm⁻¹, nowhere near the measured 1332",
        optical[0]
    );
}

/// A k mesh is a calculation, not a refusal.
///
/// v0.2.1 refused here and pointed at a supercell or `numerical_hessian`. The refusal was not
/// wrong about the physics — a k mesh does make the response couple `k` with `k + q` — but that is
/// exactly what `dynamical_matrix_dfpt` solves, and at `q = 0` its force constants **are** the
/// k-point zone-centre Hessian. Refusing was refusing to make one call.
///
/// The check is that the two routes give the same matrix, not merely that neither errors: they are
/// meant to be the same calculation reached two ways.
#[test]
fn a_k_mesh_gives_the_zone_centre_response_rather_than_an_error() {
    let mut opts = options("pm7");
    opts.pbc = Some(PbcOptions {
        kmesh: pm7_rs::KMesh::grid(2, 1, 1),
        ..PbcOptions::default()
    });
    let molecule = ch2_chain();
    let hessian = analytic_hessian(&molecule, &params("pm7"), &opts, 1.0e-4)
        .expect("a k mesh is the zone-centre response now, not a refusal");
    let response = pm7_rs::dynamical_matrix_dfpt(
        &molecule,
        &params("pm7"),
        &opts,
        [0.0; 3],
        &pm7_rs::dfpt::DfptOptions::default(),
    )
    .unwrap();

    assert_eq!(hessian.rows, response.force_constants.n);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..hessian.rows {
        for j in 0..hessian.cols {
            let (re, _) = response.force_constants.get(i, j);
            worst = worst.max((hessian[(i, j)] - re).abs());
            scale = scale.max(re.abs());
        }
    }
    assert!(
        scale > 1.0,
        "force constants are trivially small: {scale:.3e}"
    );
    assert!(
        worst < 1.0e-10 * scale,
        "analytic_hessian on a k mesh differs from D(q = 0) by {worst:.3e} against a scale of \
         {scale:.3e}; they are meant to be the same calculation"
    );
}

#[test]
fn a_huge_cell_reproduces_the_molecular_hessian() {
    // The periodic machinery has to reduce to the molecular one when the images are far enough
    // away to stop interacting. This is the cleanest statement that the image sums, the monopole
    // subtraction and the Ewald terms all cancel correctly in the isolated limit.
    let molecule = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.2400 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    let mut boxed = molecule.clone();
    boxed.cell = Some(Cell::cubic(24.0 * a0()).unwrap());

    let free =
        analytic_hessian(&molecule, &params("pm7"), &options("pm7"), 1.0e-4).expect("molecular");
    let periodic =
        analytic_hessian(&boxed, &params("pm7"), &options("pm7"), 1.0e-4).expect("periodic");
    let mut worst = 0.0_f64;
    for i in 0..free.rows {
        for j in 0..free.cols {
            worst = worst.max((free[(i, j)] - periodic[(i, j)]).abs());
        }
    }
    // A 24 Å box still leaves a small monopole interaction between images; the bound is what a
    // water molecule's residual quadrupole field can plausibly do at that separation.
    assert!(
        worst < 5.0e-3,
        "a water molecule in a 24 Å box differs from the free molecule by {worst:.3e} eV/Bohr²"
    );
    // And the SCF energies must agree to the same order.
    let e_free = run_pm7(&molecule, &params("pm7"), &options("pm7"))
        .unwrap()
        .total_ev;
    let e_box = run_pm7(&boxed, &params("pm7"), &options("pm7"))
        .unwrap()
        .total_ev;
    assert!(
        (e_free - e_box).abs() < 5.0e-3,
        "energies differ by {:.3e} eV",
        (e_free - e_box).abs()
    );
}

// ---------------------------------------------------------------------------------------------
// Polarization vectors.
// ---------------------------------------------------------------------------------------------

/// The eigenvectors are the ones the frequencies came from, and they mean what they claim to.
///
/// Three claims that a wrong convention would break separately. **Unitarity** — they diagonalize a
/// Hermitian matrix, so the columns are orthonormal, and a normalization applied to the wrong axis
/// fails here. **Alignment** — column `n` goes with frequency `n`, which is the one thing a
/// reordering would break and nothing else would notice; `D e_n = lambda_n e_n` is the statement.
/// And the **acoustic modes at Gamma are uniform translations**, which is not a convention at all
/// but the physics: the cell can be translated for free, so those three eigenvectors have to put
/// the same Cartesian displacement on every atom.
#[test]
fn the_zone_centre_polarization_vectors_are_the_modes_they_claim_to_be() {
    let molecule = ch2_chain();
    let fc = pm7_rs::force_constants(&molecule, &params("pm7"), &options("pm7"), [1, 1, 1])
        .expect("force constants");
    let modes = fc.modes([0.0, 0.0, 0.0]).expect("modes");
    let n = modes.eigenvectors.n;
    assert_eq!(modes.frequencies_cm.len(), n);

    // Unitary: `E* E = I`.
    let mut worst_gram = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for k in 0..n {
                let (ar, ai) = modes.eigenvectors.get(k, i);
                let (br, bi) = modes.eigenvectors.get(k, j);
                // conj(a) * b
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            worst_gram = worst_gram.max((re - expected).abs()).max(im.abs());
        }
    }
    assert!(
        worst_gram < 1.0e-10,
        "the columns are not orthonormal: |E*E - I| reaches {worst_gram:.3e}"
    );

    // Alignment: `D e_n` is `lambda_n e_n`, with `lambda` recovered from the reported frequency.
    let d = fc.dynamical_matrix([0.0, 0.0, 0.0]);
    let scale = pm7_rs::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
    let mut worst_residual = 0.0_f64;
    for column in 0..n {
        // `f = sign(lambda) * sqrt(|lambda|) * scale`, so `lambda = sign(f) * (f / scale)^2`.
        let f = modes.frequencies_cm[column];
        let lambda = (f / scale) * (f / scale) * if f < 0.0 { -1.0 } else { 1.0 };
        for row in 0..n {
            let (mut re, mut im) = (0.0, 0.0);
            for k in 0..n {
                let (dr, di) = d.get(row, k);
                let (er, ei) = modes.eigenvectors.get(k, column);
                re += dr * er - di * ei;
                im += dr * ei + di * er;
            }
            let (er, ei) = modes.eigenvectors.get(row, column);
            worst_residual = worst_residual
                .max((re - lambda * er).abs())
                .max((im - lambda * ei).abs());
        }
    }
    assert!(
        worst_residual < 1.0e-8,
        "column n is not the eigenvector of frequency n: |D e - lambda e| reaches \
         {worst_residual:.3e}"
    );

    // The acoustic modes move every atom identically.
    let nat = molecule.atoms.len();
    let mut by_magnitude: Vec<usize> = (0..n).collect();
    by_magnitude.sort_by(|a, b| {
        modes.frequencies_cm[*a]
            .abs()
            .partial_cmp(&modes.frequencies_cm[*b].abs())
            .unwrap()
    });
    // A chain has one acoustic branch, not three: only the periodic direction has one.
    let acoustic = by_magnitude[0];
    let mut worst_spread = 0.0_f64;
    for axis in 0..3 {
        let (reference, _) = modes.cartesian_modes.get(axis, acoustic);
        for atom in 1..nat {
            let (v, _) = modes.cartesian_modes.get(3 * atom + axis, acoustic);
            worst_spread = worst_spread.max((v - reference).abs());
        }
    }
    assert!(
        worst_spread < 1.0e-8,
        "the acoustic mode at {:.4} cm^-1 is not a uniform translation: atoms differ by \
         {worst_spread:.3e}",
        modes.frequencies_cm[acoustic]
    );
}

/// The two phonon routes report the same modes, not two conventions for them.
///
/// At a `q` the supercell holds exactly, the perturbation solver and the supercell force constants
/// are computing the same dynamical matrix by different means. Frequencies are the comparable
/// quantity: an eigenvector is defined only up to a phase, and within a degenerate set only up to
/// a rotation, so comparing vectors directly would be comparing arbitrary choices.
#[test]
fn both_phonon_routes_agree_where_the_supercell_is_exact() {
    let molecule = ch2_chain();
    let mut opts = options("pm7");
    opts.pbc = Some(PbcOptions {
        kmesh: pm7_rs::pbc::KMesh::grid(2, 1, 1),
        ..PbcOptions::default()
    });
    let fc = pm7_rs::force_constants(&molecule, &params("pm7"), &options("pm7"), [2, 1, 1])
        .expect("force constants");
    let supercell = sorted(fc.frequencies_cm([0.5, 0.0, 0.0]).expect("frequencies"));

    let response = pm7_rs::dfpt::dynamical_matrix_dfpt(
        &molecule,
        &params("pm7"),
        &opts,
        [0.5, 0.0, 0.0],
        &pm7_rs::dfpt::DfptOptions::default(),
    )
    .expect("perturbation solve");
    let modes = response.modes().expect("modes");
    let from_dfpt = sorted(modes.frequencies_cm.clone());

    assert_eq!(supercell.len(), from_dfpt.len());
    let mut worst = 0.0_f64;
    for (a, b) in supercell.iter().zip(&from_dfpt) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1.0e-3,
        "the two routes disagree by {worst:.3e} cm^-1 at a commensurate q:\n  supercell \
         {supercell:?}\n  dfpt      {from_dfpt:?}"
    );

    // And `frequencies_cm` is `modes().frequencies_cm` -- the same diagonalization, not a second
    // one that could drift from it.
    let direct = response.frequencies_cm().expect("frequencies");
    assert_eq!(direct, modes.frequencies_cm);
}
