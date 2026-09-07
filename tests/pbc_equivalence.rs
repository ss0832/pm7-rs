// SPDX-License-Identifier: GPL-3.0-or-later
//! Equivalences that a correct periodic implementation must satisfy exactly, or must converge
//! to at a known rate. These are not regression values baked in from a previous run — each one
//! is an identity that follows from the formalism, so a failure points at a specific defect
//! rather than at "the number moved".

use pm7_rs::cell::Cell;
use pm7_rs::math::Vec3;
use pm7_rs::pbc::{KMesh, PbcOptions};
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters};

fn params() -> Pm7Parameters {
    Pm7Parameters::standard().expect("PM7 parameters")
}

/// Water, in Ångström.
fn water() -> Molecule {
    Molecule::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.958400 0.000000 0.000000\nH -0.239987 0.927846 0.000000\n",
        0.0,
    )
    .unwrap()
}

/// A single atom, useful because its periodic energy is pure self-image interaction.
fn lone_atom(z: u8) -> Molecule {
    Molecule::new(vec![pm7_rs::Atom {
        z,
        position: Vec3::zero(),
    }])
}

fn energy(molecule: &Molecule, options: &Pm7Options) -> f64 {
    run_pm7(molecule, &params(), options).expect("SCF").total_ev
}

fn periodic_options(cutoff_angstrom: f64) -> Pm7Options {
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    Pm7Options {
        pbc: Some(PbcOptions {
            short_range_cutoff: 9.0 * a,
            exchange_cutoff: cutoff_angstrom * a,
            correction_cutoff: cutoff_angstrom * a,
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    }
}

#[test]
fn a_molecule_in_a_large_box_converges_to_the_molecular_energy() {
    // The isolated-molecule limit. Everything in the periodic path — the image pair list, the
    // monopole subtraction in the core Hamiltonian and the Fock, the Ewald sum, the
    // `+½ Z·V` energy bookkeeping — has to be right for this to land, and the residual has to
    // *shrink with the box*, not merely be small once.
    let mol = water();
    let reference = energy(&mol, &Pm7Options::default());

    let mut previous = f64::INFINITY;
    for edge_angstrom in [16.0_f64, 24.0, 32.0] {
        let edge = edge_angstrom * pm7_rs::constants::ANGSTROM_TO_BOHR;
        let periodic = mol.clone().with_cell(Cell::cubic(edge).unwrap());
        let e = energy(&periodic, &periodic_options(12.0));
        let residual = (e - reference).abs();
        eprintln!("edge {edge_angstrom:>5.1} A: E = {e:.9} eV, |ΔE| = {residual:.3e} eV");
        assert!(
            residual < previous,
            "residual grew with the box: {residual:.3e} at {edge_angstrom} A vs {previous:.3e} before"
        );
        previous = residual;
    }
    assert!(
        previous < 5.0e-3,
        "32 A box still {previous:.3e} eV from the molecular energy"
    );
}

#[test]
fn a_neutral_atom_in_a_box_has_no_electronic_periodic_energy() {
    // A lone neutral atom carries no net charge and no dipole, so its NDDO interaction with its
    // own images is only the short-ranged remainder — identically zero past 7 A. With PM7-minus
    // (post-SCF corrections off) the periodic and molecular energies must therefore agree to
    // machine precision once the box exceeds twice the feather range. Anything else would mean
    // the image sum, the monopole subtraction, or the Ewald bookkeeping had left a residue.
    let params_minus = Pm7Parameters::method(pm7_rs::Pm7Method::Pm7Minus).unwrap();
    for z in [1_u8, 6, 8] {
        let mol = lone_atom(z).with_multiplicity(if z == 1 { 2 } else { 3 });
        let mut opts = Pm7Options {
            method: pm7_rs::Pm7Method::Pm7Minus,
            multiplicity: mol.multiplicity,
            ..Pm7Options::default()
        };
        let reference = run_pm7(&mol, &params_minus, &opts).unwrap().total_ev;

        let edge = 20.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
        let periodic = mol.clone().with_cell(Cell::cubic(edge).unwrap());
        opts.pbc = Some(PbcOptions::default());
        let e = run_pm7(&periodic, &params_minus, &opts).unwrap().total_ev;
        assert!(
            (e - reference).abs() < 1e-10,
            "Z={z}: periodic {e:.12} vs molecular {reference:.12}"
        );
    }
}

#[test]
fn image_dispersion_is_the_only_residue_for_a_lone_atom() {
    // With PM7's corrections on, the same lone atom *should* differ from its molecular
    // counterpart, because it genuinely disperses against its own images. The residue must
    // therefore be small, negative (dispersion is attractive), and shrink like the R^-6 tail as
    // the box grows — which distinguishes a real physical term from a leftover bug.
    let mol = lone_atom(6).with_multiplicity(3);
    let opts_mol = Pm7Options {
        multiplicity: 3,
        ..Pm7Options::default()
    };
    let reference = energy(&mol, &opts_mol);

    let mut previous = 0.0_f64;
    for edge_angstrom in [12.0_f64, 16.0] {
        let edge = edge_angstrom * pm7_rs::constants::ANGSTROM_TO_BOHR;
        let periodic = mol.clone().with_cell(Cell::cubic(edge).unwrap());
        let opts = Pm7Options {
            multiplicity: 3,
            pbc: Some(PbcOptions::default()),
            ..Pm7Options::default()
        };
        let residue = energy(&periodic, &opts) - reference;
        assert!(
            residue < 0.0,
            "image dispersion should lower the energy, got {residue:+.3e} eV at {edge_angstrom} A"
        );
        if previous != 0.0 {
            // Doubling-ish the box must weaken the R^-6 tail substantially.
            assert!(
                residue.abs() < previous.abs() * 0.5,
                "dispersion residue {residue:+.3e} did not fall off with the box (was {previous:+.3e})"
            );
        }
        previous = residue;
    }
}

#[test]
fn energy_is_invariant_under_translation_by_a_lattice_vector() {
    // Moving an atom by a lattice vector is the identity operation on a periodic system. If the
    // image enumeration or the minimum-image search had an off-by-one, this is where it shows.
    let edge = 12.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let cell = Cell::cubic(edge).unwrap();
    let base = water().with_cell(cell);
    let opts = periodic_options(10.0);
    let reference = energy(&base, &opts);

    for shift in [
        Vec3::new(edge, 0.0, 0.0),
        Vec3::new(0.0, -edge, 0.0),
        Vec3::new(edge, edge, -edge),
    ] {
        let mut moved = base.clone();
        moved.atoms[0].position += shift;
        let e = energy(&moved, &opts);
        assert!(
            (e - reference).abs() < 1e-8,
            "shifting an atom by {shift:?} changed the energy by {:.3e} eV",
            e - reference
        );
    }
}

#[test]
fn energy_is_invariant_under_rigid_translation_of_the_whole_cell() {
    let edge = 12.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let cell = Cell::cubic(edge).unwrap();
    let base = water().with_cell(cell);
    let opts = periodic_options(10.0);
    let reference = energy(&base, &opts);

    let mut moved = base.clone();
    let shift = Vec3::new(1.7, -3.1, 0.9);
    for atom in &mut moved.atoms {
        atom.position += shift;
    }
    let e = energy(&moved, &opts);
    assert!(
        (e - reference).abs() < 1e-9,
        "rigid translation changed the energy by {:.3e} eV",
        e - reference
    );
}

#[test]
fn a_one_dimensional_chain_is_not_the_same_as_its_isolated_unit() {
    // A sanity floor: the 1-D path must actually be doing something. A hydrogen-fluoride chain
    // with 3 A spacing has real inter-unit interaction, so the periodic energy must differ
    // from the isolated molecule by a chemically meaningful amount — and must still be finite,
    // converged, and stable against enlarging the cutoffs.
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    let mol = Molecule::from_xyz_str("2\nHF\nF 0.0 0.0 0.0\nH 0.92 0.0 0.0\n", 0.0).unwrap();
    let isolated = energy(&mol, &Pm7Options::default());

    let chain = mol.with_cell(Cell::new(&[Vec3::new(3.0 * a, 0.0, 0.0)]).unwrap());
    let e_short = energy(&chain, &periodic_options(12.0));
    let e_long = energy(&chain, &periodic_options(20.0));

    assert!(e_short.is_finite() && e_long.is_finite());
    assert!(
        (e_short - e_long).abs() < 1e-4,
        "1-D energy not converged in the cutoff: {e_short:.9} vs {e_long:.9}"
    );
    assert!(
        (e_short - isolated).abs() > 1e-3,
        "the 1-D chain reproduced the isolated molecule exactly, so the periodic path did nothing"
    );
}

#[test]
fn a_charged_cell_runs_and_reports_its_background() {
    // Charged periodic cells are supported; the absolute energy depends on the neutralizing
    // background, which must therefore be reported rather than silently folded in.
    let edge = 14.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let mol = water()
        .with_cell(Cell::cubic(edge).unwrap())
        .with_charge(1.0);
    let opts = Pm7Options {
        charge: 1.0,
        multiplicity: 2,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    let out = run_pm7(&mol, &params(), &opts).expect("charged periodic SCF");
    assert!(out.total_ev.is_finite());
    assert!(out.converged);
    let bg = out
        .background_ev
        .expect("charged cell reports a background");
    assert!(
        bg < 0.0 && bg.is_finite(),
        "jellium background should be a finite negative constant, got {bg}"
    );
    assert!(
        out.makov_payne_ev.expect("Makov-Payne diagnostic") > 0.0,
        "the Makov-Payne estimate of the finite-size error should be positive for a +1 cell"
    );
    // The net Mulliken charge must still integrate to the requested cell charge.
    let q: f64 = out.charges.iter().sum();
    assert!((q - 1.0).abs() < 1e-8, "cell charge came out as {q}");
}

#[test]
fn uhf_is_supported_periodically_on_the_same_terms_as_rhf() {
    // Open-shell periodic systems go through exactly the same path: the periodic core
    // Hamiltonian and the Ewald context are spin-independent (the monopole potential is built
    // from the *total* Mulliken charge), so `−V_A` lands on both spins' Fock diagonals and the
    // `+½ Z·V` energy correction is the same expression as for RHF.
    let edge = 14.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let cell = Cell::cubic(edge).unwrap();

    // A methyl radical in a box: genuinely open shell, so the UHF path is taken.
    let mol = Molecule::from_xyz_str(
        "4\nmethyl\nC 0.0 0.0 0.0\nH 1.0790 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n",
        0.0,
    )
    .unwrap();
    let opts_mol = Pm7Options {
        multiplicity: 2,
        ..Pm7Options::default()
    };
    let molecular = run_pm7(&mol, &params(), &opts_mol).expect("molecular UHF");
    assert!(molecular.unrestricted, "the reference run must be UHF");

    // The open-shell isolated-molecule limit, checked the same way as the closed-shell one: the
    // residual has to *shrink with the box*, not merely be small at one size.
    let opts = Pm7Options {
        multiplicity: 2,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    let mut previous = f64::INFINITY;
    for edge_angstrom in [14.0_f64, 20.0, 28.0] {
        let e = edge_angstrom * pm7_rs::constants::ANGSTROM_TO_BOHR;
        let periodic_mol = mol.clone().with_cell(Cell::cubic(e).unwrap());
        let out = run_pm7(&periodic_mol, &params(), &opts).expect("periodic UHF");
        assert!(out.unrestricted, "the periodic run must also be UHF");
        assert!(out.spin_density.is_some(), "UHF must report a spin density");
        let residual = (out.total_ev - molecular.total_ev).abs();
        eprintln!("UHF edge {edge_angstrom:>5.1} A: |ΔE| = {residual:.3e} eV");
        assert!(
            residual < previous,
            "UHF residual grew with the box: {residual:.3e} at {edge_angstrom} A vs {previous:.3e}"
        );
        previous = residual;
    }
    assert!(
        previous < 5.0e-3,
        "28 A box still {previous:.3e} eV from the molecular UHF energy"
    );

    // And a charged open-shell periodic cell, the combination most likely to expose a
    // spin/charge bookkeeping mistake.
    let charged = mol.with_cell(cell).with_charge(1.0);
    let opts_charged = Pm7Options {
        charge: 1.0,
        multiplicity: 3,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    let out = run_pm7(&charged, &params(), &opts_charged).expect("charged periodic UHF");
    assert!(out.converged && out.unrestricted && out.total_ev.is_finite());
    let q: f64 = out.charges.iter().sum();
    assert!(
        (q - 1.0).abs() < 1e-8,
        "charged UHF cell charge came out as {q}"
    );
}

#[test]
fn unimplemented_periodic_modes_fail_loudly() {
    let edge = 10.0 * pm7_rs::constants::ANGSTROM_TO_BOHR;
    let mol = water().with_cell(Cell::cubic(edge).unwrap());
    let opts = Pm7Options {
        pbc: Some(PbcOptions::mopac_compatible()),
        ..Pm7Options::default()
    };
    assert!(
        run_pm7(&mol, &params(), &opts).is_err(),
        "MOPAC-compatibility mode should be refused until it is implemented"
    );
}

/// A 1-D hydrogen-fluoride chain — small enough to supercell cheaply, polar enough that the
/// electrostatics is not trivial.
fn hf_chain(period_angstrom: f64) -> Molecule {
    let a = pm7_rs::constants::ANGSTROM_TO_BOHR;
    Molecule::from_xyz_str("2\nHF\nF 0.0 0.0 0.0\nH 0.92 0.0 0.0\n", 0.0)
        .unwrap()
        .with_cell(Cell::new(&[Vec3::new(period_angstrom * a, 0.0, 0.0)]).unwrap())
}

/// Repeat a periodic system `n` times along its first lattice vector.
fn supercell_along_first(molecule: &Molecule, n: usize) -> Molecule {
    let cell = molecule.cell.expect("periodic");
    let (super_cell, shifts) = cell.supercell([n, 1, 1]).unwrap();
    let mut atoms = Vec::with_capacity(molecule.atoms.len() * n);
    for shift in &shifts {
        let t = cell.translation(*shift);
        for atom in &molecule.atoms {
            atoms.push(pm7_rs::Atom {
                z: atom.z,
                position: atom.position + t,
            });
        }
    }
    Molecule::new(atoms).with_cell(super_cell)
}

#[test]
fn a_one_by_one_by_one_mesh_reproduces_the_gamma_point() {
    // `KMesh::Gamma` and `KMesh::grid(1,1,1)` are the same physical model, so they must give the
    // same number — the k-point path and the Γ path have to agree where they overlap, or one of
    // them is doing something the other is not.
    let mol = hf_chain(3.0);
    // The k path damps linearly rather than using DIIS, so it needs a tighter density tolerance
    // to reach the same energy precision as the Γ path; without that the comparison measures the
    // convergence criterion instead of the physics.
    let gamma = Pm7Options {
        p_tol: 1.0e-10,
        pbc: Some(PbcOptions::default()),
        ..Pm7Options::default()
    };
    let mesh = Pm7Options {
        p_tol: 1.0e-10,
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(1, 1, 1),
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    };
    let a = run_pm7(&mol, &params(), &gamma).expect("Gamma");
    let b = run_pm7(&mol, &params(), &mesh).expect("1x1x1 mesh");
    assert!(b.converged, "the 1x1x1 mesh did not converge");
    assert!(
        (a.total_ev - b.total_ev).abs() < 1e-8,
        "Gamma {:.9} vs 1x1x1 mesh {:.9} (difference {:.2e})",
        a.total_ev,
        b.total_ev,
        (a.total_ev - b.total_ev).abs()
    );
    assert_eq!(b.n_kpoints, Some(1));
}

#[test]
fn born_von_karman_identity_holds_for_a_one_dimensional_chain() {
    // The sharpest available test of the k-point machinery: an `n×1×1` k mesh on a cell must
    // give the same energy *per unit cell* as the Γ point of the `n×1×1` supercell. The two
    // calculations share no code path — one folds phases over a k mesh, the other enumerates
    // more atoms — so agreement pins the phase conventions, the weights, the time-reversal
    // folding, and the real-space back-transform all at once.
    let mol = hf_chain(3.2);
    for n in [2_usize, 3] {
        let mesh = Pm7Options {
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(n, 1, 1),
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        let per_cell = run_pm7(&mol, &params(), &mesh).expect("k mesh");
        assert!(per_cell.converged, "n={n}: k-mesh SCF did not converge");

        let big = supercell_along_first(&mol, n);
        let gamma = Pm7Options {
            pbc: Some(PbcOptions::default()),
            ..Pm7Options::default()
        };
        let supercell = run_pm7(&big, &params(), &gamma).expect("supercell Gamma");
        assert!(supercell.converged, "n={n}: supercell SCF did not converge");

        let expected = supercell.total_ev / n as f64;
        eprintln!(
            "n={n}: k-mesh {:.9} eV/cell, supercell/{n} {:.9} eV/cell, diff {:.3e}",
            per_cell.total_ev,
            expected,
            per_cell.total_ev - expected
        );
        assert!(
            (per_cell.total_ev - expected).abs() < 2.0e-3,
            "n={n}: BvK identity violated — k mesh gives {:.9} eV/cell but the {n}x supercell \
             gives {expected:.9} eV/cell",
            per_cell.total_ev
        );
    }
}

#[test]
fn a_k_mesh_converges_and_reports_bands() {
    let mol = hf_chain(3.2);
    let opts = Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(6, 1, 1),
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    };
    let out = run_pm7(&mol, &params(), &opts).expect("k mesh");
    assert!(out.converged);
    let bands = out.band_energies.expect("k-point runs report bands");
    let nk = out.n_kpoints.expect("k-point count");
    assert_eq!(bands.len(), nk, "one band list per k point");
    // Time-reversal folding must have reduced a 6-point mesh below 6.
    assert!(nk < 6, "6x1x1 mesh kept {nk} points; folding did nothing");
    for e in &bands {
        assert!(
            e.windows(2).all(|w| w[0] <= w[1] + 1e-12),
            "bands not sorted"
        );
        assert!(e.iter().all(|v| v.is_finite()));
    }
    assert!(out.fermi_ev.expect("Fermi level").is_finite());
    assert_eq!(
        out.entropy_ev,
        Some(0.0),
        "no smearing means no entropy term"
    );
}

#[test]
fn smearing_runs_and_reports_a_negative_entropy() {
    let mol = hf_chain(3.2);
    let opts = Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: KMesh::grid(4, 1, 1),
            smearing: pm7_rs::pbc::Smearing::FermiDirac { width_ev: 0.2 },
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    };
    let out = run_pm7(&mol, &params(), &opts).expect("smeared k mesh");
    assert!(out.converged);
    let s = out.entropy_ev.expect("entropy");
    assert!(
        s <= 0.0 && s.is_finite(),
        "smearing entropy {s} should be finite and non-positive"
    );
}

/// The electronic free energy is the internal energy **plus** the entropy term, and equals it
/// exactly without smearing.
///
/// `entropy_ev` has been reported since v0.2.0 "so the free energy and the internal energy stay
/// distinguishable" (`docs/pbc.md`), but nothing ever assembled the free energy: the ASE
/// calculator set `free_energy` to the internal energy, so with smearing on it returned a number
/// that is not what the forces differentiate. This pins both halves — that the sum is formed, and
/// that forming it moves nothing when there is no entropy to add.
#[test]
fn the_free_energy_is_the_internal_energy_plus_the_entropy_term() {
    let mol = hf_chain(3.2);
    let mesh = || KMesh::grid(4, 1, 1);

    let cold = Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: mesh(),
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    };
    let out = run_pm7(&mol, &params(), &cold).expect("aufbau k mesh");
    assert_eq!(
        out.free_energy_ev().to_bits(),
        out.total_ev.to_bits(),
        "without smearing the free energy must be the internal energy, bit for bit"
    );

    let warm = Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: mesh(),
            smearing: pm7_rs::pbc::Smearing::FermiDirac { width_ev: 0.5 },
            ..PbcOptions::default()
        }),
        ..Pm7Options::default()
    };
    let out = run_pm7(&mol, &params(), &warm).expect("smeared k mesh");
    let entropy = out.entropy_ev.expect("entropy");
    assert_eq!(
        out.free_energy_ev().to_bits(),
        (out.total_ev + entropy).to_bits(),
        "the free energy must be E + (-TS) exactly"
    );
    // The identity above holds trivially if the entropy is zero, which is the failure mode that
    // would let the whole thing pass on a gapped system. Insist that this fixture actually smears.
    assert!(
        entropy < -1.0e-9,
        "fixture does not smear: entropy {entropy} is not negative, so the test is vacuous"
    );
    assert!(
        out.free_energy_ev() < out.total_ev,
        "the entropy term lowers the free energy"
    );
}

/// **Per-axis periodicity.** A slab is the same system whichever lattice slot its open direction
/// sits in, so all three two-periodic patterns must give one energy — and likewise all three
/// one-periodic patterns.
///
/// Through 0.2.2 there was no way to ask: `Cell::from_flags`, `native`'s `pbc=` and the ASE
/// calculator each refused anything but a leading pattern, so an `Atoms` object built as a slab
/// along *y* — an ordinary thing for an ASE user to have — was rejected rather than run. The fix
/// is a cyclic rotation of the lattice vectors, and this is what says the rotation is a
/// relabelling and not a change of system.
#[test]
fn every_axis_pattern_describes_the_same_slab() {
    let molecule = Molecule::from_xyz_str("2\nx\nC 0.0 0.0 0.0\nC 0.7 0.7 0.7\n", 0.0).unwrap();
    let options = periodic_options(7.0);

    // The same physical lattice — x and y periodic at 2.5 Å, z open at 12 Å — written with its
    // open direction in each of the three slots.
    let slabs = [
        (
            [true, true, false],
            [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 12.0]],
        ),
        (
            [true, false, true],
            [[2.5, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 2.5, 0.0]],
        ),
        (
            [false, true, true],
            [[0.0, 0.0, 12.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]],
        ),
    ];
    let mut energies = Vec::new();
    for (pbc, rows) in slabs {
        let (cell, rotation) = Cell::from_angstrom_rows_pbc(&rows, pbc).expect("cell");
        let cell = cell.expect("a slab is periodic");
        assert_eq!(cell.dim(), 2, "{pbc:?} is a slab");
        // The rotation is what the caller has to put its own per-lattice-vector inputs through.
        assert_eq!(rotation.apply(pbc), [true, true, false], "{pbc:?}");
        energies.push(energy(&molecule.clone().with_cell(cell), &options));
    }
    for (k, e) in energies.iter().enumerate().skip(1) {
        assert!(
            (e - energies[0]).abs() < 1.0e-9,
            "slab pattern {k} gave {e} against {} for the leading one",
            energies[0]
        );
    }

    // And the same for a chain: one periodic direction, in each of the three slots.
    let chains = [
        (
            [true, false, false],
            [[2.5, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ),
        (
            [false, true, false],
            [[0.0, 10.0, 0.0], [2.5, 0.0, 0.0], [0.0, 0.0, 10.0]],
        ),
        (
            [false, false, true],
            [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [2.5, 0.0, 0.0]],
        ),
    ];
    let mut energies = Vec::new();
    for (pbc, rows) in chains {
        let (cell, _) = Cell::from_angstrom_rows_pbc(&rows, pbc).expect("cell");
        let cell = cell.expect("a chain is periodic");
        assert_eq!(cell.dim(), 1, "{pbc:?} is a chain");
        energies.push(energy(&molecule.clone().with_cell(cell), &options));
    }
    for (k, e) in energies.iter().enumerate().skip(1) {
        assert!(
            (e - energies[0]).abs() < 1.0e-9,
            "chain pattern {k} gave {e} against {} for the leading one",
            energies[0]
        );
    }
}

/// An all-false pattern is a molecule, not a degenerate cell.
#[test]
fn no_periodic_direction_means_no_cell() {
    let rows = [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 12.0]];
    let (cell, rotation) = Cell::from_angstrom_rows_pbc(&rows, [false; 3]).expect("no cell");
    assert!(cell.is_none(), "nothing periodic must yield no cell");
    assert!(rotation.is_identity());
}

/// The extended-XYZ `pbc=` key accepts a non-leading pattern, and reading one back gives the
/// same system. Reading is the surface an ASE user actually reaches: `atoms.write("x.xyz")` on a
/// slab built along *y* writes `pbc="T F T"`, and that file used to be rejected on load.
#[test]
fn a_non_leading_pbc_key_round_trips_through_extended_xyz() {
    let text = "2\nLattice=\"2.5 0 0 0 0 12.0 0 2.5 0\" \
                Properties=species:S:1:pos:R:3 pbc=\"T F T\"\n\
                C 0.0 0.0 0.0\nC 0.7 0.7 0.7\n";
    let (molecule, rotation) = Molecule::from_xyz_str_with_axes(text, 0.0).expect("parse");
    let cell = molecule.cell.expect("periodic");
    assert_eq!(cell.dim(), 2);
    assert_eq!(rotation.order(), [2, 0, 1], "a3 leads, then a1, then a2");

    let leading = "2\nLattice=\"2.5 0 0 0 2.5 0 0 0 12.0\" \
                   Properties=species:S:1:pos:R:3 pbc=\"T T F\"\n\
                   C 0.0 0.0 0.0\nC 0.7 0.7 0.7\n";
    let reference = Molecule::from_xyz_str(leading, 0.0).expect("parse");
    let options = periodic_options(7.0);
    let (a, b) = (energy(&molecule, &options), energy(&reference, &options));
    assert!(
        (a - b).abs() < 1.0e-9,
        "pbc=\"T F T\" gave {a}, its reordered twin {b}"
    );
}
