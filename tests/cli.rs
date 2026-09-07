// SPDX-License-Identifier: GPL-3.0-or-later
//! The command-line interface, exercised as a subprocess.
//!
//! The CLI is the one surface with no type checking between it and the user: a renamed flag or a
//! mode that stopped being wired compiles perfectly and fails only when someone runs it. These
//! tests run the real binary and read its real output.

use std::path::PathBuf;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_pm7_rs_cli")
}

fn scratch(name: &str, contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("pm7_cli_{name}"));
    std::fs::write(&path, contents).expect("write scratch file");
    path
}

/// A diamond primitive cell as extended XYZ, so the `Lattice="..."` path is exercised too.
fn diamond() -> PathBuf {
    scratch(
        "diamond.xyz",
        "2\n\
         Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" \
         Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n\
         C 0.000000 0.000000 0.000000\n\
         C 0.891750 0.891750 0.891750\n",
    )
}

fn water() -> PathBuf {
    scratch(
        "water.xyz",
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
    )
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(binary())
        .args(args)
        .output()
        .expect("run pm7_rs_cli");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_molecular_single_point_reports_the_heat_of_formation() {
    let path = water();
    let (ok, stdout, stderr) = run(&["energy", path.to_str().unwrap(), "--json"]);
    assert!(ok, "energy failed: {stderr}");
    assert!(
        stdout.contains("\"heat_of_formation_kcal\":-57.78"),
        "unexpected JSON: {stdout}"
    );
}

#[test]
fn a_periodic_single_point_reads_the_lattice_from_the_file_and_takes_a_k_mesh() {
    let path = diamond();
    let (ok, stdout, stderr) = run(&[
        "energy",
        path.to_str().unwrap(),
        "--kpoints",
        "2",
        "2",
        "2",
        "--json",
    ]);
    assert!(ok, "periodic energy failed: {stderr}");
    // Eight points before folding; the count reported is what was actually diagonalized.
    assert!(
        stdout.contains("\"n_kpoints\""),
        "no k-point count: {stdout}"
    );
    assert!(stdout.contains("\"fermi_ev\""), "no Fermi level: {stdout}");
}

#[test]
fn stress_reports_voigt_order_and_a_pressure() {
    let path = diamond();
    let (ok, stdout, stderr) = run(&["stress", path.to_str().unwrap(), "--kpoints", "2", "2", "2"]);
    assert!(ok, "stress failed: {stderr}");
    assert!(stdout.contains("Voigt [xx yy zz yz xz xy]"), "{stdout}");
    // Diamond under its own equilibrium mismatch: a real number, and cubic symmetry means the
    // three shear components vanish.
    let voigt: Vec<f64> = stdout
        .lines()
        .find(|l| l.starts_with("Voigt"))
        .and_then(|l| l.split(':').nth(1))
        .expect("a Voigt line")
        .split_whitespace()
        .map(|v| v.parse().expect("a number"))
        .collect();
    assert_eq!(voigt.len(), 6);
    assert!(
        (voigt[0] - voigt[1]).abs() < 1e-6 && (voigt[1] - voigt[2]).abs() < 1e-6,
        "a cubic cell must have equal diagonal stress: {voigt:?}"
    );
    for shear in &voigt[3..] {
        assert!(shear.abs() < 1e-6, "cubic shear should vanish: {voigt:?}");
    }
    assert!(stdout.contains("pressure:"), "{stdout}");
}

#[test]
fn phonons_reports_the_acoustic_residual_and_frequencies_at_each_q() {
    let path = diamond();
    let (ok, stdout, stderr) = run(&[
        "phonons",
        path.to_str().unwrap(),
        "--supercell",
        "2",
        "2",
        "2",
        "--qpoints",
        "0,0,0",
        "0.5,0,0",
        "--acoustic-sum-rule",
    ]);
    assert!(ok, "phonons failed: {stderr}");
    assert!(stdout.contains("acoustic sum-rule residual"), "{stdout}");
    assert!(
        stdout.contains("q = (0.000000, 0.000000, 0.000000)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("q = (0.500000, 0.000000, 0.000000)"),
        "{stdout}"
    );
    // With the sum rule imposed, the three zone-centre acoustic modes must be zero.
    let gamma_block: Vec<f64> = stdout
        .lines()
        .skip_while(|l| !l.starts_with("q = (0.000000"))
        .skip(1)
        .take(6)
        .map(|l| {
            l.split_whitespace()
                .next()
                .unwrap()
                .parse()
                .expect("a frequency")
        })
        .collect();
    assert_eq!(gamma_block.len(), 6);
    for acoustic in &gamma_block[..3] {
        assert!(
            acoustic.abs() < 1.0,
            "acoustic mode should be zero after the projection: {gamma_block:?}"
        );
    }
    assert!(
        gamma_block[5] > 1000.0,
        "diamond's optical mode should be over 1000 cm^-1: {gamma_block:?}"
    );
}

#[test]
fn bands_diagonalizes_the_converged_fock_along_a_path() {
    let path = diamond();
    let (ok, stdout, stderr) = run(&[
        "bands",
        path.to_str().unwrap(),
        "--kpoints",
        "2",
        "2",
        "2",
        "--qpoints",
        "0,0,0",
        "0.5,0,0",
    ]);
    assert!(ok, "bands failed: {stderr}");
    assert!(stdout.contains("Fermi level"), "{stdout}");
    assert!(
        stdout.contains("k = (0.000000, 0.000000, 0.000000)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("k = (0.500000, 0.000000, 0.000000)"),
        "{stdout}"
    );
}

#[test]
fn the_cell_flag_overrides_and_accepts_every_periodicity() {
    // A 1-D chain given entirely on the command line: three numbers, not nine.
    let path = scratch("chain.xyz", "2\nH2 chain\nH 0.0 0.0 0.0\nH 0.76 0.0 0.0\n");
    let (ok, stdout, stderr) = run(&["energy", path.to_str().unwrap(), "--cell", "3.2,0,0"]);
    assert!(ok, "1-D --cell failed: {stderr}");
    assert!(stdout.contains("total energy"), "{stdout}");
}

#[test]
fn divide_and_conquer_is_reachable_from_the_command_line() {
    let path = water();
    let (ok, stdout, stderr) = run(&["energy", path.to_str().unwrap(), "--dandc", "15.0"]);
    assert!(ok, "--dandc failed: {stderr}");
    assert!(stdout.contains("divide and conquer"), "{stdout}");
    assert!(stdout.contains("subsystems:"), "{stdout}");
}

#[test]
fn a_periodic_mode_on_a_molecule_explains_itself() {
    let path = water();
    let (ok, _stdout, stderr) = run(&["stress", path.to_str().unwrap()]);
    assert!(!ok, "stress on a molecule should fail");
    // The message has to say what to do, not just that something is missing.
    assert!(
        stderr.contains("needs a periodic cell") && stderr.contains("--cell"),
        "unhelpful error: {stderr}"
    );
}

#[test]
fn an_unknown_flag_is_rejected_rather_than_ignored() {
    let path = water();
    let (ok, _stdout, stderr) = run(&["energy", path.to_str().unwrap(), "--kpoint", "2"]);
    assert!(!ok, "a misspelled flag must not be silently accepted");
    assert!(stderr.contains("unknown option"), "{stderr}");
}

/// The frontier markers are the whole point of the human-readable form: an orbital list
/// without them is a column of numbers.
#[test]
fn orbitals_reports_the_frontier_and_the_occupations() {
    let path = water();
    let (ok, out, err) = run(&["orbitals", path.to_str().unwrap()]);
    assert!(ok, "{err}");
    assert!(out.contains("<-- HOMO"), "{out}");
    assert!(out.contains("<-- LUMO"), "{out}");
    assert!(out.contains("orbitals reported at: molecular"), "{out}");

    let (ok, json, err) = run(&["orbitals", path.to_str().unwrap(), "--json"]);
    assert!(ok, "{err}");
    assert!(json.contains("\"mo_coefficients\""), "{json}");
    assert!(json.contains("\"n_occ\":4"), "{json}");
}

/// `--ir` must take the spectrum path, which produces frequencies and intensities from one
/// CPHF solve rather than running the Hessian twice.
#[test]
fn frequencies_with_ir_reports_intensities() {
    let path = water();
    let (ok, out, err) = run(&["frequencies", path.to_str().unwrap(), "--ir"]);
    assert!(ok, "{err}");
    assert!(out.contains("IR (km/mol)"), "{out}");
    assert!(out.contains("DIPT"), "{out}");

    let (ok, json, err) = run(&["frequencies", path.to_str().unwrap(), "--ir", "--json"]);
    assert!(ok, "{err}");
    assert!(json.contains("\"ir_intensities_km_per_mol\""), "{json}");
    assert!(json.contains("\"dipole_derivatives_e\""), "{json}");
}

/// A field reaches the CLI, changes the energy, and reports its own contribution separately.
#[test]
fn an_external_field_changes_the_energy_and_is_reported() {
    let path = water();
    let (ok, plain, err) = run(&["energy", path.to_str().unwrap(), "--json"]);
    assert!(ok, "{err}");
    let (ok, fielded, err) = run(&[
        "energy",
        path.to_str().unwrap(),
        "--field",
        "0.5,0,0",
        "--json",
    ]);
    assert!(ok, "{err}");
    assert!(fielded.contains("\"field_ev\""), "{fielded}");
    assert!(plain != fielded, "the field must change the answer");
    assert!(!plain.contains("\"field_ev\""), "no field, no key");
}

/// `forces` is exactly the negated `gradient`, to the last printed digit.
///
/// Both exist because a force is what an optimizer wants and a gradient is what a derivative
/// check wants, and a sign error between them is silent in exactly the cases that matter.
#[test]
fn forces_are_the_negated_gradient() {
    let path = water();
    let (ok, gradient, err) = run(&["gradient", path.to_str().unwrap()]);
    assert!(ok, "gradient failed: {err}");
    let (ok, forces, err) = run(&["forces", path.to_str().unwrap()]);
    assert!(ok, "forces failed: {err}");

    let numbers = |text: &str| -> Vec<f64> {
        text.split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let g = numbers(&gradient);
    let f = numbers(&forces);
    assert_eq!(g.len(), 9, "water has nine components");
    assert_eq!(g.len(), f.len());
    for (a, b) in g.iter().zip(&f) {
        assert!(
            (a + b).abs() < 1.0e-12,
            "gradient {a} and force {b} are not opposite"
        );
    }
    assert!(
        g.iter().any(|v| v.abs() > 1.0e-6),
        "an all-zero gradient would pass vacuously"
    );
}

/// `hessian` prints a square, symmetric matrix of the right size.
#[test]
fn the_hessian_mode_prints_a_symmetric_matrix() {
    let path = water();
    let (ok, stdout, err) = run(&["hessian", path.to_str().unwrap()]);
    assert!(ok, "hessian failed: {err}");
    let rows: Vec<Vec<f64>> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), 9, "3N = 9 rows for water");
    for row in &rows {
        assert_eq!(row.len(), 9, "3N = 9 columns");
    }
    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert!(
                (value - rows[j][i]).abs() < 1.0e-6,
                "the Hessian must be symmetric at ({i},{j})"
            );
        }
    }
}

/// `molden` writes a file a reader can parse, and `--molden-basis sto` switches representation.
#[test]
fn the_molden_mode_writes_both_basis_forms() {
    let path = water();
    let (ok, gto, err) = run(&["molden", path.to_str().unwrap()]);
    assert!(ok, "molden failed: {err}");
    assert!(
        gto.starts_with("[Molden Format]"),
        "{}",
        &gto[..40.min(gto.len())]
    );
    for section in ["[Title]", "[Atoms] AU", "[GTO]", "[MO]"] {
        assert!(gto.contains(section), "missing {section}");
    }
    // The caveat travels with the file, not only with the documentation.
    assert!(
        gto.contains("orthonormal AO basis"),
        "the NDDO caveat is missing"
    );

    let (ok, sto, err) = run(&["molden", path.to_str().unwrap(), "--molden-basis", "sto"]);
    assert!(ok, "molden --molden-basis sto failed: {err}");
    assert!(
        sto.contains("[STO]") && sto.contains("[Atoms] Angs"),
        "STO files are Angstrom"
    );
    assert!(!sto.contains("[GTO]"));

    let (ok, _, err) = run(&[
        "molden",
        path.to_str().unwrap(),
        "--molden-basis",
        "nonsense",
    ]);
    assert!(!ok, "an unknown basis should be refused");
    assert!(
        err.contains("sto-nG"),
        "the error should say what is accepted: {err}"
    );
}

/// `dfpt` reaches the CLI, and refuses a molecule rather than inventing a box.
#[test]
fn the_dfpt_mode_runs_periodic_and_refuses_a_molecule() {
    let (ok, stdout, err) = run(&[
        "dfpt",
        diamond().to_str().unwrap(),
        "--kpoints",
        "2",
        "2",
        "2",
        "--qpoints",
        "0.3,-0.15,0.42",
    ]);
    assert!(ok, "dfpt failed: {err}");
    assert!(stdout.contains("cm^-1"), "{stdout}");

    let (ok, _, err) = run(&["dfpt", water().to_str().unwrap()]);
    assert!(!ok, "dfpt on a molecule should be refused");
    assert!(err.contains("periodic cell"), "{err}");
}

/// `born` reports Born charges that obey the acoustic sum rule.
#[test]
fn the_born_mode_reports_charges_and_the_sum_rule() {
    let (ok, stdout, err) = run(&[
        "born",
        diamond().to_str().unwrap(),
        "--kpoints",
        "2",
        "2",
        "2",
    ]);
    assert!(ok, "born failed: {err}");
    assert!(stdout.contains("acoustic sum rule residual"), "{stdout}");
    assert!(stdout.contains("dielectric tensor"), "{stdout}");

    let (ok, _, err) = run(&["born", water().to_str().unwrap()]);
    assert!(!ok, "born on a molecule should be refused");
    assert!(err.contains("periodic cell"), "{err}");
}

/// `--reference` reaches the SCF: forcing UHF on a closed shell is legal and changes nothing
/// physical, while forcing RHF on an odd electron count is refused rather than silently ignored.
#[test]
fn the_reference_flag_reaches_the_scf() {
    let path = water();
    let (ok, auto, err) = run(&["energy", path.to_str().unwrap(), "--json"]);
    assert!(ok, "{err}");
    let (ok, uhf, err) = run(&[
        "energy",
        path.to_str().unwrap(),
        "--reference",
        "uhf",
        "--json",
    ]);
    assert!(ok, "forcing UHF on a closed shell is legal: {err}");
    assert!(auth_energy(&auto).is_finite() && auth_energy(&uhf).is_finite());
    assert!(
        (auth_energy(&auto) - auth_energy(&uhf)).abs() < 1.0e-6,
        "a closed-shell UHF solution should collapse onto the RHF one"
    );

    let (ok, _, err) = run(&["energy", path.to_str().unwrap(), "--reference", "nonsense"]);
    assert!(!ok, "an unknown reference should be refused");
    assert!(err.contains("auto, rhf or uhf"), "{err}");
}

/// A file with a UTF-8 byte-order mark parses, and gives the same answer.
///
/// The BOM is invisible in every editor, so the failure it caused was "invalid XYZ atom count: 3"
/// pointing at a line that plainly reads `3` — the user is told the wrong thing about the wrong
/// character. And it is not an exotic file: on Windows, PowerShell's `Set-Content -Encoding UTF8`
/// and Notepad's "UTF-8" both write one, which makes this the ordinary way to produce an XYZ file
/// on the platform this crate is developed on. It cost a measurement in this very session.
#[test]
fn a_byte_order_mark_does_not_break_a_structure_file() {
    let plain = water();
    let (ok, expected, stderr) = run(&["energy", plain.to_str().unwrap(), "--json"]);
    assert!(ok, "the plain file failed: {stderr}");

    let marked = scratch(
        "water_bom.xyz",
        "\u{feff}3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
    );
    let (ok, stdout, stderr) = run(&["energy", marked.to_str().unwrap(), "--json"]);
    assert!(ok, "a BOM-prefixed XYZ file was refused: {stderr}");
    assert_eq!(
        auth_energy(&stdout),
        auth_energy(&expected),
        "the BOM changed the answer"
    );
}

/// Pull `heat_of_formation_kcal` out of the JSON without a JSON parser.
fn auth_energy(json: &str) -> f64 {
    let key = "\"heat_of_formation_kcal\":";
    let start = json.find(key).expect("no heat of formation in the output") + key.len();
    let rest = &json[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse()
        .expect("unparseable heat of formation")
}

/// Escaping to UHF says so on stderr, and reports `<S^2>` on stdout.
///
/// **A silent change of spin reference is the failure mode this guards.** `--stability follow` on
/// stretched H₂ returns an unrestricted answer where the caller asked for a restricted one; it is
/// the lower and the correct one, but it is also a different model, and a geometry scan that
/// switches partway through is discontinuous exactly there. `PM7_QUIET` silences the warning
/// without changing the number, the same contract every other warning in this crate has.
#[test]
fn following_a_triplet_instability_warns_that_the_answer_became_unrestricted() {
    let path = scratch(
        "h2_stretched.xyz",
        "2\nstretched H2\nH 0.0 0.0 0.0\nH 0.0 0.0 2.5\n",
    );
    let stretched = path.to_str().unwrap();

    let (ok, out, err) = run(&["energy", stretched, "--stability", "follow"]);
    assert!(ok, "energy --stability follow failed: {err}");
    for expected in ["UNRESTRICTED", "RHF->UHF", "broken-symmetry", "PM7_QUIET"] {
        assert!(
            err.contains(expected),
            "the RHF->UHF switch must be reported and name {expected:?}; stderr was:\n{err}"
        );
    }
    assert!(
        out.contains("<S^2>") && out.contains("(SZ)"),
        "an unrestricted answer must report its spin contamination; stdout was:\n{out}"
    );

    // Silenced, but not changed: the same energy line, and no warning.
    let quiet = Command::new(binary())
        .args(["energy", stretched, "--stability", "follow"])
        .env("PM7_QUIET", "1")
        .output()
        .expect("run pm7_rs_cli");
    let quiet_err = String::from_utf8_lossy(&quiet.stderr).into_owned();
    assert!(
        !quiet_err.contains("UNRESTRICTED"),
        "PM7_QUIET must silence the switch warning; stderr was:\n{quiet_err}"
    );
    let quiet_out = String::from_utf8_lossy(&quiet.stdout).into_owned();
    let energy = |text: &str| {
        text.lines()
            .find(|l| l.starts_with("total energy:"))
            .expect("an energy line")
            .to_string()
    };
    assert_eq!(
        energy(&out),
        energy(&quiet_out),
        "silencing a warning must not move the answer"
    );

    // And `check` reports without moving: the restricted energy, unchanged.
    let (ok, checked, err) = run(&["energy", stretched, "--stability", "check"]);
    assert!(ok, "energy --stability check failed: {err}");
    assert!(
        !err.contains("UNRESTRICTED"),
        "`check` must not switch the reference, and so must not warn; stderr was:\n{err}"
    );
    assert!(
        !checked.contains("<S^2>"),
        "`check` leaves the restricted solution in place, which has no <S^2> to report"
    );
    assert_ne!(
        energy(&checked),
        energy(&out),
        "`follow` must reach a different (lower) solution than `check` leaves standing"
    );
}
