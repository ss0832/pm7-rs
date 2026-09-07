// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line interface for PM7: single points, gradients, stress, optimization, frequencies,
//! phonons and band structures, for molecules and for periodic cells.
//!
//! A periodic run needs a cell. It comes from the structure file when that is an extended XYZ with
//! a `Lattice="..."` key — which is what ASE writes — or from `--cell`, which overrides it. Without
//! either, every mode runs molecular and the periodic-only ones say so rather than inventing a box.

use pm7_rs::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM};
use pm7_rs::{
    analytic_stress, band_structure, closed_form_gradient, force_constants, optimize, run_dandc,
    run_pm7, Cell, DandcOptions, KMesh, Molecule, OptOptions, PbcMode, PbcOptions, Pm7Method,
    Pm7Options, Pm7Parameters, Smearing,
};
use std::process::exit;
use std::str::FromStr;

fn main() {
    if let Err(error) = run() {
        eprintln!("pm7-rs: {error}");
        exit(1);
    }
}

/// Everything the flags can set, before it is turned into a [`Pm7Options`].
struct Cli {
    mode: String,
    path: String,
    charge: f64,
    multiplicity: usize,
    method: Pm7Method,
    opt_output: Option<String>,
    json: bool,
    use_diis: bool,
    /// SCF density-convergence tolerance. Present on every PyO3 signature and on `PM7(...)` since
    /// 0.2.0, and on neither command line until 0.2.3 -- so a stalling periodic cell could be
    /// diagnosed from Python and not from here.
    scf_tolerance: Option<f64>,
    max_scf: Option<usize>,
    /// Operator applications one CPHF solve may spend. A private constant through 0.2.2, so an
    /// ill-conditioned cell could only be told to use a different mode.
    cphf_max_iterations: Option<usize>,
    /// Whether to ask if the converged SCF solution is a minimum, and escape it if not.
    stability: pm7_rs::stability::ScfStability,
    exchange_cutoff: Option<(f64, f64)>,
    cell: Option<Cell>,
    /// The full 3x3 lattice, kept only when `--cell` was given all nine numbers. `--pbc` selects
    /// among three lattice vectors, so it needs all three; three or six numbers have already
    /// committed to a leading pattern.
    cell_rows: Option<[[f64; 3]; 3]>,
    /// `--pbc TFT`: which lattice directions are periodic, in the user's axis order.
    pbc: Option<[bool; 3]>,
    /// Where a charged species' reported dipole is measured from. Present on the Python CLI since
    /// 0.2.1 and on this one only from 0.2.3, so the same molecule gave two different dipoles
    /// depending on which command line asked.
    dipole_origin: pm7_rs::DipoleOrigin,
    /// How the lattice vectors were reordered to put the periodic ones first. Everything the user
    /// indexes by lattice vector (`--kpoints`, `--kshift`, `--supercell`, fractional `--qpoints`)
    /// is given in *their* order and has to be rotated into this one.
    axes: pm7_rs::AxisRotation,
    kpoints: Option<[usize; 3]>,
    kshift: [f64; 3],
    smearing: Option<Smearing>,
    pbc_mode: PbcMode,
    dandc: Option<f64>,
    /// `--opt-cell`: relax the lattice as well as the atoms. Opt-in, because turning it on by
    /// default would move every published periodic `optimize` number.
    opt_cell: bool,
    /// `--gtol`, `--stress-tol`, `--opt-max-iter`. None of `OptOptions` was reachable from either
    /// command line through 0.2.2.
    gtol: Option<f64>,
    stress_tol: Option<f64>,
    opt_max_iter: Option<usize>,
    /// `--stability-every N`: re-run the stability analysis every `N`th optimizer step. A geometry
    /// step changes the orbitals, so a solution that was a minimum at the start can stop being one
    /// on the way. `0` (the default) never checks.
    stability_every: Option<usize>,
    supercell: [usize; 3],
    qpoints: Vec<[f64; 3]>,
    /// Enforce the acoustic sum rule on the force constants. **On by default since 0.2.3**: a
    /// uniform translation of the whole crystal costs no energy, so the residual is numerical
    /// noise, and leaving it in reports a non-zero acoustic frequency at the zone centre. On
    /// diamond it is the difference between `0.0001` and `-0.0000 cm⁻¹`. `--no-acoustic-sum-rule`
    /// keeps the raw force constants.
    acoustic_sum_rule: bool,
    field: Option<pm7_rs::ExternalField>,
    ir: bool,
    /// What `frequencies` removes before diagonalizing. The default projects out the rigid-body
    /// subspace, so water reports its three vibrations; `--projection none` is the opt-out that
    /// returns the raw 3N set, for seeing what was removed and how far from zero it was.
    projection: pm7_rs::Projection,
    reference: pm7_rs::ScfReference,
    molden_basis: String,
    lo_to: Option<[f64; 3]>,
    slab_thickness: Option<f64>,
    wire_cross_section: Option<f64>,
}

fn run() -> pm7_rs::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        print_usage();
        exit(2);
    }
    let mut cli = parse_args(&args)?;

    // The file may itself carry `pbc="T F T"`, in which case the parser has already rotated the
    // lattice vectors and hands back the rotation it used.
    let (mut molecule, mut axes) = Molecule::from_xyz_file_with_axes(&cli.path, cli.charge)?;
    if let Some(cell) = cli.cell {
        molecule = molecule.with_cell(cell);
        axes = pm7_rs::AxisRotation::IDENTITY;
    }
    if let Some(pbc) = cli.pbc {
        let (cell, rotation) = apply_pbc_flag(&cli, &molecule, pbc)?;
        molecule.cell = cell;
        axes = rotation;
    }
    cli.axes = axes;
    // Per-lattice-vector inputs are given in the user's axis order; move them into the cell's.
    if let Some(n) = cli.kpoints {
        cli.kpoints = Some(axes.apply(n));
    }
    cli.kshift = axes.apply(cli.kshift);
    cli.supercell = axes.apply(cli.supercell);
    for q in &mut cli.qpoints {
        *q = axes.apply(*q);
    }
    let cli = cli;
    // Which long flags were typed, read off the raw arguments rather than tracked per parse arm:
    // a `Cli` field cannot distinguish "not given" from "given as the default".
    let seen: Vec<String> = args[3..]
        .iter()
        .filter(|a| a.starts_with("--"))
        .cloned()
        .collect();
    validate_flags(&cli.mode, &seen, molecule.cell.is_some())?;
    let parameters = Pm7Parameters::method(cli.method)?;
    let pbc = molecule.cell.map(|_| PbcOptions {
        mode: cli.pbc_mode,
        kmesh: match cli.kpoints {
            None => KMesh::Gamma,
            // `1 1 1` collapses to Γ only when it is *unshifted*. A 1×1×1 mesh at a half shift is
            // one k point that is not Γ -- a perfectly ordinary request, and one the Python CLI
            // has always forwarded. Folding it to Γ here silently answered a different question.
            Some([1, 1, 1]) if cli.kshift == [0.0; 3] => KMesh::Gamma,
            Some(n) => KMesh::MonkhorstPack {
                n,
                shift: cli.kshift,
                gamma_centred: true,
            },
        },
        smearing: cli.smearing.unwrap_or_default(),
        ..PbcOptions::default()
    });
    let options = Pm7Options {
        method: cli.method,
        charge: cli.charge,
        multiplicity: cli.multiplicity,
        use_diis: cli.use_diis,
        exchange_cutoff: cli.exchange_cutoff,
        pbc,
        field: cli.field,
        reference: cli.reference,
        dipole_origin: cli.dipole_origin,
        p_tol: cli.scf_tolerance.unwrap_or(Pm7Options::default().p_tol),
        max_scf: cli.max_scf.unwrap_or(Pm7Options::default().max_scf),
        cphf_max_iterations: cli
            .cphf_max_iterations
            .unwrap_or(Pm7Options::default().cphf_max_iterations),
        stability: cli.stability,
        ..Pm7Options::default()
    };

    match cli.mode.as_str() {
        "energy" | "charges" => energy(&cli, &molecule, &parameters, &options),
        "gradient" | "forces" => gradient(&cli, &molecule, &parameters, &options),
        "stress" => stress(&cli, &molecule, &parameters, &options),
        "optimize" => optimise(&cli, &molecule, &parameters, &options),
        "frequencies" => frequencies(&cli, &molecule, &parameters, &options),
        "hessian" => hessian(&cli, &molecule, &parameters, &options),
        "orbitals" => orbitals(&cli, &molecule, &parameters, &options),
        "molden" => molden(&cli, &molecule, &parameters, &options),
        "phonons" => phonons(&cli, &molecule, &parameters, &options),
        "dfpt" => dfpt(&cli, &molecule, &parameters, &options),
        "born" => born(&cli, &molecule, &parameters, &options),
        "bands" => bands(&cli, &molecule, &parameters, &options),
        "dielectric" => dielectric(&cli, &molecule, &parameters, &options),
        other => Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "unknown mode `{other}`"
        ))),
    }
}

/// Which flags each mode actually reads, beyond the ones every mode reads.
///
/// A flag a mode never looks at used to be accepted in silence: `energy --supercell 2 2 2` ran a
/// single point and said nothing, and `energy structure.xyz --opt-output out.xyz` exited
/// successfully having written no file. `tests/cli_matrix.rs` has the full ledger of them.
///
/// Refusing is better than ignoring for the reason the crate refuses everywhere else: a flag that
/// is accepted and discarded reads, to the person who typed it, exactly like a flag that worked.
const MODE_FLAGS: &[(&str, &[&str])] = &[
    ("energy", &["--dandc"]),
    ("charges", &["--dandc"]),
    ("gradient", &[]),
    ("forces", &[]),
    ("stress", &[]),
    (
        "optimize",
        &[
            "--dandc",
            "--opt-output",
            "--output",
            "--opt-cell",
            "--gtol",
            "--stress-tol",
            "--opt-max-iter",
            "--stability-every",
        ],
    ),
    ("frequencies", &["--ir", "--projection"]),
    ("hessian", &[]),
    ("orbitals", &[]),
    ("molden", &["--molden-basis", "--opt-output", "--output"]),
    (
        "phonons",
        &[
            "--supercell",
            "--qpoints",
            "--acoustic-sum-rule",
            "--no-acoustic-sum-rule",
            "--lo-to",
        ],
    ),
    ("dfpt", &["--qpoints", "--lo-to"]),
    ("born", &["--lo-to"]),
    ("bands", &["--qpoints"]),
    (
        "dielectric",
        &["--slab-thickness", "--wire-cross-section", "--lo-to"],
    ),
];

/// Flags every mode reads, so they never need naming per mode.
const GENERAL_FLAGS: &[&str] = &[
    "--method",
    "--charge",
    "--multiplicity",
    "--reference",
    "--json",
    "--no-diis",
    "--scf-tolerance",
    "--max-scf",
    "--field",
    "--cell",
    "--pbc",
    "--dipole-origin",
    // Reaches `Pm7Options` and so travels with every SCF, even though today only the CPHF
    // response-Fock consults it. Listing it per mode would refuse it on `energy` and refuse a
    // *malformed* one with the wrong complaint.
    "--exchange-cutoff",
    // The same argument, for the same solver: it is an `Pm7Options` field, and a mode that never
    // reaches a CPHF simply carries a number it does not read — where naming the CPHF modes here
    // would turn a mistyped budget on `energy` into "this flag does nothing for mode `energy`".
    "--cphf-max-iterations",
    // Same argument again: an `Pm7Options` field, so a mode that never runs an SCF stability
    // analysis carries a setting it does not read rather than refusing it with the wrong reason.
    "--stability",
];

/// Flags that only mean something once the structure has a cell.
///
/// `--pbc` is *not* here: it is one of the ways to give a run its lattice in the first place
/// (against `--cell`'s nine numbers), so refusing it for want of a cell would refuse it for want
/// of the thing it supplies. `apply_pbc_flag` raises its own refusal when there is no lattice for
/// it to select from, and that message can say which of the two ways to use.
const PERIODIC_FLAGS: &[&str] = &["--kpoints", "--kshift", "--smearing", "--pbc-mode"];

/// Refuse a flag the mode will not read, and a periodic flag on a molecule.
fn validate_flags(mode: &str, seen: &[String], periodic: bool) -> pm7_rs::Result<()> {
    let allowed = MODE_FLAGS
        .iter()
        .find(|(m, _)| *m == mode)
        .map(|(_, f)| *f)
        .unwrap_or(&[]);
    // `molden` writes a Molden file: a text format with its own sections and no JSON form. It is
    // the one mode `--json` does not mean anything for, so it is refused there rather than
    // accepted and dropped.
    if mode == "molden" && seen.iter().any(|f| f == "--json") {
        return Err(pm7_rs::Pm7Error::InvalidInput(
            "`--json` does nothing for mode `molden`: a Molden file is a text format with its own \
             sections, not a value that can be wrapped in JSON. Use `orbitals --json` for the \
             orbital energies and coefficients as data."
                .into(),
        ));
    }
    for flag in seen {
        if GENERAL_FLAGS.contains(&flag.as_str()) || allowed.contains(&flag.as_str()) {
            continue;
        }
        if PERIODIC_FLAGS.contains(&flag.as_str()) {
            if periodic {
                continue;
            }
            return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                "`{flag}` describes Brillouin-zone sampling, and this structure has no cell, so \
                 there is no zone to sample. Give a lattice with `--cell` or an extended-XYZ \
                 `Lattice=\"...\"` key, or drop `{flag}`."
            )));
        }
        // `--json` is refused on `molden` by falling through to here: a Molden file is a text
        // format with its own sections, and there is nothing to wrap it in.
        let hint = MODE_FLAGS
            .iter()
            .filter(|(_, f)| f.contains(&flag.as_str()))
            .map(|(m, _)| *m)
            .collect::<Vec<_>>();
        let where_it_works = if hint.is_empty() {
            String::new()
        } else {
            format!(" It applies to `{}`.", hint.join("`, `"))
        };
        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "`{flag}` does nothing for mode `{mode}`, so it is refused rather than ignored.\
             {where_it_works}"
        )));
    }
    Ok(())
}

fn energy(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    // Divide and conquer is a different driver, not a flag on the SCF, so it branches here.
    if let Some(buffer) = cli.dandc {
        let dandc = DandcOptions {
            buffer: buffer * ANGSTROM_TO_BOHR,
            ..DandcOptions::default()
        };
        let result = run_dandc(molecule, parameters, options, &dandc)?;
        if cli.json {
            println!(
                "{{\"method\":\"{}\",\"electronic_ev\":{:.12},\"fermi_ev\":{:.12},\
                 \"subsystems\":{},\"largest_subsystem\":{},\"iterations\":{},\"converged\":{}}}",
                cli.method,
                result.electronic_ev,
                result.fermi_ev,
                result.subsystems,
                result.largest_subsystem,
                result.iterations,
                result.converged
            );
        } else {
            println!("method: {} (divide and conquer)", cli.method);
            println!("electronic energy: {:.12} eV", result.electronic_ev);
            println!("Fermi level: {:.12} eV", result.fermi_ev);
            println!(
                "subsystems: {} (largest {} atoms)",
                result.subsystems, result.largest_subsystem
            );
            println!(
                "converged: {} after {} iterations",
                result.converged, result.iterations
            );
        }
        return Ok(());
    }

    let result = run_pm7(molecule, parameters, options)?;
    if cli.json {
        let mut fields = format!(
            "\"method\":\"{}\",\"energy_ev\":{:.12},\"heat_of_formation_kcal\":{:.12},\"charges\":[{}]",
            cli.method,
            result.total_ev,
            result.heat_of_formation_kcal,
            result
                .charges
                .iter()
                .map(|q| format!("{q:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        fields.push_str(&format!(
            ",\"dipole_debye\":[{:.12},{:.12},{:.12}]\
             ,\"dipole_point_charge_debye\":[{:.12},{:.12},{:.12}]\
             ,\"dipole_sp_hybrid_debye\":[{:.12},{:.12},{:.12}]\
             ,\"dipole_pd_hybrid_debye\":[{:.12},{:.12},{:.12}]\
             ,\"orbital_source\":\"{}\"",
            result.dipole_debye.x,
            result.dipole_debye.y,
            result.dipole_debye.z,
            result.dipole.point_charge.x,
            result.dipole.point_charge.y,
            result.dipole.point_charge.z,
            result.dipole.sp_hybrid.x,
            result.dipole.sp_hybrid.y,
            result.dipole.sp_hybrid.z,
            result.dipole.pd_hybrid.x,
            result.dipole.pd_hybrid.y,
            result.dipole.pd_hybrid.z,
            result.orbital_source.as_str(),
        ));
        if let Some(e) = result.homo_ev {
            fields.push_str(&format!(",\"homo_ev\":{e:.12}"));
        }
        if let Some(e) = result.lumo_ev {
            fields.push_str(&format!(",\"lumo_ev\":{e:.12}"));
        }
        if let Some(e) = result.homo_ev_beta {
            fields.push_str(&format!(",\"homo_ev_beta\":{e:.12}"));
        }
        if let Some(e) = result.lumo_ev_beta {
            fields.push_str(&format!(",\"lumo_ev_beta\":{e:.12}"));
        }
        if let Some(e) = result.gap_ev() {
            fields.push_str(&format!(",\"gap_ev\":{e:.12}"));
        }
        if let Some(e) = result.field_ev {
            fields.push_str(&format!(",\"field_ev\":{e:.12}"));
        }
        if let Some(s2) = result.spin_squared() {
            fields.push_str(&format!(",\"spin_squared\":{s2:.12}"));
        }
        if let Some(n) = result.n_kpoints {
            fields.push_str(&format!(",\"n_kpoints\":{n}"));
        }
        if let Some(e) = result.fermi_ev {
            fields.push_str(&format!(",\"fermi_ev\":{e:.12}"));
        }
        if let Some(e) = result.entropy_ev {
            fields.push_str(&format!(",\"entropy_ev\":{e:.12}"));
            fields.push_str(&format!(
                ",\"free_energy_ev\":{:.12}",
                result.free_energy_ev()
            ));
        }
        println!("{{{fields}}}");
    } else {
        println!("method: {}", cli.method);
        println!("total energy: {:.12} eV", result.total_ev);
        println!(
            "heat of formation: {:.12} kcal/mol",
            result.heat_of_formation_kcal
        );
        println!("SCF iterations: {}", result.iterations);
        println!(
            "dipole: {:.6} {:.6} {:.6} Debye  (|mu| = {:.6})",
            result.dipole_debye.x,
            result.dipole_debye.y,
            result.dipole_debye.z,
            result.dipole_magnitude
        );
        if let (Some(homo), Some(lumo)) = (result.homo_ev, result.lumo_ev) {
            println!(
                "HOMO / LUMO: {homo:.6} / {lumo:.6} eV  (gap {:.6} eV, from the {} orbitals)",
                result.gap_ev().unwrap_or(f64::NAN),
                result.orbital_source.as_str()
            );
        }
        if let Some(e) = result.field_ev {
            println!("external-field energy: {e:.12} eV");
        }
        if let Some(s2) = result.spin_squared() {
            // MOPAC's `(SZ)` / `(S**2)` pair. Printed together because the gap between them is
            // the whole point: an unrestricted determinant is not a spin eigenfunction, and how
            // far `<S^2>` sits above `S(S+1)` is the only measure of that in the output.
            let n_beta = result.n_occ_beta.unwrap_or(result.n_occ);
            let sz = 0.5 * (result.n_occ as f64 - n_beta as f64);
            println!(
                "(SZ) = {sz:.6}   <S^2> = {s2:.6}   (exact {:.6})",
                sz * (sz + 1.0)
            );
        }
        if let Some(n) = result.n_kpoints {
            println!("k points (after time-reversal folding): {n}");
        }
        if let Some(e) = result.fermi_ev {
            println!("Fermi level: {e:.12} eV");
        }
        if let Some(e) = result.entropy_ev {
            println!("entropy contribution -TS: {e:.12} eV");
            // The electronic (Mermin) free energy, which is what the forces differentiate under
            // smearing. Not a thermochemical Gibbs energy; this program derives no thermochemistry
            // from its frequencies.
            println!(
                "electronic free energy E-TS: {:.12} eV",
                result.free_energy_ev()
            );
        }
        if cli.mode == "charges" {
            for (atom, charge) in molecule.atoms.iter().zip(&result.charges) {
                println!(
                    "{} {charge:+.8}",
                    pm7_rs::z_to_symbol(atom.z).unwrap_or("X")
                );
            }
        }
    }
    Ok(())
}

/// `gradient` prints `dE/dR`; `forces` prints its negation.
///
/// Both, rather than making the caller remember the sign: a force is what every MD and optimizer
/// consumer wants, a gradient is what every derivative check wants, and a sign error between them
/// is silent in exactly the cases that matter.
fn gradient(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    let result = closed_form_gradient(molecule, parameters, options)?;
    let sign = if cli.mode == "forces" { -1.0 } else { 1.0 };
    if cli.json {
        let rows: Vec<String> = result
            .gradient
            .iter()
            .map(|v| format!("[{:.12},{:.12},{:.12}]", sign * v.x, sign * v.y, sign * v.z))
            .collect();
        let key = if cli.mode == "forces" {
            "forces_ev_per_bohr"
        } else {
            "gradient_ev_per_bohr"
        };
        println!(
            "{{\"{key}\":[{}],\"energy_ev\":{:.12},\"max_gradient_ev_per_bohr\":{:.12}}}",
            rows.join(","),
            result.energy_ev,
            result.max_gradient
        );
        return Ok(());
    }
    for vector in result.gradient {
        println!(
            "{:.12} {:.12} {:.12}",
            sign * vector.x,
            sign * vector.y,
            sign * vector.z
        );
    }
    Ok(())
}

/// The analytic Cartesian Hessian, `3N x 3N`, in eV/Bohr².
fn hessian(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    let h = pm7_rs::analytic_hessian(molecule, parameters, options, 1.0e-3)?;
    if cli.json {
        println!("{{\"hessian_ev_per_bohr2\": {}}}", json_matrix(&h));
    } else {
        for i in 0..h.rows {
            let row: Vec<String> = (0..h.cols).map(|j| format!("{:.12}", h[(i, j)])).collect();
            println!("{}", row.join(" "));
        }
    }
    Ok(())
}

/// The converged wavefunction in Molden format.
fn molden(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    let basis = match cli.molden_basis.to_ascii_lowercase().as_str() {
        "sto" | "slater" => pm7_rs::MoldenBasis::Sto,
        other => {
            let n = other
                .strip_prefix("sto-")
                .and_then(|rest| rest.strip_suffix('g'))
                .and_then(|digits| digits.parse::<usize>().ok())
                .filter(|n| (1..=12).contains(n))
                .ok_or_else(|| {
                    pm7_rs::Pm7Error::InvalidInput(format!(
                        "unknown --molden-basis `{}`; use `sto` or `sto-nG` (e.g. sto-6g)",
                        cli.molden_basis
                    ))
                })?;
            pm7_rs::MoldenBasis::StoNg { n }
        }
    };
    let scf = run_pm7(molecule, parameters, options)?;
    let molden_options = pm7_rs::MoldenOptions {
        basis,
        comment: None,
    };
    // Through the library's own writer rather than a second `std::fs::write` here. `write_molden`
    // had no caller anywhere -- the CLI wrote the file itself and so did the ASE calculator -- and
    // an exported function nothing exercises is the class of defect this repository has been
    // caught by before (`frequencies_cm_lo_to`, noted in `python/tests/test_api_coverage.py`).
    match &cli.opt_output {
        Some(path) => {
            pm7_rs::write_molden(path, molecule, parameters, &scf, &molden_options)?;
            println!("wrote {path}");
        }
        None => print!(
            "{}",
            pm7_rs::to_molden(molecule, parameters, &scf, &molden_options)?
        ),
    }
    Ok(())
}

/// Phonons at arbitrary q by perturbation theory, with no supercell.
fn dfpt(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "dfpt")?;
    let qpoints = if cli.qpoints.is_empty() {
        vec![[0.0, 0.0, 0.0]]
    } else {
        cli.qpoints.clone()
    };
    let settings = pm7_rs::dfpt::DfptOptions::default();
    // The LO-TO material constants, solved once for the whole q list: Z* and eps^inf are
    // properties of the cell, not of a wavevector. The term belongs only at the zone centre --
    // away from it the macroscopic field is already inside Phi(q) through the phased Ewald sum.
    let non_analytic = match cli.lo_to {
        None => None,
        Some(q_hat) => {
            let field =
                pm7_rs::dfpt::born_and_dielectric(molecule, parameters, options, &settings)?;
            Some((field.non_analytic()?, q_hat))
        }
    };
    let mut all = Vec::with_capacity(qpoints.len());
    for q in &qpoints {
        let out =
            pm7_rs::dfpt::dynamical_matrix_dfpt(molecule, parameters, options, *q, &settings)?;
        let split = match &non_analytic {
            Some((term, q_hat)) if q.iter().all(|c| c.abs() <= 1.0e-12) => {
                Some(out.frequencies_cm_lo_to(term, *q_hat)?)
            }
            _ => None,
        };
        all.push((
            *q,
            out.frequencies_cm()?,
            out.iterations,
            out.residual,
            split,
        ));
    }
    if cli.json {
        let entries: Vec<String> = all
            .iter()
            .map(|(q, f, iters, residual, split)| {
                let freq: Vec<String> = f.iter().map(|v| format!("{v:.6}")).collect();
                let lo_to = match split {
                    None => String::new(),
                    Some(values) => {
                        let listed: Vec<String> =
                            values.iter().map(|v| format!("{v:.6}")).collect();
                        format!(", \"frequencies_cm_lo_to\": [{}]", listed.join(", "))
                    }
                };
                let reported = cli.axes.undo(*q);
                format!(
                    "{{\"q\": [{:.10}, {:.10}, {:.10}], \"frequencies_cm\": [{}], \
                     \"iterations\": {iters}, \"residual\": {residual:.6e}{lo_to}}}",
                    reported[0],
                    reported[1],
                    reported[2],
                    freq.join(", ")
                )
            })
            .collect();
        println!("{{\"qpoints\": [{}]}}", entries.join(", "));
    } else {
        for (q, f, iters, _, split) in &all {
            let reported = cli.axes.undo(*q);
            println!(
                "q = {:.6} {:.6} {:.6}   ({iters} iterations)",
                reported[0], reported[1], reported[2]
            );
            match split {
                None => {
                    for frequency in f {
                        println!("  {frequency:12.4} cm^-1");
                    }
                }
                Some(values) => {
                    println!("  {:>12}  {:>16}", "cm^-1", "with LO-TO");
                    for (frequency, split) in f.iter().zip(values) {
                        println!("  {frequency:12.4}  {split:16.4}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Born effective charges and the electronic dielectric tensor.
fn born(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "born")?;
    let out = pm7_rs::dfpt::born_and_dielectric(
        molecule,
        parameters,
        options,
        &pm7_rs::dfpt::DfptOptions::default(),
    )?;
    if cli.json {
        let charges: Vec<String> = out.born.iter().map(json_mat3).collect();
        print!(
            "{{\"born_charges\": [{}], \"dielectric\": {}, \"polarizability\": {}, \
             \"acoustic_residual\": {:.6e}, \"volume_bohr3\": {:.10}",
            charges.join(", "),
            json_mat3(&out.dielectric),
            json_mat3(&out.polarizability),
            out.acoustic_residual(),
            out.volume_bohr3
        );
        if let (Some(q), Ok(na)) = (cli.lo_to, out.non_analytic()) {
            let matrix = na.matrix(q)?;
            let mut rows = Vec::with_capacity(matrix.n);
            for i in 0..matrix.n {
                let row: Vec<String> = (0..matrix.n)
                    .map(|j| format!("{:.12}", matrix.get(i, j).0))
                    .collect();
                rows.push(format!("[{}]", row.join(", ")));
            }
            print!(
                ", \"lo_to_force_constants_ev_per_bohr2\": [{}]",
                rows.join(", ")
            );
        }
        println!("}}");
    } else {
        for (index, z) in out.born.iter().enumerate() {
            println!("atom {index}  Z* (rows = field, columns = displacement)");
            for a in 0..3 {
                println!(
                    "  {:12.6} {:12.6} {:12.6}",
                    z.get(a, 0),
                    z.get(a, 1),
                    z.get(a, 2)
                );
            }
        }
        println!("acoustic sum rule residual {:.3e}", out.acoustic_residual());
        if out.volume_bohr3 > 0.0 {
            println!("dielectric tensor (electronic, clamped ion)");
            for a in 0..3 {
                println!(
                    "  {:12.6} {:12.6} {:12.6}",
                    out.dielectric.get(a, 0),
                    out.dielectric.get(a, 1),
                    out.dielectric.get(a, 2)
                );
            }
        } else {
            println!("dielectric tensor: not defined without a 3-D cell; raw d(mu)/d(f):");
            for a in 0..3 {
                println!(
                    "  {:12.6e} {:12.6e} {:12.6e}",
                    out.polarizability.get(a, 0),
                    out.polarizability.get(a, 1),
                    out.polarizability.get(a, 2)
                );
            }
        }
    }
    Ok(())
}

/// `eps^inf` for a chain or a slab, which needs an extent the cell cannot supply.
fn dielectric(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "dielectric")?;
    let extent =
        match (cli.slab_thickness, cli.wire_cross_section) {
            (Some(_), Some(_)) => return Err(pm7_rs::Pm7Error::InvalidInput(
                "--slab-thickness and --wire-cross-section are two different conventions; pass \
                 the one that matches the cell's dimensionality"
                    .into(),
            )),
            (Some(d), None) => {
                pm7_rs::ExtentConvention::SlabThickness(d * pm7_rs::constants::ANGSTROM_TO_BOHR)
            }
            (None, Some(s)) => pm7_rs::ExtentConvention::WireCrossSection(
                s * pm7_rs::constants::ANGSTROM_TO_BOHR * pm7_rs::constants::ANGSTROM_TO_BOHR,
            ),
            (None, None) => {
                return Err(pm7_rs::Pm7Error::InvalidInput(
                    "dielectric needs the material's extent: --slab-thickness A for a layer, or \
                 --wire-cross-section A^2 for a chain. A supercell says where the atoms are, not \
                 where the material ends, so there is nothing to infer it from; doubling the \
                 vacuum would otherwise change the answer. For a 3-D cell the volume is already \
                 the extent -- use `born`."
                        .into(),
                ))
            }
        };
    let out = pm7_rs::dielectric_with_extent(
        molecule,
        parameters,
        options,
        &pm7_rs::dfpt::DfptOptions::default(),
        extent,
    )?;
    let slab = matches!(out.extent, pm7_rs::ExtentConvention::SlabThickness(_));
    if cli.json {
        println!(
            "{{\"dielectric\": {}, \"polarizability\": {}, \"axis\": [{:.12}, {:.12}, {:.12}], \
             \"measure_bohr\": {:.10}, \"extent\": {:.10}, \"extent_convention\": \"{}\", \
             \"sheet_parallel_bohr\": {:.10}, \"sheet_perpendicular_bohr\": {:.10}, \
             \"axis_mixing\": {:.6e}}}",
            json_mat3(&out.dielectric),
            json_mat3(&out.polarizability),
            out.axis.x,
            out.axis.y,
            out.axis.z,
            out.measure,
            out.extent.value(),
            if slab {
                "slab_thickness"
            } else {
                "wire_cross_section"
            },
            out.invariants.parallel,
            out.invariants.perpendicular,
            out.axis_mixing
        );
    } else {
        // The cell's own periodic measure is an area for a slab and a length for a wire, so the
        // unit is printed rather than assumed.
        println!(
            "{} {:.6} {}   (assigned, not derived)",
            if slab {
                "slab thickness:"
            } else {
                "wire cross-section:"
            },
            out.extent.value(),
            if slab { "Bohr" } else { "Bohr^2" }
        );
        println!(
            "cell measure       {:.6} {}",
            out.measure,
            if slab { "Bohr^2" } else { "Bohr" }
        );
        println!(
            "axis               ({:+.6}, {:+.6}, {:+.6})",
            out.axis.x, out.axis.y, out.axis.z
        );
        println!("dielectric tensor (electronic, clamped ion, depolarization corrected)");
        for a in 0..3 {
            println!(
                "  {:12.6} {:12.6} {:12.6}",
                out.dielectric.get(a, 0),
                out.dielectric.get(a, 1),
                out.dielectric.get(a, 2)
            );
        }
        // The two numbers the assigned extent cannot move.
        println!(
            "sheet parallel      (eps_par - 1) d  {:.6} Bohr",
            out.invariants.parallel
        );
        println!(
            "sheet perpendicular (1 - 1/eps) d    {:.6} Bohr",
            out.invariants.perpendicular
        );
        println!(
            "axis mixing (0 = the axis is an eigenvector) {:.3e}",
            out.axis_mixing
        );
    }
    Ok(())
}

/// A `3x3` tensor as a JSON array of arrays. No serde: `Cargo.toml` is deliberately three
/// dependencies, and one helper is cheaper than a derive macro and a parser.
fn json_mat3(m: &pm7_rs::math::Mat3) -> String {
    let rows: Vec<String> = (0..3)
        .map(|i| {
            let row: Vec<String> = (0..3).map(|j| format!("{:.12}", m.get(i, j))).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

/// A real matrix as a JSON array of arrays.
fn json_matrix(m: &pm7_rs::Matrix) -> String {
    let rows: Vec<String> = (0..m.rows)
        .map(|i| {
            let row: Vec<String> = (0..m.cols).map(|j| format!("{:.12}", m[(i, j)])).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

fn stress(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "stress")?;
    let scf = run_pm7(molecule, parameters, options)?;
    let result = analytic_stress(molecule, parameters, options, &scf)?;
    let s = result.stress;
    // ASE's Voigt order, which is what every consumer of a stress tensor expects.
    let voigt = [
        s.get(0, 0),
        s.get(1, 1),
        s.get(2, 2),
        s.get(1, 2),
        s.get(0, 2),
        s.get(0, 1),
    ];
    if cli.json {
        let mut fields = format!(
            "\"energy_ev\":{:.12},\"stress_ev_per_bohr3\":[{}]",
            scf.total_ev,
            voigt
                .iter()
                .map(|v| format!("{v:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        if let Some(p) = result.pressure_gpa(molecule) {
            fields.push_str(&format!(",\"pressure_gpa\":{p:.12}"));
        }
        println!("{{{fields}}}");
    } else {
        println!("total energy: {:.12} eV", scf.total_ev);
        println!("stress tensor (eV/Bohr^{}):", molecule.cell.unwrap().dim());
        for i in 0..3 {
            println!(
                "  {:>16.12} {:>16.12} {:>16.12}",
                s.get(i, 0),
                s.get(i, 1),
                s.get(i, 2)
            );
        }
        println!(
            "Voigt [xx yy zz yz xz xy]: {}",
            voigt
                .iter()
                .map(|v| format!("{v:.9}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        match result.pressure_gpa(molecule) {
            Some(p) => println!("pressure: {p:.9} GPa"),
            None => println!("pressure: n/a (a pressure needs a volume, so 3-D only)"),
        }
    }
    Ok(())
}

fn optimise(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    // A geometry optimization is tens to hundreds of energy-and-gradient evaluations, which is
    // exactly the workload linear scaling is for -- and it was the one driver `--dandc` could not
    // reach, because `optimise` went straight to the exact gradient.
    let defaults = OptOptions::default();
    let opt = OptOptions {
        dandc: cli.dandc.map(|buffer| DandcOptions {
            buffer: buffer * ANGSTROM_TO_BOHR,
            ..DandcOptions::default()
        }),
        // Through 0.2.2 every one of these was `OptOptions::default()` at both call sites, so a
        // run that needed a tighter convergence or more iterations had no way to say so from
        // either command line.
        relax_cell: cli.opt_cell,
        gtol: cli.gtol.unwrap_or(defaults.gtol),
        stress_tol: cli.stress_tol.unwrap_or(defaults.stress_tol),
        max_iter: cli.opt_max_iter.unwrap_or(defaults.max_iter),
        // A geometry step changes the orbitals, so a solution that was a minimum at the start can
        // stop being one on the way. `--stability-every N` re-checks every `N`th step and switches
        // the run to `follow` for good once an instability is found; `0` (the default) never checks.
        stability_every: cli.stability_every.unwrap_or(defaults.stability_every),
        ..defaults
    };
    let result = optimize(molecule, parameters, options, &opt)?;
    let a0 = BOHR_TO_ANGSTROM;
    if cli.json {
        let positions: Vec<String> = result
            .molecule
            .atoms
            .iter()
            .map(|a| {
                format!(
                    "[{:.12},{:.12},{:.12}]",
                    a.position.x * a0,
                    a.position.y * a0,
                    a.position.z * a0
                )
            })
            .collect();
        let numbers: Vec<String> = result
            .molecule
            .atoms
            .iter()
            .map(|a| a.z.to_string())
            .collect();
        let mut fields = format!(
            "\"converged\":{},\"iterations\":{},\"energy_ev\":{:.12},\
             \"heat_of_formation_kcal\":{:.12},\"max_gradient_ev_per_bohr\":{:.12},\
             \"driver\":\"{}\",\"numbers\":[{}],\"positions_angstrom\":[{}]",
            result.converged,
            result.iterations,
            result.energy_ev,
            result.heat_of_formation_kcal,
            result
                .trajectory
                .last()
                .map(|s| s.max_gradient)
                .unwrap_or(f64::NAN),
            if cli.dandc.is_some() {
                "divide-and-conquer"
            } else {
                "exact"
            },
            numbers.join(","),
            positions.join(",")
        );
        // A periodic relaxation has to say which cell the coordinates belong to, or the output
        // cannot be read back.
        if let Some(cell) = result.molecule.cell {
            let rows: Vec<String> = cell
                .completed_vectors()
                .iter()
                .map(|v| format!("[{:.12},{:.12},{:.12}]", v.x * a0, v.y * a0, v.z * a0))
                .collect();
            fields.push_str(&format!(",\"cell_angstrom\":[{}]", rows.join(",")));
        }
        println!("{{{fields}}}");
    } else {
        if cli.dandc.is_some() {
            println!("driver: divide and conquer");
        }
        println!(
            "converged: {} after {} iterations",
            result.converged, result.iterations
        );
        println!("energy: {:.12} eV", result.energy_ev);
        println!(
            "heat of formation: {:.12} kcal/mol",
            result.heat_of_formation_kcal
        );
        if let Some(step) = result.trajectory.last() {
            println!("max gradient: {:.6e} eV/Bohr", step.max_gradient);
            // Only with `--opt-cell`: without it there is no stress degree of freedom, and
            // printing the fixed-cell stress here would read as something the run was driving on.
            if cli.opt_cell {
                println!(
                    "max free stress: {:.6e} eV/Bohr^{}",
                    step.max_stress,
                    molecule.cell.map(|c| c.dim()).unwrap_or(3)
                );
            }
        }
        // The relaxed geometry is the point of the mode. Printing only "converged: true" and an
        // energy left the answer reachable solely through `--output`, so a run that forgot the
        // flag did the work and threw it away.
        if cli.opt_output.is_none() {
            println!();
            print!("{}", result.molecule.to_xyz_string("pm7-rs optimized"));
        }
    }
    if let Some(output) = &cli.opt_output {
        std::fs::write(output, result.molecule.to_xyz_string("pm7-rs optimized"))?;
    }
    Ok(())
}

fn frequencies(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    if !cli.ir {
        let modes = pm7_rs::vibrational_analysis_projected(
            molecule,
            parameters,
            options,
            1.0e-3,
            cli.projection,
        )?;
        for frequency in modes.frequencies_cm {
            println!("{frequency:.6}");
        }
        return Ok(());
    }
    // `--ir` takes the spectrum path, which produces the frequencies and the intensities from the
    // *same* CPHF solve rather than running the Hessian twice.
    let spectrum =
        pm7_rs::ir_spectrum_projected(molecule, parameters, options, 1.0e-3, cli.projection)?;
    if cli.json {
        let list = |values: &[f64]| {
            values
                .iter()
                .map(|v| format!("{v:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let tensor = (0..spectrum.dipole_derivatives.rows)
            .map(|i| {
                let row: Vec<f64> = (0..spectrum.dipole_derivatives.cols)
                    .map(|j| spectrum.dipole_derivatives[(i, j)])
                    .collect();
                format!("[{}]", list(&row))
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"frequencies_cm\":[{}],\"ir_intensities_km_per_mol\":[{}],\
             \"mopac_dipt\":[{}],\"dipole_derivatives_e\":[{tensor}]}}",
            list(&spectrum.frequencies_cm),
            list(&spectrum.intensities_km_per_mol),
            list(&spectrum.mopac_dipt),
        );
    } else {
        println!(
            "{:>18}  {:>12}  {:>11}",
            "frequency (cm^-1)", "IR (km/mol)", "DIPT (D/A)"
        );
        for i in 0..spectrum.frequencies_cm.len() {
            println!(
                "{:18.4}  {:12.4}  {:11.5}",
                spectrum.frequencies_cm[i],
                spectrum.intensities_km_per_mol[i],
                spectrum.mopac_dipt[i]
            );
        }
    }
    Ok(())
}

/// Orbital energies, occupations and (with `--json`) the coefficient matrix.
fn orbitals(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    let result = run_pm7(molecule, parameters, options)?;
    let occupations = result.occupations();
    if cli.json {
        let list = |values: &[f64]| {
            values
                .iter()
                .map(|v| format!("{v:.12}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let matrix = |m: &pm7_rs::Matrix| {
            (0..m.rows)
                .map(|i| {
                    let row: Vec<f64> = (0..m.cols).map(|j| m[(i, j)]).collect();
                    format!("[{}]", list(&row))
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        // Brought to parity with `native.orbitals`, which returned all of this while the CLI
        // returned five keys. A caller scripting the binary had no way to reach the frontier
        // energies, the gap, the AO labelling or either beta channel.
        let mut fields = format!(
            "\"orbital_source\":\"{}\",\"unrestricted\":{},\"n_occ\":{},\
             \"mo_energies_ev\":[{}],\"occupations\":[{}],\"mo_coefficients\":[{}]",
            result.orbital_source.as_str(),
            result.unrestricted,
            result.n_occ,
            list(&result.mo_energies),
            list(&occupations),
            matrix(&result.mo_coeff),
        );
        for (key, value) in [
            ("homo_ev", result.homo_ev),
            ("lumo_ev", result.lumo_ev),
            ("gap_ev", result.gap_ev()),
            ("homo_ev_beta", result.homo_ev_beta),
            ("lumo_ev_beta", result.lumo_ev_beta),
            ("fermi_ev", result.fermi_ev),
            ("entropy_ev", result.entropy_ev),
        ] {
            if let Some(v) = value {
                fields.push_str(&format!(",\"{key}\":{v:.12}"));
            }
        }
        if let Some(beta) = &result.mo_energies_beta {
            fields.push_str(&format!(",\"mo_energies_beta_ev\":[{}]", list(beta)));
        }
        if let Some(occ) = result.occupations_beta() {
            fields.push_str(&format!(",\"occupations_beta\":[{}]", list(&occ)));
        }
        if let Some(n) = result.n_occ_beta {
            fields.push_str(&format!(",\"n_occ_beta\":{n}"));
        }
        if let Some(c) = &result.mo_coeff_beta {
            fields.push_str(&format!(",\"mo_coefficients_beta\":[{}]", matrix(c)));
        }
        // Which AO each coefficient row belongs to. Without it a coefficient matrix is a block of
        // numbers whose rows the caller has to re-derive from the element list and the basis rules.
        if let Ok(basis) = pm7_rs::basis::Basis::build(molecule, parameters) {
            const LABELS: [&str; 9] = ["s", "px", "py", "pz", "dx2-y2", "dxz", "dz2", "dyz", "dxy"];
            let names: Vec<String> = basis
                .aos
                .iter()
                .map(|a| format!("\"{}\"", LABELS[a.orb as usize]))
                .collect();
            let atoms: Vec<String> = basis.aos.iter().map(|a| a.atom.to_string()).collect();
            fields.push_str(&format!(
                ",\"ao_labels\":[{}],\"ao_atom_index\":[{}]",
                names.join(","),
                atoms.join(",")
            ));
        }
        println!("{{{fields}}}");
    } else {
        if let (Some(homo), Some(lumo)) = (result.homo_ev, result.lumo_ev) {
            println!(
                "HOMO / LUMO: {homo:.6} / {lumo:.6} eV  (gap {:.6} eV)",
                result.gap_ev().unwrap_or(f64::NAN)
            );
        }
        println!("orbitals reported at: {}", result.orbital_source.as_str());
        println!("{:>5} {:>6} {:>14}", "index", "occ", "energy (eV)");
        for (index, (energy, occupation)) in result.mo_energies.iter().zip(&occupations).enumerate()
        {
            let marker = if index + 1 == result.n_occ {
                "  <-- HOMO"
            } else if index == result.n_occ {
                "  <-- LUMO"
            } else {
                ""
            };
            println!("{index:5} {occupation:6.3} {energy:14.6}{marker}");
        }
        // The beta channel used to print energies with a blank occupation column and no frontier
        // markers, although `occupations_beta`, `homo_ev_beta` and `lumo_ev_beta` all existed. An
        // open-shell run's SOMO is the interesting orbital and it was the one you could not see.
        if let Some(beta) = &result.mo_energies_beta {
            println!();
            if let (Some(homo), Some(lumo)) = (result.homo_ev_beta, result.lumo_ev_beta) {
                println!(
                    "beta HOMO / LUMO: {homo:.6} / {lumo:.6} eV  (gap {:.6} eV)",
                    lumo - homo
                );
            }
            let beta_occupations = result.occupations_beta().unwrap_or_default();
            let n_beta = result.n_occ_beta.unwrap_or(0);
            println!("{:>5} {:>6} {:>14}   beta", "index", "occ", "energy (eV)");
            for (index, energy) in beta.iter().enumerate() {
                let occupation = beta_occupations.get(index).copied().unwrap_or(0.0);
                let marker = if index + 1 == n_beta {
                    "  <-- HOMO"
                } else if index == n_beta {
                    "  <-- LUMO"
                } else {
                    ""
                };
                println!("{index:5} {occupation:6.3} {energy:14.6}{marker}");
            }
        }
        if let Some(fermi) = result.fermi_ev {
            println!("\nFermi level: {fermi:.6} eV");
        }
        // The occupation column above is an aufbau count over the Γ states of a converged k-mesh
        // Hamiltonian (see `Pm7Result::occupations`), which is the mesh's filling only when a gap
        // straddles the Fermi level. A non-zero entropy says it does not, and printing a column of
        // clean 2.000/0.000 next to a Fermi level that contradicts them is how a reader is misled.
        if let Some(entropy) = result.entropy_ev {
            if entropy.abs() > 0.0 {
                println!(
                    "entropy contribution -TS: {entropy:.6} eV -- states are fractionally \
                     occupied on the mesh, so the `occ` column above is a per-k band count and \
                     not this run's filling"
                );
            }
        }
    }
    Ok(())
}

fn phonons(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "phonons")?;
    let mut fc = force_constants(molecule, parameters, options, cli.supercell)?;
    let residual = fc.acoustic_residual();
    if cli.acoustic_sum_rule {
        fc.enforce_acoustic_sum_rule();
    }
    let qpoints: &[[f64; 3]] = if cli.qpoints.is_empty() {
        &[[0.0, 0.0, 0.0]]
    } else {
        &cli.qpoints
    };
    // --lo-to adds the non-analytic term at the zone centre. The material constants come from
    // a field response, so this costs one on top of the force constants; it is opt-in for that
    // reason as well as because the q -> 0 limit has no default direction.
    let non_analytic = match cli.lo_to {
        None => None,
        Some(q_hat) => {
            let field = pm7_rs::dfpt::born_and_dielectric(
                molecule,
                parameters,
                options,
                &pm7_rs::dfpt::DfptOptions::default(),
            )?;
            Some((field.non_analytic()?, q_hat))
        }
    };
    let split_at = |q: &[f64; 3]| -> pm7_rs::Result<Option<Vec<f64>>> {
        match &non_analytic {
            Some((term, q_hat)) if q.iter().all(|c| c.abs() <= 1.0e-12) => {
                Ok(Some(fc.frequencies_cm_lo_to(term, *q_hat)?))
            }
            _ => Ok(None),
        }
    };
    if cli.json {
        let mut entries = Vec::new();
        for q in qpoints {
            let f = fc.frequencies_cm(*q)?;
            let lo_to = match split_at(q)? {
                None => String::new(),
                Some(values) => format!(
                    ",\"frequencies_cm_lo_to\":[{}]",
                    values
                        .iter()
                        .map(|v| format!("{v:.6}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            };
            let reported = cli.axes.undo(*q);
            entries.push(format!(
                "{{\"q\":[{},{},{}],\"frequencies_cm\":[{}]{lo_to}}}",
                reported[0],
                reported[1],
                reported[2],
                f.iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        // Echoed in the axis order the user typed, not the internal one: `--pbc TFT --supercell
        // 2 1 2` should read back as `2 1 2`.
        let repeat = cli.axes.undo(cli.supercell);
        println!(
            "{{\"supercell\":[{},{},{}],\"acoustic_residual\":{residual:.6e},\"qpoints\":[{}]}}",
            repeat[0],
            repeat[1],
            repeat[2],
            entries.join(",")
        );
    } else {
        let repeat = cli.axes.undo(cli.supercell);
        println!("supercell: {}x{}x{}", repeat[0], repeat[1], repeat[2]);
        // Reported whether or not it was projected out: it is the honest measure of how well the
        // force constants respect translational invariance, and projecting hides it.
        println!("acoustic sum-rule residual: {residual:.6e} eV/Bohr^2");
        if cli.acoustic_sum_rule {
            println!("(acoustic sum rule projected out; --no-acoustic-sum-rule keeps the raw set)");
        } else {
            println!(
                "(acoustic sum rule NOT projected out: the residual above is in the spectrum)"
            );
        }
        for q in qpoints {
            let f = fc.frequencies_cm(*q)?;
            let reported = cli.axes.undo(*q);
            println!(
                "q = ({:.6}, {:.6}, {:.6})",
                reported[0], reported[1], reported[2]
            );
            match split_at(q)? {
                None => {
                    for v in f {
                        println!("  {v:12.4} cm^-1");
                    }
                }
                Some(values) => {
                    println!("  {:>12}  {:>16}", "cm^-1", "with LO-TO");
                    for (v, split) in f.iter().zip(&values) {
                        println!("  {v:12.4}  {split:16.4}");
                    }
                }
            }
        }
    }
    Ok(())
}

fn bands(
    cli: &Cli,
    molecule: &Molecule,
    parameters: &Pm7Parameters,
    options: &Pm7Options,
) -> pm7_rs::Result<()> {
    require_cell(molecule, "bands")?;
    if cli.qpoints.is_empty() {
        return Err(pm7_rs::Pm7Error::InvalidInput(
            "bands needs a k path: --qpoints kx,ky,kz kx,ky,kz ... (fractional)".into(),
        ));
    }
    // The density and the Fermi level come from the sampling mesh; the path only asks the
    // converged Hamiltonian for its eigenvalues elsewhere. See `scf_pbc::band_structure`.
    let out = band_structure(molecule, parameters, options, &cli.qpoints)?;
    let bands = out.energies;
    let fermi = out.fermi_ev;
    if cli.json {
        let rows: Vec<String> = cli
            .qpoints
            .iter()
            .zip(&bands)
            .map(|(q, e)| {
                let k = cli.axes.undo(*q);
                format!(
                    "{{\"k\":[{},{},{}],\"energies_ev\":[{}]}}",
                    k[0],
                    k[1],
                    k[2],
                    e.iter()
                        .map(|v| format!("{v:.9}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        println!(
            "{{\"fermi_ev\":{fermi:.12},\"bands\":[{}]}}",
            rows.join(",")
        );
    } else {
        println!("Fermi level (from the sampling mesh): {fermi:.9} eV");
        for (q, e) in cli.qpoints.iter().zip(&bands) {
            let k = cli.axes.undo(*q);
            println!("k = ({:.6}, {:.6}, {:.6})", k[0], k[1], k[2]);
            for v in e {
                println!("  {v:14.8} eV");
            }
        }
    }
    Ok(())
}

/// Resolve `--pbc` against whichever lattice the run has, and say exactly which combinations are
/// meaningful rather than letting precedence decide.
///
/// `--pbc` picks periodic directions out of **three** lattice vectors, so all three have to be on
/// the table. Two ways to get them, and one way not to:
///
/// * `--cell` with nine numbers — the rows are used directly.
/// * a structure file whose `Lattice=` is fully periodic — `--pbc` overrides the file's `pbc=`
///   key, which is the useful direction of the override (a file written by someone else, run as a
///   slab without editing it).
/// * `--cell` with three or six numbers, or a file whose `pbc=` already drops a direction, is
///   **refused**. Those lattices have fewer than three real vectors; `completed_vectors` would
///   hand back synthetic unit normals for the rest, and selecting a "periodic direction" out of a
///   made-up normal is not a thing the user asked for. Silently doing it would produce a cell
///   whose second lattice vector was invented by this program.
fn apply_pbc_flag(
    cli: &Cli,
    molecule: &Molecule,
    pbc: [bool; 3],
) -> pm7_rs::Result<(Option<Cell>, pm7_rs::AxisRotation)> {
    let rows = match cli.cell_rows {
        Some(rows) => rows,
        None => {
            let cell = molecule.cell.ok_or_else(|| {
                pm7_rs::Pm7Error::InvalidInput(
                    "`--pbc` says which lattice directions are periodic, and this run has no \
                     lattice at all. Give one with `--cell a11,...,a33` (nine numbers) or use a \
                     structure file with a Lattice=\"...\" key."
                        .into(),
                )
            })?;
            if cell.dim() != 3 {
                let source = if cli.cell.is_some() {
                    "`--cell` was given fewer than nine numbers"
                } else {
                    "the structure file's pbc= key already dropped the others"
                };
                return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                    "`--pbc` chooses among three lattice vectors and this lattice has {}, because \
                     {source}. Give all nine numbers to `--cell`, or set the pattern in the \
                     file's pbc= key instead.",
                    cell.dim()
                )));
            }
            let rows = cell.angstrom_rows();
            [rows[0], rows[1], rows[2]]
        }
    };
    Cell::from_angstrom_rows_pbc(&rows, pbc)
}

fn require_cell(molecule: &Molecule, mode: &str) -> pm7_rs::Result<()> {
    if molecule.cell.is_none() {
        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "`{mode}` needs a periodic cell: use an extended-XYZ file with a Lattice=\"...\" key, \
             or pass --cell a11,a12,a13,a21,...,a33 (Angstrom)"
        )));
    }
    Ok(())
}

fn parse_args(args: &[String]) -> pm7_rs::Result<Cli> {
    let mut cli = Cli {
        mode: args[1].clone(),
        path: args[2].clone(),
        charge: 0.0,
        multiplicity: 1,
        method: Pm7Method::Pm7,
        opt_output: None,
        json: false,
        use_diis: true,
        scf_tolerance: None,
        max_scf: None,
        cphf_max_iterations: None,
        stability: pm7_rs::stability::ScfStability::Off,
        exchange_cutoff: None,
        cell: None,
        cell_rows: None,
        pbc: None,
        dipole_origin: pm7_rs::DipoleOrigin::default(),
        axes: pm7_rs::AxisRotation::IDENTITY,
        kpoints: None,
        kshift: [0.0; 3],
        smearing: None,
        pbc_mode: PbcMode::Ewald,
        dandc: None,
        opt_cell: false,
        gtol: None,
        stress_tol: None,
        opt_max_iter: None,
        stability_every: None,
        supercell: [1, 1, 1],
        qpoints: Vec::new(),
        acoustic_sum_rule: true,
        field: None,
        ir: false,
        projection: pm7_rs::Projection::default(),
        reference: pm7_rs::ScfReference::Auto,
        molden_basis: "sto-6g".to_string(),
        lo_to: None,
        slab_thickness: None,
        wire_cross_section: None,
    };
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--exchange-cutoff" => {
                // Two Bohr values: inner (full exchange) and outer (zero exchange), smooth between.
                let inner: f64 = parse(args, index + 1, "--exchange-cutoff inner")?;
                let outer: f64 = parse(args, index + 2, "--exchange-cutoff outer")?;
                cli.exchange_cutoff = Some((inner, outer));
                index += 2;
            }
            "--field" => {
                // Three comma- or space-separated components in volts/Angstrom, MOPAC's
                // `FIELD=` convention (the potential gradient, so `E = +F.mu`).
                index += 1;
                let text = argument(args, index, "--field")?;
                let triple = parse_triple(text, "--field")?;
                cli.field = Some(pm7_rs::ExternalField::new(triple[0], triple[1], triple[2]));
            }
            "--ir" => cli.ir = true,
            "--projection" => {
                index += 1;
                cli.projection = pm7_rs::Projection::parse(argument(args, index, "--projection")?)?;
            }
            "--reference" => {
                index += 1;
                cli.reference = match argument(args, index, "--reference")? {
                    "auto" => pm7_rs::ScfReference::Auto,
                    "rhf" | "restricted" => pm7_rs::ScfReference::Restricted,
                    "uhf" | "unrestricted" => pm7_rs::ScfReference::Unrestricted,
                    other => {
                        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                            "unknown --reference `{other}`; expected auto, rhf or uhf"
                        )))
                    }
                };
            }
            "--molden-basis" => {
                index += 1;
                cli.molden_basis = argument(args, index, "--molden-basis")?.to_string();
            }
            // The loop adds its own `index += 1` for the flag itself, so an arm advances past its
            // *values* only. Advancing past both here skips the next flag entirely.
            "--slab-thickness" => {
                index += 1;
                cli.slab_thickness = Some(parse(args, index, "--slab-thickness ANGSTROM")?);
            }
            "--wire-cross-section" => {
                index += 1;
                cli.wire_cross_section =
                    Some(parse(args, index, "--wire-cross-section ANGSTROM2")?);
            }
            "--lo-to" => {
                index += 1;
                let text = argument(args, index, "--lo-to")?;
                cli.lo_to = Some(parse_triple(text, "--lo-to")?);
            }
            "--charge" => {
                index += 1;
                cli.charge = parse(args, index, "--charge")?;
            }
            "--multiplicity" => {
                index += 1;
                cli.multiplicity = parse(args, index, "--multiplicity")?;
            }
            "--method" => {
                index += 1;
                cli.method = Pm7Method::from_str(argument(args, index, "--method")?)?;
            }
            // Three spellings for one destination, because all three were already promised.
            // `--opt-output` is what the Rust CLI parsed; `--output`/`-o` is what the Python CLI
            // parses, what `MODE_FLAGS` listed for `optimize` and `molden`, and what the usage
            // text advertises for `molden` -- and neither of the latter two existed, so
            // `optimize x.xyz --output out.xyz` failed with "unknown option `--output`" against a
            // help text that had just named it.
            "--output" | "-o" | "--opt-output" => {
                let flag = args[index].clone();
                index += 1;
                cli.opt_output = Some(argument(args, index, &flag)?.to_owned());
            }
            "--cell" => {
                index += 1;
                let (cell, rows) = parse_cell(argument(args, index, "--cell")?)?;
                cli.cell = Some(cell);
                cli.cell_rows = rows;
            }
            "--pbc" => {
                index += 1;
                cli.pbc = Some(parse_pbc(argument(args, index, "--pbc")?)?);
            }
            "--dipole-origin" => {
                index += 1;
                cli.dipole_origin = match argument(args, index, "--dipole-origin")? {
                    "coordinates" | "origin" => pm7_rs::DipoleOrigin::Coordinates,
                    "com" | "centre-of-mass" | "center-of-mass" | "mass" => {
                        pm7_rs::DipoleOrigin::CentreOfMass
                    }
                    "charge" | "centre-of-charge" | "center-of-charge" => {
                        pm7_rs::DipoleOrigin::CentreOfCharge
                    }
                    other => {
                        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                            "unknown --dipole-origin `{other}`; expected coordinates, com or charge"
                        )))
                    }
                };
            }
            "--kpoints" => {
                let n1: usize = parse(args, index + 1, "--kpoints n1")?;
                let n2: usize = parse(args, index + 2, "--kpoints n2")?;
                let n3: usize = parse(args, index + 3, "--kpoints n3")?;
                cli.kpoints = Some([n1, n2, n3]);
                index += 3;
            }
            "--kshift" => {
                for k in 0..3 {
                    cli.kshift[k] = parse(args, index + 1 + k, "--kshift")?;
                }
                index += 3;
            }
            "--smearing" => {
                let kind = argument(args, index + 1, "--smearing KIND")?.to_owned();
                let width: f64 = parse(args, index + 2, "--smearing WIDTH")?;
                cli.smearing = Some(match kind.as_str() {
                    "fermi" => Smearing::FermiDirac { width_ev: width },
                    "gauss" | "gaussian" => Smearing::Gaussian { width_ev: width },
                    "mp" => Smearing::MethfesselPaxton {
                        width_ev: width,
                        order: 1,
                    },
                    "none" => Smearing::None,
                    other => {
                        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                            "unknown smearing `{other}`; expected fermi, gauss, mp or none"
                        )))
                    }
                });
                index += 2;
            }
            "--pbc-mode" => {
                index += 1;
                cli.pbc_mode = match argument(args, index, "--pbc-mode")? {
                    "ewald" => PbcMode::Ewald,
                    "mopac" => PbcMode::MopacCluster,
                    other => {
                        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                            "unknown --pbc-mode `{other}`; expected ewald or mopac"
                        )))
                    }
                };
            }
            // On/off: a bare `--dandc` turns divide and conquer on at the default buffer, and an
            // optional value overrides it. The value used to be mandatory, which made the common
            // case ("just use it") require knowing a number.
            //
            // The lookahead asks whether the next token *parses as a number*, not whether it
            // starts with a dash. `--dandc -5` is a negative buffer and has to reach the
            // validation that rejects it; reading it as the next flag would answer a bad request
            // with `unknown option -5`, which names the wrong thing.
            "--dandc"
                if args
                    .get(index + 1)
                    .map(|a| a.parse::<f64>().is_err())
                    .unwrap_or(true) =>
            {
                cli.dandc = Some(DandcOptions::default().buffer * BOHR_TO_ANGSTROM);
            }
            "--dandc" => {
                index += 1;
                cli.dandc = Some(parse(args, index, "--dandc BUFFER")?);
            }
            "--opt-cell" => cli.opt_cell = true,
            "--gtol" => {
                index += 1;
                cli.gtol = Some(parse(args, index, "--gtol")?);
            }
            "--stress-tol" => {
                index += 1;
                cli.stress_tol = Some(parse(args, index, "--stress-tol")?);
            }
            "--opt-max-iter" => {
                index += 1;
                cli.opt_max_iter = Some(parse(args, index, "--opt-max-iter")?);
            }
            "--stability-every" => {
                cli.stability_every = Some(parse(args, index, "--stability-every")?);
            }
            "--supercell" => {
                for k in 0..3 {
                    cli.supercell[k] = parse(args, index + 1 + k, "--supercell")?;
                }
                index += 3;
            }
            "--qpoints" => {
                // Every following `x,y,z` triple, up to the next flag.
                while index + 1 < args.len() && !args[index + 1].starts_with("--") {
                    index += 1;
                    cli.qpoints.push(parse_triple(&args[index], "--qpoints")?);
                }
                if cli.qpoints.is_empty() {
                    return Err(pm7_rs::Pm7Error::InvalidInput(
                        "--qpoints needs at least one `kx,ky,kz`".into(),
                    ));
                }
            }
            "--acoustic-sum-rule" => cli.acoustic_sum_rule = true,
            "--no-acoustic-sum-rule" => cli.acoustic_sum_rule = false,
            "--json" => cli.json = true,
            "--no-diis" => cli.use_diis = false,
            "--scf-tolerance" => {
                index += 1;
                cli.scf_tolerance = Some(parse(args, index, "--scf-tolerance TOL")?);
            }
            "--max-scf" => {
                index += 1;
                cli.max_scf = Some(parse(args, index, "--max-scf N")?);
            }
            "--stability" => {
                index += 1;
                let name = args.get(index).map(String::as_str).unwrap_or("");
                cli.stability = match name.to_ascii_lowercase().as_str() {
                    "off" | "none" => pm7_rs::stability::ScfStability::Off,
                    "check" => pm7_rs::stability::ScfStability::Check,
                    "follow" => pm7_rs::stability::ScfStability::Follow,
                    other => {
                        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                            "`--stability` wants `off`, `check` or `follow`, not `{other}`. \
                             `check` reports whether the converged SCF solution is a minimum; \
                             `follow` also re-converges from a rotated guess when it is not, and \
                             keeps the lower of the two."
                        )))
                    }
                };
            }
            "--cphf-max-iterations" => {
                index += 1;
                let n: usize = parse(args, index, "--cphf-max-iterations N")?;
                if n == 0 {
                    return Err(pm7_rs::Pm7Error::InvalidInput(
                        "`--cphf-max-iterations` must be at least 1: the orbital response is \
                         solved iteratively, and a budget of zero asks for no iterations at all \
                         rather than for a cheap answer."
                            .into(),
                    ));
                }
                cli.cphf_max_iterations = Some(n);
            }
            flag => {
                return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                    "unknown option `{flag}`"
                )))
            }
        }
        index += 1;
    }
    Ok(cli)
}

/// Nine comma-separated Ångström components, row major: `a_x,a_y,a_z,b_x,...,c_z`.
///
/// Also accepts three (a 1-D chain) or six (a 2-D sheet), so the flag covers every periodicity
/// the code does rather than only the 3-D case.
fn parse_cell(text: &str) -> pm7_rs::Result<(Cell, Option<[[f64; 3]; 3]>)> {
    let values: Vec<f64> = text
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| {
            pm7_rs::Pm7Error::InvalidInput(format!(
                "--cell wants comma-separated numbers: `{text}`"
            ))
        })?;
    if values.len() % 3 != 0 || values.is_empty() || values.len() > 9 {
        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "--cell wants 3, 6 or 9 numbers (1-D, 2-D or 3-D); got {}",
            values.len()
        )));
    }
    let rows: Vec<[f64; 3]> = values.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    // The full 3x3 is kept only when it was actually given: `--pbc` needs three real lattice
    // vectors to choose between, and three or six numbers have already chosen.
    let full = (rows.len() == 3).then(|| [rows[0], rows[1], rows[2]]);
    Ok((Cell::from_angstrom_rows(&rows)?, full))
}

/// `--pbc TTF`, spelled exactly as the Python CLI and extended XYZ spell it.
fn parse_pbc(text: &str) -> pm7_rs::Result<[bool; 3]> {
    let tokens: Vec<&str> = if text.contains([',', ' ']) {
        text.split([',', ' ']).filter(|t| !t.is_empty()).collect()
    } else {
        // `TFT` as three characters, which is how the Python CLI and every extended-XYZ file
        // written by ASE spell it.
        text.split("").filter(|t| !t.is_empty()).collect()
    };
    if tokens.len() != 3 {
        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "--pbc wants three flags, one per lattice vector, e.g. `TTF` or `T,T,F`; got `{text}`"
        )));
    }
    let mut flags = [false; 3];
    for (slot, token) in flags.iter_mut().zip(&tokens) {
        *slot = match *token {
            "T" | "t" | "True" | "true" | "1" => true,
            "F" | "f" | "False" | "false" | "0" => false,
            other => {
                return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                    "--pbc flags are T or F; got `{other}` in `{text}`"
                )))
            }
        };
    }
    Ok(flags)
}

fn parse_triple(text: &str, flag: &str) -> pm7_rs::Result<[f64; 3]> {
    let values: Vec<f64> = text
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| pm7_rs::Pm7Error::InvalidInput(format!("{flag} wants `x,y,z`: `{text}`")))?;
    if values.len() != 3 {
        return Err(pm7_rs::Pm7Error::InvalidInput(format!(
            "{flag} wants `x,y,z`, got {} numbers in `{text}`",
            values.len()
        )));
    }
    Ok([values[0], values[1], values[2]])
}

fn argument<'a>(args: &'a [String], index: usize, flag: &str) -> pm7_rs::Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| pm7_rs::Pm7Error::InvalidInput(format!("{flag} needs a value")))
}

fn parse<T: FromStr>(args: &[String], index: usize, flag: &str) -> pm7_rs::Result<T> {
    argument(args, index, flag)?
        .parse()
        .map_err(|_| pm7_rs::Pm7Error::InvalidInput(format!("invalid value for {flag}")))
}

fn print_usage() {
    eprintln!(
        "usage: pm7_rs_cli <mode> structure.xyz [options]\n\
         \n\
         modes\n  \
           energy        total energy and heat of formation\n  \
           charges       energy plus Mulliken charges\n  \
           gradient      analytic nuclear gradient dE/dR (eV/Bohr)\n  \
           forces        the same, negated: -dE/dR\n  \
           stress        analytic stress tensor and pressure   [periodic]\n  \
           optimize      L-BFGS geometry optimization\n  \
           frequencies   harmonic vibrational frequencies; --ir adds intensities\n  \
           hessian       the analytic Cartesian Hessian (eV/Bohr^2)\n  \
           orbitals      orbital energies and occupations; --json adds the coefficients\n  \
           molden        the wavefunction in Molden format; -o writes it to a file\n  \
           phonons       force constants and frequencies at one or more q  [periodic]\n  \
           dfpt          phonons at arbitrary q, no supercell   [periodic]\n  \
           born          Born effective charges and eps_inf     [periodic]\n  \
           bands         band energies along a k path           [periodic]\n  \
           dielectric    eps_inf for a chain or a slab          [periodic]\n\
         \n\
         general options\n  \
           --method NAME              pm7 (default), pm7-ts, pm7-, pm7-hh, pm7-sparkle\n  \
           --reference auto|rhf|uhf   SCF reference; auto picks RHF or UHF by electron count\n  \
           --charge Q                 net charge\n  \
           --multiplicity M           spin multiplicity\n  \
           --molden-basis B           sto-6g (default) or sto, for `molden`\n  \
           --lo-to X,Y,Z              LO-TO direction for `born`, `dfpt` and `phonons`\n                             \
                             (3-D only; no default, because the q -> 0 limit is\n                             \
                             direction dependent)\n  \
           --json                     machine-readable output\n  \
           --no-diis                  plain SCF iteration\n  \
           --output FILE, -o FILE     write the result to a file (optimize, molden)\n  \
           --opt-output FILE          accepted alias for --output\n  \
           --opt-cell                 relax the lattice too, not just the atoms (optimize)\n  \
           --gtol EV_PER_BOHR         optimizer force convergence (default 1e-3)\n  \
           --stress-tol EV_PER_BOHR3  optimizer stress convergence, with --opt-cell\n  \
           --opt-max-iter N           optimizer iteration limit (default 200)\n  \
  --stability-every N        re-check SCF stability every Nth optimizer step\n                             \
                             (default 0, never)\n  \
           --exchange-cutoff IN OUT   smooth long-range-exchange cutoff (Bohr) for the CPHF;\n                             \
                             omitted = exact and bit-identical\n  \
           --scf-tolerance TOL        SCF density convergence (default 1e-7)\n  \
           --max-scf N                SCF iteration budget (default 200)\n  \
           --cphf-max-iterations N    CPHF orbital-response budget (default 100); raise it when\n                             \
                             a small-gap cell reports an unconverged response\n  \
           --stability MODE           off (default), check, or follow: is the converged SCF\n                             \
                             solution a minimum, and escape it if not\n  \
           --dandc [BUFFER]           divide-and-conquer SCF, for `energy`, `charges` and\n                             \
                             `optimize`. BUFFER is the buffer radius in Angstrom and\n                             \
                             defaults to 7.9; do not go below 7\n  \
           --field FX,FY,FZ           uniform electric field in V/Angstrom, MOPAC FIELD= sign\n  \
           --dipole-origin WHERE      coordinates | com | charge, for a charged species' dipole\n  \
           --ir                       with frequencies, also report IR intensities\n  \
           --projection KIND          rigid (default) | translations | none, for frequencies;\n                             \
                             `none` is the raw 3N set with the rigid motions left in\n\
         \n\
         periodic options (the cell comes from an extended-XYZ Lattice=\"...\" key, or --cell)\n  \
           --cell A,B,C,...           3, 6 or 9 comma-separated Angstrom components, row major\n  \
           --pbc TTF                  which lattice vectors are periodic; any pattern, e.g. TFT\n  \
           --kpoints N1 N2 N3         Gamma-centred Monkhorst-Pack mesh (default: Gamma)\n  \
           --kshift S1 S2 S3          mesh shift in fractional units\n  \
           --smearing KIND WIDTH      fermi | gauss | mp | none, width in eV\n  \
           --pbc-mode ewald|mopac     ewald (default) or the MOPAC-compatible truncated sum\n  \
           --supercell N1 N2 N3       force-constant supercell (phonons; default 1 1 1)\n  \
           --qpoints X,Y,Z [X,Y,Z...] fractional q (phonons) or k (bands) points\n  \
           --acoustic-sum-rule        project the acoustic sum rule out (the default)\n  \
           --no-acoustic-sum-rule     keep the raw force constants, residual and all\n  \
           --slab-thickness A         material thickness for dielectric on a 2-D cell\n  \
           --wire-cross-section A2    material cross-section for dielectric on a 1-D cell\n                             \
                             (required: a supercell says where the atoms are, not\n                             \
                             where the material ends)"
    );
}
