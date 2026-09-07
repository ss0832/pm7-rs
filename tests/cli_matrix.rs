// SPDX-License-Identifier: GPL-3.0-or-later
//! Every mode of the command line against every flag that mode accepts.
//!
//! `tests/cli.rs` asks whether each mode gives the right answer. This asks something cheaper and
//! more mechanical: whether any *combination* of flags can make the binary behave badly. A CLI is
//! the one surface with no type checking on it, and the failure that matters is not a wrong number
//! — it is a panic, a silent success with a flag the mode never read, or an error message that
//! names an internal instead of the flag the user got wrong.
//!
//! "All combinations" is not the power set: twenty-five flags would be thirty-three million runs
//! and most of them meaningless. It is, per mode, the cross-product over the flags that mode
//! accepts with two or three representative values each — the same thing a person means by
//! "every combination of the k-mesh, the smearing and the reference". [`BLOCKS`] is that table.
//!
//! Three tiers:
//!
//! * the default tier, which every `cargo test` runs: a representative subset covering every mode
//!   and every flag at least once, the documented rejections, and the no-op ledger below. Under
//!   twenty seconds.
//! * `cargo test --release --test cli_matrix -- --ignored`, the full cross-product.
//! * [`a_flag_either_changes_the_answer_or_is_a_pinned_no_op`], which is the only test here that
//!   can see a *silently ignored* flag at all. The invariant one run can check is "succeeded, or
//!   failed and said why"; "the flag was read" needs the same command run twice, so it lives in
//!   its own differential test with an explicit ledger of the flags that today do nothing.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_pm7_rs_cli")
}

// --- the systems ---------------------------------------------------------------------------
//
// Small enough that a single point is a few tens of milliseconds, and between them they cover
// every periodicity the code has: a molecule, an open shell, a chain, a sheet and a crystal.

/// The fixture files, written once per test process.
struct Fixtures {
    water: PathBuf,
    methyl: PathBuf,
    h2: PathBuf,
    diamond: PathBuf,
    chain: PathBuf,
    sheet: PathBuf,
    missing: PathBuf,
}

fn scratch(name: &str, contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("pm7_matrix_{name}"));
    std::fs::write(&path, contents).expect("write scratch file");
    path
}

fn fixtures() -> &'static Fixtures {
    static FIXTURES: OnceLock<Fixtures> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let mut missing = std::env::temp_dir();
        missing.push("pm7_matrix_there_is_no_such_file.xyz");
        let _ = std::fs::remove_file(&missing);
        Fixtures {
            water: scratch(
                "water.xyz",
                "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
            ),
            // A doublet, for --multiplicity and --reference.
            methyl: scratch(
                "methyl.xyz",
                "4\nmethyl radical\n\
                 C  0.0000  0.0000 0.0\n\
                 H  1.0790  0.0000 0.0\n\
                 H -0.5395  0.9345 0.0\n\
                 H -0.5395 -0.9345 0.0\n",
            ),
            // Plain XYZ, so that --cell is the only possible source of a lattice.
            h2: scratch("h2.xyz", "2\nH2\nH 0.0 0.0 0.0\nH 0.75 0.0 0.0\n"),
            diamond: scratch(
                "diamond.xyz",
                "2\n\
                 Lattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" \
                 Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n\
                 C 0.000000 0.000000 0.000000\n\
                 C 0.891750 0.891750 0.891750\n",
            ),
            // 1-D: nine lattice numbers, but `pbc="T F F"` makes only the first periodic.
            chain: scratch(
                "chain.xyz",
                "2\n\
                 Lattice=\"2.6 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0\" \
                 Properties=species:S:1:pos:R:3 pbc=\"T F F\"\n\
                 H 0.000000 0.000000 0.000000\n\
                 H 0.750000 0.000000 0.000000\n",
            ),
            // 2-D.
            sheet: scratch(
                "bn.xyz",
                "2\n\
                 Lattice=\"2.510 0.0 0.0 -1.255 2.1738 0.0 0.0 0.0 20.0\" \
                 Properties=species:S:1:pos:R:3 pbc=\"T T F\"\n\
                 B 0.000000 0.000000 0.000000\n\
                 N 1.255000 0.724600 0.000000\n",
            ),
            missing,
        }
    })
}

/// Expand one token of an axis value or a rejection template.
///
/// Substitution happens *after* splitting on whitespace, so a temporary directory with a space in
/// its name stays a single argument.
fn expand(token: &str, index: usize) -> String {
    let f = fixtures();
    let path = |p: &PathBuf| p.to_str().expect("utf-8 temp path").to_owned();
    match token {
        "{water}" => path(&f.water),
        "{methyl}" => path(&f.methyl),
        "{h2}" => path(&f.h2),
        "{diamond}" => path(&f.diamond),
        "{chain}" => path(&f.chain),
        "{sheet}" => path(&f.sheet),
        "{missing}" => path(&f.missing),
        // Unique per case: the matrix runs several workers at once, and two cases sharing an
        // output path would race rather than test anything.
        "{out}" => {
            let mut p = std::env::temp_dir();
            p.push(format!("pm7_matrix_out_{index}.xyz"));
            path(&p)
        }
        other => other.to_owned(),
    }
}

fn tokens(text: &str, index: usize) -> Vec<String> {
    text.split_whitespace().map(|t| expand(t, index)).collect()
}

/// A structure and whatever flags it takes to make it a system worth asking about.
///
/// Charge and multiplicity travel with the structure rather than forming their own axis: a
/// doublet multiplicity on water is not a combination, it is an arithmetic error, and the matrix
/// would spend a sixth of itself on the same parity message.
const SYSTEMS: &[(&str, &str)] = &[
    ("water", "{water}"),
    ("water-cation", "{water} --charge 1 --multiplicity 2"),
    ("methyl", "{methyl} --multiplicity 2"),
    ("diamond", "{diamond}"),
    ("chain", "{chain}"),
    ("sheet", "{sheet}"),
    // The two systems whose cell comes from --cell rather than from the file, in the nine-number
    // and the three-number forms.
    ("boxed-water", "{water} --cell 8.0,0,0,0,8.0,0,0,0,8.0"),
    ("cell-chain", "{h2} --cell 2.6,0,0"),
];

fn system(name: &str) -> &'static str {
    SYSTEMS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .unwrap_or_else(|| panic!("no system named `{name}`"))
}

// --- the axes ------------------------------------------------------------------------------
//
// An empty string means the flag is absent, which is always one of the representative values:
// "the default" is a value like any other and half the regressions live there.

const JSON: &[&str] = &["", "--json"];
const DIIS: &[&str] = &["", "--no-diis"];
const METHOD: &[&str] = &["", "--method pm7-ts"];
const REFERENCE: &[&str] = &["", "--reference rhf", "--reference uhf"];
const SCF: &[&str] = &["", "--scf-tolerance 1e-5", "--max-scf 400"];
// The CPHF budget, a private constant through 0.2.2. Crossed on `phonons` rather than on the
// molecular block because that is where it binds: a small-gap cell is the case that needs it.
const CPHF: &[&str] = &["", "--cphf-max-iterations 400"];
const FIELD: &[&str] = &["", "--field 0.2,0,0", "--field 0,0,-0.5"];
const EXCHANGE: &[&str] = &["", "--exchange-cutoff 6 12", "--exchange-cutoff 0 8"];
const IR: &[&str] = &["", "--ir"];
const PROJECTION: &[&str] = &["", "--projection none", "--projection translations"];
const DIPOLE_ORIGIN: &[&str] = &["", "--dipole-origin com", "--dipole-origin charge"];
const MOLDEN_BASIS: &[&str] = &["", "--molden-basis sto", "--molden-basis sto-3g"];
// Three spellings for one destination. `--output`/`-o` is what the Python CLI uses and
// what MODE_FLAGS listed; neither had a parse arm before 0.2.3.
const OPT_OUTPUT: &[&str] = &["", "--opt-output {out}", "--output {out}"];
// Variable-cell relaxation and the optimizer knobs, none of which reached either
// command line before 0.2.3 -- `OptOptions` was `::default()` at every call site.
const OPT: &[&str] = &[
    "",
    "--opt-cell",
    "--gtol 1e-2",
    "--opt-max-iter 3",
    "--stress-tol 1e-4",
    "--stability-every 2",
];
// Is the converged SCF a minimum? `follow` is the one that can *change* the answer, so it belongs
// in the matrix rather than only in the H2 test: on a stable molecule it must return the same
// number, which is the property that makes the option safe to switch on.
const STABILITY: &[&str] = &["", "--stability check", "--stability follow"];
// `2 1 1` is legal for every periodicity; `2 2 2` asks a chain and a sheet to repeat a direction
// they do not have, which is a rejection the matrix should see rather than avoid.
const KMESH: &[&str] = &["", "--kpoints 2 1 1", "--kpoints 2 2 2"];
// Only 0 and 0.5 keep the mesh closed under k -> -k; 0.25 is in the rejection table instead.
const KSHIFT: &[&str] = &["", "--kshift 0.5 0.0 0.0"];
// Every per-axis pattern, including the ones that need the lattice vectors reordered.
const PBC: &[&str] = &["", "--pbc TTT", "--pbc TFT", "--pbc FTF", "--pbc TTF"];
const SMEARING_A: &[&str] = &["", "--smearing fermi 0.2", "--smearing mp 0.15"];
const SMEARING_B: &[&str] = &["", "--smearing gauss 0.1", "--smearing none 0.0"];
const PBC_MODE_ALL: &[&str] = &["", "--pbc-mode ewald", "--pbc-mode mopac"];
const PBC_MODE: &[&str] = &["", "--pbc-mode mopac"];
const SUPERCELL: &[&str] = &["", "--supercell 2 1 1", "--supercell 2 2 2"];
// The sum rule is enforced by default since 0.2.3, so the axis is its negation:
// `--acoustic-sum-rule` now names the default and is inert.
const ASR: &[&str] = &["", "--no-acoustic-sum-rule", "--acoustic-sum-rule"];
const LO_TO: &[&str] = &["", "--lo-to 1,0,0"];
const QPOINTS_PAIR: &[&str] = &["", "--qpoints 0,0,0 0.5,0,0"];
const QPOINTS: &[&str] = &["", "--qpoints 0,0,0", "--qpoints 0,0,0 0.5,0,0"];
const KPATH: &[&str] = &["", "--qpoints 0,0,0", "--qpoints 0,0,0 0.5,0,0 0.5,0.5,0.0"];
const EXTENT: &[&str] = &[
    "",
    "--slab-thickness 3.33",
    "--wire-cross-section 9.0",
    "--slab-thickness 3.33 --wire-cross-section 9.0",
];
/// Flags that belong to some *other* mode. Passing one is how a user finds out whether the CLI
/// reads its arguments or merely parses them.
const STRAY: &[&str] = &[
    "--supercell 2 1 1",
    "--qpoints 0,0,0",
    "--acoustic-sum-rule",
    "--ir",
    "--molden-basis sto",
    "--opt-output {out}",
    "--lo-to 1,0,0",
    "--slab-thickness 3.33",
    "--wire-cross-section 9.0",
    "--dandc 15.0",
];

/// One rectangle of the matrix: some modes, some systems, and the axes those modes accept.
struct Block {
    name: &'static str,
    modes: &'static [&'static str],
    systems: &'static [&'static str],
    axes: &'static [&'static [&'static str]],
}

const MOLECULAR_MODES: &[&str] = &[
    "energy",
    "charges",
    "gradient",
    "forces",
    "orbitals",
    "hessian",
    "frequencies",
    "optimize",
    "molden",
];

const BLOCKS: &[Block] = &[
    // The SCF itself, crossed with everything that changes how it is solved.
    Block {
        name: "molecular-core",
        modes: MOLECULAR_MODES,
        systems: &["water", "water-cation", "methyl"],
        axes: &[METHOD, REFERENCE, JSON, DIIS, SCF],
    },
    // The flags that only matter once a derivative or a file is involved.
    Block {
        name: "molecular-extras",
        modes: &["energy", "frequencies", "hessian", "molden", "optimize"],
        systems: &["water", "methyl"],
        axes: &[FIELD, EXCHANGE, IR, MOLDEN_BASIS, OPT_OUTPUT],
    },
    // Everything a cell adds, on all three periodicities plus a box supplied by --cell.
    Block {
        name: "periodic-core",
        modes: &[
            "energy", "charges", "gradient", "forces", "stress", "orbitals", "hessian",
        ],
        systems: &["diamond", "chain", "sheet", "boxed-water"],
        axes: &[KMESH, KSHIFT, SMEARING_A, PBC_MODE, JSON],
    },
    Block {
        name: "phonons",
        modes: &["phonons"],
        systems: &["diamond", "chain", "sheet"],
        axes: &[
            SUPERCELL,
            QPOINTS_PAIR,
            ASR,
            LO_TO,
            JSON,
            &["", "--kpoints 2 1 1"],
            CPHF,
        ],
    },
    Block {
        name: "dfpt",
        modes: &["dfpt"],
        systems: &["diamond", "chain", "sheet"],
        axes: &[
            QPOINTS,
            LO_TO,
            KMESH,
            JSON,
            &["", "--exchange-cutoff 6 12"],
            &["", "--scf-tolerance 1e-5"],
        ],
    },
    Block {
        name: "born",
        modes: &["born"],
        systems: &["diamond", "chain", "sheet"],
        axes: &[
            KMESH,
            SMEARING_B,
            LO_TO,
            JSON,
            &["", "--exchange-cutoff 6 12"],
            DIIS,
        ],
    },
    Block {
        name: "bands",
        modes: &["bands"],
        systems: &["diamond", "chain", "sheet"],
        axes: &[KPATH, KMESH, SMEARING_B, PBC_MODE_ALL, JSON],
    },
    Block {
        name: "dielectric",
        modes: &["dielectric"],
        systems: &["chain", "sheet", "diamond", "cell-chain"],
        axes: &[EXTENT, KMESH, JSON, METHOD, DIIS],
    },
    // Per-axis periodicity, and the two flags that only reached this CLI in 0.2.3.
    Block {
        name: "per-axis-pbc",
        modes: &["energy", "orbitals", "stress"],
        systems: &["boxed-water"],
        axes: &[PBC, KMESH, JSON],
    },
    Block {
        name: "stability",
        modes: &["energy", "charges", "orbitals", "optimize"],
        systems: &["water", "methyl", "water-cation"],
        axes: &[STABILITY, JSON],
    },
    Block {
        name: "optimizer",
        modes: &["optimize"],
        systems: &["water", "boxed-water"],
        axes: &[OPT, JSON],
    },
    Block {
        name: "projection-and-dipole-origin",
        modes: &["energy", "frequencies"],
        systems: &["water", "water-cation"],
        axes: &[PROJECTION, DIPOLE_ORIGIN],
    },
    // The part that is deliberately nonsense: a flag from another mode, on every mode.
    Block {
        name: "stray-flags",
        modes: MODES,
        systems: &["water", "diamond"],
        axes: &[STRAY, JSON],
    },
];

const MODES: &[&str] = &[
    "energy",
    "charges",
    "gradient",
    "forces",
    "stress",
    "optimize",
    "frequencies",
    "hessian",
    "orbitals",
    "molden",
    "phonons",
    "dfpt",
    "born",
    "bands",
    "dielectric",
];

/// Every flag `--help` advertises. The coverage assertion is against this list, so a flag added
/// to the CLI and not to an axis fails the default tier rather than going untested.
const FLAGS: &[&str] = &[
    "--acoustic-sum-rule",
    "--no-acoustic-sum-rule",
    "--cell",
    "--charge",
    "--dandc",
    "--exchange-cutoff",
    "--field",
    "--ir",
    "--json",
    "--kpoints",
    "--kshift",
    "--lo-to",
    "--max-scf",
    "--cphf-max-iterations",
    "--method",
    "--molden-basis",
    "--multiplicity",
    "--no-diis",
    "--opt-output",
    "--pbc",
    "--pbc-mode",
    "--projection",
    "--dipole-origin",
    "--output",
    "--opt-cell",
    "--gtol",
    "--stress-tol",
    "--opt-max-iter",
    "--stability",
    "--stability-every",
    "--qpoints",
    "--reference",
    "--scf-tolerance",
    "--slab-thickness",
    "--smearing",
    "--supercell",
    "--wire-cross-section",
];

// --- generation ----------------------------------------------------------------------------

struct Case {
    args: Vec<String>,
    label: String,
}

/// The whole matrix, in a fixed order so that a case index means the same thing on every run.
fn matrix() -> Vec<Case> {
    let mut cases = Vec::new();
    for block in BLOCKS {
        for mode in block.modes {
            for name in block.systems {
                let head = system(name);
                let radix: Vec<usize> = block.axes.iter().map(|a| a.len()).collect();
                let total: usize = radix.iter().product();
                for mut n in 0..total {
                    let index = cases.len();
                    let mut args = vec![(*mode).to_owned()];
                    args.extend(tokens(head, index));
                    for (axis, base) in block.axes.iter().zip(&radix) {
                        args.extend(tokens(axis[n % base], index));
                        n /= base;
                    }
                    let label = format!("[{}] {}", block.name, args.join(" "));
                    cases.push(Case { args, label });
                }
            }
        }
    }
    cases
}

/// Invocations that must be refused, and the words the refusal has to contain.
///
/// A rejection is worth more than an acceptance here: the whole point of the invariant is that a
/// wrong flag stops the run and says so, and these are the wrong flags.
const REJECTIONS: &[(&str, &str)] = &[
    ("energy {water} --kpoint 2", "unknown option"),
    ("energy {water} --KPOINTS 2 2 2", "unknown option"),
    ("energy {water} --help", "unknown option"),
    ("energy {water} -j", "unknown option"),
    ("nonsense {water}", "unknown mode"),
    ("energy {water} --kpoints", "needs a value"),
    ("energy {water} --kpoints 2", "needs a value"),
    ("energy {water} --kpoints 2 2", "needs a value"),
    ("energy {water} --smearing", "needs a value"),
    ("energy {water} --smearing fermi", "needs a value"),
    ("energy {water} --method", "needs a value"),
    ("energy {water} --method nonsense", "unknown PM7 method"),
    ("energy {water} --reference nonsense", "auto, rhf or uhf"),
    (
        "energy {water} --pbc-mode nonsense",
        "expected ewald or mopac",
    ),
    (
        "energy {diamond} --smearing bogus 0.2",
        "expected fermi, gauss, mp or none",
    ),
    ("molden {water} --molden-basis nonsense", "sto-nG"),
    ("molden {water} --molden-basis sto-13g", "sto-nG"),
    ("energy {water} --field junk", "wants `x,y,z`"),
    ("energy {water} --field 1,2", "wants `x,y,z`"),
    ("energy {water} --lo-to 1,0", "wants `x,y,z`"),
    ("energy {water} --cell 1,2", "3, 6 or 9 numbers"),
    ("energy {water} --cell 0,0,0", "non-zero length"),
    ("energy {water} --scf-tolerance abc", "invalid value"),
    ("energy {water} --scf-tolerance 0", "p_tol"),
    ("energy {water} --max-scf 0", "max_scf"),
    ("energy {water} --max-scf -1", "invalid value"),
    ("frequencies {water} --cphf-max-iterations 0", "at least 1"),
    (
        "frequencies {water} --cphf-max-iterations -1",
        "invalid value",
    ),
    // Small enough that water's own response cannot finish, which is the refusal the flag exists
    // to let a caller lift. It has to *name* the flag: a budget the caller set is the first thing
    // to reconsider when the budget runs out.
    (
        "frequencies {water} --cphf-max-iterations 2",
        "cphf-max-iterations",
    ),
    (
        "energy {water} --multiplicity 0",
        "multiplicity must be >= 1",
    ),
    ("energy {water} --multiplicity -1", "invalid value"),
    // Eight electrons: a doublet is arithmetically impossible, a triplet is merely excited.
    ("energy {water} --multiplicity 2", "parity"),
    ("energy {water} --charge 1", "parity"),
    (
        "energy {methyl} --multiplicity 2 --reference rhf",
        "needs a closed shell",
    ),
    ("energy {water} --dandc 0", "buffer must be finite"),
    ("energy {water} --dandc -5", "buffer must be finite"),
    (
        "energy {water} --exchange-cutoff 12 6",
        "0 <= inner < outer",
    ),
    ("energy {water} --exchange-cutoff 0 0", "0 <= inner < outer"),
    ("energy {diamond} --kpoints 0 0 0", "must be >= 1"),
    (
        "energy {diamond} --kpoints 2 2 2 --smearing fermi -1",
        "smearing width",
    ),
    (
        "energy {diamond} --kpoints 2 2 2 --smearing fermi 0",
        "smearing width",
    ),
    (
        "energy {diamond} --kpoints 2 2 2 --kshift 0.25 0.25 0.25",
        "only 0 and 0.5",
    ),
    ("stress {water}", "needs a periodic cell"),
    ("phonons {water}", "needs a periodic cell"),
    ("dfpt {water}", "needs a periodic cell"),
    ("born {water}", "needs a periodic cell"),
    ("bands {water}", "needs a periodic cell"),
    ("dielectric {water}", "needs a periodic cell"),
    ("bands {diamond} --qpoints", "at least one"),
    ("phonons {diamond} --supercell 0 0 0", "at least 1"),
    ("dielectric {diamond}", "--slab-thickness"),
    (
        "dielectric {sheet} --slab-thickness 3.3 --wire-cross-section 9",
        "two different conventions",
    ),
    (
        "dielectric {sheet} --wire-cross-section 9",
        "periodic direction",
    ),
    (
        "dielectric {sheet} --slab-thickness 0",
        "must be a positive",
    ),
    (
        "dielectric {diamond} --slab-thickness 3.3",
        "already has a volume",
    ),
    ("frequencies {diamond} --ir", "born"),
    ("phonons {sheet} --lo-to 1,0,0", "needs a 3-D cell"),
    ("phonons {diamond} --lo-to 0,0,0", "non-zero vector"),
    // Advertised in --help and refused on purpose: the truncated MOPAC lattice sum was never
    // finished. The refusal is the contract, so it is pinned here rather than left to chance.
    (
        "energy {diamond} --pbc-mode mopac",
        "MopacCluster is not implemented yet",
    ),
    (
        "energy {diamond} --pbc-mode mopac --kpoints 2 2 2",
        "PbcMode::Ewald",
    ),
];

// --- the invariant -------------------------------------------------------------------------

/// Failure messages that name neither the flag that caused them nor a flag to reach for instead.
///
/// This is a defect ledger, not a permission slip: each entry is a message the CLI could say
/// better, pinned so that the matrix stays green while a *new* mute failure turns it red. None
/// of them is a panic and all of them are true; they simply leave the reader to guess which of
/// the flags they typed the complaint is about.
const MUTE_FAILURES: &[&str] = &[
    // `--kpoints 2 2 2` on a chain or a sheet. Names neither --kpoints nor --cell.
    "cannot repeat non-periodic direction",
    // `bands` with no --kpoints. "run with a k mesh" is the remedy, spelled without the flag.
    "a band structure needs the translation-resolved density",
    // `--max-scf` too small. Does not suggest raising it.
    "did not converge after",
    // A structure file that is not there: the OS message, localized, with no path in it.
    "(os error 2)",
    // `--kshift` off the allowed set: a paragraph on time reversal that never says --kshift.
    "is not supported: only 0 and 0.5",
];

/// Lowercase and drop every separator, so that `--pbc-mode` matches `PbcMode` and `--max-scf`
/// matches `max_scf`. A message is allowed to spell a flag the way the code spells it.
fn squash(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Words a message may use in place of the flag itself. Prose is allowed to be prose.
const ALIASES: &[(&str, &str)] = &[
    ("--dandc", "divide-and-conquer"),
    ("--cell", "lattice"),
    ("--scf-tolerance", "p_tol"),
    ("--kpoints", "Monkhorst"),
    ("--ir", "infrared"),
    ("--slab-thickness", "thickness"),
    ("--slab-thickness", "extent"),
    ("--wire-cross-section", "cross-section"),
    ("--wire-cross-section", "extent"),
    ("--multiplicity", "electron count"),
    ("--charge", "electron count"),
    ("--reference", "closed shell"),
];

/// Does this message let the reader find the flag they got wrong, or the one to reach for?
fn explains(args: &[String], stderr: &str) -> bool {
    // A message that names any flag has said what to do, whether or not it is the one passed.
    if stderr.contains("--") {
        return true;
    }
    let flat = squash(stderr);
    // The mode is the other half of the invocation, and naming it is naming the problem.
    if flat.contains(&squash(&args[0])) {
        return true;
    }
    for arg in args {
        let Some(bare) = arg.strip_prefix("--") else {
            continue;
        };
        let key = squash(bare);
        // Two-letter stems like `ir` match half the dictionary; those go through ALIASES.
        if key.len() >= 4 && flat.contains(&key) {
            return true;
        }
        if ALIASES
            .iter()
            .any(|(flag, word)| flag == arg && flat.contains(&squash(word)))
        {
            return true;
        }
    }
    false
}

/// The invariant, for one invocation: it succeeded, or it failed and said why.
fn check(args: &[String]) -> Result<(), String> {
    let out = Command::new(binary())
        .args(args)
        .output()
        .map_err(|e| format!("could not run the binary: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // A panic is never an answer, however wrong the flags were.
    if stderr.contains("panicked at") || stdout.contains("panicked at") {
        return Err(format!("panicked: {}", first_line(&stderr)));
    }
    if out.status.code() == Some(101) {
        return Err(format!("exit 101 (panic): {}", first_line(&stderr)));
    }
    if out.status.success() {
        return Ok(());
    }
    match out.status.code() {
        // 1 is the CLI's own refusal; 2 is the usage banner, which only <3 arguments can reach.
        Some(1) => {
            if !stderr.starts_with("pm7-rs: ") {
                return Err(format!(
                    "a refusal must come through the `pm7-rs:` channel: {}",
                    first_line(&stderr)
                ));
            }
        }
        Some(2) => {
            if !stderr.starts_with("usage: pm7_rs_cli") {
                return Err(format!(
                    "exit 2 without the usage banner: {}",
                    first_line(&stderr)
                ));
            }
            return Ok(());
        }
        other => {
            return Err(format!(
                "unexpected exit code {other:?}: {}",
                first_line(&stderr)
            ))
        }
    }
    if stderr.trim().len() < "pm7-rs: ".len() + 8 {
        return Err(format!("failed with no explanation: {stderr:?}"));
    }
    if !explains(args, &stderr) && !MUTE_FAILURES.iter().any(|m| stderr.contains(m)) {
        return Err(format!(
            "the refusal names neither the flag nor a remedy: {}",
            first_line(&stderr)
        ));
    }
    Ok(())
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_owned()
}

/// Run a batch of cases across a few workers and return every violation, sorted.
///
/// A handful of workers rather than one: each invocation is a whole process, mostly spent
/// starting up, and the full matrix is thousands of them. More than four oversubscribes the SCF's
/// own thread pool and stops helping.
fn run_cases(cases: &[Case]) -> Vec<String> {
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 4);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(case) = cases.get(index) else { break };
                if let Err(why) = check(&case.args) {
                    failures
                        .lock()
                        .expect("failure list")
                        .push(format!("{}\n      {why}", case.label));
                }
            });
        }
    });
    let mut failures = failures.into_inner().expect("failure list");
    failures.sort();
    failures
}

fn report(what: &str, ran: usize, failures: &[String]) {
    if failures.is_empty() {
        return;
    }
    // A broken invariant tends to break for a whole axis at once, and a panic message with three
    // thousand entries in it is not a report. The first forty say what happened.
    const SHOWN: usize = 40;
    let listed = failures
        .iter()
        .take(SHOWN)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n  ");
    let rest = failures.len().saturating_sub(SHOWN);
    let more = if rest == 0 {
        String::new()
    } else {
        format!("\n  ... and {rest} more")
    };
    panic!(
        "{} of {ran} {what} invocations broke the invariant:\n  {listed}{more}",
        failures.len()
    );
}

// --- tier one: what every `cargo test` runs --------------------------------------------------

/// The greedy cover: the first case that brings in a mode or a flag nothing before it used.
///
/// Greedy over the matrix in its natural order rather than a hand-written list, so that a new
/// axis is represented the moment it is added to [`BLOCKS`] and nobody has to remember.
fn representative(cases: &[Case]) -> Vec<&Case> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut chosen = Vec::new();
    for case in cases {
        let novel: Vec<&str> = case
            .args
            .iter()
            .filter(|a| a.starts_with("--") || MODES.contains(&a.as_str()))
            .map(String::as_str)
            .filter(|a| !seen.contains(a))
            .collect();
        if !novel.is_empty() {
            seen.extend(novel);
            chosen.push(case);
        }
    }
    chosen
}

#[test]
fn the_representative_subset_covers_every_mode_and_every_flag() {
    let cases = matrix();
    let chosen = representative(&cases);
    let used: BTreeSet<&str> = chosen
        .iter()
        .flat_map(|c| c.args.iter().map(String::as_str))
        .collect();

    let missing_modes: Vec<&&str> = MODES.iter().filter(|m| !used.contains(**m)).collect();
    assert!(
        missing_modes.is_empty(),
        "the subset never runs {missing_modes:?}"
    );
    let missing_flags: Vec<&&str> = FLAGS.iter().filter(|f| !used.contains(**f)).collect();
    assert!(
        missing_flags.is_empty(),
        "the subset never passes {missing_flags:?}; add an axis to BLOCKS"
    );

    // Cheap enough to keep in the default tier: this is the number that has to stay small.
    assert!(
        chosen.len() < 80,
        "the cover grew to {} cases; the default tier is meant to be seconds",
        chosen.len()
    );

    let owned: Vec<Case> = chosen
        .into_iter()
        .map(|c| Case {
            args: c.args.clone(),
            label: c.label.clone(),
        })
        .collect();
    let failures = run_cases(&owned);
    report("representative", owned.len(), &failures);
}

#[test]
fn every_documented_rejection_says_what_is_wrong() {
    let mut failures = Vec::new();
    for (index, (template, expected)) in REJECTIONS.iter().enumerate() {
        let args = tokens(template, 1_000_000 + index);
        let out = Command::new(binary())
            .args(&args)
            .output()
            .expect("run pm7_rs_cli");
        let stderr = String::from_utf8_lossy(&out.stderr);
        if out.status.success() {
            failures.push(format!(
                "{template}\n      was accepted; it must be refused"
            ));
            continue;
        }
        if stderr.contains("panicked at") || out.status.code() == Some(101) {
            failures.push(format!(
                "{template}\n      panicked: {}",
                first_line(&stderr)
            ));
            continue;
        }
        if !stderr.to_lowercase().contains(&expected.to_lowercase()) {
            failures.push(format!(
                "{template}\n      expected {expected:?} in: {}",
                first_line(&stderr)
            ));
        }
    }
    report("rejection", REJECTIONS.len(), &failures);
}

/// An unknown flag is rejected on *every* mode, not only on the one somebody tested.
///
/// There is no `--help` on the option surface — `pm7_rs_cli` with fewer than three arguments
/// prints the usage banner and that is the whole of it — so a misspelling has nothing to fall
/// back on and rejection is the only thing standing between the user and a silently different
/// calculation.
#[test]
fn an_unknown_flag_is_rejected_on_every_mode() {
    for mode in MODES {
        let args = tokens(&format!("{mode} {{water}} --kpoint 2"), 0);
        let out = Command::new(binary()).args(&args).output().expect("run");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "`{mode}` accepted the misspelled --kpoint"
        );
        assert!(
            stderr.contains("unknown option `--kpoint`"),
            "`{mode}`: {stderr}"
        );
    }
}

/// `--pbc-mode mopac` is advertised and refused, and the refusal is the documented one.
#[test]
fn the_advertised_mopac_lattice_sum_refuses_with_its_explanation() {
    let args = tokens("energy {diamond} --pbc-mode mopac", 0);
    let out = Command::new(binary()).args(&args).output().expect("run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "mopac mode is not implemented");
    assert!(
        stderr.contains("MopacCluster is not implemented yet") && stderr.contains("PbcMode::Ewald"),
        "the refusal must name the alternative: {stderr}"
    );
    // On a molecule the same flag is refused earlier and for a different reason: there is no cell,
    // so there is no lattice sum for it to describe. Through v0.2.2 it was accepted and silently
    // discarded, which is the `no-cell-pbc-mode` entry this ledger opened with.
    let args = tokens("energy {water} --pbc-mode mopac", 0);
    let out = Command::new(binary()).args(&args).output().expect("run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a molecule has no zone to sample");
    assert!(
        stderr.contains("--pbc-mode") && stderr.contains("no cell"),
        "the refusal must say why the flag cannot apply: {stderr}"
    );
}

/// The matrix is the size it was designed to be.
///
/// Not vanity: the number is the difference between "every combination" and "the combinations
/// somebody happened to list", and it drifts silently when an axis gains a value. The upper bound
/// went from 8000 to 12000 in 0.2.3, which added three blocks — per-axis periodicity, the
/// optimizer knobs, and the projection crossed with the dipole origin.
#[test]
fn the_matrix_is_the_size_it_claims_to_be() {
    let cases = matrix();
    assert!(
        (3_000..=12_000).contains(&cases.len()),
        "the matrix is {} invocations; it is meant to be a few thousand",
        cases.len()
    );
    // Every mode really is in it.
    let modes: BTreeSet<&str> = cases.iter().map(|c| c.args[0].as_str()).collect();
    for mode in MODES {
        assert!(modes.contains(mode), "no case runs `{mode}`");
    }
}

// --- the ledger: flags that are read, and flags that are not ---------------------------------

/// `(label, base, added)` — run `base`, run `base added`, and see whether the answer moved.
const DIFFERENTIAL: &[(&str, &str, &str)] = &[
    // Flags that must change the answer on a mode that accepts them.
    ("kpoints", "energy {diamond}", "--kpoints 2 2 2"),
    (
        "kshift",
        "energy {diamond} --kpoints 2 2 2",
        "--kshift 0.5 0.5 0.5",
    ),
    (
        "smearing",
        "energy {diamond} --kpoints 2 2 2",
        "--smearing fermi 0.5",
    ),
    ("cell", "energy {h2}", "--cell 2.6,0,0"),
    ("method", "energy {water}", "--method pm7-ts"),
    ("charge", "energy {water}", "--charge 1 --multiplicity 2"),
    (
        "multiplicity",
        "energy {methyl} --multiplicity 2",
        "--multiplicity 4",
    ),
    ("field", "energy {water}", "--field 0.5,0,0"),
    ("json", "energy {water}", "--json"),
    ("no-diis", "energy {water}", "--no-diis"),
    (
        "scf-tolerance",
        "energy {diamond} --kpoints 2 2 2",
        "--scf-tolerance 1e-3",
    ),
    ("dandc", "energy {water}", "--dandc 15.0"),
    ("ir", "frequencies {water}", "--ir"),
    (
        "exchange-cutoff",
        "frequencies {water} --ir",
        "--exchange-cutoff 0 4",
    ),
    ("molden-basis", "molden {water}", "--molden-basis sto"),
    ("supercell", "phonons {diamond}", "--supercell 2 1 1"),
    (
        "no-acoustic-sum-rule",
        "phonons {diamond}",
        "--no-acoustic-sum-rule",
    ),
    // Naming the default, which must not move the answer.
    (
        "acoustic-sum-rule-default",
        "phonons {diamond}",
        "--acoustic-sum-rule",
    ),
    ("qpoints", "phonons {diamond}", "--qpoints 0.5,0,0"),
    ("lo-to", "phonons {diamond}", "--lo-to 1,0,0"),
    (
        "slab-thickness",
        "dielectric {sheet}",
        "--slab-thickness 3.33",
    ),
    (
        "wire-cross-section",
        "dielectric {chain}",
        "--wire-cross-section 9.0",
    ),
    // Honoured, but only just: forcing UHF on a closed shell reaches the same solution by a
    // longer road, and it is the iteration count and the last digits that say so.
    ("reference-uhf", "energy {water}", "--reference uhf"),
    // Flags that are inert here and are meant to be: a default named explicitly, and an
    // iteration budget the run was never going to need.
    ("pbc-mode-ewald", "energy {diamond}", "--pbc-mode ewald"),
    ("max-scf-headroom", "energy {water}", "--max-scf 400"),
    // Same shape one level down: water's orbital response converges well inside the default 100,
    // so raising the budget buys headroom the run never spends. The flag earns its keep on a cell
    // that *does* run out — `tests/phonons.rs` holds that case, because it is a k-mesh phonon run
    // and does not belong in a matrix that has to finish in minutes.
    (
        "cphf-max-iterations-headroom",
        "frequencies {water}",
        "--cphf-max-iterations 400",
    ),
    // The molecular SCF with DIIS on: `--scf-tolerance` reaches `Pm7Options::p_tol` and then
    // makes no difference in either direction, because `scf.rs` accepts convergence on
    // `de < e_tol && (dp < p_tol || residual_ok)` and the DIIS residual satisfies the second
    // clause first. 1e-1 and 1e-12 both stop water at eleven iterations. On the periodic path,
    // and on this same molecule with `--no-diis`, the flag does what it says.
    (
        "scf-tolerance-molecular",
        "energy {water}",
        "--scf-tolerance 1e-1",
    ),
    // Flags belonging to another mode entirely. Every one of these is read by the parser,
    // accepted, and then never looked at again.
    ("stray-supercell", "energy {water}", "--supercell 2 2 2"),
    ("stray-qpoints", "energy {water}", "--qpoints 0,0,0"),
    ("stray-asr", "energy {water}", "--no-acoustic-sum-rule"),
    ("stray-ir", "energy {water}", "--ir"),
    ("stray-molden-basis", "energy {water}", "--molden-basis sto"),
    ("stray-lo-to", "energy {water}", "--lo-to 1,0,0"),
    ("stray-slab", "energy {water}", "--slab-thickness 3.33"),
    ("stray-wire", "energy {water}", "--wire-cross-section 9.0"),
    ("stray-opt-output", "energy {water}", "--opt-output {out}"),
    ("stray-json", "gradient {water}", "--json"),
    ("stray-json-forces", "forces {water}", "--json"),
    ("stray-json-optimize", "optimize {water}", "--json"),
    ("stray-json-molden", "molden {water}", "--json"),
    // Periodic flags on a molecule: there is no cell, so the whole periodic block is discarded
    // without a word.
    ("no-cell-kpoints", "energy {water}", "--kpoints 2 2 2"),
    ("no-cell-kshift", "energy {water}", "--kshift 0.5 0.5 0.5"),
    ("no-cell-smearing", "energy {water}", "--smearing fermi 0.5"),
    ("no-cell-pbc-mode", "energy {water}", "--pbc-mode mopac"),
];

/// Flags that today produce byte-identical output with and without them.
///
/// The first three are correct: naming the default `--pbc-mode ewald`, and leaving headroom in an
/// iteration budget the run never approached, must not move the answer, and the molecular DIIS
/// path reaches its own residual criterion before `--scf-tolerance` can bind. Everything after
/// them is the defect this test exists to record: the flag is parsed, accepted, and never read.
/// `--opt-output` on a mode other than `optimize` or `molden` is the worst of them, because the
/// user is waiting for a file that will never be written and the exit status says the run
/// succeeded.
/// Only three remain, and all three are correct behaviour rather than defects: naming the default
/// `--pbc-mode ewald`, leaving headroom in an iteration budget the run never approached, and
/// `--scf-tolerance` on a molecular path that reaches its own residual criterion first.
///
/// The other seventeen this ledger opened with were real. They are gone as of 0.2.3: a flag the
/// mode will not read is now **refused** rather than accepted and discarded (`validate_flags` in
/// `src/bin/pm7_rs.rs`), and `--json` reaches `gradient`, `forces` and `optimize` instead of being
/// swallowed. A flag that stops being a no-op without this list being updated fails the test
/// below, which is what makes each fix checkable.
const KNOWN_NO_OPS: &[&str] = &[
    "pbc-mode-ewald",
    // Same shape as `pbc-mode-ewald`: naming a default. The sum rule is enforced from 0.2.3, so
    // `--acoustic-sum-rule` asks for what already happens; `--no-acoustic-sum-rule` is the one
    // that moves the answer, and it is exercised above.
    "acoustic-sum-rule-default",
    "max-scf-headroom",
    "cphf-max-iterations-headroom",
    "scf-tolerance-molecular",
];

/// The one thing a single invocation cannot see: a flag that was accepted and never read.
///
/// Every flag is run twice, with and without, and the ones whose output does not move are
/// compared against [`KNOWN_NO_OPS`]. A flag that starts being ignored fails this; a flag that
/// stops being ignored fails it too, which is the point — the ledger is meant to shrink.
#[test]
fn a_flag_either_changes_the_answer_or_is_a_pinned_no_op() {
    let run = |template: &str, index: usize| -> String {
        let args = tokens(template, index);
        let out = Command::new(binary()).args(&args).output().expect("run");
        assert!(
            !String::from_utf8_lossy(&out.stderr).contains("panicked at"),
            "`{template}` panicked"
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let mut inert = Vec::new();
    for (index, (label, base, added)) in DIFFERENTIAL.iter().enumerate() {
        let plain = run(base, 2_000_000 + index);
        let with = run(&format!("{base} {added}"), 2_000_000 + index);
        if plain == with {
            inert.push(*label);
        }
    }
    let found: BTreeSet<&str> = inert.into_iter().collect();
    let pinned: BTreeSet<&str> = KNOWN_NO_OPS.iter().copied().collect();

    let newly_ignored: Vec<&&str> = found.difference(&pinned).collect();
    assert!(
        newly_ignored.is_empty(),
        "these flags are now accepted and never read: {newly_ignored:?}"
    );
    let now_honoured: Vec<&&str> = pinned.difference(&found).collect();
    assert!(
        now_honoured.is_empty(),
        "these flags are honoured now; take them out of KNOWN_NO_OPS: {now_honoured:?}"
    );
}

/// `--opt-output` writes a file on the modes that own it, and is **refused** on the rest.
///
/// It used to be accepted everywhere and honoured on two modes, so `energy --opt-output out.xyz`
/// exited successfully having written nothing. Since 0.2.3 a mode that will not read a flag says
/// so; see `validate_flags` in `src/bin/pm7_rs.rs`.
#[test]
fn the_output_flag_writes_a_file_only_where_it_means_something() {
    for (mode, writes) in [
        ("optimize", true),
        ("molden", true),
        ("energy", false),
        ("frequencies", false),
    ] {
        let mut target = std::env::temp_dir();
        target.push(format!("pm7_matrix_written_by_{mode}.xyz"));
        let _ = std::fs::remove_file(&target);
        let args = tokens(
            &format!("{mode} {{water}} --opt-output {}", target.display()),
            0,
        );
        let out = Command::new(binary()).args(&args).output().expect("run");
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(
            out.status.success(),
            writes,
            "`{mode} --opt-output` should {} -- stderr: {stderr}",
            if writes { "succeed" } else { "be refused" }
        );
        if !writes {
            assert!(
                stderr.contains("does nothing for mode") && stderr.contains(mode),
                "the refusal must name the flag and the mode: {stderr}"
            );
        }
        assert_eq!(
            target.exists(),
            writes,
            "`{mode} --opt-output` wrote a file: {}, expected {writes}",
            target.exists()
        );
        let _ = std::fs::remove_file(&target);
    }
}

// --- tier two: the whole thing ---------------------------------------------------------------

/// Every mode against every combination of the flags it accepts.
///
/// `cargo test --release --test cli_matrix -- --ignored`. Several thousand processes; a few
/// minutes across four workers, and far too long for a pull request.
#[test]
#[ignore = "the full cross-product: thousands of subprocesses, minutes to run"]
fn the_full_matrix_holds_the_invariant_on_every_combination() {
    let cases = matrix();
    let failures = run_cases(&cases);
    report("matrix", cases.len(), &failures);
}
