# SPDX-License-Identifier: GPL-3.0-or-later
"""Every mode of ``python -m pm7_rs`` against every flag that mode accepts.

The companion of ``tests/cli_matrix.rs``, against the other command line. The two surfaces are
built from different parsers over the same library — argparse here, a hand-written loop there —
and as of 0.2.3 they accept the same flags;
``test_the_two_command_lines_offer_the_same_flags`` is what keeps that true. They still *behave*
differently in places, because argparse has no per-mode flag table and the Rust CLI does, so a
flag belonging to another mode is refused there and accepted-and-dropped here. Those places are
recorded in ``KNOWN_NO_OPS`` rather than assumed away.

"All combinations" is, per mode, the cross-product over the flags that mode accepts with two or
three representative values each: `BLOCKS` is that table, and it comes to a few thousand runs.

Two tiers:

* the default one, which every ``pytest`` runs: a representative subset covering every mode and
  every flag at least once, the documented rejections, and the no-op ledger. Well under a minute.
* ``pytest -m matrix``, the full cross-product. Several thousand interpreter starts; minutes.

The invariant one run can check is "it succeeded, or it failed and said which flag and what to
do". Whether a flag was *read* takes the same command twice, so that lives in
``test_a_flag_either_changes_the_answer_or_is_a_pinned_no_op`` with an explicit ledger.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

# --- the systems ------------------------------------------------------------------------------

_DIR = pathlib.Path(tempfile.mkdtemp(prefix="pm7_matrix_"))

_FILES = {
    "water": "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
    # A doublet, for --multiplicity and --reference.
    "methyl": (
        "4\nmethyl radical\n"
        "C  0.0000  0.0000 0.0\n"
        "H  1.0790  0.0000 0.0\n"
        "H -0.5395  0.9345 0.0\n"
        "H -0.5395 -0.9345 0.0\n"
    ),
    # Plain XYZ, so --cell is the only possible source of a lattice.
    "h2": "2\nH2\nH 0.0 0.0 0.0\nH 0.75 0.0 0.0\n",
    "diamond": (
        '2\nLattice="0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "C 0.000000 0.000000 0.000000\n"
        "C 0.891750 0.891750 0.891750\n"
    ),
    # 1-D: nine lattice numbers, but pbc="T F F" makes only the first periodic.
    "chain": (
        '2\nLattice="2.6 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" '
        'Properties=species:S:1:pos:R:3 pbc="T F F"\n'
        "H 0.000000 0.000000 0.000000\n"
        "H 0.750000 0.000000 0.000000\n"
    ),
    # 2-D.
    "sheet": (
        '2\nLattice="2.510 0.0 0.0 -1.255 2.1738 0.0 0.0 0.0 20.0" '
        'Properties=species:S:1:pos:R:3 pbc="T T F"\n'
        "B 0.000000 0.000000 0.000000\n"
        "N 1.255000 0.724600 0.000000\n"
    ),
}

for _name, _text in _FILES.items():
    (_DIR / f"{_name}.xyz").write_text(_text, encoding="ascii")


def _expand(token: str, index: int) -> str:
    """Substitute one placeholder. Applied *after* splitting, so a path may contain spaces."""
    if token == "{missing}":
        return str(_DIR / "there_is_no_such_file.xyz")
    # Unique per case: the matrix runs several workers at once, and two cases sharing an output
    # path would race rather than test anything.
    if token == "{out}":
        return str(_DIR / f"out_{index}.xyz")
    if token.startswith("{") and token.endswith("}"):
        return str(_DIR / f"{token[1:-1]}.xyz")
    return token


def _tokens(text: str, index: int = 0) -> list[str]:
    return [_expand(t, index) for t in text.split()]


# A structure and whatever flags make it a system worth asking about. Charge and multiplicity
# travel with the structure rather than forming their own axis: a doublet multiplicity on water is
# not a combination, it is an arithmetic error, and the matrix would spend a sixth of itself on
# the same parity message.
SYSTEMS = {
    "water": "{water}",
    "water-cation": "{water} --charge 1 --multiplicity 2",
    "methyl": "{methyl} --multiplicity 2",
    "diamond": "{diamond}",
    "chain": "{chain}",
    "sheet": "{sheet}",
    # The cell from --cell rather than from the file, in the nine-number and the three-number
    # forms. The short forms were Rust-CLI-only until 0.2.3.
    "boxed-water": "{water} --cell 8.0,0,0,0,8.0,0,0,0,8.0",
    "cell-chain": "{h2} --cell 2.6,0,0",
}

# --- the axes ---------------------------------------------------------------------------------
#
# "" means the flag is absent, which is always one of the representative values: the default is a
# value like any other and half the regressions live there.

JSON = ("", "--json")
METHOD = ("", "--method pm7-ts")
REFERENCE = ("", "--reference rhf", "--reference uhf")
DIPOLE_ORIGIN = ("", "--dipole-origin com", "--dipole-origin charge")
FIELD = ("", "--field 0.2,0,0", "--field 0,0,-0.5")
IR = ("", "--ir")
PROJECTION = ("", "--projection none", "--projection translations")
# The four solver knobs that reached this CLI in 0.2.3. The values are chosen to be accepted and
# to change nothing measurable: what is under test is that the flag is read at all.
SOLVER = ("", "--no-diis", "--scf-tolerance 1e-7", "--max-scf 400", "--exchange-cutoff 6 12")
MOLDEN_BASIS = ("", "--molden-basis sto", "--molden-basis sto-3g")
OUTPUT = ("", "--output {out}")
# Variable-cell relaxation and the optimizer knobs, none of which reached either command
# line before 0.2.3 -- `OptOptions` was `::default()` at every call site.
OPT = ("", "--opt-cell", "--gtol 1e-2", "--opt-max-iter 3", "--stress-tol 1e-4")
# `2 1 1` is legal for every periodicity; `2 2 2` asks a chain and a sheet to repeat a direction
# they do not have, which is a rejection the matrix should see rather than avoid.
KMESH = ("", "--kpoints 2 1 1", "--kpoints 2 2 2")
# Only 0 and 0.5 keep the mesh closed under k -> -k; 0.25 is in the rejection table instead.
KSHIFT = ("", "--kshift 0.5 0.0 0.0")
SMEARING_A = ("", "--smearing fermi 0.2", "--smearing mp 0.15")
SMEARING_B = ("", "--smearing gauss 0.1", "--smearing none 0.0")
PBC_MODE = ("", "--pbc-mode ewald", "--pbc-mode mopac")
# Every per-axis pattern, including the ones that need the lattice vectors reordered. `TFT` and
# `FTF` were refused outright until 0.2.3.
PBC = ("", "--pbc TTT", "--pbc TTF", "--pbc TFT", "--pbc FTF")
SUPERCELL = ("", "--supercell 2 1 1", "--supercell 2 2 2")
# The sum rule is enforced by default since 0.2.3, so the axis is its negation:
# `--acoustic-sum-rule` now names the default and is inert.
ASR = ("", "--no-acoustic-sum-rule")
LO_TO = ("", "--lo-to 1,0,0")
QPOINTS_PAIR = ("", "--qpoints 0,0,0 0.5,0,0")
QPOINTS = ("", "--qpoints 0,0,0", "--qpoints 0,0,0 0.5,0,0")
KPATH = ("", "--qpoints 0,0,0", "--qpoints 0,0,0 0.5,0,0 0.5,0.5,0.0")
EXTENT = (
    "",
    "--slab-thickness 3.33",
    "--wire-cross-section 9.0",
    "--slab-thickness 3.33 --wire-cross-section 9.0",
)
# Flags that belong to some *other* mode. Passing one is how a user finds out whether the CLI
# reads its arguments or merely parses them.
STRAY = (
    "--supercell 2 1 1",
    "--qpoints 0,0,0",
    "--acoustic-sum-rule",
    "--ir",
    "--molden-basis sto",
    "--output {out}",
    "--lo-to 1,0,0",
    "--slab-thickness 3.33",
    "--dandc 15.0",
)

MODES = (
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
)

MOLECULAR_MODES = (
    "energy",
    "charges",
    "gradient",
    "forces",
    "orbitals",
    "hessian",
    "frequencies",
    "optimize",
    "molden",
)

# Every flag this parser advertises. The coverage assertion is against this list, so a flag added
# to the CLI and not to an axis fails the default tier rather than going untested.
FLAGS = (
    "--acoustic-sum-rule",
    "--no-acoustic-sum-rule",
    "--cell",
    "--charge",
    "--dandc",
    "--dipole-origin",
    "--field",
    "--ir",
    "--json",
    "--kpoints",
    "--kshift",
    "--lo-to",
    "--method",
    "--molden-basis",
    "--multiplicity",
    "--output",
    "--opt-cell",
    "--gtol",
    "--stress-tol",
    "--opt-max-iter",
    "--pbc",
    "--pbc-mode",
    "--projection",
    "--no-diis",
    "--scf-tolerance",
    "--max-scf",
    "--exchange-cutoff",
    "--qpoints",
    "--reference",
    "--slab-thickness",
    "--smearing",
    "--supercell",
    "--wire-cross-section",
)

# One rectangle of the matrix: some modes, some systems, and the axes those modes accept.
BLOCKS = (
    # The SCF itself, crossed with everything that changes how it is solved.
    ("molecular-core", MOLECULAR_MODES, ("water", "water-cation", "methyl"),
     (METHOD, REFERENCE, JSON, DIPOLE_ORIGIN)),
    # The flags that only matter once a derivative or a file is involved.
    ("molecular-extras", ("energy", "frequencies", "hessian", "molden", "optimize"),
     ("water", "methyl"), (FIELD, IR, MOLDEN_BASIS, OUTPUT)),
    # The vibrational projection, and the four SCF knobs on the modes that read them.
    ("projection-and-solver", ("energy", "frequencies"), ("water",),
     (PROJECTION, SOLVER)),
    # The optimizer knobs, on a molecule and on a cell.
    ("optimizer", ("optimize",), ("water", "boxed-water"), (OPT, JSON)),
    # Everything a cell adds, on all three periodicities plus a box supplied by --cell.
    ("periodic-core", ("energy", "charges", "gradient", "forces", "stress", "hessian"),
     ("diamond", "chain", "sheet"), (KMESH, SMEARING_A, PBC_MODE, JSON)),
    # --pbc and --kshift, which only the periodic modes reach.
    ("periodicity-overrides", ("energy", "orbitals", "optimize"),
     ("diamond", "boxed-water"), (PBC, KSHIFT, KMESH, JSON)),
    ("phonons", ("phonons",), ("diamond", "chain", "sheet"),
     (SUPERCELL, QPOINTS_PAIR, ASR, LO_TO, JSON)),
    ("dfpt", ("dfpt",), ("diamond", "chain", "sheet"), (QPOINTS, LO_TO, KMESH, JSON)),
    ("born", ("born",), ("diamond", "chain", "sheet"), (KMESH, SMEARING_B, LO_TO, JSON)),
    ("bands", ("bands",), ("diamond", "chain", "sheet"), (KPATH, KMESH, SMEARING_B, JSON)),
    ("dielectric", ("dielectric",), ("chain", "sheet", "diamond", "boxed-water"),
     (EXTENT, KMESH, JSON)),
    # The part that is deliberately nonsense: a flag from another mode, on every mode.
    ("stray-flags", MODES, ("water", "diamond"), (STRAY, JSON)),
)


def matrix():
    """The whole matrix, in a fixed order so a case index means the same thing on every run."""
    cases = []
    for name, modes, systems, axes in BLOCKS:
        for mode in modes:
            for system in systems:
                head = SYSTEMS[system]
                for combination in itertools.product(*axes):
                    index = len(cases)
                    args = [mode, *_tokens(head, index)]
                    for value in combination:
                        args += _tokens(value, index)
                    cases.append((f"[{name}] " + " ".join(args), args))
    return cases


# --- running ----------------------------------------------------------------------------------

# `text=True` alone decodes with `locale.getencoding()`, which is the console code page on
# Windows -- cp932 on a Japanese install. The CLI writes UTF-8, so the first message containing an
# en dash or a Greek letter kills the *reader thread* with UnicodeDecodeError rather than failing
# the assertion it was meant to check. Naming the encoding makes these tests say the same thing on
# every locale; forcing it in the child's environment makes the child agree.
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(args):
    """Run the CLI as a subprocess and return `(returncode, stdout, stderr)`."""
    done = subprocess.run(
        [sys.executable, "-m", "pm7_rs", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_ENV,
        timeout=1800,
    )
    return done.returncode, done.stdout, done.stderr


def _workers():
    return min(8, max(2, (os.cpu_count() or 2)))


def run_cases(cases):
    """Check a batch across a few workers and return every violation, sorted.

    Threads rather than one at a time: each invocation is a whole interpreter, nearly a second of
    which is spent importing numpy and the extension module before any chemistry happens.
    """
    def one(case):
        label, args = case
        why = check(args)
        return None if why is None else f"{label}\n      {why}"

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        failures = [f for f in pool.map(one, cases) if f]
    return sorted(failures)


# --- the invariant ----------------------------------------------------------------------------

# Failure messages that name neither the flag that caused them nor a flag to reach for instead.
#
# A defect ledger, not a permission slip: each entry is a message the CLI could say better, pinned
# so the matrix stays green while a *new* mute failure turns it red. None of them is a crash and
# all of them are true; they simply leave the reader to guess which of the flags they typed the
# complaint is about.
MUTE_FAILURES = (
    # `--kpoints 2 2 2` on a chain or a sheet. Names neither --kpoints nor --cell.
    "cannot repeat non-periodic direction",
    # `bands` with no --kpoints. "run with a k mesh" is the remedy, spelled without the flag.
    "a band structure needs the translation-resolved density",
    # `--kshift` off the allowed set: a paragraph on time reversal that never says --kshift.
    "is not supported: only 0 and 0.5",
)

# Exception types that mean the CLI fell over rather than refused. A `ValueError` is how the
# library says no and how this CLI passes that on; anything else is an internal reaching daylight.
CRASH_TYPES = (
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "ZeroDivisionError",
    "AssertionError",
    "RecursionError",
    "UnboundLocalError",
    "NameError",
    "OverflowError",
    "StopIteration",
    "NotImplementedError",
    "ArithmeticError",
)

# Words a message may use in place of the flag itself. Prose is allowed to be prose.
ALIASES = (
    ("--dandc", "divide-and-conquer"),
    ("--cell", "lattice"),
    ("--kpoints", "Monkhorst"),
    ("--ir", "infrared"),
    ("--slab-thickness", "thickness"),
    ("--slab-thickness", "extent"),
    ("--wire-cross-section", "cross-section"),
    ("--wire-cross-section", "extent"),
    ("--multiplicity", "electron count"),
    ("--charge", "electron count"),
    ("--reference", "closed shell"),
    ("--pbc", "periodic direction"),
)


def _squash(text):
    """Lowercase and drop every separator, so `--pbc-mode` matches `PbcMode`."""
    return "".join(c for c in text.lower() if c.isalnum())


def _explains(args, stderr):
    """Does this message let the reader find the flag they got wrong, or the one to reach for?"""
    # A message that names any flag has said what to do, whether or not it is the one passed.
    if "--" in stderr:
        return True
    flat = _squash(stderr)
    # The mode is the other half of the invocation, and naming it is naming the problem.
    if _squash(args[0]) in flat:
        return True
    for arg in args:
        if not arg.startswith("--"):
            continue
        key = _squash(arg[2:])
        # Two-letter stems like `ir` match half the dictionary; those go through ALIASES.
        if len(key) >= 4 and key in flat:
            return True
        if any(flag == arg and _squash(word) in flat for flag, word in ALIASES):
            return True
    return False


def check(args):
    """The invariant, for one invocation: it succeeded, or it failed and said why.

    Returns `None` when the run is fine, or a one-line reason when it is not.
    """
    code, stdout, stderr = run(args)
    for kind in CRASH_TYPES:
        if f"\n{kind}:" in stderr or stderr.startswith(f"{kind}:"):
            return f"crashed with {kind}: {_last_line(stderr)}"
    if code == 0:
        return None
    if code not in (1, 2):
        return f"unexpected exit code {code}: {_last_line(stderr)}"
    if not stderr.strip():
        return "failed with an empty stderr"
    message = _last_line(stderr)
    if len(message) < 12:
        return f"failed with no explanation: {stderr!r}"
    if not _explains(args, stderr) and not any(m in stderr for m in MUTE_FAILURES):
        return f"the refusal names neither the flag nor a remedy: {message}"
    return None


def _last_line(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _report(what, ran, failures):
    # A broken invariant tends to break for a whole axis at once, and an assertion message with
    # three thousand entries in it is not a report. The first forty say what happened.
    shown = "\n  ".join(failures[:40])
    rest = max(0, len(failures) - 40)
    more = "" if not rest else f"\n  ... and {rest} more"
    assert not failures, (
        f"{len(failures)} of {ran} {what} invocations broke the invariant:\n  {shown}{more}"
    )


# --- tier one: what every pytest runs -----------------------------------------------------------


def representative(cases):
    """The greedy cover: the first case bringing in a mode or a flag nothing before it used.

    Greedy over the matrix in its natural order rather than a hand-written list, so a new axis is
    represented the moment it is added to `BLOCKS` and nobody has to remember.
    """
    seen: set[str] = set()
    chosen = []
    for label, args in cases:
        novel = {a for a in args if (a.startswith("--") or a in MODES) and a not in seen}
        if novel:
            seen |= novel
            chosen.append((label, args))
    return chosen


def test_the_representative_subset_covers_every_mode_and_every_flag():
    cases = matrix()
    chosen = representative(cases)
    used = {a for _, args in chosen for a in args}

    missing_modes = [m for m in MODES if m not in used]
    assert not missing_modes, f"the subset never runs {missing_modes}"
    missing_flags = [f for f in FLAGS if f not in used]
    assert not missing_flags, f"the subset never passes {missing_flags}; add an axis to BLOCKS"
    # Cheap enough to keep in the default tier: this is the number that has to stay small.
    assert len(chosen) < 60, f"the cover grew to {len(chosen)} cases; the default tier is seconds"

    _report("representative", len(chosen), run_cases(chosen))


def test_the_matrix_is_the_size_it_claims_to_be():
    """Not vanity: the number is the difference between "every combination" and "the combinations
    somebody happened to list", and it drifts silently when an axis gains a value."""
    cases = matrix()
    assert 2_000 <= len(cases) <= 8_000, (
        f"the matrix is {len(cases)} invocations; it is meant to be a few thousand"
    )
    modes = {args[0] for _, args in cases}
    for mode in MODES:
        assert mode in modes, f"no case runs `{mode}`"


# Invocations that must be refused, and the words the refusal has to contain.
#
# A rejection is worth more than an acceptance here: the whole point of the invariant is that a
# wrong flag stops the run and says so, and these are the wrong flags.
REJECTIONS = (
    ("energy {water} --kpoint 2", "unrecognized arguments"),
    ("energy {water} --nonsense", "unrecognized arguments"),
    ("nonsense {water}", "invalid choice"),
    # `--no-diis`, `--scf-tolerance`, `--max-scf` and `--exchange-cutoff` used to be here, as the
    # four flags the Rust CLI had and this one did not. They are accepted in 0.2.3 -- since the
    # wheel ships no Rust binary, "rejected rather than accepted-and-ignored" was the best of two
    # bad outcomes for a pip user, and the four are exactly what one reaches for when an SCF will
    # not converge. `test_the_two_command_lines_offer_the_same_flags` keeps the sets equal now.
    # `--opt-output` used to be here as a flag this CLI did not have. Since 0.2.3 it is an accepted
    # alias of `--output`, so the pair below is what remains: a genuinely unknown flag is refused,
    # and the alias is not one.
    ("energy {water} --opt-outpt {out}", "unrecognized arguments"),
    ("energy {water} --dipole-origin nonsense", "invalid choice"),
    ("energy {water} --pbc-mode nonsense", "invalid choice"),
    ("energy {water} --kpoints", "expected 3 arguments"),
    ("energy {water} --kpoints 2", "expected 3 arguments"),
    ("energy {water} --smearing fermi", "expected 2 arguments"),
    ("energy {water} --method nonsense", "unknown PM7 method"),
    ("energy {water} --reference nonsense", "unknown SCF reference"),
    ("energy {diamond} --smearing bogus 0.2", "unknown smearing"),
    ("energy {water} --field junk", "wants `fx,fy,fz`"),
    # `--cell 3,0,0` is a 1-D chain, which the Rust CLI has always taken and this one refused.
    # Four numbers is not any dimensionality, and is still refused.
    ("energy {water} --cell 3,0,0,1", "3, 6 or 9 numbers"),
    ("energy {water} --pbc TT", "three flags"),
    ("energy {water} --multiplicity 0", "multiplicity must be >= 1"),
    # Eight electrons: a doublet is arithmetically impossible.
    ("energy {water} --multiplicity 2", "parity"),
    ("energy {water} --charge 1", "parity"),
    ("energy {methyl} --multiplicity 2 --reference rhf", "needs a closed shell"),
    # This CLI refuses a k mesh with no cell where the Rust one drops it without a word.
    ("energy {water} --kpoints 2 2 2", "--kpoints needs a cell"),
    ("energy {diamond} --kpoints 2 2 2 --kshift 0.25 0.25 0.25", "only 0 and 0.5"),
    ("stress {water}", "needs a periodic cell"),
    ("phonons {water}", "needs a periodic cell"),
    ("bands {water}", "needs a periodic cell"),
    ("dielectric {water}", "needs a periodic cell"),
    ("bands {diamond}", "needs a k path"),
    ("born {diamond} --lo-to 1,0", "wants `x,y,z`"),
    ("dielectric {sheet}", "--slab-thickness"),
    ("dielectric {sheet} --wire-cross-section 9", "periodic direction"),
    ("dielectric {diamond} --slab-thickness 3.3", "already has a volume"),
    ("frequencies {diamond} --ir", "born"),
    ("phonons {sheet} --lo-to 1,0,0", "needs a 3-D cell"),
    # This CLI says which mode --dandc does not apply to; the Rust one ignores it there.
    ("phonons {diamond} --dandc 15", "does not apply"),
    # Advertised in --help and refused on purpose: the truncated MOPAC lattice sum was never
    # finished. The refusal is the contract, so it is pinned rather than left to chance.
    ("energy {diamond} --pbc-mode mopac", "MopacCluster is not implemented yet"),
)


def test_every_documented_rejection_says_what_is_wrong():
    def one(entry):
        template, expected = entry
        code, _out, err = run(_tokens(template))
        if code == 0:
            return f"{template}\n      was accepted; it must be refused"
        if any(f"\n{kind}:" in err for kind in CRASH_TYPES):
            return f"{template}\n      crashed: {_last_line(err)}"
        if expected.lower() not in err.lower():
            return f"{template}\n      expected {expected!r} in: {_last_line(err)}"
        return None

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        failures = sorted(f for f in pool.map(one, REJECTIONS) if f)
    _report("rejection", len(REJECTIONS), failures)


def test_an_unknown_flag_is_rejected_on_every_mode():
    """A misspelling is refused on *every* mode, not only on the one somebody tested.

    `allow_abbrev=False` is what makes this true: argparse would otherwise read `--kpoint` as a
    short form of `--kpoints`, so a typo becomes a different flag and the calculation quietly
    changes.
    """
    def one(mode):
        code, _out, err = run(_tokens(f"{mode} {{water}} --kpoint 2"))
        if code == 0:
            return f"`{mode}` accepted the misspelled --kpoint"
        if "unrecognized arguments: --kpoint" not in err:
            return f"`{mode}`: {_last_line(err)}"
        return None

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        failures = sorted(f for f in pool.map(one, MODES) if f)
    _report("unknown-flag", len(MODES), failures)


def test_the_advertised_mopac_lattice_sum_refuses_with_its_explanation():
    """`--pbc-mode mopac` is offered by the parser and refused by the library, on purpose."""
    code, _out, err = run(_tokens("energy {diamond} --pbc-mode mopac"))
    assert code != 0, "mopac mode is not implemented"
    assert "MopacCluster is not implemented yet" in err, err
    assert "PbcMode::Ewald" in err, err
    # And it is a refusal, not a crash: the same flag on a molecule is simply inert, because
    # there is no lattice sum to do.
    code, _out, err = run(_tokens("energy {water} --pbc-mode mopac"))
    assert code == 0, err


# --- the ledger: flags that are read, and flags that are not ------------------------------------

# `(label, base, added)` -- run `base`, run `base added`, and see whether the answer moved.
DIFFERENTIAL = (
    # Flags that must change the answer on a mode that accepts them.
    ("kpoints", "energy {diamond}", "--kpoints 2 1 1"),
    ("kshift", "energy {diamond} --kpoints 2 2 2", "--kshift 0.5 0.5 0.5"),
    ("smearing", "energy {diamond} --kpoints 2 2 2", "--smearing fermi 0.5"),
    ("pbc", "energy {diamond}", "--pbc TTF"),
    ("cell", "energy {h2}", "--cell 2.6,0,0,0,20,0,0,0,20"),
    ("method", "energy {water}", "--method pm7-ts"),
    ("charge", "energy {water}", "--charge 1 --multiplicity 2"),
    ("multiplicity", "energy {methyl} --multiplicity 2", "--multiplicity 4"),
    ("reference", "energy {water}", "--reference uhf"),
    ("field", "energy {water}", "--field 0.5,0,0"),
    ("json", "energy {water}", "--json"),
    ("dandc", "energy {water}", "--dandc 15.0"),
    ("ir", "frequencies {water}", "--ir"),
    ("molden-basis", "molden {water}", "--molden-basis sto"),
    ("supercell", "phonons {diamond}", "--supercell 2 1 1"),
    # A 1x1x1 supercell already satisfies the sum rule to the digits printed, so turning the
    # projection off only shows on a real one.
    (
        "no-acoustic-sum-rule",
        "phonons {diamond} --supercell 2 1 1",
        "--no-acoustic-sum-rule",
    ),
    # Naming the default, which must not move the answer.
    (
        "acoustic-sum-rule-default",
        "phonons {diamond} --supercell 2 1 1",
        "--acoustic-sum-rule",
    ),
    ("qpoints", "phonons {diamond}", "--qpoints 0.5,0,0"),
    ("lo-to", "phonons {diamond}", "--lo-to 1,0,0"),
    ("slab-thickness", "dielectric {sheet}", "--slab-thickness 3.33"),
    ("wire-cross-section", "dielectric {chain}", "--wire-cross-section 9.0"),
    ("output-molden", "molden {water}", "--output {out}"),
    # Inert on purpose: a default named explicitly.
    ("pbc-mode-ewald", "energy {diamond}", "--pbc-mode ewald"),
    # Flags belonging to another mode entirely.
    ("stray-supercell", "energy {water}", "--supercell 2 1 1"),
    ("stray-qpoints", "energy {water}", "--qpoints 0,0,0"),
    ("stray-asr", "energy {water}", "--no-acoustic-sum-rule"),
    ("stray-ir", "energy {water}", "--ir"),
    # Same shape as --ir above: argparse has no per-mode flag table, so a flag only `frequencies`
    # reads is accepted and dropped on every other mode. The Rust CLI refuses it by name through
    # MODE_FLAGS. Ledgered rather than fixed, because fixing it means giving this CLI a per-mode
    # table of its own, which is a larger change than the divergence justifies.
    ("stray-projection", "energy {water}", "--projection none"),
    ("stray-molden-basis", "energy {water}", "--molden-basis sto"),
    ("stray-lo-to", "energy {water}", "--lo-to 1,0,0"),
    ("stray-slab", "energy {water}", "--slab-thickness 3.33"),
    ("stray-wire", "energy {water}", "--wire-cross-section 9.0"),
    ("stray-output", "energy {water}", "--output {out}"),
    ("stray-json-molden", "molden {water}", "--json"),
    # The one flag `optimize` ought to own: this CLI writes the file only for `molden`.
    ("output-optimize", "optimize {water}", "--output {out}"),
    # The origin of a point-charge dipole is only a choice for a charged species, and only `--json`
    # reports a dipole at all here -- the human form of `energy` prints none, where the Rust CLI's
    # does.
    (
        "dipole-origin",
        "energy {water} --charge 1 --multiplicity 2 --json",
        "--dipole-origin charge",
    ),
    # ... and on the modes that drop it before it reaches the library.
    ("dipole-origin-born", "born {diamond}", "--dipole-origin charge"),
    # Periodic flags on a molecule. `--kpoints` is refused here (see REJECTIONS); the rest are not.
    ("no-cell-kshift", "energy {water}", "--kshift 0.5 0.5 0.5"),
    ("no-cell-smearing", "energy {water}", "--smearing fermi 0.5"),
    ("no-cell-pbc-mode", "energy {water}", "--pbc-mode mopac"),
    ("no-cell-pbc", "energy {water}", "--pbc TTF"),
)

# Flags that today produce byte-identical output with and without them.
#
# `pbc-mode-ewald` is correct: naming the default must not move the answer. Everything after it is
# the defect this test exists to record -- the flag is parsed, accepted, and never read.
# `output-optimize` used to be the sharpest of them -- the Rust CLI's `--opt-output` wrote the
# optimized geometry and this one silently did not, so the same intent expressed to the two command
# lines gave a file on one and nothing on the other. Fixed in 0.2.3 and gone from this list.
KNOWN_NO_OPS = {
    "pbc-mode-ewald",
    "acoustic-sum-rule-default",
    "stray-supercell",
    "stray-qpoints",
    "stray-asr",
    "stray-ir",
    "stray-projection",
    "stray-molden-basis",
    "stray-lo-to",
    "stray-slab",
    "stray-wire",
    "stray-output",
    "stray-json-molden",
    "dipole-origin-born",
    "no-cell-kshift",
    "no-cell-smearing",
    "no-cell-pbc-mode",
    "no-cell-pbc",
}


def test_a_flag_either_changes_the_answer_or_is_a_pinned_no_op():
    """The one thing a single invocation cannot see: a flag accepted and never read.

    Every flag runs twice, with and without, and the ones whose output does not move are compared
    against `KNOWN_NO_OPS`. A flag that starts being ignored fails this; a flag that stops being
    ignored fails it too, which is the point -- the ledger is meant to shrink.
    """
    def one(entry):
        index, (label, base, added) = entry
        _c, plain, _e = run(_tokens(base, 900_000 + index))
        _c, with_flag, _e = run(_tokens(f"{base} {added}", 900_001 + index))
        return label if plain == with_flag else None

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        inert = {label for label in pool.map(one, enumerate(DIFFERENTIAL)) if label}

    newly_ignored = sorted(inert - KNOWN_NO_OPS)
    assert not newly_ignored, f"these flags are now accepted and never read: {newly_ignored}"
    now_honoured = sorted(KNOWN_NO_OPS - inert)
    assert not now_honoured, (
        f"these flags are honoured now; take them out of KNOWN_NO_OPS: {now_honoured}"
    )


def test_the_output_flag_writes_a_file_where_it_means_something():
    """`--output` writes on the two modes that produce a file, and nowhere else.

    This asserted the opposite until 0.2.3, and said so: `optimize --output` was accepted, exited
    0, and wrote nothing, while the Rust CLI's `--opt-output` on the same mode wrote the relaxed
    geometry. That is the worst shape a flag can have -- the user waits for a file that was never
    going to arrive and the exit status says it worked -- so it is fixed rather than pinned.

    `energy --output` still writes nothing. Unlike the Rust CLI, argparse has no per-mode flag
    table to refuse it with; that divergence is in `KNOWN_NO_OPS` as `stray-output`.
    """
    for mode, writes in (("molden", True), ("optimize", True), ("energy", False)):
        target = _DIR / f"written_by_{mode}.out"
        target.unlink(missing_ok=True)
        code, _out, err = run(_tokens(f"{mode} {{water}} --output {target}"))
        assert code == 0, err
        assert target.exists() is writes, (
            f"`{mode} --output` wrote a file: {target.exists()}, expected {writes}"
        )
        target.unlink(missing_ok=True)


def test_a_library_refusal_arrives_as_a_message_rather_than_a_traceback():
    """A refusal raised below the argparse layer reads like a refusal.

    Until 0.2.3 this test asserted the opposite, and said so: `main` let the `ValueError`
    propagate, so `python -m pm7_rs energy crystal.xyz --method nonsense` answered a typo with
    twenty lines of interpreter frames and the message on the last one, while the Rust CLI printed
    `pm7-rs: <message>` and exited 1 for the same input. The message was right and the presentation
    was not. It was pinned rather than left alone so that fixing it would be a visible change.
    """
    code, _out, err = run(_tokens("energy {water} --method nonsense"))
    assert code == 1
    assert "Traceback (most recent call last)" not in err, err
    assert err.strip().startswith("pm7-rs: unknown PM7 method"), err

    # The missing-file case was the worse half: an OSError, where the traceback was the only thing
    # that named the file.
    code, _out, err = run(_tokens("energy {missing}"))
    assert code == 1
    assert "Traceback (most recent call last)" not in err, err
    assert err.strip().startswith("pm7-rs: "), err
    assert "missing" in err or "No such file" in err, err


# --- tier two: the whole thing ------------------------------------------------------------------


@pytest.mark.matrix
def test_the_full_matrix_holds_the_invariant_on_every_combination():
    """Every mode against every combination of the flags it accepts.

    `pytest -m matrix`. Several thousand interpreter starts; minutes, and far too long for a pull
    request.
    """
    cases = matrix()
    _report("matrix", len(cases), run_cases(cases))
