# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared plumbing for the MOPAC oracle scripts.

Every oracle does the same four things: write a geometry, run MOPAC on it, run `pm7-rs` on it, and
tabulate the difference. This module owns those four so the individual scripts stay short enough to
read in one screen, and so a change to (say) how the MOPAC executable is located happens once.

Locating MOPAC, in order:

1. ``$MOPAC_EXE`` — an explicit path, which is the only thing that works if you have MOPAC
   installed somewhere unusual;
2. the vendored Windows tree under ``tools/oracle/mopac-23.2.5-win`` (excluded from the published
   crate, present in the development checkout);
3. ``mopac`` on ``$PATH`` — how a Linux or macOS install normally looks.

If none of those finds it the scripts say so and exit, rather than reporting every molecule as a
failure and leaving you to work out why.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import json
from dataclasses import dataclass, field as field_
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]

_MOPAC_HOF = re.compile(r"FINAL HEAT OF FORMATION\s*=\s*(-?[0-9.]+)\s*KCAL", re.IGNORECASE)
_RS_HOF = re.compile(r"heat of formation[^:]*:\s*(-?[0-9.]+)\s*kcal", re.IGNORECASE)


def mopac_executable() -> Path:
    """The MOPAC 23.2.5 binary, or exit with an explanation."""
    override = os.environ.get("MOPAC_EXE")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        sys.exit(f"MOPAC_EXE is set to {override!r}, which is not a file")

    vendored = (
        ROOT / "tools/oracle/mopac-23.2.5-win/mopac-23.2.5-win/bin/mopac.exe",
        ROOT / "tools/oracle/mopac-23.2.5-win/bin/mopac.exe",
    )
    for candidate in vendored:
        if candidate.is_file():
            return candidate

    found = shutil.which("mopac")
    if found:
        return Path(found)

    sys.exit(
        "MOPAC not found. Set MOPAC_EXE to the executable, put `mopac` on PATH, or unpack the\n"
        "vendored build under tools/oracle/mopac-23.2.5-win/."
    )


def pm7_rs_command() -> list[str]:
    """The `pm7-rs` CLI invocation: the release binary if it is built, else `cargo run`.

    The release binary is roughly a hundred times faster to start, which matters for the sweeps
    that run several hundred diatomics.
    """
    suffix = ".exe" if os.name == "nt" else ""
    built = ROOT / "target" / "release" / f"pm7_rs_cli{suffix}"
    if built.is_file():
        return [str(built)]
    return ["cargo", "run", "--quiet", "--release", "--bin", "pm7_rs_cli", "--"]


def write_xyz(path: Path, name: str, atoms: Sequence[str]) -> Path:
    """An XYZ file for `pm7-rs`. `atoms` are ``"El x y z"`` strings in Ångström."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"{len(atoms)}\n{name}\n" + "\n".join(atoms) + "\n"
    path.write_text(body, encoding="ascii", newline="\n")
    return path


def write_mop(path: Path, name: str, atoms: Sequence[str], keywords: str) -> Path:
    """A MOPAC input file: three header lines, then ``El x 1 y 1 z 1`` per atom.

    The trailing ``1`` on each coordinate marks it optimizable; with ``1SCF`` nothing moves, but
    MOPAC still wants the flags.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [keywords, name, ""]
    for atom in atoms:
        element, x, y, z = atom.split()
        lines.append(f"{element}  {x} 1  {y} 1  {z} 1")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return path


def run_mopac(mopac: Path, mop_path: Path) -> float | None:
    """Run MOPAC and return the final heat of formation in kcal/mol, or `None` if it failed."""
    try:
        subprocess.run(
            [str(mopac), str(mop_path)],
            cwd=mop_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out_path = mop_path.with_suffix(".out")
    if not out_path.is_file():
        return None
    match = _MOPAC_HOF.search(out_path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def run_pm7_rs(xyz_path: Path, *extra: str) -> float | None:
    """Run the `pm7-rs` CLI and return the heat of formation in kcal/mol, or `None`."""
    command = pm7_rs_command() + ["energy", str(xyz_path), *extra]
    try:
        done = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _RS_HOF.search(done.stdout)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------------------------
# Parsing MOPAC output beyond the heat of formation
#
# Every regex below was written against real MOPAC v23.2.5 output rather than the manual, and the
# printed precision is recorded next to each one because it sets the floor on any threshold that
# compares against it. See `thresholds()`.
# ---------------------------------------------------------------------------------------------

# "          HOMO LUMO ENERGIES (EV) =        -12.082  4.025"          -- 3 dp, closed shell
_MOPAC_HOMO_LUMO = re.compile(
    r"HOMO\s+LUMO\s+ENERGIES\s*\(EV\)\s*=\s*(-?[0-9.]+)\s+(-?[0-9.]+)", re.IGNORECASE
)
# An open shell prints the two channels separately instead:
#   " ALPHA SOMO LUMO (EV)    =         -9.897  6.053"
#   " BETA  SOMO LUMO (EV)    =        -14.026  0.613"
_MOPAC_ALPHA_SOMO = re.compile(
    r"ALPHA\s+SOMO\s+LUMO\s*\(EV\)\s*=\s*(-?[0-9.]+)\s+(-?[0-9.]+)", re.IGNORECASE
)
_MOPAC_BETA_SOMO = re.compile(
    r"BETA\s+SOMO\s+LUMO\s*\(EV\)\s*=\s*(-?[0-9.]+)\s+(-?[0-9.]+)", re.IGNORECASE
)
# An unrestricted run also prints how far its determinant is from a spin eigenfunction:
#   "          (SZ)    =    0.500000"
#   "          (S**2)  =    0.753039"
# Only UHF prints it, so a `None` here means "restricted", not "not parsed".
_MOPAC_SPIN_SQUARED = re.compile(r"\(S\*\*2\)\s*=\s*(-?[0-9.]+)")
# "           THE ELECTRIC FIELD IS   0.50000   0.00000   0.00000 VOLTS/ANGSTROM"
_MOPAC_FIELD = re.compile(
    r"THE ELECTRIC FIELD IS\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s*VOLTS/ANGSTROM",
    re.IGNORECASE,
)
# " POINT-CHG.     1.115     1.441     0.000     1.822"                -- 3 dp, Debye
_MOPAC_DIPOLE_ROW = re.compile(
    r"^\s*(POINT-CHG\.|HYBRID|SUM)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s*$",
    re.MULTILINE,
)
# "      1          1  O    CARTESIAN X     0.000000     -2.345404  KCAL/ANGSTROM"  -- 6 dp
_MOPAC_GRADIENT_ROW = re.compile(
    r"^\s*\d+\s+(\d+)\s+\S+\s+CARTESIAN\s+([XYZ])\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+KCAL/ANGSTROM",
    re.MULTILINE,
)
# "     1          O          -0.645142        6.6451     1.81893     4.82622"      -- 6 dp
_MOPAC_CHARGE_ROW = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(-?[0-9.]+)\s+[0-9.]+", re.MULTILINE
)
# The `FORCE LARGE` block: rows of six, keyed by label. 4 dp on FREQ, 5 dp on the dipoles.
_MOPAC_VIB_ROW = re.compile(
    r"^\s*(FREQ|MASS|DIPX|DIPY|DIPZ|DIPT)\(I\)((?:\s+-?[0-9.]+)+)\s*$", re.MULTILINE
)


@dataclass
class VibrationalMode:
    """One row of MOPAC's `FREQUENCIES, REDUCED MASSES AND VIBRATIONAL DIPOLES` block.

    MOPAC emits these in its own internal order, which is **not** sorted by frequency and mixes
    the vibrations with the translations and rotations. Match by frequency, never by index.
    """

    freq_cm: float
    reduced_mass: float
    dip: tuple[float, float, float]
    dipt: float


@dataclass
class MopacRun:
    """Everything the oracle reads out of one MOPAC run. `None` means "not printed"."""

    hof_kcal: float | None = None
    homo_ev: float | None = None
    lumo_ev: float | None = None
    homo_ev_beta: float | None = None
    lumo_ev_beta: float | None = None
    #: `<S^2>`, printed only by an unrestricted run. `None` therefore means "restricted".
    spin_squared: float | None = None
    field_v_per_angstrom: tuple[float, float, float] | None = None
    dipole_point_charge: tuple[float, float, float] | None = None
    dipole_hybrid: tuple[float, float, float] | None = None
    dipole_sum: tuple[float, float, float] | None = None
    dipole_total: float | None = None
    charges: list[float] = field_(default_factory=list)
    # `gradient_kcal_per_angstrom[i]` is atom `i`'s (x, y, z), in MOPAC's printed sign.
    gradient_kcal_per_angstrom: list[tuple[float, float, float]] = field_(default_factory=list)
    orbital_energies_ev: list[float] = field_(default_factory=list)
    vibrations: list[VibrationalMode] = field_(default_factory=list)
    text: str = ""


def _triple(match: re.Match[str], start: int) -> tuple[float, float, float]:
    return (float(match.group(start)), float(match.group(start + 1)), float(match.group(start + 2)))


def _parse_orbital_energies(text: str) -> list[float]:
    """The eigenvalue rows of the `EIGENVECTORS` block.

    MOPAC prints the block in column groups of six: a `Root No.` header, a symmetry-label row, then
    the energies. The energies are the only all-numeric row in the group, and the coefficient rows
    that follow always start with an AO label (`S`, `Px`, ...), so "a line of nothing but numbers
    inside the EIGENVECTORS section" identifies them without needing to count lines.
    """
    start = text.find("EIGENVECTORS")
    if start < 0:
        return []
    stop = text.find("NET ATOMIC CHARGES", start)
    section = text[start : stop if stop > 0 else len(text)]
    energies: list[float] = []
    for line in section.splitlines():
        fields = line.split()
        if not fields or len(fields) > 8:
            continue
        try:
            values = [float(f) for f in fields]
        except ValueError:
            continue
        # A `Root No.` header is also all-numeric, but its entries are consecutive small integers.
        if all(v == int(v) and 1 <= v <= 10_000 for v in values):
            continue
        energies.extend(values)
    return energies


def _parse_vibrations(text: str) -> list[VibrationalMode]:
    """The `FORCE LARGE` vibrational-dipole block, transposed into one record per mode."""
    if "VIBRATIONAL DIPOLES" not in text:
        return []
    section = text[text.find("VIBRATIONAL DIPOLES") :]
    columns: dict[str, list[float]] = {}
    for match in _MOPAC_VIB_ROW.finditer(section):
        columns.setdefault(match.group(1), []).extend(float(v) for v in match.group(2).split())
    if "FREQ" not in columns:
        return []
    count = min(len(v) for v in columns.values())
    zero = [0.0] * count
    return [
        VibrationalMode(
            freq_cm=columns["FREQ"][i],
            reduced_mass=columns.get("MASS", zero)[i],
            dip=(
                columns.get("DIPX", zero)[i],
                columns.get("DIPY", zero)[i],
                columns.get("DIPZ", zero)[i],
            ),
            dipt=columns.get("DIPT", zero)[i],
        )
        for i in range(count)
    ]


def parse_mopac(text: str) -> MopacRun:
    """Read every quantity the oracle compares out of one MOPAC `.out` file."""
    run = MopacRun(text=text)

    if match := _MOPAC_HOF.search(text):
        run.hof_kcal = float(match.group(1))
    if match := _MOPAC_HOMO_LUMO.search(text):
        run.homo_ev, run.lumo_ev = float(match.group(1)), float(match.group(2))
    # An open shell has no single HOMO/LUMO line; the two channels are printed separately, and
    # the alpha SOMO is the quantity that plays the HOMO's role.
    if match := _MOPAC_ALPHA_SOMO.search(text):
        run.homo_ev, run.lumo_ev = float(match.group(1)), float(match.group(2))
    if match := _MOPAC_BETA_SOMO.search(text):
        run.homo_ev_beta, run.lumo_ev_beta = float(match.group(1)), float(match.group(2))
    # Only an unrestricted run prints this, so `None` distinguishes "MOPAC used RHF" from
    # "MOPAC used UHF and got S(S+1)" -- which is a distinction the comparison relies on.
    if match := _MOPAC_SPIN_SQUARED.search(text):
        run.spin_squared = float(match.group(1))
    if match := _MOPAC_FIELD.search(text):
        run.field_v_per_angstrom = _triple(match, 1)

    for match in _MOPAC_DIPOLE_ROW.finditer(text):
        vector = _triple(match, 2)
        if match.group(1).startswith("POINT"):
            run.dipole_point_charge = vector
        elif match.group(1) == "HYBRID":
            run.dipole_hybrid = vector
        else:
            run.dipole_sum, run.dipole_total = vector, float(match.group(5))

    # The charge table sits under `NET ATOMIC CHARGES`; restricting to that section keeps the
    # deliberately loose row pattern from matching arbitrary numeric tables elsewhere. The
    # terminator has to be looked for *past the header line*, because the header itself reads
    # "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS" and would otherwise close the section
    # immediately, leaving no rows at all.
    charge_start = text.find("NET ATOMIC CHARGES")
    if charge_start >= 0:
        body_start = text.find("\n", charge_start) + 1
        charge_stop = text.find("DIPOLE", body_start)
        section = text[body_start : charge_stop if charge_stop > 0 else len(text)]
        run.charges = [float(m.group(3)) for m in _MOPAC_CHARGE_ROW.finditer(section)]

    gradient: dict[int, list[float]] = {}
    for match in _MOPAC_GRADIENT_ROW.finditer(text):
        atom = int(match.group(1)) - 1
        axis = "XYZ".index(match.group(2))
        gradient.setdefault(atom, [0.0, 0.0, 0.0])[axis] = float(match.group(4))
    run.gradient_kcal_per_angstrom = [
        tuple(gradient[i]) for i in range(len(gradient)) if i in gradient  # type: ignore[misc]
    ]

    run.orbital_energies_ev = _parse_orbital_energies(text)
    run.vibrations = _parse_vibrations(text)
    return run


def run_mopac_full(mopac: Path, mop_path: Path) -> MopacRun | None:
    """Run MOPAC and parse everything, or `None` if it produced no output file at all."""
    try:
        subprocess.run(
            [str(mopac), str(mop_path)],
            cwd=mop_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out_path = mop_path.with_suffix(".out")
    if not out_path.is_file():
        return None
    return parse_mopac(out_path.read_text(encoding="utf-8", errors="replace"))


def run_pm7_rs_json(xyz_path: Path, mode: str, *extra: str) -> dict | None:
    """Run the `pm7-rs` CLI in `--json` mode and return the parsed object, or `None`."""
    command = pm7_rs_command() + [mode, str(xyz_path), "--json", *extra]
    try:
        done = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------------------------


@dataclass
class Breach:
    case: str
    quantity: str
    observed: float
    limit: float


class Checker:
    """Accumulates threshold breaches so a script can exit non-zero instead of being eyeballed.

    The scripts used to print a delta column and leave the judgement to whoever ran them, which
    is fine for a person investigating one molecule and useless as a regression gate.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.breaches: list[Breach] = []
        self.checked = 0
        self.skipped: list[str] = []

    def check(self, case: str, quantity: str, observed: float | None, limit: float) -> bool:
        """Record `|observed| <= limit`. A `None` observation is a skip, not a pass."""
        if observed is None:
            self.skipped.append(f"{case}: {quantity}")
            return False
        self.checked += 1
        if abs(observed) > limit:
            self.breaches.append(Breach(case, quantity, abs(observed), limit))
            return False
        return True

    def report(self) -> int:
        """Print the verdict and return the process exit code."""
        print()
        if self.skipped:
            print(f"{len(self.skipped)} comparison(s) skipped (a side produced no number):")
            for item in self.skipped[:20]:
                print(f"  - {item}")
            if len(self.skipped) > 20:
                print(f"  ... and {len(self.skipped) - 20} more")
        if not self.breaches:
            print(f"{self.label}: {self.checked} comparison(s) within threshold.")
            return 0
        print(f"{self.label}: {len(self.breaches)} of {self.checked} comparison(s) OVER threshold:")
        for breach in sorted(self.breaches, key=lambda b: -b.observed / b.limit):
            print(
                f"  {breach.case:<24} {breach.quantity:<28} "
                f"|delta| = {breach.observed:.6g}  >  {breach.limit:.6g}"
            )
        return 1


def thresholds() -> dict[str, float]:
    """Default oracle tolerances, in the natural unit of each quantity.

    Each one is set at or above MOPAC's **printed** precision, because nothing finer than what the
    `.out` file shows can be compared at all:

    | quantity          | MOPAC prints | threshold          |
    |-------------------|--------------|--------------------|
    | heat of formation | 5 dp kcal    | 1e-4 kcal/mol      |
    | dipole            | 3 dp D       | 2e-3 D             |
    | orbital energies  | 3 dp eV      | 2e-3 eV            |
    | charges           | 6 dp e       | 1e-4 e             |
    | gradient          | 6 dp kcal/A  | 1e-3 kcal/mol/A    |
    | vibrational DIPT  | 5 dp         | 1 % relative       |
    | `<S^2>`           | 6 dp         | 1e-5               |

    `docs/fidelity.md` records the one grandfathered exception: a heat-of-formation residual up to
    about 0.36 kcal/mol for some four-coordinate, highly d-populated transition-metal compounds.
    """
    return {
        "hof_kcal": 1.0e-4,
        "dipole_debye": 2.0e-3,
        "orbital_ev": 2.0e-3,
        "charge_e": 1.0e-4,
        "gradient_kcal_per_angstrom": 1.0e-3,
        "dipt_relative": 1.0e-2,
        # `<S^2>` is printed to six decimals and both sides compute it from the same converged
        # densities, so this is a print-precision threshold rather than a tolerance for a
        # difference anybody expects.
        "spin_squared": 1.0e-5,
    }


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------

BASELINE_DIR = ROOT / "tools/oracle/baselines"


def load_baseline(name: str) -> dict | None:
    """A recorded baseline, or `None` if it has never been written."""
    path = BASELINE_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(name: str, payload: dict) -> Path:
    """Record a baseline. Tracked in git: it is what later runs are compared against."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

@dataclass
class Row:
    """One molecule's comparison. `None` means that side did not produce a number."""

    name: str
    values: dict[str, float | int | str | None]


def delta(a: float | None, b: float | None, places: int = 4) -> float | None:
    """`a − b`, rounded, or `None` if either side is missing."""
    if a is None or b is None:
        return None
    return round(a - b, places)


def table(rows: Iterable[Row], columns: Sequence[str]) -> str:
    """A fixed-width table. Nothing fancy — it is read in a terminal, not parsed."""
    rows = list(rows)
    headers = ["mol", *columns]

    def cell(value: float | int | str | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    body = [[row.name, *(cell(row.values.get(c)) for c in columns)] for row in rows]
    widths = [
        max(len(headers[i]), *(len(line[i]) for line in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    out.append("  ".join("-" * w for w in widths))
    out.extend("  ".join(c.ljust(w) for c, w in zip(line, widths)) for line in body)
    return "\n".join(out)


def compare(
    name: str,
    atoms: Sequence[str],
    directory: Path,
    mopac: Path,
    *,
    mopac_keywords: str = "PM7 1SCF PRECISE",
    rs_extra: Sequence[str] = (),
    suffix: str = "",
) -> tuple[float | None, float | None]:
    """Write both inputs, run both programs, return `(mopac_hof, pm7_rs_hof)` in kcal/mol."""
    xyz_path = write_xyz(directory / f"{name}.xyz", name, atoms)
    mop_path = write_mop(directory / f"{name}{suffix}.mop", name, atoms, mopac_keywords)
    return run_mopac(mopac, mop_path), run_pm7_rs(xyz_path, *rs_extra)
