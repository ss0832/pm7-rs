# SPDX-License-Identifier: GPL-3.0-or-later
"""Element-pair value sweep against MOPAC.

For every PM7 element A, build the A–H and A–F diatomics at 1.05 × the sum of covalent radii,
choose the multiplicity from the valence-electron parity, and compare heats of formation. A pair
whose ΔHf is off by more than a threshold is the signature of a core-core or pair-scaling bug in
that element's parameters — the kind of thing a curated molecule set walks straight past.

Pairs that converge on one side only are reported separately: "MOPAC converged and we did not" is
an SCF robustness finding, not a value finding, and mixing the two hides both.

    python tools/oracle/pair_sweep.py [--threshold 1.0] [--partners H,F]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle import (  # noqa: E402
    ROOT,
    Row,
    delta,
    mopac_executable,
    run_mopac,
    run_pm7_rs,
    table,
    write_mop,
    write_xyz,
)

# Element symbols, indexed by Z.
SYMBOL = (
    "",
    *"H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split(),
)

# Covalent radii in Ångström, indexed by Z (Cordero/Pyykkö, matching `constants.rs` RAD_A).
RADIUS = (
    0.00, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
    1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
    1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,
    1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
    2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62,
    1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
)


def core_charges() -> dict[int, float]:
    """`Z -> core_charge` for every element that has **orbitals**, from the parameter CSV.

    The table also carries MOPAC's pseudo-elements, and they are not chemistry:

    * ``Z = 87..101`` and ``Z = 103..107`` are **sparkles** — pure point charges with no basis
      functions at all, used to model counter-ions and lanthanides. They carry ``n = 0``.
    * ``Z = 102`` is ``Tv``, the translation-vector marker. It has ``n = 3`` and so survives an
      orbital test; what marks it is the sentinel ``beta_s = -9999999``.
    * ``Z = 98`` is ``Cb``, MOPAC's capped-bond link atom. It has genuine parameters
      (``u_ss = -3``, ``beta_s = -99``, one s orbital) but they are chosen to terminate a bond in
      a QM/MM partition, not to reproduce an element. A ``Cb-H`` diatomic is a number, not
      chemistry.

    All three are skipped, so the sweep reports coverage of the **elements** and nothing else.
    Without a filter it walked off the end of ``SYMBOL`` and died with an ``IndexError`` before
    running a single comparison; with a symbol table merely long enough to index, it would instead
    have built diatomics out of two point charges, a lattice vector, or a link atom and counted
    them as element coverage. So the exclusions are named here rather than the bound quietly
    widened — an oracle that silently tests the wrong thing is worse than one that crashes.
    """
    path = ROOT / "src/data/pm7_elements.csv"
    out: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle):
            if not fields or fields[0].lstrip().startswith("#") or fields[0] == "z":
                continue
            if len(fields) <= 13:
                continue
            z = int(fields[0])
            has_orbitals = int(fields[1]) > 0
            # `SYMBOL` stops at Rn (86); everything past it is a special-purpose species.
            is_element = 1 <= z < len(SYMBOL)
            if has_orbitals and is_element:
                out[z] = float(fields[5])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="flag a pair whose |delta| exceeds this, in kcal/mol (default 1.0)",
    )
    parser.add_argument(
        "--partners",
        default="H,F",
        help="comma-separated partner element symbols (default H,F)",
    )
    args = parser.parse_args()

    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/pairsweep"
    tore = core_charges()
    partners = [p.strip() for p in args.partners.split(",") if p.strip()]
    partner_z = {p: SYMBOL.index(p) for p in partners}

    rows: list[Row] = []
    for za in sorted(tore):
        symbol_a = SYMBOL[za]
        for partner, zp in partner_z.items():
            if za == zp or zp not in tore:
                continue
            n_electrons = tore[za] + tore[zp]
            if n_electrons < 2:
                continue
            odd = int(round(n_electrons)) % 2 != 0
            multiplicity = 2 if odd else 1
            r = round(1.05 * (RADIUS[za] + RADIUS[zp]), 4)
            name = f"{symbol_a}{partner}"
            atoms = [f"{symbol_a} 0.0 0.0 0.0", f"{partner} 0.0 0.0 {r}"]

            xyz_path = write_xyz(directory / f"{name}.xyz", name, atoms)
            keywords = "PM7 1SCF PRECISE UHF DOUBLET" if odd else "PM7 1SCF PRECISE"
            mop_path = write_mop(directory / f"{name}.mop", name, atoms, keywords)
            mopac_hof = run_mopac(mopac, mop_path)
            rs_hof = run_pm7_rs(xyz_path, "--multiplicity", str(multiplicity))
            rows.append(
                Row(
                    name,
                    {
                        "mult": multiplicity,
                        "mopac": mopac_hof,
                        "pm7rs": rs_hof,
                        "delta": delta(rs_hof, mopac_hof),
                    },
                )
            )

    columns = ["mult", "mopac", "pm7rs", "delta"]
    both = [r for r in rows if r.values["delta"] is not None]
    flagged = sorted(
        (r for r in both if abs(float(r.values["delta"])) > args.threshold),
        key=lambda r: -abs(float(r.values["delta"])),
    )
    only_mopac = [r for r in rows if r.values["mopac"] is not None and r.values["pm7rs"] is None]
    only_rs = [r for r in rows if r.values["mopac"] is None and r.values["pm7rs"] is not None]

    print(
        f"pairs tested: {len(rows)}; converged in both: {len(both)}; "
        f"|delta| > {args.threshold}: {len(flagged)}"
    )
    print(f"\n--- flagged (|delta| > {args.threshold} kcal/mol) ---")
    print(table(flagged, columns) if flagged else "(none)")
    print("\n--- pm7-rs failed but MOPAC converged (SCF robustness, not a value residual) ---")
    print(table(only_mopac, ["mult", "mopac"]) if only_mopac else "(none)")
    print("\n--- MOPAC failed but pm7-rs converged ---")
    print(table(only_rs, ["mult", "pm7rs"]) if only_rs else "(none)")
    print("\n--- worst 15 by |delta| (converged in both) ---")
    worst = sorted(both, key=lambda r: -abs(float(r.values["delta"])))[:15]
    print(table(worst, columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
