# SPDX-License-Identifier: GPL-3.0-or-later
"""pm7-rs vs MOPAC for **Sparkle/PM7**, over every lanthanide the model covers.

    python tools/oracle/sparkles.py
    python tools/oracle/sparkles.py --distances 1.9 2.1 2.4

A Sparkle is a `+3` point core with **no atomic orbitals** (`src/data/pm7_sparkles.csv`,
`Z = 57..71`), so it exercises a path nothing else does: the two-electron assembly has to skip an
atom that occupies no rows of the density, the core-core repulsion has to use the Sparkle's own
`alpha`/`g` Gaussians rather than the ordinary element table, and the electron-core attraction has
to keep acting even though the atom contributes no electrons.

`tests/molecules.rs` pins two of these against MOPAC (EuF3 and GdF3) at a tolerance of 2 kcal/mol,
which is what could be asserted from two hand-copied numbers. This sweep is the measurement that
tolerance was standing in for: every element, at several bond lengths, against MOPAC run here.

The geometry is a planar `D3h` `LnF3` at a fixed `Ln-F`. It is not the equilibrium structure of
anything, and it does not need to be -- both programs evaluate the same Hamiltonian at the same
coordinates, so any disagreement is an implementation difference and nothing else. Sweeping the
distance is what separates a constant offset (a heat-of-formation reference) from a distance
dependence (the core-core Gaussians or the electron-core attraction).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle import ROOT, Row, compare, delta, mopac_executable, table  # noqa: E402

# Z = 57..71, the Sparkle/PM7 range in `src/data/pm7_sparkles.csv`.
LANTHANIDES = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
]

DEFAULT_DISTANCES = (2.1,)


def trifluoride(symbol: str, r: float) -> list[str]:
    """Planar D3h `LnF3` with the metal at the origin and `Ln-F = r`."""
    c, s = math.cos(math.radians(120.0)), math.sin(math.radians(120.0))
    return [
        f"{symbol} 0.0 0.0 0.0",
        f"F {r:.4f} 0.0 0.0",
        f"F {r * c:.4f} {r * s:.4f} 0.0",
        f"F {r * c:.4f} {-r * s:.4f} 0.0",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCES),
        metavar="ANGSTROM",
        help="Ln-F distances to sweep (default: 2.1)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=LANTHANIDES,
        metavar="SYMBOL",
        help="which lanthanides to run (default: all fifteen)",
    )
    args = parser.parse_args()

    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/dcases"
    directory.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    worst = 0.0
    worst_name = ""
    for r in args.distances:
        for symbol in args.elements:
            name = f"{symbol.lower()}f3_{r:.2f}".replace(".", "p")
            atoms = trifluoride(symbol, r)
            mopac_hof, rs_hof = compare(
                name,
                atoms,
                directory,
                mopac,
                mopac_keywords="PM7 SPARKLE 1SCF PRECISE",
                rs_extra=("--method", "pm7-sparkle"),
            )
            d = delta(rs_hof, mopac_hof)
            rows.append(
                Row(
                    f"{symbol}F3 @ {r:.2f}",
                    {"mopac": mopac_hof, "pm7rs": rs_hof, "delta": d},
                )
            )
            if d is not None and abs(d) > abs(worst):
                worst, worst_name = d, f"{symbol}F3 @ {r:.2f}"

    print(table(rows, ["mopac", "pm7rs", "delta"]))
    missing = [row.name for row in rows if row.values.get("pm7rs") is None]
    if missing:
        print(f"\npm7-rs produced no answer for: {', '.join(missing)}")
    print(f"\nworst |delta| = {abs(worst):.4f} kcal/mol at {worst_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
