# SPDX-License-Identifier: GPL-3.0-or-later
"""pm7-rs vs MOPAC PM7 heat of formation for a few s/p-only molecules.

    python tools/oracle/spmols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle import ROOT, Row, compare, delta, mopac_executable, table  # noqa: E402

MOLECULES = {
    "nh3": [
        "N 0.0 0.0 0.0",
        "H 0.0 0.9377 0.3816",
        "H 0.8121 -0.4689 0.3816",
        "H -0.8121 -0.4689 0.3816",
    ],
    "ch4": [
        "C 0.0 0.0 0.0",
        "H 0.6276 0.6276 0.6276",
        "H 0.6276 -0.6276 -0.6276",
        "H -0.6276 0.6276 -0.6276",
        "H -0.6276 -0.6276 0.6276",
    ],
    "hf": ["F 0.0 0.0 0.0", "H 0.0 0.0 0.9169"],
    "co": ["C 0.0 0.0 0.0", "O 0.0 0.0 1.1283"],
}


def main() -> int:
    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/dcases"
    rows = []
    for name, atoms in MOLECULES.items():
        mopac_hof, rs_hof = compare(name, atoms, directory, mopac)
        rows.append(
            Row(
                name,
                {"mopac": mopac_hof, "pm7rs": rs_hof, "delta": delta(rs_hof, mopac_hof)},
            )
        )
    print(table(rows, ["mopac", "pm7rs", "delta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
