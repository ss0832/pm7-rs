# SPDX-License-Identifier: GPL-3.0-or-later
"""pm7-rs vs MOPAC PM7 heat of formation for molecules that use the d shell.

    python tools/oracle/dmols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle import ROOT, Row, compare, delta, mopac_executable, table  # noqa: E402

MOLECULES = {
    "sih4": [
        "Si 0.0 0.0 0.0",
        "H 0.8545 0.8545 0.8545",
        "H 0.8545 -0.8545 -0.8545",
        "H -0.8545 0.8545 -0.8545",
        "H -0.8545 -0.8545 0.8545",
    ],
    "so2": ["S 0.0 0.0 0.0", "O 1.2340 0.0 0.7195", "O -1.2340 0.0 0.7195"],
    "ph3": [
        "P 0.0 0.0 0.0",
        "H 0.0 1.1932 0.7715",
        "H 1.0333 -0.5966 0.7715",
        "H -1.0333 -0.5966 0.7715",
    ],
    "hcl": ["Cl 0.0 0.0 0.0", "H 0.0 0.0 1.2746"],
    "pf3": [
        "P 0.0 0.0 0.0",
        "F 0.0 1.2860 0.8043",
        "F 1.1137 -0.6430 0.8043",
        "F -1.1137 -0.6430 0.8043",
    ],
}


def main() -> int:
    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/dcases"
    rows = []
    for name, atoms in MOLECULES.items():
        mopac_hof, rs_hof = compare(
            name, atoms, directory, mopac, mopac_keywords="PM7 1SCF PRECISE AUX"
        )
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
