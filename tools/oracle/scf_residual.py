# SPDX-License-Identifier: GPL-3.0-or-later
"""The pure SCF + core-core residual per monomer: MOPAC PM7- vs pm7-rs PM7-minus.

Running both sides with the corrections switched off separates "the SCF differs" from "a
correction differs", which is the first thing worth knowing when a heat of formation is off.

    python tools/oracle/scf_residual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle import ROOT, Row, compare, delta, mopac_executable, table  # noqa: E402

MOLECULES = {
    "h2o": ["O 0.0000 0.0000 0.0000", "H 0.9584 0.0000 0.0000", "H -0.2400 0.9278 0.0000"],
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
    "n2": ["N 0.0 0.0 0.0", "N 0.0 0.0 1.0977"],
    "hcn": ["H 0.0 0.0 0.0", "C 0.0 0.0 1.0640", "N 0.0 0.0 2.2200"],
    "ch3nh2": [
        "C -0.6870 0.0176 0.0000",
        "N 0.7379 -0.1170 0.0000",
        "H -1.0980 0.5203 0.8800",
        "H -1.0980 0.5203 -0.8800",
        "H -1.0350 -1.0142 0.0000",
        "H 1.1476 0.3639 0.7912",
        "H 1.1476 0.3639 -0.7912",
    ],
    "hcooh": [
        "C 0.0 0.0 0.0",
        "O 0.0 1.2015 0.0",
        "O -1.1042 -0.7142 0.0",
        "H -1.9182 -0.1926 0.0",
        "H 0.9367 -0.5599 0.0",
    ],
}


def main() -> int:
    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/scfres"
    rows = []
    for name, atoms in MOLECULES.items():
        mopac_hof, rs_hof = compare(
            name,
            atoms,
            directory,
            mopac,
            mopac_keywords="PM7- 1SCF PRECISE",
            rs_extra=["--method", "pm7-"],
        )
        rows.append(
            Row(
                name,
                {"mopac": mopac_hof, "pm7rs": rs_hof, "d_scf": delta(rs_hof, mopac_hof)},
            )
        )
    print(table(rows, ["mopac", "pm7rs", "d_scf"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
