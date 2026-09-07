# SPDX-License-Identifier: GPL-3.0-or-later
"""The PM7 hydrogen-bond correction against MOPAC, on H-bonded dimers.

Each dimer is run twice on each side — full PM7 and PM7- — so the correction itself can be read
off as the difference. That matters because a full-PM7 disagreement says nothing about *which*
piece disagrees: `d_scf` is the uncorrected residual and `d_corr` is the correction's own.

    python tools/oracle/hbond_check.py
"""

from __future__ import annotations

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

MOLECULES = {
    # Classic Cs water dimer (O–O ≈ 2.91 Å).
    "water_dimer": [
        "O -1.551007 -0.114520 0.000000",
        "H -1.934259 0.762503 0.000000",
        "H -0.599677 0.040712 0.000000",
        "O 1.350625 0.111469 0.000000",
        "H 1.680398 -0.373741 -0.758561",
        "H 1.680398 -0.373741 0.758561",
    ],
    "ammonia_dimer": [
        "N -1.578718 0.000000 0.100792",
        "H -1.983213 0.811863 -0.356218",
        "H -1.983213 -0.811863 -0.356218",
        "H -1.884372 0.000000 1.062051",
        "N 1.578718 0.000000 -0.100792",
        "H 2.166021 0.000000 0.719402",
        "H 1.983213 0.811863 -0.556218",
        "H 1.983213 -0.811863 -0.556218",
    ],
    # Formic acid dimer: a double hydrogen bond, O–O ≈ 2.67 Å.
    "formic_dimer": [
        "C -1.888000 0.000000 0.117000",
        "O -1.286000 0.000000 1.176000",
        "O -1.310000 0.000000 -1.096000",
        "H -0.335000 0.000000 -0.981000",
        "H -2.976000 0.000000 0.140000",
        "C 1.888000 0.000000 -0.117000",
        "O 1.286000 0.000000 -1.176000",
        "O 1.310000 0.000000 1.096000",
        "H 0.335000 0.000000 0.981000",
        "H 2.976000 0.000000 -0.140000",
    ],
}


def main() -> int:
    mopac = mopac_executable()
    directory = ROOT / "tools/oracle/hbcases"
    rows = []
    for name, atoms in MOLECULES.items():
        xyz_path = write_xyz(directory / f"{name}.xyz", name, atoms)

        full = run_mopac(
            mopac, write_mop(directory / f"{name}.mop", name, atoms, "PM7 1SCF PRECISE")
        )
        minus = run_mopac(
            mopac, write_mop(directory / f"{name}_m.mop", name, atoms, "PM7- 1SCF PRECISE")
        )
        rs_full = run_pm7_rs(xyz_path)
        rs_minus = run_pm7_rs(xyz_path, "--method", "pm7-")

        rows.append(
            Row(
                name,
                {
                    "mop_corr": delta(full, minus),
                    "rs_corr": delta(rs_full, rs_minus),
                    "d_corr": delta(delta(rs_full, rs_minus), delta(full, minus)),
                    "d_scf": delta(rs_minus, minus),
                    "mop_full": full,
                    "rs_full": rs_full,
                },
            )
        )
    print(table(rows, ["mop_corr", "rs_corr", "d_corr", "d_scf", "mop_full", "rs_full"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
