#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract reproducible PM7 tables from a pinned MOPAC source checkout.

The tables this emits are derived from MOPAC v23.2.5 (Apache-2.0, (c) 2021 Virginia Tech);
`THIRD_PARTY_NOTICES.md` records the attribution and `third_party/mopac/LICENSE` carries the
license text that has to travel with anything built from them.

The MOPAC PM7 parameter files are Fortran ``data`` statements.  This script keeps
the generated Rust inputs auditable: it accepts only a source tree, performs no
network access, and emits stable CSV files under ``src/data``.  The generated
tables preserve MOPAC's native eV and Angstrom-based parameter units.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ELEMENT_FIELDS = (
    "u_ss", "u_pp", "u_dd", "zeta_s", "zeta_p", "zeta_d", "beta_s",
    "beta_p", "beta_d", "g_ss", "g_sp", "g_pp", "g_p2", "h_sp",
    "zsn", "zpn", "zdn", "f0sd", "g2sd", "alpha", "poc", "polvol",
)

FORTRAN_TO_CSV = {
    "u_ss": "uss7", "u_pp": "upp7", "u_dd": "udd7",
    "zeta_s": "zs7", "zeta_p": "zp7", "zeta_d": "zd7",
    "beta_s": "betas7", "beta_p": "betap7", "beta_d": "betad7",
    "g_ss": "gss7", "g_sp": "gsp7", "g_pp": "gpp7", "g_p2": "gp27",
    "h_sp": "hsp7", "zsn": "zsn7", "zpn": "zpn7", "zdn": "zdn7",
    "f0sd": "f0sd7", "g2sd": "g2sd7", "alpha": "alp7",
    "poc": "poc_7", "polvol": "polvo7",
}


def strip_fortran_comments(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def fortran_float(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def assignments(text: str) -> dict[str, dict[tuple[int, int], float]]:
    """Read scalar Fortran data statements, including two-index arrays."""
    out: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    pattern = re.compile(
        r"\bdata\s+([A-Za-z][A-Za-z0-9_]*)\s*\(\s*(\d+)\s*"
        r"(?:,\s*(\d+)\s*)?\)\s*/\s*([^/\s]+)\s*/",
        flags=re.IGNORECASE,
    )
    cleaned = strip_fortran_comments(text)
    for name, first, second, value in pattern.findall(cleaned):
        try:
            parsed = fortran_float(value)
        except ValueError:
            continue
        out[name.lower()][(int(first), int(second or "1"))] = parsed
    assignment_pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*(\d+)\s*"
        r"(?:,\s*(\d+)\s*)?\)\s*=\s*([^\s]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for name, first, second, value in assignment_pattern.findall(cleaned):
        try:
            parsed = fortran_float(value)
        except ValueError:
            continue
        out[name.lower()][(int(first), int(second or "1"))] = parsed
    return out


def vector_block(text: str, name: str) -> list[int]:
    """Expand a Fortran ``data name / ... /`` integer vector."""
    cleaned = strip_fortran_comments(text)
    match = re.search(
        rf"\bdata\s+{re.escape(name)}(?:\s*\([^)]*\))?\s*&?\s*(?:&\s*)?/\s*(.*?)/",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing Fortran data vector {name}")
    values: list[int] = []
    for token in match.group(1).replace("&", " ").split(","):
        token = token.strip()
        if not token:
            continue
        repeated = re.fullmatch(r"([+-]?\d+)\s*\*\s*([+-]?\d+)", token)
        if repeated:
            values.extend([int(repeated.group(2))] * int(repeated.group(1)))
        else:
            values.append(int(token))
    if len(values) != 107:
        raise ValueError(f"{name} has {len(values)} entries, expected 107")
    return values


def value(table: dict[str, dict[tuple[int, int], float]], name: str, z: int, slot: int = 1) -> float:
    return table.get(name.lower(), {}).get((z, slot), 0.0)


def fmt(value: float) -> str:
    return f"{value:.12g}"


def write_elements(
    output: Path,
    source: Path,
    suffix: str,
    occupations: tuple[list[int], list[int], list[int], list[int]],
    eheat: dict[tuple[int, int], float],
) -> None:
    table = assignments(source.read_text(encoding="utf-8"))
    ios, iop, iod, npq = occupations
    stem = "7" if not suffix else "7_TS"
    mapping = {field: raw.replace("7", stem) for field, raw in FORTRAN_TO_CSV.items()}
    header = ["z", "n", "ios", "iop", "iod", "core_charge", "eheat_kcal", *ELEMENT_FIELDS]
    header += [f"g{k}_{slot}" for k in (1, 2, 3) for slot in range(1, 5)]
    rows: list[list[str]] = []
    for z in range(1, 108):
        row = [str(z), str(npq[z - 1]), str(ios[z - 1]), str(iop[z - 1]), str(iod[z - 1]), str(ios[z - 1] + iop[z - 1] + iod[z - 1]), fmt(eheat.get((z, 1), 0.0))]
        row.extend(fmt(value(table, mapping[field], z)) for field in ELEMENT_FIELDS)
        for k in (1, 2, 3):
            raw = f"gues7{suffix}{k}" if suffix else f"gues7{k}"
            row.extend(fmt(value(table, raw, z, slot)) for slot in range(1, 5))
        rows.append(row)
    write_csv(output, header, rows, source.name)


def write_pairs(output: Path, source: Path) -> None:
    table = assignments(source.read_text(encoding="utf-8"))
    pairs: dict[tuple[int, int], list[float]] = {}
    for key in set(table.get("alpb", {})) | set(table.get("xfac", {})):
        z_hi, z_lo = key
        pairs[(max(z_hi, z_lo), min(z_hi, z_lo))] = [
            value(table, "alpb", z_hi, z_lo), value(table, "xfac", z_hi, z_lo)
        ]
    rows = [[str(hi), str(lo), fmt(values[0]), fmt(values[1])] for (hi, lo), values in sorted(pairs.items())]
    write_csv(output, ["z_hi", "z_lo", "alpb", "xfac"], rows, source.name)


def write_vpar(output: Path, source: Path, name: str) -> None:
    table = assignments(source.read_text(encoding="utf-8"))
    rows = [[str(index), fmt(value(table, name, index))] for index in range(1, 61) if value(table, name, index) != 0.0]
    write_csv(output, ["index", "value"], rows, source.name)


def write_sparkles(output: Path, source: Path) -> None:
    table = assignments(source.read_text(encoding="utf-8"))
    rows: list[list[str]] = []
    for z in range(57, 72):
        row = [str(z), fmt(value(table, "alp7sp", z)), fmt(value(table, "gss7sp", z))]
        for slot in (1, 2):
            row.extend(
                [
                    fmt(value(table, "gues7sp1", z, slot)),
                    fmt(value(table, "gues7sp2", z, slot)),
                    fmt(value(table, "gues7sp3", z, slot)),
                ]
            )
        rows.append(row)
    write_csv(
        output,
        ["z", "alpha", "g_ss", "g1_1", "g2_1", "g3_1", "g1_2", "g2_2", "g3_2"],
        rows,
        source.name,
    )


def write_csv(path: Path, header: list[str], rows: list[list[str]], source_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# Generated by tools/extract_params/extract_pm7.py; do not hand edit.\n")
        handle.write(f"# Source: MOPAC v23.2.5 src/models/{source_name} (Apache-2.0).\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mopac-root", type=Path, default=Path(".mopac-source"))
    parser.add_argument("--output", type=Path, default=Path("src/data"))
    args = parser.parse_args()
    models = args.mopac_root / "src" / "models"
    pm7 = models / "parameters_for_PM7_C.F90"
    ts = models / "parameters_for_PM7_TS_C.F90"
    sparkle = models / "parameters_for_PM7_Sparkles_C.F90"
    parameters = models / "parameters_C.F90"
    required = [pm7, ts, sparkle, parameters]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("MOPAC source files not found:\n" + "\n".join(missing))

    common = parameters.read_text(encoding="utf-8")
    occupations = tuple(vector_block(common, field) for field in ("ios", "iop", "iod", "npq"))
    eheat = assignments(common).get("eheat", {})
    write_elements(args.output / "pm7_elements.csv", pm7, "", occupations, eheat)
    write_pairs(args.output / "pm7_pairs.csv", pm7)
    write_vpar(args.output / "pm7_vpar.csv", pm7, "v_par7")
    write_elements(args.output / "pm7ts_elements.csv", ts, "_TS", occupations, eheat)
    write_pairs(args.output / "pm7ts_pairs.csv", ts)
    write_vpar(args.output / "pm7ts_vpar.csv", ts, "v_par7_TS")
    write_sparkles(args.output / "pm7_sparkles.csv", sparkle)


if __name__ == "__main__":
    main()
