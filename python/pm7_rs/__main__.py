# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line interface: ``pm7-rs <mode> structure.xyz [options]``.

The Rust ``pm7_rs_cli`` binary is not shipped inside the Python wheel, so this provides the same
capabilities to anyone who installed with ``pip``. Extended-XYZ ``Lattice=`` and ``pbc=`` keys in
the comment line are honoured, so a periodic structure written by ASE runs as a periodic
calculation with no extra flags.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import native

_MODES = (
    "energy",
    "charges",
    "gradient",
    "forces",
    "stress",
    "optimize",
    "frequencies",
    "hessian",
    "phonons",
    "dfpt",
    "born",
    "bands",
    "orbitals",
    "molden",
    "dielectric",
)


def _write_xyz(path, symbols, positions, cell=None, pbc=None, title="pm7-rs"):
    """Write an extended-XYZ file that :func:`_read_xyz` reads back to the same system.

    The cell is written whenever there is one. A geometry printed without its lattice cannot be
    read back as the system it came from -- the coordinates alone do not say what they repeat in --
    and that is what the ``optimize`` output used to do.
    """
    header = title.replace('"', " ").replace("\n", " ")
    if cell is not None:
        numbers = " ".join(f"{value:.12f}" for row in cell for value in row)
        flags = pbc if pbc is not None else (True, True, True)
        marks = " ".join("T" if f else "F" for f in flags)
        header = (
            f'Lattice="{numbers}" Properties=species:S:1:pos:R:3 pbc="{marks}" {header}'
        )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(symbols)}\n{header}\n")
        for symbol, p in zip(symbols, positions):
            handle.write(f"{symbol:<3} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")


def _read_xyz(path):
    """Read an XYZ file, honouring the extended-XYZ ``Lattice``/``pbc`` keys."""
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if len(lines) < 2:
        raise SystemExit(f"{path}: not an XYZ file")
    count = int(lines[0].split()[0])
    comment = lines[1]
    symbols, positions = [], []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit(f"{path}: malformed atom line: {line!r}")
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(positions, dtype=float), _parse_lattice(comment)


def _quoted(text, key):
    """Value of ``key="..."`` (or a bare ``key=value``) in an extended-XYZ comment line."""
    for token in (key + "=",):
        index = text.find(token)
        while index != -1:
            if index == 0 or text[index - 1].isspace():
                rest = text[index + len(token) :]
                if rest.startswith('"'):
                    return rest[1:].split('"', 1)[0]
                return rest.split()[0] if rest.split() else ""
            index = text.find(token, index + 1)
    return None


def _parse_lattice(comment):
    lattice = _quoted(comment, "Lattice")
    if lattice is None:
        return None, None
    values = [float(x) for x in lattice.split()]
    if len(values) != 9:
        raise SystemExit("extended-XYZ Lattice needs 9 numbers")
    cell = np.asarray(values, dtype=float).reshape(3, 3)
    flags = _quoted(comment, "pbc")
    pbc = None
    if flags is not None:
        pbc = [f.lower() in ("t", "true", "1") for f in flags.split()]
        if len(pbc) != 3:
            raise SystemExit("extended-XYZ pbc needs 3 flags")
    return cell, pbc


_SYMBOLS: dict[str, int] = {}


def _z(symbol):
    if not _SYMBOLS:
        table = (
            "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
            "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
        ).split()
        _SYMBOLS.update({s: i + 1 for i, s in enumerate(table)})
    try:
        return int(symbol)
    except ValueError:
        pass
    key = symbol[:1].upper() + symbol[1:].lower()
    if key not in _SYMBOLS:
        raise SystemExit(f"unknown element {symbol!r}")
    return _SYMBOLS[key]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="pm7-rs",
        description="PM7-family semiempirical calculations on molecules and periodic systems.",
        # No prefix abbreviation: by default argparse accepts `--kpoint` as a short form of
        # `--kpoints`, so a typo silently becomes a different flag — and if a future flag makes
        # the abbreviation ambiguous, a command that used to work starts failing. The Rust CLI
        # rejects unknown flags outright; this makes the two agree.
        allow_abbrev=False,
    )
    parser.add_argument("mode", choices=_MODES)
    parser.add_argument("structure", help="XYZ or extended-XYZ file (Angstrom)")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--method", default="pm7")
    parser.add_argument("--reference", default="auto")
    parser.add_argument(
        "--cell",
        help="3, 6 or 9 comma- or space-separated lattice components in Angstrom (1-D, 2-D or "
        "3-D; nine is row-major); overrides an extended-XYZ Lattice key",
    )
    parser.add_argument("--pbc", help='per-direction periodicity, e.g. "TTF"')
    parser.add_argument(
        "--kpoints",
        nargs=3,
        type=int,
        metavar=("N1", "N2", "N3"),
        help="Monkhorst-Pack divisions (default: the Gamma point)",
    )
    parser.add_argument(
        "--smearing",
        nargs=2,
        metavar=("KIND", "WIDTH_EV"),
        help="occupation broadening for metals: none | fermi | gauss | mp",
    )
    parser.add_argument(
        "--kshift",
        nargs=3,
        type=float,
        metavar=("S1", "S2", "S3"),
        help="fractional offset of the Monkhorst-Pack mesh",
    )
    parser.add_argument("--pbc-mode", choices=("ewald", "mopac"), default=None)
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        metavar=("N1", "N2", "N3"),
        help="force-constant supercell (phonons; default 1 1 1)",
    )
    parser.add_argument(
        "--qpoints",
        nargs="+",
        metavar="X,Y,Z",
        help="fractional q points (phonons) or k points (bands)",
    )
    parser.add_argument(
        "--acoustic-sum-rule",
        action="store_true",
        help="project the acoustic sum rule out of the force constants (the default since 0.2.3)",
    )
    parser.add_argument(
        "--no-acoustic-sum-rule",
        action="store_true",
        help="keep the raw force constants, residual and all",
    )
    parser.add_argument(
        "--lo-to",
        nargs=1,
        metavar="X,Y,Z",
        help="Cartesian direction for the LO-TO term (born; 3-D only, no default because the "
        "q -> 0 limit is direction dependent)",
    )
    parser.add_argument(
        "--molden-basis",
        default="sto-6g",
        help='molden basis: "sto-6g" (default, readable everywhere) or "sto" (exact Slater, '
        "s/p only)",
    )
    parser.add_argument(
        "--output",
        "-o",
        # The Rust CLI parsed only `--opt-output` before 0.2.3 and now takes all three spellings;
        # accepting the old name here too keeps the two sets of accepted flags equal, which
        # `test_the_two_command_lines_offer_the_same_flags` requires.
        "--opt-output",
        default=None,
        help="write the result here (the relaxed geometry, or the molden file) instead of "
        "standard output",
    )
    parser.add_argument(
        "--dandc",
        type=float,
        metavar="BUFFER",
        help="divide-and-conquer SCF with this buffer radius in Angstrom; "
        "do not go below 7, where the accuracy falls off a cliff",
    )
    parser.add_argument(
        "--field",
        metavar="FX,FY,FZ",
        help="uniform external electric field in volts/Angstrom, in MOPAC's FIELD= convention "
        "(the potential gradient, so the interaction energy is +F.mu)",
    )
    parser.add_argument(
        "--dipole-origin",
        choices=("coordinates", "com", "charge"),
        default=None,
        help="origin for the point-charge dipole of a charged species (default: com, as MOPAC)",
    )
    parser.add_argument(
        "--ir",
        action="store_true",
        help="with `frequencies`, also report IR intensities and dipole derivatives",
    )
    parser.add_argument(
        "--projection",
        choices=("rigid", "translations", "none"),
        default=None,
        help="what to project out of the spectrum before diagonalizing (default: rigid, which "
        "removes the translations and the rotations the system actually has). `none` returns the "
        "raw 3N set",
    )
    # On every PyO3 signature and on ``PM7(...)`` since 0.2.0, and on neither command line until
    # 0.2.3. A periodic cell that stalls is exactly the case someone reaches for a command line
    # for, and the two knobs that diagnose it were the ones a command line could not set.
    parser.add_argument(
        "--scf-tolerance",
        type=float,
        metavar="TOL",
        help="SCF density-convergence threshold (default: the built-in one)",
    )
    parser.add_argument(
        "--max-scf",
        type=int,
        metavar="N",
        help="maximum SCF iterations",
    )
    # A private constant through 0.2.2, which meant a small-gap cell whose orbital response needed
    # more than 100 applications could only be told to use a different mode.
    parser.add_argument(
        "--cphf-max-iterations",
        type=int,
        metavar="N",
        help="CPHF orbital-response budget (default 100); raise it when a small-gap cell "
             "reports an unconverged response",
    )
    parser.add_argument(
        "--stability",
        choices=("off", "check", "follow"),
        help="is the converged SCF solution a minimum? `check` reports the orbital-Hessian "
             "curvature; `follow` also re-converges from a rotated guess when it is negative "
             "and keeps the lower of the two. Off by default: the check costs about a CPHF solve",
    )
    parser.add_argument(
        "--no-diis",
        action="store_true",
        help="turn off DIIS acceleration (the first thing to try when an SCF oscillates)",
    )
    parser.add_argument(
        "--opt-cell",
        action="store_true",
        help="with `optimize`, relax the lattice as well as the atoms",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        metavar="EV_PER_BOHR",
        help="optimizer force convergence (default 1e-3)",
    )
    parser.add_argument(
        "--stress-tol",
        type=float,
        metavar="EV_PER_BOHR3",
        help="optimizer stress convergence, with --opt-cell",
    )
    parser.add_argument(
        "--opt-max-iter",
        type=int,
        metavar="N",
        help="optimizer iteration limit (default 200)",
    )
    parser.add_argument(
        "--stability-every",
        type=int,
        metavar="N",
        help="re-check SCF stability every Nth optimizer step (default 0, never)",
    )
    parser.add_argument(
        "--exchange-cutoff",
        nargs=2,
        type=float,
        metavar=("INNER", "OUTER"),
        help="smooth long-range-exchange cutoff in Bohr for the analytic Hessian; omitting it "
        "keeps the Hessian bit-identical",
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        metavar="ANGSTROM",
        help="thickness of the material for `dielectric` on a 2-D cell. Required: a supercell "
        "says where the atoms are, not where the material ends",
    )
    parser.add_argument(
        "--wire-cross-section",
        type=float,
        metavar="ANGSTROM2",
        help="cross-sectional area of the material for `dielectric` on a 1-D cell",
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    return parser


def _triples(values, flag):
    """Parse a list of ``x,y,z`` strings into a list of three-element lists."""
    out = []
    for text in values:
        parts = [p for p in text.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise SystemExit(f"{flag} wants `x,y,z`, got {text!r}")
        out.append([float(p) for p in parts])
    return out


def main(argv=None) -> int:
    """Run one mode, reporting a refusal the way a command line should.

    Every validation failure this module raises itself is a one-line ``SystemExit``. A refusal
    from the extension module -- an unusable cell, an unsupported combination, an SCF that will
    not converge -- used to escape as a raw ``ValueError`` traceback, which is a different user
    experience for the same class of problem and buries the sentence that says what to fix. So did
    a missing structure file, which is worse: ``python -m pm7_rs energy nope.xyz`` answered with
    twenty frames of interpreter, and the traceback was the only thing that named the file.

    ``OSError`` is caught for that reason and not to swallow I/O problems generally: at this level
    there is nothing to recover, and a command line reporting "no such file" is more useful than
    the same fact spelled as a stack. The Rust CLI has always printed ``pm7-rs: <message>`` and
    exited 1; both now leave through the same door.
    """
    try:
        return _main(argv)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"pm7-rs: {error}") from None


def _main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    symbols, positions, (cell, pbc) = _read_xyz(args.structure)
    numbers = [_z(s) for s in symbols]

    if args.cell:
        values = [float(x) for x in args.cell.replace(",", " ").split()]
        if len(values) not in (3, 6, 9) or not values:
            raise SystemExit("--cell wants 3, 6 or 9 numbers (1-D, 2-D or 3-D)")
        # Three or six numbers name a chain or a slab, which is what the Rust CLI has always
        # accepted and this one refused. The native layer takes a full 3x3 plus `pbc`, so the
        # missing rows are filled with placeholders that `pbc` then marks non-periodic; they are
        # never used as lattice vectors, only as slots the reordering can skip.
        given = len(values) // 3
        rows = [values[3 * k : 3 * k + 3] for k in range(given)]
        placeholders = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        while len(rows) < 3:
            rows.append(placeholders[len(rows)])
        cell = np.asarray(rows, dtype=float)
        if given < 3:
            pbc = [k < given for k in range(3)]
    if args.pbc:
        if args.cell and len(args.cell.replace(",", " ").split()) != 9:
            # The short forms of --cell have already chosen a pattern, and the rows --pbc would be
            # choosing among do not all exist. Same refusal as the Rust CLI's.
            raise SystemExit(
                "--pbc chooses among three lattice vectors, so --cell needs all nine numbers"
            )
        pbc = [c.upper() in ("T", "1", "Y") for c in args.pbc if not c.isspace()]
        if len(pbc) != 3:
            raise SystemExit('--pbc needs three flags, e.g. "TTF"')

    common = dict(
        charge=args.charge,
        multiplicity=args.multiplicity,
        method=args.method,
        reference=args.reference,
    )
    field = None  # noqa: F841 - assigned just below
    if args.field:
        parts = [p for p in args.field.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise SystemExit("--field wants `fx,fy,fz` in volts/Angstrom")
        field = [float(p) for p in parts]
    common["field"] = field
    common["dipole_origin"] = args.dipole_origin
    # These reach every native entry point, molecular and periodic alike, which is why they go in
    # `common` and not in the periodic block below.
    if args.scf_tolerance is not None:
        common["scf_tolerance"] = args.scf_tolerance
    if args.max_scf is not None:
        common["max_scf"] = args.max_scf
    if args.cphf_max_iterations is not None:
        if args.cphf_max_iterations < 1:
            raise SystemExit(
                "--cphf-max-iterations must be at least 1: the orbital response is solved "
                "iteratively, and a budget of zero asks for no iterations at all rather than "
                "for a cheap answer"
            )
        common["cphf_max_iterations"] = args.cphf_max_iterations
    if args.stability is not None:
        common["stability"] = args.stability
    if args.no_diis:
        common["use_diis"] = False
    if args.exchange_cutoff:
        common["exchange_cutoff"] = (
            float(args.exchange_cutoff[0]),
            float(args.exchange_cutoff[1]),
        )
    periodic = {}
    if cell is not None:
        periodic["cell"] = cell
        if pbc is not None:
            periodic["pbc"] = pbc
        if args.kpoints:
            periodic["kpoints"] = args.kpoints
        if args.kshift:
            periodic["kpoint_shift"] = args.kshift
        if args.smearing:
            periodic["smearing"] = (args.smearing[0], float(args.smearing[1]))
        if args.pbc_mode:
            periodic["pbc_mode"] = args.pbc_mode
    elif args.kpoints:
        raise SystemExit("--kpoints needs a cell")

    # `born` and `dfpt` belong on this list and were missing from it. Without a cell they reached
    # `np.asarray(None, dtype=float)`, which is a 0-d array of `nan` whose `.tolist()` is a bare
    # float, and the extension then said `argument 'cell': 'float' object cannot be converted to
    # 'Sequence'` from inside a twenty-frame traceback. True, unhelpful, and a crash rather than a
    # refusal. The Rust CLI has refused both by name all along (`require_cell`); the full
    # `pytest -m matrix` tier is what found the divergence.
    if args.mode in ("phonons", "bands", "born", "dfpt") and cell is None:
        raise SystemExit(
            f"{args.mode} needs a periodic cell: use an extended-XYZ file with a "
            'Lattice="..." key, or pass --cell'
        )

    # Divide and conquer is a different driver rather than a flag on the SCF.
    if args.dandc is not None:
        if args.mode not in ("energy", "charges", "gradient", "forces"):
            raise SystemExit(f"--dandc does not apply to `{args.mode}`")
        dandc_periodic = {k: v for k, v in periodic.items() if k in ("cell", "pbc", "pbc_mode")}
        out = native.divide_and_conquer(
            numbers, positions, buffer=args.dandc, **common, **dandc_periodic
        )
    elif args.mode in ("energy", "charges"):
        # `dict(...)` because `single_point` is typed as a `TypedDict`: it *is* a dict at run
        # time, but a checker will not let one flow into a variable the branch above already
        # inferred as a plain dict.
        out = dict(native.single_point(numbers, positions, **common, **periodic))
    elif args.mode == "gradient":
        out = native.gradient(numbers, positions, **common, **periodic)
    elif args.mode == "forces":
        out = native.forces(numbers, positions, **common, **periodic)
    elif args.mode == "stress":
        if cell is None:
            raise SystemExit("stress needs a periodic cell")
        out = native.stress(numbers, positions, **common, **periodic)
    elif args.mode == "optimize":
        out = native.optimize(
            numbers,
            positions,
            relax_cell=args.opt_cell,
            gtol=args.gtol,
            stress_tol=args.stress_tol,
            max_iter=args.opt_max_iter,
            stability_every=args.stability_every,
            **common,
            **periodic,
        )
        if args.output:
            # The Rust CLI's `--opt-output` has always written the relaxed geometry; here the same
            # flag was accepted, exited 0, and wrote nothing -- the worst shape a flag can have,
            # because the user waits for a file that was never going to arrive and the exit status
            # says the run worked. The Python CLI also dropped the cell when printing, so a
            # periodic optimization could not round-trip at all; the extended-XYZ header fixes
            # that too.
            _write_xyz(
                args.output,
                symbols,
                out["positions_angstrom"],
                out.get("cell_angstrom"),
                out.get("pbc"),
                title="pm7-rs optimized",
            )
            print(f"wrote {args.output}")
    elif args.mode == "phonons":
        qpoints = _triples(args.qpoints, "--qpoints") if args.qpoints else [[0.0, 0.0, 0.0]]
        out = native.phonons(
            numbers,
            positions,
            cell,
            qpoints,
            # `(1, 1, 1)` when the flag is absent, not `None`. `native.phonons` defaults to the
            # zone centre, but passing `None` **overrides** that default rather than falling back
            # to it, so `python -m pm7_rs phonons cell.xyz` without `--supercell` died on
            # "supercell must have three entries" — the Rust CLI has defaulted to 1x1x1 all along.
            # Every existing test passed the flag, so the default path was never run.
            supercell=tuple(args.supercell) if args.supercell else (1, 1, 1),
            # On unless explicitly turned off. `--acoustic-sum-rule` still names the default, which
            # is harmless and keeps a script written against 0.2.2 working.
            acoustic_sum_rule=not args.no_acoustic_sum_rule,
            lo_to_direction=_triples(args.lo_to, "--lo-to")[0] if args.lo_to else None,
            **common,
            **{k: v for k, v in periodic.items() if k in ("pbc",)},
        )
    elif args.mode == "dfpt":
        qpoints = _triples(args.qpoints, "--qpoints") if args.qpoints else [[0.0, 0.0, 0.0]]
        out = native.dfpt(
            numbers,
            positions,
            cell,
            qpoints,
            lo_to_direction=_triples(args.lo_to, "--lo-to")[0] if args.lo_to else None,
            **common,
            **{k: v for k, v in periodic.items() if k != "cell"},
        )
        # The force constants are `nq * (2 * 3N * 3N)` floats and nobody reads them off a
        # terminal; the frequencies are the answer. `--json` still carries everything.
        if not args.json:
            out = {k: v for k, v in out.items() if k != "force_constants_ev_per_bohr2"}
    elif args.mode == "born":
        out = native.born_charges(
            numbers,
            positions,
            cell,
            lo_to_direction=_triples(args.lo_to, "--lo-to")[0] if args.lo_to else None,
            **{k: v for k, v in common.items() if k not in ("field", "dipole_origin")},
            **{k: v for k, v in periodic.items() if k != "cell"},
        )
    elif args.mode == "molden":
        text = native.molden(
            numbers,
            positions,
            basis=args.molden_basis,
            **{
                k: v
                for k, v in common.items()
                if k in ("charge", "multiplicity", "method", "reference", "field", "dipole_origin")
            },
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
            print(f"wrote {args.output}")
        else:
            print(text, end="")
        return 0
    elif args.mode == "orbitals":
        out = native.orbitals(numbers, positions, **common, **periodic)
    elif args.mode == "bands":
        if not args.qpoints:
            raise SystemExit("bands needs a k path: --qpoints kx,ky,kz kx,ky,kz ...")
        out = native.band_structure(
            numbers,
            positions,
            cell,
            _triples(args.qpoints, "--qpoints"),
            **common,
            **{k: v for k, v in periodic.items() if k != "cell"},
        )
    elif args.mode == "hessian":
        out = native.hessian(numbers, positions, **common, **periodic)
    elif args.mode == "dielectric":
        if cell is None:
            raise SystemExit("dielectric needs a periodic cell")
        if args.slab_thickness is None and args.wire_cross_section is None:
            raise SystemExit(
                "dielectric needs the material's extent: --slab-thickness A for a layer, or "
                "--wire-cross-section A^2 for a chain. A supercell says where the atoms are, "
                "not where the material ends, so there is nothing to infer it from. Pass "
                "neither only for a 3-D cell, where the cell volume *is* the extent; use "
                "`born` for that."
            )
        # The library takes the extent in **Bohr**; the flag is documented in Angstrom, as the
        # Rust CLI's is. Passing the raw value made the two command lines disagree by a factor of
        # 1.889 on a flag whose metavar says ANGSTROM -- and the disagreement was visible in the
        # two outputs side by side (6.292788 Bohr against 3.33) without being noticed.
        a0 = 1.8897261254578281
        thickness = None if args.slab_thickness is None else args.slab_thickness * a0
        section = None if args.wire_cross_section is None else args.wire_cross_section * a0 * a0
        out = native.dielectric_with_extent(
            numbers,
            positions,
            cell,
            slab_thickness=thickness,
            wire_cross_section=section,
            **{k: v for k, v in common.items() if k not in ("field", "dipole_origin")},
            **{k: v for k, v in periodic.items() if k != "cell"},
        )
    else:
        # The periodic keywords are forwarded here too. They were not until v0.2.2, so
        # `python -m pm7_rs frequencies crystal.xyz` answered with the frequencies of those atoms
        # **as an isolated molecule** — a two-atom diamond cell came back as a C2 diatomic at
        # -772, 654, 654 cm^-1 where the crystal's zone centre is 0, 0, 0, 1248, 1248, 1248. The
        # library layers were fixed first and this one was missed, because no test ran this mode
        # on a cell; `test_cli.py` now compares it against `phonons` on the same file.
        projection = {"projection": args.projection} if args.projection else {}
        if args.ir:
            out = native.vibrations(
                numbers, positions, ir=True, modes=True, **projection, **common, **periodic
            )
        else:
            # Every keyword is forwarded, `field` included. Calling positionally and dropping the
            # rest is how `--field frequencies` used to return the field-free spectrum without
            # saying so.
            out = native.frequencies(numbers, positions, **projection, **common, **periodic)

    if args.json:
        print(json.dumps(_jsonable(out), indent=2))
        return 0
    _print_human(args.mode, symbols, out)
    return 0


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _print_q_frequencies(out):
    """One block per q: the frequencies, and the LO-TO split beside them when it exists.

    `frequencies_cm_lo_to` is `None` for a q away from the zone centre -- the non-analytic
    term is a `q -> 0` limit and does not belong there -- so those q points print one column.
    """
    split = out.get("frequencies_cm_lo_to")
    for index, (q, frequencies) in enumerate(zip(out["qpoints"], out["frequencies_cm"])):
        print(f"q = ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f})")
        here = None if split is None else split[index]
        if here is None:
            for f in frequencies:
                print(f"  {f:12.4f} cm^-1")
        else:
            print(f"  {'cm^-1':>12}  {'with LO-TO':>16}")
            for f, lo in zip(frequencies, here):
                print(f"  {f:12.4f}  {lo:16.4f}")

def _print_human(mode, symbols, out):
    if "energy_ev" in out:
        print(f"total energy:       {out['energy_ev']:.12f} eV")
    if "heat_of_formation_kcal" in out:
        print(f"heat of formation:  {out['heat_of_formation_kcal']:.12f} kcal/mol")
    if out.get("n_kpoints") is not None:
        print(f"k points:           {out['n_kpoints']}")
    # `bands` prints its own Fermi level next to the path it belongs to.
    if out.get("fermi_ev") is not None and mode != "bands":
        print(f"Fermi level:        {out['fermi_ev']:.6f} eV")
    if mode == "charges":
        # `--dandc` returns no Mulliken charges — the assembled density is sparse and the
        # subsystem populations are not summed into per-atom charges — so this used to end in
        # `KeyError: 'charges'` after printing a perfectly good energy. A missing property is a
        # thing to say, not a thing to index.
        charges = out.get("charges")
        if charges is None:
            print(
                "charges: not available from the divide-and-conquer driver; drop --dandc for "
                "Mulliken charges"
            )
        else:
            for symbol, q in zip(symbols, charges):
                print(f"{symbol:<3} {q:+.8f}")
    if mode in ("gradient", "forces"):
        # `--dandc` sends every mode it applies to through `native.divide_and_conquer`, whose
        # result carries `forces_ev_per_angstrom` and **not** `gradient_ev_per_angstrom`, so
        # `gradient --dandc` ended in `KeyError` after a good energy had already been printed.
        #
        # The gradient is the negated force, so it is derived rather than refused: refusing would
        # be withholding a number the run already computed, and the sign is the only thing between
        # them.
        key = "gradient_ev_per_angstrom" if mode == "gradient" else "forces_ev_per_angstrom"
        vectors = out.get(key)
        if vectors is None and mode == "gradient" and "forces_ev_per_angstrom" in out:
            vectors = [[-c for c in v] for v in out["forces_ev_per_angstrom"]]
        if vectors is None:
            print(f"{mode}: not reported by this driver; drop --dandc for per-atom {mode}")
        else:
            for symbol, v in zip(symbols, vectors):
                print(f"{symbol:<3} {v[0]:+.9f} {v[1]:+.9f} {v[2]:+.9f}")
    if mode == "stress":
        xx, yy, zz, yz, xz, xy = out["stress_voigt"]
        print("stress (eV/A^3, Voigt xx yy zz yz xz xy):")
        print(f"  {xx:+.9f} {yy:+.9f} {zz:+.9f} {yz:+.9f} {xz:+.9f} {xy:+.9f}")
        if "pressure_gpa" in out:
            print(f"pressure:           {out['pressure_gpa']:.6f} GPa")
    if mode == "optimize":
        print(f"converged: {out['converged']} after {out['iterations']} iterations")
        for symbol, p in zip(symbols, out["positions_angstrom"]):
            print(f"{symbol:<3} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}")
        # The cell was dropped here, so a periodic optimization could not round-trip: the printed
        # coordinates say nothing about what they repeat in.
        if out.get("cell_angstrom") is not None:
            print("cell (Angstrom):")
            for row in out["cell_angstrom"]:
                print(f"    {row[0]:.9f} {row[1]:.9f} {row[2]:.9f}")
    if mode == "frequencies":
        intensities = out.get("ir_intensities_km_per_mol")
        if intensities is None:
            for f in out["frequencies_cm"]:
                print(f"{f:.6f}")
        else:
            print(f"{'frequency (cm^-1)':>18}  {'IR (km/mol)':>12}  {'DIPT (D/A)':>11}")
            dipt = out.get("mopac_dipt") or [float('nan')] * len(intensities)
            for f, intensity, d in zip(out["frequencies_cm"], intensities, dipt):
                print(f"{f:18.4f}  {intensity:12.4f}  {d:11.5f}")
    if mode == "orbitals":
        homo, lumo = out.get("homo_ev"), out.get("lumo_ev")
        if homo is not None and lumo is not None:
            print(f"HOMO / LUMO:        {homo:.6f} / {lumo:.6f} eV  (gap {out['gap_ev']:.6f} eV)")
        print(f"orbitals reported at: {out['orbital_source']}")
        print(f"{'index':>5} {'occ':>6} {'energy (eV)':>14}")
        occupations = out["occupations"]
        for index, (energy, occupation) in enumerate(zip(out["mo_energies_ev"], occupations)):
            marker = ""
            if index + 1 == out["n_occ"]:
                marker = "  <-- HOMO"
            elif index == out["n_occ"]:
                marker = "  <-- LUMO"
            print(f"{index:5d} {occupation:6.3f} {energy:14.6f}{marker}")
        if out.get("mo_energies_beta_ev") is not None:
            print("beta:")
            for index, energy in enumerate(out["mo_energies_beta_ev"]):
                print(f"{index:5d} {'':6} {energy:14.6f}")
    if mode == "phonons":
        supercell = out["supercell"]
        print(f"supercell:          {supercell[0]}x{supercell[1]}x{supercell[2]}")
        # Reported whether or not it was projected out: projecting hides how well the force
        # constants respected translational invariance in the first place.
        print(f"acoustic residual:  {out['acoustic_residual_ev_per_bohr2']:.6e} eV/Bohr^2")
        _print_q_frequencies(out)
    if mode == "bands":
        print(f"Fermi level:        {out['fermi_ev']:.6f} eV")
        for k, energies in zip(out["kpath"], out["energies_ev"]):
            print(f"k = ({k[0]:.6f}, {k[1]:.6f}, {k[2]:.6f})")
            for e in energies:
                print(f"  {e:14.8f} eV")
    if mode == "dfpt":
        # `dfpt` and `born` share none of the generic keys above — no `energy_ev`, no
        # `heat_of_formation_kcal`, no `n_kpoints` — so without these two branches
        # `python -m pm7_rs dfpt cell.xyz` exited 0 having printed nothing at all. The Rust CLI
        # printed both all along; no test ran either mode through this one.
        print(f"response iterations: {out['iterations']}  (converged: {out['converged']})")
        print(f"response residual:   {out['residual']:.6e}")
        _print_q_frequencies(out)
    if mode == "born":
        print(f"response iterations: {out['iterations']}  (converged: {out['converged']})")
        print(f"acoustic residual:   {out['acoustic_residual']:.6e} e")
        print("Born effective charges Z* (rows: field a, columns: displacement b)")
        for symbol, z in zip(symbols, out["born_charges"]):
            print(f"{symbol:<3}")
            for row in z:
                print(f"     {row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}")
        volume = out.get("volume_bohr3") or 0.0
        if volume > 0.0:
            print("dielectric tensor (electronic, clamped ion)")
            for row in out["dielectric"]:
                print(f"     {row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}")
        else:
            print("dielectric tensor: not defined without a 3-D cell; raw d(mu)/d(f):")
            for row in out["polarizability"]:
                print(f"     {row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}")
        if out.get("lo_to_force_constants_ev_per_bohr2") is not None:
            direction = out["lo_to_direction"]
            print(
                "LO-TO force constants along "
                f"({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}) "
                "in eV/Bohr^2 (real part)"
            )
            # `complex_rows` hands back `(real, imag)` as two whole matrices, not a matrix of
            # pairs — numpy builds a complex array from that in one step. The real part is what
            # is worth reading here; `D^NA` is real for a real `q̂`.
            real, _imag = out["lo_to_force_constants_ev_per_bohr2"]
            for row in real:
                print("     " + " ".join(f"{value:+.6f}" for value in row))
    if mode == "dielectric":
        slab = out["extent_convention"] == "slab_thickness"
        kind = "slab thickness" if slab else "wire cross-section"
        unit = "Bohr" if slab else "Bohr^2"
        # The cell's own periodic measure: an **area** for a slab, a **length** for a wire. The
        # units differ per dimensionality, so they are printed rather than assumed.
        measure_unit = "Bohr^2" if slab else "Bohr"
        print(f"{kind}: {out['extent']:.6f} {unit}   (assigned, not derived)")
        print(f"cell measure:       {out['measure_bohr']:.6f} {measure_unit}")
        axis = out["axis"]
        print(f"axis:               ({axis[0]:+.6f}, {axis[1]:+.6f}, {axis[2]:+.6f})")
        print("dielectric tensor (electronic, clamped ion, depolarization corrected)")
        for row in out["dielectric"]:
            print(f"     {row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}")
        # The two numbers that do *not* depend on the thickness you assumed. If a reader takes
        # one number away from this mode it should be one of these, not a tensor element whose
        # value moves with a guess.
        print(f"sheet parallel      (eps_par - 1) d: {out['sheet_parallel_bohr']:.6f} Bohr")
        print(f"sheet perpendicular (1 - 1/eps) d:   {out['sheet_perpendicular_bohr']:.6f} Bohr")
        print(f"axis mixing (0 = the axis is an eigenvector): {out['axis_mixing']:.3e}")

    if mode == "hessian":
        rows = out["hessian_ev_per_angstrom2"]
        print(f"Hessian, {len(rows)}x{len(rows[0])} eV/Angstrom^2")
        for row in rows:
            print("     " + " ".join(f"{value:+.6f}" for value in row))

    if "subsystems" in out:
        print(f"subsystems:         {out['subsystems']} (divide and conquer)")


if __name__ == "__main__":
    sys.exit(main())
