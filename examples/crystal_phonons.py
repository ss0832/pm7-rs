# SPDX-License-Identifier: GPL-3.0-or-later
"""Zone-centre phonons across nine structure types, plus a coordination polymer.

    python examples/crystal_phonons.py

Green tests are not the same as right numbers, and a phonon spectrum has a property no unit test
asserts for it: **the degeneracies its space group requires**. A cubic `Td` or `Oh` cell must put
its zone-centre optical modes into triplets; a wurtzite cell into singlets and doublets; and every
crystal must put three modes at exactly zero. Those are statements about the Hamiltonian's
symmetry, and they fail loudly when a two-centre rotation, a lattice sum or an image list is wrong
in a way that no energy comparison would notice.

Structures come from ASE (`bulk`, `spacegroup.crystal`) rather than hand-typed coordinates: a
mistyped Wyckoff position produces a *plausible* spectrum with the wrong degeneracies, which is
exactly the failure this script exists to detect and would then be reporting on itself.

The acoustic sum rule is on by default from 0.2.3, so the three acoustic modes should come back at
0 rather than merely near it.
"""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    from ase.build import bulk
    from ase.spacegroup import crystal
except ImportError:  # pragma: no cover
    sys.exit("this script needs ASE: python -m pip install ase")

import pm7_rs.native as native

# `(label, builder, k mesh, expected)`, where `expected` is one of:
#
# * a list of multiplicities — what the space group requires of the *optical* modes, in any order;
# * `None` — the structure's irreps were not worked out, so only the acoustic modes and the general
#   sanity of the spectrum are checked;
# * a **string** — a reason no degeneracy can be required of this cell, printed with the result.
#   That is a different thing from `None` and worth distinguishing: it means the answer PM7 gives
#   is not one the symmetry argument applies to.
STRUCTURES = [
    (
        "diamond (Fd-3m, cubic)",
        lambda: bulk("C", "diamond", a=3.567),
        3,
        [3],
    ),
    (
        # **4x4x4, not 3x3x3, and the mesh is the physics here.** PM7 gives ZnS a gap that depends
        # strongly on the sampling: 0.000 eV at 3x3x3 and 2.575 eV at 4x4x4. On the coarse mesh the
        # cell is spuriously *metallic*, and a zone-centre response of a partially occupied cell is
        # incomplete -- the q = 0 Fermi-level shift is not implemented -- so `dfpt` refuses it. It
        # used to answer, with a smeared fill covering a third of a state, and the degeneracies
        # came out right, which is exactly the kind of plausible-looking incomplete number this
        # script exists to stop trusting.
        "zinc blende ZnS (F-43m)",
        lambda: bulk("ZnS", "zincblende", a=5.41),
        4,
        [3],
    ),
    (
        "rocksalt NaCl (Fm-3m)",
        lambda: bulk("NaCl", "rocksalt", a=5.64),
        3,
        [3],
    ),
    (
        "fluorite CaF2 (Fm-3m)",
        lambda: crystal(
            ["Ca", "F"],
            [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            spacegroup=225,
            cellpar=[5.463, 5.463, 5.463, 90, 90, 90],
            primitive_cell=True,
        ),
        4,
        # **PM7 has no usable gap for fluorite, and that is the finding.** Real CaF2 is an
        # insulator with a ~12 eV gap; PM7 gives it 0.012 eV at a 3x3x3 mesh, 0.092 at 4x4x4,
        # 0.054 at 5x5x5 and 0.032 at 6x6x6, and every smeared fill at 0.02, 0.05 or 0.10 eV comes
        # out genuinely partially occupied -- a hundredth to a third of a state on either side of
        # E_F. So the zone-centre response is refused: a q = 0 perturbation of a partially occupied
        # cell moves the Fermi level, and that term is not implemented.
        #
        # This entry used to require [3, 3] and get it, from a 3x3x3 run smeared at 0.10 eV whose
        # occupations were a third of a state from integral. The degeneracies were right and the
        # calculation was incomplete, which is the most convincing way for a number to be wrong.
        # Nothing is required of this cell now, and the reason is printed with it.
        "PM7 leaves fluorite gapless (<= 0.09 eV at every mesh tried), so its zone-centre "
        "response is not well defined and no degeneracy can be required of it",
    ),
    (
        "perovskite SrTiO3 (Pm-3m)",
        lambda: crystal(
            ["Sr", "Ti", "O"],
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)],
            spacegroup=221,
            cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
            primitive_cell=True,
        ),
        3,
        [3, 3, 3, 3],
    ),
    (
        "wurtzite ZnO (P6_3mc)",
        lambda: bulk("ZnO", "wurtzite", a=3.25, c=5.207),
        3,
        # 2A1 + 2B1 + 2E1 + 2E2 total; one A1 and one E1 are acoustic, leaving
        # A1 + 2B1 + E1 + 2E2 optical = three singlets and three doublets.
        [1, 1, 1, 2, 2, 2],
    ),
    (
        "rutile TiO2 (P4_2/mnm)",
        lambda: crystal(
            ["Ti", "O"],
            [(0.0, 0.0, 0.0), (0.3053, 0.3053, 0.0)],
            spacegroup=136,
            cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
            primitive_cell=True,
        ),
        3,
        None,
    ),
    (
        "spinel MgAl2O4 (Fd-3m)",
        # `setting=2` is not optional: ASE defaults to origin choice 1, where (1/8,1/8,1/8) is not
        # the 8a site, and the cell comes back as Al2Mg4O8 — the cations swapped, with the right
        # atom count and the wrong compound. That is precisely the kind of quiet error this script
        # exists to catch, so it is worth catching in the script itself.
        lambda: crystal(
            ["Mg", "Al", "O"],
            [(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (0.2624, 0.2624, 0.2624)],
            spacegroup=227,
            setting=2,
            cellpar=[8.083, 8.083, 8.083, 90, 90, 90],
            primitive_cell=True,
        ),
        2,
        None,
    ),
    (
        "layered rocksalt LiCoO2 (R-3m)",
        lambda: crystal(
            ["Li", "Co", "O"],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.2395)],
            spacegroup=166,
            cellpar=[2.815, 2.815, 14.05, 90, 90, 120],
            primitive_cell=True,
        ),
        3,
        None,
    ),
    (
        "Zn(CN)2 coordination polymer (P-43m)",
        lambda: crystal(
            ["Zn", "Zn", "C", "N"],
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.193, 0.193, 0.193), (0.305, 0.305, 0.305)],
            spacegroup=215,
            cellpar=[5.9227, 5.9227, 5.9227, 90, 90, 90],
            primitive_cell=True,
        ),
        2,
        None,
    ),
]

# Two frequencies closer than this are one degenerate set. Chosen against the *gaps*: on these
# cells the within-set spread is below 0.05 cm^-1 and the nearest genuine separation is tens, so
# nothing sits near the line. Reported alongside the result so the reader can judge it.
DEGENERACY_TOLERANCE_CM = 1.0


def multiplicities(frequencies, tolerance=DEGENERACY_TOLERANCE_CM):
    """Group a sorted spectrum into degenerate sets and return their sizes."""
    out = []
    for f in frequencies:
        if out and abs(f - out[-1][-1]) <= tolerance:
            out[-1].append(f)
        else:
            out.append([f])
    return [len(g) for g in out], out


def main() -> int:
    problems = []
    for label, build, mesh, expected in STRUCTURES:
        atoms = build()
        numbers = atoms.get_atomic_numbers().tolist()
        positions = atoms.get_positions().tolist()
        cell = atoms.get_cell()[:].tolist()
        n = len(numbers)

        print(f"=== {label} — {atoms.get_chemical_formula()}, "
              f"{n} atoms, {mesh}x{mesh}x{mesh} mesh ===")
        started = time.time()
        # `dfpt`, not `phonons`. `phonons` takes its Brillouin-zone sampling from `supercell` and
        # refuses `kpoints` outright, so a properly sampled zone centre would mean a 3x3x3 repeat
        # — 54 atoms for diamond, 378 for spinel. DFPT solves the response at `q = 0` directly on a
        # k mesh: the same physics for a fraction of the cost, and the right tool when the zone
        # centre is all that is wanted.
        #
        # **Smearing is a fallback, not a default.** An unsmeared aufbau fill is discontinuous in
        # the band energies, and on a cell whose frontier states are nearly degenerate it can
        # settle into a solution that breaks the crystal's own symmetry — which then shows up here
        # as a split triplet. PM7 gives both ZnS and CaF2 an almost-zero gap and both do exactly
        # that. So: run unsmeared first, and only if that fails or produces the wrong degeneracies
        # retry with a Fermi width, reporting which was used.
        #
        # A width is not a free pass, and from 0.2.3 it is not treated as one. If the smeared fill
        # is *genuinely* partial the zone-centre response is refused, because a q = 0 perturbation
        # of a partially occupied cell moves the Fermi level and that term is not implemented. So a
        # rung can fail for two quite different reasons — the SCF would not converge, or it did and
        # the answer would have been incomplete — and both land in `note`.
        out, used = None, "unsmeared"
        note = None
        for attempt, extra in (
            ("unsmeared", {}),
            ("Fermi 0.10 eV", {"smearing": ("fermi", 0.10)}),
            # A narrower rung, because the wide one can leave a *gapped* cell's occupations
            # fractional enough that the zone-centre response is refused. That refusal is right --
            # the q = 0 Fermi-level shift is missing for a partially occupied cell -- and the
            # remedy is a width the gap can dominate, not a coarser check.
            ("Fermi 0.02 eV", {"smearing": ("fermi", 0.02)}),
        ):
            try:
                candidate = native.dfpt(
                    numbers, positions, cell, [[0.0, 0.0, 0.0]],
                    kpoints=(mesh, mesh, mesh),
                    scf_tolerance=1e-9, dfpt_tolerance=1e-10, dfpt_max_iterations=400,
                    **extra,
                )
            except Exception as error:
                note = str(error).strip().splitlines()[0]
                continue
            sizes_try, _ = multiplicities(
                [f for f in sorted(candidate["frequencies_cm"][0])][3:]
            )
            out, used = candidate, attempt
            if not isinstance(expected, list) or sorted(sizes_try) == sorted(expected):
                break
        if out is None:
            print(f"    REFUSED after {time.time() - started:.1f}s: {note[:150]}\n")
            problems.append(f"{label}: {note[:110]}")
            continue

        freqs = sorted(out["frequencies_cm"][0])
        elapsed = time.time() - started

        # The acoustic modes are the three closest to zero, which is **not** the same as the three
        # lowest: a structure away from its minimum has imaginary optical modes below them, and
        # SrTiO3 at the experimental lattice constant has nine of them.
        by_magnitude = sorted(range(len(freqs)), key=lambda i: abs(freqs[i]))
        acoustic_index = set(by_magnitude[:3])
        acoustic = [freqs[i] for i in sorted(acoustic_index)]
        optical = [f for i, f in enumerate(freqs) if i not in acoustic_index]
        worst_acoustic = max(abs(f) for f in acoustic)
        sizes, groups = multiplicities(optical)

        print(f"    {len(freqs)} modes in {elapsed:.1f}s, {used}, "
              f"response converged in {out['iterations']} (residual {out['residual']:.1e})")
        if used != "unsmeared":
            print(f"    (the unsmeared run did not serve: {(note or 'wrong degeneracies')[:110]})")
        print(f"    acoustic: {[round(f, 4) for f in acoustic]}   (worst |f| = {worst_acoustic:.2e})")
        print(f"    optical degeneracies: {sizes}")
        print(f"    optical (one per set): "
              f"{[round(g[0], 1) for g in groups]}")

        if worst_acoustic > 1.0e-2:
            problems.append(
                f"{label}: acoustic modes are not zero (worst {worst_acoustic:.3e} cm^-1)"
            )
        if isinstance(expected, str):
            print(f"    (no degeneracy required: {expected})")
        elif expected is not None and sorted(sizes) != sorted(expected):
            problems.append(
                f"{label}: degeneracies {sorted(sizes)} where the space group requires "
                f"{sorted(expected)}"
            )
            print(f"    ** the space group requires {sorted(expected)} **")
        imaginary = [f for f in optical if f < -1.0]
        if imaginary:
            print(f"    (note: {len(imaginary)} imaginary optical modes, lowest "
                  f"{min(imaginary):.1f} cm^-1 — this geometry is not PM7's minimum)")
        print()

    print("=" * 72)
    if problems:
        print(f"{len(problems)} structure(s) to look at:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("every structure: three acoustic modes at zero, degeneracies as required")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
