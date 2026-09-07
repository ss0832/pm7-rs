# SPDX-License-Identifier: GPL-3.0-or-later
"""Record and re-check the MOPAC oracle baseline.

Two jobs, deliberately in one script:

1. **Oracle** — compare `pm7-rs` against MOPAC v23.2.5 on a fixed case set, with a numeric
   threshold per quantity and a non-zero exit when one is breached. The older scripts print a
   delta column and leave the judgement to whoever ran them, which is right for investigating one
   molecule and useless as a regression gate.
2. **Self-baseline** — record `pm7-rs`'s own numbers so a later change can be diffed against them.
   That is what makes "this optimization is exact" a checkable claim rather than an assertion:
   an exact change moves nothing, and an approximation moves something by a stated amount.

    python tools/oracle/baseline.py --record    # run both sides and write the baseline
    python tools/oracle/baseline.py             # run both sides, compare, exit non-zero on breach

The case set deliberately includes systems that are **large, polar, and charged**. The pre-existing
sweeps are small molecules and diatomics, in which a long-range electrostatic change is invisible:
nothing in them is bigger than the 7 A range past which PM7's feathering makes every two-centre
integral exactly a point charge, so a change to how that regime is summed cannot show up at all.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle  # noqa: E402

WORKDIR = oracle.ROOT / "tools/oracle/lrcases"


def _bond(x: float, y: float, z: float, element: str) -> str:
    return f"{element} {x:.6f} {y:.6f} {z:.6f}"


def alkane(n_carbon: int) -> list[str]:
    """A trans zigzag n-alkane, built from idealized sp3 geometry.

    Large and essentially apolar: it isolates the *size* of a system from its polarity, so a
    change that is fine here but not on `hf_chain` is a long-range electrostatics problem rather
    than a cutoff problem.
    """
    cc, angle = 1.526, math.radians(112.0)
    dx, dz = cc * math.sin(angle / 2.0), cc * math.cos(angle / 2.0)
    atoms, carbons = [], []
    for i in range(n_carbon):
        x, z = i * dx, (i % 2) * dz
        carbons.append((x, 0.0, z))
        atoms.append(_bond(x, 0.0, z, "C"))
    # Two hydrogens per carbon out of the backbone plane, plus one cap at each end along it.
    ch, hy = 1.09, 0.89
    for i, (x, _, z) in enumerate(carbons):
        flip = 1.0 if i % 2 == 0 else -1.0
        atoms.append(_bond(x, hy, z + 0.45 * flip, "H"))
        atoms.append(_bond(x, -hy, z + 0.45 * flip, "H"))
    atoms.append(_bond(carbons[0][0] - ch, 0.0, carbons[0][2], "H"))
    atoms.append(_bond(carbons[-1][0] + ch, 0.0, carbons[-1][2], "H"))
    return atoms


def hf_chain(n_units: int) -> list[str]:
    """`n` aligned HF molecules on a line: a large, strongly polar, non-periodic system.

    Every monomer's dipole points the same way, so the total dipole grows with `n` and the
    monopole-monopole tail is exactly what a truncation would get wrong.
    """
    atoms = []
    for i in range(n_units):
        base = i * 2.80
        atoms.append(_bond(base, 0.0, 0.0, "F"))
        atoms.append(_bond(base + 0.93, 0.0, 0.0, "H"))
    return atoms


def water_wire(n_units: int) -> list[str]:
    """A hydrogen-bonded water chain, slightly kinked.

    Exercises the EH+ hydrogen-bond correction at scale as well as the electrostatics. The kink is
    deliberate: a perfectly collinear donor-H...acceptor arrangement sits on the EH+ dihedral
    singularity documented in `docs/singularities.md`.
    """
    atoms = []
    for i in range(n_units):
        base, lift = i * 2.85, 0.15 * (i % 2)
        atoms.append(_bond(base, lift, 0.0, "O"))
        atoms.append(_bond(base + 0.96, lift + 0.02, 0.0, "H"))
        atoms.append(_bond(base - 0.24, lift + 0.90, 0.30, "H"))
    return atoms


# --- geometry builders ---------------------------------------------------------------------
#
# **These are idealized shapes, not optimized geometries, and that is not a shortcut.** A fidelity
# oracle evaluates *both* programs at the *same* point, so what the point is does not enter the
# comparison -- only that the two agree there. Optimizing every case would cost an hour of MOPAC
# time to change nothing, and would make the case set depend on which program did the optimizing.
#
# What the geometry does have to be is *chemically sane*: bond lengths near equilibrium, so the SCF
# converges and lands in the solution a reader would expect rather than in whichever basin a
# distorted structure happens to reach. That is what `validate_geometry` below enforces, and it was
# not a hypothetical -- three cases in the first draft of this table placed a hydrogen 0.6 A from
# the atom it was meant to be bonded to, because they were assembled by appending hand-typed
# coordinates to a generated fragment. Generators and a checker, not hand-typed coordinates.

Point = tuple[float, float, float, str]

_TETRAHEDRAL = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
_OCTAHEDRAL = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# cos of the tetrahedral angle: the H-C-H angle is 109.47 degrees, so a bond makes 70.53 degrees
# with the axis pointing away from the fourth substituent, and cos(70.53) = 1/3 exactly.
_TETRAHEDRAL_COS = 1.0 / 3.0
_TETRAHEDRAL_SIN = math.sqrt(1.0 - _TETRAHEDRAL_COS**2)


def _lines(points: Sequence[Point]) -> list[str]:
    return [_bond(x, y, z, element) for x, y, z, element in points]


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n)


def _perpendicular(u: tuple[float, float, float]) -> tuple[float, float, float]:
    """Any unit vector orthogonal to `u`, chosen so the cross product is never degenerate."""
    seed = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
    v = (
        u[1] * seed[2] - u[2] * seed[1],
        u[2] * seed[0] - u[0] * seed[2],
        u[0] * seed[1] - u[1] * seed[0],
    )
    return _normalize(v)


def cap(
    centre: tuple[float, float, float],
    neighbour: tuple[float, float, float],
    element: str,
    r: float,
    count: int = 3,
    twist: float = 0.0,
) -> list[Point]:
    """`count` substituents on `centre`, arranged tetrahedrally about the axis *away* from
    `neighbour`.

    This is how every -CH3, -NH3 and -OH in the table is built. Doing it by formula rather than by
    hand is the difference between a methyl group and three hydrogens near a carbon.
    """
    u = _normalize((centre[0] - neighbour[0], centre[1] - neighbour[1], centre[2] - neighbour[2]))
    v = _perpendicular(u)
    w = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    points: list[Point] = []
    for k in range(count):
        phi = math.radians(twist + 360.0 * k / count)
        direction = [
            _TETRAHEDRAL_COS * u[i]
            + _TETRAHEDRAL_SIN * (math.cos(phi) * v[i] + math.sin(phi) * w[i])
            for i in range(3)
        ]
        points.append(
            (centre[0] + r * direction[0], centre[1] + r * direction[1],
             centre[2] + r * direction[2], element)
        )
    return points


def bridge_cap(
    centre: tuple[float, float, float],
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    element: str,
    r: float,
    angle: float = 109.47,
) -> list[Point]:
    """Two substituents on a centre that already has **two** heavy neighbours: a backbone CH2.

    [`cap`] cannot do this and must not be used for it. It arranges substituents about the axis
    away from *one* neighbour, and with `count=2` it puts the first of them at an arbitrary
    azimuth — which, on glycine's alpha carbon, landed a hydrogen 0.38 A from the nitrogen. The
    two hydrogens of a methylene go in the plane that bisects the two heavy bonds and is
    perpendicular to them, which is what this builds: along the bisector, split by `angle` about
    the normal of the three-atom plane.
    """
    to_first = _normalize(tuple(centre[i] - first[i] for i in range(3)))  # type: ignore[arg-type]
    to_second = _normalize(tuple(centre[i] - second[i] for i in range(3)))  # type: ignore[arg-type]
    bisector = _normalize(tuple(to_first[i] + to_second[i] for i in range(3)))  # type: ignore[arg-type]
    normal = _normalize((
        to_first[1] * to_second[2] - to_first[2] * to_second[1],
        to_first[2] * to_second[0] - to_first[0] * to_second[2],
        to_first[0] * to_second[1] - to_first[1] * to_second[0],
    ))
    half = math.radians(angle) / 2.0
    points: list[Point] = []
    for sign in (1.0, -1.0):
        direction = [
            math.cos(half) * bisector[i] + sign * math.sin(half) * normal[i] for i in range(3)
        ]
        points.append((
            centre[0] + r * direction[0], centre[1] + r * direction[1],
            centre[2] + r * direction[2], element,
        ))
    return points


def diatomic(a: str, b: str, r: float) -> list[str]:
    return _lines([(0.0, 0.0, 0.0, a), (r, 0.0, 0.0, b)])


def linear(elements: Sequence[str], bonds: Sequence[float]) -> list[str]:
    """A chain along x: `elements[i+1]` sits `bonds[i]` further along than `elements[i]`."""
    points, x = [(0.0, 0.0, 0.0, elements[0])], 0.0
    for element, step in zip(elements[1:], bonds):
        x += step
        points.append((x, 0.0, 0.0, element))
    return _lines(points)


def bent(centre: str, ligand: str, r: float, angle: float) -> list[str]:
    half = math.radians(angle) / 2.0
    return _lines([
        (0.0, 0.0, 0.0, centre),
        (r * math.sin(half), r * math.cos(half), 0.0, ligand),
        (-r * math.sin(half), r * math.cos(half), 0.0, ligand),
    ])


def trigonal_planar(centre: str, ligand: str, r: float) -> list[str]:
    points = [(0.0, 0.0, 0.0, centre)]
    for k in range(3):
        theta = math.radians(120.0 * k)
        points.append((r * math.cos(theta), r * math.sin(theta), 0.0, ligand))
    return _lines(points)


def pyramidal(centre: str, ligand: str, r: float, angle: float) -> list[str]:
    """`angle` is the ligand-centre-ligand angle: 120 is planar, 109.47 tetrahedral.

    Derived rather than tabulated. Three ligands on a cone of half-angle `p` about the C3 axis sit
    `2 r sin(p) sin(60 deg)` apart, and the same pair is `2 r sin(angle/2)` apart by definition, so
    `sin(p) = sin(angle/2) / sin(60 deg)`.
    """
    half = math.radians(angle) / 2.0
    polar = math.asin(min(1.0, math.sin(half) / math.sin(math.radians(60.0))))
    points = [(0.0, 0.0, 0.0, centre)]
    for k in range(3):
        theta = math.radians(120.0 * k)
        points.append((
            r * math.sin(polar) * math.cos(theta),
            r * math.sin(polar) * math.sin(theta),
            r * math.cos(polar),
            ligand,
        ))
    return _lines(points)


def substituted_methane(ligands: Sequence[tuple[str, float]], centre: str = "C") -> list[str]:
    """A tetrahedral centre with each position given its own element and bond length."""
    points = [(0.0, 0.0, 0.0, centre)]
    for (element, r), (x, y, z) in zip(ligands, _TETRAHEDRAL):
        scale = r / math.sqrt(3.0)
        points.append((scale * x, scale * y, scale * z, element))
    return _lines(points)


def tetrahedral(centre: str, ligand: str, r: float) -> list[str]:
    return substituted_methane([(ligand, r)] * 4, centre)


def octahedral(centre: str, ligand: str, r: float) -> list[str]:
    points = [(0.0, 0.0, 0.0, centre)]
    points += [(r * x, r * y, r * z, ligand) for x, y, z in _OCTAHEDRAL]
    return _lines(points)


def square_planar(centre: str, ligand: str, r: float) -> list[str]:
    points = [(0.0, 0.0, 0.0, centre)]
    points += [(r * x, r * y, 0.0, ligand) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return _lines(points)


def carbonyl(metal: str, m_c: float, c_o: float, shape: str) -> list[str]:
    """A homoleptic metal carbonyl: `M(CO)n`, with every CO pointing radially outward.

    Linear M–C–O is what carbonyls actually do, and it is also what keeps the geometry unambiguous:
    the only free parameters are the two bond lengths.
    """
    directions = {
        "tetrahedral": [
            tuple(c / math.sqrt(3.0) for c in v) for v in _TETRAHEDRAL
        ],
        "octahedral": [tuple(float(c) for c in v) for v in _OCTAHEDRAL],
        # Trigonal bipyramid: three in the xy plane, two on the axis.
        "bipyramidal": [
            (1.0, 0.0, 0.0), (-0.5, 0.866_025_403_784, 0.0), (-0.5, -0.866_025_403_784, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        ],
    }[shape]
    points: list[Point] = [(0.0, 0.0, 0.0, metal)]
    for u in directions:
        points.append((m_c * u[0], m_c * u[1], m_c * u[2], "C"))
        reach = m_c + c_o
        points.append((reach * u[0], reach * u[1], reach * u[2], "O"))
    return _lines(points)


def metallocene(metal: str, height: float, ring_radius: float, c_h: float,
                tilt: float = 0.0) -> list[str]:
    """Two cyclopentadienyl rings on a metal: eclipsed sandwich, or bent when `tilt` is non-zero.

    `tilt` moves the two ring centroids off the axis by that much in `y`, which is what turns a
    ferrocene into a titanocene-dichloride fragment with room for two more ligands below.
    """
    points: list[Point] = [(0.0, 0.0, 0.0, metal)]
    for sign in (1.0, -1.0):
        centre_z = sign * height if tilt == 0.0 else height
        centre_y = 0.0 if tilt == 0.0 else sign * tilt
        for k in range(5):
            theta = 2.0 * math.pi * k / 5.0 + (0.0 if sign > 0 else math.pi / 5.0)
            x = ring_radius * math.cos(theta)
            y = ring_radius * math.sin(theta) + centre_y
            points.append((x, y, centre_z, "C"))
            reach = ring_radius + c_h
            points.append((
                reach * math.cos(theta), reach * math.sin(theta) + centre_y, centre_z, "H",
            ))
    return _lines(points)


def dumbbell(
    a: str, b: str, bond: float, cap_a: tuple[str, float, int], cap_b: tuple[str, float, int],
    twist: float = 60.0,
) -> list[str]:
    """Two bonded heavy atoms, each capped tetrahedrally: ethane, methanol, methylamine, ...

    `cap_a` is `(element, bond length, how many)`. `twist` staggers the second cap against the
    first, which is the conformation these molecules actually adopt.
    """
    first, second = (0.0, 0.0, 0.0), (bond, 0.0, 0.0)
    points: list[Point] = [(*first, a), (*second, b)]
    points += cap(first, second, cap_a[0], cap_a[1], cap_a[2])
    points += cap(second, first, cap_b[0], cap_b[1], cap_b[2], twist=twist)
    return _lines(points)


def _ring_points(elements: Sequence[str], radius: float, hydrogens: Sequence[bool],
                 h_radius: float, height: float = 0.0) -> list[Point]:
    """A regular planar ring, with a hydrogen radially outside each flagged position.

    Regular rather than experimental, per the note above. What has to be right for an aromatic is
    that it is *planar and conjugated*, and a regular polygon is both.
    """
    points, n = [], len(elements)
    for k, element in enumerate(elements):
        theta = 2.0 * math.pi * k / n
        points.append((radius * math.cos(theta), radius * math.sin(theta), height, element))
    for k, carries in enumerate(hydrogens):
        if carries:
            theta = 2.0 * math.pi * k / n
            points.append((h_radius * math.cos(theta), h_radius * math.sin(theta), height, "H"))
    return points


def ring(elements: Sequence[str], radius: float, hydrogens: Sequence[bool],
         h_radius: float) -> list[str]:
    return _lines(_ring_points(elements, radius, hydrogens, h_radius))


def substituted_benzene(tail: Sequence[Point]) -> list[str]:
    """Benzene with position 0's hydrogen replaced by `tail`, given in the same frame."""
    core = _ring_points(["C"] * 6, 1.397, [False, True, True, True, True, True], 2.483)
    return _lines(core + list(tail))


def benzene() -> list[str]:
    return ring(["C"] * 6, 1.397, [True] * 6, 2.483)


def stacked_benzene(separation: float) -> list[str]:
    """Two parallel benzenes, one directly above the other.

    Nothing holds this together but dispersion, so it is the case that moves first if the
    dispersion term or its damping is wrong -- and it is invisible to every covalent case here.
    """
    lower = _ring_points(["C"] * 6, 1.397, [True] * 6, 2.483)
    upper = _ring_points(["C"] * 6, 1.397, [True] * 6, 2.483, height=separation)
    return _lines(lower + upper)


def water_dimer() -> list[str]:
    """The canonical hydrogen bond: one donor O-H pointing at the acceptor's lone pair."""
    return _lines([
        (0.00, 0.00, 0.00, "O"),
        (0.96, 0.00, 0.00, "H"),
        (-0.24, 0.93, 0.00, "H"),
        (2.90, 0.10, 0.00, "O"),
        (3.30, -0.30, 0.75, "H"),
        (3.30, -0.30, -0.75, "H"),
    ])


def formic_acid_dimer() -> list[str]:
    """Two hydrogen bonds at once, in the cyclic arrangement that makes carboxylic acids dimerize."""
    monomer: list[Point] = [
        (0.00, 0.00, 0.00, "C"),
        (0.00, 1.20, 0.00, "O"),
        (1.20, -0.65, 0.00, "O"),
        (1.20, -1.61, 0.00, "H"),
        (-0.94, -0.55, 0.00, "H"),
    ]
    centre_x, centre_y = 1.20, 0.28
    partner: list[Point] = [
        (2 * centre_x - x, 2 * centre_y - y, z, element) for x, y, z, element in monomer
    ]
    return _lines(monomer + partner)


def glycine() -> list[str]:
    """The smallest amino acid: an amine, a carboxyl and an sp3 carbon in one molecule."""
    nitrogen, alpha, carboxyl = (0.00, 0.00, 0.00), (1.47, 0.00, 0.00), (2.04, 1.41, 0.00)
    points: list[Point] = [(*nitrogen, "N"), (*alpha, "C"), (*carboxyl, "C")]
    points += cap(nitrogen, alpha, "H", 1.01, count=2)
    # The alpha carbon has two heavy neighbours, so its hydrogens go perpendicular to them.
    points += bridge_cap(alpha, nitrogen, carboxyl, "H", 1.09)
    points += [(3.25, 1.55, 0.00, "O"), (1.20, 2.44, 0.00, "O"), (1.70, 3.28, 0.00, "H")]
    return _lines(points)


def ethene() -> list[str]:
    return _lines([
        (0.00, 0.67, 0.00, "C"),
        (0.00, -0.67, 0.00, "C"),
        (0.92, 1.23, 0.00, "H"),
        (-0.92, 1.23, 0.00, "H"),
        (0.92, -1.23, 0.00, "H"),
        (-0.92, -1.23, 0.00, "H"),
    ])


def butadiene() -> list[str]:
    """s-trans 1,3-butadiene: alternating bond lengths, so the conjugation is in the geometry."""
    return _lines([
        (0.00, 0.00, 0.00, "C"),
        (1.34, 0.00, 0.00, "C"),
        (2.10, 1.23, 0.00, "C"),
        (3.44, 1.23, 0.00, "C"),
        (-0.55, 0.93, 0.00, "H"),
        (-0.55, -0.93, 0.00, "H"),
        (1.89, -0.93, 0.00, "H"),
        (1.55, 2.16, 0.00, "H"),
        (3.99, 0.30, 0.00, "H"),
        (3.99, 2.16, 0.00, "H"),
    ])


def cubane() -> list[str]:
    """Eight carbons on a cube, each with a hydrogen along the body diagonal.

    Every C-C-C angle is 90 degrees, which is 19.5 from tetrahedral -- deliberately outside the
    range any parameterization was fitted in.
    """
    half, reach = 0.785, 1.415
    points: list[Point] = []
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    points += [(half * x, half * y, half * z, "C") for x, y, z in corners]
    points += [(reach * x, reach * y, reach * z, "H") for x, y, z in corners]
    return _lines(points)


def naphthalene_like() -> list[str]:
    """Two regular hexagons sharing an edge: a genuinely bicyclic conjugated pi system."""
    a = 1.397
    dx, dy = a * 1.5, a * math.sqrt(3.0) / 2.0
    core = [(a, 0.0), (a / 2, dy), (-a / 2, dy), (-a, 0.0), (-a / 2, -dy), (a / 2, -dy)]
    fused = [(x + dx, y + dy) for x, y in core]
    # The two rings share an edge, so two of `fused` are already in `core`. Deduplicate by
    # **distance**, not by tuple equality: the shared pair is generated by two different
    # arithmetic paths and agrees to about 1e-16 rather than exactly, so `p not in core` keeps
    # both and puts two carbons on top of each other.
    ring_atoms = list(core)
    for point in fused:
        if all(math.hypot(point[0] - x, point[1] - y) > 1.0e-6 for x, y in ring_atoms):
            ring_atoms.append(point)
    points: list[Point] = [(x, y, 0.0, "C") for x, y in ring_atoms]
    # A hydrogen radially outward from the bicyclic centroid on every carbon with two ring
    # neighbours; the two shared carbons have three and get none.
    cx = sum(x for x, _ in ring_atoms) / len(ring_atoms)
    cy = sum(y for _, y in ring_atoms) / len(ring_atoms)
    for x, y in ring_atoms:
        neighbours = sum(
            1 for u, v in ring_atoms if 0.1 < math.hypot(x - u, y - v) < 1.5 * a
        )
        if neighbours > 2:
            continue
        ux, uy = _normalize((x - cx, y - cy, 0.0))[:2]
        points.append((x + 1.09 * ux, y + 1.09 * uy, 0.0, "H"))
    return _lines(points)


def validate_geometry(name: str, atoms: Sequence[str]) -> list[str]:
    """Complaints about one case's geometry, or an empty list.

    Two checks, and both earned their place by catching something in this file's first draft:

    * **nothing overlaps** -- no two atoms closer than 0.65 A, which is shorter than the shortest
      bond PM7 has parameters for (H-H at 0.74);
    * **nothing floats away** -- every atom has a neighbour within 3.2 A. A hydrogen appended to a
      generated fragment at hand-typed coordinates lands somewhere plausible-looking and bonded to
      nothing, and the SCF converges anyway, to the energy of a different molecule.

    A deliberately non-bonded case -- the dimers -- still passes the second: 3.2 A is wider than
    any hydrogen bond and wider than the stacked-benzene separation.
    """
    parsed = []
    for line in atoms:
        element, x, y, z = line.split()
        parsed.append((element, float(x), float(y), float(z)))
    if len(parsed) < 2:
        return []

    complaints = []
    for i, (element_i, xi, yi, zi) in enumerate(parsed):
        nearest = math.inf
        for j, (element_j, xj, yj, zj) in enumerate(parsed):
            if i == j:
                continue
            d = math.dist((xi, yi, zi), (xj, yj, zj))
            nearest = min(nearest, d)
            if i < j and d < 0.65:
                complaints.append(
                    f"{name}: {element_i}{i} and {element_j}{j} are {d:.3f} A apart, which is "
                    f"shorter than any bond PM7 is parameterized for"
                )
        if nearest > 3.2:
            complaints.append(
                f"{name}: {element_i}{i} has no neighbour within 3.2 A (nearest {nearest:.3f} A), "
                f"so it is a separate fragment rather than part of the molecule"
            )
    return complaints

# name -> (atoms, charge, multiplicity)
#
# **One hundred cases, and the number is a coverage budget rather than a target.** Thirteen was
# enough to catch the long-range electrostatics regression this file was written for, and not
# enough to catch anything else: it exercised eight elements, one bond order above single, no
# halogen heavier than fluorine, no aromatic ring, no triplet, and nothing held together by
# dispersion. An oracle that only looks where you already looked cannot tell you that you looked in
# the wrong place.
#
# The groups are the axes, and each is here because a defect can hide in it and nowhere else:
#
# * **elements** -- every parameter row is its own chance of a transcription error, and a wrong
#   `alpb`/`xfac` pair shows up only in a molecule containing both partners;
# * **bond order and conjugation** -- the resonance integral and the two-centre two-electron block
#   are probed differently by a triple bond than by a single one;
# * **d shells** -- Si, P, S, Cl and heavier route through the MNDO/d kernel and the one-centre p-d
#   dipole term, which no sp molecule touches;
# * **charge** -- the dipole origin convention and an ion's core-core terms, at both signs;
# * **open shells** -- the UHF path, at doublet and triplet;
# * **hydrogen bonding and dispersion** -- the two post-SCF corrections, invisible to every
#   covalent case in the set;
# * **size** -- past PM7's 7 A feathering, where every two-centre integral is exactly a point
#   charge and a change to how that tail is summed can hide.
CASES: dict[str, tuple[list[str], int, int]] = {
    # -- small neutral molecules: the regime the older sweeps cover, kept so a regression here is
    # distinguishable from one that only appears at size.
    "water": (["O 0.0 0.0 0.0", "H 0.96 0.0 0.0", "H -0.24 0.93 0.0"], 0, 1),
    "ammonia": (
        ["N 0.0 0.0 0.0", "H 0.94 0.0 0.38", "H -0.47 0.81 0.38", "H -0.47 -0.81 0.38"],
        0,
        1,
    ),
    "formaldehyde": (
        ["C 0.0 0.0 0.0", "O 0.0 0.0 1.21", "H 0.94 0.0 -0.54", "H -0.94 0.0 -0.54"],
        0,
        1,
    ),
    "methane": (tetrahedral("C", "H", 1.09), 0, 1),
    "hydrogen": (diatomic("H", "H", 0.74), 0, 1),
    "nitrogen": (diatomic("N", "N", 1.10), 0, 1),
    "carbon_monoxide": (diatomic("C", "O", 1.13), 0, 1),
    "hydrogen_fluoride": (diatomic("H", "F", 0.92), 0, 1),
    "carbon_dioxide": (linear(["O", "C", "O"], [1.16, 1.16]), 0, 1),
    "hydrogen_cyanide": (linear(["H", "C", "N"], [1.07, 1.16]), 0, 1),
    "acetylene": (linear(["H", "C", "C", "H"], [1.06, 1.20, 1.06]), 0, 1),
    "ethene": (ethene(), 0, 1),
    "ethane": (dumbbell("C", "C", 1.53, ("H", 1.09, 3), ("H", 1.09, 3)), 0, 1),
    "methanol": (dumbbell("C", "O", 1.42, ("H", 1.09, 3), ("H", 0.96, 1)), 0, 1),
    "methylamine": (dumbbell("C", "N", 1.47, ("H", 1.09, 3), ("H", 1.01, 2)), 0, 1),
    "hydrogen_peroxide": (dumbbell("O", "O", 1.45, ("H", 0.97, 1), ("H", 0.97, 1), 90.0), 0, 1),
    "hydrazine": (dumbbell("N", "N", 1.45, ("H", 1.02, 2), ("H", 1.02, 2), 90.0), 0, 1),
    "nitrous_oxide": (linear(["N", "N", "O"], [1.13, 1.19]), 0, 1),
    "ozone": (bent("O", "O", 1.28, 117.0), 0, 1),
    "allene": (
        linear(["C", "C", "C"], [1.31, 1.31])
        + _lines([
            (-0.55, 0.93, 0.00, "H"), (-0.55, -0.93, 0.00, "H"),
            (3.17, 0.00, 0.93, "H"), (3.17, 0.00, -0.93, "H"),
        ]),
        0,
        1,
    ),
    # -- second row and heavier main group: Si, P, S, Cl and below route through the MNDO/d kernel,
    # whose one-centre p-d dipole term no sp molecule reaches.
    "hydrogen_sulfide": (["S 0.0 0.0 0.0", "H 1.34 0.0 0.0", "H -0.35 1.29 0.0"], 0, 1),
    "sulfur_dioxide": (["S 0.0 0.0 0.0", "O 1.24 0.0 0.72", "O -1.24 0.0 0.72"], 0, 1),
    "phosphorus_trifluoride": (
        ["P 0.0 0.0 0.0", "F 1.24 0.0 0.51", "F -0.62 1.07 0.51", "F -0.62 -1.07 0.51"],
        0,
        1,
    ),
    "silane": (tetrahedral("Si", "H", 1.48), 0, 1),
    "phosphine": (pyramidal("P", "H", 1.42, 93.5), 0, 1),
    "borane": (trigonal_planar("B", "H", 1.19), 0, 1),
    "boron_trifluoride": (trigonal_planar("B", "F", 1.31), 0, 1),
    "aluminium_trichloride": (trigonal_planar("Al", "Cl", 2.06), 0, 1),
    "silicon_tetrafluoride": (tetrahedral("Si", "F", 1.55), 0, 1),
    "carbon_tetrachloride": (tetrahedral("C", "Cl", 1.77), 0, 1),
    "sulfur_hexafluoride": (octahedral("S", "F", 1.56), 0, 1),
    "carbon_disulfide": (linear(["S", "C", "S"], [1.55, 1.55]), 0, 1),
    "carbonyl_sulfide": (linear(["O", "C", "S"], [1.16, 1.56]), 0, 1),
    "thioformaldehyde": (
        ["C 0.0 0.0 0.0", "S 0.0 0.0 1.61", "H 0.93 0.0 -0.56", "H -0.93 0.0 -0.56"],
        0,
        1,
    ),
    "methanethiol": (dumbbell("C", "S", 1.82, ("H", 1.09, 3), ("H", 1.34, 1)), 0, 1),
    "silanol": (dumbbell("Si", "O", 1.63, ("H", 1.48, 3), ("H", 0.96, 1)), 0, 1),
    # -- halogens, including the heavy ones nothing else in the set contains.
    "hydrogen_chloride": (diatomic("H", "Cl", 1.27), 0, 1),
    "hydrogen_bromide": (diatomic("H", "Br", 1.41), 0, 1),
    "hydrogen_iodide": (diatomic("H", "I", 1.61), 0, 1),
    "chlorine": (diatomic("Cl", "Cl", 1.99), 0, 1),
    "bromine": (diatomic("Br", "Br", 2.28), 0, 1),
    "iodine": (diatomic("I", "I", 2.67), 0, 1),
    "chloromethane": (
        substituted_methane([("H", 1.09), ("H", 1.09), ("H", 1.09), ("Cl", 1.78)]), 0, 1),
    "bromomethane": (
        substituted_methane([("H", 1.09), ("H", 1.09), ("H", 1.09), ("Br", 1.93)]), 0, 1),
    "iodomethane": (
        substituted_methane([("H", 1.09), ("H", 1.09), ("H", 1.09), ("I", 2.14)]), 0, 1),
    "chloroform": (
        substituted_methane([("H", 1.09), ("Cl", 1.77), ("Cl", 1.77), ("Cl", 1.77)]), 0, 1),
    "chlorobenzene": (substituted_benzene([(3.17, 0.0, 0.0, "Cl")]), 0, 1),
    # -- alkali and alkaline earth: ionic bonding, and core parameters nothing covalent reaches.
    "lithium_hydride": (diatomic("Li", "H", 1.60), 0, 1),
    "lithium_fluoride": (diatomic("Li", "F", 1.56), 0, 1),
    "sodium_chloride": (diatomic("Na", "Cl", 2.36), 0, 1),
    "potassium_bromide": (diatomic("K", "Br", 2.82), 0, 1),
    "beryllium_dihydride": (linear(["H", "Be", "H"], [1.33, 1.33]), 0, 1),
    "magnesium_dichloride": (linear(["Cl", "Mg", "Cl"], [2.18, 2.18]), 0, 1),
    "calcium_difluoride": (bent("Ca", "F", 2.10, 140.0), 0, 1),
    # -- aromatic and conjugated: the resonance integral and the two-centre block under
    # delocalization, which no localized molecule probes.
    "benzene": (benzene(), 0, 1),
    "butadiene": (butadiene(), 0, 1),
    "pyridine": (
        ring(["N", "C", "C", "C", "C", "C"], 1.395, [False, True, True, True, True, True], 2.48),
        0,
        1,
    ),
    "furan": (ring(["O", "C", "C", "C", "C"], 1.12, [False, True, True, True, True], 2.21), 0, 1),
    "thiophene": (
        ring(["S", "C", "C", "C", "C"], 1.25, [False, True, True, True, True], 2.34), 0, 1),
    "pyrrole": (ring(["N", "C", "C", "C", "C"], 1.17, [True, True, True, True, True], 2.26), 0, 1),
    "imidazole": (ring(["N", "C", "N", "C", "C"], 1.17, [True, True, False, True, True], 2.26),
                  0, 1),
    "naphthalene_like": (naphthalene_like(), 0, 1),
    "phenol": (substituted_benzene([(2.76, 0.0, 0.0, "O"), (3.10, 0.90, 0.0, "H")]), 0, 1),
    "aniline": (
        substituted_benzene([(2.80, 0.0, 0.0, "N"), (3.34, 0.82, 0.0, "H"),
                             (3.34, -0.82, 0.0, "H")]),
        0,
        1,
    ),
    "toluene": (
        substituted_benzene(
            [(2.90, 0.0, 0.0, "C")]
            + cap((2.90, 0.0, 0.0), (1.397, 0.0, 0.0), "H", 1.09, 3)
        ),
        0,
        1,
    ),
    # -- strained and cage: the angular terms, far from the geometries any parameterization was
    # fitted to.
    "cyclopropane": (
        _lines(
            [(0.870, 0.000, 0.0, "C"), (-0.435, 0.753, 0.0, "C"), (-0.435, -0.753, 0.0, "C")]
            + [(1.520, 0.000, 0.885, "H"), (1.520, 0.000, -0.885, "H"),
               (-0.760, 1.317, 0.885, "H"), (-0.760, 1.317, -0.885, "H"),
               (-0.760, -1.317, 0.885, "H"), (-0.760, -1.317, -0.885, "H")]
        ),
        0,
        1,
    ),
    "cubane": (cubane(), 0, 1),
    # -- charged: the dipole origin convention and an ion's core terms, at both signs.
    "hydroxide": (["O 0.0 0.0 0.0", "H 0.96 0.0 0.0"], -1, 1),
    "ammonium": (
        ["N 0.0 0.0 0.0", "H 0.60 0.60 0.60", "H -0.60 -0.60 0.60", "H -0.60 0.60 -0.60",
         "H 0.60 -0.60 -0.60"],
        1,
        1,
    ),
    "hydronium": (pyramidal("O", "H", 0.98, 113.0), 1, 1),
    "fluoride": (["F 0.0 0.0 0.0"], -1, 1),
    "chloride": (["Cl 0.0 0.0 0.0"], -1, 1),
    "cyanide": (diatomic("C", "N", 1.17), -1, 1),
    "nitrate": (trigonal_planar("N", "O", 1.26), -1, 1),
    "carbonate": (trigonal_planar("C", "O", 1.29), -2, 1),
    "sulfate": (tetrahedral("S", "O", 1.49), -2, 1),
    "acetate": (
        _lines(
            [(0.00, 0.00, 0.00, "C"), (1.52, 0.00, 0.00, "C"),
             (2.13, 1.08, 0.00, "O"), (2.13, -1.08, 0.00, "O")]
            + cap((0.0, 0.0, 0.0), (1.52, 0.0, 0.0), "H", 1.09, 3)
        ),
        -1,
        1,
    ),
    "methyl_cation": (trigonal_planar("C", "H", 1.09), 1, 1),
    "methyl_anion": (pyramidal("C", "H", 1.10, 108.0), -1, 1),
    "methylammonium": (dumbbell("N", "C", 1.50, ("H", 1.02, 3), ("H", 1.09, 3)), 1, 1),
    # -- open shell: the UHF path, at both multiplicities it supports.
    "methyl_radical": (
        ["C 0.0 0.0 0.0", "H 1.08 0.0 0.0", "H -0.54 0.94 0.0", "H -0.54 -0.94 0.0"],
        0,
        2,
    ),
    "hydroxyl_radical": (diatomic("O", "H", 0.97), 0, 2),
    "nitric_oxide": (diatomic("N", "O", 1.15), 0, 2),
    "nitrogen_dioxide": (bent("N", "O", 1.19, 134.0), 0, 2),
    "chlorine_dioxide": (bent("Cl", "O", 1.47, 117.0), 0, 2),
    "ethyl_radical": (
        _lines(
            [(0.00, 0.00, 0.00, "C"), (1.49, 0.00, 0.00, "C")]
            + cap((1.49, 0.0, 0.0), (0.0, 0.0, 0.0), "H", 1.09, 3)
            + [(-0.55, 0.93, 0.00, "H"), (-0.55, -0.93, 0.00, "H")]
        ),
        0,
        2,
    ),
    "oxygen_triplet": (diatomic("O", "O", 1.21), 0, 3),
    "methylene_triplet": (bent("C", "H", 1.08, 134.0), 0, 3),
    "water_dimer": (water_dimer(), 0, 1),
    "formic_acid_dimer": (formic_acid_dimer(), 0, 1),
    "ammonia_water": (
        _lines(
            [(0.00, 0.00, 0.00, "N"), (2.95, 0.00, 0.00, "O")]
            + cap((0.0, 0.0, 0.0), (2.95, 0.0, 0.0), "H", 1.02, 3)
            + [(2.00, 0.05, 0.00, "H"), (3.25, 0.80, 0.45, "H")]
        ),
        0,
        1,
    ),
    "benzene_dimer_stacked": (stacked_benzene(3.60), 0, 1),
    "methane_dimer": (
        tetrahedral("C", "H", 1.09)
        + _lines(
            [(3.70, 0.0, 0.0, "C")]
            + [(3.70 + 1.09 * x / math.sqrt(3.0), 1.09 * y / math.sqrt(3.0),
                1.09 * z / math.sqrt(3.0), "H") for x, y, z in _TETRAHEDRAL]
        ),
        0,
        1,
    ),
    "glycine": (glycine(), 0, 1),
    # -- large, and the reason this file exists: past the 7 A feather range, where every two-centre
    # integral is exactly a point charge.
    "alkane_c20": (alkane(20), 0, 1),
    "alkane_c30": (alkane(30), 0, 1),
    "hf_chain_8": (hf_chain(8), 0, 1),
    "water_wire_8": (water_wire(8), 0, 1),
    "water_wire_12": (water_wire(12), 0, 1),
    # A hydroxide-doped water wire: the wire minus one terminal proton, at charge -1. Dropping the
    # proton as well as adding the charge is what keeps the electron count even -- an 8-water wire
    # at charge -1 has 65 electrons and is not a singlet at all.
    "water_wire_8_anion": (water_wire(8)[:-1], -1, 1),
}


# --- every parameterized element ------------------------------------------------------------
#
# **The set above covers nineteen elements; PM7 is parameterized for seventy-three.** An oracle that
# exercises a quarter of the parameter table can say nothing about the rest of it, and the rest of
# it is where a transcription error would sit unnoticed — one wrong `alpb`, `xfac` or `u_ss` row is
# invisible until a molecule containing that element is run.
#
# So the block below adds a compound for every element the table parameterizes and the set above
# does not reach, and `validate_coverage` at the end turns that into a **contract**: the assertion
# reads `src/data/pm7_elements.csv`, so an element added to PM7 support without an oracle case
# fails here rather than going untested.
#
# **Spin states are chosen, not derived from electron parity.** `docs/fidelity.md` records what
# happens otherwise: the diatomic sweep assigns multiplicity by parity alone, asks for Co(d⁷)H as a
# singlet, and the two codes settle in different SCF basins — 22 of 125 pairs, up to 90 kcal/mol,
# and none of it a statement about the integrals. Every transition metal here is therefore in a
# formal oxidation state whose ground configuration is unambiguous: d⁰ (Sc³⁺, Ti⁴⁺, V⁵⁺, Cr⁶⁺,
# Mn⁷⁺, Zr⁴⁺, Nb⁵⁺, Mo⁶⁺, Tc⁷⁺, Hf⁴⁺, Ta⁵⁺, W⁶⁺, Re⁷⁺, Os⁸⁺), d⁶ low spin in a strong carbonyl
# field, d⁸ square planar, or d¹⁰.
COVERAGE: dict[str, tuple[list[str], int, int]] = {
    # -- noble gases. Bare atoms on purpose: `docs/fidelity.md` records that noble-gas *hydrides*
    # and *fluorides* are exactly where the two codes pick different SCF basins, and a bare closed
    # shell exercises the element's parameters and its heat-of-formation reference without that
    # confound.
    "helium": (["He 0.0 0.0 0.0"], 0, 1),
    "neon": (["Ne 0.0 0.0 0.0"], 0, 1),
    "argon": (["Ar 0.0 0.0 0.0"], 0, 1),
    "krypton": (["Kr 0.0 0.0 0.0"], 0, 1),
    "xenon": (["Xe 0.0 0.0 0.0"], 0, 1),
    # -- the rest of the alkali and alkaline earth metals.
    "rubidium_chloride": (diatomic("Rb", "Cl", 2.79), 0, 1),
    "caesium_chloride": (diatomic("Cs", "Cl", 2.91), 0, 1),
    "strontium_dichloride": (linear(["Cl", "Sr", "Cl"], [2.63, 2.63]), 0, 1),
    "barium_dichloride": (linear(["Cl", "Ba", "Cl"], [2.78, 2.78]), 0, 1),
    # -- the heavier p block: hydrides and halides, all closed shell.
    "gallium_trichloride": (trigonal_planar("Ga", "Cl", 2.09), 0, 1),
    "germane": (tetrahedral("Ge", "H", 1.53), 0, 1),
    "arsine": (pyramidal("As", "H", 1.51, 92.1), 0, 1),
    "hydrogen_selenide": (bent("Se", "H", 1.46, 91.0), 0, 1),
    "indium_trichloride": (trigonal_planar("In", "Cl", 2.29), 0, 1),
    "stannane": (tetrahedral("Sn", "H", 1.71), 0, 1),
    "stibine": (pyramidal("Sb", "H", 1.70, 91.7), 0, 1),
    "hydrogen_telluride": (bent("Te", "H", 1.66, 90.3), 0, 1),
    "thallium_chloride": (diatomic("Tl", "Cl", 2.48), 0, 1),
    "lead_dichloride": (bent("Pb", "Cl", 2.44, 96.0), 0, 1),
    "bismuth_trichloride": (pyramidal("Bi", "Cl", 2.42, 100.0), 0, 1),
    # -- d0 transition-metal halides and oxides: an empty d shell is a closed shell, so the spin
    # state is not in question and neither code has a basin to choose.
    "scandium_trifluoride": (trigonal_planar("Sc", "F", 1.85), 0, 1),
    "titanium_tetrachloride": (tetrahedral("Ti", "Cl", 2.17), 0, 1),
    "vanadium_oxytrichloride": (
        _lines([(0.0, 0.0, 0.0, "V"), (0.0, 0.0, 1.57, "O")]
               + [(2.14 * x / math.sqrt(3.0), 2.14 * y / math.sqrt(3.0),
                   -2.14 * abs(z) / math.sqrt(3.0), "Cl")
                  for x, y, z in _TETRAHEDRAL[:3]]),
        0,
        1,
    ),
    "chromyl_chloride": (
        _lines([(0.0, 0.0, 0.0, "Cr"), (1.58, 0.0, 0.79, "O"), (-1.58, 0.0, 0.79, "O"),
                (0.0, 1.76, -0.88, "Cl"), (0.0, -1.76, -0.88, "Cl")]),
        0,
        1,
    ),
    "permanganate": (tetrahedral("Mn", "O", 1.63), -1, 1),
    "yttrium_trichloride": (trigonal_planar("Y", "Cl", 2.47), 0, 1),
    "zirconium_tetrachloride": (tetrahedral("Zr", "Cl", 2.32), 0, 1),
    "niobium_pentachloride": (
        _lines([(0.0, 0.0, 0.0, "Nb")]
               + [(2.30 * x, 2.30 * y, 0.0, "Cl") for x, y in
                  ((1.0, 0.0), (-0.5, 0.866025), (-0.5, -0.866025))]
               + [(0.0, 0.0, 2.34, "Cl"), (0.0, 0.0, -2.34, "Cl")]),
        0,
        1,
    ),
    "molybdenum_hexafluoride": (octahedral("Mo", "F", 1.82), 0, 1),
    "pertechnetate": (tetrahedral("Tc", "O", 1.71), -1, 1),
    "lanthanum_trichloride": (trigonal_planar("La", "Cl", 2.59), 0, 1),
    "lutetium_trifluoride": (trigonal_planar("Lu", "F", 2.02), 0, 1),
    "hafnium_tetrachloride": (tetrahedral("Hf", "Cl", 2.32), 0, 1),
    "tantalum_pentachloride": (
        _lines([(0.0, 0.0, 0.0, "Ta")]
               + [(2.28 * x, 2.28 * y, 0.0, "Cl") for x, y in
                  ((1.0, 0.0), (-0.5, 0.866025), (-0.5, -0.866025))]
               + [(0.0, 0.0, 2.32, "Cl"), (0.0, 0.0, -2.32, "Cl")]),
        0,
        1,
    ),
    "tungsten_hexafluoride": (octahedral("W", "F", 1.83), 0, 1),
    "perrhenate": (tetrahedral("Re", "O", 1.72), -1, 1),
    "osmium_tetroxide": (tetrahedral("Os", "O", 1.71), 0, 1),
    # -- d10 and d8: closed by a full shell or by a square-planar field.
    "copper_chloride": (diatomic("Cu", "Cl", 2.05), 0, 1),
    "zinc_dichloride": (linear(["Cl", "Zn", "Cl"], [2.07, 2.07]), 0, 1),
    "silver_chloride": (diatomic("Ag", "Cl", 2.28), 0, 1),
    "cadmium_dichloride": (linear(["Cl", "Cd", "Cl"], [2.28, 2.28]), 0, 1),
    "mercury_dichloride": (linear(["Cl", "Hg", "Cl"], [2.25, 2.25]), 0, 1),
    "tetrachloropalladate": (square_planar("Pd", "Cl", 2.31), -2, 1),
    "tetrachloroplatinate": (square_planar("Pt", "Cl", 2.32), -2, 1),
    "gold_chloride": (diatomic("Au", "Cl", 2.20), 0, 1),
    # -- the actinide and lanthanide sparkles the parameter table carries.
    "thorium_tetrafluoride": (tetrahedral("Th", "F", 2.14), 0, 1),
    "californium_trifluoride": (trigonal_planar("Cf", "F", 2.16), 0, 1),
    "nobelium_trifluoride": (trigonal_planar("No", "F", 2.18), 0, 1),
    # -- organometallics: a metal bonded to carbon, which is a different test from a metal bonded
    # to a halide. The carbonyls are d6 or d10 low spin, the metallocene is Fe(II) d6 low spin, and
    # every one of them is a closed shell whose spin state is not in doubt.
    "nickel_tetracarbonyl": (carbonyl("Ni", 1.84, 1.13, "tetrahedral"), 0, 1),
    "iron_pentacarbonyl": (carbonyl("Fe", 1.81, 1.15, "bipyramidal"), 0, 1),
    "chromium_hexacarbonyl": (carbonyl("Cr", 1.92, 1.14, "octahedral"), 0, 1),
    "molybdenum_hexacarbonyl": (carbonyl("Mo", 2.06, 1.15, "octahedral"), 0, 1),
    "tungsten_hexacarbonyl": (carbonyl("W", 2.06, 1.15, "octahedral"), 0, 1),
    "ruthenium_pentacarbonyl": (carbonyl("Ru", 1.94, 1.14, "bipyramidal"), 0, 1),
    "osmium_pentacarbonyl": (carbonyl("Os", 1.95, 1.14, "bipyramidal"), 0, 1),
    "manganese_carbonyl_hydride": (
        carbonyl("Mn", 1.85, 1.14, "octahedral")[: 1 + 2 * 5]
        + _lines([(0.0, 0.0, -1.58, "H")]),
        0,
        1,
    ),
    "cobalt_tetracarbonyl_anion": (carbonyl("Co", 1.80, 1.15, "tetrahedral"), -1, 1),
    "rhodium_dicarbonyl_chloride": (
        _lines([(0.0, 0.0, 0.0, "Rh"), (1.85, 0.0, 0.0, "C"), (2.99, 0.0, 0.0, "O"),
                (0.0, 1.85, 0.0, "C"), (0.0, 2.99, 0.0, "O"), (-2.32, 0.0, 0.0, "Cl"),
                (0.0, -2.32, 0.0, "Cl")]),
        -1,
        1,
    ),
    "iridium_dicarbonyl_dichloride": (
        _lines([(0.0, 0.0, 0.0, "Ir"), (1.87, 0.0, 0.0, "C"), (3.01, 0.0, 0.0, "O"),
                (0.0, 1.87, 0.0, "C"), (0.0, 3.01, 0.0, "O"), (-2.35, 0.0, 0.0, "Cl"),
                (0.0, -2.35, 0.0, "Cl")]),
        -1,
        1,
    ),
    "ferrocene": (metallocene("Fe", 1.66, 1.43, 1.08), 0, 1),
    "dimethylzinc": (
        _lines([(0.0, 0.0, 0.0, "Zn"), (1.93, 0.0, 0.0, "C"), (-1.93, 0.0, 0.0, "C")])
        + _lines(cap((1.93, 0.0, 0.0), (0.0, 0.0, 0.0), "H", 1.09, 3)
                 + cap((-1.93, 0.0, 0.0), (0.0, 0.0, 0.0), "H", 1.09, 3, twist=60.0)),
        0,
        1,
    ),
    "tetramethyltin": (
        _lines([(0.0, 0.0, 0.0, "Sn")]
               + [(2.14 * x / math.sqrt(3.0), 2.14 * y / math.sqrt(3.0),
                   2.14 * z / math.sqrt(3.0), "C") for x, y, z in _TETRAHEDRAL])
        + _lines([p for x, y, z in _TETRAHEDRAL
                  for p in cap((2.14 * x / math.sqrt(3.0), 2.14 * y / math.sqrt(3.0),
                                2.14 * z / math.sqrt(3.0)), (0.0, 0.0, 0.0), "H", 1.09, 3)]),
        0,
        1,
    ),
    "methylmagnesium_chloride": (
        _lines([(0.0, 0.0, 0.0, "Mg"), (2.10, 0.0, 0.0, "C"), (-2.20, 0.0, 0.0, "Cl")])
        + _lines(cap((2.10, 0.0, 0.0), (0.0, 0.0, 0.0), "H", 1.09, 3)),
        0,
        1,
    ),
    "titanocene_dichloride": (
        metallocene("Ti", 2.06, 1.42, 1.08, tilt=1.30)
        + _lines([(0.0, 2.10, -1.55, "Cl"), (0.0, -2.10, -1.55, "Cl")]),
        0,
        1,
    ),
}

# --- open shells ------------------------------------------------------------------------------
#
# The UHF path deserves more than the six doublets and two triplets above, and the cheapest way to
# reach a **high-spin** state whose configuration is not in doubt is a bare atom: a nitrogen's
# ground term is ⁴S and a carbon's is ³P, with no geometry and no basin to choose. Those are the
# cases that would catch a spin-density or an exchange-term defect that a doublet does not.
OPEN_SHELL: dict[str, tuple[list[str], int, int]] = {
    "carbon_atom_triplet": (["C 0.0 0.0 0.0"], 0, 3),
    "nitrogen_atom_quartet": (["N 0.0 0.0 0.0"], 0, 4),
    "oxygen_atom_triplet": (["O 0.0 0.0 0.0"], 0, 3),
    "fluorine_atom_doublet": (["F 0.0 0.0 0.0"], 0, 2),
    "phosphorus_atom_quartet": (["P 0.0 0.0 0.0"], 0, 4),
    "sulfur_atom_triplet": (["S 0.0 0.0 0.0"], 0, 3),
    "silicon_atom_triplet": (["Si 0.0 0.0 0.0"], 0, 3),
    # Molecular open shells beyond the doublets above: a difluoroamino radical, a cyanide radical,
    # and the allyl radical, whose singly occupied orbital is delocalized over three carbons rather
    # than localized on one — a different thing for the UHF density to get right.
    "difluoroamino_radical": (bent("N", "F", 1.36, 103.0), 0, 2),
    "cyanide_radical": (diatomic("C", "N", 1.17), 0, 2),
    "allyl_radical": (
        _lines([(0.00, 0.00, 0.00, "C"), (1.39, 0.00, 0.00, "C"), (2.09, 1.20, 0.00, "C"),
                (-0.55, 0.93, 0.00, "H"), (-0.55, -0.93, 0.00, "H"),
                (1.94, -0.93, 0.00, "H"),
                (1.54, 2.13, 0.00, "H"), (3.18, 1.20, 0.00, "H")]),
        0,
        2,
    ),
    # A triplet dication and a quintet: charge and high spin at once, which nothing else here does.
    "oxygen_dication_triplet": (diatomic("O", "O", 1.12), 2, 3),
    "carbon_dimer_quintet": (diatomic("C", "C", 1.50), 0, 5),
}

CASES.update(COVERAGE)
CASES.update(OPEN_SHELL)


# The symbols the parameter table is indexed by. Kept here rather than imported because the oracle
# scripts are standalone: they run against an installed `pm7-rs` or a built binary, not against the
# crate's source.
_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
    20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
    29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb",
    38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd",
    47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
    56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
    74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa",
    92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No",
}


def parameterized_elements() -> set[str]:
    """Every element `src/data/pm7_elements.csv` actually parameterizes.

    Read from the shipped table rather than listed here, which is what makes the coverage check a
    contract instead of a comment: an element added to PM7 support and not to this file fails
    `validate_coverage` on the next run.
    """
    path = oracle.ROOT / "src/data/pm7_elements.csv"
    rows = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    header = rows[0].split(",")
    z_at, zeta_at, beta_at = (header.index(n) for n in ("z", "zeta_s", "beta_s"))
    out = set()
    for row in rows[1:]:
        cells = row.split(",")
        # The table carries a row per atomic number; an unparameterized one is all zeros. This is
        # the same test `Pm7Parameters::from_tables` applies when it decides to skip a row.
        if float(cells[zeta_at]) == 0.0 and float(cells[beta_at]) == 0.0:
            continue
        symbol = _SYMBOLS.get(int(cells[z_at]))
        if symbol:
            out.add(symbol)
    return out


def elements_in_cases() -> set[str]:
    return {line.split()[0] for atoms, _, _ in CASES.values() for line in atoms}


def validate_coverage() -> list[str]:
    """Complaints about what the case set fails to reach."""
    wanted = parameterized_elements()
    have = elements_in_cases()
    missing = sorted(wanted - have)
    complaints = []
    if missing:
        complaints.append(
            f"{len(missing)} parameterized element(s) appear in no case: {', '.join(missing)}. "
            f"An element PM7 has parameters for and the oracle never runs is an untested "
            f"parameter row."
        )
    stray = sorted(have - wanted - {"H"})
    if stray:
        complaints.append(f"case(s) use element(s) PM7 has no parameters for: {', '.join(stray)}")
    return complaints


def keywords(charge: int, multiplicity: int) -> str:
    """MOPAC keywords for one case.

    `RELSCF=0.0001` is not decoration. `PRECISE` alone leaves MOPAC's own SCF the *looser* of the
    two sides on a large system, and because the heat of formation is variational while the atomic
    charges are not, that shows up in the charges long before it shows up in the energy. Measured
    on the 62-atom `alkane_c20` case, against `pm7-rs` converged to 1e-12:

    | MOPAC SCF          | worst Mulliken charge delta | heat-of-formation delta |
    |--------------------|-----------------------------|-------------------------|
    | `PRECISE`          | 2.23e-4 e                   | -6.6e-6 kcal/mol        |
    | `+ RELSCF=0.01`    | 2.39e-5 e                   | +3.4e-6 kcal/mol        |
    | `+ RELSCF=0.0001`  | 1.89e-6 e                   | +3.4e-6 kcal/mol        |

    The residual falls to MOPAC's 6-decimal print precision, so the 2.2e-4 seen at `PRECISE` was
    measuring MOPAC's convergence, not this implementation's fidelity.
    """
    words = ["PM7", "1SCF", "PRECISE", "GRADIENTS", "RELSCF=0.0001"]
    if charge:
        words.append(f"CHARGE={charge}")
    # Through 0.2.3 this stopped at TRIPLET, which quietly capped what the case set could contain:
    # a nitrogen atom's ground state is a quartet and there was no way to ask for one.
    spin = {2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUINTET", 6: "SEXTET"}.get(multiplicity)
    if spin:
        words += ["UHF", spin]
    elif multiplicity != 1:
        raise ValueError(f"multiplicity {multiplicity} has no MOPAC keyword here")
    return " ".join(words)


def run_case(name: str, mopac: Path) -> dict:
    """Run both programs on one case and return every number both sides produced."""
    atoms, charge, multiplicity = CASES[name]
    xyz = oracle.write_xyz(WORKDIR / f"{name}.xyz", name, atoms)
    mop = oracle.write_mop(WORKDIR / f"{name}.mop", name, atoms, keywords(charge, multiplicity))

    reference = oracle.run_mopac_full(mopac, mop)
    extra = ["--charge", str(charge), "--multiplicity", str(multiplicity)]
    ours = oracle.run_pm7_rs_json(xyz, "energy", *extra)

    record: dict = {"charge": charge, "multiplicity": multiplicity, "atoms": len(atoms)}
    if reference is not None:
        record["mopac"] = {
            "hof_kcal": reference.hof_kcal,
            "homo_ev": reference.homo_ev,
            "lumo_ev": reference.lumo_ev,
            "homo_ev_beta": reference.homo_ev_beta,
            "lumo_ev_beta": reference.lumo_ev_beta,
            "dipole_sum": reference.dipole_sum,
            "dipole_total": reference.dipole_total,
            "charges": reference.charges,
            # Printed only by an unrestricted run, so `None` here means "MOPAC used RHF" and the
            # comparison below skips rather than failing.
            "spin_squared": reference.spin_squared,
        }
    if ours is not None:
        record["pm7_rs"] = ours
    return record


TSV_COLUMNS = (
    "case", "atoms", "charge", "multiplicity",
    "mopac_hof_kcal", "pm7rs_hof_kcal", "d_hof_kcal",
    "mopac_homo_ev", "pm7rs_homo_ev", "d_homo_ev",
    "mopac_lumo_ev", "pm7rs_lumo_ev", "d_lumo_ev",
    "mopac_dipole_debye", "pm7rs_dipole_debye", "d_dipole_debye",
    "worst_charge_e",
    "mopac_s2", "pm7rs_s2", "d_s2",
)


def write_tsv(fresh: dict, path: Path) -> Path:
    """One row per case, both sides side by side, with the difference already taken.

    `reference.json` is the machine-readable record and keeps everything -- every orbital energy,
    every Mulliken charge. This is the *readable* one: it opens in a spreadsheet, sorts by the delta
    column, and answers "which case is worst" without a JSON path. Tab-separated rather than comma,
    because several of these fields are already comma-free numbers and a molecule name never
    contains a tab.
    """
    def cell(value: float | None, digits: int = 6) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    rows = ["\t".join(TSV_COLUMNS)]
    for name, record in fresh.get("cases", {}).items():
        reference, ours = record.get("mopac") or {}, record.get("pm7_rs") or {}
        mopac_charges, our_charges = reference.get("charges") or [], ours.get("charges") or []
        worst_charge = (
            max((abs(a - b) for a, b in zip(our_charges, mopac_charges)), default=None)
            if len(mopac_charges) == len(our_charges) and mopac_charges
            else None
        )
        # MOPAC prints the dipole as a vector and a total; the comparable scalar is the total.
        mopac_dipole = reference.get("dipole_total")
        ours_dipole = ours.get("dipole_debye")
        ours_total = (
            math.sqrt(sum(c * c for c in ours_dipole)) if ours_dipole is not None else None
        )
        row = [
            name,
            str(record.get("atoms", "")),
            str(record.get("charge", "")),
            str(record.get("multiplicity", "")),
        ]
        for ours_key, mopac_key, digits in (
            ("heat_of_formation_kcal", "hof_kcal", 6),
            ("homo_ev", "homo_ev", 4),
            ("lumo_ev", "lumo_ev", 4),
        ):
            a, b = reference.get(mopac_key), ours.get(ours_key)
            row += [cell(a, digits), cell(b, digits),
                    cell(None if a is None or b is None else b - a, digits)]
        row += [
            cell(mopac_dipole, 4), cell(ours_total, 4),
            cell(None if mopac_dipole is None or ours_total is None
                 else ours_total - mopac_dipole, 4),
        ]
        row.append(cell(worst_charge, 8))
        # `<S^2>`, blank on both sides for a restricted case rather than filled with `S(S+1)`:
        # sorting the column then brings the open-shell cases together instead of burying them.
        mopac_s2, our_s2 = reference.get("spin_squared"), ours.get("spin_squared")
        row += [
            cell(mopac_s2, 6), cell(our_s2, 6),
            cell(None if mopac_s2 is None or our_s2 is None else our_s2 - mopac_s2, 6),
        ]
        rows.append("\t".join(row))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def compare(name: str, record: dict, checker: oracle.Checker, limits: dict[str, float]) -> None:
    """Check one case's `pm7-rs` numbers against MOPAC's."""
    reference, ours = record.get("mopac"), record.get("pm7_rs")
    if reference is None or ours is None:
        checker.skipped.append(f"{name}: one side produced nothing")
        return

    checker.check(
        name,
        "heat of formation (kcal/mol)",
        oracle.delta(ours.get("heat_of_formation_kcal"), reference.get("hof_kcal"), 9),
        limits["hof_kcal"],
    )

    mopac_charges, our_charges = reference.get("charges") or [], ours.get("charges") or []
    if len(mopac_charges) == len(our_charges) and mopac_charges:
        worst = max(abs(a - b) for a, b in zip(our_charges, mopac_charges))
        checker.check(name, "worst Mulliken charge (e)", worst, limits["charge_e"])
    else:
        checker.skipped.append(f"{name}: charge vectors differ in length")

    # `dipole_debye` and the orbital energies reach the CLI only once the property stack of
    # v0.2.1 is wired; until then these are skips rather than silent passes.
    if "dipole_debye" in ours and reference.get("dipole_sum"):
        worst = max(abs(a - b) for a, b in zip(ours["dipole_debye"], reference["dipole_sum"]))
        checker.check(name, "worst dipole component (D)", worst, limits["dipole_debye"])
    else:
        checker.skipped.append(f"{name}: dipole not on the CLI yet")

    for label in ("homo_ev", "lumo_ev", "homo_ev_beta", "lumo_ev_beta"):
        if label in ours and reference.get(label) is not None:
            checker.check(
                name,
                f"{label} (eV)",
                oracle.delta(ours[label], reference[label], 9),
                limits["orbital_ev"],
            )
        else:
            checker.skipped.append(f"{name}: {label} not reported by both sides")

    # `<S^2>` exists only for an unrestricted solution, and both sides decide that the same way --
    # by shell, from the multiplicity -- so a case where one has it and the other does not is a
    # disagreement about the *spin path*, which is worth a complaint rather than a skip.
    mopac_s2, our_s2 = reference.get("spin_squared"), ours.get("spin_squared")
    if mopac_s2 is not None and our_s2 is not None:
        checker.check(
            name, "<S^2>", oracle.delta(our_s2, mopac_s2, 9), limits["spin_squared"]
        )
    elif mopac_s2 is None and our_s2 is None:
        pass  # Both restricted. Nothing to compare, and nothing wrong.
    else:
        # Not a tolerance -- there is no small version of "the two codes ran different methods".
        # A limit of zero would be the honest number and divides by zero in the report's ranking,
        # so this is the smallest one that reads as a breach and sorts.
        checker.check(
            name,
            f"spin path disagrees: MOPAC {'UHF' if mopac_s2 is not None else 'RHF'}, "
            f"pm7-rs {'UHF' if our_s2 is not None else 'RHF'}",
            1.0,
            0.5,
        )


def drift(recorded: dict, fresh: dict, checker: oracle.Checker, limits: dict[str, float]) -> None:
    """Check `pm7-rs` against its own recorded baseline: has anything moved that should not have?"""
    for name, old in recorded.get("cases", {}).items():
        new = fresh.get("cases", {}).get(name)
        if new is None or "pm7_rs" not in old or "pm7_rs" not in new:
            continue
        before, after = old["pm7_rs"], new["pm7_rs"]
        checker.check(
            name,
            "self-drift: heat of formation",
            oracle.delta(
                after.get("heat_of_formation_kcal"), before.get("heat_of_formation_kcal"), 12
            ),
            limits["self_drift_kcal"],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the baseline and exit 0")
    parser.add_argument("--only", help="run one case by name")
    parser.add_argument(
        "--self-drift-kcal",
        type=float,
        default=1.0e-6,
        help="how far pm7-rs's own heat of formation may move from the recorded baseline",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check every case's geometry and exit, without running either program",
    )
    parser.add_argument(
        "--tsv",
        default=str(oracle.ROOT / "tools/oracle/baselines/reference.tsv"),
        help="where to write the one-row-per-case summary (default: beside reference.json)",
    )
    args = parser.parse_args(argv)

    # **Geometry first, before either program runs.** A case with a hydrogen 0.6 A from its carbon
    # still converges, to the energy of a different molecule, and both programs agree on it
    # perfectly — so every check downstream of here would pass and the case would be worthless.
    # This is the only check in the file that cannot be made by comparing the two sides.
    complaints = [c for name, (atoms, _, _) in CASES.items()
                  for c in validate_geometry(name, atoms)]
    complaints += validate_coverage()
    if complaints:
        for complaint in complaints:
            print(f"  {complaint}", file=sys.stderr)
        print(f"\n{len(complaints)} geometry problem(s); nothing was run", file=sys.stderr)
        return 1
    if args.validate_only:
        print(f"{len(CASES)} case geometries are sane, covering "
              f"{len(elements_in_cases() & parameterized_elements())} of "
              f"{len(parameterized_elements())} parameterized elements")
        return 0

    mopac = oracle.mopac_executable()
    names = [args.only] if args.only else list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        return parser.error(f"unknown case(s): {', '.join(unknown)}")

    fresh = {"cases": {}}
    rows = []
    for name in names:
        record = run_case(name, mopac)
        fresh["cases"][name] = record
        reference, ours = record.get("mopac", {}), record.get("pm7_rs", {})
        rows.append(
            oracle.Row(
                name,
                {
                    "atoms": record["atoms"],
                    "mopac": reference.get("hof_kcal"),
                    "pm7-rs": ours.get("heat_of_formation_kcal"),
                    "delta": oracle.delta(
                        ours.get("heat_of_formation_kcal"), reference.get("hof_kcal"), 6
                    ),
                },
            )
        )
    print(oracle.table(rows, ["atoms", "mopac", "pm7-rs", "delta"]))

    tsv = write_tsv(fresh, Path(args.tsv))
    print(f"\nwrote the side-by-side summary to {tsv}")

    if args.record:
        path = oracle.save_baseline("reference", fresh)
        print(f"recorded {len(fresh['cases'])} case(s) to {path}")
        return 0

    limits = oracle.thresholds()
    limits["self_drift_kcal"] = args.self_drift_kcal
    checker = oracle.Checker("oracle")
    for name in names:
        compare(name, fresh["cases"][name], checker, limits)
    recorded = oracle.load_baseline("reference")
    if recorded is not None:
        drift(recorded, fresh, checker, limits)
    else:
        print("\nno recorded baseline; run with --record to create one")
    return checker.report()


if __name__ == "__main__":
    raise SystemExit(main())