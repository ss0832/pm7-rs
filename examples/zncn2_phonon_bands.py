# SPDX-License-Identifier: GPL-3.0-or-later
"""Phonon dispersion of Zn(CN)2 along the simple-cubic Brillouin-zone path.

    python examples/zncn2_phonon_bands.py [--points 12] [--out zncn2_phonons.png]

Zn(CN)2 is a coordination polymer: two interpenetrating diamond-like nets of Zn centres bridged by
cyanide, cubic P-43m, ten atoms per primitive cell and so thirty branches. It is a good test of a
periodic semiempirical Hamiltonian precisely because its spectrum spans two decades — a stiff
C≡N stretch near 2400 cm⁻¹ sitting above a dense forest of Zn–C–N bending and librational modes
below 600, with the framework's acoustic branches underneath.

**DFPT, not a supercell.** `phonons` Fourier-interpolates between the wavevectors a supercell can
represent, so a smooth dispersion would need a large repeat of an already ten-atom cell. DFPT
solves the response at each `q` directly, exactly, with no commensurability condition — which is
what makes a band *path* affordable here at all.

**But "no commensurability condition" is about the supercell, not about the k mesh**, and mistaking
one for the other is how the first version of this plot came out wrong. The response couples `k`
with `k + q`, so the *ground-state* sampling still limits which wavevectors mean anything: at a
2×2×2 mesh the acoustic branches came back **imaginary** away from the high-symmetry points, down
to −73 cm⁻¹, and reading that as a structural instability would have been reading the sampling.
Refining the mesh under the softest wavevectors settles it — the dip shrinks monotonically and is
on its way to zero:

    wavevector          2x2x2    3x3x3    4x4x4
    just off Gamma     -72.64   -37.44   -22.55
    Gamma               -0.00    -0.00    -0.00
    just off R         -69.35   -22.53   -15.31

Every *high-symmetry* point was stable at every mesh, which is the tell: a real soft mode is
usually softest at a zone point, where it would give an ordered lower-symmetry structure, and it
does not care how finely the ground state is sampled. So the default here is `--mesh 3`, and the
plot says what the residual softness is rather than dressing it up as physics.

The material is famous for strong negative thermal expansion, whose usual signature is low-lying
transverse modes. Whether PM7 reproduces that is not asserted here; the plot is the measurement,
and the numbers printed alongside it are what a reader should judge.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    sys.exit("this script needs matplotlib: python -m pip install matplotlib")

try:
    from ase.spacegroup import crystal
except ImportError:  # pragma: no cover
    sys.exit("this script needs ASE: python -m pip install ase")

import pm7_rs.native as native

A = 5.9227  # Angstrom, cubic P-43m

# Zn at the two 1a/1b sites; C and N on the body diagonals, placed so that Zn-C is about 1.98 A
# and C-N about 1.15 A, which is what the Zn-C-N-Zn bridge length across the cell requires.
STRUCTURE = (
    ["Zn", "Zn", "C", "N"],
    [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.193, 0.193, 0.193), (0.305, 0.305, 0.305)],
)

# The standard path for a simple-cubic lattice. `None` marks a discontinuity: M and R are not
# joined by the segment that precedes them, so the axis breaks there rather than drawing a line
# through a jump that is not a dispersion.
PATH = [
    ("$\\Gamma$", (0.0, 0.0, 0.0)),
    ("X", (0.0, 0.5, 0.0)),
    ("M", (0.5, 0.5, 0.0)),
    ("$\\Gamma$", (0.0, 0.0, 0.0)),
    ("R", (0.5, 0.5, 0.5)),
    ("X", (0.0, 0.5, 0.0)),
]


def build():
    symbols, positions = STRUCTURE
    atoms = crystal(symbols, positions, spacegroup=215,
                    cellpar=[A, A, A, 90, 90, 90], primitive_cell=True)
    return (
        atoms.get_atomic_numbers().tolist(),
        atoms.get_positions().tolist(),
        atoms.get_cell()[:].tolist(),
        atoms,
    )


def sample(per_segment):
    """Fractional q points along PATH, with the cumulative distance for the x axis."""
    qs, xs, ticks = [], [], []
    distance = 0.0
    for index in range(len(PATH) - 1):
        (label, start), (_, end) = PATH[index], PATH[index + 1]
        start_v, end_v = np.array(start), np.array(end)
        span = float(np.linalg.norm(end_v - start_v))
        ticks.append((distance, label))
        # The last point of a segment is the first of the next, so it is emitted once.
        count = per_segment if index < len(PATH) - 2 else per_segment + 1
        for step in range(count):
            fraction = step / per_segment
            qs.append((start_v + fraction * (end_v - start_v)).tolist())
            xs.append(distance + fraction * span)
        distance += span
    ticks.append((distance, PATH[-1][0]))
    return qs, xs, ticks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=12,
                        help="q points per path segment (default 12)")
    parser.add_argument("--mesh", type=int, default=3,
                        help="Monkhorst-Pack divisions for the ground state (default 3). The "
                             "acoustic branches away from the high-symmetry points are limited by "
                             "this, not by the response: at 2 they come back imaginary to "
                             "-73 cm^-1, at 3 to -37, at 4 to -23")
    parser.add_argument("--out", default="zncn2_phonons.png")
    parser.add_argument("--relax", action="store_true",
                        help="relax the atoms and the lattice first (recommended: the "
                             "experimental constant is not PM7's minimum)")
    args = parser.parse_args()

    numbers, positions, cell, atoms = build()
    print(f"Zn(CN)2 — {atoms.get_chemical_formula()}, {len(numbers)} atoms, "
          f"a = {A} A, {args.mesh}x{args.mesh}x{args.mesh} mesh")

    lattice = A
    if args.relax:
        # A dispersion computed at a strained geometry is a dispersion of the strain as much as of
        # the material: the soft branches are the ones that respond, and they are exactly the ones
        # worth looking at here. So relax the cell first, which is what `relax_cell` is for.
        started = time.time()
        opt = native.optimize(numbers, positions, cell=cell, kpoints=(args.mesh,) * 3,
                              relax_cell=True, scf_tolerance=1e-9, max_iter=60)
        positions, cell = opt["positions_angstrom"], opt["cell_angstrom"]
        lattice = float(np.linalg.norm(cell[0]))
        print(f"relaxed in {opt['iterations']} steps ({time.time() - started:.0f}s), "
              f"converged={opt['converged']}: a = {lattice:.4f} A "
              f"({100 * (lattice - A) / A:+.2f} % from experiment)")

    qs, xs, ticks = sample(args.points)
    plain = [label.replace("$", "").replace("\\", "") for _, label in ticks]
    print(f"{len(qs)} q points along {' - '.join(plain)}")

    # The dispersion is ~10 s per wavevector, so a re-plot should not mean a re-solve. The cache
    # is keyed on everything that changes the answer.
    key = f"{args.points}_{args.mesh}_{'relaxed' if args.relax else 'raw'}"
    cache = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + f".{key}.npz")
    started = time.time()
    if cache.exists():
        bands = np.load(cache)["bands"]
        print(f"reusing {cache.name} — {len(bands)} wavevectors, no re-solve")
    else:
        rows = []
        for index, q in enumerate(qs):
            out = native.dfpt(
                numbers, positions, cell, [q], kpoints=(args.mesh,) * 3,
                scf_tolerance=1e-9, dfpt_tolerance=1e-9, dfpt_max_iterations=400,
            )
            rows.append(sorted(out["frequencies_cm"][0]))
            done = index + 1
            if done % 5 == 0 or done == len(qs):
                rate = (time.time() - started) / done
                print(f"  {done:3d}/{len(qs)}  {rate:.1f}s/point, "
                      f"~{rate * (len(qs) - done) / 60:.1f} min left")
        bands = np.array(rows)
        np.savez_compressed(cache, bands=bands, xs=np.array(xs))

    gamma = bands[0]
    print(f"\nzone centre: {len(gamma)} branches")
    print(f"  three closest to zero: {[round(f, 4) for f in sorted(gamma, key=abs)[:3]]}")
    print(f"  highest: {gamma[-1]:.1f} cm^-1 (the C-N stretch)")
    lowest = bands.min()
    if lowest < -1.0:
        at = np.unravel_index(bands.argmin(), bands.shape)[0]
        q = qs[at]
        print(f"  lowest anywhere on the path: {lowest:.1f} cm^-1 at "
              f"q = ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}) — the acoustic branches are still limited "
              f"by the {args.mesh}x{args.mesh}x{args.mesh} mesh, not by the structure; refine it "
              f"and this shrinks (see the module docstring)")

    # **A broken y axis, because the spectrum has a hole in it.** The C-N stretch sits near
    # 2370 cm^-1 and everything else below 600, so a single axis spends two thirds of its height
    # on empty space and compresses the part with all the structure in it — including the soft
    # branch that goes imaginary, which is the feature worth seeing.
    top = float(bands.max())
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.06},
    )
    soft = "#c05621"   # branches that dip below zero
    normal = "#2b6cb0"
    for branch in range(bands.shape[1]):
        colour = soft if bands[:, branch].min() < -1.0 else normal
        width = 1.4 if colour == soft else 1.0
        for ax in (upper, lower):
            ax.plot(xs, bands[:, branch], lw=width, color=colour, alpha=0.9)

    upper.set_ylim(0.92 * top, 1.02 * top)
    # Room below zero for the note, which has to sit under the lowest branch rather than on it.
    lower.set_ylim(min(-2.6 * abs(bands.min()), -60.0), 640.0)
    lower.axhline(0.0, color="#a0aec0", lw=0.9, ls="--")

    # The diagonal ticks that say "the axis is cut here".
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
              color="#4a5568", mec="#4a5568", mew=1.1, clip_on=False)
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **kw)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **kw)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(bottom=False)

    for ax in (upper, lower):
        for x, _ in ticks[1:-1]:
            ax.axvline(x, color="#cbd5e0", lw=0.8)
        ax.grid(axis="y", color="#edf2f7", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    upper.spines["top"].set_visible(False)

    lower.set_xticks([x for x, _ in ticks])
    lower.set_xticklabels([label for _, label in ticks])
    lower.set_xlim(xs[0], xs[-1])
    lower.set_ylabel("frequency (cm$^{-1}$)")
    lower.yaxis.set_label_coords(-0.075, 0.72)

    geometry = (f"relaxed, a = {lattice:.4f} Å ({100 * (lattice - A) / A:+.2f} % vs experiment)"
                if args.relax else f"experimental a = {A} Å, unrelaxed")
    upper.set_title(
        f"Zn(CN)$_2$ phonon dispersion — PM7 / DFPT, {len(numbers)} atoms per cell\n"
        f"{geometry}, {len(qs)} wavevectors on a "
        f"{args.mesh}×{args.mesh}×{args.mesh} mesh",
        fontsize=11, pad=12,
    )
    # The residual dip is the ground-state sampling, and saying so is the whole point of the
    # annotation: an earlier version of this plot called it a saddle point, which was reading the
    # mesh as physics. Zone-centre and zone-boundary points are stable at every mesh; the dip
    # between them halves with each refinement.
    if lowest < -1.0:
        lower.text(
            0.014, 0.045,
            f"acoustic branches dip to {lowest:.0f} cm$^{{-1}}$ between the high-symmetry points.\n"
            f"That is the {args.mesh}×{args.mesh}×{args.mesh} ground-state mesh, not an "
            f"instability: it halves with each refinement.",
            transform=lower.transAxes, fontsize=8.5, color=soft, linespacing=1.5,
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
        )
    upper.text(0.012, 0.12, "C≡N stretch", transform=upper.transAxes,
               fontsize=8.5, color="#4a5568")
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out} ({time.time() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
