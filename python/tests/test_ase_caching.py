# SPDX-License-Identifier: GPL-3.0-or-later
"""What the ASE calculator actually recomputes, counted rather than asserted from a docstring.

A calculator that *says* it caches and does not is worse than one that says nothing: the cost is
invisible until someone profiles a workflow. These tests count real calls into the native layer by
wrapping it, so a regression in the grouping shows up as a number.
"""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from pm7_rs import native  # noqa: E402
from pm7_rs.ase import PM7  # noqa: E402


def water():
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2400, 0.9278, 0.0]],
    )


class Counter:
    """Wrap a `native` entry point and count how often it is really called."""

    def __init__(self, monkeypatch, name):
        self.calls = 0
        original = getattr(native, name)

        def wrapper(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        # Patch it where the calculator looks it up, which is the module object itself.
        monkeypatch.setattr(native, name, wrapper)


def test_the_whole_vibrational_group_costs_one_solve(monkeypatch):
    """Frequencies, modes, IR intensities, dipole derivatives and the Hessian share one CPHF.

    They all come out of the same solve, so asking for several must not pay for several. This is
    the claim `get_ir_intensities`'s docstring makes; here it is measured.
    """
    counter = Counter(monkeypatch, "vibrations")
    atoms = water()
    atoms.calc = PM7(method="pm7-")

    frequencies = atoms.calc.get_frequencies()
    modes = atoms.calc.get_normal_modes()
    intensities = atoms.calc.get_ir_intensities()
    derivatives = atoms.calc.get_dipole_derivatives()
    hessian = atoms.calc.get_hessian()

    assert counter.calls == 1, f"the group was computed {counter.calls} times, not once"
    # Water: 3N-6 = 3 vibrations since 0.2.3, with the mode-indexed arrays shrinking together.
    assert frequencies.shape == (3,)
    # Modes are **columns**: 3N rows, one column per retained mode.
    assert modes.shape == (9, 3)
    assert intensities.shape == (3,)
    # Indexed by Cartesian coordinate rather than by mode, so this one keeps its 3N.
    assert derivatives.shape == (3, 9)
    # The raw Hessian is untouched by the projection: it is the second-derivative matrix, and
    # projecting it here would change what `get_hessian` means.
    assert hessian.shape == (9, 9)


def test_moving_the_atoms_invalidates_the_group(monkeypatch):
    """The cache must be per geometry, or it is not a cache but a bug."""
    counter = Counter(monkeypatch, "vibrations")
    atoms = water()
    atoms.calc = PM7(method="pm7-")

    first = atoms.calc.get_frequencies()
    assert counter.calls == 1
    atoms.calc.get_ir_intensities()
    assert counter.calls == 1, "no move, no recompute"

    atoms.positions[1, 0] += 0.05
    second = atoms.calc.get_frequencies()
    assert counter.calls == 2, "a moved atom must invalidate the group"
    assert not np.allclose(first, second), "the frequencies should have changed"


def test_an_energy_cycle_never_touches_the_vibrational_group(monkeypatch):
    """The group is lazy: an ordinary energy or forces run must not pay for a Hessian."""
    counter = Counter(monkeypatch, "vibrations")
    atoms = water()
    atoms.calc = PM7(method="pm7-")

    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_charges()
    atoms.get_dipole_moment()

    assert counter.calls == 0, "an energy cycle computed a Hessian it was never asked for"


def test_phonons_and_dfpt_are_cached_per_argument(monkeypatch):
    """The periodic methods take arguments, so they cache on the arguments too.

    `get_property` cannot key on a q path, so these keep their own cache. Asking for the same q
    path twice must not solve it twice, and asking for a different one must not return the first
    one's answer.
    """
    dfpt = Counter(monkeypatch, "dfpt")
    a = 3.2
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.76, 0.0, 0.0]],
        cell=[[a, 0, 0], [0, 12.0, 0], [0, 0, 12.0]],
        pbc=[True, False, False],
    )
    atoms.calc = PM7(method="pm7-", kpts=(3, 1, 1))

    first = atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])
    assert dfpt.calls == 1
    again = atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])
    assert dfpt.calls == 1, "the same q path was solved twice"
    assert again["frequencies_cm"] == first["frequencies_cm"]

    other = atoms.calc.get_dfpt([[1.0 / 3.0, 0.0, 0.0]])
    assert dfpt.calls == 2, "a different q path must actually be solved"
    assert other["frequencies_cm"] != first["frequencies_cm"]

    # And moving the atoms invalidates it.
    atoms.positions[1, 0] += 0.02
    atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])
    assert dfpt.calls == 3, "a moved atom must invalidate the q-path cache"


def test_orbitals_and_born_charges_are_cached(monkeypatch):
    orbitals = Counter(monkeypatch, "orbitals")
    atoms = water()
    atoms.calc = PM7(method="pm7-")
    atoms.calc.get_orbitals()
    atoms.calc.get_orbitals()
    assert orbitals.calls == 1, "orbitals were computed twice for one geometry"

    atoms.positions[2, 1] += 0.03
    atoms.calc.get_orbitals()
    assert orbitals.calls == 2
