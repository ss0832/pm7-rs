# SPDX-License-Identifier: GPL-3.0-or-later
"""Native PM7-family API (atomic units at the public boundary).

Every entry point accepts:

* ``charge``       — formal molecular charge (e).
* ``multiplicity`` — spin multiplicity 2S+1 (1 = singlet, 2 = doublet, …).
* ``method``      — ``"pm7"``, ``"pm7-ts"``, ``"pm7-hh"``, ``"pm7-"`` (PM7-minus)…
* ``reference``    — spin treatment, independent of ``multiplicity``:
  ``"auto"`` (RHF closed shell / UHF open shell), ``"rhf"``/``"r"`` (force RHF),
  ``"uhf"``/``"u"`` (force UHF, e.g. a UHF singlet).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _native


def _input(numbers, positions):
    numbers = [int(z) for z in np.asarray(numbers).reshape(-1)]
    positions = np.asarray(positions, dtype=float).reshape(len(numbers), 3).tolist()
    return numbers, positions


def single_point(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
                 method: str = "pm7", reference: str = "auto") -> dict:
    """Single point; positions in Angstrom. Returns atomic-unit energetics plus the
    accepted ``charge``/``multiplicity``/``reference`` and whether UHF was used."""
    numbers, positions = _input(numbers, positions)
    return _native.single_point(numbers, positions, float(charge), int(multiplicity), method, reference)


def gradient(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
             method: str = "pm7", reference: str = "auto") -> dict:
    """Energy and analytic **gradient** ∂E/∂x (``gradient_hartree_per_bohr`` and
    ``gradient_ev_per_angstrom``); positions in Angstrom."""
    numbers, positions = _input(numbers, positions)
    return _native.gradient(numbers, positions, float(charge), int(multiplicity), method, reference)


def forces(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
           method: str = "pm7", reference: str = "auto") -> dict:
    """Energy and analytic **forces** −∂E/∂x (``forces_hartree_per_bohr`` and
    ``forces_ev_per_angstrom``); positions in Angstrom."""
    numbers, positions = _input(numbers, positions)
    return _native.forces(numbers, positions, float(charge), int(multiplicity), method, reference)


def optimize(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
             method: str = "pm7", reference: str = "auto") -> dict:
    """Optimize a structure on the requested PM7-family surface."""
    numbers, positions = _input(numbers, positions)
    return _native.optimize(numbers, positions, float(charge), int(multiplicity), method, reference)


def frequencies(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
                method: str = "pm7", reference: str = "auto") -> dict:
    """Harmonic frequencies (cm⁻¹) at the supplied geometry."""
    numbers, positions = _input(numbers, positions)
    return _native.frequencies(numbers, positions, float(charge), int(multiplicity), method, reference)


def hessian(numbers: Sequence[int], positions, charge: float = 0.0, multiplicity: int = 1,
            method: str = "pm7", reference: str = "auto") -> dict:
    """Analytic Cartesian **Hessian** (3N×3N) as ``hessian_hartree_per_bohr2`` (atomic
    units) and ``hessian_ev_per_angstrom2``; positions in Angstrom."""
    numbers, positions = _input(numbers, positions)
    return _native.hessian(numbers, positions, float(charge), int(multiplicity), method, reference)
