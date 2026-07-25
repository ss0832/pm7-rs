# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for PM7-family methods (eV / Angstrom conventions)."""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install `pm7-rs-python[ase]` to use pm7_rs.ase.PM7.") from exc

from . import native


class PM7(Calculator):
    """PM7-family calculator.

    Parameters
    ----------
    charge : int
        Formal molecular charge (e); an integer (number of electrons removed/added).
    multiplicity : int
        Spin multiplicity 2S+1 (1 = singlet, 2 = doublet, …).
    method : str
        ``"pm7"``, ``"pm7-ts"``, ``"pm7-hh"``, ``"pm7-"`` (PM7-minus), …
    reference : str
        Spin treatment independent of ``multiplicity``: ``"auto"`` (RHF/UHF by shell),
        ``"rhf"`` (force RHF), or ``"uhf"`` (force UHF, e.g. a UHF singlet).

    All four inputs are honoured at every property evaluation. Energies are eV,
    forces eV/Å, the Hessian eV/Å², charges e, dipole e·Å.
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges", "dipole", "hessian"]

    def __init__(self, charge: int = 0, multiplicity: int = 1, method: str = "pm7",
                 reference: str = "auto", **kwargs):
        # ASE's base Calculator absorbs unknown keywords into `self.parameters`, so a stale
        # `variant=` would be silently ignored and quietly downgrade the run to plain PM7.
        # Fail loudly instead: this was renamed in 0.1.2 and the accepted values are unchanged.
        if "variant" in kwargs:
            raise TypeError(
                "PM7(variant=...) was renamed to PM7(method=...) in pm7-rs 0.1.2; "
                f"use PM7(method={kwargs['variant']!r}) instead."
            )
        super().__init__(**kwargs)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.method = method
        self.reference = reference

    def _numbers_positions(self, atoms):
        atoms = self.atoms if atoms is None else atoms
        return atoms.get_atomic_numbers(), atoms.get_positions()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        numbers = self.atoms.get_atomic_numbers()
        positions = self.atoms.get_positions()
        # A gradient call already performs the SCF and returns its energy. Reuse it for normal
        # ASE force/optimization steps instead of doing an extra single-point SCF first.
        if "forces" in properties:
            force = native.forces(numbers, positions, self.charge, self.multiplicity,
                                  self.method, self.reference)
            self.results["forces"] = np.asarray(force["forces_ev_per_angstrom"], dtype=float)
            self.results["energy"] = force["energy_ev"]
            self.results["free_energy"] = force["energy_ev"]
            self.results["heat_of_formation_kcal"] = force["heat_of_formation_kcal"]

        need_point = (
            "charges" in properties
            or "dipole" in properties
            or (("energy" in properties or "free_energy" in properties) and "forces" not in properties)
        )
        if need_point:
            point = native.single_point(numbers, positions, self.charge, self.multiplicity,
                                         self.method, self.reference)
            self.results["energy"] = point["energy_ev"]
            self.results["free_energy"] = point["energy_ev"]
            self.results["charges"] = np.asarray(point["charges"], dtype=float)
            self.results["dipole"] = np.asarray(point["dipole_debye"], dtype=float) * 0.2081943
            self.results["heat_of_formation_kcal"] = point["heat_of_formation_kcal"]
            self.results["unrestricted"] = point["unrestricted"]

        # The expensive Hessian is computed only when explicitly requested.
        if "hessian" in properties:
            h = native.hessian(numbers, positions, self.charge, self.multiplicity,
                               self.method, self.reference)
            self.results["hessian"] = np.asarray(h["hessian_ev_per_angstrom2"], dtype=float)

    def get_gradient(self, atoms=None) -> np.ndarray:
        """Energy **gradient** ∂E/∂x in eV/Å (= −forces), shape (N, 3). Forces themselves
        come from the standard ASE ``get_forces(atoms)`` on the base ``Calculator``."""
        numbers, positions = self._numbers_positions(atoms)
        grad = native.gradient(numbers, positions, self.charge, self.multiplicity,
                               self.method, self.reference)
        return np.asarray(grad["gradient_ev_per_angstrom"], dtype=float)

    def get_hessian(self, atoms=None) -> np.ndarray:
        """Analytic Cartesian **Hessian** in eV/Å², shape (3N, 3N). Declared in
        ``implemented_properties`` but computed **lazily** — only this call (or
        ``get_property("hessian")``) triggers it, never a normal energy/forces cycle. Routed
        through ASE's property machinery so the result is cached per geometry."""
        return self.get_property("hessian", atoms)


__all__ = ["PM7"]
