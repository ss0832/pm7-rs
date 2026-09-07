# SPDX-License-Identifier: GPL-3.0-or-later
"""The ASE calculator: molecules, periodic systems, stress conventions, and cell relaxation."""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from pm7_rs.ase import PM7  # noqa: E402


def water():
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.239987, 0.927846, 0.0]],
    )


def water_crystal():
    """Two waters in a cubic cell — polar, so the Ewald terms matter."""
    a = 5.6
    positions = []
    numbers = []
    for shift, flip in [(np.zeros(3), 1.0), (np.full(3, a / 2), -1.0)]:
        numbers += [8, 1, 1]
        positions += [
            shift,
            shift + np.array([0.9584 * flip, 0.0, 0.0]),
            shift + np.array([-0.24 * flip, 0.9278, 0.0]),
        ]
    return Atoms(numbers=numbers, positions=positions, cell=np.eye(3) * a, pbc=True)


def test_molecule_energy_forces_charges_dipole():
    atoms = water()
    atoms.calc = PM7()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    # Newton's third law: an isolated molecule feels no net force.
    assert np.abs(forces.sum(axis=0)).max() < 1e-8
    charges = atoms.get_charges()
    assert charges.shape == (3,)
    assert charges.sum() == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(atoms.get_dipole_moment()).all()
    assert "heat_of_formation_kcal" in atoms.calc.results


def test_periodic_energy_forces_and_stress_come_from_one_scf():
    atoms = water_crystal()
    atoms.calc = PM7()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.isfinite(energy)
    assert stress.shape == (6,), "ASE wants the Voigt 6-vector"
    assert np.abs(forces.sum(axis=0)).max() < 1e-7, "no net force on a periodic cell"
    assert atoms.calc.results["n_kpoints"] == 1


def test_stress_matches_a_finite_difference_of_the_energy():
    """The single most valuable check on the ASE layer.

    Sign and Voigt ordering are exactly the things that do not fail loudly: get either wrong and
    a cell relaxation walks the wrong way while every individual number still looks plausible.
    Differencing the energy under an applied strain pins both.
    """
    atoms = water_crystal()
    atoms.calc = PM7()
    stress = atoms.get_stress()
    volume = atoms.get_volume()
    h = 5e-4

    voigt_index = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (0, 2): 4, (0, 1): 5}
    for (i, j), slot in voigt_index.items():
        energies = []
        for sign in (+1.0, -1.0):
            eps = np.eye(3)
            eps[i, j] += 0.5 * sign * h
            eps[j, i] += 0.5 * sign * h
            strained = atoms.copy()
            strained.set_cell(atoms.get_cell() @ eps, scale_atoms=True)
            strained.calc = PM7()
            energies.append(strained.get_potential_energy())
        fd = (energies[0] - energies[1]) / (2.0 * h) / volume
        assert stress[slot] == pytest.approx(fd, abs=2e-4), (
            f"stress component {slot} (index {i},{j}): analytic {stress[slot]:.9f} "
            f"vs finite difference {fd:.9f}"
        )


def test_cell_relaxation_converges_and_lowers_the_stress():
    """A variable-cell relaxation only converges if the stress has the right sign and order.

    This is the integration test for the whole periodic derivative stack: ASE's filter feeds the
    stress straight into the optimizer, so a wrong sign diverges and a wrong Voigt order shears
    the cell instead of relaxing it.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    atoms = bulk("Si", "diamond", a=5.2)  # deliberately compressed
    atoms.calc = PM7(kpts=(2, 2, 2))
    initial_pressure = -np.mean(atoms.get_stress()[:3])

    optimizer = BFGS(FrechetCellFilter(atoms), logfile=None)
    optimizer.run(fmax=0.05, steps=25)

    final_pressure = -np.mean(atoms.get_stress()[:3])
    assert abs(final_pressure) < abs(initial_pressure), (
        f"cell relaxation did not reduce the pressure: {initial_pressure:.6f} -> "
        f"{final_pressure:.6f} eV/A^3"
    )
    assert atoms.get_volume() > 0.0
    assert np.isfinite(atoms.get_potential_energy())


def test_k_points_change_the_answer_and_are_reported():
    atoms = bulk("Si", "diamond", a=5.43)
    gamma = atoms.copy()
    gamma.calc = PM7()
    meshed = atoms.copy()
    meshed.calc = PM7(kpts=(3, 3, 3))
    e_gamma = gamma.get_potential_energy()
    e_mesh = meshed.get_potential_energy()
    assert np.isfinite(e_gamma) and np.isfinite(e_mesh)
    assert gamma.calc.results["n_kpoints"] == 1
    assert meshed.calc.results["n_kpoints"] > 1
    assert e_gamma != pytest.approx(e_mesh, abs=1e-6), (
        "a k mesh should change the energy of a two-atom primitive cell"
    )


def test_a_slab_is_treated_as_two_dimensional():
    a = 2.5
    atoms = Atoms(
        numbers=[5, 7],
        positions=[[0.0, 0.0, 0.0], [1.25, 0.7217, 0.0]],
        cell=[[a, 0, 0], [a / 2, a * 0.8660254, 0], [0, 0, 20.0]],
        pbc=[True, True, False],
    )
    atoms.calc = PM7()
    assert np.isfinite(atoms.get_potential_energy())
    stress = atoms.get_stress()
    # No lattice vector along the surface normal means no stress conjugate to it.
    assert abs(stress[2]) < 1e-10, "a slab must have no zz stress"
    assert abs(stress[3]) < 1e-10 and abs(stress[4]) < 1e-10


def test_a_slab_built_along_y_runs_instead_of_being_rejected():
    """``pbc=(True, False, True)`` is an ordinary ``Atoms``, and used to be refused.

    Through 0.2.2 this raised "pm7-rs needs the periodic directions first" and left reordering the
    cell to the user -- for a pattern ASE produces whenever someone builds a slab along *y*. The
    native layer now rotates the lattice vectors itself, so the calculator passes ``pbc`` straight
    through.

    The assertion is an equality rather than "it did not raise": rotating the lattice vectors is a
    relabelling, so the energy must be the one the reordered cell gives, to round-off.
    """
    atoms = water()
    atoms.set_cell([[10.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 10.0, 0.0]])
    atoms.set_pbc([True, False, True])
    atoms.calc = PM7()
    got = atoms.get_potential_energy()

    reordered = water()
    reordered.set_cell([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]])
    reordered.set_pbc([True, True, False])
    reordered.calc = PM7()
    assert got == pytest.approx(reordered.get_potential_energy(), abs=1.0e-9)


def test_the_renamed_variant_keyword_still_fails_loudly():
    with pytest.raises(TypeError, match="renamed"):
        PM7(variant="pm7-ts")


def test_molecular_hessian_through_the_calculator():
    atoms = water()
    atoms.calc = PM7()
    h = atoms.calc.get_hessian(atoms)
    assert h.shape == (9, 9)
    assert np.allclose(h, h.T, atol=1e-6)
