# SPDX-License-Identifier: GPL-3.0-or-later
"""The two v0.2.0 entry points that are not just a keyword on an existing call."""

from __future__ import annotations

import numpy as np
import pytest

from pm7_rs import native

ase = pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402


def alkane(n_carbon: int):
    """A linear alkane, as (numbers, positions)."""
    lines = []
    for i in range(n_carbon):
        x, y = i * 1.26, 0.44 * (i % 2)
        hy = y + (-0.5 if i % 2 == 0 else 0.5)
        lines += [("C", x, y, 0.0), ("H", x, hy, 0.89), ("H", x, hy, -0.89)]
    lines += [("H", -1.09, 0.0, 0.0), ("H", (n_carbon - 1) * 1.26 + 1.09, 0.44 * ((n_carbon - 1) % 2), 0.0)]
    numbers = [{"C": 6, "H": 1}[s] for s, *_ in lines]
    positions = [[x, y, z] for _, x, y, z in lines]
    return numbers, positions


def test_diamond_phonons_reproduce_the_raman_line():
    """The physical check: diamond's zone-centre optical mode is the 1332 cm⁻¹ Raman line, three
    times degenerate, with three acoustic modes at zero below it. Nothing here is fitted to that
    number, so hitting it exercises the force constants, the mass weighting and the units at
    once."""
    prim = bulk("C", "diamond", a=3.567, cubic=False)
    out = native.phonons(
        prim.get_atomic_numbers(),
        prim.get_positions(),
        prim.get_cell(),
        [[0, 0, 0], [0.5, 0.0, 0.0]],
        supercell=(2, 2, 2),
        acoustic_sum_rule=True,
        scf_tolerance=1e-8,
    )
    assert out["supercell"] == [2, 2, 2]
    assert out["acoustic_residual_ev_per_bohr2"] < 1e-10

    gamma = sorted(out["frequencies_cm"][0])
    assert all(abs(f) < 1e-2 for f in gamma[:3]), f"acoustic modes are not zero: {gamma}"
    optical = gamma[3:]
    assert max(optical) - min(optical) < 1.0, f"the optical branch is not degenerate: {gamma}"
    assert 1150.0 < optical[0] < 1500.0, f"the Raman mode is at {optical[0]:.1f} cm-1"

    # The zone boundary has to disperse away from the zone centre, or the interpolation is not
    # doing anything.
    boundary = sorted(out["frequencies_cm"][1])
    assert boundary[0] > 100.0, f"the acoustic branch did not disperse: {boundary}"


def _complex(pair):
    """A ``(real, imag)`` pair as one complex array."""
    return np.asarray(pair[0]) + 1j * np.asarray(pair[1])


@pytest.mark.parametrize("route", ["phonons", "dfpt"])
def test_the_polarization_vectors_come_back_and_mean_what_they_claim(route):
    """Both phonon routes return eigenvectors, and they are the right objects.

    They were computed on every call and discarded through 0.2.2, which left the frequency the only
    thing either route could tell you. Three claims here, each of which a different mistake would
    break: the columns are orthonormal (they diagonalize a Hermitian matrix, so a normalization
    applied to the wrong axis fails); ``cartesian_modes`` columns are unit length; and the three
    acoustic modes at Γ put the **same** displacement on both atoms, which is not a convention but
    the statement that translating the crystal costs no energy.
    """
    prim = bulk("C", "diamond", a=3.567, cubic=False)
    z, p, c = prim.get_atomic_numbers(), prim.get_positions(), prim.get_cell()
    if route == "phonons":
        out = native.phonons(z, p, c, [[0, 0, 0]], supercell=(2, 2, 2), scf_tolerance=1e-8)
    else:
        out = native.dfpt(z, p, c, [[0, 0, 0]], kpoints=(2, 2, 2), scf_tolerance=1e-8)

    e = _complex(out["modes"][0])
    u = _complex(out["cartesian_modes"][0])
    f = np.asarray(out["frequencies_cm"][0])
    n = 3 * len(z)
    assert e.shape == (n, n) and u.shape == (n, n)
    assert f.shape == (n,)

    gram = e.conj().T @ e
    assert np.abs(gram - np.eye(n)).max() < 1e-10, "the columns are not orthonormal"
    assert np.abs(1.0 - np.linalg.norm(u, axis=0)).max() < 1e-10, "columns are not unit length"

    # The three closest to zero, which for a cell at its minimum are the three lowest as well.
    acoustic = np.argsort(np.abs(f))[:3]
    for column in acoustic:
        displacement = u[:, column].real.reshape(-1, 3)
        spread = np.abs(displacement - displacement[0]).max()
        assert spread < 1e-8, (
            f"the acoustic mode at {f[column]:.4f} cm^-1 is not a uniform translation: "
            f"atoms differ by {spread:.2e}"
        )


def test_the_two_phonon_routes_report_the_same_modes():
    """A supercell Hessian and the perturbation solver, at a q the supercell holds exactly.

    Frequencies are what is comparable: an eigenvector is fixed only up to a phase, and inside a
    degenerate set only up to a rotation of it, so comparing the vectors themselves would be
    comparing arbitrary choices. Diamond's zone centre has a triplet, which is exactly such a set.
    """
    prim = bulk("C", "diamond", a=3.567, cubic=False)
    z, p, c = prim.get_atomic_numbers(), prim.get_positions(), prim.get_cell()
    q = [[0.5, 0.0, 0.0]]
    a = np.sort(
        native.phonons(z, p, c, q, supercell=(2, 2, 2), scf_tolerance=1e-8)["frequencies_cm"][0]
    )
    b = np.sort(native.dfpt(z, p, c, q, kpoints=(2, 2, 2), scf_tolerance=1e-8)["frequencies_cm"][0])
    assert np.abs(a - b).max() < 1e-2, f"the routes disagree:\n  supercell {a}\n  dfpt      {b}"


def test_divide_and_conquer_agrees_with_the_exact_scf():
    numbers, positions = alkane(24)
    d = native.divide_and_conquer(
        numbers, positions, buffer=9.0, core_size=8, scf_tolerance=1e-7
    )
    exact = native.single_point(numbers, positions)
    assert d["subsystems"] > 1, "the system was not actually partitioned"
    assert d["largest_subsystem"] < len(numbers), "a subsystem swallowed the whole molecule"
    assert abs(d["energy_ev"] - exact["energy_ev"]) < 0.05, (
        f"D&C {d['energy_ev']:.4f} eV vs exact {exact['energy_ev']:.4f} eV"
    )
    forces = np.asarray(d["forces_ev_per_angstrom"])
    assert forces.shape == (len(numbers), 3)
    assert np.isfinite(forces).all()
    # A molecule feels no net force, whatever the partitioning did.
    assert np.abs(forces.sum(axis=0)).max() < 5e-3, "the D&C forces do not sum to zero"


def test_a_wider_buffer_gets_closer():
    numbers, positions = alkane(16)
    exact = native.single_point(numbers, positions)["energy_ev"]
    errors = [
        abs(
            native.divide_and_conquer(
                numbers, positions, buffer=b, core_size=6, scf_tolerance=1e-7
            )["energy_ev"]
            - exact
        )
        for b in (4.0, 12.0)
    ]
    assert errors[1] < errors[0], f"widening the buffer made it worse: {errors}"


def test_divide_and_conquer_reports_a_stress_for_a_periodic_cell():
    prim = bulk("C", "diamond", a=3.567, cubic=False).repeat((2, 2, 2))
    out = native.divide_and_conquer(
        prim.get_atomic_numbers(),
        prim.get_positions(),
        cell=prim.get_cell(),
        pbc=(True, True, True),
        buffer=6.0,
        core_size=8,
        scf_tolerance=1e-6,
    )
    stress = np.asarray(out["stress_voigt"])
    assert stress.shape == (6,)
    assert np.isfinite(stress).all()


def test_phonons_need_a_cell_and_reject_a_bad_supercell():
    prim = bulk("C", "diamond", a=3.567, cubic=False)
    with pytest.raises(ValueError, match="three entries"):
        native.phonons(
            prim.get_atomic_numbers(),
            prim.get_positions(),
            prim.get_cell(),
            [[0, 0, 0]],
            supercell=(2, 2),
        )
