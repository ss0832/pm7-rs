# SPDX-License-Identifier: GPL-3.0-or-later
"""SCF stability and spin contamination, through the Python layer.

The Rust suite already tests the physics (``tests/stability.rs``). What these check is that the
whole chain reaches Python intact: the standalone call, the option on an ordinary entry point, the
switch of spin reference that a ``follow`` run performs, and the ``<S^2>`` that comes back with it.

The numbers are borrowed rather than invented. ``0.753039`` and ``2.002664`` are MOPAC's ``(S**2)``
for CH3 and O2 at PM7/UHF; ``104.204`` is two hydrogen atoms out of the PM7 parameter table; and a
broken-symmetry singlet of two separated H atoms has ``<S^2> = 1`` exactly, which follows from the
two determinants it is an equal mixture of.
"""

from __future__ import annotations

import numpy as np
import pytest

import pm7_rs
from pm7_rs import native

H2_Z = [1, 1]
CH3_Z = [6, 1, 1, 1]
CH3_R = [[0.0, 0.0, 0.0], [1.079, 0.0, 0.0], [-0.5395, 0.9345, 0.0], [-0.5395, -0.9345, 0.0]]
O2_Z = [8, 8]
O2_R = [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0]]

# MOPAC v23.2.5, `(S**2)` on these exact geometries.
MOPAC_CH3_S2 = 0.753039
MOPAC_O2_S2 = 2.002664
# 2 x 52.102 kcal/mol, PM7's heat of formation for a hydrogen atom.
TWO_H_ATOMS = 104.204


def h2(r: float) -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [0.0, 0.0, r]]


def test_the_standalone_call_reports_both_channels():
    """`scf_stability` measures without moving, and both channels have to be asked."""
    bound = native.scf_stability(H2_Z, h2(0.74))
    assert bound["analysed"] is True
    assert bound["lowest_ev"] > 0.0
    assert bound["lowest_triplet_ev"] > 0.0
    assert bound["unstable"] is False

    # Stretched: a minimum among closed-shell solutions, a saddle once the spins may differ. A
    # check that ran only the singlet channel would call this a minimum.
    stretched = native.scf_stability(H2_Z, h2(4.0))
    assert stretched["lowest_ev"] > 0.0
    assert stretched["lowest_triplet_ev"] < 0.0
    assert stretched["unstable"] is True

    # The energy it reports is the one it was handed, unmoved.
    assert stretched["energy_ev"] == pytest.approx(
        native.single_point(H2_Z, h2(4.0))["energy_ev"], abs=1e-12
    )


def test_the_top_level_alias_exists():
    """A function reachable only as `pm7_rs.native.x` reads as a private one."""
    assert pm7_rs.scf_stability is native.scf_stability


def test_following_the_instability_dissociates_h2_and_says_it_switched(capfd):
    """`follow` reaches the right answer, and reports that the model changed to get there."""
    plain = native.single_point(H2_Z, h2(4.0))
    assert plain["unrestricted"] is False
    assert plain["spin_squared"] is None
    assert plain["heat_of_formation_kcal"] - TWO_H_ATOMS > 100.0

    escaped = native.single_point(H2_Z, h2(4.0), stability="follow")
    assert escaped["unrestricted"] is True
    assert escaped["heat_of_formation_kcal"] == pytest.approx(TWO_H_ATOMS, abs=1.0)
    # An equal mixture of the singlet and the triplet: exactly 1, from the two determinants.
    assert escaped["spin_squared"] == pytest.approx(1.0, abs=1e-3)

    # The switch of spin reference is announced, because the answer is now a different model from
    # the one that was asked for.
    warned = capfd.readouterr().err
    assert "UNRESTRICTED" in warned, f"the RHF->UHF switch must be reported; stderr was:\n{warned}"

    # `check` measures the same thing without moving anything.
    checked = native.single_point(H2_Z, h2(4.0), stability="check")
    assert checked["unrestricted"] is False
    assert checked["energy_ev"] == pytest.approx(plain["energy_ev"], abs=1e-12)


def test_spin_squared_matches_mopac():
    """`<S^2>` against the two cases MOPAC prints it for."""
    methyl = native.single_point(CH3_Z, CH3_R, multiplicity=2)
    assert methyl["unrestricted"] is True
    assert methyl["spin_squared"] == pytest.approx(MOPAC_CH3_S2, abs=5e-6)

    dioxygen = native.single_point(O2_Z, O2_R, multiplicity=3)
    assert dioxygen["unrestricted"] is True
    assert dioxygen["spin_squared"] == pytest.approx(MOPAC_O2_S2, abs=5e-6)

    # `orbitals` reports it too, from the same solution.
    assert native.orbitals(CH3_Z, CH3_R, multiplicity=2)["spin_squared"] == pytest.approx(
        MOPAC_CH3_S2, abs=5e-6
    )


def test_a_stable_molecule_is_left_exactly_alone():
    """The property that makes the option safe: a well-behaved answer must not move."""
    water_z = [8, 1, 1]
    water_r = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    plain = native.single_point(water_z, water_r)
    followed = native.single_point(water_z, water_r, stability="follow")
    assert followed["energy_ev"] == plain["energy_ev"]
    assert followed["unrestricted"] is False


def test_the_optimizer_can_recheck_along_the_way():
    """`stability_every` reaches `OptOptions`, and a stable case is unaffected by it."""
    water_z = [8, 1, 1]
    water_r = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    plain = native.optimize(water_z, water_r, gtol=1e-3)
    watched = native.optimize(water_z, water_r, gtol=1e-3, stability_every=2)
    assert watched["heat_of_formation_kcal"] == pytest.approx(
        plain["heat_of_formation_kcal"], abs=1e-9
    )


def test_the_ase_calculator_carries_both():
    """`PM7(stability=...)` is accepted, forwarded, and reports through `results`."""
    ase = pytest.importorskip("ase")
    from pm7_rs.ase import PM7

    atoms = ase.Atoms("H2", positions=h2(4.0))
    atoms.calc = PM7(stability="follow")
    atoms.get_potential_energy()
    assert atoms.calc.results["unrestricted"] is True
    assert atoms.calc.results["spin_squared"] == pytest.approx(1.0, abs=1e-3)

    measured = PM7()
    measured.atoms = atoms
    found = measured.get_scf_stability(atoms)
    assert found["unstable"] is True
    assert found["lowest_triplet_ev"] < 0.0
