# SPDX-License-Identifier: GPL-3.0-or-later
"""The native Python API: molecules, periodic systems, and the keyword contract."""

from __future__ import annotations

import numpy as np
import pytest

import pm7_rs

WATER_Z = [8, 1, 1]
WATER_XYZ = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.239987, 0.927846, 0.0]]


def test_molecular_single_point_is_unchanged_by_periodic_support():
    """The molecular path must give the same numbers it always did.

    The reference is MOPAC's PM7 heat of formation for this water geometry, which the Rust suite
    also pins; repeating it here catches a Python-layer unit or ordering mistake that the Rust
    tests cannot see.
    """
    out = pm7_rs.single_point(WATER_Z, WATER_XYZ)
    assert out["converged"]
    assert out["heat_of_formation_kcal"] == pytest.approx(-57.7893, abs=1e-3)
    assert not out["unrestricted"]
    # A molecule gets no periodic keys at all, so their presence is a reliable discriminator.
    for key in ("stress", "stress_voigt", "n_kpoints", "cell_angstrom"):
        assert key not in out


def test_gradient_and_forces_are_negatives_of_each_other():
    g = pm7_rs.gradient(WATER_Z, WATER_XYZ)
    f = pm7_rs.forces(WATER_Z, WATER_XYZ)
    assert np.allclose(
        np.asarray(g["gradient_ev_per_angstrom"]),
        -np.asarray(f["forces_ev_per_angstrom"]),
    )
    assert g["energy_ev"] == pytest.approx(f["energy_ev"])


def test_a_cell_switches_the_same_call_to_a_periodic_calculation():
    big = np.eye(3) * 30.0
    molecular = pm7_rs.single_point(WATER_Z, WATER_XYZ)
    periodic = pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=big)
    assert periodic["converged"]
    assert periodic["periodicity"] == 3
    assert periodic["n_kpoints"] == 1
    # A 30 A box is effectively isolated.
    assert periodic["energy_ev"] == pytest.approx(molecular["energy_ev"], abs=5e-3)
    assert np.asarray(periodic["stress_voigt"]).shape == (6,)
    assert "pressure_gpa" in periodic


def test_orthorhombic_cell_shorthand_matches_the_full_matrix():
    a, b, c = 12.0, 13.0, 14.0
    full = pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=np.diag([a, b, c]))
    short = pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=[a, b, c])
    assert short["energy_ev"] == pytest.approx(full["energy_ev"], abs=1e-9)


def test_lower_dimensional_cells_run():
    chain_cell = [[3.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    wire = pm7_rs.single_point(
        [9, 1], [[0, 0, 0], [0.92, 0, 0]], cell=chain_cell, pbc=[True, False, False]
    )
    assert wire["periodicity"] == 1
    assert wire["converged"]
    # A 1-D cell has a length, not a volume, so no pressure is reported.
    assert "pressure_gpa" not in wire

    sheet_cell = [[2.5, 0.0, 0.0], [1.25, 2.165, 0.0], [0.0, 0.0, 1.0]]
    sheet = pm7_rs.single_point(
        [5, 7], [[0, 0, 0], [1.25, 0.7217, 0]], cell=sheet_cell, pbc=[True, True, False]
    )
    assert sheet["periodicity"] == 2
    assert sheet["converged"]


def test_k_points_fold_and_report_bands():
    out = pm7_rs.single_point(
        [9, 1],
        [[0, 0, 0], [0.92, 0, 0]],
        cell=[[3.2, 0, 0], [0, 1, 0], [0, 0, 1]],
        pbc=[True, False, False],
        kpoints=(6, 1, 1),
    )
    assert out["converged"]
    # Time-reversal symmetry must have folded the 6-point mesh.
    assert out["n_kpoints"] < 6
    bands = out["band_energies_ev"]
    assert len(bands) == out["n_kpoints"]
    assert out["fermi_ev"] is not None


def test_stress_needs_a_cell_and_returns_both_forms():
    with pytest.raises(Exception):
        pm7_rs.stress(WATER_Z, WATER_XYZ, cell=None)
    out = pm7_rs.stress(WATER_Z, WATER_XYZ, cell=np.eye(3) * 12.0)
    voigt = np.asarray(out["stress_voigt"])
    full = np.asarray(out["stress"])
    assert voigt.shape == (6,)
    assert full.shape == (3, 3)
    # Voigt order is [xx, yy, zz, yz, xz, xy].
    assert voigt[0] == pytest.approx(full[0, 0])
    assert voigt[1] == pytest.approx(full[1, 1])
    assert voigt[2] == pytest.approx(full[2, 2])
    assert voigt[3] == pytest.approx(full[1, 2])
    assert voigt[4] == pytest.approx(full[0, 2])
    assert voigt[5] == pytest.approx(full[0, 1])
    assert np.allclose(full, full.T), "stress must be symmetric"


def test_charged_periodic_cell_reports_its_background():
    out = pm7_rs.single_point(
        WATER_Z, WATER_XYZ, charge=1.0, multiplicity=2, cell=np.eye(3) * 14.0
    )
    assert out["converged"]
    assert out["background_ev"] < 0.0
    assert out["makov_payne_ev"] > 0.0
    assert sum(out["charges"]) == pytest.approx(1.0, abs=1e-8)


def test_open_shell_periodic_runs_unrestricted():
    methyl_z = [6, 1, 1, 1]
    methyl = [[0, 0, 0], [1.079, 0, 0], [-0.5395, 0.9344, 0], [-0.5395, -0.9344, 0]]
    out = pm7_rs.single_point(methyl_z, methyl, multiplicity=2, cell=np.eye(3) * 14.0)
    assert out["converged"]
    assert out["unrestricted"]


def test_invalid_keywords_are_rejected_rather_than_ignored():
    with pytest.raises(TypeError):
        pm7_rs.single_point(WATER_Z, WATER_XYZ, not_a_keyword=1)
    with pytest.raises(ValueError):
        pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        # kpoints without a cell is a mistake, not something to silently ignore.
        pm7_rs.single_point(WATER_Z, WATER_XYZ, kpoints=(2, 2, 2))
    with pytest.raises(ValueError):
        # `pbc` has to have three entries. Which *pattern* it holds is no longer a constraint --
        # a non-leading one is reordered rather than refused since 0.2.3, so asserting a refusal
        # for `[True, False, True]` here would now be asserting a bug.
        pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=np.eye(3) * 10.0, pbc=[True, False])
    with pytest.raises(Exception):
        pm7_rs.single_point(WATER_Z, WATER_XYZ, cell=np.eye(3) * 10.0, smearing=("nope", 0.1))


def test_a_periodic_hessian_is_the_periodic_one_and_not_the_molecular_one():
    """A cell must change the Hessian, not be accepted and ignored.

    This replaces a test that asserted `hessian(cell=...)` raised `TypeError`. The instinct behind
    it was exactly right — its comment read "refusing the keyword is better than accepting it and
    quietly returning the molecular Hessian of a periodic system" — but the refusal lived in this
    one wrapper, while `frequencies` and `vibrations` took no periodic keywords at all and so did
    precisely the feared thing: a periodic `Atoms` through the ASE calculator got the frequencies
    of its atoms as an isolated molecule, silently.

    So the property worth pinning is not the refusal. It is that a cell **arrives**.
    """
    molecular = np.asarray(pm7_rs.hessian(WATER_Z, WATER_XYZ)["hessian_ev_per_angstrom2"])

    def residual(edge):
        periodic = np.asarray(
            pm7_rs.hessian(WATER_Z, WATER_XYZ, cell=np.eye(3) * edge)[
                "hessian_ev_per_angstrom2"
            ]
        )
        assert periodic.shape == molecular.shape
        return float(np.abs(periodic - molecular).max())

    # Stated as a *decay* rather than an absolute bound, which is the form
    # `tests/pbc_equivalence.rs` uses for the same question about the energy: there is no
    # threshold worth guessing, and "the residual shrinks with the box" is the property that
    # separates a real image interaction from a leftover bug. A 24 A box still differs by about
    # 6e-3 eV/Bohr^2 through the R^-6 dispersion tail between images, which is physics.
    tight, roomy = residual(4.5), residual(24.0)
    assert tight > 1.0e-3, f"a 4.5 A box gave the molecular Hessian ({tight:.3e}): cell ignored"
    assert roomy < 0.05 * tight, (
        f"the box dependence did not decay: {tight:.3e} at 4.5 A, {roomy:.3e} at 24 A"
    )
    assert "frequencies" in pm7_rs.__all__


def test_molecular_hessian_and_frequencies_still_work():
    h = pm7_rs.hessian(WATER_Z, WATER_XYZ)
    matrix = np.asarray(h["hessian_ev_per_angstrom2"])
    assert matrix.shape == (9, 9)
    assert np.allclose(matrix, matrix.T, atol=1e-6)
    vib = pm7_rs.frequencies(WATER_Z, WATER_XYZ)
    freqs = np.asarray(vib["frequencies_cm"])
    # 3N-6. The old assertion was `size == 9` followed by "the last three are above 500", which is
    # the magnitude test 0.2.3 removes: it decided a mode was a rigid motion from how small its
    # frequency came out, where the projection decides it from the geometry.
    assert freqs.size == 3, f"water has three vibrations, got {freqs}"
    assert (freqs > 500.0).all(), "every returned mode is a genuine vibration"
    # The opt-out still hands back the unprojected set, for anyone diagnosing a Hessian.
    raw = np.asarray(
        pm7_rs.frequencies(WATER_Z, WATER_XYZ, projection="none")["frequencies_cm"]
    )
    assert raw.size == 9
    # `translations` is the middle setting: the acoustic sum rule and nothing else, so the three
    # rotations stay in. It is what a periodic system gets, and is reachable for a molecule too.
    partial = np.asarray(
        pm7_rs.frequencies(WATER_Z, WATER_XYZ, projection="translations")["frequencies_cm"]
    )
    assert partial.size == 6
