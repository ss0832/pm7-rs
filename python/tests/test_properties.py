# SPDX-License-Identifier: GPL-3.0-or-later
"""The v0.2.1 property stack through the Python layer: field, dipole, orbitals, IR, laziness.

The laziness tests are the interesting ones. "Computed on demand and cached" is easy to claim and
easy to get wrong in a way nobody notices — a second call that silently recomputes just makes
things slow, not wrong — so they count the calls into the extension rather than trusting the
docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

import pm7_rs
from pm7_rs import native

WATER_Z = [8, 1, 1]
WATER_R = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
H2S_Z = [16, 1, 1]
H2S_R = [[0.0, 0.0, 0.0], [1.34, 0.0, 0.0], [-0.35, 1.29, 0.0]]

# MOPAC v23.2.5 on these exact geometries.
MOPAC_WATER_HOF = -57.78228
MOPAC_WATER_FIELD_HOF = -54.78395
MOPAC_WATER_DIPOLE = [1.312, 1.695, 0.000]
MOPAC_H2S_DIPOLE = [1.109, 1.443, 0.000]


def test_the_external_field_reaches_the_engine_and_matches_mopac():
    without = native.single_point(WATER_Z, WATER_R)
    assert abs(without["heat_of_formation_kcal"] - MOPAC_WATER_HOF) < 1.0e-4

    with_field = native.single_point(WATER_Z, WATER_R, field=[0.5, 0.0, 0.0])
    assert abs(with_field["heat_of_formation_kcal"] - MOPAC_WATER_FIELD_HOF) < 1.0e-4
    assert with_field["field_ev"] is not None
    assert without["field_ev"] is None


def test_a_zero_field_is_indistinguishable_from_no_field():
    a = native.single_point(WATER_Z, WATER_R)
    b = native.single_point(WATER_Z, WATER_R, field=[0.0, 0.0, 0.0])
    assert a["energy_ev"] == b["energy_ev"]


def test_a_bad_field_is_rejected():
    with pytest.raises(ValueError):
        native.single_point(WATER_Z, WATER_R, field=[1.0, 2.0])
    with pytest.raises(ValueError):
        native.single_point(WATER_Z, WATER_R, field=[float("nan"), 0.0, 0.0])


def test_the_dipole_breakdown_matches_mopac_including_the_pd_term():
    water = native.single_point(WATER_Z, WATER_R)
    assert np.allclose(water["dipole_debye"], MOPAC_WATER_DIPOLE, atol=2.0e-3)
    # No d orbitals, so no p-d term at all.
    assert np.allclose(water["dipole_pd_hybrid_debye"], 0.0)

    h2s = native.single_point(H2S_Z, H2S_R)
    assert np.allclose(h2s["dipole_debye"], MOPAC_H2S_DIPOLE, atol=2.0e-3)
    # Sulfur has d orbitals, and the term is large: about 1 D, not a rounding correction.
    assert np.linalg.norm(h2s["dipole_pd_hybrid_debye"]) > 0.5

    # The three parts add up to the reported total.
    total = (
        np.asarray(h2s["dipole_point_charge_debye"])
        + np.asarray(h2s["dipole_sp_hybrid_debye"])
        + np.asarray(h2s["dipole_pd_hybrid_debye"])
    )
    assert np.allclose(total, h2s["dipole_debye"], atol=1.0e-12)


def test_a_charged_species_is_recentred_like_mopac():
    hydroxide = dict(numbers=[8, 1], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], charge=-1.0)
    default = native.single_point(**hydroxide)
    raw = native.single_point(**hydroxide, dipole_origin="coordinates")
    assert not np.allclose(default["dipole_debye"], raw["dipole_debye"])
    assert not np.allclose(default["dipole_origin_bohr"], 0.0)
    # A neutral molecule cannot see the setting at all.
    a = native.single_point(WATER_Z, WATER_R, dipole_origin="coordinates")
    b = native.single_point(WATER_Z, WATER_R, dipole_origin="com")
    assert a["dipole_debye"] == b["dipole_debye"]


def test_an_unknown_dipole_origin_is_rejected():
    with pytest.raises(ValueError, match="dipole_origin"):
        native.single_point(WATER_Z, WATER_R, dipole_origin="middle")


def test_orbitals_carry_labels_and_both_spin_channels():
    out = native.orbitals(WATER_Z, WATER_R)
    nao = len(out["ao_labels"])
    assert nao == 6  # O(s,px,py,pz) + 2 H(s)
    assert out["ao_labels"][0] == "1 O s"
    assert np.asarray(out["mo_coefficients"]).shape == (nao, nao)
    assert sum(out["occupations"]) == pytest.approx(8.0)
    assert out["orbital_source"] == "molecular"
    assert out["mo_energies_beta_ev"] is None if "mo_energies_beta_ev" in out else True

    methyl = native.orbitals(
        [6, 1, 1, 1],
        [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.94, 0], [-0.54, -0.94, 0]],
        multiplicity=2,
        reference="uhf",
    )
    assert methyl["unrestricted"]
    assert methyl["mo_energies_beta_ev"] is not None
    assert sum(methyl["occupations"]) == pytest.approx(4.0)
    assert sum(methyl["occupations_beta"]) == pytest.approx(3.0)


def test_the_coefficients_are_orthonormal():
    out = native.orbitals(WATER_Z, WATER_R)
    c = np.asarray(out["mo_coefficients"])
    assert np.allclose(c.T @ c, np.eye(c.shape[0]), atol=1.0e-10)


def test_vibrations_returns_the_ir_spectrum():
    """Water returns **3N-6 = 3** frequencies, and every mode-indexed array shrinks with them.

    Not 9. Since 0.2.3 the translations and rotations are projected out geometrically, so they are
    not in the array to be filtered. The old shape assertion said 9 and the old count assertion
    said "three of them are above 500 cm-1", which is the magnitude heuristic this release exists
    to remove: it asks "is this number small", which has no right answer, in place of "is this
    displacement a rigid motion", which does.

    The alignment is the part worth pinning. `frequencies_cm`, `ir_intensities_km_per_mol` and
    `mode_dipole_derivatives` are indexed by mode and must shrink *together*; if one kept 9 entries
    the intensities would silently belong to different modes than the frequencies.
    `dipole_derivatives_e` is indexed by **Cartesian coordinate**, not by mode, so it stays 3x3N.
    """
    out = native.vibrations(WATER_Z, WATER_R, ir=True, modes=True)
    frequencies = np.asarray(out["frequencies_cm"])
    intensities = np.asarray(out["ir_intensities_km_per_mol"])
    assert frequencies.shape == (3,), "water has 3N-6 = 3 vibrations"
    assert intensities.shape == (3,)
    assert np.asarray(out["mode_dipole_derivatives"]).shape == (3, 3)
    # 3 x 3N: one row per Cartesian dipole component, one column per nuclear coordinate. The
    # projection does not touch this one, and a shrunk one would mean the wrong thing shrank.
    assert np.asarray(out["dipole_derivatives_e"]).shape == (3, 9)
    # Every returned mode is a genuine vibration, so every one has a finite intensity -- no
    # filtering step, which is the point.
    assert (frequencies > 500.0).all(), f"a rigid motion survived the projection: {frequencies}"
    assert (intensities > 0.0).all()


def test_the_dipole_derivative_sum_rule_holds_through_python():
    out = native.vibrations(WATER_Z, WATER_R, ir=True)
    d = np.asarray(out["dipole_derivatives_e"]).reshape(3, 3, 3)
    # Sum over atoms: zero for a neutral molecule, exactly.
    assert np.allclose(d.sum(axis=1), 0.0, atol=1.0e-8)


class _Counter:
    """Wrap `native.vibrations` and count how often it is actually called."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = native.vibrations

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(native, "vibrations", counted)


def test_constructing_vibrations_computes_nothing(monkeypatch):
    counter = _Counter(monkeypatch)
    pm7_rs.Vibrations(WATER_Z, WATER_R)
    assert counter.calls == 0


def test_many_attribute_accesses_cost_one_calculation(monkeypatch):
    counter = _Counter(monkeypatch)
    v = pm7_rs.Vibrations(WATER_Z, WATER_R)
    _ = v.frequencies_cm
    _ = v.ir_intensities_km_per_mol
    _ = v.hessian_ev_per_angstrom2
    _ = v.modes
    _ = v.dipole_derivatives
    assert counter.calls == 1, "the group must be computed once and cached"


def test_the_orbital_response_is_opt_in_and_says_so():
    v = pm7_rs.Vibrations(WATER_Z, WATER_R)
    with pytest.raises(AttributeError, match="orbital_response=True"):
        _ = v.orbital_response

    wanted = pm7_rs.Vibrations(WATER_Z, WATER_R, orbital_response=True)
    u = wanted.orbital_response
    # (3N, n_vir, n_occ): water has 9 degrees of freedom, 4 occupied and 2 virtual orbitals.
    assert u.shape == (9, 2, 4)


def test_the_lazy_wrapper_agrees_with_the_plain_function():
    plain = native.vibrations(WATER_Z, WATER_R, ir=True, modes=True)
    lazy = pm7_rs.Vibrations(WATER_Z, WATER_R)
    assert np.allclose(lazy.frequencies_cm, plain["frequencies_cm"])
    assert np.allclose(lazy.ir_intensities_km_per_mol, plain["ir_intensities_km_per_mol"])
