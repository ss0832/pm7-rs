# SPDX-License-Identifier: GPL-3.0-or-later
"""The periodic-response surface: DFPT phonons, Born charges, and Molden output.

These are the v0.2.1 additions that reach Python. The physics is validated in the Rust suite
(`tests/born.rs`, `tests/dfpt.rs`); what is checked here is that the **binding** carries it
faithfully — the right keys, the right shapes, the right units, and a refusal where a refusal is
owed rather than a plausible number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pm7_rs import native

BOHR = 1.8897261254578281


def lif():
    """Rocksalt LiF: the textbook ionic crystal, where `Z*` is close to ±1."""
    a = 4.03 * BOHR
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    return [3, 9], [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0]], cell


def water():
    a = BOHR
    return [8, 1, 1], [[0.0, 0.0, 0.0], [0.96 * a, 0.0, 0.0], [-0.24 * a, 0.93 * a, 0.0]]


def test_born_charges_come_through_with_the_sum_rule_intact():
    numbers, positions, cell = lif()
    out = native.born_charges(numbers, positions, cell, method="pm7-", kpoints=(3, 3, 3))

    z = np.asarray(out["born_charges"])
    assert z.shape == (2, 3, 3), "one 3x3 tensor per atom"
    # Li is the cation; the two must be equal and opposite in a neutral two-atom cell.
    assert z[0][0][0] > 0.8, f"Z*(Li) should be near +1, got {z[0][0][0]}"
    assert abs(z[0][0][0] + z[1][0][0]) < 1e-8
    assert out["acoustic_residual"] < 1e-8
    assert out["converged"] is True

    eps = np.asarray(out["dielectric"])
    assert eps.shape == (3, 3)
    # A polarizable insulator: strictly above 1, and cubic so isotropic. `>= 1` would pass on a
    # dead solver returning the identity, which is exactly the bug this shape of test once missed.
    assert eps[0][0] > 1.0 + 1e-4, f"eps must exceed 1, got {eps[0][0]}"
    assert abs(eps[0][0] - eps[1][1]) < 1e-5

    # The raw response is defined in any dimension; the dielectric tensor is not.
    assert np.asarray(out["polarizability"]).shape == (3, 3)


def test_the_lo_to_term_is_opt_in_and_needs_a_direction():
    numbers, positions, cell = lif()
    without = native.born_charges(numbers, positions, cell, method="pm7-", kpoints=(2, 2, 2))
    assert without["lo_to_force_constants_ev_per_bohr2"] is None

    withq = native.born_charges(
        numbers, positions, cell, method="pm7-", kpoints=(2, 2, 2), lo_to_direction=(1, 0, 0)
    )
    matrix = withq["lo_to_force_constants_ev_per_bohr2"]
    assert matrix is not None
    real, imag = matrix
    assert np.asarray(real).shape == (6, 6), "3N x 3N for two atoms"
    # An outer product of real vectors: real, symmetric, positive semi-definite.
    assert np.allclose(imag, 0.0, atol=1e-12)
    assert np.allclose(real, np.transpose(real), atol=1e-10)
    assert min(real[i][i] for i in range(6)) > -1e-10


def test_dfpt_matches_the_supercell_at_a_commensurate_q():
    """The binding's own version of the Rust suite's decisive check.

    `q = 1/3` is commensurate with a 3x supercell, so `phonons` gives the exact answer there and
    `dfpt` must reproduce it — and unlike `q = 0` or `q = 1/2` the phases are genuinely complex,
    which is where the two used to disagree.
    """
    a = 3.2 * BOHR
    cell = [[a, 0.0, 0.0], [0.0, 12.0 * BOHR, 0.0], [0.0, 0.0, 12.0 * BOHR]]
    numbers = [1, 1]
    positions = [[0.0, 0.0, 0.0], [0.76 * BOHR, 0.0, 0.0]]
    q = [[1.0 / 3.0, 0.0, 0.0]]
    common = dict(method="pm7-", pbc=(True, False, False))

    # The supercell repeat and the k mesh are the same sampling seen from two sides: Gamma of a
    # 3x supercell is the 3-point mesh of the cell. So these two calls are the same calculation.
    reference = native.phonons(numbers, positions, cell, q, supercell=(3, 1, 1), **common)
    mine = native.dfpt(numbers, positions, cell, q, kpoints=(3, 1, 1), **common)
    assert mine["converged"] is True

    got = sorted(mine["frequencies_cm"][0])
    want = sorted(reference["frequencies_cm"][0])
    scale = max(1.0, max(abs(v) for v in want))
    assert all(abs(x - y) < 5e-2 * scale for x, y in zip(got, want)), f"{got} vs {want}"


def test_dfpt_returns_complex_force_constants_as_two_real_matrices():
    numbers, positions, cell = lif()
    out = native.dfpt(
        numbers, positions, cell, [[0.3, -0.15, 0.42]], method="pm7-", kpoints=(2, 2, 2)
    )
    real, imag = out["force_constants_ev_per_bohr2"][0]
    real, imag = np.asarray(real), np.asarray(imag)
    assert real.shape == imag.shape == (6, 6)
    # Hermitian: the real part symmetric, the imaginary part antisymmetric. This is the invariant
    # whose violation went unnoticed from v0.2.0 because the assembly symmetrized it away.
    assert np.allclose(real, real.T, atol=1e-9)
    assert np.allclose(imag, -imag.T, atol=1e-9)
    # And genuinely complex, or the check above is vacuous.
    assert np.abs(imag).max() > 1e-6


def test_a_diverged_response_raises_rather_than_returning_a_number():
    numbers, positions, cell = lif()
    with pytest.raises(Exception) as excinfo:
        native.dfpt(
            numbers,
            positions,
            cell,
            [[0.3, -0.15, 0.42]],
            method="pm7-",
            kpoints=(2, 2, 2),
            dfpt_max_iterations=1,
            dfpt_tolerance=1e-16,
        )
    message = str(excinfo.value)
    assert "response" in message.lower(), message


def test_molden_carries_its_own_caveat_and_both_basis_forms():
    numbers, positions = water()
    gto = native.molden(numbers, positions)
    for section in ("[Molden Format]", "[Title]", "[Atoms] AU", "[GTO]", "[MO]"):
        assert section in gto, f"missing {section}"
    assert "orthonormal AO basis" in gto, "the NDDO caveat must travel with the file"
    assert "<STO|STO-nG> =" in gto, "the file should state its own fit quality"
    # Water has no d functions, so declaring [5D] would be wrong.
    assert "[5D]" not in gto

    sto = native.molden(numbers, positions, basis="sto")
    assert "[STO]" in sto and "[GTO]" not in sto
    # The two sections use different units and no file may mix them.
    assert "[Atoms] Angs" in sto

    with pytest.raises(Exception):
        native.molden(numbers, positions, basis="not-a-basis")


def test_molden_refuses_a_periodic_system():
    numbers, positions, cell = lif()
    # `molden` takes no cell at all, so the refusal has to come from the shape of the API: there
    # is no way to ask for a periodic Molden file, which is the point.
    assert "cell" not in native.molden.__code__.co_varnames


def test_phonons_refuses_a_k_mesh_and_says_what_to_use():
    """`kpoints` cannot affect the supercell route, so it is refused rather than accepted.

    `force_constants` solves the ``supercell`` repeat at its Gamma point, and Gamma of an
    ``n1 x n2 x n3`` supercell *is* the ``n1 x n2 x n3`` mesh of the cell. Silently accepting
    ``kpoints`` would leave a caller believing a mesh had applied when the answer could not have
    depended on it — the same failure as dropping the argument, with better manners.
    """
    a = 3.2 * BOHR
    cell = [[a, 0.0, 0.0], [0.0, 12.0 * BOHR, 0.0], [0.0, 0.0, 12.0 * BOHR]]
    numbers = [1, 1]
    positions = [[0.0, 0.0, 0.0], [0.76 * BOHR, 0.0, 0.0]]
    common = dict(method="pm7-", pbc=(True, False, False))

    with pytest.raises(Exception) as excinfo:
        native.phonons(
            numbers, positions, cell, [[0, 0, 0]], supercell=(1, 1, 1), kpoints=(4, 1, 1), **common
        )
    message = str(excinfo.value)
    assert "supercell" in message and "dfpt" in message, message

    # `supercell` is the knob that does work, and it must change the answer.
    one = native.phonons(numbers, positions, cell, [[0, 0, 0]], supercell=(1, 1, 1), **common)
    three = native.phonons(numbers, positions, cell, [[0, 0, 0]], supercell=(3, 1, 1), **common)
    assert any(
        not math.isclose(x, y, abs_tol=1e-6)
        for x, y in zip(one["frequencies_cm"][0], three["frequencies_cm"][0])
    ), "a larger force-constant supercell changed nothing"
