# SPDX-License-Identifier: GPL-3.0-or-later
"""Every property reachable from `native` is reachable from ASE, and vice versa.

The two surfaces are meant to mirror each other and differ only in units. Nothing enforced that,
so a feature could reach one and not the other and only a reader of both files would notice —
which is how `get_phonons`, `get_dfpt` and `get_born_charges` shipped raising `TypeError` on any
periodic `Atoms`: the tests exercised `native.*` directly and never went through the calculator.
These call the calculator.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator  # noqa: E402

from pm7_rs import native  # noqa: E402
from pm7_rs.ase import PM7  # noqa: E402

BOHR = 1.8897261254578281


def water():
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2400, 0.9278, 0.0]],
    )


def hydrogen_sulfide():
    """Has d functions, so the p–d dipole term is non-zero and the Molden file needs `[5D]`."""
    return Atoms(
        "SH2",
        positions=[[0.0, 0.0, 0.0], [1.34, 0.0, 0.0], [-0.33, 1.29, 0.0]],
    )


def hydrogen_chain():
    a = 3.2
    return Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.76, 0.0, 0.0]],
        cell=[[a, 0, 0], [0, 12.0, 0], [0, 0, 12.0]],
        pbc=[True, False, False],
    )


# --- dipole -----------------------------------------------------------------------------


def test_the_dipole_reaches_both_surfaces_with_its_terms_separate():
    atoms = hydrogen_sulfide()
    atoms.calc = PM7(method="pm7-")

    # ASE's standard property, in its own units.
    total = atoms.get_dipole_moment()
    assert total.shape == (3,)

    # And the breakdown, in Debye, which the standard property cannot express.
    parts = atoms.calc.get_dipole_breakdown()
    for key in (
        "dipole_debye",
        "dipole_point_charge_debye",
        "dipole_sp_hybrid_debye",
        "dipole_pd_hybrid_debye",
    ):
        assert key in parts, f"{key} missing from the ASE breakdown"

    # The terms must add up to the total, or the split is decorative.
    rebuilt = (
        parts["dipole_point_charge_debye"]
        + parts["dipole_sp_hybrid_debye"]
        + parts["dipole_pd_hybrid_debye"]
    )
    assert np.allclose(rebuilt, parts["dipole_debye"], atol=1e-9)
    # H2S has d functions, so the p–d term is the one that used to be missing entirely.
    assert np.abs(parts["dipole_pd_hybrid_debye"]).max() > 1e-6

    # The same numbers from `native`.
    point = native.single_point(
        atoms.get_atomic_numbers(), atoms.get_positions(), method="pm7-"
    )
    assert np.allclose(point["dipole_debye"], parts["dipole_debye"], atol=1e-9)


# --- IR ---------------------------------------------------------------------------------


def test_the_ir_spectrum_reaches_both_surfaces():
    atoms = water()
    atoms.calc = PM7(method="pm7-")

    spectrum = atoms.calc.get_ir_spectrum()
    assert set(spectrum) >= {
        "frequencies_cm",
        "ir_intensities_km_per_mol",
        "cartesian_modes",
        "dipole_derivatives_debye_per_angstrom",
    }
    # 3N-6 since 0.2.3: the rigid motions are projected out rather than filtered by magnitude.
    assert spectrum["frequencies_cm"].shape == (3,)
    assert spectrum["ir_intensities_km_per_mol"].shape == (3,)
    # Modes are **columns**: 3N rows, one column per retained mode.
    assert spectrum["cartesian_modes"].shape == (9, 3)
    # The individual accessors return the same arrays, from the same cached solve.
    assert np.allclose(atoms.calc.get_frequencies(), spectrum["frequencies_cm"])
    assert np.allclose(atoms.calc.get_ir_intensities(), spectrum["ir_intensities_km_per_mol"])

    native_out = native.vibrations(
        atoms.get_atomic_numbers(), atoms.get_positions(), method="pm7-", ir=True, modes=True
    )
    assert np.allclose(
        native_out["ir_intensities_km_per_mol"], spectrum["ir_intensities_km_per_mol"]
    )
    # Intensities are non-negative by construction; a negative one is a sign error.
    assert spectrum["ir_intensities_km_per_mol"].min() >= -1e-9


# --- Molden -----------------------------------------------------------------------------


def test_molden_reaches_both_surfaces_and_writes_a_file(tmp_path):
    atoms = hydrogen_sulfide()
    atoms.calc = PM7(method="pm7-")

    path = tmp_path / "orbitals.molden"
    text = atoms.calc.write_molden(path)
    assert path.read_text(encoding="utf-8") == text
    for section in ("[Molden Format]", "[Atoms] AU", "[GTO]", "[MO]", "[5D]"):
        assert section in text, f"missing {section}"
    assert "orthonormal AO basis" in text, "the NDDO caveat must travel with the file"

    direct = native.molden(atoms.get_atomic_numbers(), atoms.get_positions(), method="pm7-")
    assert direct == text


# --- DFPT and Born charges ---------------------------------------------------------------


def test_dfpt_and_born_charges_reach_both_surfaces():
    atoms = hydrogen_chain()
    atoms.calc = PM7(method="pm7-", kpts=(3, 1, 1))

    out = atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])
    assert out["converged"] is True
    assert len(out["frequencies_cm"][0]) == 6

    direct = native.dfpt(
        atoms.get_atomic_numbers(),
        atoms.get_positions(),
        atoms.cell[:],
        [[0.0, 0.0, 0.0]],
        method="pm7-",
        pbc=(True, False, False),
        kpoints=(3, 1, 1),
    )
    assert np.allclose(direct["frequencies_cm"][0], out["frequencies_cm"][0], atol=1e-6)

    # Born charges, and the acoustic sum rule as the check that they mean anything.
    born = atoms.calc.get_born_charges()
    assert np.asarray(born["born_charges"]).shape == (2, 3, 3)
    assert born["acoustic_residual"] < 1e-8


def test_phonons_reach_ase_even_with_a_k_mesh_on_the_calculator():
    """`PM7(kpts=...)` is the normal way to set up a periodic calculator, and it must not make
    `get_phonons` unusable.

    `native.phonons` refuses an explicit `kpoints` — correctly, since the supercell's Γ point *is*
    the mesh and a second sampling knob would contradict the first. But forwarding the
    calculator's own `kpts` into it turned that refusal on a user who had done nothing wrong. The
    method drops it and says so in its docstring; `kpts` still governs energies and forces from
    the same calculator, and still applies to `get_dfpt`.
    """
    atoms = hydrogen_chain()
    atoms.calc = PM7(method="pm7-", kpts=(3, 1, 1))
    out = atoms.calc.get_phonons([[0.0, 0.0, 0.0]], supercell=(2, 1, 1))
    assert len(out["frequencies_cm"][0]) == 6
    # The mesh still reaches the things it does apply to.
    assert atoms.get_potential_energy() < 0.0
    assert atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])["converged"] is True


def test_phonon_eigenvectors_reach_ase_as_complex_arrays():
    """`get_phonon_modes` hands back what a caller actually wants to do something with.

    The dicts carry `modes` as `(real, imag)` pairs, which is the right wire format and the wrong
    thing to have to assemble by hand every time. This is the same data through both routes, with
    the pairs joined and nothing else changed — and the check that it is the same data is that a
    chain's one acoustic mode moves every atom together, which no packing mistake would preserve.
    """
    atoms = hydrogen_chain()
    atoms.calc = PM7(method="pm7-")
    for route, extra in (("supercell", {"supercell": (2, 1, 1)}), ("dfpt", {})):
        freqs, modes, cartesian = atoms.calc.get_phonon_modes(
            [[0.0, 0.0, 0.0]], route=route, **extra
        )
        n = 3 * len(atoms)
        assert freqs[0].shape == (n,)
        assert modes[0].shape == (n, n) and np.iscomplexobj(modes[0])
        assert cartesian[0].shape == (n, n)
        # A chain has one acoustic branch — only its periodic direction has one.
        acoustic = int(np.argmin(np.abs(freqs[0])))
        displacement = cartesian[0][:, acoustic].real.reshape(-1, 3)
        assert np.abs(displacement - displacement[0]).max() < 1e-8, (
            f"{route}: the acoustic mode is not a uniform translation"
        )
    with pytest.raises(ValueError, match="route must be"):
        atoms.calc.get_phonon_modes([[0.0, 0.0, 0.0]], route="supercel")


def test_a_calculator_with_no_atoms_says_so_rather_than_crashing():
    calc = PM7(method="pm7-")
    with pytest.raises(ValueError) as excinfo:
        calc.get_orbitals()
    assert "no Atoms yet" in str(excinfo.value)

# --- the export lists themselves --------------------------------------------------------
#
# `dfpt`, `born_charges` and `molden` all shipped in v0.2.1 working through `pm7_rs.native` and
# missing from `native.__all__` *and* from the package namespace, so `pm7_rs.dfpt` raised
# `AttributeError`. Read alongside a `scope.md` that said perturbation theory "reaches no
# binding", that looked exactly like a feature which existed only in Rust. It is a two-line
# defect that no test could see, because nothing compared the export lists against the modules.


def test_native_dunder_all_lists_every_public_function():
    import inspect

    public = {
        name
        for name, value in vars(native).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", "") == native.__name__
    }
    missing = sorted(public - set(native.__all__))
    assert not missing, f"defined in pm7_rs.native but absent from its __all__: {missing}"
    extra = sorted(set(native.__all__) - public)
    assert not extra, f"listed in pm7_rs.native.__all__ but not defined there: {extra}"


def test_the_package_namespace_re_exports_everything_native_has():
    import pm7_rs

    missing = sorted(name for name in native.__all__ if not hasattr(pm7_rs, name))
    assert not missing, (
        "reachable as pm7_rs.native.<name> but not as pm7_rs.<name>: "
        f"{missing}. A user who finds a function in the docs and cannot import it from the "
        "package concludes the feature is not there."
    )
    for name in native.__all__:
        assert getattr(pm7_rs, name) is getattr(native, name), f"{name} is a different object"
        assert name in pm7_rs.__all__, f"{name} is re-exported but missing from pm7_rs.__all__"


def test_dfpt_and_born_charges_are_callable_from_the_package_namespace():
    """The specific claim that was false: perturbation theory is reachable without `native`."""
    import pm7_rs

    a = 4.03
    cell = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    numbers = [3, 9]
    positions = [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0]]

    out = pm7_rs.dfpt(numbers, positions, cell, [[0.0, 0.0, 0.0]], kpoints=(2, 2, 2))
    assert np.asarray(out["frequencies_cm"]).shape == (1, 6)
    born = pm7_rs.born_charges(numbers, positions, cell, kpoints=(2, 2, 2))
    assert np.asarray(born["born_charges"]).shape == (2, 3, 3)


# --- the two free energies --------------------------------------------------------------


def test_free_energy_is_the_internal_energy_without_smearing():
    atoms = water()
    atoms.calc = PM7()
    energy = atoms.get_potential_energy()
    free = atoms.get_potential_energy(force_consistent=True)
    assert free == energy, "with no entropy the two energies must be the same number"
    assert atoms.calc.results["free_energy"] == atoms.calc.results["energy"]


def test_smearing_separates_the_free_energy_from_the_internal_one():
    """ASE's force-consistent energy is the Mermin `E - TS`, not the internal energy.

    Through v0.2.1 the calculator set `free_energy` to `energy_ev`, so this difference was
    identically zero however hard the cell was smeared -- and an optimizer following the
    force-consistent energy was following a quantity the forces are not the gradient of.
    """
    a = 3.0
    atoms = Atoms(
        "Li2",
        positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc=True,
    )
    atoms.calc = PM7(kpts=(4, 4, 4), smearing=("fermi-dirac", 0.3))
    energy = atoms.get_potential_energy()
    free = atoms.get_potential_energy(force_consistent=True)
    entropy = atoms.calc.results["entropy_ev"]

    assert entropy < -1.0e-6, f"fixture does not smear: entropy {entropy}, so the test is vacuous"
    assert free == pytest.approx(energy + entropy, abs=1.0e-12)
    assert free < energy, "the entropy term lowers the free energy"

# --- LO-TO, which was reachable from nowhere ---------------------------------------------
#
# `DfptResult::frequencies_cm_lo_to` and `ForceConstants::frequencies_cm_lo_to` had **no caller
# anywhere in the repository** through v0.2.1 -- not in the bindings, not in either CLI, not in a
# test -- while `docs/properties.md` described them as the way to use the non-analytic term. What
# was bound was the raw `D^NA` matrix, leaving the caller to add it to the force constants,
# mass-weight and re-diagonalize by hand.


def rocksalt():
    a = 4.03
    return (
        [3, 9],
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0]],
        [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
    )


def _one_branch_moved(plain, split):
    """Indices where the LO-TO term changed a frequency by more than a wavenumber."""
    plain = np.sort(np.asarray(plain, dtype=float))
    split = np.sort(np.asarray(split, dtype=float))
    assert np.all(split >= plain - 1e-6), "the LO-TO term must not lower any branch"
    return np.flatnonzero(np.abs(split - plain) > 1.0)


def test_lo_to_frequencies_reach_both_phonon_routes_from_native():
    numbers, positions, cell = rocksalt()
    for name, out in [
        (
            "dfpt",
            native.dfpt(
                numbers,
                positions,
                cell,
                [[0.0, 0.0, 0.0]],
                lo_to_direction=(1, 0, 0),
                kpoints=(3, 3, 3),
            ),
        ),
        (
            "phonons",
            native.phonons(
                numbers,
                positions,
                cell,
                [[0.0, 0.0, 0.0]],
                lo_to_direction=(1, 0, 0),
                supercell=(2, 2, 2),
            ),
        ),
    ]:
        assert out["frequencies_cm_lo_to"] is not None, f"{name} returned no LO-TO frequencies"
        moved = _one_branch_moved(out["frequencies_cm"][0], out["frequencies_cm_lo_to"][0])
        assert moved.size == 1, f"{name}: expected one longitudinal branch to move, got {moved}"


def test_the_lo_to_split_is_absent_without_a_direction_and_away_from_gamma():
    numbers, positions, cell = rocksalt()
    plain = native.dfpt(numbers, positions, cell, [[0.0, 0.0, 0.0]], kpoints=(3, 3, 3))
    assert plain["frequencies_cm_lo_to"] is None, "no direction should mean no split"

    # Away from the zone centre the macroscopic field is already inside Phi(q) through the phased
    # Ewald sum, so the term does not belong there and the entry is None rather than a number.
    off = native.dfpt(
        numbers,
        positions,
        cell,
        [[0.0, 0.0, 0.0], [1 / 3, 0.0, 0.0]],
        lo_to_direction=(1, 0, 0),
        kpoints=(3, 3, 3),
    )
    assert off["frequencies_cm_lo_to"][0] is not None, "the zone centre should carry the term"
    assert off["frequencies_cm_lo_to"][1] is None, "q != 0 must not carry the term"


def test_lo_to_reaches_the_ase_calculator_too():
    numbers, positions, cell = rocksalt()
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    atoms.calc = PM7(kpts=(3, 3, 3))
    out = atoms.calc.get_dfpt([[0.0, 0.0, 0.0]], lo_to_direction=(1, 0, 0))
    moved = _one_branch_moved(out["frequencies_cm"][0], out["frequencies_cm_lo_to"][0])
    assert moved.size == 1

    # And that the cache keys on the direction rather than handing back the previous answer.
    without = atoms.calc.get_dfpt([[0.0, 0.0, 0.0]])
    assert without["frequencies_cm_lo_to"] is None

# --- the vibrational group on a periodic cell ---------------------------------------------
#
# `native.vibrations` took no periodic keywords at all through v0.2.1, and `PM7._vibrational`
# passed none, so `PM7(...).get_frequencies()` on a periodic `Atoms` computed the frequencies of
# those atoms **as an isolated molecule**. For a two-atom diamond cell that is a C2 diatomic --
# `-772, 654, 654` where the crystal's zone centre is `0, 0, 0, 1248, 1248, 1248`. No error, no
# warning, and a spectrum that looks like an answer.


def diamond_cell():
    a = 3.567
    return (
        [6, 6],
        [[0.0, 0.0, 0.0], [a / 4, a / 4, a / 4]],
        [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
    )


def test_periodic_frequencies_are_the_crystals_and_not_the_isolated_molecules():
    numbers, positions, cell = diamond_cell()
    isolated = np.sort(np.asarray(native.frequencies(numbers, positions)["frequencies_cm"]))

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    atoms.calc = PM7()
    periodic = np.sort(np.asarray(atoms.calc.get_frequencies()))

    # The zone centre of the crystal, which `phonons` reaches by an independent route.
    reference = np.sort(
        np.asarray(native.phonons(numbers, positions, cell, [[0, 0, 0]], supercell=(1, 1, 1))[
            "frequencies_cm"
        ][0])
    )
    # A millikayser. The three acoustic modes are numerical zeros that land anywhere within about
    # 1e-5 cm^-1 of zero depending on the route, and comparing those to 1e-6 would be comparing
    # noise; the optical modes agree to every printed digit.
    assert np.allclose(periodic, reference, atol=1e-3), (
        f"ASE gave {periodic}, the phonon route {reference}"
    )
    # And it is emphatically not the isolated-molecule answer, which is what it used to be.
    assert not np.allclose(periodic, isolated, atol=1.0), (
        "the periodic frequencies are still the isolated-molecule ones"
    )
    assert periodic[-1] > 1000.0, f"diamond's optical mode should be near 1250, got {periodic[-1]}"


def test_a_k_mesh_reaches_the_frequencies_too():
    numbers, positions, cell = diamond_cell()
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    atoms.calc = PM7(kpts=(3, 3, 3))
    ase_side = np.sort(np.asarray(atoms.calc.get_frequencies()))
    response = np.sort(
        np.asarray(native.dfpt(numbers, positions, cell, [[0, 0, 0]], kpoints=(3, 3, 3))[
            "frequencies_cm"
        ][0])
    )
    assert np.allclose(ase_side, response, atol=1e-3), f"{ase_side} vs {response}"
    # A k mesh must actually change the answer, or this test would pass on the Gamma path.
    gamma = np.sort(
        np.asarray(native.phonons(numbers, positions, cell, [[0, 0, 0]], supercell=(1, 1, 1))[
            "frequencies_cm"
        ][0])
    )
    assert not np.allclose(ase_side, gamma, atol=1.0), "the k mesh changed nothing"


def test_infrared_is_refused_for_a_periodic_cell_rather_than_answered_molecularly():
    numbers, positions, cell = diamond_cell()
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    atoms.calc = PM7()
    with pytest.raises(NotImplementedError, match="molecular"):
        atoms.calc.get_ir_intensities()
    with pytest.raises(NotImplementedError, match="molecular"):
        atoms.calc.get_dipole_derivatives()


def test_the_periodic_hessian_reaches_native_and_ase():
    numbers, positions, cell = diamond_cell()
    from_native = np.asarray(
        native.hessian(numbers, positions, cell=cell)["hessian_ev_per_angstrom2"]
    )
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    atoms.calc = PM7()
    from_ase = np.asarray(atoms.calc.get_hessian())
    assert from_native.shape == (6, 6)
    assert np.allclose(from_native, from_ase, atol=1e-8), "native and ASE Hessians disagree"


# --- surface parity, by enumeration ------------------------------------------------------
#
# The tests above check named pairs. Every gap this project has actually shipped got through
# because nothing *enumerated* a surface and asked what was missing from the next one: `pm7_rs.dfpt`
# was absent for a whole release, `vibrations` took no periodic keywords at any layer, and the
# command line went on returning an isolated molecule's frequencies for a crystal after the library
# had been fixed. The three tests here enumerate.


#: Every name in ``native.__all__``, mapped to the ASE calculator method that covers it — or to
#: ``None`` with the reason it deliberately has none.
#:
#: A name missing from this table fails the test. That is the point: adding an entry point without
#: deciding whether it belongs on the calculator is exactly how the last set of gaps happened, and
#: an explicit "no, because" is a decision while an absent key is an oversight.
ASE_ROUTES = {
    "single_point": "calculate",
    "gradient": "get_gradient",
    "forces": "get_gradient",
    "stress": "get_stress",
    "optimize": None,  # ASE's own optimizers drive the calculator; a second one would compete.
    "frequencies": "get_frequencies",
    "hessian": "get_hessian",
    "orbitals": "get_orbitals",
    "scf_stability": "get_scf_stability",
    "vibrations": "get_normal_modes",
    "Vibrations": "get_normal_modes",
    "phonons": "get_phonons",
    "band_structure": "get_band_structure",
    "dfpt": "get_dfpt",
    "born_charges": "get_born_charges",
    "dielectric_with_extent": "get_dielectric_with_extent",
    "divide_and_conquer": "calculate",  # via PM7(dandc=...)
    "molden": "write_molden",
    "polarizability": "get_polarizability",
    "static_dielectric": "get_static_dielectric",
    "dielectric_origin_sensitivity": "get_dielectric_origin_sensitivity",
    "berry_polarization": "get_berry_polarization",
    "finite_field": "get_finite_field",
}


def test_every_native_entry_point_has_a_decided_ase_route():
    unclassified = sorted(set(native.__all__) - set(ASE_ROUTES))
    assert not unclassified, (
        f"{unclassified} reached pm7_rs.native without anyone deciding whether the ASE "
        "calculator should expose it. Add it to ASE_ROUTES — with a method name, or with None "
        "and a comment saying why not."
    )
    stale = sorted(set(ASE_ROUTES) - set(native.__all__))
    assert not stale, f"ASE_ROUTES names {stale}, which no longer exist in pm7_rs.native"

    missing = []
    for name, method in ASE_ROUTES.items():
        if method is None:
            continue
        if not hasattr(PM7, method):
            missing.append(f"{name} -> PM7.{method} (does not exist)")
    assert not missing, "\n".join(missing)


def test_the_ase_band_structure_method_is_ours_and_not_ases_inherited_one():
    """``PM7.band_structure`` exists and is **not** the route to ``native.band_structure``.

    ASE's ``Calculator`` base class defines ``band_structure()``, which rebuilds one from a
    finished SCF through ``get_eigenvalues``/``get_ibz_k_points`` — so the name is present on the
    calculator whatever pm7-rs does. An enumeration test that only checked ``hasattr`` would
    report this surface as covered while no pm7-rs code was reachable at all.
    """
    assert hasattr(PM7, "band_structure"), "ASE's own method should still be inherited"
    assert PM7.band_structure is Calculator.band_structure, (
        "if pm7-rs ever overrides this name, the test below is checking the wrong thing"
    )
    assert "band_structure" not in ASE_ROUTES.values() or ASE_ROUTES["band_structure"] == (
        "get_band_structure"
    )


def test_the_two_command_lines_offer_the_same_modes():
    """The Rust binary and ``python -m pm7_rs`` are documented as interchangeable."""
    root = Path(__file__).resolve().parents[2]
    rust = set()
    for a, b in re.findall(r'^\s+"([a-z]+)"(?: \| "([a-z]+)")? => \w',
                           (root / "src/bin/pm7_rs.rs").read_text(encoding="utf-8"), re.M):
        rust.add(a)
        if b:
            rust.add(b)
    # The dispatch match also contains string arms that are not modes (flag values); keep only
    # what the Python side could plausibly mirror by intersecting with its own vocabulary plus
    # anything Python has that Rust lacks, which is what the assertion is about.
    from pm7_rs.__main__ import _MODES

    python = set(_MODES)
    rust &= python | {"hessian", "dielectric", "bands", "born", "dfpt", "phonons", "molden"}

    assert not (rust - python), (
        f"the Rust CLI has modes the Python CLI does not: {sorted(rust - python)}. "
        "The wheel does not ship the Rust binary, so a pip user has no other route to them."
    )


def test_the_two_command_lines_offer_the_same_flags():
    """Mode parity was enforced; flag parity was not, and had drifted in seven places.

    The wheel does not ship the Rust binary, so anything only the Rust CLI accepts is unreachable
    to a ``pip install`` user -- which was true of ``--scf-tolerance``, ``--max-scf``,
    ``--no-diis`` and ``--exchange-cutoff``, the four knobs someone reaches for when an SCF will
    not converge. In the other direction ``--pbc`` and ``--dipole-origin`` were Python-only. And
    ``--output`` was listed in the Rust CLI's own per-mode flag table and in its usage text
    without a parse arm behind either, so ``optimize x.xyz --output out.xyz`` failed with
    "unknown option `--output`" against a help text that had just named it.

    Aliases are the reason this compares sets of *accepted spellings* rather than of arms: the
    Rust CLI takes ``--output``, ``-o`` and ``--opt-output`` for one destination.
    """
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/bin/pm7_rs.rs").read_text(encoding="utf-8")

    # Every long flag any parse arm accepts, aliases included: an arm is
    # `"--a" | "-b" | "--c" => {`.
    rust = set()
    for arm in re.findall(r'^\s+((?:"-{1,2}[a-z0-9-]+"\s*\|?\s*)+)=>', source, re.M):
        rust.update(f for f in re.findall(r'"(--[a-z0-9-]+)"', arm))

    from pm7_rs.__main__ import _build_parser

    python = set()
    for action in _build_parser()._actions:
        python.update(o for o in action.option_strings if o.startswith("--"))
    python -= {"--help"}

    rust_only = rust - python
    python_only = python - rust
    assert not rust_only, (
        f"the Rust CLI accepts flags the Python CLI does not: {sorted(rust_only)}. "
        "The wheel ships no Rust binary, so a pip user cannot reach them."
    )
    assert not python_only, (
        f"the Python CLI accepts flags the Rust CLI does not: {sorted(python_only)}"
    )
