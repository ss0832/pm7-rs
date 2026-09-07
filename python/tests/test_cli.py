# SPDX-License-Identifier: GPL-3.0-or-later
"""The console script that `pip install` provides.

The Rust `pm7_rs_cli` binary is not inside the wheel, so someone who installed with pip gets this
one instead — and it is the surface with no type checking between it and the user. These run it
the way a user would: as a subprocess, reading its real output.

`pm7-rs` and `python -m pm7_rs` are the same entry point, so testing the module form covers both
without depending on whether the scripts directory happens to be on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

DIAMOND = """2
Lattice="0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0" \
Properties=species:S:1:pos:R:3 pbc="T T T"
C 0.000000 0.000000 0.000000
C 0.891750 0.891750 0.891750
"""

WATER = """3
water
O 0.0000 0.0000 0.0000
H 0.9584 0.0000 0.0000
H -0.2400 0.9278 0.0000
"""


BN_SHEET = """2
Lattice="2.510 0.0 0.0 -1.255 2.1738 0.0 0.0 0.0 20.0" \
Properties=species:S:1:pos:R:3 pbc="T T F"
B 0.000000 0.000000 0.000000
N 1.255000 0.724600 0.000000
"""


def run(*args):
    """Run the CLI as a subprocess and return `(returncode, stdout, stderr)`."""
    done = subprocess.run(
        [sys.executable, "-m", "pm7_rs", *args],
        capture_output=True,
        text=True,
        # `text=True` alone decodes with `locale.getencoding()`, which is the console code page on
        # Windows — cp932 on a Japanese install. The CLI writes UTF-8, so the first message
        # containing an en dash or a Greek letter made the *reader thread* die with
        # `UnicodeDecodeError` rather than failing the assertion it was meant to check. Naming the
        # encoding makes these tests say the same thing on every locale.
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=1800,
    )
    return done.returncode, done.stdout, done.stderr


@pytest.fixture(scope="module")
def diamond(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "diamond.xyz"
    path.write_text(DIAMOND, encoding="ascii")
    return str(path)


@pytest.fixture(scope="module")
def water(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "water.xyz"
    path.write_text(WATER, encoding="ascii")
    return str(path)


@pytest.fixture(scope="module")
def bn_sheet(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "bn.xyz"
    path.write_text(BN_SHEET, encoding="ascii")
    return str(path)


def test_the_console_script_is_declared_and_points_somewhere_real():
    """`pip install` must give a `pm7-rs` command, not just an importable module."""
    from importlib import metadata

    entries = metadata.entry_points(group="console_scripts")
    script = {e.name: e.value for e in entries}.get("pm7-rs")
    assert script == "pm7_rs.__main__:main", f"console script is {script!r}"
    # And the target has to resolve — a declared entry point to a missing function installs fine
    # and fails only when someone runs it.
    from pm7_rs.__main__ import main

    assert callable(main)


def test_a_molecular_single_point(water):
    code, out, err = run("energy", water)
    assert code == 0, err
    assert "-57.78" in out, out


def test_json_output_is_parseable(water):
    code, out, err = run("energy", water, "--json")
    assert code == 0, err
    payload = json.loads(out)
    assert payload["heat_of_formation_kcal"] == pytest.approx(-57.789, abs=1e-2)
    assert len(payload["charges"]) == 3


def test_a_periodic_single_point_reads_the_lattice_and_takes_a_k_mesh(diamond):
    code, out, err = run("energy", diamond, "--kpoints", "2", "2", "2", "--json")
    assert code == 0, err
    payload = json.loads(out)
    assert payload["n_kpoints"] == 8
    assert payload["fermi_ev"] < 0.0


def test_stress_reports_voigt_and_a_pressure(diamond):
    code, out, err = run("stress", diamond, "--kpoints", "2", "2", "2", "--json")
    assert code == 0, err
    voigt = json.loads(out)["stress_voigt"]
    assert len(voigt) == 6
    # Cubic symmetry: equal diagonal, vanishing shear. A Voigt ordering mistake breaks this.
    assert voigt[0] == pytest.approx(voigt[1], abs=1e-6)
    assert voigt[1] == pytest.approx(voigt[2], abs=1e-6)
    for shear in voigt[3:]:
        assert shear == pytest.approx(0.0, abs=1e-6)


def test_phonons_gives_three_zero_acoustic_modes_at_gamma(diamond):
    code, out, err = run(
        "phonons",
        diamond,
        "--supercell",
        "2",
        "2",
        "2",
        "--qpoints",
        "0,0,0",
        "--acoustic-sum-rule",
        "--json",
    )
    assert code == 0, err
    frequencies = sorted(json.loads(out)["frequencies_cm"][0])
    assert len(frequencies) == 6
    for acoustic in frequencies[:3]:
        assert abs(acoustic) < 1.0, frequencies
    # Diamond's optical mode; PM7 need not hit 1332 exactly but must be in the right decade.
    assert 1000.0 < frequencies[-1] < 1600.0, frequencies


def test_bands_walks_a_path_without_re_converging_on_it(diamond):
    code, out, err = run(
        "bands", diamond, "--kpoints", "2", "2", "2", "--qpoints", "0,0,0", "0.5,0,0", "--json"
    )
    assert code == 0, err
    payload = json.loads(out)
    assert len(payload["energies_ev"]) == 2
    assert len(payload["energies_ev"][0]) == 8  # two carbons, four AOs each
    assert payload["fermi_ev"] < 0.0


def test_divide_and_conquer_is_reachable(water):
    code, out, err = run("energy", water, "--dandc", "8.0")
    assert code == 0, err
    assert "divide and conquer" in out


def test_a_periodic_mode_on_a_molecule_says_what_to_do(water):
    code, _out, err = run("phonons", water)
    assert code != 0
    assert "needs a periodic cell" in err and "--cell" in err, err


def test_an_unknown_flag_is_rejected(water):
    code, _out, err = run("energy", water, "--kpoint", "2")
    assert code != 0
    assert "unrecognized arguments" in err or "invalid" in err, err


# The console script is the surface a `pip install` user actually gets.
def test_the_installed_console_script_runs_every_mode(water):
    """`pip install` must give a working `pm7-rs` command, not just `python -m pm7_rs`.

    The two share `main()`, but only the console script exercises the entry-point wiring: a wheel
    can install the module perfectly and still ship a `pm7-rs` that does not resolve.
    """
    script = shutil.which("pm7-rs")
    if script is None:
        pytest.skip("pm7-rs console script is not on PATH in this environment")
    for mode in ("energy", "charges", "orbitals"):
        done = subprocess.run(
            [script, mode, water], capture_output=True, text=True, timeout=1800
        )
        assert done.returncode == 0, f"{mode}: {done.stderr}"
        assert done.stdout.strip(), f"{mode} printed nothing"

    ir = subprocess.run(
        [script, "frequencies", water, "--ir"], capture_output=True, text=True, timeout=1800
    )
    assert ir.returncode == 0, ir.stderr
    assert "IR (km/mol)" in ir.stdout

    field = subprocess.run(
        [script, "energy", water, "--field", "0.5,0,0"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert field.returncode == 0, field.stderr
    plain = subprocess.run(
        [script, "energy", water], capture_output=True, text=True, timeout=1800
    )
    assert field.stdout != plain.stdout, "the field must change the answer"

# --- the modes that printed nothing ------------------------------------------------------
#
# `dfpt`, `born` and `molden` were never run through this CLI by any test. Two of the three
# produced **no output at all** without `--json`: `_print_human` had no branch for them, and the
# generic keys it falls back on -- `energy_ev`, `heat_of_formation_kcal`, `n_kpoints` -- are not
# in either result dict. The command exited 0 having written zero bytes, which is the failure
# mode a test that only checks the return code cannot see.


def test_dfpt_prints_frequencies_without_json(diamond):
    code, out, err = run("dfpt", diamond, "--kpoints", "2", "2", "2", "--qpoints", "0,0,0")
    assert code == 0, err
    assert out.strip(), "dfpt printed nothing at all"
    assert "cm^-1" in out
    assert "q = (" in out


def test_born_prints_charges_and_the_dielectric_tensor(diamond):
    code, out, err = run("born", diamond, "--kpoints", "2", "2", "2")
    assert code == 0, err
    assert out.strip(), "born printed nothing at all"
    assert "Born effective charges" in out
    assert "dielectric tensor" in out
    assert "acoustic residual" in out


def test_born_prints_the_lo_to_block_when_a_direction_is_given(diamond):
    code, out, err = run("born", diamond, "--kpoints", "2", "2", "2", "--lo-to", "1,0,0")
    assert code == 0, err
    assert "LO-TO force constants" in out


def test_molden_writes_a_file_and_says_so(water, tmp_path):
    target = tmp_path / "water.molden"
    code, out, err = run("molden", water, "--output", str(target))
    assert code == 0, err
    assert target.exists(), "molden mode wrote no file"
    text = target.read_text(encoding="utf-8")
    assert "[Atoms]" in text and "[MO]" in text


def test_dfpt_and_born_still_emit_json_when_asked(diamond):
    for mode in ("dfpt", "born"):
        args = [mode, diamond, "--kpoints", "2", "2", "2", "--json"]
        if mode == "dfpt":
            args += ["--qpoints", "0,0,0"]
        code, out, err = run(*args)
        assert code == 0, err
        payload = json.loads(out)
        assert payload, f"{mode} --json produced an empty object"

def test_dfpt_prints_the_lo_to_column_when_a_direction_is_given(diamond):
    """`--lo-to` reaches `dfpt` and `phonons`, not only `born`.

    The methods behind it -- `DfptResult::frequencies_cm_lo_to` and the pair on `ForceConstants` --
    had no caller anywhere in the repository through v0.2.1, so this is the first thing that runs
    them from a command line.
    """
    code, out, err = run(
        "dfpt", diamond, "--kpoints", "2", "2", "2", "--qpoints", "0,0,0", "--lo-to", "1,0,0"
    )
    assert code == 0, err
    assert "with LO-TO" in out, out


def test_phonons_prints_the_lo_to_column_when_a_direction_is_given(diamond):
    code, out, err = run("phonons", diamond, "--qpoints", "0,0,0", "--lo-to", "1,0,0")
    assert code == 0, err
    assert "with LO-TO" in out, out


def test_phonons_defaults_to_the_zone_centre_without_a_supercell(diamond):
    """`--supercell` is optional, and omitting it means 1x1x1 as it does on the Rust CLI.

    It used to raise "supercell must have three entries": the Python CLI passed `None`, which
    *overrides* `native.phonons`'s own `(1, 1, 1)` default rather than falling back to it. Every
    other test in this file passes the flag, so the default path had never been run.
    """
    code, out, err = run("phonons", diamond, "--qpoints", "0,0,0")
    assert code == 0, err
    assert "cm^-1" in out, out


def test_a_q_finer_than_the_mesh_warns(diamond):
    """The sampling limit is announced rather than returned silently.

    An `n x n x n` mesh cannot resolve `q << 1/n`: the response couples k with a k+q the sampling
    cannot tell from k. The solve still converges -- to the answer for a question the mesh could
    not pose -- and diamond on a 3^3 mesh at q = 1/160 comes back with acoustic modes near
    -3000 cm^-1. It used to come back with no indication at all.
    """
    code, out, err = run(
        "dfpt", diamond, "--kpoints", "3", "3", "3", "--qpoints", "0.00625,0,0"
    )
    assert code == 0, err
    assert "finer than half the k-mesh step" in err, err


def test_the_warning_can_be_silenced(diamond, monkeypatch):
    import os
    import subprocess
    import sys

    environment = dict(os.environ, PM7_QUIET="1")
    done = subprocess.run(
        [
            sys.executable, "-m", "pm7_rs", "dfpt", diamond,
            "--kpoints", "3", "3", "3", "--qpoints", "0.00625,0,0",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    assert "finer than half the k-mesh step" not in done.stderr


# --- the command line is a fourth layer, and it was missed ---------------------------------


def test_frequencies_on_a_cell_are_the_crystals_and_not_the_isolated_molecules(diamond):
    """`frequencies` on a periodic file must agree with `phonons` at the zone centre.

    v0.2.2 fixed `vibrations` dropping the cell in the PyO3 binding, in `native.py` and in the ASE
    calculator — and left this layer alone, because no test ran this mode on a periodic file. It
    went on answering with the frequencies of those atoms **as an isolated molecule**: a two-atom
    diamond cell came back as a C2 diatomic at -772, 654, 654 cm^-1 where the crystal's zone centre
    is 0, 0, 0, 1248, 1248, 1248. Plausible numbers, no warning, wrong physics.

    Comparing against `phonons` rather than against pinned values is the point: the two routes
    compute the zone centre differently, so agreement is evidence and a pinned list is only a
    record of what the code did the day it was written.
    """
    code, out, err = run("frequencies", diamond, "--json")
    assert code == 0, err
    got = sorted(json.loads(out)["frequencies_cm"])

    code, out, err = run("phonons", diamond, "--json")
    assert code == 0, err
    want = sorted(json.loads(out)["frequencies_cm"][0])

    assert len(got) == len(want) == 6
    for a, b in zip(got, want):
        assert abs(a - b) < 1.0e-3, f"frequencies {got} vs phonons {want}"
    # The optical branch is the part that moves if the cell is dropped: 654 vs 1248.
    assert max(want) > 1000.0, "fixture no longer distinguishes the two answers"


def test_infrared_on_a_cell_is_refused_by_name_rather_than_by_an_internal(diamond):
    """The refusal has to say what to do instead.

    This used to die on "dipole derivatives need the retained CPHF orbital response" — true, and
    useless: it names an internal rather than the fact that a crystal's dipole is not a function of
    its density. The ASE calculator refused it properly and the library did not, so every other
    surface got the internal message.
    """
    code, out, err = run("frequencies", diamond, "--ir")
    assert code != 0
    message = (out + err).lower()
    assert "born" in message, message
    assert "orbital response" not in message, message


def test_the_hessian_mode_exists_and_is_square(diamond):
    """The Rust CLI has had `hessian` since 0.2.0; the Python one did not."""
    code, out, err = run("hessian", diamond, "--json")
    assert code == 0, err
    rows = json.loads(out)["hessian_ev_per_angstrom2"]
    assert len(rows) == 6 and all(len(r) == 6 for r in rows)


def test_dielectric_refuses_a_slab_with_no_thickness_and_says_why(bn_sheet):
    code, out, err = run("dielectric", bn_sheet)
    assert code != 0
    message = out + err
    assert "extent" in message and "--slab-thickness" in message, message


def test_dielectric_reports_the_invariants_that_the_thickness_cannot_change(bn_sheet):
    """Doubling the assigned thickness must move `eps` and leave the sheet invariants alone.

    That is the whole reason they are reported: `eps` for a layer is only defined once someone has
    asserted where the material stops, and the two invariants are what survives the assertion.
    """
    first = {}
    for thickness in ("3.33", "6.66"):
        code, out, err = run(
            "dielectric", bn_sheet, "--slab-thickness", thickness, "--kpoints", "3", "3", "1",
            "--json",
        )
        assert code == 0, err
        first[thickness] = json.loads(out)

    a, b = first["3.33"], first["6.66"]
    assert a["extent_convention"] == "slab_thickness"
    assert abs(a["sheet_parallel_bohr"] - b["sheet_parallel_bohr"]) < 1e-9
    assert abs(a["sheet_perpendicular_bohr"] - b["sheet_perpendicular_bohr"]) < 1e-9
    assert abs(a["dielectric"][0][0] - b["dielectric"][0][0]) > 1e-3, (
        "eps must depend on the assigned thickness, or the invariants are not saying anything"
    )
