# SPDX-License-Identifier: GPL-3.0-or-later
"""The package must work where the locale is not UTF-8.

This is a portability test, not a formatting one. A `pip install` on a machine whose preferred
encoding is `cp932`, `cp1252` or `C` is an ordinary situation, and two things break there if
nobody is watching:

* a printed non-ASCII character raises `UnicodeEncodeError` the moment stdout is **redirected**,
  because Python then encodes with `locale.getpreferredencoding()` rather than talking to the
  console directly. That makes it a bug you only see in a pipeline or a CI log, never
  interactively;
* a source or metadata file read without an explicit `encoding=` decodes with that same locale
  encoding and fails on any non-ASCII byte.

`subprocess.run(capture_output=True)` reproduces the first case exactly — captured output *is*
redirected output — so these tests exercise the failing path rather than approximating it.

Non-ASCII in comments and docstrings is fine and deliberately not policed: Python 3 source is
UTF-8 by definition, and the physics reads far better with `Å`, `Γ` and `∂` in it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

WATER = """3
water
O 0.0000 0.0000 0.0000
H 0.9584 0.0000 0.0000
H -0.2400 0.9278 0.0000
"""

# Every mode that prints something. `optimize` is included because it writes a geometry.
MODES = ["energy", "charges", "gradient", "forces", "optimize", "frequencies"]


def legacy_environment() -> dict[str, str]:
    """An environment that forces the pre-UTF-8 behaviour.

    `PYTHONUTF8=0` disables UTF-8 mode (which Python 3.15 turns on by default, and which would
    otherwise mask the very problem this file is about); `LC_ALL`/`LANG` pin the POSIX locale;
    `PYTHONIOENCODING` is deliberately **not** set, because setting it would paper over the
    default that real users get.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env.pop("PYTHONIOENCODING", None)
    return env


@pytest.fixture(scope="module")
def water(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("encoding") / "water.xyz"
    path.write_text(WATER, encoding="ascii")
    return str(path)


@pytest.mark.parametrize("mode", MODES)
def test_the_cli_runs_with_a_non_utf8_locale_and_redirected_output(mode, water):
    done = subprocess.run(
        [sys.executable, "-m", "pm7_rs", mode, water],
        capture_output=True,
        env=legacy_environment(),
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr
    # The bytes that actually reached the pipe must be ASCII: anything else is a character this
    # package would have failed to emit on a `cp932` machine.
    done.stdout.decode("ascii")


def test_the_json_output_is_ascii(water):
    done = subprocess.run(
        [sys.executable, "-m", "pm7_rs", "energy", water, "--json"],
        capture_output=True,
        env=legacy_environment(),
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    done.stdout.decode("ascii")


def test_importing_the_package_needs_no_utf8_locale():
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pm7_rs, pm7_rs.native, pm7_rs.__main__;"
            "from pm7_rs import native;"
            "print(native.single_point([8,1,1],"
            "[[0,0,0],[0.96,0,0],[-0.24,0.93,0]])['heat_of_formation_kcal'])",
        ],
        capture_output=True,
        env=legacy_environment(),
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert abs(float(done.stdout.decode("ascii")) + 57.78228) < 1.0e-4


def test_the_ase_calculator_imports_without_a_utf8_locale():
    pytest.importorskip("ase")
    done = subprocess.run(
        [sys.executable, "-c", "import pm7_rs.ase; print(pm7_rs.ase.PM7.implemented_properties)"],
        capture_output=True,
        env=legacy_environment(),
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    done.stdout.decode("ascii")


def test_no_runtime_string_in_the_package_is_non_ascii():
    """Comments and docstrings may be anything; *emitted* text may not.

    Checked lexically as well as end-to-end, because the subprocess tests above only cover the
    code paths they happen to run, and a rarely-taken error branch is exactly where a stray
    `Å` survives.
    """
    package = Path(__file__).resolve().parents[1] / "pm7_rs"
    offenders: list[str] = []
    for source in sorted(package.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        # Strip docstrings and comments, then look at what is left.
        stripped = _without_docstrings_and_comments(text)
        for number, line in enumerate(stripped.splitlines(), start=1):
            if any(ord(character) > 127 for character in line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, "non-ASCII in emitted text:\n" + "\n".join(offenders)


def _without_docstrings_and_comments(text: str) -> str:
    """Blank out comments and triple-quoted strings, keeping line numbers intact."""
    import io
    import tokenize

    out = text.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:  # pragma: no cover - a syntax error is a different test's job
        return text
    for token in tokens:
        is_docstring = token.type == tokenize.STRING and token.string.startswith(('"""', "'''"))
        if token.type != tokenize.COMMENT and not is_docstring:
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = out[row - 1]
            begin = start_col if row == start_row else 0
            finish = end_col if row == end_row else len(line)
            out[row - 1] = line[:begin] + " " * (finish - begin) + line[finish:]
    return "\n".join(out)
