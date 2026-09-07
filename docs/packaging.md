# Packaging and release

`pm7-rs` ships as two things from one repository: a Rust crate (`pm7-rs` on crates.io) and a Python
wheel (`pm7-rs-python` on PyPI, imported as `pm7_rs`). The distribution name and the import name
differ, and deliberately: renaming either would break existing users for no gain. `pip install
pm7-rs-python`, then `import pm7_rs`.

## The version lives in one place

`Cargo.toml` holds it. `pyproject.toml` declares `dynamic = ["version"]` and maturin reads it from
the crate; `pm7_rs.__version__` reads it back with `importlib.metadata`. Three copies of a version
number means two of them are eventually wrong, so there is one.

To bump: edit `Cargo.toml`, run `cargo check` so `Cargo.lock` follows, and nothing else.

## What is in the sdist, and why it matters

The parameter tables are compiled in with `include_str!`, so `src/data/*.csv` must be *inside* the
sdist or the build fails at install time — on the user's machine, which is the worst place to find
out. `.github/workflows/ci.yml` therefore unpacks the sdist into a directory with no repository
around it, installs from that alone, and runs a single point. A green CI means someone who has only
the tarball can build it.

`Cargo.toml`'s `exclude` keeps `.mopac-source/` and the whole of `tools/oracle/` out of both
artifacts — the first is Apache-2.0 source used only for the fidelity audit, the second holds a
vendored MOPAC build alongside the oracle's own scratch output. Neither belongs in a published
package. The list also drops `*.zip` and the build leavings `**/__pycache__/`, `**/*.pyc` and
`**/*.pyd`, so a local `maturin develop` cannot leak a compiled extension into a source package.

## Licences travel with the binary, and that is a packaging job

Both shipped artifacts contain other people's code, and neither fact is visible from a source file:

* `src/data/pm7*.csv` are generated from MOPAC v23.2.5 (Apache-2.0) and pulled in with
  `include_str!`, so `_native.pyd` **contains** Apache-2.0-derived material and shipping it is a
  redistribution under §4(a) and §4(b).
* `cargo build` links statically, so the same binary is a copy of a substantial portion of ~120
  crates — MIT or Apache-2.0 almost throughout, both of which require the notice to travel with
  the copy.

`pyproject.toml`'s `license-files` is what puts all four files into the wheel's
`dist-info/licenses/`, and through 0.2.2 it listed two of them: a `pip install pm7-rs-python`
delivered MOPAC-derived tables with no MOPAC licence anywhere in it, and no crate notices at all.
This is exactly the kind of thing that decays silently — a file moves, an entry is not updated, and
every test still passes — so `tests/attribution.rs` checks the chain instead: every licence file in
the tree reaches `license-files`, and every crate reachable from `Cargo.lock` over non-dev edges
has an entry in `third_party/rust/NOTICES.md`.

Regenerate the crate notices after any dependency change:

```powershell
python tools/licenses/collect_rust_notices.py
```

**A binary-only archive has to carry them too.** An archive containing `pm7_rs_cli.exe` and nothing
else satisfies neither licence, so `LICENSE`, `THIRD_PARTY_NOTICES.md`, `third_party/mopac/LICENSE`
and `third_party/rust/NOTICES.md` go into the release archive alongside the executable.

## Wheels

`abi3-py310`, so **one wheel per platform covers Python 3.10 and everything after it**. Building
per minor version would multiply the matrix by five for byte-identical binaries.

Platforms built by `.github/workflows/wheels.yml`:

| OS | targets |
|---|---|
| Linux (manylinux) | x86_64, aarch64 |
| macOS | x86_64, arm64 |
| Windows | x64 |

plus the sdist. Everything goes through `twine check` before any publish job runs.

## Publishing

Publishing uses **Trusted Publishing** (OIDC), so no API token is stored in the repository or in
GitHub secrets. Configure the publisher once on the PyPI project page, pointing it at this
repository, the workflow file `wheels.yml`, and the environment name (`pypi` or `testpypi`).

### Release checklist

1. **Version** — bump `Cargo.toml`, run `cargo check`.
2. **Changelog** — add the section. Breaking changes go under their own heading and say what to
   change, not just what moved.
3. **Green** — `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
   `cargo test --release`, and `python -m pytest python/tests -q` against a fresh
   `maturin develop --release --features python`.
4. **Numbers** — re-run `tests/perf_report.rs` and `tests/dandc_scaling.rs` and update
   [`performance.md`](performance.md) if they moved. A documented timing that no longer matches
   the code is worse than no timing.
5. **Oracle** — `python tools/oracle/spmols.py`, `dmols.py`, `scf_residual.py`, `hbond_check.py`.
   Every delta should read `0.0000`.
6. **Dry runs** — `cargo publish --dry-run`, and `maturin sdist --out dist` followed by
   `twine check dist/*`.
7. **TestPyPI first** — run the `Wheels` workflow manually with `publish = testpypi`, then install
   from there into a clean environment and run a single point. This is the step that catches a
   metadata problem while it is still fixable; PyPI does not allow re-uploading a version.
8. **Tag** — `git tag vX.Y.Z && git push --tags`. The tag push builds and publishes to PyPI.
9. **crates.io** — `cargo publish`. Separate from the wheel, and it can happen either side of it.

### If something is wrong after publishing

PyPI versions are immutable: you cannot replace `X.Y.Z`, only yank it and publish `X.Y.Z+1`. That
is the reason step 7 exists.

## Local development

```bash
# Rust only
cargo test --release

# Python bindings into the current environment
maturin develop --release --features python
python -m pytest python/tests -q
```

The `python` feature is not default, so a plain `cargo build` does not pull in pyo3 and the crate
stays usable as a pure Rust dependency.

## Dependencies

Deliberately few. `faer` for the dense linear algebra (pure Rust — no LAPACK or BLAS to find at
build time, which is most of why a wheel builds cleanly on five platforms), `rayon` for the
parallelism, and `pyo3` only under the `python` feature. On the Python side, `numpy` (1.x and 2.x
are both exercised in CI) and, for the calculator, `ase`.
