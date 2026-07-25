<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Changelog

All notable changes to `pm7-rs` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-07-25

### Changed (breaking)

- Renamed the PM7-family selector from `variant` to `method` on every API surface, matching the
  terminology used by MOPAC itself (`method_pm7`, `method_pm7_hh`, …). The accepted values are
  unchanged (`"pm7"`, `"pm7-ts"`, `"pm7-"`, `"pm7-hh"`, `"pm7-sparkle"`, …).
  - Rust: `Pm7Variant` → `Pm7Method`, module `variant` → `method`, `Pm7Options::variant` →
    `Pm7Options::method`, `Pm7Parameters::variant(..)` → `Pm7Parameters::method(..)`, and the
    `Pm7Parameters::variant` field → `::method`.
  - Python: the `variant=` keyword becomes `method=` on `single_point`, `gradient`, `forces`,
    `hessian`, `frequencies`, and `optimize`; `single_point` echoes the key `"method"`.
  - ASE: `PM7(variant=...)` becomes `PM7(method=...)`, and the attribute is `calc.method`.
    Because ASE's base `Calculator` absorbs unknown keywords instead of rejecting them, a stale
    `variant=` would otherwise be silently ignored and quietly downgrade the run to plain PM7;
    it now raises a `TypeError` naming the replacement. The native functions already reject
    unknown keywords on their own.
  - CLI: `--variant` becomes `--method`; `--json` output reports `"method"`.

### Documentation

- Rewrote `docs/scope.md`: the milestone table now records M0–M10 as complete instead of listing
  the d-orbital, Sparkle, and post-SCF-correction work as pending, and it states the actual
  boundaries (non-periodic only, no thermochemistry, fixed-topology hydrogen-bond derivatives,
  the localized frame-singularity fallback, and the TiF4-class residual).
- Rewrote `docs/theory.md`: it described only the M0–M2 s/p core and declared d orbitals,
  Sparkles, and the corrections unimplemented. It now documents the s/p/d MNDO/d basis, the
  Hamiltonian terms, point-charge feathering, the MOPAC initial guess, the post-SCF corrections,
  the heat-of-formation assembly, and the AD/CPHF derivative scheme including the `Dual2N<27>`
  hydrogen-bond Hessian.

## [0.1.1] - 2026-07-22

### Fixed

- Included PM7 post-SCF correction derivatives in the public fixed-density validation gradient.
- Replaced the EH+ hydrogen-bond finite-difference gradient with first-order AD of the same
  fixed-topology energy expression.
- Rejected empty, non-finite, coincident-atom, invalid-tolerance, and invalid exchange-cutoff
  inputs before integral allocation; SCF failures now report their final density residual.
- Corrected README MOPAC reference values and removed corrupted text.

### Performance and memory

- Bounded transient Fock J/K contribution storage to 2048 atom pairs per batch.
- Based the automatic memory budget on currently available RAM and increased the conservative
  CPHF per-worker estimate.
- Capped the large-stack EH+ Hessian pool at four workers.
- Kept the release profile at Fat LTO with one codegen unit for maximum calculation throughput;
  OOM safeguards apply to electronic-structure calculations rather than compilation.
- Avoided a duplicate SCF in normal ASE force evaluations and added `free_energy` support.

### Documentation

- Added primary PM7, MOPAC, hydrogen-bond, ASE, and PySEQM references.

## [0.1.0] — 2026-07-21

First public release: a Rust-native implementation of the **PM7** semiempirical
NDDO method (Stewart, *J. Mol. Model.* **19**, 1 (2013)) and its PM7-family
methods, validated against MOPAC v23.2.5.

### Methods & elements
- Full **MNDO/d s/p/d NDDO kernel** — H through the heavy main-group and
  transition-metal PM7 elements (9-AO `spd` two-center integrals, one-center
  Slater–Condon integrals, d-orbital overlaps).
- Methods: **PM7**, **PM7-TS**, **PM7-minus** (corrections off), **PM7-HH**,
  and **Sparkle/PM7** (Ln(III) lanthanide sparkles).
- Post-SCF corrections — PM6-DH **dispersion** and the PM7 **hydrogen-bond**
  ("EH+") correction, both bit-exact vs MOPAC; PM7-HH H–H repulsion.
- RHF and UHF; RHF/UHF selectable independently of multiplicity.

### Properties
- Energies / heats of formation, **analytic gradients**, and **analytic
  Hessians** (forward-mode dual-number AD). The many-body H-bond ("EH+")
  correction's Hessian is analytic too, via multi-variable second-order AD
  (`Dual2N<27>`): each bond's geometric energy yields its exact ≤27×27 local
  second-derivative block in one pass (no finite differences; its gradient uses a
  fixed-topology central difference). With vibrational analysis.
- **L-BFGS** geometry optimization.

### Interfaces
- Rust library, a `pm7_rs_cli` command-line tool, and Python — native
  (`pm7_rs.native`) and ASE (`pm7_rs.ase.PM7`, eV/Å) — bindings via maturin.

### Numerics & performance
- **Point-charge feathering** (`l_feather`): every two-center NDDO integral
  (core–core, electron–core, two-electron) smoothly transitions to the exact
  point-charge value as atoms separate, exactly as MOPAC does for all PM7 runs.
  This makes heats of formation reproduce MOPAC PM7 to the printed 1e-5 kcal/mol
  across sp, d, transition-metal, and lanthanide-sparkle species, and makes all
  five methods (PM7, PM7-TS, PM7-minus, PM7-HH, Sparkle) bit-exact vs MOPAC.
- **MOPAC initial-density guess** (`moldat.F90`): the SCF guess smears the core
  charge over the sp orbitals as MOPAC does, so pm7-rs converges to MOPAC's SCF
  stationary point even for the ionic/covalent-**bistable** diatomics (BF, AsF,
  AlN) that otherwise settle in a different valid basin.
- Sparkle/PM7 lanthanide core–core pairs are zeroed as in MOPAC `switch.F90`, so
  every Ln–X interaction completes identically (fixes a Gd-only +13 kcal/mol
  outlier from its stray atomic pair parameters).
- The analytic gradient/Hessian is exact for bonds exactly on a Cartesian axis
  too: the singular pair's derivatives come from a finite difference of the
  correct-value integrals (`rotfix`), so no orientation loses the perpendicular
  force/curvature component.
- Linear algebra via **faer** (pure-Rust; no LAPACK/BLAS); hot loops
  parallelized with **rayon** bit-identically.
- 2018-CODATA constants matching MOPAC v23.2.5.
- Pre-flight **memory guard** (returns a clean error instead of OOM-crashing on
  oversized systems) and an optional smooth long-range-**exchange cutoff** for
  the analytic-Hessian CPHF (off by default → bit-identical).

### Provenance
- PM7 parameter tables generated from the pinned MOPAC v23.2.5 source
  (Apache-2.0); attribution in `THIRD_PARTY_NOTICES.md`. No fitted parameter is
  transcribed by hand.

### Known limitations
- A ≤0.36 kcal/mol heat-of-formation residual remains for four-coordinate
  transition-metal species with a highly populated d shell (e.g. TiF4). The
  Mulliken charges still match MOPAC, so it is a small d-integral-value
  difference at the same SCF solution, not an SCF-basin or scaling error.
