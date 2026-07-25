# Scope and status

`pm7-rs` implements the PM7 semiempirical NDDO method and its PM7-family methods for
**molecular, non-periodic** systems, validated against the official MOPAC v23.2.5 executable.

## Milestones

All planned milestones are complete.

| Milestone | Status |
|---|---|
| M0: crate/API/method scaffold | complete |
| M1: MOPAC-derived PM7/TS/Sparkle tables and regeneration tool | complete |
| M2: s/p PM7 core SCF and oracle gate | complete |
| M3: dispersion, EH+ hydrogen-bond, PM7-HH post-SCF corrections | complete |
| M4: MNDO/d 9-AO kernel | complete |
| M5: Sparkle zero-AO +3-core execution path | complete |
| M6–M8: analytic gradients, Hessians, geometry optimization | complete |
| M9: PM7-TS / PM7-minus / PM7-HH oracle suites | complete |
| M10: packaging, Python/ASE bindings, documentation | complete |

## What is implemented

- **Elements** — the full minimal-valence s/p/d MNDO/d kernel: 1 AO (s), 4 AOs (s,p), 9 AOs
  (s,p,d), and 0 AOs for Sparkle lanthanide sites (Z 58–70, treated as +3 point cores).
- **Methods** — `Pm7`, `Pm7Ts`, `Pm7Minus` (corrections off), `Pm7Hh`, and `Pm7Sparkle`, selected
  through the `method` field/keyword on every API surface.
- **SCF** — RHF and UHF, with the spin treatment selectable independently of the multiplicity
  via `ScfReference::{Auto, Restricted, Unrestricted}`.
- **Post-SCF corrections** — PM6-DH-style D2 dispersion, the PM7 EH+ hydrogen-bond correction,
  and the PM7-HH H–H repulsion.
- **Derivatives** — forward-mode AD gradients and AD/CPHF(UCPHF) Hessians for both RHF and UHF,
  plus harmonic vibrational analysis and L-BFGS geometry optimization.
- **Interfaces** — Rust library, a `pm7_rs_cli` command-line tool, Python native bindings
  (`pm7_rs.native`), and an ASE calculator (`pm7_rs.ase.PM7`, eV/Å).

## Boundaries and known limitations

These are properties of the current implementation, stated explicitly rather than silently
approximated:

- **Non-periodic only.** There is no PBC/lattice path; all calculations are molecular.
- **No thermochemistry.** Harmonic frequencies are computed, but no entropy, enthalpy, or
  free-energy quantities are derived from them.
- **Hydrogen-bond topology is perceived from geometry**, and the EH+ derivatives are evaluated at
  *fixed* topology. A topology change is therefore a piecewise-smooth boundary on the potential
  energy surface, as it is in MOPAC.
- **Singular two-center frame orientations.** When a bond lies exactly on the singular axis of a
  local two-center rotation frame, the integral *value* stays analytic while that one pair's
  derivative comes from a localized symmetric finite difference.
- **Residual for highly d-populated transition metals.** A heat-of-formation residual of up to
  about 0.36 kcal/mol remains for some four-coordinate, highly d-populated transition-metal
  compounds such as TiF4. The Mulliken charges still match MOPAC, so this is a small
  d-integral-value difference at the *same* SCF solution — not an SCF-basin or scaling error.

PM7 is an empirical model. Agreement with a MOPAC implementation does not imply ab-initio
accuracy outside the model's parameterization domain.

See [theory.md](theory.md) for the formalism, and the [Rust](rust-api.md) / [Python](python-api.md)
API references for usage.
