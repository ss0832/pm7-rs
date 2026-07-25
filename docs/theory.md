# PM7 implementation notes

This document describes the formalism as actually implemented. See [scope.md](scope.md) for the
implementation's boundaries and known limitations.

## Model

PM7 (Stewart, *J. Mol. Model.* **19**, 1 (2013)) is an NDDO semiempirical model built on a
minimal valence basis of Slater orbitals. Each atom contributes

| shell present | AOs | basis |
|---|---|---|
| s only | 1 | s |
| s, p | 4 | s, p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub> |
| s, p, d | 9 | + the five d functions (MNDO/d) |
| Sparkle lanthanide (Z 58–70) | 0 | +3 point core, no AOs |

The NDDO working basis is treated as orthonormal, so the SCF Fock problem is an **ordinary**
symmetric eigenproblem; the overlap matrix enters the resonance terms rather than acting as a
generalized-eigenproblem metric.

## Hamiltonian terms

- **One-center terms** — `U_ss`/`U_pp`/`U_dd`, the one-center two-electron integrals
  (`g_ss`, `g_sp`, `g_pp`, `g_p2`, `h_sp`), and for d elements the Slater–Condon-derived
  one-center d integrals of the MNDO/d formalism.
- **Resonance** — off-diagonal core terms are `H_µν = S_µν · (β_µ + β_ν)/2` with per-element
  `β_s`/`β_p`/`β_d`. There is no additional pairwise resonance factor.
- **Two-center two-electron integrals** — the NDDO multipole expansion with Klopman–Ohno
  screening; for d elements the full MNDO/d two-center machinery (local multipole integrals
  assembled in a local diatomic frame, then rotated into the molecular frame).
- **Electron–core attraction** — evaluated with the core Klopman–Ohno radii, using PM7's
  dedicated core radius parameter where an element defines one.
- **Core–core repulsion** — MOPAC's `ccrep` form: a pairwise `ALPB`/`XFAC`-scaled exponential,
  the special-cased O–H / C–H / N–H / C–C / Si–O forms, per-element Gaussian terms, and the
  unpolarizable-core short-range guard. Missing pairs complete from the diagonal average.

### Point-charge feathering

PM7 always enables MOPAC's `l_feather` path. As two atoms separate, every two-center integral
smoothly transitions to its exact point-charge limit:

```text
I_feathered = I·c + point·(1 - c)      (monopole/charge-charge integrals)
I_feathered = I·c                      (higher multipoles)
c = 1 - exp(-(r_Angstrom - 7)^2 · 0.22)   for r < 7 Angstrom, else 0
```

The switch is C¹ at 7 Å, so gradients and Hessians remain smooth. This applies to the core–core,
electron–core, and two-electron blocks alike, and is what makes heats of formation reproduce
MOPAC to the printed 1e-5 kcal/mol.

### SCF initial guess

The initial density follows MOPAC's own guess: the core charge is smeared over the sp orbitals
rather than using ground-state atomic occupations. This matters for systems with more than one
valid SCF solution — several polar diatomics (BF, AsF, AlN) otherwise converge to a different,
equally aufbau-valid basin than MOPAC's.

## Post-SCF corrections

Applied according to the selected method (`Pm7Minus` disables all of them):

- **Dispersion** — a D2-style `-C6/R^6` term with Fermi damping, Slater–Kirkwood `C6` combination
  and per-element `R0`; the carbon `C6` depends on the perceived coordination number.
- **Hydrogen bond (EH+)** — a purely *geometric* many-body term over each perceived
  donor–H···acceptor arrangement (angles, torsions, and distance dampings). Unlike the PM6-DH2
  form it carries no dependence on the SCF charges.
- **PM7-HH** — an additional short-range H–H repulsion, used only by the `Pm7Hh` method.

## Heat of formation

```text
Delta Hf = (E_electronic + E_core-core + E_corrections - sum(E_isol) + sum(E_heat)) * eV_to_kcal_per_mol
```

with 2018-CODATA constants matching MOPAC v23.2.5.

## Derivatives

The integral kernels are written **once**, generic over a `Scalar` trait, and instantiated at
three types:

| type | yields |
|---|---|
| `f64` | the energy |
| `Dual` | value + first derivatives (analytic gradient) |
| `Dual2` | value + first and second derivatives (analytic Hessian skeleton) |

- **Gradient** — forward-mode AD contracted against the converged density (no SCF re-runs, no
  Pulay terms, since the basis is orthonormal).
- **Hessian** — the `Dual2` skeleton plus the orbital response from a CPHF (RHF) / UCPHF (UHF)
  fixed-point solve, accelerated with Pulay DIIS. Because CPHF is a *linear* system, DIIS reaches
  the same solution in far fewer iterations.
- **Correction derivatives** — dispersion and the PM7-HH repulsion are pairwise and differentiate
  directly at `Dual2`. The hydrogen-bond term is many-body, so its Hessian uses a multi-variable
  second-order dual, `Dual2N<27>`: each bond couples at most nine atoms (27 Cartesian degrees of
  freedom), so one AD pass yields that bond's exact local second-derivative block, and the total
  is the sum over bonds.
- **Frame singularities** — a bond lying exactly on the singular axis of a local two-center
  rotation frame would make the forward-mode derivative of that rotation collapse. Those pairs
  (and only those) fall back to a symmetric finite difference of the correct-value integrals, so
  no orientation loses its perpendicular force or curvature component.

## Numerics

Dense linear algebra uses `faer` (pure Rust; no LAPACK/BLAS). Pair- and coordinate-level loops
are parallelized with `rayon` in a **bit-identical** way: work is split across threads but each
result accumulates in the original order, so the thread count never changes the numbers.
