<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Rust API

`pm7-rs` is a Rust-native implementation of the **PM7** semiempirical NDDO method family
(PM7, PM7-TS, PM7-minus, PM7-HH, Sparkle/PM7), with a shared RHF/UHF SCF, fully
analytic gradients and Hessians (forward-mode dual numbers; the many-body H-bond term's Hessian
uses multi-variable second-order AD, `Dual2N`, and its gradient a fixed-topology central
difference), L-BFGS optimization, and vibrational analysis. Heats of formation reproduce MOPAC
PM7 to the printed 1e-5 kcal/mol.

**Units.** Coordinates are Angstrom at the XYZ boundary; `Molecule` stores Bohr internally.
`Pm7Result`/`GradientResult` energies are eV (except `heat_of_formation_kcal`), gradients are
eV/Bohr, Hessians eV/Bohr². Constants are 2018-CODATA, matching MOPAC v23.2.5.

All fallible calls return `pm7_rs::Result<T>` (`= std::result::Result<T, Pm7Error>`).

---

## Quick start

```rust
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters, Pm7Method};

let molecule = Molecule::from_xyz_file("examples/water.xyz", 0.0)?;   // (path, charge)
let method = Pm7Method::Pm7;
let parameters = Pm7Parameters::method(method)?;
let options = Pm7Options { method, ..Pm7Options::default() };

let result = run_pm7(&molecule, &parameters, &options)?;
println!("ΔHf   = {:.5} kcal/mol", result.heat_of_formation_kcal);
println!("dipole= {:.3} D", result.dipole_magnitude);
println!("q(O)  = {:+.4} e", result.charges[0]);
# Ok::<(), pm7_rs::Pm7Error>(())
```

`Pm7Parameters::standard()` is exactly `Pm7Parameters::method(Pm7Method::Pm7)`. **Always use
the same method in the parameters and in `Pm7Options`** — a mismatch is rejected.

---

## Building a molecule

```rust
use pm7_rs::{Molecule, Atom};

// From an XYZ file or an in-memory XYZ string (second arg is the formal charge, e):
let a = Molecule::from_xyz_file("examples/methane.xyz", 0.0)?;
let b = Molecule::from_xyz_str("3\nwater\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n", 0.0)?;

// Or build one directly (`Atom` has public fields; positions are Bohr here):
use pm7_rs::{Atom, math::Vec3};
let mol = Molecule::new(vec![
    Atom { z: 8, position: Vec3::new(0.0, 0.0, 0.0) },
]);
# Ok::<(), pm7_rs::Pm7Error>(())
```

`symbol_to_z("Fe")` / `z_to_symbol(26)` convert between element symbols and atomic numbers.

---

## Methods — `Pm7Method`

```rust
use pm7_rs::Pm7Method;

let v: Pm7Method = "pm7-ts".parse()?;          // FromStr; also "pm7ts"
assert_eq!(v, Pm7Method::Pm7Ts);
assert_eq!(v.as_str(), "pm7-ts");
assert!(Pm7Method::Pm7Hh.has_hh_repulsion());
assert!(!Pm7Method::Pm7Minus.has_post_scf_corrections());
# Ok::<(), pm7_rs::Pm7Error>(())
```

| Method          | `parse()` names                    | Parameters      | Post-SCF corrections            |
|------------------|------------------------------------|-----------------|---------------------------------|
| `Pm7`            | `pm7`                              | base PM7        | dispersion + H-bond             |
| `Pm7Ts`          | `pm7-ts`, `pm7ts`                  | PM7-TS tables   | dispersion + H-bond             |
| `Pm7Minus`       | `pm7-`, `pm7-minus`, `pm7minus`   | base PM7        | none (uncorrected NDDO)         |
| `Pm7Hh`          | `pm7-hh`, `pm7hh`                  | base PM7        | dispersion + H-bond + H–H rep.  |
| `Pm7Sparkle`     | `pm7-sparkle`, `sparkle`          | base + sparkles | dispersion + H-bond             |

Sparkle/PM7 treats Ln(III) (Z 58–70) as +3 point cores with zero AOs:

```rust
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters, Pm7Method};
let mol = Molecule::from_xyz_str(
    "4\nEuF3\nEu 0 0 0\nF 2.1 0 0\nF -1.05 1.818 0\nF -1.05 -1.818 0\n", 0.0)?;
let p = Pm7Parameters::method(Pm7Method::Pm7Sparkle)?;
let o = Pm7Options { method: Pm7Method::Pm7Sparkle, ..Pm7Options::default() };
let r = run_pm7(&mol, &p, &o)?;   // ΔHf ≈ −13.02 kcal/mol
# Ok::<(), pm7_rs::Pm7Error>(())
```

---

## Options — `Pm7Options`

```rust
use pm7_rs::{Pm7Options, Pm7Method, ScfReference, ScfAccelerator};
use pm7_rs::stability::ScfStability;

let options = Pm7Options {
    method: Pm7Method::Pm7,
    charge: 0.0,             // formal molecular charge (e)
    multiplicity: 1,         // 2S+1: 1 singlet, 2 doublet, …
    reference: ScfReference::Auto,   // Auto | Restricted (RHF) | Unrestricted (UHF)
    accelerator: ScfAccelerator::AdiisCdiis, // AdiisCdiis | Cdiis | None
    max_scf: 200,
    e_tol: 1.0e-8,           // SCF energy convergence (eV)
    p_tol: 1.0e-7,           // SCF density convergence
    max_memory_mb: None,     // pre-flight OOM guard budget (else PM7_MEM_BUDGET_MB / 80% RAM)
    exchange_cutoff: None,   // optional smooth long-range exchange cutoff (Bohr) for the Hessian CPHF
    cphf_max_iterations: 100, // orbital-response budget, per perturbation
    stability: ScfStability::Off, // Off | Check | Follow -- is the solution a minimum?
    ..Pm7Options::default()
};
```

- `reference` selects the spin treatment **independently of `multiplicity`**, so a closed-shell
  singlet can be forced through the UHF path with `ScfReference::Unrestricted`.
- `exchange_cutoff = Some((inner, outer))` (Bohr) drops the two-center exchange beyond `outer`
  with a C²-smooth switch, only in the analytic-Hessian CPHF response; `None` (default) is
  bit-identical / cutoff-free. Coulomb is never cut.
- `cphf_max_iterations` is how many operator applications one orbital-response solve may spend.
  How many it *needs* is set by the conditioning of the orbital Hessian, which is set by the
  frontier gap — a property of the system. 100 covers an ordinary molecule and most cells; a
  small-gap cell can need several times that and **refuses** rather than returning an unconverged
  response, since a Hessian built on one is wrong by an amount nothing downstream can see. Raising
  it costs iterations and moves nothing else: the tolerance is unchanged, so a solve that converged
  at 100 returns the same matrix at 400. Reachable as `--cphf-max-iterations` on both command lines
  and `cphf_max_iterations=` in Python.
- `stability` asks whether the converged solution is a **minimum** rather than merely a stationary
  point, which the SCF residual cannot distinguish. `Off` (default) does not look; `Check` fills in
  `Pm7Result::stability`; `Follow` rotates along the unstable direction, re-converges, and keeps
  whichever solution is lower. See "Stability analysis" below.

---

## Single point — `run_pm7` → `Pm7Result`

```rust
let r = run_pm7(&molecule, &parameters, &options)?;
```

`Pm7Result` fields:

| Field                     | Meaning                                                        |
|---------------------------|---------------------------------------------------------------|
| `heat_of_formation_kcal`  | ΔHf (kcal/mol)                                                 |
| `total_ev`                | electronic + core-core (+ corrections) energy (eV)            |
| `electronic_ev`, `core_ev`| energy decomposition (eV)                                     |
| `charges`                 | Mulliken net charges (e), one per atom                        |
| `dipole_debye`, `dipole_magnitude` | dipole vector and magnitude (Debye)                  |
| `mo_energies`, `mo_coeff` | MO eigenvalues (eV) and coefficients                          |
| `n_occ`                   | number of occupied MOs (α count for RHF)                      |
| `homo_ev`, `lumo_ev`      | frontier orbital energies (eV; `None` if undefined)           |
| `density`, `spin_density` | AO density; spin density is `Some` only for UHF               |
| `iterations`, `converged` | SCF iteration count and convergence flag                      |
| `unrestricted`            | `true` when the UHF path was used                             |
| `spin_squared()`          | `⟨S²⟩` (MOPAC's `(S**2)`); `None` for RHF and for a k mesh     |
| `stability`               | what `stability::check` found, or `None` if nobody asked      |

A UHF doublet cation:

```rust
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters};
let mol = Molecule::from_xyz_str("5\nCH3\nC 0 0 0\nH 1.08 0 0\nH -0.54 0.935 0\nH -0.54 -0.935 0\nH 0 0 1.08\n", 0.0)?;
let o = Pm7Options { multiplicity: 2, ..Pm7Options::default() };
let r = run_pm7(&mol, &Pm7Parameters::standard()?, &o)?;
assert!(r.unrestricted);
# Ok::<(), pm7_rs::Pm7Error>(())
```

---

## Gradient and forces

```rust
use pm7_rs::{closed_form_gradient, numerical_gradient};

let g = closed_form_gradient(&molecule, &parameters, &options)?;
// GradientResult: .energy_ev, .gradient (Vec<Vec3>, eV/Bohr), .forces (= −gradient),
//                 .max_gradient, .scf (the full Pm7Result)
let fmax = g.max_gradient;

// Full-SCF finite-difference reference (for validation; `step` in Bohr):
let gn = numerical_gradient(&molecule, &parameters, &options, 1.0e-4)?;
```

The analytic gradient is exact (forward-mode dual numbers, no SCF re-runs); a bond lying exactly
on a Cartesian axis is handled by a finite-difference fallback for that pair only, so no
orientation loses the perpendicular component.

---

## Hessian and vibrational analysis

```rust
use pm7_rs::{analytic_hessian, numerical_hessian, vibrational_analysis};

// Analytic Cartesian Hessian (eV/Bohr², 3N×3N Matrix); `step` seeds the CPHF finite tolerance.
let h = analytic_hessian(&molecule, &parameters, &options, 1.0e-3)?;
let hn = numerical_hessian(&molecule, &parameters, &options, 1.0e-3)?;   // reference

// Harmonic frequencies + normal modes in one call. `frequencies_cm` is 3N−6 for a non-linear
// molecule: the rigid-body subspace is projected out of the mass-weighted Hessian, not filtered
// out of the spectrum afterwards.
let modes = vibrational_analysis(&molecule, &parameters, &options, 1.0e-3)?;
// VibrationalModes: .hessian (Matrix), .frequencies_cm (Vec<f64>, cm⁻¹; negative = imaginary),
//                   .eigenvalues (mass-weighted), .modes (3N × n, mass-weighted, one per column),
//                   .cartesian_modes (m^−1/2 L, columns renormalized — MOPAC's `cnorml`),
//                   .removed (what the projector took, or None)
println!("{:?}", modes.frequencies_cm);

// The raw 3N set, for looking at what was removed and how far from zero it was.
use pm7_rs::{vibrational_analysis_projected, Projection};
let raw = vibrational_analysis_projected(
    &molecule, &parameters, &options, 1.0e-3, Projection::None,
)?;
```

`analytic_hessian` is untouched by any of this: it returns the raw `3N × 3N` second-derivative
matrix, and the projection happens only where a *spectrum* is formed. A **periodic** system keeps
its `3N` length too — the removed generators come back at exactly zero rather than being dropped,
because a phonon branch index has to mean the same thing at every `q`.

The Hessian uses CPHF/UCPHF orbital response, solved by **preconditioned conjugate gradient** with
the damped DIIS fixed point as the fallback for an orbital Hessian that is not positive definite. A
response that does not converge is an **error** (`Pm7Error::ResponseFailed`), not a silently
returned last iterate: the relaxation term is linear in `U`, so a Hessian built on an unconverged
response is wrong in proportion to the residual and looks exactly like a converged one. How many
applications it takes is set by the frontier gap — `Pm7Options::cphf_max_iterations` (default 100)
is the budget, and a small-gap cell can need more than an order of magnitude above it.

The dispersion and PM7-HH corrections contribute analytic Hessians, and the many-body H-bond
("EH+") term an analytic Hessian too — each bond's geometric energy is instantiated at
multi-variable second-order AD (`Dual2N<27>`) to give its exact ≤27×27 local block in a single pass
(no finite differences).

On a periodic `Molecule` this is the **zone centre**. For a closed shell at Γ it is
`hessian_pbc::analytic_hessian_periodic`; on a k mesh or for an open shell it delegates to
`dfpt::dynamical_matrix_dfpt` at `q = 0`, which is the same calculation reached the way the
`k ↔ k + q` coupling requires and is the only route that has an unrestricted CPHF.

---

## Geometry optimization — `optimize`

```rust
use pm7_rs::{optimize, OptOptions};

let opt = OptOptions {
    max_iter: 200,
    gtol: 1.0e-4,      // max |gradient| convergence (eV/Bohr)
    history: 8,        // L-BFGS memory
    dandc: None,       // Some(DandcOptions) to drive the steps with divide and conquer
    ..OptOptions::default()
};
let res = optimize(&molecule, &parameters, &options, &opt)?;
// OptResult: .molecule (optimized), .scf (Option<Pm7Result> — None under `dandc`, which has no
//            single converged SCF to hand back), .converged, .iterations, .energy_ev,
//            .heat_of_formation_kcal, .trajectory (Vec<OptStep>{ energy_ev,
//            heat_of_formation_kcal, max_gradient, max_stress, positions, cell })
println!("opt ΔHf = {:.5}", res.heat_of_formation_kcal);
```

`OptOptions::default()` is a sensible starting point. L-BFGS with an Armijo line search; the
trajectory is retained for inspection.

### Variable cell

```rust
let relaxed = OptOptions { relax_cell: true, ..OptOptions::default() };
let res = optimize(&crystal, &parameters, &options, &relaxed)?;
let cell = res.molecule.cell.unwrap();
```

The variables are the strain components inside the periodic subspace — `dim(dim+1)/2` of them —
with the cell `h = (I + ε)h₀` and the atoms carried affinely. That makes `analytic_stress` their
exact conjugate (`∂E/∂ε = Ωσ` is the definition of the stress tensor), so nothing is
finite-differenced and no second implementation of the same derivative exists to drift.

Opt-in, and the reason is worth a number: with a fixed cell, diamond at `a = 3.75 Å` on a 2×2×2
mesh reports `converged` after one iteration with a max gradient of `2e-14 eV/Bohr` and −29.6 GPa
standing. Relaxing the cell reaches `a = 3.6751 Å` in four iterations, 0.076 eV lower, and the same
lattice constant to 6e-6 relative from a start at `a = 3.40 Å`.

`gtol` and `stress_tol` are separate tests and both must pass; a mixed norm over the combined
vector would let a converged force hide an unconverged stress. `analytic_stress`'s refusals — an
external field, `PbcMode::MopacCluster` — surface here rather than becoming a silent atoms-only run.

`grad_step` was declared and never read; it is gone as of 0.2.3.

### Re-checking stability along the way

```rust
let watched = OptOptions { stability_every: 5, ..OptOptions::default() };
```

A geometry step changes the orbitals, so a solution that was a minimum at the starting geometry can
stop being one on the way — and the optimizer would then converge on a stationary point of the
wrong surface without saying so. `stability_every: n` runs the analysis every `n`-th step and
switches the run to `ScfStability::Follow` **permanently** once an instability is found, rather than
per step: a run that alternated between two solutions would have a discontinuous energy and no
optimizer can converge on that. `0` (the default) never checks, which is what every published number
was computed with. `--stability-every N` on both command lines, `stability_every=` in Python.

---

## Stability analysis — `stability::check` → `Stability`

```rust
use pm7_rs::stability::{self, ScfStability};

// As its own call, on a solution you already have:
let found = stability::check(&molecule, &parameters, &options, &result)?;
if let Some(s) = found {
    println!("singlet {:.4} eV, triplet {:?}", s.lowest_ev, s.lowest_triplet_ev);
}

// Or as part of the SCF, which also lets it act:
let escaping = Pm7Options { stability: ScfStability::Follow, ..options.clone() };
let r = run_pm7(&molecule, &parameters, &escaping)?;
```

An SCF converges on `[F, P] = 0`, which is a **stationary** condition. A saddle point converges as
cleanly as a minimum, to as tight a residual, and reports the same `converged: true`. This is the
only thing in the crate that tells them apart.

| `Stability` field | Meaning |
|---|---|
| `lowest_ev` | lowest singlet (RHF→RHF) orbital-Hessian eigenvalue, eV |
| `lowest_triplet_ev` | lowest triplet (RHF→UHF) eigenvalue, eV; `None` if already unrestricted |
| `unstable` | either eigenvalue below the tolerance |
| `lowered_ev` | how far `Follow` got, or `None` |

Both channels are asked because they are different questions: the singlet asks whether the solution
is a minimum among *closed-shell* solutions, the triplet whether it still is once α and β may
differ. Stretched H₂ is stable in the first and unstable in the second, and an analysis that ran
only the first would call it a minimum — 121.6 kcal/mol above the right answer.

`Follow` returns an **unrestricted** result when it escapes through the triplet channel, and warns
on stderr that it did (`PM7_QUIET` silences it). That is a change of model, not just of number: the
solution is broken-symmetry rather than a spin eigenfunction — check `spin_squared()` — and a
geometry scan that switches partway through is discontinuous there. The lower of the two solutions
is always what comes back, so following can never make an answer worse.

The complex (RHF→CHF) channel is deliberately not implemented; see [scope.md](scope.md) for why.

---

## Periodic systems

Giving a `Molecule` a `Cell` is the whole change. `run_pm7`, `closed_form_gradient`,
`analytic_hessian` and `optimize` stay single entry points and dispatch on it.

```rust
use pm7_rs::{run_pm7, Cell, KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters, Vec3};

let cell = Cell::from_angstrom_rows(&[
    [0.0, 1.7835, 1.7835],
    [1.7835, 0.0, 1.7835],
    [1.7835, 1.7835, 0.0],
])?;
let molecule = Molecule::new(vec![
    pm7_rs::Atom { z: 6, position: Vec3::zero() },
    pm7_rs::Atom { z: 6, position: Vec3::new(1.685, 1.685, 1.685) },
])
.with_cell(cell);

let options = Pm7Options {
    pbc: Some(PbcOptions { kmesh: KMesh::grid(4, 4, 4), ..PbcOptions::default() }),
    ..Pm7Options::default()
};
let result = run_pm7(&molecule, &Pm7Parameters::standard()?, &options)?;
println!("{} k points, E_F = {:?}", result.n_kpoints.unwrap(), result.fermi_ev);
# Ok::<(), pm7_rs::Pm7Error>(())
```

A slab or a chain whose open direction is *not* the last lattice vector goes through
`Cell::from_angstrom_rows_pbc`, which takes all three rows plus an ASE-style `[bool; 3]`:

```rust
use pm7_rs::Cell;

// A slab periodic in a1 and a3, open along a2.
let (cell, axes) = Cell::from_angstrom_rows_pbc(
    &[[2.5, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 2.5, 0.0]],
    [true, false, true],
)?;
assert_eq!(cell.unwrap().dim(), 2);
// The lattice vectors were rotated to (a3, a1, a2), so anything you index by lattice vector goes
// through the same rotation — and `undo` puts a result back in your own axis order.
assert_eq!(axes.apply([4, 1, 4]), [4, 4, 1]);
assert_eq!(axes.undo(axes.apply([4, 1, 4])), [4, 1, 4]);
# Ok::<(), pm7_rs::Pm7Error>(())
```

`Cell` keeps storing its periodic vectors contiguously, so every periodic module still reaches the
lattice through a count and a slice rather than a per-direction mask. All eight patterns are
reachable by one of the three cyclic rotations, and cyclic rotations preserve handedness — which
`Cell::reciprocal` depends on, since it divides by a signed determinant.

`Pm7Result` gains `Option` fields that are `None` for a molecule: `n_kpoints`, `fermi_ev`,
`entropy_ev`, `band_energies`, `bloch_density`, `bloch_spin_density`, and — the one a caller is
most likely to want off a plain single point — `stress`, plus the energy decomposition `ewald_ev`,
`background_ev` and the `makov_payne_ev` diagnostic.

`Pm7Result::free_energy_ev()` is `total_ev + entropy_ev`, the **Mermin electronic** free energy
`E − TS`. Under smearing the analytic force is `−∂F/∂R`, so that is the energy to pair with a
gradient; without smearing it equals `total_ev` bit for bit. It is not a thermochemical free
energy — nothing vibrational enters it. See [pbc.md](pbc.md#metals).

**Use a k mesh for a small cell.** `KMesh::Gamma` is the default and is qualitatively wrong for
the long-range exchange of a small primitive cell — 29 eV/atom for two-atom diamond. See
[pbc.md](pbc.md).

## Stress — `analytic_stress` → `StressResult`

```rust
let scf = run_pm7(&molecule, &params, &options)?;
let stress = pm7_rs::analytic_stress(&molecule, &params, &options, &scf)?;
println!("{:?} GPa", stress.pressure_gpa(&molecule));
```

`StressResult` separates the contributions (`electronic`, `core`, `ewald`, `correction`) as well
as their sum, because a stress that disagrees with a finite difference is nearly always one term.
`σ = (1/Ω) ∂E/∂ε`, positive under tension; `Ω` is the volume in 3-D, the area in 2-D, the length
in 1-D.

## Phonons — `force_constants` → `ForceConstants`

```rust
let mut fc = pm7_rs::force_constants(&molecule, &params, &options, [2, 2, 2])?;
println!("acoustic residual {:.3e}", fc.acoustic_residual());
fc.enforce_acoustic_sum_rule();
for f in fc.frequencies_cm([0.5, 0.0, 0.0])? {
    println!("{f:.2} cm^-1");
}

// Frequencies *and* the motions that go with them.
let modes: pm7_rs::PhononModes = fc.modes([0.5, 0.0, 0.0])?;
// PhononModes: .frequencies_cm (Vec<f64>, ascending; negative = imaginary),
//              .eigenvectors (CMatrix, one mode per column, unitary, mass-weighted),
//              .cartesian_modes (m^−1/2 e, columns renormalized — the ones to displace along)
```

Exact at every `q` commensurate with the supercell, Fourier-interpolated between them.

`frequencies_cm` is `modes(q)?.frequencies_cm` with a field taken, so the two cannot come from
diagonalizations that drift apart. The eigenvectors are **complex** away from the zone centre: a
phonon at `q` is `u_A ∝ e_A e^{iq·R_A}`, and the phase is what separates branches sharing a `|q|`.
`DfptResult::modes()` returns the same type from the other route.

## Perturbation theory at arbitrary `q` — `dynamical_matrix_dfpt`

```rust
use pm7_rs::dfpt::{dynamical_matrix_dfpt, DfptOptions};

let out = dynamical_matrix_dfpt(&molecule, &params, &options, [0.5, 0.0, 0.0], &DfptOptions::default())?;
assert!(out.converged);
let frequencies = out.frequencies_cm()?;
let modes = out.modes()?;          // the same PhononModes the supercell route returns
let ground_state = &out.scf;       // the converged SCF the response was built on
```

This couples `k` with `k + q` rather than enlarging the cell, so its cost does not grow with the
supercell a commensurate `q` would otherwise need, and **any** `q` is reachable rather than only
those a supercell happens to fold onto.

`DfptOptions::require_convergence` is **true** by default and makes a diverged response an error.
That matters more than it sounds: the response is a *linear* fixed point, so a failure is not a
near miss but a geometric divergence, and the force constants come back at `1e33` with
`frequencies_cm` ready to take their square root. Set it to `false` only to inspect a failure.

The dynamical matrix is also checked for Hermiticity — which it is by construction — before the
assembly averages out the rounding, so a construction defect surfaces instead of being symmetrized
into a plausible matrix. `DfptResult::hermiticity` reports the measured departure, so a run sitting
just under the threshold is visible rather than merely allowed.

Restricted, unrestricted and **metallic** cells, the last provided `PbcOptions::smearing` is set: a
gapless mesh with no smearing is refused, because every band pair straddling `E_F` then contributes
a `0/0`. The response itself makes no integer-occupation assumption — it weights band pairs by
`Δf/Δε`, which is the metallic form.

The one omission is at **`q = 0` on a genuinely partially occupied cell**: a uniform perturbation
moves the Fermi level, and the intraband term that goes with holding the electron count fixed
(de Gironcoli 1995) is not implemented. That is **refused** from 0.2.3 rather than answered, because
an incomplete zone-centre response looks exactly like a complete one. At `q ≠ 0` the term vanishes
by symmetry and nothing is refused; a smearing that leaves the occupations integral — the usual case,
a width applied to a gapped cell to escape a symmetry-broken solution — is admitted. See
[pbc.md](pbc.md#metals-need-smearing-and-that-is-the-gate).

Two more knobs on `DfptOptions`:

* **`long_range: LongRange::{Auto, Require, Off}`** — whether the long-range monopole term is
  carried. It is all three of the sites it touches or none. `Off` exists so the term's effect is a
  number rather than an argument: on LiF at `q = (¼, 0, 0)` the lowest mode moves from −50.3 to
  74.7 cm⁻¹ without it.
* **`keep_response`** — retains the first-order densities in `DfptResult::response`, moved rather
  than cloned. Off by default; it is `3N × n_k × 2·nao²` floats and the force constants never need
  it kept.

## Born charges, `ε^∞` and LO–TO — `born_and_dielectric`

```rust
use pm7_rs::dfpt::{born_and_dielectric, DfptOptions};

let out = born_and_dielectric(&molecule, &params, &options, &DfptOptions::default())?;
println!("Z*(0)_xx = {:.4}", out.born[0].get(0, 0));
println!("acoustic sum rule residual {:.2e}", out.acoustic_residual());

// LO–TO is 3-D only and needs a direction; nothing is added implicitly.
let na = out.non_analytic()?;
let split = result.frequencies_cm_lo_to(&na, [1.0, 0.0, 0.0])?;
```

`born[A].get(a, b)` is `∂²E/∂f_a ∂R_{A,b}` with **`a` the field index and `b` the displacement
index** (C-5); it is not symmetric in general. `polarizability` is the raw `∂μ/∂f` and is defined in
any dimension; `dielectric` is its 3-D reading and is left as the identity without a cell volume.
Restricted and unrestricted. See [properties.md](properties.md) for what PM7 gets right here (`Z*`)
and what it does not (`ε^∞`).

## The rest of the field response

```rust
use pm7_rs::{
    dielectric_origin_sensitivity, dielectric_with_extent, polarizability,
    static_dielectric_tensor, ExtentConvention,
};

let alpha = polarizability(&molecule, &params, &options, &dfpt)?;
let eps0 = static_dielectric_tensor(&molecule, &params, &options, &dfpt)?;
let sheet = dielectric_with_extent(
    &molecule, &params, &options, &dfpt, ExtentConvention::SlabThickness(6.3),
)?;
let drift = dielectric_origin_sensitivity(&molecule, &params, &options, &dfpt, offset)?;
```

| entry point | what it is | dimensionality |
|---|---|---|
| `polarizability` | `α = ∂μ/∂f`, without computing `Z*` to get it | any |
| `dielectric_with_extent` | `ε^∞` once you say where the material stops | 1-D and 2-D |
| `static_dielectric_tensor` | `ε⁰ = ε^∞ +` the ionic `1/ω²` sum | 3-D |
| `dielectric_origin_sensitivity` | how much `α` moves when the cell is translated | any |

`ExtentConvention` is a **thickness in Bohr** or a **cross-section in Bohr²**, and it is required:
a supercell says where the atoms are, not where the material stops, so doubling the vacuum must not
change `ε`. `ExtentDielectric::invariants` carries the two combinations the extent cannot change.

`StaticDielectric::skipped_modes` is **always three** from 0.2.3 — the acoustic modes, identified by
their overlap with the mass-weighted uniform translations rather than by being small. The field to
read before quoting `ε⁰` is now `soft_optical_modes`: non-zero says the geometry is not a minimum,
and those modes are *reported and kept* rather than dropped. Through 0.2.2 a magnitude floor
(`SOFT_MODE_FLOOR`, removed) did the job, so the count rose above three on such a geometry and the
`1/ω²` sum quietly lost the modes that dominate it.
`dielectric_origin_sensitivity` measures the approximation the position operator
rests on instead of arguing it — near machine precision means the argument holds for this system.

## Polarization and a finite field — `berry_polarization`, `run_finite_field`

```rust
use pm7_rs::{berry_polarization, run_finite_field, FiniteFieldOptions, Vec3};

let p = berry_polarization(&molecule, &params, &options, 16)?;
let moved = p.difference(&q);          // reduced onto the nearest branch, never `q.total - p.total`

let ff = run_finite_field(
    &molecule, &params, &options, [6, 6, 6], Vec3::new(1.0e-4, 0.0, 0.0),
    &FiniteFieldOptions::default(),
)?;
```

Both are 3-D and closed-shell. `berry_polarization` is defined **modulo** the per-axis
`BerryPolarization::quantum`, so the only meaningful operation on a pair of them is `difference`,
which reduces onto the branch nearest zero; subtracting the totals is wrong by exactly one quantum
whenever a displacement crossed a branch. `strings` is the convergence parameter.

`run_finite_field` is for a field along a **periodic** direction, where `𝓔·R` is unbounded and the
ground state does not exist; a field orthogonal to every lattice vector is an ordinary calculation
and goes through `Pm7Options::field`, which the two treatments refuse to combine with. `divisions`
is the k mesh and the Berry string length at once. See
[pbc.md](pbc.md#polarization-and-a-field-along-a-lattice-vector).

## Wavefunction output — `to_molden`

```rust
let scf = pm7_rs::run_pm7(&molecule, &params, &options)?;
let text = pm7_rs::to_molden(&molecule, &params, &scf, &pm7_rs::MoldenOptions::default())?;
```

Returns the file as a `String`; `write_molden` takes a path. Molecules only. The default is an
STO-6G rendering basis with `[5D]`; `MoldenBasis::Sto` writes the exact Slater basis but is refused
for a d-bearing molecule, because Molden's `[STO]` primitive cannot express `x²−y²` or `2z²−x²−y²`.
The NDDO orthogonality caveat is written into the file itself — see [properties.md](properties.md).

## Band structure — `band_structure`

```rust
let bands = pm7_rs::band_structure(&molecule, &params, &options, &[[0.0; 3], [0.5, 0.0, 0.0]])?;
println!("E_F = {:.3} eV", bands.fermi_ev);
```

Not an SCF: the density and the Fermi level come from the sampling mesh in `options`, and the path
only asks the converged Hamiltonian for its eigenvalues elsewhere.

## Divide and conquer — `run_dandc`

```rust
use pm7_rs::{dandc_derivatives, run_dandc, DandcOptions};

let dandc = DandcOptions { buffer: 15.0, ..DandcOptions::default() };
let result = run_dandc(&molecule, &params, &options, &dandc)?;
let derivatives = dandc_derivatives(&molecule, &params, &options, &result)?;
```

`buffer` is in **Bohr** and is the accuracy knob; do not go below 7 Å (13.2 Bohr), where the
accuracy falls off a cliff. Below roughly 350 atoms the exact SCF is faster *and* exact. No
Hessian: `run_dandc` returns an explicit error rather than a wrong one.
## Parameters — `Pm7Parameters`

```rust
use pm7_rs::{Pm7Parameters, Pm7Method};

let p = Pm7Parameters::method(Pm7Method::Pm7)?;
let carbon = p.element(6)?;          // &Pm7Element (u_ss, betas, zeta_s, gss, poc, dshell, …)
let cc = p.pair(6, 6);               // Pm7Pair { alpb, xfac } (diagonal-average fallback if absent)
```

Parameter tables are generated from the pinned MOPAC v23.2.5 source (`tools/extract_params/`)
and embedded via `include_str!`; no fitted parameter is transcribed by hand.

---

## Advanced: fixed-density helpers

The free functions `run_pm7`, `closed_form_gradient`, `analytic_hessian`, `optimize`, and
`vibrational_analysis` are the primary entry points; `Pm7Calculator::new(parameters)` bundles a
parameter set for reuse across calls. For custom analyses, `energy_at_fixed_density`,
`electronic_gradient_fixed_density`, and `fixed_density_gradient` expose the Hellmann–Feynman
machinery at an arbitrary frozen density (this is what the analytic-Hessian skeleton is built on).

See the CLI examples in the [README](../README.md), the Python bindings in
[python-api.md](python-api.md), and the implementation boundary in [scope.md](scope.md).
