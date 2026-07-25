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

let options = Pm7Options {
    method: Pm7Method::Pm7,
    charge: 0.0,             // formal molecular charge (e)
    multiplicity: 1,         // 2S+1: 1 singlet, 2 doublet, …
    reference: ScfReference::Auto,   // Auto | Restricted (RHF) | Unrestricted (UHF)
    accelerator: ScfAccelerator::AdiisCdiis, // AdiisCdiis | Cdiis | None
    max_scf: 200,
    e_tol: 1.0e-8,           // SCF energy convergence (eV)
    p_tol: 1.0e-7,           // SCF density convergence
    max_memory_mb: None,     // pre-flight OOM guard budget (else PM7_MEM_BUDGET_MB / 90% RAM)
    exchange_cutoff: None,   // optional smooth long-range exchange cutoff (Bohr) for the Hessian CPHF
    ..Pm7Options::default()
};
```

- `reference` selects the spin treatment **independently of `multiplicity`**, so a closed-shell
  singlet can be forced through the UHF path with `ScfReference::Unrestricted`.
- `exchange_cutoff = Some((inner, outer))` (Bohr) drops the two-center exchange beyond `outer`
  with a C²-smooth switch, only in the analytic-Hessian CPHF response; `None` (default) is
  bit-identical / cutoff-free. Coulomb is never cut.

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

// Harmonic frequencies + normal modes in one call:
let modes = vibrational_analysis(&molecule, &parameters, &options, 1.0e-3)?;
// VibrationalModes: .hessian (Matrix), .frequencies_cm (Vec<f64>, cm⁻¹; negative = imaginary),
//                   .eigenvalues (mass-weighted)
println!("{:?}", modes.frequencies_cm);
```

The Hessian uses CPHF/UCPHF (Pulay-DIIS-accelerated) orbital response; the dispersion and PM7-HH
corrections contribute analytic Hessians, and the many-body H-bond ("EH+") term an analytic
Hessian too — each bond's geometric energy is instantiated at multi-variable second-order AD
(`Dual2N<27>`) to give its exact ≤27×27 local block in a single pass (no finite differences).

---

## Geometry optimization — `optimize`

```rust
use pm7_rs::{optimize, OptOptions};

let opt = OptOptions {
    max_iter: 200,
    gtol: 1.0e-4,     // max |gradient| convergence (eV/Bohr)
    grad_step: 5.0e-4,
    history: 8,       // L-BFGS memory
};
let res = optimize(&molecule, &parameters, &options, &opt)?;
// OptResult: .molecule (optimized), .scf (final Pm7Result), .converged, .iterations,
//            .trajectory (Vec<OptStep>{ energy_ev, heat_of_formation_kcal, max_gradient, positions })
println!("opt ΔHf = {:.5}", res.scf.heat_of_formation_kcal);
```

`OptOptions::default()` is a sensible starting point. L-BFGS with an Armijo line search; the
trajectory is retained for inspection.

---

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
