# pm7-rs

`pm7-rs` is a Rust implementation of the PM7 semiempirical NDDO method and its PM7-TS,
PM7-minus, PM7-HH, and Sparkle/PM7 methods. It provides RHF/UHF energies, heats of formation,
Mulliken charges, gradients, Hessians, geometry optimization, harmonic frequencies, a CLI, and
Python/ASE interfaces.

The parameter tables are generated from a pinned MOPAC v23.2.5 source tree. The implementation
uses `faer` for dense eigensolvers and `rayon` for deterministic pair- and coordinate-level
parallelism. The project is GPL-3.0-or-later.

## Status and physical scope

- Full minimal-valence s/p/d MNDO/d kernel for supported PM7 elements.
- PM7 dispersion, EH+ hydrogen-bond correction, and PM7-HH H-H correction.
- RHF and UHF with explicit `ScfReference::{Auto, Restricted, Unrestricted}` control.
- Forward-mode AD gradients and AD/CPHF Hessians for RHF and UHF.
- Sparkle/PM7 lanthanide sites and PM7-family parameter selection.
- Molecular, non-periodic calculations only.

PM7 is an empirical semiempirical model; agreement with a MOPAC implementation does not imply
ab-initio accuracy outside the model's parameterization domain. Hydrogen-bond topology is
perceived from geometry and its derivatives are evaluated at fixed topology, so a topology
change is a piecewise-smooth boundary. At the singular orientation of a two-center local frame,
the value remains analytic and a localized symmetric finite-difference fallback supplies that
pair's derivative. A residual of up to about 0.36 kcal/mol is still known for some
four-coordinate, highly d-populated transition-metal compounds such as TiF4.

## Build and test

Rust 1.82 or newer is required.

```powershell
$env:CARGO_BUILD_JOBS = '1'
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release
```

Release builds use Fat LTO and one codegen unit as a project policy, prioritizing calculation
throughput:

```powershell
cargo build --release
```

## Rust API

```rust
use pm7_rs::{run_pm7, Molecule, Pm7Options, Pm7Parameters, Pm7Method};

let molecule = Molecule::from_xyz_file("examples/water.xyz", 0.0)?;
let method = Pm7Method::Pm7;
let parameters = Pm7Parameters::method(method)?;
let options = Pm7Options {
    method,
    charge: 0.0,
    multiplicity: 1,
    ..Pm7Options::default()
};
let result = run_pm7(&molecule, &parameters, &options)?;
println!("Delta Hf = {:.6} kcal/mol", result.heat_of_formation_kcal);
# Ok::<(), pm7_rs::Pm7Error>(())
```

`Pm7Parameters` and `Pm7Options` must select the same method. Coordinates stored in `Molecule`
are Bohr; XYZ input is read as Angstrom. Energies are eV and gradients are eV/Bohr unless a
field documents another unit.

## CLI

```powershell
cargo run --release --bin pm7_rs_cli -- energy examples/water.xyz --json
cargo run --release --bin pm7_rs_cli -- gradient examples/methane.xyz
cargo run --release --bin pm7_rs_cli -- energy examples/methyl.xyz --multiplicity 2
cargo run --release --bin pm7_rs_cli -- frequencies examples/water.xyz
```

## Python and ASE

Build the extension with maturin:

```powershell
python -m pip install maturin
maturin develop --release --features python
```

The native Python API accepts Angstrom coordinates and exposes both atomic-unit and eV/Angstrom
results:

```python
from pm7_rs import native

numbers = [8, 1, 1]
positions = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]

point = native.single_point(numbers, positions, method="pm7")
force = native.forces(numbers, positions, method="pm7")
print(point["heat_of_formation_kcal"])
print(force["forces_ev_per_angstrom"])
```

The ASE calculator follows ASE's eV/Angstrom conventions. Force evaluations reuse the energy
from the gradient calculation and therefore require one SCF, rather than a separate energy SCF
followed by a gradient SCF.

```python
from ase.build import molecule
from pm7_rs.ase import PM7

atoms = molecule("H2O")
atoms.calc = PM7(charge=0, multiplicity=1, reference="auto", method="pm7")
energy_ev = atoms.get_potential_energy()
forces_ev_per_angstrom = atoms.get_forces()
```

## Calculation-time memory and performance controls

Before dense SCF or CPHF allocations, the library estimates calculation-time peak memory. The
budget is resolved
in this order:

1. `Pm7Options::max_memory_mb`
2. the `PM7_MEM_BUDGET_MB` environment variable
3. 80% of currently available physical RAM on Windows/Linux

If the estimate exceeds the budget, the calculation returns `Pm7Error::InsufficientMemory`
instead of relying on the operating system's OOM handling. The Fock builder processes at most
2048 atom-pair contribution blocks at a time, and the large-stack EH+ Hessian pool is capped at
four workers. Finite-difference Hessian validation also derives its concurrent SCF-column count
from the same memory budget instead of launching one full SCF per Rayon worker. Set
`RAYON_NUM_THREADS` to control the upper bound on global parallelism.

For very large Hessians, `Pm7Options::exchange_cutoff = Some((inner, outer))` enables a smooth
long-range exchange approximation in the CPHF response build. Distances are Bohr, Coulomb terms
are never cut, and `None` is the exact default.

## MOPAC validation

Fixed-geometry v0.1.2 results compared with the official MOPAC v23.2.5 executable:

| System | pm7-rs (kcal/mol) | MOPAC (kcal/mol) |
|---|---:|---:|
| H2O | -57.78933784 | -57.78934 |
| CH4 | -14.39979910 | -14.39980 |
| H2S | -3.05174959 | -3.05175 |
| CH3 doublet | 28.46324808 | 28.46325 |

The regression suite also covers PM7 methods, d-shell species, Sparkle/PM7 fluorides,
hydrogen-bond corrections, SCF-basin selection, all supported element pairs, and gradient/Hessian
finite-difference checks.

## References

1. J. J. P. Stewart, “Optimization of parameters for semiempirical methods VI: more
   modifications to the NDDO approximations and re-optimization of parameters,” *J. Mol. Model.*
   **19**, 1-32 (2013). [doi:10.1007/s00894-012-1667-x](https://doi.org/10.1007/s00894-012-1667-x)
2. J. E. Moussa and J. J. P. Stewart, “MOPAC: An open-source semiempirical molecular orbital
   program,” *J. Open Source Softw.* **11**, 8025 (2026).
   [doi:10.21105/joss.08025](https://doi.org/10.21105/joss.08025)
3. M. Korth, “Third-Generation Hydrogen-Bonding Corrections for Semiempirical QM Methods and
   Force Fields,” *J. Chem. Theory Comput.* **6**, 3808-3816 (2010).
   [doi:10.1021/ct100408b](https://doi.org/10.1021/ct100408b)
4. A. H. Larsen et al., “The atomic simulation environment-a Python library for working with
   atoms,” *J. Phys.: Condens. Matter* **29**, 273002 (2017).
   [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)
5. G. Zhou et al., “Graphics Processing Unit-Accelerated Semiempirical Born Oppenheimer
   Molecular Dynamics Using PyTorch,” *J. Chem. Theory Comput.* **16**, 4951-4962 (2020).
   [doi:10.1021/acs.jctc.0c00243](https://doi.org/10.1021/acs.jctc.0c00243)

Software sources: [MOPAC v23.2.5](https://github.com/openmopac/mopac/releases/tag/v23.2.5),
[ASE](https://wiki.fysik.dtu.dk/ase/), [faer](https://docs.rs/faer/), and
[Rayon](https://github.com/rayon-rs/rayon). Parameter and code provenance details are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
