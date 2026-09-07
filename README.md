# pm7-rs

`pm7-rs` is a Rust implementation of the PM7 semiempirical NDDO method and its PM7-TS,
PM7-minus, PM7-HH, and Sparkle/PM7 methods, for **molecules and for periodic systems in one, two
and three dimensions**. It provides RHF/UHF energies, heats of formation, Mulliken charges,
analytic gradients, analytic stress, Hessians, phonons, geometry optimization, harmonic
frequencies, a linear-scaling divide-and-conquer SCF, a CLI, and Python/ASE interfaces.

The parameter tables are generated from a pinned MOPAC v23.2.5 source tree. The implementation
uses `faer` for dense eigensolvers and `rayon` for deterministic pair- and coordinate-level
parallelism. The project is GPL-3.0-or-later.

## Status and physical scope

- Full minimal-valence s/p/d MNDO/d kernel for supported PM7 elements.
- PM7 dispersion, EH+ hydrogen-bond correction, and PM7-HH H-H correction — all periodic.
- RHF and UHF with explicit `ScfReference::{Auto, Restricted, Unrestricted}` control.
- Forward-mode AD gradients and AD/CPHF Hessians for RHF and UHF.
- **SCF stability analysis** (Seeger–Pople), on the same orbital Hessian the CPHF already applies:
  is the converged solution a minimum, or only a stationary point? Singlet and triplet channels,
  and — opt-in — an escape along the unstable direction that keeps whichever solution is lower.
  It removes the classic RHF dissociation failure: stretched H₂ goes from 121.6 kcal/mol above
  two hydrogen atoms to within 0.02 of them. `⟨S²⟩` is reported for every unrestricted solution.
  **Off by default.**
- Sparkle/PM7 lanthanide sites and PM7-family parameter selection.
- **Periodic boundary conditions** in 1-D, 2-D and 3-D: Gamma point and Monkhorst-Pack k meshes,
  neutral and charged cells, energy, analytic gradient, analytic stress, analytic zone-centre
  force constants. See [`docs/pbc.md`](docs/pbc.md).
- **Phonons and perturbation theory at arbitrary wavevector** (an NDDO density-functional
  perturbation theory coupling `k` with `k + q`), for restricted, unrestricted and smeared metallic
  cells, plus band structures along a k path.
- **Molecular properties**: a uniform external electric field with an exact analytic gradient and
  Hessian, a corrected dipole with its terms reported separately, orbital energies and
  coefficients, IR spectra, and Molden output. See [`docs/properties.md`](docs/properties.md).
- **Born effective charges, the polarizability, both dielectric tensors and LO–TO splitting**, from
  the same perturbation solver with a homogeneous field taken through the commutator `[H, r]`. The
  electronic `eps^inf` in 3-D and, with an assigned extent, for a chain or a slab; the static
  `eps^0` with the ionic term added.
- **Berry-phase polarization and a finite field along a lattice vector** (the Nunes-Gonze electric
  enthalpy), which exist as independent checks on the perturbative field response: the finite-field
  polarizability agrees with the CPHF one to a ratio of 1.0063 on two formalisms sharing only the
  SCF.
- **Divide and conquer**: a linear-scaling SCF with gradient and stress, measured log-log slope
  0.99 above 200 atoms. See [`docs/divide_and_conquer.md`](docs/divide_and_conquer.md).

Two limits worth knowing before you start: sampling a small cell at the Gamma point alone gets
the exchange **quantitatively wrong**, not merely coarse (29 eV/atom for a two-atom diamond
cell), so use a k mesh; and the EH+ hydrogen-bond gradient diverges when the acceptor's dihedral
axis goes collinear, which is a defect in PM7's published functional form rather than in this
implementation. Both are measured in [`docs/scope.md`](docs/scope.md) and
[`docs/singularities.md`](docs/singularities.md).

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
cargo run --release --bin pm7_rs_cli -- frequencies examples/water.xyz --ir
cargo run --release --bin pm7_rs_cli -- forces examples/methane.xyz
cargo run --release --bin pm7_rs_cli -- hessian examples/water.xyz
cargo run --release --bin pm7_rs_cli -- molden examples/water.xyz -o water.molden
cargo run --release --bin pm7_rs_cli -- energy examples/water.xyz --field 0.5,0,0
```

Periodic modes take the cell from an extended-XYZ `Lattice="..."` key — which is what
`atoms.write()` produces in ASE — or from `--cell`:

```powershell
cargo run --release --bin pm7_rs_cli -- energy diamond.xyz --kpoints 4 4 4
cargo run --release --bin pm7_rs_cli -- stress diamond.xyz --kpoints 4 4 4
cargo run --release --bin pm7_rs_cli -- phonons diamond.xyz --supercell 2 2 2 --qpoints 0,0,0 0.5,0,0
cargo run --release --bin pm7_rs_cli -- dfpt diamond.xyz --kpoints 4 4 4 --qpoints 0.3,-0.15,0.42
cargo run --release --bin pm7_rs_cli -- born diamond.xyz --kpoints 4 4 4 --lo-to 1,0,0
cargo run --release --bin pm7_rs_cli -- bands diamond.xyz --kpoints 4 4 4 --qpoints 0,0,0 0.5,0,0
cargo run --release --bin pm7_rs_cli -- dielectric bn_sheet.xyz --slab-thickness 3.33 --kpoints 4 4 1
cargo run --release --bin pm7_rs_cli -- energy chain.xyz --cell 3.2,0,0
cargo run --release --bin pm7_rs_cli -- energy slab.xyz --cell 3.2,0,0,0,0,12,0,3.2,0 --pbc TFT
cargo run --release --bin pm7_rs_cli -- optimize diamond.xyz --kpoints 2 2 2 --opt-cell
cargo run --release --bin pm7_rs_cli -- energy big.xyz --dandc 15.0
```

`--opt-cell` relaxes the lattice as well as the atoms, driven by the analytic stress. Without it a
periodic `optimize` reports success with whatever stress the fixed cell implies — diamond at
`a = 3.75 Å` comes back converged after one iteration with −29.6 GPa standing.

`--pbc` takes any per-axis pattern; the lattice vectors are reordered so the periodic ones lead,
and `--kpoints`, `--supercell` and fractional `--qpoints` move with them.

`--stability check|follow` asks whether the converged SCF is a minimum rather than merely a
stationary point, and — with `follow` — escapes it. Off by default:

```powershell
cargo run --release --bin pm7_rs_cli -- energy h2_stretched.xyz                      # 225.79 kcal/mol
cargo run --release --bin pm7_rs_cli -- energy h2_stretched.xyz --stability follow   # 104.19, two H atoms
```

`pm7_rs_cli` with no arguments prints every mode and flag. Both command lines offer the same fifteen
modes — `energy`, `charges`, `gradient`, `forces`, `stress`, `optimize`, `frequencies`, `hessian`,
`phonons`, `dfpt`, `born`, `bands`, `orbitals`, `molden`, `dielectric` — and the same flags, both
enumerated by tests to keep it that way.

## Python and ASE

Build the extension with maturin:

```powershell
python -m pip install maturin
maturin develop --release --features python
```

Installing from PyPI also puts a **`pm7-rs`** command on your path, covering every mode the Rust
binary does (the Rust binary is not shipped inside the wheel):

```bash
pip install pm7-rs-python
pm7-rs energy water.xyz --json
pm7-rs phonons diamond.xyz --supercell 2 2 2 --qpoints 0,0,0 0.5,0,0
```

`python -m pm7_rs` is the same entry point, and works even when the scripts directory is not on
`PATH`.
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

Fixed-geometry results compared with the official MOPAC v23.2.5 executable. The oracle scripts in
`tools/oracle/` regenerate these against a real MOPAC install and currently report a difference of
`0.0000` kcal/mol on every molecule they cover:

| System | pm7-rs (kcal/mol) | MOPAC (kcal/mol) |
|---|---:|---:|
| H2O | -57.78933784 | -57.78934 |
| CH4 | -14.39979910 | -14.39980 |
| H2S | -3.05174959 | -3.05175 |
| CH3 doublet | 28.46324808 | 28.46325 |

The regression suite also covers PM7 methods, d-shell species, Sparkle/PM7 fluorides,
hydrogen-bond corrections, SCF-basin selection, all supported element pairs, and gradient/Hessian
finite-difference checks.

`tools/oracle/baseline.py` carries **176** cases covering **all 73 elements PM7 is parameterized
for** — a contract enforced by reading `src/data/pm7_elements.csv`, not a comment — across five
multiplicities, charges −2 to +2 and sixteen organometallics. It compares the heat of formation,
every Mulliken charge, both frontier channels, the dipole and `⟨S²⟩`, at MOPAC's printed precision:
1057 of 1086 comparisons pass, and [`docs/fidelity.md`](docs/fidelity.md) accounts for each of the
29 that do not, case by case.

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
