<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Python API

`pm7_rs` exposes the PM7-family engine through two layers:

- **`pm7_rs.native`** (also re-exported at the top level `pm7_rs.*`) — thin functions returning
  plain `dict`s of NumPy-friendly lists, in atomic units *and* eV/Å.
- **`pm7_rs.ase.PM7`** — an ASE `Calculator` (eV / Å conventions).

## Build

```bash
python -m venv .venv && . .venv/Scripts/activate     # or source .venv/bin/activate
maturin develop --release --features python           # build the native extension
pip install ase                                       # optional, only for pm7_rs.ase
```

**Inputs.** `numbers` is a length-N sequence of atomic numbers; `positions` is an (N, 3) array in
**Angstrom**. Every function takes the same keyword set:

| kwarg          | default  | meaning                                                              |
|----------------|----------|----------------------------------------------------------------------|
| `charge`       | `0.0`    | formal molecular charge (e)                                          |
| `multiplicity` | `1`      | spin multiplicity 2S+1 (1 singlet, 2 doublet, …)                     |
| `method`      | `"pm7"`  | `"pm7"`, `"pm7-ts"`, `"pm7-"` (PM7-minus), `"pm7-hh"`, `"pm7-sparkle"` |
| `reference`    | `"auto"` | spin path: `"auto"` (RHF/UHF by shell), `"rhf"`/`"r"`, `"uhf"`/`"u"` |

`reference` is independent of `multiplicity`, so a closed-shell singlet can be forced through UHF
with `reference="uhf"`.

---

## `single_point` — energy, charges, dipole, orbitals

```python
import pm7_rs

numbers   = [8, 1, 1]                                            # H2O
positions = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]

sp = pm7_rs.single_point(numbers, positions, charge=0.0, multiplicity=1, method="pm7")
print(sp["heat_of_formation_kcal"])   # ΔHf, kcal/mol
print(sp["energy_ev"], sp["energy_hartree"])
print(sp["charges"])                  # Mulliken net charges (e), per atom
print(sp["dipole_debye"])             # [x, y, z], Debye
```

Returned keys: `energy_ev`, `energy_hartree`, `heat_of_formation_kcal`, `electronic_ev`,
`core_ev`, `charges`, `dipole_debye`, `homo_ev`, `lumo_ev`, `converged`, `unrestricted`, and the
echoed `method`, `charge`, `multiplicity`, `reference`.

---

## `gradient` / `forces`

```python
g = pm7_rs.gradient(numbers, positions)     # ∂E/∂x
g["gradient_hartree_per_bohr"]              # (N, 3), atomic units
g["gradient_ev_per_angstrom"]               # (N, 3), eV/Å

f = pm7_rs.forces(numbers, positions)       # −∂E/∂x
f["forces_ev_per_angstrom"]                 # (N, 3), eV/Å
```

Both also return `energy_ev`, `energy_hartree`, and `heat_of_formation_kcal`. The gradient is
fully analytic (dual-number contraction, no SCF re-runs); axis-aligned bonds are handled exactly.

---

## `hessian`

```python
h = pm7_rs.hessian(numbers, positions)
h["hessian_hartree_per_bohr2"]              # (3N, 3N), atomic units
h["hessian_ev_per_angstrom2"]               # (3N, 3N), eV/Å²
```

Analytic Cartesian Hessian via CPHF/UCPHF. This is the expensive property (O(N⁴) with the CPHF
solve) — call it only when you need curvature.

---

## `frequencies` — harmonic vibrational analysis

```python
vib = pm7_rs.frequencies(numbers, positions)
vib["frequencies_cm"]      # cm⁻¹, ascending; negative entries are imaginary (saddle / non-minimum)
vib["eigenvalues"]         # mass-weighted Hessian eigenvalues
```

Only harmonic frequencies are computed — **no** thermochemistry (entropy/enthalpy/free energy) is
derived from them.

---

## `optimize` — L-BFGS geometry optimization

```python
opt = pm7_rs.optimize(numbers, positions, method="pm7")
opt["positions_angstrom"]        # optimized geometry, (N, 3) Å
opt["heat_of_formation_kcal"]    # ΔHf at the minimum
opt["converged"], opt["iterations"]
```

---

## Method and open-shell examples

```python
# PM7-TS parameters, PM7-minus (uncorrected), PM7-HH (adds H–H repulsion):
pm7_rs.single_point(numbers, positions, method="pm7-ts")
pm7_rs.single_point(numbers, positions, method="pm7-")
pm7_rs.single_point(numbers, positions, method="pm7-hh")

# Sparkle/PM7 lanthanide (Eu is a +3 point core):
pm7_rs.single_point([63, 9, 9, 9],
                    [[0,0,0], [2.1,0,0], [-1.05,1.818,0], [-1.05,-1.818,0]],
                    method="pm7-sparkle")

# Open-shell doublet cation, and a forced-UHF closed-shell singlet:
pm7_rs.single_point(numbers, positions, charge=1.0, multiplicity=2)
pm7_rs.single_point(numbers, positions, reference="uhf")
```

The same functions are available as `pm7_rs.native.single_point(...)` etc.; the top-level
`pm7_rs.single_point` is an alias.

---

## ASE calculator — `pm7_rs.ase.PM7`

Follows ASE's eV / Å conventions.

```python
from ase.build import molecule
from pm7_rs.ase import PM7

atoms = molecule("H2O")
atoms.calc = PM7(method="pm7", charge=0.0, multiplicity=1, reference="auto")

energy = atoms.get_potential_energy()      # eV
forces = atoms.get_forces()                # eV/Å  (standard ASE, from the results cache)
charges = atoms.get_charges()              # e
dipole = atoms.get_dipole_moment()         # e·Å

grad = atoms.calc.get_gradient()           # eV/Å   (= −forces), convenience
hess = atoms.calc.get_hessian()            # eV/Å², (3N, 3N) — computed lazily, only on this call
```

`PM7.implemented_properties == ["energy", "forces", "charges", "dipole", "hessian"]`. `energy`,
`charges`, and `dipole` are produced on every evaluation; `forces` and the (expensive) `hessian`
are computed **lazily** — only when requested (via `get_forces()` / `get_hessian()` or
`get_property(...)`), so a plain energy cycle never pays for the Hessian. `get_hessian()` goes
through ASE's property machinery, so the result is cached per geometry. The heat of formation is
available after any evaluation as `atoms.calc.results["heat_of_formation_kcal"]`.

Constructor keywords `charge`, `multiplicity`, `method`, and `reference` match the native API and
are honoured on every property evaluation. Use ASE's own drivers for optimization and vibrations:

```python
from ase.optimize import BFGS
BFGS(atoms).run(fmax=0.02)                  # uses PM7 forces
```

See the [Rust API](rust-api.md) for the underlying engine and the [README](../README.md) for the
CLI.
