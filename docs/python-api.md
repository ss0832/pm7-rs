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

| kwarg                 | default  | meaning                                                          |
|-----------------------|----------|------------------------------------------------------------------|
| `charge`              | `0.0`    | formal molecular charge (e)                                      |
| `multiplicity`        | `1`      | spin multiplicity 2S+1 (1 singlet, 2 doublet, …)                 |
| `method`              | `"pm7"`  | `"pm7"`, `"pm7-ts"`, `"pm7-"` (PM7-minus), `"pm7-hh"`, `"pm7-sparkle"` |
| `reference`           | `"auto"` | spin path: `"auto"` (RHF/UHF by shell), `"rhf"`/`"r"`, `"uhf"`/`"u"` |
| `scf_tolerance`       | `None`   | SCF density convergence; `None` keeps the built-in `1e-7`        |
| `max_scf`             | `None`   | SCF iteration budget (default 200)                               |
| `use_diis`            | `None`   | `False` turns off DIIS — the first thing to try when an SCF oscillates |
| `cphf_max_iterations` | `None`   | CPHF orbital-response budget (default 100); raise it when a small-gap cell reports an unconverged response |
| `stability`           | `None`   | `"off"` (default), `"check"`, `"follow"` — is the converged solution a minimum, or only a stationary point? |
| `exchange_cutoff`     | `None`   | `(inner, outer)` in **Bohr** for the Hessian CPHF's smooth long-range-exchange cutoff; `None` is exact |
| `field`               | `None`   | uniform external field `[x, y, z]` in **V/Å**, MOPAC's `FIELD=` sign convention |
| `dipole_origin`       | `None`   | `"coordinates"`, `"com"`, `"charge"` — only affects a charged species |

Plus the periodic set — `cell`, `pbc`, `kpoints`, `kpoint_shift`, `smearing`, `pbc_mode` — which is
documented under [periodic systems](#periodic-systems) and defaults to molecular behaviour
throughout.

`reference` is independent of `multiplicity`, so a closed-shell singlet can be forced through UHF
with `reference="uhf"`. The four SCF and CPHF knobs are on every entry point rather than only where
they bite: they reach `Pm7Options`, so a mode that never runs a CPHF simply carries a number it does
not read — which is better than refusing a mistyped budget with a complaint about the wrong thing.

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

Returned keys: `energy_ev`, `energy_hartree`, `free_energy_ev`, `heat_of_formation_kcal`,
`electronic_ev`, `core_ev`, `field_ev`, `charges`, `dipole_debye` and its three decomposed parts
(`dipole_point_charge_debye`, `dipole_sp_hybrid_debye`, `dipole_pd_hybrid_debye`) with
`dipole_origin_bohr`, the frontier set `homo_ev`, `lumo_ev`, `gap_ev` and their `_beta` twins,
`mo_energies_ev`, `n_occ`, `orbital_source`, `converged`, `iterations`, `unrestricted`, and the
echoed `method`, `charge`, `multiplicity`, `reference`. A periodic cell adds the keys under
[periodic systems](#periodic-systems).

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

Pass a `cell` for a periodic system and it is the **zone-centre** force constants: from the
periodic analytic Hessian for a closed shell at Γ, and from the perturbation solver at `q = 0` on a
k mesh or for an open shell — the periodic analytic Hessian has no unrestricted CPHF, and the
perturbation solver carries a band set per spin. Other wavevectors come from
[`dfpt`](#dfpt--phonons-at-arbitrary-q-no-supercell) or
[`phonons`](#phonons--force-constants-and-frequencies).

---

## `frequencies` — harmonic vibrational analysis

```python
vib = pm7_rs.frequencies(numbers, positions)
vib["frequencies_cm"]      # 3N−6 entries, ascending; negative ones are imaginary (saddle point)
vib["eigenvalues"]         # mass-weighted Hessian eigenvalues

# The opt-out: the raw 3N set, translations and rotations included.
pm7_rs.frequencies(numbers, positions, projection="none")
```

**The rigid-body motions are projected out, not filtered out.** Water returns **three** frequencies,
not nine of which six happen to be small. Through 0.2.2 it returned nine, three of them
(`71.35`, `121.33`, `181.31 cm⁻¹`) unprojected rotations that a caller had to recognize by
magnitude. Whether the molecule is linear is decided from its inertia tensor, so a linear one gives
3N−5 and a single atom gives none — a geometric tolerance asks "are these atoms collinear", which
has an answer, where a frequency cutoff asks "is this number small", which does not.

`projection` takes `"rigid"` (the default), `"translations"` (the acoustic sum rule alone, leaving
any rotations in) or `"none"`. Every mode-indexed array shrinks together and stays aligned:
`frequencies_cm`, `ir_intensities_km_per_mol`, `cartesian_modes` (by column), and
`mode_dipole_derivatives`. `dipole_derivatives` is indexed by Cartesian coordinate, not by mode,
and keeps its `3 × 3N`; so does `hessian` at `3N × 3N`.

A `cell` gives the **zone-centre** frequencies of a periodic system, agreeing with `phonons` and
`dfpt(q = 0)` to every digit. Through v0.2.1 the periodic keywords did not reach this function or
`vibrations`, so a periodic cell silently got the frequencies of its atoms as an isolated
molecule — for a two-atom diamond cell, a C₂ diatomic at `−772, 654, 654` where the crystal's zone
centre is `0, 0, 0, 1248, 1248, 1248`. Infrared intensities stay molecular and are **refused** for
a periodic cell rather than answered that way.

Only harmonic frequencies are computed — **no** thermochemistry is derived from them. There is no
zero-point energy, no vibrational partition function, and no Gibbs or Helmholtz free energy of the
kind a normal-mode analysis would give.

Note that `free_energy_ev` (and ASE's `free_energy`) is a **different quantity with the same
name**: the Mermin *electronic* free energy `E − TS` of a smeared band structure. It is about how
a metal's bands are occupied, not about molecular vibrations, and it exists whether or not a
Hessian was ever computed. See [pbc.md](pbc.md#metals).

---

## `optimize` — L-BFGS geometry optimization

```python
opt = pm7_rs.optimize(numbers, positions, method="pm7")
opt["positions_angstrom"]        # optimized geometry, (N, 3) Å
opt["heat_of_formation_kcal"]    # ΔHf at the minimum
opt["converged"], opt["iterations"]
opt["trajectory"]                # one entry per step: energy, max gradient, geometry, cell
```

### Variable cell

```python
opt = pm7_rs.optimize(numbers, positions, cell=cell, kpoints=(2, 2, 2), relax_cell=True)
opt["cell_angstrom"], opt["pbc"]     # the relaxed lattice, in your axis order
opt["trajectory"][-1]["max_stress"]  # eV/Bohr^dim
```

**Opt-in, and the reason is worth a number.** With `relax_cell=False` — the default — the atoms
relax inside a fixed cell, and the run reports success with whatever stress that cell implies.
Diamond at `a = 3.75 Å` on a 2×2×2 mesh comes back `converged: True` after one iteration with a max
gradient of `2e-14 eV/Bohr` and **−29.6 GPa** of pressure. Both are true: the atoms are at their
minimum for that cell, and the cell is 2 % too big.

With `relax_cell=True` the same structure relaxes in four iterations to `a = 3.6751 Å`, 0.076 eV
lower — and starting instead from `a = 3.40 Å` reaches the same lattice constant to 6e-6 relative.
The variables are the strain components inside the periodic subspace, so the analytic stress is
their exact conjugate and the atoms move affinely with the cell. A chain relaxes along its own axis
and acquires nothing perpendicular to it; a slab does not strain its vacuum direction.

`gtol` (force, eV/Bohr) and `stress_tol` (largest free stress component) are **separate** tests and
both must pass. A single mixed norm would let a converged force hide an unconverged stress.

ASE users have `FrechetCellFilter` and can keep using it — this is for everyone else.

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

An unrestricted result also carries `spin_squared` — `⟨S²⟩`, MOPAC's `(S**2)`. A UHF determinant is
not a spin eigenfunction, and this is the only thing in the output that says how far from one it is:
a clean doublet radical is near `0.75` and a broken-symmetry singlet runs up to `1.0`. It is `None`
for a restricted run (where the value is `S(S+1)` by construction) and for a k-mesh run.

---

## `scf_stability` — is the converged solution a minimum?

```python
found = pm7_rs.scf_stability(numbers, positions)
found["lowest_ev"]          # singlet (RHF→RHF) orbital-Hessian eigenvalue, eV
found["lowest_triplet_ev"]  # triplet (RHF→UHF); None if the solution is already unrestricted
found["unstable"]           # either one below the tolerance
found["analysed"]           # False for a periodic cell or an unconverged solve — an answer, not an error
```

An SCF converges on `[F, P] = 0`, which is a **stationary** condition, not a minimum one: a saddle
point converges as cleanly as a minimum, to as tight a residual, and reports the same
`converged: True`. This is what tells them apart. Seeger and Pople's test, on the same orbital
Hessian the CPHF already applies — see [scope.md](scope.md) for which of their channels are covered.

To act on it rather than measure it, pass `stability="follow"` to any entry point:

```python
# Stretched H2: RHF dissociates it 121.6 kcal/mol too high.
pm7_rs.single_point([1, 1], [[0, 0, 0], [0, 0, 4.0]])                       # 225.79 kcal/mol
pm7_rs.single_point([1, 1], [[0, 0, 0], [0, 0, 4.0]], stability="follow")   # 104.19 — two H atoms
```

`"follow"` rotates along the unstable direction, re-converges, and keeps whichever solution is
lower, so it can never make an answer worse. When it escapes through the triplet channel the result
is **unrestricted** where you asked for restricted — a different model, not just a different number
— and it says so on stderr (`PM7_QUIET=1` silences it). Check `spin_squared` to see the
contamination that came with it. Off by default on every surface.

`optimize` additionally takes `stability_every=n`, which re-checks every `n`-th step: a geometry step
changes the orbitals, so a solution that was a minimum at the start can stop being one on the way.

---

## The command line, from a pip install

`pip install pm7-rs-python` puts a **`pm7-rs`** command on your path. It is the same tool as
`python -m pm7_rs`, and it covers everything the Rust `pm7_rs_cli` binary does — the Rust binary is
not shipped inside the wheel, so this is what a pip user gets.

```bash
pm7-rs energy water.xyz --json
pm7-rs energy diamond.xyz --kpoints 4 4 4
pm7-rs stress diamond.xyz --kpoints 4 4 4
pm7-rs phonons diamond.xyz --supercell 2 2 2 --qpoints 0,0,0 0.5,0,0 --acoustic-sum-rule
pm7-rs bands diamond.xyz --kpoints 4 4 4 --qpoints 0,0,0 0.5,0,0
pm7-rs dfpt diamond.xyz --kpoints 3 3 3 --qpoints 0.25,0,0
pm7-rs born lif.xyz --kpoints 3 3 3 --lo-to 1,0,0
pm7-rs dielectric bn_sheet.xyz --slab-thickness 3.33 --kpoints 4 4 1
pm7-rs energy big.xyz --dandc 8.0
```

Modes: `energy`, `charges`, `gradient`, `forces`, `stress`, `optimize`, `frequencies`, `hessian`,
`phonons`, `dfpt`, `born`, `bands`, `orbitals`, `molden`, `dielectric`. `pm7-rs --help` lists every
flag.

The two command lines offer the **same** modes, and a test enumerates both to keep it that way.
Until v0.2.2 the Rust binary had `hessian` and this one did not, which matters more than it sounds:
the wheel does not ship the Rust binary, so a pip user had no route to it at all.

`dielectric` needs the material's extent (`--slab-thickness` for a layer, `--wire-cross-section` for
a chain) and refuses to guess. A supercell says where the atoms are, not where the material stops,
so doubling the vacuum would otherwise change ε. It prints the two **sheet invariants** alongside
the tensor — those are the numbers that do not move with the thickness you assumed. Both command
lines take Ångström and convert; through 0.2.2 this one passed the value straight through and was
Bohr in practice, a silent factor of 1.889. See
[the units note below](#polarizability-static_dielectric-dielectric_with_extent).

The cell comes from an extended-XYZ `Lattice="..."` key — which is what `atoms.write()` produces —
or from `--cell`. Flag abbreviation is off, so `--kpoint` is an error rather than a silent match
for `--kpoints`.

If the command is not found after installing, the scripts directory is not on your `PATH`;
`python -m pm7_rs` works regardless and is exactly the same entry point.
## Periodic systems

Every function takes the same six periodic keywords. Passing `cell` is what turns a molecular call
into a periodic one; the rest have defaults that reduce to the Γ point of a neutral, unsmeared,
Ewald-mode cell.

| keyword | type | meaning |
|---|---|---|
| `cell` | `(3,3)` array or `None` | lattice vectors in Ångström, one row each |
| `pbc` | `(bool, bool, bool)` | which directions are periodic; any pattern, e.g. `(True, False, True)` |
| `kpoints` | `(n1, n2, n3)` or `None` | Monkhorst–Pack divisions; `None` is Γ |
| `kpoint_shift` | `(3,)` | fractional mesh offset |
| `smearing` | `(kind, width_ev, order)` | `"none"`, `"fermi"`, `"gauss"`, `"mp"` |
| `pbc_mode` | `"ewald"` or `"mopac"` | rigorous (default) or MOPAC-compatible |

```python
from pm7_rs import native

a = 3.567 / 2
cell = [[0, a, a], [a, 0, a], [a, a, 0]]
numbers = [6, 6]
positions = [[0, 0, 0], [a / 2, a / 2, a / 2]]

point = native.single_point(numbers, positions, cell=cell, kpoints=(4, 4, 4))
point["energy_ev"], point["fermi_ev"], point["n_kpoints"]
```

Periodic results carry extra keys: `stress` (3×3, eV/Å³), `stress_voigt`
(`[xx, yy, zz, yz, xz, xy]`), `pressure_gpa`, `volume_angstrom3`, `fermi_ev`, `entropy_ev`,
`n_kpoints`, and for a charged cell `ewald_ev`, `background_ev` and `makov_payne_ev`.

Every result also carries `free_energy_ev` alongside `energy_ev`: the Mermin **electronic** free
energy `E − TS`, equal to `energy_ev` exactly when there is no smearing. With `smearing` set it is
the quantity the forces differentiate, so it is the one an optimizer should follow. It is not a
thermochemical free energy — see [`frequencies`](#frequencies--harmonic-vibrational-analysis) above
and [pbc.md](pbc.md#metals).

**Use a k mesh for a small cell.** Γ alone gets the long-range exchange quantitatively wrong —
29 eV/atom for this two-atom diamond cell. See [pbc.md](pbc.md).

## `stress` — the stress tensor

```python
out = native.stress(numbers, positions, cell=cell, kpoints=(4, 4, 4))
out["stress_voigt"]   # eV/A^3, ASE order, positive under tension
out["pressure_gpa"]
```

## `phonons` — force constants and frequencies

```python
out = native.phonons(
    numbers, positions, cell,
    qpoints=[[0, 0, 0], [0.5, 0, 0]],
    supercell=(2, 2, 2),
    acoustic_sum_rule=True,
)
out["frequencies_cm"]      # one list per q point
out["acoustic_residual_ev_per_bohr2"]   # reported whether or not it was projected out

real, imag = out["modes"][0]            # 3N x 3N, one mode per COLUMN, same order as the
e = np.asarray(real) + 1j * np.asarray(imag)    # frequencies. Unitary; mass-weighted.
u = out["cartesian_modes"][0]           # the same set as displacements, m^-1/2 e, renormalized
```

Also returned: `qpoints`, `supercell`, `translations`, and — with `lo_to_direction` —
`frequencies_cm_lo_to` and `lo_to_direction`.

**The polarization vectors are new in 0.2.3.** They were computed on every call and discarded, which
left the frequency the only thing this route could tell you: no way to say which atoms a soft branch
moves, to displace a structure along a mode, or to check that two branches at one frequency are the
degenerate pair a space group requires. `cartesian_modes` is the one to move atoms along, and its
convention is the molecular `vibrations`' exactly. Away from the zone centre both are genuinely
complex — a phonon at `q` is `u_A ∝ e_A e^{iq·R_A}`, and the phase separates branches that share a
`|q|`, so taking `.real` is only meaningful where every Bloch phase is `±1`.

`supercell` is the Brillouin-zone sampling knob here, not `kpoints`: the force constants come from
the supercell's Γ point, and Γ of an `n₁×n₂×n₃` supercell **is** the `n₁×n₂×n₃` mesh of the cell.
Passing `kpoints` raises rather than being accepted and ignored — it could not have changed the
answer, and a silent no-op is how a caller comes to believe a mesh applied when it did not.

## `dfpt` — phonons at arbitrary `q`, no supercell

```python
out = native.dfpt(numbers, positions, cell, qpoints=[[0.3, -0.15, 0.42]], kpoints=(4, 4, 4))
out["frequencies_cm"][0]
real, imag = out["force_constants_ev_per_bohr2"][0]   # two real matrices, for numpy
out["modes"][0], out["cartesian_modes"][0]            # same layout and meaning as `phonons`
out["iterations"], out["converged"], out["residual"], out["hermiticity"]
```

Solves the linear response directly at each `q`, so there is no commensurability condition and
`kpoints` does apply. Restricted, unrestricted, and — since 0.2.2 — **metallic**, provided
`smearing` is set: a gapless mesh with no smearing is refused, because every band pair straddling
`E_F` then contributes a `0/0` and the answer would be decided by a numerical floor rather than by
the physics. The `q = 0` Fermi-level shift is still missing; at `q ≠ 0` it vanishes by symmetry.
See [pbc.md](pbc.md#metals-need-smearing-and-that-is-the-gate).

A response that fails to converge **raises**: it is a linear fixed point, so a failure is a
divergence rather than a near miss.

**The mesh has to be able to hold your `q`.** The response couples `k` with `k + q`, so an
`n × n × n` mesh cannot resolve `q ≪ 1/n` — and the solve still converges, to the answer for a
question the mesh could not pose. Diamond on a `3³` mesh at `q = 1/160` returns acoustic modes at
−2977 cm⁻¹. A warning names the mesh step and the divisions the wavevector would need; `PM7_QUIET`
silences it. See [pbc.md](pbc.md#the-k-mesh-has-to-be-able-to-hold-your-q).

### LO–TO on either phonon route

```python
out = pm7_rs.dfpt(numbers, positions, cell, [[0, 0, 0]],
                  lo_to_direction=(1, 0, 0), kpoints=(3, 3, 3))
out["frequencies_cm"][0]        # 245.7  245.7  245.7
out["frequencies_cm_lo_to"][0]  # 245.7  245.7  288.5
```

`lo_to_direction` works the same on `phonons`, on `PM7.get_dfpt`/`get_phonons`, and as `--lo-to` on
both command lines. One branch — the longitudinal one along `q̂` — is pushed up and the transverse
pair is untouched. **3-D only**, and there is no default direction, because the `q → 0` limit is
direction dependent. Entries for a q away from the zone centre are `None`: the macroscopic field is
already inside `Φ(q)` there, and adding the term again would count it twice.

## `born_charges` — `Z*`, `ε^∞` and LO–TO

```python
out = native.born_charges(numbers, positions, cell, kpoints=(3, 3, 3), lo_to_direction=(1, 0, 0))
out["born_charges"][0]      # a 3x3 tensor per atom: [field index][displacement index]
out["dielectric"]           # 3-D only; the identity without a cell volume
out["polarizability"]       # the raw d(mu)/d(f), defined in any dimension
out["acoustic_residual"]    # sum_A Z*_A, which must vanish for a neutral cell
```

`lo_to_direction` is optional and has no default, because the `q → 0` limit is direction dependent.
Without it `lo_to_force_constants_ev_per_bohr2` is `None` rather than an error, so the 1-D and 2-D
cases need no `try`/`except`.

`Z*` is quantitative in PM7; `ε^∞` is not. See [properties.md](properties.md).

`born_charges` and `dfpt` also take `long_range=` (`"auto"`, `"require"`, `"off"`) and
`keep_response=`. `"off"` drops the long-range monopole term from all three places it enters, which
is how its effect is measured rather than argued; `keep_response=True` returns the first-order
densities, which are the largest array the calculation touches and are otherwise discarded.

## `polarizability`, `static_dielectric`, `dielectric_with_extent`

```python
alpha = native.polarizability(numbers, positions, cell, kpoints=(3, 3, 3))["polarizability"]
eps0  = native.static_dielectric(numbers, positions, cell, kpoints=(3, 3, 3))
eps0["dielectric"], eps0["electronic"], eps0["ionic"]      # eps^0, eps^inf, and the ionic part
eps0["skipped_modes"]        # always 3: the acoustic modes, by overlap with the translations
eps0["soft_optical_modes"]   # non-zero means the geometry is not a minimum
eps0["softest_kept"]         # smallest omega^2 actually used in the 1/omega^2 sum

sheet = native.dielectric_with_extent(numbers, positions, cell,
                                      slab_thickness=6.3, kpoints=(4, 4, 1))
sheet["dielectric"], sheet["sheet_parallel_bohr"], sheet["extent_convention"]
```

* **`polarizability`** is the same tensor `born_charges` reports, without computing the Born charges
  to get it. It exists separately because `α` is defined in **every** dimensionality where `ε^∞` is
  not. It is in the MOPAC `FIELD=` sign convention, so it carries the opposite sign to the physical
  polarizability.
* **`static_dielectric`** is `ε⁰ = ε^∞ +` the ionic term, 3-D only. `skipped_modes` is **always
  three** from 0.2.3 — the acoustic modes, picked out by their overlap with the uniform
  translations rather than by being small. Through 0.2.2 a magnitude floor decided it, so the count
  rose above three on a geometry that is not a minimum, quietly dropping genuine soft optical modes
  from the `1/ω²` sum. **Read `soft_optical_modes`** for that now: it counts them and keeps them.
* **`dielectric_with_extent`** needs the extent and refuses to guess, because a supercell says where
  the atoms are and not where the material stops. `extent_convention` comes back beside the value,
  because a bare `extent` number cannot distinguish a slab thickness from a wire cross-section —
  different units, different depolarization factors, unrecoverable from the number.

**Units for the extent differ by surface, deliberately.** `native.dielectric_with_extent` and
`PM7.get_dielectric_with_extent` take `slab_thickness` in **Bohr** and `wire_cross_section` in
**Bohr²**, matching `ExtentConvention` in the Rust API and the crate's internal coordinates. Both
**command lines** take Ångström and Ångström² and convert, because that is the unit their other
length flags use. Through 0.2.2 the Python CLI documented Ångström and passed the value straight
through, so it was Bohr in practice — a silent factor of 1.889 (3.571 for the area) against the
Rust CLI, fixed in 0.2.3.

## `dielectric_origin_sensitivity` — measuring the position-operator approximation

```python
worst = native.dielectric_origin_sensitivity(numbers, positions, cell, (0.3, 0.1, 0.0))
```

Every atom is displaced by `offset` — a **Bohr** vector, like the crate's internal coordinates and
unlike `positions`, which is Ångström — `α` is recomputed, and the largest change in any component
comes back as a bare float. Near machine precision says the periodicity argument the field
perturbation rests on holds for this system; a large value says the polarizability being reported is
a statement about where the origin was put. `PM7.get_dielectric_origin_sensitivity` forwards the
same vector, in the same units.

## `berry_polarization` and `finite_field`

```python
p = native.berry_polarization(numbers, positions, cell, strings=16)
p["polarization"], p["quantum"], p["phase"]          # e/Bohr^2; quantum is one vector per axis
p["electronic"], p["ionic"], p["string_length"]

ff = native.finite_field(numbers, positions, cell, (6, 6, 6), (1e-4, 0.0, 0.0))
ff["energy_ev"], ff["enthalpy_ev"]                   # E, and the E - Omega*E.P actually minimized
ff["polarization"], ff["resolved"], ff["converged"], ff["iterations"]
```

`berry_polarization` gives the polarization **modulo** the returned `quantum`. Compare two of them
by reducing their difference onto the nearest branch, never by subtracting `polarization` directly:
a finite displacement commonly crosses a branch and the raw difference is then wrong by exactly one
quantum — a number that looks like a catastrophic error rather than a bookkeeping choice. `strings`
is the convergence parameter and the answer has to stop moving with it.

`finite_field` is for a field applied **along** a periodic direction, where `𝓔·R` is unbounded and
the ground state does not exist; it minimizes the Nunes–Gonze electric enthalpy `E − Ω 𝓔·P`
instead. For a field orthogonal to every lattice vector, pass `field=` to the ordinary entry points
— that is an ordinary calculation and needs none of this. `divisions` is the k mesh **and** the
Berry string length at once; `resolved` says which axes had the three points a phase needs, because
an axis that cannot carry one is reported unresolved rather than as zero.

Both are 3-D and restricted closed-shell. Neither is a mode on either command line.

## `molden` — the wavefunction as a string

```python
text = native.molden(numbers, positions)                 # STO-6G rendering basis, [5D]
exact = native.molden(numbers, positions, basis="sto")   # exact Slater; s/p only
```

Returns the text rather than writing a file. Molecules only. The file carries its own caveat: NDDO
assumes an orthonormal AO basis, so the coefficients are in an implicitly orthogonalized basis while
the listed functions are the raw non-orthogonal ones.

## `divide_and_conquer` — the linear-scaling SCF

```python
out = native.divide_and_conquer(numbers, positions, buffer=8.0, core_size=12)
out["energy_ev"], out["forces_ev_per_angstrom"], out["subsystems"]
```

`buffer` is in **Ångström** here (the Rust API uses Bohr). Do not go below 7 Å: that is where
PM7's feathering makes every excluded interaction exactly a monopole, and below it the accuracy
falls off a cliff. Below roughly 350 atoms the ordinary SCF is faster and exact.
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

`PM7.implemented_properties == ["energy", "free_energy", "forces", "stress", "charges", "dipole", "hessian"]`. `energy`,
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

### Vibrational properties share one solve

`frequencies`, `normal_modes`, `ir_intensities` and `dipole_derivatives` all come out of the same
CPHF solve as the Hessian, so requesting **any** of them computes the whole group once and caches
it against the geometry. Moving an atom invalidates it.

```python
atoms.calc.get_frequencies()          # cm^-1, 3N-6 of them
atoms.calc.get_normal_modes()         # (3N, 3N-6), Cartesian columns
atoms.calc.get_ir_intensities()       # km/mol, 3N-6
atoms.calc.get_ir_spectrum()          # all of the above in one dict
atoms.calc.get_dipole_derivatives()   # (3, 3N) — by coordinate, not by mode; about the input
                                      #           origin (C-8)
atoms.calc.get_dipole_breakdown()     # Debye, point-charge / s-p / p-d separately
atoms.calc.get_orbitals()             # its own group: one plain SCF
atoms.calc.get_scf_stability()        # is the converged solution a minimum? Also its own SCF
```

These deliberately live **outside** `implemented_properties`, which stays
`["energy", "free_energy", "forces", "stress", "charges", "dipole", "hessian"]` — code that
iterates that list, ASE included, must keep working. That is also why they are *not* reached
through `get_property`, which raises for any name outside the list: they use an explicit cache
instead. `python/tests/test_ase_caching.py` counts the real calls into the native layer rather
than taking this paragraph's word for it. `ase.vibrations.Infrared` also works against this
calculator by finite differences if you prefer its machinery.

### Periodic properties take an argument, so they are plain methods

A q path, a supercell or a LO–TO direction is not something `get_property` can key on. These are
ordinary methods, each caching on the geometry **and** the argument — the same q path twice is one
solve, a different one is two:

```python
atoms.calc.get_phonons([[0, 0, 0], [0.5, 0, 0]], supercell=(2, 2, 2))
atoms.calc.get_dfpt([[0.3, -0.15, 0.42]])          # arbitrary q, no supercell
atoms.calc.get_born_charges(lo_to_direction=(1, 0, 0))
atoms.calc.write_molden("orbitals.molden")          # returns the text as well
```

The 0.2.2 field-response methods cache the same way — on the geometry and on their own arguments:

```python
atoms.calc.get_polarizability()                     # alpha, in every dimensionality
atoms.calc.get_static_dielectric()                  # eps^0; read skipped_modes
atoms.calc.get_dielectric_with_extent(slab_thickness=6.3)   # eps^inf for a slab, Bohr
atoms.calc.get_dielectric_origin_sensitivity((0.3, 0.1, 0.0))
atoms.calc.get_berry_polarization(strings=16)
atoms.calc.get_finite_field((1e-4, 0.0, 0.0), (6, 6, 6))
```

`get_finite_field` deliberately drops the calculator's `kpts`, `smearing` and `field`: `divisions`
*is* the mesh here (and the Berry string length), and `PM7(field=...)` is the other, incompatible
field treatment. `get_dielectric_with_extent` drops `field` and `dipole_origin` for the same
reason — a field is already in the perturbation.

`get_phonons` does **not** use the calculator's `kpts`. Force constants come from the supercell's
Γ point, and Γ of an `n₁×n₂×n₃` supercell *is* the `n₁×n₂×n₃` mesh — so `supercell` is the
sampling knob there and `kpts` would be a second, contradictory one. `kpts` still governs energies
and forces from the same calculator, and it does apply to `get_dfpt`.

The calculator learns its `Atoms` from `atoms.calc = PM7(...)` via ASE's `set_atoms` hook, so
these work immediately; call one on a calculator that has never been attached to anything and it
says so rather than failing on a `None`.

### The linear-scaling solver from ASE

`PM7(dandc=True)` runs the divide-and-conquer SCF instead of the exact one; `PM7(dandc={"buffer":
10.0, "core_size": 16})` sets its knobs. It supplies **energy and forces only**, and every other property is
refused by name:

```python
atoms.calc = PM7(dandc={"buffer": 8.0})
atoms.get_potential_energy()          # fine
atoms.get_forces()                    # fine
atoms.get_stress()                    # PropertyNotImplementedError, saying which property
```

That refusal is the design, not a gap. Quietly substituting the exact SCF for the property it
cannot do would hand back a number the caller has no way to distinguish from the approximate one —
and someone who reached for this solver did so because the exact one is unaffordable at their size.

The history is worth keeping: the 0.2.0 changelog advertised this keyword, it did not exist,
`**kwargs` swallowed it, and the run quietly stayed exact. v0.2.1 made it an error; 0.2.2
implements it.

### Band structures

`calc.get_band_structure("GXWL")` returns an ASE `BandStructure` — pass a path string, a `BandPath`
from `atoms.cell.bandpath(...)`, or a bare list of fractional k points (which returns the raw dict,
since a `BandStructure` needs a path to label its axis).

**This is not ASE's inherited `Calculator.band_structure()`.** That one reconstructs a band
structure from a finished SCF through `get_eigenvalues`/`get_ibz_k_points`, so it can only return
the mesh the SCF already ran on. `get_band_structure` diagonalizes non-self-consistently at
whatever k points you ask for, which is what a dispersion needs. The two names sit side by side in
the namespace and are different calculations.

See the [Rust API](rust-api.md) for the underlying engine, [properties.md](properties.md) for the
conventions behind the dipole, field and Born charges, and the [README](../README.md) for the CLI.
