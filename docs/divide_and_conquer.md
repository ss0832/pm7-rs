# Divide and conquer

An ordinary SCF spends most of its time diagonalizing an `nao × nao` Fock matrix, which costs
`O(N³)`. Divide and conquer (Dixon & Merz) replaces that with one small diagonalization per
subsystem, and because a subsystem's size is fixed by a buffer radius rather than by the system,
doubling the system doubles the *number* of subsystems and leaves each one's cost alone.

```python
from pm7_rs import native
out = native.divide_and_conquer(numbers, positions, buffer=8.0, core_size=12)
out["energy_ev"], out["forces_ev_per_angstrom"], out["subsystems"]
```

```rust
use pm7_rs::{dandc_derivatives, run_dandc, DandcOptions};
let result = run_dandc(&molecule, &params, &options, &DandcOptions::default())?;
let d = dandc_derivatives(&molecule, &params, &options, &result)?;
```

From ASE, since 0.2.2: `PM7(dandc=True)`, or `PM7(dandc={"buffer": 10.0, "core_size": 16})` for the
knobs. It supplies **energy and forces only**, and every other property is refused by name rather
than served from a quietly substituted exact SCF — someone who reached for this solver did so
because the exact one is unaffordable at their size, and a silent substitution is indistinguishable
from success. Measured against the exact result on a 24-atom water wire: energy within `9.1e-5` eV,
forces within `1.9e-4` eV/Å.

The keyword's history is worth keeping. The 0.2.0 changelog advertised it, it did not exist,
`**kwargs` swallowed it, and the run quietly stayed exact; v0.2.1 made it an explicit `TypeError`;
0.2.2 implements it.

## How it works

Atoms are cut into **cores** of about `core_size` by recursive bisection along the widest axis,
and each core is given a **buffer** of everything within `buffer` of it. A subsystem is core ∪
buffer: the core is the part whose density it is responsible for, and the buffer is there so the
core's orbitals see a chemically complete environment.

Two things hold the subsystems together, and they are what make this a method rather than a set of
independent fragment calculations:

* **One global Fermi level.** Every subsystem is filled according to the same `E_F`, found by
  bisection on the total electron count. Filling each subsystem to its own aufbau count instead
  would let charge pool in whichever fragment happened to have the lowest levels.
* **The environment potential.** Every subsystem sees the monopole field of the atoms it does not
  contain, rebuilt from the current global charges each iteration. That is what carries
  polarization across a subsystem boundary.

The global density is reassembled with the Dixon–Merz weights: 1 for a core–core pair, ½ for a
core–buffer one. It is stored as **atom-pair blocks**, never as a dense matrix — see below.

## Accuracy: put the buffer past 7 Å

The buffer is the only approximation, and it has a cliff in it. Decane (32 atoms), against the
exact SCF:

| buffer (Å) | buffer (Bohr) | error (eV) | meV/atom |
|---|---|---|---|
| 3 | 5.7 | 2.5e+1 | 778 |
| 5 | 9.4 | 4.6e+0 | 143 |
| **7** | **13.2** | **1.2e-3** | **0.04** |
| 9 | 17.0 | 1.1e-5 | 0.00 |
| 11 | 20.8 | 9.1e-6 | 0.00 |
| 15 | 28.3 | 2.0e-11 | 0.00 |

Four orders of magnitude between 5 Å and 7 Å, and that is not a coincidence: **7 Å is the range at
which PM7's feathering makes every two-centre integral exactly a point-charge monopole**. Below it
the buffer is discarding real integrals; above it, everything the buffer excludes is a monopole
that the environment field supplies exactly, and the only remaining error is the truncation of the
density matrix itself.

So: **do not use a buffer below 7 Å.** The default is 15 Bohr (7.9 Å), just past the cliff.

At a fixed buffer the error grows **linearly with the system**, because it is a per-boundary error
and the number of boundaries grows with N. On alkanes at the default buffer it runs at about
0.018 meV/atom. Quote it per atom, not per system.

The gradient inherits that error and adds one of its own: the NDDO energy is stationary with
respect to the *exact* density, so a Hellmann–Feynman gradient taken at an approximate one carries
a **non-variational residual** proportional to how far the density is from self-consistent. It
shrinks with the buffer at the same rate the energy error does, and `tests/dandc.rs` shows both
sequences rather than asserting a bound.

## Scaling

Linear alkanes, `buffer = 15` Bohr, `core_size = 8`, 16 cores, single wall-clock run
(`cargo test --release --test dandc_scaling -- --nocapture --ignored`):

| C | atoms | D&C (s) | s/iteration | exact SCF (s) | ΔE (eV) | stored density | subsystems | iterations |
|---|---|---|---|---|---|---|---|---|
| 40 | 122 | 2.08 | 0.039 | 0.54 | 0.0019 | 36.4 % | 16 | 54 |
| 80 | 242 | 5.07 | 0.098 | 2.48 | 0.0043 | 19.2 % | 32 | 52 |
| 160 | 482 | 9.53 | 0.187 | 15.42 | 0.0089 | 9.9 % | 64 | 51 |
| 320 | 962 | 19.32 | 0.379 | — | — | 5.0 % | 128 | 51 |
| 640 | 1922 | 39.51 | 0.790 | — | — | 2.5 % | 256 | 50 |

**log–log slope 1.06 overall, 0.99 above 200 atoms, 1.01 per iteration**, against **2.19** for the
exact SCF over the same range. The SCF iteration count is flat, which matters on its own — a linear
cost per iteration would not help if the number of iterations grew.

The **crossover is near 350 atoms**. Below that the ordinary SCF is both exact and faster, and this
is the wrong tool.

Wall-clock numbers bounce by a few tens of percent between runs on a loaded machine; the 160-atom
point read 17.3 s on one run and 9.5 s on the next. The linear-scaling *gate* in `tests/dandc.rs`
therefore asserts on arithmetic — `Σ n³`, `Σ n²`, stored elements — which is deterministic and
means the same thing on any machine. `tests/dandc_scaling.rs` reports the timings and asserts
nothing.

## Why the density has to be sparse

This is the part that cannot be retrofitted. A dense `nao × nao` global density costs `O(N²)` to
store and `O(N²)` to assemble, so a method whose every other step is `O(N)` still comes out
quadratic — and at 5 000 atoms the dense matrix alone is gigabytes.

`SparseDensity` stores one `na × nb` block per atom pair that any subsystem holds, in a CSR-like
layout. It fits the rest of the crate without translation: the Fock build, the gradient, the stress
and the energy contraction all already walk atom pairs.

The stored fraction shrinks as the system grows — 53 % at 62 atoms, 2.5 % at 1922 — which is the
sparsity being real rather than nominal.

## Three things that had to be right

Each was found by measuring, and each is worth knowing about if you read the code.

**The environment potential splits into nuclear and electronic halves.** `½ Tr P(H + F)` counts a
term in `H` once and a term in `F` half — which is exactly the difference between a nuclear
attraction (linear in the density) and a mean-field electron repulsion. Folding the whole of
`−V_A = −Σ_B M_AB (Z_B − P_B)` into the core Hamiltonian, which is the obvious reading of "external
potential", double counts the electronic half once per subsystem. The error grows as `N²`; it
reached 55 000 eV on a 240-atom chain.

**The far field and the neighbour search were both `O(N²)`.** Computing the global potential inside
the per-subsystem loop repeats the same arithmetic about five times over, and scanning every atom
to grow every buffer is quadratic on its own. Hoisting the potential and putting the buffer search
on a uniform grid took the measured slope from 1.22 to 1.13.

**Dixon–Merz weights are not a partition of unity at buffer edges.** They sum to 1 for a pair whose
two ends see each other — the two owning subsystems contribute ½ apiece — but an atom can lie in
another subsystem's buffer without that subsystem's cores lying in its own, and such a pair
collects a single ½. Left alone that halves the density on exactly the pairs the method is least
sure about. The driver normalizes by the realized weight total, which makes the weights a partition
of unity by construction and turns those pairs into an average of the estimates that saw them.

## Periodic systems

The periodic path differs in one place: buffer atoms are drawn from periodic images. A subsystem is
therefore a **finite cluster** even though the system is not, so no complex arithmetic is involved
and Γ and k-mesh parents are handled identically — periodicity enters through the buffer atoms and
through the Ewald far field.

The environment subtraction is pairwise over the cluster's actual images. Subtracting the lattice
sum `M_AB` instead would remove every image of B rather than the one the cluster holds.

## Unrestricted

UHF is supported on the same terms as RHF: each subsystem diagonalizes an α and a β Fock, and a
**single global Fermi level** fills both channels. The Coulomb term sees the total density and the
exchange sees the same spin, exactly as in the molecular code. `spin_density` comes back as a
second `SparseDensity`.

## What is not supported

**Second derivatives.** A Hessian needs the coupled-perturbed response of the whole system, which
does not decompose the way the density does; `analytic_hessian` is not routed through this path and
`run_dandc` does not offer one.

## Choosing the parameters

* `buffer` — the accuracy knob. Cost grows with the buffer cubed (it sets the subsystem size), so
  this is the trade-off to tune. Start at 8 Å; use the table above to decide whether you need more.
* `core_size` — the constant in the `O(N)`, not its exponent. Smaller cores mean more subsystems of
  smaller size; the total work has a shallow minimum.
* `fermi_width_ev` — a little smearing keeps the occupation a continuous function of the subsystem
  eigenvalues, which is what lets the bisection converge when a fragment has levels at `E_F`.
