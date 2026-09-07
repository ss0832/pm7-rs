# Periodic boundary conditions

`pm7-rs` runs PM7 on 1-D chains, 2-D sheets and 3-D crystals, at the Γ point or on a k mesh, for
neutral and charged cells, restricted and unrestricted. Energy, analytic gradient, analytic stress
and analytic zone-centre force constants are all available, and the post-SCF corrections
(dispersion, EH+ hydrogen bonding, PM7-HH) are periodic too.

Giving a `Molecule` a `Cell` is the whole API change: `run_pm7`, `closed_form_gradient`,
`analytic_hessian` and `optimize` stay single entry points and dispatch on it.

A `Cell` stores its periodic translation vectors contiguously, which is what lets every periodic
module reach the lattice through a count (`cell.dim()`) and a slice (`cell.vectors()`) rather than
a per-direction flag each of them has to remember to honour. **Any** per-axis pattern is still
expressible: `Cell::from_angstrom_rows_pbc` takes an ASE-style `[bool; 3]` and cyclically reorders
the three vectors so the periodic ones lead, returning the `AxisRotation` it used so the caller can
put its own per-lattice-vector inputs — `--kpoints`, `--supercell`, fractional `q` — into the same
order. All eight patterns are reachable by one of the three cyclic rotations, and a rotation
preserves handedness where a transposition would not; handedness is load-bearing for
`Cell::reciprocal`, which divides by a signed determinant.

Through 0.2.2 a non-leading pattern was refused on every surface, so an ASE `Atoms` built as a slab
along *y* — `pbc=(True, False, True)` — was rejected rather than run.

```rust
use pm7_rs::{run_pm7, Cell, KMesh, Molecule, PbcOptions, Pm7Options, Pm7Parameters};

let cell = Cell::from_angstrom_rows(&[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])?;
let molecule = silicon_atoms().with_cell(cell);
let options = Pm7Options {
    pbc: Some(PbcOptions { kmesh: KMesh::grid(4, 4, 4), ..PbcOptions::default() }),
    ..Pm7Options::default()
};
let result = run_pm7(&molecule, &Pm7Parameters::standard()?, &options)?;
```

```python
from ase.build import bulk
from pm7_rs.ase import PM7

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = PM7(kpts=(4, 4, 4))
atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
```

## What makes this tractable: PM7 feathers to a point charge

Beyond 7 Å every two-centre integral in PM7 becomes **exactly** the point-charge monopole
`q_A q_B / r` — not approximately, exactly, because the feathering in `integrals.rs` interpolates
onto `to_point` and stays there. So the two-electron integral splits with no truncation anywhere:

```text
I_{μν,λσ}(R) = [ I_{μν,λσ}(R) − δ_μν δ_λσ v(R) ]  +  δ_μν δ_λσ v(R)
               └── identically 0 beyond 7 Å ──┘      └── 1/r, summed by Ewald ──┘
```

The first bracket has compact support, so it is a finite neighbour-list sum with the ordinary NDDO
kernels. The second is a conditionally convergent lattice sum of monopoles, which is exactly what
Ewald is for. Nothing is cut off and no parameter decides where the split happens; PM7's own
functional form does.

The same decomposition handles the core–core repulsion and the electron–core attraction.

## Two modes

| `PbcMode` | What it is | When to use it |
|---|---|---|
| `Ewald` (default) | The split above. Absolutely convergent, independent of the Ewald splitting parameter, defined for charged cells, and the only mode with a stress. | Everything. |
| `MopacCluster` | MOPAC's truncated image sum (`hcore.F90`, `solrot.F90`, `trunk`). | Reproducing MOPAC's own solid-state numbers. Γ point only; the absolute energy depends on the truncation distance. |

## k points

`KMesh::Gamma` (the default) is the Γ point. `KMesh::grid(n1, n2, n3)` is a Γ-centred
Monkhorst–Pack mesh; `KMesh::Explicit` takes a list of fractional points and weights. Time-reversal
folding halves the number of points actually diagonalized.

**Use a k mesh for a small cell.** Γ-only sampling is not merely coarse, it is qualitatively wrong
for the exchange: at Γ the density matrix `P(T)` does not decay with `T`, so the long-range
exchange is summed as though every image were as correlated as the home cell. For a two-atom
diamond cell the effect is 29 eV per atom:

| mesh | k points | E/atom (eV) |
|---|---|---|
| 1×1×1 (Γ) | 1 | −94.07 |
| 2×2×2 | 8 | −121.56 |
| 3×3×3 | 14 | −123.19 |
| 4×4×4 | 36 | −123.14 |
| 6×6×6 | 112 | −123.07 |
| 8×8×8 | 260 | −123.05 |

Converged by 3×3×3 to about 0.15 eV/atom, and Γ is nowhere near. The same cell described as a
large supercell at Γ is fine — Born–von Kármán says an `n₁×n₂×n₃` mesh *is* the Γ point of the
corresponding supercell, and `tests/pbc_equivalence.rs` checks that identity holds to 1e-8 eV/atom.

Γ-only sampling of a small cell is also numerically harder: it flattens the SCF surface enough that
plain iteration oscillates and DIIS alone settles into a limit cycle. The level-shift controller in
`scf.rs` exists for that, and `tests/scf_convergence.rs` pins the case that motivated it.

### Metals

Set `Smearing::{FermiDirac, Gaussian, MethfesselPaxton}` with a width. Without smearing the
occupations are aufbau, which is right for an insulator and will oscillate for a metal. The
entropy term is reported separately as `entropy_ev`, already carrying the sign that lowers the
free energy, so the free energy and the internal energy stay distinguishable.

Smearing is also what makes the *response* of a metal well defined, and the perturbation solver
refuses a gapless mesh without it — see
[below](#metals-need-smearing-and-that-is-the-gate).

Both are available, and they are not interchangeable:

| | |
|---|---|
| `total_ev` / `energy_ev` | the internal electronic energy at the converged occupations |
| `Pm7Result::free_energy_ev()` / `free_energy_ev` | the **Mermin electronic** free energy `E − TS` |

`free_energy_ev` is what ASE returns for `get_potential_energy(force_consistent=True)`, and it is
the one to use with an optimizer or an MD run: with smeared occupations the analytic force is
`−∂F/∂R`, not `−∂E/∂R`. Without smearing `entropy_ev` is exactly zero and the two are the same
number, bit for bit, so nothing about an insulator or a molecule changes.

Neither is a **thermochemical** free energy. No vibrational partition function enters either, so
neither is the Gibbs free energy a normal-mode analysis would give — this crate computes harmonic
frequencies and derives nothing from them (see [scope.md](scope.md)). The name is shared; the
physics is not.

Through v0.2.1 the free energy was described here but never assembled anywhere: the ASE
calculator set `free_energy` to the internal energy, so a smeared run returned a force-inconsistent
number (0.073 eV out on a two-atom lithium cell at a 0.3 eV width). Fixed in 0.2.2, with
`tests/pbc_equivalence.rs` pinning both that the sum is formed and that forming it moves nothing
when there is no entropy to add.

### The k mesh has to be able to hold your `q`

The response couples `k` with `k + q`. An `n₁ × n₂ × n₃` mesh samples the zone in steps of `1/nᵢ`,
so a `q` well inside one step asks the sampling to tell apart two points it cannot, and the answer
degrades — **while still reporting `converged: true`**, because the linear solve did converge. It
converged to the answer for a question the mesh could not pose.

`max_A |Σ_B Φ_{Aα,Bβ}(q)|`, which the formalism requires to vanish as `q → 0`, on LiF at
`q = 1/160` along `a₁`:

| mesh | residue (eV/Bohr²) |
|---|---|
| 3³ | 2.96e-1 |
| 5³ | 6.75e-2 |
| 7³ | 3.21e-2 |

It converges away with the mesh, so this is a sampling limit and not a defect in the construction:
at `q = 0` the sum rule holds to `1.8e-15`, the Born sum rule `Σ_A Z*_A` to `2.2e-15`, and the
non-analytic term's own `Σ_B D^NA(q̂)` to `2.6e-16`. On a mesh that resolves the wavevector the
residue falls as `O(q)` — halving with each halving of `q`.

What it looks like when it goes wrong is worth recognizing. Diamond, `3³` mesh, `q = 1/160`:

```text
q = 0.006250 0.000000 0.000000   (17 iterations)
    -2977.1642 cm^-1
     -231.3106 cm^-1
       10.9946 cm^-1
```

Three near-zero acoustic modes are what belongs there. `pm7-rs` now prints a warning naming the
mesh step and the divisions the wavevector would need; `PM7_QUIET` silences it. A warning rather
than an error, because there is no sharp threshold — only a degradation — and a deliberately
coarse survey is a legitimate thing to run.

`tests/dfpt_long_wavelength.rs` pins all of it: the exact identities, the `O(q)` approach, that the
non-analytic term is scale-free in `q̂` (it is homogeneous of degree zero, so it cannot introduce a
`1/q²`), and that the residue converges away with the mesh.

## Charged cells

`charge ≠ 0` works in every dimensionality. The `G = 0` divergence is removed by a uniform
neutralizing background (jellium), and the background's energy is reported separately as
`background_ev` because it makes the absolute energy convention-dependent — that is physics, not
an implementation choice, and hiding it would be worse.

* The background exerts **no force** (it is uniform) but does contribute to the **stress**, through
  its volume dependence. `tests/stress.rs` finite-differences a charged cell's stress for exactly
  this reason.
* The total energy is independent of the Ewald splitting parameter `α` and of the reciprocal
  cutoff, for charged cells as well as neutral ones. That invariance is tested.
* `makov_payne_ev` is reported as a **diagnostic** of the finite-size error. It is never added:
  adding it would make the energy inconsistent with its own gradient and stress, and its validity
  rests on assumptions (near-cubic cell, localized charge) the code cannot check.

## Stress

`σ = (1/Ω) ∂E/∂ε`, positive under tension, in ASE's Voigt order `[xx, yy, zz, yz, xz, xy]`. `Ω` is
the volume in 3-D, the area in 2-D and the length in 1-D, so the units follow.

Every PM7 term depends on the nuclei only through pair displacement vectors, so the virial form
`σ_αβ = (1/Ω) Σ (∂E/∂d_α) d_β` is **exact**, not an approximation — and the NDDO basis is
orthonormal and carries no cell dependence, so unlike a Gaussian-basis code there is no Pulay
stress term at all. The long-range part needs more than the pair virial, because the Ewald
reciprocal sum depends on the cell through `1/Ω` and through the reciprocal vectors; those
derivatives are in `pbc/ewald.rs`.

The stress is checked against finite differences of the **full SCF energy** under strain, in 1-D,
2-D and 3-D, at Γ and on k meshes, neutral and charged.

## Phonons

`force_constants(molecule, params, options, supercell)` returns `Φ(0A, TB)` in real space, from the
analytic Hessian of the supercell repeat. From those:

* `dynamical_matrix(q)` and `frequencies_cm(q)` at any fractional `q` — exact at every `q`
  commensurate with the supercell, Fourier-interpolated between them;
* `modes(q)` for the frequencies **and** their polarization vectors, mass-weighted and as Cartesian
  displacements. Both phonon routes return the same `PhononModes`, and `frequencies_cm` is that
  object with one field taken, so the two cannot drift apart;
* `acoustic_residual()` as the honesty check, and `enforce_acoustic_sum_rule()` to project it out.

```python
out = native.phonons(numbers, positions, cell, [[0, 0, 0], [0.5, 0, 0]],
                     supercell=(2, 2, 2))                    # sum rule on by default
e = np.asarray(out["modes"][0][0]) + 1j * np.asarray(out["modes"][0][1])   # one mode per column
u = out["cartesian_modes"][0]                                # the displacements to move atoms by
raw = native.phonons(numbers, positions, cell, [[0, 0, 0]],
                     supercell=(2, 2, 2), acoustic_sum_rule=False)
```

**The sum rule is enforced by default since 0.2.3.** A uniform translation of the whole crystal
costs no energy, so a non-zero residual is numerical noise, and leaving it in reports a non-zero
acoustic frequency at the zone centre — on diamond, `0.0001` where the projected answer is
`-0.0000 cm⁻¹`. It is a genuine two-sided projector `Φ(0) += PΓP − Γ`, not a shift. The residual is
reported either way, because it is the honest measure of how well the force constants respect
translational invariance and projecting hides it.

PM7's converged zone-centre optical mode for diamond is **1238 cm⁻¹**, triply degenerate, 7 % below
the measured Raman line at 1332. A 2×2×2 supercell gives 1318, which looks like much better
agreement and is under-converged — the near-match is a coincidence of that supercell, not a result.

`analytic_hessian` on a periodic cell gives the `q = 0` force constants directly, on a k mesh as
well as at Γ. Through v0.2.1 it refused a k mesh and pointed at a supercell or `numerical_hessian`.
That refusal was not wrong about the physics — a phonon at wavevector `q` does couple `k` with
`k + q` — but that is exactly what `dynamical_matrix_dfpt` solves, and at `q = 0` its force
constants **are** the k-point zone-centre Hessian, so refusing was refusing to make one call. It
delegates now.

The delegation also covers an **unrestricted Γ cell**, which `analytic_hessian_periodic` refuses for
want of an unrestricted CPHF: the perturbation solver has carried a band set per spin since the UHF
field response landed, so a Γ mesh is not a special case for it but the ordinary path with one k
point. An open-shell CH₂ chain comes back as a `9×9` symmetric to `0.00e+00`.

The imaginary part is **checked** to vanish rather than dropped. At `q = 0` every Bloch phase is 1,
so a non-zero imaginary part would be a defect in the construction, and this is the one place it
could hide.

## Perturbation theory: phonons at arbitrary `q`, with no supercell

`dynamical_matrix_dfpt` solves the linear response at each `q` directly, coupling `k` with `k + q`.
There is no commensurability condition — any `q` is reachable, not only those a supercell folds
onto — and no `n₁n₂n₃`-times-larger SCF.

```python
out = native.dfpt(numbers, positions, cell, [[0.3, -0.15, 0.42]], kpoints=(4, 4, 4))
```

It takes **restricted and unrestricted** cells. The unrestricted path carries a band set and a
response density per spin: Coulomb couples each channel to the total density, exchange only to its
own.

### Metals need smearing, and that is the gate

A partially occupied band is the *normal* state of a metal treated this way and is not by itself a
reason to stop. What makes the response ill-defined is a band crossing `E_F` with **no smearing**:
each pair contributes `Δf/Δε`, which is a `0/0` for a step occupation, and the answer would be
decided by which pairs happened to fall inside the `1e-8` denominator floor rather than by the
physics. A smeared occupation makes it finite, because `Δf` then goes to zero with `Δε` at a rate
the smearing function fixes.

So the refusal is on **gaplessness without smearing**, and it names the highest occupied level, the
lowest empty one and `E_F` when it fires. bcc Li at a 0.3 eV Fermi–Dirac width converges to residual
`1.6e-11` with `D(q)` Hermitian to `1.4e-17`. Through v0.2.1 the gate was on *any* fractional
occupation, which turned away every smeared metal — including the ones where smearing is exactly
what makes the response well defined.

**The `q = 0` Fermi-level shift is still missing, and from 0.2.3 that case is refused rather than
answered.** A uniform perturbation of a metal moves `E_F`, and the intraband term that goes with
holding the electron count fixed (de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995)) is not included.
At `q ≠ 0` it vanishes by symmetry — the perturbation has no uniform component to shift the chemical
potential with — so the wavevectors a dispersion is made of are complete and the zone centre of a
metal was not.

Through 0.2.2 that incompleteness was documented and otherwise silent: `phonons`, `born_charges`
and `static_dielectric` all reach the zone centre, and an incomplete response there returns numbers
that look exactly like complete ones. It is now an error, on a test of the **occupations** rather
than of the entropy, because a Methfessel–Paxton entropy can pass through zero with the occupations
still fractional.

**A smearing that leaves the occupations integral is admitted, which is the ordinary case.** A width
applied to a gapped cell — to escape a symmetry-broken solution, or because the SCF's own ladder
chose one — changes the path and not the answer. The two populations are far apart, measured on PM7
ZnS, whose gap depends strongly on the mesh:

| mesh | width | gap | entropy | worst departure from an integer |
|---|---|---|---|---|
| 3×3×3 | 0.10 eV | 0.000 eV | −7.1e-3 | **0.333** — refused |
| 4×4×4 | 0.05 eV | 0.000 eV | −1.5e-3 | **0.334** — refused |
| 4×4×4 | 0.10 eV | 2.575 eV | −2.6e-7 | 5.7e-6 — admitted |
| 5×5×5 | 0.05 eV | 2.242 eV | −6.3e-12 | ~1e-11 — admitted |
| 4×4×4 | 0.02 eV | 2.915 eV | −5.9e-18 | ~1e-17 — admitted |

A third of a state against `5.7e-6`: the threshold sits in nearly five orders of magnitude of empty
space. The refusal names all three ways forward — any `q ≠ 0`, a finer mesh that opens a gap a
coarse one missed, or a narrower width on a cell that does have a gap. See [scope.md](scope.md).

### What it still refuses

A **diverged response**. It is a linear fixed point, so a failure is a geometric divergence
returning numbers around `1e33`, not a near miss. `DfptOptions::require_convergence` is on by
default; the solver also detects divergence early and retries on a damping ladder first.

`D(q)` is checked for Hermiticity — which it is by construction — *before* the assembly averages
out the rounding, so a defect surfaces instead of being symmetrized into a plausible matrix. That
check is what caught the exchange response building its `F(T)` and `F(−T)` blocks from the same
`Δp(T)`, which was valid only at `q = 0` and `q = ½` and wrong at every other wavevector from
v0.2.0 until v0.2.1.

**The threshold is `1e-6`, and it was calibrated rather than chosen.** The residual asymmetry is set
by how well the eigenvectors are determined, and inside a near-degenerate manifold that is poorly:
NaCl's levels are degenerate to `1.4e-14` eV and its `D(q)` is asymmetric at `3.5e-19`, while MgO's
are split by `9.5e-10` eV and its `D(q)` is asymmetric at `7e-11` — nine orders of magnitude, same
code. A cubic CsPbI₃ went past the old `1e-8` outright with a response converged to `1e-10` and a
value stable to three figures across four decades of SCF tolerance, which is a conditioning limit
and not a defect. `tests/dfpt_hermiticity.rs` pins both ends, with the well-conditioned case at
`1e-12` — far tighter than the threshold, because that is where a real error would show.
`DfptResult::hermiticity` reports the measured value, so a run sitting close to the line is visible
rather than merely allowed.

### The long-range monopole term can be switched off, so its effect is a number

`DfptOptions::long_range` is `Auto` (carry it wherever there is a lattice to sum over — what v0.2.1
did with no way to say otherwise), `Require` (a cell whose periodic mode has no lattice sum is an
error rather than a quiet difference in what was computed), or `Off`.

The switch is **all three sites or none** — the fixed-charge second derivative, the bare
perturbation's per-atom channel, and the `∂q_A(q)` shift in the coupled-perturbed kernel — because
leaving it out of any one alone would let the skeleton carry a term the response could not screen.
The point of `Off` is that it makes the term's effect measurable instead of arguable: on LiF at
`q = (¼, 0, 0)` the lowest mode moves from **−50.3 to 74.7 cm⁻¹** with it off.

`DfptOptions::keep_response` retains the first-order densities in `DfptResult::response`, moved
rather than cloned. Off by default: it is `3N × n_k × 2·nao²` floats, the largest array the
calculation touches, and the force constants never need it kept.

Born effective charges, `α`, `ε^∞`, `ε⁰` and LO–TO come from the same solver with a homogeneous
field as the perturbation; see [properties.md](properties.md).

## Polarization, and a field along a lattice vector

The dipole of a crystal is not a function of its density — moving the cell boundary moves charge
across it — so `Σ q r` is not the polarization and the Berry phase is.

```python
out = native.berry_polarization(numbers, positions, cell, strings=16)
out["polarization"], out["quantum"], out["phase"]
```

`berry_polarization` returns the electronic and ionic halves, the raw phase, and the polarization
**quantum** along each lattice vector. What it gives is defined *modulo* that quantum: a different
branch of the logarithm puts the electrons in a different cell, which is an equally valid choice.
Compare two polarizations by reducing their difference onto the nearest branch
(`BerryPolarization::difference`), never by subtracting the totals — a finite displacement commonly
crosses a branch, and the raw difference is then wrong by exactly one quantum, a number that looks
like a catastrophic error rather than a bookkeeping choice.

`strings` is the number of k points along each Brillouin-zone string, and it is **the** convergence
parameter: the answer has to become independent of it. 3-D and closed-shell; the quantum is `a_α/Ω`
and `Ω` has to be a volume.

The reason to have it is that it reaches the Born charge by a route sharing only the Hamiltonian and
the basis with the CPHF, and `tests/pbc_berry.rs` makes that comparison.

### A finite field along a periodic direction

`Pm7Options::field` handles a field orthogonal to every lattice vector — a slab's normal, a chain's
transverse axes — and refuses one along a periodic direction, correctly: `𝓔·R` shifts by `𝓔·T`
under a lattice translation, so the potential is unbounded and the ground state of `H − 𝓔·R` on a
lattice does not exist. No care in the assembly repairs that.

`run_finite_field` minimizes the **Nunes–Gonze electric enthalpy** `F = E − Ω 𝓔·P` instead, with
`P` the Berry-phase polarization above. Because `P` is built from overlaps between *neighbouring* k
points, its derivative couples them: the field term at `k` reads the coefficients at `k ± b`, so the
k points can no longer be solved one at a time. That is the structural reason this is not a small
change to the SCF, and why `scf_pbc::run_kpoint_scf_with_terms` takes a per-k operator that is not a
Bloch sum of any `H(T)`.

`divisions` is the k mesh **and** the string length at once, so it is the convergence parameter for
the polarization as well as for the zone integral; `resolved` says which axes had at least the three
points a phase needs, because an axis that cannot carry one is reported unresolved rather than as
zero.

**What validates it is not the derivation.** The coupling `λ_α = (𝓔·a_α) J / 4π`, and the
Hermitization `M + M†` rather than half of it, are choices that fail silently — a wrong factor of
two, a missing `J` or a flipped sign all give a converged calculation and a plausible
polarizability. `tests/pbc_finite_field.rs` takes `α = Ω ∂P/∂𝓔` by finite differences and compares
it against the CPHF polarizability: `α_xx = 0.146830` against `0.145913`, a **ratio of 1.0063** on
two formalisms that share only the SCF. Half that ratio would be the Hermitization. The polarization
is exactly antisymmetric in the sign of the field (`±2.0146e-8`), which no factor error would
produce by accident.

## Post-SCF corrections

All three are periodic, in the energy, the gradient, the stress and the force constants.

* **Dispersion** — `−C6/R⁶` converges absolutely, so it is an image sum with a cutoff. The carbon
  `C6` is coordination-number dependent, so the bond counts are taken over the minimum image;
  leaving that molecular would get diamond and graphite wrong by whole coefficients.
* **PM7-HH** — a short-range pair potential, image-summed the same way.
* **EH+ hydrogen bonding** — a many-body geometric term. Donor/H/acceptor perception runs over the
  minimum image, each bond's ≤9 atoms are unwrapped into a concrete image cluster, and the forces
  are folded back onto the central cell. A molecule can hydrogen bond to its own image (ice does),
  so `A == B` with `T ≠ 0` is allowed.

The image cutoff is a public parameter (`PbcOptions::correction_cutoff`), and the sums are tapered
with a C² smootherstep over the last 15 % of it. Without the taper the energy steps as an image
crosses the cutoff, and a strain finite difference then diverges as `1/h` — which is how the taper
came to be there.

A k mesh does not change the corrections: they are purely geometric, so the per-cell correction
energy is the same real-space image sum computed once. They do contribute to phonons at every `q`,
through their real-space force constants.

### One sharp edge

The EH+ functional form contains a dihedral about the `R–X` axis of the acceptor, and that dihedral
is **undefined** when the hydrogen lies on the axis. Nothing in the expression vanishes there to
damp it, so the gradient diverges as `1/ρ`: at half a degree off the axis the EH+ force is already
50 kcal/mol/Bohr against about 3 for a normal hydrogen bond. Within `1e-6` rad, MOPAC's own
degenerate branch switches formula and the energy jumps by 0.26 kcal/mol.

This is a property of the model — MOPAC's included — not of this implementation, which reproduces
it faithfully. A symmetric hand-built geometry sits on it; relaxed structures do not, because the
divergent force pushes them off. [singularities.md](singularities.md) has the measurements, the
mechanism, and the one-line repair that would remove it at the cost of changing the model.

## Convergence checklist

1. **k mesh** — increase until the energy per atom stops moving. For anything smaller than a few
   nanometres, Γ is not enough.
2. **`short_range_cutoff`** — the compact-support remainder. The default covers the 7 Å feather
   range with room; there is nothing to converge beyond that, by construction.
3. **`correction_cutoff`** — the dispersion and hydrogen-bond image sums. Widen it and check the
   energy and stress settle.
4. **`ewald_accuracy`** — the Ewald split. The energy must not depend on it, or on `ewald_alpha`;
   if it does, something is wrong rather than unconverged.
5. **Charged cells** — compare `makov_payne_ev` against the energy you care about, and if it is not
   small, use a bigger cell.
6. **The mesh, again, if you are computing phonons at a general `q`.** DFPT frees you from the
   *supercell*, not from the k mesh: the response couples `k` with `k + q`, so the ground-state
   sampling still decides which wavevectors mean anything, and away from the high-symmetry points an
   under-sampled acoustic branch comes back **imaginary** — which reads as a structural instability
   and is not one. On Zn(CN)₂ the softest wavevector on a Γ–X–M–Γ–R–X path gives −72.6, −37.4 and
   −22.6 cm⁻¹ at 2×2×2, 3×3×3 and 4×4×4, halving each time, while Γ stays at −0.00 and every zone
   point stays stable. Refine and watch the dip shrink before concluding anything from it.

## Two examples worth running

* `examples/crystal_phonons.py` — the zone centre of ten structure types (diamond, zinc blende,
  rocksalt, fluorite, perovskite, wurtzite, rutile, spinel, layered rocksalt, and the Zn(CN)₂
  coordination polymer), each checked against **the degeneracies its space group requires**. That is
  a property no unit test in this repository asserts and one that fails loudly when a two-centre
  rotation, a lattice sum or an image list is wrong in a way no energy comparison notices.
* `examples/zncn2_phonon_bands.py` — a full dispersion along Γ–X–M–Γ–R–X by DFPT, over a
  variable-cell relaxation, with the broken-axis plot a spectrum spanning two decades needs. It is
  also where the mesh caveat above was found.
