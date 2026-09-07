<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Molecular and periodic properties

Everything beyond an energy, a gradient and a Hessian: an external electric field, the dipole and
its derivatives, orbital data, Molden output, IR spectra, Born effective charges, the polarizability
and both dielectric tensors, the Berry-phase polarization, and a finite field applied along a
lattice vector.

Conventions used throughout are fixed in [theory.md](theory.md) §0 and referred to here by their
labels (C-1 … C-8). They are load-bearing: the sign of the field, which dipole operator a given
quantity uses, and which index of `Z*` is the field index all follow from them.

---

## External electric field

`Pm7Options::field` takes an `ExternalField` in **volts per Ångström**, in MOPAC's `FIELD=`
convention — the vector is the *potential gradient*, not the physical field, so

```text
E(F) = E₀ + F · μ            (C-1;  F = −E_physical)
```

`pm7-rs` keeps MOPAC's sign because `FIELD=` is what the number means, and a heat of formation
computed under a field is then directly comparable to MOPAC's.

| quantity | status |
|---|---|
| energy | exact, folded into `h_core` at the point MOPAC folds it (`hcore.F90:184-208`) |
| gradient | exact and closed form: `∂E/∂R_A = q_A f` |
| Hessian | exact; the *skeleton* second derivative is identically zero, so the whole contribution arrives through the CPHF's perturbed `∂h/∂R` |
| stress | **refused** — the field's strain derivative is not in the virial, and returning the field-free stress would look like an answer |

### Periodic cells

A uniform field is not a periodic operator: `−f·r` grows without bound under a lattice translation.
So a field component **along a periodic direction is refused** by this operator, with a message
pointing at `dfpt::born_and_dielectric`, which takes the field through the commutator `[H, r]`
instead — and, for a *finite* field along the lattice, at `run_finite_field` (see
[pbc.md](pbc.md#a-finite-field-along-a-periodic-direction)), which minimizes the electric enthalpy
`E − Ω 𝓔·P` rather than an energy that has no lower bound.

A component along a **non-periodic** direction is accepted and is exactly as well defined as it is
for a molecule — a chain's transverse axes, a slab's normal. This is not a convenience: it is the
only way to check the periodic field response against something independent, and it is what
`tests/born.rs` uses.

The two treatments **refuse to be combined**: a cell carrying `Pm7Options::field` cannot also be
handed to `run_finite_field`, because that would apply two different field operators to the same
Hamiltonian and neither answer would be either one.

Divide and conquer refuses a field outright, because each subsystem builds its own core.

---

## Dipole moment

`Pm7Result::dipole` is a `DipoleBreakdown` — point-charge, s–p hybrid and p–d hybrid terms
separately, in Debye, mirroring MOPAC's `POINT-CHG./HYBRID/SUM` print. `dipole_debye` remains and
equals `total()`.

### Three operators, and which is which

This is the part that catches people out, and it is MOPAC's inconsistency rather than ours:

| `DipoleTerms` | terms | what uses it |
|---|---|---|
| `PointCharge` | `Σ_A q_A R_A` | diagnostics only |
| `FieldConjugate` | + s–p hybrid | **MOPAC's `FIELD` operator** (`hcore.F90:191-202` has no p–d term) |
| `Full` | + p–d hybrid | **MOPAC's printed dipole** (`dipole.F90:118-148`), and the physical operator |

So in MOPAC's own convention `∂E/∂F` equals `FieldConjugate`, **not** the dipole MOPAC prints, for
any molecule containing an atom with d orbitals. `pm7-rs` reproduces that rather than quietly
repairing it, and names both so a cross-check compares like with like (C-2).

IR intensities and Born charges use `Full`, the physical operator. The molecular field defaults to
`FieldConjugate`, for MOPAC fidelity.

### Origin

`DipoleOrigin` is `CentreOfMass` by default, as MOPAC does outside `FORCE`/`THERMO`/`IRC`. It only
ever changes a **charged** molecule: for a neutral one `Σ_A q_A (R_A − c)` is independent of `c`,
and the implementation short-circuits the origin to exactly zero below MOPAC's own 0.5 e threshold
so no published neutral number moves by even one bit.

Per **C-8**, the raw `∂μ/∂R` tensor is *always* about the input coordinate origin regardless of this
setting. A moving origin would add `−q_tot m_B/M` and make the tensor convention-dependent for an
ion. MOPAC does the same thing by disabling recentring under `FORCE`.

### Periodic cells

The point-charge term is kept **only along non-periodic directions**. Along a lattice vector it is
not an observable — shifting an atom by one translation changes it — so what is reported there is
not a dipole at all. Across a non-periodic direction it is an ordinary dipole per cell, and it is
exactly what the transverse Born charges measure. The mask is per axis for that reason.

The quantity that *is* defined along a lattice vector is the **Berry-phase polarization**, and it is
a separate entry point (`berry_polarization`) rather than a repair to the dipole, because it is
defined only modulo a quantum and a `DipoleBreakdown` has nowhere to say so. See
[pbc.md](pbc.md#polarization-and-a-field-along-a-lattice-vector).

---

## Orbitals

`orbitals(...)` returns MO energies (eV and Hartree), coefficients, occupations, `gap_ev`, and
`ao_labels` / `ao_atom_index` / `ao_shell` so the coefficient matrix is interpretable without
re-deriving the basis. Both spin channels for a UHF result.

It also reports `fermi_ev` and `entropy_ev` from 0.2.3, and `orbital_source` says which Hamiltonian
the orbitals diagonalize (`"molecular"`, `"gamma"`, `"kmesh-gamma"`). That combination is the point:
on a k-mesh run the occupations are an **aufbau fill by index at Γ**, and the zone-wide answer is
the Fermi level — so a smeared metal used to return occupations that disagreed with its own Fermi
level, with nothing in the result saying so.

`single_point` carries only the cheap scalars — HOMO, LUMO, gap. Not the coefficients: that is an
`nao × nao` matrix and putting it on every single point would marshal a matrix nobody asked for on
every step of an MD run.

### `orbital_source`, and why it exists

On a k mesh there is no such thing as "the" molecular orbital set. `Pm7Result::orbital_source`
records which one you have:

| value | meaning |
|---|---|
| `Molecular` | an ordinary molecular SCF |
| `Gamma` | a periodic Γ-point run |
| `KMeshGamma` | a k-mesh run, **re-diagonalized at Γ from the converged Hamiltonian** |

Before v0.2.1 a k-mesh run reported `mo_energies` from the first *expanded* k point (which on a
shifted mesh is not even Γ) alongside `mo_coeff` from the pre-SCF Γ solve — eigen-data of two
different matrices, one of which never converged.

---

## `⟨S²⟩` and SCF stability

Two questions an unrestricted or a converged solution cannot answer about itself.

### `⟨S²⟩` — how far the determinant is from a spin eigenfunction

`Pm7Result::spin_squared()`, `spin_squared` in the `single_point`, `orbitals` and `scf_stability`
dicts, `PM7.results["spin_squared"]`, and `(SZ)` / `<S^2>` on both command lines. MOPAC's `(S**2)`.

```text
⟨S²⟩ = S_z(S_z + 1) + n_β − Σ_ij |⟨φ_i^α|φ_j^β⟩|²  =  S_z(S_z + 1) + n_β − Tr(P^α P^β)
```

The second form holds because the NDDO basis is **orthonormal by construction**, so the overlap
that normally sits between the two densities is the identity. That is a property of the model, not
an approximation made here — but it is also why this is not a drop-in for an ab-initio code, where
`Tr(P^α S P^β S)` is required.

A UHF determinant minimizes the energy without being an eigenfunction of `S²`, and nothing else in
the output says how contaminated it is: a clean doublet radical and a broken-symmetry singlet both
report `converged: true` in the same words. Measured:

| case | `S(S+1)` | `⟨S²⟩` | MOPAC |
|---|---|---|---|
| CH₃ (doublet) | 0.75 | 0.753039 | 0.753039 |
| O₂ (triplet) | 2.00 | 2.002664 | 2.002664 |
| H₂ at 1.5 Å, `follow` | 0.00 | 0.5502 | — |
| H₂ at 2.5 Å, `follow` | 0.00 | 0.9914 | — |
| H₂ at 4.0 Å, `follow` | 0.00 | **1.0000** | — |

The last row is the sharper check, because MOPAC has no say in it: a broken-symmetry singlet built
from two separated hydrogen atoms is an equal mixture of a singlet and a triplet, so `⟨S²⟩ = 1`
exactly. It is also the case where the number matters most — following a triplet instability lands
on a spin-contaminated solution *on purpose*, and that is what makes H₂ dissociate correctly.

`None` for a restricted run, where the answer is `S(S+1)` by construction, and for a k-mesh run,
where the density available is the `T = 0` block rather than the whole solution.

### Stability — is the converged solution a minimum?

`stability="check"` on any entry point, `pm7_rs.scf_stability(...)` / `PM7.get_scf_stability()` /
`pm7_rs::stability::check` on their own, `--stability check` on both command lines. **Off by
default.** See [scope.md](scope.md) for which of Seeger and Pople's channels are implemented and
[theory.md](theory.md) for the operator.

`"follow"` acts instead of reporting: rotate along the unstable eigenvector, re-converge, keep
whichever solution is lower. It cannot make an answer worse, and when it changes the spin reference
from restricted to unrestricted it **says so on stderr** — the result is then a different model from
the one asked for, and a geometry scan that switches partway through is discontinuous there.
`--stability-every N` / `stability_every=N` re-checks every `N`-th step of the built-in optimizer,
because a geometry step changes the orbitals.

---

## Molden output

`to_molden` returns the file as a `String`; `write_molden` writes it. Molecules only — Molden's
`[MO]` section is a list of molecular orbitals, not Bloch states, so a periodic system is refused.

**The caveat, which also travels inside the file:** NDDO *assumes* an orthonormal AO basis. Its
working equations are `F C = C ε` with no overlap matrix, so the `[MO]` coefficients live in an
implicitly orthogonalized basis while the listed functions are the raw, non-orthogonal Slater ones.
They differ by `S^{−1/2}`. Shapes, nodes and symmetry are faithful; bonding-region amplitudes are
approximate. This is inherent to writing an NDDO wavefunction in a format that presumes a real
basis, and MOPAC's own Molden output makes the same compromise.

### Two basis sections

`MoldenBasis::Sto` writes `[STO]`, which represents PM7's s and p shells **exactly**. It cannot
represent the d shell: Molden's primitive is a single Cartesian monomial `x^kx y^ky z^kz r^kr
e^{−αr}`, and `x²−y²` and `2z²−x²−y²` are not monomials. A d-bearing molecule is refused rather
than written with three of the five functions.

`MoldenBasis::StoNg { n }` (default 6) writes `[GTO]`, a least-squares Gaussian expansion, plus
`[5D]`. Every viewer reads it. It is a **rendering** basis: MO coefficients are unchanged so shapes
and densities are right, but anything *integrated* from the file is an STO-nG quantity.

The expansion is **fitted, not tabulated**, and reports its own quality in `[Title]` — the
normalized overlap `⟨STO|STO-nG⟩` and the radial residual. Two reasons: PM7 reaches Bi and so needs
6s and 6p, for which no published STO-nG expansion exists; and a long numerical table copied by hand
would be its own reference, with no test able to tell a transcription slip from the truth. The
worst overlap across every PM7 element at `n = 6` is asserted above 0.995.

Units follow the basis section — `[STO]` is Ångström as Molden documents it, `[GTO]` is atomic
units as every basis-set file is — so no file mixes the two.

---

## IR spectra

Frequencies, normal modes, dipole derivatives and intensities all come from the **same** CPHF
solve, so they are one call (`vibrations` in Python, `analytic_hessian_with` in Rust), never two.

```text
∂μ_α/∂R_{B,β} = q_B δ_{αβ} + Tr[ D_α · ∂P/∂R_{B,β} ]
```

— an explicit term plus a response term. Both output forms are available: the dense raw `3 × 3N`
tensor, and the mode-projected `∂μ/∂Q` with intensities in km/mol.

**The spectrum is `3N − 6` lines** (`3N − 5` linear). The translations and rotations are removed
from the mass-weighted Hessian by an Eckart projector before it is diagonalized, so there is
nothing to filter afterwards and every returned line is a vibration with a meaningful intensity.
`ir_spectrum_projected` / `projection="none"` returns the raw `3N` set when the question is what
the projector took out. Rotations of a polar molecule are **not** IR-dark, so a raw spectrum has
real intensity on those rows and reading it as spectroscopy is a mistake — which is the reason the
projected form is the default rather than an option.

### What validates it

* The **translational sum rule** `Σ_B ∂μ_α/∂R_{Bβ} = q_tot δ_{αβ}`, an exact analytic identity that
  catches a sign or index slip immediately.
* A **finite-field cross-check**: `[g(f = +h e_α) − g(f = −h e_α)]/2h` against
  `dipole_derivatives(.., FieldConjugate)`. Exact for every element including d, because both sides
  then use the same operator (C-1/C-2). A companion test asserts that running the same comparison
  against `Full` fails by *exactly* the p–d amount — the executable statement of the C-2 mismatch.
* Rotations of a *polar* molecule are **not** IR-dark, so the identity pinned is the exact one,
  `Σ_B (n × R_B)·∇_B μ = n × μ`.

MOPAC's `DIPT` is compared per mode, matched by frequency, with degenerate sets compared as
`Σ|trdip|²` over the whole set — component-wise comparison is ill-posed because eigenvector signs
are arbitrary and degenerate modes mix arbitrarily. Note that MOPAC's printed `DIPT` is **half** the
true derivative; see [fidelity.md](fidelity.md).

---

## Born effective charges and `ε^∞`

`dfpt::born_and_dielectric` returns `DfptFieldResult`:

| field | definition | convention |
|---|---|---|
| `born[A][(a, b)]` | `∂²E / ∂f_a ∂R_{A,b}` | **C-5: `a` is the field index, `b` the displacement index.** Not symmetric in general. |
| `polarizability[(a, b)]` | `∂μ_a/∂f_b` per cell, no volume factor | defined in **any** dimension |
| `dielectric[(a, b)]` | `δ_ab − (4π/Ω) ∂²E/∂f_a ∂f_b` | C-6; the minus follows from C-1. **3-D only** — with no volume it is left as the identity. For a chain or a slab use `dielectric_with_extent`, below. |

### `polarizability()` on its own

`α` is the quantity defined in **every** dimensionality, where `ε^∞` is not: a chain and a slab have
a polarizability, and neither has a dielectric constant until someone says where the material stops.
`polarizability(...)` returns exactly the tensor `born_and_dielectric` reports, without computing
the Born charges to get there.

It is in the MOPAC `FIELD=` convention (C-1), so `∂μ/∂f` carries the **opposite sign** to the
physical polarizability. `epsilon_from_polarizability` applies that flip once, internally; a caller
doing its own conversion has to.

### `dielectric_origin_sensitivity()` — measuring the approximation instead of arguing it

The position operator the field perturbation is built on is not a well-defined periodic operator.
The argument that the *response* is nevertheless well defined — an origin shift adds a constant to
the diagonal and the occupied–virtual projection annihilates it — is an argument.
`dielectric_origin_sensitivity` displaces every atom by `offset`, recomputes `α`, and returns the
largest change in any component.

A value near machine precision says the argument holds for this system. A large one says it does
not, and that the polarizability being reported is a statement about where the origin was put.
Which of those is the case is worth knowing before quoting the number.

### `ε^∞` for a chain or a slab

A slab's cell has an area, not a volume, so there is nothing to divide by. The missing ingredient
is a **thickness** or a **cross-section**, and `dielectric_with_extent` takes it as a *required*
argument: a supercell says where the atoms are, not where the material stops, so doubling the
vacuum must not change `ε` — and it would if the code took the cell height.

The conversion is a depolarization problem rather than a division. `α` here is the response to the
**external** field, so for a slab polarized along its normal the depolarizing field is already
inside it; dividing and adding one would count the screening once and the shape not at all. With
`χ = −(∂μ/∂f)/(measure · extent)` and `N` the depolarization factor,

```text
ε = 1 + 4πχ / (1 − 4πNχ)
```

| body | in plane / along the axis | across |
|---|---|---|
| slab, thickness `d` | `N = 0` | `N = 1` |
| wire, cross-section `S` | `N = 0` | `N = ½` (circular section) |
| crystal | `N = 0` | `N = 0` |

The crystal is the `N = 0` row of the same table rather than a separate rule — three-dimensional
tin-foil summation removes the macroscopic depolarizing field — and the first thing
`tests/dielectric_extent.rs` checks is that the low-dimensional formula closes on
`born_and_dielectric`'s own `ε`.

Two combinations come back alongside, and the extent **cannot change them**:

```text
(ε_∥ − 1) d = 4π α_∥ / A          (1 − 1/ε_⊥) d = 4π α_⊥ / A
```

They are what a slab can quote with no convention at all; half the first is the Rytova–Keldysh
screening length. `axis_mixing` reports how much of `α` couples the distinguished axis to its
complement, because a depolarization factor is a per-principal-axis quantity and this says how much
the conversion assumed. Zero means the axis is a principal direction and nothing was lost.

Restricted **and unrestricted**. An unrestricted cell carries a commutator `[H^σ, r]`, a band
basis and a response density per spin, and the dipole is each channel contracted against its own —
three separate things, and doing any subset of them would converge and be wrong. Forcing UHF on a
closed shell reproduces the restricted `Z*` and `α` to `10⁻⁸`, and a genuine doublet gives an
answer that is not the closed-shell one.

The explicit term in `Z*` is the **net Mulliken charge** `q_A`, not the core charge `Z_A`. With
`Z_A` a two-atom ionic cell would report `Z* = +Z` on both sites and the acoustic sum rule would
fail badly — and neither the "both contraction orders agree" check nor a convergence check would
notice, because the wrong term is common to both. Only the sum rule and a finite-difference check
can catch it, and both are in `tests/born.rs`.

### `ε⁰` — the static tensor, not just the clamped-ion one

`static_dielectric_tensor` adds what the ions contribute:

```text
ε⁰_ab = ε^∞_ab + (4π/Ω) Σ_m (Z*·e_m)_a (Z*·e_m)_b / ω_m²
```

summed over the **optical** modes of the zone-centre dynamical matrix, with `e_m` the mass-weighted
eigenvector. This is what a measured dielectric constant is usually compared against; `ε^∞` alone is
the high-frequency limit, and for an ionic crystal the two differ by a lot.

Nothing new is solved. Both ingredients — the Born charges and the Γ dynamical matrix — were
already here, so this is a contraction of quantities the crate had rather than another response.
Semiempirical codes commonly stop at `ε^∞` for want of Born charges, not for want of this formula.
The result reports `electronic` and `ionic` separately alongside `epsilon`.

**Which modes are acoustic is decided by their eigenvectors.** At `q = 0` the acoustic modes *are*
the mass-weighted uniform translations, a three-dimensional subspace known before the matrix is
diagonalized, so the three largest overlaps with it are the acoustic branch — no threshold, and
exactly three whatever the frequencies do. `skipped_modes` is therefore always three, and asserting
it checks that the projection is wired in.

Through 0.2.2 this was a magnitude test: `SOFT_MODE_FLOOR = 1e-6` in eV/(Å²·amu), with everything
below it dropped. The argument for it was that it sat in a measured gap — the acoustic branch landed
between `1e-17` and `5e-15` on the cells it was developed against, and the softest genuine optical
mode at about `2.5e-2`. But that is a statement about those cells, not about the quantity: a
frequency carries real physics at every magnitude, and a ferroelectric near its transition has
exactly the soft optical mode the floor would have swallowed — from the sum that mode dominates.

**A soft mode still makes this meaningless, and now says so.** An optical mode with `ω² ≤ 0` is
counted in `soft_optical_modes`, which is non-zero only when the geometry is not at a minimum; the
ionic term is then missing whatever those modes would have carried, which for a soft mode is most
of it. `softest_kept` is the smallest `ω²` actually used — small means one nearly-soft mode
dominates the answer. 3-D only: both halves carry a `4π/Ω` that needs `Ω` to be a volume.

#### The unit conversion was wrong by 347, and only Lyddane–Sachs–Teller said so

Worth recording, because it is the shape of error a dimensional argument cannot catch. The first
version of this returned an ionic contribution of `0.000256` for NaCl on top of an `ε^∞` of
`1.0119` — which reads as a small correction to a weakly polarizable crystal, and is wrong by two
and a half orders of magnitude.

**LST is what said so.** `ε⁰/ε^∞ = (ω_LO/ω_TO)²` for a cubic diatomic crystal, and its right-hand
side comes from the dynamical matrix with and without the non-analytic term — sharing none of the
unit conversion being checked. It required `0.0889` where the code gave `0.000256`, a ratio of
**347.0121**, which is `HARTREE_TO_EV · a₀⁴` to five figures. A clean constant ratio is a unit error
rather than physics. The conversion is `4π a₀²/Ω` now and LST agrees to `1.0000`
(`1.100809` against `1.100810`).

A dimensional re-derivation would have been checking the arithmetic against itself: the derivation
said `HARTREE_TO_EV · a₀²`, one factor of `HARTREE_TO_EV` off what the phonons require.
`tests/static_dielectric.rs` pins it from both sides — the ionic crystal against LST, and diamond
against zero. The second is not redundant: any constant times a vanishing Born charge is still zero,
so a scale test cannot catch a spurious contribution and the diamond test can.

### LO–TO

```text
D^NA_{Aα,Bβ}(q̂) = (4π/Ω) · (Σ_γ q̂_γ Z*_{A,γα})(Σ_δ q̂_δ Z*_{B,δβ}) / (q̂ · ε^∞ · q̂)
```

The **field** index of `Z*` contracts with `q̂`; the displacement index is the force-constant index
(C-7). Added to the force constants in eV/Bohr² **before** mass weighting.

`NonAnalytic::matrix(q_hat)` builds it; `DfptResult::force_constants_with_lo_to` and
`frequencies_cm_lo_to` apply it, and `ForceConstants` has the same two so either phonon route can
use it. **3-D only**, and the caller must supply `q̂` — nothing is added implicitly, because the
`q → 0` limit is direction dependent and a silently chosen direction is a wrong answer. Applying it
away from the zone centre is refused: the macroscopic field is already inside `Φ(q)` there, and
adding it again would count it twice.

### Reaching it

Pass `lo_to_direction` to either phonon route and the split frequencies come back beside the
unsplit ones, as `frequencies_cm_lo_to` — one list per q, `None` for a q away from the zone centre:

```python
out = pm7_rs.dfpt(numbers, positions, cell, [[0, 0, 0]],
                  lo_to_direction=(1, 0, 0), kpoints=(3, 3, 3))
out["frequencies_cm"][0]        # 245.7  245.7  245.7
out["frequencies_cm_lo_to"][0]  # 245.7  245.7  288.5
```

The same argument works on `pm7_rs.phonons`, on `PM7.get_dfpt` and `PM7.get_phonons`, and as
`--lo-to X,Y,Z` on both command lines. One branch — the longitudinal one along `q̂` — is pushed up
and the transverse pair is untouched, which is the whole observable content of the splitting and is
what `tests/dfpt_long_wavelength.rs` asserts.

Through v0.2.1 none of that was reachable. The four methods above had **no caller anywhere in the
repository**; what the bindings returned was the raw `D^NA` matrix, leaving a caller to add it to
the force constants, mass-weight and re-diagonalize by hand — while this document said they were
the way to use the term.

### Two independent routes to check the field response against

Everything above comes out of one CPHF. A cross-check that reuses the same solver checks the
contraction and not the physics, so 0.2.2 added two routes that share as little with it as possible:

| route | what it checks | shares with CPHF | measured |
|---|---|---|---|
| `berry_polarization` | `Z*` from `∂P/∂R` instead of `∂²E/∂f∂R` | the Hamiltonian and the basis | `tests/pbc_berry.rs` |
| `run_finite_field` | `α = Ω ∂P/∂𝓔` at finite field | only the SCF | ratio **1.0063** against the CPHF `α_xx` (0.146830 vs 0.145913) |

Both live in [pbc.md](pbc.md#polarization-and-a-field-along-a-lattice-vector), because they are
periodic-structure machinery rather than properties in their own right. Neither is a prediction of
experiment — this is a semiempirical model. What they check is that the crate computes its own
model's response consistently by more than one formalism.

### What PM7 gets right here, and what it does not

`Z*` is **quantitative**: LiF gives `+1.033` against a measured ≈1.04.

`ε^∞` is **qualitative**: 1.12 for diamond against 5.7, 1.01 for LiF against 1.92. PM7's minimal
valence basis has no polarization functions, so the response is built entirely from transitions
inside an s/p space and comes out low by a factor of two to five. This is the model, not the k mesh
— both numbers are converged to four figures by a 5×5×5 mesh. The LO–TO splitting is built from
both and inherits the weaker. See [fidelity.md](fidelity.md).
