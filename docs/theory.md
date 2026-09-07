# PM7 implementation notes

This document describes the formalism as actually implemented. See [scope.md](scope.md) for the
implementation's boundaries and known limitations.

## Conventions

Signs, factors and index orders are fixed **here**, once. They are not independent: several of
them look wrong in isolation and are only correct together, so a reader who "fixes" one of them
alone will break the others. Every property in this document cites this section rather than
restating a convention locally.

All eight are implemented. C-3, C-5, C-6 and C-7 were written ahead of the code and carried a
*"(specified; not implemented)"* marker; v0.2.1 implemented all four — the commutator with its
`(ε_m − ε_n)` denominator in `dfpt::field_commutator_blocks`, the Born-charge index order in
`DfptFieldResult::born`, the dielectric sign in `born_and_dielectric`, and the non-analytic term in
`NonAnalytic::matrix` — and did not come back to delete the markers, which therefore outlived the
code they described by a whole release. They are gone as of 0.2.2. A convention labelled
unimplemented next to code that implements it is worse than no label, because it invites the reader
to conclude the whole feature is absent.

### C-1 · The electric field is MOPAC's, and its sign is the potential gradient

MOPAC's `FIELD=(x,y,z)` vector is the gradient of the potential, not the physical field.
`hcore.F90:204` puts `−(F·R_A)` on the electron diagonal and `+Z_A (F·R_A)` into `enuclr`, so the
field energy is

```text
E(F) = E_0 + F · mu           (not −F·mu)      hence  F = −E_physical
```

`pm7-rs` keeps MOPAC's sign, because `FIELD=` is what the number means. The input unit is
volts/Ångström; internally `f = F · a_0`, in eV per e·Bohr, so positions stay in Bohr.

### C-2 · There are three dipole operators, and they are not interchangeable

| `DipoleTerms` | terms | who uses it |
|---|---|---|
| `PointCharge` | `Σ_A q_A R_A` | diagnostics only |
| `FieldConjugate` | + the one-centre s–p hybrid (`−2 dd_A`) | **MOPAC's `FIELD` operator** |
| `Full` | + the one-centre p–d hybrid (`ddp(5)`) | **MOPAC's printed dipole**, and the physical operator |

MOPAC's field operator (`hcore.F90:191-202`) has **no p–d term**, while MOPAC's dipole
(`dipole.F90:118-148`) does. So in MOPAC's own convention `∂E/∂F = mu_FieldConjugate`, which
differs from the reported `mu` for any atom carrying d orbitals. That is a property of the model as
published, not a defect here; `pm7-rs` names the two operators separately rather than silently
picking one, and every cross-check states which it uses.

### C-3 · The commutator identity

Because `H|n⟩ = ε_n|n⟩`,

```text
⟨m|[H, r]|n⟩ = (ε_m − ε_n) ⟨m|r|n⟩        hence     ⟨m|r_a|n⟩ = ⟨m|[H, r_a]|n⟩ / (ε_m − ε_n)
```

for `m ≠ n`. The denominator is `(ε_m − ε_n)` — row index first. This is how a homogeneous field
enters a periodic calculation, because `r` itself is not a periodic operator while `[H, r]` is. The
molecular field needs none of this: there `r` is perfectly well defined and the operator is built
directly (see [`crate::field`] and the *Post-SCF corrections* section below).

The identity above is exact; what it rests on is that the *response* is insensitive to the origin,
because an origin shift adds a constant to the diagonal and the occupied–virtual projection
annihilates it. That is an argument, so `dielectric_origin_sensitivity` measures it instead.

There is also a route that uses no position operator at all — the Berry phase of the occupied
manifold (`crate::pbc::berry`), which is a product of overlaps between neighbouring k points. It is
what gives a **finite** field along a lattice vector something to couple to, since `−f·r` is
unbounded there and the commutator identity is a statement about linear response rather than about
a finite perturbation.

### C-4 · Spin factors and k weights

The restricted density is `P = 2 Σ_occ c cᵀ`, so response traces carry the same factor of 2; the
`4` in the Hessian's relaxation term is that factor times the `U`/`Uᵀ` pair. k-point weights sum to
1, and the Bloch phase is the **cell** convention `e^{i k·T}` taken from fractional coordinates, so
`k·T` is exact for integer translations.

### C-5 · Born-charge index order

`Z*_{A,ab} = ∂²E / ∂f_a ∂R_{A,b}` — **`a` is the field (polarization) index, `b` is the
displacement index**. It is not symmetric in general, so the order matters wherever `Z*` is
contracted.

### C-6 · The dielectric tensor's minus sign is correct

With C-1, the polarizability is `alpha_ab = −∂²E/∂f_a ∂f_b`, and `eps = 1 + (4π/Ω) alpha`:

```text
eps_ab = delta_ab − (4π/Ω) · ∂²E/∂f_a ∂f_b
```

The minus sign follows from C-1 and must not be "corrected" to a plus.

### C-7 · The LO–TO non-analytic term

```text
D^NA_{Aa,Bb}(q̂) = (4π/Ω) · (Σ_c q̂_c Z*_{A,ca}) (Σ_d q̂_d Z*_{B,db}) / (q̂ · eps · q̂)
```

The **field** index contracts with `q̂`; the **displacement** index is the force-constant index. It
is added to the force constants (eV/Bohr²) **before** mass weighting, and only for a 3-D cell.
Nothing adds it implicitly, because the `q → 0` limit is direction dependent and a silently chosen
direction is a wrong answer rather than an approximate one.

### C-8 · The raw dipole-derivative tensor uses a fixed origin

`∂mu/∂R` is **always** taken about the input coordinate origin, whatever `DipoleOrigin` says. A
moving origin would add `−q_tot · m_B/M` and make the tensor convention-dependent for an ion.
Mode-projected IR intensities are origin-independent for a neutral molecule and origin-dependent
for a charged one — both facts follow from the translational sum rule below, and MOPAC arrives at
the same place by disabling its centre-of-mass recentring under `FORCE` (`dipole.F90:86-88`).

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
  solve. Because CPHF is a *linear* system whose operator is symmetric positive definite at a
  stable closed-shell solution, the restricted solver is a **preconditioned conjugate gradient**,
  which converges on `√κ` rather than on a fixed-point spectral radius. The DIIS-accelerated fixed
  point that preceded it is kept as the fallback for an operator that is not positive definite — a
  saddle point, or an SCF solution that is not a minimum — where CG has no meaning. Either way an
  unconverged response is an **error**, not a returned matrix, and the budget both spend is
  `Pm7Options::cphf_max_iterations`. See [performance.md](performance.md#cphf-by-conjugate-gradient).
- **Correction derivatives** — dispersion and the PM7-HH repulsion are pairwise and differentiate
  directly at `Dual2`. The hydrogen-bond term is many-body, so its Hessian uses a multi-variable
  second-order dual, `Dual2N<27>`: each bond couples at most nine atoms (27 Cartesian degrees of
  freedom), so one AD pass yields that bond's exact local second-derivative block, and the total
  is the sum over bonds.
- **Frame singularities** — a bond lying exactly on the singular axis of a local two-center
  rotation frame would make the forward-mode derivative of that rotation collapse. Those pairs
  (and only those) fall back to a symmetric finite difference of the correct-value integrals, so
  no orientation loses its perpendicular force or curvature component.

## Stability of the SCF solution

The same operator, asked a different question. A stationary point of the energy with respect to
occupied–virtual orbital rotations has second derivative

```text
A U = (ε_a − ε_i) U + [G(ΔP(U))]_ov
```

— which is exactly what the CPHF multiplies by when it solves `A U = −G_skel`. The CPHF asks for
`A⁻¹` applied to a fixed right-hand side; the stability analysis asks for the **sign of the lowest
eigenvalue** of the same `A`. Positive is a minimum, negative is a saddle, and the eigenvector is
the direction out. No new integrals and no new kernel.

`A` here is `A + B` in the usual RPA labelling: the second derivative with respect to **real**
orbital rotations. Two channels are built from it, differing only in the kernel `G`:

| channel | kernel | asks |
|---|---|---|
| singlet | full `G` (Coulomb + exchange) | is this a minimum among closed-shell solutions? |
| triplet | exchange only, since `J(0) − K(Δ) = −K(Δ)` | is it still one when α and β may differ? |

Both are needed. Stretched H₂ is a perfectly good minimum in the first and a saddle in the
second — the classic RHF dissociation failure — and a check that ran only the singlet channel would
report `+10.70 eV` and a heat of formation 121.6 kcal/mol above the right answer. The complex
channel (`A − B`) is not implemented; see [scope.md](scope.md).

### Spin contamination

An unrestricted determinant minimizes the energy without being an eigenfunction of `S²`:

```text
⟨S²⟩ = S_z(S_z + 1) + n_β − Tr(P^α P^β),    S_z = (n_α − n_β)/2
```

The trace form rather than `Tr(P^α S P^β S)` because the NDDO basis is orthonormal by construction.
This is what makes following a triplet instability meaningful and also what it costs: the escaped
solution at H₂'s dissociation limit is `⟨S²⟩ = 1` exactly, an equal mixture of the singlet and the
triplet, which is the correct energy from the wrong kind of wavefunction.

## Periodic formulation

### The split that makes it exact

PM7's feathering interpolates every two-centre integral onto the point charge and *stays there*
beyond 7 Å. So

```text
I_{μν,λσ}(R) = [ I_{μν,λσ}(R) − δ_μν δ_λσ v(R) ]  +  δ_μν δ_λσ v(R)
```

is an identity in which the first bracket has **compact support** — identically zero past 7 Å —
and the second is a lattice sum of `1/r`, which is what Ewald is for. No parameter decides the
split point; PM7's functional form does. The same decomposition handles the core–core repulsion
and the electron–core attraction.

### The exchange, and why the naive treatment fails

MOPAC evaluates the exchange from the nearest image only. That is sound inside a `makpol`
supercell, where every chemically meaningful pair is at `T = 0`, and wrong in a primitive cell,
where symmetry-equivalent partners sit at *different* translations: picking one breaks the
symmetry and produces a spurious force (1.3 eV/Å on a 2-D BN sheet). Summing all images naively
diverges instead, because at Γ the density matrix does not decay.

The resolution is to regularize the exchange monopole with the same Ewald sum the Coulomb term
uses:

```text
K[μ_A, λ_B] = −Σ_{T short} P^σ(ν_A, σ_B)·[w_T(μν|λσ) − δ_μν δ_λσ v_T]  −  P^σ(μ_A, λ_B)·Φ_AB
```

`Φ_AB` is the regularized lattice sum already built for the Coulomb term, so the extra cost is one
matrix lookup. Symmetric, finite, and exact in the molecular limit.

### k points

`H(k) = Σ_T H(T) e^{ik·T}` with real `H(T)`, so `H(−k) = H(k)*` and the two fold onto one
representative with double weight. The density comes back as
`P(T) = Σ_k w_k Re[e^{−ik·T} P(k)]`. Occupations use one Fermi level across the whole mesh — and
across both spin channels for UHF — because per-k aufbau filling lets charge pool wherever the
levels happen to be lowest.

### Stress is exact, and there is no Pulay term

Every PM7 term depends on the nuclei only through pair displacement vectors `d = R_B − R_A + T`,
so the virial form `σ_αβ = (1/Ω) Σ (∂E/∂d_α) d_β` is an identity rather than an approximation. And
the NDDO basis is orthonormal with no cell dependence, so unlike a Gaussian-basis code there is no
basis-set (Pulay) stress at all. The long-range part needs more than the pair virial because the
reciprocal sum depends on the cell through `1/Ω` and through the reciprocal vectors.

### Perturbation theory at arbitrary `q`

A phonon at wavevector `q` mixes `k` with `k + q`, which is a different calculation rather than a
bigger one. For each `k`,

```text
ΔP(k)_{mn} = [f_n(k) − f_m(k+q)] / [ε_n(k) − ε_m(k+q)] · ⟨ψ_{m,k+q}| ΔV |ψ_{n,k}⟩
```

and the response feeds back through the same two-electron kernel at momentum transfer `q`, solved
as a complex fixed point with Pulay acceleration. The force constants are then

```text
Φ_{jj'}(q) = Φ^skeleton_{jj'}(q) + Σ_k w_k Tr[ Δh^{j†}(k) ΔP^{j'}(k) ]
```

— the *bare* perturbation against the *self-consistent* response, which is what the 2n+1 theorem
leaves once the two-electron double counting cancels. Two details are load-bearing and easy to get
wrong: the k mesh must **not** be time-reversal folded (folding pairs `k` with `−k`, and a `q ≠ 0`
response pairs `k` with `k + q`), and the second-order energy must contract `ΔP(k)` in k space
rather than round-tripping through real space, where the block set is far larger than the mesh and
a Bloch sum over it over-counts.

### Divide and conquer

Subsystems are cores plus buffers, each diagonalized separately, and two things hold them
together: **one global Fermi level**, found by bisection on the total electron count, and an
**environment potential** carrying the monopole field of everything a subsystem does not contain.
The global density is reassembled with Dixon–Merz weights (1 for core–core, ½ for core–buffer),
normalized by the realized weight total so they are a partition of unity even at buffer edges.

The environment potential splits into a nuclear half that belongs in `h_core` and an electronic
half that belongs in the Fock. `½ Tr P(H + F)` counts the first once and the second half — folding
both into `h_core`, which is the obvious reading of "external potential", double counts the
electronic half once per subsystem, an `N²` error.
## Numerics

Dense linear algebra uses `faer` (pure Rust; no LAPACK/BLAS). Pair- and coordinate-level loops
are parallelized with `rayon` in a **bit-identical** way: work is split across threads but each
result accumulates in the original order, so the thread count never changes the numbers.
