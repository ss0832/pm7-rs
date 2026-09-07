# Scope and status

`pm7-rs` implements the PM7 semiempirical NDDO method and its PM7-family methods for **molecules
and for periodic systems in one, two and three dimensions**, validated against the official MOPAC
v23.2.5 executable.

## What is implemented

- **Elements** — the full minimal-valence s/p/d MNDO/d kernel: 1 AO (s), 4 AOs (s,p), 9 AOs
  (s,p,d), and 0 AOs for Sparkle lanthanide sites (Z 58–70, treated as +3 point cores).
- **Methods** — `Pm7`, `Pm7Ts`, `Pm7Minus` (corrections off), `Pm7Hh`, and `Pm7Sparkle`, selected
  through the `method` field/keyword on every API surface.
- **SCF** — RHF and UHF, with the spin treatment selectable independently of the multiplicity via
  `ScfReference::{Auto, Restricted, Unrestricted}`. A level-shift controller engages when a run
  stalls; see [performance.md](performance.md).
- **Post-SCF corrections** — PM6-DH-style D2 dispersion, the PM7 EH+ hydrogen-bond correction, and
  the PM7-HH H–H repulsion. All three are periodic, in the energy, gradient, stress and force
  constants.
- **Periodic boundary conditions** — 1-D, 2-D and 3-D; Γ point and Monkhorst–Pack k meshes;
  neutral and charged cells; restricted and unrestricted. Energy, analytic gradient, analytic
  stress and analytic zone-centre force constants. See [pbc.md](pbc.md).

  **Any per-axis pattern**, on every surface: `--pbc TFT` on both command lines, `pbc=(True,
  False, True)` in Python, and an ASE `Atoms` built as a slab along *y*. A `Cell` still stores its
  periodic vectors contiguously — that is what keeps `cell.dim()` and `cell.vectors()` sufficient
  for every periodic module — and a non-leading pattern is reached by cyclically reordering the
  three lattice vectors, with `--kpoints`, `--kshift`, `--supercell` and fractional `q` reordered
  with them so everything stays in the caller's axis order. Through 0.2.2 all three surfaces
  refused it and told the user to reorder the cell themselves.
- **Perturbation theory at arbitrary `q`** — `dynamical_matrix_dfpt` solves the linear response
  directly at each wavevector, coupling `k` with `k + q` through a complex CPHF, with no supercell
  and no commensurability condition, for **restricted and unrestricted** cells and for **smeared
  metals**. The unrestricted path carries a band set and a response density per spin: Coulomb
  couples each channel to the total density, exchange only to its own. Validated against the
  zone-centre analytic Hessian, against the numerical Hessian on multi-point meshes, and against a
  commensurate supercell. A response that fails to converge is an error, not a flag — it is a
  linear fixed point, so a failure is a divergence rather than a near miss.
- **Phonons** — real-space force constants `Φ(0A, TB)` from a supercell's analytic Hessian, the
  dynamical matrix `D(q)` at any wavevector, frequencies in cm⁻¹, and the acoustic sum rule as
  both a diagnostic and a projection — **enforced by default since 0.2.3**, with the residual
  reported either way and `--no-acoustic-sum-rule` / `acoustic_sum_rule=False` to keep the raw set.
- **Geometry optimization** — L-BFGS on the analytic gradient, with the **lattice** as an optional
  degree of freedom (`--opt-cell`, `relax_cell=True`). The variables are the strain components
  inside the periodic subspace, which makes the analytic stress their exact conjugate; the atoms
  are carried affinely so a cell step and an atomic step are independent directions. Two separate
  convergence tests, on the force and on the largest free stress component. Opt-in, because the
  fixed-cell behaviour is what every published periodic number was computed with.
- **Divide and conquer** — a linear-scaling SCF for large systems, with gradient and stress, for
  molecules and for periodic cells, restricted and unrestricted. See
  [divide_and_conquer.md](divide_and_conquer.md).
- **Derivatives** — forward-mode AD gradients and AD/CPHF(UCPHF) Hessians for both RHF and UHF,
  harmonic vibrational analysis, and L-BFGS geometry optimization. The CPHF is solved by
  preconditioned conjugate gradient, with the damped DIIS fixed point as the fallback for an
  orbital Hessian that is not positive definite; a response that does not converge is an **error**,
  not a silently returned last iterate.
- **SCF stability analysis** — Seeger–Pople, on the same orbital Hessian the CPHF already applies:
  is the converged solution a minimum, or only a stationary point? Singlet and triplet channels for
  a restricted solution, the internal channel for an unrestricted one, and — with
  `stability="follow"` — a rotation along the unstable eigenvector and a re-converged SCF, keeping
  whichever solution is lower. Reachable as its own call (`scf_stability`, `stability::check`), as
  an option on every entry point, and every `n`-th step of the built-in optimizer
  (`stability_every`). **Off by default.** See the limitation note below for the channel it omits.
- **External electric field** — MOPAC's `FIELD=(x,y,z)` operator in volts/Ångström, with the
  energy, an exact closed-form analytic gradient (`∂E/∂R_A = q_A f`), and an analytic Hessian
  through the CPHF. Molecules, and any **non-periodic** direction of a periodic cell; a component
  along a periodic direction is refused, because the potential `−f·r` is unbounded there — that
  case is `run_finite_field`, below.
- **Properties** — the dipole moment split into its point-charge, s–p hybrid and p–d hybrid terms
  (matching MOPAC's `POINT-CHG. / HYBRID / SUM`), orbital energies and coefficients for both spin
  channels, and the frontier gap. See [theory.md](theory.md) for the sign and operator conventions,
  which are load-bearing and not all obvious.
- **Field response of a cell** — Born effective charges, the polarizability `α = ∂μ/∂f`, the
  clamped-ion `ε^∞` (3-D) and `dielectric_with_extent` for a chain or a slab, the static
  `static_dielectric_tensor` (`ε⁰ = ε^∞ +` the ionic term), the LO–TO non-analytic term, and
  `dielectric_origin_sensitivity`, which measures the approximation the position operator rests on
  rather than arguing it. Restricted and unrestricted. See [properties.md](properties.md).
- **Polarization and a finite field along a periodic direction** — `berry_polarization` gives the
  King-Smith–Vanderbilt polarization modulo its quantum, and `run_finite_field` converges a cell in
  a finite field applied *along* a lattice vector by minimizing the Nunes–Gonze electric enthalpy
  `E − Ω 𝓔·P`. Both are 3-D, closed-shell, and exist as **independent checks**: the Berry route
  shares only the Hamiltonian and the basis with the CPHF Born charges, and the finite field agrees
  with the CPHF polarizability to a ratio of 1.0063 on formalisms that share only the SCF.
- **Band structures** — `band_structure` diagonalizes the converged Fock along a k path, reachable
  from ASE as `PM7.get_band_structure`.
- **Interfaces** — Rust library, a `pm7_rs_cli` command-line tool, Python native bindings
  (`pm7_rs.native`), a `pm7-rs` console script (`python -m pm7_rs`), and an ASE calculator
  (`pm7_rs.ase.PM7`, eV/Å, with stress in Voigt order so `FrechetCellFilter` and `NPT` work). The
  two command lines offer the **same** fifteen modes — `energy`, `charges`, `gradient`, `forces`,
  `stress`, `optimize`, `frequencies`, `hessian`, `phonons`, `dfpt`, `born`, `bands`, `orbitals`,
  `molden`, `dielectric` — and the same flags, both enumerated by tests to keep it that way.

  The matrix's `pytest -m matrix` and `--ignored` tiers are the ones that earn their keep: between
  them they found the divide-and-conquer OOM abort, `born`/`dfpt` crashing on a molecule instead of
  refusing, and `charges --dandc` ending in a `KeyError` after printing a good energy. All three
  are the same shape — a crash where a refusal belongs — and none was reachable from the default
  tier.

  Mode parity was tested from 0.2.1; **flag** parity was not, and had drifted in seven places by
  0.2.2. Four of them (`--scf-tolerance`, `--max-scf`, `--no-diis`, `--exchange-cutoff`) were
  Rust-CLI-only, and since the wheel ships no Rust binary they were unreachable to anyone who
  installed with `pip` — precisely the four knobs someone reaches for when an SCF will not
  converge. Closed in 0.2.3, along with a `--output` that the Rust CLI's own per-mode flag table
  and usage text both advertised with no parse arm behind either.

## Boundaries and known limitations

These are properties of the current implementation, stated explicitly rather than silently
approximated.

### Periodic

- **Γ-only sampling of a small cell is qualitatively wrong for the exchange**, not merely coarse:
  at Γ the density matrix does not decay with distance, so the long-range exchange is over-counted.
  For a two-atom diamond cell it costs 29 eV/atom. Use a k mesh, or a large enough supercell.
- **The analytic Hessian is the zone centre.** For a closed shell at Γ it is the periodic analytic
  Hessian; on a k mesh **or for an open shell** it delegates to `dynamical_matrix_dfpt` at `q = 0`,
  which is the same calculation reached the way the `k ↔ k + q` coupling requires. The open-shell
  arm is why the delegation covers a Γ mesh too: `analytic_hessian_periodic` has no unrestricted
  CPHF at all and refuses, while the perturbation solver has carried a band set per spin since the
  UHF field response landed, so at `q = 0` it produces exactly the matrix that refusal was standing
  in front of. An open-shell CH₂ chain returns a `9×9` symmetric to `0.00e+00`. A second UCPHF
  written to reach the same number would be two implementations to keep in step. Other wavevectors
  come from `dynamical_matrix_dfpt` or from a supercell at Γ, which stays exact at every
  commensurate `q`.

  The k-mesh route is **much more expensive** than the Γ one: it is `3N` linear-response solves over
  the unfolded mesh rather than one CPHF at a single point. Through 0.2.2 it also ran the ground
  state **twice** — once for the result's `scf` field and once inside the response. It no longer
  does: `DfptResult` carries the ground state it converged, and the k-mesh arm takes that instead of
  running its own. The arm is chosen from the mesh alone, which is what made the first SCF
  removable; the Γ-plus-open-shell arm still runs one of its own, because `KMesh::Gamma` does not
  build the translation-resolved Hamiltonian the response needs and the solver converges a
  different object.

  Through v0.2.1 a k mesh was refused here and the vibrational group was worse than that: the
  `vibrations` entry point took no periodic keywords at all, so `PM7(...).get_frequencies()` on a
  periodic cell returned the frequencies of its atoms **as an isolated molecule**, silently. Fixed
  in 0.2.2, with infrared refused for a periodic cell rather than answered molecularly.
- **The k-point gradient and stress no longer cost more than the energy.** Through v0.2.1 they went
  as roughly the square of the mesh size where the SCF goes as the first power, because the
  long-range exchange derivative needed one shifted Ewald sum per Born–von Kármán residue class. In
  0.2.2 every class's reciprocal sum is done in one pass over `G`, using a structure factor over
  lattice translations; the measured exponent in the class count fell from 2.05 to 0.65, and a
  7×7×7 stress from 2.81 s to 0.018 s. The fold is **3-D only** — in one and two dimensions the
  class count is too small for the square to matter, and those keep the per-class loop.
  [performance.md](performance.md) has the numbers and the reasoning.
- **A coarse mesh can break a degeneracy the symmetry requires**, and that is the mesh rather than
  the solver. Along `[1,0,0]` in diamond the two transverse acoustic branches are degenerate; at
  `q = (0.25, 0, 0)` on a 2×2×2 mesh DFPT splits them by 0.098 cm⁻¹ (`377.2476` against
  `377.3460`). Refining closes it exactly — 0.0000 at 3×3×3, 4×4×4 and 5×5×5 — while tightening
  `dfpt_tolerance` from `1e-8` to `1e-12` or `scf_tolerance` from `1e-7` to `1e-11` moves it not at
  all. The 2×2×2 mesh does not offer `k` and `k + q` sets related by the symmetry the degeneracy
  rests on; it is also the mesh that puts the frequency 22 cm⁻¹ below the converged value.
- **A coarse mesh can also make an acoustic branch imaginary away from the high-symmetry points**,
  which is the same limitation wearing a much more alarming costume: an imaginary frequency reads as
  a structural instability, and a soft-mode hunt is an expensive thing to start for no reason.
  `dfpt` needs no supercell, and it is easy to carry that over to the k mesh, which it does *not*
  free you from — the response couples `k` with `k + q`, so the ground-state sampling still decides
  which wavevectors are resolved. Measured on Zn(CN)₂ at the relaxed cubic cell, taking the softest
  wavevector on the Γ–X–M–Γ–R–X path: **−72.64 cm⁻¹ at 2×2×2, −37.44 at 3×3×3, −22.55 at 4×4×4** —
  halving with each refinement and on its way to zero, while Γ stays at −0.00 (the acoustic sum
  rule holds it there) and every zone point stays stable. The distinguishing test is that trend
  plus where the softness sits: a genuine soft mode is usually softest *at* a zone point, because
  that is where it would give an ordered lower-symmetry structure, and it does not shrink when the
  ground state is sampled more finely. `dfpt` already warns when `q` is finer than half the mesh
  step; that catches the small-`q` half of this and not the rest.
- **A small frontier gap makes the supercell CPHF slow, and the budget for it is now a knob.**
  Through 0.2.2 it was a private constant of 100 that no interface could raise, so a cell whose
  orbital Hessian is ill-conditioned failed `phonons` outright and the refusal could only suggest a
  different mode. `--cphf-max-iterations` / `cphf_max_iterations` / `Pm7Options` set it from 0.2.3.

  Measured on cubic SrTiO₃, whose PM7 gap is 0.28 eV, at `phonons --supercell 2 2 2`: residual
  `6e-9` at the default 100, `9e-7` at 200, `1e-7` at 400, `2e-9` at 800, and **converged at
  1600** — 144 s, fifteen modes, the lowest at −289.0 cm⁻¹, which is the instability the
  ten-crystal audit finds for this cell by its own route. Two things are worth reading off that
  sequence. The budget genuinely was the binding constraint. And the residual is **not monotone in
  it**, so a run that stops at 9e-7 is not evidence that the answer is out of reach; the conjugate
  gradient hands over to the DIIS fixed point when the operator stops looking positive definite,
  and a fixed point's last iterate is not its best one.

  `dfpt` is still far cheaper here — 13 iterations to `9.2e-11` in 4 s — because it solves the
  response directly rather than through a supercell Hessian. It is **not** a cross-check on the
  number, though: `--supercell 2 2 2` samples the zone on a 2×2×2 mesh and a bare `dfpt` call
  samples it at Γ, so the two spectra differ (−412.1 to 448.8 against −289.0 to 512.5) for the
  ordinary reason that they are different calculations. Tightening `--scf-tolerance` also improves
  the conditioning, and on its own was not sufficient here.
- **Charged cells depend on the background convention.** The neutralizing jellium is reported
  separately (`background_ev`) because the absolute energy is convention-dependent; the
  Makov–Payne estimate is reported as a diagnostic and never added.
- **MOPAC-compatibility mode is Γ-only and has no stress.** Its absolute energy depends on the
  truncation distance. It exists to reproduce MOPAC's own solid-state numbers, not as a default.
- **Only k-mesh shifts of 0 and ½ are supported**, per periodic direction. The mesh `{(i+s)/n}` is
  closed under `k → −k` only when `2s` is an integer, and both the real-part density assembly
  `P(T) = Σ_k w_k Re[e^{−ik·T}P(k)]` and the Born–von Kármán exchange construction require that
  closure. Any other shift is now **refused**; before v0.2.1 it was accepted and made the SCF stall
  at a small non-zero residual (5.8e-6 on a 1-D HF chain at `shift = 0.25`, unchanged after 2000
  iterations and unaffected by smearing, in a system with a 19 eV gap) — which reads as a
  convergence problem and is really an unrepresentable request.
- **The EH+ gradient diverges on the acceptor's dihedral axis.** The correction uses a dihedral
  about the acceptor's `R–X` axis, which is undefined when the hydrogen sits on that axis, and
  nothing damps it there — the force grows as `1/ρ`, reaching 50 kcal/mol/Bohr half a degree out
  against about 3 for a normal hydrogen bond, and MOPAC's degenerate branch adds a 0.26 kcal/mol
  step within `1e-6` rad. A property of the model, MOPAC's included; measured in
  [singularities.md](singularities.md), which also gives the repair and why it is not applied.

### Perturbation theory and fields

- **A `q` finer than the k mesh gives a wrong answer, with a warning rather than a refusal.** The
  response couples `k` with `k + q`, so an `n × n × n` mesh cannot resolve `q ≪ 1/n`. Diamond on a
  `3³` mesh at `q = 1/160` returns acoustic modes at −2977 and −231 cm⁻¹ where three near-zero ones
  belong. It is a sampling limit, not a defect in the construction — the residue converges away
  with the mesh and every identity stays exact — so it warns and proceeds, because there is no
  sharp threshold and a coarse survey is a legitimate thing to run. `PM7_QUIET` silences it. See
  [pbc.md](pbc.md#the-k-mesh-has-to-be-able-to-hold-your-q).

  (Two entries stood here through v0.2.1 and both were **wrong**. One said `dynamical_matrix_dfpt`
  reached no binding: it had reached all five surfaces since 0.2.1 shipped. The other was true —
  the LO–TO application methods had no caller anywhere — and is fixed in 0.2.2, where
  `lo_to_direction` on either phonon route returns the split frequencies directly. What was
  genuinely missing was the *package* namespace: `pm7_rs.dfpt` raised `AttributeError` while
  `pm7_rs.frequencies` worked, which read as the feature being absent.)
- ~~**Metallic DFPT is not supported.**~~ **Supported since 0.2.2, with smearing, except for the
  Fermi-level shift at `q = 0`.** The general band-pair form `[f_n(k) − f_m(k+q)]/[ε_n(k) −
  ε_m(k+q)]` was always here and the `k` and `k + q` meshes were always filled together against one
  Fermi level. What refused a metal was the **gate**: any fractional occupation was turned away,
  which excludes every smeared metal — including the ones where smearing is exactly what makes the
  response well defined.

  The gate is now on **gaplessness without smearing**, which is the condition that actually makes
  the denominators singular. Where a band crosses `E_F`, `Δf/Δε` is a `0/0`; a smeared occupation
  makes it finite because `Δf` then goes to zero with `Δε` at a rate the smearing function fixes,
  while an unsmeared step leaves the answer decided by which pairs fell inside the `1e-8`
  denominator floor — a property of the floor rather than of the system. bcc Li is refused without
  smearing, with the message naming the highest occupied level, the lowest empty one and `E_F`;
  with Fermi–Dirac at 0.3 eV it converges to residual `1.6e-11` with `D(q)` Hermitian to `1.4e-17`.

  **What is still missing is the Fermi-level shift and its intraband term** (de Gironcoli, *Phys.
  Rev. B* **51**, 6773 (1995)). A `q = 0` perturbation of a metal moves `E_F`; for `q ≠ 0` that term
  vanishes by symmetry, because the perturbation has no uniform component to shift the chemical
  potential with. So the wavevectors a phonon dispersion is made of are complete and the zone centre
  of a metal is not.

  **From 0.2.3 that zone centre is refused rather than answered.** Documenting an incompleteness is
  not the same as reporting it: `phonons`, `born_charges` and `static_dielectric` all reach `q = 0`,
  and an incomplete response there is indistinguishable from a complete one in the result. The gate
  is on the occupations — not on the entropy, which Methfessel–Paxton can take through zero with the
  occupations still fractional — at a threshold of `1e-3`, chosen from the gap between the two
  populations rather than from a derivation. On PM7 ZnS, whose gap is a strong function of the mesh:
  a *gapless* 3×3×3 or 4×4×4 mesh puts **a third of a state** on either side of `E_F`, while every
  gapped case sits at `5.7e-6` or below. The refusal names the three ways on — any `q ≠ 0`, a finer
  mesh that opens a gap a coarse one missed, or a narrower width on a cell that does have one. A
  smearing that leaves the occupations integral, which is the ordinary use, is unaffected.

  This is why `examples/crystal_phonons.py` runs ZnS and CaF₂ at 4×4×4 rather than 3×3×3: on the
  coarse mesh PM7 makes them spuriously metallic, and the zone-centre spectra they used to return —
  with the right degeneracies, which is what made them convincing — were incomplete.
- **LO–TO splitting is 3-D only.** The non-analytic term's `q → 0` limit has a different form in
  one and two dimensions, where the macroscopic field of a polarization wave is not
  `4π (q·Z*)² / (q·ε^∞·q)`. `DfptFieldResult::non_analytic` returns an error for a cell with no
  volume rather than inventing one, and the caller must supply `q̂` — the limit is
  direction-dependent, so a silently chosen direction would be a wrong answer.
- ~~**Born charges and `ε^∞` are closed-shell only.**~~ **Unrestricted since 0.2.2.** The field
  reaches the response through the commutator `[H, r]`, and an unrestricted cell has two different
  `H` — so it now carries a commutator, a band basis and a response density **per spin**, and the
  dipole is the sum of each channel contracted against its own. That was exactly what the refusal
  said had to happen; it is what happens.

  Validated the way the phonon path was: forcing UHF on a **closed shell** reproduces the
  restricted Born charges and polarizability to `10⁻⁸`, which is the check that would catch a
  factor of two, a channel counted twice, one commutator serving both spins, or a contraction done
  in the wrong band basis. A genuine doublet satisfies `Σ_A Z*_A = 0` to `10⁻⁸` and gives an answer
  that is **not** the closed-shell one — the second half matters, because an implementation that
  quietly averaged the channels would pass every sum rule while returning the restricted result.
- **`ε^∞` is qualitative, `Z*` is quantitative.** PM7's minimal valence basis has no polarization
  functions, so the dielectric tensor comes out low by a factor of two to five (diamond 1.12 vs
  5.7; LiF 1.01 vs 1.92) while Born charges land on experiment (LiF `Z* = +1.03` vs ~1.04). This
  is the model, not the k mesh: both are converged to four figures by `5×5×5`. See
  [fidelity.md](fidelity.md).
- **`Z*` along a periodic direction needs a converged k mesh, and a folded supercell needs more
  of one.** A cell and its commensurate double agree to `10⁻⁶` once the mesh is reasonable
  (`dbl n` vs `prim 2n` for `n ≥ 3` on an HF chain), but disagree by ~5 % at the two coarsest
  meshes, where the primitive cell is itself far from converged. The **transverse** components
  agree to `10⁻⁹` at every mesh including the coarsest, so the effect is specific to the periodic
  axis, where a folded cell has to recover from degenerate band pairs at one k what the primitive
  cell reads off distinct k points. Both facts are asserted in `tests/born.rs`.
- **No stress under an external field.** The field's strain derivative is not part of the virial,
  so `analytic_stress` refuses rather than returning the field-free stress. Forces under a field
  are exact and available from `closed_form_gradient`.
- **`Pm7Options::field` is transverse only, and there is a separate route along the lattice.** The
  ordinary field operator takes field components along **non-periodic** directions; along a periodic
  direction `−f·r` is unbounded, the spectrum has no lower bound, and the ground state of `H − 𝓔·R`
  on a lattice does not exist, so it is refused rather than approximated. `run_finite_field`
  covers the other case by minimizing the electric enthalpy `E − Ω 𝓔·P` instead, with `P` the
  Berry-phase polarization. It is **3-D and closed-shell**, its `divisions` argument is the k mesh
  and the Berry string length at once, and an axis with fewer than three points is reported
  unresolved rather than as zero. The two treatments refuse to be combined.

  Divide and conquer refuses a field outright, because each subsystem builds its own core
  Hamiltonian.
- **The Berry-phase polarization is 3-D and closed-shell.** The quantum is `a_α/Ω` and `Ω` has to
  be a volume; a slab or a chain has a polarization along its periodic directions only, which this
  does not separate out. Fewer than three points on a string is refused: the discretized phase is a
  product of nearest-neighbour overlaps and two points cannot resolve a winding. `strings` is the
  convergence parameter and the answer has to stop moving with it.
- **`ε⁰` is 3-D, and a soft mode makes it meaningless.** Both halves of
  `static_dielectric_tensor` carry a `4π/Ω` that needs `Ω` to be a volume, and the ionic term's
  `1/ω²` means a mode near zero dominates. The acoustic modes are removed by their overlap with the
  mass-weighted uniform translations — a subspace of dimension three, so `skipped_modes` is three by
  construction rather than by a threshold. `soft_optical_modes` counts optical modes at or below
  zero: non-zero means the geometry is not at a minimum and the number returned is missing most of
  the ionic term while still looking like an answer. `SOFT_MODE_FLOOR`, the `1e-6` cut this used
  through 0.2.2, is **removed** — a breaking change, and the second of the two magnitude tests
  0.2.3 exists to abolish.

### Divide and conquer

- **No second derivatives.** A Hessian needs the coupled-perturbed response of the whole system,
  which does not decompose the way the density does.
- **The energy is non-variational.** The assembled density is not the exact SCF density, so the
  energy and the gradient both carry a residual that shrinks with the buffer. Below a 7 Å buffer
  the accuracy falls off a cliff.
- **Below roughly 350 atoms it is slower than the exact SCF**, which is also exact.

### General

- **No thermochemistry.** Harmonic frequencies are computed, but no entropy, enthalpy, or
  free-energy quantities are derived from them.
- **A 1-D cell has a free rotation at Γ that a 3-D acoustic sum rule does not remove.** The
  admissible rotation generators are those satisfying `ω × T = 0` for every lattice vector, which
  the lattice decides and no magnitude test can: a chain has exactly one (about its own axis), and
  a slab or a crystal has none. `Projection::Rigid` removes that set; `projection="translations"`,
  which is the acoustic sum rule alone, leaves the chain's rotation standing.
- **A periodic spectrum keeps its `3N` length**, unlike a molecular one. The removed generators are
  re-inserted at *exactly* zero in ascending order rather than dropped, because a phonon branch
  index has to mean the same thing at every `q` and a `q`-dependent array length would break every
  band plot. So a wire returns `3N` numbers of which four are `0.000000`, where the same molecule
  without a cell would return `3N − 6`. The molecular case drops them because there is no branch
  index to keep.
- **Hydrogen-bond topology is perceived from geometry**, and the EH+ derivatives are evaluated at
  *fixed* topology. A topology change is therefore a piecewise-smooth boundary on the potential
  energy surface, as it is in MOPAC.
- **Singular two-center frame orientations.** When a bond lies exactly on the singular axis of a
  local two-center rotation frame, the integral *value* stays analytic while that one pair's
  derivative comes from a localized symmetric finite difference.
- **The stability analysis covers the real channels, not the complex one.** It is Seeger and
  Pople's test (*J. Chem. Phys.* **66** (1977) 3045): the singlet channel asks whether the solution
  is a minimum among closed-shell ones, the triplet channel whether it still is once α and β may
  differ, and for an already-unrestricted solution the internal UHF channel asks the same question
  in both spin blocks. All three are eigenvalues of `A + B`, the second derivative with respect to
  **real** orbital rotations. The third of Seeger and Pople's channels — the complex, RHF→CHF
  instability, governed by `A − B` — is **not implemented**, deliberately: the rest of the crate is
  real throughout, so a complex solution is not one any other part of pm7-rs could carry, and an
  instability that cannot be followed is worse than one not reported. The analysis is **off by
  default**; `stability="check"` reports, `"follow"` acts.
- **Residual for highly d-populated transition metals.** A heat-of-formation residual of up to
  about 0.36 kcal/mol remains for some four-coordinate, highly d-populated transition-metal
  compounds such as TiF₄. The Mulliken charges still match MOPAC, so this is a small
  d-integral-value difference at the *same* SCF solution — not an SCF-basin or scaling error.

PM7 is an empirical model. Agreement with a MOPAC implementation does not imply ab-initio accuracy
outside the model's parameterization domain.

## Where to read next

| | |
|---|---|
| The formalism | [theory.md](theory.md) |
| Where the model is not smooth | [singularities.md](singularities.md) |
| Building and releasing | [packaging.md](packaging.md) |
| Periodic systems | [pbc.md](pbc.md) |
| Linear scaling | [divide_and_conquer.md](divide_and_conquer.md) |
| Measured timings and scaling | [performance.md](performance.md) |
| The MOPAC audit | [fidelity.md](fidelity.md) |
| API | [Rust](rust-api.md) · [Python](python-api.md) |
