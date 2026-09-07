<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Changelog

All notable changes to `pm7-rs` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.4]

The version flows to the Python wheel through maturin, so `pyproject.toml` needs no change.

**0.2.4 rather than 0.2.3 because published numbers move.** Everything below was prepared as 0.2.3;
then the oracle grew from thirteen cases to a hundred and found two MOPAC corrections this crate had
never implemented. Fixing them changes the heat of formation of every molecule with an acetylenic
C–C bond or an Si–O–H group — acetylene by 12 kcal/mol — and a release that moves a published number
should say so in its version.

### Fixed: two molecular-mechanics corrections that were missing entirely

MOPAC adds corrections to the heat of formation **outside the SCF and outside the core–core
repulsion**, in `compfg.F90:370-378` under the comment "Add in any molecular-mechanics type
corrections here". `pm7-rs` implemented none of them, and nothing had ever looked: the thirteen-case
oracle contained no alkyne, no allene, no five-membered aromatic and no silanol.

- **`C_triple_bond_C`** — an empirical stabilization of acetylenic bonds, `12 kcal/mol` per C–C bond
  shorter than 1.21 Å, tapering to zero at 1.33 Å through a quintic-plus-cubic switch. MOPAC's own
  comment: "(The value 12 was determined empirically". It moves acetylene by **12.00000 kcal/mol**,
  allene by 1.87 and furan by 1.02.
- **`Si_O_H_correction`** — `15 (θ − 125°)²` kcal/mol per Si–O–H, damped by Gaussians in the Si–O
  and O–H distances. PM7-only. It moves silanol by 1.10 kcal/mol.

Both are now implemented in `src/mm_corrections.rs`, with their gradients and virials, so the energy
and the forces stay consistent and `optimize` does not pull against `energy`.

**How they were found is the point.** Both codes agreed on **every orbital energy to five decimals
and every Mulliken charge to six**, and — once MOPAC was asked for its energy partition with
`ENPART` — on the total energy (`−237.4756 eV` against `−237.475790`) and the nuclear–nuclear
repulsion (`149.8327` against `149.83265541`) as well. Identical electronic structure, identical
core repulsion, different heat of formation. That combination has exactly one explanation, and it
is the one MOPAC's source states in a comment. `examples/cc_core_probe.rs` is the measurement that
ruled the core–core repulsion out, term by term, before the source was read.

### Not implemented, deliberately: the MMOK amide correction

The same MOPAC block adds `htype·Σ sin²(dihedral)` over every N–H–C=O linkage, with
`htype = 3.1595` for PM7, and **MOPAC applies it by default** — the check that would have demanded
an explicit `MMOK` or `NOMM` keyword is commented out in `moldat.F90:876-879`. It is off here by
choice: it is a molecular-mechanics term on a torsion, MOPAC itself offers `NOMM` to remove it, and
a peptide energy resting on a torsional fudge is not what a caller of this crate is asking for.

The divergence is recorded rather than hidden: an amide will differ from a default MOPAC run by
that sum, and the two agree exactly when MOPAC is given `NOMM`. `nsp2_correction`, in the same
block, is PM6-only and unreachable from PM7.

Where 0.2.2 finished the periodic response, this release closes gaps between what the code computes
and what a user can actually get out of it, and removes two places where a number was decided by a
magnitude comparison instead of by physics.

### Added

#### SCF stability analysis, and the RHF dissociation failure it removes

An SCF iteration converges on `[F, P] = 0`, which is a **stationary** condition, not a minimum one.
A saddle point converges as cleanly as a minimum, to as tight a residual, and reports the same
`converged: true`. `src/stability.rs` asks the question the iteration cannot: is the lowest
eigenvalue of the orbital Hessian positive?

The framework is Seeger and Pople's (*J. Chem. Phys.* **66** (1977) 3045), and two of their three
channels are implemented: the **singlet** (RHF→RHF, is this a minimum among closed-shell solutions)
and the **triplet** (RHF→UHF, is it still one when α and β may differ), plus the internal channel
for an already-unrestricted solution. All are eigenvalues of `A + B`, the second derivative with
respect to *real* orbital rotations — which is exactly the operator the CPHF already applies, so
the analysis needs no new integrals and no new kernel, only the lowest eigenvalue of something the
crate can already multiply by. The **complex** channel (`A − B`, RHF→CHF) is deliberately absent:
the rest of the crate is real throughout, so a complex solution is not one any other part of
pm7-rs could carry.

`ScfStability::Follow` acts on what it finds — rotate the occupied orbitals along the unstable
eigenvector, re-converge, and keep whichever solution is lower, so following can never make an
answer worse. Stretched H₂ is the case that says it works:

| H₂ | singlet | triplet | ΔHf |
|---|---|---|---|
| 0.74 Å | +13.64 | **+8.23** | −31.748, unchanged |
| 1.50 Å | +10.36 | **−2.04** | 85.93 → 78.17 |
| 2.50 Å | +9.63 | **−7.99** | 187.08 → 103.48 |
| 4.00 Å | +10.70 | **−10.60** | 225.79 → **104.19** |

Two hydrogen atoms are `2 × 52.102 = 104.204` kcal/mol from the parameter table, so the escaped
solution dissociates correctly while the restricted one sat 121.6 kcal/mol above it. That is the
classic RHF dissociation failure, found and removed.

**It could not have been removed without fixing the UHF guess.** `uhf_loop` scaled the atomic
density by `n_α/n` and `n_β/n`, which for a closed shell is the *same number twice*: the spin
density started at exactly zero and the UHF equations kept it there forever, so `reference="uhf"`
on stretched H₂ returned the restricted energy and no amount of level shifting or DIIS tuning
changed that. `Pm7Options::initial_spin_density` is the way in, and the triplet eigenvector is what
goes there — the analysis does not merely report the instability, it supplies the direction out.

**Off by default**, on all surfaces. `check` reports through `Pm7Result::stability`; `follow` acts.
Reachable as `--stability off|check|follow` on both command lines, `stability=` on every Python
entry point and on `PM7(...)`, `Pm7Options::stability` in Rust, and as its own call —
`pm7_rs.scf_stability(...)`, `PM7.get_scf_stability()`, `pm7_rs::stability::check`. The built-in
optimizer takes `stability_every=n` / `--stability-every N` to re-check every `n`-th step, because
a geometry step changes the orbitals and a solution that was a minimum at the start can stop being
one on the way; once an instability is found the run stays on `follow`.

**A `follow` run that switches the spin reference says so on stderr.** The answer that comes back is
unrestricted where the caller asked for restricted — lower and more correct, but a different model,
and a geometry scan that switches partway through is discontinuous exactly there. The warning names
the triplet eigenvalue that forced it and how far the solution dropped, and `PM7_QUIET` silences it
on the same terms as every other warning in the crate.

**What it was built for, it refuted.** CuCl and AgCl are the two oracle cases where `pm7-rs` lands
*above* MOPAC (by 5.18 and 3.82 kcal/mol) with Mulliken charges differing by 0.019 e — the profile
of a different SCF solution. CuCl's lowest orbital-Hessian eigenvalue is `+6.05 eV`: a genuine local
minimum, nothing to follow. Thirty-six combinations of level shift, accelerator and spin reference
reach the same number, and so do forty-eight perturbed starting densities. `docs/fidelity.md`
reclassifies both cases accordingly.

#### `⟨S²⟩` for every unrestricted solution

A UHF determinant is not a spin eigenfunction, and nothing in the output said how far from one it
was. `Pm7Result::spin_squared()` reports MOPAC's `(S**2)`,
`⟨S²⟩ = S_z(S_z+1) + n_β − Tr(P^α P^β)` — the trace form rather than `Tr(P^α S P^β S)` because the
NDDO basis is orthonormal by construction, which is a property of the model and not an
approximation made here.

It agrees with MOPAC where MOPAC prints it: CH₃ `0.753039` and O₂ `2.002664`, both reproduced to
the printed precision. The sharper check is a limit MOPAC has no say in — a broken-symmetry singlet
of two separated hydrogen atoms is `⟨S²⟩ = 1` exactly, and following the triplet instability at
4 Å gives `1.0000`.

`None` for a restricted run, where the value is `S(S+1)` by construction and filling it in would
make a tautology look like a measurement, and for a k-mesh run, where the density available is the
`T = 0` block rather than the whole solution. On both command lines (`(SZ)` and `<S^2>` beside the
exact value, `spin_squared` in JSON), in the `single_point`, `orbitals` and `scf_stability` dicts,
and in `PM7.results["spin_squared"]`.

**The oracle compares it now**, on all twenty of its open-shell cases — ten doublets, seven
triplets, two quartets and a quintet. All twenty agree to `1e-6` or better. The check also catches
something no other column can: a **disagreement about the spin path**. Both codes decide RHF-or-UHF
from the multiplicity and the shell, so a case where one prints `(S**2)` and the other does not is
not a small numerical gap — it is the two codes running different methods on the same input, and it
is reported with no tolerance to hide behind.

#### Translations and rotations leave the spectrum, geometrically

`pm7_rs_cli frequencies examples/water.xyz` printed **nine** numbers, three of which (`71.35`,
`121.33`, `181.31 cm⁻¹`) were unprojected rotations. There was no Eckart projector anywhere in the
crate, and a unit test conceded the point by asserting only that the six lowest were below
300 cm⁻¹.

- `src/projection.rs` builds the rigid-body subspace and its complement: three translations
  `t_α ∝ √m_A ê_α` always, plus the rotations whose generator is admissible. **Linearity is decided
  from the inertia tensor**, not from a frequency. That distinction is the point of the whole item:
  a geometric tolerance asks "are these atoms collinear", which has a right answer, where a
  frequency cutoff asks "is this number small", which does not.
- A **1-D** cell has a genuine free rotation about its own axis, and a 3-D acoustic sum rule does
  not remove it. The admissible generators are those satisfying `ω × T = 0` for every lattice
  vector, which the lattice decides and no magnitude test can: a chain has exactly one, and a slab
  or a crystal has none.
- A molecule's arrays are **3N − 6** (3N − 5 linear). A **periodic** system keeps its `3N` length —
  the removed generators are re-inserted at exactly zero rather than dropped, because a phonon
  branch index has to mean the same thing at every `q` and a `q`-dependent length would break every
  band plot. `--projection none` on both command lines and `projection="none"` in Python return the
  raw set either way. All seven `IrSpectrum` arrays shrink together and stay positionally aligned;
  `dipole_derivatives` is indexed by Cartesian coordinate rather than by mode and keeps its
  `3 × 3N`.
- Every magnitude filter in the tests went with it — `filter(|f| *f > 500.0)` in `tests/ir.rs` and
  its three Python twins were the same heuristic in the tests, and a projection that let a rigid
  motion through would have passed all four.
- **The acoustic sum rule is enforced by default** for `force_constants` and `phonons`, on all four
  surfaces, with the residual still reported and `--no-acoustic-sum-rule` /
  `acoustic_sum_rule=False` to keep the raw set. It is a genuine two-sided projector
  `Φ(0) += PΓP − Γ`; on diamond it turns `0.0001` into `-0.0000 cm⁻¹`. This is the periodic half of
  "output noise-removed frequencies", and it defaulted to `false` in all three places.

#### Per-axis periodicity, on every surface

`--pbc TFT` could not be expressed at all. `Cell::from_flags` refused a non-leading pattern, the
Rust CLI had no `--pbc` flag, and the ASE calculator raised for an `Atoms` built as a slab along
*y* — an ordinary thing to have — rather than running it.

- `AxisRotation` (`src/cell.rs`) reorders the three lattice-vector slots cyclically so the periodic
  ones lead. **All eight patterns are reachable by one of the three cyclic rotations**, which is
  what makes this cheap rather than merely cheaper, and rotations preserve handedness where a
  transposition would not — handedness is load-bearing for `Cell::reciprocal`, which divides by a
  signed determinant.
- A mask on `Cell` was the alternative and was rejected: `Periodicity` is in the type precisely so
  no downstream module has to remember to honour a flag, and a mask puts the flag back into ~25
  dimension branches in `pbc/ewald.rs` alone. A reordering is invisible to all of them, because
  they reach the lattice through `cell.dim()` and `cell.vectors()`.
- `--kpoints`, `--kshift`, `--supercell` and fractional `--qpoints` are rotated with the lattice
  and un-rotated on output, so everything stays in the caller's axis order.
- Reached through `--pbc` on both command lines, `pbc=` in Python, the extended-XYZ `pbc=` key, and
  the ASE calculator. Measured: all three slab patterns and all three chain patterns agree to
  ≤ 1e-12 eV.

#### Variable-cell optimization

`optimize` was L-BFGS over `3·nat` Cartesian coordinates and never mentioned `src/stress.rs`. A
periodic run relaxed the atoms and reported success with whatever stress the fixed cell implied.
Measured on diamond at `a = 3.75 Å`, 2×2×2 mesh: `converged: true after 1 iterations`, max gradient
`2e-14 eV/Bohr`, and **−29.6 GPa** of pressure. Both statements are true — the atoms are at their
minimum for that cell, and the cell is 2 % too big.

`--opt-cell` (`relax_cell=True`) makes the lattice a degree of freedom. From `a = 3.75 Å` it
converges in 4 iterations and from `a = 3.40 Å` in 6, to `a/2 = 1.837553` and `1.837543 Å`
respectively — a 10 % spread in the starting cell collapsing to a 6e-6 relative disagreement, with
the energies agreeing to `4e-10 eV`. The relaxed structure is 0.076 eV/cell below the fixed-cell
one.

- **Strain, not lattice vectors.** `σ = (1/Ω)∂E/∂ε` is the definition of the stress tensor, so the
  analytic stress *is* the gradient of the strain variables — no chain rule, and no second
  implementation of a quantity the crate already has. The atoms move affinely with the cell, which
  is what makes a cell step and an atomic step independent directions rather than two descriptions
  of the same motion.
- **Only the periodic subspace strains.** The generators are built in an orthonormal basis of the
  span of the lattice vectors, giving `dim(dim+1)/2` of them. A slab has no lattice vector along
  its normal, so straining that direction is not a degree of freedom, it is a request to stretch
  vacuum. A chain relaxes along its own axis and acquires nothing perpendicular to it.
- **Two convergence tests, both of which must pass** — `gtol` on the force and `stress_tol` on the
  largest free stress component. A single mixed norm over the combined vector would let a converged
  force hide an unconverged stress, which is the failure this item exists to remove.
- `analytic_stress`'s refusals (an external field, `PbcMode::MopacCluster`) surface as an optimizer
  refusal instead of silently becoming an atoms-only run, and `relax_cell` on a molecule is refused
  by name.
- Divide and conquer drives it too: `dandc_derivatives` returns a stress for a periodic cell.
- `OptResult::trajectory` records the cell and the stress per step, and `native.optimize` returns
  the trajectory at all — it was populated and dropped.

Not available from any interface before this: `gtol`, `max_iter` and the rest were
`OptOptions::default()` at every call site, so a run that needed tighter convergence had no way to
ask. `--gtol`, `--stress-tol` and `--opt-max-iter` now exist on both command lines. `grad_step` was
declared and never read; it is gone.

ASE users were already covered by `FrechetCellFilter` and still are. This is for the Rust library,
both command lines, and Python callers not going through ASE — which is the population the request
named.

#### The flags a `pip` user could not reach

Mode parity between the two command lines was tested from 0.2.1. **Flag** parity was not, and had
drifted in seven places.

- `--scf-tolerance`, `--max-scf`, `--no-diis` and `--exchange-cutoff` were Rust-CLI-only. The wheel
  ships no Rust binary, so they were unreachable to anyone who installed with `pip` — and they are
  exactly the four knobs someone reaches for when an SCF will not converge. They now exist on both
  command lines and on every PyO3 entry point.
- `--pbc` and `--dipole-origin` were Python-only, so the same charged molecule gave two different
  dipoles depending on which command line asked.
- `--output`/`-o` was listed in the Rust CLI's own per-mode flag table **and** in its usage text,
  with no parse arm behind either: `optimize x.xyz --output out.xyz` failed with "unknown option
  `--output`" against a help text that had just named it. All three spellings now reach one
  destination.
- `--cell` takes 3, 6 or 9 numbers on both, where the Python one demanded nine.
- `--kshift` is no longer discarded when `--kpoints` is `1 1 1`: a 1×1×1 mesh at a half shift is one
  k point that is not Γ, which is a perfectly ordinary request the Rust CLI silently answered
  differently.
- `test_the_two_command_lines_offer_the_same_flags` compares the accepted spellings of both
  parsers, so this class of divergence cannot come back.

#### Orbital and Molden reporting

- `orbitals --json` emitted five keys where `native.orbitals` returned seventeen. A caller
  scripting the binary could not reach the frontier energies, the gap, the AO labelling or either
  beta channel.
- Text mode printed beta energies with a blank occupation column and no HOMO/LUMO markers, hiding
  the SOMO — the one orbital an open-shell run is about.
- `native.orbitals` accepts `smearing` and a k mesh and returned neither `fermi_ev` nor
  `entropy_ev`, while its own docstring called `fermi_ev` "the zone-wide answer". Both are returned
  now, on both surfaces, and the CLI says in words when the occupation column is a per-k band count
  rather than the mesh's filling. fcc Al on a 4×4×4 mesh with 0.3 eV Fermi smearing marks orbitals
  3–5 as `2.000` occupied at −5.307 eV, above its own Fermi level of −7.229 eV; that is the case
  the note is for.
- `pm7_rs::write_molden` had no caller anywhere — the CLI did its own `fs::write`. Both callers go
  through it now.

#### Licensing: what a binary actually redistributes

- `pyproject.toml` did not list `third_party/mopac/LICENSE`, so a `pip install pm7-rs-python`
  delivered `_native.pyd` — which embeds MOPAC-derived parameter tables through `include_str!` —
  with no copy of the MOPAC license anywhere in it.
- One layer up, and easier to miss because no source file records it: `cargo build` links
  statically, so both shipped binaries are copies of a substantial portion of ~120 crates. MIT asks
  for its notice in "all copies or substantial portions"; Apache-2.0 asks by §4(a) and §4(b).
  Neither binary carried any of it. `third_party/rust/NOTICES.md` is generated from the resolved
  dependency graph by `tools/licenses/collect_rust_notices.py`.
- The generator reads each crate's **own license files** rather than trusting the SPDX field, which
  is what turned up `faer` — this project's dense linear algebra, so in every binary — declaring
  `MIT` while additionally shipping `COPYING.EIGEN.MPL2` and three BSD notices for algorithms
  ported from Eigen, LAPACK and SuiteSparse. A manifest-driven notice would have dropped an
  MPL-2.0 obligation silently.
- `serde`, `serde_json` and `clap` are in `Cargo.lock` and deliberately absent from the notice:
  they arrive only through `criterion`, a dev-dependency. Of the lock's 160 packages, 122 reach a
  released artifact.
- `tests/attribution.rs` walks `Cargo.lock` independently of the generator, so a new dependency
  without a notice is a test failure rather than something to catch at release time. Both guards
  were checked by breaking them.

#### The periodic pair list stops being quadratic — above 32 atoms

`PairList::periodic` looped over every `A ≤ B` pair and, inside that, over every lattice image:
`O(N²·I)` distance evaluations for a list whose content is `O(N·I)`, reached from about seven
places per energy-and-gradient evaluation. One line of algebra fixes it — `|p_B + T − p_A| < r` is
`|p_B − (p_A − T)| < r`, so one spatial grid over the home-cell positions serves every translation.
The grid is the one `dandc/partition.rs` already had, moved to `src/spatial.rs` to serve both.

Diamond supercells at a 7 Å cutoff (`cargo run --release --example pairlist_scaling`): 2.67× at 54
atoms, 3.54× at 128, 5.64× at 250, 8.84× at 432.

**And 8.16× slower at two atoms**, which is why both paths are kept. The scan costs `N²/2` distance
evaluations per image and the grid costs `N` bucket queries; a bucket query is worth ten or twenty
distance evaluations, so the grid only pays above roughly 32 atoms — and a two-atom primitive cell
is what most of this crate's own tests are. Shipping the grid unconditionally would have been a
regression sold as an optimization, so `periodic` dispatches at a measured threshold.

That crossover number needed its own measurement, and the first version of this entry did not have
one. It quoted the example's two-atom row, which goes through `PairList::build` — the *dispatcher* —
and below 32 atoms that column is the scan timed against a count-only reimplementation of itself,
carrying no information about the grid at all. A new `#[ignore]`d unit test calls the two
constructors directly, because only a unit test can see them: 8.16× at 2 atoms, 1.18× at 16, then
0.48× at 54, 0.33× at 128 and 0.20× at 250. The crossover is between 16 and 54, which is where 32
sits. The conclusion held; the evidence for it did not exist.

Both paths produce **bit-identical** lists, entry for entry and in the same order, which
`tests/pair_list.rs` checks against a copy of the old enumeration on four cells, three cutoffs and
both sides of the threshold. The order was the part that had to be preserved deliberately: every
consumer sums over `pairs` in order, so a reordering is a different floating-point summation and a
different last bit in every published number.

#### The CPHF iteration budget is a knob, on every surface

`CPHF_MAX_ITERATIONS` was a private `const` of 100 that no interface could raise, and the 0.2.2
refusal said so and told the caller to use a different entry point instead. That is a strange thing
for a program to say: how many operator applications an orbital response needs is set by the
conditioning of the orbital Hessian, which is set by the frontier gap, which is a property of the
*system* — so a fixed budget is a limit on which systems the mode works for, not a safety rail.

- `Pm7Options::cphf_max_iterations` (default 100) is read by all three CPHF paths: the restricted
  molecular solver, the coupled α/β one, and the periodic `relaxation_hessian`. Reachable as
  `--cphf-max-iterations` on both command lines, `cphf_max_iterations=` on every Python entry point
  and on `PM7(...)`.
- Raising it costs iterations and moves nothing else — the tolerance is untouched, so a solve that
  converged at 100 returns the same matrix at 400. A budget of zero is refused rather than treated
  as a cheap answer.
- The refusal now names the flag it wants raised, instead of naming a different mode.

**It buys the case it was asked for.** Cubic SrTiO₃ — 0.28 eV PM7 gap, the cell 0.2.2's refusal was
written about — now goes through `phonons --supercell 2 2 2` at a budget of **1600**: 144 s,
fifteen modes, the lowest at −289.0 cm⁻¹, which is the same instability the ten-crystal audit finds
independently (nine imaginary optical modes between −290 and −134). The cubic cell at the
experimental 3.905 Å is far from PM7's minimum, and this route can now say so instead of refusing.

The residual on the way there is worth recording, because it is not what one would guess: `6e-9` at
100, `9e-7` at 200, `1e-7` at 400, `2e-9` at 800, converged at 1600 (and at 3200, to the same
spectrum). **Not monotone in the budget** — so a run that stopped at `9e-7` is not evidence the
answer is out of reach. The conjugate gradient hands over to the DIIS fixed point when the operator
stops looking positive definite, and a fixed point's last iterate is not its best one. The refusal
says to expect this rather than leaving the reader to conclude from one rung that more iterations
will not help.

`dfpt` is still the cheaper route on this cell by a wide margin — 13 iterations to `9.2e-11` in
**4 s** against 144 — but it is not the same number and should not be read as a cross-check:
`--supercell 2 2 2` samples the zone on a 2×2×2 mesh, a bare `dfpt` call samples it at Γ, and on
this cell the Γ SCF also has to fall back to a Fermi width to converge at all. Different sampling,
different spectrum (−412.1 to 448.8 against −289.0 to 512.5). The comparison that *is* meaningful
is against the audit's own 3×3×3 run, above.

#### A partially occupied zone centre is refused, not answered

The periodic response makes **no integer-occupation assumption** — it weights band pairs by
`Δf/Δε`, which is the metallic form, and a gapless mesh with no smearing has been refused since
0.2.2 because that quantity is then a `0/0`. So metallic DFPT works.

One thing was missing and silent. A `q = 0` perturbation of a partially occupied cell **moves the
Fermi level**, and the intraband term that goes with holding the electron count fixed (de Gironcoli,
*Phys. Rev. B* **51**, 6773 (1995)) is not implemented. `docs/pbc.md` and `docs/scope.md` both said
so, which is not the same as reporting it — and the zone centre is reached from more places than
`dfpt`: `born_charges`, `static_dielectric` and `polarizability` are field responses at `q = 0` by
construction, and `frequencies`/`hessian` on a k mesh delegate there too. An incomplete response is
indistinguishable in the result from a complete one. It is now an **error**. (The supercell
`phonons` route is a different solver and is not affected.)

- The gate is on the **occupations**, not the entropy. A Methfessel–Paxton entropy can pass through
  zero with the occupations still fractional, so the entropy is reported in the message rather than
  tested. `q ≠ 0` is never gated: the term vanishes there by symmetry.
- **A smearing that leaves the occupations integral is admitted**, which is the ordinary use — a
  width applied to a gapped cell to escape a symmetry-broken solution changes the path, not the
  answer. The threshold (`1e-3`) comes from the gap between the two populations, measured on PM7
  ZnS because its gap depends strongly on the mesh: a gapless 3×3×3 or 4×4×4 mesh puts **a third of
  a state** on either side of `E_F`, while every gapped case sits at `5.7e-6` or below. Nearly five
  orders of magnitude of empty space, with the threshold in the middle.
- **It caught something, and the thing it caught was a published number.**
  `examples/crystal_phonons.py` ran ZnS and CaF₂ at 3×3×3 with a 0.10 eV width, where PM7 makes them
  spuriously metallic — and got the `[3]` and `[3, 3]` their space groups require. Right
  degeneracies, incomplete calculation: the most convincing way for a number to be wrong. ZnS is
  fine once the mesh is 4×4×4, where PM7 gives it a 2.575 eV gap, and comes back `[3]` at
  232.6 cm⁻¹. CaF₂ is not: **PM7 has no usable gap for fluorite at any mesh** — 0.012 eV at 3×3×3,
  0.092 at 4×4×4, 0.054 at 5×5×5, 0.032 at 6×6×6, against a ~12 eV experimental gap — and every
  smeared fill at 0.02, 0.05 or 0.10 eV is genuinely partial. Its zone-centre response is therefore
  not well defined, the audit no longer requires a degeneracy of it, and it prints the reason. That
  is a finding about PM7's fluorite, recorded rather than worked around.

#### Phonon eigenvectors, from both routes and from ASE

`phonons` and `dfpt` returned frequencies and nothing else, though both **computed the eigenvectors
on every call and threw them away** — `let (eigenvalues, _) = d.hermitian_eigen()?` in each. That
left a whole class of question unanswerable without leaving the library: which atoms a soft branch
moves, how to displace a structure along a mode to look for a lower-symmetry minimum, whether two
branches at one frequency are the degenerate pair a space group requires.

- `ForceConstants::modes(q)` and `DfptResult::modes()` return a `PhononModes` — frequencies,
  eigenvectors of the mass-weighted dynamical matrix (one mode per column, unitary), and
  `cartesian_modes`, the same set as displacements `m^−1/2 e` renormalized per column. That last
  convention is `VibrationalModes::cartesian_modes`'s exactly, which is MOPAC's `cnorml`, so a
  molecular mode and a zone-centre phonon mean the same thing by the same rule.
- Both routes go through one function, so `frequencies_cm` is now that function's output with a
  field taken — the frequencies and the vectors cannot come from two diagonalizations that drift.
- In Python: `modes` and `cartesian_modes` on both dicts, one `(real, imag)` pair per q, the same
  complex convention `dfpt` already used for its force constants. They are genuinely complex away
  from the zone centre — a phonon at `q` is `u_A ∝ e_A e^{iq·R_A}`, and the phase is what separates
  branches sharing a `|q|`.
- In ASE: they arrive in `get_phonons`/`get_dfpt` for free, and `get_phonon_modes(...)` returns
  `(frequencies, modes, cartesian_modes)` as complex arrays for callers who would rather not
  assemble the pairs themselves.

#### One ground state per k-mesh Hessian, not two

`analytic_hessian_with` ran `run_pm7` before deciding which arm to take, and the k-mesh arm then
called `dynamical_matrix_dfpt`, which ran the ground state **again** — the whole periodic SCF twice
for one Hessian, and on a k mesh the SCF is the expensive half. `docs/scope.md` had recorded it as
"a straightforward improvement that has not been made" since 0.2.2.

The reason it was structured that way was that the branch tested `scf.unrestricted`, so it looked
like the SCF had to come first. It does not: **a real k mesh has only one arm**, so the mesh alone
decides, and only the Γ case needs the question asked. `DfptResult` now carries the ground state it
converged and the k-mesh arm takes it, bit for bit — the solver leaves the caller's options alone
whenever the mesh is not Γ, so it is the same state the removed call produced.

The Γ-plus-open-shell path still runs two, and that one is not a duplicate: `KMesh::Gamma` does not
build the translation-resolved Hamiltonian the response needs, so the solver promotes the sampling
to `grid(1, 1, 1)` and converges a different object.

Removing a call changes which error a user meets first, and the full CLI matrix caught that in 96 of
its 8924 invocations: `hessian <sheet> --kpoints 2 1 1 --pbc-mode mopac` used to be stopped by the
ground state and is now stopped by the perturbation solver, whose refusal said only "the
perturbation solver needs the Ewald periodic mode" — true, and it names neither the flag to change
nor what to change it to. Rewritten to name `--pbc-mode`, say why a cluster truncation has no
derivative, and point at the default.

#### Two example scripts that check the physics rather than the exit code

- `examples/crystal_phonons.py` runs the zone centre of **ten structure types** — diamond, zinc
  blende, rocksalt, fluorite, perovskite, wurtzite, rutile, spinel, layered rocksalt and the
  Zn(CN)₂ coordination polymer — and asserts the property no unit test in this repository asserts:
  **the degeneracies the space group requires**. A cubic `Td` or `Oh` cell must put its optical
  modes into triplets, a wurtzite cell into singlets and doublets, and every crystal must put three
  modes at exactly zero. Those are statements about the Hamiltonian's symmetry, and they fail
  loudly when a two-centre rotation, a lattice sum or an image list is wrong in a way no energy
  comparison notices. Structures come from ASE rather than hand-typed coordinates, because a
  mistyped Wyckoff position produces a *plausible* spectrum with the wrong degeneracies — which is
  exactly the failure the script exists to detect and would otherwise be reporting on itself.
- `examples/zncn2_phonon_bands.py` plots the **full dispersion** of Zn(CN)₂ along Γ–X–M–Γ–R–X, 71
  wavevectors of DFPT with no commensurability condition, over a cell relaxed with `relax_cell` to
  `a = 5.9228 Å` against the experimental `5.9227`. Thirty branches spanning two decades, from a
  C≡N stretch near 2370 cm⁻¹ down to the framework's acoustic modes, so the y axis is broken.

  It also records a mistake worth keeping, because it is the one this whole surface invites. At a
  2×2×2 ground-state mesh the acoustic branches come back **imaginary between the high-symmetry
  points**, to −73 cm⁻¹, and the obvious reading — that PM7 puts the cubic structure at a saddle
  point — is wrong. "No commensurability condition" is a statement about the *supercell*; the
  response still couples `k` with `k + q`, so the ground-state mesh limits which wavevectors mean
  anything. Refining it: −72.64, −37.44, −22.55 cm⁻¹ at 2×2×2, 3×3×3, 4×4×4, halving each time and
  on its way to zero, while Γ stays at −0.00 and the zone points stay stable throughout. A real
  soft mode does not behave that way — it is usually softest *at* a zone point, and it does not
  care how finely the ground state is sampled. The default mesh is 3, and the plot names the
  residual dip as the sampling rather than dressing it up as physics.

### Removed (breaking)

- **`SOFT_MODE_FLOOR`.** `1e-6` in eV/(Å²·amu), below which `static_dielectric_tensor` dropped a
  mode from the `1/ω²` ionic sum as acoustic. It was exported from `src/lib.rs` and reported to
  Python as `soft_mode_floor`, so removing it is a breaking change; it is the second of the two
  "below this number it must be a rigid motion" rules this release exists to abolish.

  The argument for it was that it sat in a measured gap: on the cells it was developed against the
  acoustic branch landed between `1e-17` and `5e-15`, and the softest genuine optical mode at about
  `2.5e-2` — thirteen orders of magnitude, with the cut in the middle. But that is a statement
  about those cells, not about the quantity. A frequency carries real physics at every magnitude,
  and a ferroelectric near its transition has exactly the soft *optical* mode the floor would have
  swallowed — from the sum that mode dominates, since `1/ω²` weights it hardest.

  The replacement is the statement the physics makes: at `q = 0` the acoustic modes **are** the
  mass-weighted uniform translations, a three-dimensional subspace known before the matrix is
  diagonalized. `acoustic_modes` takes the three largest overlaps with it, which needs no threshold
  at all — the overlap is 1 for a translation and 0 for anything orthogonal to it, and there are
  exactly three. `skipped_modes` is therefore always three, and the signal the old count carried by
  accident — "more than three means the geometry is not a minimum" — moves to its own field,
  `soft_optical_modes`. A mode the floor used to hide is now reported rather than dropped.

### Fixed

- **`born` and `dfpt` on a molecule crashed the Python CLI instead of refusing.** Without a cell
  they reached `np.asarray(None, dtype=float)` — a 0-d array of `nan` whose `.tolist()` is a bare
  float — and the extension answered `argument 'cell': 'float' object cannot be converted to
  'Sequence'` from inside a twenty-frame traceback. The Rust CLI has refused both by name all
  along; the two were simply missing from the Python CLI's cell guard.
- **`charges --dandc` and `gradient --dandc` ended in a `KeyError`** after printing a perfectly
  good energy. `--dandc` sends all four modes it applies to through `divide_and_conquer`, whose
  result carries `forces_ev_per_angstrom` and neither `charges` nor `gradient_ev_per_angstrom`, and
  the printer indexed the keys rather than checking for them. The gradient is now *derived* from
  the forces it does return — the sign is all that separates them, and refusing would withhold a
  number the run already computed — and the absent charges are reported as absent.
- **Divide and conquer had no memory guard, and a large buffer on a small periodic cell aborted
  the process.** `run_pm7`'s pre-flight guard is on the whole molecule's basis, which is the wrong
  quantity here: the dense work is per *subsystem*, and for a periodic cell a subsystem is larger
  than the cell because the buffer pulls in lattice images. A two-atom diamond cell has 8 AOs and
  passes every check `run_pm7` makes; `--dandc 15.0` on its 2.5 Å lattice gives a largest subsystem
  of 2674 atoms and 10696 orbitals, wanting ~34 GB. It consumed 11.5 GB over several minutes and
  then aborted on a 4 MB allocation — no `Pm7Error`, no message, nothing naming the flag.

  It now refuses in 1.2 s with the buffer, the core size, the subsystem it could not fit, and what
  to do instead. Found by `tests/cli_matrix.rs`, whose invariant is that a refusal is a message and
  never a crash — the full matrix run is what surfaced it, six of 8492 invocations.
- **A UTF-8 byte-order mark broke every structure file that had one.** The BOM is invisible in
  every editor and turns the first line into something `parse::<usize>()` rejects, so the error read
  `invalid XYZ atom count: 3` and pointed at a line that plainly reads `3`. PowerShell's
  `Set-Content -Encoding UTF8` and Notepad's "UTF-8" both write one, which makes it the ordinary
  way to produce an XYZ file on Windows.
- **A refusal from the extension module reached the Python CLI user as a traceback.** Its own
  validation failures exited through `SystemExit` with a one-line message while a `ValueError` from
  the native layer escaped as twenty frames of interpreter — a different presentation for the same
  class of problem. A missing structure file was worse: the traceback was the only thing that named
  the file.
- **A flag the mode will not read is refused, not swallowed.** `energy structure.xyz --opt-output
  out.xyz` used to exit successfully having written no file, which is the worst of the class: the
  user waits for something that was never going to arrive and the exit status says it worked. The
  no-op ledger opened with twenty entries and is down to three, all three correct behaviour.
- **`--json` reaches `gradient`, `forces` and `optimize`**, which parsed it and printed text
  anyway; and `optimize` prints the relaxed geometry rather than only `converged: true` and an
  energy.
- **The open-shell CPHF still returned unconverged answers as `Ok`.** 0.2.2's release notes say
  "both solvers" were fixed to refuse rather than hand back their last iterate; both meant both
  *restricted* solvers. The coupled α/β solver behind every UHF Hessian does not share a loop with
  them and was not looked at — it ran a hard-coded hundred iterations and returned whatever it had.
  A Hessian assembled from an unconverged response is wrong in proportion to the residual and looks
  exactly like a converged one, which is the whole reason the restricted path refuses. It refuses
  now, through the same message and on the same (now configurable) budget.
- **Sparkle parameters were installed over MOPAC's non-SPARKLE atomic-number range.** All 15 LnF₃
  now match MOPAC to 0.0000 kcal/mol, against errors of −195.3 (LaF₃) and +161.2 (LuF₃).
- **Two real bugs in the periodic density mixer.** The Pulay Gram matrix was unscaled, so a
  relative pivot test was an absolute one and a well-conditioned history was rejected for being
  *small*; and one rejection dropped the mixer to plain damping for the rest of the run. Diamond
  now converges in 11 iterations, NaCl 12, CaF₂ 12, SrTiO₃ 13, Zn(CN)₂ 14, spinel 17.
- **An incommensurate `q` was Fourier-interpolated in silence.** `phonons --supercell 2 2 2
  --qpoints 0.25,0,0` reported `798.5 cm⁻¹` where DFPT gives `670.0`, a 19 % error printed with no
  warning. It now warns and names the supercell that would make the requested `q` exact.

### Findings that changed no code

- **The DFPT degeneracy breaking at `q = (0.25, 0, 0)` is the sampling mesh**, not the CG tolerance
  and not the Hermiticity threshold — the two suspects the release plan named. Diamond's transverse
  acoustic pair splits by 0.098 cm⁻¹ on a 2×2×2 mesh (`377.2476` against `377.3460`) and is exactly
  degenerate from 3×3×3 up, while tightening `dfpt_tolerance` across four decades and
  `scf_tolerance` across four more moves it by nothing at all. `tests/dfpt.rs` pins the resolution
  rather than the number.
- **Cubic SrTiO₃ at the experimental 3.905 Å is far from PM7's minimum**: −24.2 GPa standing, nine
  imaginary optical modes between −290 and −134 cm⁻¹, and three acoustic modes at exactly zero. The
  variable-cell optimizer takes it to `a = 3.7080 Å` in six steps.

- **Diamond's 1317.9 cm⁻¹ zone-centre optical mode was under-converged.** Converged PM7 is
  **1238 cm⁻¹**, 7 % below the 1332 experimental value; the near-match was a 2×2×2 coincidence.
- **Every structure in the audit whose irreps the zone centre can legitimately be asked for
  reproduces them**, spinel's 42 modes (three singlets, three doublets, ten triplets and three
  acoustic) included, with three acoustic modes at exactly zero everywhere. ZnS needs a Fermi width
  and a 4×4×4 mesh to get there; CaF₂ is the one structure PM7 cannot be asked at all, for the
  reason above.
- **The ZnS and CaF₂ triplet splitting is not a solver defect.** `R D Rᵀ − D` is 0 for diamond and
  ~1e-11 for GaAs and CdS, but 1.23 for ZnS. The *ground-state* Hamiltonian already lacks the
  symmetry — ZnS's Γ valence triplet is split by 0.28 eV — and of the two zero-entropy SCF
  solutions the symmetry-broken one is variationally **lower by 9.6 meV**. The SCF finds the
  correct minimum; PM7's variational minimum for this cell simply is not cubic. What restores the
  required triplet is a **smeared** fill: PM7 gives both cells a near-zero gap, and an unsmeared
  aufbau fill is discontinuous in the band energies, so it settles into the symmetry-broken
  solution.

  How far that argument carries is settled by the partial-occupation gate above, and it does not
  carry as far as it first appeared. For **ZnS** it holds: at a 4×4×4 mesh PM7 gives the cell a
  2.575 eV gap, a 0.10 eV width leaves the occupations integral to `5.7e-6`, and the `[3]` that
  comes back is a complete calculation. For **CaF₂** it does not: PM7 leaves fluorite gapless at
  every mesh from 3×3×3 to 6×6×6, every width gives a genuinely partial fill, and the `[3, 3]` this
  entry originally credited to smearing was a zone-centre response missing its Fermi-level term.
  A smeared fill can restore a symmetry the unsmeared one broke; it cannot supply a gap the method
  does not have.
- **The periodic SCF's slow cases are stiff, not unstable**, so **Kerker preconditioning was
  measured and then dropped rather than built.** The residual falls 0.993 to 0.9993 per step,
  monotonically, with no oscillation — and split by channel, the **charge** channel is the
  best-converged of the three on the cases that actually stall. Kerker preconditions exactly that
  channel. `docs/performance.md`'s "Kerker for the cause" was written from the textbook diagnosis of
  charge sloshing, which is real on NaCl (1.1 electrons oscillating, the chemical potential swinging
  12 eV) and which NaCl converges through anyway. Implementing it to close the plan item would have
  put a knob in the release aimed at the wrong half of the problem. The stiff cases are served by the
  entropy-gated smearing fallback that is already there: ZnS converges in 19 iterations at a 0.05 eV
  width and not at all without, across a 6.4 eV gap where a smearing cannot change the answer.
- **A periodic force is only as converged as its density**, with roughly 10³ amplification. The
  default `p_tol = 1e-7` leaves a 1.1e-4 eV/Bohr symmetry-breaking force on a hard cell — which is
  why `--scf-tolerance` had to reach the command line, and why an earlier claim of a ZnS gradient
  defect was withdrawn.
- **CaF₂ and ZnS have multiple SCF solutions**, so a finite-difference Hessian across geometries can
  compare different electronic states. That produced a 52 %-asymmetric FD Hessian and one wrong
  conclusion, since corrected.

## [0.2.2] - never tagged

Prepared but never released: `Cargo.toml` still said `0.2.1` when the 0.2.3 work began, so
everything below ships as part of 0.2.3. Kept as its own section because it is a coherent body of
work and squashing it into the one above would lose what belongs to what.

The periodic response, finished off. Metals and open shells reach the perturbation solver; the cell
gets a polarization and a static dielectric constant; and the field response acquires two
independent routes to be checked against, which is how the largest bug in this release was found.
The k-mesh gradient and stress stop being quadratic in the mesh.

### Added

#### Polarization, and two independent checks on the field response

- **Berry-phase polarization** — `berry_polarization` (`src/pbc/berry.rs`), in `e/Bohr²`, with the
  electronic and ionic halves, the raw phase, and the polarization quantum along each lattice
  vector. `BerryPolarization::difference` reduces two polarizations onto the branch nearest zero,
  which is the only physically meaningful thing to do with a pair of them.

  The reason to have it is that it shares the Hamiltonian and the basis with the CPHF Born charges
  and essentially nothing else, so `tests/pbc_berry.rs` can compare the two. Three errors it caught,
  all mine and all found by measurement rather than by reading:

  * **The lattice reduction was only correct for an orthogonal lattice.** Projecting onto each
    quantum in turn and rounding carries a component of the others when the basis is not orthogonal,
    and an fcc cell has its vectors at 60°. On rocksalt LiF — zero by inversion symmetry — that left
    `1.43e-1 e/Bohr²` standing against a quantum of `4.88e-2`, nearly three quanta of "polarization"
    that was entirely the reduction failing. Solving `Δ = Σ_i n_i q_i` for the coefficients and
    rounding those is exact for any lattice, and is three rows of Cramer's rule.
  * **Scaling the quantum by the occupancy was wrong.** The argument for it — a restricted
    calculation moves two electrons at once — is true of the branch of the logarithm and false of
    the quantity. Fluorine at fractional `(−½, ½, ½)` with core charge 7 puts `P_ionic` at exactly
    `3.5 (a, 0, 0)/Ω`, a half-integer, which only exists against the **single-electron** quantum.
  * **A wrong premise in my own test.** I asserted that a centrosymmetric crystal has `P = 0`.
    Inversion gives `P = −P` modulo the quantum, so `2P = 0`, which admits zero **and** half a
    quantum — and this crystal sits at the half. The test now asserts the symmetry statement rather
    than the intuition, and separately pins the electronic phase at zero, which is the part that
    catches a defect in the string, the overlap, or the closing link.

- **A finite field along a periodic direction** — `run_finite_field`, by the Nunes–Gonze electric
  enthalpy `F = E − Ω 𝓔·P`, with `P` the Berry phase above. A field orthogonal to every lattice
  vector is an ordinary calculation and already went through `Pm7Options::field`; along a periodic
  direction `𝓔·R` is unbounded and the ground state of `H − 𝓔·R` on a lattice **does not exist**, so
  there is nothing to converge and no care in the assembly repairs it.

  The field term couples neighbouring k points, so the k points can no longer be solved one at a
  time. `scf_pbc::run_kpoint_scf_with_terms` takes a per-k Hermitian operator that is not a Bloch
  sum of any `H(T)`; `None` is the ordinary path and is unchanged.

  **What the test is for is the coupling factor, which cannot be checked by reading the
  derivation.** `λ_α = (𝓔·a_α) J / 4π`, and the Hermitization `M + M†` rather than half of it, are
  both choices that fail silently: a wrong factor of two, a missing `J`, or a flipped sign all give
  a converged calculation and a plausible polarizability. Measured against CPHF,
  `α_xx = 0.146830` against `0.145913` — **ratio 1.0063**, on two formalisms that share only the
  SCF. Half that ratio would be the Hermitization. The polarization is exactly antisymmetric in the
  sign of the field (`±2.0146e-8`), which no factor error would produce by accident.

  Neither number is a prediction of experiment — this is a semiempirical model. What is checked is
  that the crate computes its own model's polarizability consistently by two routes.

#### Dielectric constants

- **`static_dielectric_tensor()` — `ε⁰`, the static tensor, not just the clamped-ion one.**

  ```text
  ε⁰_ab = ε∞_ab + (4π/Ω) Σ_m (Z*·e_m)_a (Z*·e_m)_b / ω_m²
  ```

  over the optical modes of the zone-centre dynamical matrix. This is the quantity a measured
  dielectric constant is usually compared against; `ε∞` alone is the high-frequency limit and for an
  ionic crystal the two differ by a lot. Nothing new is solved — both ingredients were already here,
  so it is a contraction rather than another response. Semiempirical codes commonly stop at `ε∞` for
  want of Born charges, not for want of this formula.

  It reports `skipped_modes`, because the `1/ω²` means a mode near zero dominates and a mode near
  zero is the geometry saying it is not at a minimum. Three skipped is the acoustic branch and
  expected; more than three means the ionic term is missing whatever those modes would have
  contributed, which for a soft mode is most of it. `SOFT_MODE_FLOOR` sits in a measured gap: on the
  cells this was developed against, the acoustic branch lands between `1e-17` and `5e-15` and the
  softest genuine optical mode at about `2.5e-2`, so the floor sits in the middle of thirteen orders
  of magnitude and neither classification is near the line.

- **`polarizability()` on its own.** The same tensor `born_and_dielectric` reports, without
  computing the Born charges to get it. It exists as its own entry point because `α` is defined in
  **every** dimensionality where `ε∞` is not: a chain and a slab have a polarizability, and neither
  has a dielectric constant until someone says where the material stops.

- **`dielectric_origin_sensitivity()`** — measures the approximation the position operator rests on
  instead of arguing it. The argument that the response is well defined even though `r` is not a
  periodic operator — an origin shift adds a constant to the diagonal and the occupied–virtual
  projection annihilates it — is an argument. This displaces every atom by `offset`, recomputes `α`,
  and returns the largest change in any component. Near machine precision says the argument holds
  for this system; a large value says the polarizability being reported is a statement about where
  the origin was put, which is worth knowing before quoting it.

- **`ε^∞` for a chain or a slab** — `dielectric_with_extent`, on every surface. Below three
  dimensions the cell has no volume, so `born_and_dielectric` left `dielectric` as the identity and
  only the raw `polarizability` was available. The missing ingredient is a thickness or a
  cross-section, and it is a **required** argument: a supercell says where the atoms are, not where
  the material stops, so doubling the vacuum must not change `ε`.

  The conversion carries the depolarization factor of the assumed body rather than being a
  division — `α` is the response to the *external* field, so for a slab polarized along its normal
  the depolarizing field is already inside it. The three-dimensional case is the `N = 0` row of the
  same table, and `tests/dielectric_extent.rs` checks that the low-dimensional formula closes on
  `born_and_dielectric`'s own `ε` before it checks anything else. Reported alongside are the two
  extent-free combinations `(ε_∥ − 1)d` and `(1 − 1/ε_⊥)d`, which are what a slab can quote without
  choosing a convention, and `axis_mixing`, which says how much the per-principal-axis treatment
  assumed.

  The sign is the one place this could have gone quietly wrong, and did in a first draft: `α` here
  is `∂μ/∂f` in MOPAC's field convention, so the physical susceptibility is its **negation** (C-1,
  C-6). The negation now happens once, inside `epsilon_from_polarizability`, so a caller holding a
  `polarizability` cannot pair it with the wrong sign — and the closure test is what holds that
  fixed. Without it the draft returned `2 − ε`, which is a perfectly plausible-looking number.

- **Born charges, `α` and `ε^∞` take an unrestricted cell.** v0.2.1 refused, and the refusal was
  precise about why: the field reaches the response through the commutator `[H, r]`, an
  unrestricted cell has two different `H`, and so each spin needs its own commutator *and* its own
  band-basis contraction — a different contraction from the spin-independent one the phonons use.
  All three are per spin now, and `solve_one` returns the spin-resolved response rather than the
  sum so the contraction can be done channel by channel.

  Forcing UHF on a **closed shell** reproduces the restricted Born charges and polarizability to
  `10⁻⁸` — the check that would catch a factor of two, a channel counted twice, one commutator
  serving both spins, or a contraction in the wrong band basis. A genuine doublet satisfies
  `Σ_A Z*_A = 0` to `10⁻⁸` **and** differs from the closed-shell answer, which is the half that
  matters: an implementation that quietly averaged the channels would pass every sum rule while
  returning the restricted result.

#### Perturbation theory

- **Metallic DFPT.** This was not the large port it looked like. The general band-pair form
  `[f_n(k) − f_m(k+q)] / [ε_n(k) − ε_m(k+q)]` was already here, and the `k` and `k + q` meshes were
  already filled together against one Fermi level with smearing. What stood in the way was the
  **gate**: any fractional occupation was refused outright, which turns away every smeared metal —
  including the ones where smearing is exactly what makes the response well defined.

  The gate is now on **gaplessness without smearing**, which is the condition that actually makes
  the energy denominators singular. Where a band crosses `E_F`, `Δf/Δε` is a `0/0`; a smeared
  occupation makes it finite because `Δf` then goes to zero with `Δε` at a rate the smearing
  function fixes, while an unsmeared step leaves the answer decided by which pairs happened to fall
  inside the `1e-8` denominator floor. bcc Li: refused without smearing, with the message naming the
  highest occupied level, the lowest empty one and `E_F`; with Fermi–Dirac at 0.3 eV it converges to
  residual `1.6e-11` with `D(q)` Hermitian to `1.4e-17`.

  **Still absent is the Fermi-level shift and its intraband term**, which matters at `q = 0` and
  vanishes by symmetry at `q ≠ 0` — the perturbation has no uniform component to shift the chemical
  potential with — so the wavevectors a phonon dispersion is made of are complete.
  `docs/scope.md` says which is which.

- **The unrestricted periodic Hessian**, by delegation rather than by a second UCPHF.
  `analytic_hessian_periodic` refused an open shell for want of one; `dynamical_matrix_dfpt` has
  carried a band set per spin since the UHF field response landed, so at `q = 0` on a Γ mesh it
  produces exactly the matrix that refusal was standing in front of. An open-shell CH₂ chain now
  returns a `9×9` that is symmetric to `0.00e+00`. A second UCPHF written to reach the same number
  would be two implementations to keep in step. The SCF is hoisted above the branch, because both
  arms ran their own and choosing the arm needs its answer.

- **`LongRange {Auto, Require, Off}`** on `DfptOptions`, wired to all three sites the long-range
  monopole term touches — the fixed-charge second derivative, the bare perturbation's per-atom
  channel, and the `∂q_A(q)` shift in the coupled-perturbed kernel. It is all three or none:
  leaving it out of any one alone would let the skeleton carry a term the response could not screen.

  The point of `Off` is that it makes the term's effect **measurable** rather than arguable. LiF at
  `q = (¼, 0, 0)`: the lowest mode moves from **−50.3 to 74.7 cm⁻¹** with the term off. `Require`
  makes a cell whose periodic mode has no lattice sum an error rather than a quiet difference in
  what was computed.

- **`DfptOptions::keep_response` and `DfptResult::response`** — the first-order densities, moved
  rather than cloned. Off by default: it is `3N × n_k × 2·nao²` floats, the largest array the
  calculation touches, and the force constants never need it kept.

- **`DfptResult::hermiticity`** — the worst departure of `D(q)` from Hermiticity *before* the
  assembly symmetrized it, relative to the matrix's own largest element. Reported rather than kept
  private because it is the one number that says how much the symmetrization had to clean up: a run
  sitting just under the refusal threshold is one whose force constants are held together by that
  threshold.

- **LO–TO frequencies are reachable.** `lo_to_direction` on `dfpt` and `phonons` — Rust, PyO3,
  `pm7_rs.native`, `PM7.get_dfpt`/`get_phonons`, and `--lo-to` on both command lines — returns
  `frequencies_cm_lo_to` beside the unsplit frequencies, `None` for a q away from the zone centre
  where the term does not belong.

  `DfptResult::frequencies_cm_lo_to`, `force_constants_with_lo_to` and the matching pair on
  `ForceConstants` had **no caller anywhere in the repository** — not in the bindings, not in either
  CLI, not in a test — while `docs/properties.md` described them as the way to use the non-analytic
  term. What was bound was the raw `D^NA` matrix, leaving the caller to add it to the force
  constants, mass-weight and re-diagonalize by hand.

- `pbc::ewald::ewald_reciprocal_hessian_bvk` — the Hessian half of the k-mesh fold below. Writing it
  turned up a **third** `G` loop in the 1-D arm that the gradient version had missed.
- `scf_pbc::density_from_fock_with` — an optional per-k operator, which the finite field needs and
  nothing else does. `None` is the ordinary path.

#### Interfaces

- The five new entry points reach **Python and ASE**: `polarizability`, `static_dielectric`,
  `berry_polarization`, `finite_field` and `dielectric_origin_sensitivity` on `pm7_rs.native`, with
  `get_polarizability`, `get_static_dielectric`, `get_berry_polarization`, `get_finite_field` and
  `get_dielectric_origin_sensitivity` on `pm7_rs.ase.PM7`. `long_range` and `keep_response` reach
  `dfpt` and `born_charges`.
- **`PM7.get_band_structure`** returns an ASE `BandStructure` from a path string, a `BandPath`, or a
  bare list of fractional k points. It did not exist. Note that `PM7` already had a
  `band_structure` **attribute** — ASE's own base-class method, which reconstructs a band structure
  from the mesh the SCF already ran on — so a naive name comparison reported this surface as covered
  while no `pm7-rs` code was reachable. A test pins that the two are different calculations.
- **`PM7(dandc=...)` runs the linear-scaling SCF.** It was an explicit `TypeError` (v0.2.1 made it
  one; the 0.2.0 changelog had advertised a keyword that `**kwargs` silently swallowed). Measured on
  a 24-atom water wire: energy within `9.1e-5` eV and forces within `1.9e-4` eV/Å of the exact
  result. Properties it cannot supply are refused **by name** rather than served from a quietly
  substituted exact SCF — someone who reached for this solver did so because the exact one is
  unaffordable, and a silent substitution is indistinguishable from success.
- **The Python CLI gained `hessian`**, which the Rust binary has had since 0.2.0 and which the wheel
  gives a pip user no other route to. **Both CLIs gained `dielectric`** for the new low-dimensional
  `ε∞`. A test enumerates the two command lines and requires them to offer the same modes.
- `dielectric_with_extent` now reports **`extent_convention`** alongside the extent. The dict
  carried a bare `extent` number with no way to tell a slab thickness in Bohr from a wire
  cross-section in Bohr² — different units, different depolarization factors, and unrecoverable
  from the value.

#### Tests and tooling

- `tests/static_dielectric.rs` — `ε⁰` pinned from both sides: an ionic crystal against
  Lyddane–Sachs–Teller, and diamond against zero. The second is **not** redundant. Any constant
  times a vanishing Born charge is still zero, so a scale test cannot catch a spurious contribution
  and this one can.
- `tests/pbc_berry.rs` — string-length convergence, the inversion-symmetry statement (see above),
  and the Berry Born charge against the CPHF one.
- `tests/pbc_finite_field.rs` — a zero field reproduces the field-free state, the two field
  treatments refuse to be combined, an unresolvable axis is refused, and the finite-field
  polarizability agrees with the CPHF one.
- `tests/dfpt_hermiticity.rs` — both ends of the conditioning range, with the well-conditioned case
  pinned at `1e-12`, far tighter than the threshold, because that is where a real error shows.
- `tests/dfpt_long_wavelength.rs` — the `q → 0` behaviour of `D(q)`: the sum rules that stay exact,
  the `O(q)` approach on a mesh that resolves the wavevector, that the non-analytic term is
  homogeneous of degree zero in `q̂`, that it raises exactly one branch and lowers none, and that
  the small-`q` residue is a mesh artifact that converges away.
- `tests/cphf_convergence.rs` — the response solver against systems harder than the well-behaved
  organics the Hessian tests use: `d` orbitals, a conjugated ring, a cation, an open shell. Each
  compares the analytic Hessian against `numerical_hessian`, which shares the SCF and the gradient
  but no part of the CPHF, so a solver that stopped early disagrees with it. A unit test in
  `src/hessian.rs` additionally pins that conjugate gradient and the fixed point land on the same
  `U` — the property that lets one carry the other as a fallback without the answer depending on
  which branch ran.
- `tests/determinism.rs` grew DFPT and Born-charge coverage, which it had none of, and now pins the
  periodic gradient and stress to be bit-identical across thread counts. That was added **before**
  the Ewald fold below, so that the difference between "reordered" and "thread-dependent" stayed
  enforceable while the reordering was made.
- **Three enumeration tests**, so the next API gap fails a test instead of shipping: every native
  entry point must have a decided ASE route (a method name, or `None` with a reason); the two
  command lines must offer the same modes; and ASE's inherited `band_structure` must not be mistaken
  for ours. These exist because an enumeration of the surfaces **at run time** — the extension
  module, `native.__all__`, the package namespace, the ASE calculator and both command lines — found
  four missing things, one of which was a bug already claimed as fixed.
- `tests/perf_report.rs` benches the perturbation path — `D(q)` against system size and mesh, and
  the Born-charge field response. Nothing benched DFPT, `force_constants` or `run_kpoint_scf`
  before, so a release claiming a scaling-order change had no measurement that would show it. It
  also reports **local** log–log slopes between consecutive points rather than one fit over the
  range; see Performance below for why that decided whether the work happened at all.
- `profile::stage` instrumentation in `src/dfpt.rs`, which had none.
- `tools/make_release_zip.py` builds from `git ls-files` rather than a tree walk, and refuses a
  dirty tree unless `--allow-dirty` is given. The 0.2.1 archive carried `src/variant.rs`, deleted
  six commits earlier, and would have carried an uncommitted `src/molden.rs` while a clean clone
  of the same revision did not compile at all.

### Changed

- **The `D(q)` Hermiticity threshold moved from `1e-8` to `1e-6`**, calibrated against measured
  eigenvector conditioning rather than raised to make something pass. The residual asymmetry of
  `D(q)` is set by how well the eigenvectors are determined, and inside a near-degenerate manifold
  that is poorly:

  | cell | level splitting | `D(q)` asymmetry |
  |---|---|---|
  | NaCl | degenerate to `1.4e-14` eV | `3.5e-19` |
  | MgO | split by `9.5e-10` eV | `7e-11` |

  Nine orders of magnitude, same code. A cubic CsPbI₃ went past `1e-8` outright with a response
  converged to `1e-10` and a value stable to three figures across four decades of SCF tolerance,
  which is a conditioning limit and not a defect. `tests/dfpt_hermiticity.rs` pins both ends, and
  `DfptResult::hermiticity` reports the measured value so a run sitting close to the line is visible
  rather than merely allowed.

- **The k-point SCF rebuilds the Fock at the density it reports.** The loop contracted the
  *output* density against a Fock built from the *input* one, where the molecular path has always
  spent one extra build at the converged density (`scf.rs`, `f_final`). `E[P] = ½ Tr[P(H + F[P])]`
  is stationary only on the idempotent manifold, so a mismatched pair leaves an error **first
  order** in the gap — and one that still looks like an energy.

  **It was not doing measurable damage, and saying so is the point.** The loop exits when the two
  densities agree to `p_tol`, which makes the first-order term `p_tol`-sized; rebuilding moved the
  gradient-versus-finite-difference agreement by `3 × 10⁻⁸` eV/Bohr. The rebuild is kept because it
  makes the invariant hold on the *unconverged* exit path too — where the reported density is the
  mixed one and the energy came from the unmixed — and because two SCF paths differing in that
  discipline is what made the question hard to answer at all.

  `tests/kpoint_variational.rs` states the property as a **convergence rate** rather than a
  tolerance: the disagreement with a central difference falls by exactly four per halving of `h`,
  over sixteen-fold in `h`, which is `h²` truncation with no constant term to find. An absolute
  bound cannot tell "the gradient is right and my step was coarse" from "the gradient is wrong by a
  constant" — a first draft of this test asserted one and failed on its own step size.

- **The CPHF is a preconditioned conjugate gradient.** The equations are linear and, at a stable
  closed-shell SCF solution, the orbital Hessian `A U = (ε_a − ε_i) U + [G(ΔP(U))]_ov` is symmetric
  positive definite — CG's hypothesis exactly, and it converges on `√κ` rather than on the spectral
  radius of a fixed-point map. The damped DIIS fixed point stays as the fallback for `p·Ap ≤ 0`
  (written `!(pap > 0.0)` so a `NaN` takes it too), which is what a saddle point or a
  not-quite-stationary density produces.

  The convergence measure is `‖M⁻¹ r‖`, and that is not a cosmetic choice: with `M = ε_a − ε_i`,
  `M⁻¹ r` **is** the fixed-point step `f(U) − U` the old solver tested, algebraically. So the
  tolerance means what it always meant, the two solvers are comparable iteration for iteration, and
  nothing downstream needed retuning.

  Measured back to back on the 102-atom Hessian: **11883 → 11211 operator applications** and
  **2.965 s → 2.738 s**. A modest gain, and the reason is worth recording rather than glossing: the
  fixed point being replaced already had depth-8 DIIS on it, so it was not the plain iteration the
  textbook comparison assumes, and `bench102` is a well-behaved organic molecule. The place CG
  earns its keep is a small frontier gap, which is why `tests/cphf_convergence.rs` now covers
  benzene, H₂S, ammonium and a methyl radical against a finite difference — and why the honest
  headline for this item is the error in **Fixed** below, not the seven percent.

- **Perturbation theory: the response Fock stopped rebuilding what does not change.** Three
  quantities that are functions of `(geometry, q)` alone were recomputed on each of
  `3N × iterations × spins` calls, and are now built once per wavevector in `ResponseTables`:
  the block index of a pair's translation `T` and of `−T` — a `HashMap` lookup that was being done
  **once per matrix element**, about sixty-six per atom pair; the Bloch phase `e^{iq·T}`; and the
  phased long-range Coulomb table `Φ_q`, whose lattice sum also computed three gradient and nine
  Hessian components that the kernel never reads. The bare perturbation's projection into the band
  basis moved out of the CPHF iteration for the same reason, and the `D(q)` assembly stopped
  rebuilding `Δh^j(k)` inside its `j'` loop, where it was a factor of `3N` too often.

  All of it is the same arithmetic in the same order, so results are bit-identical.

  `PM7_PROFILE=1` now reaches the periodic path at all, which is how this was found: the profiler
  put `dfpt: response Fock` at 64.6 % of a phonon run and, once broken down, the two-centre pair
  loop at 74 % of *that*. It also refuted the first guess — hoisting the phased Ewald sum, which
  the release plan had named as the largest single win, turns out to be 0.012 s over sixteen calls
  on these cells. It is kept because it is free and will matter at large `nat`, but it is not the
  win, and saying so is the point of having a profiler.

### Fixed

- **The static dielectric tensor's ionic term was low by a factor of 347.** Found by re-running the
  crystal validation over this release's own additions. NaCl came back with an ionic contribution of
  `0.000256` on top of an `ε∞` of `1.0119`, which reads as a small correction to a weakly
  polarizable crystal and is wrong by two and a half orders of magnitude.

  **Lyddane–Sachs–Teller is what said so.** `ε⁰/ε∞ = (ω_LO/ω_TO)²` for a cubic diatomic crystal,
  and the right-hand side comes from the dynamical matrix with and without the non-analytic term —
  sharing none of the unit conversion being checked. It required `0.0889` where the code gave
  `0.000256`, a ratio of **347.0121**. That is `HARTREE_TO_EV · a₀⁴` to five figures, and a clean
  constant ratio is a unit error rather than physics. The conversion is now `4π a₀²/Ω`, and LST
  agrees to **1.0000** (`1.100809` against `1.100810`).

  A dimensional re-derivation would have been checking the arithmetic against itself: my own
  derivation said the factor should be `HARTREE_TO_EV · a₀²`, which is off by another
  `HARTREE_TO_EV` from what the phonons require. The identity was doing real work here.

- **Five new entry points compiled, existed, and were absent from the extension module.** One of
  them was registered by a string replacement that matched nothing — the module binder says
  `module.add_function`, not `m.add_function` — so the functions were reachable from Rust and not
  from Python until a runtime call went looking. This is the class of defect the run-time surface
  enumeration above was written to catch.

- **ASE's `free_energy` was the internal energy, not the free energy.** `entropy_ev` has been
  reported since v0.2.0 "so the free energy and the internal energy stay distinguishable"
  (`docs/pbc.md`) — but nothing ever assembled the free energy, and the calculator set
  `results["free_energy"] = energy_ev`. With smearing on, the analytic force is `−∂F/∂R` and not
  `−∂E/∂R`, so `get_potential_energy(force_consistent=True)` — which ASE optimizers and several
  MD integrators use — returned a number inconsistent with the forces beside it. Measured on a
  two-atom lithium cell at a 0.3 eV Fermi–Dirac width: `entropy_ev = −0.073 eV` while
  `free_energy − energy` was exactly `0.0`.

  `Pm7Result::free_energy_ev()` now forms `E − TS` in one place and it reaches every surface:
  `free_energy_ev` on every Python result dict, `free_energy` in the ASE calculator, and a printed
  line in the Rust CLI. **No existing number moves** — without smearing `entropy_ev` is exactly
  zero and the free energy is the internal energy bit for bit, which is what
  `tests/pbc_equivalence.rs` asserts alongside the case where it is not.

  Named to stay distinguishable from the *thermochemical* free energy: this one is the **Mermin
  electronic** free energy of a smeared band structure, and no vibrational partition function
  enters it. There is still no thermochemistry — harmonic frequencies are computed and nothing is
  derived from them.

- **`get_frequencies()` on a periodic cell returned the isolated-molecule spectrum.** `vibrations`
  took no periodic keywords at all — not in the PyO3 binding, not in `native.vibrations`, not in
  the ASE calculator's vibrational cache — so a periodic `Atoms` was handed to the *molecular*
  Hessian path with the cell dropped on the floor. A two-atom diamond cell came back as a C₂
  diatomic: **−772, 654, 654 cm⁻¹** where the crystal's zone centre is **0, 0, 0, 1248, 1248,
  1248**. No error, no warning, and a plausible-looking spectrum. `native.frequencies` and
  `native.hessian` dropped them the same way, though their bindings had accepted them since 0.2.0.

  All three now forward the cell, and the zone-centre frequencies agree with `phonons` and
  `dfpt(q = 0)` to every digit.

  Found by the k-mesh delegation below: with the k mesh newly reaching `analytic_hessian`, the
  frequencies it produced were *identical* to the Γ ones — which they cannot be — and the reason
  was that neither had ever been periodic.

- **The command line was a fourth layer, and it still had the bug.** `python -m pm7_rs frequencies
  crystal.xyz` went on returning the frequencies of those atoms as an isolated molecule after the
  binding, `native.py` and the ASE calculator were fixed and the bug reported closed: a two-atom
  diamond cell came back as a C₂ diatomic at **−772, 654, 654 cm⁻¹** where `phonons` on the same
  file gives **0, 0, 0, 1248, 1248, 1248**. No test ran that mode on a cell. It forwards the
  periodic keywords now, and `test_cli.py` compares it against `phonons` rather than against pinned
  numbers, so the two routes have to keep agreeing.

- **Infrared on a periodic cell died on an internal message.** It named the retained CPHF orbital
  response — true, and useless, because it names an implementation detail rather than the fact that
  a crystal's dipole is not a function of its density. The ASE layer refused it properly and the
  library did not, so every other surface got the internal message. `ir.rs` now refuses by name and
  points at `born_and_dielectric`.

- **`analytic_hessian` refused a k mesh.** It now delegates to `dynamical_matrix_dfpt` at `q = 0`,
  which is the same calculation reached the way the `k ↔ k + q` coupling requires, and brings the
  unrestricted case with it — the perturbation solver carries a band set per spin where
  `analytic_hessian_periodic` has no unrestricted CPHF at all. The imaginary part is *checked* to
  vanish rather than dropped: at `q = 0` every Bloch phase is 1, so a non-zero one would be a
  defect in the construction and this is the one place it could hide.

  The test that pinned the old refusal asserted on the *wording* of the error. Its replacement
  asserts that the two routes produce the same matrix, which is the thing actually worth holding
  fixed — a refusal message is a decision, and a decision is allowed to change.

- **A long-wavelength `q` the k mesh could not represent returned nonsense silently.** The response
  couples `k` with `k + q`, so an `n × n × n` mesh cannot resolve `q ≪ 1/n` — and reported
  `converged: true` anyway, because the linear solve had converged; it converged to the answer for
  a question the mesh could not pose. Diamond on a `3³` mesh at `q = 1/160` came back with acoustic
  modes at **−2977 and −231 cm⁻¹** where three near-zero ones belong. There is now a warning naming
  the mesh step and the divisions that `q` would need (`PM7_QUIET` silences it).

  Investigated because of a report that `D(q)`'s acoustic sum rule diverges as `1/q²`. It does not:
  at `q = 0` the sum rule holds to `1.8e-15`, `Σ_A Z*_A` to `2.2e-15`, `Σ_B D^NA(q̂)` to `2.6e-16`,
  and on a mesh that resolves the wavevector the residue falls as `O(q)`. The residue *does* plateau
  on a mesh that cannot — 2.96e-1 at `3³`, 6.75e-2 at `5³`, 3.21e-2 at `7³` — which is a sampling
  limit rather than a defect in the construction, and is what the warning is for.

- **The CPHF returned unconverged answers as `Ok`.** Both solvers ran their iteration budget out
  and handed back the last iterate, with nothing recording that it was not a solution. The
  relaxation term is `4 G:U`, linear in `U`, so a Hessian built on an unconverged response is
  wrong in proportion to the residual — and it came back looking exactly like a converged one, with
  frequencies to match. It is now a `ResponseFailed` error naming the residual and the budget.

  No system in the test suite was hitting this, which is worth stating: the defect was that a
  failure had nowhere to go, not that failures were common.

- **`python -m pm7_rs phonons cell.xyz` without `--supercell` died** on "supercell must have three
  entries". The CLI passed `None`, which *overrides* `native.phonons`'s own `(1, 1, 1)` default
  rather than falling back to it, and every existing test passed the flag — so the default path had
  never been run. The Rust CLI has defaulted to `1×1×1` all along.

- **The CLI test harness decoded subprocess output with `locale.getencoding()`** — cp932 on this
  machine. The first error message containing a non-ASCII character killed the reader thread with
  `UnicodeDecodeError` instead of failing the assertion it was checking.

### Performance

- **The k-point gradient and stress were quadratic in the k mesh; now they are not.** The
  long-range exchange derivative ran one shifted Ewald pair sum per Born–von Kármán residue class,
  and the supercell's own reciprocal lattice is `C = n₁n₂n₃` times denser than the primitive one —
  so the derivative cost went as `C²` where the energy goes as `C`. `docs/performance.md` had named
  this as the largest known order defect in the periodic code and recorded the fix as "not
  implemented".

  `pbc::ewald::ewald_reciprocal_bvk` now does **every class's reciprocal sum in one pass over `G`**.
  The reciprocal kernels reach the pair separation only through `cos(G·d)` and `sin(G·d)`, and a
  lattice translation shifts that argument and nothing else, so the class sum collapses into a
  rotation of that pair by the translation structure factor `S_AB(G) = Σ_t c_AB(t) e^{iG·T_t}`.
  Because `G·T_t = 2π Σ_j m_j t_j / n_j`, `S` takes only `C` distinct values however many `G` there
  are, and those are a 3-D DFT of `c_AB(t)` that separates into three one-dimensional passes.

  Measured on diamond, timing the **stress** — it takes a converged SCF, so it is derivative work
  and nothing else:

  | mesh | classes | before | after |
  |---|---|---|---|
  | 4×4×4 | 64 | 0.107 s | 0.0080 s |
  | 5×5×5 | 125 | 0.379 s | 0.0100 s |
  | 6×6×6 | 216 | 1.090 s | 0.0130 s |
  | 7×7×7 | 343 | 2.809 s | 0.0175 s |

  Local exponent in the class count, over the last pair of points: **2.05 → 0.65**.

  **A single least-squares fit would have said there was nothing to fix.** One line through
  `log(time)` against `log(C)` over the whole mesh range reports **1.01**: the small meshes sit on a
  floor of costs that do not scale with the mesh at all, and a global fit averages the floor
  together with the asymptote. The consecutive-point slopes climbed 1.18 → 1.77 → 1.93 and showed
  the quadratic the fit had hidden. `perf_report` reports local slopes now.

  **Numbers move in the last digits.** The reciprocal sum is accumulated in a different order, so a
  k-point gradient, stress or cell optimization can differ at the level of summation rounding. The
  finite-difference agreement is unchanged — `tests/stress.rs`'s 3-D k-point gradient and stress
  checks pass at the same tolerances — and `tests/determinism.rs` pins the periodic gradient and
  stress to be bit-identical across thread counts.

  3-D only: the reduction is in `C`, and a wire with a 20-point mesh has `C = 20`. The 1-D and 2-D
  virial kernels also carry the separation explicitly and would need three further transforms for a
  case that does not have the problem; those keep the per-class loop.

  `pbc::ewald::the_folded_reciprocal_sum_reproduces_the_class_loop` pins the two against each other
  on a **3×2×4 triclinic** mesh. A cubic mesh cannot see a transposed axis in the separable
  transform or in the `G → m mod n` lookup — every permutation gives the same answer — so it would
  pass while the code was wrong for every cell anyone would actually use.

- **The same fold now covers the periodic Hessian**, through `ewald_reciprocal_hessian_bvk`, so the
  force constants stop paying the `C²` the gradient stopped paying. Writing it found a **third** `G`
  loop in the 1-D arm that the gradient version had missed.

## [0.2.1]

Molecular properties: an external electric field, the dipole MOPAC actually prints, orbitals, and
infrared spectra -- plus the bugs that turned up while building them.

### Added

#### External electric field
- **`Pm7Options::field`**, in MOPAC's `FIELD=(x,y,z)` volts/Angstrom convention and sign: energy,
  an exact closed-form analytic gradient, and an analytic Hessian. Validated against a MOPAC run
  to the printed digit (water at 0.5 V/A: -54.78395 kcal/mol).
- The gradient is `dE/dR_A = q_A f` in closed form and the **skeleton second derivative is
  identically zero**, because the field energy is linear in `R` at fixed density and the hybrid
  term has no coordinate dependence at all. The entire Hessian contribution therefore arrives
  through the CPHF's perturbed `dh/dR`, which is one constant diagonal block per atom.
- Accepted for a periodic cell **along non-periodic directions only** -- a chain's transverse
  axes, a slab's normal. Along a periodic direction the potential `-f.r` is unbounded and the
  request is refused with a message saying so.

#### Properties
- **`src/dipole.rs`** -- one dipole operator shared by the field, the reported dipole and the IR
  intensities, so they cannot drift apart. `DipoleBreakdown` reports the point-charge, s-p hybrid
  and p-d hybrid terms separately, mirroring MOPAC's `POINT-CHG. / HYBRID / SUM`.
- **Orbital output**: energies, coefficients, occupations and the frontier gap for **both** spin
  channels, with `ao_labels` so a coefficient matrix is readable without rebuilding the basis.
- **`src/ir.rs`** -- dipole derivatives and infrared intensities, in both requested forms: the
  dense `3 x 3N` raw tensor and the mode-projected intensities in km/mol. Costs one extra
  dipole-operator build on top of a Hessian, because it reuses the CPHF response that Hessian was
  already solving for and throwing away.
- **`OrbitalResponse`** -- the CPHF first-order orbital response, retained rather than discarded.
  Retaining it costs **no arithmetic**; the opt-in bounds memory (`3N x n_vir x n_occ`).

#### Interfaces
- Everything above reaches all five surfaces: Rust, PyO3, `pm7_rs.native`, `pm7_rs.ase.PM7`, and
  both CLIs. New: `native.orbitals`, `native.vibrations`, `native.Vibrations`, ASE's
  `get_frequencies`/`get_normal_modes`/`get_ir_intensities`/`get_dipole_derivatives`/`get_orbitals`,
  CLI mode `orbitals` and flags `--field` and `--ir` on both CLIs, plus `--dipole-origin` on the
  Python one.
- **One lazy mechanism, not three.** Quantities are grouped by the calculation that produces them;
  requesting any member computes the whole group once and caches it. `native.Vibrations` does this
  in plain Python, `ase.PM7` through ASE's own property cache. Pinned by tests that *count* calls
  into the extension rather than trusting the docstring.
- `pm7_rs.constants` -- the unit conversions the Python layer needs, derived from the same CODATA
  values the Rust core uses instead of transcribed. The ASE dipole conversion was a bare
  `0.2081943` literal with nothing tying it to the constants it comes from.

#### Validation
- `tools/oracle/baseline.py` -- a MOPAC oracle with **numeric thresholds and a non-zero exit**,
  replacing a delta column that a human had to read. 13 cases, 80 comparisons, covering heats of
  formation, Mulliken charges, dipoles and orbital energies.
- The case set now includes systems that are **large, polar and charged** (a C20 alkane, an
  aligned HF chain, a water wire, a hydroxide-doped water wire). Nothing in the previous sweeps
  was bigger than the 7 A range past which PM7 feathers to a point charge, so a long-range change
  could not show up at all.

### Fixed

- **Four ASE accessors raised on every call.** `get_frequencies`, `get_normal_modes`,
  `get_ir_intensities` and `get_dipole_derivatives` all went through ASE's `get_property`, which
  refuses any name outside `implemented_properties` -- and those four are deliberately outside it,
  so code iterating that list keeps seeing only names ASE understands. So each raised
  `PropertyNotImplementedError` on its first call while its docstring described a per-geometry
  cache that had never run once. No test called them. They now share one solve through an explicit
  cache keyed on the geometry's bytes, and `python/tests/test_ase_caching.py` **counts native
  calls** rather than trusting a docstring: asking for all five costs exactly one.
- **`get_phonons`, `get_dfpt` and `get_born_charges` raised `TypeError` on any periodic `Atoms`.**
  They spread `_periodic_kwargs` alongside `_common`, which already contains it, so `pbc` and
  `kpoints` were passed twice and Python rejected the call. The tests exercised `native.*`
  directly and never went through the calculator; `python/tests/test_api_coverage.py` now does.
- **`PM7(kpts=...)` made `get_phonons` unusable.** The calculator's k mesh was forwarded into the
  supercell route, which correctly refuses one -- so a user who set up a periodic calculator the
  normal way got a refusal for something they had not asked for. It is dropped there, with the
  reason in the docstring: Gamma of an `n1 x n2 x n3` supercell *is* that mesh, so `supercell` is
  the sampling knob and `kpts` would be a second, contradictory one. It still governs energies and
  forces, and still applies to `get_dfpt`.
- **Any accessor called before the first energy evaluation crashed** on
  `'NoneType' object has no attribute 'get_atomic_numbers'`. ASE only populates `calc.atoms`
  inside `calculate`, so `atoms.calc = PM7(); atoms.calc.get_orbitals()` died. The calculator now
  implements ASE's `set_atoms` hook -- kept in its own attribute, **not** `self.atoms`, because
  `check_state` compares that against the live object and a live reference there would stop the
  standard properties ever invalidating -- and says what to do when it genuinely has no system.
- **The periodic methods cache per argument.** A q path is not something `get_property` can key
  on, so `get_phonons`, `get_dfpt` and `get_born_charges` keep their own cache on the geometry
  **and** the argument: the same q path twice is one solve, a different one is two, and moving an
  atom invalidates both.
- **Perturbation theory now takes an unrestricted cell.** `dynamical_matrix_dfpt` used to refuse
  one and point at a supercell Hessian. It carries a band set and a response density **per spin**
  now: Coulomb couples each channel to the total density, exchange only to its own, and each
  channel is filled from its own chemical potential — which is right for both, because with
  integer occupations the response moves no electrons across `E_F` at all. The `½` the exchange
  terms used to carry as a literal is now the caller passing `Δp/2` as the same-spin density, so
  the closed-shell path is unchanged.

  Validated three ways: forcing UHF on a **closed shell** reproduces the RHF response to `10⁻⁶`
  relative (an already-validated reference, and the check that would catch a factor of two, a
  double-counted channel, or an exchange term reading the total density); an open-shell methyl
  chain matches `numerical_hessian`, which involves no perturbation theory at all; and an
  open-shell cell at complex-phase `q` still produces a Hermitian `D(q)`.

  Born charges and `ε^∞` remain closed-shell, and now say why: the field enters through `[H, r]`,
  an unrestricted cell has two different `H`, and each spin would need its own commutator *and*
  its own band-basis contraction. One commutator for both channels would converge and be wrong.
- **DFPT was wrong at every wavevector except `q = 0` and `q = ½`, and hid it.** The exchange part
  of the response kernel built its `F(T)` and `F(-T)` blocks from the same `Δp(T)`. That is valid
  for the ground state, where the density blocks are real and `P(-T) = P(T)ᵀ`, so one number serves
  twice. A response **amplitude** at wavevector `q` obeys no such relation: `Δp(T)` is
  `Σ_k w_k e^{-ik·T} ΔP(k)` with `ΔP(k)` an off-diagonal `(k+q <- k)` object whose adjoint runs the
  other way, so `Δp(-T) != Δp(T)†` unless `q` is zero. Each block is now built from its own
  response density.

  The symptom was a **non-Hermitian `D(q)`** — 2 % of the matrix scale on LiF, 80 % on a CH₂ chain
  — and, on stiffer systems, a self-consistency iteration that diverged outright and returned
  numbers around `1e33`. It survived from v0.2.0 for two compounding reasons, both now fixed:

  * every test used `q = 0` or `q = ½`, where the phases are `1` and `±1` and the two forms
    coincide. `tests/dfpt.rs` now checks `q = ⅓` and `q = ¼` against 3× and 4× supercells, and
    **asserts the reference's imaginary part is non-zero** so the test cannot quietly degenerate
    into the real-phase case it was written to escape;
  * the assembly **symmetrized `D(q)` unconditionally** on the way out, laundering any construction
    defect into a plausible Hermitian matrix. The deviation is now measured *before* the averaging
    and refused when it exceeds what rounding on a Bloch sum can explain. The averaging remains,
    because genuine rounding is real and belongs cleaned up — but it is no longer a place for bugs
    to hide.
- **A diverged linear response is now an error, not a flag.** `DfptOptions::require_convergence`
  (default **true**) makes `dynamical_matrix_dfpt` and `born_and_dielectric` refuse a result whose
  response did not converge, rather than returning `Ok` with a `converged: false` field nothing was
  obliged to read while `frequencies_cm` took the square root of `1e33`. The solver also detects
  divergence early instead of burning the whole iteration budget, and retries on a **damping
  ladder** — which genuinely rescues the common failure (a real, negative eigenvalue of the
  self-consistency map, where damping moves `λ` to `(1-a) + aλ`) and cannot rescue `λ > 1`, so the
  remaining failures are reported honestly instead of being iterated into garbage.
- **A periodic cell silently ignored an external field.** `build_core_periodic` passed `None` for
  the field, so a `Pm7Options::field` accepted by validation never reached `h_core`: the SCF, the
  energy and the forces all came back field-free. The periodic branch of
  `fixed_density_gradient_blocks` was likewise missing the one-centre `q_A f` term the molecular
  branch has. Both are fixed, and `analytic_stress` now **refuses** a field rather than returning
  the field-free virial, because the field's strain derivative is not in it.
- **The field's s-p hybrid was dropped on a k mesh.** The `T = 0` Bloch block copied only the
  *diagonal* of `h_core`, which was correct while `h_core` held nothing but the `U` diagonal. The
  field's `<s|r|p>` term is one-centre but **off**-diagonal, so it was lost -- leaving a k-point
  run whose `dE/df` was the point-charge dipole alone. The whole on-site block is copied now; a
  field-free run is bit-identical, since every element outside the diagonal is an exact zero at
  that point.
- **A periodic system reported no point-charge dipole at all, in any direction.** The Berry-phase
  argument that makes the point-charge sum meaningless applies **along a lattice vector**, not
  across one: a chain's transverse dipole and a slab's out-of-plane dipole are ordinary
  observables, and they are exactly what the transverse Born charges measure. The term is now
  masked per axis instead of wholesale.
- **`eps^inf` was wrong for every periodic system.** The DFPT solver treated the electric-field
  perturbation as an ordinary local potential: it fed the raw commutator `[H, r]` straight in,
  applying neither the convention C-3 denominator `(eps_m - eps_n)` nor the sign of `D = -r`. The
  bare term and the self-consistent kernel are now projected separately, since only the bare half
  needs converting. Born charges were **not** affected -- they use the field on one side only,
  which is why the acoustic sum rule, the supercell check and both contraction orders all passed
  throughout. Diamond used to report `eps = 1.000000` exactly (a semiconductor with no
  polarizability); it now gives 1.12, and LiF's Born charge comes out at +1.03 against a measured
  ~1.04.
- **`tools/oracle/pair_sweep.py` had never run.** It indexed a 87-entry symbol table with atomic
  numbers from a parameter file that goes to 107, and died with an `IndexError` before the first
  comparison. The sweep now selects the elements and names its exclusions -- sparkles (`Z = 87..101`,
  `103..107`), the translation-vector marker `Tv` (`Z = 102`), and the capped-bond link atom `Cb`
  (`Z = 98`), which has real parameters but is not an element. See `docs/fidelity.md` for what the
  sweep found once it could run.
- **The p-d one-centre dipole term was missing entirely.** MOPAC's `dipole.F90:118-148` applies it
  to every 9-AO atom; `pm7-rs` did not. For H2S the reported hybrid dipole was (0.917, 1.202) D
  where MOPAC gives (0.285, 0.374) -- a missing term the size of the answer, not a rounding
  difference. Affects every molecule containing S, P, Cl or a transition metal.
- **A charged molecule's dipole is now measured about the centre of mass**, as MOPAC does, because
  about the input origin it is not an observable at all. Neutral molecules are unaffected **bit
  for bit** -- the origin is short-circuited to zero below MOPAC's own 0.5 e threshold, so no
  published number moves.
- **On a k mesh the reported orbitals were not an eigenpair of anything.** `mo_energies` came from
  `band_energies[0]` -- the first *expanded* k point, which on a shifted mesh is not even Gamma --
  while `mo_coeff` was carried over from the pre-SCF Gamma solve, whose Fock was built from the
  *starting* density. They are now re-derived at Gamma from the converged Hamiltonian and labelled
  with `OrbitalSource`.
- **k-point shifts other than 0 and 1/2 are refused.** The mesh `{(i+s)/n}` is closed under
  `k -> -k` only when `2s` is an integer, which both the real-part density assembly and the
  Born-von Karman exchange require. An unsupported shift used to make the SCF stall at a small
  non-zero residual (5.8e-6 on a 1-D HF chain, unchanged after 2000 iterations, in a system with a
  19 eV gap) -- a failure that reads as a convergence problem and is really an unrepresentable
  request.
- `run_dandc` now validates its input like `run_pm7` does. A non-finite coordinate reached
  `partition`, whose median bisection sorts by position, and panicked inside a comparator instead
  of returning a `Pm7Error` naming the atom.
- The divide-and-conquer spatial grid clamps its bucket count. `DandcOptions::validate` only
  required `buffer > 0`, so `buffer = 1e-9` on a 100-Bohr system asked for ~1e24 buckets.
- Type stubs promised `Literal` method names that the wrapper passes plain `str` to, so
  `mypy` failed on 34 pre-existing errors. Nobody had seen them because `.github/` was untracked
  and **CI had never run**.

### Changed

- `#![deny(unsafe_code)]`, with one `#[allow]` and a `// SAFETY:` note on the single
  `GlobalMemoryStatusEx` call. "There is exactly one unsafe block" is now an invariant the
  compiler enforces rather than a claim.
- `analytic_hessian_with` / `HessianRequest` / `HessianResult` sit alongside `analytic_hessian`,
  which is unchanged and now a two-line wrapper.
- `VibrationalModes` keeps its eigenvectors (`modes`, `cartesian_modes`). They were computed and
  discarded, which left nothing able to say which way a mode moves.
- The MOPAC oracle runs with `RELSCF=0.0001`. At plain `PRECISE`, MOPAC's own SCF is the looser
  side on a large system, and a charge threshold tight enough to be useful fails on its noise --
  see `docs/fidelity.md`.

### Packaging

- **Verified installable and runnable where the locale is not UTF-8.** `pip install` from the
  sdist under `PYTHONUTF8=0` / `LC_ALL=C` on a `cp932` machine, then every CLI mode with stdout
  **redirected** -- which is when Python encodes with the locale rather than talking to the
  console, and therefore the only way the failure shows up. `pyproject.toml` is now pure ASCII, no
  file carries a BOM, and `python/tests/test_encoding.py` plus a CI job keep it that way. Non-ASCII
  in comments and docstrings is unaffected and deliberately not policed.

### Performance

- **The pair list is memoized on the exact geometry.** One energy-and-gradient evaluation built it
  from about seven places -- the core Hamiltonian, core-core repulsion and its gradient, dispersion
  energy and gradient, the H-H repulsion, the electronic gradient -- each re-running the same
  enumeration. `PairList::cached` keys on the **bit pattern** of every position, the cell and the
  cutoff, compared element by element rather than hashed, so there is no collision to reason about
  and a geometry differing in the last bit misses. Thread-local, so no lock and no way to hand one
  thread's geometry to another; four slots, because one evaluation legitimately wants several
  cutoffs at once. `PairList::build` is unchanged and still public.
- **The SCF history writes into reused buffers.** At depth 8 the loop freed and reallocated three
  `nao x nao` matrices every iteration, plus a fresh commutator on top; the history now rotates and
  overwrites in place. It stays chronological -- both extrapolators treat the last entry as the
  current one -- so the arithmetic is untouched. The UHF loop also stopped building a
  `2*nao x nao` stacked error matrix on every iteration to produce one scalar; its norm chains the
  two commutator slices instead, in the same order over the same values, so `err_norm` is
  bit-identical and the stall guard behaves exactly as before.
- **The divide-and-conquer gradient no longer densifies its density** — the memory-order reduction.
  `dandc_derivatives` called `to_dense()` on the `O(N)` sparse density, allocating `N_ao^2` (about
  512 MB at two thousand atoms) at the one step *after* the linear-scaling SCF had finished, which
  put a quadratic memory bound on a method whose whole purpose is to avoid one. Every read the pair
  loop makes is block-local, so it reads the sparse store directly now; the blocks are copied into
  small per-pair arrays, which also beats striding an `N_ao x N_ao` matrix in the innermost of four
  nested loops. `tests/dandc.rs` pins the sparse result against the densified one to `1e-9`.
  Periodic divide and conquer would need a translation-resolved sparse density, which does not
  exist, so it keeps the dense path.
- **`scatter_density` and `ewald_matrix` are parallel**, both over a fixed partition with an
  ordered combination, so nothing depends on `RAYON_NUM_THREADS`. `scatter_density` chunks the
  subsystems rather than giving each one its own accumulator, because a `SparseDensity` is an
  `O(N)` allocation and there is one subsystem per atom.
- **A-DIIS stopped allocating `2k` difference matrices per call.** The subtraction is fused into
  the dot product, which is the same arithmetic in the same order and therefore bit-identical.
  Expanding `<A-A'|B-B'>` into four dot products would not be -- that reassociates the rounding --
  so it is deliberately not done that way.
- **A transcendental reduction was implemented and then reverted**, with the reasoning recorded in
  `docs/performance.md` because the negative result is worth as much as the change would have been.
  Factoring `cos(G.(R_b - R_a))` into `cos.cos + sin.sin` makes `ewald_matrix` **N** transcendentals
  per `G` instead of `N^2` -- and less accurate, because `G.R_a` grows with `|G|` and with the
  atom's distance from the origin while `G.(R_b - R_a)` is bounded by the cell diameter. The lost
  digits land in the matrix every periodic SCF iterates against, and were enough to stop a rattled
  Gamma-point cell converging inside its budget. `tests/scf_convergence.rs` caught it.
- **The hydrogen-bond topology is now linear**, and 6.0x faster at 960 atoms (21.99 -> 3.68 ms on a
  320-monomer water wire); its measured exponent fell from `N^1.8` to `N^1.0`. Every energy is
  bit-identical: the spatial grid only proposes candidates, and each search sorts them back into
  the ascending index order the full scans used -- which is load-bearing, because the first
  acceptor to claim a hydrogen keeps it.
  The culprit was **not** the loop that looks quadratic. Gridding the acceptor-pair search and the
  candidate dedup first changed the total time by nothing at all; `PM7_PROFILE=1` then showed
  `neighbours_capped` -- a full scan run twice per candidate bond -- was 90 % of it.
- **Divide and conquer no longer rebuilds the Ewald far field every SCF iteration.** The matrix and
  its nuclear contraction depend only on the geometry, so ~55 repetitions of a serial `O(N^2 n_G)`
  lattice sum per run are gone. `find_fermi` no longer recomputes a Fermi-independent weighted norm
  on each of its 200 bisection steps.
- `symmetric_eigen` borrows its input instead of walking a row-major buffer column-major, and skips
  the ascending-order permutation when faer already returned ascending eigenvalues -- two `N^2`
  strided copies per diagonalization, and there is one per SCF iteration, per k point, and per
  divide-and-conquer subsystem.
- The long-range Ewald **exchange** loop in the Fock build is parallel; it was the only serial,
  unbatched loop left in that routine, and it runs on every Fock build.
- `hydrogen_bond_energy` is parallel over bonds, matching the gradient and Hessian in the same file.
- The coincident-atom check at the top of every `run_pm7` is `O(N)` rather than `O(N^2)`.
- **`tests/determinism.rs`** pins the thread-count independence the project has always claimed:
  a gradient, an analytic Hessian, a k-point energy and a divide-and-conquer SCF, compared
  **bit for bit** across rayon pools of 1, 2 and 7 threads. Nothing checked it before.

### Documentation

- `docs/theory.md` gains a normative **Conventions** section (C-1 to C-8): the field sign, the
  three dipole operators, the commutator identity, spin and k-weight factors, the Born-charge
  index order, the dielectric sign, the LO-TO contraction, and the dipole-derivative origin. They
  are not independent -- several look wrong in isolation and are only correct together.
- `docs/fidelity.md` records two findings about MOPAC itself: its printed vibrational dipoles are
  **half** the derivative (`fmat.F90:197-252` divides a difference over `delta` by `2*delta`;
  measured ratios 1.995, 2.002, 1.995), and its default SCF is the looser side on a large system.
## [0.2.0]

Periodic boundary conditions, linear scaling, and perturbation theory at arbitrary wavevector.

### Added

#### Periodic boundary conditions
- **1-D chains, 2-D sheets and 3-D crystals**, at the Γ point or on a Monkhorst–Pack k mesh, for
  neutral and charged cells, restricted and unrestricted. Energy, analytic gradient, analytic
  stress and analytic zone-centre force constants.
- Two modes: `PbcMode::Ewald` (default, absolutely convergent, independent of the splitting
  parameter, the only mode with a stress) and `PbcMode::MopacCluster` for reproducing MOPAC's own
  solid-state numbers. The Ewald split is exact rather than truncated, because beyond 7 Å PM7's
  feathering makes every two-centre integral *identically* a point-charge monopole.
- Charged cells in every dimensionality, with a uniform neutralizing background reported
  separately as `background_ev` and a Makov–Payne value reported as a diagnostic and never added.
- Metals: Fermi–Dirac, Gaussian and Methfessel–Paxton smearing, with the entropy term reported
  separately as `entropy_ev`.
- All three post-SCF corrections (D2 dispersion, EH+ hydrogen bonding, PM7-HH) are periodic in the
  energy, gradient, stress and force constants. Image sums are tapered with a C² smootherstep;
  without it the energy steps as an image crosses the cutoff and a strain finite difference
  diverges as `1/h`.

#### Lattice dynamics
- `force_constants` — real-space `Φ(0A, TB)` from a supercell's analytic Hessian, with
  `dynamical_matrix(q)`, `frequencies_cm(q)`, `acoustic_residual()` and
  `enforce_acoustic_sum_rule()`.
- **`dynamical_matrix_dfpt` — density-functional perturbation theory at arbitrary `q`**, coupling
  `k` with `k + q` through a complex CPHF with Pulay acceleration. Validated against the
  zone-centre analytic Hessian, against the numerical Hessian on 2- and 4-point meshes, and
  against a commensurate supercell — three independent references.
- `band_structure` — the converged Fock diagonalized along a k path. Deliberately not an SCF: a
  high-symmetry path is the wrong set of points to build a density from.

#### Linear scaling
- **Divide and conquer** (Dixon–Merz) for molecules and for periodic cells, Γ and k-point, energy
  and gradient and stress, restricted and unrestricted. Measured log–log slope **0.99** above 200
  atoms against **2.19** for the exact SCF, with a crossover near 350 atoms.
- `SparseDensity`, a CSR-like atom-pair block store. A dense global density would make the method
  quadratic in memory and assembly however linear everything else was.

#### Interfaces
- CLI: new `stress`, `phonons` and `bands` modes; new `--cell`, `--kpoints`, `--kshift`,
  `--smearing`, `--pbc-mode`, `--dandc`, `--supercell`, `--qpoints` and `--acoustic-sum-rule`
  flags. An extended-XYZ `Lattice="..."` key is read automatically.
- Python: `stress`, `phonons`, `band_structure` and `divide_and_conquer`; every existing function
  takes `cell`, `pbc`, `kpoints`, `kpoint_shift`, `smearing` and `pbc_mode`.
- **The `pm7-rs` console script that `pip install` provides now matches the Rust CLI mode for
  mode** — `phonons`, `bands` and `--dandc` were missing, so a pip user could not reach them at
  all. `python -m pm7_rs` is the same entry point. Flag abbreviation is now off, so a typo like
  `--kpoint` is an error rather than a silent match for `--kpoints`.
- ASE: `"stress"` in `implemented_properties`, in Voigt order and eV/Å³, so `FrechetCellFilter`
  and `NPT` work. `kpts=`, `smearing=`, `pbc_mode=` and `dandc=` on the constructor.
- `python/pm7_rs/_native.pyi` — PEP 561 type stubs for the compiled extension.

#### Diagnostics
- `profile` — a staged wall-clock timer behind `PM7_PROFILE`. It exists because reading the source
  and picking the loop that looks expensive is how you optimize a stage that was 0.1 % of the run;
  the planned `G:U` GEMM turned out to be exactly that.

### Changed (breaking)

- `Molecule` gained a `cell: Option<Cell>` field. Struct-literal construction needs
  `..Default::default()` or the `Molecule::new(..).with_cell(..)` builder; every other API is
  source-compatible, and `run_pm7`, `closed_form_gradient`, `analytic_hessian` and `optimize` stay
  single entry points that dispatch on the cell rather than splitting into periodic twins.
- `requires-python` raised to `>= 3.10` (3.9 reached end of life in October 2025); the pyo3 ABI is
  `abi3-py310`, so one wheel per platform still covers everything from 3.10 up.
- `pyproject.toml` moved to PEP 639 (`license = "GPL-3.0-or-later"` plus `license-files`) and to
  `dynamic = ["version"]`, so the version lives only in `Cargo.toml`.

### Performance

- **The 102-atom analytic Hessian is 2.2× faster** (9.0 s → 4.2 s), and phonons with it. The CPHF's
  two matrix products now go through faer's blocked GEMM instead of a hand-written `ikj` loop, with
  transposes taken as views rather than copies — the response-density product got 4.3× faster and
  the occupied–virtual projection 5.1×. Summation order changes, so results move in the last ulp;
  runs stay reproducible.
- A staged profiler (`PM7_PROFILE=1`) drove this, and immediately overturned the plan: the `G:U`
  double loop this release had named as the GEMM candidate is **0.1 %** of a Hessian. Three further
  optimizations were implemented, measured and reverted — gathering density sub-blocks, packing the
  Coulomb contraction onto lower-triangle indices, and batching the response Fock across degrees of
  freedom. All three reduce work and all three are slower, because at NDDO block sizes the pair
  loop is bound by per-pair overhead rather than by memory or arithmetic.
  [`performance.md`](docs/performance.md) has the numbers and the code says so in place.
- `Pm7Parameters::method` caches its parsed tables per method (`Pm7Parameters::shared` returns the
  cached copy without cloning). Parsing a thousand rows of CSV is nothing next to an SCF and not
  nothing next to a *water* SCF, which is what a molecular-dynamics step is.
- The two-centre Fock contraction reads a precomputed pack-index table instead of recomputing a
  branchy `pack()` on every innermost iteration.

### Fixed

- **The phased Ewald sum dropped its zero-wavevector term whenever `q` was a nonzero reciprocal
  lattice vector.** `Φ_q` is periodic in `q` under the reciprocal lattice, so `q = G` must give
  exactly the `q = 0` answer, but the 2-D sheet term, the 1-D log term and the 3-D background were
  keyed off `q == 0` while the reciprocal sum dropped `|G + q| ≈ 0`. This is the common case, not
  a corner: the long-range exchange sums a *supercell* lattice at a `q` commensurate with the k
  mesh, and every such `q` is a supercell reciprocal lattice vector. It cost a rank-one transverse
  error of 2.7 eV/Bohr² in `D(½,0,0)` for a 1-D chain whose true transverse force constants are
  0.09.
- The 1-D Ewald log kernel's small-ρ limit was `−α²/2` where the series gives `−α²`. Invisible in
  the gradient, where it is multiplied by the vanishing perpendicular offset; the Hessian's
  transverse term picks it up undamped.
- The k-point gradient and stress used a single density matrix for every image pair, giving a
  19.3 eV/Å force on a perfect crystal and a pressure off by 436 GPa. Fixed with a Born–von
  Kármán-class-resolved density.
- Force-constant translations were not closed under negation, making `D(q)` non-Hermitian away
  from the commensurate points; the acoustic-sum-rule projection had a fixed point at half the
  residual.
- The divide-and-conquer environment potential double counted its electronic half once per
  subsystem — an `N²` error that reached 55 000 eV on a 240-atom chain.
- CDIIS settled into a limit cycle around `1e-6` once its error vectors turned linearly dependent;
  the `B` matrix is now scaled to a unit diagonal so its pivot guard is a relative test.

### Documentation

- New: [`pbc.md`](docs/pbc.md), [`divide_and_conquer.md`](docs/divide_and_conquer.md),
  [`performance.md`](docs/performance.md), [`singularities.md`](docs/singularities.md),
  [`packaging.md`](docs/packaging.md).
- `singularities.md` separates the singularities that are removable from the ones that belong to
  PM7 itself, with the measurement that tells them apart. The EH+ gradient divergence turns out to
  be the acceptor's dihedral going undefined on its own axis — a defect in the published
  functional form, shared with MOPAC, that no reformulation removes.

### Tests

- **d orbitals in periodic systems** now have their own suite (`tests/pbc_d_orbitals.rs`): silicon
  (3-D, spd/spd), an H₂S chain (1-D, mixed sp/spd) and a ZnO sheet (2-D, two d-bearing elements),
  covering the k-mesh energy, the analytic gradient against a finite difference, the stress against
  a strain finite difference, and the phonons' acoustic sum rule. The MNDO/d kernel and the
  periodic machinery were written at different times and the places they meet — `has_any_d` routing
  inside the image-pair loop, the 45×45 packed one-centre block under a Bloch sum, the `Dual2`
  d-path in the periodic skeleton Hessian — were reached by neither a d-bearing *molecule* test nor
  an s/p *crystal* test. All four passed unchanged, so this pins behaviour rather than fixing it.
- `tests/cli.rs` and `python/tests/test_cli.py` run both command-line interfaces as subprocesses.
  A renamed flag or an unwired mode compiles perfectly and fails only when someone runs it.

### Tooling

- `tools/oracle/*.ps1` became `tools/oracle/*.py`, so the MOPAC oracle runs on any platform. MOPAC
  is located from `$MOPAC_EXE`, then the vendored Windows tree, then `PATH`.
## [0.1.2]

### Changed (breaking)

- Renamed the PM7-family selector from `variant` to `method` on every API surface, matching the
  terminology used by MOPAC itself (`method_pm7`, `method_pm7_hh`, …). The accepted values are
  unchanged (`"pm7"`, `"pm7-ts"`, `"pm7-"`, `"pm7-hh"`, `"pm7-sparkle"`, …).
  - Rust: `Pm7Variant` → `Pm7Method`, module `variant` → `method`, `Pm7Options::variant` →
    `Pm7Options::method`, `Pm7Parameters::variant(..)` → `Pm7Parameters::method(..)`, and the
    `Pm7Parameters::variant` field → `::method`.
  - Python: the `variant=` keyword becomes `method=` on `single_point`, `gradient`, `forces`,
    `hessian`, `frequencies`, and `optimize`; `single_point` echoes the key `"method"`.
  - ASE: `PM7(variant=...)` becomes `PM7(method=...)`, and the attribute is `calc.method`.
    Because ASE's base `Calculator` absorbs unknown keywords instead of rejecting them, a stale
    `variant=` would otherwise be silently ignored and quietly downgrade the run to plain PM7;
    it now raises a `TypeError` naming the replacement. The native functions already reject
    unknown keywords on their own.
  - CLI: `--variant` becomes `--method`; `--json` output reports `"method"`.

### Documentation

- Rewrote `docs/scope.md`: the milestone table now records M0–M10 as complete instead of listing
  the d-orbital, Sparkle, and post-SCF-correction work as pending, and it states the actual
  boundaries (non-periodic only, no thermochemistry, fixed-topology hydrogen-bond derivatives,
  the localized frame-singularity fallback, and the TiF4-class residual).
- Rewrote `docs/theory.md`: it described only the M0–M2 s/p core and declared d orbitals,
  Sparkles, and the corrections unimplemented. It now documents the s/p/d MNDO/d basis, the
  Hamiltonian terms, point-charge feathering, the MOPAC initial guess, the post-SCF corrections,
  the heat-of-formation assembly, and the AD/CPHF derivative scheme including the `Dual2N<27>`
  hydrogen-bond Hessian.

## [0.1.1]

### Fixed

- Included PM7 post-SCF correction derivatives in the public fixed-density validation gradient.
- Replaced the EH+ hydrogen-bond finite-difference gradient with first-order AD of the same
  fixed-topology energy expression.
- Rejected empty, non-finite, coincident-atom, invalid-tolerance, and invalid exchange-cutoff
  inputs before integral allocation; SCF failures now report their final density residual.
- Corrected README MOPAC reference values and removed corrupted text.

### Performance and memory

- Bounded transient Fock J/K contribution storage to 2048 atom pairs per batch.
- Based the automatic memory budget on currently available RAM and increased the conservative
  CPHF per-worker estimate.
- Capped the large-stack EH+ Hessian pool at four workers.
- Kept the release profile at Fat LTO with one codegen unit for maximum calculation throughput;
  OOM safeguards apply to electronic-structure calculations rather than compilation.
- Avoided a duplicate SCF in normal ASE force evaluations and added `free_energy` support.

### Documentation

- Added primary PM7, MOPAC, hydrogen-bond, ASE, and PySEQM references.

## [0.1.0]

First public release: a Rust-native implementation of the **PM7** semiempirical
NDDO method (Stewart, *J. Mol. Model.* **19**, 1 (2013)) and its PM7-family
methods, validated against MOPAC v23.2.5.

### Methods & elements
- Full **MNDO/d s/p/d NDDO kernel** — H through the heavy main-group and
  transition-metal PM7 elements (9-AO `spd` two-center integrals, one-center
  Slater–Condon integrals, d-orbital overlaps).
- Methods: **PM7**, **PM7-TS**, **PM7-minus** (corrections off), **PM7-HH**,
  and **Sparkle/PM7** (Ln(III) lanthanide sparkles).
- Post-SCF corrections — PM6-DH **dispersion** and the PM7 **hydrogen-bond**
  ("EH+") correction, both bit-exact vs MOPAC; PM7-HH H–H repulsion.
- RHF and UHF; RHF/UHF selectable independently of multiplicity.

### Properties
- Energies / heats of formation, **analytic gradients**, and **analytic
  Hessians** (forward-mode dual-number AD). The many-body H-bond ("EH+")
  correction's Hessian is analytic too, via multi-variable second-order AD
  (`Dual2N<27>`): each bond's geometric energy yields its exact ≤27×27 local
  second-derivative block in one pass (no finite differences; its gradient uses a
  fixed-topology central difference). With vibrational analysis.
- **L-BFGS** geometry optimization.

### Interfaces
- Rust library, a `pm7_rs_cli` command-line tool, and Python — native
  (`pm7_rs.native`) and ASE (`pm7_rs.ase.PM7`, eV/Å) — bindings via maturin.

### Numerics & performance
- **Point-charge feathering** (`l_feather`): every two-center NDDO integral
  (core–core, electron–core, two-electron) smoothly transitions to the exact
  point-charge value as atoms separate, exactly as MOPAC does for all PM7 runs.
  This makes heats of formation reproduce MOPAC PM7 to the printed 1e-5 kcal/mol
  across sp, d, transition-metal, and lanthanide-sparkle species, and makes all
  five methods (PM7, PM7-TS, PM7-minus, PM7-HH, Sparkle) bit-exact vs MOPAC.
- **MOPAC initial-density guess** (`moldat.F90`): the SCF guess smears the core
  charge over the sp orbitals as MOPAC does, so pm7-rs converges to MOPAC's SCF
  stationary point even for the ionic/covalent-**bistable** diatomics (BF, AsF,
  AlN) that otherwise settle in a different valid basin.
- Sparkle/PM7 lanthanide core–core pairs are zeroed as in MOPAC `switch.F90`, so
  every Ln–X interaction completes identically (fixes a Gd-only +13 kcal/mol
  outlier from its stray atomic pair parameters).
- The analytic gradient/Hessian is exact for bonds exactly on a Cartesian axis
  too: the singular pair's derivatives come from a finite difference of the
  correct-value integrals (`rotfix`), so no orientation loses the perpendicular
  force/curvature component.
- Linear algebra via **faer** (pure-Rust; no LAPACK/BLAS); hot loops
  parallelized with **rayon** bit-identically.
- 2018-CODATA constants matching MOPAC v23.2.5.
- Pre-flight **memory guard** (returns a clean error instead of OOM-crashing on
  oversized systems) and an optional smooth long-range-**exchange cutoff** for
  the analytic-Hessian CPHF (off by default → bit-identical).

### Provenance
- PM7 parameter tables generated from the pinned MOPAC v23.2.5 source
  (Apache-2.0); attribution in `THIRD_PARTY_NOTICES.md`. No fitted parameter is
  transcribed by hand.

### Known limitations
- A ≤0.36 kcal/mol heat-of-formation residual remains for four-coordinate
  transition-metal species with a highly populated d shell (e.g. TiF4). The
  Mulliken charges still match MOPAC, so it is a small d-integral-value
  difference at the same SCF solution, not an SCF-basin or scaling error.
