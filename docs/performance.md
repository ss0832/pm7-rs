# Performance

Measured numbers, not claims. Everything here comes from a test you can run:

```bash
cargo test --release --test perf_report -- --nocapture --ignored --test-threads=1
cargo test --release --test dandc_scaling -- --nocapture --ignored
```

Timings are the **minimum of three runs** on 16 logical cores (Windows 11, rustc 1.94). Minimum,
not mean, because the failure mode of a loaded machine is one-sided: a run can be slowed
arbitrarily and never speeded. Even so they bounce by tens of percent — the 482-atom
divide-and-conquer point read 17.3 s once and 9.5 s the next time — so treat the exponents as the
result and the absolute times as indicative.

**If a number here looks wrong, check what else is running before believing it.** A measurement
taken while an unrelated training job had the machine at 100 % showed a change making the Hessian
60 % *slower* when it had made it 40 % faster.

## What 0.2.2 changed, and the guess it refuted

### The profiler reached the periodic path, and the plan was wrong

`src/dfpt.rs` had **no** `profile::stage` calls at all, so the most expensive routine in the crate
was invisible to `PM7_PROFILE=1`. The release plan, written from reading the code, named one item
as the largest single win: `fock_response_q` rebuilt its phased Ewald table on every call, `3N ×
iterations × spins` times, and discarded twelve of the thirteen components it computed.

Instrumented and measured, that item is **0.012 s over sixteen calls**. On cells of a few atoms the
`nat²` lattice sum is simply cheap. The hoist is kept — it is free, bit-identical, and will matter
at large `nat` — but it is not the win, and the plan that predicted it was reasoning from operation
counts without a denominator.

Where the time actually was, on a phonon run over the diamond series:

| stage | thread-seconds | share |
|---|---|---|
| `dfpt: response Fock` | 97.2 | 42.8 % |
| — of which `kernel two-centre pairs` | **71.7** | **31.6 %** |
| — of which `kernel BvK exchange` | 1.34 | 0.6 % |
| — of which `kernel long-range Coulomb` | 0.03 | 0.0 % |
| `dfpt: long-range kernel at q` (the hoisted table) | 0.012 | 0.0 % |

Inside that pair loop, `out.add(t, ..)` and `delta_same.get(t)` each did a `HashMap<[i32; 3],
usize>` lookup **per matrix element** — about sixty-six hashes per atom pair, on every call. The
translation of a pair, its Bloch phase and its block index are all functions of `(geometry, q)`
alone, so they are precomputed once per wavevector in `ResponseTables` now. Per call:

| | before | after |
|---|---|---|
| `kernel two-centre pairs` | 23.44 ms | **20.79 ms** |
| `dfpt: response Fock` | 31.75 ms | **29.64 ms** |

Eleven percent of the dominant loop, not the factor of two the hash-count suggested — the rest of
it is the four-index `two_e` contraction, which is real arithmetic. Two further hoists (the bare
term's band-basis projection out of the CPHF iteration, `at_k` out of the assembly's inner loop,
where it was a factor of `3N` too often) are in the same commit and are also bit-identical.

### CPHF by conjugate gradient

The CPHF is a linear system whose operator is symmetric positive definite at a stable closed-shell
SCF solution, so preconditioned conjugate gradient applies and converges on `√κ` rather than on a
fixed-point spectral radius. Measured back to back on the 102-atom Hessian:

| solver | operator applications | wall (best of 3) |
|---|---|---|
| damped fixed point + DIIS (0.2.1) | 11883 | 2.965 s |
| **preconditioned conjugate gradient** | **11211** | **2.738 s** |

Six percent fewer applications and eight percent faster — a modest gain, and the reason belongs
here rather than in a footnote: the fixed point being replaced already had depth-8 DIIS on it, so
it was not the plain iteration the textbook comparison assumes, and `bench102` is a well-behaved
organic molecule with a comfortable frontier gap. CG earns its keep where the gap is small;
`tests/cphf_convergence.rs` is the set that would show it.

The larger finding from that work was not a timing. Both solvers used to run their iteration budget
out and return the last iterate as `Ok`. The relaxation term is `4 G:U`, linear in `U`, so a
Hessian assembled from an unconverged response is wrong in proportion to the residual and looks
exactly like a converged one. It is an error now.

**Both solvers** meant both *restricted* solvers. The coupled α/β solver behind every open-shell
Hessian kept the old behaviour two releases longer, because it does not share a loop with them and
was not looked at when they were fixed — it ran a hard-coded hundred iterations and returned `ua,
ub` as `Ok` whatever the error had reached. Fixed in 0.2.3, through the same refusal; the budget
both paths spend is now [`Pm7Options::cphf_max_iterations`](rust-api.md#options--pm7options), so it
is one number rather than two constants that were equal by coincidence.

### Benchmarks that reach the regime

`benches/scaling.rs` topped out at 98 atoms for a single point and **26 atoms** for a Hessian, and
nothing benched DFPT, `force_constants` or `run_kpoint_scf` at all — so a release claiming a
scaling-order change had no measurement that would show it. `tests/perf_report.rs` now has a
perturbation section over system size and mesh, and every timed case is a **best of three**: the
same DFPT binary measured 15.3 s and 19.2 s on consecutive runs, which is wider than most changes
worth making.

## What 0.2.1 changed, and how it was found

Four changes, three of them found by the profiler after a first guess had already been written and
measured to do nothing. That sequence is the point of this section.

### The hydrogen-bond topology was `O(N^1.8)`, and the obvious fix missed it

The EH+ correction's *derivatives* were already linear; its **topology perception** was not. On a
water wire, `hydrogen_bond_energy`:

| monomers | atoms | before | after | |
|---|---|---|---|---|
| 80 | 240 | 1.75 ms | 1.39 ms | |
| 160 | 480 | 5.82 ms | 2.34 ms | |
| 320 | 960 | **21.99 ms** | **3.68 ms** | 6.0× |
| 640 | 1920 | — | 10.47 ms | |

Ratios per doubling went from 3.3–3.8 (i.e. `N^1.8`) to about 1.6–2.0.

The instructive part is the order in which it was fixed. The acceptor-pair search and the
candidate dedup were gridded first — the two loops that *look* quadratic — and the total time did
not move at all (21.99 → 22.47 ms). Only then was `PM7_PROFILE=1` turned on, which said topology
was 90 % of the cost, and the actual culprit turned out to be `neighbours_capped`: a full
`0..numat` scan called **twice per candidate bond**, `O(P·N)`. Gridding that took the topology
stage from 0.103 s to 0.019 s over five calls and dropped its exponent from 1.77 to 0.98.

Every energy in the table above is **bit-identical** before and after. The grid only proposes
candidates; each search sorts them back into ascending index order before testing, which is what
the full scans did — and the order is load-bearing, because the first acceptor to claim a hydrogen
keeps it and the neighbour list drops its longest bond only once a fifth arrives.

### Divide and conquer: the far field was rebuilt every iteration

`environment_field` called `ewald_matrix` on every SCF iteration, and `ewald_matrix`'s own
documentation says to build it once per geometry — which the molecular path already did. At the
50–57 iterations a divide-and-conquer run takes, that was ~55 repetitions of a serial `O(N² n_G)`
lattice sum. The nuclear half `Σ_B M_AB Z_B` is constant too, so it is now hoisted with it; only
the electronic contraction is redone.

`find_fermi`'s 200-step bisection recomputed `Σ_μ w_μμ |C_μ,index|²` inside its eigenvalue loop,
although that norm depends only on `(subsystem, state)`. Precomputing it turns the search from
`O(200 · Σ_α m_α²)` into `O(Σ_α m_α² + 200 · Σ_α m_α)`.

### Two constant-factor items

`symmetric_eigen` built its faer input with `Mat::from_fn`, which walks a row-major buffer
column-major — an `N²` strided read on every diagonalization, of which there is one per SCF
iteration, per k point, and per divide-and-conquer subsystem. It now borrows the buffer. The
ascending-order permutation is also skipped when faer already returned ascending eigenvalues,
which it does in practice.

The long-range Ewald **exchange** loop in the Fock build was the one serial, unbatched loop left
in an otherwise rayon-parallel routine, and it runs on every Fock build. Splitting it by row atom
is bit-identical rather than merely close: atom `A` writes only its own rows, and within them each
element belongs to exactly one column atom, so every element is written once by one task and there
is no reduction whose order a thread count could change.

### The guarantee that had no test

`tests/determinism.rs` runs a molecular gradient, an analytic Hessian, a k-point energy and a
divide-and-conquer SCF inside rayon pools of 1, 2 and 7 threads and compares **bit patterns**. The
"thread count never changes the numbers" claim in [theory.md](theory.md) had nothing behind it
until now, which is exactly the sort of promise that holds until one `par_iter().sum()` slips in
and then fails only on someone else's machine.

## Molecular

| system | atoms | single point | gradient | analytic Hessian | Hessian in v0.1.2 |
|---|---|---|---|---|---|
| water | 3 | 0.011 s | 0.004 s | 0.006 s | 0.012 s |
| ethanol | 9 | 0.008 s | 0.008 s | 0.017 s | 0.032 s |
| `bench102.xyz` | 102 | 0.29 s | 0.31 s | **4.64 s** | 9.01 s |

The gradient runs its own SCF, so it is not an increment on the single point; at these sizes the
two are within noise of each other and the Hessian's CPHF dominates.

### Where the Hessian's time goes, and what moved it

`PM7_PROFILE=1` turns on a staged timer (`src/profile.rs`). On the 102-atom Hessian, before any of
this work:

| stage | share |
|---|---|
| CPHF: response Fock | 46.0 % |
| CPHF: response density `Cv U Coᵀ` | 24.8 % |
| CPHF: projection `Cvᵀ F Co` | 23.0 % |
| **`H_relax = 4 G:U` assembly** | **0.1 %** |

That last row is the point of having a profiler. The plan for this release named the `G:U` double
loop as a GEMM candidate; it is one twentieth of one percent of the run. Half the time was in two
matrix products written as a hand-rolled `ikj` loop.

Routing them through faer's blocked GEMM, with the transposes taken as **views rather than copies**
(`Cvᵀ` was being materialized on every one of ~4000 calls), gave:

| stage | before | after | speedup |
|---|---|---|---|
| response density | 32.0 | 7.4 thread-s | 4.3× |
| projection to the ov block | 29.6 | 5.8 thread-s | 5.1× |
| whole Hessian, 102 atoms | 9.0 s | 4.2 s | 2.2× |

### Four more attempts, three of them wrong

Kept:

* **A pack-index table** in the two-centre Fock contraction, replacing a branchy `pack()` on every
  innermost iteration. Bit-identical and strictly less work.

Measured and reverted. Each of these looks obviously right on paper, which is the reason to write
them down:

| attempt | idea | result |
|---|---|---|
| Gather density sub-blocks | do the strided reads once instead of `n_a²` times | 5.2 → 9.8 s |
| Pack the Coulomb contraction | `J` is symmetric in both index pairs: 100 steps instead of 256 | 4.17 → 4.39 s |
| Batch the response Fock across DOFs | one pass over the pairs serves all `3N` densities | 5.2 → 8.8 s |

The batching did what it was meant to structurally — 70 Fock passes instead of 3961 — and was
still slower. The premise was that the pair loop is memory bound; at 102 atoms the whole integral
set is about 4 MB, so it sits in L3 across calls and there was no traffic to save, while the
batching cost the per-DOF parallelism the `par_iter` had been getting for free.

The packing is the sharpest case: strictly less arithmetic, still slower, because building the
packed density vectors is two heap allocations per pair per build and there are tens of millions of
those.

The common thread is that at NDDO block sizes this loop is neither memory bound nor arithmetic
bound — it is bound by per-pair overhead. Anything that adds a fixed per-pair cost loses even when
it removes work from the inner loop. All three are commented in place so nobody spends the
afternoon again.

## Periodic

Diamond, two-atom primitive cell, as the k mesh is refined:

| mesh | k points (folded) | single point | gradient | stress | E/atom (eV) |
|---|---|---|---|---|---|
| Γ | 1 | 0.018 s | 0.025 s | 0.026 s | −94.07 |
| 2×2×2 | 8 | 0.108 s | 0.122 s | 0.103 s | −121.56 |
| 3×3×3 | 14 | 0.110 s | 0.147 s | 0.173 s | −123.19 |
| 4×4×4 | 36 | 0.171 s | 0.424 s | 0.414 s | −123.14 |
| 6×6×6 | 112 | 0.690 s | 3.18 s | 3.58 s | −123.07 |

**The single point is close to linear in the number of k points** (112× the points for 31× the
time, helped by time-reversal folding).

**The gradient and the stress used to be quadratic in the mesh, and in 0.2.2 they are not.** The
cause was the long-range exchange derivative: each Born–von Kármán residue class needed its own
shifted Ewald pair sum over the supercell, and there are `C = n₁n₂n₃` classes while the supercell's
own reciprocal lattice is `C` times denser — so the derivative cost went as `C²` where the energy
goes as `C`.

`pbc::ewald::ewald_reciprocal_bvk` now does every class's reciprocal sum in **one** pass over `G`.
The reciprocal kernels reach the pair separation only through `cos(G·d)` and `sin(G·d)`, and a
lattice translation shifts that argument without touching anything else, so the class sum collapses
into a rotation of that pair by the **translation structure factor** `S_AB(G) = Σ_t c_AB(t)
e^{iG·T_t}`. Since `G·T_t = 2π Σ_j m_j t_j / n_j`, `S` takes only `C` distinct values however many
`G` there are, and those values are a 3-D DFT of `c_AB(t)` that separates into three
one-dimensional passes.

### What it measured

Diamond, two atoms, mesh only. The **stress** is the honest column: it takes a converged SCF, so it
times derivative work and nothing else. (The gradient runs its own SCF, so subtracting one best-of-N
from another would leave the difference of two noise floors.)

| mesh | C | stress before | stress after | speedup |
|---|---|---|---|---|
| 2×2×2 | 8 | 0.0074 s | 0.0059 s | 1.3× |
| 3×3×3 | 27 | 0.0238 s | 0.0064 s | 3.7× |
| 4×4×4 | 64 | 0.107 s | 0.0080 s | 13× |
| 5×5×5 | 125 | 0.379 s | 0.0100 s | 38× |
| 6×6×6 | 216 | 1.090 s | 0.0130 s | 84× |
| 7×7×7 | 343 | 2.809 s | 0.0175 s | **160×** |

Local exponent in `C`, over the last pair of points: **2.05 → 0.65**.

`tests/perf_report.rs::kpoint_derivative_scaling` reports the table above, and reports **local**
slopes between consecutive points rather than one fit over the range. That distinction decided
whether this work happened at all: a single least-squares line through the pre-change timings
reports **1.01**, because the small meshes sit on a floor of costs that do not scale with the mesh
and a global fit averages the floor together with the asymptote. The consecutive slopes climbed
1.18 → 1.77 → 1.93 and showed the quadratic the fit had hidden.

### The Hessian half of the same fold

`ewald_reciprocal_hessian_bvk` does for the force constants what `ewald_reciprocal_bvk` does for the
gradient and stress, so the periodic Hessian stops paying the `C²` they stopped paying. No separate
timing table: the same class-count argument applies unchanged, and the equivalence test against the
class loop is what pins it.

Writing it turned up a **third** `G` loop in the 1-D arm that the gradient version had missed. That
is the argument for doing the second half at all rather than leaving it — the first fold had already
been reviewed and measured, and the omission survived both.

### Note on the earlier rejected structure factor

An attempt to fold this sum into `|S(G)|²`, the way the *charge* path does, is recorded below as
rejected, and it stays rejected: a general coefficient matrix has no rank-1 factorization over
atoms, so the `N²` pair loop cannot be removed and the large-argument phase `G·(R_b − R_a)` loses
precision under trigonometric range reduction. The factorization above is over **lattice
translations**, a different index; its phases are rational multiples of `2π`, bounded and exactly
periodic, so neither objection applies.

### Scope

3-D only. The order reduction is in `C = n₁n₂n₃`, and only a three-dimensional mesh makes `C` large
enough for the square to hurt — a wire with a 20-point mesh has `C = 20`. The 1-D and 2-D virial
kernels also carry the separation *explicitly* (the `− dpar·G_i·d_j` term of the Parry kernel),
which shifts with the translation and would need three further transforms, for a case that does not
have the problem. Those dimensionalities keep the class loop.

`ewald_pair_matrix_with(.., reciprocal: false)` is what the class loop runs when the fold is active:
the real-space sum, whose distances genuinely change with the translation, plus the self and
background terms. Those are already linear in `C`, because a class whose translation carries every
pair past the real-space cutoff contributes nothing.

The **stress is nearly free given the gradient** — the two differ by 10–20 %, because the virial is
an outer product accumulated in the same pair loop rather than a separate calculation.

## Phonons

Diamond, force constants from the analytic Hessian of a supercell:

| supercell | atoms | force constants | in v0.1.2 | Γ optical mode |
|---|---|---|---|---|
| 1×1×1 | 2 | 0.033 s | 0.046 s | 1248 cm⁻¹ |
| 2×2×2 | 16 | 1.63 s | 3.04 s | 1318 cm⁻¹ |
| 3×3×3 | 54 | 18.3 s | 27.7 s | 1254 cm⁻¹ |

Cubic in the supercell, as a Hessian is. The measured Raman line is 1332 cm⁻¹; PM7 lands in the
1250–1320 cm⁻¹ range and the residual spread between supercells is the electronic k-sampling, not
the phonon machinery — a larger supercell samples the Brillouin zone more finely for the *density*
as well as resolving more force constants.

## Divide and conquer

The full table is in [`divide_and_conquer.md`](divide_and_conquer.md). The headline:

| | log–log slope |
|---|---|
| divide and conquer, ≥200 atoms | **0.99** |
| divide and conquer, per SCF iteration | **1.01** |
| exact SCF, same range | 2.19 |

Crossover near 350 atoms; below that the exact SCF is faster *and* exact.

Getting there took three fixes, each found by measuring rather than by reading the code:

1. The environment potential was double counting its electronic half once per subsystem — an `N²`
   error, 55 000 eV on a 240-atom chain.
2. The far field was recomputed inside the per-subsystem loop, and the buffer search scanned every
   atom. Hoisting the first and putting the second on a uniform grid took the slope from 1.22
   to 1.13.
3. The buffer had to go past 7 Å. Below that the accuracy falls off a cliff, because 7 Å is where
   PM7's feathering makes every excluded interaction *exactly* a monopole.

### What is still not linear

Stated rather than hidden:

* **The far field for a molecule** is a direct `O(N²)` charge sum, hoisted so it runs once per SCF
  iteration rather than once per subsystem. At 5 000 atoms that is 2.5e7 operations per iteration,
  which is not the bottleneck; past roughly 10⁴ atoms it becomes one. A Barnes–Hut or fast
  multipole treatment would remove it. For a periodic cell the Ewald sum already handles it better.
* **`dandc_derivatives` densifies the sparse density** to reuse the existing pair loops, which is
  `O(N²)` in memory. That bounds the size a gradient can be taken for, well below the size an
  energy can. Teaching the gradient loops to read pair blocks directly is the fix; they are already
  organized by pair, so it is mechanical rather than structural.
* **`PairList::build`** enumerates all pairs to apply its cutoff. It is `O(N²)` and shared with the
  ordinary SCF path.

## SCF convergence

Two changes to the SCF itself, both found by running molecular dynamics rather than single points —
a trajectory has to converge at *every* step, so it samples geometries a curated test set never
reaches.

* **CDIIS stagnation.** Once the error vectors turn linearly dependent, the bordered solve still
  succeeds but returns huge cancelling weights, and the run settles into a limit cycle around
  `1e-6` — converged for the energy, not converged by the density criterion. The `B` matrix is now
  scaled to a unit diagonal so its pivot guard is a relative test, and the oldest vectors are
  dropped when the weights blow up.
* **A level-shift controller** engages when a run has genuinely stopped moving. It watches the
  density step *and* the energy, because a hard molecular SCF can creep downhill for hundreds of
  iterations with a flat step size, and shifting that only slows it. `tests/scf_convergence.rs`
  pins both cases; a controller tuned on either alone breaks the other.

The SCF iteration count in the divide-and-conquer series is flat at 50–57 from 62 to 1922 atoms,
which is worth checking for separately: a linear cost per iteration is no use if the number of
iterations grows.

### The k-point SCF uses none of this, and the reason it should not simply inherit it

The molecular SCF runs A-DIIS → CDIIS on the `[F, P]` commutator. `run_kpoint_scf` runs neither: it
uses `DensityMixer`, Pulay/Anderson on the real-space density blocks.

CDIIS **would** generalize — in NDDO the basis is orthonormal, so the per-k error vector is just
`e(k) = [F(k), P(k)]` and the inner product is `Σ_k w_k Re Tr[e(k)† e(k')]`. It should beat
density-residual mixing near convergence, because it measures the stationarity condition rather
than self-consistency of the density map.

But it is not the binding constraint here. The failure `DensityMixer` was written for is recorded
in its own comment: silicon stalls at `3e-4` and *gets worse as the k mesh is refined*. Degrading
with refinement is the signature of **charge sloshing** — the long-wavelength, small-`G`
instability — and no DIIS variant addresses it, because it is a preconditioning problem. The
standard treatment is **Kerker** damping of the small-`G` residual components alongside
Pulay/Broyden.

So the identified next step was CDIIS for the tail and Kerker for the cause: the same two-stage
shape as A-DIIS → CDIIS, with the right early-stage tool for a solid. A-DIIS itself ranks lowest of
the three — it needs `E[P]`, which is available, but its job is escaping bad basins, and a gapped
insulator started from a sensible guess rarely needs that.

**Kerker was measured and then dropped, deliberately, in 0.2.3.** `grep -i kerker src/` still finds
nothing, and that is now a decision rather than a backlog item. The paragraph above diagnoses charge
sloshing, which is an *oscillation*; when the stalling cases were actually instrumented they turned
out to be **stiff** — the residual falling 0.993 to 0.9993 per step, monotonically, with no
oscillation at all — and the residual splits with the **charge channel best converged of the three**.
Kerker preconditions exactly that channel. Building it would have been shipping a knob aimed at the
wrong half of the problem in order to satisfy a plan written before the measurement. See
[the mixer bugs](#two-real-bugs-in-the-periodic-density-mixer) for what the measurement did find, and
`docs/scope.md` for the smearing fallback that handles the stiff case.

### 0.2.3: the periodic pair list stops being quadratic — above 32 atoms

`PairList::periodic` looped over every `A ≤ B` pair and, inside that, over every lattice image:
`O(N²·I)` distance evaluations for a list whose content is `O(N·I)`. It is reached from about seven
places per energy-and-gradient evaluation.

The rearrangement is one line of algebra. `|p_B + T − p_A| < r` is `|p_B − (p_A − T)| < r`, so one
spatial grid over the home-cell positions serves every translation — the translation moves the
query point, not the index. The grid is the one `dandc/partition.rs` already had, moved to
`src/spatial.rs` to serve both.

**It is a pessimization below about 32 atoms, and that is why both paths are kept.** Diamond
supercells at a 7 Å cutoff, calling the two constructors directly — `cargo test --release --lib --
--ignored --nocapture pair_list_crossover`:

| atoms | scan | grid | grid/scan |
|---|---|---|---|
| 2 | 0.159 ms | 1.294 ms | **8.16×** |
| 16 | 2.81 ms | 3.33 ms | 1.18× |
| 54 | 29.5 ms | 14.2 ms | 0.48× |
| 128 | 119 ms | 39.9 ms | 0.33× |
| 250 | 480 ms | 94.0 ms | 0.20× |

Eight times slower on a two-atom primitive cell — which is what most of this crate's own tests and a
great deal of ordinary use are. The reason is not subtle: the scan costs `N²/2` distance
evaluations per image and the grid costs `N` bucket queries, and a bucket query is worth ten or
twenty distance evaluations. Shipping the grid unconditionally would have been a regression sold as
an optimization. `PairList::periodic` therefore dispatches on the atom count at a threshold of 32,
which is where the crossover between 16 and 54 atoms puts it.

**That table had to be re-measured to say this.** The one it replaces came from
`examples/pairlist_scaling.rs`, which goes through `PairList::build` — the *dispatcher* — so below
32 atoms its "binned" column was the scan timed against a count-only reimplementation of itself and
carried no information about the grid whatsoever. It happened to point the same way, which is how it
survived. The example still earns its keep as the **user-facing** measurement, since dispatching is
what a caller actually gets:

| atoms | `PairList::build` | old nested scan | speedup |
|---|---|---|---|
| 2 | 0.139 ms | 0.128 ms | 0.93× |
| 16 | 2.51 ms | 3.16 ms | 1.26× |
| 54 | 10.4 ms | 27.9 ms | 2.67× |
| 128 | 25.9 ms | 91.5 ms | 3.54× |
| 250 | 61.6 ms | 347 ms | 5.64× |
| 432 | 103 ms | 911 ms | 8.84× |

Both paths produce **bit-identical** lists — same entries, same order, same last bit — and
`tests/pair_list.rs` checks that against a copy of the old enumeration on four cells, three cutoffs
and both sides of the threshold. The order is the part that had to be preserved deliberately:
every consumer sums over `pairs` in order, so a reordered list is a different floating-point
summation, and a "harmless speedup" would have moved every published number.

One thing this does **not** fix: the image range comes from `cell_diameter`, the longest diagonal,
so a slab or a wire with a large vacuum gap asks for far more images than it needs — a 100×
elongated cell asks for ~10⁵ images in its short directions. Both implementations then spend all
their time there and the pair list is not the bottleneck. That is a separate defect in the margin
heuristic and is recorded rather than fixed.

### 0.2.3: the SCF diagnosis above was aimed at the wrong channel

Measuring the residual per channel — charge, on-site, inter-cell — changed the conclusion. On the
cases that actually stall, the **charge** channel is the best converged of the three, so Kerker,
which preconditions exactly that channel, would not have helped them. And the stalling cases are
not oscillating at all: the residual falls monotonically at 0.993/step against a 1e-7 threshold.
That is a **stiff** iteration, not an unstable one, and the two are not fixed by the same thing.

What did fix them were two ordinary bugs in `DensityMixer`, both found by looking at why a
well-behaved history was being rejected:

* the Pulay Gram matrix was **unscaled**, so a relative pivot test was operating on entries around
  `1e-24` and behaving as an absolute one. A perfectly conditioned history was discarded for being
  *small*.
* one rejection dropped the mixer to plain damping **for the rest of the run**, so a single bad
  step at iteration 4 cost every iteration after it.

Iterations to convergence after the fix: diamond 11, NaCl 12, CaF₂ 12, SrTiO₃ 13, Zn(CN)₂ 14,
spinel 17.

Charge sloshing is real and was measured on NaCl — 1.1 electrons oscillating with the chemical
potential swinging 12 eV — but NaCl was never one of the failing cases. Kerker is **not
implemented, and will not be**: it is aimed at a case that converges rather than at one that does
not, and adding it to satisfy a plan written before the measurement would put a knob in the release
that the release's own data says points the wrong way. The stiff cases are handled by the
entropy-gated smearing fallback instead, which is measured: ZnS converges in 19 iterations at a
0.05 eV width and not at all without, across a 6.4 eV gap where a smearing cannot change the answer.

## v0.2.1: what changed, what was rejected, and what is still open

**Memory order, which is the one that was asked for.** `dandc_derivatives` reads the sparse
density directly instead of calling `to_dense()` first. That removed an `N_ao²` allocation — about
512 MB at two thousand atoms — from the one step *after* the linear-scaling SCF had finished, and
it put a quadratic memory bound on a method whose entire purpose is to avoid one. Every read the
pair loop makes is block-local, so nothing about it needed the dense form; the blocks are now
copied into small per-pair arrays, which is also better for cache than striding an `N_ao × N_ao`
matrix in the innermost of four nested loops. `tests/dandc.rs` pins the sparse result against the
densified one to `10⁻⁹` relative. Periodic divide and conquer would need a translation-resolved
sparse density, which does not exist, so it keeps the dense path.

**Parallelism, with the determinism discipline.** `scatter_density` now runs over a fixed partition
of subsystems with per-chunk `SparseDensity` accumulators folded back **in chunk order**, so the
summation order does not depend on `RAYON_NUM_THREADS`. Chunked rather than one accumulator per
subsystem, because a `SparseDensity` is an `O(N)` allocation and there is one subsystem per atom.
`ewald_matrix` is parallel by row, each row written by exactly one thread.

**A-DIIS allocation churn.** `adiis_extrapolate` built `2k` difference matrices per call — sixteen
at the default depth — dotted them and discarded them. The subtraction is now fused into the dot
product, which is the *same arithmetic in the same order* and therefore bit-identical. Expanding
`⟨A−A'|B−B'⟩` into four dot products would **not** be, because it reassociates the rounding, so it
is deliberately not done that way.

### A transcendental reduction that was implemented and then removed

`ewald_matrix`'s 3-D reciprocal block does `N²` cosines per `G`. The structure factor separates —
`cos(G·(R_b − R_a)) = cos(G·R_a)cos(G·R_b) + sin(G·R_a)sin(G·R_b)` — which makes it **N**
transcendentals per `G` and two multiply-adds in the inner loop. It was written, and then reverted.

It is faster and **less accurate**. `G·R_a` grows with `|G|` and with how far the atom sits from
the coordinate origin, so each `sin`/`cos` pays a large-argument reduction; `G·(R_b − R_a)` is
bounded by `|G|` times the cell diameter wherever the cell happens to be placed. The lost digits
are small, but they land in the matrix every periodic SCF iterates against, and they were enough
to stop a rattled Γ-point cell converging inside its iteration budget —
`tests/scf_convergence.rs` failed at `1.1e-7` after 200 iterations. Fewer transcendentals is not
worth an SCF that does not converge, and the algebraic identity being exact on paper is not the
same as the two expressions being interchangeable in floating point.

### Still open, and why

* **The molecular point-charge collapse.** Past the 7 Å feather range a two-centre block is its
  point-charge monopole, and the periodic path already exploits that (`hamiltonian.rs`). Doing the
  same for molecules would turn `n_a²n_b²` block work into one scalar for most pairs of a large
  molecule. It is **not implemented**, for two reasons that are worth stating rather than
  gesturing at. First, it is not exact: the branch also drops the resonance `β·S` term, which is
  about `10⁻¹¹` eV at 7 Å — negligible, but not zero, so enabling it by default would move every
  published molecular number on a premise that does not hold. Second, the collapse must be applied
  **identically** in the core build, the closed-shell and unrestricted gradients, and the molecular
  Hessian, or the energy and its derivatives are computed from different Hamiltonians; the
  molecular Hessian has no such branch today. It belongs behind an opt-in flag with its residual
  measured, not switched on quietly. `examples/collapse_reach.rs` prints how much of a molecule's
  pair work is actually past the feather range, which is the number that decides whether three new
  implementations are worth writing.
* ~~**`PairList` caching.**~~ **Done**, and this entry was stale when it was written:
  `PairList::cached` (`src/pbc/images.rs:156`) is a thread-local four-slot LRU keyed on the exact
  bit pattern of every position, and every caller in `src/` goes through it — nineteen sites across
  seven modules; the only remaining `PairList::build` calls are in its own unit tests. The cost of
  *building* one is done too as of 0.2.3 — see the spatial-binning section above, including why the
  old scan is still there below 32 atoms.
* **A DIIS ring buffer.** `Vec::remove(0)` on the history shifts a handful of `Matrix` *headers*,
  not their data, so the win is smaller than the entry in the plan implies. The allocation the
  history genuinely costs is the clones themselves, which are inherent to keeping a history.

## Measuring it yourself

```bash
cargo test --release --test perf_report -- --nocapture --ignored --test-threads=1
cargo test --release --test dandc_scaling -- --nocapture --ignored
cargo bench --bench scaling                 # criterion, with statistics
PM7_PROFILE=1 cargo test --release --test perf_report hessian_profile -- --ignored --nocapture
```

`PM7_PROFILE` reports **thread-seconds**, so a stage on 16 cores shows about 16× its elapsed time.
That is the right unit for finding where the work is and the wrong one for predicting speedup; the
report says so in its own header.

The criterion benches (`benches/scaling.rs`) cover the hot operations rather than whole workflows,
because criterion's value is its statistics and a 30-second workflow gets three samples.

## What was measured and left alone

Recording the things that did *not* need work is as useful as recording the things that did — it is
what stops the next person re-deriving them.

* **Parameter loading.** `Pm7Parameters::method` now caches per method, because a molecular-dynamics
  step is a *water* SCF and a thousand rows of CSV is not negligible next to one.
  `Pm7Parameters::shared` hands back the cached copy without even cloning.
* **`PairTwoElecG::w` flattening** was already done in v0.1.2 — a `Vec<Vec<S>>` costs one allocation
  per bra pair and a pointer chase per read.
* **The `G:U` GEMM**, planned for this release, is 0.1 % of a Hessian. Not done, and the profile
  above is why.
* **The `rotfix` frame-singularity fallback** costs nothing measurable: it triggers only inside a
  0.8° window around each singular axis. Its exact replacement (Wigner-matrix conjugation) is
  described in [singularities.md](singularities.md) and was not implemented for the same reason.
## Regression gates

Wall-clock assertions are unreliable in CI, so the gates assert on arithmetic instead. In
`tests/dandc.rs`:

| quantity | gate |
|---|---|
| diagonalization work `Σ n³` | slope ≤ 1.2 |
| Fock work `Σ n²` | slope ≤ 1.2 |
| stored density elements | slope ≤ 1.2 |

These are deterministic and mean the same thing on any machine. The timing tests report and assert
nothing.
