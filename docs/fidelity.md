<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# PM7 fidelity audit

This document records the line-by-line audit of `pm7-rs` against the pinned **MOPAC v23.2.5**
source (`.mopac-source/`, Apache-2.0). Every entry names the MOPAC file and line range that the
Rust code was checked against, so the claim can be re-verified rather than trusted.

Status legend:

| status | meaning |
|---|---|
| **match** | the Rust expression is algebraically identical to MOPAC's, including branch conditions |
| **match (bounded)** | identical except for a difference that can only change the result on a measure-zero set of inputs, stated explicitly |
| **residual** | a numeric difference remains; the size and the isolated cause are recorded |
| **open** | not yet audited at this level of detail |

---

## Core–core repulsion (`ccrep`)

Rust: [`src/repulsion.rs`](../src/repulsion.rs) `pair_core_energy_scalar`
MOPAC: `.mopac-source/src/integrals/ccrep.F90:16-262`

| term | MOPAC | status |
|---|---|---|
| monopole `E = Z_i Z_j · gab` with feathered `gab` | `ccrep.F90:48` (`gab` passed in from `mndod`/`reppd`) | match |
| defined-pair scale `1 + 2·xfac·exp(−alpb·(r + 3e-4 r⁶))` | `ccrep.F90:80` | match |
| `alpb < 1e-6 → 1.2` | `ccrep.F90:75` | match |
| H–C / H–N: `1 + 2·xfac·exp(−alpb·r²)` | `ccrep.F90:93` | match |
| H–O: `… − par3·exp(−2·par4·r)` | `ccrep.F90:98` | match — MOPAC writes `exp(-par4*r*2)`, i.e. `exp(−2·par4·r)`, and the Rust `(-2.0*vpar(4)*r).exp()*vpar(3)` is the same |
| C–C: `+ par1·exp(−par2·r)` | `ccrep.F90:104` | match |
| O–Si: `− 7e-4·exp(−(r − 2.9)²)` | `ccrep.F90:116` | match |
| H–H, N–N: **no** special case (fall through to the general form) | `ccrep.F90:91,107-110` (empty `case` bodies) | match — the Rust `_ => {}` arm reproduces this |
| **the special-case set is exactly** `{(1,6), (1,7), (1,8), (6,6), (8,14)}` for PM7 | `ccrep.F90:86-120` | **match — confirmed complete**; no PM7 branch is missing |
| undefined pair: `scale = 10·exp(−2.18 r)`, or `−3.0` for Z 57–71 | `ccrep.F90:156-162` | match — MOPAC's `ni > 56 .and. ni < 72` is Z 57–71, and `enuclr = abs(scale·enuc) + enuc` equals `enuc·(1+scale)` because `enuc > 0` for positive cores and positive `gab` |
| "probable dead code" `nt = ni+nj ∈ {8,9}` | `ccrep.F90:173-179` | match — for PM6/PM7 `eni = enj = 0` (`ccrep.F90:164-165`), so the term is identically zero; the Rust code correctly omits it |
| missing-pair completion `xfac = ½(xfac_ii + xfac_jj)`, `alpb = ½(alpb_ii + alpb_jj)` | `ccrep.F90:56-69` | match — `Pm7Parameters::pair` ([`src/params.rs:149`](../src/params.rs)) does the same average. MOPAC applies it only when `\|xfac(ni,nj)\| < 1e-5`; **no shipped PM7 pair row has `\|xfac\| < 1e-5`** (verified: 0 of 1028 rows in `src/data/pm7_pairs.csv`), so returning the stored entry and testing it afterwards is equivalent |
| core Gaussians: slot 1 always added for **both** atoms, plus the full `ig = 1..4` loop **only** when the pair is undefined | `ccrep.F90:186-192` (always) and `ccrep.F90:193-196, 231-243` (`i = 0` if `abond > 1e-4`, else `i = 4`) | **match — the apparent double-count of slot 0 is what MOPAC does.** MOPAC adds the "VdW term" (slot 1) unconditionally and then, for an undefined pair (`abond = 0`), runs `ig = 1..4` which re-adds slot 1. `pair_core_energy_scalar` reproduces this exactly. The gate is equivalent: a defined pair always has `abond ≥ 1.2` after the `< 1e-6 → 1.2` clamp |
| unpolarizable-core `1e-8/ax¹²` guard, `ax = r/(Z_i^⅓ + Z_j^⅓) < 3`, capped at `1e5` | `ccrep.F90:252-260` | match |
| Gaussian exponent cutoff | `ccrep.F90:189` uses `ax < 25`, `ccrep.F90:234,239` use `ax <= 25` | **match (bounded)** — `core_gaussian_slot` uses `ax < 25` everywhere. The two differ only at exactly `ax == 25.0`, where the term is `e⁻²⁵ ≈ 1.4e-11` times a prefactor. Not reachable in practice from double-precision geometry. |

**Conclusion:** `ccrep` is a faithful port. The two suspicions raised while reading the Rust code
(an incomplete special-case set, and a double-counted Gaussian slot) are both **false alarms** —
MOPAC does exactly what the Rust code does.

---

## One-centre two-electron Fock contribution

Rust: [`src/fock.rs`](../src/fock.rs) `oc_two_electron` and the `n == 9` branch of `build_fock_spin_x`
MOPAC: `.mopac-source/src/SCF/fock1.F90:16-89` and `wstore` in `.mopac-source/src/integrals/mndod.F90`

MOPAC builds a packed one-centre integral matrix `w(ilim, ilim)` in `wstore` and contracts it in
`fock1` as `F_ij += Σ_kl [ P_tot(kl)·w(ij,kl) − P_α(kl)·w(ik,jl) ]`. The Rust code contracts
`p_tot·g(μν,λσ) − p_spin·g(μλ,νσ)` — the same J and K index patterns.

`wstore` sets, in the lower-triangle pair order `1=(s,s), 2=(pₓ,s), 3=(pₓ,pₓ), 4=(p_y,s),
5=(p_y,pₓ), 6=(p_y,p_y), 7=(p_z,s), 8=(p_z,pₓ), 9=(p_z,p_y), 10=(p_z,p_z)`:

| `wstore` entry | value | `oc_two_electron` branch | status |
|---|---|---|---|
| `w(1,1)` | `gss` | `(a==b, c==d, both s)` | match |
| `w(3,1) w(6,1) w(10,1)` + transposes | `gsp` | `(a==b, c==d, one s)` | match |
| `w(3,3) w(6,6) w(10,10)` | `gpp` | `(a==b, c==d, a==c, both p)` | match |
| `w(6,3) w(10,3) w(10,6)` + transposes | `gp2` | `(a==b, c==d, a!=c, both p)` | match |
| `w(2,2) w(4,4) w(7,7)` | `hsp` | `(s pᵢ \| s pᵢ)` | match |
| `w(5,5) w(8,8) w(9,9)` | `½(gpp − gp2)` | `(pᵢ pⱼ \| pᵢ pⱼ), i ≠ j` | match |
| every other entry | `0` (the matrix is zeroed first, `mndod.F90` `wstore`) | falls through to `0.0` | match |
| `ilim > 10` d block, filled from `repd` via `intij/intkl/intrep` | | `elem.dshell.onecenter` (45×45) | match in structure; the numeric `repd` table is covered by the residual entry below |
| H (`natorb == 1`): only `w(1,1) = gss` | `wstore` guards the p/d fill on `natorb(ni) > 2` | Rust `n == 1` reaches only `(0,0,0,0)` | match |

**Conclusion:** the zeros returned by `oc_two_electron` are genuine MOPAC zeros. The second
suspicion (that MOPAC might have non-zero integrals where the Rust returns 0) is a **false alarm**.

---

## SCF initial density guess

Rust: [`src/scf.rs`](../src/scf.rs) `sad_density`
MOPAC: `.mopac-source/src/moldat.F90:731-789`

| item | MOPAC | status |
|---|---|---|
| `yy = charge / (norbs + 1e-10)` | `moldat.F90:731` | match — the Rust `(total_tore − n_electrons)/nao` is the same quantity |
| sparkle (0 AOs) skipped | `moldat.F90:736` (`nlast − nfirst == -1` → cycle) | match |
| H: `P = tore − yy` | `moldat.F90:742` | match |
| sp: `P = tore/4 − yy` on all four AOs | `moldat.F90:747-748` | match |
| main-group d test `Z<21 ∨ 30<Z<39 ∨ 48<Z<57` | `moldat.F90:751` | match |
| main-group spd: sp gets `tore/4 − yy`, d gets `−yy` | `moldat.F90:757-760` | match |
| transition metal: `sum = tore − 9yy`; s ← `clamp(sum,0,2)`; `sum −= 2`; then 5 d ← `clamp(0.2·sum,0,2)`; `sum −= 10`; then 3 p ← `sum/3` | `moldat.F90:765-786` | match — identical order, identical clamps |

**Conclusion:** the MOPAC starting point is reproduced exactly, which is what makes the
bistable diatomics (BF, AsF, AlN) land in MOPAC's SCF basin.

---

## Point-charge feathering (`l_feather`)

Rust: [`src/integrals.rs`](../src/integrals.rs) `feather_to_point`, `feather_ri`
MOPAC: `to_point` in `.mopac-source/src/integrals/mndod.F90`, `nddo_to_point` in
`.mopac-source/src/integrals/solrot.F90:108-138`

Status: **match** (already validated bit-wise by `tests/molecules.rs::pm7_feathering_matches_mopac_bitwise`).

Audit note for the periodic work: in a solid-state run MOPAC composes feathering with a *second*
mechanism — `solrot` switches to a pure `point()` beyond `clower` (13 Å default) and `trunk(r)`
compresses distances toward `cutofp` (30 Å default for solids). The two must not be conflated;
see [pbc.md](pbc.md).

---

## What the hundred-and-seventy-six-case oracle changed

`tools/oracle/baseline.py` carried **thirteen** cases through 0.2.3. It now carries **176**, covering
**all 73 elements PM7 is parameterized for** — a contract, not a comment: `validate_coverage` reads
`src/data/pm7_elements.csv`, so an element gains a parameter row and loses an oracle case only by
failing the run. It also spans five multiplicities (to a quintet), charges −2 to +2, sixteen
organometallics, and a geometry validator that refuses to run a case whose atoms overlap or float
free — which caught three malformed structures in its own first draft.

Expanding it found two missing corrections and fixed them (see the changelog). What it also did was
**measure the transition-metal residual properly for the first time**, and the picture is more
structured than the single row below used to suggest. Of **1086** comparisons, 29 breach, and they
fall into two kinds that want different responses:

| case | ΔHf gap | charges agree? | reading |
|---|---|---|---|
| `copper_chloride` | +5.18 | no (0.019 e) | MNDO/d value residual — **not** a basin, see below |
| `silver_chloride` | +3.82 | no (0.012 e) | MNDO/d value residual — **not** a basin, see below |
| `titanocene_dichloride` | −0.79 | no (0.004 e) | different SCF solution, **`pm7-rs`'s is lower** |
| `iridium_dicarbonyl_dichloride` | −0.19 | no (0.002 e) | different SCF solution, **`pm7-rs`'s is lower** |
| `titanium_tetrachloride` | −0.357 | yes | same state: the MNDO/d value residual |
| `methylmagnesium_chloride` | +0.039 | yes | same state |
| `cobalt_tetracarbonyl_anion` | +0.033 | yes | same state |
| `perrhenate` | −0.025 | yes | same state |
| `formic_acid_dimer` | +0.024 | yes | same state |
| `lutetium_trifluoride` | −0.012 | yes | same state |
| `silanol` | +0.00044 | yes | same state |

**The Mulliken charges are what separate the two.** Matching charges mean both codes are on the same
electronic state, so the gap is a value — an integral, or a term one side has and the other does not.
That is the class the C–C triple-bond and Si–O–H corrections turned out to be, and it is the class
worth chasing. Differing charges mean different basins, and then the only question is which is lower.

`pm7-rs` is **lower** on titanocene dichloride and the iridium carbonyl, which is a variational
method doing its job and not a defect.

### CuCl and AgCl are not a basin problem, and finding that out took a stability analysis

They looked like one — 5.2 and 3.8 kcal/mol above MOPAC with charges differing by 0.019 and
0.012 e, which is the signature of a different SCF solution. Three measurements say otherwise:

* **thirty-six** combinations of level shift (0–20 eV), spin reference and accelerator all converge
  to the same 13.39233 (`examples/cucl_basin.rs`);
* **forty-eight** perturbed starting densities do too, none finding anything lower
  (`examples/cucl_search.rs`);
* the **stability analysis** puts the lowest orbital-Hessian eigenvalue at **+6.05 eV**, so the
  solution is a genuine local minimum rather than a saddle (`src/stability.rs`).

A second basin that no perturbation, no level shift and no negative curvature can reach is not a
second basin. Both codes also start from the *same* guess — `sad_density` is MOPAC's own diagonal
guess, `moldat.F90:731-790` — and their frontier orbitals differ by only 0.055 and 0.126 eV, which
is far too little for qualitatively different solutions.

What is left is a small difference in the Hamiltonian itself, shifting the converged density a
little and the energy by more. **MNDO/d is implemented** — `src/mndod.rs`, `mndod_tables.rs` and
`mndod_twocenter.rs`, and every element with a `d` shell goes through it, which is why H₂S, SF₆,
PF₃, SiF₄ and the whole halide series match to `1e-6` kcal/mol. So this is a *value* residual inside
that kernel and not a missing one, which puts it in the same row as `TiF₄`/`TiCl₄`.

**That placement is an inference, not a localization.** For titanium the row below records positive
evidence — matching Mulliken charges at the same SCF solution, pointing at the two-centre `w` and
one-centre `repd` values. For copper and silver the basin explanation has been ruled out and no
alternative has been ruled *in*: the element-by-element `w`/`repd` comparison against MOPAC that the
row calls for has still not been done, and until it is, "the same residual, larger for the late
transition metals" is the most that should be claimed.

**The stability analysis earned its place by ruling a hypothesis out.** It is in the crate as
`Pm7Options::stability`, off by default because the check costs about a CPHF solve; `Check` reports
the curvature and `Follow` re-converges from a rotated guess and keeps the lower of the two. Water,
methane and benzene report `+7.6`, `+10.8` and positive curvature and are left untouched, which is
the property that makes it safe to switch on.

`titanium_tetrachloride`'s −0.357 lands exactly on the ≤ 0.36 kcal/mol bound the row below records
for four-coordinate d-populated species, which is the first independent confirmation of that number
since it was written. `titanocene_dichloride`'s 0.79 is **outside** it, so the bound as stated is
too tight — but that case is a basin difference rather than a value one, so the two are not
comparable and the row stands.

---

## Known residual

| item | size | isolated cause | status |
|---|---|---|---|
| Heat of formation for four-coordinate, highly d-populated transition-metal species (e.g. TiF₄) | ≤ 0.36 kcal/mol | Mulliken charges match MOPAC at the same SCF solution, so this is a value difference in the MNDO/d two-centre `w` / one-centre `repd` integrals, not an SCF-basin or scaling error | **residual** — element-by-element `w`/`repd` comparison against MOPAC pending |
| Diatomic hydrides and fluorides of the **noble gases** and of **open-shell transition metals** | up to 90 kcal/mol, 22 of 125 pairs | different SCF solution, not a different integral: the sweep assigns multiplicity by electron parity alone (`2` if odd, else `1`), so Co(d⁷)H is asked for as a singlet and the two codes settle in different basins. `pm7-rs` finds the lower solution in some (XeH, CoF, NeH) and the higher in others (HeF, HeH) | **residual** — needs a multiplicity scan per pair before it means anything about the integrals |

| Bare monatomic **anions**' orbital energies (`F⁻`, `Cl⁻`) | 5–9 eV on individual levels | an isolated atom has no two-centre terms, so its orbital energies come entirely from one-centre quantities that the *energy* combines differently from the way the eigenvalues do; the heat of formation is exact to MOPAC's 5 dp and every one of the 175 polyatomic cases agrees to better than `1e-3` eV | **specified behaviour** — a bare monatomic ion's reported eigenvalues are not claimed to match MOPAC's, and nothing with a bond is affected |
| `CuCl`, `AgCl` heats of formation | 5.2 and 3.8 kcal/mol | **not** an SCF basin — ruled out by 36 solver settings, 48 perturbed starting densities and a `+6.05 eV` stability eigenvalue. MNDO/d is implemented and exact for the main-group `d` elements, so this is a value residual inside it rather than a missing kernel; which integral has not been established | **residual** — needs the same element-by-element `w`/`repd` comparison as the row above |

The second row is **newly measured**, not newly caused: `pair_sweep.py` had never executed (see
the changelog), so nothing in `pm7-rs` ≤ 0.2.0 was ever compared against MOPAC across the periodic
table. The v0.2.1 fixes cannot have caused it — they touch the periodic Bloch block, the periodic
dipole report and the DFPT field response, and a molecular diatomic reaches none of the three.

The **molecular** oracle set (`spmols`, `dmols`, `scf_residual`, `hbond_check`, and `baseline.py`'s
176 cases) is exact to MOPAC's printed precision throughout, with the exceptions tabulated above:
1057 of 1086 comparisons pass at `1e-4` kcal/mol on the heat of formation, `2e-3` eV on the orbital
energies, `2e-3` D on the dipole, `1e-4` e on every Mulliken charge and `1e-5` on `⟨S²⟩`.

---

## `eps^inf` is systematically too small, and that is the model

| system | `pm7-rs` | experiment |
|---|---|---|
| diamond | 1.12 | 5.7 |
| LiF | 1.01 | 1.92 |

PM7's valence basis is minimal: one s and three p functions per heavy atom, no polarization
functions. The electronic dielectric response is built entirely from transitions inside that
space, and the `d`-like character that a real crystal polarizes into is not in it. So `eps^inf`
comes out low by a factor of two to five, and no amount of k-point refinement changes that — the
values above are converged to four figures by a `5x5x5` mesh.

**Born effective charges do not share the problem.** They are a charge-transfer quantity rather
than a polarization one, and LiF's `Z* = +1.033` sits right on the measured ~1.04. That asymmetry
is worth stating plainly: `Z*` from this code is quantitative, `eps^inf` is qualitative, and the
LO–TO splitting built from both (convention C-7) inherits the weaker of the two.

---

## MOPAC's printed vibrational dipoles are half the derivative

`FORCE LARGE` prints `DIPX`/`DIPY`/`DIPZ`/`DIPT` per mode, built from `deldip = ∂mu/∂x`. That
derivative is formed in `src/forces/fmat.F90:197-252`:

```text
xparam(i) += 0.5*delta      →  dipole → deldip(:,i)
xparam(i) -= delta          →  dipole → del2          (so the two points are ±delta/2)
xparam(i) += 0.5*delta      →  back to the origin
deldip = (deldip - del2) * 0.5 / delta
```

The two evaluations are separated by `delta`, so the central difference is
`(mu(+d/2) − mu(−d/2)) / delta`. MOPAC divides by `2*delta` instead, making the printed quantity
**half** the true dipole derivative.

Found empirically before being found in the source: water's three vibrations gave ratios of
**1.995, 2.002 and 1.995** against this implementation's analytic derivative. `IrSpectrum::mopac_dipt`
reproduces the factor deliberately, so a `FORCE LARGE` run can be compared number for number
(`tests/ir.rs::the_vibrational_dipoles_match_mopac_force_large`, agreement better than 0.01 D/Å);
`IrSpectrum::intensities_km_per_mol` is the physical quantity and carries no such factor.

`deldip` feeds only the printed table and `anavib`, so nothing else in MOPAC is affected.

| status | scope |
|---|---|
| **residual (MOPAC)** | the printed `DIPT`/`DIPX`/`DIPY`/`DIPZ` table only |

---

## MOPAC's own SCF is the looser side on a large system

Worth stating because it changes how the oracle has to be run. The heat of formation is
variational and so is second-order insensitive to a density error; the Mulliken charges are not,
and move at first order. On the 62-atom `alkane_c20` case of `tools/oracle/baseline.py`, with
`pm7-rs` converged to `1e-12`:

| MOPAC keywords | worst Mulliken charge difference | heat-of-formation difference |
|---|---|---|
| `PM7 1SCF PRECISE` | 2.23e-4 e | −6.6e-6 kcal/mol |
| `… RELSCF=0.01` | 2.39e-5 e | +3.4e-6 kcal/mol |
| `… RELSCF=0.0001` | **1.89e-6 e** | +3.4e-6 kcal/mol |

The residual falls to MOPAC's six-decimal print precision and the heat of formation stops moving,
so the 2.2e-4 seen at plain `PRECISE` was measuring **MOPAC's convergence, not this
implementation's fidelity**. The oracle therefore runs with `RELSCF=0.0001`; without it a charge
threshold tight enough to be useful would fail on MOPAC's own noise.

---

## Audit coverage

| area | status |
|---|---|
| `ccrep` core–core repulsion | audited — match |
| one-centre two-electron Fock (`fock1` + `wstore`) | audited — match |
| SCF initial guess (`moldat`) | audited — match |
| point-charge feathering | audited — match (bit-wise test exists) |
| MNDO/d two-centre `w` / one-centre `repd` numeric values | **open** (see residual) |
| sp two-centre `rotate`/`reppd` | **open** at the line level; covered numerically by the MOPAC oracle suite |
| dispersion / EH+ / PM7-HH corrections | **open** at the line level; covered numerically by the oracle suite |
| solid-state image sums (`hcore` `id ≠ 0`, `solrot`, `trunk`) | audited for the v0.2.0 periodic port — see [pbc.md](pbc.md) |
| `⟨S²⟩` (`(S**2)`) on all 20 open-shell cases | audited — match to 1e-6, and the oracle now compares it |

### `⟨S²⟩` joined the oracle in 0.2.4

Twenty of the 176 cases are open-shell — ten doublets, seven triplets, two quartets and a
quintet — and MOPAC prints `(S**2)` for every one. All twenty agree to **1e-6 or better**, the
worst being the quintet C₂ at `6.013492` against `6.013491`. The threshold is `1e-5`, set by
MOPAC's six-decimal print rather than by any expected difference.

It also does one thing the other columns cannot: it catches a **disagreement about the spin path**.
Both codes decide RHF-or-UHF from the multiplicity and the shell, so a case where MOPAC prints
`(S**2)` and `pm7-rs` does not — or the reverse — is not a small numerical gap, it is the two codes
running different methods on the same input. That is reported as a failure rather than a skip, with
no tolerance to hide behind. Zero cases disagree today.
