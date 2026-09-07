# Singularities

PM7 is smooth almost everywhere and not smooth in a handful of specific places. Some of those are
properties of the **published model**, which MOPAC shares and no implementation can remove without
changing the answer. Others are properties of the **formula chosen to evaluate it** — the function
is perfectly smooth, the parameterization is not, and the arithmetic goes through `0/0` on the way
to a limit that exists.

Only the second kind can be removed. Telling them apart takes a measurement, not an argument: it is
easy to look at `acos` followed by `cos` and conclude it must be catastrophic, and be wrong.

## The test that separates them

Sweep the geometry towards the singular point and watch the analytic derivative against a central
difference of the **energy**, which knows nothing about how the derivative is computed. Three
outcomes, three different conclusions:

| what you see | what it means |
|---|---|
| both stay finite and agree | nothing is wrong; the limit exists and both routes find it |
| the finite difference stays finite, the analytic one degrades | a parameterization artefact — removable |
| **both** blow up together | the model itself has no derivative there — not removable |

The third is the one that matters, and it is the one that is easy to miss, because an implementation
that faithfully reproduces a divergence looks exactly like a broken implementation until you check
the energy.

`tests/hbond_singularity.rs` runs this sweep.

## Measured: the EH+ correction

Two of its angles can go singular, and they behave completely differently.

### D–H···A → 180°: nothing wrong

This is the *ideal* hydrogen bond, and `bangle(..).cos()` looked like a textbook removable
singularity: `d(acos)/dt = −1/√(1−t²)` diverges exactly where `d(cos)/dθ = −sin θ` vanishes, so the
chain rule evaluates `0 × ∞`.

Measured, over four decades of tilt down to `1e-10` rad, the error against a finite difference of
the energy is **flat at 5.5e-9 relative** — pure finite-difference truncation. There is no
degradation to fix.

The reason is worth knowing, because it generalizes: `sin(acos t)` and `√(1−t²)` are the same
quantity computed two ways, and both inherit the *same* relative error from `t`. In the ratio the
errors cancel. A `0 × ∞` written as a product of two correlated quantities is often benign.

`cos θ` is still computed directly now (`bangle_cos_g`, with `bangle_sin_g` beside it) — it is
exact, cheaper, and immune to how `clamp2` and `acos` interact at the endpoint. Removing the round
trip left `bangle_g` with no callers at all, and it was **deleted**; the name survives only in a
comment in `src/hbond.rs` recording why. That says something about the original: *every* use of the
angle was `bangle(..).cos()`. The angle was never needed.

### R–X···H → 180°: the model diverges

This is the acceptor-side reference angle, and it is a different story. Rotating one acceptor O–H
until it is anti-parallel to the O···H vector:

| tilt (rad) | E (kcal/mol) | largest ∂E/∂x (kcal/mol/Bohr) | analytic vs FD |
|---|---|---|---|
| 1e-1 | −1.329197 | 5.9 | 4.8e-9 |
| 1e-2 | −1.153574 | 50.2 | 7.4e-7 |
| 1e-3 | −1.134236 | 492 | 7.7e-5 |
| 1e-4 | −1.132284 | 4874 | 7.7e-3 |
| 1e-5 | −1.132089 | 27400 | 0.79 |
| 1e-6 | −1.132069 | 9208 | 53 |
| 1e-7 | **−0.873427** | 933 | 1.0 |

Two separate things are happening.

**The gradient genuinely diverges, like `1/ρ`.** The energy converges to a limit along the sweep
(`ΔE ∝ tilt`), but the *out-of-plane* component blows up — the largest entry is the acceptor
oxygen's `z`, the azimuthal direction. That is the **dihedral**. `dihed(r2, r1, X, H)` measures a
rotation about the `r1–X` axis, and when H lies on that axis the rotation is undefined: moving H
sideways by `ε` swings the torsion by O(1), so `∂τ/∂x ~ 1/ρ`. The EH+ energy depends on
`cos(shift − τ)` and nothing in the expression vanishes as `ρ → 0` to damp it.

So the energy is bounded but not Lipschitz, and the analytic gradient is *correct* — it reproduces
the model's own divergence. The growing analytic-vs-FD column is the finite difference failing, not
the analytic path: FD error goes as `h²E'''`, and `E'''` diverges faster than `E'`.

At half a degree off the axis the EH+ force is already 50 kcal/mol/Bohr against about 3 for a normal
hydrogen bond — a 15× inflation. This is reachable geometry, not a pathological limit. It is the
mechanism behind the note that a symmetric water crystal "sat exactly on a point where the analytic
gradient diverges".

**And below `1e-6` the energy jumps by 0.26 kcal/mol.** That is MOPAC's `dihed` degenerate branch:
`yxdist ≤ 1e-6` switches to a different formula (`go to 10` in the Fortran). An unannounced
discontinuity in the potential energy surface, present in MOPAC and reproduced here for fidelity.

**Neither is removable by expansion.** A Taylor series cannot assign a value to a rotation angle
about an axis the atom is sitting on; every value is equally valid, and the energy really does have
different limits from different directions. This is a defect in the *functional form* of PM7's EH+
term, not in any implementation of it.

What a well-posed torsional term does — and this one does not — is weight the dihedral by something
that vanishes where the dihedral becomes undefined. The repair is correspondingly small: multiply
`torsion_cos` by `w = ρ²/(ρ² + ρ₀²)`, where `ρ` is H's perpendicular distance from the `r1–X` axis.
Then `w·∂τ/∂x ~ ρ/ρ₀²` is bounded and tends to zero, the second derivative is bounded too, and away
from the axis (`ρ ≫ ρ₀`) the change is `O(ρ₀²/ρ²)` — nothing. With `ρ₀ ≈ 0.05 Å` it acts only within
about 3° of the axis, a region where the current model is meaningless anyway.

That is a **model change**, so it is documented here rather than applied. It is not wired to an
option yet.

## Removable — fixed exactly

### The zero wavevector of a phased Ewald sum

`Φ_q(d) = Σ'_T e^{iq·T} v(d+T)` is periodic in `q` under the reciprocal lattice, so `q = G` has to
give exactly the `q = 0` answer. The reciprocal sum drops `|G + q| ≈ 0` by construction and special
terms stand in for it — the 2-D sheet term, the 1-D log term, the 3-D neutralizing background.
Keying those off `q == 0` rather than off `q ∈ reciprocal lattice` drops them silently whenever `q`
is a nonzero lattice vector: the sum stays finite and merely comes out short.

That is the common case, not a curiosity. The long-range exchange sums a **supercell** lattice at a
`q` commensurate with the k mesh, and every such `q` is a supercell reciprocal lattice vector. The
symptom was a rank-one transverse error of about 2.7 eV/Bohr² in `D(½,0,0)` for a 1-D chain whose
true transverse force constants are 0.09 — invisible at `q = 0`, and scaling as `1/N_k`, which is
what a `1/Ω_supercell` reciprocal-space term does. It made the arbitrary-`q` DFPT disagree with the
supercell reference by 5 % while every component of it tested clean.

**Fix:** `is_reciprocal_lattice_vector`, with the phase set to exactly 1 there so the reduction is
bit-exact rather than good to 1e-16 per image.
`the_phased_sum_at_a_reciprocal_lattice_vector_is_the_gamma_sum` checks value, gradient and Hessian
in 1-D, 2-D and 3-D.

### The 1-D Ewald log kernel at zero transverse offset

`(−1 + e^{−α²u})/u` at `u → 0` is `0/0` with limit `−α²`. It had been written as `−α²/2`. Invisible
in the gradient, where it is multiplied by the vanishing perpendicular offset; the Hessian's
transverse term picks it up undamped.

This is the one entry where a Taylor expansion *is* the answer, because there is no algebraic
identity available — the function is defined by its series at the origin.

### The two-centre local-frame rotation

The MNDO/d rotations are singular for a bond along ±z, and the sp path has **two** singular
directions, because its two-electron rotation and its overlap rotation feed opposite arguments to
the same routine: `+x` for one, `−x` for the other. The integral *values* stay correct (the f64
special case is a valid rotation); the forward-mode derivative collapses to the constant branch and
loses the perpendicular component.

**Handled, not removed.** `rotfix` takes such a pair's derivatives by central finite differences of
the f64 integrals, inside a window of about 0.8° around each singular axis — exact to about `1e-6`
relative, well under the SCF's own convergence noise, and rare enough to cost nothing.

The exact fix exists: the integrals are tensors, so one can evaluate at `R d` for a fixed rotation
`R` that moves the bond off the singular axis and transform back with the s/p/d Wigner matrices of
`R`. It was not done, because the measured error of the present treatment is already below every
other error in the calculation.
`every_axis_direction_is_guarded_against_the_frame_singularity` sweeps all six axis directions on
both paths; guarding only `+x`, as v0.1.2 did, cost 0.33 eV/Bohr on a `−x`-aligned bond.

## Not removable — the model's own

| Where | What | Consequence |
|---|---|---|
| EH+ torsion at a collinear `R–X···H` | dihedral undefined on the axis | **Gradient diverges as `1/ρ`** |
| `dihed`'s degenerate branch | MOPAC switches formula at `yxdist ≤ 1e-6` | 0.26 kcal/mol jump in the energy |
| EH+ angle term, `cos(shift − θ)`, `shift ∉ πZ` | `θ` has a cone point at collinear | Gradient bounded, direction-discontinuous |
| EH+ `max(angle2_cos, angle2_cos_2)` | branch switch | Gradient kink where the branches cross |
| EH+ `\|cos τ\|` on the acceptor torsion | absolute value | Gradient kink at `τ = ±π/2` |
| Hydrogen-bond topology perception | the *set* of terms changes with geometry | Piecewise-smooth surface; derivatives taken at fixed topology |
| Feathering onto the point charge | `C¹` at 7 Å by construction | Second derivative continuous but not smooth |

For the angle terms the implementation still does better than the naive route: `bangle_sin_g`
computes `sin θ` from `|r_ji × r_jk|` rather than from `√(1 − cos²θ)`. Both are the same number away
from collinearity; at it, the second is a `0/0` hiding a finite limit, while the first is the norm
of a vector that vanishes linearly and yields the correct **one-sided** derivative. The corner does
not go away — it is the model's — but a correct one-sided derivative is a much better thing to hand
an optimizer than catastrophic cancellation.

## What this means in practice

* An optimizer or MD run that walks onto the EH+ torsion axis will see forces inflated by one to
  four orders of magnitude, and a small energy discontinuity if it gets within `1e-6`. The forces
  are *correct for PM7*. If this happens, the structure is telling you something about the model.
* Symmetric hand-built geometries hit it; relaxed real structures generally do not, because the
  divergent force pushes them off the axis.
* If you suspect a new one: build the geometry, sweep towards it, and compare the analytic
  derivative with a central difference of the **energy**. The three signatures in the table above
  are distinct, and the test takes a few minutes to write.
