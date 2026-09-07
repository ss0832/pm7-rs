// SPDX-License-Identifier: GPL-3.0-or-later

//! Density-functional perturbation theory for phonons at **arbitrary** wavevector.
//!
//! # What this buys over a supercell
//!
//! Real-space force constants read off an `n₁×n₂×n₃` supercell give `D(q)` exactly at the `q`
//! commensurate with that supercell, and Fourier interpolation between them. The interpolation is
//! only as good as the decay of `Φ(0, T)` inside the supercell, and the cost of enlarging it is
//! cubic. Perturbation theory computes `D(q)` at any `q` from the primitive cell, at a cost that
//! does not depend on `q` at all.
//!
//! # The perturbation
//!
//! Displace atom `A` along `α` with a phase:
//!
//! ```text
//! u_{Aα}(T) = ε e^{i q·T}
//! ```
//!
//! Because every real-space block `H_{μν}(T)` is translation-covariant, its response factorizes as
//! `δH_{μν}(T₁, T₂) = ε e^{i q·T₁} h^q_{μν}(T₂ − T₁)`, and the Bloch transform couples exactly the
//! pair `(k + q, k)` — the entire content of "a phonon at `q` mixes `k` with `k + q`".
//!
//! The first-order density matrix follows from ordinary linear response, in the band basis of the
//! two k points:
//!
//! ```text
//! ΔP(k)_{mn} = [f_n(k) − f_m(k+q)] / [ε_n(k) − ε_m(k+q)] · ⟨ψ_{m,k+q}| ΔV |ψ_{n,k}⟩
//! ```
//!
//! Both the occupied–empty and the empty–occupied blocks contribute; the second comes from the
//! response of the *bra* at `k + q` to the `−q` component of a real displacement, and dropping it
//! halves the answer.
//!
//! # What is short-ranged and what is not
//!
//! The Coulomb kernel couples the **on-site** block of one atom to that of another in cell `T`,
//! and the on-site block at cell `T` carries the perturbation phase `e^{i q·T}` — so the Coulomb
//! response needs a *phased* lattice sum, [`crate::pbc::ewald::ewald_phased`]. The exchange
//! couples the `(0, T)` block, whose prefactor is `e^{i q·0} = 1`, so it needs no phased sum at
//! all and reuses the Born–von Kármán class sums the ground state already builds. Getting that
//! distinction backwards is the easiest way to produce a plausible and wrong dispersion.
//!
//! # Scope
//!
//! Closed-shell (RHF), `PbcMode::Ewald`, 1-D, 2-D and 3-D. The k mesh must be an explicit
//! Monkhorst–Pack grid — `KMesh::Gamma` does not build the translation-resolved Hamiltonian this
//! needs, so use `KMesh::grid(1, 1, 1)` for a Γ-sampled calculation.

use crate::basis::Basis;
use crate::cmatrix::CMatrix;
use crate::error::{Pm7Error, Result};
use crate::hamiltonian::CoreHamiltonian;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::pbc::{KPoint, PairList, PbcMode, PbcOptions};
use crate::scf::{Pm7Options, Pm7Result};
use crate::scf_pbc::BlochBlocks;
use crate::system::Molecule;

/// Complex translation-resolved blocks: the perturbation's analogue of [`BlochBlocks`].
#[derive(Clone, Debug)]
pub struct ComplexBlocks {
    re: BlochBlocks,
    im: BlochBlocks,
}

impl ComplexBlocks {
    fn new(translations: Vec<[i32; 3]>, nao: usize) -> Result<Self> {
        Ok(Self {
            re: BlochBlocks::new(translations.clone(), nao)?,
            im: BlochBlocks::new(translations, nao)?,
        })
    }

    fn zeros_like(other: &Self) -> Self {
        let mut out = other.clone();
        out.re.clear();
        out.im.clear();
        out
    }

    /// Add `value` (as `[re, im]`) to element `(row, col)` of the block at `t`, ignoring
    /// translations outside the set.
    #[inline]
    fn add(&mut self, t: [i32; 3], row: usize, col: usize, value: [f64; 2]) {
        if let Some(index) = self.re.position(t) {
            self.add_at(index, row, col, value);
        }
    }

    /// [`Self::add`] with the block already resolved.
    ///
    /// [`BlochBlocks::position`] is a `HashMap<[i32; 3], usize>` lookup, and the response Fock's
    /// pair loop called it **once per matrix element** — about sixty-six hashes per atom pair, on
    /// every one of `3N × iterations × spins` calls. The translation of a pair does not change
    /// during a solve, so it is resolved once in [`ResponseTables`] and passed in.
    #[inline]
    fn add_at(&mut self, index: usize, row: usize, col: usize, value: [f64; 2]) {
        self.re.block_mut(index)[(row, col)] += value[0];
        self.im.block_mut(index)[(row, col)] += value[1];
    }

    /// The real and imaginary blocks at a resolved index.
    #[inline]
    fn at(&self, index: usize) -> (&Matrix, &Matrix) {
        (self.re.block(index), self.im.block(index))
    }

    /// `Σ_T B(T) e^{i k·T}`.
    fn at_k(&self, k: &KPoint) -> CMatrix {
        let n = self.re.nao();
        let mut out = CMatrix::zeros(n);
        for (index, t) in self.re.translations().iter().enumerate() {
            let phase = k.phase(*t);
            let (c, s) = (phase.cos(), phase.sin());
            let (br, bi) = (self.re.block(index), self.im.block(index));
            for i in 0..n {
                for j in 0..n {
                    let (x, y) = (br[(i, j)], bi[(i, j)]);
                    let (re, im) = out.get(i, j);
                    out.set(i, j, re + x * c - y * s, im + x * s + y * c);
                }
            }
        }
        out
    }

    fn translations(&self) -> &[[i32; 3]] {
        self.re.translations()
    }

    /// RMS difference over every stored element, real and imaginary.
    fn rms_diff(&self, other: &Self) -> f64 {
        let r = self.re.rms_diff(&other.re);
        let i = self.im.rms_diff(&other.im);
        (r * r + i * i).sqrt()
    }

    /// `w·self + (1 − w)·other`.
    fn mixed(&self, other: &Self, w: f64) -> Self {
        Self {
            re: crate::scf_pbc::mix_blocks(&other.re, &self.re, w),
            im: crate::scf_pbc::mix_blocks(&other.im, &self.im, w),
        }
    }

    /// The `T = 0` blocks, real and imaginary.
    fn onsite(&self) -> (&Matrix, &Matrix) {
        let index = self
            .re
            .position([0, 0, 0])
            .expect("the block set contains the zero translation");
        (self.re.block(index), self.im.block(index))
    }

    /// Real and imaginary parts of the block at `t`, or `None` outside the set.
    /// `self + other`, block for block.
    fn plus(&self, other: &Self) -> Self {
        Self {
            re: crate::scf_pbc::add_blocks(&self.re, &other.re),
            im: crate::scf_pbc::add_blocks(&self.im, &other.im),
        }
    }

    /// `self * factor`.
    fn scaled_by(&self, factor: f64) -> Self {
        Self {
            re: crate::scf_pbc::scale_blocks(&self.re, factor),
            im: crate::scf_pbc::scale_blocks(&self.im, factor),
        }
    }

    fn get(&self, t: [i32; 3]) -> Option<(&Matrix, &Matrix)> {
        let index = self.re.position(t)?;
        Some((self.re.block(index), self.im.block(index)))
    }
}

/// The dynamical matrix at one wavevector, plus what it took to get there.
#[derive(Clone, Debug)]
pub struct DfptResult {
    /// Force-constant matrix `D(q)`, `3N × 3N`, Hermitian, in eV/Bohr².
    pub force_constants: CMatrix,
    /// Atomic masses in amu, for mass weighting.
    pub masses: Vec<f64>,
    /// Fractional wavevector.
    pub q_frac: [f64; 3],
    /// Self-consistency iterations the response took.
    pub iterations: usize,
    pub converged: bool,
    /// Largest change in the response density on the last iteration.
    pub residual: f64,
    /// The first-order density per degree of freedom, when `DfptOptions::keep_response` asked for
    /// it. `response[dof][k]` is `ΔP` at that k point, in the AO basis.
    ///
    /// `None` by default, because it is the largest array the calculation touches and nothing in
    /// the force constants needs it kept.
    pub response: Option<Vec<Vec<CMatrix>>>,
    /// The converged ground state the response was built on.
    ///
    /// Returned rather than dropped because the caller usually needs exactly this SCF and has no
    /// way to get it without running a second one. `analytic_hessian_with` did precisely that: on a
    /// k mesh it ran `run_pm7` for its own `HessianResult::scf` and then called this function,
    /// which ran the same ground state again — the whole SCF, twice, for one Hessian. Handing it
    /// back is what let that become one.
    pub scf: crate::scf::Pm7Result,
    /// Worst departure of `D(q)` from Hermiticity, **before** it was symmetrized, relative to the
    /// matrix's own largest element.
    ///
    /// `D(q)` is Hermitian by construction, so this is pure accumulated rounding on the Bloch sums
    /// plus whatever the response failed to converge. It is reported rather than kept private
    /// because it is the one number that says how much the symmetrization at the end had to clean
    /// up: a run whose value sits just under the refusal threshold is one whose force constants
    /// are held together by that threshold.
    pub hermiticity: f64,
}

impl DfptResult {
    /// Mass-weighted dynamical matrix in eV/(Å²·amu) — the same units
    /// [`crate::hessian_pbc::ForceConstants::dynamical_matrix`] returns.
    pub fn dynamical_matrix(&self) -> CMatrix {
        let n = self.force_constants.n;
        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let mut out = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let w = a0_sq / (self.masses[i / 3] * self.masses[j / 3]).sqrt();
                let (re, im) = self.force_constants.get(i, j);
                out.set(i, j, re * w, im * w);
            }
        }
        out
    }

    /// Phonon frequencies in cm⁻¹, ascending; a negative value is an imaginary mode.
    pub fn frequencies_cm(&self) -> Result<Vec<f64>> {
        Ok(self.modes()?.frequencies_cm)
    }

    /// The frequencies **and the polarization vectors** at this wavevector.
    ///
    /// The same [`crate::hessian_pbc::PhononModes`] the supercell route returns, from the same
    /// diagonalization the frequencies come from — so a caller reads a mode the same way whichever
    /// route produced it, and the two cannot disagree about which vector belongs to which
    /// frequency.
    pub fn modes(&self) -> Result<crate::hessian_pbc::PhononModes> {
        crate::hessian_pbc::phonon_modes(&self.dynamical_matrix(), &self.masses)
    }

    /// `Phi(q) + Phi^NA(q_hat)` in eV/Bohr², **before** mass weighting (convention C-7).
    ///
    /// The caller supplies `q_hat`; nothing is added implicitly. The `q -> 0` limit of the
    /// macroscopic field is **direction dependent**, so there is no such thing as "the" LO–TO
    /// correction at `q = 0` — a silently chosen direction would be a wrong answer rather than a
    /// default. `q_hat` need not be normalized; only its direction is used.
    ///
    /// Only meaningful **at** `q = 0`. Away from the zone centre the macroscopic field is already
    /// in `Phi(q)` through the Ewald sum, and adding this term as well would double-count it, so a
    /// non-zero `q` is refused rather than quietly corrected twice.
    pub fn force_constants_with_lo_to(&self, na: &NonAnalytic, q_hat: [f64; 3]) -> Result<CMatrix> {
        if self.q_frac.iter().any(|c| c.abs() > 1.0e-12) {
            return Err(Pm7Error::InvalidInput(format!(
                "the non-analytic term is the q -> 0 limit and belongs only at the zone centre; \
                 this result is at q = {:?}, where the macroscopic field is already inside the \
                 Ewald sum and adding it again would count it twice",
                self.q_frac
            )));
        }
        let extra = na.matrix(q_hat)?;
        if extra.n != self.force_constants.n {
            return Err(Pm7Error::InvalidInput(format!(
                "the non-analytic term is {}x{} but these force constants are {}x{}; they are not \
                 from the same cell",
                extra.n, extra.n, self.force_constants.n, self.force_constants.n
            )));
        }
        let mut out = self.force_constants.clone();
        for i in 0..out.n {
            for j in 0..out.n {
                let (ar, ai) = out.get(i, j);
                let (br, bi) = extra.get(i, j);
                out.set(i, j, ar + br, ai + bi);
            }
        }
        Ok(out)
    }

    /// [`Self::frequencies_cm`] with the LO–TO term added along `q_hat`.
    ///
    /// The longitudinal branch is pushed up and the transverse ones are untouched, which is the
    /// whole observable content of the splitting.
    pub fn frequencies_cm_lo_to(&self, na: &NonAnalytic, q_hat: [f64; 3]) -> Result<Vec<f64>> {
        let phi = self.force_constants_with_lo_to(na, q_hat)?;
        let n = phi.n;
        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let mut dynamical = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let w = a0_sq / (self.masses[i / 3] * self.masses[j / 3]).sqrt();
                let (re, im) = phi.get(i, j);
                dynamical.set(i, j, re * w, im * w);
            }
        }
        let (eigenvalues, _) = dynamical.hermitian_eigen()?;
        Ok(eigenvalues
            .into_iter()
            .map(|lambda| {
                let w = lambda.abs().sqrt() * crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
                if lambda < 0.0 {
                    -w
                } else {
                    w
                }
            })
            .collect())
    }
}

/// Whether the long-range monopole term is carried by the response.
///
/// The term is the phased lattice sum's own field, and it is wired into three places that have to
/// agree: the fixed-charge second derivative, the bare perturbation's per-atom channel, and the
/// `∂q_A(q)` shift in the coupled-perturbed kernel. Leaving it out of any one of them alone would
/// let the skeleton carry a long-range term the response could not screen, so the switch is all
/// three or none.
///
/// The point of `Off` is that it makes the term's effect **measurable** instead of arguable. A
/// claim that the long-range channel matters is worth a number, and the only way to get one is to
/// run without it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LongRange {
    /// Carry it wherever there is a lattice to sum over. The default, and what v0.2.1 did with no
    /// way to say otherwise.
    #[default]
    Auto,
    /// Require it. A cell whose periodic mode has no lattice sum is an error rather than a quiet
    /// difference in what was computed.
    Require,
    /// Leave it out, so its effect can be measured.
    Off,
}

/// Settings for the perturbation solve.
#[derive(Clone, Copy, Debug)]
pub struct DfptOptions {
    /// Whether the long-range monopole term is carried. See [`LongRange`].
    pub long_range: LongRange,
    /// How many perturbations to solve and contract before releasing their response densities.
    ///
    /// `None` (the default) holds all `3N`, which is the previous behaviour arithmetic for
    /// arithmetic. A `Some(b)` caps the response side of the assembly at `b * n_k * nao^2`.
    ///
    /// The knob exists because the contraction needs **both** indices -- `D_{jj'}` pairs every
    /// bare column with every response -- so neither side can be released before the other is
    /// done, and both grow as `N^3 n_k`. Measured on diamond supercells at a 4x4x4 mesh: 24 MB
    /// each at 8 atoms, 81 MB at 12, extrapolating to roughly 650 MB at 24 atoms and 5 GB at 48.
    ///
    /// It is a trade, not a free win: the phased columns are rebuilt once per block, so the Bloch
    /// sums are repeated `ceil(3N/b)` times. A small `b` buys memory with assembly time.
    pub response_block: Option<usize>,

    /// Keep the first-order densities in [`DfptResult::response`].
    ///
    /// Off by default: it is `3N × n_k × 2·nao²` floats, the largest array in the calculation, and
    /// the force constants themselves never need it kept.
    pub keep_response: bool,
    pub max_iterations: usize,
    /// Convergence threshold on the response density, RMS.
    pub tolerance: f64,
    /// Linear mixing factor for the response density.
    ///
    /// The **first** rung of a damping ladder, not the only one: if the iteration diverges the
    /// solver retries with progressively heavier damping before giving up. See `solve_one`.
    pub mixing: f64,
    /// Refuse to return a result whose response did not converge. **On by default.**
    ///
    /// A diverged linear response does not produce a slightly-wrong dynamical matrix, it produces
    /// numbers like `1e33`, and `frequencies_cm` will cheerfully take the square root of one. The
    /// old behaviour was to return `Ok` with a `converged: false` field that nothing was obliged
    /// to read, which is how a general `q` could report garbage as a phonon spectrum.
    ///
    /// Set it to `false` only to inspect a failure — the result is not a physical answer.
    pub require_convergence: bool,
}

impl Default for DfptOptions {
    fn default() -> Self {
        Self {
            // The response is a linear fixed point whose map has a spectral radius around 0.7 for
            // a covalent solid, so plain iteration converges geometrically but not fast; the
            // budget is set so that even a stiff case finishes rather than reporting failure.
            long_range: LongRange::Auto,
            response_block: None,
            keep_response: false,
            max_iterations: 200,
            tolerance: 1.0e-10,
            mixing: 0.9,
            require_convergence: true,
        }
    }
}

/// Refuse a response that did not converge, naming the numbers.
fn check_converged(
    what: &str,
    dfpt: &DfptOptions,
    converged: bool,
    iterations: usize,
    residual: f64,
) -> Result<()> {
    if converged || !dfpt.require_convergence {
        return Ok(());
    }
    Err(Pm7Error::ResponseFailed(format!(
        "the {what} response did not converge after {iterations} iterations across the damping \
         ladder: residual {residual:.3e} against a tolerance of {:.1e}. The response is a linear \
         fixed point, so this means the spectral radius of the self-consistency map is at or above \
         one for this system and wavevector, and the result is not a slightly-wrong answer but a \
         diverged one. Try a denser k mesh, a larger `tolerance`, or a smaller `mixing`; set \
         `require_convergence: false` only to inspect the failure.",
        dfpt.tolerance
    )))
}

/// The largest violation of `M† = M`, and the largest element, of a complex matrix.
///
/// `D(q)` is Hermitian for any `q` by construction — `Phi(0A, TB)` is real and symmetric under
/// exchanging the two ends, so its Bloch sum satisfies `D(q)† = D(q)`. That makes this a free and
/// very sharp invariant: a phase error, an index transposition or a diverged response all break it
/// long before the frequencies look obviously wrong.
fn hermiticity(m: &CMatrix) -> (f64, f64) {
    let (mut worst, mut scale) = (0.0_f64, 0.0_f64);
    for i in 0..m.n {
        for j in 0..m.n {
            let (a, b) = m.get(i, j);
            let (c, d) = m.get(j, i);
            worst = worst.max((a - c).abs()).max((b + d).abs());
            scale = scale.max(a.abs()).max(b.abs());
        }
    }
    (worst, scale)
}

/// Say so when `q` is finer than the k mesh can represent.
///
/// The response couples `k` with `k + q`. An `n₁ × n₂ × n₃` mesh samples the zone in steps of
/// `1/nᵢ`, so a `q` well inside one step asks the sampling to distinguish two points it cannot
/// tell apart, and the answer degrades smoothly towards meaningless — while still reporting
/// `converged: true`, because the *linear solve* did converge. It converged to the answer for a
/// question the mesh could not pose.
///
/// Measured on LiF at `q = 1/160` along `a₁`, as `max_A |Σ_B Φ_{Aα,Bβ}(q)|`, which the formalism
/// requires to vanish as `q → 0`:
///
/// | mesh | residue |
/// |---|---|
/// | 3³ | 2.96e-1 |
/// | 5³ | 6.75e-2 |
/// | 7³ | 3.21e-2 |
///
/// It converges away, so this is a sampling limit and not a defect in the construction —
/// `tests/dfpt_long_wavelength.rs` pins both that and the identities that stay exact throughout.
/// But nothing said so, and a caller who asks for a long-wavelength phonon on a coarse mesh gets
/// a plausible number with no indication that the mesh, not the physics, produced it.
///
/// A warning rather than an error: there is no sharp threshold, only a degradation, and refusing
/// would break the legitimate case of a deliberately coarse survey. It goes to stderr and can be
/// silenced with `PM7_QUIET`.
fn warn_if_q_outruns_the_mesh(pbc: &PbcOptions, q_frac: [f64; 3]) {
    if std::env::var_os("PM7_QUIET").is_some() {
        return;
    }
    let divisions = pbc.kmesh.divisions();
    for (axis, &n) in divisions.iter().enumerate() {
        let q = q_frac[axis].abs();
        // Nothing to say about the zone centre, which is exact, or about a `q` the mesh resolves.
        if q <= 0.0 || n == 0 {
            continue;
        }
        let step = 1.0 / n as f64;
        if q < 0.5 * step {
            eprintln!(
                "pm7-rs: q = {q:.6} along axis {axis} is finer than half the k-mesh step \
                 {step:.6} ({n} divisions). The response couples k with k+q, so the mesh cannot \
                 resolve this wavevector and the long-wavelength limit will be wrong by an amount \
                 that shrinks only when the mesh is refined. Use at least {} divisions on this \
                 axis, or a larger q. Set PM7_QUIET to silence this.",
                (1.0 / q).ceil() as usize
            );
        }
    }
}

/// The dynamical matrix at `q_frac` (fractional reciprocal coordinates) by perturbation theory.
pub fn dynamical_matrix_dfpt(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    q_frac: [f64; 3],
    dfpt: &DfptOptions,
) -> Result<DfptResult> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput(
            "perturbation theory needs a periodic system; this molecule has no cell".into(),
        )
    })?;
    let pbc = options.pbc_for(molecule).expect("cell implies pbc options");
    if pbc.mode != PbcMode::Ewald {
        // Naming the flag matters more here than it looks. This refusal is reached from `dfpt`,
        // from `born`, and — since the k-mesh Hessian stopped running a redundant ground state —
        // from `hessian` and `frequencies` on a k mesh, where it is now the *first* thing the user
        // meets. Saying only "the perturbation solver needs the Ewald periodic mode" leaves them
        // to work out that `--pbc-mode` is the knob and that dropping it is the fix.
        return Err(Pm7Error::InvalidInput(
            "`--pbc-mode mopac` (`pbc_mode=\"mopac\"`) cannot be perturbed: the perturbation \
             solver differentiates the long-range lattice sum, and MOPAC-compatibility mode \
             replaces that sum with a finite cluster truncation, which has no derivative to take. \
             Drop the flag — `ewald` is the default, and is the mode every response, stress and \
             phonon in this crate is built on. MOPAC-compatibility mode exists to reproduce \
             MOPAC's own solid-state energies, and is Gamma-only and stress-free for the same \
             reason."
                .into(),
        ));
    }
    // `KMesh::Gamma` does not build the translation-resolved Hamiltonian the response needs, but
    // `grid(1, 1, 1)` is the same sampling and does. Promote rather than refuse: telling a caller
    // to write a different spelling of the thing they already asked for is a worse error message
    // than no error at all.
    let mut pbc = pbc;
    if matches!(pbc.kmesh, crate::pbc::KMesh::Gamma) {
        pbc.kmesh = crate::pbc::KMesh::grid(1, 1, 1);
    }
    let pbc = pbc;
    let mut options = options.clone();
    options.pbc = Some(pbc.clone());
    let options = &options;

    let basis = Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        &basis,
        params,
        options.force_dpath,
        &pbc,
        options.active_field(),
    )?;
    let scf = crate::scf::run_pm7(molecule, params, options)?;
    let density = scf.bloch_density.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput("the perturbation solver needs a k-resolved density".into())
    })?;
    let core_bloch = core.bloch.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput("the perturbation solver needs the k-resolved core".into())
    })?;

    // Cartesian q, built only from the periodic reciprocal directions.
    let b = cell.reciprocal_2pi();
    let mut q_cart = Vec3::zero();
    for (k, frac) in q_frac.iter().enumerate().take(cell.dim()) {
        q_cart += b[k] * *frac;
    }

    warn_if_q_outruns_the_mesh(&pbc, q_frac);

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let translations: Vec<[i32; 3]> = core_bloch.translations().to_vec();

    // --- the bare perturbation and the skeleton term, from one pair pass ------------------
    let (bare, skeleton) = {
        let _t = crate::profile::stage("dfpt: bare + skeleton");
        bare_and_skeleton(
            molecule,
            params,
            &basis,
            &core,
            &scf,
            density,
            &pbc,
            q_cart,
            &translations,
            dfpt.long_range,
        )?
    };

    // --- band structure at k and k+q -----------------------------------------------------
    // The full mesh, not the time-reversal-folded one: folding pairs `k` with `−k`, and a `q ≠ 0`
    // response relates `k` to `k + q`, which is a different pairing entirely.
    let kpoints = response_mesh(&pbc.kmesh, &cell)?;
    // The converged Fock, built once per spin: `H(k)` at any `k` is a Bloch sum of the same
    // blocks. An unrestricted cell has two, built from `(P ± Δ)/2` the way the SCF built them, so
    // the response is expanded on exactly the orbitals the ground state converged to.
    let spin_focks = converged_spin_focks(molecule, &basis, params, &core, &scf, density)?;
    let electrons = spin_electron_counts(molecule, params, options, scf.unrestricted)?;
    let refs: Vec<&crate::scf_pbc::BlochBlocks> = spin_focks.iter().collect();
    let per_orbital = if scf.unrestricted { 1.0 } else { 2.0 };
    let channels = {
        let _t = crate::profile::stage("dfpt: bands at k and k+q");
        build_bands(
            &kpoints,
            &refs,
            &electrons,
            per_orbital,
            &pbc,
            q_frac,
            q_cart,
        )?
    };
    let bands = &channels[0];

    // --- self-consistent response ---------------------------------------------------------
    //
    // Parallel over degrees of freedom, as the molecular and Γ-periodic CPHF already are. Each
    // `solve_one` is independent — it reads the bands and the bare perturbation and writes
    // nothing shared — and `collect` preserves order, so the assembled `D(q)` does not depend on
    // the thread count.
    let tables = {
        let _t = crate::profile::stage("dfpt: response tables at q");
        ResponseTables::build(
            molecule,
            &core,
            &pbc,
            q_cart,
            &translations,
            dfpt.long_range != LongRange::Off,
        )
        .expect("a periodic cell was checked above")
    };
    // --- solve and assemble, in blocks of perturbations ------------------------------------
    //
    // `D^resp_{jj'} = Σ_k Tr[ Δh^j(k)† ΔP^{j'}(k) ]`, the standard non-variational form: the
    // *bare* perturbation against the *self-consistent* density response, which is what the
    // 2n+1 theorem leaves after the two-electron double counting cancels.
    //
    // # Why this is blocked
    //
    // The contraction needs **both** indices, so neither side can be dropped before the other is
    // finished, and both are `3N × n_k × nao²` complex. Measured on diamond supercells that is
    // 24 MB apiece at 8 atoms on a 4×4×4 mesh and 81 MB at 12 atoms — it grows as `N³ n_k`, so
    // 24 atoms is about 650 MB and 48 atoms about 5 GB, which is where a calculation stops being
    // possible rather than merely slow.
    //
    // Holding only `block` responses at a time cuts that side to `block × n_k × nao²`. It is not
    // free: the phased columns `Δh^j(k)` are then rebuilt once per block rather than once in
    // total, so the Bloch sums are repeated `ceil(3N/block)` times. `DfptOptions::response_block`
    // is that knob, and `None` — the default — keeps everything in one block, which is the
    // previous behaviour arithmetic for arithmetic.
    let block = dfpt
        .response_block
        .map(|b| b.max(1))
        .unwrap_or(ndof)
        .min(ndof.max(1));
    let mut total = skeleton;
    let weight = 1.0 / kpoints.len() as f64;
    let mut iterations = 0;
    let mut converged = true;
    let mut residual = 0.0_f64;
    let mut kept: Vec<Vec<CMatrix>> = Vec::new();

    let mut start = 0usize;
    while start < ndof {
        let stop = (start + block).min(ndof);
        let solved: Vec<(Vec<CMatrix>, usize, bool, f64)> = {
            use rayon::prelude::*;
            let _t = crate::profile::stage("dfpt: response solve (all DOFs)");
            (start..stop)
                .into_par_iter()
                .map(|j| {
                    solve_one(
                        BareTerm::Local(&bare[j]),
                        &channels,
                        molecule,
                        params,
                        &basis,
                        &core,
                        &tables,
                        &translations,
                        dfpt,
                    )
                    .map(|(per_spin, iters, ok, res)| (total_per_k(&per_spin), iters, ok, res))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let mut responses: Vec<Vec<CMatrix>> = Vec::with_capacity(stop - start);
        for (per_k, iters, ok, res) in solved {
            iterations = iterations.max(iters);
            converged &= ok;
            residual = residual.max(res);
            responses.push(per_k);
        }

        {
            let _t = crate::profile::stage("dfpt: assemble D(q)");
            for (j, column) in bare.iter().enumerate() {
                // `Δh^j(k)` depends on `(j, k)` and not on `j'`, so it is built once per column
                // per block rather than inside the `j'` loop, where it was a Bloch sum over every
                // translation `3N` times more often than there are distinct values.
                let phased: Vec<CMatrix> = bands.iter().map(|band| column.at_k(&band.k)).collect();
                for (offset, delta) in responses.iter().enumerate() {
                    let jp = start + offset;
                    let mut acc = (0.0, 0.0);
                    for slot in 0..bands.len() {
                        let h = &phased[slot];
                        let p = &delta[slot];
                        // Tr[h† p] = Σ_{μν} conj(h_{μν}) p_{μν}
                        for mu in 0..h.n {
                            for nu in 0..h.n {
                                let (hr, hi) = h.get(mu, nu);
                                let (pr, pi) = p.get(mu, nu);
                                acc.0 += hr * pr + hi * pi;
                                acc.1 += hr * pi - hi * pr;
                            }
                        }
                    }
                    let (re, im) = total.get(j, jp);
                    total.set(j, jp, re + weight * acc.0, im + weight * acc.1);
                }
            }
        }
        if dfpt.keep_response {
            kept.extend(responses);
        }
        start = stop;
    }
    let response_density = kept;

    // The post-SCF corrections. They are classical, pairwise or many-body in the *positions*
    // alone, so they never entered the electronic response — but they are part of `D(q)` and were
    // simply missing before, invisibly, because every test used the corrections-off method.
    crate::gradient::add_correction_hessian_phased(molecule, options, q_cart, &mut total);

    // `D(q)` is Hermitian by construction, so **measure the deviation before averaging it away**.
    //
    // Symmetrizing unconditionally is what made this the least useful invariant in the crate: a
    // phase error, a transposed index or a diverged response all break Hermiticity badly, and all
    // of them used to be laundered into a plausible-looking Hermitian matrix on the way out. The
    // averaging is still done — genuine rounding on a Bloch sum is real and belongs cleaned up —
    // but only after the deviation has been checked against the matrix's own scale.
    let (violation, scale) = hermiticity(&total);
    // `1e-6` relative, not the `1e-8` this started at, because `1e-8` refuses correct answers.
    //
    // The residual asymmetry is set by how well the *eigenvectors* are determined, and inside a
    // near-degenerate manifold that is not very well at all: any unitary rotation of a degenerate
    // block is an equally valid eigenbasis, so the response is invariant under it algebraically
    // and only to rounding numerically. Measured on two-atom rocksalt cells, identical geometry,
    // changing only the elements:
    //
    // | cell   | smallest level splitting | asymmetry of D(q) |
    // |--------|--------------------------|-------------------|
    // | NaCl   | 1.4e-14 eV               | 3.5e-19           |
    // | CsI    | 2.7e-14 eV               | 5.8e-18           |
    // | MgO    | 9.5e-10 eV               | 3.5e-15 .. 7e-11  |
    // | ZnS    | 4.7e-10 eV               | 8.3e-12 .. 7e-9   |
    //
    // A cubic CsPbI3 cell went past `1e-8` outright and was refused, with a response converged to
    // 1e-10 and a value reproducible to three significant figures across four decades of SCF
    // tolerance -- which is what a conditioning limit looks like, and is not what a defect looks
    // like. The defects this check exists to catch (a phase error, a transposed index, a diverged
    // response) put the asymmetry within a couple of orders of the matrix itself, so `1e-6` still
    // catches them with five orders to spare while leaving correct near-degenerate systems alone.
    //
    // `DfptResult::hermiticity` reports the measured value, so a run sitting close to the line is
    // visible rather than merely allowed.
    let allowed = 1.0e-6 * scale.max(1.0);
    if !violation.is_finite() || violation > allowed {
        // Name the response residual, and say whether it is the explanation.
        //
        // An unconverged response leaves a non-Hermiticity a small multiple of its own residual:
        // measured on LiF at `q = (1/4, 0, 0)` on a 4x4x4 mesh, the ratio runs 2.6 to 4.8 across
        // four decades of tolerance. So this check needs a response a few times tighter than its
        // own allowance, which is why the default tolerance is `1e-10` and why `1e-8` fails.
        //
        // The message used to end "this is a defect in the construction or a diverged response,
        // not a tolerance to widen". That is true of the *Hermiticity* tolerance and actively
        // misleading about the cause: a caller who set `dfpt_tolerance = 1e-8` -- a documented
        // knob, exposed on every surface -- was told their construction was broken when the fix
        // was to put the tolerance back.
        //
        // `10 x residual` as the dividing line, because the observed amplification is under 5 and
        // a factor of two of headroom keeps a genuine construction error from being excused.
        let explained_by_residual = residual.is_finite() && violation <= 10.0 * residual;
        let blame = if explained_by_residual {
            format!(
                " The response itself only reached {residual:.3e} against the {:.1e} it was asked \
                 for, and an unconverged response leaves a non-Hermiticity a few times its own \
                 residual -- which is all of this one. Tighten `dfpt_tolerance` to 1e-8 or below \
                 (the default is 1e-10) before looking further.",
                dfpt.tolerance
            )
        } else {
            format!(
                " The response reached {residual:.3e}, too tight to account for a deviation this \
                 large: D(q) is Hermitian by construction, so this is a phase error or a \
                 transposed index, not a tolerance to widen."
            )
        };
        return Err(Pm7Error::ResponseFailed(format!(
            "the dynamical matrix at q = {q_frac:?} is not Hermitian: worst deviation {violation:.3e} \
             against a largest element of {scale:.3e}, which is {:.1e} times the {allowed:.1e} that \
             rounding on a Bloch sum can explain.{blame}",
            violation / allowed.max(f64::MIN_POSITIVE)
        )));
    }
    let n = total.n;
    let mut hermitian = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = total.get(i, j);
            let (br, bi) = total.get(j, i);
            hermitian.set(i, j, 0.5 * (ar + br), 0.5 * (ai - bi));
        }
    }
    check_converged("phonon", dfpt, converged, iterations, residual)?;

    Ok(DfptResult {
        force_constants: hermitian,
        masses: molecule
            .atoms
            .iter()
            .map(|a| crate::data_tables::MASS[a.z as usize])
            .collect(),
        q_frac,
        iterations,
        converged,
        residual,
        scf,
        hermiticity: violation / scale.max(1.0),
        // Moved rather than cloned: `response_density` is dead after the contraction above, and it
        // is the largest array in the calculation, so cloning it to hand it back would double the
        // peak exactly where the peak matters.
        response: dfpt.keep_response.then_some(response_density),
    })
}

struct BandPair {
    k: KPoint,
    eps_k: Vec<f64>,
    c_k: CMatrix,
    eps_kq: Vec<f64>,
    c_kq: CMatrix,
    /// Occupations at `k` and at `k + q`, in electrons per orbital, from one chemical potential
    /// shared by the whole mesh.
    ///
    /// Filling the lowest `n_occ` bands at every k independently is right for an insulator and
    /// wrong for anything else — and wrong *silently*, because the response still converges to a
    /// plausible number. A 1-D chain is exactly where bands cross between mesh points.
    occ_k: Vec<f64>,
    occ_kq: Vec<f64>,
}

/// Refuse a gapless mesh that carries no smearing to regularize it.
///
/// The response weights each band pair by `Δf/Δε`. Where a band crosses the Fermi level that is a
/// `0/0`: two states arbitrarily close in energy with occupations arbitrarily close together. A
/// **smeared** occupation makes it finite, because `Δf` then goes to zero with `Δε` at a rate the
/// smearing function fixes. An unsmeared step does not: the answer comes out depending on which
/// pairs happened to fall inside the `1e-8` denominator floor, which is a property of the floor
/// rather than of the system.
///
/// So the gate is on **gaplessness without smearing**, not on fractional occupations. Through
/// v0.2.1 it was the latter: any fractional occupancy was refused outright, which turned every
/// smeared metal away — including the ones where smearing is exactly what makes the response
/// well defined. A partially occupied band is the *normal* state of a metal treated this way and
/// is not by itself a reason to stop.
///
/// What is still missing is the Fermi-level shift: a `q = 0` perturbation of a metal moves `E_F`,
/// and the intraband term that goes with it (de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995)) is
/// not included. For `q ≠ 0` that term vanishes by symmetry — the perturbation has no uniform
/// component to shift the chemical potential with — so the wavevectors a phonon dispersion is
/// actually made of are complete. `docs/scope.md` says which is which.
/// An occupation this far from 0 or 1 is a genuinely partial one.
///
/// **Chosen against the gap between the two populations, not against a derivation.** A smearing
/// applied to a cell that *has* a gap leaves occupations within roughly `exp(−gap / 2w)` of an
/// integer, and a cell whose bands actually cross the Fermi level puts a whole fraction of a state
/// on either side of it. Measured on PM7 ZnS, whose gap is a strong function of the mesh, at three
/// Fermi widths — the worst departure from an integer, with the entropy beside it:
///
/// | mesh | width | gap | entropy | worst departure |
/// |---|---|---|---|---|
/// | 3×3×3 | 0.10 | 0.000 eV | −7.1e-3 | **0.333** |
/// | 4×4×4 | 0.05 | 0.000 eV | −1.5e-3 | **0.334** |
/// | 5×5×5 | 0.10 | 0.000 eV | −1.5e-3 | **0.333** |
/// | 4×4×4 | 0.10 | 2.575 eV | −2.6e-7 | 5.7e-6 |
/// | 5×5×5 | 0.05 | 2.242 eV | −6.3e-12 | ~1e-11 |
/// | 4×4×4 | 0.02 | 2.915 eV | −5.9e-18 | ~1e-17 |
///
/// The gapless cases sit at **a third of a state** and the gapped ones at `5.7e-6` and below —
/// nearly five orders of magnitude of empty space, and this threshold is in the middle of it. The
/// same split holds for SrTiO₃ (6.4 eV gap at a 0.10 eV width, entropy `−6e-15`).
///
/// The value also carries its own meaning: the omitted intraband term is weighted by the states at
/// the Fermi level, so a departure below `1e-3` bounds the omission at about a tenth of a percent,
/// which is far under any accuracy claimed for a PM7 phonon.
const PARTIAL_OCCUPATION_TOLERANCE: f64 = 1.0e-3;

/// Refuse a `q = 0` response on a cell whose bands genuinely cross the Fermi level.
///
/// The response itself does **not** assume integer occupations: it weights each band pair by
/// `Δf/Δε`, which is exactly the metallic form, and [`refuse_gapless_without_smearing`] is the gate
/// that keeps that quantity finite. What is missing is one specific term, and only at one specific
/// wavevector.
///
/// A `q = 0` perturbation of a partially occupied cell **moves the Fermi level**. Keeping the
/// electron count fixed then costs an intraband contribution — de Gironcoli, *Phys. Rev. B* **51**,
/// 6773 (1995) — which is not implemented here. For `q ≠ 0` the perturbation has no uniform
/// component to shift the chemical potential with and the term vanishes by symmetry, so a phonon
/// dispersion away from the zone centre is complete. At `q = 0` on a real metal it is not, and the
/// answer that comes back is wrong by an amount nothing in the result reports.
///
/// So this refuses rather than warning: an incomplete zone-centre response looks exactly like a
/// complete one, and the zone centre is reached from more places than `dfpt` — [`born_and_dielectric`]
/// is a `q = 0` field response by construction, and `analytic_hessian` on a k mesh delegates here.
///
/// **Smearing that leaves the occupations integral is fine, and that is the common case.** A width
/// applied to a gapped cell — to escape a symmetry-broken solution, or because the SCF ladder chose
/// one — changes the path and not the answer, and this lets it through. The test is on the
/// occupations rather than on the entropy because Methfessel–Paxton entropies can pass through zero
/// with the occupations still fractional; the entropy is reported alongside because it is the
/// quantity the SCF's own smearing ladder gates on.
fn refuse_partial_occupations_at_zone_centre(
    occupations: &[Vec<f64>],
    entropy_ev: f64,
    fermi_ev: f64,
    q_frac: [f64; 3],
) -> Result<()> {
    // Exactly the zone centre. The statement above — that the term vanishes for `q ≠ 0` — is about
    // the perturbation having no uniform component, which is true of any non-zero `q`.
    if q_frac.iter().any(|c| *c != 0.0) {
        return Ok(());
    }
    let mut worst = 0.0_f64;
    let mut count = 0usize;
    for band in occupations {
        for &f in band {
            // These are per-orbital fractions, so an integral fill is 0 or 1.
            let departure = f.min(1.0 - f);
            if departure > PARTIAL_OCCUPATION_TOLERANCE {
                count += 1;
                worst = worst.max(departure);
            }
        }
    }
    if count == 0 {
        return Ok(());
    }
    Err(Pm7Error::InvalidInput(format!(
        "this cell is partially occupied at the Fermi level ({count} states are fractionally \
         filled, the furthest {worst:.3e} from an integer; E_F = {fermi_ev:.6} eV, entropy \
         {entropy_ev:.3e} eV), and the zone-centre response for such a cell is incomplete. A \
         q = 0 perturbation moves the Fermi level, and the intraband term that goes with holding \
         the electron count fixed (de Gironcoli, Phys. Rev. B 51, 6773 (1995)) is not implemented. \
         It is refused rather than reported because the result would look exactly like a complete \
         one. Three ways on: any q != 0 is complete, since the term vanishes there by symmetry; a \
         finer k mesh often opens a gap that a coarse one missed (PM7 ZnS is gapless at 3x3x3 and \
         has a 2.9 eV gap at 4x4x4); and a narrower smearing width restores integral occupations \
         on a cell that does have a gap."
    )))
}

fn refuse_gapless_without_smearing(
    occupations: &[Vec<f64>],
    energies: &[Vec<f64>],
    fermi_ev: f64,
    smearing: crate::pbc::Smearing,
) -> Result<()> {
    if !matches!(smearing, crate::pbc::Smearing::None) {
        return Ok(());
    }
    // Highest occupied and lowest empty across the whole mesh. A negative or zero difference is a
    // metal; the bands overlap and no single gap exists.
    let mut homo = f64::NEG_INFINITY;
    let mut lumo = f64::INFINITY;
    for (bands, occ) in energies.iter().zip(occupations) {
        for (&e, &f) in bands.iter().zip(occ) {
            if f > 0.5 {
                homo = homo.max(e);
            } else {
                lumo = lumo.min(e);
            }
        }
    }
    if homo.is_finite() && lumo.is_finite() && lumo - homo <= 0.0 {
        return Err(Pm7Error::InvalidInput(format!(
            "the mesh found no band gap (highest occupied {homo:.6} eV, lowest empty \
             {lumo:.6} eV, E_F = {fermi_ev:.6} eV), and there is no smearing to regularize the \
             response. Every pair straddling the Fermi level contributes `Δf/Δε`, which is a 0/0 \
             for a step occupation, so the answer would be decided by which pairs fell inside a \
             numerical floor rather than by the physics. Set `PbcOptions::smearing` (Python: \
             `smearing=(\"fermi\", width_ev)`), which is what makes those terms finite."
        )));
    }
    Ok(())
}

/// The **SCF's own** k set, unfolded.
///
/// This used to reconstruct a Γ-centred, unshifted grid from `KMesh::divisions()`, which discards
/// `shift` and `gamma_centred` and returns `[1, 1, 1]` for `KMesh::Explicit`. The SCF meanwhile
/// used `KMesh::expand()`, which honours all three — so a shifted mesh, an original
/// Monkhorst–Pack mesh, or an explicit list ran the response on a *different set of k points from
/// the one the density was converged on*, and said nothing about it.
///
/// `KPoint::cart` is filled by the expansion too. It was left at zero here, which made
/// `kq.cart = k.cart + q_cart` wrong; nothing read `cart` so nothing broke, but the field
/// perturbation does read it.
fn response_mesh(kmesh: &crate::pbc::KMesh, cell: &crate::cell::Cell) -> Result<Vec<KPoint>> {
    Ok(kmesh.expand_unfolded(cell)?.points)
}

#[inline]
fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// The bare (fixed-density) perturbation blocks for all `3N` degrees of freedom, and the skeleton
/// contribution to `D(q)`, from one pass over the image pairs.
///
/// Both come from the same derivatives, so they are built together: the first derivative of each
/// pair term feeds the perturbation, the second feeds the skeleton.
///
/// The phase bookkeeping is the whole difficulty. Every contribution lands in a block
/// `H(T_block)` whose *row* atom sits in cell 0, and carries the factor `e^{i q·S}` where `S` is
/// the cell of the atom being displaced, measured from that row atom's cell. Summing the result
/// over `T` at `q = 0` has to reproduce `hessian_pbc`'s ordinary derivative Fock, which is what
/// `the_bare_perturbation_reduces_to_the_gamma_derivative_fock` checks.
#[allow(clippy::too_many_arguments)]
fn bare_and_skeleton(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    // The pair list comes from `pbc`, not from here; the core is taken so the signature matches
    // the rest of the family and stays right if the pair enumeration ever moves onto it.
    _core: &CoreHamiltonian,
    scf: &Pm7Result,
    density: &BlochBlocks,
    pbc: &PbcOptions,
    q: Vec3,
    translations: &[[i32; 3]],
    long_range: LongRange,
) -> Result<(Vec<ComplexBlocks>, CMatrix)> {
    use crate::dual::Scalar;
    use crate::dual2::Dual2;

    let cell = molecule.cell.expect("periodic");
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let nao = basis.nao;
    let divisions = pbc.kmesh.divisions();
    let has_any_d = molecule
        .atoms
        .iter()
        .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));

    let mut bare: Vec<ComplexBlocks> = Vec::with_capacity(ndof);
    let template = ComplexBlocks::new(translations.to_vec(), nao)?;
    for _ in 0..ndof {
        bare.push(ComplexBlocks::zeros_like(&template));
    }
    let mut skeleton = CMatrix::zeros(ndof);

    let onsite = density
        .get([0, 0, 0])
        .expect("the density has a zero translation");
    let list = PairList::cached(molecule, pbc.short_range_cutoff);

    for pair in &list.pairs {
        let (u, v) = (pair.a, pair.b);
        let eu = params.element(molecule.atoms[u].z)?;
        let ev = params.element(molecule.atoms[v].z)?;
        let (a, b, d, t) = if eu.n_orb >= ev.n_orb {
            (u, v, pair.d, pair.t)
        } else {
            (v, u, pair.d * -1.0, [-pair.t[0], -pair.t[1], -pair.t[2]])
        };
        let neg_t = [-t[0], -t[1], -t[2]];
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        let far = pair.r > crate::pbc::FEATHER_RANGE_BOHR;
        let pt = density
            .folded(t, divisions)
            .expect("the density covers the pair translations");
        let pt_neg = density
            .folded(neg_t, divisions)
            .expect("the density covers the negated pair translations");

        // `e^{i q·T}` for this pair's translation, and its conjugate.
        let shift = cell.translation(t);
        let angle = q.dot(shift);
        let phase = [angle.cos(), angle.sin()];
        let conj = [phase[0], -phase[1]];
        let one = [1.0, 0.0];

        // --- first derivatives, for the perturbation ------------------------------------
        let (te1, s1) = if far {
            (
                crate::integrals::point_charge_pair_dual(ea, eb, d),
                [[crate::dual::Dual::constant(0.0); 9]; 9],
            )
        } else {
            crate::gradient::pair_dual(ea, eb, Vec3::zero(), d, has_any_d)?
        };
        let dv: [f64; 3] = if pbc.mode == PbcMode::Ewald {
            let c = -crate::constants::PM7_EV / (pair.r * pair.r * pair.r);
            [d.x * c, d.y * c, d.z * c]
        } else {
            [0.0; 3]
        };
        let pop_a: f64 = (0..na).map(|k| onsite[(oa + k, oa + k)]).sum();
        let pop_b: f64 = (0..nb).map(|k| onsite[(ob + k, ob + k)]).sum();
        let subtract = pbc.mode == PbcMode::Ewald;

        for axis in 0..3 {
            // A displacement of `b` moves `d` forwards, one of `a` moves it backwards. The four
            // (atom, phase) combinations below are the whole of the phase bookkeeping.
            let cases: [(usize, f64, [f64; 2], bool); 4] = [
                (a, -1.0, one, true),   // a in cell 0, for blocks whose row atom is a
                (b, 1.0, phase, true),  // b in cell T, likewise
                (b, 1.0, one, false),   // b in cell 0, for blocks whose row atom is b
                (a, -1.0, conj, false), // a in cell −T, likewise
            ];
            for (atom, sign, ph, row_is_a) in cases {
                let target = &mut bare[3 * atom + axis];
                let scale = |x: f64| -> [f64; 2] { [ph[0] * sign * x, ph[1] * sign * x] };

                if row_is_a {
                    // Resonance into H(T)[a, b].
                    for i in 0..na {
                        let bi = crate::hessian::beta_of(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = crate::hessian::beta_of(eb, basis.aos[ob + j].orb);
                            let val = 0.5 * (bi + bj) * s1[i][j].d[axis];
                            target.add(t, oa + i, ob + j, scale(val));
                        }
                    }
                    // Electron–core attraction of b's core on a, into H(0)[a, a].
                    for i in 0..na {
                        for j in 0..na {
                            target.add([0, 0, 0], oa + i, oa + j, scale(te1.e1b[i][j].d[axis]));
                        }
                    }
                    // Coulomb from b's density, into H(0)[a, a].
                    for mu in 0..na {
                        for nu in 0..na {
                            let mut acc = 0.0;
                            for la in 0..nb {
                                for si in 0..nb {
                                    acc += onsite[(ob + la, ob + si)]
                                        * te1.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            if subtract && mu == nu {
                                acc -= dv[axis] * pop_b;
                                acc += dv[axis] * eb.core_charge;
                            }
                            target.add([0, 0, 0], oa + mu, oa + nu, scale(acc));
                        }
                    }
                    // Exchange into H(T)[a, b].
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acc = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    acc +=
                                        pt[(oa + nu, ob + si)] * te1.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            if subtract {
                                acc -= pt[(oa + mu, ob + la)] * dv[axis];
                            }
                            target.add(t, oa + mu, ob + la, scale(-0.5 * acc));
                        }
                    }
                } else {
                    // The mirror blocks, whose row atom is b sitting in cell 0.
                    for i in 0..na {
                        let bi = crate::hessian::beta_of(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = crate::hessian::beta_of(eb, basis.aos[ob + j].orb);
                            let val = 0.5 * (bi + bj) * s1[i][j].d[axis];
                            target.add(neg_t, ob + j, oa + i, scale(val));
                        }
                    }
                    for k in 0..nb {
                        for l in 0..nb {
                            target.add([0, 0, 0], ob + k, ob + l, scale(te1.e2a[k][l].d[axis]));
                        }
                    }
                    for la in 0..nb {
                        for si in 0..nb {
                            let mut acc = 0.0;
                            for mu in 0..na {
                                for nu in 0..na {
                                    acc += onsite[(oa + mu, oa + nu)]
                                        * te1.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            if subtract && la == si {
                                acc -= dv[axis] * pop_a;
                                acc += dv[axis] * ea.core_charge;
                            }
                            target.add([0, 0, 0], ob + la, ob + si, scale(acc));
                        }
                    }
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acc = 0.0;
                            for nu in 0..na {
                                for si in 0..nb {
                                    acc += pt_neg[(ob + si, oa + nu)]
                                        * te1.two_e(mu, nu, la, si).d[axis];
                                }
                            }
                            if subtract {
                                acc -= pt_neg[(ob + la, oa + mu)] * dv[axis];
                            }
                            target.add(neg_t, ob + la, oa + mu, scale(-0.5 * acc));
                        }
                    }
                }
            }
        }

        // --- second derivatives, for the skeleton ---------------------------------------
        let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
        let (te2, s2) = if far {
            (
                crate::integrals::point_charge_pair_dual2(ea, eb, d),
                [[Dual2::constant(0.0); 9]; 9],
            )
        } else {
            let (te, s, _) = crate::hessian::pair_dual2_at(ea, eb, d, has_any_d)?;
            (te, s)
        };
        let mut epair = Dual2::constant(0.0);
        for i in 0..na {
            let bi = crate::hessian::beta_of(ea, basis.aos[oa + i].orb);
            for j in 0..nb {
                let bj = crate::hessian::beta_of(eb, basis.aos[ob + j].orb);
                epair = epair + s2[i][j] * (pt[(oa + i, ob + j)] * (bi + bj));
            }
        }
        for i in 0..na {
            for j in 0..na {
                epair = epair + te2.e1b[i][j] * onsite[(oa + i, oa + j)];
            }
        }
        for k in 0..nb {
            for l in 0..nb {
                epair = epair + te2.e2a[k][l] * onsite[(ob + k, ob + l)];
            }
        }
        for mu in 0..na {
            for nu in 0..na {
                for la in 0..nb {
                    for si in 0..nb {
                        let coul = onsite[(oa + mu, oa + nu)] * onsite[(ob + la, ob + si)];
                        let exch = -0.5 * pt[(oa + mu, ob + la)] * pt[(oa + nu, ob + si)];
                        epair = epair + te2.two_e(mu, nu, la, si) * (coul + exch);
                    }
                }
            }
        }
        let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
        if subtract {
            let exch_mono: f64 = (0..na)
                .flat_map(|mu| (0..nb).map(move |la| (mu, la)))
                .map(|(mu, la)| {
                    let x = pt[(oa + mu, ob + la)];
                    -0.5 * x * x
                })
                .sum();
            let c = -(pop_a * pop_b + exch_mono) + eb.core_charge * pop_a + ea.core_charge * pop_b;
            epair = epair + r.recip() * (c * crate::constants::PM7_EV);
        }
        epair = epair
            + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                ea,
                eb,
                molecule.atoms[a].z,
                molecule.atoms[b].z,
                r,
                params,
            );
        if subtract {
            let zz = ea.core_charge * eb.core_charge * crate::constants::PM7_EV;
            epair = epair - r.recip() * zz;
        }
        scatter_skeleton(&mut skeleton, a, b, &epair.h, pair.weight, phase);
    }

    // --- the long-range monopole, phased ---------------------------------------------------
    add_long_range(
        molecule,
        basis,
        scf,
        pbc,
        q,
        &mut bare,
        &mut skeleton,
        density,
        long_range,
    )?;
    Ok((bare, skeleton))
}

/// Scatter one pair's 3×3 second-derivative block into `D(q)` with the phases a wavevector
/// demands.
///
/// The diagonal entries carry no phase and the off-diagonal ones carry `e^{±i q·T}`. A self-image
/// pair, which cancels to nothing at `q = 0`, contributes `2h(1 − cos q·T)` away from it — an atom
/// really does feel its own images when they move out of step with it.
fn scatter_skeleton(
    out: &mut CMatrix,
    a: usize,
    b: usize,
    block: &[[f64; 3]; 3],
    weight: f64,
    phase: [f64; 2],
) {
    let conj = [phase[0], -phase[1]];
    for (i, row) in block.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let w = weight * value;
            let bump = |m: &mut CMatrix, r: usize, c: usize, v: [f64; 2]| {
                let (re, im) = m.get(r, c);
                m.set(r, c, re + v[0], im + v[1]);
            };
            bump(out, 3 * a + i, 3 * a + j, [w, 0.0]);
            bump(out, 3 * b + i, 3 * b + j, [w, 0.0]);
            bump(out, 3 * a + i, 3 * b + j, cmul([-w, 0.0], phase));
            bump(out, 3 * b + i, 3 * a + j, cmul([-w, 0.0], conj));
        }
    }
}

/// Everything the long-range monopole contributes: the phased lattice sums the pair loop left out.
///
/// Two different sums are needed and confusing them is the classic error. The **Coulomb** couples
/// the on-site block of `A` to that of `B` in cell `S`, and the on-site block at cell `S` carries
/// the perturbation phase, so the sum is phased: `∇Φ_q`. Displacing `A` itself moves the row
/// atom's own cell, which carries no phase, so that half uses `∇Φ_0`. The **exchange** couples the
/// `(0, T)` block, and within one Born–von Kármán class the phase factors out — leaving the
/// supercell sums the ground state already has, and their phased derivatives.
#[allow(clippy::too_many_arguments)]
fn add_long_range(
    molecule: &Molecule,
    basis: &Basis,
    scf: &Pm7Result,
    pbc: &PbcOptions,
    q: Vec3,
    bare: &mut [ComplexBlocks],
    skeleton: &mut CMatrix,
    density: &BlochBlocks,
    long_range: LongRange,
) -> Result<()> {
    // All three sites the term touches are switched together; see [`LongRange`].
    match long_range {
        LongRange::Off => return Ok(()),
        LongRange::Require if pbc.mode != PbcMode::Ewald => {
            return Err(Pm7Error::InvalidInput(format!(
                "LongRange::Require asks for the long-range monopole term, and the {:?} periodic \
                 mode has no lattice sum to take it from. Use PbcMode::Ewald, or LongRange::Auto \
                 to accept its absence.",
                pbc.mode
            )))
        }
        _ => {}
    }
    if pbc.mode != PbcMode::Ewald {
        return Ok(());
    }
    let cell = molecule.cell.expect("periodic");
    let nat = molecule.atoms.len();
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let charges = &scf.charges;
    let divisions = pbc.kmesh.divisions();

    // --- Coulomb, on the primitive lattice ---------------------------------------------
    let ep = crate::pbc::EwaldParameters::new(&cell, nat, pbc.ewald_accuracy, pbc.ewald_alpha);
    let displacements: Vec<Vec3> = (0..nat)
        .flat_map(|a| (0..nat).map(move |b| (a, b)))
        .map(|(a, b)| positions[b] - positions[a])
        .collect();
    let phased = crate::pbc::ewald::ewald_phased(&cell, &displacements, q, &ep);
    let plain = crate::pbc::ewald::ewald_phased(&cell, &displacements, Vec3::zero(), &ep);

    for ia in 0..nat {
        let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
        for ib in 0..nat {
            let slot = ia * nat + ib;
            for axis in 0..3 {
                // ∂(−V_A)/∂u_C, with C = B carrying the phase and C = A not.
                let from_b = phased[slot].gradient[axis];
                let from_a = plain[slot].gradient[axis][0];
                let value_b = [-charges[ib] * from_b[0], -charges[ib] * from_b[1]];
                let value_a = [charges[ib] * from_a, 0.0];
                for mu in 0..na {
                    bare[3 * ib + axis].add([0, 0, 0], oa + mu, oa + mu, value_b);
                    bare[3 * ia + axis].add([0, 0, 0], oa + mu, oa + mu, value_a);
                }
            }
            // Skeleton: ½ Σ_AB q_A q_B ∂²Φ_q/∂d∂d, scattered as a pair term.
            let c = 0.5 * charges[ia] * charges[ib];
            for i in 0..3 {
                for j in 0..3 {
                    let h = phased[slot].hessian[i][j];
                    let h0 = plain[slot].hessian[i][j][0];
                    let bump = |m: &mut CMatrix, r: usize, cc: usize, v: [f64; 2]| {
                        let (re, im) = m.get(r, cc);
                        m.set(r, cc, re + v[0], im + v[1]);
                    };
                    // The `AA`/`BB` diagonals are unphased; the cross terms carry `Φ_q`.
                    bump(skeleton, 3 * ia + i, 3 * ia + j, [c * h0, 0.0]);
                    bump(skeleton, 3 * ib + i, 3 * ib + j, [c * h0, 0.0]);
                    bump(skeleton, 3 * ia + i, 3 * ib + j, [-c * h[0], -c * h[1]]);
                    bump(skeleton, 3 * ib + i, 3 * ia + j, [-c * h[0], c * h[1]]);
                }
            }
        }
    }

    // --- exchange, per Born–von Kármán residue class on the supercell -------------------
    let classes = crate::hamiltonian::bvk_representatives(divisions);
    let (super_cell, _) = cell.supercell(divisions)?;
    let super_ep = crate::pbc::EwaldParameters::new(
        &super_cell,
        nat * classes.len(),
        pbc.ewald_accuracy,
        pbc.ewald_alpha,
    );
    for t in &classes {
        let shift = cell.translation(*t);
        let angle = q.dot(shift);
        let class_phase = [angle.cos(), angle.sin()];
        let shifted: Vec<Vec3> = (0..nat)
            .flat_map(|a| (0..nat).map(move |b| (a, b)))
            .map(|(a, b)| positions[b] + shift - positions[a])
            .collect();
        let ph = crate::pbc::ewald::ewald_phased(&super_cell, &shifted, q, &super_ep);
        let pl = crate::pbc::ewald::ewald_phased(&super_cell, &shifted, Vec3::zero(), &super_ep);
        let pt = density
            .folded(*t, divisions)
            .expect("the density covers the residue classes");

        for ia in 0..nat {
            let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
            for ib in 0..nat {
                let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
                let slot = ia * nat + ib;
                // The same-spin density is half the total for a closed shell, and the two spins
                // put the factor back, so the exchange coefficient carries a plain ½.
                for axis in 0..3 {
                    let db = cmul(class_phase, ph[slot].gradient[axis]);
                    let da = -pl[slot].gradient[axis][0];
                    for mu in 0..na {
                        for la in 0..nb {
                            let p = 0.5 * pt[(oa + mu, ob + la)];
                            bare[3 * ib + axis].add(*t, oa + mu, ob + la, [-p * db[0], -p * db[1]]);
                            bare[3 * ia + axis].add(*t, oa + mu, ob + la, [-p * da, 0.0]);
                        }
                    }
                }
                // Skeleton: `−¼ Σ_{μλ} P(t)² ∂²Φ/∂d∂d`, from `E_x = −¼ Σ P² Φ`.
                let coefficient: f64 = (0..na)
                    .flat_map(|mu| (0..nb).map(move |la| (mu, la)))
                    .map(|(mu, la)| {
                        let x = pt[(oa + mu, ob + la)];
                        -0.25 * x * x
                    })
                    .sum();
                for i in 0..3 {
                    for j in 0..3 {
                        let h = cmul(class_phase, ph[slot].hessian[i][j]);
                        let h0 = pl[slot].hessian[i][j][0];
                        let bump = |m: &mut CMatrix, r: usize, cc: usize, v: [f64; 2]| {
                            let (re, im) = m.get(r, cc);
                            m.set(r, cc, re + v[0], im + v[1]);
                        };
                        bump(skeleton, 3 * ia + i, 3 * ia + j, [coefficient * h0, 0.0]);
                        bump(skeleton, 3 * ib + i, 3 * ib + j, [coefficient * h0, 0.0]);
                        bump(
                            skeleton,
                            3 * ia + i,
                            3 * ib + j,
                            [-coefficient * h[0], -coefficient * h[1]],
                        );
                        bump(
                            skeleton,
                            3 * ib + i,
                            3 * ia + j,
                            [-coefficient * h[0], coefficient * h[1]],
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// The phased long-range Coulomb kernel `Φ_q(R_b − R_a)`, for every ordered atom pair.
///
/// One `nat × nat` table of complex numbers, built **once per wavevector** and read by every
/// response Fock. Through v0.2.1 [`fock_response_q`] rebuilt it on entry: fresh
/// [`EwaldParameters`], a fresh `nat²` displacement list, and a fresh [`ewald_phased`] over every
/// real-space image and every reciprocal vector — on each of `3N × iterations × spins` calls, for a
/// geometry and a `q` that do not change once `dynamical_matrix_dfpt` has started. `PM7_PROFILE=1`
/// put that at **64.6 %** of a phonon run.
///
/// It also asked for twelve things it did not want: [`ewald_phased`] returns the value, three
/// gradient components and nine Hessian components, and the kernel reads `value` alone. Computing
/// them once makes that a rounding error rather than a reason to write a second, near-duplicate
/// lattice sum whose agreement with the first nothing would check.
pub(crate) struct LongRangeQ {
    /// Row-major `nat × nat`, `[re, im]`, in the units [`ewald_phased`] returns.
    values: Vec<[f64; 2]>,
    nat: usize,
}

impl LongRangeQ {
    #[inline]
    fn get(&self, a: usize, b: usize) -> [f64; 2] {
        debug_assert!(a < self.nat && b < self.nat);
        self.values[a * self.nat + b]
    }
}

/// One pair's constants: which blocks it writes, and the Bloch phase it carries.
#[derive(Clone, Copy)]
struct PairTables {
    /// Block index of the pair's translation `T`, `None` when it is outside the set.
    forward: Option<usize>,
    /// Block index of `−T`.
    backward: Option<usize>,
    /// `e^{i q·T}`.
    phase: [f64; 2],
}

/// Everything the response Fock needs that is a function of `(geometry, q)` alone.
///
/// Built **once per wavevector**, in place of the three things the kernel used to recompute on
/// each of its `3N × iterations × spins` calls:
///
/// * the phased long-range Coulomb table `Φ_q(R_b − R_a)` — a fresh [`EwaldParameters`], a fresh
///   `nat²` displacement list and a full lattice sum, of which only `value` was read (the other
///   twelve components, three gradients and nine Hessian entries, were computed and discarded);
/// * `e^{i q·T}` per pair — two transcendentals for a `T` shared by many pairs;
/// * the block index of `T` and of `−T` — a `HashMap` lookup **per matrix element**, which
///   `PM7_PROFILE=1` showed was a large part of the pair loop's 74 % share of the kernel.
///
/// The pair list and the block set are both fixed for the duration of a solve, so all three are
/// table lookups here. The arithmetic is unchanged and in the same order, so results are
/// bit-identical.
pub(crate) struct ResponseTables {
    long_range: Option<LongRangeQ>,
    /// Indexed by position in `core.pairs`.
    pairs: Vec<PairTables>,
    /// Block index of `T = 0`, which every on-site Coulomb write lands in.
    zero: Option<usize>,
}

impl ResponseTables {
    fn build(
        molecule: &Molecule,
        core: &CoreHamiltonian,
        pbc: &PbcOptions,
        q: Vec3,
        translations: &[[i32; 3]],
        long_range_on: bool,
    ) -> Option<Self> {
        let cell = molecule.cell?;
        let index: std::collections::HashMap<[i32; 3], usize> = translations
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, i))
            .collect();
        let nat = molecule.atoms.len();
        let long_range = (long_range_on && pbc.mode == PbcMode::Ewald).then(|| {
            let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
            let ep =
                crate::pbc::EwaldParameters::new(&cell, nat, pbc.ewald_accuracy, pbc.ewald_alpha);
            let displacements: Vec<Vec3> = (0..nat)
                .flat_map(|a| (0..nat).map(move |b| (a, b)))
                .map(|(a, b)| positions[b] - positions[a])
                .collect();
            let phased = crate::pbc::ewald::ewald_phased(&cell, &displacements, q, &ep);
            LongRangeQ {
                values: phased.iter().map(|k| k.value).collect(),
                nat,
            }
        });
        let pairs = core
            .pairs
            .iter()
            .map(|pair| {
                let t = pair.t;
                let angle = q.dot(cell.translation(t));
                PairTables {
                    forward: index.get(&t).copied(),
                    backward: index.get(&[-t[0], -t[1], -t[2]]).copied(),
                    phase: [angle.cos(), angle.sin()],
                }
            })
            .collect();
        Some(Self {
            long_range,
            pairs,
            zero: index.get(&[0, 0, 0]).copied(),
        })
    }
}

/// The two-electron response of the Fock to a density change at wavevector `q`.
///
/// This is the linear kernel `K[Δp]` — no core Hamiltonian, no nuclear charges, nothing that does
/// not move with the density. It mirrors [`crate::fock::build_fock_spin_bloch`] term for term, and
/// the only additions are the `e^{±i q·T}` factors on the **Coulomb** couplings. The exchange gets
/// none: it connects the `(0, T)` block, whose perturbation prefactor is `e^{i q·0}`.
///
/// # Spin
///
/// `delta_p` is the **total** response and `delta_same` the **same-spin** one, which is what the
/// two halves of the kernel see: Coulomb couples a spin channel to the whole density, exchange
/// only to its own. This is one function for RHF and UHF rather than two, because the difference
/// between them is entirely in what those two arguments are:
///
/// * closed shell — `delta_same = delta_p / 2`, which reproduces the `½` that used to be written
///   into the exchange terms as a literal;
/// * unrestricted — `delta_same = delta_p_alpha` or `delta_p_beta`, with `delta_p` their sum.
///
/// Writing it as two arguments rather than a `½` makes the RHF path bit-identical while giving the
/// UHF one somewhere to put a genuinely different same-spin density.
#[allow(clippy::too_many_arguments)]
fn fock_response_q(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    core: &CoreHamiltonian,
    tables: &ResponseTables,
    delta_p: &ComplexBlocks,
    delta_same: &ComplexBlocks,
) -> Result<ComplexBlocks> {
    let nao = basis.nao;
    let nat = molecule.atoms.len();
    let mut out = ComplexBlocks::new(delta_p.translations().to_vec(), nao)?;
    let (p0r, p0i) = delta_p.onsite();

    let (psr, psi) = delta_same.onsite();

    // One-centre: local, so no phase. The kernel is linear, so the real and imaginary parts go
    // through the same routine independently.
    let _t_one = crate::profile::stage("dfpt:   kernel one-centre");
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let br = crate::fock::one_centre_block(elem, off, n, p0r, psr);
        let bi = crate::fock::one_centre_block(elem, off, n, p0i, psi);
        if let Some(zero_block) = tables.zero {
            for mu in 0..n {
                for nu in 0..n {
                    out.add_at(
                        zero_block,
                        off + mu,
                        off + nu,
                        [br[mu * n + nu], bi[mu * n + nu]],
                    );
                }
            }
        }
    }

    drop(_t_one);
    let _t_two = crate::profile::stage("dfpt:   kernel two-centre pairs");
    for (pair, entry) in core.pairs.iter().zip(&tables.pairs) {
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
        let (na, nb) = (basis.atom_norb[pair.a], basis.atom_norb[pair.b]);
        let phase = entry.phase;
        let conj = [phase[0], -phase[1]];

        // Coulomb: A's on-site block sees B's density in cell T (phase `e^{+i q·T}`), and B's
        // sees A's in cell −T (phase `e^{−i q·T}`).
        let pop = |m: &Matrix, off: usize, n: usize| -> f64 {
            (0..n).map(|k| m[(off + k, off + k)]).sum()
        };
        let (pop_b_r, pop_b_i) = (pop(p0r, ob, nb), pop(p0i, ob, nb));
        let (pop_a_r, pop_a_i) = (pop(p0r, oa, na), pop(p0i, oa, na));
        if let Some(zero_block) = tables.zero {
            for mu in 0..na {
                for nu in 0..na {
                    let mut acc = [0.0, 0.0];
                    for la in 0..nb {
                        for si in 0..nb {
                            let w = te.two_e(mu, nu, la, si);
                            acc[0] += p0r[(ob + la, ob + si)] * w;
                            acc[1] += p0i[(ob + la, ob + si)] * w;
                        }
                    }
                    if pair.v_point != 0.0 && mu == nu {
                        acc[0] -= pair.v_point * pop_b_r;
                        acc[1] -= pair.v_point * pop_b_i;
                    }
                    out.add_at(zero_block, oa + mu, oa + nu, cmul(acc, phase));
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    let mut acc = [0.0, 0.0];
                    for mu in 0..na {
                        for nu in 0..na {
                            let w = te.two_e(mu, nu, la, si);
                            acc[0] += p0r[(oa + mu, oa + nu)] * w;
                            acc[1] += p0i[(oa + mu, oa + nu)] * w;
                        }
                    }
                    if pair.v_point != 0.0 && la == si {
                        acc[0] -= pair.v_point * pop_a_r;
                        acc[1] -= pair.v_point * pop_a_i;
                    }
                    out.add_at(zero_block, ob + la, ob + si, cmul(acc, conj));
                }
            }
        }

        // Exchange, unphased: it connects the `(0, T)` block, whose perturbation prefactor is
        // `e^{i q·0}`.
        //
        // The `T` and `−T` blocks are built from **their own** response densities. The ground
        // state can build both from one block, because there `P` is real and `P(−T) = P(T)ᵀ`, so
        // the second contribution is the transpose of the first and the same number serves twice.
        // A response *amplitude* at wavevector `q` obeys no such relation: `Δp(T)` is
        // `Σ_k w_k e^{−ik·T} ΔP(k)` with `ΔP(k)` an off-diagonal `(k+q ← k)` object, and its
        // adjoint runs `(k ← k+q)`, so `Δp(−T) ≠ Δp(T)†` unless `q` is zero.
        //
        // Reusing one block for both wrote a Fock response whose `F(−T)` did not match its `F(T)`,
        // which made `F(k)` non-Hermitian and, through the response, `D(q)` non-Hermitian too —
        // but **only where the phases are genuinely complex**. At `q = 0` and `q = ½` the blocks
        // are real and the two forms coincide, which is exactly the set the tests covered, so this
        // survived from v0.2.0 while every general `q` came back wrong and then got symmetrized on
        // the way out.
        // Exchange reads the **same-spin** response, so there is no `½` here: for a closed shell
        // the caller passes `Δp/2` and the arithmetic is identical to what the literal did.
        let Some(forward) = entry.forward else {
            continue;
        };
        let (ptr, pti) = delta_same.at(forward);
        let negative = entry.backward.map(|index| delta_same.at(index));
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = [0.0, 0.0];
                for nu in 0..na {
                    for si in 0..nb {
                        let w = te.two_e(mu, nu, la, si);
                        acc[0] += ptr[(oa + nu, ob + si)] * w;
                        acc[1] += pti[(oa + nu, ob + si)] * w;
                    }
                }
                if pair.v_point != 0.0 {
                    acc[0] -= ptr[(oa + mu, ob + la)] * pair.v_point;
                    acc[1] -= pti[(oa + mu, ob + la)] * pair.v_point;
                }
                out.add_at(forward, oa + mu, ob + la, [-acc[0], -acc[1]]);

                if let Some((pnr, pni)) = negative {
                    // Same contraction with the two ends exchanged, reading `Δp(−T)`.
                    let mut back = [0.0, 0.0];
                    for nu in 0..na {
                        for si in 0..nb {
                            let w = te.two_e(mu, nu, la, si);
                            back[0] += pnr[(ob + si, oa + nu)] * w;
                            back[1] += pni[(ob + si, oa + nu)] * w;
                        }
                    }
                    if pair.v_point != 0.0 {
                        back[0] -= pnr[(ob + la, oa + mu)] * pair.v_point;
                        back[1] -= pni[(ob + la, oa + mu)] * pair.v_point;
                    }
                    out.add_at(
                        entry.backward.expect("negative block resolved above"),
                        ob + la,
                        oa + mu,
                        [-back[0], -back[1]],
                    );
                }
            }
        }
    }

    drop(_t_two);
    // Long-range Coulomb, phased over the primitive lattice. The kernel table is built once per
    // wavevector by `LongRangeQ::build`, not once per call; see its documentation.
    if let Some(phased) = tables.long_range.as_ref() {
        let _t_lr = crate::profile::stage("dfpt:   kernel long-range Coulomb");
        for ia in 0..nat {
            let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
            let mut potential = [0.0, 0.0];
            for ib in 0..nat {
                let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
                let dp = [
                    (0..nb).map(|k| p0r[(ob + k, ob + k)]).sum::<f64>(),
                    (0..nb).map(|k| p0i[(ob + k, ob + k)]).sum::<f64>(),
                ];
                let m = phased.get(ia, ib);
                let term = cmul(m, dp);
                potential[0] += term[0];
                potential[1] += term[1];
            }
            for mu in 0..na {
                out.add([0, 0, 0], oa + mu, oa + mu, potential);
            }
        }

        drop(_t_lr);
        // Long-range exchange, per residue class, using the class sums the ground state built.
        let _t_ex = crate::profile::stage("dfpt:   kernel BvK exchange");
        if let Some(ew) = &core.ewald {
            if let Some(bvk) = &ew.exchange {
                for (class, t) in bvk.translations.iter().enumerate() {
                    let values = &bvk.values[class];
                    // Long-range exchange, like the short-range kind, sees only its own spin.
                    let Some((ptr, pti)) = delta_same.get(*t) else {
                        continue;
                    };
                    for ia in 0..nat {
                        let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
                        for ib in 0..nat {
                            let m = values[ia][ib];
                            if m == 0.0 {
                                continue;
                            }
                            let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
                            for mu in 0..na {
                                for la in 0..nb {
                                    out.add(
                                        *t,
                                        oa + mu,
                                        ob + la,
                                        [
                                            -ptr[(oa + mu, ob + la)] * m,
                                            -pti[(oa + mu, ob + la)] * m,
                                        ],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Diagonalize the converged Fock at every `k` and `k + q`, and fill both from one chemical
/// potential.
///
/// Shared by the phonon and field paths so they cannot disagree about which states the response
/// is built on. The `k` set and the `k + q` set are the same Brillouin zone sampled from two
/// origins, so filling them together at half weight each leaves the electron count right and
/// gives one `E_F` both are consistent with; filling each k by band index would be an
/// aufbau-per-k assumption, wrong the moment a band crosses `E_F` between mesh points.
/// One band set per spin channel: a single entry for a closed shell, two for an unrestricted one.
///
/// `focks` and `electrons` are per spin and must agree in length. `per_orbital` is the occupancy of
/// a filled orbital — `2` when one Fock describes both spins, `1` when each channel is its own.
/// Each channel is filled from **its own** chemical potential, which is right for both: with
/// integer occupations the response moves no electrons across `E_F`, so the shared ground-state
/// `E_F` never enters the response, and a UHF system whose two channels have different Fermi
/// levels is perfectly ordinary.
#[allow(clippy::too_many_arguments)]
fn build_bands(
    kpoints: &[KPoint],
    focks: &[&crate::scf_pbc::BlochBlocks],
    electrons: &[f64],
    per_orbital: f64,
    pbc: &PbcOptions,
    q_frac: [f64; 3],
    q_cart: Vec3,
) -> Result<Vec<Vec<BandPair>>> {
    debug_assert_eq!(focks.len(), electrons.len());
    let mut channels = Vec::with_capacity(focks.len());
    for (converged_fock, &n_elec) in focks.iter().zip(electrons) {
        let mut raw: Vec<(KPoint, Vec<f64>, CMatrix, Vec<f64>, CMatrix)> =
            Vec::with_capacity(kpoints.len());
        for k in kpoints {
            let kq = KPoint {
                frac: [
                    k.frac[0] + q_frac[0],
                    k.frac[1] + q_frac[1],
                    k.frac[2] + q_frac[2],
                ],
                cart: k.cart + q_cart,
                weight: k.weight,
                time_reversal_pair: false,
            };
            let (eps_k, c_k) = converged_fock.at_k(k).hermitian_eigen()?;
            let (eps_kq, c_kq) = converged_fock.at_k(&kq).hermitian_eigen()?;
            raw.push((*k, eps_k, c_k, eps_kq, c_kq));
        }

        // One chemical potential for the whole mesh, and for the shifted mesh with it.
        //
        // The `k` set and the `k + q` set are the same Brillouin zone sampled from two origins, so
        // filling them together at half weight each leaves the electron count right and produces
        // one `E_F` that both sets are consistent with. Filling each k independently by band index
        // would be an aufbau-per-k assumption, which is wrong the moment a band crosses `E_F`
        // between mesh points — and wrong silently, because the response still converges.
        let mut energies: Vec<Vec<f64>> = Vec::with_capacity(2 * raw.len());
        let mut weights: Vec<f64> = Vec::with_capacity(2 * raw.len());
        let half_weight = 0.5 / raw.len() as f64;
        for (_, eps_k, _, _, _) in &raw {
            energies.push(eps_k.clone());
            weights.push(half_weight);
        }
        for (_, _, _, eps_kq, _) in &raw {
            energies.push(eps_kq.clone());
            weights.push(half_weight);
        }
        let filled =
            crate::scf_pbc::fill_bands(&energies, &weights, n_elec, per_orbital, pbc.smearing)?;
        refuse_gapless_without_smearing(
            &filled.occupations,
            &energies,
            filled.fermi_ev,
            pbc.smearing,
        )?;
        refuse_partial_occupations_at_zone_centre(
            &filled.occupations,
            filled.entropy_ev,
            filled.fermi_ev,
            q_frac,
        )?;

        let n_k = raw.len();
        let mut bands = Vec::with_capacity(n_k);
        for (slot, (k, eps_k, c_k, eps_kq, c_kq)) in raw.into_iter().enumerate() {
            // Electrons per filled orbital, matching the normalization of the density this
            // channel's response carries.
            let occ_k = filled.occupations[slot]
                .iter()
                .map(|f| per_orbital * f)
                .collect();
            let occ_kq = filled.occupations[n_k + slot]
                .iter()
                .map(|f| per_orbital * f)
                .collect();
            bands.push(BandPair {
                k,
                eps_k,
                c_k,
                eps_kq,
                c_kq,
                occ_k,
                occ_kq,
            });
        }
        channels.push(bands);
    }
    Ok(channels)
}

/// The converged Fock per spin channel: one for a closed shell, two for an unrestricted one.
///
/// `build_fock_spin_bloch` takes the **total** density and the **same-spin** one, so a closed shell
/// passes `(P, P/2)` and an unrestricted cell `(P, (P ± Δ)/2)` — exactly the pair the SCF itself
/// converged on, which is what makes the response expand on the right orbitals.
pub(crate) fn converged_spin_focks(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    scf: &Pm7Result,
    density: &crate::scf_pbc::BlochBlocks,
) -> Result<Vec<crate::scf_pbc::BlochBlocks>> {
    match scf.bloch_spin_density.as_ref() {
        Some(spin) if scf.unrestricted => {
            let alpha =
                crate::scf_pbc::scale_blocks(&crate::scf_pbc::add_blocks(density, spin), 0.5);
            let beta =
                crate::scf_pbc::scale_blocks(&crate::scf_pbc::sub_blocks(density, spin), 0.5);
            Ok(vec![
                crate::fock::build_fock_spin_bloch(molecule, basis, params, core, density, &alpha)?,
                crate::fock::build_fock_spin_bloch(molecule, basis, params, core, density, &beta)?,
            ])
        }
        _ => {
            let half = crate::scf_pbc::scale_blocks(density, 0.5);
            Ok(vec![crate::fock::build_fock_spin_bloch(
                molecule, basis, params, core, density, &half,
            )?])
        }
    }
}

/// The `(alpha, beta)` electron counts, or `(total,)` for a closed shell.
fn spin_electron_counts(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    unrestricted: bool,
) -> Result<Vec<f64>> {
    let total = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.core_charge))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<f64>()
        - options.charge;
    if !unrestricted {
        return Ok(vec![total]);
    }
    // `multiplicity = 2S + 1`, so the excess of alpha over beta is `multiplicity - 1`.
    let excess = options.multiplicity as f64 - 1.0;
    let beta = (total - excess) / 2.0;
    Ok(vec![beta + excess, beta])
}

/// Pulay acceleration for the response fixed point.
///
/// The Sternheimer equation is **linear**, so plain iteration converges geometrically at whatever
/// the coupling's spectral radius happens to be — 0.86 at the zone centre of a CH₂ chain and 0.96
/// at its zone boundary, which is 500 iterations for ten digits. DIIS on a linear system converges
/// in about as many steps as the subspace is deep, so it turns that into a dozen.
struct ResponseDiis {
    inputs: Vec<ComplexBlocks>,
    images: Vec<ComplexBlocks>,
    errors: Vec<ComplexBlocks>,
    depth: usize,
}

impl ResponseDiis {
    fn new(depth: usize) -> Self {
        Self {
            inputs: Vec::new(),
            images: Vec::new(),
            errors: Vec::new(),
            depth,
        }
    }

    fn dot(a: &ComplexBlocks, b: &ComplexBlocks) -> f64 {
        a.re.frobenius_dot(&b.re) + a.im.frobenius_dot(&b.im)
    }

    /// Given the trial `x` and its image `f(x)`, propose the next trial.
    fn next(&mut self, x: &ComplexBlocks, fx: &ComplexBlocks, fallback: f64) -> ComplexBlocks {
        let error = ComplexBlocks {
            re: crate::scf_pbc::sub_blocks(&fx.re, &x.re),
            im: crate::scf_pbc::sub_blocks(&fx.im, &x.im),
        };
        self.inputs.push(x.clone());
        self.images.push(fx.clone());
        self.errors.push(error);
        if self.images.len() > self.depth {
            self.inputs.remove(0);
            self.images.remove(0);
            self.errors.remove(0);
        }
        match self.coefficients() {
            Some(c) => {
                let mut out = ComplexBlocks::zeros_like(fx);
                for (weight, image) in c.iter().zip(&self.images) {
                    out.re = crate::scf_pbc::add_blocks(
                        &out.re,
                        &crate::scf_pbc::scale_blocks(&image.re, *weight),
                    );
                    out.im = crate::scf_pbc::add_blocks(
                        &out.im,
                        &crate::scf_pbc::scale_blocks(&image.im, *weight),
                    );
                }
                out
            }
            // Fall back to plain mixing, which can never do worse than not accelerating.
            None => fx.mixed(x, fallback),
        }
    }

    /// Solve `min ‖Σ cᵢ eᵢ‖` subject to `Σ cᵢ = 1`, dropping the oldest vectors while the weights
    /// look unreliable.
    fn coefficients(&self) -> Option<Vec<f64>> {
        let n = self.errors.len();
        if n < 2 {
            return None;
        }
        let mut gram = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = Self::dot(&self.errors[i], &self.errors[j]);
                gram[i * n + j] = v;
                gram[j * n + i] = v;
            }
        }
        for skip in 0..(n - 1) {
            let m = n - skip;
            let scale = (0..m)
                .map(|i| gram[(skip + i) * n + (skip + i)].abs())
                .fold(0.0_f64, f64::max);
            if !scale.is_finite() || scale <= 0.0 {
                continue;
            }
            let dim = m + 1;
            let mut a = vec![0.0; dim * dim];
            for i in 0..m {
                for j in 0..m {
                    a[i * dim + j] = gram[(skip + i) * n + (skip + j)] / scale;
                }
                a[i * dim + m] = -1.0;
                a[m * dim + i] = -1.0;
            }
            let mut rhs = vec![0.0; dim];
            rhs[m] = -1.0;
            if let Some(solution) = solve_small(&mut a, &mut rhs, dim) {
                let weight: f64 = solution[..m].iter().map(|v| v.abs()).sum();
                if solution.iter().all(|v| v.is_finite()) && weight <= 25.0 {
                    let mut c = vec![0.0; n];
                    c[skip..].copy_from_slice(&solution[..m]);
                    return Some(c);
                }
            }
        }
        None
    }
}

fn solve_small(a: &mut [f64], rhs: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= a[col * n + j] * x[j];
        }
        x[col] = sum / a[col * n + col];
    }
    Some(x)
}

/// Solve the self-consistent response for one degree of freedom.
///
/// Returns the real-space response density, the iteration count, whether it converged, and the
/// final residual.
#[allow(clippy::too_many_arguments)]
/// Born effective charges and the electronic dielectric tensor at `q = 0`.
///
/// Runs `3N + 3` self-consistent responses: one per nuclear displacement, and one per Cartesian
/// field direction. The displacement responses are the same objects a `q = 0` phonon calculation
/// produces, so the Born charges cost almost nothing on top of them; the dielectric tensor is
/// what the three field responses are for.
///
/// See [`DfptFieldResult`] for the sign and index conventions, which are not guessable and are
/// fixed in `docs/theory.md` (C-1 and C-5 to C-7).
pub fn born_and_dielectric(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dfpt: &DfptOptions,
) -> Result<DfptFieldResult> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput(
            "Born charges and a dielectric tensor need a periodic system; this molecule has no \
             cell. For a molecule, `ir::dipole_derivatives` is the corresponding quantity."
                .into(),
        )
    })?;
    let pbc = options.pbc_for(molecule).expect("cell implies pbc options");
    if pbc.mode != PbcMode::Ewald {
        return Err(Pm7Error::InvalidInput(
            "the field response needs the Ewald periodic mode".into(),
        ));
    }
    let mut pbc = pbc;
    if matches!(pbc.kmesh, crate::pbc::KMesh::Gamma) {
        pbc.kmesh = crate::pbc::KMesh::grid(1, 1, 1);
    }
    let pbc = pbc;
    let mut options = options.clone();
    options.pbc = Some(pbc.clone());
    let options = &options;

    let basis = Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        &basis,
        params,
        options.force_dpath,
        &pbc,
        options.active_field(),
    )?;
    let scf = crate::scf::run_pm7(molecule, params, options)?;
    // An unrestricted cell is supported here on the same terms as the phonons, but it takes more
    // than passing two densities: the field reaches the response through the commutator `[H, r]`,
    // and `H_alpha != H_beta`, so each channel needs its **own** commutator — and the `<r>·dP`
    // contraction has to be evaluated in each channel's own band basis and summed, which is a
    // different contraction from the spin-independent one the phonons use. Doing half of it (one
    // commutator serving both channels) converges and is wrong, which is why v0.2.1 refused rather
    // than approximated. All three are per spin below.
    let density = scf.bloch_density.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput("the field response needs a k-resolved density".into())
    })?;
    let core_bloch = core.bloch.as_ref().ok_or_else(|| {
        Pm7Error::InvalidInput("the field response needs the k-resolved core".into())
    })?;

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let translations: Vec<[i32; 3]> = core_bloch.translations().to_vec();
    let q_cart = Vec3::zero();

    // The displacement perturbations, exactly as a zone-centre phonon uses them.
    let (bare, _skeleton) = bare_and_skeleton(
        molecule,
        params,
        &basis,
        &core,
        &scf,
        density,
        &pbc,
        q_cart,
        &translations,
        dfpt.long_range,
    )?;

    let kpoints = response_mesh(&pbc.kmesh, &cell)?;
    let spin_focks = converged_spin_focks(molecule, &basis, params, &core, &scf, density)?;
    let electrons = spin_electron_counts(molecule, params, options, scf.unrestricted)?;
    let refs: Vec<&crate::scf_pbc::BlochBlocks> = spin_focks.iter().collect();
    let per_orbital = if scf.unrestricted { 1.0 } else { 2.0 };
    let channels = build_bands(
        &kpoints,
        &refs,
        &electrons,
        per_orbital,
        &pbc,
        [0.0; 3],
        q_cart,
    )?;

    // The field perturbations, through the commutator — **one per spin channel**.
    //
    // The position operator **r** is not a periodic operator, so a field reaches the response
    // through [H, r] (convention C-3). An unrestricted cell has two different H, so it has two
    // different commutators, and the <r>.dP contraction has to be evaluated in each channel's
    // own band basis and summed. Doing half of that -- one commutator serving both channels --
    // converges and is wrong, which is why v0.2.1 refused the unrestricted case outright rather
    // than approximating it.
    let commutators: Vec<[ComplexBlocks; 3]> = spin_focks
        .iter()
        .map(|fock| field_commutator_blocks(molecule, params, &basis, &cell, fock))
        .collect::<Result<Vec<_>>>()?;
    // Regrouped as [axis][spin], which is the shape BareTerm::Field takes: one solve per
    // Cartesian axis, carrying every channel's commutator for that axis.
    let by_axis: Vec<Vec<ComplexBlocks>> = (0..3)
        .map(|axis| commutators.iter().map(|c| c[axis].clone()).collect())
        .collect();

    let tables = {
        let _t = crate::profile::stage("dfpt: response tables at q");
        ResponseTables::build(
            molecule,
            &core,
            &pbc,
            q_cart,
            &translations,
            dfpt.long_range != LongRange::Off,
        )
        .expect("a periodic cell was checked above")
    };
    let solve = |term: BareTerm<'_>| -> Result<(Vec<Vec<CMatrix>>, usize, bool, f64)> {
        solve_one(
            term,
            &channels,
            molecule,
            params,
            &basis,
            &core,
            &tables,
            &translations,
            dfpt,
        )
    };

    use rayon::prelude::*;
    let displacement: Vec<(Vec<Vec<CMatrix>>, usize, bool, f64)> = (0..ndof)
        .into_par_iter()
        .map(|j| solve(BareTerm::Local(&bare[j])))
        .collect::<Result<Vec<_>>>()?;
    let field: Vec<(Vec<Vec<CMatrix>>, usize, bool, f64)> = (0..3)
        .into_par_iter()
        .map(|axis| solve(BareTerm::Field(&by_axis[axis])))
        .collect::<Result<Vec<_>>>()?;

    let mut iterations = 0;
    let mut converged = true;
    let mut residual = 0.0_f64;
    for (_, iters, ok, res) in displacement.iter().chain(field.iter()) {
        iterations = iterations.max(*iters);
        converged &= ok;
        residual = residual.max(*res);
    }

    // Born charges. The **explicit** term is the net Mulliken charge, not the core charge: the
    // field's one-electron term `-(f.R_A)` is bilinear in `f` and `R_A`, so the cross derivative
    // keeps `-n_A delta_ab` alongside `+Z_A delta_ab`. Dropping it would break the acoustic sum
    // rule — and, because it is common to both contraction orders, the "both orders agree" check
    // would not notice.
    let mut born = Vec::with_capacity(nat);
    for atom in 0..nat {
        let mut z = crate::math::Mat3::zero();
        for b in 0..3 {
            let response = &displacement[3 * atom + b].0;
            let electronic = dipole_response(&channels, &commutators, response);
            for a in 0..3 {
                let explicit = if a == b { scf.charges[atom] } else { 0.0 };
                z.set(a, b, explicit + electronic[a]);
            }
        }
        born.push(z);
    }

    // The raw response `d(mu_a)/d(f_b)` per cell, before any volume factor. This is the quantity
    // the field solve actually produces; `eps` is a 3-D reading of it. Keeping it separate is not
    // bookkeeping — in one and two dimensions `eps` is not defined at all (there is no `Omega`)
    // while this tensor still is, and it is the only part a finite field can be checked against.
    let mut polarizability = crate::math::Mat3::zero();
    for b in 0..3 {
        let electronic = dipole_response(&channels, &commutators, &field[b].0);
        for a in 0..3 {
            polarizability.set(a, b, electronic[a]);
        }
    }

    // Dielectric tensor: `eps_ab = delta_ab - (4 pi / Omega) d(mu_a)/d(f_b)`.
    let volume_bohr3 = cell.volume().unwrap_or(0.0);
    let mut dielectric = crate::math::Mat3::identity();
    if volume_bohr3 > 0.0 {
        let scale = 4.0 * std::f64::consts::PI / volume_bohr3;
        for b in 0..3 {
            for a in 0..3 {
                let base = if a == b { 1.0 } else { 0.0 };
                dielectric.set(a, b, base - scale * polarizability.get(a, b));
            }
        }
    }

    check_converged("electric-field", dfpt, converged, iterations, residual)?;

    Ok(DfptFieldResult {
        born,
        dielectric,
        polarizability,
        volume_bohr3,
        iterations,
        converged,
        residual,
    })
}

/// `Tr[D_a dP]` per Cartesian axis, contracted in the **band** basis.
///
/// It cannot be done in the AO basis. `D_a = -r_a` has an unbounded `R_A` diagonal, so an AO trace
/// against it is origin- and cell-choice-dependent and means nothing (convention C-3 / A4.2). In
/// the band basis the position operator's *interband* elements are well defined through the
/// commutator, and `dP` is purely interband because it carries a factor `f_n - f_m`, so the trace
/// is finite and unambiguous.
fn dipole_response(
    channels: &[Vec<BandPair>],
    commutators: &[[ComplexBlocks; 3]],
    per_spin_per_k: &[Vec<CMatrix>],
) -> [f64; 3] {
    debug_assert_eq!(channels.len(), commutators.len());
    debug_assert_eq!(channels.len(), per_spin_per_k.len());
    let mut out = [0.0_f64; 3];
    // Each channel contracts **its own** response density against **its own** position operator,
    // in **its own** band basis, and the dipole is the sum. All three have to be per spin: the
    // commutator because `H_α ≠ H_β`, the band basis because each channel has its own orbitals,
    // and the response because the field moves the two channels differently. For a closed shell
    // there is one channel carrying both spins through `per_orbital = 2`, and this reduces to what
    // it always was.
    for ((bands, commutator), per_k) in channels.iter().zip(commutators).zip(per_spin_per_k) {
        let weight = 1.0 / bands.len() as f64;
        for (slot, band) in bands.iter().enumerate() {
            // Back into the band basis: `C† dP_AO C`, exact because `C` is unitary.
            let rho = band.c_k.adjoint().matmul(&per_k[slot].matmul(&band.c_k));
            let n = rho.n;
            for (axis, slot_out) in out.iter_mut().enumerate() {
                let v = commutator[axis].at_k(&band.k);
                let projected = band.c_k.adjoint().matmul(&v.matmul(&band.c_k));
                let mut acc = 0.0;
                for m in 0..n {
                    for p in 0..n {
                        let de = band.eps_k[m] - band.eps_k[p];
                        if de.abs() < 1.0e-8 {
                            continue;
                        }
                        // r_{mp} = [C† v C]_{mp} / (eps_m - eps_p); D = -r;
                        // Tr[D dP] = -Σ r_{mp} rho_{pm}.
                        let (vr, vi) = projected.get(m, p);
                        let (pr, pi) = rho.get(p, m);
                        acc -= (vr * pr - vi * pi) / de;
                    }
                }
                *slot_out += weight * acc;
            }
        }
    }
    out
}

/// Born effective charges and the electronic dielectric tensor, from the `q = 0` responses.
///
/// Both come from the same machinery the phonons use, with one extra perturbation kind: a
/// homogeneous electric field, which enters through the commutator `[H, r]` because `r` itself is
/// not periodic (convention C-3).
///
/// | quantity | definition | convention |
/// |---|---|---|
/// | `born[A][(a, b)]` | `d^2 E / d f_a d R_{A,b}` | C-5: **`a` is the field index, `b` the displacement index** |
/// | `dielectric[(a, b)]` | `delta_ab - (4 pi / Omega) d^2 E / d f_a d f_b` | C-6: the minus follows from C-1 |
///
/// The Born charges are nearly free: their electronic part contracts the *displacement* responses,
/// which a phonon calculation already produces. The dielectric tensor needs the three field
/// responses in addition.
#[derive(Clone, Debug)]
pub struct DfptFieldResult {
    /// One 3x3 Born effective charge tensor per atom, in units of the elementary charge.
    pub born: Vec<crate::math::Mat3>,
    /// The electronic (clamped-ion) dielectric tensor.
    ///
    /// **3-D only.** With no cell volume there is no `Omega` to divide by and this is left as the
    /// identity; use [`Self::polarizability`], which is defined in any dimension.
    pub dielectric: crate::math::Mat3,
    /// `d(mu_a)/d(f_b)` per cell, in `e·Bohr` per `eV/(e·Bohr)` — the raw field response, with no
    /// volume factor applied.
    ///
    /// This is what the solver computes; [`Self::dielectric`] is its 3-D reading. It is the tensor
    /// to compare against a finite field along a **non-periodic** direction, which is the only
    /// independent check of the field response that exists (there is no periodic `FIELD=` in
    /// MOPAC, and in 3-D no direction is free).
    pub polarizability: crate::math::Mat3,
    /// Cell volume in Bohr^3, the `Omega` of the formulae above.
    pub volume_bohr3: f64,
    pub iterations: usize,
    pub converged: bool,
    pub residual: f64,
}

impl DfptFieldResult {
    /// The acoustic sum rule `sum_A Z*_A = q_tot`, as a residual.
    ///
    /// Zero for a neutral cell. This is the check that catches a missing **explicit** term in the
    /// Born charge — the fixed-density `q_A delta_ab` piece — which the two contraction orders
    /// agree on and therefore cannot detect between them.
    pub fn acoustic_residual(&self) -> f64 {
        let mut worst = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let total: f64 = self.born.iter().map(|z| z.get(a, b)).sum();
                worst = worst.max(total.abs());
            }
        }
        worst
    }

    /// The material data a LO-TO correction needs.
    ///
    /// **3-D only.** The non-analytic term's `q -> 0` limit has a different form in one and two
    /// dimensions, where the macroscopic field of a polarization wave is not simply
    /// `4 pi (q.Z*) / (q.eps.q)`.
    pub fn non_analytic(&self) -> Result<NonAnalytic> {
        if self.volume_bohr3 <= 0.0 {
            return Err(Pm7Error::InvalidInput(
                "the LO-TO non-analytic term needs a 3-D cell with a finite volume; in one and \
                 two dimensions the q -> 0 limit takes a different form"
                    .into(),
            ));
        }
        Ok(NonAnalytic {
            born: self.born.clone(),
            dielectric: self.dielectric,
            volume_bohr3: self.volume_bohr3,
        })
    }
}

/// `ε^∞` for a chain or a slab, together with what the assigned extent could not change.
#[derive(Clone, Debug)]
pub struct ExtentDielectric {
    /// The dielectric tensor under the assigned extent.
    pub dielectric: crate::math::Mat3,
    /// The raw `∂μ_a/∂f_b` per cell it was built from, which carries no convention at all.
    pub polarizability: crate::math::Mat3,
    /// The slab normal or wire axis used, normalized.
    pub axis: Vec3,
    /// The cell's own periodic measure: an area for a slab, a length for a wire, in Bohr powers.
    pub measure: f64,
    /// The convention the caller supplied.
    pub extent: crate::pbc::ExtentConvention,
    /// `(ε_∥ − 1)d` and `(1 − 1/ε_⊥)d`, both free of the extent. See
    /// [`crate::pbc::SheetInvariants`].
    pub invariants: crate::pbc::SheetInvariants,
    /// How much of `α` couples the distinguished axis to its complement, relative to the largest
    /// diagonal entry. Zero means the axis is a principal direction of the response and the
    /// per-principal-axis depolarization factors lost nothing.
    pub axis_mixing: f64,
}

/// The cell's electronic polarizability `α_ab = ∂μ_a/∂f_b`, on its own.
///
/// The same tensor [`born_and_dielectric`] reports, without the Born charges or the dielectric
/// conversion. It exists as its own entry point because `α` is the quantity that is defined in
/// **every** dimensionality: a chain and a slab have one, and neither has an `ε∞` until someone
/// says where the material stops (see [`dielectric_with_extent`]). Reaching it through the field
/// result meant computing Born charges to get a number that does not need them.
///
/// In the sign convention this crate uses throughout, which is MOPAC's `FIELD=` (convention C-1),
/// so `∂μ/∂f` carries the opposite sign to the physical polarizability. [`epsilon_from_polarizability`]
/// applies that flip once, internally; a caller doing its own conversion has to.
pub fn polarizability(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dfpt: &DfptOptions,
) -> Result<crate::math::Mat3> {
    Ok(born_and_dielectric(molecule, params, options, dfpt)?.polarizability)
}

/// How much `α` moves when the whole cell is translated — the size of the approximation.
///
/// The position operator the field perturbation is built on is **not** a well-defined periodic
/// operator. The argument that the *response* is nevertheless well defined — an origin shift adds
/// a constant to the diagonal, and the occupied–virtual projection annihilates it — is an
/// argument, and this measures it instead. Every atom is displaced by `offset`, `α` is recomputed,
/// and the largest change in any component comes back.
///
/// A value near machine precision says the argument holds for this system. A large one says it
/// does not, and that the polarizability being reported is a statement about where the origin was
/// put. Which of those is the case is worth knowing before quoting the number.
pub fn dielectric_origin_sensitivity(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dfpt: &DfptOptions,
    offset: Vec3,
) -> Result<f64> {
    let here = polarizability(molecule, params, options, dfpt)?;
    let mut shifted = molecule.clone();
    for atom in &mut shifted.atoms {
        atom.position += offset;
    }
    let there = polarizability(&shifted, params, options, dfpt)?;
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((here.get(a, b) - there.get(a, b)).abs());
        }
    }
    Ok(worst)
}

// `SOFT_MODE_FLOOR` lived here through 0.2.2: `1e-6` in eV/(Å²·amu), below which a mode was
// dropped from the ionic sum as acoustic. It is removed, and its removal is a breaking change.
//
// The argument for it was that it sat in a measured gap — the acoustic branch landed between 1e-17
// and 5e-15 on the cells it was developed against, and the softest genuine optical mode at about
// 2.5e-2, so the floor sat in the middle of thirteen orders of magnitude. That argument is about
// the cells it was tried on, not about the quantity: a frequency has real physics at every
// magnitude, so any cut through it is a guess that some other crystal will fall on the wrong side
// of. A ferroelectric near its transition has exactly the soft optical mode this would have
// swallowed, and it would have been swallowed from the sum it dominates.
//
// `acoustic_modes` replaces it with the statement the physics actually makes: at `q = 0` the
// acoustic modes are the mass-weighted uniform translations, there are three of them, and the
// three largest overlaps with that subspace are they.

/// `ε⁰`, the **static** dielectric tensor: the clamped-ion `ε∞` plus what the ions contribute.
///
/// ```text
/// ε⁰_ab = ε∞_ab + (4π/Ω) Σ_m (Z*·e_m)_a (Z*·e_m)_b / ω_m²
/// ```
///
/// summed over the optical modes `m` of the zone-centre dynamical matrix, with `e_m` the
/// mass-weighted eigenvector. This is the quantity a measured dielectric constant is usually
/// compared against; `ε∞` alone is the high-frequency limit, and for an ionic crystal the two
/// differ by a lot.
///
/// Nothing new is solved. Both ingredients — the Born charges and the Γ dynamical matrix — are
/// already here, so this is a contraction of quantities the crate had rather than another
/// response. Semiempirical codes commonly stop at `ε∞` for want of Born charges, not for want of
/// this formula.
///
/// **A soft mode makes this large and the geometry is what to check.** The `1/ω²` means a mode
/// near zero dominates, and a mode near zero in a structure that is not at its minimum is the
/// structure telling you so. Such a mode is **kept** in the sum and counted in
/// `soft_optical_modes`, which is reported rather than hidden because an unrelaxed geometry
/// otherwise returns a number that looks like an answer. `skipped_modes` is the three acoustic
/// modes and is always three — through 0.2.2 a magnitude floor decided both, so a genuine soft
/// optical mode was silently dropped from the sum it dominates.
///
/// 3-D only: both halves carry a `4π/Ω` that needs `Ω` to be a volume.
pub fn static_dielectric_tensor(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dfpt: &DfptOptions,
) -> Result<StaticDielectric> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput("a static dielectric tensor needs a periodic cell".into())
    })?;
    if cell.dim() != 3 {
        return Err(Pm7Error::InvalidInput(
            "the static dielectric tensor is three-dimensional: both halves carry a 4*pi/Omega \
             that needs Omega to be a volume, and a chain or a slab has only a length or an area. \
             `dielectric_with_extent` handles those, for the electronic half, once you say where \
             the material stops."
                .into(),
        ));
    }

    let field = born_and_dielectric(molecule, params, options, dfpt)?;
    let phonons = dynamical_matrix_dfpt(molecule, params, options, [0.0; 3], dfpt)?;

    // Mass-weighted, and diagonalized once: the frequencies and the eigenvectors have to come from
    // the same decomposition, or the `1/omega^2` and the mode it weights are from different
    // matrices.
    let dynamical = phonons.dynamical_matrix();
    let (eigenvalues, vectors) = dynamical.hermitian_eigen()?;
    let ndof = dynamical.n;
    let masses = &phonons.masses;

    // **Which modes are acoustic is decided by the eigenvectors, not by their frequencies.**
    //
    // Until 0.2.3 this dropped every mode with `lambda <= SOFT_MODE_FLOOR = 1e-6`. That is the
    // "below this number it must be acoustic" rule this release exists to remove, and it was
    // wrong in both directions: a genuine soft optical mode below the floor was silently dropped
    // from a sum it dominates, and on a cell whose acoustic branch happened to land above it the
    // `1/omega^2` would have blown up instead.
    //
    // At `q = 0` the acoustic modes *are* the mass-weighted uniform translations, so the
    // classification is an overlap with a known three-dimensional subspace, and there are exactly
    // three of them. Taking the three largest overlaps needs no threshold at all: the overlap is
    // 1 for a translation and 0 for anything orthogonal to it, which is a different kind of
    // quantity from a frequency, where genuine physics lives at every magnitude.
    let acoustic = acoustic_modes(&vectors, masses, ndof);

    let mut ionic = crate::math::Mat3::zero();
    let mut softest = f64::INFINITY;
    let mut soft_optical: Vec<(usize, f64)> = Vec::new();
    for (mode, &lambda) in eigenvalues.iter().enumerate() {
        if acoustic.contains(&mode) {
            continue;
        }
        // An optical mode at or below zero means the geometry is not a minimum. The floor used to
        // hide these too; `1/lambda` on a negative eigenvalue silently returns a tensor with the
        // wrong sign, which reads as an answer.
        if lambda <= 0.0 {
            soft_optical.push((mode, lambda));
            continue;
        }
        softest = softest.min(lambda);
        // Mode effective charge `(Z*·e)_a = Σ_{A,b} Z*_A[a][b] e_{Ab} / sqrt(m_A)`. The eigenvector
        // is mass weighted, so undoing the weighting here is what makes this a Cartesian dipole.
        let mut charge = [0.0_f64; 3];
        for a in 0..3 {
            for row in 0..ndof {
                let atom = row / 3;
                let b = row % 3;
                let (re, _) = vectors.get(row, mode);
                charge[a] += field.born[atom].get(a, b) * re / masses[atom].sqrt();
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                ionic.set(a, b, ionic.get(a, b) + charge[a] * charge[b] / lambda);
            }
        }
    }

    // `4*pi*a0^2/Omega`, which turns the mode sum into a dimensionless susceptibility.
    //
    // The conversion is pinned by **Lyddane-Sachs-Teller**, not by dimensional bookkeeping:
    // `eps_0/eps_inf = (omega_LO/omega_TO)^2` for a cubic diatomic crystal, and every quantity in
    // that identity reaches this point by a route that does not share this constant -- the LO and
    // TO frequencies come from the dynamical matrix with and without the non-analytic term.
    //
    // The first version of this line was low by 347.01, which is `HARTREE_TO_EV * a0^4` to five
    // figures. Nothing but the identity said so: an ionic term of 0.000256 against an `eps_inf`
    // of 1.0119 reads as a small correction to a weakly polarizable crystal, where LST requires
    // 0.0889 -- two and a half orders of magnitude, wearing the face of a plausible number.
    //
    // `tests/static_dielectric.rs` keeps it pinned. A dimensional argument for this factor would
    // be checking the arithmetic against itself; LST checks it against the phonons.
    let volume_bohr3 = field.volume_bohr3;
    let a0 = crate::constants::ANGSTROM_TO_BOHR;
    let scale = 2.0 * std::f64::consts::TAU * a0 * a0 / volume_bohr3;
    let mut epsilon = field.dielectric;
    for a in 0..3 {
        for b in 0..3 {
            epsilon.set(a, b, epsilon.get(a, b) + scale * ionic.get(a, b));
        }
    }
    Ok(StaticDielectric {
        epsilon,
        electronic: field.dielectric,
        ionic: {
            let mut m = crate::math::Mat3::zero();
            for a in 0..3 {
                for b in 0..3 {
                    m.set(a, b, scale * ionic.get(a, b));
                }
            }
            m
        },
        skipped_modes: acoustic.len(),
        soft_optical_modes: soft_optical.len(),
        softest_kept: if softest.is_finite() { softest } else { 0.0 },
    })
}

/// The three acoustic modes at `q = 0`, by overlap with the mass-weighted uniform translations.
///
/// The replacement for `SOFT_MODE_FLOOR`. At the zone centre a uniform translation of the whole
/// cell costs nothing, so the acoustic eigenvectors are `t_α ∝ √m_A ê_α` — a subspace that is
/// known before the matrix is diagonalized, and has dimension three whatever the frequencies come
/// out as.
///
/// There are exactly three, so this takes the three largest overlaps rather than testing against a
/// cut. A translation has overlap 1 and anything orthogonal to it has overlap 0; the middle is
/// empty for a well-formed problem, which is what makes "the three largest" well defined and what
/// makes a frequency floor the wrong tool — a frequency has genuine physics at every magnitude.
fn acoustic_modes(vectors: &crate::cmatrix::CMatrix, masses: &[f64], ndof: usize) -> Vec<usize> {
    // The three mass-weighted translation generators, orthonormal by construction: their supports
    // are disjoint across Cartesian directions and each has norm `sqrt(Σ_A m_A)`.
    let total: f64 = masses.iter().sum();
    let norm = total.sqrt();
    let overlap = |mode: usize| -> f64 {
        let mut weight = 0.0;
        for direction in 0..3 {
            let (mut re, mut im) = (0.0, 0.0);
            for row in (direction..ndof).step_by(3) {
                let mass = masses[row / 3].sqrt();
                let (r, i) = vectors.get(row, mode);
                re += mass * r;
                im += mass * i;
            }
            weight += (re * re + im * im) / (norm * norm);
        }
        weight
    };

    let mut ranked: Vec<(usize, f64)> = (0..ndof).map(|m| (m, overlap(m))).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<usize> = ranked.into_iter().take(3).map(|(m, _)| m).collect();
    chosen.sort_unstable();
    chosen
}

/// The static dielectric tensor and the pieces it was built from.
#[derive(Clone, Copy, Debug)]
pub struct StaticDielectric {
    /// `ε⁰ = ε∞ + ionic`, dimensionless.
    pub epsilon: crate::math::Mat3,
    /// The clamped-ion half, as [`born_and_dielectric`] reports it.
    pub electronic: crate::math::Mat3,
    /// What the ions added.
    pub ionic: crate::math::Mat3,
    /// Modes left out of the ionic sum because they are **acoustic** — identified by their overlap
    /// with the mass-weighted uniform translations, not by their frequency. Always three.
    ///
    /// Kept as a field because it is a useful assertion for a caller, but it no longer carries the
    /// information the old one did: through 0.2.2 this counted everything below `SOFT_MODE_FLOOR`,
    /// so "more than three" was how a caller learned the geometry was not a minimum. That signal
    /// now has its own field, [`Self::soft_optical_modes`], which is the honest place for it — the
    /// old count conflated "this mode is a translation" with "this structure is not relaxed".
    ///
    pub skipped_modes: usize,
    /// **Optical** modes with `ω² ≤ 0`, which the ionic sum cannot use.
    ///
    /// Non-zero means the geometry is not at a minimum, and the ionic term is missing whatever
    /// those modes would have contributed — which for a soft mode is most of it, since `1/ω²`
    /// weights them hardest. Reported rather than hidden: a `1/ω²` over a negative eigenvalue
    /// returns a tensor with the wrong sign, wearing the face of an answer.
    pub soft_optical_modes: usize,
    /// The smallest `ω²` that *was* kept, in eV/(Å²·amu). Small means the answer is dominated by
    /// one nearly-soft mode and is correspondingly sensitive to the geometry.
    pub softest_kept: f64,
}

/// `ε^∞` for a periodic system with fewer than three periodic directions.
///
/// [`born_and_dielectric`] leaves `dielectric` as the identity there, because `ε` needs a volume
/// and a slab's cell has an area. The missing ingredient is a **thickness** or a
/// **cross-section**, and it is a required argument rather than something to infer: a supercell
/// says where the atoms are, not where the material stops, so doubling the vacuum must not change
/// `ε` — and it would if the code took the cell height.
///
/// The conversion is a depolarization problem rather than a division; see
/// [`crate::pbc::extent`] for the law, the factors, and why the three-dimensional case is the
/// `N = 0` row of the same table rather than a separate rule.
///
/// The axis is derived from the cell: `â₁ × â₂` for a slab, `â₁` for a wire. A three-dimensional
/// cell is refused — it has a volume already, and [`born_and_dielectric`] gives its `ε` directly.
pub fn dielectric_with_extent(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dfpt: &DfptOptions,
    extent: crate::pbc::ExtentConvention,
) -> Result<ExtentDielectric> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm7Error::InvalidInput(
            "an assigned extent needs a periodic system; this molecule has no cell".into(),
        )
    })?;
    let dim = cell.dim();
    if dim == 3 {
        return Err(Pm7Error::InvalidInput(
            "a 3-D cell already has a volume: `born_and_dielectric` returns its dielectric tensor \
             directly, and assigning an extent on top would be a second, contradictory convention"
                .into(),
        ));
    }
    if dim != extent.periodic_directions() {
        return Err(Pm7Error::InvalidInput(format!(
            "this cell has {dim} periodic direction(s) but the convention describes {}: a \
             thickness belongs to a slab and a cross-section to a wire",
            extent.periodic_directions()
        )));
    }

    let vectors = cell.vectors();
    let axis = match dim {
        1 => vectors[0],
        2 => vectors[0].cross(vectors[1]),
        _ => unreachable!("dimensionality checked above"),
    };
    if axis.norm() < 1.0e-12 {
        return Err(Pm7Error::InvalidInput(
            "the lattice vectors are degenerate, so the slab normal or wire axis is undefined"
                .into(),
        ));
    }
    let axis = axis * (1.0 / axis.norm());

    let field = born_and_dielectric(molecule, params, options, dfpt)?;
    let measure = cell.measure();
    let dielectric =
        crate::pbc::epsilon_from_polarizability(&field.polarizability, axis, measure, extent)?;
    Ok(ExtentDielectric {
        dielectric,
        polarizability: field.polarizability,
        axis,
        measure,
        extent,
        invariants: crate::pbc::SheetInvariants::of(&field.polarizability, axis, measure),
        axis_mixing: crate::pbc::extent_axis_mixing(&field.polarizability, axis),
    })
}

/// The material constants of the LO-TO non-analytic term, ready to add to a `D(q)`.
#[derive(Clone, Debug)]
pub struct NonAnalytic {
    pub born: Vec<crate::math::Mat3>,
    pub dielectric: crate::math::Mat3,
    pub volume_bohr3: f64,
}

impl NonAnalytic {
    /// `D^NA_{Aa,Bb}(q_hat)` per convention C-7, in eV/Bohr^2 — i.e. **force constants**, before
    /// any mass weighting.
    ///
    /// The **field** index of `Z*` contracts with `q_hat`; the displacement index is the
    /// force-constant index. Nothing applies this implicitly: the `q -> 0` limit is direction
    /// dependent, so a silently chosen direction is a wrong answer rather than an approximate one.
    pub fn matrix(&self, q_hat: [f64; 3]) -> Result<CMatrix> {
        let q = Vec3::new(q_hat[0], q_hat[1], q_hat[2]);
        let norm = q.norm();
        if !norm.is_finite() || norm < 1.0e-12 {
            return Err(Pm7Error::InvalidInput(
                "the LO-TO term needs a direction: q_hat must be a non-zero vector".into(),
            ));
        }
        let q = q * (1.0 / norm);
        let mut denominator = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                denominator += q.get(a) * self.dielectric.get(a, b) * q.get(b);
            }
        }
        if denominator.abs() < 1.0e-12 {
            return Err(Pm7Error::InvalidInput(
                "the LO-TO denominator q.eps.q vanished; the dielectric tensor is singular along \
                 this direction"
                    .into(),
            ));
        }
        // `qz[A][b] = sum_a q_a Z*_{A,ab}` — the field index summed away, the displacement index
        // left as the force-constant index.
        let qz: Vec<[f64; 3]> = self
            .born
            .iter()
            .map(|z| {
                let mut row = [0.0; 3];
                for (b, slot) in row.iter_mut().enumerate() {
                    *slot = (0..3).map(|a| q.get(a) * z.get(a, b)).sum();
                }
                row
            })
            .collect();
        let n = 3 * self.born.len();
        let mut out = CMatrix::zeros(n);
        let prefactor = 4.0 * std::f64::consts::PI / (self.volume_bohr3 * denominator);
        for (ia, za) in qz.iter().enumerate() {
            for (ib, zb) in qz.iter().enumerate() {
                for a in 0..3 {
                    for b in 0..3 {
                        out.set(3 * ia + a, 3 * ib + b, prefactor * za[a] * zb[b], 0.0);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// The commutator `[H, r_a]` in the real-space block basis, for each Cartesian axis.
///
/// A homogeneous field cannot be added to a periodic Hamiltonian: `-f·r` grows without bound
/// under lattice translation, so there is no periodic matrix to add. The commutator is bounded
/// and periodic, and convention C-3 recovers the position operator's interband elements from it.
///
/// With `r` in the AO basis being `R_A` on the diagonal plus the intra-atomic hybrid dipole `d`
/// (the s–p `dd` and p–d `ddp(5)` terms — [`crate::dipole::dipole_operator`] builds exactly this),
///
/// ```text
/// [H, r_a](mu 0, nu T) = H(mu 0, nu T) (R_B + T - R_A)_a
///                      + sum_lam [ H(mu 0, lam T) d_B(lam nu, a) - d_A(mu lam, a) H(lam 0, nu T) ]
/// ```
///
/// The first term is why this is bounded at all: `H` decays with separation faster than the
/// displacement grows. The second is the one-centre hybridization, which does not commute with
/// `H` and would be silently dropped by treating `r` as purely diagonal.
///
/// Indexed in the **same** real-space convention as `H`, so `at_k` Bloch-sums it with the same
/// `e^{ik·T}`.
fn field_commutator_blocks(
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    cell: &crate::cell::Cell,
    fock: &crate::scf_pbc::BlochBlocks,
) -> Result<[ComplexBlocks; 3]> {
    let nao = basis.nao;
    let translations = fock.translations().to_vec();
    let mut out = [
        ComplexBlocks::new(translations.clone(), nao)?,
        ComplexBlocks::new(translations.clone(), nao)?,
        ComplexBlocks::new(translations.clone(), nao)?,
    ];
    // The intra-atomic part of `r`, about the coordinate origin. Only its on-site blocks are
    // non-zero, which is what makes the commutator above a two-term expression rather than a full
    // matrix product.
    let d = crate::dipole::dipole_operator(
        molecule,
        basis,
        params,
        Vec3::zero(),
        crate::dipole::DipoleTerms::Full,
    )?;
    // `dipole_operator` returns the *electronic* operator `D = -r`, so the hybrid part of `r` is
    // its negation; the diagonal `-R_A` part is handled explicitly by the displacement term and
    // must not be counted twice.
    let hybrid = |axis: usize, i: usize, j: usize| -> f64 {
        if basis.aos[i].atom == basis.aos[j].atom && i != j {
            -d[axis][(i, j)]
        } else {
            0.0
        }
    };

    for (slot, t) in translations.iter().enumerate() {
        let block = fock.block(slot);
        let shift = cell.translation(*t);
        for mu in 0..nao {
            let a = basis.aos[mu].atom;
            for nu in 0..nao {
                let b = basis.aos[nu].atom;
                let displacement = molecule.atoms[b].position + shift - molecule.atoms[a].position;
                let h = block[(mu, nu)];
                for axis in 0..3 {
                    let mut value = h * displacement.get(axis);
                    // + sum_lam H(mu 0, lam T) d_B(lam nu)
                    for lam in 0..nao {
                        if basis.aos[lam].atom == b {
                            value += block[(mu, lam)] * hybrid(axis, lam, nu);
                        }
                    }
                    // - sum_lam d_A(mu lam) H(lam 0, nu T)
                    for lam in 0..nao {
                        if basis.aos[lam].atom == a {
                            value -= hybrid(axis, mu, lam) * block[(lam, nu)];
                        }
                    }
                    out[axis].add(*t, mu, nu, [value, 0.0]);
                }
            }
        }
    }
    Ok(out)
}

/// What the perturbation's *bare* (fixed-density) term is, in the form the response can use it.
///
/// A nuclear displacement gives an ordinary local AO operator. A homogeneous electric field does
/// not: `r` is unbounded under lattice translation, so there is no periodic AO matrix to hand
/// over. What *is* periodic is the commutator `[H, r]`, and convention C-3 turns it back into the
/// position operator's interband elements after projection into the band basis:
///
/// ```text
/// <m|r_a|n> = <m|[H, r_a]|n> / (eps_m - eps_n),    m != n
/// ```
///
/// So the two variants differ only by one extra division, applied *after* the projection. The
/// self-consistent kernel `K[dP]` is an ordinary local potential either way and is untouched.
#[derive(Clone, Copy)]
enum BareTerm<'a> {
    /// A local AO operator, Bloch-summed at each k. Spin-independent: a nucleus does not know
    /// which channel it is perturbing.
    Local(&'a ComplexBlocks),
    /// `-r_a`, supplied as the commutator `[H^σ, r_a]`, **one per spin channel**. The minus is
    /// the sign of the electronic dipole operator `D_a = -r_a` that a field couples to
    /// (convention C-2).
    ///
    /// Per spin because `H_α ≠ H_β` in an unrestricted cell, and `r` reaches the response only
    /// through the commutator (convention C-3) — so each channel's position operator is built
    /// from its own Hamiltonian. Using one commutator for both channels converges and is wrong,
    /// which is why the unrestricted field response was refused rather than approximated before
    /// v0.2.2.
    Field(&'a [ComplexBlocks]),
}

impl BareTerm<'_> {
    /// The perturbation this spin channel sees.
    fn blocks(&self, spin: usize) -> &ComplexBlocks {
        match self {
            Self::Local(b) => b,
            Self::Field(per_spin) => {
                debug_assert!(spin < per_spin.len(), "one commutator per spin channel");
                &per_spin[spin]
            }
        }
    }

    /// Whether the blocks are a commutator needing the C-3 conversion rather than a potential.
    fn is_field(&self) -> bool {
        matches!(self, Self::Field(_))
    }
}

/// Solve the self-consistent response to one perturbation.
///
/// `channels` holds **one band set per spin**: a single entry for a closed shell, two for an
/// unrestricted one. That is the only structural difference between RHF and UHF here, because
/// with integer occupations there is no chemical potential in the response at all — the shared
/// `E_F` is how the *ground state* was filled, and the response of a gapped system moves no
/// electrons across it. What the two channels do share is the **Coulomb** half of the kernel;
/// exchange sees only its own spin. So the loop below carries two densities and builds two
/// kernels from `(total, same-spin)`.
///
/// The returned per-k response is the **total**, summed over spins, which is what every caller
/// contracts against — the perturbation itself is spin-independent.
/// `C(k+q)† Δh C(k)`, the bare perturbation in the band basis of the coupled pair.
///
/// Rows sit at `k + q`, columns at `k`. The bare term and the kernel are projected **separately**,
/// because only one of them needs converting. A displacement gives a local AO operator and needs
/// nothing. A field arrives as the commutator `[H, r_a]`, and conventions C-3 and C-2 turn that
/// back into the operator a field actually couples to:
///
/// ```text
/// <m|r_a|n> = <m|[H, r_a]|n> / (eps_m - eps_n),        D_a = -r_a
/// ```
///
/// The kernel `K[dP]` is an ordinary local potential either way and must **not** be divided.
/// Summing the two in the AO basis and projecting once — which is what this used to do — fed the
/// raw commutator in as though it were a potential, with neither the denominator nor the sign.
/// Born charges did not notice, because they use the field on one side only; the dielectric tensor
/// came out with the wrong sign.
fn project_bare_term(bare: &BareTerm<'_>, spin: usize, band: &BandPair, nao: usize) -> CMatrix {
    let mut projected = band
        .c_kq
        .adjoint()
        .matmul(&bare.blocks(spin).at_k(&band.k).matmul(&band.c_k));
    if bare.is_field() {
        for m in 0..nao {
            for n in 0..nao {
                let de = band.eps_kq[m] - band.eps_k[n];
                let (re, im) = projected.get(m, n);
                if de.abs() < 1.0e-8 {
                    // Degenerate pair: the interband position element is not defined. The response
                    // weight `f_n - f_m` vanishes on it too, so dropping it removes nothing that
                    // the next loop would have kept.
                    projected.set(m, n, 0.0, 0.0);
                } else {
                    projected.set(m, n, -re / de, -im / de);
                }
            }
        }
    }
    projected
}

#[allow(clippy::too_many_arguments)]
fn solve_one(
    bare: BareTerm<'_>,
    channels: &[Vec<BandPair>],
    molecule: &Molecule,
    params: &Pm7Parameters,
    basis: &Basis,
    core: &CoreHamiltonian,
    tables: &ResponseTables,
    translations: &[[i32; 3]],
    dfpt: &DfptOptions,
) -> Result<(Vec<Vec<CMatrix>>, usize, bool, f64)> {
    let nao = basis.nao;
    let spins = channels.len();
    debug_assert!(spins == 1 || spins == 2, "one or two spin channels");
    let n_k = channels[0].len();
    let weight = 1.0 / n_k as f64;
    let fresh = || -> Result<Vec<ComplexBlocks>> {
        (0..spins)
            .map(|_| ComplexBlocks::new(translations.to_vec(), nao))
            .collect()
    };
    let mut delta_p = fresh()?;
    let mut previous = delta_p.clone();
    let mut per_k: Vec<Vec<CMatrix>> = vec![vec![CMatrix::zeros(nao); n_k]; spins];
    let mut diis: Vec<ResponseDiis> = (0..spins).map(|_| ResponseDiis::new(12)).collect();

    // The response is a **linear** fixed point, `dp = L[bare + K[dp]]`. Plain iteration converges
    // if and only if the spectral radius of `L K` is below one — and when it is not, it does not
    // wander, it **diverges geometrically**. Left alone it will run the full iteration budget and
    // hand back a number like `1e33` with `converged: false` in a field nobody reads.
    //
    // Damping to `dp <- (1-a) dp + a F[dp]` turns an eigenvalue `lam` into `(1-a) + a lam`, which
    // is inside the unit circle for small `a` whenever `lam` is real and **negative** — the common
    // case, since an over-screening kernel overshoots and alternates. It cannot rescue a real
    // `lam > 1`: no `a > 0` shrinks `1 + a(lam - 1)`. So the ladder is a genuine fix for one
    // failure mode and a fast, honest detector for the other, rather than a hope.
    let ladder = [dfpt.mixing, 0.35, 0.12, 0.04];
    let mut best: Option<(Vec<Vec<CMatrix>>, usize, f64)> = None;
    let mut spent = 0usize;
    let trace = std::env::var_os("PM7_DFPT_TRACE").is_some();

    // The bare term's projection into the band basis is a function of the perturbation and the
    // bands alone, so it is the same on every iteration *and* on every rung of the damping ladder.
    // It used to be rebuilt inside the iteration loop — a Bloch sum over every translation, two
    // complex products and an adjoint, per k point, per iteration, per retry. Hoisting it is the
    // same arithmetic in the same order, so the result is bit-identical; the table costs one
    // response density's worth of memory, which `per_k` already spends.
    let bare_projected: Vec<Vec<CMatrix>> = {
        let _t = crate::profile::stage("dfpt: project the bare term");
        channels
            .iter()
            .enumerate()
            .map(|(spin, bands)| {
                bands
                    .iter()
                    .map(|band| project_bare_term(&bare, spin, band, nao))
                    .collect()
            })
            .collect()
    };

    for (rung, &mixing) in ladder.iter().enumerate() {
        if rung > 0 {
            // A fresh start: a diverged history is worse than none, and DIIS extrapolating from
            // it would carry the divergence straight into the retry.
            delta_p = fresh()?;
            previous = delta_p.clone();
            per_k = vec![vec![CMatrix::zeros(nao); n_k]; spins];
            diis = (0..spins).map(|_| ResponseDiis::new(12)).collect();
            if trace {
                eprintln!("dfpt: retrying with mixing {mixing}");
            }
        }
        let mut floor = f64::INFINITY;
        let mut diverged = false;

        for iteration in 0..dfpt.max_iterations {
            // The self-consistent perturbing potential: bare plus the kernel applied to the current
            // response. On the first pass the response is zero, so this is the uncoupled answer.
            //
            // Coulomb couples each channel to the **total** density, exchange only to its own. For
            // a closed shell the same-spin density is half the total, which is exactly the factor
            // the kernel used to carry as a literal `½`.
            let kernels: Option<Vec<ComplexBlocks>> = if iteration == 0 {
                None
            } else {
                let _t = crate::profile::stage("dfpt: response Fock");
                let total = if spins == 1 {
                    delta_p[0].clone()
                } else {
                    delta_p[0].plus(&delta_p[1])
                };
                let mut built = Vec::with_capacity(spins);
                for spin in 0..spins {
                    let same = if spins == 1 {
                        total.scaled_by(0.5)
                    } else {
                        delta_p[spin].clone()
                    };
                    built.push(fock_response_q(
                        molecule, params, basis, core, tables, &total, &same,
                    )?);
                }
                Some(built)
            };

            let mut next = fresh()?;
            let mut residual = 0.0_f64;
            for spin in 0..spins {
                let bands = &channels[spin];
                let kernel = kernels.as_ref().map(|k| &k[spin]);
                for (slot, band) in bands.iter().enumerate() {
                    // Into the band basis of the coupled pair: rows at `k + q`, columns at `k`.
                    // The bare half was projected once, before the iteration; see
                    // [`project_bare_term`] for why the two halves are projected separately.
                    let combined;
                    let projected: &CMatrix = if let Some(k) = kernel {
                        let extra = band
                            .c_kq
                            .adjoint()
                            .matmul(&k.at_k(&band.k).matmul(&band.c_k));
                        let mut sum = bare_projected[spin][slot].clone();
                        for m in 0..nao {
                            for n in 0..nao {
                                let (ar, ai) = sum.get(m, n);
                                let (br, bi) = extra.get(m, n);
                                sum.set(m, n, ar + br, ai + bi);
                            }
                        }
                        combined = sum;
                        &combined
                    } else {
                        &bare_projected[spin][slot]
                    };

                    // Linear response. Both the occupied–empty and empty–occupied blocks contribute: the
                    // second is the response of the bra at `k + q`, and dropping it halves the answer.
                    let mut response = CMatrix::zeros(nao);
                    for m in 0..nao {
                        let f_m = band.occ_kq[m];
                        for n in 0..nao {
                            let f_n = band.occ_k[n];
                            let df = f_n - f_m;
                            if df == 0.0 {
                                continue;
                            }
                            let de = band.eps_k[n] - band.eps_kq[m];
                            if de.abs() < 1.0e-8 {
                                continue;
                            }
                            let (re, im) = projected.get(m, n);
                            response.set(m, n, df * re / de, df * im / de);
                        }
                    }
                    let ao = band.c_kq.matmul(&response.matmul(&band.c_k.adjoint()));
                    per_k[spin][slot] = ao.clone();

                    // Back to real space, for the kernel: `Δp(T) = Σ_k w_k e^{−i k·T} ΔP(k)`.
                    //
                    // This is a *replicating* transform — the result is the same in every cell of a
                    // Born–von Kármán residue class, which is what the kernel wants when it reads one
                    // block at a time. It is emphatically **not** invertible by summing the stored
                    // translations back up: the block set is far larger than the k mesh, so a Bloch sum
                    // over it multiplies by the number of translations per class. The second-order energy
                    // therefore contracts `ΔP(k)` directly, never a round trip through real space.
                    for t in translations {
                        let phase = -band.k.phase(*t);
                        let (c, s) = (phase.cos(), phase.sin());
                        for i in 0..nao {
                            for j in 0..nao {
                                let (xr, xi) = ao.get(i, j);
                                next[spin].add(
                                    *t,
                                    i,
                                    j,
                                    [weight * (xr * c - xi * s), weight * (xr * s + xi * c)],
                                );
                            }
                        }
                    }
                }
                residual = residual.max(next[spin].rms_diff(&previous[spin]));
            }

            if trace {
                eprintln!("dfpt iter {iteration:3} mixing {mixing} residual {residual:.6e}");
            }
            spent += 1;
            for spin in 0..spins {
                delta_p[spin] = diis[spin].next(&previous[spin], &next[spin], mixing);
            }
            previous = delta_p.clone();
            if residual < dfpt.tolerance {
                return Ok((per_k, spent, true, residual));
            }
            // Diverging, and a linear map that is diverging will not stop. Bail to the next rung
            // now instead of grinding out the remaining iterations to produce a larger wrong
            // number: by the time the residual has grown a millionfold past its own best, the
            // answer carries no information and the only question left is how loudly to say so.
            if !residual.is_finite() || residual > floor * 1.0e6 {
                if trace {
                    eprintln!("dfpt: diverged at mixing {mixing} (residual {residual:.3e}, best {floor:.3e})");
                }
                diverged = true;
                break;
            }
            floor = floor.min(residual);
            if iteration + 1 == dfpt.max_iterations {
                break;
            }
        }

        // Keep the least-bad attempt, so a caller who opts out of the convergence check still gets
        // the most converged result rather than the last one tried.
        let residual = floor;
        if !diverged && best.as_ref().map(|(_, _, r)| residual < *r).unwrap_or(true) {
            best = Some((per_k.clone(), spent, residual));
        }
    }

    let (per_k, iterations, residual) = best.unwrap_or((
        vec![vec![CMatrix::zeros(nao); n_k]; spins],
        spent,
        f64::INFINITY,
    ));
    Ok((per_k, iterations, false, residual))
}

/// The same for the k-resolved response the energy contraction uses.
fn total_per_k(per_spin: &[Vec<CMatrix>]) -> Vec<CMatrix> {
    match per_spin {
        [only] => only.clone(),
        [a, b] => a
            .iter()
            .zip(b)
            .map(|(x, y)| {
                let mut sum = CMatrix::zeros(x.n);
                for i in 0..x.n {
                    for j in 0..x.n {
                        let (ar, ai) = x.get(i, j);
                        let (br, bi) = y.get(i, j);
                        sum.set(i, j, ar + br, ai + bi);
                    }
                }
                sum
            })
            .collect(),
        _ => unreachable!("one or two spin channels"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::KMesh;

    fn chain() -> Molecule {
        let a = crate::constants::ANGSTROM_TO_BOHR;
        Molecule::new(vec![
            crate::system::Atom {
                z: 6,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.0, 0.63 * a, 0.89 * a),
            },
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.0, 0.63 * a, -0.89 * a),
            },
        ])
        .with_cell(Cell::new(&[Vec3::new(2.55 * a, 0.0, 0.0)]).unwrap())
    }

    fn setup_mesh(
        mesh: KMesh,
    ) -> (
        Molecule,
        Pm7Parameters,
        Pm7Options,
        Basis,
        CoreHamiltonian,
        Pm7Result,
        PbcOptions,
    ) {
        let molecule = chain();
        let params = Pm7Parameters::method("pm7-".parse().unwrap()).unwrap();
        let options = Pm7Options {
            method: "pm7-".parse().unwrap(),
            e_tol: 1.0e-12,
            p_tol: 1.0e-10,
            max_scf: 800,
            pbc: Some(PbcOptions {
                kmesh: mesh,
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        let basis = Basis::build(&molecule, &params).unwrap();
        let pbc = options.pbc_for(&molecule).unwrap();
        let core = crate::hamiltonian::build_core_periodic(&molecule, &basis, &params, false, &pbc)
            .unwrap();
        let scf = crate::scf::run_pm7(&molecule, &params, &options).unwrap();
        (molecule, params, options, basis, core, scf, pbc)
    }

    /// `[H, r]` is **anti-Hermitian**, and along a free axis its band diagonal vanishes.
    ///
    /// Two invariants, no reference values, both violated by any slip in the hybrid terms — a
    /// wrong sign, a missing transpose, one of the two sums dropped, or the on-site restriction
    /// put on the wrong index.
    ///
    /// 1. `v(k)† = −v(k)` at every k. `H` and `r` are Hermitian, so their commutator is
    ///    anti-Hermitian; in block form that needs `v(−T) = −v(T)ᵀ`, which is a real constraint on
    ///    how the `(R_B + T − R_A)` displacement and the two hybrid sums are laid out.
    /// 2. Along a **non-periodic** axis the band diagonal is exactly zero, because
    ///    `[C† [H,r] C]_mm = (ε_m − ε_m) r_mm`.
    ///
    /// Along a **periodic** axis the diagonal is *not* zero and must not be asserted to be: `r_x`
    /// is not an operator there, and what the construction produces is the band velocity
    /// `∂ε/∂k`, which is exactly the intraband quantity the response then discards (its weight
    /// `f_n − f_m` vanishes). Anti-Hermiticity forces that diagonal to be purely imaginary, and
    /// that *is* checked — an earlier version of this test asserted a zero diagonal on all three
    /// axes and failed on the periodic one for entirely correct physics.
    /// The **skeleton** term alone must be Hermitian at any `q`.
    ///
    /// `D(q) = skeleton(q) + response(q)`, and each half is separately Hermitian: the skeleton is
    /// `Σ_T Φ_skel(0A, TB) e^{iq·T}` over real blocks obeying `Φ(0A,TB) = Φ(0B,−T A)ᵀ`, so its
    /// Bloch sum is Hermitian for the same reason `H(k)` is.
    ///
    /// Splitting the check in two is the whole point. `D(q)` used to be symmetrized on the way out
    /// unconditionally, so a defect in *either* half was averaged away and returned as a plausible
    /// matrix. This says which half.
    #[test]
    fn the_skeleton_is_hermitian_at_a_general_q() {
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(KMesh::grid(6, 1, 1));
        let density = scf.bloch_density.as_ref().unwrap();
        let translations: Vec<[i32; 3]> = core.bloch.as_ref().unwrap().translations().to_vec();
        let cell = molecule.cell.unwrap();
        let b = cell.reciprocal_2pi();
        for q_frac in [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ] {
            let mut q_cart = Vec3::zero();
            for (k, frac) in q_frac.iter().enumerate().take(cell.dim()) {
                q_cart += b[k] * *frac;
            }
            let (_bare, skeleton) = bare_and_skeleton(
                &molecule,
                &params,
                &basis,
                &core,
                &scf,
                density,
                &pbc,
                q_cart,
                &translations,
                LongRange::Auto,
            )
            .unwrap();
            let (violation, scale) = hermiticity(&skeleton);
            assert!(
                violation < 1.0e-9 * scale.max(1.0),
                "q = {q_frac:?}: the skeleton is not Hermitian, worst deviation {violation:.3e} \
                 against a largest element of {scale:.3e}"
            );
        }
    }

    #[test]
    fn the_commutator_is_anti_hermitian_and_has_no_free_axis_diagonal() {
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(KMesh::grid(3, 1, 1));
        let density = scf.bloch_density.as_ref().unwrap();
        let half = crate::scf_pbc::scale_blocks(density, 0.5);
        let fock =
            crate::fock::build_fock_spin_bloch(&molecule, &basis, &params, &core, density, &half)
                .unwrap();
        let cell = molecule.cell.unwrap();
        let v = field_commutator_blocks(&molecule, &params, &basis, &cell, &fock).unwrap();

        for k in response_mesh(&pbc.kmesh, &cell).unwrap() {
            let (_eps, c) = fock.at_k(&k).hermitian_eigen().unwrap();
            for (axis, block) in v.iter().enumerate() {
                let vk = block.at_k(&k);
                for i in 0..vk.n {
                    for j in 0..vk.n {
                        let (ar, ai) = vk.get(i, j);
                        let (br, bi) = vk.get(j, i);
                        // Anti-Hermitian: `v_ij = -conj(v_ji)`.
                        assert!(
                            (ar + br).abs() < 1.0e-9 && (ai - bi).abs() < 1.0e-9,
                            "axis {axis}, k {:?}, ({i},{j}): v = ({ar:.3e},{ai:.3e}) and \
                             v[{j}][{i}] = ({br:.3e},{bi:.3e}) are not anti-Hermitian",
                            k.frac
                        );
                    }
                }
                let projected = c.adjoint().matmul(&vk).matmul(&c);
                // The chain is periodic along x only, so `y` and `z` are free axes.
                let free = axis != 0;
                for band in 0..projected.n {
                    let (re, im) = projected.get(band, band);
                    assert!(
                        re.abs() < 1.0e-9,
                        "axis {axis}, band {band}: an anti-Hermitian matrix has an imaginary \
                         diagonal, got a real part {re:.3e}"
                    );
                    if free {
                        assert!(
                            im.abs() < 1.0e-9,
                            "axis {axis}, k {:?}, band {band}: [C^d v C]_mm = {im:.3e}i, but on a \
                             free axis `r` is an operator and the diagonal must vanish",
                            k.frac
                        );
                    }
                }
            }
        }
    }

    /// The commutator reproduces the **actual position operator** along a non-periodic axis.
    ///
    /// The test above says `v` is a commutator with something; this says *what*. Along `y` and `z`
    /// the chain has no lattice translation, so `r` itself is a perfectly ordinary bounded AO
    /// matrix and can be built directly — which makes
    ///
    /// ```text
    /// [C† v_a C]_mn = (eps_m - eps_n) [C† r_a C]_mn
    /// ```
    ///
    /// an exact algebraic identity with an independently constructed right-hand side, for every
    /// off-diagonal band pair. This is the single check that pins the hybrid term's sign and
    /// magnitude, because convention C-3 divides by that denominator and any error in `r` comes
    /// straight back out as an error in `Z*` and `eps^inf`.
    #[test]
    fn the_commutator_reproduces_the_transverse_position_operator() {
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(KMesh::grid(3, 1, 1));
        let density = scf.bloch_density.as_ref().unwrap();
        let half = crate::scf_pbc::scale_blocks(density, 0.5);
        let fock =
            crate::fock::build_fock_spin_bloch(&molecule, &basis, &params, &core, density, &half)
                .unwrap();
        let cell = molecule.cell.unwrap();
        let v = field_commutator_blocks(&molecule, &params, &basis, &cell, &fock).unwrap();

        // `r` in the AO basis, on-site only: `R_A delta_{mu nu}` plus the intra-atomic hybrid.
        // `dipole_operator` returns the electronic operator `D = -r`, so `r = -D`.
        let d = crate::dipole::dipole_operator(
            &molecule,
            &basis,
            &params,
            Vec3::zero(),
            crate::dipole::DipoleTerms::Full,
        )
        .unwrap();

        // The chain is periodic along x only, so only `y` and `z` have a bounded `r`.
        for axis in [1usize, 2] {
            let mut r = crate::linalg::Matrix::zeros(basis.nao, basis.nao);
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    if basis.aos[i].atom == basis.aos[j].atom {
                        r[(i, j)] = -d[axis][(i, j)];
                    }
                }
            }
            for k in response_mesh(&pbc.kmesh, &cell).unwrap() {
                let (eps, c) = fock.at_k(&k).hermitian_eigen().unwrap();
                let vk = c.adjoint().matmul(&v[axis].at_k(&k)).matmul(&c);
                // `r` is on-site and `y` is non-periodic, so its Bloch sum is `r` itself at every
                // k: there is no translation for which it has an off-site block to phase.
                let mut r_bloch = CMatrix::zeros(basis.nao);
                r_bloch.add_phase(&r, 1.0, 0.0);
                let rk = c.adjoint().matmul(&r_bloch).matmul(&c);
                for m in 0..eps.len() {
                    for n in 0..eps.len() {
                        if m == n {
                            continue;
                        }
                        let gap = eps[m] - eps[n];
                        let (vr, vi) = vk.get(m, n);
                        let (rr, ri) = rk.get(m, n);
                        assert!(
                            (vr - gap * rr).abs() < 1.0e-8 && (vi - gap * ri).abs() < 1.0e-8,
                            "axis {axis}, k {:?}, bands ({m},{n}): [C^d v C] = ({vr:.6e}, \
                             {vi:.6e}) but (eps_m - eps_n) [C^d r C] = ({:.6e}, {:.6e})",
                            k.frac,
                            gap * rr,
                            gap * ri
                        );
                    }
                }
            }
        }
    }

    /// The bare perturbation, summed over translations at `q = 0`, is the ordinary derivative
    /// Fock. This isolates the phase bookkeeping from everything downstream: if it fails, the
    /// tuple of (block, row, column, displaced atom) is wrong somewhere and no amount of
    /// debugging the response will help.
    #[test]
    fn the_bare_perturbation_reduces_to_the_gamma_derivative_fock() {
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(KMesh::grid(1, 1, 1));
        let density = scf.bloch_density.as_ref().unwrap();
        let translations: Vec<[i32; 3]> = core.bloch.as_ref().unwrap().translations().to_vec();
        let (bare, _) = bare_and_skeleton(
            &molecule,
            &params,
            &basis,
            &core,
            &scf,
            density,
            &pbc,
            Vec3::zero(),
            &translations,
            LongRange::Auto,
        )
        .unwrap();
        let reference = crate::hessian_pbc::derivative_fock(
            &molecule,
            &params,
            &basis,
            &scf.density,
            &scf.charges,
            &pbc,
        )
        .unwrap();
        let gamma = KPoint {
            frac: [0.0; 3],
            cart: Vec3::zero(),
            weight: 1.0,
            time_reversal_pair: false,
        };
        for (j, block) in bare.iter().enumerate() {
            let summed = block.at_k(&gamma);
            let mut worst = 0.0_f64;
            let mut imaginary = 0.0_f64;
            for i in 0..basis.nao {
                for k in 0..basis.nao {
                    let (re, im) = summed.get(i, k);
                    worst = worst.max((re - reference[j][(i, k)]).abs());
                    imaginary = imaginary.max(im.abs());
                }
            }
            assert!(
                imaginary < 1e-12,
                "dof {j}: the Γ perturbation has an imaginary part {imaginary:.3e}"
            );
            assert!(
                worst < 1e-9,
                "dof {j}: the Γ perturbation differs from the derivative Fock by {worst:.3e}"
            );
        }
    }

    /// The bare perturbation is the derivative of the **Bloch** Fock blocks at fixed density, and
    /// a finite difference of those blocks says so directly — on a real k mesh, where the class
    /// structure of the long-range exchange is actually exercised. The `q = 0` comparison against
    /// `derivative_fock` cannot see that: at a single k point every residue class collapses into
    /// the one full lattice sum.
    #[test]
    fn the_bare_perturbation_differentiates_the_bloch_fock_on_a_k_mesh() {
        let molecule = chain();
        let params = Pm7Parameters::method("pm7-".parse().unwrap()).unwrap();
        let options = Pm7Options {
            method: "pm7-".parse().unwrap(),
            e_tol: 1.0e-12,
            p_tol: 1.0e-8,
            max_scf: 800,
            pbc: Some(PbcOptions {
                kmesh: KMesh::grid(2, 1, 1),
                ..PbcOptions::default()
            }),
            ..Pm7Options::default()
        };
        let basis = Basis::build(&molecule, &params).unwrap();
        let pbc = options.pbc_for(&molecule).unwrap();
        let core = crate::hamiltonian::build_core_periodic(&molecule, &basis, &params, false, &pbc)
            .unwrap();
        let scf = crate::scf::run_pm7(&molecule, &params, &options).unwrap();
        let density = scf.bloch_density.as_ref().unwrap();
        let translations: Vec<[i32; 3]> = core.bloch.as_ref().unwrap().translations().to_vec();
        let (bare, _) = bare_and_skeleton(
            &molecule,
            &params,
            &basis,
            &core,
            &scf,
            density,
            &pbc,
            Vec3::zero(),
            &translations,
            LongRange::Auto,
        )
        .unwrap();

        // The Fock at a displaced geometry, at the *same* density: the skeleton derivative.
        let fock_at = |shifted: &Molecule| -> BlochBlocks {
            let b = Basis::build(shifted, &params).unwrap();
            let c =
                crate::hamiltonian::build_core_periodic(shifted, &b, &params, false, &pbc).unwrap();
            crate::fock::build_fock_spin_bloch(
                shifted,
                &b,
                &params,
                &c,
                density,
                &crate::scf_pbc::scale_blocks(density, 0.5),
            )
            .unwrap()
        };

        let h = 1.0e-5;
        let gamma = KPoint {
            frac: [0.0; 3],
            cart: Vec3::zero(),
            weight: 1.0,
            time_reversal_pair: false,
        };
        let boundary = KPoint {
            frac: [0.5, 0.0, 0.0],
            cart: Vec3::zero(),
            weight: 1.0,
            time_reversal_pair: false,
        };
        for dof in [0usize, 4, 8] {
            let (atom, axis) = (dof / 3, dof % 3);
            let displaced = |s: f64| -> Molecule {
                let mut m = molecule.clone();
                match axis {
                    0 => m.atoms[atom].position.x += s * h,
                    1 => m.atoms[atom].position.y += s * h,
                    _ => m.atoms[atom].position.z += s * h,
                }
                m
            };
            let up = fock_at(&displaced(1.0));
            let down = fock_at(&displaced(-1.0));
            for k in [&gamma, &boundary] {
                let mine = bare[dof].at_k(k);
                let a = up.at_k(k);
                let b = down.at_k(k);
                let mut worst = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..basis.nao {
                    for j in 0..basis.nao {
                        let (ar, ai) = a.get(i, j);
                        let (br, bi) = b.get(i, j);
                        let fd = [(ar - br) / (2.0 * h), (ai - bi) / (2.0 * h)];
                        let (mr, mi) = mine.get(i, j);
                        scale = scale.max(fd[0].abs()).max(fd[1].abs());
                        worst = worst.max((mr - fd[0]).abs()).max((mi - fd[1]).abs());
                    }
                }
                assert!(
                    worst < 1e-4 * scale.max(1.0),
                    "dof {dof} at k = {:?}: the bare perturbation differs from a finite \
                     difference of the Bloch Fock by {worst:.3e} (values up to {scale:.3e})",
                    k.frac
                );
            }
        }
    }

    /// At `q = 0` the response kernel is just `F(P + ΔP) − F(P)`, which the ordinary Fock builder
    /// can produce independently. This is the one piece of the response that is checkable without
    /// running the whole perturbation, so it is worth checking on its own.
    #[test]
    fn the_response_kernel_reduces_to_a_difference_of_fock_builds() {
        for mesh in [KMesh::grid(1, 1, 1), KMesh::grid(2, 1, 1)] {
            check_kernel(mesh);
        }
    }

    fn check_kernel(mesh: KMesh) {
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(mesh);
        let density = scf.bloch_density.as_ref().unwrap();
        let translations: Vec<[i32; 3]> = core.bloch.as_ref().unwrap().translations().to_vec();

        // A small, deterministic, symmetric perturbation of the density.
        let mut delta = ComplexBlocks::new(translations.clone(), basis.nao).unwrap();
        let mut real_delta = BlochBlocks::new(translations.clone(), basis.nao).unwrap();
        let mut seed = 1u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02
        };
        for index in 0..translations.len() {
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    let v = next();
                    real_delta.block_mut(index)[(i, j)] = v;
                }
            }
        }
        real_delta.symmetrize();
        for (index, t) in translations.iter().enumerate() {
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    delta.add(*t, i, j, [real_delta.block(index)[(i, j)], 0.0]);
                }
            }
        }

        // Closed shell: the same-spin response is half the total, which is the pair the kernel
        // takes now that it serves both RHF and UHF.
        let half_delta = delta.scaled_by(0.5);
        let tables =
            ResponseTables::build(&molecule, &core, &pbc, Vec3::zero(), &translations, true)
                .unwrap();
        let got = fock_response_q(
            &molecule,
            &params,
            &basis,
            &core,
            &tables,
            &delta,
            &half_delta,
        )
        .unwrap();

        let shifted = crate::scf_pbc::add_blocks(density, &real_delta);
        let f1 = crate::fock::build_fock_spin_bloch(
            &molecule,
            &basis,
            &params,
            &core,
            &shifted,
            &crate::scf_pbc::scale_blocks(&shifted, 0.5),
        )
        .unwrap();
        let f0 = crate::fock::build_fock_spin_bloch(
            &molecule,
            &basis,
            &params,
            &core,
            density,
            &crate::scf_pbc::scale_blocks(density, 0.5),
        )
        .unwrap();

        let mut worst = 0.0_f64;
        let mut imaginary = 0.0_f64;
        let mut scale = 0.0_f64;
        for (index, _t) in translations.iter().enumerate() {
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    let expected = f1.block(index)[(i, j)] - f0.block(index)[(i, j)];
                    let mine = got.re.block(index)[(i, j)];
                    scale = scale.max(expected.abs());
                    worst = worst.max((mine - expected).abs());
                    imaginary = imaginary.max(got.im.block(index)[(i, j)].abs());
                }
            }
        }
        assert!(
            imaginary < 1e-12,
            "the Γ kernel produced an imaginary part {imaginary:.3e}"
        );
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "the Γ kernel differs from F(P+ΔP) − F(P) by {worst:.3e} (values up to {scale:.3e})"
        );
    }

    /// The skeleton at `q = 0` is the fixed-density part of the zone-centre Hessian — on a real k
    /// mesh as well as at a single point, which is where the residue-class structure of the
    /// long-range exchange actually differs from one full lattice sum.
    #[test]
    fn the_skeleton_reduces_to_the_gamma_fixed_density_hessian() {
        for mesh in [KMesh::grid(1, 1, 1), KMesh::grid(2, 1, 1)] {
            check_skeleton(mesh);
        }
    }

    fn check_skeleton(mesh: KMesh) {
        let divisions = mesh.divisions();
        let (molecule, params, _options, basis, core, scf, pbc) = setup_mesh(mesh);
        let density = scf.bloch_density.as_ref().unwrap();
        let translations: Vec<[i32; 3]> = core.bloch.as_ref().unwrap().translations().to_vec();
        let (_, skeleton) = bare_and_skeleton(
            &molecule,
            &params,
            &basis,
            &core,
            &scf,
            density,
            &pbc,
            Vec3::zero(),
            &translations,
            LongRange::Auto,
        )
        .unwrap();

        let total = crate::gradient::TranslatedDensity::Bloch(density, divisions);
        let half = crate::scf_pbc::scale_blocks(density, 0.5);
        let spin = crate::gradient::TranslatedDensity::Bloch(&half, divisions);
        let short =
            crate::hessian_pbc::skeleton_hessian(&molecule, &params, &basis, total, None, &pbc)
                .unwrap();
        let long = crate::hessian_pbc::long_range_hessian(
            &molecule,
            &basis,
            &scf.charges,
            &[spin, spin],
            &pbc,
        )
        .unwrap();

        let mut worst = 0.0_f64;
        let mut imaginary = 0.0_f64;
        for i in 0..short.rows {
            for j in 0..short.cols {
                let (re, im) = skeleton.get(i, j);
                worst = worst.max((re - short[(i, j)] - long[(i, j)]).abs());
                imaginary = imaginary.max(im.abs());
            }
        }
        assert!(
            imaginary < 1e-10,
            "mesh {divisions:?}: the Γ skeleton has an imaginary part {imaginary:.3e}"
        );
        assert!(
            worst < 1e-8,
            "mesh {divisions:?}: the Γ skeleton differs from the fixed-density Hessian by {worst:.3e}"
        );
    }
}
