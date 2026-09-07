// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO SCF driver — restricted (RHF) closed shell and unrestricted (UHF) open shell.
//!
//! Because NDDO assumes an orthonormal AO basis, the working equations are the plain
//! eigenproblem `F C = C ε` (no `S`); overlap enters only the resonance term of `H_core`.
//! The initial density is a **superposition of atomic densities** (internal `sad_density`) — the
//! exact free-atom density in a minimal valence basis, far better than the bare-core guess —
//! and charge convergence is accelerated with the A-DIIS→CDIIS hybrid on the `[F,P]` commutator.

use crate::basis::Basis;
use crate::constants::{AU_DIPOLE_TO_DEBYE, EV_TO_KCAL};
use crate::error::{Pm7Error, Result};
use crate::fock::{build_fock, build_fock_spin};
use crate::hamiltonian::CoreHamiltonian;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::method::Pm7Method;
use crate::params::Pm7Parameters;
use crate::repulsion::core_core_energy;
use crate::system::Molecule;

/// SCF charge-convergence accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfAccelerator {
    /// No extrapolation (plain iteration).
    None,
    /// Pulay CDIIS on the `[F,P]` commutator throughout.
    Cdiis,
    /// **A-DIIS** (Hu & Yang, *J. Chem. Phys.* **132**, 054109 (2010)) while far from
    /// convergence, switching to CDIIS once the commutator error drops below a threshold —
    /// the robust hybrid recommended for hard cases (radicals, small gaps, poor guesses).
    AdiisCdiis,
}

/// Choice of SCF reference (spin treatment), independent of the spin multiplicity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScfReference {
    /// Restricted (RHF) for a closed shell, unrestricted (UHF) for an open shell.
    #[default]
    Auto,
    /// Force restricted closed-shell RHF; errors on an open-shell electron count.
    Restricted,
    /// Force unrestricted UHF, even for a closed-shell singlet (allows spin-symmetry
    /// breaking, e.g. singlet diradicals or bond dissociation).
    Unrestricted,
}

impl std::fmt::Display for ScfReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScfReference::Auto => "auto",
            ScfReference::Restricted => "rhf",
            ScfReference::Unrestricted => "uhf",
        })
    }
}

impl std::str::FromStr for ScfReference {
    type Err = Pm7Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(ScfReference::Auto),
            "r" | "rhf" | "restricted" => Ok(ScfReference::Restricted),
            "u" | "uhf" | "unrestricted" => Ok(ScfReference::Unrestricted),
            other => Err(Pm7Error::InvalidInput(format!(
                "unknown SCF reference `{other}` (expected auto, rhf, or uhf)"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pm7Options {
    /// PM7 family selection. It must match the parameter table supplied to the calculator.
    pub method: Pm7Method,
    pub charge: f64,
    pub multiplicity: usize,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Legacy flag: `false` forces [`ScfAccelerator::None`] regardless of `accelerator`.
    pub use_diis: bool,
    pub accelerator: ScfAccelerator,
    /// Commutator-error norm below which the ADIIS→CDIIS hybrid switches to CDIIS.
    pub adiis_switch: f64,
    /// Static level shift in eV applied to the virtual manifold (MOPAC's `SHIFT`). `0.0`
    /// (default) leaves the SCF unshifted until it stalls, at which point the loop raises a
    /// shift on its own and relaxes it again as progress resumes. Raising the floor makes the
    /// shift unconditional, which helps a system known to be hard. The shift never moves the
    /// converged solution — see [`apply_level_shift`].
    pub level_shift: f64,
    /// Spin treatment (RHF/UHF), independent of `multiplicity`. Default [`ScfReference::Auto`].
    pub reference: ScfReference,
    /// Diagnostic: route every pair through the MNDO/d two-center + overlap kernel even
    /// for a pure-sp molecule (lets the validated sp path act as an oracle for the d path).
    pub force_dpath: bool,
    /// Optional peak-memory budget in MiB for the pre-flight OOM guard. `None` falls back to
    /// the `PM7_MEM_BUDGET_MB` env var, then to 80 % of currently available physical RAM (see
    /// [`crate::memory`]).
    /// Interpolate the dispersion `C6` continuously in a smooth neighbour count.
    ///
    /// **Off by default, and off is bit-identical to MOPAC.** `c6_atom` is a step on an integer
    /// bond count: a carbon with four neighbours takes 0.95 and anything else 1.65. That is what
    /// MOPAC does and what the parity tests pin, and it makes the energy **discontinuous** where a
    /// distance crosses `1.3(r_i + r_j)`. Measured on a methane whose fourth C-H bond is stretched
    /// across the threshold, the post-SCF correction steps by 0.086 meV against a local slope of
    /// 0.0002 meV per 0.002 A -- 430 times the gradient, at a single point, where the force is
    /// therefore not defined.
    ///
    /// Switching this on replaces the step with a smooth coordination number and a switch sharp
    /// enough to reproduce the discrete coefficient to nine digits at an integer count, so the two
    /// paths differ only where the discrete one has no derivative. See
    /// [`crate::dispersion::c6_atom_smooth`].
    ///
    /// It is opt-in rather than the default because it moves published numbers by ~1e-9 relative
    /// everywhere and by more near a threshold, and because MOPAC parity is a property this crate
    /// is tested against. Use it for geometry optimization or molecular dynamics that may sit near
    /// a coordination boundary, where a step function costs more than the parity is worth.
    pub smooth_dispersion: bool,
    pub max_memory_mb: Option<usize>,
    /// Optional smooth long-range-**exchange** cutoff `(inner, outer)` in **Bohr**, applied only
    /// to the analytic Hessian's CPHF response-Fock builds (its dominant cost). The two-center
    /// exchange between atoms farther than `outer` apart is dropped, with a C²-smooth switch
    /// between `inner` and `outer`. `None` (default) keeps the Hessian **bit-identical** /
    /// cutoff-free; Coulomb is never cut. Trades a controlled approximation for speed on large
    /// systems.
    pub exchange_cutoff: Option<(f64, f64)>,
    /// Operator applications one CPHF solve may spend, over both of its solvers.
    ///
    /// The orbital response behind every analytic Hessian is a linear system, and how many
    /// applications it takes is set by the conditioning of the orbital Hessian — which is set by
    /// the frontier gap, which is a property of the system and not of this crate. 100 is enough for
    /// an ordinary molecule and for most cells, and it is **not** enough for an ill-conditioned
    /// one: cubic SrTiO₃ has a 0.28 eV PM7 gap and `phonons --supercell 2 2 2` stops at a residual
    /// of 6e-9 against the 1e-9 target, while `dfpt` converges the same zone-centre response to
    /// 8e-11 in fourteen iterations through a different solver. Through 0.2.2 the budget was a
    /// private constant, so the only remedy for that was to use the other entry point.
    ///
    /// Raising it costs iterations and nothing else: the tolerance is unchanged, so a solve that
    /// converged at 100 converges at the same iterate with 400 and returns the same Hessian. It
    /// buys nothing for a system that is not converging at all — a diverging response is refused
    /// either way, later.
    pub cphf_max_iterations: usize,
    /// Whether to ask if the converged SCF solution is a **minimum**, and what to do if it is not.
    ///
    /// An SCF iteration converges to a stationary point, and a saddle converges as cleanly as a
    /// minimum — same residual, same `converged: true`. [`crate::stability`] has the measurement
    /// that made this worth a knob: CuCl converges 5.18 kcal/mol above the solution MOPAC finds,
    /// and no level shift, accelerator or spin reference reaches the lower one.
    ///
    /// `Off` by default because the check costs roughly a CPHF solve, which is the same order as
    /// the SCF it follows; a caller who wants the guarantee asks for it.
    pub stability: crate::stability::ScfStability,
    /// Start the SCF from this density instead of the superposition-of-atomic-densities guess.
    ///
    /// For a caller who already has a density worth starting from — the previous step of a
    /// trajectory, a converged solution at a nearby geometry, or an orbital rotation that escapes a
    /// saddle, which is what [`crate::stability`] uses it for. `None` keeps the SAD guess, which is
    /// the right starting point when there is nothing better.
    pub initial_density: Option<crate::linalg::Matrix>,
    /// Start a UHF run with this **spin** density `P^α − P^β`, alongside `initial_density`.
    ///
    /// Without it a closed-shell UHF run cannot break spin symmetry at all. The guess in
    /// [`uhf_loop`] is the atomic density scaled by `n_α/n` and `n_β/n`, which for `n_α = n_β` gives
    /// `P^α = P^β` exactly — and the UHF equations preserve a zero spin density forever, so
    /// `reference = "uhf"` on a closed shell returns the RHF answer no matter what else is set.
    /// That is correct when RHF is the right answer and a trap when it is not, which is why
    /// [`crate::stability`] measures the triplet instability and seeds this from its eigenvector.
    pub initial_spin_density: Option<crate::linalg::Matrix>,
    /// Periodic-boundary settings. Ignored when the molecule has no cell; required (and filled
    /// in with [`crate::pbc::PbcOptions::default`] if left `None`) when it has one.
    ///
    /// Keeping this on the options rather than on the geometry means `run_pm7`,
    /// `closed_form_gradient`, `analytic_hessian`, and `optimize` stay single entry points that
    /// dispatch on `Molecule::cell`.
    pub pbc: Option<crate::pbc::PbcOptions>,
    /// A uniform external electric field, in MOPAC's `FIELD=(x,y,z)` volts/Ångström convention.
    ///
    /// On the options rather than on the geometry for the same reason `pbc` is: it is a model
    /// setting, and keeping it here leaves `run_pm7`, `closed_form_gradient`, `analytic_hessian`
    /// and `optimize` as single entry points. See [`crate::field`] for the sign convention, which
    /// is MOPAC's and is not the physical one.
    pub field: Option<crate::field::ExternalField>,
    /// Origin for the point-charge part of the reported dipole. Only affects a **charged**
    /// molecule; see [`crate::dipole::DipoleOrigin`].
    pub dipole_origin: crate::dipole::DipoleOrigin,
}

impl Pm7Options {
    /// The external field, if one is set and it is not exactly zero.
    ///
    /// Every field-aware path goes through this, so `Some(ExternalField::default())` costs
    /// nothing and changes nothing — which is what the bit-identity regression test relies on.
    pub fn active_field(&self) -> Option<&crate::field::ExternalField> {
        self.field.as_ref().filter(|f| !f.is_zero())
    }

    /// The periodic settings to use for `molecule`: the caller's, or the defaults when the
    /// molecule is periodic and none were given. `None` for a molecule.
    pub fn pbc_for(&self, molecule: &Molecule) -> Option<crate::pbc::PbcOptions> {
        molecule.cell?;
        Some(self.pbc.clone().unwrap_or_default())
    }
}

impl Default for Pm7Options {
    fn default() -> Self {
        Self {
            method: Pm7Method::Pm7,
            charge: 0.0,
            multiplicity: 1,
            max_scf: 200,
            e_tol: 1.0e-8,
            p_tol: 1.0e-7,
            use_diis: true,
            accelerator: ScfAccelerator::AdiisCdiis,
            adiis_switch: 0.1,
            level_shift: 0.0,
            reference: ScfReference::Auto,
            force_dpath: false,
            smooth_dispersion: false,
            max_memory_mb: None,
            exchange_cutoff: None,
            cphf_max_iterations: 100,
            stability: crate::stability::ScfStability::Off,
            initial_density: None,
            initial_spin_density: None,
            pbc: None,
            field: None,
            dipole_origin: crate::dipole::DipoleOrigin::default(),
        }
    }
}

/// Which Hamiltonian the reported orbitals are eigenvectors of.
///
/// Exists so that a k-mesh run cannot silently pass off a Γ quantity as a zone-wide one. The gap
/// implied by `homo_ev`/`lumo_ev` is a Γ gap in the `KMeshGamma` case; the zone-wide answers are
/// `fermi_ev` and [`crate::scf_pbc::band_structure`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrbitalSource {
    /// A molecule: the SCF's own eigenvectors.
    Molecular,
    /// A Γ-point-only periodic solve: again the SCF's own eigenvectors.
    Gamma,
    /// A k-mesh run, reported at **Γ**, re-diagonalized from the converged Hamiltonian.
    KMeshGamma,
}

impl OrbitalSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Molecular => "molecular",
            Self::Gamma => "gamma",
            Self::KMeshGamma => "kmesh-gamma",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pm7Result {
    pub density: Matrix,
    /// Spin density `P_α − P_β` (open-shell UHF only; `None` for RHF).
    pub spin_density: Option<Matrix>,
    /// Translation-resolved density blocks `P(T)` from a k-point run.
    ///
    /// `None` for a molecule or a Γ-point cell, where `P(T)` is the same matrix for every `T` and
    /// `density` says all there is to say. A k mesh makes the density decay with distance, and
    /// the resonance and exchange terms of the gradient and the stress have to see the block **at
    /// the translation they act on** — contracting them against `P(0)` instead is wrong by whole
    /// eV/Å.
    pub bloch_density: Option<crate::scf_pbc::BlochBlocks>,
    /// Translation-resolved spin density from an unrestricted k-point run.
    pub bloch_spin_density: Option<crate::scf_pbc::BlochBlocks>,
    pub mo_energies: Vec<f64>,
    pub mo_coeff: Matrix,
    pub n_occ: usize,
    /// The β orbital set, for an unrestricted run only. `homo_ev`/`lumo_ev` keep their existing
    /// meaning (α); [`Self::gap_ev`] is the spin-resolved frontier gap.
    pub mo_energies_beta: Option<Vec<f64>>,
    pub mo_coeff_beta: Option<Matrix>,
    pub n_occ_beta: Option<usize>,
    pub homo_ev_beta: Option<f64>,
    pub lumo_ev_beta: Option<f64>,
    /// Which Hamiltonian [`Self::mo_energies`] and [`Self::mo_coeff`] belong to.
    pub orbital_source: OrbitalSource,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub dipole_debye: Vec3,
    pub dipole_magnitude: f64,
    /// The dipole split into point-charge, s–p hybrid and p–d hybrid terms (Debye), mirroring
    /// MOPAC's `POINT-CHG. / HYBRID / SUM` print. [`Self::dipole_debye`] is its total.
    pub dipole: crate::dipole::DipoleBreakdown,
    /// `f · mu_FieldConjugate` in eV when an external field is applied, else `None`.
    ///
    /// Already inside [`Self::total_ev`] and the heat of formation; reported separately so it can
    /// be subtracted. See [`crate::field`] for why this is the field-conjugate dipole rather than
    /// the full one — MOPAC's `FIELD=` operator has no p–d term even though its dipole does.
    pub field_ev: Option<f64>,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// What [`crate::stability`] found, when [`Pm7Options::stability`] asked it to look.
    ///
    /// `None` means nobody asked — **not** that the solution is a minimum. The distinction matters:
    /// an SCF converges to a stationary point either way, and this is the only field that says
    /// which kind.
    pub stability: Option<crate::stability::Stability>,
    /// True when the UHF (open-shell) path was used.
    pub unrestricted: bool,
    /// Analytic stress tensor `σ = (1/V) ∂E/∂ε` in eV/Bohr³ (eV/Bohr² for a 2-D cell, eV/Bohr
    /// for 1-D — the cell measure it is divided by is length, area, or volume). `None` for a
    /// molecule.
    pub stress: Option<crate::math::Mat3>,
    /// Long-range monopole (Ewald) energy in eV, including the neutralizing background for a
    /// charged cell. `None` for a molecule or in MOPAC-compatibility mode.
    pub ewald_ev: Option<f64>,
    /// The neutralizing-background part of `ewald_ev`. Non-zero only for a charged 3-D cell,
    /// and reported separately because it is a convention-dependent constant, not an
    /// interaction.
    pub background_ev: Option<f64>,
    /// Makov–Payne finite-size estimate for a charged cell (eV), reported as a diagnostic and
    /// never added to the energy.
    pub makov_payne_ev: Option<f64>,
    /// Number of k points actually diagonalized (after time-reversal folding). `None` for a
    /// molecule.
    pub n_kpoints: Option<usize>,
    /// Fermi level in eV. `None` for a molecule or a Γ-point run, where the HOMO/LUMO pair
    /// already says everything a Fermi level would.
    pub fermi_ev: Option<f64>,
    /// Electronic entropy contribution `−TS` in eV from the smearing, if any. Zero without
    /// smearing; reported separately so the free energy and the internal energy stay
    /// distinguishable.
    pub entropy_ev: Option<f64>,
    /// Band energies per k point, in the order of the expanded [`crate::pbc::KPointSet`].
    pub band_energies: Option<Vec<Vec<f64>>>,
}

impl Pm7Result {
    /// Occupation numbers for the α (or, restricted, the spatial) orbitals.
    ///
    /// # Aufbau by index, and why it stays that way
    ///
    /// The lowest [`Self::n_occ`] orbitals are full. For a molecule or a Γ-point run that is the
    /// filling, exactly.
    ///
    /// For a **k-mesh** run it is a count rather than a filling, and the difference is worth
    /// stating. The orbitals reported are the Γ states of the converged Hamiltonian
    /// ([`OrbitalSource::KMeshGamma`]) while `n_occ` is the *per-k band count* carried over from
    /// the molecular-style pre-SCF state. Filling the lowest `n_occ` of them is right for a gapped
    /// insulator, where every k point has the same bands occupied, and wrong for a metal, where a
    /// band can sit below the Fermi level at one k and above it at Γ.
    ///
    /// **Filling by `ε ≤ fermi_ev` instead was tried, and is wrong at the boundary.** Without
    /// smearing `fill_bands` sets `E_F` to the energy of the last state it filled, so for an
    /// insulator the Fermi level *is* the valence-band maximum — and the comparison then turns on
    /// whether the Γ eigenvalue and the mesh-wide maximum agree in their last bits. Diamond
    /// reported its threefold valence top (three orbitals at −10.090991 eV, its own HOMO) as
    /// `0.000` occupied. That fixes metals by breaking the most ordinary insulator there is, which
    /// is not a trade worth making.
    ///
    /// So the count stays and the caller is handed what it needs to see past it. Every surface
    /// that returns these occupations reports `fermi_ev` beside them — `orbitals` on both the
    /// Python API and the CLI was the one that did not, until 0.2.3 — and reports `entropy_ev`,
    /// which is zero exactly when every state is full or empty. A non-zero entropy with a
    /// `KMeshGamma` source says these numbers are a band count and not the mesh's filling. The
    /// fractional occupations themselves are not reconstructible here in any case: they need the
    /// smearing width and the whole mesh, and `mo_energies` is one k point.
    pub fn occupations(&self) -> Vec<f64> {
        let filled = if self.unrestricted { 1.0 } else { 2.0 };
        (0..self.mo_energies.len())
            .map(|i| if i < self.n_occ { filled } else { 0.0 })
            .collect()
    }

    /// Occupation numbers for the β orbitals, for an unrestricted run.
    pub fn occupations_beta(&self) -> Option<Vec<f64>> {
        let energies = self.mo_energies_beta.as_ref()?;
        let occupied = self.n_occ_beta?;
        Some(
            (0..energies.len())
                .map(|i| if i < occupied { 1.0 } else { 0.0 })
                .collect(),
        )
    }

    /// `⟨S²⟩` for an unrestricted solution — MOPAC's `(S**2)`. `None` for anything else.
    ///
    /// # Why it has to be reported and not assumed
    ///
    /// A UHF solution is not a spin eigenfunction. The determinant that minimizes the energy is
    /// generally a mixture of the spin state asked for and higher ones, and nothing in the SCF
    /// prevents that or announces it: an ordinary doublet radical comes back near the exact
    /// `S(S+1) = 0.75` and a broken-symmetry singlet comes back near `1.0`, and both report
    /// `converged: true` in the same words. The number is the only thing that distinguishes them.
    ///
    /// It matters most exactly where [`crate::stability`] is useful. Following a triplet
    /// instability *deliberately* lands on a broken-symmetry solution — that is what makes
    /// stretched H₂ dissociate correctly — so a `follow` run hands back an answer whose spin
    /// contamination is the point, and a caller that cannot see `⟨S²⟩` cannot tell that from a
    /// clean one.
    ///
    /// # The formula
    ///
    /// ```text
    /// ⟨S²⟩ = S_z(S_z + 1) + n_β − Σ_ij |⟨φ_i^α|φ_j^β⟩|²  =  S_z(S_z + 1) + n_β − Tr(P^α P^β)
    /// ```
    ///
    /// The second form holds because the NDDO basis is **orthonormal by construction**, so the
    /// overlap that normally sits between the two densities is the identity. That is a property of
    /// the model, not an approximation made here — but it is also why this is not a drop-in for an
    /// ab-initio code, where `Tr(P^α S P^β S)` is required.
    ///
    /// `None` for a restricted run (where the answer is exactly `S(S+1)` by construction and
    /// saying so would add nothing) and for a **k-mesh** run, where the density this would use is
    /// the `T = 0` block rather than the whole solution and the product of two blocks is not the
    /// quantity in the formula.
    pub fn spin_squared(&self) -> Option<f64> {
        if !self.unrestricted || self.bloch_spin_density.is_some() {
            return None;
        }
        let spin = self.spin_density.as_ref()?;
        let n_beta = self.n_occ_beta?;
        let sz = 0.5 * (self.n_occ as f64 - n_beta as f64);
        // `P^α = (P + P^spin)/2`, `P^β = (P − P^spin)/2`, so
        // `Tr(P^α P^β) = ¼[Tr(P P) − Tr(P^spin P^spin)]` — one pass, no temporaries.
        let total_sq = self.density.frobenius_dot(&self.density);
        let spin_sq = spin.frobenius_dot(spin);
        let overlap = 0.25 * (total_sq - spin_sq);
        Some(sz * (sz + 1.0) + n_beta as f64 - overlap)
    }

    /// The **Mermin electronic** free energy `E − TS` in eV: [`Self::total_ev`] plus the smearing's
    /// entropy term.
    ///
    /// This is the *electronic* free energy of a partially occupied band structure at the smearing
    /// temperature, and it is the quantity the forces differentiate — with smeared occupations
    /// `∂F/∂R` is the Hellmann–Feynman force while `∂E/∂R` is not, which is why ASE asks for it
    /// under `free_energy` and why an optimizer running `force_consistent=True` needs this and not
    /// [`Self::total_ev`].
    ///
    /// It is **not** a thermochemical free energy. No vibrational partition function enters it, so
    /// it is not the Gibbs free energy `G = H − TS_vib` a normal-mode analysis would give; this
    /// crate computes harmonic frequencies but derives no thermochemistry from them (see
    /// `docs/scope.md`). The two share a name and nothing else: this one is electronic, at the
    /// smearing width, and equals [`Self::total_ev`] **exactly** whenever there is no smearing —
    /// which is every molecule and every insulator run at aufbau occupations.
    pub fn free_energy_ev(&self) -> f64 {
        self.total_ev + self.entropy_ev.unwrap_or(0.0)
    }

    /// The frontier gap in eV, resolved across both spin channels.
    ///
    /// For an unrestricted system the physical HOMO is the higher of the two channels' occupied
    /// frontiers and the LUMO the lower of their virtual ones, so taking α alone can report a gap
    /// no electron actually sees.
    pub fn gap_ev(&self) -> Option<f64> {
        let homo = match (self.homo_ev, self.homo_ev_beta) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }?;
        let lumo = match (self.lumo_ev, self.lumo_ev_beta) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }?;
        Some(lumo - homo)
    }
}

pub struct Pm7Calculator {
    pub params: Pm7Parameters,
    pub options: Pm7Options,
}

impl Pm7Calculator {
    pub fn new(params: Pm7Parameters) -> Self {
        Self {
            params,
            options: Pm7Options::default(),
        }
    }
    pub fn with_options(params: Pm7Parameters, options: Pm7Options) -> Self {
        Self { params, options }
    }
    pub fn calculate(&self, molecule: &Molecule) -> Result<Pm7Result> {
        run_pm7(molecule, &self.params, &self.options)
    }
}

struct ScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    mo_energies: Vec<f64>,
    mo_coeff: Matrix,
    n_occ: usize,
    /// The beta orbital set, for an unrestricted run only.
    mo_energies_beta: Option<Vec<f64>>,
    mo_coeff_beta: Option<Matrix>,
    n_occ_beta: Option<usize>,
    electronic_ev: f64,
    converged: bool,
    density_error: f64,
    iterations: usize,
    unrestricted: bool,
}

pub fn run_pm7(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
) -> Result<Pm7Result> {
    let result = run_pm7_once(molecule, params, options)?;
    // Ask whether what converged is a minimum, when the caller asked. The analysis re-enters this
    // function with `stability: Off` to re-converge from a rotated guess, which is where the
    // recursion stops.
    crate::stability::analyse(molecule, params, options, result)
}

fn run_pm7_once(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
) -> Result<Pm7Result> {
    validate_input(molecule, options)?;
    if options.method != params.method {
        return Err(Pm7Error::InvalidInput(format!(
            "options method `{}` does not match parameter method `{}`",
            options.method, params.method
        )));
    }
    if options.multiplicity < 1 {
        return Err(Pm7Error::InvalidInput(
            "multiplicity must be >= 1".to_string(),
        ));
    }
    let basis = Basis::build(molecule, params)?;
    // Pre-flight OOM guard: fail cleanly before the O(n²) integral cache / dense working set.
    let n_atoms = molecule.atoms.len();
    let has_d = options.force_dpath
        || molecule
            .atoms
            .iter()
            .any(|a| params.element(a.z).map(|e| e.has_d()).unwrap_or(false));
    crate::memory::guard(
        basis.nao,
        n_atoms * (n_atoms.saturating_sub(1)) / 2,
        n_atoms,
        has_d,
        false,
        options.max_memory_mb,
    )?;
    // One entry point for molecules and periodic systems: dispatch on the cell, not on which
    // function the caller happened to reach for.
    let pbc = options.pbc_for(molecule);
    if let Some(p) = &pbc {
        p.validate()?;
        if p.mode == crate::pbc::PbcMode::MopacCluster {
            return Err(Pm7Error::InvalidInput(
                "PbcMode::MopacCluster is not implemented yet; use the default \
                 PbcMode::Ewald (see docs/pbc.md)"
                    .into(),
            ));
        }
    }
    let core = match &pbc {
        None => crate::hamiltonian::build_core_field(
            molecule,
            &basis,
            params,
            options.force_dpath,
            options.active_field(),
        )?,
        Some(p) => crate::hamiltonian::build_core_periodic_field(
            molecule,
            &basis,
            params,
            options.force_dpath,
            p,
            options.active_field(),
        )?,
    };

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= options.charge;
    let n_elec_int = n_elec.round() as i64;
    if (n_elec - n_elec_int as f64).abs() > 1.0e-6 || n_elec_int < 0 {
        return Err(Pm7Error::InvalidInput(format!(
            "invalid electron count {n_elec}"
        )));
    }
    let n_unpaired = (options.multiplicity - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Pm7Error::InvalidInput(format!(
            "electron count {n_elec_int} is incompatible with multiplicity {} (need same parity)",
            options.multiplicity
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as usize;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as usize;

    // RHF/UHF selection honours an explicit reference; `Auto` restricts only closed shells.
    let use_uhf = match options.reference {
        ScfReference::Auto => n_alpha != n_beta,
        ScfReference::Unrestricted => true,
        ScfReference::Restricted => {
            if n_alpha != n_beta {
                return Err(Pm7Error::InvalidInput(format!(
                    "restricted (RHF) needs a closed shell, but multiplicity {} gives {n_alpha}α/{n_beta}β; use UHF",
                    options.multiplicity
                )));
            }
            false
        }
    };
    let state = if use_uhf {
        uhf_loop(molecule, &basis, params, &core, n_alpha, n_beta, options)?
    } else {
        rhf_loop(molecule, &basis, params, &core, n_alpha, options)?
    };
    // A k-point mesh replaces the Γ solution with the k-resolved one. The Γ solve above is not
    // wasted: it costs one diagonalization and gives the k path a converged starting density
    // and the molecular-style MO output the result type reports.
    let kstate = match (&pbc, &core.bloch) {
        (Some(p), Some(_)) => {
            let cell = molecule.cell.expect("periodic");
            let set = p.kmesh.expand(&cell)?;
            Some(crate::scf_pbc::run_kpoint_scf(
                molecule, &basis, params, &core, &set, n_alpha, n_beta, options,
            )?)
        }
        _ => None,
    };
    let orbital_source = match (&pbc, &kstate) {
        (None, _) => OrbitalSource::Molecular,
        (Some(_), None) => OrbitalSource::Gamma,
        (Some(_), Some(_)) => OrbitalSource::KMeshGamma,
    };
    let state = match &kstate {
        None => state,
        Some(k) => {
            let zero = k.density.position([0, 0, 0]).expect("zero translation");
            // The orbitals are re-derived at Γ from the **converged** Hamiltonian rather than
            // carried over from the pre-SCF Γ solve. Before this they were mismatched: the
            // energies came from `band_energies[0]` (the first expanded k point, which on a
            // shifted mesh is not Γ at all) while the coefficients came from a Fock built out of
            // the *starting* density. The two were eigen-data of different matrices, one of
            // which had never converged.
            let gamma = crate::scf_pbc::gamma_orbitals(molecule, &basis, params, &core, k)?;
            ScfState {
                // Downstream molecular-style consumers (charges, dipole, the fixed-density
                // gradient) all want the on-site block, which is the T = 0 density.
                density: k.density.block(zero).clone(),
                spin_density: k
                    .spin_density
                    .as_ref()
                    .map(|s| s.block(s.position([0, 0, 0]).unwrap()).clone()),
                mo_energies: gamma.energies,
                mo_coeff: gamma.coefficients,
                n_occ: state.n_occ,
                mo_energies_beta: gamma.energies_beta,
                mo_coeff_beta: gamma.coefficients_beta,
                n_occ_beta: k.unrestricted.then_some(n_beta),
                electronic_ev: k.electronic_ev,
                converged: k.converged,
                density_error: k.density_error,
                iterations: k.iterations,
                unrestricted: k.unrestricted,
            }
        }
    };

    let mut core_ev = match &pbc {
        None => core_core_energy(molecule, params)?,
        Some(p) => crate::repulsion::core_core_energy_periodic(molecule, params, p)?,
    };
    // The nuclear half of the field energy, `Σ_A Z_A (f·R_A)`. MOPAC folds it into `enuclr`
    // (`hcore.F90:208`), so the heat of formation picks the field up without anything else
    // changing — which is what makes a `FIELD=` heat of formation directly comparable.
    if let Some(f) = options.active_field() {
        core_ev += f.core_energy(molecule, params)?;
    }

    // Long-range monopole electrostatics. `E_ew = ½ qᵀ M q` is quadratic in the density but not
    // homogeneous (because `q_A = Z_A − P_A` carries the core charge), so the usual
    // `½ Tr P(H + F)` does not account for it correctly. Working the algebra through, the whole
    // discrepancy is a single `+½ Σ_A Z_A V_A`:
    //
    //   ½ Tr P·F_ewald = ½ Σ_A (Z_A − q_A)(−V_A) = −½ Z·V + E_ew
    //   ⇒  E_remainder + E_ew = [½ Tr P(H + F)] + ½ Z·V
    let (electronic_ev, ewald_ev, background_ev, makov_payne_ev) = match &core.ewald {
        None => (state.electronic_ev, None, None, None),
        Some(ew) => {
            let charges = ew.charges(&basis, &state.density);
            let potential = ew.potential(&charges);
            let e_ew: f64 = 0.5
                * charges
                    .iter()
                    .zip(&potential)
                    .map(|(q, v)| q * v)
                    .sum::<f64>();
            let z_dot_v: f64 = ew
                .core_charge
                .iter()
                .zip(&potential)
                .map(|(z, v)| z * v)
                .sum();
            let cell = molecule.cell.expect("Ewald context implies a cell");
            let mp = pbc
                .as_ref()
                .filter(|p| p.report_makov_payne)
                .and_then(|_| crate::pbc::ewald::makov_payne(&cell, &charges));
            // The background term is reported from a direct evaluation rather than tracked
            // through the matrix, so it stays meaningful if the matrix ever changes.
            let bg = {
                let q_tot: f64 = charges.iter().sum();
                if q_tot.abs() < 1.0e-14 || cell.dim() != 3 {
                    0.0
                } else {
                    let positions: Vec<crate::math::Vec3> =
                        molecule.atoms.iter().map(|a| a.position).collect();
                    let p = pbc.as_ref().expect("periodic");
                    let ep = crate::pbc::EwaldParameters::new(
                        &cell,
                        positions.len(),
                        p.ewald_accuracy,
                        p.ewald_alpha,
                    );
                    -std::f64::consts::PI * q_tot * q_tot
                        / (2.0 * ep.alpha * ep.alpha * cell.measure())
                        * crate::constants::PM7_EV
                }
            };
            (
                state.electronic_ev + 0.5 * z_dot_v,
                Some(e_ew),
                Some(bg),
                mp,
            )
        }
    };

    // Post-SCF corrections (dispersion + H-bond), reported in kcal/mol and added to
    // both the total energy and the heat of formation. Disabled for PM7-minus.
    //
    // These are purely geometric — they do not depend on the density or on k — so a periodic
    // system evaluates them once per cell as a real-space image sum, at the same value for
    // every k-point mesh. They still carry q-dependence into a phonon spectrum through their
    // real-space force constants.
    let correction_kcal = correction_energy(molecule, options);
    let total_ev = electronic_ev + core_ev + correction_kcal * crate::constants::KCAL_TO_EV;

    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let heat_of_formation_kcal =
        (electronic_ev + core_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL + correction_kcal;

    if !state.converged {
        return Err(Pm7Error::ScfNotConverged {
            iterations: state.iterations,
            error: state.density_error,
        });
    }

    // Mulliken net charges from the total density.
    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let mut pop = 0.0;
        for mu in 0..n {
            pop += state.density[(off + mu, off + mu)];
        }
        charges[ia] = params.element(atom.z)?.core_charge - pop;
    }

    // Dipole: point-charge term plus the one-centre s–p and p–d hybrid polarizations.
    //
    // For a periodic system the point-charge part is origin-dependent and therefore not a
    // physical observable (the polarization needs a Berry-phase treatment), so it is left at
    // zero rather than reported as if it meant something.
    let dipole = crate::dipole::breakdown(
        molecule,
        &basis,
        params,
        &state.density,
        &charges,
        options.dipole_origin,
        pbc.is_some(),
    )?;
    let dipole_debye = dipole.total();
    let dipole_magnitude = dipole_debye.norm();

    // The field energy, `f · mu_FieldConjugate`, reported so a caller can subtract it. Note the
    // operator: MOPAC's `FIELD=` has no p–d term even though its dipole does, so this is
    // deliberately **not** `dipole.total()`. See `docs/theory.md` convention C-2.
    let field_ev = options.active_field().map(|f| {
        f.internal()
            .dot(dipole.field_conjugate() * (1.0 / AU_DIPOLE_TO_DEBYE))
    });

    let nao = basis.nao;
    let homo_ev = (state.n_occ >= 1).then(|| state.mo_energies[state.n_occ - 1]);
    let lumo_ev = (state.n_occ < nao).then(|| state.mo_energies[state.n_occ]);
    let (homo_ev_beta, lumo_ev_beta) = match (&state.mo_energies_beta, state.n_occ_beta) {
        (Some(eps), Some(occupied)) => (
            (occupied >= 1).then(|| eps[occupied - 1]),
            (occupied < nao).then(|| eps[occupied]),
        ),
        _ => (None, None),
    };

    Ok(Pm7Result {
        stability: None,
        density: state.density,
        spin_density: state.spin_density,
        bloch_density: kstate.as_ref().map(|k| k.density.clone()),
        bloch_spin_density: kstate.as_ref().and_then(|k| k.spin_density.clone()),
        mo_energies: state.mo_energies,
        mo_coeff: state.mo_coeff,
        n_occ: state.n_occ,
        mo_energies_beta: state.mo_energies_beta,
        mo_coeff_beta: state.mo_coeff_beta,
        n_occ_beta: state.n_occ_beta,
        homo_ev_beta,
        lumo_ev_beta,
        orbital_source,
        electronic_ev,
        core_ev,
        total_ev,
        heat_of_formation_kcal,
        charges,
        dipole_debye,
        dipole_magnitude,
        dipole,
        field_ev,
        homo_ev,
        lumo_ev,
        iterations: state.iterations,
        converged: state.converged,
        unrestricted: state.unrestricted,
        // Filled in by `crate::stress::analytic_stress`, which needs the converged density.
        stress: None,
        ewald_ev,
        background_ev,
        makov_payne_ev,
        n_kpoints: kstate
            .as_ref()
            .map(|k| k.n_kpoints)
            .or(pbc.as_ref().map(|_| 1)),
        fermi_ev: kstate.as_ref().map(|k| k.fermi_ev),
        entropy_ev: kstate.as_ref().map(|k| k.entropy_ev),
        band_energies: kstate.as_ref().map(|k| k.band_energies.clone()),
    })
}

/// Post-SCF correction energy (kcal/mol) for a molecule or a periodic cell.
///
/// The image cutoff comes from [`crate::pbc::PbcOptions::correction_cutoff`]; a molecule sums
/// every pair regardless, so this is the same number the molecular path always produced.
/// Heat of formation in kcal/mol from a **total** energy, for a driver that does not produce a
/// [`Pm7Result`].
///
/// `run_pm7` computes this inline from its own `electronic + core` split. Divide and conquer has
/// no such split to hand back -- `dandc_derivatives` returns the total -- so the same reference
/// subtraction is expressed here in terms of the total instead, and the two agree by construction:
///
/// ```text
/// total   = electronic + core + correction_kcal * KCAL_TO_EV
/// dHf     = (electronic + core - e_isol + eheat) * EV_TO_KCAL + correction_kcal
///         = (total - correction_kcal * KCAL_TO_EV - e_isol + eheat) * EV_TO_KCAL + correction_kcal
/// ```
///
/// Having one function rather than a second transcription of the reference sums is the point: a
/// divide-and-conquer optimization that reported a heat of formation off by the isolated-atom
/// terms would look entirely plausible.
pub fn heat_of_formation_from_total(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    total_ev: f64,
) -> Result<f64> {
    let correction_kcal = correction_energy(molecule, options);
    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let without_correction = total_ev - correction_kcal * crate::constants::KCAL_TO_EV;
    Ok((without_correction - e_isol_sum + eheat_sum) * EV_TO_KCAL + correction_kcal)
}

pub fn correction_energy(molecule: &Molecule, options: &Pm7Options) -> f64 {
    if !options.method.has_post_scf_corrections() {
        return 0.0;
    }
    let cutoff = options
        .pbc_for(molecule)
        .map(|p| p.correction_cutoff)
        .unwrap_or(f64::INFINITY);
    let mut c = if options.smooth_dispersion {
        crate::dispersion::dispersion_energy_smooth_cut(molecule, cutoff)
    } else {
        crate::dispersion::dispersion_energy_cut(molecule, cutoff)
    };
    c += crate::hbond::hydrogen_bond_energy(molecule);
    // PM7-HH adds an extra hydrogen–hydrogen repulsion term.
    if options.method.has_hh_repulsion() {
        c += crate::hh_rep::hh_repulsion_energy_cut(molecule, cutoff);
    }
    // MOPAC's molecular-mechanics corrections to the heat of formation (`compfg.F90:370-378`).
    // They are not part of the SCF and not part of the core–core repulsion, which is why the
    // hundred-case oracle found them as a heat-of-formation gap on molecules whose orbital
    // energies and Mulliken charges agreed exactly. See [`crate::mm_corrections`].
    c += crate::mm_corrections::c_triple_bond_energy(molecule);
    c += crate::mm_corrections::si_o_h_energy(molecule);
    c
}

/// Reject the inputs that would otherwise fail deep inside a kernel, where the message would name
/// a matrix index rather than an atom. Every entry point that owns a whole calculation calls this
/// first — [`run_pm7`] and [`crate::dandc::run_dandc`] alike.
pub(crate) fn validate_input(molecule: &Molecule, options: &Pm7Options) -> Result<()> {
    if molecule.is_empty() {
        return Err(Pm7Error::InvalidInput(
            "a PM7 calculation needs at least one atom".into(),
        ));
    }
    if !options.charge.is_finite() {
        return Err(Pm7Error::InvalidInput("charge must be finite".into()));
    }
    if options.max_scf == 0 {
        return Err(Pm7Error::InvalidInput(
            "max_scf must be greater than zero".into(),
        ));
    }
    if !options.e_tol.is_finite() || options.e_tol <= 0.0 {
        return Err(Pm7Error::InvalidInput(
            "e_tol must be finite and positive".into(),
        ));
    }
    if !options.p_tol.is_finite() || options.p_tol <= 0.0 {
        return Err(Pm7Error::InvalidInput(
            "p_tol must be finite and positive".into(),
        ));
    }
    if !options.level_shift.is_finite() || options.level_shift < 0.0 {
        return Err(Pm7Error::InvalidInput(
            "level_shift must be finite and non-negative".into(),
        ));
    }
    if !options.adiis_switch.is_finite() || options.adiis_switch < 0.0 {
        return Err(Pm7Error::InvalidInput(
            "adiis_switch must be finite and non-negative".into(),
        ));
    }
    if let Some((inner, outer)) = options.exchange_cutoff {
        if !inner.is_finite() || !outer.is_finite() || inner < 0.0 || outer <= inner {
            return Err(Pm7Error::InvalidInput(
                "exchange_cutoff must satisfy 0 <= inner < outer (Bohr)".into(),
            ));
        }
    }
    if let Some(f) = &options.field {
        crate::field::validate_for(molecule, f)?;
    }
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let p = atom.position;
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Err(Pm7Error::InvalidInput(format!(
                "atom {} has non-finite coordinates",
                index + 1
            )));
        }
    }
    coincident_atoms(molecule)?;
    Ok(())
}

/// Reject two atoms at the same point, in `O(N)` rather than `O(N²)`.
///
/// The pairwise double loop this replaces ran at the top of **every** `run_pm7`, so a
/// 2000-atom divide-and-conquer job spent two million iterations before any physics happened.
/// Bucketing on a lattice whose spacing is far larger than the coincidence threshold makes the
/// check exact rather than approximate: two atoms closer than `1e-10` Bohr cannot be more than
/// one bucket apart, so scanning a bucket and its 26 neighbours cannot miss a pair.
fn coincident_atoms(molecule: &Molecule) -> Result<()> {
    use std::collections::HashMap;
    // Threshold is `norm² < 1e-20`, i.e. a separation below `1e-10` Bohr.
    const SPACING: f64 = 1.0e-6;
    let key = |p: crate::math::Vec3| -> [i64; 3] {
        [
            (p.x / SPACING).floor() as i64,
            (p.y / SPACING).floor() as i64,
            (p.z / SPACING).floor() as i64,
        ]
    };
    let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::with_capacity(molecule.atoms.len());
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let cell = key(atom.position);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbour = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    let Some(others) = buckets.get(&neighbour) else {
                        continue;
                    };
                    for &j in others {
                        if (atom.position - molecule.atoms[j].position).norm2() < 1.0e-20 {
                            // Report in the original order, so the message does not depend on
                            // which atom happened to be inserted first.
                            let (a, b) = (j.min(i), j.max(i));
                            return Err(Pm7Error::InvalidInput(format!(
                                "atoms {} and {} occupy the same position",
                                a + 1,
                                b + 1
                            )));
                        }
                    }
                }
            }
        }
        buckets.entry(cell).or_default().push(i);
    }
    Ok(())
}

/// MOPAC's diagonal-density initial guess (`moldat.F90:731-790`): the core charge is smeared
/// **evenly over the sp orbitals** (each gets `tore/4`, *not* the atomic ground-state occupation),
/// the d shell is empty for main-group atoms / aufbau-filled for transition metals, and the net
/// molecular charge `yy = charge/n_ao` is spread over every AO. Reproducing MOPAC's exact starting
/// point makes pm7-rs converge to the **same SCF stationary point** as MOPAC. This is decisive for
/// the ionic/covalent-bistable diatomics (BF, AsF, AlN) where the SCF has two aufbau-valid
/// solutions and the guess selects the basin; unique-minimum systems (the vast majority) are
/// unaffected. `n_electrons` is the total electron count (`2·n_occ` RHF, `n_alpha + n_beta` UHF).
pub(crate) fn sad_density(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    n_electrons: f64,
) -> Result<Matrix> {
    let nao = basis.nao;
    let mut p = Matrix::zeros(nao, nao);
    let total_tore: f64 = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum();
    let yy = if nao > 0 {
        (total_tore - n_electrons) / nao as f64
    } else {
        0.0
    };
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let tore = elem.core_charge;
        let z = atom.z;
        match n {
            1 => p[(off, off)] = tore - yy,
            4 => {
                let w = tore * 0.25 - yy;
                for k in 0..4 {
                    p[(off + k, off + k)] = w;
                }
            }
            9 => {
                let main_group = z < 21 || (z > 30 && z < 39) || (z > 48 && z < 57);
                if main_group {
                    let w = tore * 0.25 - yy;
                    for k in 0..4 {
                        p[(off + k, off + k)] = w;
                    }
                    for k in 4..9 {
                        p[(off + k, off + k)] = -yy;
                    }
                } else {
                    // Transition metal: fill s (≤2), then the d shell, then the p shell.
                    let mut sum = tore - 9.0 * yy;
                    p[(off, off)] = sum.clamp(0.0, 2.0);
                    sum -= 2.0;
                    if sum > 0.0 {
                        for k in 4..9 {
                            p[(off + k, off + k)] = (sum * 0.2).clamp(0.0, 2.0);
                        }
                        sum -= 10.0;
                        if sum > 0.0 {
                            for k in 1..4 {
                                p[(off + k, off + k)] = sum / 3.0;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(p)
}

/// Build a density `P = w Σ_{k<n_occ} c_k c_kᵀ` from MO coefficients (`w` = 2 for RHF, 1 for UHF).
///
/// `P[μ,ν] = w · (row μ of C, occupied part) · (row ν of C, occupied part)`. Each output row is
/// an independent set of contiguous dot products, so this parallelizes over rows (race-free) and
/// the inner dot auto-vectorizes — the dominant per-SCF-iteration cost at large `n_basis`.
fn density_from_coeff(c: &Matrix, n_occ: usize, weight: f64) -> Matrix {
    use rayon::prelude::*;
    let nao = c.rows;
    let mut p = Matrix::zeros(nao, nao);
    if nao == 0 {
        return p;
    }
    let cs = c.as_slice(); // row-major nao×nao; occupied MOs are columns 0..n_occ
    p.as_mut_slice()
        .par_chunks_mut(nao)
        .enumerate()
        .for_each(|(mu, prow)| {
            let cmu = &cs[mu * nao..mu * nao + n_occ];
            for (nu, pv) in prow.iter_mut().enumerate() {
                let cnu = &cs[nu * nao..nu * nao + n_occ];
                let dot: f64 = cmu.iter().zip(cnu).map(|(a, b)| a * b).sum();
                *pv = weight * dot;
            }
        });
    p
}

/// SCF error `[F, P] = FP − PF` in the orthonormal NDDO basis. Both `F` and `P` are symmetric,
/// so `PF = (FP)ᵀ` **exactly** (term-by-term identical sums), and the antisymmetric commutator is
/// `FP − (FP)ᵀ`. This needs only one matmul instead of two — bit-identical to the two-matmul form.
/// Written into a caller-supplied buffer, so an SCF history slot can be reused rather than
/// reallocated. Every caller has somewhere to put it, so there is no owning variant.
fn commutator_into(f: &Matrix, p: &Matrix, out: &mut Matrix) {
    let fp = f.matmul(p);
    let n = fp.rows;
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = fp[(i, j)] - fp[(j, i)];
        }
    }
}

/// Make room at the end of a bounded history and hand back the slot to write into.
///
/// Below `max` this grows the history; **at** `max` it rotates the oldest entry to the end and
/// returns its buffer, so the steady state touches the allocator not at all. Rotating rather than
/// indexing a ring keeps the history in **chronological order**, which both extrapolators depend
/// on — each treats the last entry as the current iteration's — so the accelerators need no
/// changes and the arithmetic is untouched.
///
/// What this saves is real but bounded: at depth 8 the loop used to free and reallocate three
/// `nao × nao` matrices every iteration. The `rotate_left` that replaces it moves eight `Matrix`
/// *headers*, not their data.
fn advance_history(history: &mut Vec<Matrix>, n: usize, max: usize) -> &mut Matrix {
    if history.len() < max {
        history.push(Matrix::zeros(n, n));
    } else {
        history.rotate_left(1);
    }
    history.last_mut().expect("just pushed or rotated")
}

fn rms_diff(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.as_slice().len().max(1);
    (a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n as f64)
        .sqrt()
}

fn rhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    n_occ: usize,
    options: &Pm7Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    // The caller's density when there is one, and the superposition of atomic densities otherwise.
    // A seeded guess has to be the right shape or it would corrupt the run silently, so a wrong one
    // is refused rather than reshaped.
    let mut density = match &options.initial_density {
        Some(d) if d.rows == nao && d.cols == nao => d.clone(),
        Some(d) => {
            return Err(Pm7Error::InvalidInput(format!(
                "the supplied initial density is {}x{} and this system's basis is {nao}x{nao}",
                d.rows, d.cols
            )))
        }
        None => sad_density(molecule, basis, params, 2.0 * n_occ as f64)?,
    };
    let mut e_old = 0.0;
    let mut mo_energies = vec![0.0; nao];
    let mut mo_coeff = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut density_error = f64::INFINITY;
    let mut iterations = 0;
    let mut diis_f: Vec<Matrix> = Vec::new();
    let mut diis_e: Vec<Matrix> = Vec::new();
    let mut diis_d: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let accel = if options.use_diis {
        options.accelerator
    } else {
        ScfAccelerator::None
    };
    let mut guard = StallGuard::new(options.level_shift);
    // Per-iteration SCF trace to stderr. Hoisted out of the loop: an env lookup per iteration
    // is a real cost on the small-molecule runs this crate is fastest at.
    let trace = std::env::var_os("PM7_SCF_TRACE").is_some();

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let f = {
            let _t = crate::profile::stage("scf: Fock build");
            build_fock(molecule, basis, params, core, &density)?
        };
        let e_elec = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f));

        // History (Fock, commutator, density) for CDIIS / A-DIIS, written **into reused buffers**.
        // Pushing clones and dropping the oldest freed and reallocated three `nao × nao` matrices
        // every iteration once the history was full; the commutator was allocated fresh on top of
        // that. The history stays chronological, so the extrapolators are unchanged.
        advance_history(&mut diis_f, nao, max_diis).copy_from(&f);
        commutator_into(&f, &density, advance_history(&mut diis_e, nao, max_diis));
        advance_history(&mut diis_d, nao, max_diis).copy_from(&density);
        let err_norm = diis_e
            .last()
            .expect("just written")
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        let mut f_use = match accel {
            ScfAccelerator::None => f,
            ScfAccelerator::Cdiis => {
                diis_extrapolate(&diis_f, &diis_e).unwrap_or_else(|| f.clone())
            }
            ScfAccelerator::AdiisCdiis => {
                if err_norm > options.adiis_switch {
                    adiis_extrapolate(&diis_d, &diis_f).unwrap_or_else(|| f.clone())
                } else {
                    diis_extrapolate(&diis_f, &diis_e).unwrap_or_else(|| f.clone())
                }
            }
        };
        apply_level_shift(&mut f_use, &density, 2.0, guard.shift);
        let (eps, c) = {
            let _t = crate::profile::stage("scf: diagonalize");
            symmetric_eigen(&f_use)?
        };
        let p_new = density_from_coeff(&c, n_occ, 2.0);
        let dp = rms_diff(&p_new, &density);
        density_error = dp;
        let de = (e_elec - e_old).abs();

        mo_energies = eps;
        mo_coeff = c;
        density = p_new;
        e_old = e_elec;
        // `err_norm` is only a faithful residual on the unshifted iteration (see `StallGuard`).
        let residual_ok = guard.shift == 0.0 && err_norm < 1.0e-7;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || residual_ok) {
            converged = true;
            break;
        }
        guard.update(dp, e_elec);
        if trace {
            eprintln!(
                "scf {iterations:4} E {e_elec:.10} dp {dp:.3e} err {err_norm:.3e} shift {:.2}",
                guard.shift
            );
        }
    }
    let f_final = build_fock(molecule, basis, params, core, &density)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final));
    if guard.shift > 0.0 {
        // The last diagonalization was of a shifted Fock, whose virtual eigenvalues are all
        // `shift` too high. Report the canonical orbitals of the real Fock instead.
        let (eps, c) = symmetric_eigen(&f_final)?;
        mo_energies = eps;
        mo_coeff = c;
    }

    Ok(ScfState {
        density,
        spin_density: None,
        mo_energies,
        mo_coeff,
        n_occ,
        mo_energies_beta: None,
        mo_coeff_beta: None,
        n_occ_beta: None,
        electronic_ev,
        converged,
        density_error,
        iterations,
        unrestricted: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn uhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    n_alpha: usize,
    n_beta: usize,
    options: &Pm7Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    // MOPAC's guess split by spin population. The different α/β aufbau counts break spin symmetry
    // **for an open shell**; for `n_alpha == n_beta` the two halves are identical, the spin density
    // is exactly zero, and the UHF equations keep it there — so a forced-UHF closed shell returns
    // the RHF solution and cannot reach a spin-broken one. `initial_spin_density` is how a caller
    // who knows better says so; `crate::stability` gets it from the triplet instability's
    // eigenvector.
    let total = match &options.initial_density {
        Some(d) if d.rows == nao && d.cols == nao => d.clone(),
        Some(d) => {
            return Err(Pm7Error::InvalidInput(format!(
                "the supplied initial density is {}x{} and this system's basis is {nao}x{nao}",
                d.rows, d.cols
            )))
        }
        None => sad_density(molecule, basis, params, (n_alpha + n_beta) as f64)?,
    };
    let n_tot = (n_alpha + n_beta).max(1) as f64;
    let (fa, fb) = (n_alpha as f64 / n_tot, n_beta as f64 / n_tot);
    let mut pa = total.clone();
    for v in pa.as_mut_slice() {
        *v *= fa;
    }
    let mut pb = total;
    for v in pb.as_mut_slice() {
        *v *= fb;
    }
    if let Some(spin) = &options.initial_spin_density {
        if spin.rows != nao || spin.cols != nao {
            return Err(Pm7Error::InvalidInput(format!(
                "the supplied initial spin density is {}x{} and this system's basis is {nao}x{nao}",
                spin.rows, spin.cols
            )));
        }
        // `P^α += s/2`, `P^β -= s/2`, so the total is untouched and the spin density becomes `s`.
        for ((a, b), s) in pa
            .as_mut_slice()
            .iter_mut()
            .zip(pb.as_mut_slice())
            .zip(spin.as_slice())
        {
            *a += 0.5 * *s;
            *b -= 0.5 * *s;
        }
    }
    let mut e_old = 0.0;
    let mut eps_a = vec![0.0; nao];
    let mut c_a = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut density_error = f64::INFINITY;
    let mut iterations = 0;

    let mut hist_fa: Vec<Matrix> = Vec::new();
    let mut hist_fb: Vec<Matrix> = Vec::new();
    let mut hist_err: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let mut guard = StallGuard::new(options.level_shift);
    let trace = std::env::var_os("PM7_SCF_TRACE").is_some();
    // Reused across iterations rather than allocated inside the loop.
    let (mut ea, mut eb) = (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao));

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let mut p_tot = pa.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
            *t += *b;
        }
        let fa = build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
        let fb = build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;

        let e_elec = 0.5
            * (p_tot.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

        // Combined DIIS error = [F_a,P_a] ⊕ [F_b,P_b].
        //
        // The two commutators go into buffers reused across iterations, and the stacked
        // `2·nao × nao` matrix is built **only** where it is consumed — inside the CDIIS branch
        // below, which is currently disabled. It used to be allocated, filled and dropped on every
        // iteration to produce one scalar.
        //
        // The norm chains the two slices rather than summing them separately, so the accumulation
        // runs in the same order over the same values as the stacked matrix did and `err_norm` is
        // bit-identical. `err_norm` drives the stall guard, and a last-bit change there can move
        // an iteration count.
        commutator_into(&fa, &pa, &mut ea);
        commutator_into(&fb, &pb, &mut eb);
        let err_norm = ea
            .as_slice()
            .iter()
            .chain(eb.as_slice())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        // A plain CDIIS extrapolation can converge UHF radicals to a higher-energy
        // spin-broken fixed point (PM7 CH3 is a reproducible example).  Keep the
        // UHF path variationally stable until the shared energy-monotone A-DIIS
        // controller is extended to coupled alpha/beta histories; RHF retains its
        // existing A-DIIS/CDIIS accelerator.
        const UHF_CDIIS_ENABLED: bool = false;
        let (fa_use, fb_use) = if UHF_CDIIS_ENABLED && options.use_diis {
            let mut err = Matrix::zeros(2 * nao, nao);
            for i in 0..nao {
                for j in 0..nao {
                    err[(i, j)] = ea[(i, j)];
                    err[(nao + i, j)] = eb[(i, j)];
                }
            }
            hist_fa.push(fa.clone());
            hist_fb.push(fb.clone());
            hist_err.push(err);
            if hist_fa.len() > max_diis {
                hist_fa.remove(0);
                hist_fb.remove(0);
                hist_err.remove(0);
            }
            match diis_coeffs(&hist_err) {
                Some((skip, coeffs)) => (
                    combine(&hist_fa[skip..], &coeffs),
                    combine(&hist_fb[skip..], &coeffs),
                ),
                None => (fa, fb),
            }
        } else {
            (fa, fb)
        };

        let (mut fa_use, mut fb_use) = (fa_use, fb_use);
        apply_level_shift(&mut fa_use, &pa, 1.0, guard.shift);
        apply_level_shift(&mut fb_use, &pb, 1.0, guard.shift);
        let (ea_eps, ca) = symmetric_eigen(&fa_use)?;
        let (_eb_eps, cb) = symmetric_eigen(&fb_use)?;
        let pa_new = density_from_coeff(&ca, n_alpha, 1.0);
        let pb_new = density_from_coeff(&cb, n_beta, 1.0);

        let dp = rms_diff(&pa_new, &pa) + rms_diff(&pb_new, &pb);
        density_error = dp;
        let de = (e_elec - e_old).abs();
        eps_a = ea_eps;
        c_a = ca;
        pa = pa_new;
        pb = pb_new;
        e_old = e_elec;
        // `err_norm` is only a faithful residual on the unshifted iteration (see `StallGuard`).
        let residual_ok = guard.shift == 0.0 && err_norm < 1.0e-7;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || residual_ok) {
            converged = true;
            break;
        }
        guard.update(dp, e_elec);
        if trace {
            eprintln!(
                "scf {iterations:4} E {e_elec:.10} dp {dp:.3e} err {err_norm:.3e} shift {:.2} (uhf)",
                guard.shift
            );
        }
    }

    let mut density = pa.clone();
    for (t, b) in density.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *t += *b;
    }
    let mut spin = pa.clone();
    for (s, b) in spin.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *s -= *b;
    }
    // Final energy.
    let fa = build_fock_spin(molecule, basis, params, core, &density, &pa)?;
    let fb = build_fock_spin(molecule, basis, params, core, &density, &pb)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));
    if guard.shift > 0.0 {
        // Report the canonical α orbitals of the real Fock, not the shifted one.
        let (eps, c) = symmetric_eigen(&fa)?;
        eps_a = eps;
        c_a = c;
    }
    // The β set, from the same converged Fock. The loop itself never needs it — it propagates
    // densities, not orbitals — but a caller asking for the orbitals of an open-shell system
    // wants both spins, and `analytic_hessian_uhf` was already re-diagonalizing here to recover
    // them.
    let (eps_b, c_b) = symmetric_eigen(&fb)?;

    Ok(ScfState {
        density,
        spin_density: Some(spin),
        mo_energies: eps_a,
        mo_coeff: c_a,
        n_occ: n_alpha,
        mo_energies_beta: Some(eps_b),
        mo_coeff_beta: Some(c_b),
        n_occ_beta: Some(n_beta),
        electronic_ev,
        converged,
        density_error,
        iterations,
        unrestricted: true,
    })
}

/// Largest automatic level shift, in eV. Deliberately generous: the shift never moves the
/// converged answer, so the only cost of a large one is a slower approach, whereas the cost of
/// too small a cap is a run that never converges at all.
const MAX_LEVEL_SHIFT: f64 = 64.0;

/// Iterations per comparison window. Progress is judged between two consecutive windows rather
/// than between neighbouring iterations: a healthy DIIS run's step size bounces around by an
/// order of magnitude from one iteration to the next, so a per-iteration test reads ordinary
/// convergence as a stall and shifts a well-behaved molecule for no reason.
const STALL_WINDOW: usize = 5;

/// Energy improvement (eV) over one window that still counts as progress. Deliberately tiny:
/// its job is only to separate "creeping downhill" from "not moving", and the two are orders
/// of magnitude apart in practice — a slow molecular descent gains ~1e-2 eV per window while a
/// stalled periodic cell gains nothing at all (its energy drifts *up* by ~1e-8).
const ENERGY_PROGRESS_EV: f64 = 1.0e-9;

/// Add `shift · (1 − Q)` to a Fock matrix, where `Q = P / occupancy` projects onto the
/// occupied subspace.
///
/// Raising the virtual manifold damps the density's response to the Fock **without moving the
/// fixed point**: at self-consistency `Q` projects onto `F`'s own occupied eigenvectors, so the
/// shift relabels virtual eigenvalues and leaves the occupied subspace — hence the density —
/// exactly where it was. The NDDO basis is orthonormal, so no overlap matrix appears.
fn apply_level_shift(f: &mut Matrix, density: &Matrix, occupancy: f64, shift: f64) {
    if shift <= 0.0 {
        return;
    }
    let n = f.rows;
    let inv = 1.0 / occupancy;
    for i in 0..n {
        for j in 0..n {
            let q = density[(i, j)] * inv;
            let delta = if i == j { 1.0 - q } else { -q };
            f[(i, j)] += shift * delta;
        }
    }
}

/// Tracks whether the accelerator is still making progress and, when it is not, raises a level
/// shift to contract the SCF map.
///
/// A periodic Γ-point cell couples every atom's charge to every other through the Madelung
/// potential, which can make the bare SCF map expansive even at a healthy 5 eV gap: undamped
/// iteration on a 4-atom silicon cell oscillates at `1e-1`, and DIIS on its own settles into a
/// limit cycle around `1e-6` — good enough for the energy, fatal for an MD run that has to
/// converge at every step.
struct StallGuard {
    floor: f64,
    shift: f64,
    steps: Vec<f64>,
    energies: Vec<f64>,
}

impl StallGuard {
    fn new(floor: f64) -> Self {
        Self {
            floor,
            shift: floor,
            steps: Vec::new(),
            energies: Vec::new(),
        }
    }

    /// Feed the current density change and re-decide the shift.
    ///
    /// The density change, not the commutator norm: shifting by `b` makes the commutator of the
    /// *unshifted* Fock pick up a `b/2 · [P_prev, P_new]` term, so `‖[F,P]‖` stays flat while
    /// the shift doubles and the step halves. Reading that as a stall makes the controller
    /// chase its own tail all the way to the cap.
    ///
    /// The extrapolation history needs no attention when the shift moves: it stores the raw
    /// Focks and their commutators, and the shift is applied *after* extrapolation.
    fn update(&mut self, step: f64, energy: f64) {
        self.steps.push(step);
        self.energies.push(energy);
        let n = self.steps.len();
        if n < 2 * STALL_WINDOW {
            return;
        }
        let smallest = |w: &[f64]| w.iter().copied().fold(f64::INFINITY, f64::min);
        let step_now = smallest(&self.steps[n - STALL_WINDOW..]);
        let step_was = smallest(&self.steps[n - 2 * STALL_WINDOW..n - STALL_WINDOW]);
        let energy_now = smallest(&self.energies[n - STALL_WINDOW..]);
        let energy_was = smallest(&self.energies[n - 2 * STALL_WINDOW..n - STALL_WINDOW]);

        let moved = if step_now <= step_was * 0.1 {
            // Converging fast; the shift is not earning its keep. Ease it off so the last
            // iterations are unshifted ones.
            if self.shift <= self.floor {
                false
            } else {
                let relaxed = (self.shift * 0.5).max(self.floor);
                self.shift = if relaxed < 0.25 { self.floor } else { relaxed };
                true
            }
        } else if step_now > step_was * 0.9
            && energy_now > energy_was - ENERGY_PROGRESS_EV
            && self.shift < MAX_LEVEL_SHIFT
        {
            // Neither the step nor the energy improved across two windows. Both tests matter:
            // a hard *molecular* SCF can creep along a shallow valley for hundreds of
            // iterations with a flat step size while the energy falls steadily, and shifting
            // that run only slows the descent. Only a genuinely motionless run gets a shift.
            self.shift = if self.shift < 1.0 {
                1.0
            } else {
                (self.shift * 2.0).min(MAX_LEVEL_SHIFT)
            };
            true
        } else {
            false
        };
        if moved {
            // Judge the new shift on its own evidence, not the window that provoked it.
            self.steps.clear();
            self.energies.clear();
        } else {
            self.steps.remove(0);
            self.energies.remove(0);
        }
    }
}

fn combine(fs: &[Matrix], coeffs: &[f64]) -> Matrix {
    let (r, c) = (fs[0].rows, fs[0].cols);
    let mut out = Matrix::zeros(r, c);
    for (i, f) in fs.iter().enumerate() {
        let ci = coeffs[i];
        for (o, v) in out.as_mut_slice().iter_mut().zip(f.as_slice()) {
            *o += ci * v;
        }
    }
    out
}

/// Largest `Σ|c_i|` an extrapolation may have before its window is judged unreliable.
///
/// A healthy DIIS step mixes the history with weights of order one. Once the error vectors
/// turn linearly dependent the bordered solve still *succeeds*, but returns huge weights that
/// cancel to something meaningless — which is precisely how DIIS stagnates in a limit cycle
/// rather than failing outright.
const DIIS_MAX_WEIGHT: f64 = 25.0;

/// Solve the Pulay DIIS coefficient system from a stack of error matrices.
///
/// Returns `(skip, coeffs)`: the extrapolation uses `es[skip..]`, i.e. the newest
/// `es.len() - skip` entries. Dropping the oldest vectors is the standard cure for the
/// linear dependence that builds up as the errors shrink; without it a run can sit at
/// `~1e-5` forever, which is what a periodic Γ-point SCF did during MD.
fn diis_coeffs(es: &[Matrix]) -> Option<(usize, Vec<f64>)> {
    let n = es.len();
    if n < 2 {
        return None;
    }
    use rayon::prelude::*;
    // The error-overlap matrix is symmetric (`⟨e_i|e_j⟩ = ⟨e_j|e_i⟩`), so evaluate only the
    // upper triangle — one frobenius dot per (i,j) — in parallel, then mirror. Every
    // candidate window is a trailing sub-block of this one matrix, so it is built once.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let vals: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| ((i, j), es[i].frobenius_dot(&es[j])))
        .collect();
    let mut gram = vec![0.0; n * n];
    for ((i, j), v) in vals {
        gram[i * n + j] = v;
        gram[j * n + i] = v;
    }
    for skip in 0..(n - 1) {
        if let Some(c) = diis_window_coeffs(&gram, n, skip) {
            if c[..n - skip].iter().map(|v| v.abs()).sum::<f64>() <= DIIS_MAX_WEIGHT {
                return Some((skip, c));
            }
        }
    }
    None
}

/// Coefficients for the trailing window `gram[skip.., skip..]` of the error Gram matrix.
fn diis_window_coeffs(gram: &[f64], n: usize, skip: usize) -> Option<Vec<f64>> {
    let m = n - skip;
    if m < 2 {
        return None;
    }
    // Scale the Gram block to a unit diagonal before bordering it. The coefficients are
    // invariant under this (it only rescales the Lagrange multiplier), but it puts the block
    // and the `-1` border on the same footing, so the pivot guard below is a *relative*
    // test rather than one that passes trivially once the errors are ~1e-5.
    let mut scale = 0.0f64;
    for i in 0..m {
        scale = scale.max(gram[(skip + i) * n + (skip + i)].abs());
    }
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let dim = m + 1;
    let mut b = Matrix::zeros(dim, dim);
    for i in 0..m {
        for j in 0..m {
            b[(i, j)] = gram[(skip + i) * n + (skip + j)] / scale;
        }
        b[(i, m)] = -1.0;
        b[(m, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[m] = -1.0;
    // The DIIS matrix (a small bordered saddle-point system) becomes singular near
    // convergence when the error vectors turn linearly dependent. A pivot-guarded
    // Gaussian elimination returns `None` there so the caller falls back to the plain
    // Fock — faer's LU instead returns a degenerate solution that derails DIIS. The heavy
    // O(n^3) eigendecomposition still uses faer; only this tiny solve is bespoke.
    let c = solve_bordered_small(&b, &rhs)?;
    if c.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(c)
}

/// Gaussian elimination with partial pivoting for the small DIIS system; returns `None`
/// if the matrix is (near-)singular.
fn solve_bordered_small(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    let mut m = a.clone();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[(col, col)].abs();
        for row in (col + 1)..n {
            let v = m[(row, col)].abs();
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
                let t = m[(col, j)];
                m[(col, j)] = m[(pivot, j)];
                m[(pivot, j)] = t;
            }
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = m[(row, col)] / m[(col, col)];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                let v = m[(col, j)];
                m[(row, j)] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= m[(col, j)] * x[j];
        }
        x[col] = sum / m[(col, col)];
    }
    Some(x)
}

fn diis_extrapolate(fs: &[Matrix], es: &[Matrix]) -> Option<Matrix> {
    let (skip, coeffs) = diis_coeffs(es)?;
    Some(combine(&fs[skip..], &coeffs))
}

/// A-DIIS (Hu & Yang 2010): minimize `f(c) = 2 Σ c_i ⟨D_i−D_n|F_n⟩ + Σ c_i c_j ⟨D_i−D_n|F_j−F_n⟩`
/// over the simplex `{c ≥ 0, Σc = 1}`, then return the extrapolated Fock `Σ c_i F_i`.
/// Robust far from convergence (nonnegative weights prevent the runaway extrapolation that
/// plain DIIS can produce with a poor initial guess).
fn adiis_extrapolate(densities: &[Matrix], focks: &[Matrix]) -> Option<Matrix> {
    let n = densities.len();
    if n < 2 {
        return None;
    }
    use rayon::prelude::*;
    let dn = &densities[n - 1];
    let fnl = &focks[n - 1];

    // The differences are **not** materialized. Building `D_i − D_n` and `F_j − F_n` allocated
    // `2k` matrices per call — sixteen at the default depth — filled them, dotted them and threw
    // them away, on every iteration far from convergence.
    //
    // Fusing the subtraction into the dot product is the *same arithmetic in the same order*:
    // `frobenius_dot` is a plain in-order fold over the slice, so `Σ (a−a')(b−b')` computed in one
    // pass is bit-for-bit what the two-step version produced. Expanding it algebraically into four
    // dot products would not be — that reassociates the rounding — so it is deliberately not done
    // that way, and the A-DIIS weights are unchanged to the last bit.
    let dot_diff = |a: &Matrix, a0: &Matrix, b: &Matrix, b0: &Matrix| -> f64 {
        a.as_slice()
            .iter()
            .zip(a0.as_slice())
            .zip(b.as_slice().iter().zip(b0.as_slice()))
            .map(|((x, x0), (y, y0))| (x - x0) * (y - y0))
            .sum()
    };
    // `d[i] = <D_i − D_n | F_n>`, with the same fusion.
    let d: Vec<f64> = densities
        .par_iter()
        .map(|di| {
            di.as_slice()
                .iter()
                .zip(dn.as_slice())
                .zip(fnl.as_slice())
                .map(|((x, x0), y)| (x - x0) * y)
                .sum()
        })
        .collect();
    // s[i][j] = ⟨D_i−D_n | F_j−F_n⟩; parallelize over rows i (each row is independent).
    let s: Vec<Vec<f64>> = densities
        .par_iter()
        .map(|di| focks.iter().map(|fj| dot_diff(di, dn, fj, fnl)).collect())
        .collect();
    let c = solve_adiis_simplex(&d, &s);
    Some(combine(focks, &c))
}

/// Projected-gradient minimization of the A-DIIS quadratic on the probability simplex.
fn solve_adiis_simplex(d: &[f64], s: &[Vec<f64>]) -> Vec<f64> {
    let n = d.len();
    // Lipschitz estimate for the step size from (S + Sᵀ).
    let mut l: f64 = 1.0e-12;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += (s[i][j] + s[j][i]).abs();
        }
        l = l.max(row);
    }
    let lr = 1.0 / l;
    // Start from the latest point (all weight on the newest Fock/density).
    let mut c = vec![0.0; n];
    c[n - 1] = 1.0;
    for _ in 0..400 {
        // grad_k = 2 d_k + Σ_j (s_kj + s_jk) c_j
        let mut g = vec![0.0; n];
        for k in 0..n {
            let mut acc = 2.0 * d[k];
            for j in 0..n {
                acc += (s[k][j] + s[j][k]) * c[j];
            }
            g[k] = acc;
        }
        let trial: Vec<f64> = (0..n).map(|i| c[i] - lr * g[i]).collect();
        let proj = simplex_project(&trial);
        let mut delta = 0.0;
        for i in 0..n {
            delta += (proj[i] - c[i]).abs();
        }
        c = proj;
        if delta < 1.0e-12 {
            break;
        }
    }
    c
}

/// Euclidean projection of `v` onto the probability simplex `{c ≥ 0, Σc = 1}`.
fn simplex_project(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut css = 0.0;
    let mut rho = 0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            rho = j + 1;
            theta = t;
        }
    }
    let _ = rho;
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(xyz: &str, charge: f64) -> Pm7Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        run_pm7(&mol, &params, &Pm7Options::default()).unwrap()
    }

    fn run_mult(xyz: &str, charge: f64, mult: usize) -> Pm7Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options {
            charge,
            multiplicity: mult,
            ..Pm7Options::default()
        };
        run_pm7(&mol, &params, &opts).unwrap()
    }

    #[test]
    #[ignore = "diagnostic: locate the sp vs d-path h_core difference for water"]
    fn debug_water_hcore_diff() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let basis = Basis::build(&mol, &params).unwrap();
        let sp = crate::hamiltonian::build_core_with(&mol, &basis, &params, false).unwrap();
        let dp = crate::hamiltonian::build_core_with(&mol, &basis, &params, true).unwrap();
        let n = sp.h_core.rows;
        for i in 0..n {
            for j in 0..n {
                let d = (sp.h_core[(i, j)] - dp.h_core[(i, j)]).abs();
                if d > 1.0e-6 {
                    eprintln!(
                        "h_core[{i}][{j}]: sp={:+.5} d={:+.5} diff={:.2e}",
                        sp.h_core[(i, j)],
                        dp.h_core[(i, j)],
                        d
                    );
                }
            }
        }
    }

    #[test]
    fn dpath_reproduces_sp_energy_for_formaldehyde() {
        // Formaldehyde has a heavy–heavy C=O pair (both carry p orbitals, so both
        // dipole/quadrupole charge separations are nonzero) — this exercises the full
        // reppd heavy–heavy branch, unlike water (H has no p).
        let mol = Molecule::from_xyz_str(
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.03 0.0 1.25\nH 0.95 0.02 -0.55\nH -0.94 -0.03 -0.52\n",
            0.0,
        )
        .unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let sp = run_pm7(&mol, &params, &Pm7Options::default()).unwrap();
        let dp = run_pm7(
            &mol,
            &params,
            &Pm7Options {
                force_dpath: true,
                ..Pm7Options::default()
            },
        )
        .unwrap();
        eprintln!(
            "H2CO sp={:.6} d-path={:.6} delta={:.3e}",
            sp.total_ev,
            dp.total_ev,
            (sp.total_ev - dp.total_ev).abs()
        );
        assert!(
            (sp.total_ev - dp.total_ev).abs() < 1.0e-4,
            "d-path {} != sp-path {}",
            dp.total_ev,
            sp.total_ev
        );
    }

    #[test]
    fn dpath_reproduces_sp_energy_for_water() {
        // Routing a pure-sp molecule through the MNDO/d two-center + overlap kernel
        // must give the same total energy as the validated sp path — an end-to-end
        // oracle for the d-path rotation (independent of any d-specific integrals).
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let sp = run_pm7(&mol, &params, &Pm7Options::default()).unwrap();
        let dp = run_pm7(
            &mol,
            &params,
            &Pm7Options {
                force_dpath: true,
                ..Pm7Options::default()
            },
        )
        .unwrap();
        eprintln!(
            "water sp total={:.6} eV, d-path total={:.6} eV, delta={:.2e}",
            sp.total_ev,
            dp.total_ev,
            (sp.total_ev - dp.total_ev).abs()
        );
        assert!(
            (sp.total_ev - dp.total_ev).abs() < 1.0e-4,
            "d-path energy {} != sp-path energy {}",
            dp.total_ev,
            sp.total_ev
        );
    }

    #[test]
    #[ignore = "diagnostic: PH3 eigenvalues vs MOPAC"]
    fn debug_ph3_eigenvalues() {
        let xyz = "4\nph3\nP 0.0 0.0 0.0\nH 0.0 1.1932 0.7715\nH 1.0333 -0.5966 0.7715\nH -1.0333 -0.5966 0.7715\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let r = run_pm7(&mol, &params, &Pm7Options::default()).unwrap();
        eprintln!(
            "pm7-rs PH3 mo = {:?}",
            r.mo_energies
                .iter()
                .map(|v| (v * 1e3).round() / 1e3)
                .collect::<Vec<_>>()
        );
        eprintln!("MOPAC  PH3 mo = [-48.617,-16.070,-16.070,-11.870,0.962,0.962,4.228,99.603,99.603,99.669,100.644,100.644]");
        eprintln!(
            "pm7-rs charges = {:?}",
            r.charges
                .iter()
                .map(|v| (v * 1e4).round() / 1e4)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn h2s_d_orbital_matches_mopac() {
        // Hypervalent sulfur exercises the full MNDO/d path (9-AO S). MOPAC v23.2.5
        // PM7 1SCF at this fixed geometry: ΔHf = −3.05175 kcal/mol, S charge −0.330,
        // HOMO/d eigenvalues within a few meV.
        let xyz = "3\nH2S\nS 0.0 0.0 0.0\nH 0.0 0.9705 0.9430\nH 0.0 -0.9705 0.9430\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let r = run_pm7(&mol, &params, &Pm7Options::default()).unwrap();
        assert!(r.converged);
        assert!(
            (r.heat_of_formation_kcal - (-3.05175)).abs() < 0.1,
            "H2S dHf = {} kcal/mol (MOPAC -3.05175)",
            r.heat_of_formation_kcal
        );
        assert!(
            (r.charges[0] - (-0.330)).abs() < 5.0e-3,
            "S charge {}",
            r.charges[0]
        );
        // The five low-lying virtual d orbitals reproduce MOPAC.
        let mopac = [
            -26.944, -14.107, -11.796, -9.370, -0.834, -0.237, -0.147, 0.014, 0.038,
        ];
        for (i, &m) in mopac.iter().enumerate() {
            assert!(
                (r.mo_energies[i] - m).abs() < 0.05,
                "MO {i}: pm7-rs {} vs MOPAC {m}",
                r.mo_energies[i]
            );
        }
    }

    #[test]
    fn reference_forces_uhf_or_rhf() {
        // A closed-shell singlet defaults to RHF, but UHF can be forced explicitly;
        // the forced-UHF singlet reproduces the RHF energy (no symmetry breaking here).
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();

        let rhf = run_pm7(&mol, &params, &Pm7Options::default()).unwrap();
        assert!(!rhf.unrestricted);

        let uhf = run_pm7(
            &mol,
            &params,
            &Pm7Options {
                reference: ScfReference::Unrestricted,
                ..Pm7Options::default()
            },
        )
        .unwrap();
        assert!(uhf.unrestricted, "reference=uhf must select the UHF path");
        assert!((uhf.total_ev - rhf.total_ev).abs() < 1.0e-4);

        // Forcing RHF on an odd-electron (open-shell) system is rejected.
        let methyl =
            "4\nCH3\nC 0.0 0.0 0.0\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\nH -0.545 -0.944 0.0\n";
        let mol_r = Molecule::from_xyz_str(methyl, 0.0).unwrap();
        let err = run_pm7(
            &mol_r,
            &params,
            &Pm7Options {
                multiplicity: 2,
                reference: ScfReference::Restricted,
                ..Pm7Options::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Pm7Error::InvalidInput(_)));
    }

    #[test]
    fn reference_parses_from_str() {
        assert_eq!("auto".parse::<ScfReference>().unwrap(), ScfReference::Auto);
        assert_eq!(
            "rhf".parse::<ScfReference>().unwrap(),
            ScfReference::Restricted
        );
        assert_eq!(
            "R".parse::<ScfReference>().unwrap(),
            ScfReference::Restricted
        );
        assert_eq!(
            "uhf".parse::<ScfReference>().unwrap(),
            ScfReference::Unrestricted
        );
        assert_eq!(
            "U".parse::<ScfReference>().unwrap(),
            ScfReference::Unrestricted
        );
        assert!("bogus".parse::<ScfReference>().is_err());
    }

    #[test]
    fn water_heat_of_formation() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "H2O: dHf={:.3} kcal/mol  elec={:.4} eV core={:.4} eV  dipole={:.3} D  charges={:?}  iters={}",
            r.heat_of_formation_kcal, r.electronic_ev, r.core_ev, r.dipole_magnitude, r.charges, r.iterations
        );
        assert!(r.converged);
        // MOPAC v23.2.5 PM7 oracle at this fixed geometry: -57.78934 kcal/mol.
        assert!((r.heat_of_formation_kcal - (-57.78934)).abs() < 1.0e-4);
        let qsum: f64 = r.charges.iter().sum();
        assert!(qsum.abs() < 1e-6);
        assert!(r.charges[0] < 0.0 && r.charges[1] > 0.0);
    }

    #[test]
    fn water_no_diis_debug() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let opts = Pm7Options {
            use_diis: false,
            max_scf: 500,
            ..Pm7Options::default()
        };
        let r = run_pm7(&mol, &params, &opts).unwrap();
        eprintln!(
            "H2O(no-diis): dHf={:.3} conv={} iters={}",
            r.heat_of_formation_kcal, r.converged, r.iterations
        );
    }

    #[test]
    fn methane_heat_of_formation() {
        let xyz = "5\nmethane\nC 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "CH4: dHf={:.3} kcal/mol dipole={:.3} D iters={}",
            r.heat_of_formation_kcal, r.dipole_magnitude, r.iterations
        );
        assert!(r.converged);
        // MOPAC v23.2.5 PM7 oracle at this fixed geometry: -14.39980 kcal/mol.
        assert!((r.heat_of_formation_kcal - (-14.39980)).abs() < 1.0e-4);
    }

    #[test]
    fn accelerators_agree_on_energy() {
        // A-DIIS→CDIIS, plain CDIIS, and no acceleration must reach the same converged
        // energy (same SCF fixed point); the hybrid should not need more iterations.
        let xyz =
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm7Parameters::standard().unwrap();
        let run = |acc: ScfAccelerator| {
            let opts = Pm7Options {
                accelerator: acc,
                ..Pm7Options::default()
            };
            run_pm7(&mol, &params, &opts).unwrap()
        };
        let hybrid = run(ScfAccelerator::AdiisCdiis);
        let cdiis = run(ScfAccelerator::Cdiis);
        let none = run(ScfAccelerator::None);
        eprintln!(
            "iters: hybrid={} cdiis={} none={}  E(hybrid)={:.6}",
            hybrid.iterations, cdiis.iterations, none.iterations, hybrid.total_ev
        );
        assert!((hybrid.total_ev - cdiis.total_ev).abs() < 1e-6);
        assert!((hybrid.total_ev - none.total_ev).abs() < 1e-6);
        // Both accelerated paths should be far faster than plain iteration.
        assert!(hybrid.iterations < none.iterations);
    }

    #[test]
    fn methyl_radical_uhf() {
        // Planar CH3 radical (doublet): UHF must converge, be open-shell, and carry net spin.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let r = run_mult(xyz, 0.0, 2);
        eprintln!(
            "CH3.: dHf={:.3} kcal/mol unrestricted={} iters={}",
            r.heat_of_formation_kcal, r.unrestricted, r.iterations
        );
        assert!(r.converged);
        assert!(r.unrestricted);
        // Total spin population (∫ P_α − P_β) should be ≈ 1 unpaired electron.
        let spin = r.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!((n_spin - 1.0).abs() < 1e-6, "net spin {n_spin}");
        // MOPAC v23.2.5 PM7 oracle at this fixed geometry: +28.46325 kcal/mol.
        assert!((r.heat_of_formation_kcal - 28.46325).abs() < 1.0e-4);
    }

    #[test]
    fn invalid_physical_inputs_fail_before_integral_build() {
        let params = Pm7Parameters::standard().unwrap();
        let coincident =
            Molecule::from_xyz_str("2\ninvalid\nH 0.0 0.0 0.0\nH 0.0 0.0 0.0\n", 0.0).unwrap();
        assert!(matches!(
            run_pm7(&coincident, &params, &Pm7Options::default()),
            Err(Pm7Error::InvalidInput(_))
        ));

        let water =
            Molecule::from_xyz_str("3\nwater\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n", 0.0)
                .unwrap();
        let invalid_cutoff = Pm7Options {
            exchange_cutoff: Some((10.0, 5.0)),
            ..Pm7Options::default()
        };
        assert!(matches!(
            run_pm7(&water, &params, &invalid_cutoff),
            Err(Pm7Error::InvalidInput(_))
        ));
    }
}
