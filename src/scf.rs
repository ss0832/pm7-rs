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
use crate::params::Pm7Parameters;
use crate::repulsion::core_core_energy;
use crate::system::Molecule;
use crate::method::Pm7Method;

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
    /// Spin treatment (RHF/UHF), independent of `multiplicity`. Default [`ScfReference::Auto`].
    pub reference: ScfReference,
    /// Diagnostic: route every pair through the MNDO/d two-center + overlap kernel even
    /// for a pure-sp molecule (lets the validated sp path act as an oracle for the d path).
    pub force_dpath: bool,
    /// Optional peak-memory budget in MiB for the pre-flight OOM guard. `None` falls back to
    /// the `PM7_MEM_BUDGET_MB` env var, then to 80 % of currently available physical RAM (see
    /// [`crate::memory`]).
    pub max_memory_mb: Option<usize>,
    /// Optional smooth long-range-**exchange** cutoff `(inner, outer)` in **Bohr**, applied only
    /// to the analytic Hessian's CPHF response-Fock builds (its dominant cost). The two-center
    /// exchange between atoms farther than `outer` apart is dropped, with a C²-smooth switch
    /// between `inner` and `outer`. `None` (default) keeps the Hessian **bit-identical** /
    /// cutoff-free; Coulomb is never cut. Trades a controlled approximation for speed on large
    /// systems.
    pub exchange_cutoff: Option<(f64, f64)>,
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
            reference: ScfReference::Auto,
            force_dpath: false,
            max_memory_mb: None,
            exchange_cutoff: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pm7Result {
    pub density: Matrix,
    /// Spin density `P_α − P_β` (open-shell UHF only; `None` for RHF).
    pub spin_density: Option<Matrix>,
    pub mo_energies: Vec<f64>,
    pub mo_coeff: Matrix,
    pub n_occ: usize,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub dipole_debye: Vec3,
    pub dipole_magnitude: f64,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// True when the UHF (open-shell) path was used.
    pub unrestricted: bool,
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
    let core = crate::hamiltonian::build_core_with(molecule, &basis, params, options.force_dpath)?;

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

    let core_ev = core_core_energy(molecule, params)?;

    // Post-SCF corrections (dispersion + H-bond), reported in kcal/mol and added to
    // both the total energy and the heat of formation. Disabled for PM7-minus.
    let correction_kcal = if options.method.has_post_scf_corrections() {
        let mut c = crate::dispersion::dispersion_energy(molecule);
        c += crate::hbond::hydrogen_bond_energy(molecule);
        // PM7-HH adds an extra hydrogen–hydrogen repulsion term.
        if options.method.has_hh_repulsion() {
            c += crate::hh_rep::hh_repulsion_energy(molecule);
        }
        c
    } else {
        0.0
    };
    let total_ev = state.electronic_ev + core_ev + correction_kcal * crate::constants::KCAL_TO_EV;

    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let heat_of_formation_kcal =
        (state.electronic_ev + core_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL + correction_kcal;

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

    // Dipole: point-charge term + s–p hybrid polarization (both in e·Bohr).
    let mut dip = Vec3::zero();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        dip += atom.position * charges[ia];
        let elem = params.element(atom.z)?;
        if elem.has_p() {
            let off = basis.atom_offset[ia];
            let hyb = -2.0 * elem.dd;
            dip += Vec3::new(
                hyb * state.density[(off, off + 1)],
                hyb * state.density[(off, off + 2)],
                hyb * state.density[(off, off + 3)],
            );
        }
    }
    let dipole_debye = dip * AU_DIPOLE_TO_DEBYE;
    let dipole_magnitude = dipole_debye.norm();

    let nao = basis.nao;
    let homo_ev = (state.n_occ >= 1).then(|| state.mo_energies[state.n_occ - 1]);
    let lumo_ev = (state.n_occ < nao).then(|| state.mo_energies[state.n_occ]);

    Ok(Pm7Result {
        density: state.density,
        spin_density: state.spin_density,
        mo_energies: state.mo_energies,
        mo_coeff: state.mo_coeff,
        n_occ: state.n_occ,
        electronic_ev: state.electronic_ev,
        core_ev,
        total_ev,
        heat_of_formation_kcal,
        charges,
        dipole_debye,
        dipole_magnitude,
        homo_ev,
        lumo_ev,
        iterations: state.iterations,
        converged: state.converged,
        unrestricted: state.unrestricted,
    })
}

fn validate_input(molecule: &Molecule, options: &Pm7Options) -> Result<()> {
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
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let p = atom.position;
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Err(Pm7Error::InvalidInput(format!(
                "atom {} has non-finite coordinates",
                index + 1
            )));
        }
    }
    for i in 0..molecule.atoms.len() {
        for j in (i + 1)..molecule.atoms.len() {
            if (molecule.atoms[i].position - molecule.atoms[j].position).norm2() < 1.0e-20 {
                return Err(Pm7Error::InvalidInput(format!(
                    "atoms {} and {} occupy the same position",
                    i + 1,
                    j + 1
                )));
            }
        }
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
fn sad_density(
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
fn commutator(f: &Matrix, p: &Matrix) -> Matrix {
    let fp = f.matmul(p);
    let n = fp.rows;
    let mut e = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            e[(i, j)] = fp[(i, j)] - fp[(j, i)];
        }
    }
    e
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
    let mut density = sad_density(molecule, basis, params, 2.0 * n_occ as f64)?; // atomic-density guess
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

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let f = build_fock(molecule, basis, params, core, &density)?;
        let e_elec = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f));
        let err = commutator(&f, &density);
        let err_norm = err.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        // History (Fock, commutator, density) for CDIIS / A-DIIS.
        diis_f.push(f.clone());
        diis_e.push(err);
        diis_d.push(density.clone());
        if diis_f.len() > max_diis {
            diis_f.remove(0);
            diis_e.remove(0);
            diis_d.remove(0);
        }

        let f_use = match accel {
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
        let (eps, c) = symmetric_eigen(&f_use)?;
        let p_new = density_from_coeff(&c, n_occ, 2.0);
        let dp = rms_diff(&p_new, &density);
        density_error = dp;
        let de = (e_elec - e_old).abs();

        mo_energies = eps;
        mo_coeff = c;
        density = p_new;
        e_old = e_elec;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
        }
    }
    let f_final = build_fock(molecule, basis, params, core, &density)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final));

    Ok(ScfState {
        density,
        spin_density: None,
        mo_energies,
        mo_coeff,
        n_occ,
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
    // MOPAC guess split by spin population; the different α/β aufbau counts break spin symmetry.
    let sad = sad_density(molecule, basis, params, (n_alpha + n_beta) as f64)?;
    let n_tot = (n_alpha + n_beta).max(1) as f64;
    let (fa, fb) = (n_alpha as f64 / n_tot, n_beta as f64 / n_tot);
    let mut pa = sad.clone();
    for v in pa.as_mut_slice() {
        *v *= fa;
    }
    let mut pb = sad;
    for v in pb.as_mut_slice() {
        *v *= fb;
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
        let ea = commutator(&fa, &pa);
        let eb = commutator(&fb, &pb);
        let mut err = Matrix::zeros(2 * nao, nao);
        for i in 0..nao {
            for j in 0..nao {
                err[(i, j)] = ea[(i, j)];
                err[(nao + i, j)] = eb[(i, j)];
            }
        }
        let err_norm = err.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        // A plain CDIIS extrapolation can converge UHF radicals to a higher-energy
        // spin-broken fixed point (PM7 CH3 is a reproducible example).  Keep the
        // UHF path variationally stable until the shared energy-monotone A-DIIS
        // controller is extended to coupled alpha/beta histories; RHF retains its
        // existing A-DIIS/CDIIS accelerator.
        const UHF_CDIIS_ENABLED: bool = false;
        let (fa_use, fb_use) = if UHF_CDIIS_ENABLED && options.use_diis {
            hist_fa.push(fa.clone());
            hist_fb.push(fb.clone());
            hist_err.push(err);
            if hist_fa.len() > max_diis {
                hist_fa.remove(0);
                hist_fb.remove(0);
                hist_err.remove(0);
            }
            match diis_coeffs(&hist_err) {
                Some(coeffs) => (combine(&hist_fa, &coeffs), combine(&hist_fb, &coeffs)),
                None => (fa, fb),
            }
        } else {
            (fa, fb)
        };

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
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
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

    Ok(ScfState {
        density,
        spin_density: Some(spin),
        mo_energies: eps_a,
        mo_coeff: c_a,
        n_occ: n_alpha,
        electronic_ev,
        converged,
        density_error,
        iterations,
        unrestricted: true,
    })
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

/// Solve the Pulay DIIS coefficient system from a stack of error matrices.
fn diis_coeffs(es: &[Matrix]) -> Option<Vec<f64>> {
    let n = es.len();
    if n < 2 {
        return None;
    }
    use rayon::prelude::*;
    let dim = n + 1;
    let mut b = Matrix::zeros(dim, dim);
    // The error-overlap matrix is symmetric (`⟨e_i|e_j⟩ = ⟨e_j|e_i⟩`), so evaluate only the
    // upper triangle — one frobenius dot per (i,j) — in parallel, then mirror.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let vals: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| ((i, j), es[i].frobenius_dot(&es[j])))
        .collect();
    for ((i, j), v) in vals {
        b[(i, j)] = v;
        b[(j, i)] = v;
    }
    for i in 0..n {
        b[(i, n)] = -1.0;
        b[(n, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    // The DIIS matrix (a small bordered saddle-point system) becomes singular near
    // convergence when the error vectors turn linearly dependent. A pivot-guarded
    // Gaussian elimination returns `None` there so the caller falls back to the plain
    // Fock — faer's LU instead returns a degenerate solution that derails DIIS. The heavy
    // O(n^3) eigendecomposition still uses faer; only this tiny solve is bespoke.
    solve_bordered_small(&b, &rhs)
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
    let coeffs = diis_coeffs(es)?;
    Some(combine(fs, &coeffs))
}

fn mat_sub(a: &Matrix, b: &Matrix) -> Matrix {
    let mut o = a.clone();
    for (ov, bv) in o.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *ov -= *bv;
    }
    o
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
    let dd: Vec<Matrix> = densities.par_iter().map(|d| mat_sub(d, dn)).collect();
    let ff: Vec<Matrix> = focks.par_iter().map(|f| mat_sub(f, fnl)).collect();
    let d: Vec<f64> = dd.par_iter().map(|ddi| ddi.frobenius_dot(fnl)).collect();
    // s[i][j] = ⟨D_i−D_n | F_j−F_n⟩; parallelize over rows i (each row is independent).
    let s: Vec<Vec<f64>> = dd
        .par_iter()
        .map(|ddi| ff.iter().map(|ffj| ddi.frobenius_dot(ffj)).collect())
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
