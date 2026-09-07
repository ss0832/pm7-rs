// SPDX-License-Identifier: GPL-3.0-or-later

//! Divide-and-conquer SCF (Dixon & Merz).
//!
//! # The idea
//!
//! The cost of an ordinary SCF is dominated by diagonalizing an `nao × nao` Fock matrix, which is
//! `O(N³)`. Divide and conquer replaces that with one small diagonalization per subsystem: the
//! system is cut into cores of bounded size, each core is given a buffer shell so its orbitals see
//! a complete environment, and the global density is reassembled from the subsystem densities with
//! partition weights.
//!
//! Because a subsystem's size is fixed by the buffer radius and not by the system, doubling the
//! system doubles the number of subsystems and leaves each one's cost alone — the diagonalization
//! becomes `O(N)`. Everything else has to keep up: the density is stored as
//! [`SparseDensity`] atom-pair blocks rather than a dense matrix, and the far field comes from the
//! monopole machinery (Ewald for a crystal, a direct charge sum for a molecule) instead of a full
//! pair enumeration.
//!
//! # What holds the subsystems together
//!
//! Two things, and they are what make this a method rather than a set of independent calculations:
//!
//! * **One global Fermi level.** Each subsystem contributes occupied states according to the same
//!   `E_F`, found by bisection on the total electron count. Filling each subsystem to its own
//!   aufbau count instead would let charge pool in whichever fragment happened to have the lowest
//!   levels.
//! * **The environment potential.** Every subsystem sees `−V_A^ext`, the monopole field of the
//!   atoms it does *not* contain, built from the current global charges. That is what carries
//!   polarization across subsystem boundaries.
//!
//! # What it costs
//!
//! The energy is variational in each subsystem but the assembled density is not the exact SCF
//! density, so the total energy carries a non-variational residual that shrinks as the buffer
//! grows. [`DandcOptions::buffer`] is the knob, and `tests/dandc.rs` shows the convergence rather
//! than asserting a bound out of thin air.
//!
//! Second derivatives are **not** supported: a Hessian needs the coupled-perturbed response of the
//! whole system, which does not decompose the way the density does.

pub mod partition;
pub mod sparse;

use crate::basis::Basis;
use crate::error::{Pm7Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::scf::{Pm7Options, ScfReference};
use crate::system::{Atom, Molecule};
use partition::{partition, Partitioning, Subsystem};
pub use sparse::{BlockView, SparseDensity};

/// Settings for the divide-and-conquer driver.
#[derive(Clone, Copy, Debug)]
pub struct DandcOptions {
    /// Buffer radius in **Bohr**. The accuracy knob: everything outside a core's buffer reaches it
    /// only as a monopole. The default is a little beyond the 7 Å range at which PM7's feathering
    /// makes every two-centre integral *exactly* a monopole, so the truncation is in the density
    /// rather than in the integrals.
    pub buffer: f64,
    /// Target atoms per core. Sets the subsystem size, and with it the constant in the `O(N)`.
    pub core_size: usize,
    /// Fermi smearing width in eV for the common chemical potential. A little smearing keeps the
    /// occupation a continuous function of the subsystem eigenvalues, which is what lets the
    /// bisection converge for a system whose fragments have levels crossing `E_F`.
    pub fermi_width_ev: f64,
    /// Maximum SCF iterations.
    pub max_scf: usize,
    /// Density convergence threshold (RMS over the stored blocks).
    pub p_tol: f64,
    /// Linear mixing factor for the density.
    pub mixing: f64,
}

impl Default for DandcOptions {
    fn default() -> Self {
        Self {
            buffer: 15.0,
            core_size: 12,
            fermi_width_ev: 0.1,
            max_scf: 200,
            p_tol: 1.0e-6,
            mixing: 0.4,
        }
    }
}

impl DandcOptions {
    pub fn validate(&self) -> Result<()> {
        if !self.buffer.is_finite() || self.buffer <= 0.0 {
            return Err(Pm7Error::InvalidInput(
                "the divide-and-conquer buffer must be finite and positive".into(),
            ));
        }
        if self.core_size == 0 {
            return Err(Pm7Error::InvalidInput(
                "the divide-and-conquer core size must be at least 1".into(),
            ));
        }
        if !self.fermi_width_ev.is_finite() || self.fermi_width_ev <= 0.0 {
            return Err(Pm7Error::InvalidInput(
                "the divide-and-conquer Fermi width must be finite and positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.mixing) || self.mixing <= 0.0 {
            return Err(Pm7Error::InvalidInput(
                "the divide-and-conquer mixing factor must lie in (0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// What a divide-and-conquer SCF produces.
#[derive(Clone, Debug)]
pub struct DandcResult {
    /// Global density, as atom-pair blocks.
    pub density: SparseDensity,
    /// Spin density `P_α − P_β`, for an unrestricted run.
    pub spin_density: Option<SparseDensity>,
    /// The common chemical potential, in eV.
    pub fermi_ev: f64,
    /// Electronic energy in eV.
    pub electronic_ev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub density_error: f64,
    /// Number of subsystems, and the size of the largest — the two numbers that decide whether
    /// this is actually linear.
    pub subsystems: usize,
    pub largest_subsystem: usize,
    pub unrestricted: bool,
}

/// Run a divide-and-conquer SCF.
pub fn run_dandc(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    dandc: &DandcOptions,
) -> Result<DandcResult> {
    // The same input validation `run_pm7` does. Without it a non-finite coordinate reaches
    // `partition`, whose median bisection sorts by position — so the failure was a panic inside a
    // comparator rather than a `Pm7Error` naming the atom.
    crate::scf::validate_input(molecule, options)?;
    // An external field is **refused**, not ignored. `solve_subsystem` builds each subsystem's
    // core Hamiltonian with `build_core`, which knows nothing about a field, so accepting one
    // here would return a perfectly plausible field-free answer to someone who asked for a field.
    // Threading the field through the subsystem cores is not merely mechanical either: the
    // environment potential is built from monopoles, and a linear `−f·R` term across subsystem
    // boundaries needs its own treatment.
    if options.active_field().is_some() {
        return Err(Pm7Error::InvalidInput(
            "an external electric field is not supported by the divide-and-conquer SCF: each \
             subsystem builds its own core Hamiltonian, which the field does not reach. Use the \
             exact SCF (`run_pm7`) for a field."
                .into(),
        ));
    }
    dandc.validate()?;
    let basis = Basis::build(molecule, params)?;
    let n_atoms = molecule.atoms.len();
    let norb: Vec<usize> = basis.atom_norb.clone();
    let parts = partition(molecule, dandc.core_size, dandc.buffer);

    // **Pre-flight OOM guard, on the largest subsystem.**
    //
    // `run_pm7` guards on the whole molecule's basis, and that guard is useless here: it is the
    // subsystems that carry the dense work, and for a periodic cell a subsystem is *larger* than
    // the cell, because the buffer pulls in lattice images. A two-atom diamond cell has 8 AOs and
    // sails through every check in `run_pm7`; ask for a 15 Å buffer on its 2.5 Å lattice and each
    // subsystem is some thousands of atoms, whose dense blocks ran the process out of memory and
    // aborted it — `memory allocation of 4177920 bytes failed`, with no `Pm7Error` and nothing
    // naming the flag that caused it. Found by `tests/cli_matrix.rs`, which asks for exactly that
    // combination and whose invariant is that a refusal is a message and not a crash.
    //
    // The guard goes here rather than inside `solve_subsystem` because the `neighbours` loop just
    // below is already `O(subsystem²)` per subsystem and allocates from it.
    let largest = parts
        .subsystems
        .iter()
        .map(|s| s.parent.iter().map(|&a| norb[a]).sum::<usize>())
        .max()
        .unwrap_or(0);
    let largest_atoms = parts.subsystems.iter().map(|s| s.len()).max().unwrap_or(0);
    let has_d = options.force_dpath
        || molecule
            .atoms
            .iter()
            .any(|a| params.element(a.z).map(|e| e.has_d()).unwrap_or(false));
    crate::memory::guard(
        largest,
        largest_atoms * largest_atoms.saturating_sub(1) / 2,
        largest_atoms,
        has_d,
        false,
        options.max_memory_mb,
    )
    .map_err(|e| {
        // The bare `InsufficientMemory` names a basis size the caller never chose. Say which knob
        // produced it and which way to turn it.
        Pm7Error::InvalidInput(format!(
            "{e}\nThis is a divide-and-conquer subsystem, not the whole system: a {:.2} Å buffer \
             around a {}-atom core gives a largest subsystem of {largest_atoms} atoms and \
             {largest} orbitals. Reduce the buffer, or drop divide and conquer and use the exact \
             SCF — below roughly 350 atoms the exact SCF is both faster and exact.",
            dandc.buffer * crate::constants::BOHR_TO_ANGSTROM,
            dandc.core_size,
        ))
    })?;

    // The sparsity pattern is the union of the subsystems' own atom pairs — exactly the pairs
    // divide and conquer can say anything about, and nothing more.
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for s in &parts.subsystems {
        for i in 0..s.len() {
            for j in 0..s.len() {
                if s.weight(i, j) > 0.0 {
                    neighbours[s.parent[i]].push(s.parent[j]);
                }
            }
        }
    }
    let mut density = SparseDensity::with_pattern(&norb, &neighbours);

    // Dixon–Merz weights are 1 for a core–core pair and ½ for a core–buffer one, and the textbook
    // claim is that they sum to 1 over the subsystems. They do — for a pair whose two ends see
    // *each other*. At the edge of a buffer they need not: an atom can lie inside another
    // subsystem's buffer without that subsystem's cores lying inside its own, and such a pair
    // collects a single ½. Left alone that halves the density on exactly the pairs the method is
    // least sure about. Normalizing by the realized total makes the weights a partition of unity
    // by construction, and turns those pairs into an average of the estimates that saw them.
    let mut weight_total = vec![0.0_f64; density.stored_pairs()];
    for s in &parts.subsystems {
        for i in 0..s.len() {
            for j in 0..s.len() {
                let w = s.weight(i, j);
                if w > 0.0 {
                    if let Some(slot) = density.pair_slot(s.parent[i], s.parent[j]) {
                        weight_total[slot] += w;
                    }
                }
            }
        }
    }
    let normalizer: Vec<f64> = weight_total
        .iter()
        .map(|w| if *w > 1.0e-12 { 1.0 / w } else { 0.0 })
        .collect();

    // Electron count, and whether the run is unrestricted.
    let core_electrons: f64 = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum();
    let n_elec = core_electrons - options.charge;
    if n_elec < -1.0e-9 {
        return Err(Pm7Error::InvalidInput(
            "the divide-and-conquer driver was given a system with negative electron count".into(),
        ));
    }
    let unrestricted = match options.reference {
        ScfReference::Unrestricted => true,
        ScfReference::Restricted => false,
        ScfReference::Auto => options.multiplicity > 1,
    };
    let n_beta_extra = (options.multiplicity as f64 - 1.0) * 0.5;

    // Atomic-density starting guess, scattered into the sparse diagonal blocks.
    let guess = crate::scf::sad_density(molecule, &basis, params, n_elec)?;
    for a in 0..n_atoms {
        let na = norb[a];
        let off = basis.atom_offset[a];
        let mut block = vec![0.0; na * na];
        for i in 0..na {
            for j in 0..na {
                block[i * na + j] = guess[(off + i, off + j)];
            }
        }
        density.accumulate(a, a, 1.0, &block);
    }
    let mut spin_density = unrestricted.then(|| {
        let mut s = SparseDensity::with_pattern(&norb, &neighbours);
        // Start the spin density from a scaled copy of the total: the driver only needs a
        // symmetry-broken seed, and the SCF finds the rest.
        for a in 0..n_atoms {
            let na = norb[a];
            let mut block = vec![0.0; na * na];
            let view = density.block(a, a);
            let scale = if core_electrons > 0.0 {
                2.0 * n_beta_extra / core_electrons
            } else {
                0.0
            };
            for i in 0..na {
                block[i * na + i] = scale * view.get(i, i);
            }
            s.accumulate(a, a, 1.0, &block);
        }
        s
    });

    let mut fermi_ev = 0.0;
    let mut converged = false;
    let mut density_error = f64::INFINITY;
    let mut iterations = 0;
    let mut electronic_ev = 0.0;

    // Geometry-only, so it is built once rather than once per iteration.
    let far = far_field(molecule, params, options)?;

    for iteration in 0..dandc.max_scf {
        iterations = iteration + 1;
        let field = environment_field(molecule, &far, &density, &parts)?;

        // Solve every subsystem. Independent by construction, so this is the parallel step.
        let solved: Result<Vec<SubsystemSolution>> = {
            use rayon::prelude::*;
            parts
                .subsystems
                .par_iter()
                .zip(&field)
                .map(|(s, v_ext)| {
                    solve_subsystem(
                        molecule,
                        params,
                        options,
                        s,
                        v_ext,
                        &density,
                        spin_density.as_ref(),
                    )
                })
                .collect()
        };
        let solved = solved?;

        // One chemical potential for the whole system.
        fermi_ev = find_fermi(&solved, &parts, n_elec, dandc.fermi_width_ev)?;

        // Reassemble the global density from the subsystem solutions.
        let mut next = SparseDensity::with_pattern(&norb, &neighbours);
        let mut next_spin = unrestricted.then(|| SparseDensity::with_pattern(&norb, &neighbours));
        // Parallel over subsystems, with a **fixed partition and an ordered combination**: each
        // chunk accumulates into its own `SparseDensity` and the partials are folded back in
        // chunk order, so the summation order — and therefore the last bit of every element — is
        // the same whatever `RAYON_NUM_THREADS` says. `tests/determinism.rs` pins that.
        //
        // Chunked rather than one accumulator per subsystem: a `SparseDensity` is an `O(N)`
        // allocation and there are as many subsystems as atoms, so per-subsystem partials would
        // trade the quadratic memory this method exists to avoid for a different one.
        {
            use rayon::prelude::*;
            let pairs: Vec<(&Subsystem, &SubsystemSolution)> =
                parts.subsystems.iter().zip(&solved).collect();
            let threads = rayon::current_num_threads().max(1);
            let chunk = pairs.len().div_ceil(threads).max(1);
            let blank = (
                SparseDensity::with_pattern(&norb, &neighbours),
                unrestricted.then(|| SparseDensity::with_pattern(&norb, &neighbours)),
            );
            let partials: Vec<(SparseDensity, Option<SparseDensity>)> = pairs
                .par_chunks(chunk)
                .map(|group| {
                    let mut local = blank.clone();
                    for (s, solution) in group {
                        scatter_density(
                            s,
                            solution,
                            fermi_ev,
                            dandc.fermi_width_ev,
                            &mut local.0,
                            false,
                        );
                        if let Some(spin) = local.1.as_mut() {
                            scatter_density(
                                s,
                                solution,
                                fermi_ev,
                                dandc.fermi_width_ev,
                                spin,
                                true,
                            );
                        }
                    }
                    local
                })
                .collect();
            for (density, spin) in &partials {
                next.add_assign(density);
                if let (Some(target), Some(source)) = (next_spin.as_mut(), spin.as_ref()) {
                    target.add_assign(source);
                }
            }
        }
        next.scale_pairs(&normalizer);
        if let Some(spin) = next_spin.as_mut() {
            spin.scale_pairs(&normalizer);
        }

        density_error = density.rms_diff(&next);
        density.mix(&next, dandc.mixing);
        if let (Some(current), Some(new)) = (spin_density.as_mut(), next_spin.as_ref()) {
            current.mix(new, dandc.mixing);
        }
        electronic_ev = solved.iter().map(|s| s.energy_contribution).sum();
        if density_error < dandc.p_tol {
            converged = true;
            break;
        }
    }

    Ok(DandcResult {
        density,
        spin_density,
        fermi_ev,
        electronic_ev,
        converged,
        iterations,
        density_error,
        subsystems: parts.subsystems.len(),
        largest_subsystem: parts.largest(),
        unrestricted,
    })
}

/// Total energy, gradient and stress at a divide-and-conquer density.
///
/// The derivatives are the ordinary fixed-density (Hellmann–Feynman) expressions evaluated at the
/// divide-and-conquer density rather than the exact one. That is legitimate and it is also the
/// only honest option: the NDDO energy is stationary with respect to the *exact* density, so a
/// gradient taken at an approximate one carries a **non-variational residual** — a term
/// proportional to how far the density is from self-consistent. It shrinks with the buffer at the
/// same rate the energy error does, and `tests/dandc.rs` shows both together rather than leaving
/// the reader to assume.
pub struct DandcDerivatives {
    pub energy_ev: f64,
    pub gradient: Vec<Vec3>,
    /// `None` for a molecule.
    pub stress: Option<crate::math::Mat3>,
}

/// Energy, gradient and (for a periodic cell) stress from a converged divide-and-conquer density.
pub fn dandc_derivatives(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    result: &DandcResult,
) -> Result<DandcDerivatives> {
    let basis = Basis::build(molecule, params)?;
    // The pair loop reads the sparse density directly. It used to `to_dense()` it first, which
    // allocated `N_ao²` — around 512 MB at two thousand atoms — at the one step *after* the
    // linear-scaling SCF had finished, and put a quadratic memory bound on a method whose whole
    // purpose is to avoid one. Every read the loop makes is block-local, so nothing about it
    // needed the dense form.
    //
    // Periodic divide and conquer would need a translation-resolved sparse density; there is no
    // such thing yet, so a periodic cell keeps the dense path and is bounded as before. Molecular
    // is what divide and conquer is actually used for.
    let (gradient, virial) = match options.pbc_for(molecule) {
        None => (
            crate::gradient::electronic_gradient_sparse(molecule, params, &basis, &result.density)?
                .into_iter()
                .zip(crate::repulsion::core_core_gradient(molecule, params)?)
                .map(|(e, c)| e + c)
                .collect::<Vec<Vec3>>(),
            crate::math::Mat3::zero(),
        ),
        Some(_) => {
            let dense = result.density.to_dense(&basis);
            let density = crate::gradient::TranslatedDensity::Uniform(&dense);
            crate::gradient::fixed_density_gradient_blocks(molecule, params, density, options)?
        }
    };
    let correction = crate::gradient::correction_gradient_and_virial(molecule, options);
    let gradient: Vec<Vec3> = gradient
        .iter()
        .zip(&correction.0)
        .map(|(g, c)| *g + *c)
        .collect();

    let core_ev = match options.pbc_for(molecule) {
        None => crate::repulsion::core_core_energy(molecule, params)?,
        Some(p) => crate::repulsion::core_core_energy_periodic(molecule, params, &p)?,
    };
    let correction_ev =
        crate::scf::correction_energy(molecule, options) * crate::constants::KCAL_TO_EV;
    let energy_ev = result.electronic_ev + core_ev + correction_ev;

    let stress = molecule.cell.map(|cell| {
        virial
            .plus(&correction.1)
            .symmetrized()
            .scaled(1.0 / cell.measure())
    });
    Ok(DandcDerivatives {
        energy_ev,
        gradient,
        stress,
    })
}

/// One subsystem's eigenvalues, orbitals, and the energy it contributes.
struct SubsystemSolution {
    eigenvalues: Vec<f64>,
    coefficients: Matrix,
    /// α channel, when unrestricted. `None` means both spins share the restricted solution.
    beta: Option<(Vec<f64>, Matrix)>,
    /// Orbital offset of each subsystem atom in the subsystem basis.
    offset: Vec<usize>,
    norb: Vec<usize>,
    /// `½ Tr D∘p (H + F)`, accumulated with the partition weights.
    energy_contribution: f64,
}

/// Build and diagonalize one subsystem.
fn solve_subsystem(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    s: &Subsystem,
    v_ext: &EnvironmentField,
    density: &SparseDensity,
    spin: Option<&SparseDensity>,
) -> Result<SubsystemSolution> {
    // The subsystem as a finite cluster. For a periodic parent the buffer atoms carry their image
    // positions, so this is geometrically faithful without ever needing a cell.
    let atoms: Vec<Atom> = s
        .parent
        .iter()
        .zip(&s.positions)
        .map(|(&index, position)| Atom {
            z: molecule.atoms[index].z,
            position: *position,
        })
        .collect();
    let cluster = Molecule::new(atoms);
    let basis = Basis::build(&cluster, params)?;
    let mut core = crate::hamiltonian::build_core(&cluster, &basis, params)?;

    // The nuclear half of the environment goes into the core Hamiltonian, where `½ Tr P(H + F)`
    // counts it once in full; the electronic half is added to the Fock below, where the same
    // formula gives it the ½ a mean-field term needs. See [`EnvironmentField`].
    for (i, v) in v_ext.nuclear.iter().enumerate() {
        let off = basis.atom_offset[i];
        for mu in 0..basis.atom_norb[i] {
            core.h_core[(off + mu, off + mu)] -= v;
        }
    }

    // The electronic half of the environment: a Coulomb repulsion from the electrons outside, so
    // it goes on the Fock diagonal with a plus sign and stays out of `h_core`.
    let add_environment = |f: &mut Matrix| {
        for (i, v) in v_ext.electronic.iter().enumerate() {
            let off = basis.atom_offset[i];
            for mu in 0..basis.atom_norb[i] {
                f[(off + mu, off + mu)] += v;
            }
        }
    };

    // The current global density, restricted to this subsystem.
    let p_local = local_density(s, &basis, density);
    let mut fock = match spin {
        None => crate::fock::build_fock(&cluster, &basis, params, &core, &p_local)?,
        Some(spin) => {
            let s_local = local_density(s, &basis, spin);
            let mut pa = p_local.clone();
            for (v, sv) in pa.as_mut_slice().iter_mut().zip(s_local.as_slice()) {
                *v = 0.5 * (*v + *sv);
            }
            crate::fock::build_fock_spin(&cluster, &basis, params, &core, &p_local, &pa)?
        }
    };
    add_environment(&mut fock);
    let (eigenvalues, coefficients) = symmetric_eigen(&fock)?;
    let beta = match spin {
        None => None,
        Some(spin) => {
            let s_local = local_density(s, &basis, spin);
            let mut pb = p_local.clone();
            for (v, sv) in pb.as_mut_slice().iter_mut().zip(s_local.as_slice()) {
                *v = 0.5 * (*v - *sv);
            }
            let mut fb =
                crate::fock::build_fock_spin(&cluster, &basis, params, &core, &p_local, &pb)?;
            add_environment(&mut fb);
            Some(symmetric_eigen(&fb)?)
        }
    };

    // Energy: `½ Σ_{μν} w_μν P_μν (H + F)_μν`, over this subsystem's weighted pairs. Summing this
    // over subsystems gives the whole electronic energy because the weights are a partition of
    // unity over the pairs any subsystem holds.
    let mut energy_contribution = 0.0;
    for i in 0..s.len() {
        for j in 0..s.len() {
            let w = s.weight(i, j);
            if w == 0.0 {
                continue;
            }
            let (oi, oj) = (basis.atom_offset[i], basis.atom_offset[j]);
            for mu in 0..basis.atom_norb[i] {
                for nu in 0..basis.atom_norb[j] {
                    let p = p_local[(oi + mu, oj + nu)];
                    energy_contribution +=
                        0.5 * w * p * (core.h_core[(oi + mu, oj + nu)] + fock[(oi + mu, oj + nu)]);
                }
            }
        }
    }
    let _ = options;

    Ok(SubsystemSolution {
        eigenvalues,
        coefficients,
        beta,
        offset: basis.atom_offset.clone(),
        norb: basis.atom_norb.clone(),
        energy_contribution,
    })
}

/// Copy the global density's blocks into a dense matrix over the subsystem's basis.
fn local_density(s: &Subsystem, basis: &Basis, density: &SparseDensity) -> Matrix {
    let mut p = Matrix::zeros(basis.nao, basis.nao);
    for i in 0..s.len() {
        for j in 0..s.len() {
            let block = density.block(s.parent[i], s.parent[j]);
            if block.is_empty() {
                continue;
            }
            let (oi, oj) = (basis.atom_offset[i], basis.atom_offset[j]);
            for mu in 0..basis.atom_norb[i] {
                for nu in 0..basis.atom_norb[j] {
                    p[(oi + mu, oj + nu)] = block.get(mu, nu);
                }
            }
        }
    }
    p
}

/// Fermi–Dirac occupation, in electrons per spin-degenerate orbital.
#[inline]
fn occupation(e: f64, fermi: f64, width: f64) -> f64 {
    let x = (e - fermi) / width;
    if x > 40.0 {
        0.0
    } else if x < -40.0 {
        2.0
    } else {
        2.0 / (1.0 + x.exp())
    }
}

/// One `(eigenvalue, weighted norm, spin scale)` per state, precomputed for the Fermi bisection.
///
/// The norm `Σ_μ w_μμ |C_μ,index|²` is what makes a subsystem's electron count a *weighted* trace
/// rather than a plain occupation sum — and it depends only on `(subsystem, state)`, not on the
/// chemical potential. Recomputing it inside the bisection meant redoing `O(Σ_α m_α²)` work on
/// every one of 200 steps; hoisting it makes the search `O(Σ m_α² + 200 Σ m_α)`, and turns what
/// was the largest remaining serial stage into a scan over a flat array.
fn weighted_spectrum(solved: &[SubsystemSolution], parts: &Partitioning) -> Vec<Vec<(f64, f64)>> {
    use rayon::prelude::*;
    // Grouped **per subsystem**, not flattened. `count` sums each subsystem to a partial and then
    // sums the partials, exactly as the previous per-subsystem `map(...).sum()` did — flattening
    // would re-associate the additions and move the answer in the last bits for no reason.
    // Collected in order, so the thread count cannot change it either.
    let per_subsystem: Vec<Vec<(f64, f64)>> = solved
        .par_iter()
        .zip(&parts.subsystems)
        .map(|(solution, s)| {
            let channels: [(&[f64], &Matrix, f64); 2] = match &solution.beta {
                None => [
                    (&solution.eigenvalues, &solution.coefficients, 1.0),
                    (&[], &solution.coefficients, 0.0),
                ],
                Some((eb, cb)) => [
                    (&solution.eigenvalues, &solution.coefficients, 0.5),
                    (eb, cb, 0.5),
                ],
            };
            let mut out = Vec::new();
            for (eps, c, scale) in channels {
                if scale == 0.0 {
                    continue;
                }
                for (index, &e) in eps.iter().enumerate() {
                    let mut norm = 0.0;
                    for i in 0..s.len() {
                        let w = s.weight(i, i);
                        if w == 0.0 {
                            continue;
                        }
                        let off = solution.offset[i];
                        for mu in 0..solution.norb[i] {
                            let v = c[(off + mu, index)];
                            norm += w * v * v;
                        }
                    }
                    out.push((e, scale * norm));
                }
            }
            out
        })
        .collect();
    per_subsystem
}

/// The chemical potential that puts `n_elec` electrons into the weighted subsystem spectra.
fn find_fermi(
    solved: &[SubsystemSolution],
    parts: &Partitioning,
    n_elec: f64,
    width: f64,
) -> Result<f64> {
    let spectrum = weighted_spectrum(solved, parts);
    let count = |fermi: f64| -> f64 {
        spectrum
            .iter()
            .map(|states| {
                states
                    .iter()
                    .map(|(e, weight)| {
                        let f = occupation(*e, fermi, width);
                        // The original skipped a negligible occupation rather than adding it.
                        // Adding an exact zero is bit-identical, so keep the branch and keep the
                        // arithmetic identical to what it replaced.
                        if f < 1.0e-14 {
                            0.0
                        } else {
                            f * weight
                        }
                    })
                    .sum::<f64>()
            })
            .sum()
    };
    let (mut lo, mut hi) = (-500.0_f64, 500.0_f64);
    if count(lo) > n_elec || count(hi) < n_elec {
        return Err(Pm7Error::InvalidInput(format!(
            "no chemical potential in [-500, 500] eV holds {n_elec} electrons; the subsystem \
             spectra span {:.3} to {:.3} eV",
            solved
                .iter()
                .filter_map(|s| s.eigenvalues.first().copied())
                .fold(f64::INFINITY, f64::min),
            solved
                .iter()
                .filter_map(|s| s.eigenvalues.last().copied())
                .fold(f64::NEG_INFINITY, f64::max),
        )));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count(mid) < n_elec {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1.0e-12 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}
/// Add this subsystem's weighted density into the global one.
fn scatter_density(
    s: &Subsystem,
    solution: &SubsystemSolution,
    fermi: f64,
    width: f64,
    out: &mut SparseDensity,
    spin: bool,
) {
    let channels: Vec<(&[f64], &Matrix, f64)> = match &solution.beta {
        None => vec![(
            solution.eigenvalues.as_slice(),
            &solution.coefficients,
            if spin { 0.0 } else { 1.0 },
        )],
        Some((eb, cb)) => vec![
            (solution.eigenvalues.as_slice(), &solution.coefficients, 0.5),
            (eb, cb, if spin { -0.5 } else { 0.5 }),
        ],
    };
    for i in 0..s.len() {
        for j in 0..s.len() {
            let w = s.weight(i, j);
            if w == 0.0 {
                continue;
            }
            let (na, nb) = (solution.norb[i], solution.norb[j]);
            let (oi, oj) = (solution.offset[i], solution.offset[j]);
            let mut block = vec![0.0; na * nb];
            for (eps, c, scale) in &channels {
                if *scale == 0.0 {
                    continue;
                }
                for (index, &e) in eps.iter().enumerate() {
                    let f = occupation(e, fermi, width) * scale;
                    if f.abs() < 1.0e-14 {
                        continue;
                    }
                    for mu in 0..na {
                        let cm = c[(oi + mu, index)];
                        for nu in 0..nb {
                            block[mu * nb + nu] += f * cm * c[(oj + nu, index)];
                        }
                    }
                }
            }
            out.accumulate(s.parent[i], s.parent[j], w, &block);
        }
    }
}

/// Net atomic charges from the sparse density.
pub fn mulliken_charges(
    molecule: &Molecule,
    params: &Pm7Parameters,
    density: &SparseDensity,
) -> Result<Vec<f64>> {
    molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(a, atom)| Ok(params.element(atom.z)?.core_charge - density.population(a)))
        .collect()
}

/// The monopole field each subsystem atom feels from the atoms it does **not** contain, split
/// into its nuclear and electronic halves.
///
/// The split is not cosmetic. A mean-field electron–electron term has to be counted **once** in
/// the energy, and `½ Tr P(H + F)` does that only for a term living in `F` alone; a term in `H` —
/// where the nuclear attraction belongs, being linear in the density — is counted in full.
/// Folding the whole of `−V_A = −Σ_B M_AB (Z_B − P_B)` into the core Hamiltonian, which is the
/// obvious reading of "external potential", double counts the electronic half once per subsystem.
/// That error grows as `N²`: it reached 55 000 eV on a 240-atom chain before this split.
///
/// For a periodic cell the field is the Ewald potential minus what the subsystem already treats
/// exactly; for a molecule it is a direct sum, which is the one remaining `O(N²)` step —
/// `docs/performance.md` records where that shows up and what replaces it.
struct EnvironmentField {
    /// `Σ_{B∉α} M_AB Z_B`, subtracted from the core Hamiltonian's diagonal.
    nuclear: Vec<f64>,
    /// `Σ_{B∉α} M_AB P_B`, added to the Fock alone so the energy picks up its half.
    electronic: Vec<f64>,
}

/// The part of the environment potential that depends only on the **geometry**, built once.
///
/// The Ewald interaction matrix `M_AB` is a lattice sum over fixed positions, and the nuclear
/// half `Σ_B M_AB Z_B` contracts it with fixed core charges: neither moves during an SCF. They
/// used to be rebuilt on every iteration, which at the 50–57 iterations a divide-and-conquer run
/// takes meant repeating a serial `O(N² n_G)` lattice sum ~55 times for no reason —
/// `ewald_matrix`'s own doc says to build it once per geometry, and the molecular path already
/// did.
struct FarField {
    /// Flattened `M_AB`, stride `n`. `None` outside Ewald mode, where the direct `1/r` sum is
    /// preferable to storing the `N × N` matrix a linear-scaling method exists to avoid.
    ewald: Option<Vec<f64>>,
    /// `Σ_B M_AB Z_B` (or its direct-sum equivalent). Constant across the SCF.
    nuclear: Vec<f64>,
    positions: Vec<Vec3>,
    cores: Vec<f64>,
}

fn far_field(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
) -> Result<FarField> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let ev = crate::constants::PM7_EV;
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let cores: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.core_charge))
        .collect::<Result<Vec<_>>>()?;

    match options.pbc_for(molecule) {
        Some(pbc) if pbc.mode == crate::pbc::PbcMode::Ewald => {
            let cell = molecule.cell.expect("periodic");
            let ep =
                crate::pbc::EwaldParameters::new(&cell, n, pbc.ewald_accuracy, pbc.ewald_alpha);
            let rows = crate::pbc::ewald::ewald_matrix(&cell, &positions, &ep);
            // Flattened: the row-of-rows form is `n` separate heap allocations and a pointer
            // chase per element, on a matrix walked `n` times per iteration.
            let mut flat = Vec::with_capacity(n * n);
            for row in &rows {
                flat.extend_from_slice(row);
            }
            let nuclear = (0..n)
                .into_par_iter()
                .map(|a| (0..n).map(|b| flat[a * n + b] * cores[b]).sum())
                .collect();
            Ok(FarField {
                ewald: Some(flat),
                nuclear,
                positions,
                cores,
            })
        }
        _ => {
            let nuclear = (0..n)
                .into_par_iter()
                .map(|a| {
                    (0..n)
                        .filter(|b| *b != a)
                        .map(|b| ev * cores[b] / (positions[b] - positions[a]).norm())
                        .sum()
                })
                .collect();
            Ok(FarField {
                ewald: None,
                nuclear,
                positions,
                cores,
            })
        }
    }
}

impl FarField {
    /// The electronic half, `Σ_B M_AB P_B`. The only part that changes between iterations.
    fn electronic(&self, populations: &[f64]) -> Vec<f64> {
        use rayon::prelude::*;
        let n = populations.len();
        let ev = crate::constants::PM7_EV;
        match &self.ewald {
            Some(flat) => (0..n)
                .into_par_iter()
                .map(|a| {
                    let row = &flat[a * n..(a + 1) * n];
                    row.iter().zip(populations).map(|(m, p)| m * p).sum()
                })
                .collect(),
            None => (0..n)
                .into_par_iter()
                .map(|a| {
                    (0..n)
                        .filter(|b| *b != a)
                        .map(|b| {
                            ev * populations[b] / (self.positions[b] - self.positions[a]).norm()
                        })
                        .sum()
                })
                .collect(),
        }
    }
}

fn environment_field(
    molecule: &Molecule,
    far: &FarField,
    density: &SparseDensity,
    parts: &Partitioning,
) -> Result<Vec<EnvironmentField>> {
    use rayon::prelude::*;
    let n = molecule.atoms.len();
    let ev = crate::constants::PM7_EV;
    let cores = &far.cores;
    let populations: Vec<f64> = (0..n).map(|a| density.population(a)).collect();

    // The potential of *everything* at every atom, computed **once** per iteration rather than
    // once per subsystem. Doing it inside the subsystem loop is the same arithmetic repeated
    // `Σ_α |α| / N ≈ 5` times over, and it was what pushed the measured scaling exponent from
    // 1.1 to 1.5 past a thousand atoms. The nuclear half is now hoisted further still — out of
    // the SCF loop entirely, since it cannot change.
    let global_nuclear = &far.nuclear;
    let global_electronic = far.electronic(&populations);

    // What each subsystem already treats exactly is then removed pair by pair, over the
    // subsystem's *own* geometry. For a periodic parent that matters: a buffer atom is a specific
    // image, and the cluster accounts for that image alone — subtracting the whole lattice sum
    // `M_AB`, which covers every image of B, would take out far more than the subsystem put in.
    Ok(parts
        .subsystems
        .par_iter()
        .map(|s| {
            let mut field = EnvironmentField {
                nuclear: Vec::with_capacity(s.len()),
                electronic: Vec::with_capacity(s.len()),
            };
            for (i, position) in s.positions.iter().enumerate() {
                let (mut z, mut p) = (0.0, 0.0);
                for (j, other) in s.positions.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let r = (*other - *position).norm();
                    if r < 1.0e-8 {
                        continue;
                    }
                    z += ev * cores[s.parent[j]] / r;
                    p += ev * populations[s.parent[j]] / r;
                }
                field.nuclear.push(global_nuclear[s.parent[i]] - z);
                field.electronic.push(global_electronic[s.parent[i]] - p);
            }
            field
        })
        .collect())
}
