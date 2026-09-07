// SPDX-License-Identifier: GPL-3.0-or-later

//! **Berry-phase electronic polarization** — King-Smith and Vanderbilt's modern theory.
//!
//! # Why polarization is not `Σ q r`
//!
//! In a periodic crystal the dipole per cell is not a function of the charge density. Moving the
//! cell boundary moves a charge from one side to the other and changes `Σ q r` by a lattice vector
//! times that charge, so "the dipole of the unit cell" depends on where the cell was drawn. No
//! amount of care with the sum fixes that — the quantity is genuinely not defined, which is what
//! [`crate::dipole`] says when it declines to report one for a periodic system.
//!
//! What *is* defined is the **change** in polarization along an adiabatic path, and the object
//! whose changes those are is a Berry phase of the occupied Bloch states:
//!
//! ```text
//! P_el,α = (1/Ω) a_α · (1/N_⊥) Σ_{k_⊥} (1/2π) Im ln Π_j det S(k_j, k_{j+1})
//! ```
//!
//! The product runs along a **string** of k points spanning the Brillouin zone in direction `α`,
//! and `S` is the overlap of the occupied manifolds at neighbouring points. The result is defined
//! only **modulo the quantum** `a_α/Ω`: a different branch of the logarithm assigns the electrons
//! to a different cell, which is an equally valid choice. That ambiguity is the physics rather
//! than a defect, and [`BerryPolarization::quantum`] reports it so that
//! [`BerryPolarization::difference`] can put a difference on the branch nearest zero.
//!
//! # Why this exists when CPHF already gives Born charges
//!
//! It is an **independent route to the same number**. `Ω ∂P/∂τ_A` is the Born effective charge,
//! and this computes `P` with no linear response anywhere: a string of ordinary diagonalizations
//! and a determinant. A CPHF `Z*` and a Berry `Z*` share the Hamiltonian and the basis and share
//! essentially nothing else, so their agreement is evidence about the response solver in a way
//! that no internal consistency check can be. `tests/pbc_berry.rs` makes that comparison.
//!
//! # The overlap in an NDDO basis
//!
//! `S_mn(k, k+b) = ⟨u_mk | u_n,k+b⟩` is over the cell-periodic parts. This crate builds
//! `H(k) = Σ_T e^{ik·T} H(T)` — the phase carried on the lattice translation alone — so the
//! coefficients are in the **cell gauge** and
//!
//! ```text
//! S_mn(k, k+b) = Σ_μ c*_{μm}(k) e^{−i b·τ_μ} c_{μn}(k+b)
//! ```
//!
//! with `τ_μ` the position of the atom carrying orbital `μ`. NDDO assumes an orthonormal AO basis
//! and puts each orbital at its atom, which is the same approximation
//! [`crate::dipole::dipole_operator`] makes when it writes `R_A` on the diagonal of atom `A`'s
//! block. Using anything else here would make the Berry phase and the dipole disagree about where
//! an orbital sits.
//!
//! What it drops is the intra-atomic `s`–`p` hybridization moment that the dipole operator does
//! carry, and for PM7 also the `p`–`d` one. That is a real difference between the two routes; it
//! is measured rather than argued away.
//!
//! # Closing the string, and why it needs no extra factor
//!
//! In the cell gauge `H(k) = Σ_T e^{ik·T} H(T)` is **exactly periodic** in `k`: `G·T` is a
//! multiple of `2π` for every lattice translation, so `H(k + G) = H(k)` term by term and the
//! coefficients at `k_0 + G` are the coefficients at `k_0`. The last link is therefore an ordinary
//! link that happens to reuse the `k_0` vectors, with the same `e^{−ib·τ_μ}` as every other, and
//! the `e^{−iG·τ_μ}` that closes the loop appears on its own — `J` links each carrying `b = G/J`
//! multiply to exactly that.
//!
//! Worth stating because the *atomic* gauge, `H(k) = Σ_T e^{ik·(T + τ_ν − τ_μ)} H(T)`, is not
//! periodic in `k`, and there the closing link does need an explicit `e^{−iG·τ_μ}` on top. Adding
//! that correction to this gauge instead is a mistake whose signature is a Born charge tens of
//! electrons in size, which is what the cross-check against CPHF is for.
//!
//! Either way the product is manifestly gauge invariant in the other sense: whatever phase the
//! diagonalizer put on an eigenvector at an interior point appears once as `c*` and once as `c`
//! and cancels.

// The loop variables below are Cartesian directions, atom indices, or the rows and columns of a
// matrix being eliminated. The index *is* the meaning: `for axis in 0..3` says which direction,
// where an enumerate over an iterator says it less clearly and no more safely.
#![allow(clippy::needless_range_loop)]

use crate::basis::Basis;
use crate::cmatrix::CMatrix;
use crate::error::{Pm7Error, Result};
use crate::math::Vec3;
use crate::params::Pm7Parameters;
use crate::pbc::kpoints::KPoint;
use crate::scf::Pm7Options;
use crate::system::Molecule;

/// The polarization of a periodic cell, in `e/Bohr²` (atomic units: `e·Bohr` per `Bohr³`).
#[derive(Clone, Debug)]
pub struct BerryPolarization {
    /// Electronic contribution, from the Berry phase of the occupied manifold.
    pub electronic: Vec3,
    /// Ionic contribution, `(1/Ω) Σ_A Z_A τ_A` over the core charges.
    pub ionic: Vec3,
    /// Their sum. Defined **modulo** [`Self::quantum`].
    pub total: Vec3,
    /// The Berry phase along each lattice direction, in units of `2π`. The raw, branch-dependent
    /// number, reported because everything else here is derived from it.
    pub phase: [f64; 3],
    /// The polarization quantum along each lattice vector, `a_α/Ω`. Two polarizations describe the
    /// same physical state if they differ by an integer combination of these.
    pub quantum: [Vec3; 3],
    /// How many k points each string used.
    pub string_length: usize,
}

impl BerryPolarization {
    /// `other − self`, reduced onto the branch nearest zero along each lattice direction.
    ///
    /// The only physically meaningful thing to do with two polarizations. Subtracting the `total`
    /// fields is wrong whenever the two landed on different branches, which for a finite
    /// displacement is common and gives an answer off by exactly one quantum — a number that looks
    /// like a catastrophic error rather than like a bookkeeping choice.
    pub fn difference(&self, other: &Self) -> Vec3 {
        let delta = other.total - self.total;
        // A lattice reduction, done in the quanta's own basis rather than by projection.
        //
        // Projecting onto each quantum in turn and rounding is exact only for an **orthogonal**
        // lattice. The quanta are the lattice vectors over `Ω`, and for anything but a cubic cell
        // they are not orthogonal: an fcc cell has them at 60°, where the projection of a vector
        // onto one of them carries a component of the others and the greedy round lands on the
        // wrong lattice point. Measured on rocksalt LiF, whose polarization is zero by inversion
        // symmetry, the projection route left 1.43e-1 e/Bohr² standing against a quantum of
        // 4.88e-2 — nearly three quanta of "polarization" that is entirely the reduction failing.
        //
        // Solving `delta = Σ_i n_i q_i` for the coefficients and rounding those is exact for any
        // lattice, and is three rows of Cramer's rule.
        let [q0, q1, q2] = self.quantum;
        let det = q0.dot(q1.cross(q2));
        if det.abs() < 1.0e-30 {
            return delta;
        }
        let n0 = (delta.dot(q1.cross(q2)) / det).round();
        let n1 = (q0.dot(delta.cross(q2)) / det).round();
        let n2 = (q0.dot(q1.cross(delta)) / det).round();
        delta - q0 * n0 - q1 * n1 - q2 * n2
    }
}

/// Determinant of a complex matrix by Gaussian elimination with partial pivoting.
///
/// Only ever applied to an occupied-by-occupied overlap block, which is small — the cost here is
/// nothing beside the diagonalization that produced its inputs.
///
/// Returns zero for a singular matrix rather than failing: a vanishing overlap between adjacent
/// points on a string is a real condition (the string is too coarse to follow the manifold), and
/// the caller reports it against that cause rather than as an error from inside a loop.
fn determinant(mut a: Vec<Vec<[f64; 2]>>) -> [f64; 2] {
    let n = a.len();
    let mul = |x: [f64; 2], y: [f64; 2]| [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
    let norm = |x: [f64; 2]| (x[0] * x[0] + x[1] * x[1]).sqrt();
    let div = |x: [f64; 2], y: [f64; 2]| {
        let d = y[0] * y[0] + y[1] * y[1];
        [
            (x[0] * y[0] + x[1] * y[1]) / d,
            (x[1] * y[0] - x[0] * y[1]) / d,
        ]
    };
    let mut det = [1.0, 0.0];
    for col in 0..n {
        let mut pivot = col;
        let mut best = norm(a[col][col]);
        for row in (col + 1)..n {
            let size = norm(a[row][col]);
            if size > best {
                best = size;
                pivot = row;
            }
        }
        if best == 0.0 {
            return [0.0, 0.0];
        }
        if pivot != col {
            a.swap(pivot, col);
            det = [-det[0], -det[1]];
        }
        det = mul(det, a[col][col]);
        let diagonal = a[col][col];
        for row in (col + 1)..n {
            let factor = div(a[row][col], diagonal);
            if factor == [0.0, 0.0] {
                continue;
            }
            for k in col..n {
                let value = mul(a[col][k], factor);
                a[row][k][0] -= value[0];
                a[row][k][1] -= value[1];
            }
        }
    }
    det
}

/// `S_mn = Σ_μ c*_{μm}(left) e^{−i b·τ_μ} c_{μn}(right)`, over the lowest `n_occ` bands.
///
/// `phase_per_ao` is `e^{−i b·τ_μ}` precomputed per orbital, since it is the same for every pair
/// of bands and depends only on the step.
fn overlap_block(
    left: &CMatrix,
    right: &CMatrix,
    phase_per_ao: &[[f64; 2]],
    n_occ: usize,
) -> Vec<Vec<[f64; 2]>> {
    let nao = phase_per_ao.len();
    let mut s = vec![vec![[0.0_f64; 2]; n_occ]; n_occ];
    for (m, row) in s.iter_mut().enumerate() {
        for (n, entry) in row.iter_mut().enumerate() {
            let mut acc = [0.0_f64; 2];
            for mu in 0..nao {
                let (lr, li) = left.get(mu, m);
                let (rr, ri) = right.get(mu, n);
                // conj(left) * phase * right
                let p = phase_per_ao[mu];
                let a = [lr * p[0] + li * p[1], lr * p[1] - li * p[0]];
                acc[0] += a[0] * rr - a[1] * ri;
                acc[1] += a[0] * ri + a[1] * rr;
            }
            *entry = acc;
        }
    }
    s
}

/// The Berry-phase polarization of a periodic cell.
///
/// `strings` is the number of k points along each Brillouin-zone string, and it is **the**
/// convergence parameter: the answer has to become independent of it, which is a thing to check
/// rather than a value to assume.
///
/// The transverse sampling comes from the calculation's own k mesh. A string is a one-dimensional
/// integration for each transverse point, so the work is `strings × (transverse points)`
/// diagonalizations in a potential that is converged once.
///
/// # What it refuses
///
/// A cell that is not fully periodic. The quantum is `a_α/Ω` and `Ω` has to be a volume; a slab or
/// a chain has a polarization along its periodic directions only, which this does not separate
/// out.
pub fn berry_polarization(
    molecule: &Molecule,
    params: &Pm7Parameters,
    options: &Pm7Options,
    strings: usize,
) -> Result<BerryPolarization> {
    let cell = molecule
        .cell
        .ok_or_else(|| Pm7Error::InvalidInput("a Berry phase needs a periodic cell".into()))?;
    if cell.vectors().len() != 3 {
        return Err(Pm7Error::InvalidInput(
            "the Berry-phase polarization is defined here for a three-dimensional cell: the \
             quantum is a/Omega and Omega has to be a volume. A slab or a chain has a \
             polarization along its periodic directions only, which this does not separate out."
                .into(),
        ));
    }
    if strings < 3 {
        return Err(Pm7Error::InvalidInput(format!(
            "a Berry-phase string needs at least 3 k points and got {strings}: the discretized \
             phase is a product of nearest-neighbour overlaps, and two points cannot resolve a \
             winding."
        )));
    }

    let pbc = options
        .pbc_for(molecule)
        .ok_or_else(|| Pm7Error::InvalidInput("a Berry phase needs periodic options".into()))?;
    let scf = crate::scf::run_pm7(molecule, params, options)?;
    let basis = Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_periodic_field(
        molecule,
        &basis,
        params,
        options.force_dpath,
        &pbc,
        options.active_field(),
    )?;
    // The converged Fock, per spin channel, as Bloch blocks. `at_k` then gives `H(k)` at *any* k,
    // on or off the mesh, because in the cell gauge the Bloch sum is defined for every k.
    let density = scf
        .bloch_density
        .as_ref()
        .ok_or_else(|| {
            Pm7Error::InvalidInput(
                "a Berry phase needs a k-point ground state; run with a k mesh so the Bloch \
                 blocks of the converged density exist"
                    .into(),
            )
        })?
        .clone();
    let focks = crate::dfpt::converged_spin_focks(molecule, &basis, params, &core, &scf, &density)?;

    // Where each orbital sits: its atom's position. The same approximation the dipole operator
    // makes, and using anything else would make the two disagree about the same crystal.
    let mut tau = vec![Vec3::zero(); basis.nao];
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let start = basis.atom_offset[a];
        for slot in tau[start..start + basis.atom_norb[a]].iter_mut() {
            *slot = atom.position;
        }
    }

    // How many bands are filled, and what each holds. A restricted calculation has one manifold of
    // doubly occupied bands; an unrestricted one has two manifolds of singly occupied bands.
    let n_elec: f64 = molecule
        .atoms
        .iter()
        .map(|atom| params.element(atom.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum::<f64>()
        - options.charge;
    let manifolds: Vec<(usize, usize, f64)> = if scf.unrestricted {
        let total = n_elec.round() as usize;
        let unpaired = options.multiplicity.saturating_sub(1);
        let n_beta = (total - unpaired) / 2;
        vec![(0, total - n_beta, 1.0), (1, n_beta, 1.0)]
    } else {
        vec![(0, (n_elec / 2.0).round() as usize, 2.0)]
    };
    for (_, n_occ, _) in &manifolds {
        if *n_occ == 0 || *n_occ > basis.nao {
            return Err(Pm7Error::InvalidInput(format!(
                "the occupied manifold has {n_occ} bands against {} orbitals; a Berry phase needs \
                 a filled manifold to follow from one k point to the next",
                basis.nao
            )));
        }
    }

    let reciprocal = cell.reciprocal_2pi();
    let divisions = pbc.kmesh.divisions();
    let mut phase = [0.0_f64; 3];

    for axis in 0..3 {
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut total_phase = 0.0_f64;
        let mut transverse = 0usize;

        for i1 in 0..divisions[t1].max(1) {
            for i2 in 0..divisions[t2].max(1) {
                let mut base = [0.0_f64; 3];
                base[t1] = i1 as f64 / divisions[t1].max(1) as f64;
                base[t2] = i2 as f64 / divisions[t2].max(1) as f64;

                let mut string_phase = 0.0_f64;
                for &(spin, n_occ, weight) in &manifolds {
                    let mut coefficients: Vec<CMatrix> = Vec::with_capacity(strings);
                    for j in 0..strings {
                        let mut frac = base;
                        frac[axis] = j as f64 / strings as f64;
                        let mut cart = Vec3::zero();
                        for d in 0..3 {
                            cart += reciprocal[d] * frac[d];
                        }
                        let k = KPoint {
                            frac,
                            cart,
                            weight: 1.0,
                            time_reversal_pair: false,
                        };
                        let (_, vectors) = focks[spin].at_k(&k).hermitian_eigen()?;
                        coefficients.push(vectors);
                    }

                    // `b`, the step between adjacent points, in Cartesian reciprocal space.
                    let step = reciprocal[axis] / strings as f64;
                    let phase_per_ao: Vec<[f64; 2]> = tau
                        .iter()
                        .map(|position| {
                            let angle = -step.dot(*position);
                            [angle.cos(), angle.sin()]
                        })
                        .collect();

                    let mut accumulated = [1.0_f64, 0.0];
                    for j in 0..strings {
                        // The last link closes onto `k_0 + G`, whose coefficients *are* the `k_0`
                        // coefficients in this gauge — see the module note. So it is an ordinary
                        // link with the same step factor, and the `e^{−iG·τ}` that closes the loop
                        // is the product of the `J` steps rather than an extra term on one of them.
                        let right = if j + 1 == strings { 0 } else { j + 1 };
                        let block = overlap_block(
                            &coefficients[j],
                            &coefficients[right],
                            &phase_per_ao,
                            n_occ,
                        );
                        let d = determinant(block);
                        let size = (d[0] * d[0] + d[1] * d[1]).sqrt();
                        if size == 0.0 {
                            return Err(Pm7Error::InvalidInput(format!(
                                "the overlap between adjacent points of the string along axis \
                                 {axis} is singular, so the occupied manifold cannot be followed \
                                 from one to the next. Increase `strings` above {strings}."
                            )));
                        }
                        // Accumulated as a product and renormalized each step: `strings` complex
                        // multiplications otherwise overflow or underflow the modulus long before
                        // the argument, and the argument is the only part that matters.
                        let next = [
                            accumulated[0] * d[0] - accumulated[1] * d[1],
                            accumulated[0] * d[1] + accumulated[1] * d[0],
                        ];
                        let n = (next[0] * next[0] + next[1] * next[1]).sqrt();
                        accumulated = [next[0] / n, next[1] / n];
                    }
                    string_phase +=
                        weight * accumulated[1].atan2(accumulated[0]) / std::f64::consts::TAU;
                }
                total_phase += string_phase;
                transverse += 1;
            }
        }
        phase[axis] = total_phase / transverse as f64;
    }

    let volume = cell.measure();
    // `P_el = (1/Ω) Σ_α φ_α a_α`, with `φ` in units of `2π`.
    //
    // The sign follows from the `e^{−ib·τ_μ}` in the overlap and is fixed by a limit rather than by
    // a convention quoted from elsewhere. Take one orbital in a large box: `c = 1`, every link
    // contributes `e^{−ib·τ}`, and the `J` of them multiply to `e^{−iG·τ}`, so `φ = −τ_α/a_α` in
    // units of `2π`. An electron there carries `P = −(occupancy)·τ_α/Ω`, which is
    // `+a_α φ_α · occupancy / Ω` — a plus, because the minus of the electron's charge and the minus
    // already inside `φ` cancel. Writing the textbook `−(e/Ω) φ a` on top of *this* overlap
    // convention double-counts that sign, and its signature is a Born charge tens of electrons in
    // size.
    let mut electronic = Vec3::zero();
    for axis in 0..3 {
        electronic += cell.vectors()[axis] * (phase[axis] / volume);
    }

    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        ionic += atom.position * (params.element(atom.z)?.core_charge / volume);
    }

    Ok(BerryPolarization {
        electronic,
        ionic,
        total: electronic + ionic,
        phase,
        // The quantum is `a/Omega` per lattice vector -- the **single-electron** one.
        //
        // Tempting to scale it by the occupancy, on the argument that a spin-restricted
        // calculation moves two electrons at once so its branch ambiguity is `2a/Omega`. That is
        // true of the branch of the logarithm and false of the quantity: the indeterminacy of a
        // polarization is moving one electron by a lattice vector, and the **ionic** term realizes
        // half-integer multiples of `a/Omega` whenever an ion with an odd core charge sits at a
        // half-lattice site. Rocksalt LiF is exactly that case -- fluorine at fractional
        // (-1/2, 1/2, 1/2) with core charge 7 puts `P_ionic` at `3.5 (a,0,0)/Omega` -- and against
        // a doubled quantum neither `P` nor `2P` reduces to anything meaningful.
        quantum: [
            cell.vectors()[0] / volume,
            cell.vectors()[1] / volume,
            cell.vectors()[2] / volume,
        ],
        string_length: strings,
    })
}
