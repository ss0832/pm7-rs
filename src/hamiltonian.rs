// SPDX-License-Identifier: GPL-3.0-or-later

//! Core (one-electron) Hamiltonian assembly.
//!
//! `H_core` holds the diagonal atomic energies `U_ss/U_pp`, the electron–core attraction
//! to every other atom (from the NDDO integrals), and the inter-atomic resonance
//! `H_μν = ½(β_μ + β_ν) S_μν`. The per-pair two-electron integrals are returned alongside
//! for reuse in the Fock build.

use crate::basis::Basis;
use crate::error::Result;
use crate::integrals::{pair_two_electron, PairTwoElec};
use crate::linalg::Matrix;
use crate::mndod_twocenter::pair_two_electron_d_g;
use crate::overlap::diatom_overlap;
use crate::overlap_d::diat_overlap;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// Rotated two-electron integrals for one atom pair, tagged with the ordered atom indices
/// (`a` is the heavy atom when the other is H).
pub struct PairIntegral {
    pub a: usize,
    pub b: usize,
    pub te: PairTwoElec,
    /// Point-charge monopole value `PM7_EV / |d|` for this pair, or `0.0` when no long-range
    /// treatment is active.
    ///
    /// In a periodic Ewald run the monopole part of every Coulomb-like term is summed over the
    /// whole lattice by [`crate::pbc::ewald`], so it must not also be counted here: the Coulomb
    /// contraction uses `(μν|λσ) − δ_μν δ_λσ v_point`, and the electron–core and core–core terms
    /// have their own monopoles removed the same way. **Exchange keeps the full integral**,
    /// because the Ewald sum reconstructs a Coulomb series only.
    pub v_point: f64,
    /// True when this entry is an atom paired with one of its own periodic images (`a == b`,
    /// lattice translation ≠ 0). The Fock scatter then writes both contributions into the same
    /// atom block, which is correct and is why the block offsets may coincide.
    pub self_image: bool,
    /// Lattice translation of atom `b` relative to atom `a`, already corrected for the
    /// heavier-atom-first reordering. `[0, 0, 0]` for a molecule.
    ///
    /// Carrying it here rather than re-deriving it means the Fock, the gradient, and the force
    /// constants all index the same translation the integral was evaluated at — there is no
    /// second pair list that could enumerate images differently.
    pub t: [i32; 3],
}

/// Long-range electrostatics attached to a periodic core Hamiltonian.
///
/// The Ewald interaction matrix depends only on the geometry, so it is built once and contracted
/// against the current Mulliken charges each SCF iteration.
#[derive(Clone, Debug)]
pub struct EwaldContext {
    /// `M[a][b]` in eV per unit charge², with `E = ½ qᵀ M q` and `V = M q`.
    pub matrix: Vec<Vec<f64>>,
    /// Core charge `Z_A` (MOPAC's `tore`) of each atom.
    pub core_charge: Vec<f64>,
    /// Born–von Kármán resolved exchange lattice sums, present only for a k-point mesh.
    ///
    /// `exchange[t][a][b] = M^super(R_AB + T_t) − M^super_self`, where `M^super` is the
    /// regularized lattice sum of the **BvK supercell** and `T_t` runs over the residue classes
    /// of the primitive cell inside it. Summing over `t` reproduces the primitive lattice sum
    /// (`splitting_a_lattice_sum_over_supercell_residue_classes_reproduces_it`), so a `n×1×1`
    /// mesh and the `n`-fold supercell see the same exchange tail — which is what makes the
    /// Born–von Kármán identity hold rather than nearly hold.
    pub exchange: Option<BvkExchange>,
}

/// The exchange lattice sums of a Born–von Kármán supercell, resolved by residue class.
#[derive(Clone, Debug)]
pub struct BvkExchange {
    /// Representative translation of each residue class, in the symmetric range.
    pub translations: Vec<[i32; 3]>,
    /// `values[t][a][b]`, in eV per unit charge².
    pub values: Vec<Vec<Vec<f64>>>,
}

/// The exchange lattice sum uses `M_AB − M_self` rather than `M_AB`.
///
/// `M_self = M_AA` is the Madelung self-potential of the lattice — the same for every atom,
/// since it depends only on the cell. Subtracting it is the Γ-point **exchange divergence
/// correction** (Gygi–Baldereschi / Spencer–Alavi), and without it the periodic exchange
/// carries a spurious term that decays only as `1/L`:
///
/// * `M̃_AA = 0`, so an atom has no exchange with its own images — which is what makes a lone
///   atom in a box reproduce the isolated atom *exactly* rather than to `O(1/L)`.
/// * `M̃_AB → 1/R_AB` as the cell grows (because `M_AB → 1/R_AB + M_self`), so a molecule in a
///   large box converges to the molecular energy with an `O(R²/L³)` residual instead of `O(1/L)`.
///
/// Coulomb is deliberately left alone: there the dropped `G = 0` term multiplies `q_tot²`, which
/// is zero for a neutral cell and is the physically intended jellium for a charged one.
pub fn exchange_potential(matrix: &[Vec<f64>], a: usize, b: usize) -> f64 {
    matrix[a][b] - matrix[0][0]
}

impl EwaldContext {
    /// The Madelung self-potential `M_self` subtracted from the exchange lattice sum.
    pub fn self_potential(&self) -> f64 {
        self.matrix
            .first()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0)
    }
}

impl EwaldContext {
    /// `V_A = (M q)_A`, the potential each unit of electron population feels.
    pub fn potential(&self, charges: &[f64]) -> Vec<f64> {
        self.matrix
            .iter()
            .map(|row| row.iter().zip(charges).map(|(m, q)| m * q).sum())
            .collect()
    }

    /// Net atomic charges `q_A = Z_A − P_A` from a density matrix.
    pub fn charges(&self, basis: &Basis, density: &Matrix) -> Vec<f64> {
        self.core_charge
            .iter()
            .enumerate()
            .map(|(ia, z)| {
                let off = basis.atom_offset[ia];
                let n = basis.atom_norb[ia];
                let pop: f64 = (0..n).map(|mu| density[(off + mu, off + mu)]).sum();
                z - pop
            })
            .collect()
    }
}

pub struct CoreHamiltonian {
    pub h_core: Matrix,
    pub pairs: Vec<PairIntegral>,
    /// Present only for a periodic system in [`crate::pbc::PbcMode::Ewald`].
    pub ewald: Option<EwaldContext>,
    /// Translation-resolved `H(T)`, present only when a k-point mesh is in use.
    ///
    /// At the Γ point every translation enters the same sum, so `h_core` is all that is needed
    /// and these blocks are not built. `h_core == bloch.gamma_sum()` whenever both exist, which
    /// [`crate::scf_pbc`] asserts.
    pub bloch: Option<crate::scf_pbc::BlochBlocks>,
}

/// Representative translations of the residue classes of a cell inside its `n₁×n₂×n₃` Born–von
/// Kármán supercell, folded into the symmetric range so the set is closed under negation.
pub fn bvk_representatives(n: [usize; 3]) -> Vec<[i32; 3]> {
    let fold = |i: usize, m: usize| -> i32 {
        let m = m as i32;
        let i = i as i32;
        if 2 * i > m {
            i - m
        } else {
            i
        }
    };
    let mut out = Vec::with_capacity(n[0] * n[1] * n[2]);
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                out.push([fold(i, n[0]), fold(j, n[1]), fold(k, n[2])]);
            }
        }
    }
    out
}

/// Which residue class a translation belongs to, as an index into [`bvk_representatives`].
pub fn bvk_class(t: [i32; 3], n: [usize; 3]) -> usize {
    let wrap = |x: i32, m: usize| -> usize {
        let m = m as i32;
        (((x % m) + m) % m) as usize
    };
    let (i, j, k) = (wrap(t[0], n[0]), wrap(t[1], n[1]), wrap(t[2], n[2]));
    (i * n[1] + j) * n[2] + k
}

/// Build the Born–von Kármán resolved exchange lattice sums for a k-point mesh.
fn bvk_exchange(
    cell: &crate::cell::Cell,
    positions: &[crate::math::Vec3],
    n: [usize; 3],
    opts: &crate::pbc::PbcOptions,
) -> Result<BvkExchange> {
    let (super_cell, _) = cell.supercell(n)?;
    let n_atoms = positions.len();
    let ep = crate::pbc::EwaldParameters::new(
        &super_cell,
        n_atoms * n[0] * n[1] * n[2],
        opts.ewald_accuracy,
        opts.ewald_alpha,
    );
    let translations = bvk_representatives(n);
    // One flat list of displacements so the reciprocal sum is walked once, not once per pair.
    let mut displacements = Vec::with_capacity(translations.len() * n_atoms * n_atoms + 1);
    displacements.push(crate::math::Vec3::zero()); // the self-potential, for the correction
    for t in &translations {
        let shift = cell.translation(*t);
        for a in 0..n_atoms {
            for b in 0..n_atoms {
                displacements.push(positions[b] + shift - positions[a]);
            }
        }
    }
    let phi = crate::pbc::ewald::ewald_potentials_at(&super_cell, &displacements, &ep);
    let self_potential = phi[0];
    let mut values = Vec::with_capacity(translations.len());
    let mut cursor = 1usize;
    for _ in &translations {
        let mut block = vec![vec![0.0_f64; n_atoms]; n_atoms];
        for row in block.iter_mut() {
            for v in row.iter_mut() {
                *v = phi[cursor] - self_potential;
                cursor += 1;
            }
        }
        values.push(block);
    }
    Ok(BvkExchange {
        translations,
        values,
    })
}

/// Return the resonance β for orbital index `orb` (0 = s, 1..3 = p, 4..8 = d).
#[inline]
fn beta_of(elem: &crate::params::Pm7Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

pub fn build_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
) -> Result<CoreHamiltonian> {
    build_core_impl(molecule, basis, params, false)
}

/// Like [`build_core`] but with an explicit `force_dpath` diagnostic flag.
pub fn build_core_with(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
) -> Result<CoreHamiltonian> {
    build_core_impl(molecule, basis, params, force_dpath)
}

fn build_core_impl(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
) -> Result<CoreHamiltonian> {
    build_core_generic(molecule, basis, params, force_dpath, None, None)
}

/// Like [`build_core`] but with a uniform external electric field folded into `h_core`.
///
/// Molecular callers that honour [`crate::scf::Pm7Options::field`] use this; the three-argument
/// [`build_core`] is the no-field path and stays bit-identical.
pub fn build_core_field(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
    field: Option<&crate::field::ExternalField>,
) -> Result<CoreHamiltonian> {
    build_core_generic(molecule, basis, params, force_dpath, None, field)
}

/// Core Hamiltonian for a periodic system at the Γ point.
///
/// At the Γ point the real-space density matrix is the same for every lattice translation, so
/// `H_Γ = Σ_T H(T)` and the whole SCF reduces to the molecular one with image-summed integrals.
/// That is why there is no separate periodic SCF driver: this function produces a
/// [`CoreHamiltonian`] the existing loops consume unchanged.
pub fn build_core_periodic(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
    pbc: &crate::pbc::PbcOptions,
) -> Result<CoreHamiltonian> {
    build_core_generic(molecule, basis, params, force_dpath, Some(pbc), None)
}

/// Like [`build_core_periodic`] but with a uniform external field folded into `h_core`.
///
/// A field is only admissible along a **non-periodic** direction — `−f·r` is unbounded under a
/// lattice translation, so a component along a periodic axis is not a well-defined operator at
/// all. [`crate::field::validate_for`] enforces that before the SCF starts; by the time a field
/// reaches here it has already been checked, and the term added is the same one the molecular
/// path adds. Every periodic caller goes through this, so a field can no longer be accepted at
/// the API boundary and then quietly dropped on the way to `h_core`.
pub fn build_core_periodic_field(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
    pbc: &crate::pbc::PbcOptions,
    field: Option<&crate::field::ExternalField>,
) -> Result<CoreHamiltonian> {
    build_core_generic(molecule, basis, params, force_dpath, Some(pbc), field)
}

fn build_core_generic(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    force_dpath: bool,
    pbc: Option<&crate::pbc::PbcOptions>,
    field: Option<&crate::field::ExternalField>,
) -> Result<CoreHamiltonian> {
    use crate::pbc::{PairList, PbcMode};
    let nao = basis.nao;
    let mut h = Matrix::zeros(nao, nao);

    // Diagonal one-electron energies U_ss / U_pp / U_dd.
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h[(mu, mu)] = match ao.orb {
            0 => elem.u_ss,
            1..=3 => elem.u_pp,
            _ => elem.u_dd,
        };
    }

    // A uniform external field, at exactly the point MOPAC adds it (`hcore.F90:184-206`): after
    // the `U` diagonal and before any two-centre term, so nothing downstream can tell the
    // difference between a field and a shifted set of one-electron energies. The `is_zero` guard
    // is exact, so a zero field leaves `h` untouched bit for bit.
    if let Some(f) = field.filter(|f| !f.is_zero()) {
        let one_electron = f.one_electron(molecule, basis, params)?;
        for i in 0..nao {
            for j in 0..nao {
                h[(i, j)] += one_electron[(i, j)];
            }
        }
    }

    use rayon::prelude::*;

    // A molecule containing any d-bearing atom is evaluated entirely in the
    // MNDO/d rotation frame (two-electron *and* overlap) so that H_core mixes no
    // frames; a pure sp molecule uses the faster sp Dewar–Sabelli–Klopman path.
    let has_any_d = force_dpath
        || molecule
            .atoms
            .iter()
            .any(|a| params.element(a.z).map(|e| e.n_orb == 9).unwrap_or(false));

    // The image pair list subsumes the molecular enumeration: with no cell it is exactly the
    // `a < b, T = 0` list the molecular code used to build inline, in the same order, so the
    // molecular path stays bit-identical.
    //
    // One cutoff is enough. Past the feather range *every* two-centre term — Coulomb and
    // exchange alike — is exactly its point-charge monopole, and both monopoles are summed over
    // the whole lattice by Ewald. What is left here has compact support, so the short-range
    // cutoff is the only one the pair list needs.
    let cutoff = pbc.map(|p| p.short_range_cutoff).unwrap_or(f64::INFINITY);
    let list = PairList::cached(molecule, cutoff);
    // Ewald removes the monopole from every Coulomb-like term; MOPAC-compat mode does not
    // (it sums the monopoles in real space with its own truncation instead).
    let subtract_monopole = matches!(pbc.map(|p| p.mode), Some(PbcMode::Ewald));

    type Computed = (
        usize,
        usize,
        bool,
        f64,
        PairTwoElec,
        [[f64; 9]; 9],
        [i32; 3],
    );
    let computed: Vec<Computed> = list
        .pairs
        .par_iter()
        .map(|pair| -> Result<Computed> {
            let (u, v) = (pair.a, pair.b);
            let eu = params.element(molecule.atoms[u].z)?;
            let ev = params.element(molecule.atoms[v].z)?;
            // Ordered pair: heavier atom (more AOs) first. Swapping the atoms reverses the
            // displacement, which is what keeps the integral frames consistent.
            let (a, b, d) = if eu.n_orb >= ev.n_orb {
                (u, v, pair.d)
            } else {
                (v, u, pair.d * -1.0)
            };
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let r = pair.r;
            // Beyond the feather range the two-electron block is *exactly* a point charge, so
            // for a periodic run it is built in O(1) rather than evaluated. The resonance
            // `beta.S` is dropped with it: at 7 A a valence Slater overlap is already ~1e-11 eV
            // once multiplied by beta, five orders below the 1e-5 kcal/mol agreement this crate
            // holds itself to. A molecule keeps the full evaluation even past 7 A, so published
            // molecular numbers stay bit-identical.
            if pbc.is_some() && r > crate::pbc::FEATHER_RANGE_BOHR {
                let v_point = if subtract_monopole {
                    crate::constants::PM7_EV / r
                } else {
                    0.0
                };
                return Ok((
                    a,
                    b,
                    pair.is_self_image(),
                    v_point,
                    crate::integrals::point_charge_pair(ea, eb, r),
                    [[0.0; 9]; 9],
                    if a == pair.a {
                        pair.t
                    } else {
                        [-pair.t[0], -pair.t[1], -pair.t[2]]
                    },
                ));
            }
            let (te, s_block): (PairTwoElec, [[f64; 9]; 9]) = if has_any_d {
                (
                    pair_two_electron_d_g::<f64>(ea, eb, [d.x, d.y, d.z]),
                    diat_overlap::<f64>(ea, eb, [d.x, d.y, d.z]),
                )
            } else {
                let mut s9 = [[0.0; 9]; 9];
                // `diatom_overlap` takes positions; feeding it the origin and the displacement
                // is equivalent and is what makes an image pair work without inventing an
                // image atom.
                let s4 = diatom_overlap(ea, crate::math::Vec3::zero(), eb, d)?;
                for i in 0..4 {
                    s9[i][..4].copy_from_slice(&s4[i][..4]);
                }
                (pair_two_electron(ea, eb, d / r, r), s9)
            };
            let v_point = if subtract_monopole {
                crate::constants::PM7_EV / r
            } else {
                0.0
            };
            Ok((
                a,
                b,
                pair.is_self_image(),
                v_point,
                te,
                s_block,
                if a == pair.a {
                    pair.t
                } else {
                    // The pair was reordered heavier-first, which reverses the translation.
                    [-pair.t[0], -pair.t[1], -pair.t[2]]
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    // Translation-resolved blocks, needed only for a k-point mesh: at the Γ point every `H(T)`
    // enters the same sum, so building them would be pure overhead.
    let mut bloch = match pbc {
        Some(p) if !matches!(p.kmesh, crate::pbc::KMesh::Gamma) => {
            // The block set must cover both the pair list's translations *and* every Born–von
            // Kármán residue class, because the long-range exchange writes one contribution per
            // class and those classes need not appear among the interacting pairs.
            let mut ts = list.translations();
            for t in bvk_representatives(p.kmesh.divisions()) {
                if !ts.contains(&t) {
                    ts.push(t);
                }
                let neg = [-t[0], -t[1], -t[2]];
                if !ts.contains(&neg) {
                    ts.push(neg);
                }
            }
            ts.sort_unstable();
            Some(crate::scf_pbc::BlochBlocks::new(ts, nao)?)
        }
        _ => None,
    };
    if let Some(b) = bloch.as_mut() {
        // Every one-centre term written into `h` so far belongs to the zero translation: the
        // `U_ss/U_pp/U_dd` diagonal, and — when there is a field — its intra-atomic operator.
        //
        // The whole **on-site block** is copied, not just the diagonal. The field's `<s|r_k|p_k>`
        // hybrid is one-centre but *off*-diagonal, so a diagonal-only copy dropped it and left a
        // k-point SCF running with a field that had lost its hybrid term. That is invisible in
        // the energy to about a part in 10^3 and completely wrong in `dE/df`, which is how it was
        // found: the field's Hellmann-Feynman dipole came back as the point-charge term alone.
        //
        // Nothing else has touched `h` yet, so for a field-free run every element copied here
        // outside the diagonal is an exact zero and the block is bit-identical to before.
        for ia in 0..molecule.atoms.len() {
            let off = basis.atom_offset[ia];
            let norb = basis.atom_norb[ia];
            for i in 0..norb {
                for j in 0..norb {
                    b.at_mut([0, 0, 0])[(off + i, off + j)] = h[(off + i, off + j)];
                }
            }
        }
    }

    // Assemble H_core serially from the precomputed per-pair integrals, in list order.
    let mut pairs = Vec::with_capacity(computed.len());
    for (a, b, self_image, v_point, te, s_block, t) in computed {
        let (ea, eb) = (
            params.element(molecule.atoms[a].z)?,
            params.element(molecule.atoms[b].z)?,
        );
        let off_a = basis.atom_offset[a];
        let off_b = basis.atom_offset[b];
        let na = basis.atom_norb[a];
        let nb = basis.atom_norb[b];

        // Electron–core attraction: e1b onto atom a's block, e2a onto atom b's block. The
        // monopole part (`−Z·v_point` on the diagonal) is removed when Ewald will re-supply it.
        let neg_t = [-t[0], -t[1], -t[2]];
        for i in 0..na {
            for j in 0..na {
                let mono = if i == j {
                    -eb.core_charge * v_point
                } else {
                    0.0
                };
                let value = te.e1b[i][j] - mono;
                h[(off_a + i, off_a + j)] += value;
                // The electron–core attraction is an on-site block: it lives entirely at T = 0
                // however far away the attracting core is.
                if let Some(b) = bloch.as_mut() {
                    b.at_mut([0, 0, 0])[(off_a + i, off_a + j)] += value;
                }
            }
        }
        for i in 0..nb {
            for j in 0..nb {
                let mono = if i == j {
                    -ea.core_charge * v_point
                } else {
                    0.0
                };
                let value = te.e2a[i][j] - mono;
                h[(off_b + i, off_b + j)] += value;
                if let Some(b) = bloch.as_mut() {
                    b.at_mut([0, 0, 0])[(off_b + i, off_b + j)] += value;
                }
            }
        }

        // Resonance β·S. A molecular pair writes each element once, so `=` and `+=` agree
        // there; a periodic system can have several images of the same atom pair contributing
        // to the same Γ-point block, so this must accumulate.
        for i in 0..na {
            let bi = beta_of(ea, basis.aos[off_a + i].orb);
            for j in 0..nb {
                let bj = beta_of(eb, basis.aos[off_b + j].orb);
                let value = 0.5 * (bi + bj) * s_block[i][j];
                h[(off_a + i, off_b + j)] += value;
                // A self-image pair has `off_a == off_b`; writing the transpose as well is
                // right there too, because the entry stands for both `+T` and `−T`.
                h[(off_b + j, off_a + i)] += value;
                // Translation-resolved: the block for `T` and its transpose for `−T`, which is
                // what makes every `H(k)` Hermitian by construction.
                if let Some(bl) = bloch.as_mut() {
                    bl.at_mut(t)[(off_a + i, off_b + j)] += value;
                    bl.at_mut(neg_t)[(off_b + j, off_a + i)] += value;
                }
            }
        }

        pairs.push(PairIntegral {
            a,
            b,
            te,
            v_point,
            self_image,
            t,
        });
    }

    // Long-range monopole electrostatics.
    let ewald = match (molecule.cell, pbc) {
        (Some(cell), Some(opts)) if opts.mode == PbcMode::Ewald => {
            let positions: Vec<crate::math::Vec3> =
                molecule.atoms.iter().map(|a| a.position).collect();
            let ep = crate::pbc::EwaldParameters::new(
                &cell,
                positions.len(),
                opts.ewald_accuracy,
                opts.ewald_alpha,
            );
            let matrix = crate::pbc::ewald::ewald_matrix(&cell, &positions, &ep);
            let core_charge = molecule
                .atoms
                .iter()
                .map(|a| params.element(a.z).map(|e| e.core_charge))
                .collect::<Result<Vec<_>>>()?;
            let exchange = match &opts.kmesh {
                crate::pbc::KMesh::Gamma => None,
                mesh => Some(bvk_exchange(&cell, &positions, mesh.divisions(), opts)?),
            };
            Some(EwaldContext {
                matrix,
                core_charge,
                exchange,
            })
        }
        _ => None,
    };

    if let Some(b) = bloch.as_mut() {
        // Contributions were written into both `T` and `−T` as they were generated, so this
        // only removes accumulated rounding; a large residual would mean a term went into one
        // translation and not its partner, which would make `H(k)` non-Hermitian.
        let residual = b.symmetrize();
        debug_assert!(
            residual < 1.0e-9,
            "H(T) is not the transpose of H(-T): residual {residual:.3e}"
        );
        debug_assert!(
            {
                let gamma = b.gamma_sum();
                (0..nao).all(|i| (0..nao).all(|j| (gamma[(i, j)] - h[(i, j)]).abs() < 1.0e-9))
            },
            "the translation-resolved blocks do not sum to the Γ-point core Hamiltonian"
        );
    }

    Ok(CoreHamiltonian {
        h_core: h,
        pairs,
        ewald,
        bloch,
    })
}
