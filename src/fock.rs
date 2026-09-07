// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO Fock-matrix build, spin-resolved: `F^σ = H_core + J(P_tot) − K(P^σ)`.
//!
//! The Coulomb part `J` is built from the **total** density (both spins); the exchange
//! part `K` from the **same-spin** density. The RHF (closed-shell) Fock is the special case
//! `P^σ = ½ P_tot`, i.e. `F = H_core + J(P) − K(½P)`. The one-center block uses the exact
//! one-center two-electron integrals ([`oc_two_electron`]); the two-center block uses the
//! rotated integrals from [`crate::integrals`].

use crate::basis::Basis;
use crate::error::Result;
use crate::hamiltonian::CoreHamiltonian;
use crate::linalg::Matrix;
use crate::params::Pm7Parameters;
use crate::system::Molecule;

/// Lower-triangle pair index `pack(a,b)` within a 9-orbital atom block (0-based).
#[inline]
fn pack9(a: usize, b: usize) -> usize {
    let (h, l) = if a >= b { (a, b) } else { (b, a) };
    h * (h + 1) / 2 + l
}

/// `PACK[a][b] = pack9(a, b)`, precomputed.
///
/// The two-centre contraction evaluates this on every one of its `n_a² n_b²` innermost
/// iterations, and each evaluation is a compare, a swap and a multiply that the branch predictor
/// has no pattern to learn. A 81-entry table is a single load from L1.
const PACK: [[usize; 9]; 9] = {
    let mut table = [[0usize; 9]; 9];
    let mut a = 0;
    while a < 9 {
        let mut b = 0;
        while b < 9 {
            let (h, l) = if a >= b { (a, b) } else { (b, a) };
            table[a][b] = h * (h + 1) / 2 + l;
            b += 1;
        }
        a += 1;
    }
    table
};

// Three optimizations of the two-centre loop below were implemented, measured, and **removed**.
// They are recorded here because each looks obviously right on paper, and the measurement is the
// only thing that says otherwise:
//
// * **Gathering each pair's density sub-blocks into contiguous scratch**, so the strided reads
//   through a full-width matrix happen once instead of `n_a²` times. 5.2 s → 9.8 s on the 102-atom
//   Hessian: the blocks are typically 4×4, so zeroing three 81-element arrays per pair costs more
//   stores than the gather saves loads.
// * **Packing the Coulomb contraction onto lower-triangle indices.** `J_A[μν]` is symmetric in
//   both index pairs, so `n_a² n_b²` steps become `npair_a × npair_b` — 100 instead of 256 for an
//   sp/sp pair. Strictly less arithmetic, and still 4.39 s against 4.17 s, because building the
//   packed density vectors is two heap allocations per pair per build and there are tens of
//   millions of those.
// * **Batching the response Fock across degrees of freedom**, so one pass over the pairs serves
//   many densities. See `hessian::cphf_ov`.
//
// The common thread: at NDDO block sizes the loop is not memory bound and not arithmetic bound —
// it is bound by per-pair overhead, so anything that adds a fixed per-pair cost loses even when it
// removes work from the inner loop.

/// One-center two-electron integral `(a b | c d)` (all orbitals on the same atom), from the
/// PM7 one-center parameters. Orbital indices: 0 = s, 1..3 = p. Uses the NDDO index
/// symmetries `(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)`.
#[inline]
pub fn oc_two_electron(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    gss: f64,
    gsp: f64,
    gpp: f64,
    gp2: f64,
    hsp: f64,
) -> f64 {
    // Diagonal-pair cases: bra = (x,x), ket = (y,y).
    if a == b && c == d {
        return match (a == 0, c == 0) {
            (true, true) => gss,  // (ss|ss)
            (true, false) => gsp, // (ss|pp)
            (false, true) => gsp, // (pp|ss)
            (false, false) => {
                if a == c {
                    gpp // (pp|pp)
                } else {
                    gp2 // (pp|p'p')
                }
            }
        };
    }
    // Off-diagonal-pair cases: sort bra/ket index pairs.
    let (ba, bb) = (a.min(b), a.max(b));
    let (kc, kd) = (c.min(d), c.max(d));
    // (s p_i | s p_i) = H_sp
    if ba == 0 && bb != 0 && kc == 0 && kd != 0 && bb == kd {
        return hsp;
    }
    // (p_i p_j | p_i p_j) = ½(G_pp − G_p2),  i ≠ j
    if ba != 0 && bb != 0 && ba != bb && ba == kc && bb == kd {
        return 0.5 * (gpp - gp2);
    }
    0.0
}

/// Smooth long-range **exchange** switch: `1` for `r ≤ inner`, `0` for `r ≥ outer`, and a
/// C²-continuous smootherstep (`6t⁵ − 15t⁴ + 10t³`) in between. `inner`/`outer` are distances
/// in Bohr; `r2` is the squared interatomic distance. The two-center exchange contribution
/// `K(μ_a, λ_b)` is weighted by the same-spin density `P(ν_a, σ_b)` *between* the two atoms, which
/// decays with separation, so scaling it by this switch (and skipping it entirely once the switch
/// hits 0) is a controlled, smooth approximation — used only when an exchange cutoff is requested.
#[inline]
fn exchange_switch(r2: f64, inner: f64, outer: f64) -> f64 {
    if r2 <= inner * inner {
        return 1.0;
    }
    if r2 >= outer * outer {
        return 0.0;
    }
    let t = (r2.sqrt() - inner) / (outer - inner);
    1.0 - t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Build the spin-σ Fock matrix `F = H_core + J(p_tot) − K(p_spin)`.
///
/// The one-center and two-center electron-repulsion contractions are computed in parallel
/// (rayon) into small per-atom / per-pair contribution blocks, then scattered into `F`
/// serially in the original order. Because the scatter order and each block's inner
/// accumulation order are unchanged, the result is **bit-identical** to a serial build.
pub fn build_fock_spin(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Result<Matrix> {
    build_fock_spin_x(molecule, basis, params, core, p_tot, p_spin, None)
}

/// [`build_fock_spin`] with an optional smooth long-range-exchange cutoff `(inner, outer)` in
/// Bohr. `exchange_cutoff = None` takes the exact path and is **bit-identical** to
/// [`build_fock_spin`]; `Some((inner, outer))` weights each atom pair's exchange block by
/// the internal `exchange_switch` and skips it entirely beyond `outer`. Coulomb (J) is never cut. Intended
/// for the analytic-Hessian CPHF, whose per-DOF response Fock builds dominate the cost.
pub fn build_fock_spin_x(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
    p_spin: &Matrix,
    exchange_cutoff: Option<(f64, f64)>,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let mut f = core.h_core.clone();

    // One-center (intra-atomic) contributions — each atom's n×n block, computed in parallel.
    let oc_blocks: Result<Vec<(usize, usize, Vec<f64>)>> = molecule
        .atoms
        .par_iter()
        .enumerate()
        .map(|(ia, atom)| {
            let elem = params.element(atom.z)?;
            let off = basis.atom_offset[ia];
            let n = basis.atom_norb[ia];
            let mut block = vec![0.0; n * n];
            if n == 9 {
                // d-bearing atoms use the full spd one-center integral matrix (MNDO/d).
                let oc = &elem
                    .dshell
                    .as_ref()
                    .expect("d element has DShell")
                    .onecenter;
                let g = |a: usize, b: usize, c: usize, d: usize| oc[pack9(a, b) * 45 + pack9(c, d)];
                for mu in 0..9 {
                    for nu in 0..9 {
                        let mut acc = 0.0;
                        for la in 0..9 {
                            for si in 0..9 {
                                acc += p_tot[(off + la, off + si)] * g(mu, nu, la, si);
                                acc -= p_spin[(off + la, off + si)] * g(mu, la, nu, si);
                            }
                        }
                        block[mu * n + nu] = acc;
                    }
                }
            } else {
                let (gss, gsp, gpp, gp2, hsp) =
                    (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
                for mu in 0..n {
                    for nu in 0..n {
                        let mut acc = 0.0;
                        for la in 0..n {
                            for si in 0..n {
                                acc += p_tot[(off + la, off + si)]
                                    * oc_two_electron(mu, nu, la, si, gss, gsp, gpp, gp2, hsp);
                                acc -= p_spin[(off + la, off + si)]
                                    * oc_two_electron(mu, la, nu, si, gss, gsp, gpp, gp2, hsp);
                            }
                        }
                        block[mu * n + nu] = acc;
                    }
                }
            }
            Ok((off, n, block))
        })
        .collect();
    for (off, n, block) in oc_blocks? {
        for mu in 0..n {
            for nu in 0..n {
                f[(off + mu, off + nu)] += block[mu * n + nu];
            }
        }
    }

    // Two-center (inter-atomic) contributions — each pair's Coulomb and exchange blocks,
    // computed in parallel. `jaa`/`jbb` are the a-/b-diagonal Coulomb blocks; `kab` the
    // exchange block for the (a,b) off-diagonal (added symmetrically on scatter).
    type TwoCenterBlock = (usize, usize, usize, usize, Vec<f64>, Vec<f64>, Vec<f64>);
    // Bound transient J/K block storage. Previously all O(N_atoms^2) pair contributions stayed
    // live until the scatter; batching keeps peak memory roughly constant while preserving the
    // exact serial scatter order and pair-level Rayon parallelism.
    const PAIR_BATCH_SIZE: usize = 2_048;
    for pair_batch in core.pairs.chunks(PAIR_BATCH_SIZE) {
        let tc: Vec<TwoCenterBlock> = pair_batch
            .par_iter()
            .map(|pair| {
                let te = &pair.te;
                let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
                // Use the BASIS orbital counts, not `te.norb_*`: a Sparkle (0 AOs) reports `norb = 1`
                // in the integral block (a fictitious monopole s used only for the electron-core `e1b`,
                // which lives in H_core). Iterating the two-electron J/K over that fake orbital would
                // read density at the Sparkle's offset — which, having no AOs, aliases the next atom's
                // orbitals — corrupting the Fock. For non-Sparkle pairs these counts are identical, so
                // this is bit-identical there.
                let (na, nb) = (basis.atom_norb[pair.a], basis.atom_norb[pair.b]);
                let mut jaa = vec![0.0; na * na];
                let mut jbb = vec![0.0; nb * nb];
                let mut kab = vec![0.0; na * nb];
                // Mulliken populations of the two atoms, needed only to remove the monopole
                // part of the Coulomb integrals when an Ewald sum is re-supplying it. For a
                // molecule `v_point` is zero and these are multiplied by nothing.
                let (pop_a, pop_b) = if pair.v_point == 0.0 {
                    (0.0, 0.0)
                } else {
                    (
                        (0..na).map(|k| p_tot[(oa + k, oa + k)]).sum::<f64>(),
                        (0..nb).map(|k| p_tot[(ob + k, ob + k)]).sum::<f64>(),
                    )
                };
                let stride = te.npair_j;
                let w = &te.w;

                for mu in 0..na {
                    for nu in 0..na {
                        let row = PACK[mu][nu] * stride;
                        let mut acc = 0.0;
                        for la in 0..nb {
                            for si in 0..nb {
                                acc += p_tot[(ob + la, ob + si)] * w[row + PACK[la][si]];
                            }
                        }
                        if mu == nu {
                            acc -= pair.v_point * pop_b;
                        }
                        jaa[mu * na + nu] = acc;
                    }
                }
                for la in 0..nb {
                    for si in 0..nb {
                        let col = PACK[la][si];
                        let mut acc = 0.0;
                        for mu in 0..na {
                            for nu in 0..na {
                                acc += p_tot[(oa + mu, oa + nu)] * w[PACK[mu][nu] * stride + col];
                            }
                        }
                        if la == si {
                            acc -= pair.v_point * pop_a;
                        }
                        jbb[la * nb + si] = acc;
                    }
                }
                // Long-range-exchange scale (1.0 when no cutoff → bit-identical; 0.0 skips the block).
                let scale = match exchange_cutoff {
                    None => 1.0,
                    Some((inner, outer)) => {
                        let d = molecule.atoms[pair.a].position - molecule.atoms[pair.b].position;
                        exchange_switch(d.norm2(), inner, outer)
                    }
                };
                if scale != 0.0 {
                    for mu in 0..na {
                        for la in 0..nb {
                            let mut acc = 0.0;
                            for nu in 0..na {
                                let row = PACK[mu][nu] * stride;
                                for si in 0..nb {
                                    acc += p_spin[(oa + nu, ob + si)] * w[row + PACK[la][si]];
                                }
                            }
                            // The monopole part of the exchange is summed over the whole lattice
                            // by the Ewald term below, so it is removed here too — exactly as it
                            // is for Coulomb. `(μν|λσ) → δ_μν δ_λσ v` picks out ν = μ, σ = λ.
                            acc -= p_spin[(oa + mu, ob + la)] * pair.v_point;
                            kab[mu * nb + la] = -acc * scale;
                        }
                    }
                }
                (oa, ob, na, nb, jaa, jbb, kab)
            })
            .collect();
        for (oa, ob, na, nb, jaa, jbb, kab) in tc {
            for mu in 0..na {
                for nu in 0..na {
                    f[(oa + mu, oa + nu)] += jaa[mu * na + nu];
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    f[(ob + la, ob + si)] += jbb[la * nb + si];
                }
            }
            // h_core (and thus f) is symmetric here, so adding kab to both transposed
            // positions preserves the symmetry the serial `f[b,a] = f[a,b]` produced.
            for mu in 0..na {
                for la in 0..nb {
                    let v = kab[mu * nb + la];
                    f[(oa + mu, ob + la)] += v;
                    f[(ob + la, oa + mu)] += v;
                }
            }
        }
    }

    if let Some(ew) = &core.ewald {
        // Long-range Coulomb. `E_ew = ½ qᵀ M q` with `q_A = Z_A − P_A`, so `∂E_ew/∂P_μμ = −V_A`
        // for every orbital μ on atom A and nothing off-diagonal. The accompanying energy
        // bookkeeping lives in `run_pm7` — the ½-trace formula does not cover a term that is
        // quadratic but not *homogeneous* in the density.
        let charges = ew.charges(basis, p_tot);
        let potential = ew.potential(&charges);
        for (ia, v) in potential.iter().enumerate() {
            let off = basis.atom_offset[ia];
            for mu in 0..basis.atom_norb[ia] {
                f[(off + mu, off + mu)] -= v;
            }
        }

        // Long-range exchange. Beyond the feather range the two-electron integral is exactly
        // `δ_μν δ_λσ v`, so the exchange operator becomes `−P^σ(μ_A, λ_B) Σ'_T v(r_AB + T)` —
        // the *same* conditionally convergent lattice sum the Coulomb term needs, and
        // regularized the same way by `M_AB`.
        //
        // The alternative — taking only the nearest image, as MOPAC does — is right for the
        // large Born–von Kármán supercells MOPAC's `makpol` builds, but wrong for a primitive
        // cell: a boron in a BN sheet bonds to three equivalent nitrogens that live in three
        // *different* translations, so picking one breaks the symmetry and produces a spurious
        // force of over 1 eV/Å.
        //
        // The lattice sum used is `M̃_AB = M_AB − M_self`, the Γ-point exchange divergence
        // correction — see `hamiltonian::exchange_potential` for why the bare `M_AB` would leave
        // a spurious term decaying only as 1/L.
        //
        // This term is homogeneous quadratic in the density, so unlike the Coulomb one it needs
        // no correction to the ½-trace energy.
        // Parallel over the row atom. This was the one serial, unbatched loop left in an
        // otherwise rayon-parallel Fock build, and it runs on **every** Fock build.
        //
        // Splitting by row atom is safe and **bit-identical**: atom `ia` writes only rows
        // `oa..oa+na`, and within those rows each output element `(oa+mu, ob+la)` belongs to
        // exactly one `ib`. So every element is written exactly once, by one task — there is no
        // reduction whose order a thread count could change.
        use rayon::prelude::*;
        let n_atoms = basis.atom_offset.len();
        let self_potential = ew.self_potential();
        let nao = basis.nao;

        // One mutable row block per atom, carved out in order.
        let mut rest = f.as_mut_slice();
        let mut blocks: Vec<&mut [f64]> = Vec::with_capacity(n_atoms);
        let mut consumed = 0usize;
        for ia in 0..n_atoms {
            let start = basis.atom_offset[ia] * nao;
            let end = start + basis.atom_norb[ia] * nao;
            let (skip, tail) = rest.split_at_mut(start - consumed);
            debug_assert!(skip.iter().all(|_| true));
            let (block, tail) = tail.split_at_mut(end - start);
            blocks.push(block);
            rest = tail;
            consumed = end;
        }

        blocks.par_iter_mut().enumerate().for_each(|(ia, rows)| {
            let oa = basis.atom_offset[ia];
            let na = basis.atom_norb[ia];
            for ib in 0..n_atoms {
                let m = ew.matrix[ia][ib] - self_potential;
                if m == 0.0 {
                    continue;
                }
                let ob = basis.atom_offset[ib];
                let nb = basis.atom_norb[ib];
                for mu in 0..na {
                    for la in 0..nb {
                        rows[mu * nao + ob + la] -= p_spin[(oa + mu, ob + la)] * m;
                    }
                }
            }
        });
    }

    Ok(f)
}

/// Translation-resolved Fock blocks `F(T)` for a k-point calculation.
///
/// Only the **exchange** carries the translation index. Everything else is diagonal in `T`:
///
/// * the one-centre integrals are on-site;
/// * Coulomb contracts each atom's *own-cell* density block, which is `P(0)` by translational
///   invariance, and sums the integrals over images;
/// * the long-range Ewald terms are built from Mulliken charges and from `P(0)`.
///
/// So `J` and the long-range terms go into `F(0)` and `K` into `F(T)`. Summing over `T`
/// reproduces the Γ-point Fock exactly, which is what makes `KMesh::grid(1,1,1)` agree with
/// `KMesh::Gamma`.
pub fn build_fock_spin_bloch(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    p_tot: &crate::scf_pbc::BlochBlocks,
    p_spin: &crate::scf_pbc::BlochBlocks,
) -> Result<crate::scf_pbc::BlochBlocks> {
    let core_bloch = core.bloch.as_ref().ok_or_else(|| {
        crate::error::Pm7Error::InvalidInput(
            "a k-point Fock build needs the translation-resolved core Hamiltonian".into(),
        )
    })?;
    let mut f = core_bloch.clone();
    let zero = p_tot
        .position([0, 0, 0])
        .expect("density blocks contain the zero translation");
    let p_tot0 = p_tot.block(zero);
    let p_spin0 = p_spin.block(zero);

    // One-centre (intra-atomic) contributions — identical to the Γ path, on the T = 0 block.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let block = one_centre_block(elem, off, n, p_tot0, p_spin0);
        let target = f.at_mut([0, 0, 0]);
        for mu in 0..n {
            for nu in 0..n {
                target[(off + mu, off + nu)] += block[mu * n + nu];
            }
        }
    }

    // Two-centre contributions, over the very pairs the core Hamiltonian was built from — each
    // one already carrying the translation its integrals were evaluated at.
    for pair in &core.pairs {
        let te = &pair.te;
        let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
        let (na, nb) = (basis.atom_norb[pair.a], basis.atom_norb[pair.b]);
        let t = pair.t;
        let neg_t = [-t[0], -t[1], -t[2]];

        let (pop_a, pop_b) = if pair.v_point == 0.0 {
            (0.0, 0.0)
        } else {
            (
                (0..na).map(|k| p_tot0[(oa + k, oa + k)]).sum::<f64>(),
                (0..nb).map(|k| p_tot0[(ob + k, ob + k)]).sum::<f64>(),
            )
        };

        // Coulomb: on-site density blocks, into F(0).
        {
            let target = f.at_mut([0, 0, 0]);
            for mu in 0..na {
                for nu in 0..na {
                    let mut acc = 0.0;
                    for la in 0..nb {
                        for si in 0..nb {
                            acc += p_tot0[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    if mu == nu {
                        acc -= pair.v_point * pop_b;
                    }
                    target[(oa + mu, oa + nu)] += acc;
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    let mut acc = 0.0;
                    for mu in 0..na {
                        for nu in 0..na {
                            acc += p_tot0[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    if la == si {
                        acc -= pair.v_point * pop_a;
                    }
                    target[(ob + la, ob + si)] += acc;
                }
            }
        }

        // Exchange: the density block **at this translation**, into F(T) and its transpose in
        // F(−T). This is the whole reason the k-point path exists — at Γ every P(T) is the same
        // matrix and the distinction collapses.
        let p_spin_t = p_spin.block(
            p_spin
                .position(t)
                .expect("density blocks cover the core's translations"),
        );
        let mut kab = vec![0.0; na * nb];
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        acc += p_spin_t[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si);
                    }
                }
                acc -= p_spin_t[(oa + mu, ob + la)] * pair.v_point;
                kab[mu * nb + la] = -acc;
            }
        }
        for mu in 0..na {
            for la in 0..nb {
                let v = kab[mu * nb + la];
                f.at_mut(t)[(oa + mu, ob + la)] += v;
                f.at_mut(neg_t)[(ob + la, oa + mu)] += v;
            }
        }
    }

    if let Some(ew) = &core.ewald {
        // Long-range Coulomb: built from Mulliken charges, which are `P(0)` quantities, so it
        // lands entirely in `F(0)`.
        let charges = ew.charges(basis, p_tot0);
        let potential = ew.potential(&charges);
        {
            let target = f.at_mut([0, 0, 0]);
            for (ia, v) in potential.iter().enumerate() {
                let off = basis.atom_offset[ia];
                for mu in 0..basis.atom_norb[ia] {
                    target[(off + mu, off + mu)] -= v;
                }
            }
        }

        // Long-range exchange, resolved by Born–von Kármán residue class. Each class carries
        // its own lattice sum over the *supercell*, and its own density block, so the whole
        // term disperses with k instead of being frozen at `T = 0`.
        let n_atoms = basis.atom_offset.len();
        let bvk = ew.exchange.as_ref().ok_or_else(|| {
            crate::error::Pm7Error::InvalidInput(
                "a k-point Fock build needs the Born-von Karman exchange sums".into(),
            )
        })?;
        for (class, t) in bvk.translations.iter().enumerate() {
            let values = &bvk.values[class];
            let index = p_spin
                .position(*t)
                .expect("density blocks cover the BvK residue classes");
            let p_t = p_spin.block(index).clone();
            let target = f.at_mut(*t);
            for ia in 0..n_atoms {
                let (oa, na) = (basis.atom_offset[ia], basis.atom_norb[ia]);
                for ib in 0..n_atoms {
                    let m = values[ia][ib];
                    if m == 0.0 {
                        continue;
                    }
                    let (ob, nb) = (basis.atom_offset[ib], basis.atom_norb[ib]);
                    for mu in 0..na {
                        for la in 0..nb {
                            target[(oa + mu, ob + la)] -= p_t[(oa + mu, ob + la)] * m;
                        }
                    }
                }
            }
        }
    }

    Ok(f)
}

/// The one-centre `J − K` block of a single atom, shared by the Γ and k-point Fock builds.
pub(crate) fn one_centre_block(
    elem: &crate::params::Pm7Element,
    off: usize,
    n: usize,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Vec<f64> {
    let mut block = vec![0.0; n * n];
    if n == 9 {
        let oc = &elem
            .dshell
            .as_ref()
            .expect("d element has DShell")
            .onecenter;
        let g = |a: usize, b: usize, c: usize, d: usize| oc[pack9(a, b) * 45 + pack9(c, d)];
        for mu in 0..9 {
            for nu in 0..9 {
                let mut acc = 0.0;
                for la in 0..9 {
                    for si in 0..9 {
                        acc += p_tot[(off + la, off + si)] * g(mu, nu, la, si);
                        acc -= p_spin[(off + la, off + si)] * g(mu, la, nu, si);
                    }
                }
                block[mu * n + nu] = acc;
            }
        }
    } else {
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += p_tot[(off + la, off + si)]
                            * oc_two_electron(mu, nu, la, si, gss, gsp, gpp, gp2, hsp);
                        acc -= p_spin[(off + la, off + si)]
                            * oc_two_electron(mu, la, nu, si, gss, gsp, gpp, gp2, hsp);
                    }
                }
                block[mu * n + nu] = acc;
            }
        }
    }
    block
}

/// RHF (closed-shell) Fock: `F = H_core + J(P) − K(½P)`.
pub fn build_fock(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
) -> Result<Matrix> {
    build_fock_x(molecule, basis, params, core, density, None)
}

// A batched response Fock -- one pass over the atom pairs contracting many densities against
// each integral block -- was implemented here and **removed**. The premise was that the pair loop
// is memory bound; measured, it is not at the sizes a dense CPHF can reach: 102 atoms is about
// 4 MB of integral blocks, which stays in L3 across calls. See `hessian::cphf_ov`.

/// [`build_fock`] with an optional smooth long-range-exchange cutoff `(inner, outer)` in Bohr.
/// `None` is bit-identical to [`build_fock`]. See [`build_fock_spin_x`].
pub fn build_fock_x(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm7Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
    exchange_cutoff: Option<(f64, f64)>,
) -> Result<Matrix> {
    let mut half = density.clone();
    for v in half.as_mut_slice() {
        *v *= 0.5;
    }
    build_fock_spin_x(
        molecule,
        basis,
        params,
        core,
        density,
        &half,
        exchange_cutoff,
    )
}
