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
                for mu in 0..na {
                    for nu in 0..na {
                        let mut acc = 0.0;
                        for la in 0..nb {
                            for si in 0..nb {
                                acc += p_tot[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                            }
                        }
                        jaa[mu * na + nu] = acc;
                    }
                }
                for la in 0..nb {
                    for si in 0..nb {
                        let mut acc = 0.0;
                        for mu in 0..na {
                            for nu in 0..na {
                                acc += p_tot[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                            }
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
                                for si in 0..nb {
                                    acc += p_spin[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si);
                                }
                            }
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

    Ok(f)
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
