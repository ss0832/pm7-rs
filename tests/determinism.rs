// SPDX-License-Identifier: GPL-3.0-or-later
//! Results must not depend on how many threads happen to be available.
//!
//! `docs/theory.md` states this as a guarantee — "work is split across threads but each result
//! accumulates in the original order, so the thread count never changes the numbers" — and until
//! now nothing checked it. That is exactly the sort of promise that stays true until one
//! `par_iter().sum()` slips in, and then fails only on a machine with a different core count,
//! which is to say on someone else's machine.
//!
//! Each test runs the same calculation inside rayon pools of 1, 2 and 7 threads and compares
//! **bit patterns**, not tolerances: an order-dependent reduction shows up in the last ulp long
//! before it shows up anywhere a tolerance would catch it.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::math::Vec3;
use pm7_rs::{
    analytic_hessian, analytic_stress, born_and_dielectric, closed_form_gradient,
    dynamical_matrix_dfpt, run_dandc, run_pm7, Atom, Cell, DandcOptions, DfptOptions, KMesh,
    Molecule, PbcOptions, Pm7Options, Pm7Parameters,
};

fn at(z: u8, x: f64, y: f64, z_: f64) -> Atom {
    Atom {
        z,
        position: Vec3::new(
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z_ * ANGSTROM_TO_BOHR,
        ),
    }
}

/// Run `f` inside a rayon pool of exactly `threads` workers.
fn with_threads<T: Send>(threads: usize, f: impl Fn() -> T + Sync + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build a rayon pool")
        .install(f)
}

/// A hydrogen-bonded water chain: exercises the EH+ correction, which is where the many-body
/// parallel work lives, alongside the ordinary pair loops.
fn water_wire(n: usize) -> Molecule {
    let mut atoms = Vec::new();
    for i in 0..n {
        let base = i as f64 * 2.85;
        let lift = 0.15 * (i % 2) as f64;
        atoms.push(at(8, base, lift, 0.0));
        atoms.push(at(1, base + 0.96, lift + 0.02, 0.0));
        atoms.push(at(1, base - 0.24, lift + 0.90, 0.30));
    }
    Molecule::new(atoms)
}

fn diamond() -> Molecule {
    let a = 3.567 * ANGSTROM_TO_BOHR;
    let cell = Cell::new(&[
        Vec3::new(0.0, a * 0.5, a * 0.5),
        Vec3::new(a * 0.5, 0.0, a * 0.5),
        Vec3::new(a * 0.5, a * 0.5, 0.0),
    ])
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 6,
            position: Vec3::new(a * 0.25, a * 0.25, a * 0.25),
        },
    ])
    .with_cell(cell)
}

fn tight() -> Pm7Options {
    Pm7Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-9,
        max_scf: 500,
        ..Default::default()
    }
}

const THREADS: [usize; 3] = [1, 2, 7];

fn assert_same_bits(label: &str, values: &[f64]) {
    for (i, value) in values.iter().enumerate().skip(1) {
        assert_eq!(
            values[0].to_bits(),
            value.to_bits(),
            "{label}: {} threads gave {value}, 1 thread gave {}",
            THREADS[i],
            values[0]
        );
    }
}

#[test]
fn a_molecular_energy_and_gradient_are_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = water_wire(6);
    let options = tight();

    let mut energies = Vec::new();
    let mut gradients: Vec<Vec<Vec3>> = Vec::new();
    for threads in THREADS {
        let (energy, gradient) = with_threads(threads, || {
            let out = closed_form_gradient(&molecule, &params, &options).unwrap();
            (out.energy_ev, out.gradient.clone())
        });
        energies.push(energy);
        gradients.push(gradient);
    }
    assert_same_bits("total energy", &energies);
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let column: Vec<f64> = gradients.iter().map(|g| g[atom].get(axis)).collect();
            assert_same_bits(&format!("gradient atom {atom} axis {axis}"), &column);
        }
    }
}

/// The analytic Hessian is the most parallel path in the crate: the skeleton pair loop, the
/// derivative-Fock build and the CPHF solve all run on rayon.
#[test]
fn the_analytic_hessian_is_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = water_wire(3);
    let options = tight();

    let hessians: Vec<_> = THREADS
        .iter()
        .map(|&threads| {
            with_threads(threads, || {
                analytic_hessian(&molecule, &params, &options, 1.0e-3).unwrap()
            })
        })
        .collect();
    let n = hessians[0].rows;
    for i in 0..n {
        for j in 0..n {
            let column: Vec<f64> = hessians.iter().map(|h| h[(i, j)]).collect();
            assert_same_bits(&format!("hessian ({i},{j})"), &column);
        }
    }
}

/// The periodic path: the Ewald sums, the k-point diagonalizations, and the long-range exchange
/// loop that was made parallel in 0.2.1.
#[test]
fn a_periodic_k_point_energy_is_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = diamond();
    let mut options = tight();
    options.pbc = Some(PbcOptions {
        kmesh: KMesh::grid(3, 3, 3),
        ..Default::default()
    });

    let energies: Vec<f64> = THREADS
        .iter()
        .map(|&threads| {
            with_threads(threads, || {
                run_pm7(&molecule, &params, &options).unwrap().total_ev
            })
        })
        .collect();
    assert_same_bits("periodic total energy", &energies);
}

/// Divide and conquer: the subsystem solves are parallel, and 0.2.1 hoisted the far field and
/// rewrote the Fermi search, both of which touch the summation structure.
#[test]
fn the_divide_and_conquer_energy_is_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = water_wire(10);
    let options = tight();
    let dandc = DandcOptions {
        buffer: 8.0 * ANGSTROM_TO_BOHR,
        core_size: 6,
        ..Default::default()
    };

    let mut energies = Vec::new();
    let mut fermis = Vec::new();
    for threads in THREADS {
        let out = with_threads(threads, || {
            run_dandc(&molecule, &params, &options, &dandc).unwrap()
        });
        energies.push(out.electronic_ev);
        fermis.push(out.fermi_ev);
    }
    assert_same_bits("divide-and-conquer electronic energy", &energies);
    assert_same_bits("divide-and-conquer Fermi level", &fermis);
}

/// A polar 1-D chain: the cheapest periodic system whose Born charges are not zero by symmetry.
fn hf_chain() -> Molecule {
    let a = ANGSTROM_TO_BOHR;
    Molecule::new(vec![at(9, 0.0, 0.0, 0.0), at(1, 0.95, 0.0, 0.0)])
        .with_cell(Cell::new(&[Vec3::new(3.4 * a, 0.0, 0.0)]).unwrap())
}

fn periodic(mesh: KMesh) -> Pm7Options {
    Pm7Options {
        pbc: Some(PbcOptions {
            kmesh: mesh,
            ..Default::default()
        }),
        ..tight()
    }
}

/// The response solver, at a wavevector that is neither the zone centre nor the zone boundary.
///
/// This is the least protected path in the crate against an order-dependent reduction: the CPHF
/// iteration sums over k points, over lattice translations and over perturbations, and 0.2.2
/// rewrote all three of those loops — the pair loop lost its per-element hash lookups, the bare
/// term was hoisted out of the iteration, and the solver itself became a conjugate gradient. Each
/// of those was checked for bit-identity by hand at the time; none of them was checked *by a
/// test*, which is the same as not being checked at all six months from now.
///
/// `q = 1/4` is commensurate with the 4-point mesh, so the k+q set is the k set and the run
/// exercises the general complex-phase path rather than a real special case.
#[test]
fn the_dfpt_response_is_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hf_chain();
    let options = periodic(KMesh::grid(4, 1, 1));

    let matrices: Vec<_> = THREADS
        .iter()
        .map(|&threads| {
            with_threads(threads, || {
                dynamical_matrix_dfpt(
                    &molecule,
                    &params,
                    &options,
                    [0.25, 0.0, 0.0],
                    &DfptOptions::default(),
                )
                .unwrap()
                .force_constants
            })
        })
        .collect();

    let n = matrices[0].n;
    for i in 0..n * n {
        let re: Vec<f64> = matrices.iter().map(|m| m.re[i]).collect();
        let im: Vec<f64> = matrices.iter().map(|m| m.im[i]).collect();
        assert_same_bits(&format!("force constant {i} (real)"), &re);
        assert_same_bits(&format!("force constant {i} (imaginary)"), &im);
    }
}

/// The field response: three perturbations that are not atomic displacements, contracted against
/// the dipole operator rather than against a derivative Fock matrix.
#[test]
fn born_charges_and_the_dielectric_tensor_are_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = hf_chain();
    let options = periodic(KMesh::grid(4, 1, 1));

    let fields: Vec<_> = THREADS
        .iter()
        .map(|&threads| {
            with_threads(threads, || {
                born_and_dielectric(&molecule, &params, &options, &DfptOptions::default()).unwrap()
            })
        })
        .collect();

    for atom in 0..molecule.atoms.len() {
        for i in 0..3 {
            for j in 0..3 {
                let column: Vec<f64> = fields.iter().map(|f| f.born[atom].get(i, j)).collect();
                assert_same_bits(&format!("Born charge atom {atom} ({i},{j})"), &column);
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let column: Vec<f64> = fields.iter().map(|f| f.polarizability.get(i, j)).collect();
            assert_same_bits(&format!("polarizability ({i},{j})"), &column);
        }
    }
}

/// The periodic gradient and stress, on a mesh with more than one k point.
///
/// These share the long-range exchange derivative loop, which is the largest remaining `O(mesh²)`
/// structure in the crate and therefore the next thing to be rewritten. A rewrite that reorders
/// its sums is allowed to move the last digits of the published numbers; it is not allowed to make
/// the answer depend on the core count of whoever runs it. That distinction only survives if it is
/// pinned before the rewrite, not after.
#[test]
fn the_periodic_gradient_and_stress_are_thread_count_independent() {
    let params = Pm7Parameters::standard().unwrap();
    let molecule = diamond();
    let options = periodic(KMesh::grid(3, 3, 3));

    let mut gradients: Vec<Vec<Vec3>> = Vec::new();
    let mut stresses = Vec::new();
    for threads in THREADS {
        let (gradient, stress) = with_threads(threads, || {
            let g = closed_form_gradient(&molecule, &params, &options).unwrap();
            let s = analytic_stress(&molecule, &params, &options, &g.scf).unwrap();
            (g.gradient.clone(), s.stress)
        });
        gradients.push(gradient);
        stresses.push(stress);
    }

    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let column: Vec<f64> = gradients.iter().map(|g| g[atom].get(axis)).collect();
            assert_same_bits(
                &format!("periodic gradient atom {atom} axis {axis}"),
                &column,
            );
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let column: Vec<f64> = stresses.iter().map(|s| s.get(i, j)).collect();
            assert_same_bits(&format!("stress ({i},{j})"), &column);
        }
    }
}
