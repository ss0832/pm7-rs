// SPDX-License-Identifier: GPL-3.0-or-later
//! Exhaustive element-pair sweep of the empirical element-pair machinery (core-core `alpb`/`xfac`
//! scaling + the special-case O-H/C-H/N-H/C-C/Si-O terms, the missing-pair diagonal-average
//! completion, and the two-center two-electron integrals). Every supported PM7 element pair is
//! exercised without needing a physical SCF, so the whole combinatorial space is covered cheaply.
//! Catches: non-finite (NaN/Inf) energies/integrals, panics, asymmetric pair lookups, and wrong
//! analytic core-core derivatives in the special-case scaling.

use pm7_rs::constants::ANGSTROM_TO_BOHR;
use pm7_rs::integrals::pair_two_electron_g;
use pm7_rs::math::Vec3;
use pm7_rs::params::Pm7Parameters;
use pm7_rs::repulsion::{pair_core_energy, pair_core_energy_and_dr};
use pm7_rs::Pm7Method;

fn sorted_elements(params: &Pm7Parameters) -> Vec<u8> {
    let mut zs: Vec<u8> = params.elements.keys().copied().collect();
    zs.sort_unstable();
    zs
}

fn at(r_bohr: f64) -> (Vec3, Vec3) {
    (Vec3::zero(), Vec3::new(r_bohr, 0.0, 0.0))
}

/// Core-core energy and its analytic derivative, for every ordered element pair, at a spread of
/// distances (including the special-case Si-O centre at 2.9 Å and unphysically short/long ends
/// that stress the `r^-12` guard and the exp tails). No SCF required.
#[test]
fn all_element_pairs_core_core_finite_and_gradient_consistent() {
    let methods = [Pm7Method::Pm7, Pm7Method::Pm7Ts, Pm7Method::Pm7Sparkle];
    let finite_r = [0.5, 0.8, 1.2, 1.6, 2.2, 2.9, 3.6, 5.0, 8.0];
    let grad_r = [1.0, 1.5, 2.0, 2.9, 3.5];
    let mut failures: Vec<String> = Vec::new();

    for method in methods {
        let params = Pm7Parameters::method(method).unwrap();
        let zs = sorted_elements(&params);
        for (i, &a) in zs.iter().enumerate() {
            for &b in &zs[i..] {
                // Pair table must be symmetric (order-independent).
                let pab = params.pair(a, b);
                let pba = params.pair(b, a);
                if pab != pba {
                    failures.push(format!(
                        "{method}: pair({a},{b})={pab:?} != pair({b},{a})={pba:?}"
                    ));
                }
                // Finiteness of the core-core energy across distances, both orderings.
                for &ra in &finite_r {
                    let r = ra * ANGSTROM_TO_BOHR;
                    let (p0, p1) = at(r);
                    for (za, zb, x0, x1) in [(a, b, p0, p1), (b, a, p0, p1)] {
                        match pair_core_energy(za, zb, x0, x1, &params) {
                            Ok(v) if v.is_finite() => {}
                            Ok(v) => failures.push(format!(
                                "{method}: core-core Z{za}-Z{zb} r={ra}Å non-finite: {v}"
                            )),
                            Err(e) => failures.push(format!(
                                "{method}: core-core Z{za}-Z{zb} r={ra}Å errored: {e}"
                            )),
                        }
                    }
                }
                // Analytic dE/dr vs central finite difference (catches special-case derivative bugs).
                for &ra in &grad_r {
                    let r = ra * ANGSTROM_TO_BOHR;
                    let (p0, p1) = at(r);
                    let (_e, dedr) = match pair_core_energy_and_dr(a, b, p0, p1, &params) {
                        Ok(v) => v,
                        Err(e) => {
                            failures.push(format!("{method}: dr Z{a}-Z{b} r={ra}Å errored: {e}"));
                            continue;
                        }
                    };
                    let h = 1.0e-6;
                    let ep =
                        pair_core_energy(a, b, Vec3::zero(), Vec3::new(r + h, 0.0, 0.0), &params)
                            .unwrap();
                    let em =
                        pair_core_energy(a, b, Vec3::zero(), Vec3::new(r - h, 0.0, 0.0), &params)
                            .unwrap();
                    let fd = (ep - em) / (2.0 * h);
                    if !dedr.is_finite() {
                        failures.push(format!(
                            "{method}: core-core dE/dr Z{a}-Z{b} r={ra}Å non-finite: {dedr}"
                        ));
                        continue;
                    }
                    let scale = fd.abs().max(dedr.abs()).max(1.0);
                    if (dedr - fd).abs() / scale > 1.0e-5 {
                        failures.push(format!(
                            "{method}: core-core dE/dr Z{a}-Z{b} r={ra}Å analytic {dedr:.6e} vs FD {fd:.6e}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} element-pair core-core failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Two-center two-electron integrals (`w`, `e1b`, `e2a`) must be finite for every element pair
/// (both go through the sp kernel or the MNDO/d 9×9 kernel). Sparkles (0 AOs) are skipped for the
/// electronic integrals but covered by the core-core test above.
#[test]
fn all_element_pairs_two_electron_integrals_finite() {
    let methods = [Pm7Method::Pm7, Pm7Method::Pm7Ts, Pm7Method::Pm7Sparkle];
    let dist_r = [0.8, 1.5, 2.5, 4.0];
    let mut failures: Vec<String> = Vec::new();

    for method in methods {
        let params = Pm7Parameters::method(method).unwrap();
        let zs = sorted_elements(&params);
        for &a in &zs {
            for &b in &zs {
                let ea = params.element(a).unwrap();
                let eb = params.element(b).unwrap();
                if ea.n_orb == 0 || eb.n_orb == 0 {
                    continue; // sparkle: no atomic orbitals
                }
                // Match SCF ordering: the heavier (more-AO) atom is the first index.
                let (ei, ej, za, zb) = if ea.n_orb >= eb.n_orb {
                    (ea, eb, a, b)
                } else {
                    (eb, ea, b, a)
                };
                for &ra in &dist_r {
                    let r = ra * ANGSTROM_TO_BOHR;
                    let te = pair_two_electron_g::<f64>(ei, ej, [r, 0.0, 0.0]);
                    let mut bad = false;
                    for &v in &te.w {
                        bad |= !v.is_finite();
                    }
                    for row in &te.e1b {
                        for &v in row {
                            bad |= !v.is_finite();
                        }
                    }
                    for row in &te.e2a {
                        for &v in row {
                            bad |= !v.is_finite();
                        }
                    }
                    if bad {
                        failures.push(format!(
                            "{method}: two-electron Z{za}-Z{zb} r={ra}Å has a non-finite integral"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} element-pair two-electron failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
