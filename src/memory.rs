// SPDX-License-Identifier: GPL-3.0-or-later

//! Pre-flight memory estimation and an optional budget guard.
//!
//! Dense NDDO working sets grow as `O(n_basis²)` and the analytic-Hessian CPHF stacks as
//! `O(n_atoms · n_basis²)`; for a large enough system these can exhaust RAM and the process is
//! OOM-killed mid-run with no diagnostic. This module estimates the peak resident bytes *before*
//! the large allocations and, if a budget is in effect, returns a clean
//! [`Pm7Error::InsufficientMemory`] instead.
//!
//! The budget is resolved as: `Pm7Options::max_memory_mb` if set, else the `PM7_MEM_BUDGET_MB`
//! environment variable, else 80 % of currently available physical RAM (Windows/Linux), else no
//! guard. Using available rather than installed RAM leaves room for the OS, the linker, Python,
//! and other processes while the estimate itself remains deliberately conservative.

use crate::error::{Pm7Error, Result};

/// Bytes of one dense `n_basis × n_basis` f64 matrix.
#[inline]
fn matrix_bytes(n_basis: usize) -> u64 {
    (n_basis as u64)
        .saturating_mul(n_basis as u64)
        .saturating_mul(8)
}

/// Estimated peak resident bytes for a single-point SCF (`want_hessian = false`) or an analytic
/// Hessian (`want_hessian = true`). Deliberately an over-estimate.
pub fn estimate_peak_bytes(
    n_basis: usize,
    n_pairs: usize,
    n_atoms: usize,
    has_d: bool,
    want_hessian: bool,
) -> u64 {
    let mat = matrix_bytes(n_basis);
    // Per-pair two-electron cache: sp ≈ 2.3 KB, spd (45×45 `w` + 9×9 e1b/e2a) ≈ 18 KB.
    let per_pair: u64 = if has_d { 18_000 } else { 2_400 };
    let integral_cache = (n_pairs as u64).saturating_mul(per_pair);
    // SCF working set: h_core, density, Fock, MO coeffs, up to ~24 DIIS matrices, temporaries.
    let scf = mat.saturating_mul(30).saturating_add(integral_cache);
    let hess = if want_hessian {
        let threads = rayon::current_num_threads().max(1) as u64;
        // CPHF ov-block stacks (~6·n_atoms·n_occ·n_vir ≲ 6·n_atoms·nao²/4), the transient
        // per-thread skeleton Focks (~3·threads·nao², ×2 for UHF), and the 3N×3N Hessian.
        let ov_stacks = (6 * n_atoms as u64).saturating_mul(mat) / 4;
        // A CPHF worker can transiently own response, Fock, projection, DIIS and matmul buffers.
        // Ten matrices per worker is conservative for both RHF and UHF without scaling with DOF.
        let skeleton = (10 * threads).saturating_mul(mat);
        let dense_hess = (3 * n_atoms as u64)
            .saturating_mul(3 * n_atoms as u64)
            .saturating_mul(8);
        ov_stacks
            .saturating_add(skeleton)
            .saturating_add(dense_hess)
    } else {
        0
    };
    scf.saturating_add(hess)
}

/// Resolve the memory budget in bytes, or `None` for "no guard".
pub fn budget_bytes(explicit_mb: Option<usize>) -> Option<u64> {
    if let Some(mb) = explicit_mb {
        return Some((mb as u64).saturating_mul(1 << 20));
    }
    if let Ok(s) = std::env::var("PM7_MEM_BUDGET_MB") {
        if let Ok(mb) = s.trim().parse::<u64>() {
            return Some(mb.saturating_mul(1 << 20));
        }
    }
    // Default: 80 % of memory available *now*. Unknown platforms retain the previous no-guard
    // behaviour; callers that require a hard limit should set max_memory_mb explicitly.
    available_memory_bytes().map(|available| available / 5 * 4)
}

/// Maximum number of independent memory-heavy tasks that may run concurrently under the
/// current budget. At least one task is returned so its own pre-flight guard can report a useful
/// error when even a single task does not fit.
pub fn parallel_task_limit(
    per_task_bytes: u64,
    task_count: usize,
    explicit_mb: Option<usize>,
) -> usize {
    if task_count == 0 {
        return 0;
    }
    let rayon_limit = rayon::current_num_threads().max(1);
    let memory_limit = budget_bytes(explicit_mb)
        .map(|budget| (budget / per_task_bytes.max(1)).max(1) as usize)
        .unwrap_or(rayon_limit);
    task_count.min(rayon_limit).min(memory_limit).max(1)
}

/// Guard: if a budget is in effect and the estimate exceeds it, return a clean error.
pub fn guard(
    n_basis: usize,
    n_pairs: usize,
    n_atoms: usize,
    has_d: bool,
    want_hessian: bool,
    explicit_mb: Option<usize>,
) -> Result<()> {
    if let Some(budget) = budget_bytes(explicit_mb) {
        let need = estimate_peak_bytes(n_basis, n_pairs, n_atoms, has_d, want_hessian);
        if need > budget {
            return Err(Pm7Error::InsufficientMemory {
                needed_mb: need.saturating_add((1 << 20) - 1) >> 20,
                budget_mb: budget >> 20,
                what: if want_hessian {
                    "analytic Hessian"
                } else {
                    "SCF"
                },
            });
        }
    }
    Ok(())
}

/// Best-effort total physical RAM in bytes (`None` if it cannot be determined).
#[cfg(windows)]
fn available_memory_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    unsafe {
        let mut status: MemoryStatusEx = std::mem::zeroed();
        status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut status) != 0 && status.avail_phys > 0 {
            Some(status.avail_phys)
        } else {
            None
        }
    }
}

/// Best-effort currently available physical RAM in bytes from `/proc/meminfo`.
#[cfg(target_os = "linux")]
fn available_memory_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(not(any(windows, target_os = "linux")))]
fn available_memory_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_grows_quadratically_in_basis() {
        // Isolate the dense working set (no integral cache) so the O(n_basis²) growth is clean.
        let small = estimate_peak_bytes(100, 0, 50, false, false);
        let big = estimate_peak_bytes(200, 0, 50, false, false);
        // Doubling n_basis quadruples the dense working set.
        assert_eq!(big, small * 4);
    }

    #[test]
    fn explicit_budget_takes_priority_and_guards() {
        // 1 MB budget vs a large system → must error.
        let err = guard(2000, 100_000, 500, true, true, Some(1));
        assert!(matches!(err, Err(Pm7Error::InsufficientMemory { .. })));
        // A generous budget passes.
        assert!(guard(50, 100, 10, false, false, Some(100_000)).is_ok());
    }

    #[test]
    fn parallel_task_limit_respects_memory_budget() {
        let mib = 1_u64 << 20;
        let limited = parallel_task_limit(40 * mib, 20, Some(100));
        assert!((1..=2).contains(&limited));
        assert_eq!(parallel_task_limit(200 * mib, 20, Some(100)), 1);
        assert_eq!(parallel_task_limit(40 * mib, 0, Some(100)), 0);
    }
}
