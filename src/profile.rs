// SPDX-License-Identifier: GPL-3.0-or-later

//! A staged wall-clock timer, off unless `PM7_PROFILE` is set.
//!
//! The point is to answer "where did the time go" with a number instead of an argument. Reading
//! the code and picking the loop that *looks* expensive is how you end up optimizing a stage that
//! was 3 % of the run — the 102-atom analytic Hessian spends its time in one of four places and
//! they are not equally obvious from the source.
//!
//! ```text
//! PM7_PROFILE=1 cargo test --release --test perf_report -- --ignored --nocapture
//! ```
//!
//! Timings are **wall clock summed over threads**, so a stage running on 16 cores reports roughly
//! 16× its elapsed time. That is the right thing for finding where the work is; it is the wrong
//! thing for predicting speedup, and the report says so.
//!
//! When disabled, [`stage`] returns `None` and every call site is a null check the optimizer
//! removes. Nothing is allocated and no lock is taken.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PM7_PROFILE").is_some())
}

type Table = Mutex<BTreeMap<&'static str, (u128, u64)>>;

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// A running stage. Dropping it adds the elapsed time to that stage's total.
pub struct Stage {
    name: &'static str,
    start: Instant,
}

impl Drop for Stage {
    fn drop(&mut self) {
        let nanos = self.start.elapsed().as_nanos();
        if let Ok(mut map) = table().lock() {
            let entry = map.entry(self.name).or_insert((0, 0));
            entry.0 += nanos;
            entry.1 += 1;
        }
    }
}

/// Start timing a stage, or return `None` when profiling is off.
///
/// ```ignore
/// let _t = profile::stage("fock");
/// ```
#[inline]
pub fn stage(name: &'static str) -> Option<Stage> {
    if enabled() {
        Some(Stage {
            name,
            start: Instant::now(),
        })
    } else {
        None
    }
}

/// The accumulated report, or `None` if nothing was recorded.
pub fn report() -> Option<String> {
    let map = table().lock().ok()?;
    if map.is_empty() {
        return None;
    }
    let total: u128 = map.values().map(|(n, _)| *n).sum();
    let mut rows: Vec<(&str, u128, u64)> = map.iter().map(|(k, (n, c))| (*k, *n, *c)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let width = rows.iter().map(|r| r.0.len()).max().unwrap_or(5).max(5);
    let mut out = String::from(
        "\nPM7 stage profile (thread-seconds: a stage on 16 cores reports about 16x its\n\
         elapsed time, so read the ranking, not the wall clock)\n\n",
    );
    out.push_str(&format!(
        "{:width$}  {:>12}  {:>8}  {:>10}\n",
        "stage",
        "seconds",
        "share",
        "calls",
        width = width
    ));
    out.push_str(&format!(
        "{}  {}  {}  {}\n",
        "-".repeat(width),
        "-".repeat(12),
        "-".repeat(8),
        "-".repeat(10)
    ));
    for (name, nanos, calls) in rows {
        out.push_str(&format!(
            "{:width$}  {:>12.4}  {:>7.1}%  {:>10}\n",
            name,
            nanos as f64 * 1e-9,
            100.0 * nanos as f64 / total as f64,
            calls,
            width = width
        ));
    }
    Some(out)
}

/// Print [`report`] to stderr and clear the accumulator.
///
/// Clearing matters when several measurements run in one process: without it the second
/// measurement reports the first one's work as well, which is the kind of number that sends you
/// optimizing the wrong stage.
pub fn report_and_reset() {
    if let Some(text) = report() {
        eprintln!("{text}");
    }
    if let Ok(mut map) = table().lock() {
        map.clear();
    }
}
