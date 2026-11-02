//! Netlib regression: solve each instance in `data/netlib`, run the answer
//! through the independent KKT checker, and compare the objective with the
//! HiGHS value in `data/netlib/reference.csv` (see scripts/highs_reference.py).

use std::path::PathBuf;

use saddlepoint_core::{SolverParams, Status, solve};
use saddlepoint_cpu::CpuBackend;

const TOL: f64 = 1e-6;

/// Fast enough for every `cargo test`.
const QUICK: &[&str] = &[
    "afiro", "adlittle", "israel", "sctest", "shell", "etamacro", "standata", "standgub", "standmps", "forest6",
    "woodinfe",
];

/// Slower; run with `cargo test --release -- --ignored`.
const SLOW: &[&str] = &["25fv47", "80bau3b", "scrs8", "stair"];

/// Not yet solved within the iteration limit (Phase 2 work: PID primal
/// weight, better restarts, infeasibility detection on klein1).
const KNOWN_HARD: &[&str] = &["greenbea", "perold", "klein1"];

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/netlib")
}

fn reference(name: &str) -> (String, Option<f64>) {
    let csv = std::fs::read_to_string(data_dir().join("reference.csv")).expect("reference.csv");
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == name {
            return (f[1].to_string(), f.get(2).and_then(|s| s.parse().ok()));
        }
    }
    panic!("{name} missing from reference.csv");
}

fn run(name: &str) -> Result<(), String> {
    let p = saddlepoint_io::read_mps_file(data_dir().join(format!("{name}.mps"))).map_err(|e| e.to_string())?;
    let params = SolverParams { tolerance: TOL, ..Default::default() };
    let s = solve(&CpuBackend::new(), &p, &params).map_err(|e| e.to_string())?;
    let (status, objective) = reference(name);
    match status.as_str() {
        "optimal" => {
            if s.status != Status::Optimal {
                return Err(format!("{name}: status {:?} after {} iterations", s.status, s.iterations));
            }
            saddlepoint_check::check(&p, &s.x, &s.y).verify(2.0 * TOL).map_err(|e| format!("{name}: {e}"))?;
            let r = objective.unwrap();
            let err = (s.objective - r).abs() / (1.0 + r.abs());
            if err > 1e-5 {
                return Err(format!("{name}: objective {} vs HiGHS {r} (relative {err:.2e})", s.objective));
            }
        }
        "infeasible" if s.status != Status::PrimalInfeasible => {
            return Err(format!("{name}: expected PrimalInfeasible, got {:?}", s.status));
        }
        _ => {}
    }
    Ok(())
}

fn run_all(names: &[&str]) {
    let failures: Vec<String> = names.iter().filter_map(|n| run(n).err()).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn netlib_quick() {
    run_all(QUICK);
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn netlib_slow() {
    run_all(SLOW);
}

#[test]
fn every_instance_is_classified() {
    let mut listed: Vec<&str> = QUICK.iter().chain(SLOW).chain(KNOWN_HARD).copied().collect();
    listed.sort_unstable();
    let mut on_disk: Vec<String> = std::fs::read_dir(data_dir())
        .unwrap()
        .filter_map(|e| e.ok()?.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
        .filter(|s| s != "reference")
        .collect();
    on_disk.sort_unstable();
    assert_eq!(listed, on_disk.iter().map(String::as_str).collect::<Vec<_>>());
}
