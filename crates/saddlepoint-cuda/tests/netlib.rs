//! GPU and CPU agree on Netlib: for each instance both backends must reach
//! the same status, the GPU answer must pass the independent KKT check, and
//! the two objectives must agree to the tolerance. Skips without a GPU.
//!
//! The instance lists mirror `saddlepoint-cpu/tests/netlib.rs`.

use std::path::PathBuf;
use std::time::Duration;

use saddlepoint_core::{Solution, SolverParams, Status, solve};
use saddlepoint_cpu::CpuBackend;
use saddlepoint_cuda::CudaBackend;

const TOL: f64 = 1e-6;

const QUICK: &[&str] = &[
    "afiro", "adlittle", "israel", "sctest", "shell", "etamacro", "standata", "standgub", "standmps", "forest6",
    "woodinfe",
];

const SLOW: &[&str] = &["25fv47", "80bau3b", "scrs8", "stair", "perold"];

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/netlib")
}

fn fmt(s: &Solution) -> String {
    format!("{:?} in {} iterations, {:.2?}", s.status, s.iterations, s.solve_time)
}

fn run(gpu: &CudaBackend, name: &str) -> Result<(Duration, Duration), String> {
    let p = saddlepoint_io::read_mps_file(data_dir().join(format!("{name}.mps"))).map_err(|e| e.to_string())?;
    let params = SolverParams { tolerance: TOL, ..Default::default() };
    let c = solve(&CpuBackend::new(), &p, &params).map_err(|e| format!("{name} cpu: {e}"))?;
    let g = solve(gpu, &p, &params).map_err(|e| format!("{name} gpu: {e}"))?;
    eprintln!("{name:>10}: cpu {}; gpu {}", fmt(&c), fmt(&g));
    if c.status != g.status {
        return Err(format!("{name}: cpu {}, gpu {}", fmt(&c), fmt(&g)));
    }
    if g.status == Status::Optimal {
        saddlepoint_check::check(&p, &g.x, &g.y).verify(2.0 * TOL).map_err(|e| format!("{name} gpu: {e}"))?;
        // Both are within TOL of the optimum, so they can differ by twice
        // that, relative to the objective's magnitude.
        let err = (c.objective - g.objective).abs() / (1.0 + c.objective.abs() + g.objective.abs());
        if err > 2.0 * TOL {
            return Err(format!("{name}: objective cpu {} vs gpu {} (relative {err:.2e})", c.objective, g.objective));
        }
    }
    Ok((c.solve_time, g.solve_time))
}

fn run_all(names: &[&str]) {
    let gpu = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let results: Vec<_> = names.iter().map(|n| run(&gpu, n)).collect();
    let failures: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    assert!(failures.is_empty(), "{}", failures.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"));
}

#[test]
fn netlib_quick_gpu_matches_cpu() {
    run_all(QUICK);
}

#[test]
#[ignore = "slow; run with --release -- --ignored"]
fn netlib_slow_gpu_matches_cpu() {
    run_all(SLOW);
}
