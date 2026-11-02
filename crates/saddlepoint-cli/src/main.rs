//! `saddlepoint FILE.mps [options]` solves an LP on the CPU backend and
//! validates the answer with the independent KKT checker.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use saddlepoint_core::{SolverParams, Status, solve};
use saddlepoint_cpu::CpuBackend;

const USAGE: &str = "usage: saddlepoint FILE.mps [--tol T] [--max-iter N] [--time-limit SECONDS] [--log N] [--quiet]";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut params = SolverParams { log_every_checks: 10, ..Default::default() };
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        match a.as_str() {
            "--tol" => params.tolerance = parse(&value("--tol")?)?,
            "--max-iter" => params.max_iterations = parse(&value("--max-iter")?)?,
            "--time-limit" => params.time_limit = Some(Duration::from_secs_f64(parse(&value("--time-limit")?)?)),
            "--log" => params.log_every_checks = parse(&value("--log")?)?,
            "--quiet" => params.log_every_checks = 0,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            _ if file.is_none() && !a.starts_with('-') => file = Some(a),
            _ => return Err(format!("unexpected argument {a}\n{USAGE}")),
        }
    }
    let file = file.ok_or(USAGE)?;

    let t = Instant::now();
    let problem = saddlepoint_io::read_mps_file(&file).map_err(|e| format!("{file}: {e}"))?;
    let read_time = t.elapsed();
    let backend = CpuBackend::new();
    println!(
        "{}: {} rows, {} cols, {} nonzeros (read in {:.3}s), backend {}",
        problem.name,
        problem.nrows(),
        problem.ncols(),
        problem.a.nnz(),
        read_time.as_secs_f64(),
        saddlepoint_core::Backend::name(&backend)
    );
    if params.log_every_checks > 0 {
        eprintln!(
            "{:>9} {:>14} {:>14} {:>9} {:>9} {:>9} {:>9} {:>5} {:>9}",
            "iter", "primal obj", "dual obj", "p.res", "d.res", "gap", "weight", "rst", "time"
        );
    }
    let s = solve(&backend, &problem, &params).map_err(|e| e.to_string())?;
    println!(
        "status {:?}, objective {:.10e}, {} iterations, {} restarts, {:.3}s",
        s.status,
        s.objective,
        s.iterations,
        s.restarts,
        s.solve_time.as_secs_f64()
    );
    if s.status == Status::Optimal {
        let r = saddlepoint_check::check(&problem, &s.x, &s.y);
        println!(
            "check: relative primal {:.2e}, dual {:.2e}, gap {:.2e}",
            r.relative_primal, r.relative_dual, r.relative_gap
        );
    }
    Ok(if s.status == Status::Optimal { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad value {s}"))
}
