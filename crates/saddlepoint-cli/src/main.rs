//! `saddlepoint FILE.mps [options]` solves an LP on the CPU or a CUDA GPU
//! and validates the answer with the independent KKT checker.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use saddlepoint_core::{Backend, LpProblem, Solution, SolverParams, Status, solve};
use saddlepoint_cpu::CpuBackend;
use saddlepoint_cuda::CudaBackend;

const USAGE: &str = "usage: saddlepoint FILE.mps [--tol T] [--max-iter N] [--time-limit SECONDS] [--log N] [--quiet]
                   [--gpu] [--device N]
       tuning: --check-every N --reflection R --necessary B --sufficient B --artificial B
               --kp K --ki K --kd K --geo-passes N --ruiz N";

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
    let mut device: Option<usize> = None;
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        match a.as_str() {
            "--tol" => params.tolerance = parse(&value("--tol")?)?,
            "--max-iter" => params.max_iterations = parse(&value("--max-iter")?)?,
            "--time-limit" => params.time_limit = Some(Duration::from_secs_f64(parse(&value("--time-limit")?)?)),
            "--log" => params.log_every_checks = parse(&value("--log")?)?,
            "--check-every" => params.check_every = parse(&value(&a)?)?,
            "--reflection" => params.reflection = parse(&value(&a)?)?,
            "--necessary" => params.restart_necessary = parse(&value(&a)?)?,
            "--sufficient" => params.restart_sufficient = parse(&value(&a)?)?,
            "--artificial" => params.restart_artificial = parse(&value(&a)?)?,
            "--kp" => params.pid_kp = parse(&value(&a)?)?,
            "--ki" => params.pid_ki = parse(&value(&a)?)?,
            "--kd" => params.pid_kd = parse(&value(&a)?)?,
            "--geo-passes" => params.scaling.geometric_mean_passes = parse(&value(&a)?)?,
            "--ruiz" => params.scaling.ruiz_iterations = parse(&value(&a)?)?,
            "--quiet" => params.log_every_checks = 0,
            "--gpu" => device = Some(device.unwrap_or(0)),
            "--device" => device = Some(parse(&value(&a)?)?),
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
    println!(
        "{}: {} rows, {} cols, {} nonzeros (read in {:.3}s)",
        problem.name,
        problem.nrows(),
        problem.ncols(),
        problem.a.nnz(),
        read_time.as_secs_f64(),
    );
    let s = match device {
        Some(d) => run_on(&CudaBackend::new(d).map_err(|e| e.to_string())?, &problem, &params)?,
        None => run_on(&CpuBackend::new(), &problem, &params)?,
    };
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

fn run_on<B: Backend>(backend: &B, problem: &LpProblem, params: &SolverParams) -> Result<Solution, String> {
    println!("backend {}", backend.name());
    if params.log_every_checks > 0 {
        eprintln!(
            "{:>9} {:>14} {:>14} {:>9} {:>9} {:>9} {:>9} {:>5} {:>9} {:>7} {:>9}",
            "iter", "primal obj", "dual obj", "p.res", "d.res", "gap", "weight", "rst", "time", "epoch", "r/r0"
        );
    }
    solve(backend, problem, params).map_err(|e| e.to_string())
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad value {s}"))
}
