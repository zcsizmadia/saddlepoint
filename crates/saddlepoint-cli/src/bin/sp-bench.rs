//! `sp-bench [options] FILES...` solves each MPS file with each selected
//! solver and writes one CSV row per (instance, solver) to stdout, and to
//! `--out FILE` as it goes, so a long run keeps its finished rows.
//!
//! Solvers: `cpu` and `gpu` are saddlepoint's backends; their residuals come
//! from the independent KKT checker run on the returned solution. `cupdlpx`
//! runs the driver in `bench/cupdlpx` (path given by `--cupdlpx`) without
//! presolve, and its residuals are the ones it reports.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use saddlepoint_core::{Backend, LpProblem, SolverParams, solve};
use saddlepoint_cpu::CpuBackend;
use saddlepoint_cuda::CudaBackend;

const USAGE: &str = "usage: sp-bench [--solvers cpu,gpu,cupdlpx] [--tol T] [--time-limit SECONDS] [--device N]
                [--cupdlpx PATH] [--out FILE.csv] FILES...";

const HEADER: &str =
    "instance,rows,cols,nnz,solver,status,iterations,solve_s,total_s,objective,rel_primal,rel_dual,rel_gap";

struct Options {
    solvers: Vec<String>,
    tol: f64,
    time_limit: f64,
    device: usize,
    cupdlpx: Option<String>,
    out: Option<String>,
    files: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn parse_args() -> Result<Options, String> {
    let mut o = Options {
        solvers: vec!["cpu".into(), "gpu".into()],
        tol: 1e-4,
        time_limit: 600.0,
        device: 0,
        cupdlpx: None,
        out: None,
        files: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--solvers" => o.solvers = value()?.split(',').map(|s| s.trim().to_string()).collect(),
            "--tol" => o.tol = parse(&value()?)?,
            "--time-limit" => o.time_limit = parse(&value()?)?,
            "--device" => o.device = parse(&value()?)?,
            "--cupdlpx" => o.cupdlpx = Some(value()?),
            "--out" => o.out = Some(value()?),
            "-h" | "--help" => return Err(USAGE.into()),
            _ if !a.starts_with('-') => o.files.push(a),
            _ => return Err(format!("unexpected argument {a}\n{USAGE}")),
        }
    }
    if o.files.is_empty() {
        return Err(USAGE.into());
    }
    for s in &o.solvers {
        match s.as_str() {
            "cpu" | "gpu" => {}
            "cupdlpx" if o.cupdlpx.is_none() => return Err("--solvers cupdlpx needs --cupdlpx PATH".into()),
            "cupdlpx" => {}
            _ => return Err(format!("unknown solver {s}")),
        }
    }
    Ok(o)
}

fn run() -> Result<(), String> {
    let o = parse_args()?;
    let gpu = if o.solvers.iter().any(|s| s == "gpu") {
        Some(CudaBackend::new(o.device).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let cpu = CpuBackend::new();
    let mut out = match &o.out {
        Some(path) => {
            let fresh = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
            let mut f = OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{path}: {e}"))?;
            if fresh {
                writeln!(f, "{HEADER}").map_err(|e| e.to_string())?;
            }
            Some(f)
        }
        None => None,
    };
    println!("{HEADER}");
    let params = SolverParams {
        tolerance: o.tol,
        time_limit: Some(Duration::from_secs_f64(o.time_limit)),
        log_every_checks: 0,
        ..Default::default()
    };

    for file in &o.files {
        let problem = saddlepoint_io::read_mps_file(file).map_err(|e| format!("{file}: {e}"))?;
        let size = format!("{},{},{},{}", problem.name, problem.nrows(), problem.ncols(), problem.a.nnz());
        for solver in &o.solvers {
            eprintln!("{} on {solver} ...", problem.name);
            let row = match solver.as_str() {
                "cpu" => ours(&cpu, &problem, &params),
                "gpu" => ours(gpu.as_ref().expect("created above"), &problem, &params),
                _ => cupdlpx(o.cupdlpx.as_deref().expect("checked"), file, &o),
            };
            let line = format!("{size},{solver},{row}");
            println!("{line}");
            if let Some(f) = out.as_mut() {
                emit(f, &line)?;
            }
        }
    }
    Ok(())
}

fn emit(f: &mut File, line: &str) -> Result<(), String> {
    writeln!(f, "{line}").and_then(|_| f.flush()).map_err(|e| e.to_string())
}

/// `status,iterations,solve_s,total_s,objective,rel_primal,rel_dual,rel_gap`
fn ours<B: Backend>(backend: &B, problem: &LpProblem, params: &SolverParams) -> String {
    let t = Instant::now();
    match solve(backend, problem, params) {
        Ok(s) => {
            let total = t.elapsed();
            let r = saddlepoint_check::check(problem, &s.x, &s.y);
            format!(
                "{:?},{},{:.3},{:.3},{:.12e},{:.3e},{:.3e},{:.3e}",
                s.status,
                s.iterations,
                s.solve_time.as_secs_f64(),
                total.as_secs_f64(),
                s.objective,
                r.relative_primal,
                r.relative_dual,
                r.relative_gap
            )
        }
        Err(e) => format!("Error({}),,,,,,,", e.to_string().replace(',', ";")),
    }
}

/// Runs the cuPDLPx driver and maps its `RESULT` line onto the same columns.
/// `total_s` is the process wall time, which includes reading the file and
/// creating the CUDA context.
fn cupdlpx(exe: &str, file: &str, o: &Options) -> String {
    let tol = o.tol.to_string();
    let result = std::env::temp_dir().join(format!("sp-bench-cupdlpx-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&result);
    let t = Instant::now();
    let output = Command::new(exe)
        .args(["--no_presolve", "--eps_opt", &tol, "--eps_feas", &tol, "--time_limit", &o.time_limit.to_string()])
        .arg("--result")
        .arg(&result)
        .arg(file)
        .output();
    let total = t.elapsed().as_secs_f64();
    let output = match output {
        Ok(out) => out,
        Err(e) => return format!("Error({}),,,,,,,", e.to_string().replace(',', ";")),
    };
    let text = std::fs::read_to_string(&result).unwrap_or_default();
    let _ = std::fs::remove_file(&result);
    let Some(line) = text.lines().find(|l| l.starts_with("RESULT ")) else {
        return format!("Error(no result; {}),,,,,,,", output.status);
    };
    let field = |key: &str| {
        line.split_whitespace().find_map(|kv| kv.strip_prefix(key).and_then(|v| v.strip_prefix('='))).unwrap_or("")
    };
    let status = match field("status") {
        "OPTIMAL" => "Optimal",
        "TIME_LIMIT" => "TimeLimit",
        "ITERATION_LIMIT" => "IterationLimit",
        "PRIMAL_INFEASIBLE" => "PrimalInfeasible",
        "DUAL_INFEASIBLE" => "DualInfeasible",
        other => other,
    };
    format!(
        "{status},{},{},{total:.3},{},{},{},{}",
        field("iters"),
        field("time"),
        field("primal_obj"),
        field("rel_primal"),
        field("rel_dual"),
        field("rel_gap")
    )
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad value {s}"))
}
