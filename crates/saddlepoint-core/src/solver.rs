//! Restarted Halpern PDHG with reflection, after cuPDLPx.
//!
//! One iteration on the scaled problem, with `z = (x, y)`:
//!
//! ```text
//! x_t = proj_X(x - τ (c - Aᵀ y))
//! y_t = y - σ (A(2 x_t - x) - proj_C(A(2 x_t - x) - y / σ))
//! z   = w ((1 + ρ) z_t - ρ z) + (1 - w) z_anchor,      w = (k + 1) / (k + 2)
//! ```
//!
//! Every `check_every` iterations the loop computes the KKT residuals of
//! `z_t` in the original space, tests for infeasibility, and decides whether
//! to restart. A restart moves the anchor to `z_t`, resets `k` and updates the
//! primal weight `ω` (with `τ = η / ω`, `σ = η ω`).

use std::time::{Duration, Instant};

use thiserror::Error;

use crate::backend::Backend;
use crate::problem::{LpProblem, ProblemError};
use crate::scaling::{ScaledProblem, ScalingParams, finite_bound_norm, norm2};

#[derive(Clone, Debug)]
pub struct SolverParams {
    /// Relative KKT tolerance for primal residual, dual residual and gap.
    pub tolerance: f64,
    /// Tolerance on the normalised violation of an infeasibility certificate.
    pub infeasibility_tolerance: f64,
    pub max_iterations: u64,
    pub time_limit: Option<Duration>,
    /// Iterations between termination and restart checks.
    pub check_every: u64,
    /// Step size is `step_size_factor / ‖A‖₂`.
    pub step_size_factor: f64,
    /// Reflection coefficient ρ in `[0, 1]`; 0 is plain Halpern.
    pub reflection: f64,
    pub restart_sufficient: f64,
    pub restart_necessary: f64,
    pub restart_artificial: f64,
    /// Smoothing θ of the primal weight update (1 = no memory).
    pub primal_weight_smoothing: f64,
    pub power_iterations: usize,
    pub scaling: ScalingParams,
    /// Print a progress line every this many checks (0 = silent).
    pub log_every_checks: u64,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            infeasibility_tolerance: 1e-8,
            max_iterations: 1_000_000,
            time_limit: None,
            check_every: 64,
            step_size_factor: 0.998,
            reflection: 1.0,
            restart_sufficient: 0.2,
            restart_necessary: 0.8,
            restart_artificial: 0.36,
            primal_weight_smoothing: 0.5,
            power_iterations: 200,
            scaling: ScalingParams::default(),
            log_every_checks: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    IterationLimit,
    TimeLimit,
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub status: Status,
    /// Primal solution in the original space.
    pub x: Vec<f64>,
    /// Dual solution in the original space, for the minimisation form
    /// (`y > 0` on an active lower row bound).
    pub y: Vec<f64>,
    /// Objective in the original sense (maximised value for max problems).
    pub objective: f64,
    pub dual_objective: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub gap: f64,
    pub iterations: u64,
    pub restarts: u64,
    pub solve_time: Duration,
}

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("invalid problem: {0}")]
    Problem(#[from] ProblemError),
    #[error("backend error: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync>),
    #[error("numerical breakdown: {0}")]
    Numerical(&'static str),
}

/// KKT measurements of one candidate point, in the original space.
#[derive(Clone, Copy, Debug)]
struct Kkt {
    primal_obj: f64,
    dual_obj: f64,
    primal_res: f64,
    dual_res: f64,
}

impl Kkt {
    fn gap(&self) -> f64 {
        (self.primal_obj - self.dual_obj).abs()
    }

    fn converged(&self, eps: f64, b_norm: f64, c_norm: f64) -> bool {
        self.primal_res <= eps * (1.0 + b_norm)
            && self.dual_res <= eps * (1.0 + c_norm)
            && self.gap() <= eps * (1.0 + self.primal_obj.abs() + self.dual_obj.abs())
    }
}

pub fn solve<B: Backend>(backend: &B, problem: &LpProblem, params: &SolverParams) -> Result<Solution, SolveError> {
    problem.validate()?;
    Solver::new(backend, problem, params)?.run()
}

fn be<E: std::error::Error + Send + Sync + 'static>(e: E) -> SolveError {
    SolveError::Backend(Box::new(e))
}

struct Solver<'a, B: Backend> {
    backend: &'a B,
    problem: &'a LpProblem,
    params: &'a SolverParams,
    scaled: ScaledProblem,
    dev: B::Problem,
    x: B::Vector,
    y: B::Vector,
    x_t: B::Vector,
    y_t: B::Vector,
    x_bar: B::Vector,
    x_anchor: B::Vector,
    y_anchor: B::Vector,
    ax: B::Vector,
    aty: B::Vector,
    tmp_n: B::Vector,
    tmp_m: B::Vector,
}

impl<'a, B: Backend> Solver<'a, B> {
    fn new(backend: &'a B, problem: &'a LpProblem, params: &'a SolverParams) -> Result<Self, SolveError> {
        let scaled = ScaledProblem::new(problem, &params.scaling);
        let dev = backend.load(&scaled).map_err(be)?;
        let (m, n) = (problem.nrows(), problem.ncols());
        let zn = vec![0.0; n];
        let zm = vec![0.0; m];
        let v = |d: &[f64]| backend.upload(d).map_err(be);
        Ok(Self {
            x: v(&zn)?,
            y: v(&zm)?,
            x_t: v(&zn)?,
            y_t: v(&zm)?,
            x_bar: v(&zn)?,
            x_anchor: v(&zn)?,
            y_anchor: v(&zm)?,
            ax: v(&zm)?,
            aty: v(&zn)?,
            tmp_n: v(&zn)?,
            tmp_m: v(&zm)?,
            backend,
            problem,
            params,
            scaled,
            dev,
        })
    }

    /// Largest singular value of the scaled matrix by power iteration on A'ᵀA'.
    fn estimate_norm(&mut self) -> Result<f64, SolveError> {
        let b = self.backend;
        let n = self.problem.ncols();
        if n == 0 || self.scaled.a.nnz() == 0 {
            return Ok(1.0);
        }
        let start: Vec<f64> = (0..n).map(|j| 1.0 + 0.01 * (j % 7) as f64).collect();
        let mut v = b.upload(&start).map_err(be)?;
        let s = 1.0 / b.norm_sq(&v).map_err(be)?.sqrt();
        b.scale(&mut v, s).map_err(be)?;
        let mut lambda = 0.0;
        for _ in 0..self.params.power_iterations {
            b.spmv(&self.dev, &v, &mut self.tmp_m).map_err(be)?;
            b.spmv_t(&self.dev, &self.tmp_m, &mut v).map_err(be)?;
            let norm = b.norm_sq(&v).map_err(be)?.sqrt();
            if norm.is_nan() || norm <= 0.0 || norm.is_infinite() {
                return Err(SolveError::Numerical("power iteration collapsed"));
            }
            b.scale(&mut v, 1.0 / norm).map_err(be)?;
            let done = (norm - lambda).abs() <= 1e-6 * norm;
            lambda = norm;
            if done {
                break;
            }
        }
        // Power iteration underestimates; pad slightly so the step stays safe.
        Ok(lambda.sqrt() * 1.01)
    }

    fn kkt(&mut self) -> Result<Kkt, SolveError> {
        let b = self.backend;
        let f = self.scaled.objective_factor();
        b.spmv(&self.dev, &self.x_t, &mut self.tmp_m).map_err(be)?;
        let ps = b.primal_stats(&self.dev, &self.x_t, &self.tmp_m).map_err(be)?;
        b.spmv_t(&self.dev, &self.y_t, &mut self.tmp_n).map_err(be)?;
        let ds = b.dual_stats(&self.dev, &self.y_t, &self.tmp_n).map_err(be)?;
        Ok(Kkt {
            primal_obj: ps.objective * f + self.problem.c0,
            dual_obj: ds.objective * f + self.problem.c0,
            primal_res: ps.residual_sq.sqrt(),
            dual_res: ds.residual_sq.sqrt(),
        })
    }

    /// Tests `z_t - z_anchor` as an infeasibility certificate.
    fn infeasibility(&mut self) -> Result<Option<Status>, SolveError> {
        let b = self.backend;
        let eps = self.params.infeasibility_tolerance;

        b.sub(&self.y_t, &self.y_anchor, &mut self.tmp_m).map_err(be)?;
        b.spmv_t(&self.dev, &self.tmp_m, &mut self.tmp_n).map_err(be)?;
        let d = b.dual_ray_stats(&self.dev, &self.tmp_m, &self.tmp_n).map_err(be)?;
        if d.objective > 0.0 && d.violation_sq.sqrt() <= eps * d.objective && d.objective > eps * d.norm_sq.sqrt() {
            return Ok(Some(Status::PrimalInfeasible));
        }

        b.sub(&self.x_t, &self.x_anchor, &mut self.tmp_n).map_err(be)?;
        b.spmv(&self.dev, &self.tmp_n, &mut self.tmp_m).map_err(be)?;
        let p = b.primal_ray_stats(&self.dev, &self.tmp_n, &self.tmp_m).map_err(be)?;
        if p.objective < 0.0 && p.violation_sq.sqrt() <= -eps * p.objective && -p.objective > eps * p.norm_sq.sqrt() {
            return Ok(Some(Status::DualInfeasible));
        }
        Ok(None)
    }

    fn run(mut self) -> Result<Solution, SolveError> {
        let start = Instant::now();
        let p = self.params;
        let b = self.backend;

        let eta = p.step_size_factor / self.estimate_norm()?;
        let c_norm_scaled = norm2(&self.scaled.c);
        let b_norm_scaled = finite_bound_norm(&self.scaled.lc, &self.scaled.uc);
        let mut omega =
            if c_norm_scaled > 1e-10 && b_norm_scaled > 1e-10 { c_norm_scaled / b_norm_scaled } else { 1.0 };

        let b_norm = finite_bound_norm(&self.problem.lc, &self.problem.uc);
        let c_norm = norm2(&self.problem.c);

        let mut total: u64 = 0;
        let mut k: u64 = 0;
        let mut restarts: u64 = 0;
        let mut checks: u64 = 0;
        let mut ref_residual: Option<f64> = None;
        let mut prev_residual = f64::INFINITY;

        let (status, kkt) = loop {
            let (tau, sigma) = (eta / omega, eta * omega);
            b.spmv_t(&self.dev, &self.y, &mut self.aty).map_err(be)?;
            b.primal_step(&self.dev, &self.x, &self.aty, tau, &mut self.x_t, &mut self.x_bar).map_err(be)?;
            b.spmv(&self.dev, &self.x_bar, &mut self.ax).map_err(be)?;
            b.dual_step(&self.dev, &self.y, &self.ax, sigma, &mut self.y_t).map_err(be)?;

            total += 1;
            let check = total.is_multiple_of(p.check_every) || total >= p.max_iterations;
            let w = (k + 1) as f64 / (k + 2) as f64;
            let rx = b.halpern_step(&mut self.x, &self.x_t, &self.x_anchor, w, p.reflection, check).map_err(be)?;
            let ry = b.halpern_step(&mut self.y, &self.y_t, &self.y_anchor, w, p.reflection, check).map_err(be)?;
            k += 1;

            if !check {
                continue;
            }
            checks += 1;
            let residual = (omega * rx + ry / omega).sqrt();
            if !residual.is_finite() {
                return Err(SolveError::Numerical("iterates are not finite"));
            }
            let kkt = self.kkt()?;
            if p.log_every_checks > 0 && checks.is_multiple_of(p.log_every_checks) {
                eprintln!(
                    "{total:>9} {:>14.6e} {:>14.6e} {:>9.2e} {:>9.2e} {:>9.2e} {omega:>9.2e} {restarts:>5} {:>8.2}s",
                    kkt.primal_obj,
                    kkt.dual_obj,
                    kkt.primal_res,
                    kkt.dual_res,
                    kkt.gap(),
                    start.elapsed().as_secs_f64()
                );
            }
            if kkt.converged(p.tolerance, b_norm, c_norm) {
                break (Status::Optimal, kkt);
            }
            if let Some(s) = self.infeasibility()? {
                break (s, kkt);
            }
            if total >= p.max_iterations {
                break (Status::IterationLimit, kkt);
            }
            if p.time_limit.is_some_and(|t| start.elapsed() >= t) {
                break (Status::TimeLimit, kkt);
            }

            // Restart decision on the fixed-point residual.
            let r0 = *ref_residual.get_or_insert(residual);
            let restart = residual <= p.restart_sufficient * r0
                || (residual <= p.restart_necessary * r0 && residual > prev_residual)
                || k as f64 >= p.restart_artificial * total as f64;
            prev_residual = residual;
            if restart {
                let dx = b.distance_sq(&self.x_t, &self.x_anchor).map_err(be)?.sqrt();
                let dy = b.distance_sq(&self.y_t, &self.y_anchor).map_err(be)?.sqrt();
                if dx > 1e-10 && dy > 1e-10 {
                    let th = p.primal_weight_smoothing;
                    omega = (th * (dy / dx).ln() + (1.0 - th) * omega.ln()).exp();
                }
                for (src, dst) in [(&self.x_t, &mut self.x_anchor), (&self.y_t, &mut self.y_anchor)] {
                    b.copy(src, dst).map_err(be)?;
                }
                b.copy(&self.x_t, &mut self.x).map_err(be)?;
                b.copy(&self.y_t, &mut self.y).map_err(be)?;
                k = 0;
                restarts += 1;
                ref_residual = None;
                prev_residual = f64::INFINITY;
            }
        };

        let x = self.scaled.unscale_primal(&b.download(&self.x_t).map_err(be)?);
        let y = self.scaled.unscale_dual(&b.download(&self.y_t).map_err(be)?);
        let sign = if self.problem.maximize { -1.0 } else { 1.0 };
        Ok(Solution {
            status,
            x,
            y,
            objective: sign * kkt.primal_obj,
            dual_objective: sign * kkt.dual_obj,
            primal_residual: kkt.primal_res,
            dual_residual: kkt.dual_res,
            gap: kkt.gap(),
            iterations: total,
            restarts,
            solve_time: start.elapsed(),
        })
    }
}
