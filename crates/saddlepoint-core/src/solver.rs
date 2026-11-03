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
    /// PID gains for the primal-weight controller (cuPDLPx defaults).
    pub pid_kp: f64,
    pub pid_ki: f64,
    pub pid_kd: f64,
    /// Discount of the controller's integral term per restart.
    pub pid_discount: f64,
    pub power_iterations: usize,
    pub scaling: ScalingParams,
    /// Print a progress line every this many checks (0 = silent).
    pub log_every_checks: u64,
    pub termination: Termination,
}

/// How the dual side of the termination test is measured. The primal
/// residual is the same in both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Termination {
    /// Dual residual and objective of `y_t` with the reduced cost `c - Aᵀy_t`
    /// projected onto the bounds: the dual objective is a true bound, and the
    /// answer passes the independent KKT check at the same tolerance.
    #[default]
    Projected,
    /// cuPDLPx's measure: the reduced cost `λ = c - Aᵀy + (x_t - x)/τ` implied by
    /// the primal step, with dual objective `λ·x_t` plus the row terms. It stops
    /// sooner (up to about 2x on some MIPLIB LPs), but the answer can miss the projected
    /// gap by several times the tolerance. Use it to compare with cuPDLPx.
    StepImplied,
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
            restart_necessary: 0.5,
            restart_artificial: 0.36,
            pid_kp: 0.99,
            pid_ki: 0.01,
            pid_kd: 0.0,
            pid_discount: 0.3,
            power_iterations: 200,
            scaling: ScalingParams::default(),
            log_every_checks: 0,
            termination: Termination::Projected,
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
    tmp_m2: B::Vector,
    c: B::Vector,
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
            tmp_m2: v(&zm)?,
            c: v(&scaled.c)?,
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

    /// KKT measures of `(x_t, y_t)`. Leaves `A'ᵀ y_t` in `tmp_n`.
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

    /// cuPDLPx's dual objective of `y_t`: the row terms of `y_t` plus `λ·x_t`, with
    /// `λ = c - Aᵀy + (x_t - x)/τ` the reduced cost implied by the primal step.
    /// Must run before the Halpern step, while `x` and `aty` are the step's inputs.
    fn step_dual_objective(&mut self, tau: f64) -> Result<f64, SolveError> {
        let b = self.backend;
        // With `aty = c` the reduced cost is zero, so this is the row terms alone.
        let rows = b.dual_stats(&self.dev, &self.y_t, &self.c).map_err(be)?.objective;
        let dot = |u: &B::Vector, v: &B::Vector| b.dot(u, v).map_err(be);
        let lx = dot(&self.c, &self.x_t)? - dot(&self.aty, &self.x_t)?
            + (dot(&self.x_t, &self.x_t)? - dot(&self.x, &self.x_t)?) / tau;
        Ok((rows + lx) * self.scaled.objective_factor() + self.problem.c0)
    }

    /// Fixed-point residual `‖z - T(z)‖_P` in the PDHG metric
    /// `P = [ω/η I, A'ᵀ; A', 1/(ηω) I]`, where `T(z) = z_t`.
    fn fixed_point_residual(&mut self, eta: f64, omega: f64) -> Result<f64, SolveError> {
        let b = self.backend;
        b.sub(&self.x_t, &self.x, &mut self.tmp_n).map_err(be)?;
        b.sub(&self.y_t, &self.y, &mut self.tmp_m2).map_err(be)?;
        b.spmv(&self.dev, &self.tmp_n, &mut self.tmp_m).map_err(be)?;
        let dx2 = b.norm_sq(&self.tmp_n).map_err(be)?;
        let dy2 = b.norm_sq(&self.tmp_m2).map_err(be)?;
        let cross = b.dot(&self.tmp_m2, &self.tmp_m).map_err(be)?;
        let r2 = omega / eta * dx2 + dy2 / (eta * omega) + 2.0 * cross;
        if !r2.is_finite() {
            return Err(SolveError::Numerical("iterates are not finite"));
        }
        Ok(r2.max(0.0).sqrt())
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
        let b_norm = finite_bound_norm(&self.problem.lc, &self.problem.uc);
        let c_norm = norm2(&self.problem.c);

        // Initial primal weight, as in cuPDLPx: 1 when bounds and objective
        // are rescaled, otherwise the ratio of the unscaled norms.
        let mut omega = if p.scaling.bound_objective_rescaling { 1.0 } else { (1.0 + c_norm) / (1.0 + b_norm) };
        let mut pid = PidState::new(omega);

        let mut total: u64 = 0;
        let mut k: u64 = 0;
        let mut restarts: u64 = 0;
        let mut checks: u64 = 0;
        let mut anchor_residual = f64::INFINITY;
        let mut prev_residual = f64::INFINITY;

        let (status, kkt) = loop {
            let (tau, sigma) = (eta / omega, eta * omega);
            b.spmv_t(&self.dev, &self.y, &mut self.aty).map_err(be)?;
            b.primal_step(&self.dev, &self.x, &self.aty, tau, &mut self.x_t, &mut self.x_bar).map_err(be)?;
            b.spmv(&self.dev, &self.x_bar, &mut self.ax).map_err(be)?;
            b.dual_step(&self.dev, &self.y, &self.ax, sigma, &mut self.y_t).map_err(be)?;

            total += 1;
            let check = total.is_multiple_of(p.check_every) || total >= p.max_iterations;
            // At k = 0 the iterate is the anchor, so this measures r(z_anchor),
            // the reference for the restart criteria.
            let residual = if check || k == 0 { self.fixed_point_residual(eta, omega)? } else { 0.0 };
            if k == 0 {
                anchor_residual = residual;
            }
            // The step-based dual residual needs x from before the Halpern update.
            let measured = if check {
                let kkt = self.kkt()?;
                let step =
                    b.step_dual_residual_sq(&self.dev, &self.x, &self.x_t, &self.aty, &self.tmp_n, tau).map_err(be)?;
                let step_obj = match p.termination {
                    Termination::Projected => None,
                    Termination::StepImplied => Some(self.step_dual_objective(tau)?),
                };
                Some((kkt, step.sqrt(), step_obj))
            } else {
                None
            };
            let w = (k + 1) as f64 / (k + 2) as f64;
            b.halpern_step(&mut self.x, &self.x_t, &self.x_anchor, w, p.reflection).map_err(be)?;
            b.halpern_step(&mut self.y, &self.y_t, &self.y_anchor, w, p.reflection).map_err(be)?;
            k += 1;

            let Some((kkt, step_dual, step_obj)) = measured else { continue };
            checks += 1;
            if p.log_every_checks > 0 && checks.is_multiple_of(p.log_every_checks) {
                eprintln!(
                    "{total:>9} {:>14.6e} {:>14.6e} {:>9.2e} {:>9.2e} {:>9.2e} {omega:>9.2e} {restarts:>5} {:>8.2}s {k:>7} {:>9.2e}",
                    kkt.primal_obj,
                    kkt.dual_obj,
                    kkt.primal_res,
                    kkt.dual_res,
                    kkt.gap(),
                    start.elapsed().as_secs_f64(),
                    residual / anchor_residual
                );
            }
            let test = match step_obj {
                None => kkt,
                Some(dual_obj) => Kkt { dual_obj, dual_res: step_dual, ..kkt },
            };
            if test.converged(p.tolerance, b_norm, c_norm) {
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
            let r0 = anchor_residual;
            let restart = residual <= p.restart_sufficient * r0
                || (residual <= p.restart_necessary * r0 && residual > prev_residual)
                || k as f64 >= p.restart_artificial * total as f64;
            prev_residual = residual;
            if restart {
                let dx = b.distance_sq(&self.x_t, &self.x_anchor).map_err(be)?.sqrt();
                let dy = b.distance_sq(&self.y_t, &self.y_anchor).map_err(be)?.sqrt();
                let rel_primal = kkt.primal_res / (1.0 + b_norm);
                // The PID balances against cuPDLPx's step-based dual residual. The
                // projected one (used for termination) is identically zero when
                // every variable is bounded, which would pin the weight.
                let rel_dual = step_dual / (1.0 + c_norm);
                omega = pid.update(p, omega, dx, dy, rel_primal, rel_dual);
                for (src, dst) in [(&self.x_t, &mut self.x_anchor), (&self.y_t, &mut self.y_anchor)] {
                    b.copy(src, dst).map_err(be)?;
                }
                b.copy(&self.x_t, &mut self.x).map_err(be)?;
                b.copy(&self.y_t, &mut self.y).map_err(be)?;
                k = 0;
                restarts += 1;
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

/// The PID primal-weight controller from cuPDLPx, with its safeguard: when
/// the displacements or the residual balance leave a sane range, fall back to
/// the weight from the restart whose relative primal and dual residuals were
/// closest, and clear the controller state.
struct PidState {
    integral: f64,
    prev_error: Option<f64>,
    best_weight: f64,
    best_balance: f64,
}

impl PidState {
    fn new(omega: f64) -> Self {
        Self { integral: 0.0, prev_error: None, best_weight: omega, best_balance: f64::INFINITY }
    }

    /// New weight after a restart with displacements `dx = ‖x_t - x_anchor‖`
    /// and `dy = ‖y_t - y_anchor‖`.
    fn update(&mut self, p: &SolverParams, omega: f64, dx: f64, dy: f64, rel_primal: f64, rel_dual: f64) -> f64 {
        let sane = |v: f64| (1e-16..=1e12).contains(&v);
        let ratio = rel_dual / rel_primal;
        if !sane(dx) || !sane(dy) || !(1e-8..=1e8).contains(&ratio) {
            self.integral = 0.0;
            self.prev_error = None;
            return self.best_weight;
        }
        // Positive error: the scaled primal displacement is too large.
        let e = (omega * dx / dy).ln();
        self.integral = p.pid_discount * self.integral + e;
        let derivative = self.prev_error.map_or(0.0, |prev| e - prev);
        self.prev_error = Some(e);
        let next = (omega.ln() - (p.pid_kp * e + p.pid_ki * self.integral + p.pid_kd * derivative)).exp();
        let balance = ratio.log10().abs();
        if balance < self.best_balance {
            self.best_balance = balance;
            self.best_weight = next;
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_moves_weight_toward_displacement_ratio() {
        let p = SolverParams::default();
        let mut pid = PidState::new(1.0);
        // Dual displacement 100x the primal one: ω should rise to about 100.
        let w = pid.update(&p, 1.0, 1e-2, 1.0, 1e-3, 1e-3);
        assert!((w.ln() - 100f64.ln()).abs() < 0.1, "{w}");
    }

    #[test]
    fn pid_safeguard_restores_best_weight() {
        let p = SolverParams::default();
        let mut pid = PidState::new(1.0);
        let best = pid.update(&p, 1.0, 1.0, 4.0, 1e-3, 1e-3); // balanced residuals
        pid.update(&p, best, 1.0, 9.0, 1e-3, 1e-1); // less balanced
        // A tiny displacement trips the safeguard.
        assert_eq!(pid.update(&p, 7.0, 1e-20, 1.0, 1e-3, 1e-3), best);
        assert_eq!(pid.integral, 0.0);
    }
}
