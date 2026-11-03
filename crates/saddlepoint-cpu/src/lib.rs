//! CPU reference backend. It is the source of truth for tests and the
//! fallback when no GPU is present. Work above a size threshold runs on the
//! rayon thread pool; small problems stay on the calling thread, where pool
//! dispatch would cost more than the work.

use std::convert::Infallible;

use rayon::prelude::*;
use saddlepoint_core::{Backend, CsrMatrix, DualStats, PrimalStats, RayStats, ScaledProblem};

/// Below this many elements (or nonzeros) a pass runs serially.
const PAR_THRESHOLD: usize = 32 * 1024;

#[derive(Clone, Debug, Default)]
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct CpuProblem {
    a: CsrMatrix,
    at: CsrMatrix,
    c: Vec<f64>,
    lc: Vec<f64>,
    uc: Vec<f64>,
    lv: Vec<f64>,
    uv: Vec<f64>,
    primal_factor: Vec<f64>,
    dual_factor: Vec<f64>,
}

fn sum2(n: usize, f: impl Fn(usize) -> (f64, f64) + Sync + Send) -> (f64, f64) {
    let add = |a: (f64, f64), b: (f64, f64)| (a.0 + b.0, a.1 + b.1);
    if n < PAR_THRESHOLD {
        (0..n).map(f).fold((0.0, 0.0), add)
    } else {
        (0..n).into_par_iter().map(f).reduce(|| (0.0, 0.0), add)
    }
}

fn sum1(n: usize, f: impl Fn(usize) -> f64 + Sync + Send) -> f64 {
    if n < PAR_THRESHOLD { (0..n).map(f).sum() } else { (0..n).into_par_iter().map(f).sum() }
}

fn for_each(out: &mut [f64], f: impl Fn(usize, &mut f64) + Sync + Send) {
    if out.len() < PAR_THRESHOLD {
        out.iter_mut().enumerate().for_each(|(i, o)| f(i, o));
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, o)| f(i, o));
    }
}

fn csr_mul(a: &CsrMatrix, x: &[f64], out: &mut [f64]) {
    let row = |i: usize, o: &mut f64| {
        let (cols, vals) = a.row(i);
        *o = cols.iter().zip(vals).map(|(&c, &v)| v * x[c]).sum();
    };
    if a.nnz() < PAR_THRESHOLD {
        out.iter_mut().enumerate().for_each(|(i, o)| row(i, o));
    } else {
        out.par_iter_mut().with_min_len(256).enumerate().for_each(|(i, o)| row(i, o));
    }
}

/// Distance from `v` to `[lo, hi]`, signed: negative below, positive above.
#[inline]
fn excess(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        v - lo
    } else if v > hi {
        v - hi
    } else {
        0.0
    }
}

/// Bound term `v⁺ lo - v⁻ hi`, skipping infinite bounds.
#[inline]
fn bound_term(v: f64, lo: f64, hi: f64) -> f64 {
    if v > 0.0 && lo.is_finite() {
        v * lo
    } else if v < 0.0 && hi.is_finite() {
        v * hi
    } else {
        0.0
    }
}

/// Part of `v` whose sign an infinite bound forbids: `v > 0` needs a finite
/// `lo`, `v < 0` needs a finite `hi`.
#[inline]
fn forbidden(v: f64, lo: f64, hi: f64) -> f64 {
    if (v > 0.0 && lo == f64::NEG_INFINITY) || (v < 0.0 && hi == f64::INFINITY) { v } else { 0.0 }
}

impl Backend for CpuBackend {
    type Error = Infallible;
    type Vector = Vec<f64>;
    type Problem = CpuProblem;

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn load(&self, p: &ScaledProblem) -> Result<CpuProblem, Infallible> {
        Ok(CpuProblem {
            a: p.a.clone(),
            at: p.at.clone(),
            c: p.c.clone(),
            lc: p.lc.clone(),
            uc: p.uc.clone(),
            lv: p.lv.clone(),
            uv: p.uv.clone(),
            primal_factor: p.primal_residual_factors(),
            dual_factor: p.dual_residual_factors(),
        })
    }

    fn upload(&self, data: &[f64]) -> Result<Vec<f64>, Infallible> {
        Ok(data.to_vec())
    }

    fn download(&self, v: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(v.clone())
    }

    fn copy(&self, src: &Vec<f64>, dst: &mut Vec<f64>) -> Result<(), Infallible> {
        dst.copy_from_slice(src);
        Ok(())
    }

    fn spmv(&self, p: &CpuProblem, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), Infallible> {
        csr_mul(&p.a, x, out);
        Ok(())
    }

    fn spmv_t(&self, p: &CpuProblem, y: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), Infallible> {
        csr_mul(&p.at, y, out);
        Ok(())
    }

    fn primal_step(
        &self,
        p: &CpuProblem,
        x: &Vec<f64>,
        aty: &Vec<f64>,
        tau: f64,
        x_t: &mut Vec<f64>,
        x_bar: &mut Vec<f64>,
    ) -> Result<(), Infallible> {
        for_each(x_t, |j, o| *o = (x[j] - tau * (p.c[j] - aty[j])).clamp(p.lv[j], p.uv[j]));
        let x_t = &*x_t;
        for_each(x_bar, |j, o| *o = 2.0 * x_t[j] - x[j]);
        Ok(())
    }

    fn dual_step(
        &self,
        p: &CpuProblem,
        y: &Vec<f64>,
        ax_bar: &Vec<f64>,
        sigma: f64,
        y_t: &mut Vec<f64>,
    ) -> Result<(), Infallible> {
        for_each(y_t, |i, o| {
            let v = ax_bar[i] - y[i] / sigma;
            *o = y[i] - sigma * (ax_bar[i] - v.clamp(p.lc[i], p.uc[i]));
        });
        Ok(())
    }

    fn halpern_step(
        &self,
        z: &mut Vec<f64>,
        z_t: &Vec<f64>,
        anchor: &Vec<f64>,
        w: f64,
        rho: f64,
    ) -> Result<(), Infallible> {
        for_each(z, |i, o| *o = w * ((1.0 + rho) * z_t[i] - rho * *o) + (1.0 - w) * anchor[i]);
        Ok(())
    }

    fn distance_sq(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(sum1(a.len(), |i| (a[i] - b[i]).powi(2)))
    }

    fn sub(&self, a: &Vec<f64>, b: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), Infallible> {
        for_each(out, |i, o| *o = a[i] - b[i]);
        Ok(())
    }

    fn norm_sq(&self, v: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(sum1(v.len(), |i| v[i] * v[i]))
    }

    fn dot(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(sum1(a.len(), |i| a[i] * b[i]))
    }

    fn scale(&self, v: &mut Vec<f64>, alpha: f64) -> Result<(), Infallible> {
        for_each(v, |_, o| *o *= alpha);
        Ok(())
    }

    fn primal_stats(&self, p: &CpuProblem, x: &Vec<f64>, ax: &Vec<f64>) -> Result<PrimalStats, Infallible> {
        let residual_sq = sum1(ax.len(), |i| (excess(ax[i], p.lc[i], p.uc[i]) * p.primal_factor[i]).powi(2));
        let objective = sum1(x.len(), |j| p.c[j] * x[j]);
        Ok(PrimalStats { residual_sq, objective })
    }

    fn dual_stats(&self, p: &CpuProblem, y: &Vec<f64>, aty: &Vec<f64>) -> Result<DualStats, Infallible> {
        let rows = sum1(y.len(), |i| bound_term(y[i], p.lc[i], p.uc[i]));
        let (res, cols) = sum2(aty.len(), |j| {
            let r = p.c[j] - aty[j];
            ((forbidden(r, p.lv[j], p.uv[j]) * p.dual_factor[j]).powi(2), bound_term(r, p.lv[j], p.uv[j]))
        });
        Ok(DualStats { residual_sq: res, objective: rows + cols })
    }

    fn step_dual_residual_sq(
        &self,
        p: &CpuProblem,
        x: &Vec<f64>,
        x_t: &Vec<f64>,
        aty: &Vec<f64>,
        aty_t: &Vec<f64>,
        tau: f64,
    ) -> Result<f64, Infallible> {
        let inv_tau = 1.0 / tau;
        Ok(sum1(x.len(), |j| ((aty[j] - aty_t[j] - (x_t[j] - x[j]) * inv_tau) * p.dual_factor[j]).powi(2)))
    }

    fn primal_ray_stats(&self, p: &CpuProblem, dx: &Vec<f64>, adx: &Vec<f64>) -> Result<RayStats, Infallible> {
        // A primal ray keeps A dx inside the recession cone of [lc, uc] and dx
        // inside that of [lv, uv].
        let rows = sum1(adx.len(), |i| recession_violation(adx[i], p.lc[i], p.uc[i]).powi(2));
        let (cols, norm_sq) = sum2(dx.len(), |j| (recession_violation(dx[j], p.lv[j], p.uv[j]).powi(2), dx[j] * dx[j]));
        let objective = sum1(dx.len(), |j| p.c[j] * dx[j]);
        Ok(RayStats { objective, violation_sq: rows + cols, norm_sq })
    }

    fn dual_ray_stats(&self, p: &CpuProblem, dy: &Vec<f64>, atdy: &Vec<f64>) -> Result<RayStats, Infallible> {
        // A dual ray (c = 0): reduced cost r = -Aᵀdy; its sign and dy's sign
        // must respect the infinite bounds.
        let (row_viol, row_obj) =
            sum2(dy.len(), |i| (forbidden(dy[i], p.lc[i], p.uc[i]).powi(2), bound_term(dy[i], p.lc[i], p.uc[i])));
        let (col_viol, col_obj) = sum2(atdy.len(), |j| {
            let r = -atdy[j];
            (forbidden(r, p.lv[j], p.uv[j]).powi(2), bound_term(r, p.lv[j], p.uv[j]))
        });
        let norm_sq = sum1(dy.len(), |i| dy[i] * dy[i]);
        Ok(RayStats { objective: row_obj + col_obj, violation_sq: row_viol + col_viol, norm_sq })
    }
}

/// How far `v` leaves the recession cone of `[lo, hi]`: a finite `lo` needs
/// `v ≥ 0`, a finite `hi` needs `v ≤ 0`.
#[inline]
fn recession_violation(v: f64, lo: f64, hi: f64) -> f64 {
    let lo = if lo.is_finite() { 0.0 } else { f64::NEG_INFINITY };
    let hi = if hi.is_finite() { 0.0 } else { f64::INFINITY };
    excess(v, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use saddlepoint_core::{LpProblem, SolverParams, Status, solve};

    const INF: f64 = f64::INFINITY;

    /// min -x - y  s.t.  x + 2y ≤ 4,  3x + y ≤ 6,  x, y ≥ 0.  Optimum at
    /// (1.6, 1.2) with objective -2.8.
    fn tiny() -> LpProblem {
        LpProblem {
            name: "tiny".into(),
            a: CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 1.0)]),
            c: vec![-1.0, -1.0],
            c0: 0.0,
            lc: vec![-INF, -INF],
            uc: vec![4.0, 6.0],
            lv: vec![0.0, 0.0],
            uv: vec![INF, INF],
            maximize: false,
            row_names: vec![],
            col_names: vec![],
        }
    }

    #[test]
    fn solves_tiny_lp() {
        let params = SolverParams { tolerance: 1e-8, ..Default::default() };
        let s = solve(&CpuBackend::new(), &tiny(), &params).unwrap();
        assert_eq!(s.status, Status::Optimal);
        assert!((s.objective + 2.8).abs() < 1e-6, "{}", s.objective);
        assert!((s.x[0] - 1.6).abs() < 1e-5 && (s.x[1] - 1.2).abs() < 1e-5, "{:?}", s.x);
    }

    #[test]
    fn detects_primal_infeasibility() {
        // x + y ≥ 5 with x, y ≤ 1.
        let mut p = tiny();
        p.a = CsrMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
        p.lc = vec![5.0];
        p.uc = vec![INF];
        p.uv = vec![1.0, 1.0];
        let s = solve(&CpuBackend::new(), &p, &SolverParams::default()).unwrap();
        assert_eq!(s.status, Status::PrimalInfeasible);
    }

    #[test]
    fn detects_dual_infeasibility() {
        // min -x - y with only x - y ≤ 1: unbounded along (1, 1).
        let mut p = tiny();
        p.a = CsrMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, -1.0)]);
        p.lc = vec![-INF];
        p.uc = vec![1.0];
        let s = solve(&CpuBackend::new(), &p, &SolverParams::default()).unwrap();
        assert_eq!(s.status, Status::DualInfeasible);
    }
}
