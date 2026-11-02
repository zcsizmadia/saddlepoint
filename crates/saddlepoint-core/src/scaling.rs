//! Diagonal preconditioning: Ruiz equilibration, Pock-Chambolle scaling and
//! a scalar rescaling of bounds and objective.
//!
//! The scaled problem uses `A' = R A C`, and the variables relate as
//!
//! ```text
//! x = C x' / β        y = R y' / γ        (c - Aᵀy) = (c' - A'ᵀy') / (γ C)
//! ```
//!
//! where `β` scales the bounds and `γ` scales the objective.

use crate::problem::{CsrMatrix, LpProblem};

#[derive(Clone, Debug)]
pub struct ScalingParams {
    pub ruiz_iterations: usize,
    /// Pock-Chambolle exponent; `None` disables the step.
    pub pock_chambolle_alpha: Option<f64>,
    /// Scale bounds by `1 / (1 + ‖b‖)` and the objective by `1 / (1 + ‖c‖)`.
    pub bound_objective_rescaling: bool,
}

impl Default for ScalingParams {
    fn default() -> Self {
        Self { ruiz_iterations: 10, pock_chambolle_alpha: Some(1.0), bound_objective_rescaling: true }
    }
}

/// The problem after scaling, plus everything needed to map back.
#[derive(Clone, Debug)]
pub struct ScaledProblem {
    pub a: CsrMatrix,
    pub at: CsrMatrix,
    pub c: Vec<f64>,
    pub lc: Vec<f64>,
    pub uc: Vec<f64>,
    pub lv: Vec<f64>,
    pub uv: Vec<f64>,
    pub row_scale: Vec<f64>,
    pub col_scale: Vec<f64>,
    pub bound_scale: f64,
    pub obj_scale: f64,
}

impl ScaledProblem {
    pub fn new(p: &LpProblem, params: &ScalingParams) -> Self {
        let (m, n) = (p.nrows(), p.ncols());
        let mut a = p.a.clone();
        let mut row_scale = vec![1.0; m];
        let mut col_scale = vec![1.0; n];

        for _ in 0..params.ruiz_iterations {
            let (rmax, cmax) = abs_stats(&a, |v, acc| acc.max(v));
            let r: Vec<f64> = rmax.iter().map(|&v| inv_sqrt(v)).collect();
            let c: Vec<f64> = cmax.iter().map(|&v| inv_sqrt(v)).collect();
            apply(&mut a, &mut row_scale, &mut col_scale, &r, &c);
        }
        if let Some(alpha) = params.pock_chambolle_alpha {
            let (rsum, _) = abs_stats(&a, |v, acc| acc + v.powf(2.0 - alpha));
            let (_, csum) = abs_stats(&a, |v, acc| acc + v.powf(alpha));
            let r: Vec<f64> = rsum.iter().map(|&v| inv_sqrt(v)).collect();
            let c: Vec<f64> = csum.iter().map(|&v| inv_sqrt(v)).collect();
            apply(&mut a, &mut row_scale, &mut col_scale, &r, &c);
        }

        let mut c: Vec<f64> = p.c.iter().zip(&col_scale).map(|(c, s)| c * s).collect();
        let mut lc: Vec<f64> = p.lc.iter().zip(&row_scale).map(|(b, s)| b * s).collect();
        let mut uc: Vec<f64> = p.uc.iter().zip(&row_scale).map(|(b, s)| b * s).collect();
        let mut lv: Vec<f64> = p.lv.iter().zip(&col_scale).map(|(b, s)| b / s).collect();
        let mut uv: Vec<f64> = p.uv.iter().zip(&col_scale).map(|(b, s)| b / s).collect();

        let (mut bound_scale, mut obj_scale) = (1.0, 1.0);
        if params.bound_objective_rescaling {
            bound_scale = 1.0 / (1.0 + finite_bound_norm(&lc, &uc));
            obj_scale = 1.0 / (1.0 + norm2(&c));
            for v in lc.iter_mut().chain(&mut uc).chain(&mut lv).chain(&mut uv) {
                *v *= bound_scale;
            }
            for v in &mut c {
                *v *= obj_scale;
            }
        }

        let at = a.transpose();
        Self { a, at, c, lc, uc, lv, uv, row_scale, col_scale, bound_scale, obj_scale }
    }

    pub fn nrows(&self) -> usize {
        self.a.nrows
    }

    pub fn ncols(&self) -> usize {
        self.a.ncols
    }

    /// Maps a scaled primal vector back to the original space.
    pub fn unscale_primal(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().zip(&self.col_scale).map(|(x, s)| x * s / self.bound_scale).collect()
    }

    /// Maps a scaled dual vector back to the original space.
    pub fn unscale_dual(&self, ys: &[f64]) -> Vec<f64> {
        ys.iter().zip(&self.row_scale).map(|(y, s)| y * s / self.obj_scale).collect()
    }

    /// Per-row factor that maps a scaled primal residual to the original one.
    pub fn primal_residual_factors(&self) -> Vec<f64> {
        self.row_scale.iter().map(|s| 1.0 / (s * self.bound_scale)).collect()
    }

    /// Per-column factor that maps a scaled reduced cost to the original one.
    pub fn dual_residual_factors(&self) -> Vec<f64> {
        self.col_scale.iter().map(|s| 1.0 / (s * self.obj_scale)).collect()
    }

    /// Factor mapping a scaled objective value to the original one.
    pub fn objective_factor(&self) -> f64 {
        1.0 / (self.bound_scale * self.obj_scale)
    }
}

fn inv_sqrt(v: f64) -> f64 {
    if v > 0.0 && v.is_finite() { 1.0 / v.sqrt() } else { 1.0 }
}

/// Per-row and per-column folds of `|a_ij|`.
fn abs_stats(a: &CsrMatrix, f: impl Fn(f64, f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut rows = vec![0.0; a.nrows];
    let mut cols = vec![0.0; a.ncols];
    for (i, r) in rows.iter_mut().enumerate() {
        let (idx, vals) = a.row(i);
        for (&j, &v) in idx.iter().zip(vals) {
            *r = f(v.abs(), *r);
            cols[j] = f(v.abs(), cols[j]);
        }
    }
    (rows, cols)
}

fn apply(a: &mut CsrMatrix, row_scale: &mut [f64], col_scale: &mut [f64], r: &[f64], c: &[f64]) {
    a.scale(r, c);
    row_scale.iter_mut().zip(r).for_each(|(s, f)| *s *= f);
    col_scale.iter_mut().zip(c).for_each(|(s, f)| *s *= f);
}

pub(crate) fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// 2-norm of the finite constraint bounds, counting an equality row once.
pub(crate) fn finite_bound_norm(lo: &[f64], hi: &[f64]) -> f64 {
    lo.iter()
        .zip(hi)
        .map(|(&l, &u)| {
            let l = if l.is_finite() { l } else { 0.0 };
            let u = if u.is_finite() { u } else { 0.0 };
            if l == u { l * l } else { l * l + u * u }
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LpProblem {
        let a = CsrMatrix::from_triplets(2, 3, &[(0, 0, 100.0), (0, 1, 0.5), (1, 1, 3.0), (1, 2, -0.01)]);
        LpProblem {
            name: "sample".into(),
            a,
            c: vec![1.0, -2.0, 30.0],
            c0: 0.0,
            lc: vec![1.0, f64::NEG_INFINITY],
            uc: vec![1.0, 4.0],
            lv: vec![0.0, -1.0, f64::NEG_INFINITY],
            uv: vec![f64::INFINITY, 2.0, 5.0],
            maximize: false,
            row_names: vec![],
            col_names: vec![],
        }
    }

    #[test]
    fn scaled_matrix_is_r_a_c() {
        let p = sample();
        let s = ScaledProblem::new(&p, &ScalingParams::default());
        for i in 0..p.nrows() {
            let (cols, vals) = p.a.row(i);
            let (scols, svals) = s.a.row(i);
            assert_eq!(cols, scols);
            for ((&j, &v), &sv) in cols.iter().zip(vals).zip(svals) {
                let expected = v * s.row_scale[i] * s.col_scale[j];
                assert!((expected - sv).abs() <= 1e-14 * expected.abs());
            }
        }
    }

    #[test]
    fn unscaling_round_trips_feasibility_and_objective() {
        let p = sample();
        let s = ScaledProblem::new(&p, &ScalingParams::default());
        // Any original x maps to a scaled x' with the same constraint values
        // (up to the row scaling) and the same objective (up to β γ).
        let x = vec![0.01, 0.3, -2.0];
        let xs: Vec<f64> = x.iter().zip(&s.col_scale).map(|(x, c)| x / c * s.bound_scale).collect();
        let back = s.unscale_primal(&xs);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-14);
        }
        let obj: f64 = p.c.iter().zip(&x).map(|(c, x)| c * x).sum();
        let sobj: f64 = s.c.iter().zip(&xs).map(|(c, x)| c * x).sum();
        assert!((obj - sobj * s.objective_factor()).abs() < 1e-12);

        let mut ax = vec![0.0; 2];
        let mut asx = vec![0.0; 2];
        p.a.mul_vec(&x, &mut ax);
        s.a.mul_vec(&xs, &mut asx);
        let f = s.primal_residual_factors();
        for i in 0..2 {
            assert!((ax[i] - asx[i] * f[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn ruiz_equilibrates_rows_and_columns() {
        let p = sample();
        let params =
            ScalingParams { ruiz_iterations: 30, pock_chambolle_alpha: None, bound_objective_rescaling: false };
        let s = ScaledProblem::new(&p, &params);
        let (rmax, cmax) = abs_stats(&s.a, |v, acc| acc.max(v));
        for v in rmax.iter().chain(&cmax) {
            assert!((v - 1.0).abs() < 1e-3, "{v}");
        }
    }
}
