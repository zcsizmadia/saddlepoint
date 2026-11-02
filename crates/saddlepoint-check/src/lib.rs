//! Independent KKT checker.
//!
//! Recomputes residuals and the duality gap of a claimed LP solution from the
//! original problem data, with plain serial code that shares nothing with the
//! solver beyond the [`LpProblem`] type. Tests run every solver answer
//! through it.

use saddlepoint_core::LpProblem;

#[derive(Clone, Debug)]
pub struct KktReport {
    /// Objective of `x` in the minimisation form, constant included.
    pub primal_objective: f64,
    /// Dual objective of `y` in the minimisation form, constant included.
    pub dual_objective: f64,
    /// ‖violation of lc ≤ Ax ≤ uc‖₂.
    pub constraint_violation: f64,
    /// ‖violation of lv ≤ x ≤ uv‖₂.
    pub bound_violation: f64,
    /// ‖reduced-cost and dual-sign violations‖₂.
    pub dual_violation: f64,
    pub relative_primal: f64,
    pub relative_dual: f64,
    pub relative_gap: f64,
}

impl KktReport {
    pub fn max_relative(&self) -> f64 {
        self.relative_primal.max(self.relative_dual).max(self.relative_gap)
    }

    /// `Err` with a description when any relative measure exceeds `tol`.
    pub fn verify(&self, tol: f64) -> Result<(), String> {
        if self.max_relative() <= tol {
            Ok(())
        } else {
            Err(format!(
                "KKT check failed at tol {tol:e}: primal {:.3e}, dual {:.3e}, gap {:.3e}",
                self.relative_primal, self.relative_dual, self.relative_gap
            ))
        }
    }
}

fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.map(|x| x * x).sum::<f64>().sqrt()
}

fn outside(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo - v
    } else if v > hi {
        v - hi
    } else {
        0.0
    }
}

/// Checks `(x, y)` against `p`. `y` follows the solver's convention: the
/// multiplier of `lc ≤ Ax ≤ uc` in the minimisation form, positive when the
/// lower bound is active.
pub fn check(p: &LpProblem, x: &[f64], y: &[f64]) -> KktReport {
    assert_eq!(x.len(), p.ncols());
    assert_eq!(y.len(), p.nrows());
    let (m, n) = (p.nrows(), p.ncols());

    let mut ax = vec![0.0; m];
    let mut aty = vec![0.0; n];
    for i in 0..m {
        for k in p.a.row_ptr[i]..p.a.row_ptr[i + 1] {
            let (j, v) = (p.a.col_idx[k], p.a.values[k]);
            ax[i] += v * x[j];
            aty[j] += v * y[i];
        }
    }

    let primal_objective = p.c.iter().zip(x).map(|(c, x)| c * x).sum::<f64>() + p.c0;
    let constraint_violation = norm((0..m).map(|i| outside(ax[i], p.lc[i], p.uc[i])));
    let bound_violation = norm((0..n).map(|j| outside(x[j], p.lv[j], p.uv[j])));

    // Dual: y_i > 0 needs a finite lc_i, y_i < 0 a finite uc_i; the same for
    // the reduced cost r = c - Aᵀy against the variable bounds.
    let mut dual_objective = p.c0;
    let mut dual_viol = Vec::with_capacity(m + n);
    let mut term = |v: f64, lo: f64, hi: f64| {
        if v > 0.0 {
            if lo.is_finite() { dual_objective += v * lo } else { dual_viol.push(v) }
        } else if v < 0.0 {
            if hi.is_finite() { dual_objective += v * hi } else { dual_viol.push(v) }
        }
    };
    for ((&yi, &lo), &hi) in y.iter().zip(&p.lc).zip(&p.uc) {
        term(yi, lo, hi);
    }
    for (j, &atyj) in aty.iter().enumerate() {
        term(p.c[j] - atyj, p.lv[j], p.uv[j]);
    }
    let dual_violation = norm(dual_viol.into_iter());

    let b_norm = norm((0..m).flat_map(|i| {
        let (l, u) = (p.lc[i], p.uc[i]);
        let l = if l.is_finite() { l } else { 0.0 };
        let u = if u.is_finite() && p.uc[i] != p.lc[i] { u } else { 0.0 };
        [l, u]
    }));
    let c_norm = norm(p.c.iter().copied());
    let primal = (constraint_violation.powi(2) + bound_violation.powi(2)).sqrt();
    KktReport {
        primal_objective,
        dual_objective,
        constraint_violation,
        bound_violation,
        dual_violation,
        relative_primal: primal / (1.0 + b_norm),
        relative_dual: dual_violation / (1.0 + c_norm),
        relative_gap: (primal_objective - dual_objective).abs() / (1.0 + primal_objective.abs() + dual_objective.abs()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saddlepoint_core::CsrMatrix;

    const INF: f64 = f64::INFINITY;

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
    fn accepts_the_known_optimum() {
        // y solves Aᵀy = c on the two active ≤ rows: y = (-0.4, -0.2).
        let r = check(&tiny(), &[1.6, 1.2], &[-0.4, -0.2]);
        r.verify(1e-12).unwrap();
        assert!((r.primal_objective + 2.8).abs() < 1e-12);
    }

    #[test]
    fn rejects_a_suboptimal_point() {
        let r = check(&tiny(), &[1.0, 1.0], &[-0.4, -0.2]);
        assert!(r.verify(1e-6).is_err());
    }

    #[test]
    fn rejects_wrong_dual_sign() {
        let r = check(&tiny(), &[1.6, 1.2], &[0.4, 0.2]);
        assert!(r.relative_dual > 0.1);
    }
}
