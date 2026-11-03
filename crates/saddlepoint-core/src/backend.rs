//! The compute backend the PDHG loop runs on.
//!
//! The trait is deliberately small and fused: each method is one pass over
//! memory (one kernel on a GPU). Vectors live on the backend; only the
//! scalar results of reductions come back to the host.
//!
//! Every method works on the *scaled* problem held in [`Backend::Problem`].

use crate::scaling::ScaledProblem;

/// Partial KKT quantities from one fused reduction. Residuals are already
/// mapped to the original (unscaled) space; objective terms are in the
/// scaled space (multiply by [`ScaledProblem::objective_factor`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrimalStats {
    /// `Σ (dist(Ax, [lc, uc]) · factor)²` over rows.
    pub residual_sq: f64,
    /// `c'ᵀx'`.
    pub objective: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DualStats {
    /// Squared norm of the reduced-cost components that infinite variable
    /// bounds forbid, mapped to the original space.
    pub residual_sq: f64,
    /// Scaled dual objective (bound terms of rows and columns).
    pub objective: f64,
}

/// Quantities used to recognise an infeasibility certificate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RayStats {
    /// Objective of the ray: `cᵀdx` for a primal ray, the bound terms for a
    /// dual ray.
    pub objective: f64,
    /// Squared norm of the ray's violations of the homogeneous constraints.
    pub violation_sq: f64,
    /// Squared norm of the ray itself.
    pub norm_sq: f64,
}

pub trait Backend {
    type Error: std::error::Error + Send + Sync + 'static;
    type Vector;
    type Problem;

    fn name(&self) -> String;

    /// Moves the scaled problem onto the backend.
    fn load(&self, problem: &ScaledProblem) -> Result<Self::Problem, Self::Error>;

    fn upload(&self, data: &[f64]) -> Result<Self::Vector, Self::Error>;
    fn download(&self, v: &Self::Vector) -> Result<Vec<f64>, Self::Error>;
    fn copy(&self, src: &Self::Vector, dst: &mut Self::Vector) -> Result<(), Self::Error>;

    /// `out = A' x`.
    fn spmv(&self, p: &Self::Problem, x: &Self::Vector, out: &mut Self::Vector) -> Result<(), Self::Error>;
    /// `out = A'ᵀ y`.
    fn spmv_t(&self, p: &Self::Problem, y: &Self::Vector, out: &mut Self::Vector) -> Result<(), Self::Error>;

    /// `x_t = proj_[lv,uv](x - τ (c - aty))` and `x_bar = 2 x_t - x`.
    fn primal_step(
        &self,
        p: &Self::Problem,
        x: &Self::Vector,
        aty: &Self::Vector,
        tau: f64,
        x_t: &mut Self::Vector,
        x_bar: &mut Self::Vector,
    ) -> Result<(), Self::Error>;

    /// `y_t = y - σ (ax_bar - proj_[lc,uc](ax_bar - y / σ))`.
    fn dual_step(
        &self,
        p: &Self::Problem,
        y: &Self::Vector,
        ax_bar: &Self::Vector,
        sigma: f64,
        y_t: &mut Self::Vector,
    ) -> Result<(), Self::Error>;

    /// Reflected Halpern update
    /// `z = w ((1 + ρ) z_t - ρ z) + (1 - w) anchor`.
    fn halpern_step(
        &self,
        z: &mut Self::Vector,
        z_t: &Self::Vector,
        anchor: &Self::Vector,
        w: f64,
        rho: f64,
    ) -> Result<(), Self::Error>;

    /// `‖a - b‖²`.
    fn distance_sq(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64, Self::Error>;
    /// `out = a - b`.
    fn sub(&self, a: &Self::Vector, b: &Self::Vector, out: &mut Self::Vector) -> Result<(), Self::Error>;
    /// `‖v‖²`.
    fn norm_sq(&self, v: &Self::Vector) -> Result<f64, Self::Error>;
    /// `aᵀb`.
    fn dot(&self, a: &Self::Vector, b: &Self::Vector) -> Result<f64, Self::Error>;
    /// `v *= alpha`.
    fn scale(&self, v: &mut Self::Vector, alpha: f64) -> Result<(), Self::Error>;

    /// Primal residual and objective of `x` given `ax = A' x`.
    fn primal_stats(&self, p: &Self::Problem, x: &Self::Vector, ax: &Self::Vector) -> Result<PrimalStats, Self::Error>;
    /// Dual residual and objective of `y` given `aty = A'ᵀ y`.
    fn dual_stats(&self, p: &Self::Problem, y: &Self::Vector, aty: &Self::Vector) -> Result<DualStats, Self::Error>;
    /// cuPDLPx's dual residual, squared and in original units: with the
    /// reduced cost `λ = c - aty + (x_t - x) / tau` that the primal step from
    /// `x` implied, the residual of `y_t` is `c - aty_t - λ`, that is
    /// `Σ ((aty - aty_t - (x_t - x) / tau) · dual_factor)²`.
    fn step_dual_residual_sq(
        &self,
        p: &Self::Problem,
        x: &Self::Vector,
        x_t: &Self::Vector,
        aty: &Self::Vector,
        aty_t: &Self::Vector,
        tau: f64,
    ) -> Result<f64, Self::Error>;

    /// Treats `dx` as a candidate primal ray (dual infeasibility certificate).
    fn primal_ray_stats(
        &self,
        p: &Self::Problem,
        dx: &Self::Vector,
        adx: &Self::Vector,
    ) -> Result<RayStats, Self::Error>;
    /// Treats `dy` as a candidate dual ray (primal infeasibility certificate).
    fn dual_ray_stats(
        &self,
        p: &Self::Problem,
        dy: &Self::Vector,
        atdy: &Self::Vector,
    ) -> Result<RayStats, Self::Error>;
}
