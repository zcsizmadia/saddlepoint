//! Core of the saddlepoint solver: problem types, preconditioning and the
//! restarted Halpern PDHG loop, generic over a compute [`Backend`].

pub mod backend;
pub mod problem;
pub mod scaling;
pub mod solver;

pub use backend::{Backend, DualStats, PrimalStats, RayStats};
pub use problem::{CsrMatrix, LpProblem, ProblemError};
pub use scaling::{ScaledProblem, ScalingParams};
pub use solver::{Solution, SolveError, SolverParams, Status, Termination, solve};
