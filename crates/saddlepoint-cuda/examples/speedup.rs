//! GPU vs CPU time per PDHG iteration on a large random LP.
//!
//! Both backends take the same iterations (see `tests/netlib.rs`), so time
//! per iteration is the end-to-end speedup.
//!
//! ```sh
//! cargo run --release -p saddlepoint-cuda --example speedup -- [COLS] [NNZ_PER_COL] [ITERS] [BAND]
//! ```
//!
//! The LP: `min cᵀx` over `0 ≤ x ≤ 10`, `m = COLS / 2` rows, half of them
//! equalities and half `≤`, built around a known feasible point so that the
//! problem is valid. Column `j`'s nonzeros fall in rows within `BAND` of
//! `j / 2`, which gives the locality of structured LPs. `BAND = 0` scatters
//! them uniformly instead, the worst case for SpMV gathers.

use std::time::Instant;

use saddlepoint_core::{Backend, CsrMatrix, LpProblem, SolverParams, solve};
use saddlepoint_cpu::CpuBackend;
use saddlepoint_cuda::CudaBackend;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random_lp(n: usize, per_col: usize, band: usize) -> LpProblem {
    let m = n / 2;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut triplets = Vec::with_capacity(n * per_col);
    for j in 0..n {
        for _ in 0..per_col {
            let i = if band == 0 {
                (rng.next() % m as u64) as usize
            } else {
                (j / 2 + m - band + (rng.next() % (2 * band as u64 + 1)) as usize) % m
            };
            triplets.push((i, j, rng.unit() * 2.0 - 1.0));
        }
    }
    let a = CsrMatrix::from_triplets(m, n, &triplets);
    let x0: Vec<f64> = (0..n).map(|_| rng.unit() * 10.0).collect();
    let mut b = vec![0.0; m];
    a.mul_vec(&x0, &mut b);
    let lc = b.iter().enumerate().map(|(i, &v)| if i % 2 == 0 { v } else { f64::NEG_INFINITY }).collect();
    let uc = b.iter().map(|&v| v + 1.0).collect::<Vec<_>>();
    let uc = uc.iter().zip(&b).enumerate().map(|(i, (&u, &v))| if i % 2 == 0 { v } else { u }).collect();
    LpProblem {
        name: format!("random_{n}x{per_col}"),
        a,
        c: (0..n).map(|_| rng.unit() - 0.3).collect(),
        c0: 0.0,
        lc,
        uc,
        lv: vec![0.0; n],
        uv: vec![10.0; n],
        maximize: false,
        row_names: vec![],
        col_names: vec![],
    }
}

fn per_iteration<B: Backend>(backend: &B, p: &LpProblem, iters: usize) -> f64 {
    let params = SolverParams { max_iterations: iters as u64, tolerance: 1e-12, ..Default::default() };
    let t = Instant::now();
    let s = solve(backend, p, &params).expect("solve");
    let total = t.elapsed().as_secs_f64();
    println!(
        "  {:<40} {:>7} iterations  {:>8.3}s solve  {:>8.1} µs/iteration  ({:.3}s including setup)",
        backend.name(),
        s.iterations,
        s.solve_time.as_secs_f64(),
        s.solve_time.as_secs_f64() / s.iterations as f64 * 1e6,
        total
    );
    s.solve_time.as_secs_f64() / s.iterations as f64
}

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>().expect("numeric argument"));
    let n = args.next().unwrap_or(1_000_000);
    let per_col = args.next().unwrap_or(5);
    let iters = args.next().unwrap_or(2_000);
    let band = args.next().unwrap_or(1_000);
    let p = random_lp(n, per_col, band);
    println!("{}: {} rows, {} cols, {} nonzeros", p.name, p.nrows(), p.ncols(), p.a.nnz());
    let gpu = CudaBackend::new(0).expect("no usable GPU");
    let g = per_iteration(&gpu, &p, iters);
    let c = per_iteration(&CpuBackend::new(), &p, iters);
    println!("  speedup {:.1}x", c / g);
}
