//! Every `Backend` method on the GPU against the CPU reference, on scaled
//! Netlib problems and deterministic pseudo-random vectors. Skips without a
//! GPU.

use std::path::PathBuf;

use saddlepoint_core::{Backend, ScaledProblem, ScalingParams};
use saddlepoint_cpu::CpuBackend;
use saddlepoint_cuda::CudaBackend;

fn gpu() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn scaled(name: &str) -> ScaledProblem {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../data/netlib/{name}.mps"));
    let p = saddlepoint_io::read_mps_file(path).unwrap();
    ScaledProblem::new(&p, &ScalingParams::default())
}

/// Deterministic values in [-2, 2], with every fifth one zero so that sign
/// tests take all branches.
fn vector(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if i % 5 == 0 { 0.0 } else { (s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0 }
        })
        .collect()
}

fn close(what: &str, a: f64, b: f64) {
    let err = (a - b).abs() / (1.0 + a.abs().max(b.abs()));
    assert!(err < 1e-11, "{what}: cpu {a} vs gpu {b} (relative {err:.2e})");
}

fn close_vec(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = 1.0 + a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale;
    assert!(err < 1e-12, "{what}: max relative difference {err:.2e}");
}

fn compare(name: &str, gpu: &CudaBackend) {
    let cpu = CpuBackend::new();
    let sp = scaled(name);
    let (m, n) = (sp.a.nrows, sp.a.ncols);
    let pc = cpu.load(&sp).unwrap();
    let pg = gpu.load(&sp).unwrap();
    let up = |v: &[f64]| gpu.upload(v).unwrap();
    let down = |v: &<CudaBackend as Backend>::Vector| gpu.download(v).unwrap();

    let (x, y) = (vector(n, 1), vector(m, 2));
    let (xa, ya) = (vector(n, 3), vector(m, 4));
    let (gx, gy, gxa, gya) = (up(&x), up(&y), up(&xa), up(&ya));

    // SpMV.
    let (mut cm, mut cn) = (vec![0.0; m], vec![0.0; n]);
    let (mut gm, mut gn) = (up(&cm), up(&cn));
    cpu.spmv(&pc, &x, &mut cm).unwrap();
    gpu.spmv(&pg, &gx, &mut gm).unwrap();
    close_vec(&format!("{name} spmv"), &cm, &down(&gm));
    cpu.spmv_t(&pc, &y, &mut cn).unwrap();
    gpu.spmv_t(&pg, &gy, &mut gn).unwrap();
    close_vec(&format!("{name} spmv_t"), &cn, &down(&gn));

    // Steps.
    let tau = 0.37;
    let (mut ct, mut cb) = (vec![0.0; n], vec![0.0; n]);
    let (mut gt, mut gb) = (up(&ct), up(&cb));
    cpu.primal_step(&pc, &x, &cn, tau, &mut ct, &mut cb).unwrap();
    gpu.primal_step(&pg, &gx, &gn, tau, &mut gt, &mut gb).unwrap();
    close_vec(&format!("{name} primal_step x_t"), &ct, &down(&gt));
    close_vec(&format!("{name} primal_step x_bar"), &cb, &down(&gb));

    let sigma = 1.9;
    let mut cyt = vec![0.0; m];
    let mut gyt = up(&cyt);
    cpu.dual_step(&pc, &y, &cm, sigma, &mut cyt).unwrap();
    gpu.dual_step(&pg, &gy, &gm, sigma, &mut gyt).unwrap();
    close_vec(&format!("{name} dual_step"), &cyt, &down(&gyt));

    let mut cz = x.clone();
    let mut gz = up(&cz);
    cpu.halpern_step(&mut cz, &ct, &xa, 0.8, 1.0).unwrap();
    gpu.halpern_step(&mut gz, &gt, &gxa, 0.8, 1.0).unwrap();
    close_vec(&format!("{name} halpern_step"), &cz, &down(&gz));

    // Vector helpers.
    close(&format!("{name} distance_sq"), cpu.distance_sq(&x, &xa).unwrap(), gpu.distance_sq(&gx, &gxa).unwrap());
    close(&format!("{name} norm_sq"), cpu.norm_sq(&y).unwrap(), gpu.norm_sq(&gy).unwrap());
    close(&format!("{name} dot"), cpu.dot(&y, &ya).unwrap(), gpu.dot(&gy, &gya).unwrap());
    let mut cs = vec![0.0; n];
    let mut gs = up(&cs);
    cpu.sub(&x, &xa, &mut cs).unwrap();
    gpu.sub(&gx, &gxa, &mut gs).unwrap();
    close_vec(&format!("{name} sub"), &cs, &down(&gs));
    cpu.scale(&mut cs, -1.5).unwrap();
    gpu.scale(&mut gs, -1.5).unwrap();
    close_vec(&format!("{name} scale"), &cs, &down(&gs));

    // Stats. Use the projected iterate so that the bounds are active.
    let ps = (cpu.primal_stats(&pc, &ct, &cm).unwrap(), gpu.primal_stats(&pg, &gt, &gm).unwrap());
    close(&format!("{name} primal residual"), ps.0.residual_sq, ps.1.residual_sq);
    close(&format!("{name} primal objective"), ps.0.objective, ps.1.objective);
    let ds = (cpu.dual_stats(&pc, &y, &cn).unwrap(), gpu.dual_stats(&pg, &gy, &gn).unwrap());
    close(&format!("{name} dual residual"), ds.0.residual_sq, ds.1.residual_sq);
    close(&format!("{name} dual objective"), ds.0.objective, ds.1.objective);
    let pr = (cpu.primal_ray_stats(&pc, &x, &cm).unwrap(), gpu.primal_ray_stats(&pg, &gx, &gm).unwrap());
    close(&format!("{name} primal ray objective"), pr.0.objective, pr.1.objective);
    close(&format!("{name} primal ray violation"), pr.0.violation_sq, pr.1.violation_sq);
    close(&format!("{name} primal ray norm"), pr.0.norm_sq, pr.1.norm_sq);
    let dr = (cpu.dual_ray_stats(&pc, &y, &cn).unwrap(), gpu.dual_ray_stats(&pg, &gy, &gn).unwrap());
    close(&format!("{name} dual ray objective"), dr.0.objective, dr.1.objective);
    close(&format!("{name} dual ray violation"), dr.0.violation_sq, dr.1.violation_sq);
    close(&format!("{name} dual ray norm"), dr.0.norm_sq, dr.1.norm_sq);

    // Copy and round trip.
    let mut gc = up(&vec![0.0; n]);
    gpu.copy(&gx, &mut gc).unwrap();
    assert_eq!(down(&gc), x, "{name} copy");
}

#[test]
fn backend_ops_match_cpu() {
    let Some(gpu) = gpu() else { return };
    // Small, medium, and one with free variables and ranges.
    for name in ["afiro", "israel", "25fv47", "80bau3b", "perold"] {
        compare(name, &gpu);
    }
}
