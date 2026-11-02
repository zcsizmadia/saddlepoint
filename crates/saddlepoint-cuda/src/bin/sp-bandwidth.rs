//! `sp-bandwidth [DEVICE] [MILLION_ELEMENTS]`: measure memory bandwidth and
//! f32/f64 FMA rate, and estimate what that means for f64 CSR SpMV.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let device: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0);
    let millions: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(32);
    let r = saddlepoint_cuda::probe(device, millions * 1_000_000)?;
    let d = &r.device;
    println!(
        "device        {} (sm_{}{}, {} SMs)",
        d.name, d.compute_capability.0, d.compute_capability.1, d.multiprocessors
    );
    println!(
        "memory        {:.2} GiB, theoretical {:.0} GB/s",
        d.total_memory as f64 / (1u64 << 30) as f64,
        d.theoretical_bandwidth_gbs
    );
    println!("copy          f32 {:7.1} GB/s   f64 {:7.1} GB/s", r.copy_f32_gbs, r.copy_f64_gbs);
    println!("triad         f32 {:7.1} GB/s   f64 {:7.1} GB/s", r.triad_f32_gbs, r.triad_f64_gbs);
    println!(
        "fma           f32 {:7.1} GFLOP/s f64 {:7.1} GFLOP/s (ratio 1/{:.0})",
        r.fma_f32_gflops,
        r.fma_f64_gflops,
        r.fma_f32_gflops / r.fma_f64_gflops
    );
    // f64 CSR SpMV moves ~12 bytes per nonzero (8 value + 4 index) plus the
    // gathered x entry; take 16 bytes per 2 flops as the planning figure.
    let spmv_gflops = r.triad_f64_gbs / 16.0 * 2.0;
    println!(
        "f64 SpMV      ~{:.1} GFLOP/s if bandwidth-bound; f64 compute limit {:.1} GFLOP/s -> {}",
        spmv_gflops,
        r.fma_f64_gflops,
        if spmv_gflops < r.fma_f64_gflops { "bandwidth-bound" } else { "compute-bound" }
    );
    Ok(())
}
