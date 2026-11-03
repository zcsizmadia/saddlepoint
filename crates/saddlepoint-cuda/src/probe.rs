//! Device probe: memory bandwidth and f32/f64 arithmetic rate. PDHG is
//! bandwidth-bound, so these numbers say how fast the solver can be on a
//! given card.

use std::sync::Arc;

use crate::ensure_cuda_dlls_on_path;

use cudarc::driver::sys::{CUdevice_attribute as Attr, CUevent_flags};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

const PROBE_SRC: &str = r#"
template <typename T>
__device__ void copy_impl(const T* __restrict__ in, T* __restrict__ out, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = in[i];
}

template <typename T>
__device__ void triad_impl(T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c, T s, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        a[i] = b[i] + s * c[i];
}

// Each thread runs `iters` rounds of 8 independent FMA chains: 16 flops per round.
template <typename T>
__device__ void fma_impl(T* out, int iters) {
    T a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const T m = (T)0.999999, c = (T)1e-7;
    for (int k = 0; k < iters; ++k) {
        a0 = a0 * m + c; a1 = a1 * m + c; a2 = a2 * m + c; a3 = a3 * m + c;
        a4 = a4 * m + c; a5 = a5 * m + c; a6 = a6 * m + c; a7 = a7 * m + c;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

extern "C" __global__ void copy_f32(const float* in, float* out, long long n) { copy_impl(in, out, n); }
extern "C" __global__ void copy_f64(const double* in, double* out, long long n) { copy_impl(in, out, n); }
extern "C" __global__ void triad_f32(float* a, const float* b, const float* c, float s, long long n) { triad_impl(a, b, c, s, n); }
extern "C" __global__ void triad_f64(double* a, const double* b, const double* c, double s, long long n) { triad_impl(a, b, c, s, n); }
extern "C" __global__ void fma_f32(float* out, int iters) { fma_impl(out, iters); }
extern "C" __global__ void fma_f64(double* out, int iters) { fma_impl(out, iters); }
"#;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
    /// From memory clock and bus width; laptops often run below this.
    pub theoretical_bandwidth_gbs: f64,
}

#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub device: DeviceInfo,
    pub copy_f32_gbs: f64,
    pub copy_f64_gbs: f64,
    pub triad_f32_gbs: f64,
    pub triad_f64_gbs: f64,
    pub fma_f32_gflops: f64,
    pub fma_f64_gflops: f64,
}

pub fn device_info(ctx: &Arc<CudaContext>) -> Result<DeviceInfo, DriverError> {
    let attr = |a| ctx.attribute(a);
    let clock_khz = attr(Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)? as f64;
    let bus_bits = attr(Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)? as f64;
    Ok(DeviceInfo {
        name: ctx.name()?,
        total_memory: ctx.total_mem()?,
        compute_capability: (
            attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ),
        multiprocessors: attr(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        // DDR: two transfers per clock.
        theoretical_bandwidth_gbs: 2.0 * clock_khz * 1e3 * bus_bits / 8.0 / 1e9,
    })
}

/// Times `reps` launches of `launch` and returns milliseconds per launch.
fn time_ms(
    stream: &Arc<CudaStream>,
    reps: usize,
    mut launch: impl FnMut() -> Result<(), DriverError>,
) -> Result<f64, DriverError> {
    launch()?; // warm-up
    let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..reps {
        launch()?;
    }
    let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    Ok(start.elapsed_ms(&end)? as f64 / reps as f64)
}

/// Measures copy and triad bandwidth on arrays of `n` elements, and FMA rate.
pub fn probe(device: usize, n: usize) -> Result<ProbeResult, Box<dyn std::error::Error>> {
    ensure_cuda_dlls_on_path();
    let ctx = CudaContext::new(device)?;
    let info = device_info(&ctx)?;
    let stream = ctx.default_stream();
    let opts = CompileOptions { options: vec!["--std=c++17".into()], ..Default::default() };
    let module = ctx.load_module(compile_ptx_with_opts(PROBE_SRC, opts)?)?;
    let f = |name: &str| module.load_function(name);
    let (copy32, copy64, triad32, triad64, fma32, fma64): (
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
    ) = (f("copy_f32")?, f("copy_f64")?, f("triad_f32")?, f("triad_f64")?, f("fma_f32")?, f("fma_f64")?);

    let blocks = (info.multiprocessors as u32) * 8;
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let nn = n as i64;
    let reps = 20;

    let gbs = |bytes: f64, ms: f64| bytes / (ms * 1e-3) / 1e9;

    let (copy_f32_gbs, triad_f32_gbs) = {
        let a = stream.alloc_zeros::<f32>(n)?;
        let mut b = stream.alloc_zeros::<f32>(n)?;
        let c = stream.alloc_zeros::<f32>(n)?;
        let ms = time_ms(&stream, reps, || {
            let mut l = stream.launch_builder(&copy32);
            l.arg(&a).arg(&mut b).arg(&nn);
            unsafe { l.launch(cfg) }.map(|_| ())
        })?;
        let copy = gbs(2.0 * 4.0 * n as f64, ms);
        let ms = time_ms(&stream, reps, || {
            let mut l = stream.launch_builder(&triad32);
            l.arg(&mut b).arg(&a).arg(&c).arg(&1.5f32).arg(&nn);
            unsafe { l.launch(cfg) }.map(|_| ())
        })?;
        (copy, gbs(3.0 * 4.0 * n as f64, ms))
    };
    let (copy_f64_gbs, triad_f64_gbs) = {
        let a = stream.alloc_zeros::<f64>(n)?;
        let mut b = stream.alloc_zeros::<f64>(n)?;
        let c = stream.alloc_zeros::<f64>(n)?;
        let ms = time_ms(&stream, reps, || {
            let mut l = stream.launch_builder(&copy64);
            l.arg(&a).arg(&mut b).arg(&nn);
            unsafe { l.launch(cfg) }.map(|_| ())
        })?;
        let copy = gbs(2.0 * 8.0 * n as f64, ms);
        let ms = time_ms(&stream, reps, || {
            let mut l = stream.launch_builder(&triad64);
            l.arg(&mut b).arg(&a).arg(&c).arg(&1.5f64).arg(&nn);
            unsafe { l.launch(cfg) }.map(|_| ())
        })?;
        (copy, gbs(3.0 * 8.0 * n as f64, ms))
    };

    let threads = (blocks * 256) as f64;
    let fma_cfg = cfg;
    let (it32, it64) = (4096i32, 256i32);
    let mut out32 = stream.alloc_zeros::<f32>(threads as usize)?;
    let ms = time_ms(&stream, 5, || {
        let mut l = stream.launch_builder(&fma32);
        l.arg(&mut out32).arg(&it32);
        unsafe { l.launch(fma_cfg) }.map(|_| ())
    })?;
    let fma_f32_gflops = threads * it32 as f64 * 16.0 / (ms * 1e-3) / 1e9;
    let mut out64 = stream.alloc_zeros::<f64>(threads as usize)?;
    let ms = time_ms(&stream, 5, || {
        let mut l = stream.launch_builder(&fma64);
        l.arg(&mut out64).arg(&it64);
        unsafe { l.launch(fma_cfg) }.map(|_| ())
    })?;
    let fma_f64_gflops = threads * it64 as f64 * 16.0 / (ms * 1e-3) / 1e9;
    stream.synchronize()?;

    Ok(ProbeResult {
        device: info,
        copy_f32_gbs,
        copy_f64_gbs,
        triad_f32_gbs,
        triad_f64_gbs,
        fma_f32_gflops,
        fma_f64_gflops,
    })
}
