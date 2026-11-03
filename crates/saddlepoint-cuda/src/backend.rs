//! [`CudaBackend`]: the PDHG backend on an NVIDIA GPU, in f64.
//!
//! SpMV goes through cuSPARSE; everything else is one NVRTC kernel from
//! `kernels.cu` per [`Backend`] method. Reductions write per-block partials
//! into a small device buffer; one download per stats call brings them back,
//! and the host adds them.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::cusparse::result::CusparseError;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};
use saddlepoint_core::{Backend, DualStats, PrimalStats, RayStats, ScaledProblem};

use crate::ensure_cuda_dlls_on_path;
use crate::sparse::{CsrDevice, Handle};

const SRC: &str = include_str!("kernels.cu");
/// Must match `BLOCK` in `kernels.cu`.
const BLOCK: u32 = 256;
/// The most values a single stats call reduces (dual ray: 3 + 2).
const MAX_VALUES_PER_CALL: usize = 5;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
    #[error("NVRTC: {0}")]
    Nvrtc(#[from] CompileError),
    #[error("cuSPARSE: {0:?}")]
    Cusparse(CusparseError),
    #[error("problem too large for 32-bit sparse indices")]
    TooLarge,
    /// The CUDA driver, NVRTC or cuSPARSE library couldn't be loaded.
    #[error("CUDA unavailable: {0}")]
    Unavailable(String),
}

impl From<CusparseError> for CudaError {
    fn from(e: CusparseError) -> Self {
        Self::Cusparse(e)
    }
}

struct Kernels {
    primal_step: CudaFunction,
    dual_step: CudaFunction,
    halpern_step: CudaFunction,
    sub: CudaFunction,
    scale: CudaFunction,
    red_distance_sq: CudaFunction,
    red_dot: CudaFunction,
    red_primal_residual: CudaFunction,
    red_dual_rows: CudaFunction,
    red_dual_cols: CudaFunction,
    red_primal_ray_rows: CudaFunction,
    red_primal_ray_cols: CudaFunction,
    red_dual_ray_rows: CudaFunction,
    red_dual_ray_cols: CudaFunction,
    red_step_dual: CudaFunction,
}

impl Kernels {
    fn load(m: &Arc<CudaModule>) -> Result<Self, DriverError> {
        let f = |name: &str| m.load_function(name);
        Ok(Self {
            primal_step: f("primal_step")?,
            dual_step: f("dual_step")?,
            halpern_step: f("halpern_step")?,
            sub: f("sub")?,
            scale: f("scale")?,
            red_distance_sq: f("red_distance_sq")?,
            red_dot: f("red_dot")?,
            red_primal_residual: f("red_primal_residual")?,
            red_dual_rows: f("red_dual_rows")?,
            red_dual_cols: f("red_dual_cols")?,
            red_primal_ray_rows: f("red_primal_ray_rows")?,
            red_primal_ray_cols: f("red_primal_ray_cols")?,
            red_dual_ray_rows: f("red_dual_ray_rows")?,
            red_dual_ray_cols: f("red_dual_ray_cols")?,
            red_step_dual: f("red_step_dual")?,
        })
    }
}

pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    k: Kernels,
    sparse: Handle,
    /// Upper bound on the grid of element-wise kernels.
    max_blocks: u32,
    /// Grid of every reduction (fewer for short vectors).
    reduce_blocks: u32,
    /// Per-block partials of the reductions in flight.
    partials: RefCell<CudaSlice<f64>>,
}

/// The scaled problem on the device.
pub struct CudaProblem {
    a: CsrDevice,
    at: CsrDevice,
    c: CudaSlice<f64>,
    lc: CudaSlice<f64>,
    uc: CudaSlice<f64>,
    lv: CudaSlice<f64>,
    uv: CudaSlice<f64>,
    primal_factor: CudaSlice<f64>,
    dual_factor: CudaSlice<f64>,
    /// cuSPARSE scratch, sized for both products.
    buffer: RefCell<CudaSlice<u8>>,
}

/// One reduction kernel's slots in the partials buffer.
#[derive(Clone, Copy)]
struct Slots {
    offset: usize,
    blocks: usize,
    values: usize,
}

impl CudaBackend {
    /// Opens device `ordinal`, compiles the kernels and creates the cuSPARSE
    /// handle. Returns an error, rather than panicking, when there's no
    /// driver, toolkit library or GPU.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        ensure_cuda_dlls_on_path();
        // cudarc's dynamic loader panics when a shared library is missing.
        std::panic::catch_unwind(|| Self::open(ordinal)).unwrap_or_else(|e| {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "a CUDA library failed to load".into());
            Err(CudaError::Unavailable(msg))
        })
    }

    fn open(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        // One stream only, so cudarc's cross-stream event tracking is pure
        // overhead.
        // SAFETY: every operation of this backend runs on `stream`.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let opts = CompileOptions { options: vec!["--std=c++17".into()], ..Default::default() };
        let module = ctx.load_module(compile_ptx_with_opts(SRC, opts)?)?;
        let k = Kernels::load(&module)?;
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let sms = sms.max(1) as u32;
        let reduce_blocks = (sms * 4).min(1024);
        let partials = stream.alloc_zeros::<f64>(reduce_blocks as usize * MAX_VALUES_PER_CALL)?;
        let sparse = Handle::new(&stream)?;
        Ok(Self { ctx, stream, k, sparse, max_blocks: sms * 32, reduce_blocks, partials: RefCell::new(partials) })
    }

    /// Like [`CudaBackend::new`], but `None` when no usable GPU is present.
    pub fn try_default() -> Option<Self> {
        Self::new(0).ok()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    fn grid(&self, n: usize) -> LaunchConfig {
        let blocks = n.div_ceil(BLOCK as usize).clamp(1, self.max_blocks as usize) as u32;
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }

    fn reduce_grid(&self, n: usize) -> (LaunchConfig, usize) {
        let blocks = n.div_ceil(BLOCK as usize).clamp(1, self.reduce_blocks as usize) as u32;
        (LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }, blocks as usize)
    }

    /// Downloads the partials of `slots` and returns each kernel's totals.
    fn collect(&self, slots: &[Slots]) -> Result<Vec<Vec<f64>>, CudaError> {
        let used = slots.iter().map(|s| s.offset + s.blocks * s.values).max().unwrap_or(0);
        let host = self.stream.clone_dtoh(&self.partials.borrow().slice(0..used))?;
        Ok(slots
            .iter()
            .map(|s| (0..s.values).map(|k| (0..s.blocks).map(|b| host[s.offset + b * s.values + k]).sum()).collect())
            .collect())
    }
}

/// Launches `f` on `cfg` with the given arguments. Arguments must be places
/// (locals or fields), not temporaries.
macro_rules! launch {
    ($stream:expr, $f:expr, $cfg:expr; $($arg:expr),* $(,)?) => {{
        let mut l = $stream.launch_builder(&$f);
        $( l.arg($arg); )*
        // SAFETY: argument types and counts match the kernel signatures in
        // kernels.cu, and every pointer argument is a live device buffer of
        // the length passed alongside it.
        unsafe { l.launch($cfg) }.map(|_| ()).map_err(CudaError::from)
    }};
}

/// Launches a reduction kernel over `n` elements writing `values` totals per
/// block at `*offset`, and advances `*offset`.
macro_rules! reduce {
    ($self:expr, $f:expr, $n:expr, $values:expr, $offset:expr; $($arg:expr),* $(,)?) => {{
        let n: usize = $n;
        let (cfg, blocks) = $self.reduce_grid(n);
        let slots = Slots { offset: *$offset, blocks, values: $values };
        let nn = n as i64;
        let base = slots.offset as i64;
        let mut partials = $self.partials.borrow_mut();
        launch!($self.stream, $f, cfg; $($arg,)* &nn, &mut *partials, &base)?;
        *$offset += blocks * $values;
        slots
    }};
}

impl Backend for CudaBackend {
    type Error = CudaError;
    type Vector = CudaSlice<f64>;
    type Problem = CudaProblem;

    fn name(&self) -> String {
        let name = self.ctx.name().unwrap_or_else(|_| "unknown GPU".into());
        format!("cuda ({name})")
    }

    fn load(&self, p: &ScaledProblem) -> Result<CudaProblem, CudaError> {
        let s = &self.stream;
        let a = CsrDevice::new(s, p.a.nrows, p.a.ncols, &p.a.row_ptr, &p.a.col_idx, &p.a.values)?;
        let at = CsrDevice::new(s, p.at.nrows, p.at.ncols, &p.at.row_ptr, &p.at.col_idx, &p.at.values)?;
        let (m, n) = (p.a.nrows, p.a.ncols);
        let (mut xm, mut xn) = (s.alloc_zeros::<f64>(m)?, s.alloc_zeros::<f64>(n)?);
        let size = a.buffer_size(&self.sparse, s, &xn, &mut xm)?.max(at.buffer_size(&self.sparse, s, &xm, &mut xn)?);
        Ok(CudaProblem {
            c: s.clone_htod(&p.c)?,
            lc: s.clone_htod(&p.lc)?,
            uc: s.clone_htod(&p.uc)?,
            lv: s.clone_htod(&p.lv)?,
            uv: s.clone_htod(&p.uv)?,
            primal_factor: s.clone_htod(&p.primal_residual_factors())?,
            dual_factor: s.clone_htod(&p.dual_residual_factors())?,
            buffer: RefCell::new(s.alloc_zeros::<u8>(size.max(1))?),
            a,
            at,
        })
    }

    fn upload(&self, data: &[f64]) -> Result<CudaSlice<f64>, CudaError> {
        Ok(self.stream.clone_htod(data)?)
    }

    fn download(&self, v: &CudaSlice<f64>) -> Result<Vec<f64>, CudaError> {
        Ok(self.stream.clone_dtoh(v)?)
    }

    fn copy(&self, src: &CudaSlice<f64>, dst: &mut CudaSlice<f64>) -> Result<(), CudaError> {
        Ok(self.stream.memcpy_dtod(src, dst)?)
    }

    fn spmv(&self, p: &CudaProblem, x: &CudaSlice<f64>, out: &mut CudaSlice<f64>) -> Result<(), CudaError> {
        p.a.mul(&self.sparse, &self.stream, x, out, &mut p.buffer.borrow_mut())
    }

    fn spmv_t(&self, p: &CudaProblem, y: &CudaSlice<f64>, out: &mut CudaSlice<f64>) -> Result<(), CudaError> {
        p.at.mul(&self.sparse, &self.stream, y, out, &mut p.buffer.borrow_mut())
    }

    fn primal_step(
        &self,
        p: &CudaProblem,
        x: &CudaSlice<f64>,
        aty: &CudaSlice<f64>,
        tau: f64,
        x_t: &mut CudaSlice<f64>,
        x_bar: &mut CudaSlice<f64>,
    ) -> Result<(), CudaError> {
        let n = x.len() as i64;
        launch!(self.stream, self.k.primal_step, self.grid(x.len()); x, aty, &p.c, &p.lv, &p.uv, &tau, x_t, x_bar, &n)
    }

    fn dual_step(
        &self,
        p: &CudaProblem,
        y: &CudaSlice<f64>,
        ax_bar: &CudaSlice<f64>,
        sigma: f64,
        y_t: &mut CudaSlice<f64>,
    ) -> Result<(), CudaError> {
        let m = y.len() as i64;
        launch!(self.stream, self.k.dual_step, self.grid(y.len()); y, ax_bar, &p.lc, &p.uc, &sigma, y_t, &m)
    }

    fn halpern_step(
        &self,
        z: &mut CudaSlice<f64>,
        z_t: &CudaSlice<f64>,
        anchor: &CudaSlice<f64>,
        w: f64,
        rho: f64,
    ) -> Result<(), CudaError> {
        let n = z.len() as i64;
        let cfg = self.grid(z.len());
        launch!(self.stream, self.k.halpern_step, cfg; z, z_t, anchor, &w, &rho, &n)
    }

    fn distance_sq(&self, a: &CudaSlice<f64>, b: &CudaSlice<f64>) -> Result<f64, CudaError> {
        let mut off = 0;
        let s = reduce!(self, self.k.red_distance_sq, a.len(), 1, &mut off; a, b);
        Ok(self.collect(&[s])?[0][0])
    }

    fn sub(&self, a: &CudaSlice<f64>, b: &CudaSlice<f64>, out: &mut CudaSlice<f64>) -> Result<(), CudaError> {
        let n = a.len() as i64;
        let cfg = self.grid(a.len());
        launch!(self.stream, self.k.sub, cfg; a, b, out, &n)
    }

    fn norm_sq(&self, v: &CudaSlice<f64>) -> Result<f64, CudaError> {
        self.dot(v, v)
    }

    fn dot(&self, a: &CudaSlice<f64>, b: &CudaSlice<f64>) -> Result<f64, CudaError> {
        let mut off = 0;
        let s = reduce!(self, self.k.red_dot, a.len(), 1, &mut off; a, b);
        Ok(self.collect(&[s])?[0][0])
    }

    fn scale(&self, v: &mut CudaSlice<f64>, alpha: f64) -> Result<(), CudaError> {
        let n = v.len() as i64;
        let cfg = self.grid(v.len());
        launch!(self.stream, self.k.scale, cfg; v, &alpha, &n)
    }

    fn primal_stats(&self, p: &CudaProblem, x: &CudaSlice<f64>, ax: &CudaSlice<f64>) -> Result<PrimalStats, CudaError> {
        let mut off = 0;
        let r = reduce!(self, self.k.red_primal_residual, ax.len(), 1, &mut off; ax, &p.lc, &p.uc, &p.primal_factor);
        let o = reduce!(self, self.k.red_dot, x.len(), 1, &mut off; &p.c, x);
        let t = self.collect(&[r, o])?;
        Ok(PrimalStats { residual_sq: t[0][0], objective: t[1][0] })
    }

    fn dual_stats(&self, p: &CudaProblem, y: &CudaSlice<f64>, aty: &CudaSlice<f64>) -> Result<DualStats, CudaError> {
        let mut off = 0;
        let rows = reduce!(self, self.k.red_dual_rows, y.len(), 1, &mut off; y, &p.lc, &p.uc);
        let cols = reduce!(self, self.k.red_dual_cols, aty.len(), 2, &mut off; &p.c, aty, &p.lv, &p.uv, &p.dual_factor);
        let t = self.collect(&[rows, cols])?;
        Ok(DualStats { residual_sq: t[1][0], objective: t[0][0] + t[1][1] })
    }

    fn step_dual_residual_sq(
        &self,
        p: &CudaProblem,
        x: &CudaSlice<f64>,
        x_t: &CudaSlice<f64>,
        aty: &CudaSlice<f64>,
        aty_t: &CudaSlice<f64>,
        tau: f64,
    ) -> Result<f64, CudaError> {
        let mut off = 0;
        let inv_tau = 1.0 / tau;
        let s = reduce!(self, self.k.red_step_dual, x.len(), 1, &mut off; x, x_t, aty, aty_t, &p.dual_factor, &inv_tau);
        Ok(self.collect(&[s])?[0][0])
    }

    fn primal_ray_stats(
        &self,
        p: &CudaProblem,
        dx: &CudaSlice<f64>,
        adx: &CudaSlice<f64>,
    ) -> Result<RayStats, CudaError> {
        let mut off = 0;
        let rows = reduce!(self, self.k.red_primal_ray_rows, adx.len(), 1, &mut off; adx, &p.lc, &p.uc);
        let cols = reduce!(self, self.k.red_primal_ray_cols, dx.len(), 3, &mut off; dx, &p.c, &p.lv, &p.uv);
        let t = self.collect(&[rows, cols])?;
        Ok(RayStats { objective: t[1][2], violation_sq: t[0][0] + t[1][0], norm_sq: t[1][1] })
    }

    fn dual_ray_stats(
        &self,
        p: &CudaProblem,
        dy: &CudaSlice<f64>,
        atdy: &CudaSlice<f64>,
    ) -> Result<RayStats, CudaError> {
        let mut off = 0;
        let rows = reduce!(self, self.k.red_dual_ray_rows, dy.len(), 3, &mut off; dy, &p.lc, &p.uc);
        let cols = reduce!(self, self.k.red_dual_ray_cols, atdy.len(), 2, &mut off; atdy, &p.lv, &p.uv);
        let t = self.collect(&[rows, cols])?;
        Ok(RayStats { objective: t[0][1] + t[1][1], violation_sq: t[0][0] + t[1][0], norm_sq: t[0][2] })
    }
}
