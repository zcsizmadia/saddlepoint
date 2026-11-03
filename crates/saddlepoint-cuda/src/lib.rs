//! CUDA backend for saddlepoint.
//!
//! Kernels are CUDA C (`kernels.cu`) compiled at run time with NVRTC and
//! loaded through cudarc, so building needs no nvcc step. The driver, NVRTC
//! and cuSPARSE libraries are loaded dynamically, so the crate also builds on
//! machines without CUDA. Each kernel is small and isolated so it can later
//! be ported to Rust GPU kernels.
//!
//! - [`CudaBackend`] implements [`saddlepoint_core::Backend`]. Vectors stay
//!   on the device, and only reduction results come back to the host.
//! - [`probe`] measures memory bandwidth and arithmetic rate (`sp-bandwidth`).

mod backend;
pub mod probe;
mod sparse;

pub use backend::{CudaBackend, CudaError, CudaProblem};
pub use probe::{DeviceInfo, ProbeResult, device_info, probe};

/// On Windows the CUDA installer does not always put the toolkit's `bin`
/// folder (which holds `nvrtc64_*.dll` and `cusparse64_*.dll`) on `PATH`.
/// Add `%CUDA_PATH%\bin` and `%CUDA_PATH%\bin\x64` before cudarc loads them.
/// No-op elsewhere.
pub(crate) fn ensure_cuda_dlls_on_path() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if !cfg!(windows) {
            return;
        }
        let Some(cuda) = std::env::var_os("CUDA_PATH") else { return };
        let cuda = std::path::PathBuf::from(cuda);
        let mut paths: Vec<_> =
            [cuda.join("bin"), cuda.join("bin").join("x64")].into_iter().filter(|p| p.is_dir()).collect();
        if let Some(old) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&old));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            // SAFETY: runs once, before this crate loads any CUDA library.
            // Callers must not modify the environment concurrently.
            unsafe { std::env::set_var("PATH", joined) };
        }
    });
}
