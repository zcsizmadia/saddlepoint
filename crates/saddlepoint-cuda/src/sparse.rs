//! Thin safe wrapper over the cuSPARSE generic SpMV API: one handle bound to
//! the backend's stream, and one CSR descriptor per matrix (`A` and `Aᵀ`).
//! Dense-vector descriptors are host-side structs, created per call.
//!
//! cudarc exposes only raw bindings for cuSPARSE, so this module holds all of
//! the crate's cuSPARSE FFI.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cusparse::result::CusparseError;
use cudarc::cusparse::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::backend::CudaError;

/// Deterministic (bit-reproducible) CSR algorithm.
const ALG: sys::cusparseSpMVAlg_t = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2;
const F64: sys::cudaDataType = sys::cudaDataType::CUDA_R_64F;

pub(crate) struct Handle(sys::cusparseHandle_t);

impl Handle {
    pub(crate) fn new(stream: &CudaStream) -> Result<Self, CudaError> {
        let h = cudarc::cusparse::result::create()?;
        let handle = Self(h);
        unsafe { sys::cusparseSetStream(handle.0, stream.cu_stream() as _).result()? };
        Ok(handle)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::cusparse::result::destroy(self.0);
        }
    }
}

/// A CSR matrix on the device with its cuSPARSE descriptor. The device
/// arrays must outlive the descriptor, so they're owned here.
pub(crate) struct CsrDevice {
    desc: sys::cusparseSpMatDescr_t,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    _row_ptr: CudaSlice<i32>,
    _col_idx: CudaSlice<i32>,
    _values: CudaSlice<f64>,
}

impl CsrDevice {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        nrows: usize,
        ncols: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
    ) -> Result<Self, CudaError> {
        let to_i32 = |v: &[usize]| -> Result<Vec<i32>, CudaError> {
            v.iter().map(|&x| i32::try_from(x).map_err(|_| CudaError::TooLarge)).collect()
        };
        let nnz = values.len();
        let row_ptr = stream.clone_htod(&to_i32(row_ptr)?)?;
        // cuSPARSE rejects null pointers, even for an empty matrix.
        let col_idx = stream.clone_htod(&if nnz == 0 { vec![0] } else { to_i32(col_idx)? })?;
        let values = stream.clone_htod(&if nnz == 0 { vec![0.0] } else { values.to_vec() })?;
        let mut desc = std::ptr::null_mut();
        {
            let (rp, _g1) = row_ptr.device_ptr(stream);
            let (ci, _g2) = col_idx.device_ptr(stream);
            let (va, _g3) = values.device_ptr(stream);
            unsafe {
                sys::cusparseCreateCsr(
                    &mut desc,
                    nrows as i64,
                    ncols as i64,
                    nnz as i64,
                    rp as *mut c_void,
                    ci as *mut c_void,
                    va as *mut c_void,
                    sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                    F64,
                )
                .result()?;
            }
        }
        Ok(Self { desc, nrows, ncols, _row_ptr: row_ptr, _col_idx: col_idx, _values: values })
    }

    /// Scratch bytes cuSPARSE needs for `out = self · x`.
    pub(crate) fn buffer_size(
        &self,
        handle: &Handle,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
    ) -> Result<usize, CudaError> {
        let mut size = 0usize;
        self.with_vectors(stream, x, out, |vx, vy| unsafe {
            let (alpha, beta) = (1.0f64, 0.0f64);
            sys::cusparseSpMV_bufferSize(
                handle.0,
                sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha as *const f64 as *const c_void,
                self.desc as _,
                vx as _,
                &beta as *const f64 as *const c_void,
                vy,
                F64,
                ALG,
                &mut size,
            )
            .result()
        })?;
        Ok(size)
    }

    /// `out = self · x`.
    pub(crate) fn mul(
        &self,
        handle: &Handle,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        buffer: &mut CudaSlice<u8>,
    ) -> Result<(), CudaError> {
        let (buf, _gb) = buffer.device_ptr_mut(stream);
        self.with_vectors(stream, x, out, |vx, vy| unsafe {
            let (alpha, beta) = (1.0f64, 0.0f64);
            sys::cusparseSpMV(
                handle.0,
                sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha as *const f64 as *const c_void,
                self.desc as _,
                vx as _,
                &beta as *const f64 as *const c_void,
                vy,
                F64,
                ALG,
                buf as *mut c_void,
            )
            .result()
        })
    }

    fn with_vectors(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f64>,
        out: &mut CudaSlice<f64>,
        f: impl FnOnce(sys::cusparseDnVecDescr_t, sys::cusparseDnVecDescr_t) -> Result<(), CusparseError>,
    ) -> Result<(), CudaError> {
        assert_eq!(x.len(), self.ncols, "spmv: x has the wrong length");
        assert_eq!(out.len(), self.nrows, "spmv: out has the wrong length");
        let (xp, _gx) = x.device_ptr(stream);
        let (op, _go) = out.device_ptr_mut(stream);
        let mut vx = std::ptr::null_mut();
        let mut vy = std::ptr::null_mut();
        unsafe {
            sys::cusparseCreateDnVec(&mut vx, self.ncols as i64, xp as *mut c_void, F64).result()?;
            if let Err(e) = sys::cusparseCreateDnVec(&mut vy, self.nrows as i64, op as *mut c_void, F64).result() {
                let _ = sys::cusparseDestroyDnVec(vx as _);
                return Err(e.into());
            }
            let r = f(vx, vy);
            let _ = sys::cusparseDestroyDnVec(vx as _);
            let _ = sys::cusparseDestroyDnVec(vy as _);
            r?;
        }
        Ok(())
    }
}

impl Drop for CsrDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cusparseDestroySpMat(self.desc as _);
        }
    }
}
