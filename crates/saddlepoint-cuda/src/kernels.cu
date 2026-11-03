// saddlepoint CUDA kernels, compiled at run time with NVRTC.
//
// Every kernel is one pass over memory with a grid-stride loop. Reductions
// write one partial per block (K values per block, at out[base + block*K + k]);
// the host adds the partials. Each kernel mirrors one CpuBackend method, which
// is the reference.

#define BLOCK 256
typedef long long i64;

#define GRID_LOOP(i, n) for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < (n); i += (i64)gridDim.x * blockDim.x)

__device__ __forceinline__ double clampd(double v, double lo, double hi) { return fmin(fmax(v, lo), hi); }

// Distance from v to [lo, hi], signed: negative below, positive above.
__device__ __forceinline__ double excess(double v, double lo, double hi) {
    return v < lo ? v - lo : (v > hi ? v - hi : 0.0);
}

// Bound term v+ lo - v- hi, skipping infinite bounds.
__device__ __forceinline__ double bound_term(double v, double lo, double hi) {
    if (v > 0.0 && isfinite(lo)) return v * lo;
    if (v < 0.0 && isfinite(hi)) return v * hi;
    return 0.0;
}

// Part of v whose sign an infinite bound forbids.
__device__ __forceinline__ double forbidden(double v, double lo, double hi) {
    return ((v > 0.0 && isinf(lo) && lo < 0.0) || (v < 0.0 && isinf(hi) && hi > 0.0)) ? v : 0.0;
}

// How far v leaves the recession cone of [lo, hi].
__device__ __forceinline__ double recession_violation(double v, double lo, double hi) {
    if (isfinite(lo) && v < 0.0) return v;
    if (isfinite(hi) && v > 0.0) return v;
    return 0.0;
}

__device__ __forceinline__ double warp_sum(double s) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    return s;
}

// Sums v[0..K) over the block and writes the K totals for this block.
template <int K>
__device__ void block_store(const double (&v)[K], double* out, i64 base) {
    __shared__ double sh[K][BLOCK / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double s = warp_sum(v[k]);
        if (lane == 0) sh[k][warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double s = warp_sum(lane < BLOCK / 32 ? sh[k][lane] : 0.0);
            if (lane == 0) out[base + (i64)blockIdx.x * K + k] = s;
        }
    }
}

// ---- element-wise -------------------------------------------------------

// x_t = proj_[lv,uv](x - tau (c - aty)),  x_bar = 2 x_t - x
extern "C" __global__ void primal_step(const double* __restrict__ x, const double* __restrict__ aty,
                                       const double* __restrict__ c, const double* __restrict__ lv,
                                       const double* __restrict__ uv, double tau, double* __restrict__ x_t,
                                       double* __restrict__ x_bar, i64 n) {
    GRID_LOOP(j, n) {
        const double xj = x[j];
        const double t = clampd(xj - tau * (c[j] - aty[j]), lv[j], uv[j]);
        x_t[j] = t;
        x_bar[j] = 2.0 * t - xj;
    }
}

// y_t = y - sigma (ax_bar - proj_[lc,uc](ax_bar - y / sigma))
extern "C" __global__ void dual_step(const double* __restrict__ y, const double* __restrict__ ax_bar,
                                     const double* __restrict__ lc, const double* __restrict__ uc, double sigma,
                                     double* __restrict__ y_t, i64 m) {
    GRID_LOOP(i, m) {
        const double a = ax_bar[i], yi = y[i];
        const double v = a - yi / sigma;
        y_t[i] = yi - sigma * (a - clampd(v, lc[i], uc[i]));
    }
}

// z = w ((1 + rho) z_t - rho z) + (1 - w) anchor
extern "C" __global__ void halpern_step(double* __restrict__ z, const double* __restrict__ z_t,
                                        const double* __restrict__ anchor, double w, double rho, i64 n) {
    GRID_LOOP(i, n) { z[i] = w * ((1.0 + rho) * z_t[i] - rho * z[i]) + (1.0 - w) * anchor[i]; }
}

extern "C" __global__ void sub(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out,
                               i64 n) {
    GRID_LOOP(i, n) { out[i] = a[i] - b[i]; }
}

extern "C" __global__ void scale(double* __restrict__ v, double alpha, i64 n) {
    GRID_LOOP(i, n) { v[i] *= alpha; }
}

// ---- reductions ---------------------------------------------------------

extern "C" __global__ void red_distance_sq(const double* __restrict__ a, const double* __restrict__ b, i64 n,
                                           double* out, i64 base) {
    double s[1] = {0.0};
    GRID_LOOP(i, n) {
        const double d = a[i] - b[i];
        s[0] += d * d;
    }
    block_store<1>(s, out, base);
}

extern "C" __global__ void red_dot(const double* __restrict__ a, const double* __restrict__ b, i64 n, double* out,
                                   i64 base) {
    double s[1] = {0.0};
    GRID_LOOP(i, n) { s[0] += a[i] * b[i]; }
    block_store<1>(s, out, base);
}

// sum (excess(ax, lc, uc) * factor)^2
extern "C" __global__ void red_primal_residual(const double* __restrict__ ax, const double* __restrict__ lc,
                                               const double* __restrict__ uc, const double* __restrict__ factor,
                                               i64 m, double* out, i64 base) {
    double s[1] = {0.0};
    GRID_LOOP(i, m) {
        const double e = excess(ax[i], lc[i], uc[i]) * factor[i];
        s[0] += e * e;
    }
    block_store<1>(s, out, base);
}

// sum bound_term(y, lc, uc)
extern "C" __global__ void red_dual_rows(const double* __restrict__ y, const double* __restrict__ lc,
                                         const double* __restrict__ uc, i64 m, double* out, i64 base) {
    double s[1] = {0.0};
    GRID_LOOP(i, m) { s[0] += bound_term(y[i], lc[i], uc[i]); }
    block_store<1>(s, out, base);
}

// r = c - aty: [sum (forbidden(r) * factor)^2, sum bound_term(r, lv, uv)]
extern "C" __global__ void red_dual_cols(const double* __restrict__ c, const double* __restrict__ aty,
                                         const double* __restrict__ lv, const double* __restrict__ uv,
                                         const double* __restrict__ factor, i64 n, double* out, i64 base) {
    double s[2] = {0.0, 0.0};
    GRID_LOOP(j, n) {
        const double r = c[j] - aty[j];
        const double f = forbidden(r, lv[j], uv[j]) * factor[j];
        s[0] += f * f;
        s[1] += bound_term(r, lv[j], uv[j]);
    }
    block_store<2>(s, out, base);
}

// sum recession_violation(adx, lc, uc)^2
extern "C" __global__ void red_primal_ray_rows(const double* __restrict__ adx, const double* __restrict__ lc,
                                               const double* __restrict__ uc, i64 m, double* out, i64 base) {
    double s[1] = {0.0};
    GRID_LOOP(i, m) {
        const double v = recession_violation(adx[i], lc[i], uc[i]);
        s[0] += v * v;
    }
    block_store<1>(s, out, base);
}

// [sum recession_violation(dx, lv, uv)^2, sum dx^2, sum c dx]
extern "C" __global__ void red_primal_ray_cols(const double* __restrict__ dx, const double* __restrict__ c,
                                               const double* __restrict__ lv, const double* __restrict__ uv, i64 n,
                                               double* out, i64 base) {
    double s[3] = {0.0, 0.0, 0.0};
    GRID_LOOP(j, n) {
        const double d = dx[j];
        const double v = recession_violation(d, lv[j], uv[j]);
        s[0] += v * v;
        s[1] += d * d;
        s[2] += c[j] * d;
    }
    block_store<3>(s, out, base);
}

// [sum forbidden(dy, lc, uc)^2, sum bound_term(dy, lc, uc), sum dy^2]
extern "C" __global__ void red_dual_ray_rows(const double* __restrict__ dy, const double* __restrict__ lc,
                                             const double* __restrict__ uc, i64 m, double* out, i64 base) {
    double s[3] = {0.0, 0.0, 0.0};
    GRID_LOOP(i, m) {
        const double d = dy[i];
        const double f = forbidden(d, lc[i], uc[i]);
        s[0] += f * f;
        s[1] += bound_term(d, lc[i], uc[i]);
        s[2] += d * d;
    }
    block_store<3>(s, out, base);
}

// r = -atdy: [sum forbidden(r, lv, uv)^2, sum bound_term(r, lv, uv)]
extern "C" __global__ void red_dual_ray_cols(const double* __restrict__ atdy, const double* __restrict__ lv,
                                             const double* __restrict__ uv, i64 n, double* out, i64 base) {
    double s[2] = {0.0, 0.0};
    GRID_LOOP(j, n) {
        const double r = -atdy[j];
        const double f = forbidden(r, lv[j], uv[j]);
        s[0] += f * f;
        s[1] += bound_term(r, lv[j], uv[j]);
    }
    block_store<2>(s, out, base);
}
