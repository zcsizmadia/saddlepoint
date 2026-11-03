# Benchmarks

LP relaxations of nine MIPLIB 2017 instances (integrality is ignored), from
0.5M to 13.6M nonzeros, solved by saddlepoint on the CPU and the GPU and by
[cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) without presolve, so the
PDHG cores are compared on the same input.

## Reproduce

```
sh scripts/fetch_miplib.sh                       # data/miplib/*.mps, about 1.3 GB unpacked
cargo build --release -p saddlepoint-cli
target/release/sp-bench --solvers cpu,gpu,cupdlpx --cupdlpx PATH/TO/cupdlpx_run \
    --tol 1e-4 --time-limit 300 --out results.csv data/miplib/*.mps
```

`sp-bench` writes one CSV row per (instance, solver). For saddlepoint the
residuals come from the independent KKT checker (`saddlepoint-check`) run on
the returned solution; for cuPDLPx they are what it reports. `solve_s` is each
solver's own clock; `total_s` adds scaling and upload for saddlepoint, and
file reading and CUDA start-up for cuPDLPx. Building the cuPDLPx driver is
described in [`cupdlpx/README.md`](cupdlpx/README.md).

## Results

2026-11-03, saddlepoint `bench-miplib` branch, cuPDLPx `d17f53e` (defaults
except presolve off). RTX A1000 Laptop GPU (4 GB, about 167 GB/s) and
i7-12800H (20 threads, DDR5-4800). Relative tolerance 1e-4, 300 s limit;
"limit" means the time limit was hit. Times are `solve_s`; the raw rows are in [`results/2026-11-03-rtx-a1000.csv`](results/2026-11-03-rtx-a1000.csv).

| instance | nnz | CPU | GPU | cuPDLPx | iterations (ours / cuPDLPx) | µs per iteration (GPU / cuPDLPx) |
|---|--:|--:|--:|--:|--:|--:|
| rail507 | 0.47M | 3.89 s | 0.89 s | **0.32 s** | 4,288 / 1,800 | 207 / 175 |
| irish-electricity | 0.52M | 82.6 s | **26.0 s** | limit | 86,592 / >1,067,600 | 300 / 281 |
| supportcase10 | 0.56M | 219 s | 66.1 s | **5.96 s** | 193,024 / 18,000 | 342 / 331 |
| ex10 | 1.16M | 1.89 s | 0.58 s | **0.25 s** | 512 / 600 | 1,137 / 423 |
| physiciansched3-3 | 1.06M | limit | limit | limit | – | 604 / 589 |
| s100 | 1.78M | limit | 207 s | **70.9 s** | 253,184 / 92,800 | 817 / 764 |
| savsched1 | 1.77M | 7.67 s | 0.91 s | **0.80 s** | 704 / 800 | 1,297 / 996 |
| neos-5052403-cygnet | 4.90M | 222 s | 75.5 s | **13.2 s** | 60,480 / 11,200 | 1,248 / 1,181 |
| square41 | 13.6M | limit | limit | limit | – | 2,918 / 2,836 |

Objective error relative to HiGHS (highspy defaults, integrality dropped):

| instance | saddlepoint | cuPDLPx |
|---|--:|--:|
| rail507 | 5.0e-5 | 2.8e-4 |
| supportcase10 | 3.5e-4 | 3.2e-5 |
| ex10 | 5.0e-7 | 5.8e-5 |
| savsched1 | 8.1e-5 | 1.5e-5 |
| neos-5052403-cygnet | 1.0e-7 | 1.3e-4 |

HiGHS did not finish s100 or irish-electricity in 600 s. On s100 the two
PDHG answers differ (saddlepoint −0.2398, cuPDLPx −0.1710, HiGHS's
unproven incumbent −0.1792): at 1e-4 relative tolerance a small constraint
violation still moves this objective a lot, since it is small next to the
norms the tolerance is relative to.

### What this says

- **Per iteration we are within 2–7% of cuPDLPx** once a problem has a
  million or more nonzeros (s100, cygnet, physiciansched3-3, square41).
  Short runs (ex10 and savsched1 stop under 1,000 iterations) are dominated
  by fixed start-up work such as the norm estimate, and on small problems
  (rail507) kernel-launch overhead still shows.
- **The gap is iteration count.** cuPDLPx needs 5–11x fewer iterations on
  cygnet and supportcase10, and 2–3x fewer on s100 and rail507. On
  irish-electricity the opposite happens: we converge in 87k iterations and
  cuPDLPx does not in 1M. So the next large win is algorithmic (restart
  and primal-weight behaviour on these instances), not kernels.
- The GPU is 3–8x faster than our CPU backend here, iteration for iteration
  the same algorithm.
