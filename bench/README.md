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

2026-11-03, cuPDLPx `d17f53e` (defaults except presolve off). RTX A1000
Laptop GPU (4 GB, about 167 GB/s) and i7-12800H (20 threads, DDR5-4800).
Relative tolerance 1e-4, 300 s limit; "limit" means the time limit was hit.
Times are `solve_s`. Raw rows: [`results/2026-11-03-rtx-a1000-pid.csv`](results/2026-11-03-rtx-a1000-pid.csv)
(after #12), and [`results/2026-11-03-rtx-a1000.csv`](results/2026-11-03-rtx-a1000.csv) (before).

| instance | nnz | CPU | GPU | cuPDLPx | iterations (ours / cuPDLPx) | µs per iteration (GPU / cuPDLPx) |
|---|--:|--:|--:|--:|--:|--:|
| rail507 | 0.47M | 15.9 s | 0.76 s | **0.32 s** | 3,456 / 1,800 | 219 / 175 |
| irish-electricity | 0.52M | 90.0 s | **26.4 s** | limit | 86,592 / >1,067,600 | 305 / 281 |
| supportcase10 | 0.56M | 19.5 s | 6.68 s | **5.96 s** | 19,264 / 18,000 | 347 / 331 |
| ex10 | 1.16M | 0.56 s | 0.27 s | **0.25 s** | 448 / 600 | 607 / 423 |
| physiciansched3-3 | 1.06M | limit | limit | limit | – | 611 / 589 |
| s100 | 1.78M | limit | 176 s | **70.9 s** | 174,848 / 92,800 | 1,009 / 764 |
| savsched1 | 1.77M | 1.97 s | 0.91 s | **0.80 s** | 704 / 800 | 1,298 / 996 |
| neos-5052403-cygnet | 4.90M | 77.0 s | 28.6 s | **13.2 s** | 22,720 / 11,200 | 1,260 / 1,181 |
| square41 | 13.6M | limit | limit | limit | – | 2,939 / 2,836 |

CPU times on this laptop vary a lot between runs (rail507 took 3.5 s, 4.1 s
and 15.9 s for the same 3,456 iterations); iteration counts are exactly
reproducible. GPU times vary less but not nothing (s100 ran at 1,009 µs per
iteration here, 817 in the run before).

Objective error relative to HiGHS (highspy defaults, integrality dropped):

| instance | saddlepoint | cuPDLPx |
|---|--:|--:|
| rail507 | 8.9e-5 | 2.8e-4 |
| supportcase10 | 1.9e-5 | 3.2e-5 |
| ex10 | 3.8e-7 | 5.8e-5 |
| savsched1 | 8.1e-5 | 1.5e-5 |
| neos-5052403-cygnet | 1.3e-6 | 1.3e-4 |

HiGHS did not finish s100 or irish-electricity in 600 s. On s100 the PDHG
answers now agree (saddlepoint −0.1713, cuPDLPx −0.1710; HiGHS's unproven
incumbent is −0.1792). Before #12 saddlepoint stopped at −0.2398: at 1e-4
relative tolerance a small constraint violation still moves this objective
a lot, since it is small next to the norms the tolerance is relative to.

### What this says

- **Per iteration we are within 2–7% of cuPDLPx** on the large instances
  (cygnet, physiciansched3-3, square41). Short runs (ex10 and savsched1
  stop under 1,000 iterations) are dominated by fixed start-up work such as
  the norm estimate, and on small problems (rail507) kernel-launch
  overhead still shows.
- **The primal weight was stuck (fixed in #12).** On problems whose
  variables all have finite bounds, the projected dual residual is
  identically zero, so the PID controller's safeguard kept the initial
  weight forever. It now balances against cuPDLPx's step-based dual
  residual. That cut supportcase10 from 193,024 to 19,264 iterations and
  cygnet from 60,480 to 22,720.
- **The remaining iteration gap** is about 2x on cygnet, s100 and rail507.
  On irish-electricity the opposite holds: we converge in 87k iterations
  and cuPDLPx does not in 1M.
