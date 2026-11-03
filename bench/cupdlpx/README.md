# cuPDLPx driver

`main.c` is a small command-line driver for [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx)
that `sp-bench --solvers cupdlpx` runs as the reference solver. cuPDLPx's own
CLI needs `getopt` and `libgen`, so it is not built under MSVC; this driver
calls the library directly and builds on Windows and Linux.

```
cupdlpx_run [--no_presolve] [--no_active_set_boost] [--no_infeasibility_detection] [--verbose]
            [--eps_opt X] [--eps_feas X] [--iter_limit N] [--time_limit S] [--eval_freq N] FILE.mps
```

It ends with one machine-readable line:

```
RESULT file=... status=OPTIMAL iters=1800 time=0.306 primal_obj=... dual_obj=... rel_primal=... rel_dual=... rel_gap=...
```

## Build

Build cuPDLPx first (CMake, CUDA toolkit; the baseline in `bench/README.md`
used commit `d17f53e`):

```
git clone https://github.com/MIT-Lu-Lab/cuPDLPx
cmake -S cuPDLPx -B cuPDLPx/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cuPDLPx/build
```

On Windows run both commands, and the one below, from a Visual Studio
developer shell (`Enter-VsDevShell` in PowerShell; running `vcvars64.bat`
from PowerShell does not keep its environment). Then:

```
# Windows (MSVC)
cl /nologo /O2 /std:c11 /I cuPDLPx\include main.c /Fe:cupdlpx_run.exe /link cuPDLPx\build\cupdlpx.lib

# Linux (untested so far)
cc -O2 -std=c11 -I cuPDLPx/include main.c -o cupdlpx_run -L cuPDLPx/build -lcupdlpx -Wl,-rpath,$PWD/cuPDLPx/build
```

On Windows the driver needs these folders on `PATH` to find its DLLs:
`cuPDLPx\build`, `cuPDLPx\build\_deps\zlib-build`, `cuPDLPx\build\_deps\pslp-build`
and `%CUDA_PATH%\bin\x64`.
