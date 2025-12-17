# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmark suite comparing `xt::xtensor` (runtime-sized) vs `xt::xtensor_fixed` (compile-time sized) for scalar addition operations. Tests 18 compiler configurations (GCC 11-14, Clang 16-20) with/without XSIMD.

## Build Commands

```bash
# Build a specific configuration
./build.sh --compiler=gcc-14 --xsimd=ON --jobs=8

# Clean rebuild
./build.sh --compiler=clang-18 --xsimd=OFF --clean

# List existing builds
./build.sh --list
```

Build output: `build_<compiler><version>_<xsimd|noxsimd>/bench_xscalar`

## Running Benchmarks

```bash
# Run all benchmarks
./build_gcc14_xsimd/bench_xscalar

# Filter by type
./build_gcc14_xsimd/bench_xscalar --benchmark_filter="double"
./build_gcc14_xsimd/bench_xscalar --benchmark_filter="fixed"

# CSV output
./build_gcc14_xsimd/bench_xscalar --benchmark_format=csv
```

## Collecting Results Across All Configurations

```bash
# Run benchmarks and save output
./run_bench.sh > all_results.txt

# Parse results into comparison tables
python3 parse_results.py
```

## Project Files

| File | Purpose |
|------|---------|
| `bench_xscalar.cpp` | Benchmark source - tests `xtensor` and `xtensor_fixed` scalar addition |
| `build.sh` | Multi-compiler build script - downloads deps (xtl, xsimd, xtensor), builds configurations |
| `CMakeLists.txt` | CMake config - fetches Google Benchmark, links xtensor/xsimd |
| `run_bench.sh` | Runs benchmarks across all 18 build configurations |
| `parse_results.py` | Parses CSV output from `run_bench.sh`, generates comparison tables |
| `BENCHMARK_REPORT.md` | Full analysis with all data tables |

## Architecture

```
identify_xscalar_issue/
├── deps/                    # Auto-downloaded: xtl, xsimd, xtensor sources
├── install/<config>/        # Per-config installed headers
├── build_<config>/          # Per-config benchmark executables
└── *.sh, *.cpp, *.py        # Source files
```

The `build.sh` script:
1. Downloads xtl, xsimd, xtensor to `deps/`
2. Builds and installs them to `install/<config>/`
3. Builds the benchmark executable in `build_<config>/`

## Adding New Benchmark Sizes

Edit `bench_xscalar.cpp` - add sizes to `REGISTER_ALL_XTENSOR` and `REGISTER_ALL_FIXED` macros.

## Adding New Container Types

Add new benchmark template function (like `BM_XScalarAdd_xtensor`) and registration macros in `bench_xscalar.cpp`.
