# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmark suite comparing `xt::xtensor` (runtime-sized) vs `xt::xtensor_fixed` (compile-time sized) for scalar addition operations. Tests 18 compiler configurations (GCC 11-14, Clang 16-20) with/without XSIMD.

## Build Commands

```bash
# Build a specific configuration
./scripts/build.sh --compiler=gcc-14 --xsimd=ON --jobs=8

# Clean rebuild
./scripts/build.sh --compiler=clang-18 --xsimd=OFF --clean

# List existing builds
./scripts/build.sh --list
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
./scripts/run_bench.sh > all_results.txt

# Parse results into comparison tables
python3 scripts/parse_results.py
```

## VTune Profiling

```bash
# Collect VTune hotspots data
./scripts/run_vtune_analysis.sh collect --configs=gcc14_xsimd,gcc14_noxsimd

# Generate reports
./scripts/run_vtune_analysis.sh report

# List available options
./scripts/run_vtune_analysis.sh --help
```

## Project Structure

```
identify_xscalar_issue/
├── src/                         # Source code
│   └── bench_xscalar.cpp        # Benchmark source
├── scripts/                     # Automation scripts
│   ├── build.sh                 # Multi-compiler build script
│   ├── run_bench.sh             # Run benchmarks across configs
│   ├── run_all_benchmarks.sh    # CSV output for all configs
│   ├── run_vtune_analysis.sh    # VTune profiling script
│   └── parse_results.py         # Parse results into tables
├── docs/                        # Documentation
│   ├── BENCHMARK_REPORT.md      # Full analysis with data tables
│   └── reports/                 # Performance issue reports
│       └── ISSUE_001_*.md       # Detailed issue analysis
├── CMakeLists.txt               # CMake configuration
├── README.md                    # Project overview
├── CLAUDE.md                    # This file
├── deps/                        # Auto-downloaded dependencies (gitignored)
├── install/                     # Per-config installed headers (gitignored)
├── build_*/                     # Per-config build directories (gitignored)
└── analysis_results/            # VTune profiling results (gitignored)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/bench_xscalar.cpp` | Benchmark source - tests `xtensor` and `xtensor_fixed` scalar addition |
| `scripts/build.sh` | Multi-compiler build script - downloads deps, builds configurations |
| `scripts/run_vtune_analysis.sh` | VTune profiling for performance analysis |
| `docs/reports/ISSUE_001_*.md` | Detailed analysis of XSIMD overhead issue |
| `CMakeLists.txt` | CMake config - fetches Google Benchmark, links xtensor/xsimd |

## Build Process

The `scripts/build.sh` script:
1. Downloads xtl, xsimd, xtensor to `deps/`
2. Builds and installs them to `install/<config>/`
3. Builds the benchmark executable in `build_<config>/`

## Adding New Benchmark Sizes

Edit `src/bench_xscalar.cpp` - add sizes to `REGISTER_ALL_XTENSOR` and `REGISTER_ALL_FIXED` macros.

## Adding New Container Types

Add new benchmark template function (like `BM_XScalarAdd_xtensor`) and registration macros in `src/bench_xscalar.cpp`.
