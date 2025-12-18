#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "$PROJECT_DIR"

BUILDS=(
    "gcc11_xsimd"
    "gcc11_noxsimd"
    "gcc12_xsimd"
    "gcc12_noxsimd"
    "gcc13_xsimd"
    "gcc13_noxsimd"
    "gcc14_xsimd"
    "gcc14_noxsimd"
    "clang16_xsimd"
    "clang16_noxsimd"
    "clang17_xsimd"
    "clang17_noxsimd"
    "clang18_xsimd"
    "clang18_noxsimd"
    "clang19_xsimd"
    "clang19_noxsimd"
    "clang20_xsimd"
    "clang20_noxsimd"
)

echo "config,type,size,time_ns"
for build in "${BUILDS[@]}"; do
    exe="build_${build}/bench_xscalar"
    if [[ -x "$exe" ]]; then
        # Run benchmark and parse output
        "$exe" --benchmark_format=csv 2>/dev/null | tail -n +2 | while IFS=, read -r name iter real_time cpu_time time_unit rest; do
            # Parse name like XScalarAdd_double_16
            if [[ "$name" =~ XScalarAdd_([a-z]+)_([0-9]+) ]]; then
                type="${BASH_REMATCH[1]}"
                size="${BASH_REMATCH[2]}"
                echo "${build},${type},${size},${cpu_time}"
            fi
        done
    fi
done
