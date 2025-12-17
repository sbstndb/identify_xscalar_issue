#!/bin/bash
cd /home/sbstndbs/identify_xscalar_issue

BUILDS="build_gcc11_xsimd build_gcc11_noxsimd build_gcc12_xsimd build_gcc12_noxsimd build_gcc13_xsimd build_gcc13_noxsimd build_gcc14_xsimd build_gcc14_noxsimd build_clang16_xsimd build_clang16_noxsimd build_clang17_xsimd build_clang17_noxsimd build_clang18_xsimd build_clang18_noxsimd build_clang19_xsimd build_clang19_noxsimd build_clang20_xsimd build_clang20_noxsimd"

for build in $BUILDS; do
    if [[ -x "${build}/bench_xscalar" ]]; then
        echo "=== $build ==="
        "${build}/bench_xscalar" --benchmark_filter="double" --benchmark_format=csv 2>/dev/null | tail -n +2
    fi
done
