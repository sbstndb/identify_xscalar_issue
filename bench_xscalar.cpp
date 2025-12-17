#include <benchmark/benchmark.h>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/core/xnoalias.hpp>

// ============================================================================
// Benchmark for xtensor (runtime size)
// ============================================================================
template <typename T, std::size_t N>
static void BM_XScalarAdd_xtensor(benchmark::State& state) {
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({N});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({N});
    vec1.fill(1);

    for (auto _ : state) {
        xt::noalias(result) = vec1 + static_cast<T>(1.0);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetBytesProcessed(state.iterations() * N * sizeof(T));
    state.SetItemsProcessed(state.iterations() * N);
}

// ============================================================================
// Benchmark for xtensor_fixed (compile-time size)
// ============================================================================
template <typename T, std::size_t N>
static void BM_XScalarAdd_fixed(benchmark::State& state) {
    xt::xtensor_fixed<T, xt::xshape<N>> vec1;
    xt::xtensor_fixed<T, xt::xshape<N>> result;
    vec1.fill(1);

    for (auto _ : state) {
        xt::noalias(result) = vec1 + static_cast<T>(1.0);
        benchmark::DoNotOptimize(result.data());
    }

    state.SetBytesProcessed(state.iterations() * N * sizeof(T));
    state.SetItemsProcessed(state.iterations() * N);
}

// ============================================================================
// Registration macros
// ============================================================================

// Register xtensor benchmark for a specific size
#define REGISTER_XTENSOR_BENCHMARK(TYPE, SIZE) \
    BENCHMARK(BM_XScalarAdd_xtensor<TYPE, SIZE>) \
        ->Name("xtensor_" #TYPE "_" #SIZE) \
        ->Unit(benchmark::kNanosecond)

// Register xtensor_fixed benchmark for a specific size
#define REGISTER_FIXED_BENCHMARK(TYPE, SIZE) \
    BENCHMARK(BM_XScalarAdd_fixed<TYPE, SIZE>) \
        ->Name("fixed_" #TYPE "_" #SIZE) \
        ->Unit(benchmark::kNanosecond)

// Register all sizes for xtensor
#define REGISTER_ALL_XTENSOR(TYPE) \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 1); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 2); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 3); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 4); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 5); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 6); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 7); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 8); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 9); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 10); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 16); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 32); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 64); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 128); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 256); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 512); \
    REGISTER_XTENSOR_BENCHMARK(TYPE, 1024)

// Register all sizes for xtensor_fixed
#define REGISTER_ALL_FIXED(TYPE) \
    REGISTER_FIXED_BENCHMARK(TYPE, 1); \
    REGISTER_FIXED_BENCHMARK(TYPE, 2); \
    REGISTER_FIXED_BENCHMARK(TYPE, 3); \
    REGISTER_FIXED_BENCHMARK(TYPE, 4); \
    REGISTER_FIXED_BENCHMARK(TYPE, 5); \
    REGISTER_FIXED_BENCHMARK(TYPE, 6); \
    REGISTER_FIXED_BENCHMARK(TYPE, 7); \
    REGISTER_FIXED_BENCHMARK(TYPE, 8); \
    REGISTER_FIXED_BENCHMARK(TYPE, 9); \
    REGISTER_FIXED_BENCHMARK(TYPE, 10); \
    REGISTER_FIXED_BENCHMARK(TYPE, 16); \
    REGISTER_FIXED_BENCHMARK(TYPE, 32); \
    REGISTER_FIXED_BENCHMARK(TYPE, 64); \
    REGISTER_FIXED_BENCHMARK(TYPE, 128); \
    REGISTER_FIXED_BENCHMARK(TYPE, 256); \
    REGISTER_FIXED_BENCHMARK(TYPE, 512); \
    REGISTER_FIXED_BENCHMARK(TYPE, 1024)

// ============================================================================
// Register all benchmarks
// ============================================================================

// xtensor (runtime size)
REGISTER_ALL_XTENSOR(float);
REGISTER_ALL_XTENSOR(double);

// xtensor_fixed (compile-time size)
REGISTER_ALL_FIXED(float);
REGISTER_ALL_FIXED(double);
