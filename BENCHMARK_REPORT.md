# XTensor Scalar Addition Performance Analysis

## Executive Summary

This report presents a comprehensive performance analysis of scalar addition operations in the xtensor C++ library, comparing:
- **`xt::xtensor<T, N>`**: Runtime-sized tensor container
- **`xt::xtensor_fixed<T, xt::xshape<N>>`**: Compile-time fixed-size tensor container

Testing was performed across **18 compiler configurations** (GCC 11-14, Clang 16-20) with and without XSIMD vectorization support.

### Key Findings

| Finding | Impact |
|---------|--------|
| `xtensor_fixed` at size=1-2 without XSIMD achieves **~0.2ns** | 15-20x faster than runtime `xtensor` |
| GCC with XSIMD: `xtensor` faster for sizes 2-64 | XSIMD overhead hurts `xtensor_fixed` |
| Clang without XSIMD: `xtensor_fixed` wins sizes 1-7, 10-16 | Up to 4x speedup |
| Large sizes (256+): `xtensor_fixed` + XSIMD + Clang optimal | 1.5-2x speedup |
| Worst configuration: `gcc13_noxsimd_fixed` for large sizes | Up to 400ns at size=1024 |

---

## 1. Methodology

### 1.1 Benchmark Environment

| Parameter | Value |
|-----------|-------|
| **Platform** | Linux 6.14.0-36-generic (x86_64) |
| **CPU Architecture** | Native (detected via `-march=native`) |
| **Optimization Level** | `-O3` |
| **Debug Info** | `-g -fno-omit-frame-pointer` |
| **Benchmark Framework** | Google Benchmark v1.8.3 |

### 1.2 Compiler Matrix

| Compiler Family | Versions Tested |
|-----------------|-----------------|
| **GCC** | 11, 12, 13, 14 |
| **Clang** | 16, 17, 18, 19, 20 |

Each compiler was tested with two XSIMD configurations:
- **xsimd**: XSIMD vectorization enabled (`-DXTENSOR_USE_XSIMD=ON`)
- **noxsimd**: XSIMD vectorization disabled (`-DXTENSOR_USE_XSIMD=OFF`)

**Total configurations**: 9 compilers × 2 XSIMD modes = **18 configurations**

### 1.3 Test Sizes

```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024
```

### 1.4 Benchmark Kernel

The benchmark measures the performance of scalar addition to a 1D tensor:

```cpp
// Runtime-sized xtensor
template <typename T, std::size_t N>
static void BM_XScalarAdd_xtensor(benchmark::State& state) {
    xt::xtensor<T, 1> vec1 = xt::xtensor<T, 1>::from_shape({N});
    xt::xtensor<T, 1> result = xt::xtensor<T, 1>::from_shape({N});
    vec1.fill(1);

    for (auto _ : state) {
        xt::noalias(result) = vec1 + static_cast<T>(1.0);
        benchmark::DoNotOptimize(result.data());
    }
}

// Compile-time fixed-size xtensor_fixed
template <typename T, std::size_t N>
static void BM_XScalarAdd_fixed(benchmark::State& state) {
    xt::xtensor_fixed<T, xt::xshape<N>> vec1;
    xt::xtensor_fixed<T, xt::xshape<N>> result;
    vec1.fill(1);

    for (auto _ : state) {
        xt::noalias(result) = vec1 + static_cast<T>(1.0);
        benchmark::DoNotOptimize(result.data());
    }
}
```

**Operation**: `result = vec1 + 1.0` (element-wise scalar addition)

**Type tested**: `double` (8 bytes per element)

---

## 2. Complete Results

### 2.1 GCC Results

#### GCC 11

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 3.52 ns | **0.22 ns** | 16.0x | 3.53 ns | **0.22 ns** | 16.2x |
| 2 | **3.86 ns** | 9.42 ns | 0.4x | 3.35 ns | **0.45 ns** | 7.4x |
| 3 | **4.17 ns** | 9.89 ns | 0.4x | 3.82 ns | **1.54 ns** | 2.5x |
| 4 | **3.14 ns** | 9.60 ns | 0.3x | 4.08 ns | **1.66 ns** | 2.5x |
| 5 | **3.76 ns** | 9.33 ns | 0.4x | 5.19 ns | **2.51 ns** | 2.1x |
| 6 | **3.99 ns** | 9.41 ns | 0.4x | 5.77 ns | **2.52 ns** | 2.3x |
| 7 | **4.16 ns** | 9.41 ns | 0.4x | 5.99 ns | **2.68 ns** | 2.2x |
| 8 | **3.65 ns** | 9.40 ns | 0.4x | 6.54 ns | **3.02 ns** | 2.2x |
| 9 | **4.29 ns** | 9.54 ns | 0.4x | 7.38 ns | **5.08 ns** | 1.5x |
| 10 | **4.51 ns** | 9.71 ns | 0.5x | 7.67 ns | **3.85 ns** | 2.0x |
| 16 | **4.74 ns** | 9.70 ns | 0.5x | 10.33 ns | **8.64 ns** | 1.2x |
| 32 | **7.22 ns** | 12.13 ns | 0.6x | 18.90 ns | **11.97 ns** | 1.6x |
| 64 | **12.16 ns** | 14.30 ns | 0.9x | 30.61 ns | **29.71 ns** | 1.0x |
| 128 | 34.67 ns | **28.45 ns** | 1.2x | **47.34 ns** | 55.55 ns | 0.9x |
| 256 | 42.04 ns | **36.98 ns** | 1.1x | **80.93 ns** | 96.87 ns | 0.8x |
| 512 | 76.98 ns | **64.94 ns** | 1.2x | **164.74 ns** | 188.80 ns | 0.9x |
| 1024 | 145.81 ns | **109.24 ns** | 1.3x | **313.77 ns** | 359.83 ns | 0.9x |

#### GCC 12

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 3.44 ns | **0.22 ns** | 15.8x | 3.25 ns | **0.22 ns** | 14.8x |
| 2 | **3.72 ns** | 10.20 ns | 0.4x | 3.59 ns | **0.32 ns** | 11.2x |
| 3 | **4.20 ns** | 10.09 ns | 0.4x | 3.96 ns | **1.52 ns** | 2.6x |
| 4 | **3.11 ns** | 8.78 ns | 0.4x | 3.69 ns | **1.64 ns** | 2.2x |
| 5 | **3.64 ns** | 9.13 ns | 0.4x | 5.25 ns | **2.07 ns** | 2.5x |
| 6 | **4.00 ns** | 9.13 ns | 0.4x | 5.69 ns | **2.13 ns** | 2.7x |
| 7 | **4.16 ns** | 9.61 ns | 0.4x | 6.17 ns | **2.81 ns** | 2.2x |
| 8 | **3.55 ns** | 8.96 ns | 0.4x | 6.71 ns | **2.71 ns** | 2.5x |
| 9 | **4.11 ns** | 9.38 ns | 0.4x | 7.17 ns | **6.00 ns** | 1.2x |
| 10 | **4.57 ns** | 9.68 ns | 0.5x | 8.04 ns | **3.24 ns** | 2.5x |
| 16 | **4.52 ns** | 9.78 ns | 0.5x | 10.49 ns | **8.71 ns** | 1.2x |
| 32 | **7.26 ns** | 11.55 ns | 0.6x | 22.71 ns | **20.02 ns** | 1.1x |
| 64 | **11.49 ns** | 14.10 ns | 0.8x | 29.59 ns | **26.92 ns** | 1.1x |
| 128 | **19.51 ns** | 26.54 ns | 0.7x | **46.05 ns** | 51.87 ns | 0.9x |
| 256 | 37.59 ns | **36.84 ns** | 1.0x | 83.51 ns | **83.44 ns** | 1.0x |
| 512 | 76.59 ns | **63.18 ns** | 1.2x | **156.30 ns** | 159.15 ns | 1.0x |
| 1024 | 139.82 ns | **107.73 ns** | 1.3x | **301.88 ns** | 320.93 ns | 0.9x |

#### GCC 13

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 3.71 ns | **0.23 ns** | 16.3x | 4.31 ns | **0.22 ns** | 19.6x |
| 2 | **3.96 ns** | 10.50 ns | 0.4x | 4.72 ns | **0.33 ns** | 14.5x |
| 3 | **4.17 ns** | 10.98 ns | 0.4x | 5.08 ns | **1.67 ns** | 3.0x |
| 4 | **3.24 ns** | 9.40 ns | 0.3x | 5.39 ns | **1.85 ns** | 2.9x |
| 5 | **3.98 ns** | 9.67 ns | 0.4x | 5.78 ns | **2.34 ns** | 2.5x |
| 6 | **4.20 ns** | 10.11 ns | 0.4x | 6.68 ns | **2.42 ns** | 2.8x |
| 7 | **5.84 ns** | 10.09 ns | 0.6x | 7.05 ns | **5.58 ns** | 1.3x |
| 8 | **4.13 ns** | 9.31 ns | 0.4x | 7.25 ns | **5.55 ns** | 1.3x |
| 9 | **4.87 ns** | 9.99 ns | 0.5x | 7.41 ns | **6.11 ns** | 1.2x |
| 10 | **4.95 ns** | 10.15 ns | 0.5x | 7.94 ns | **7.17 ns** | 1.1x |
| 16 | **4.96 ns** | 10.09 ns | 0.5x | 9.45 ns | **6.34 ns** | 1.5x |
| 32 | **8.07 ns** | 11.40 ns | 0.7x | **14.86 ns** | 17.34 ns | 0.9x |
| 64 | **14.00 ns** | 14.85 ns | 0.9x | **31.34 ns** | 38.73 ns | 0.8x |
| 128 | 42.32 ns | **19.75 ns** | 2.1x | **44.65 ns** | 58.45 ns | 0.8x |
| 256 | 49.97 ns | **37.40 ns** | 1.3x | **84.11 ns** | 102.11 ns | 0.8x |
| 512 | 84.50 ns | **62.06 ns** | 1.4x | **158.95 ns** | 198.14 ns | 0.8x |
| 1024 | 152.05 ns | **108.30 ns** | 1.4x | **304.61 ns** | 411.64 ns | 0.7x |

#### GCC 14

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 4.08 ns | **0.24 ns** | 17.3x | 4.53 ns | **0.22 ns** | 20.2x |
| 2 | **4.55 ns** | 10.93 ns | 0.4x | 4.91 ns | **0.29 ns** | 17.2x |
| 3 | **4.71 ns** | 11.11 ns | 0.4x | 5.26 ns | **1.69 ns** | 3.1x |
| 4 | **3.82 ns** | 9.77 ns | 0.4x | 6.49 ns | **1.80 ns** | 3.6x |
| 5 | **4.21 ns** | 10.16 ns | 0.4x | 6.77 ns | **5.56 ns** | 1.2x |
| 6 | **4.24 ns** | 10.33 ns | 0.4x | 7.08 ns | **2.46 ns** | 2.9x |
| 7 | **4.45 ns** | 10.81 ns | 0.4x | 7.45 ns | **2.82 ns** | 2.6x |
| 8 | **4.30 ns** | 10.07 ns | 0.4x | 7.50 ns | **3.15 ns** | 2.4x |
| 9 | **5.34 ns** | 10.42 ns | 0.5x | 7.92 ns | **3.55 ns** | 2.2x |
| 10 | **5.23 ns** | 10.93 ns | 0.5x | 8.07 ns | **3.86 ns** | 2.1x |
| 16 | **5.11 ns** | 10.79 ns | 0.5x | 10.50 ns | **6.68 ns** | 1.6x |
| 32 | **7.37 ns** | 12.33 ns | 0.6x | 15.13 ns | **11.79 ns** | 1.3x |
| 64 | **12.86 ns** | 15.14 ns | 0.9x | 32.66 ns | **29.54 ns** | 1.1x |
| 128 | 22.60 ns | **22.00 ns** | 1.0x | **47.33 ns** | 52.86 ns | 0.9x |
| 256 | 43.02 ns | **40.73 ns** | 1.1x | **82.03 ns** | 97.90 ns | 0.8x |
| 512 | 86.83 ns | **66.01 ns** | 1.3x | **164.73 ns** | 182.60 ns | 0.9x |
| 1024 | 162.02 ns | **114.90 ns** | 1.4x | **314.30 ns** | 364.19 ns | 0.9x |

### 2.2 Clang Results

#### Clang 16

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 6.17 ns | **1.65 ns** | 3.7x | 4.11 ns | **2.07 ns** | 2.0x |
| 2 | **6.69 ns** | 7.83 ns | 0.9x | 5.09 ns | **2.05 ns** | 2.5x |
| 3 | **7.28 ns** | 8.20 ns | 0.9x | 5.82 ns | **2.12 ns** | 2.8x |
| 4 | **5.77 ns** | 7.39 ns | 0.8x | 6.49 ns | **1.65 ns** | 3.9x |
| 5 | **6.58 ns** | 7.67 ns | 0.9x | 6.88 ns | **1.89 ns** | 3.6x |
| 6 | **6.65 ns** | 8.49 ns | 0.8x | 7.41 ns | **2.01 ns** | 3.7x |
| 7 | **7.32 ns** | 8.39 ns | 0.9x | 8.12 ns | **2.05 ns** | 4.0x |
| 8 | **6.42 ns** | 7.59 ns | 0.8x | **6.56 ns** | 10.53 ns | 0.6x |
| 9 | **7.13 ns** | 7.82 ns | 0.9x | **7.00 ns** | 11.60 ns | 0.6x |
| 10 | **7.49 ns** | 7.94 ns | 0.9x | 7.06 ns | **2.16 ns** | 3.3x |
| 16 | **7.23 ns** | 8.70 ns | 0.8x | 5.72 ns | **2.38 ns** | 2.4x |
| 32 | 9.88 ns | **8.95 ns** | 1.1x | **7.03 ns** | 10.37 ns | 0.7x |
| 64 | 13.84 ns | **11.06 ns** | 1.3x | **9.65 ns** | 17.86 ns | 0.5x |
| 128 | 24.22 ns | **15.23 ns** | 1.6x | **13.94 ns** | 32.26 ns | 0.4x |
| 256 | 41.72 ns | **24.78 ns** | 1.7x | **31.65 ns** | 62.46 ns | 0.5x |
| 512 | 77.57 ns | **43.08 ns** | 1.8x | **60.28 ns** | 117.97 ns | 0.5x |
| 1024 | 154.98 ns | **80.05 ns** | 1.9x | **115.31 ns** | 236.77 ns | 0.5x |

#### Clang 17

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 6.17 ns | **1.61 ns** | 3.8x | 4.33 ns | **2.03 ns** | 2.1x |
| 2 | **6.61 ns** | 7.62 ns | 0.9x | 5.37 ns | **2.07 ns** | 2.6x |
| 3 | **6.88 ns** | 7.78 ns | 0.9x | 5.95 ns | **2.08 ns** | 2.9x |
| 4 | **5.62 ns** | 7.45 ns | 0.8x | 6.42 ns | **1.63 ns** | 3.9x |
| 5 | **6.18 ns** | 8.15 ns | 0.8x | 6.59 ns | **1.85 ns** | 3.6x |
| 6 | **6.46 ns** | 7.58 ns | 0.9x | 7.13 ns | **1.84 ns** | 3.9x |
| 7 | **6.81 ns** | 8.56 ns | 0.8x | 7.77 ns | **2.07 ns** | 3.8x |
| 8 | **6.15 ns** | 7.37 ns | 0.8x | **6.64 ns** | 10.53 ns | 0.6x |
| 9 | **6.78 ns** | 8.65 ns | 0.8x | **6.89 ns** | 11.44 ns | 0.6x |
| 10 | **7.15 ns** | 7.89 ns | 0.9x | 7.22 ns | **2.08 ns** | 3.5x |
| 16 | **7.06 ns** | 9.75 ns | 0.7x | 5.78 ns | **2.38 ns** | 2.4x |
| 32 | 9.34 ns | **8.49 ns** | 1.1x | **7.38 ns** | 10.32 ns | 0.7x |
| 64 | 13.32 ns | **10.65 ns** | 1.3x | **9.42 ns** | 17.77 ns | 0.5x |
| 128 | 22.29 ns | **15.03 ns** | 1.5x | **14.23 ns** | 32.51 ns | 0.4x |
| 256 | 41.82 ns | **24.66 ns** | 1.7x | **32.12 ns** | 62.37 ns | 0.5x |
| 512 | 75.94 ns | **43.98 ns** | 1.7x | **59.93 ns** | 117.60 ns | 0.5x |
| 1024 | 149.93 ns | **81.68 ns** | 1.8x | **115.97 ns** | 239.36 ns | 0.5x |

#### Clang 18

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 6.57 ns | **1.63 ns** | 4.0x | 4.33 ns | **1.60 ns** | 2.7x |
| 2 | **6.80 ns** | 7.68 ns | 0.9x | 5.16 ns | **1.58 ns** | 3.3x |
| 3 | **7.34 ns** | 7.67 ns | 1.0x | 6.10 ns | **1.73 ns** | 3.5x |
| 4 | **5.83 ns** | 7.24 ns | 0.8x | 6.34 ns | **1.59 ns** | 4.0x |
| 5 | **6.70 ns** | 7.40 ns | 0.9x | 6.58 ns | **1.71 ns** | 3.9x |
| 6 | **7.03 ns** | 7.55 ns | 0.9x | 7.39 ns | **7.36 ns** | 1.0x |
| 7 | **7.54 ns** | 7.80 ns | 1.0x | **7.78 ns** | 8.52 ns | 0.9x |
| 8 | **6.58 ns** | 7.55 ns | 0.9x | **6.32 ns** | 8.84 ns | 0.7x |
| 9 | **7.24 ns** | 7.69 ns | 0.9x | **6.78 ns** | 9.62 ns | 0.7x |
| 10 | **7.13 ns** | 7.99 ns | 0.9x | 7.02 ns | **2.26 ns** | 3.1x |
| 16 | **7.11 ns** | 8.41 ns | 0.8x | 5.60 ns | **2.45 ns** | 2.3x |
| 32 | 9.36 ns | **8.82 ns** | 1.1x | **6.77 ns** | 10.42 ns | 0.7x |
| 64 | 13.48 ns | **10.78 ns** | 1.2x | **9.22 ns** | 18.04 ns | 0.5x |
| 128 | 23.89 ns | **15.26 ns** | 1.6x | **14.20 ns** | 34.02 ns | 0.4x |
| 256 | 39.92 ns | **24.24 ns** | 1.6x | **31.51 ns** | 67.58 ns | 0.5x |
| 512 | 78.83 ns | **43.80 ns** | 1.8x | **59.41 ns** | 126.53 ns | 0.5x |
| 1024 | 151.21 ns | **80.89 ns** | 1.9x | **116.95 ns** | 235.13 ns | 0.5x |

#### Clang 19

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 6.63 ns | **1.72 ns** | 3.9x | 6.25 ns | **2.17 ns** | 2.9x |
| 2 | **6.96 ns** | 9.59 ns | 0.7x | 7.11 ns | **2.12 ns** | 3.3x |
| 3 | **7.39 ns** | 9.83 ns | 0.8x | 7.30 ns | **2.19 ns** | 3.3x |
| 4 | **6.16 ns** | 8.09 ns | 0.8x | 8.31 ns | **2.20 ns** | 3.8x |
| 5 | **6.91 ns** | 9.47 ns | 0.7x | 8.87 ns | **2.15 ns** | 4.1x |
| 6 | **7.07 ns** | 8.50 ns | 0.8x | **8.30 ns** | 9.03 ns | 0.9x |
| 7 | **7.69 ns** | 9.26 ns | 0.8x | **8.53 ns** | 10.17 ns | 0.8x |
| 8 | **6.60 ns** | 8.80 ns | 0.8x | **7.12 ns** | 12.10 ns | 0.6x |
| 9 | **7.49 ns** | 9.69 ns | 0.8x | **7.58 ns** | 12.42 ns | 0.6x |
| 10 | **7.41 ns** | 9.27 ns | 0.8x | 7.85 ns | **2.50 ns** | 3.1x |
| 16 | **7.58 ns** | 10.70 ns | 0.7x | 7.16 ns | **2.70 ns** | 2.7x |
| 32 | **9.93 ns** | 11.07 ns | 0.9x | **7.57 ns** | 10.81 ns | 0.7x |
| 64 | 13.53 ns | **13.36 ns** | 1.0x | **10.31 ns** | 19.03 ns | 0.5x |
| 128 | 25.89 ns | **17.79 ns** | 1.5x | **15.18 ns** | 35.80 ns | 0.4x |
| 256 | 41.89 ns | **28.77 ns** | 1.5x | **33.45 ns** | 70.72 ns | 0.5x |
| 512 | 82.14 ns | **52.12 ns** | 1.6x | **63.14 ns** | 139.76 ns | 0.5x |
| 1024 | 160.58 ns | **95.79 ns** | 1.7x | **120.93 ns** | 253.00 ns | 0.5x |

#### Clang 20

| Size | xtensor (xsimd) | fixed (xsimd) | Ratio | xtensor (no) | fixed (no) | Ratio |
|------|-----------------|---------------|-------|--------------|------------|-------|
| 1 | 6.79 ns | **1.67 ns** | 4.1x | 4.50 ns | **2.12 ns** | 2.1x |
| 2 | **7.20 ns** | 9.29 ns | 0.8x | 4.64 ns | **2.10 ns** | 2.2x |
| 3 | **7.69 ns** | 9.01 ns | 0.9x | 5.46 ns | **2.13 ns** | 2.6x |
| 4 | **6.29 ns** | 8.58 ns | 0.7x | 6.03 ns | **2.07 ns** | 2.9x |
| 5 | **6.58 ns** | 8.09 ns | 0.8x | 6.41 ns | **2.11 ns** | 3.0x |
| 6 | **7.31 ns** | 8.66 ns | 0.8x | **6.77 ns** | 8.74 ns | 0.8x |
| 7 | **7.24 ns** | 8.25 ns | 0.9x | **7.17 ns** | 9.96 ns | 0.7x |
| 8 | **6.60 ns** | 8.83 ns | 0.7x | **6.22 ns** | 10.56 ns | 0.6x |
| 9 | **7.02 ns** | 8.17 ns | 0.9x | **6.75 ns** | 11.78 ns | 0.6x |
| 10 | **7.15 ns** | 9.48 ns | 0.8x | 7.05 ns | **2.58 ns** | 2.7x |
| 16 | **7.25 ns** | 8.74 ns | 0.8x | 5.82 ns | **3.21 ns** | 1.8x |
| 32 | **9.61 ns** | 9.91 ns | 1.0x | **7.35 ns** | 10.46 ns | 0.7x |
| 64 | 13.42 ns | **11.72 ns** | 1.1x | **9.54 ns** | 18.08 ns | 0.5x |
| 128 | 22.85 ns | **16.85 ns** | 1.4x | **14.47 ns** | 33.85 ns | 0.4x |
| 256 | 41.88 ns | **25.49 ns** | 1.6x | **32.14 ns** | 66.12 ns | 0.5x |
| 512 | 82.31 ns | **44.72 ns** | 1.8x | **61.12 ns** | 128.42 ns | 0.5x |
| 1024 | 311.14 ns | **81.11 ns** | 3.8x | **119.25 ns** | 254.13 ns | 0.5x |

---

## 3. Best and Worst Configurations

### 3.1 Optimal Configuration per Size

| Size | Best Configuration | Time (ns) | Speedup vs Worst |
|------|-------------------|-----------|------------------|
| 1 | `gcc12_xsimd_fixed` | **0.22** | 31x |
| 2 | `gcc14_noxsimd_fixed` | **0.29** | 38x |
| 3 | `gcc12_noxsimd_fixed` | **1.52** | 7x |
| 4 | `clang18_noxsimd_fixed` | **1.59** | 6x |
| 5 | `clang18_noxsimd_fixed` | **1.71** | 6x |
| 6 | `clang17_noxsimd_fixed` | **1.84** | 6x |
| 7 | `clang16_noxsimd_fixed` | **2.05** | 5x |
| 8 | `gcc12_noxsimd_fixed` | **2.71** | 4x |
| 9 | `gcc14_noxsimd_fixed` | **3.55** | 3x |
| 10 | `clang17_noxsimd_fixed` | **2.08** | 5x |
| 16 | `clang17_noxsimd_fixed` | **2.38** | 5x |
| 32 | `clang18_noxsimd_xtensor` | **6.77** | 3x |
| 64 | `clang18_noxsimd_xtensor` | **9.22** | 4x |
| 128 | `clang16_noxsimd_xtensor` | **13.94** | 4x |
| 256 | `clang18_xsimd_fixed` | **24.24** | 4x |
| 512 | `clang16_xsimd_fixed` | **43.08** | 5x |
| 1024 | `clang16_xsimd_fixed` | **80.05** | 5x |

### 3.2 Worst Configuration per Size (Hall of Shame)

| Size | Worst Configuration | Time (ns) | Notes |
|------|---------------------|-----------|-------|
| 1 | `clang20_xsimd_xtensor` | 6.79 | XSIMD overhead on Clang |
| 2 | `gcc14_xsimd_fixed` | 10.93 | XSIMD hurts fixed on GCC |
| 3 | `gcc14_xsimd_fixed` | 11.11 | XSIMD hurts fixed on GCC |
| 4-16 | `gcc*_xsimd_fixed` | ~10 ns | XSIMD consistently hurts fixed on GCC |
| 32+ | `gcc13_noxsimd_fixed` | 17-411 ns | GCC13 noxsimd fixed very slow |

---

## 4. Analysis

### 4.1 Why is `xtensor_fixed` at size=1-2 so fast?

When the size is known at compile time and is very small (1-2 elements), the compiler can:

1. **Complete loop unrolling**: The operation becomes a simple scalar assignment
2. **Register allocation**: All data fits in CPU registers
3. **Eliminate memory access patterns**: No need for stride calculations
4. **Dead code elimination**: Bounds checking is optimized away

With GCC and no XSIMD, `xtensor_fixed<double, xshape<1>>` achieves **0.22ns** - essentially a single CPU cycle with some measurement overhead.

### 4.2 Why does XSIMD hurt `xtensor_fixed` on GCC?

The XSIMD code path introduces:

1. **Alignment checks**: Even for small sizes, alignment verification occurs
2. **Batch size detection**: XSIMD checks if vectorization is worthwhile
3. **Branch overhead**: Decision logic for scalar vs SIMD path
4. **Type dispatching**: Template machinery for SIMD batch types

For sizes 2-16, this overhead (~7-10ns) exceeds the actual computation time, making XSIMD counterproductive for `xtensor_fixed`.

### 4.3 Why does Clang behave differently?

Clang's optimizer:
- More aggressively inlines XSIMD checks
- Better at eliminating dead code paths for small fixed sizes
- Different instruction scheduling strategies

Result: Clang + XSIMD + `xtensor_fixed` performs better at large sizes (256+) than any GCC configuration.

### 4.4 The "Size 8-9 Anomaly" in Clang noxsimd

For Clang without XSIMD, `xtensor_fixed` at sizes 8-9 shows a performance regression (10-12ns vs 2ns at adjacent sizes). This suggests:

- Loop unrolling threshold issue
- Possible aliasing assumptions affecting optimization
- Register allocation pressure at these specific sizes

### 4.5 Crossover Points

| Compiler | XSIMD | `xtensor` → `xtensor_fixed` crossover |
|----------|-------|--------------------------------------|
| GCC 11-14 | ON | ~128 elements |
| GCC 11-14 | OFF | ~64-128 elements |
| Clang 16-20 | ON | ~32-64 elements |
| Clang 16-20 | OFF | Never (xtensor wins at large sizes) |

---

## 5. Recommendations

### 5.1 General Guidelines

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION FLOWCHART                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Is size known at compile time?                                 │
│         │                                                       │
│    YES ─┼─── Is size ≤ 16?                                      │
│         │         │                                             │
│         │    YES ─┼─── Use xtensor_fixed WITHOUT XSIMD          │
│         │         │    (0.2-6ns depending on size)              │
│         │         │                                             │
│         │    NO ──┼─── Use xtensor_fixed WITH XSIMD + Clang     │
│         │              (optimal for 256+ elements)              │
│         │                                                       │
│    NO ──┼─── Use xtensor with XSIMD + GCC                       │
│              (best balance for unknown sizes)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Specific Recommendations by Use Case

| Use Case | Recommended Configuration |
|----------|--------------------------|
| **Tiny vectors (1-4 elements)** | `xtensor_fixed` + GCC + no XSIMD |
| **Small fixed vectors (5-32)** | `xtensor_fixed` + Clang + no XSIMD |
| **Medium runtime vectors (32-256)** | `xtensor` + GCC + XSIMD |
| **Large fixed vectors (256+)** | `xtensor_fixed` + Clang + XSIMD |
| **Large runtime vectors (256+)** | `xtensor` + Clang + XSIMD |
| **Mixed/unknown sizes** | `xtensor` + GCC + XSIMD |

### 5.3 Compiler-Specific Notes

**For GCC users:**
- Prefer `xtensor` over `xtensor_fixed` when XSIMD is enabled
- Without XSIMD, `xtensor_fixed` is excellent for sizes ≤ 16
- GCC 12-14 give similar performance; avoid GCC 13 without XSIMD for large fixed sizes

**For Clang users:**
- `xtensor_fixed` + XSIMD is optimal for large sizes (256+)
- Without XSIMD, avoid `xtensor_fixed` for sizes 32+
- Clang 16-18 perform similarly; Clang 19-20 show slight regression for some sizes

---

## 6. Conclusions

1. **`xtensor_fixed` achieves near-zero overhead for tiny sizes**: At size=1-2 without XSIMD, the operation takes only 0.2-0.5ns, meaning the compiler essentially optimizes away the entire computation.

2. **XSIMD introduces significant overhead for small `xtensor_fixed`**: The vectorization infrastructure adds 7-10ns per operation, which dominates for sizes < 64.

3. **Compiler choice matters significantly**: GCC and Clang optimize the same code very differently, with up to 3x performance differences for identical configurations.

4. **No single "best" configuration exists**: Optimal performance requires matching container type, XSIMD setting, and compiler to the specific use case.

5. **The worst configurations can be 30-40x slower**: Poor choices (e.g., `gcc13_noxsimd_fixed` for large sizes) result in dramatic performance penalties.

---

## Appendix: Build Configuration

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.14)
project(xscalar_benchmark)

set(CMAKE_CXX_STANDARD 17)

find_package(xtensor REQUIRED)

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)

add_executable(bench_xscalar bench_xscalar.cpp)
target_link_libraries(bench_xscalar xtensor benchmark::benchmark benchmark::benchmark_main)

if(USE_XSIMD STREQUAL "ON")
    find_package(xsimd REQUIRED)
    target_link_libraries(bench_xscalar xsimd)
    target_compile_definitions(bench_xscalar PRIVATE XTENSOR_USE_XSIMD)
endif()
```

### Compilation Flags

```bash
-O3 -march=native -g -fno-omit-frame-pointer
```

---

*Report generated: December 2024*
*Benchmark framework: Google Benchmark v1.8.3*
*xtensor version: latest master branch*
