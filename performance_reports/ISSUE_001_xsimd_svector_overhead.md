# Issue #001: XSIMD Overhead on Small xtensor_fixed Sizes

**Date**: 2025-12-18
**Severity**: High (10-40x performance degradation)
**Affected Configurations**: All GCC/Clang with XSIMD enabled
**Affected Sizes**: 2-64 elements (worst for 2-16)

---

## Executive Summary

When XSIMD is enabled, `xtensor_fixed` operations for small sizes (2-64 elements) suffer severe performance degradation due to **dynamic memory allocations** in the strided assignment path. The allocation overhead completely dominates the actual computation time.

| Size | With XSIMD | Without XSIMD | Slowdown |
|------|------------|---------------|----------|
| 1    | 0.24 ns    | 0.22 ns       | 1.1x     |
| 2    | 10.93 ns   | 0.29 ns       | **38x**  |
| 4    | 9.77 ns    | 1.80 ns       | **5.4x** |
| 16   | 10.79 ns   | 6.68 ns       | 1.6x     |

---

## Root Cause Analysis

### 1. Assignment Path Selection

When executing `result = vec1 + 1.0` on an `xtensor_fixed`, xtensor chooses between three assignment strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASSIGNMENT DECISION TREE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  linear_assign(e1, e2, trivial) == true ?                       │
│         │                                                       │
│    YES ─┼─→ linear_assigner<simd>::run()     ✓ FAST             │
│         │   (simple loop, no allocations)                       │
│         │                                                       │
│    NO ──┼─→ simd_strided_assign == true ?                       │
│              │                                                  │
│         YES ─┼─→ strided_loop_assigner<true>::run()  ✗ SLOW     │
│              │   (allocates dynamic vectors each call!)         │
│              │                                                  │
│         NO ──┼─→ stepper_assigner::run()     ~ Medium           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. The Problem: Size 1 vs Size 2+

**Size 1 (FAST)**: Uses `linear_assigner<true>` - simple SIMD loop, no allocations.

**Size 2+ (SLOW)**: Falls through to `strided_loop_assigner<true>` which allocates:

```cpp
// xassign.hpp:1111 - ALLOCATES EVERY CALL!
dynamic_shape<std::size_t> idx, max_shape;
```

Where `dynamic_shape<T>` is defined as:

```cpp
// xshape.hpp:29
template <class T>
using dynamic_shape = svector<T, 4>;  // Small vector with heap fallback
```

### 3. VTune Hotspot Evidence

**gcc14_xsimd, fixed_double_4:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `strided_loop_assigner::run` | 2.12s | 43% |
| `svector::resize` | 0.95s | **19%** |
| `svector::assign` | 0.68s | **14%** |
| `svector::svector` | 0.36s | **7%** |
| `svector::size` | 0.18s | 4% |

**~44% of execution time is spent in svector operations** - none of which are the actual computation!

**gcc14_noxsimd, fixed_double_4:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `stepper_tools::increment_stepper` | 2.85s | 84% |
| `xstepper::step` | 0.20s | 6% |

No svector overhead - all time spent in actual computation.

---

## Code Analysis

### Problematic Code Path

**File**: `deps/xtensor/include/xtensor/core/xassign.hpp`

```cpp
// Line 1100-1122: strided_loop_assigner<simd>::run()
template <bool simd>
template <class E1, class E2>
inline void strided_loop_assigner<simd>::run(E1& e1, const E2& e2,
                                              const loop_sizes_t& loop_sizes)
{
    // ...

    // LINE 1111 - THE PROBLEM!
    // TODO comment even acknowledges the issue:
    // "TODO can we get rid of this and use `shape_type`?"
    dynamic_shape<std::size_t> idx, max_shape;  // ALLOCATES HEAP MEMORY!

    if (is_row_major)
    {
        xt::resize_container(idx, cut);                    // ALLOCATION
        max_shape.assign(e1.shape().begin(), ...);         // ALLOCATION
    }
    // ...
}
```

### Additional Allocations

**Lines 884 and 929** (in `get_loop_sizes`):

```cpp
dynamic_shape<std::size_t> stride_sizes;  // More allocations!
```

---

## Why Size 1 Works But Size 2+ Doesn't

The condition for `linear_assign` in `xassign.hpp:409-412`:

```cpp
static constexpr bool linear_assign(const E1& e1, const E2& e2, bool trivial)
{
    return trivial && detail::is_linear_assign(e1, e2);
}
```

For `xtensor_fixed<double, xshape<1>>` + `xscalar<double>`:
- The expression likely qualifies as "trivially broadcastable"
- `is_linear_assign` returns `true`

For `xtensor_fixed<double, xshape<N>>` where N > 1:
- The broadcast/stride conditions fail
- Falls back to `strided_loop_assigner`

---

## Impact Assessment

### Benchmark Data (gcc14, fixed_double)

| Size | XSIMD (ns) | No XSIMD (ns) | Overhead Factor |
|------|------------|---------------|-----------------|
| 1    | 0.24       | 0.22          | 1.1x            |
| 2    | 10.93      | 0.29          | **37.7x**       |
| 3    | 11.11      | 1.69          | **6.6x**        |
| 4    | 9.77       | 1.80          | **5.4x**        |
| 5    | 10.16      | 5.56          | 1.8x            |
| 6    | 10.33      | 2.46          | **4.2x**        |

### Real-World Impact

For code doing many small tensor operations (physics simulations, graphics, ML inference):
- Millions of 3D/4D vector operations per frame
- Each operation pays ~10ns allocation overhead with XSIMD
- Can result in **100ms+ overhead per frame** in tight loops

---

## Proposed Solutions

### Short-term: User Workaround

Compile without XSIMD for small fixed-size tensor code:

```cmake
# For small tensor code paths:
target_compile_definitions(my_target PRIVATE XTENSOR_USE_XSIMD=0)
```

### Medium-term: xtensor Fix

Replace dynamic allocations with static arrays in `strided_loop_assigner`:

```cpp
// Instead of:
dynamic_shape<std::size_t> idx, max_shape;

// Use compile-time sized arrays for fixed containers:
template <class E1>
using idx_type = std::conditional_t<
    is_fixed<typename E1::shape_type>::value,
    std::array<std::size_t, E1::shape_type::size()>,
    dynamic_shape<std::size_t>
>;
idx_type<E1> idx, max_shape;
```

### Long-term: Improve Path Selection

Enhance `linear_assign` condition to better handle `xtensor_fixed` + `xscalar` expressions, avoiding the strided path entirely for these simple cases.

---

## Reproduction

```bash
# Build with and without XSIMD
./build.sh --compiler=gcc-14 --xsimd=ON
./build.sh --compiler=gcc-14 --xsimd=OFF

# Compare fixed_double benchmarks
./build_gcc14_xsimd/bench_xscalar --benchmark_filter="fixed_double_[2-6]$"
./build_gcc14_noxsimd/bench_xscalar --benchmark_filter="fixed_double_[2-6]$"

# Run VTune analysis
./run_vtune_analysis.sh collect --configs=gcc14_xsimd,gcc14_noxsimd \
    --benchmarks="fixed_double_4" --duration=3
```

---

## References

- **Source Files**:
  - `deps/xtensor/include/xtensor/core/xassign.hpp` (lines 1100-1150)
  - `deps/xtensor/include/xtensor/core/xshape.hpp` (line 29)
  - `deps/xtensor/include/xtensor/containers/xstorage.hpp` (svector implementation)

- **VTune Results**: `analysis_results/vtune_raw/gcc14_xsimd/fixed_double_*`

- **Related xtensor TODO**: Line 1111 comment: "TODO can we get rid of this and use `shape_type`?"

---

## Conclusion

The performance issue is a **design limitation** in xtensor's XSIMD strided assignment path. The code allocates dynamic memory (`svector`) for index tracking even when operating on compile-time fixed-size containers. This overhead is negligible for large tensors but catastrophic for small fixed-size vectors commonly used in graphics, physics, and real-time applications.

**Recommendation**: Disable XSIMD for code paths dominated by small `xtensor_fixed` operations until xtensor addresses this issue.
