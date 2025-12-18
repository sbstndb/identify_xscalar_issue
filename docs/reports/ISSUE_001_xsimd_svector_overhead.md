# Issue #001: XSIMD Overhead on Small xtensor_fixed Sizes

**Date**: 2025-12-18
**Updated**: 2025-12-18
**Severity**: High (10-40x performance degradation)
**Affected Configurations**: All GCC/Clang with XSIMD enabled
**Affected Sizes**: 2-64 elements (worst for 2-16)
**Affected Container**: `xtensor_fixed` ONLY (not `xtensor` or `xarray`)
**Nature**: Compile-time vs runtime path inconsistency + allocation in slow path

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

## Critical Discovery: Problem is Specific to `xtensor_fixed`

### Container Comparison

The performance issue **only affects `xtensor_fixed`**, not `xtensor` or `xarray`:

| Config | `xtensor` (size=4) | `xtensor_fixed` (size=4) | Observation |
|--------|-------------------|-------------------------|-------------|
| XSIMD ON | 3.49 ns | 9.88 ns | **xtensor_fixed is 2.8x slower** |
| XSIMD OFF | 6.15 ns | 1.82 ns | xtensor_fixed is 3.4x faster |

With XSIMD disabled, `xtensor_fixed` has the advantage (compile-time size optimization).
With XSIMD enabled, `xtensor_fixed` loses its advantage completely and becomes slower.

### Why? Runtime vs Compile-Time Path Divergence

The root cause is an **inconsistency** between how xtensor evaluates "trivial broadcast" at runtime vs compile-time:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PATH DIVERGENCE DIAGRAM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  result = vec + 1.0  (scalar addition)                                      │
│         │                                                                   │
│         ├── vec is xtensor<T,N> or xarray<T>                                │
│         │   │                                                               │
│         │   └── RUNTIME PATH (xassign.hpp:603)                              │
│         │       │                                                           │
│         │       └── e2.broadcast_shape(shape, true)                         │
│         │           │                                                       │
│         │           └── xscalar::broadcast_shape() returns TRUE always!     │
│         │               │                                                   │
│         │               └── trivial = true → linear_assigner ✓ FAST         │
│         │                                                                   │
│         └── vec is xtensor_fixed<T,xshape<N>>                               │
│             │                                                               │
│             └── COMPILE-TIME PATH (xassign.hpp:593-595)                     │
│                 │                                                           │
│                 └── static_trivial_broadcast<true, CT...>::value            │
│                     │                                                       │
│                     └── promote_index<shape_types...>::value                │
│                         │                                                   │
│                         └── promote_fixed<shapes...>                        │
│                             │                                               │
│                             └── filter_scalar: array<T,0> → fixed_shape<1>  │
│                                 │                                           │
│                                 └── broadcast_fixed_shape<N,1>              │
│                                     │                                       │
│                                     └── (N == 1) ? → FALSE when N > 1       │
│                                         │                                   │
│                                         └── trivial = false                 │
│                                             → strided_loop_assigner ✗ SLOW  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Inconsistency in Detail

**Runtime path** (`xscalar.hpp:611-614`):
```cpp
template <class CT>
template <class S>
inline bool xscalar<CT>::broadcast_shape(S&, bool) const noexcept
{
    return true;  // ALWAYS returns true - scalar can broadcast anywhere!
}
```

**Compile-time path** (`xshape.hpp:331`):
```cpp
// In broadcast_fixed_shape_cmp_impl
static constexpr bool value = (I_v == J_v);  // STRICT equality check!
```

For `xtensor_fixed<double, xshape<4>> + xscalar`:
- Runtime evaluation: `xscalar::broadcast_shape()` → `true` (not used for fixed)
- Compile-time evaluation: `fixed_shape<4>` vs `fixed_shape<1>` (from filtered scalar)
  - At dimension 0: `I_v = 4`, `J_v = 1`
  - `value = (4 == 1) = false` ← **This is the problem!**

### Both Interpretations Are "Correct"

| Path | Interpretation of "Trivial" | Logic |
|------|---------------------------|-------|
| Runtime | "Can I broadcast to this shape?" | Scalar → Any shape = trivial |
| Compile-time | "Are shapes exactly equal?" | 4 ≠ 1 = non-trivial |

The problem is that **they should be consistent** for assignment path selection purposes.

---

## Root Cause Analysis

### 1. Assignment Path Selection

When executing `result = vec1 + 1.0` on an `xtensor_fixed`, xtensor chooses between three assignment strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASSIGNMENT DECISION TREE                      │
│                      (xassign.hpp:454-477)                       │
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

### 2. Deep Dive: Why Size 1 Works But Size 2+ Doesn't

The `linear_assign` condition requires `trivial == true`:

```cpp
// xassign.hpp:409-411
static constexpr bool linear_assign(const E1& e1, const E2& e2, bool trivial)
{
    return trivial && detail::is_linear_assign(e1, e2);
}
```

The `trivial` parameter comes from shape broadcasting logic. Here's the critical chain:

#### Step 1: Shape Promotion for Broadcasting

When computing the result shape of `xtensor_fixed<double, xshape<N>> + scalar`:

```cpp
// xshape.hpp:423-436 - filter_scalar converts 0D to 1D
template <class T>
struct filter_scalar<std::array<T, 0>>  // xscalar's shape type
{
    using type = fixed_shape<1>;  // Treated as 1D with size 1
};
```

So `xscalar` (0-dimensional) becomes `fixed_shape<1>` for broadcasting purposes.

#### Step 2: Trivial Broadcast Comparison

```cpp
// xshape.hpp:313-332 - broadcast_fixed_shape_cmp_impl
static constexpr bool value = (I_v == J_v);  // Trivial only if EQUAL
```

For `fixed_shape<N>` + `xscalar` (as `fixed_shape<1>`):
- **N = 1**: `fixed_shape<1>` vs `fixed_shape<1>` → `1 == 1` → **TRIVIAL** ✓
- **N > 1**: `fixed_shape<N>` vs `fixed_shape<1>` → `N != 1` → **NOT TRIVIAL** ✗

#### Step 3: Path Selection Consequence

| Size N | trivial_broadcast | linear_assign | Path Taken |
|--------|-------------------|---------------|------------|
| 1      | true              | true          | `linear_assigner` (fast) |
| 2+     | false             | false         | `strided_loop_assigner` (slow) |

This explains the dramatic performance cliff between size 1 and size 2.

### 3. The Allocation Problem

When `strided_loop_assigner::run()` is called:

```cpp
// xassign.hpp:1111 - ALLOCATES EVERY CALL!
// TODO comment acknowledges: "TODO can we get rid of this and use `shape_type`?"
dynamic_shape<std::size_t> idx, max_shape;
```

Where `dynamic_shape<T>` is:

```cpp
// xshape.hpp:29
template <class T>
using dynamic_shape = svector<T, 4>;  // Small vector with heap fallback
```

### 4. VTune Hotspot Evidence

**gcc14_xsimd, fixed_double_4:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `strided_loop_assigner::run` | 2.12s | 43% |
| `svector::resize` | 0.95s | **19%** |
| `svector::assign` | 0.68s | **14%** |
| `svector::svector` | 0.36s | **7%** |
| `svector::size` | 0.18s | 4% |

**~44% of execution time is spent in svector operations** - none of which are actual computation!

**gcc14_noxsimd, fixed_double_4:**

| Function | CPU Time | % of Total |
|----------|----------|------------|
| `stepper_tools::increment_stepper` | 2.85s | 84% |
| `xstepper::step` | 0.20s | 6% |

No svector overhead - all time spent in actual computation.

---

## Complete Allocation Site Inventory

### Primary Site (Critical)

**File**: `xassign.hpp:1111`
```cpp
dynamic_shape<std::size_t> idx, max_shape;  // Called per assignment!
```

### Secondary Sites (Moderate Impact)

**File**: `xassign.hpp:884` (row-major nth_idx)
```cpp
dynamic_shape<std::size_t> stride_sizes;  // For parallel iteration
```

**File**: `xassign.hpp:929` (column-major nth_idx)
```cpp
dynamic_shape<std::size_t> stride_sizes;  // For parallel iteration
```

### Other Uses (Context-dependent)

| File | Line | Usage |
|------|------|-------|
| `xstrided_view_base.hpp` | 272, 962-964 | View shape/strides |
| `xsort.hpp` | 129-174, 478, 740, 1217 | Sorting permutations |
| `xmanipulation.hpp` | 517-549, 874, 900 | Reshape operations |
| `xreducer.hpp` | 329-331 | Reduction iteration |
| `xgenerator.hpp` | 443-482 | Generator shapes |

---

## Problem Classification

### Is This a Design Issue or a Bug?

**Answer: Implementation inconsistency + allocation bug. Both are fixable locally.**

| Aspect | Classification | Root Cause |
|--------|----------------|------------|
| Compile-time path returns `false` for scalar broadcast | **Implementation inconsistency** | `broadcast_fixed_shape_cmp_impl` uses strict equality |
| Runtime path returns `true` for scalar broadcast | Correct behavior | `xscalar::broadcast_shape()` knows scalars are trivial |
| Using `dynamic_shape` in strided assigner | **Allocation bug** | Can use static types when dimension is known |
| Path divergence for `xtensor_fixed` | Design limitation | Compile-time path bypasses runtime `broadcast_shape()` |

### Why Does Size=1 Work?

When `N=1`, the compile-time path produces `fixed_shape<1>` vs `fixed_shape<1>`:
- `value = (1 == 1) = true` → trivial → fast path

This is why size=1 shows no performance regression (0.24ns vs 0.22ns).

### The Core Bug

The `filter_scalar` mechanism converts 0D scalar to 1D `fixed_shape<1>`, but doesn't preserve the semantic meaning that "this came from a scalar and should always be trivial". The subsequent equality check at `xshape.hpp:331` treats it as a regular 1D tensor.

**Fix options**:
1. Modify `broadcast_fixed_shape_cmp_impl` to recognize `J_v == 1` as always trivial
2. Create a special `scalar_shape` marker type that propagates through the system
3. Force the runtime path for scalar operations even with fixed containers

The existing TODO comment at line 1111 shows developers are aware of the allocation issue:
```cpp
// TODO can we get rid of this and use `shape_type`?
```

---

## Proposed Fixes

There are **two independent issues** that can be fixed separately:

### Fix 1: Consistent Trivial Broadcast for Scalars (Primary Fix)

The cleanest fix is to make the compile-time path consistent with the runtime path by treating broadcast from `1` to `N` as trivial (since it comes from a scalar):

**File**: `xshape.hpp:331`

```cpp
// BEFORE:
static constexpr bool value = (I_v == J_v);

// AFTER: Treat broadcast from 1 as trivial (same as xscalar::broadcast_shape)
static constexpr bool value = (I_v == J_v) || (I_v == 1) || (J_v == 1);
```

**Rationale**:
- Broadcasting from size 1 to size N is semantically trivial - just replicate the value
- This matches `xscalar::broadcast_shape()` which returns `true`
- The resulting iteration pattern for linear assignment handles this correctly

**Risk Assessment**:
- Low risk: Broadcasting from 1 is always valid and simple
- This matches the runtime behavior exactly
- The iteration logic already handles replicated values

### Fix 2: Static Shape for Index Variables (Secondary Fix)

Even if Fix 1 is applied, there may be other non-trivial broadcasts that still use the strided path. Using static shapes for index variables eliminates allocation overhead:

**File**: `xassign.hpp:1111`

xtensor already provides the `static_dimension` trait:

```cpp
// xshape.hpp:247-251
template <class S>
struct static_dimension
{
    static constexpr std::ptrdiff_t value = ...;
    // Returns -1 for std::vector (dynamic)
    // Returns N for std::array<T,N> and fixed_shape<...>
};
```

**Proposed Change**:

```cpp
// BEFORE:
dynamic_shape<std::size_t> idx, max_shape;

// AFTER:
template <class E1>
struct idx_type_selector
{
    static constexpr std::ptrdiff_t dim =
        static_dimension<typename E1::shape_type>::value;

    using type = std::conditional_t<
        (dim >= 0),  // Compile-time known dimension
        std::array<std::size_t, static_cast<std::size_t>(dim)>,
        dynamic_shape<std::size_t>  // Fallback for xarray, dynamic views
    >;
};

using idx_type = typename idx_type_selector<E1>::type;
idx_type idx{}, max_shape{};
```

### Recommended Approach

**Apply Fix 1 first** - it solves the root cause (wrong path selection) rather than just optimizing the slow path. Fix 2 is a good defense-in-depth improvement but addresses the symptom rather than the cause.

---

## Side-Effect Analysis

### Compatibility Matrix

| Scenario | E1 Type | dim value | idx_type | Risk |
|----------|---------|-----------|----------|------|
| `xtensor_fixed = expr` | `fixed_shape<N>` | N | `std::array<N>` | None |
| `xtensor<T,N> = expr` | `std::array<N>` | N | `std::array<N>` | None |
| `xarray = expr` | `std::vector` | -1 | `dynamic_shape` | None (fallback) |
| `view(xarray) = expr` | `dynamic_shape` | -1 | `dynamic_shape` | None (fallback) |
| `view(xtensor_fixed) = expr` | varies | varies | varies | Needs testing |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| xarray broken | 0% | N/A | Fallback when `dim == -1` |
| Dynamic views broken | 0% | N/A | Fallback when `dim == -1` |
| Stack overflow (large N) | Very low | Low | N is typically small for fixed |
| Array larger than needed | 100% | Negligible | `cut <= dim`, stack cost minimal |

### Why the Fix is Safe

1. **Type detection at compile-time**: `static_dimension` returns -1 for dynamic shapes
2. **Fallback preserves behavior**: Dynamic cases use original `dynamic_shape`
3. **Localized change**: Only affects `idx` and `max_shape` local variables
4. **No runtime logic change**: Same number of elements used, same iteration

### Edge Cases Handled

| Case | Behavior |
|------|----------|
| `cut < dim` | Array has unused elements (stack, negligible) |
| Mixed expressions (`fixed = xarray + scalar`) | Based on E1, which is fixed → optimized |
| Nested xfunctions | E1 determines type, not E2 |

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

### Expected Improvement After Fix

| Size | Current (ns) | Expected (ns) | Improvement |
|------|--------------|---------------|-------------|
| 2    | 10.93        | ~0.3-0.5      | ~20-35x faster |
| 4    | 9.77         | ~1.8-2.0      | ~5x faster |
| 16   | 10.79        | ~6-7          | ~1.5x faster |

---

## Workarounds (Until Fix is Merged)

### Option 1: Disable XSIMD for Small Tensor Code

```cmake
# For small tensor code paths:
target_compile_definitions(my_target PRIVATE XTENSOR_USE_XSIMD=0)
```

### Option 2: Use Size 1 Arrays and Manual Loops

```cpp
// Instead of:
xtensor_fixed<double, xshape<3>> result = vec + scalar;

// Use:
xtensor_fixed<double, xshape<3>> result;
for (size_t i = 0; i < 3; ++i) {
    result(i) = vec(i) + scalar;
}
```

### Option 3: Patch xtensor Locally

Apply the proposed fix to your local xtensor installation.

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

### Source Files

| File | Lines | Content |
|------|-------|---------|
| **Critical Path Divergence** | | |
| `xassign.hpp` | 585-595 | Compile-time path: `static_trivial_broadcast` |
| `xassign.hpp` | 603 | Runtime path: `e2.broadcast_shape(shape, true)` |
| `xassign.hpp` | 560 | `static_trivial_broadcast` → `promote_index::value` |
| **Trivial Broadcast Logic** | | |
| `xshape.hpp` | 331 | **KEY BUG**: `value = (I_v == J_v)` strict equality |
| `xshape.hpp` | 439-464 | `promote_fixed` recursion with `filter_scalar` |
| `xshape.hpp` | 423-436 | `filter_scalar`: `array<T,0>` → `fixed_shape<1>` |
| `xscalar.hpp` | 611-614 | `broadcast_shape()` always returns `true` |
| **Assignment Path Selection** | | |
| `xassign.hpp` | 409-412 | `linear_assign` condition |
| `xassign.hpp` | 454-477 | Three-way assignment dispatch |
| **Allocation Sites** | | |
| `xassign.hpp` | 1111 | **Primary allocation**: `dynamic_shape<> idx, max_shape` |
| `xassign.hpp` | 884, 929 | `nth_idx` allocations |
| `xshape.hpp` | 29 | `dynamic_shape = svector<T, 4>` |
| **Utilities** | | |
| `xshape.hpp` | 247-251 | `static_dimension` trait |

### VTune Results

`analysis_results/vtune_raw/gcc14_xsimd/fixed_double_*`

### Related xtensor TODO

Line 1111 comment: `"TODO can we get rid of this and use shape_type?"`

---

## Conclusion

The performance issue stems from **an inconsistency between runtime and compile-time evaluation of "trivial broadcast"**:

1. **Runtime path** (`xtensor`, `xarray`): `xscalar::broadcast_shape()` always returns `true`, correctly recognizing that scalar-to-any-shape is trivial. Result: `linear_assigner` (fast).

2. **Compile-time path** (`xtensor_fixed`): `broadcast_fixed_shape_cmp_impl` uses strict equality (`N == 1`), treating scalar broadcast as non-trivial for N > 1. Result: `strided_loop_assigner` (slow with allocations).

This explains why:
- `xtensor` and `xarray` don't have the problem (use runtime path)
- `xtensor_fixed` has the problem (use compile-time path)
- Size=1 doesn't have the problem (`1 == 1` is true)

### Two-Part Fix

| Fix | Target | Complexity | Impact |
|-----|--------|-----------|--------|
| Fix 1: Consistent trivial broadcast | `xshape.hpp:331` | 1 line change | **Eliminates root cause** |
| Fix 2: Static index types | `xassign.hpp:1111` | ~10 lines | Defense-in-depth |

**Recommendation**:
- **Immediate**: Apply Fix 1 (consistent trivial broadcast) - solves the root cause
- **Short-term workaround**: Disable XSIMD for small `xtensor_fixed` workloads
- **Medium-term**: Submit PR to xtensor with both fixes
- **Testing**: Verify that `linear_assigner` handles broadcast-from-1 correctly (it should, as this matches runtime behavior)
