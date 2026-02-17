# Tutorial 1: Top-Down Performance Analysis Methodology

In this tutorial you will use **Arm Total Performance (ATP)** to profile a dense matrix multiplication kernel — the fundamental operator behind fully-connected layers and attention in ML models. Starting from a naive implementation, you will use ATP's Top-Down Methodology view to identify the microarchitectural bottleneck, then apply three successive optimisations and measure their impact.

By the end you will be able to:

- Collect a top-down microarchitectural profile with ATP on a Graviton instance.
- Interpret the four top-level categories: **Frontend Bound**, **Backend Bound**, **Retiring**, and **Bad Speculation**.
- Use loop reordering, cache tiling, and an explicit Arm NEON kernel to shift a workload from memory-bound to compute-efficient.

---

## Prerequisites

- An AWS Graviton instance (Graviton2 or Graviton3) with SSH access.
- A C++ compiler (`g++` 9+ or `clang++` 14+).
- CMake 3.16+.
- Arm Total Performance installed and configured.

---

## 1. Build the Code

Clone the repository and build all four variants:

```bash
cd tutorial_1_topdown
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces four executables:

| Binary | Description |
|--------|-------------|
| `matmul_naive` | ijk loop order — poor spatial locality on B |
| `matmul_reordered` | ikj loop order — stride-1 access on both B and C |
| `matmul_tiled` | ikj + cache tiling (64×64 blocks fit in L1d) |
| `matmul_tiled_neon` | ikj + tiling + explicit Arm NEON intrinsics in inner loop |

All four compute the same result (`C = A × B` for 1024×1024 float matrices by default). You can pass a custom size as the first argument, e.g. `./matmul_naive 2048`.

---

## 2. Baseline — Naive Matmul

Run the naive version to get a baseline timing:

```bash
./matmul_naive 1024
```

Expected output (times will vary by instance type):

```
Naive matmul (1024x1024)
  Time:  <X> ms
  GFLOPS: <Y>
  Check:  C[0]=... C[N*N-1]=...
```

Record the time and GFLOPS — we will compare against the optimised versions later.

### Why is it slow?

The full naive kernel in `src/matmul_naive.cpp` is:

```cpp
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j]; // B access strides by N
            }
            C[i * N + j] = sum;
        }
    }
}
```

Each iteration accesses `B[k * N + j]` — jumping by `N` floats (4 KB for N=1024) on every step. This means every access to B lands in a different cache line, causing a cache miss on nearly every iteration. The CPU spends most of its time waiting for data from memory rather than doing useful computation.

---

## 3. Profile the Naive Version with ATP

<!-- TODO: Add specific ATP commands/UI steps and screenshots -->

Profile `matmul_naive` using ATP and open the **Top-Down Methodology** view.

```bash
# TODO: ATP profiling command
```

You should see something like:

<!-- TODO: Screenshot of ATP top-down view for naive matmul -->

The key observation: the workload is heavily **Backend Bound → Memory Bound**. The CPU's backend is stalled waiting for data from the memory hierarchy. The **Retiring** percentage (cycles doing useful work) will be low.

Note the following metrics:
- **IPC** (Instructions Per Cycle) — will be low (likely < 1.0), indicating the pipeline is frequently stalled.
- **Backend Bound %** — will dominate the top-down breakdown.
- **L1D Cache Miss Rate** — will be high due to the strided B access pattern.

---

## 4. Optimisation 1 — Loop Reordering (ikj)

The fix is simple: reorder the loops from `ijk` to `ikj`. The full kernel in `src/matmul_reordered.cpp` is:

```cpp
void matmul_ikj(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j]; // B and C both stride-1
            }
        }
    }
}
```

Now the inner loop iterates over `j` — both `B[k*N + j]` and `C[i*N + j]` are accessed sequentially in memory. The hardware prefetcher can predict this pattern and bring cache lines in ahead of time.

Run it:

```bash
./matmul_reordered 1024
```

You should see a significant speedup over the naive version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_reordered` and compare the Top-Down view to the naive version:

- **Backend Bound %** should drop substantially.
- **Retiring %** should increase — more cycles are now spent doing useful FP work.
- **IPC** should improve noticeably.
- **L1D Cache Miss Rate** should be much lower.

<!-- TODO: Side-by-side screenshot comparison -->

---

## 5. Optimisation 2 — Cache Tiling

Loop reordering helped, but for large matrices the working set of B still exceeds the L1 cache. We can do better by processing the matrices in small **tiles** that fit entirely in L1d cache.

The tiled kernel in `src/matmul_tiled.cpp` is:

```cpp
constexpr int TILE = 64;

void matmul_tiled(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int k0 = 0; k0 < N; k0 += TILE) {
            for (int j0 = 0; j0 < N; j0 += TILE) {
                int i_end = std::min(i0 + TILE, N);
                int k_end = std::min(k0 + TILE, N);
                int j_end = std::min(j0 + TILE, N);
                for (int i = i0; i < i_end; ++i) {
                    for (int k = k0; k < k_end; ++k) {
                        float a_ik = A[i * N + k];
                        for (int j = j0; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

**Why TILE = 64?** A 64×64 tile of floats is 64 × 64 × 4 = 16 KB. Three tiles (A, B, C sub-blocks) total 48 KB, which fits comfortably in the Graviton3 L1d cache (64 KB per core). This means the inner loops run almost entirely out of L1, minimising stalls.

Run it:

```bash
./matmul_tiled 1024
```

You should see a further speedup over the reordered version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_tiled` and look at the Top-Down view:

- **Backend Bound %** should drop further, and the sub-category should shift from **Memory Bound** toward **Core Bound** (the memory subsystem is no longer the primary bottleneck).
- **Retiring %** should be at its highest across the first three variants.
- **IPC** should be the best so far.

<!-- TODO: Screenshot -->

---

## 6. Optimisation 3 — Explicit Arm NEON Intrinsics

After fixing memory access (loop reordering) and locality (tiling), the next step is to increase compute throughput in the hot inner loop with explicit Arm NEON instructions.

`src/matmul_tiled_neon.cpp` keeps the same tiled structure, but the innermost loop uses NEON intrinsics directly.
`dotprod` instructions are mainly for int8-style kernels, so this float32 matmul uses NEON FMA.

```cpp
void matmul_tiled_neon(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    // ... same tile loops as matmul_tiled ...
    const float a_ik = A[i * N + k];
    float* c_row = &C[i * N + j0];
    const float* b_row = &B[k * N + j0];
    int width = j_end - j0;

    int jj = 0;
    for (; jj + 4 <= width; jj += 4) {
        float32x4_t c_vec = vld1q_f32(c_row + jj);
        float32x4_t b_vec = vld1q_f32(b_row + jj);
        c_vec = vfmaq_n_f32(c_vec, b_vec, a_ik);
        vst1q_f32(c_row + jj, c_vec);
    }
    for (; jj < width; ++jj) {
        c_row[jj] += a_ik * b_row[jj];
    }
}
```

Run it:

```bash
./matmul_tiled_neon 1024
```

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

When compared against `matmul_tiled`, you should see:

- **Retiring %** increase further (more useful work per cycle).
- **IPC** improve, because the core executes explicit NEON data-parallel operations.
- **Backend Memory Bound** remain relatively low (memory pattern was already fixed).

Keep this section focused on top-down outcome validation. A deeper instruction-type analysis (scalar vs NEON mix) is covered later in Tutorial 3.

## 7. Summary — Comparing All Four Variants

Run all four back-to-back:

```bash
./matmul_naive 1024
./matmul_reordered 1024
./matmul_tiled 1024
./matmul_tiled_neon 1024
```

| Variant | Expected Improvement | Why |
|---------|---------------------|-----|
| Naive (ijk) | Baseline | Strided B access → constant cache misses |
| Reordered (ikj) | ~3–5× faster | Stride-1 access on B and C → prefetcher-friendly |
| Tiled (ikj, 64×64) | ~5–10× faster than naive | Working set fits in L1d → near-zero cache misses |
| Tiled + NEON | Additional speedup over tiled | Explicit NEON intrinsics increase math throughput in the hot inner loop |

<!-- TODO: ATP side-by-side comparison screenshot showing all four profiles -->

### What the ATP Top-Down view tells us

| Metric | Naive | Reordered | Tiled | Tiled + NEON |
|--------|-------|-----------|-------|-------------|
| Backend Bound % | High | Medium | Low | Low |
| → Memory Bound | Dominant | Reduced | Minimal | Minimal |
| Retiring % | Low | Medium | High | Highest |
| IPC | < 1.0 | ~1–2 | ~2–3+ | Best |

The progression is clear: by fixing memory access patterns (loop reordering), then ensuring data stays in cache (tiling), then increasing per-cycle compute throughput (NEON), we move the workload from data-starved to compute-efficient. ATP's Top-Down Methodology makes it straightforward to identify the bottleneck category and verify that each optimisation addressed it.

---

## Key Takeaways

1. **Start with the Top-Down view.** It tells you *where* to look — don't guess, measure.
2. **Backend Memory Bound** means the CPU is waiting for data. Fix access patterns first.
3. **Loop reordering** is often the simplest and highest-impact optimisation for nested array computations.
4. **Cache tiling** pushes performance further by keeping the working set in the fastest level of the memory hierarchy.
5. **Explicit NEON intrinsics** can push throughput further once memory-side bottlenecks are reduced.
6. **Always re-profile after each change** to confirm the optimisation had the expected effect. ATP makes this comparison easy.
