# Tutorial 1: Top-Down Performance Analysis Methodology

In this tutorial you will use **Arm Total Performance (ATP)** to profile a dense matrix multiplication kernel — the fundamental operator behind fully-connected layers and attention in ML models. Starting from a naive implementation, you will use ATP's Top-Down Methodology view to identify the microarchitectural bottleneck, then apply two successive cache-tiling optimisations and measure their impact.

By the end you will be able to:

- Collect a top-down microarchitectural profile with ATP on a Graviton instance.
- Interpret the four top-level categories: **Frontend Bound**, **Backend Bound**, **Retiring**, and **Bad Speculation**.
- Use 1D tiling and 2D tiling to progressively shift a workload from memory-bound to compute-efficient.

---

## Prerequisites

- An AWS Graviton instance (Graviton2 or Graviton3) with SSH access.
- A C++ compiler (`g++` 9+ or `clang++` 14+).
- CMake 3.16+.
- Arm Total Performance installed and configured.

---

## 1. Build the Code

Clone the repository and build all three variants:

```bash
cd tutorial_1
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces three executables:

| Binary | Description |
|--------|-------------|
| `matmul_naive` | ijk loop order — poor spatial locality on B |
| `matmul_tiled_1d` | k-dimension tiling — B strip fits in L2 but not L1 |
| `matmul_tiled_2d` | Full i,j,k tiling — working set fits entirely in L1d |

All three compute the same result (`C = A × B` for 4096×4096 float matrices by default). You can pass a custom size as the first argument, e.g. `./matmul_naive 2048`.

---

## 2. Baseline — Naive Matmul

Run the naive version to get a baseline timing:

```bash
./matmul_naive
```

Expected output (times will vary by instance type):

```
Naive matmul (4096x4096, <N> reps)
  Time:  <X> ms
  GFLOPS: <Y>
  Check:  C[0]=... C[N*N-1]=...
```

Record the time and GFLOPS — we will compare against the optimised versions later.

### Why is it slow?

The naive kernel in `src/matmul_naive.cpp` is:

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

Each iteration accesses `B[k * N + j]` — jumping by `N` floats (16 KB for N=4096) on every step. This means every access to B lands in a different cache line, causing a cache miss on nearly every iteration. The CPU spends most of its time waiting for data from memory rather than doing useful computation.

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

## 4. Optimisation 1 — 1D Tiling (k-strip)

Rather than iterating over the full k dimension at once, we split k into blocks of size `TILE`. Within each k-block the loop order is ikj, giving stride-1 access on B and C. The kernel in `src/matmul_tiled_1d.cpp` is:

```cpp
constexpr int TILE = 64;

void matmul_tiled_1d(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int k0 = 0; k0 < N; k0 += TILE) {
        int k_end = std::min(k0 + TILE, N);
        for (int i = 0; i < N; ++i) {
            for (int k = k0; k < k_end; ++k) {
                float a_ik = A[i * N + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    }
}
```

**Why does this help?** Within each k-block, the B strip accessed is `TILE` rows × N columns = 64 × 4096 × 4 bytes = **1 MB**. This fits in Graviton3's L2 cache (1 MB per core), so each B row is fetched from main memory once per k-block rather than once per row of A. L2 misses drop significantly.

**Why isn't it enough?** The 1 MB B strip does NOT fit in L1d (64 KB). The j loop still sweeps across the full N columns of B and C, so L1 misses remain elevated.

Run it:

```bash
./matmul_tiled_1d
```

You should see a significant speedup over the naive version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_tiled_1d` and compare the Top-Down view to the naive version:

- **Backend Bound %** should drop substantially.
- **Memory Bound** sub-category should decrease — fewer L2 misses.
- **L1D miss rate** remains elevated — the B strip doesn't fit in L1.
- **IPC** should improve noticeably.

<!-- TODO: Side-by-side screenshot comparison -->

---

## 5. Optimisation 2 — 2D Tiling (full blocking)

To eliminate L1 misses we tile all three dimensions. Now the inner loops work on TILE×TILE sub-blocks of A, B, and C that fit entirely in L1d cache. The kernel in `src/matmul_tiled_2d.cpp` is:

```cpp
constexpr int TILE = 64;

void matmul_tiled_2d(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int i_end = std::min(i0 + TILE, N);
            int j_end = std::min(j0 + TILE, N);
            for (int k0 = 0; k0 < N; k0 += TILE) {
                int k_end = std::min(k0 + TILE, N);
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

**Why TILE = 64?** A 64×64 tile of floats is 64 × 64 × 4 = 16 KB. Three tiles (A, B, C sub-blocks) total 48 KB, which fits comfortably in the Graviton3 L1d cache (64 KB per core). The inner loops run almost entirely out of L1, minimising both L1 and L2 stalls.

Run it:

```bash
./matmul_tiled_2d
```

You should see a further speedup over the 1D-tiled version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_tiled_2d` and look at the Top-Down view:

- **Backend Bound %** should drop further, and the sub-category should shift from **Memory Bound** toward **Core Bound** (the memory subsystem is no longer the primary bottleneck).
- **Retiring %** should be at its highest — more cycles are spent doing useful FP work.
- **IPC** should be the best across all three variants.

<!-- TODO: Screenshot -->

---

## 6. Summary — Comparing All Three Variants

Run all three back-to-back:

```bash
./matmul_naive
./matmul_tiled_1d
./matmul_tiled_2d
```

| Variant | Expected Improvement | Why |
|---------|---------------------|-----|
| Naive (ijk) | Baseline | Strided B access → constant cache misses |
| 1D tiled (k-strip) | ~3–5× faster | B strip fits in L2 → fewer L2 misses, but L1 misses remain |
| 2D tiled (i,j,k) | ~5–10× faster than naive | Working set fits in L1d → near-zero cache misses |

<!-- TODO: ATP side-by-side comparison screenshot showing all three profiles -->

### What the ATP Top-Down view tells us

| Metric | Naive | 1D Tiled | 2D Tiled |
|--------|-------|----------|----------|
| Backend Bound % | High | Medium | Low |
| → Memory Bound | Dominant | Reduced | Minimal |
| Retiring % | Low | Medium | High |
| IPC | < 1.0 | ~1–2 | ~2–3+ |

The progression is clear: by tiling the k dimension first (1D tiling) we reduce L2 misses, then by tiling all dimensions (2D tiling) we fit the working set in L1d. Each step moves the workload from data-starved toward compute-efficient. ATP's Top-Down Methodology makes it straightforward to identify the bottleneck category and verify that each optimisation addressed it.

---

## Key Takeaways

1. **Start with the Top-Down view.** It tells you *where* to look — don't guess, measure.
2. **Backend Memory Bound** means the CPU is waiting for data. Fix cache utilisation first.
3. **1D tiling** (blocking a single loop dimension) is a good first step — it reduces pressure on outer cache levels.
4. **2D tiling** (blocking all dimensions) pushes performance further by keeping the working set in the fastest level of the memory hierarchy.
5. **Always re-profile after each change** to confirm the optimisation had the expected effect. ATP makes this comparison easy.
