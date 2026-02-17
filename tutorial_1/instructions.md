# Tutorial 1: Top-Down Performance Analysis Methodology

In this tutorial you will use **Arm Total Performance (ATP)** to profile a dense matrix multiplication kernel — the fundamental operator behind fully-connected layers and attention in ML models. Starting from a naive implementation, you will use ATP's Top-Down Methodology view to identify the microarchitectural bottleneck, then apply two successive optimisations — cache tiling and register blocking with NEON — measuring their impact at each step.

By the end you will be able to:

- Collect a top-down microarchitectural profile with ATP on a Graviton instance.
- Interpret the four top-level categories: **Frontend Bound**, **Backend Bound**, **Retiring**, and **Bad Speculation**.
- Use 2D tiling and NEON register blocking to progressively shift a workload from memory-bound to compute-efficient.

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
| `matmul_naive` | ijk loop order — strided B access causes LLC misses |
| `matmul_tiled` | 2D tiling (i,j,k) — working set fits in L2 but not L1 |
| `matmul_neon`  | L1-fitting tiles + 4×4 NEON register blocking + B-tile packing |

All three compute the same result (`C = A × B` for a 512×8192 × 8192×8192 multiplication by default). The row count of A (M=512) is kept small to limit runtime while preserving the full memory access profile on the large B matrix. You can pass custom dimensions as `./matmul_naive M K N`, e.g. `./matmul_naive 256 4096 4096`.

---

## 2. Baseline — Naive Matmul

Run the naive version to get a baseline timing:

```bash
./matmul_naive
```

Expected output (times will vary by instance type):

```
Naive matmul (512x8192 * 8192x8192)
  Time:  <X> ms
  GFLOPS: <Y>
  Check:  C[0]=... C[M*N-1]=...
```

Record the time and GFLOPS — we will compare against the optimised versions later.

### Why is it slow?

The naive kernel in `src/matmul_naive.cpp` is:

```cpp
void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j]; // B access strides by N
            }
            C[i * N + j] = sum;
        }
    }
}
```

Each iteration accesses `B[k * N + j]` — jumping by N floats (32 KB for N=8192) on every step. This means every access to B lands in a different cache line. The full B matrix is 8192 × 8192 × 4 = 256 MB, far exceeding the last-level cache (~32 MB on Graviton3). Almost every B access results in an LLC miss and a round-trip to DRAM. The CPU spends most of its time waiting for data from memory rather than doing useful computation.

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
- **LLC miss rate** — will be high due to the strided B access pattern hitting DRAM on nearly every access.

---

## 4. Optimisation 1 — 2D Tiling (cache blocking)

To eliminate LLC misses we tile all three loop dimensions (i, j, k). The inner loops work on TILE×TILE sub-blocks of A, B, and C that fit in L2 cache. The kernel in `src/matmul_tiled.cpp` is:

```cpp
constexpr int TILE = 128;

void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    std::memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int i_end = std::min(i0 + TILE, M);
            int j_end = std::min(j0 + TILE, N);
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int k_end = std::min(k0 + TILE, K);
                for (int i = i0; i < i_end; ++i) {
                    for (int k = k0; k < k_end; ++k) {
                        float a_ik = A[i * K + k];
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

**Why TILE = 128?** A 128×128 tile of floats is 128 × 128 × 4 = 64 KB. Three tiles (A, B, C sub-blocks) total 192 KB, which fits comfortably in Graviton3's L2 cache (1 MB per core). The tiles do NOT fit in L1d (64 KB), so L1 misses remain — but they now hit L2 instead of DRAM. LLC misses are largely eliminated because the sub-blocks are re-used while resident in L2.

**What remains?** The working set exceeds L1d capacity, so every access within the inner loops still misses L1 and pays the L2 latency penalty (~10 cycles vs ~4 cycles for L1). The code is also still scalar — one float multiply-add per instruction — wasting the SIMD execution width.

Run it:

```bash
./matmul_tiled
```

You should see a significant speedup over the naive version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_tiled` and compare the Top-Down view to the naive version:

- **Backend Bound %** should drop substantially.
- **Memory Bound** sub-category should shift from LLC-miss-dominated to L1-miss-dominated — the data now lives in L2 instead of DRAM.
- **Retiring %** should increase — more cycles are spent doing useful work.
- **IPC** should improve noticeably.

<!-- TODO: Side-by-side screenshot comparison -->

---

## 5. Optimisation 2 — Register Blocking with NEON

The tiled version is still limited in two ways: (1) each tile exceeds L1d so accesses pay L2 latency, and (2) each instruction processes only a single float. We fix both by shrinking the tile to fit in L1d, packing B tiles for sequential access, and adding a 4×4 NEON micro-kernel. The kernel in `src/matmul_neon.cpp` is:

```cpp
constexpr int TILE = 64;

// Pack B[k0:k_end][j0:j_end] into micro-panel format.
// Layout: for each 4-column micro-panel, all k rows are stored
// contiguously so the micro-kernel streams through them linearly.
static void pack_B_tile(const float* B, float* packed,
                        int k0, int k_end, int j0, int j_end, int N) {
    float* dst = packed;
    for (int j = j0; j < j_end; j += 4) {
        for (int k = k0; k < k_end; ++k) {
            vst1q_f32(dst, vld1q_f32(&B[k * N + j]));
            dst += 4;
        }
    }
}

void matmul_neon(const float* A, const float* B, float* C, int M, int K, int N) {
    std::memset(C, 0, M * N * sizeof(float));

    // Scratch buffer for one packed B tile (at most TILE × TILE floats)
    std::vector<float> packed_B(TILE * TILE);

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int i_end = std::min(i0 + TILE, M);
                int j_end = std::min(j0 + TILE, N);
                int k_end = std::min(k0 + TILE, K);
                int k_len = k_end - k0;

                // Pack B tile so micro-kernel reads are sequential
                pack_B_tile(B, packed_B.data(), k0, k_end, j0, j_end, N);

                // Process the tile in 4×4 micro-blocks
                for (int i = i0; i < i_end; i += 4) {
                    const float* bp = packed_B.data();
                    for (int j = j0; j < j_end; j += 4) {
                        // Load 4×4 block of C into NEON registers
                        float32x4_t c0 = vld1q_f32(&C[(i + 0) * N + j]);
                        float32x4_t c1 = vld1q_f32(&C[(i + 1) * N + j]);
                        float32x4_t c2 = vld1q_f32(&C[(i + 2) * N + j]);
                        float32x4_t c3 = vld1q_f32(&C[(i + 3) * N + j]);

                        const float* bp_k = bp;
                        for (int k = k0; k < k_end; ++k) {
                            // Packed B: sequential read of B[k][j:j+4]
                            float32x4_t b = vld1q_f32(bp_k);
                            bp_k += 4;
                            // Each vfmaq_n_f32: C_row += A[row][k] * B[k][j:j+4]
                            c0 = vfmaq_n_f32(c0, b, A[(i + 0) * K + k]);
                            c1 = vfmaq_n_f32(c1, b, A[(i + 1) * K + k]);
                            c2 = vfmaq_n_f32(c2, b, A[(i + 2) * K + k]);
                            c3 = vfmaq_n_f32(c3, b, A[(i + 3) * K + k]);
                        }

                        // Store the 4×4 result back
                        vst1q_f32(&C[(i + 0) * N + j], c0);
                        vst1q_f32(&C[(i + 1) * N + j], c1);
                        vst1q_f32(&C[(i + 2) * N + j], c2);
                        vst1q_f32(&C[(i + 3) * N + j], c3);
                        bp += k_len * 4;  // advance to next micro-panel
                    }
                }
            }
        }
    }
}
```

**Why TILE = 64?** A 64×64 tile is 16 KB. Three tiles = 48 KB, which fits in L1d (64 KB). The inner loops now run almost entirely from L1, cutting access latency from ~10 cycles (L2) to ~4 cycles (L1).

**Why pack B?** In the tiled version the innermost k-loop accessed `B[k*N+j]` with stride N — one cache-line miss every few iterations. The `pack_B_tile` function copies each B tile into a contiguous micro-panel layout before the micro-kernel runs. After packing, the k-loop reads B sequentially, turning almost every access into an L1d hit.

**Why 4×4 register blocking?** Four `float32x4_t` accumulators (c0–c3) hold a 4×4 block of C entirely in NEON registers. Each `vfmaq_n_f32` performs 4 multiply-adds in a single instruction — 4× the work per instruction compared to the scalar version. The four independent accumulators also expose instruction-level parallelism, allowing the out-of-order core to overlap FMA latencies rather than stalling on a single dependency chain.

Run it:

```bash
./matmul_neon
```

You should see a further significant speedup over the tiled version.

### Re-profile with ATP

<!-- TODO: ATP profiling command and screenshot -->

Profile `matmul_neon` and look at the Top-Down view:

- **Backend Bound %** should drop further — both memory stalls (L1 hits) and core stalls (SIMD throughput) are addressed.
- **Retiring %** should be at its highest — more cycles are spent doing useful FP work, and each instruction retires more work.
- **IPC** should be the best across all three variants.

<!-- TODO: Screenshot -->

---

## 6. Summary — Comparing All Three Variants

Run all three back-to-back:

```bash
./matmul_naive
./matmul_tiled
./matmul_neon
```

| Variant | Bottleneck Addressed | Why It Helps |
|---------|---------------------|--------------|
| Naive (ijk) | Baseline | Strided B access → LLC misses on nearly every access |
| 2D tiled | LLC misses → L2 hits | 128×128 tiles fit in L2; data re-used while cache-resident |
| NEON register-blocked | L1 misses + scalar throughput + strided B access | 64×64 tiles fit in L1; B packing gives sequential reads; NEON does 4 FMAs per instruction |

<!-- TODO: ATP side-by-side comparison screenshot showing all three profiles -->

### What the ATP Top-Down view tells us

| Metric | Naive | 2D Tiled | NEON |
|--------|-------|----------|------|
| Backend Bound % | High | Medium | Low |
| → Memory Bound | LLC misses dominate | L1 misses (hitting L2) | Minimal (data in L1) |
| → Core Bound | Hidden by memory stalls | Visible (scalar FMA) | Reduced (SIMD FMA) |
| Retiring % | Low | Medium | High |
| IPC | < 1.0 | ~1–2 | ~2–4 |

The progression is clear:
1. The naive version is **memory-bound at the LLC level** — the CPU stalls waiting for DRAM.
2. 2D tiling eliminates LLC misses by keeping tiles in L2, but L1 misses and scalar execution remain — the bottleneck shifts to **L1 memory latency and core throughput**.
3. NEON register blocking shrinks tiles to fit in L1, B-tile packing ensures sequential memory access within the micro-kernel, and SIMD does 4× more work per instruction — the workload moves from data-starved toward **compute-efficient**.

ATP's Top-Down Methodology makes it straightforward to identify the bottleneck category at each step and verify that each optimisation addressed it.

---

## Key Takeaways

1. **Start with the Top-Down view.** It tells you *where* to look — don't guess, measure.
2. **Backend Memory Bound** means the CPU is waiting for data. Fix cache utilisation first.
3. **2D tiling** reduces LLC misses by keeping the working set in L2. Look for the Memory Bound sub-category to shift from LLC-dominated to L1-dominated.
4. **Register blocking with NEON and B-tile packing** attacks three remaining bottlenecks at once: shrinking tiles to fit in L1 reduces memory latency, packing B tiles ensures sequential access within the micro-kernel, and SIMD intrinsics increase the useful work retired per instruction.
5. **Always re-profile after each change** to confirm the optimisation had the expected effect. ATP makes this comparison easy.
