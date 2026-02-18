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

## Background: The Top-Down Methodology

Before profiling anything, it helps to understand what the Top-Down view is actually telling you.

### The abstract CPU model

The Arm CPU pipeline has three major phases:

- **Frontend** — fetches instructions from the instruction cache, decodes them into micro-ops, and feeds them into the backend.
- **Backend** — executes micro-ops using the CPU's execution units (integer, FP, SIMD, load/store).
- **Retire** — completes micro-ops that were executed on the correct code path and updates the architectural state.

Each cycle, the CPU has a fixed number of *slots* — the maximum number of micro-ops it can issue to the backend per cycle (the *issue width*). The Top-Down Methodology accounts for every slot in every cycle.

### The four top-level categories

ATP's Topdown recipe divides 100% of slot capacity into four mutually exclusive buckets:

| Category | What it means | Common causes |
|---|---|---|
| **Retiring** | Slots doing *useful* work (higher = better) | — |
| **Frontend Bound** | Frontend can't supply micro-ops fast enough | Instruction cache misses, branch mispredictions |
| **Backend Bound** | Backend can't accept micro-ops (data or execution stalls) | Cache misses, memory bandwidth, execution unit contention |
| **Bad Speculation** | Slots wasted on instructions from the wrong code path | Branch mispredictions, mis-speculated loads |

A high **Retiring** percentage means the hardware is being used efficiently. A high **Backend Bound** or **Frontend Bound** percentage means the CPU is frequently stalled and waiting.

### How to use it: drill down, don't guess

The power of the methodology is its hierarchical structure. You start at the top and drill down only into the dominant bucket:

```
Topdown
├── Frontend Bound  ──► instruction cache MPKI, ITLB walks, branch MPKI
├── Backend Bound   ──► memory bound (L1/L2/LLC/DRAM miss rates) or core bound (execution unit contention)
├── Bad Speculation ──► branch misprediction ratio, Branch MPKI
└── Retiring        ──► instruction mix (scalar vs SIMD, integer vs FP)
```

For example: if **Backend Bound** dominates, you drill into *Memory Bound* vs *Core Bound*. If it's Memory Bound, you look at which cache level the misses are hitting (L1, L2, LLC, or DRAM). That tells you exactly what to fix — and where to stop optimising.

### How the Topdown recipe relates to ATP's other recipes

ATP provides several recipes. They are complementary, not competing:

- **CPU Cycle Hotspots** — tells you *where* time is spent (which functions are hot). Use this first to find targets.
- **Topdown** — tells you *why* those functions are slow (which pipeline phase is the bottleneck). Use this to diagnose.
- **Memory Access** — deep-dives into memory latency using Arm's Statistical Profiling Extension (SPE), showing per-function latency broken down by L1/L2/LLC/DRAM. Use this to confirm a memory-bound diagnosis.
- **Instruction Mix** — shows the breakdown of instruction types (scalar, SIMD, FP, loads/stores). Use this to check whether vectorisation is actually happening.

In this tutorial we focus on the **Topdown** recipe, which surfaces enough information to guide each optimisation step.

> **Note on measurement bias:** ATP's sampling method introduces a known systematic bias — Retiring is reported slightly *lower* than reality, and Frontend Bound and Bad Speculation are reported slightly *higher*. Use the values to compare runs against each other and look for trends, not as absolute ground truth.

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

Matrix multiplication (`C = A × B`) is one of the most fundamental operations in modern computing — it underpins fully-connected layers, attention mechanisms, and convolutional layers in machine learning models. The naive three-loop implementation below computes the correct result but makes no attempt to exploit the hardware efficiently:

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

Run it to get a baseline timing:

```bash
./matmul_naive
```

### Profiling with ATP

To understand *why* this is slow, run the Topdown recipe in ATP. Select **Recipes → Topdown** and choose the `matmul_naive` executable.

<img src="assets/run_recipe.png" width="600" alt="Selecting the Topdown recipe in ATP"/>

The run takes around 30 seconds. Open the **Summary** view:

<img src="assets/naive_topdown.png" width="600" alt="ATP Topdown summary for naive matmul"/>

**Backend Bound** dominates. This means the backend cannot execute micro-ops fast enough — not because the frontend is failing to supply them, but because the execution units are blocked waiting on data from memory.

To understand which level of the memory hierarchy is responsible, look at the cache effectiveness metrics in the Functions tab:

<img src="assets/naive_cache_effectiveness.png" width="600" alt="Cache miss ratios for naive matmul"/>

| Cache level | Miss ratio | Meaning |
|---|---|---|
| L1D | ~49% | Nearly half of all data loads miss L1 |
| L2 | ~24% | A quarter of those that reached L2 also miss |
| LL (Last Level) | ~61% | 61% of LLC accesses miss — data must come from DRAM |

The LLC miss rate is the critical number. A 61% miss rate at the last-level cache means the majority of memory accesses cannot be satisfied by any on-chip cache. Each miss triggers a round-trip to DRAM, which costs hundreds of cycles and stalls the pipeline while it waits.

### Why does this happen?

Look at the hot inner loop:

```cpp
for (int k = 0; k < K; ++k) {
    sum += A[i * K + k] * B[k * N + j]; // B access strides by N
}
```

For matrix **A**, the access `A[i*K + k]` steps through contiguous memory as `k` increments — the hardware prefetcher handles this well and most A accesses hit L1.

For matrix **B**, the access `B[k*N + j]` jumps by `N` elements on each step of `k`. With N = 8192, that is a stride of 8192 × 4 = **32,768 bytes** (512 cache lines) between consecutive accesses. The entire B matrix is 8192 × 8192 × 4 = **256 MB**, far exceeding Graviton3's last-level cache (~32 MB). The hardware prefetcher cannot track these large strides, so virtually every B access misses all cache levels and waits on DRAM.

This is exactly what the Topdown view confirmed: **Backend Bound → Memory Bound → LLC misses**. Now we know where to focus the first optimisation.

---

## 3. Optimisation 1 — 2D Tiling (cache blocking)

To eliminate LLC misses we tile all three loop dimensions (i, j, k). The inner loops work on TILE×TILE sub-blocks of A, B, and C that fit in cache. The kernel in `src/matmul_tiled.cpp` is:

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

**Why TILE = 128?** A 128×128 tile of floats is 128 × 128 × 4 = 64 KB. Three tiles (A, B, C sub-blocks) total 192 KB, which fits comfortably in Graviton3's L2 cache (1 MB per core). The goal is to keep the working set in L2 cache to avoid expensive DRAM accesses.

Now let's see if this change can alleviate the Backend Bound bottleneck. Start another Topdown analysis using the `matmul_tiled` executable instead of `matmul_naive`.

Run the Topdown recipe on `matmul_tiled`:

```bash
# ATP: Select Recipes → Topdown, choose matmul_tiled executable
```

You should get a Topdown analysis that looks like this:

<img src="assets/tiled_topdown.png" width="600" alt="ATP Topdown summary for tiled matmul"/>

### Analysis: What Changed?

The improvement is dramatic:

| Metric | Naive | Tiled | Improvement |
|--------|-------|-------|-------------|
| **Retiring** | ~15% | **66.49%** | 4.4× more useful work |
| **Backend Bound** | ~70% | **18.91%** | 3.7× reduction |
| **Frontend Bound** | ~13% | **13.8%** | Similar |

**Retiring** now dominates at 66.49% — this means the pipeline is successfully completing instructions and doing useful work. Backend Bound dropped from ~70% to just 18.91%. The memory bottleneck has been largely eliminated.

### What's Happening in the Caches?

Look at the cache effectiveness metrics for the tiled version:

<img src="assets/tiled_cache_effectiveness.png" width="600" alt="Cache metrics for tiled matmul"/>

| Cache level | Miss Ratio | vs. Naive |
|-------------|------------|-----------|
| L1D | 2.38% | ~49% → 2.38% |
| L2 | ~0% | ~24% → ~0% |
| LLC | ~0% | ~61% → ~0% |

**The transformation is remarkable:**

1. **L1 Cache hits**: The innermost loop works on 128 consecutive elements of B and C (512 bytes each), which fit comfortably in L1 cache (64 KB). As the loop iterates, it accesses these row segments sequentially — exactly what L1 is optimized for. The L1 miss ratio dropped from 49% to just 2.38%.

2. **L2 Cache efficiency**: When the loop moves to the next row, some data evicts from L1. However, the entire 128×128 tile (64 KB) fits easily in L2 cache (1 MB), so these L1 misses hit L2 instead of going to DRAM. L2 miss ratio is effectively 0%.

3. **LLC misses eliminated**: The working set of three tiles (A-tile, B-tile, C-tile totaling 192 KB) remains resident in L2 throughout the computation of each tile block. Data never needs to be fetched from DRAM, so LLC miss ratio dropped from 61% to near zero.

**The key insight**: The innermost loop's working set is much smaller than a full 128×128 tile. It only needs:
- 1 element of A (`a_ik`) = 4 bytes
- 128 consecutive elements of B = 512 bytes  
- 128 consecutive elements of C = 512 bytes
- **Total ≈ 1 KB** — easily fits in L1

This is why we see such excellent L1 performance. Tiling provides a hierarchy of benefits: L1 caches the hot inner loop data, L2 caches the full tiles, and DRAM access is eliminated entirely.

---

## 4. Optimisation 2 — NEON Intrinsics and Register Blocking

With the memory bottleneck eliminated, **Retiring** is now the dominant category at 66.49%. This means the pipeline is successfully completing instructions — but we can make those instructions do more work.

Look at the Speculative Operation Mix for the tiled version:

<img src="assets/tiled_retiring.png" width="400" alt="Operation mix for tiled matmul"/>

| Operation Type | Percentage |
|----------------|------------|
| Load Operations | 28.3% |
| Store Operations | 14.02% |
| Integer Operations | 29.14% |
| Floating Point Operations | 14.02% |
| **Advanced SIMD Operations** | **0%** |

**Advanced SIMD Operations are 0%** — the code is not using NEON vector instructions at all. Each floating-point multiply-add is executed as a scalar operation, processing one value at a time. ARM's NEON instruction set can process 4 floats simultaneously in a single instruction, giving us 4× the arithmetic throughput.

### The NEON Solution: Register Blocking + B-Tile Packing

The kernel in `src/matmul_neon.cpp` introduces two key optimizations:

1. **4×4 register blocking**: Process a 4×4 sub-block of C using NEON registers, with each `vfmaq_n_f32` instruction performing 4 multiply-adds simultaneously
2. **B-tile packing**: Copy the B tile into a contiguous micro-panel layout so the innermost k-loop reads B sequentially (all L1 hits) instead of striding by N

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

**Why TILE = 64?** A 64×64 tile of floats is 64 × 64 × 4 = 16 KB. Three tiles (A, B, C sub-blocks) total 48 KB, which fits comfortably in Graviton3's L1d cache (64 KB). This is smaller than our previous 128×128 tiles because the NEON micro-kernel has a larger working set of registers.

**Key insight**: Each `vfmaq_n_f32` instruction performs 4 multiply-adds in a single cycle, quadrupling arithmetic throughput. The four independent C accumulators (`c0`, `c1`, `c2`, `c3`) also expose instruction-level parallelism, allowing the out-of-order core to overlap FMA latencies.

Run the Topdown recipe on `matmul_neon`:

```bash
# ATP: Select Recipes → Topdown, choose matmul_neon executable
```

Now check the Speculative Operation Mix:

<img src="assets/neon_retiring.png" width="400" alt="Operation mix for NEON matmul"/>

| Operation Type | Percentage |
|----------------|------------|
| Load Operations | 39.8% |
| Store Operations | 0% |
| Integer Operations | 17.35% |
| **Advanced SIMD Operations** | **31.63%** |
| Floating Point Operations | 0% |

**Advanced SIMD Operations now account for 31.63%** of all operations — the NEON intrinsics are working. Scalar floating-point operations dropped to 0% because all arithmetic is now vectorized. The load percentage increased because we're loading 4 values at a time into vector registers.

**Arithmetic intensity**: Each iteration of the k-loop now performs:
- 1 vector load of B (4 floats)
- 4 scalar loads of A (1 float each)
- 4 vector FMA operations (4×4 = 16 multiply-adds)
- **16 FLOPs per ~5 memory operations** — much better compute-to-memory ratio

The combination of SIMD vectorization and improved memory access patterns (via B-tile packing) significantly increases the amount of useful computational work the processor can complete per cycle.

---
---

## 5. Summary — Comparing All Three Variants

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
