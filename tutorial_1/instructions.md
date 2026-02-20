# Tutorial 1: Top-Down Performance Analysis Methodology

In this tutorial you will learn to use **Arm Total Performance (ATP)** to diagnose and fix performance bottlenecks in a C++ workload. The example workload is dense matrix multiplication (`C = A x B`) in single-precision floating point, but the methodology applies to any compute-intensive code.

You will work through ATP's core diagnostic loop three times: **profile, diagnose, fix, re-profile**. Each iteration teaches you to read a different part of the ATP interface:

1. Run the Topdown recipe and read the Summary view to identify the bottleneck category.
2. Drill into cache effectiveness in the Functions tab to find which cache level is responsible.
3. Check the Speculative Operation Mix in the Retiring breakdown to assess SIMD utilisation.

By the end you will be comfortable using ATP's Topdown recipe to guide optimisation decisions on Graviton.

**Prerequisites:** AWS Graviton 2/3 instance, C++ compiler (g++ 9+ or clang++ 14+), CMake 3.16+, ATP installed and configured.

## Workload Definition: MatMul Operator

All three binaries in this tutorial implement the same row-major GEMM-like operator. Matrix `A` has shape `M x K`, matrix `B` has shape `K x N`, and the output `C` has shape `M x N`. Each element of C is computed as `C[i,j] = sum_{k=0..K-1} A[i,k] * B[k,j]`.

In flattened row-major memory, this is:

`C[i*N + j] += A[i*K + k] * B[k*N + j]`

The default problem size in this repo is `M=256, K=1024, N=8192`, and total floating-point work is approximately `2*M*K*N` FLOPs.

---

## Background: What the Top-Down View Shows You

Before opening ATP, it helps to understand what the numbers mean. You can skip this section and refer back to it as needed.

### The four buckets

Each cycle, the CPU has a fixed number of *slots*, which is the maximum number of micro-ops it can issue. ATP's Topdown recipe accounts for every slot in every cycle and places each one into one of four mutually exclusive categories.

**Retiring** represents slots doing useful work. A higher Retiring percentage is better. When Retiring dominates, you should check the instruction mix to see whether you are using SIMD.

**Frontend Bound** means the frontend cannot supply micro-ops fast enough. When this dominates, look at instruction-cache misses and branch MPKI.

**Backend Bound** means the backend is stalled waiting for data or execution units. When this dominates, drill into whether the stall comes from memory latency or execution unit contention.

**Bad Speculation** represents slots wasted on wrong-path instructions after a mispredicted branch. When this is high, check the branch misprediction ratio.

### The drill-down hierarchy

The power of Top-Down is that you never guess. You follow the dominant bucket downward through the hierarchy:

```
Topdown
├── Frontend Bound  → I-cache MPKI, ITLB walks, branch MPKI
├── Backend Bound   → Memory Bound (L1 / L2 / LLC / DRAM) or Core Bound
├── Bad Speculation → branch misprediction ratio
└── Retiring        → instruction mix (scalar vs SIMD, integer vs FP)
```

If Backend Bound dominates, drill into Memory Bound vs Core Bound. If Memory Bound, look at which cache level is missing. That tells you exactly what to fix.

### ATP's recipes and when to use them

ATP provides several complementary recipes. **CPU Cycle Hotspots** answers the question of *where* time is spent and is best used first to find your hot functions. **Topdown** answers *why* a function is slow and should be used second to diagnose the bottleneck category. **Memory Access** answers which cache level is stalling you and is useful for confirming a memory-bound diagnosis using SPE data. **Instruction Mix** tells you whether your code is actually vectorised and is most useful after an optimisation to verify SIMD usage.

This tutorial focuses on the **Topdown** recipe, which surfaces enough information to guide each optimisation step.

> **Note on measurement bias:** ATP's sampling reports Retiring slightly low and Frontend/Bad Speculation slightly high. Use values for **relative comparison between runs**, not as absolute ground truth.

---

## 1. Build the Code

```bash
cd tutorial_1
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces three executables: `matmul_naive`, `matmul_tiled`, and `matmul_neon`. All three compute the same result (`C = A x B`, default `256x1024` by `1024x8192`). You can pass custom dimensions as `./matmul_naive M K N`.

---

## 2. Profile the Baseline: Learning the Topdown Summary View

Run the naive implementation to get a baseline timing:

```bash
./matmul_naive
```

The naive kernel is a textbook three-loop matrix multiply. Below is the full kernel from `src/matmul_naive.cpp`:

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

The key access pattern is the innermost load `B[k*N + j]`: as `k` increments, the address jumps by `N` floats each time, which is a large stride of 32 KB per iteration.

We suspect this is slow, but *why*? This is where ATP comes in.

### Step 1: Run the Topdown recipe

Open ATP and select **Recipes -> Topdown**. Choose the `matmul_naive` executable as the target:

<img src="assets/run_recipe.png" width="850" alt="Selecting the Topdown recipe in ATP"/>

ATP will collect hardware performance counter data for approximately 30 seconds. When it finishes, it opens the **Summary** view automatically.

### Step 2: Read the Summary view

<img src="assets/naive_topdown.png" width="850" alt="ATP Topdown summary for naive matmul"/>

Look at the four top-level bars. **Backend Bound** dominates at roughly 70%. This tells you that the frontend is supplying micro-ops adequately, but the backend cannot execute them fast enough because it is stalled. Retiring is low at around 15%, meaning most of the CPU's capacity is wasted waiting rather than doing useful work.

**Your diagnosis so far:** The bottleneck is in the backend. But *what* in the backend? Memory latency? Execution unit contention? You need to drill down.

### Step 3: Drill into cache effectiveness

Look below to the **Backend Summary Breakdown**. ATP shows cache effectiveness metrics for this function:

<img src="assets/naive_cache_effectiveness.png" width="850" alt="Cache miss ratios for naive matmul"/>

The miss ratios tell a clear story. At **L1D**, nearly half of all data loads miss the cache (around 49%). Of those that reach **L2**, roughly a quarter miss again (24%). At the **LLC**, the situation is worst: around 61% of accesses miss, meaning the data has to travel all the way to DRAM.

The LLC miss ratio is the critical number here. A 61% miss rate at the last-level cache means the majority of memory requests cost hundreds of cycles each. This is why Backend Bound is so high.

### Step 4: Connect the diagnosis to the code

Now you know *what* the bottleneck is: LLC misses caused by the `B[k*N + j]` access pattern, which strides by N = 8192 elements (32 KB) per k-iteration. The full B matrix is 256 MB, far exceeding the ~32 MB LLC. The hardware prefetcher cannot track these large strides, so every access goes to DRAM.

**Complete diagnosis: Backend Bound -> Memory Bound -> LLC misses, caused by strided B access.**

This is the Top-Down workflow: Summary -> dominant bucket -> drill down -> connect to code.

---

## 3. Fix and Re-Profile: 2D Tiling

The diagnosis says to keep the working set in cache. The standard fix is **2D tiling**: partition the loops so the inner computation works on sub-blocks small enough to fit in L2.

The tiled kernel in `src/matmul_tiled.cpp` uses TILE=128. A 128x128 tile of floats is 64 KB; three tiles (A, B, and C sub-blocks) total 192 KB, fitting comfortably in Graviton3's 1 MB L2 cache:

```cpp
constexpr int TILE = 128;

void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    std::memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE)
      for (int j0 = 0; j0 < N; j0 += TILE) {
        int i_end = std::min(i0 + TILE, M);
        int j_end = std::min(j0 + TILE, N);
        for (int k0 = 0; k0 < K; k0 += TILE) {
            int k_end = std::min(k0 + TILE, K);
            for (int i = i0; i < i_end; ++i)
              for (int k = k0; k < k_end; ++k) {
                float a_ik = A[i * K + k];
                for (int j = j0; j < j_end; ++j)
                    C[i * N + j] += a_ik * B[k * N + j];
              }
        }
      }
}
```

### Re-profile: did it work?

Run the Topdown recipe again, this time on `matmul_tiled`. Always re-profile after a change and never assume your optimisation had the intended effect.

<img src="assets/tiled_topdown.png" width="850" alt="ATP Topdown summary for tiled matmul"/>

Comparing the Summary view with the naive run, the improvement is dramatic. **Retiring** jumped from ~15% to 66.5%, meaning the pipeline is now doing useful work most of the time. **Backend Bound** dropped from ~70% to 18.9%, showing that memory stalls have been largely eliminated.

Now check cache effectiveness in the Functions tab:

<img src="assets/tiled_cache_effectiveness.png" width="850" alt="Cache metrics for tiled matmul"/>

The cache numbers confirm the fix worked. L1D misses dropped from ~49% to just 2.4% because the inner loop's working set now fits in L1. L2 misses fell to effectively 0% because the full 64 KB tile fits comfortably in L2. LLC misses are also gone, meaning DRAM access has been eliminated entirely.

The LLC bottleneck is gone. The diagnosis has shifted: **Retiring now dominates**, which means the pipeline is completing instructions efficiently. But is it completing the *right kind* of instructions?

### New diagnosis: check the instruction mix

With the memory bottleneck removed, drill into the **Retiring** category. Open the **Speculative Operation Mix** panel:

<img src="assets/tiled_retiring.png" width="400" alt="Operation mix for tiled matmul"/>

The operation mix reveals the next problem. Loads account for 28.3% of operations, stores 14.0%, integer operations 29.1%, floating-point scalar operations 14.0%, and **Advanced SIMD 0%**. Every multiply-add is a scalar operation processing one float at a time. Arm NEON can process 4 floats per instruction, so the code is leaving roughly 4x throughput on the table.

**New diagnosis: Retiring is high but scalar-only. The bottleneck is now compute throughput, not memory.**

---

## 4. Fix and Re-Profile: NEON Register Blocking

The diagnosis says to vectorise the arithmetic. The NEON kernel in `src/matmul_neon.cpp` introduces two changes. First, a **4x4 register blocking** micro-kernel holds a 4x4 sub-block of C in NEON registers and uses `vfmaq_n_f32` to perform 4 multiply-adds per instruction. Second, **B-tile packing** rearranges each B tile into contiguous memory so the micro-kernel reads B sequentially (all L1 hits) instead of striding by N.

The tile size is also reduced to TILE=64. Three 64x64 tiles total 48 KB, which fits in L1d (64 KB) rather than just L2. The NEON micro-kernel can afford a smaller tile because it holds more data in registers:

```cpp
constexpr int TILE = 64;

static void pack_B_tile(const float* B, float* packed,
                        int k0, int k_end, int j0, int j_end, int N) {
    float* dst = packed;
    for (int j = j0; j < j_end; j += 4)
        for (int k = k0; k < k_end; ++k) {
            vst1q_f32(dst, vld1q_f32(&B[k * N + j]));
            dst += 4;
        }
}

void matmul_neon(const float* A, const float* B, float* C, int M, int K, int N) {
    std::memset(C, 0, M * N * sizeof(float));
    std::vector<float> packed_B(TILE * TILE);

    for (int i0 = 0; i0 < M; i0 += TILE)
      for (int j0 = 0; j0 < N; j0 += TILE)
        for (int k0 = 0; k0 < K; k0 += TILE) {
            int i_end = std::min(i0 + TILE, M), j_end = std::min(j0 + TILE, N);
            int k_end = std::min(k0 + TILE, K), k_len = k_end - k0;

            pack_B_tile(B, packed_B.data(), k0, k_end, j0, j_end, N);

            for (int i = i0; i < i_end; i += 4) {
                const float* bp = packed_B.data();
                for (int j = j0; j < j_end; j += 4) {
                    float32x4_t c0 = vld1q_f32(&C[(i+0)*N+j]);
                    float32x4_t c1 = vld1q_f32(&C[(i+1)*N+j]);
                    float32x4_t c2 = vld1q_f32(&C[(i+2)*N+j]);
                    float32x4_t c3 = vld1q_f32(&C[(i+3)*N+j]);

                    const float* bp_k = bp;
                    for (int k = k0; k < k_end; ++k) {
                        float32x4_t b = vld1q_f32(bp_k); bp_k += 4;
                        c0 = vfmaq_n_f32(c0, b, A[(i+0)*K+k]);
                        c1 = vfmaq_n_f32(c1, b, A[(i+1)*K+k]);
                        c2 = vfmaq_n_f32(c2, b, A[(i+2)*K+k]);
                        c3 = vfmaq_n_f32(c3, b, A[(i+3)*K+k]);
                    }

                    vst1q_f32(&C[(i+0)*N+j], c0);
                    vst1q_f32(&C[(i+1)*N+j], c1);
                    vst1q_f32(&C[(i+2)*N+j], c2);
                    vst1q_f32(&C[(i+3)*N+j], c3);
                    bp += k_len * 4;
                }
            }
        }
}
```

### Re-profile: is SIMD being used?

Run the Topdown recipe on `matmul_neon`. Go straight to the Speculative Operation Mix, as that is what we need to verify:

<img src="assets/neon_retiring.png" width="400" alt="Operation mix for NEON matmul"/>

The result confirms the optimisation worked. Scalar floating-point dropped from 14% to 0%, and **Advanced SIMD** jumped from 0% to 31.6%. Every multiply-add is now a NEON instruction processing 4 floats at once. Each iteration of the inner k-loop performs 4 vector FMAs (16 FLOPs) from roughly 5 memory operations, a much better compute-to-memory ratio than the scalar version.

---

## 5. The Full Picture

Run all three back-to-back and compare your ATP profiles:

```bash
./matmul_naive
./matmul_tiled
./matmul_neon
```

Across the three optimisation steps, the ATP Topdown view tells a consistent story of progress. For the naive kernel, Backend Bound dominates at around 70%, LLC misses are at 61%, and SIMD utilisation is 0%. After tiling, Retiring dominates at 66% and both LLC misses and Backend Bound drop to near zero, but SIMD remains at 0%. After adding NEON register blocking, SIMD utilisation reaches ~32% and Backend Bound stays minimal.

The diagnostic workflow follows the same pattern at each step. For the naive kernel, ATP pointed to Backend Bound, which drilled down to Memory Bound and then LLC misses caused by strided B access. The fix was 2D tiling to keep tiles in L2. For the tiled kernel, ATP showed Retiring was dominant but SIMD was 0%, pointing to scalar arithmetic as the bottleneck. The fix was a NEON 4x4 micro-kernel. After the NEON version, SIMD utilisation is at 31.6% and Backend Bound is minimal, indicating the workload is now compute-efficient.

The key is that **ATP told us what to fix at each step**. The Topdown Summary pointed to the bottleneck category, and drilling into cache effectiveness or the operation mix told us exactly what to change.

---

## Key Takeaways

**Follow the Top-Down workflow.** The pattern is: Summary view -> dominant bucket -> drill down -> connect to code -> fix -> re-profile. This is the loop you will use in every tutorial in this course.

**Backend Bound combined with a high LLC miss ratio** means your working set does not fit in cache. The fix is to tile your loops or restructure your data access so that the inner computation operates on smaller blocks.

**High Retiring with 0% SIMD** means the pipeline is busy but processing only one element at a time. Vectorising with NEON intrinsics can deliver up to 4x throughput improvement for 32-bit float arithmetic.

**Always re-profile after each change.** ATP makes it easy to compare runs and confirm your optimisation addressed the right bottleneck. Never assume a change improved performance without measuring.

**Use the right recipe for the question.** Topdown is for diagnosing bottleneck categories. CPU Cycle Hotspots is for finding which functions are hot. Memory Access provides detailed cache analysis. Instruction Mix verifies that vectorisation was applied correctly.
