# Tutorial 6: Capstone — Optimising HNSWlib Vector Search with Arm Total Performance

Performance problems in real-world systems rarely live in a single hot loop. A production workload typically involves multiple interacting subsystems — distance computation, graph traversal, memory allocation — each with its own performance profile. Optimising one bottleneck often reveals the next, and the improvement comes from fixing them one at a time with evidence from profiling.

This tutorial applies the full **profile, diagnose, fix, re-profile** workflow from the earlier tutorials to a real-world C++ library: **HNSWlib**, a widely used header-only implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest-neighbor (ANN) search. The application is **image similarity search** — given a query image embedding, find the most similar images in a database of 500,000 embeddings.

You will work through three optimisation passes. Each pass uses a different ATP recipe to diagnose a different category of hardware-level bottleneck:

1. **Instruction Mix** reveals that the L2 distance function is entirely scalar — no NEON SIMD instructions despite processing 128-dimensional float vectors.
2. **Topdown** reveals that over-provisioned graph parameters (large `M` and `ef_search`) cause elevated Backend Bound from excessive neighbor evaluation and candidate queue pressure.
3. **Memory Access** reveals poor cache locality during graph traversal because queries arrive in random order, causing each search to enter the graph at an unrelated point.

By the end of this tutorial, you will know how to:

1. Use the **Instruction Mix** recipe to detect scalar floating-point code that should be vectorised.
2. Use the **Topdown** recipe to identify Backend Bound caused by over-provisioned algorithmic parameters.
3. Use the **Memory Access** recipe to diagnose cache-unfriendly access patterns in pointer-chasing graph traversals.
4. Apply the profile → diagnose → fix → re-profile loop to a multi-component real-world workload.

## Before you begin

- An AWS Graviton 2/3 instance
- C++ compiler (g++ 9+ or clang++ 14+)
- CMake 3.16+
- Python 3 with `numpy` (for embedding generation only)
- ATP installed and configured

## Terms used in this tutorial

- **HNSW**: Hierarchical Navigable Small World — a graph-based algorithm for approximate nearest-neighbor search. It builds a multi-layer graph where each node connects to its nearest neighbors. Search starts at the top layer and greedily descends, evaluating distances at each hop.
- **ANN**: Approximate Nearest Neighbor — finding the closest vectors in a dataset without an exhaustive scan. ANN algorithms trade a small amount of accuracy (recall) for large speedups over brute force.
- **Embedding**: A fixed-length vector (e.g., 128 floats) that represents an image, text, or other data point in a high-dimensional space. Similar items have nearby embeddings.
- **Recall@K**: The fraction of true top-K nearest neighbors that the approximate search actually returns. A recall of 0.95 means 95% of the true neighbors are found.
- **M**: The maximum number of connections (edges) each node has in the HNSW graph. Larger M means more neighbors to evaluate per hop — higher recall but more work.
- **ef_search**: The size of the dynamic candidate list during search. Larger ef_search explores more candidates — higher recall but slower queries.
- **L2 distance**: The squared Euclidean distance between two vectors: `sum((a[i] - b[i])^2)`. This is the distance metric used throughout this tutorial.

---

## Background: HNSW and Why It Is a Good ATP Target

HNSWlib is a C++ header-only library that implements the HNSW algorithm for approximate nearest-neighbor search. It is the engine behind many vector databases and search libraries (Chroma, Milvus's CPU backend, and others). The entire implementation is approximately 2,000 lines of C++ in a handful of header files.

Three properties make HNSWlib an ideal target for a capstone profiling exercise:

**The distance function dominates the instruction stream.** Every search query evaluates L2 distance against hundreds or thousands of candidate vectors. With 128-dimensional float32 vectors, each distance call performs 128 subtractions, 128 multiplications, and 128 additions. On AArch64 without explicit SIMD, HNSWlib's default `L2Sqr` function processes one float at a time — a scalar loop that the Instruction Mix recipe will reveal clearly.

**Graph parameters control backend pressure.** The HNSW graph is parameterised by `M` (edges per node) and `ef_search` (candidate queue size). Over-provisioned values create a dense graph where each hop evaluates many neighbors, and the priority queue that manages candidates grows large. This generates backend pressure from heap operations and memory accesses that the Topdown recipe will expose.

**Graph traversal is pointer-chasing.** HNSW search follows edges from node to node in the graph. Nodes are allocated incrementally during index construction, so nearby nodes in graph space are not nearby in memory. When queries arrive in random order, consecutive searches enter the graph at unrelated points, preventing temporal locality. The Memory Access recipe will show this as elevated load latency and low cache hit rates.

> **Note on measurement:** ATP's sampling-based approach means that metrics are most reliable when the profiled phase runs for several seconds. The build phase (index construction) runs long enough naturally; the search phase may need a large query set to accumulate sufficient samples.

---

## The Workload

An image search system indexes 500,000 image embeddings (128-dimensional float32 vectors, simulating features from a model like ResNet or CLIP) and answers similarity queries: given a new image embedding, return the 10 most similar images.

The pipeline has two phases:

```
Embeddings file  →  [Build Index]  →  HNSW graph  →  [Search]  →  Top-10 results
```

**Build** reads 500,000 base vectors from a binary file and inserts them into an HNSW index, creating the multi-layer graph.

**Search** reads 10,000 query vectors and, for each one, traverses the HNSW graph to find the approximate top-10 nearest neighbors. Results are compared against brute-force groundtruth to compute recall.

Expected output format:

```
=== Baseline Image Search ===

Loading 500000 embeddings (dim=128)...
[1/2] Building index  (M=48, ef_construction=200)
      42.3 s

[2/2] Searching  (ef_search=200, k=10, 10000 queries)
      8.1 s  (1234.6 queries/sec)
      Recall@10: 0.982

Total: 50.4 s
```

The timings will vary by instance type.

---

## Generate the Test Data

```bash
cd tutorial_6
pip install numpy
python scripts/generate_embeddings.py
```

This creates four files in `data/`:
- `embeddings.bin` — 500,000 × 128 float32 vectors (~244 MB)
- `queries.bin` — 10,000 × 128 float32 vectors (~4.9 MB)
- `groundtruth.bin` — 10,000 × 10 int32 labels (brute-force exact top-10)
- `data_config.h` — C++ header with `#define` constants for dimensions

---

## Build the Code

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

This produces four executables in `build/`:
- `search_baseline` — scalar distance, over-provisioned parameters, random query order
- `search_pass1` — NEON distance, same parameters, random query order
- `search_pass2` — NEON distance, tuned parameters, random query order
- `search_optimised` — NEON distance, tuned parameters, sorted query order

---

## Run the Baseline

```bash
./build/search_baseline
```

Note the per-phase timings — both build and search are slower than they need to be. The baseline is deliberately sub-optimal in three ways, each of which will be diagnosed by a different ATP recipe.

---

## Profile the Baseline with ATP

### Step 1: CPU Cycle Hotspots — find where time is spent

Open ATP and select **Recipes → CPU Cycle Hotspots**. Choose `./build/search_baseline` as the target command. When it finishes, open the results.

In the **Functions** tab, look for the functions that dominate CPU time. You should see:

- `L2Sqr` (or a mangled name containing `L2Sqr`) — the scalar distance function
- Functions from `HierarchicalNSW` related to search and graph traversal
- Priority queue operations (`std::push_heap`, `std::pop_heap`)

<p align="center">
<img src="assets/baseline_hotspots.png" width="850" alt="CPU Cycle Hotspots for baseline — L2Sqr and graph traversal functions dominate"/>
</p>

The CPU Hotspots view tells you *where* time is spent, but not *why*. The distance function is hot, but you cannot tell from cycle counts alone whether the problem is lack of vectorisation, memory stalls, or branch misprediction. To understand the nature of the bottleneck, you need a specialised recipe.

**Diagnosis so far:** The hot functions are L2 distance computation and HNSW graph traversal. The next step is to characterise *what kind* of work these functions are doing.

### Step 2: Instruction Mix — diagnose the scalar distance computation

Open ATP and select **Recipes → Instruction Mix**. Choose `./build/search_baseline` as the target.

In the **Instruction Mix** summary, look at the percentage breakdown of instruction types:

<p align="center">
<img src="assets/baseline_instruction_mix.png" width="850" alt="Instruction Mix for baseline — high scalar FP %, zero NEON"/>
</p>

In the baseline, you should see something like:

| Category   | Expected Range |
|------------|---------------|
| Integer    | 35–50%        |
| Branch     | 15–25%        |
| Scalar FP  | 15–30%        |
| NEON       | 0–2%          |

The critical signal is the combination of **high scalar FP percentage** and **near-zero NEON percentage**. The L2 distance function is processing 128-dimensional float32 vectors one element at a time. Each iteration performs a subtract and a fused multiply-add on a single float — pure scalar floating point. With NEON, the same work could process 4 floats per instruction (128-bit registers) or even 16 floats per iteration with loop unrolling.

**Complete diagnosis: High scalar FP % and near-zero NEON % → the L2 distance function is entirely scalar. With 128 dimensions and millions of distance evaluations, this is the dominant bottleneck. The fix is to replace the scalar loop with NEON intrinsics.**

> **What to look for:** Compare the scalar FP % and NEON % in the baseline against the same metrics after Pass 1. A drop in scalar FP % and a rise in NEON % confirms that the distance function is now vectorised.

---

## Pass 1: NEON Distance Function (Instruction Mix)

### The problem

Open `src/search_baseline.cpp` and note the `#define NO_MANUAL_VECTORIZATION` at the top. This tells HNSWlib to skip its x86 SIMD paths (SSE/AVX). On AArch64, HNSWlib has no built-in NEON path, so it falls back to the scalar `L2Sqr` function in `hnswlib/space_l2.h`:

```cpp
static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}
```

This loop processes one float per iteration. With `dim=128`, each call executes 128 scalar subtracts, 128 scalar multiplies, and 128 scalar adds. The baseline is compiled with `-fno-tree-vectorize` to prevent the compiler from auto-vectorising this loop, ensuring a clear contrast.

### The fix

In `src/search_pass1.cpp`, we define a custom `L2SpaceNeon` class that provides a NEON-accelerated distance function using Arm NEON intrinsics:

```cpp
static float L2SqrNeon(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* a = static_cast<const float*>(pVect1v);
    const float* b = static_cast<const float*>(pVect2v);
    size_t dim = *static_cast<const size_t*>(qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Process 16 floats per iteration (4 x float32x4_t)
    for (; i + 15 < dim; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        sum = vfmaq_f32(sum, d0, d0);
        sum = vfmaq_f32(sum, d1, d1);
        sum = vfmaq_f32(sum, d2, d2);
        sum = vfmaq_f32(sum, d3, d3);
    }

    // Handle remaining 4-float chunks
    for (; i + 3 < dim; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        sum = vfmaq_f32(sum, d, d);
    }

    float result = vaddvq_f32(sum);

    // Scalar tail
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}
```

Key intrinsics:
- `vld1q_f32` — load 4 floats into a 128-bit NEON register
- `vsubq_f32` — subtract 4 float pairs in parallel
- `vfmaq_f32` — fused multiply-add on 4 floats (accumulate `d*d` into `sum`)
- `vaddvq_f32` — horizontal add (reduce 4-lane sum to a single float)

The inner loop is unrolled 4× to process 16 floats per iteration. With `dim=128`, this means 8 iterations of the main loop — 8 iterations replacing 128 scalar iterations.

HNSWlib's `SpaceInterface` allows plugging in a custom distance function. The `L2SpaceNeon` class simply returns `L2SqrNeon` as its distance function, and HNSWlib uses it for all distance evaluations during build and search.

### Run Pass 1

```bash
./build/search_pass1
```

Compare the per-phase times against the baseline. Both build and search should be faster because every distance evaluation is now vectorised.

### Re-profile: did it work?

Open ATP and select **Recipes → Instruction Mix** again, this time targeting `./build/search_pass1`.

<p align="center">
<img src="assets/pass1_instruction_mix.png" width="850" alt="Instruction Mix after Pass 1 — scalar FP % down, NEON % up"/>
</p>

Compare the instruction breakdown against the baseline:

| Category  | Baseline   | Pass 1 (expected) |
|-----------|------------|--------------------|
| Scalar FP | 15–30%     | 2–8%               |
| NEON      | 0–2%       | 20–40%             |

The scalar FP percentage should drop dramatically and the NEON percentage should rise. This confirms that the distance function is now using SIMD instructions. The distance computation bottleneck is resolved, but the pipeline is still slower than it needs to be.

---

## Pass 2: Tune HNSW Parameters (Topdown)

### Profile Pass 1 with Topdown

Now that the distance computation is vectorised, profile the Pass 1 pipeline with a different recipe. Open ATP and select **Recipes → Topdown**. Choose `./build/search_pass1` as the target.

In the Topdown summary, look at the four buckets:

<p align="center">
<img src="assets/pass1_topdown.png" width="850" alt="Topdown summary after Pass 1 — Backend Bound elevated"/>
</p>

You should see **Backend Bound** accounting for a significant share of slots during the search phase. The reason is the over-provisioned graph parameters:

- **M=48** means each node in the HNSW graph has up to 48 connections. At each hop during search, the algorithm evaluates the distance to up to 48 neighbors — far more than necessary for high recall.
- **ef_search=200** means the search maintains a candidate list of up to 200 entries. Each candidate insertion and extraction involves heap operations (`push_heap`, `pop_heap`) that access memory-resident priority queue data.

The combination creates excessive work per search query: too many distance evaluations (even though each one is now fast with NEON), too many heap operations, and too much data movement through the cache hierarchy. This shows up as Backend Bound because the CPU spends cycles waiting for data rather than doing useful arithmetic.

**Complete diagnosis: Backend Bound is elevated → the search phase performs excessive work due to over-provisioned parameters (M=48 creates too many edges, ef_search=200 creates too large a candidate queue). The fix is to reduce M and ef_search to values that maintain acceptable recall with less work per query.**

### The fix

In `src/search_pass2.cpp`, the HNSW parameters are tuned:

```
M:               48 → 16    (fewer edges per node)
ef_construction: 200 → 100  (build-time candidate list)
ef_search:       200 → 64   (search-time candidate list)
```

These values are typical for production vector search deployments. `M=16` provides enough connectivity for good recall on 128-dimensional data, and `ef_search=64` explores enough candidates to find the true top-10 with high probability.

> **Trade-off:** Reducing parameters lowers recall slightly. The key insight from Topdown is that the *hardware-level* cost of the over-provisioned parameters (Backend Bound) far exceeds the *algorithmic* benefit (marginal recall improvement). ATP gives you the evidence to make this trade-off quantitatively.

### Run Pass 2

```bash
./build/search_pass2
```

Both build and search should be significantly faster. The build is faster because `M=16` means fewer edges to maintain during insertion. The search is faster because each query evaluates fewer candidates.

### Re-profile: did it work?

Open ATP and select **Recipes → Topdown** again, this time targeting `./build/search_pass2`.

<p align="center">
<img src="assets/pass2_topdown.png" width="850" alt="Topdown summary after Pass 2 — Backend Bound reduced, Retiring increased"/>
</p>

Compare the Topdown buckets against Pass 1:

| Bucket        | Pass 1 | Pass 2 (expected) |
|---------------|--------|--------------------|
| Backend Bound | High   | Lower              |
| Retiring      | Low    | Higher             |

The shift from Backend Bound to Retiring indicates that the CPU is spending more of its time doing useful work (distance computation, result extraction) and less time waiting for memory from oversized data structures. The search throughput (queries/sec) should increase substantially while recall remains above 0.95.

Two bottlenecks down. The vectorisation and parameter overheads are fixed. The next step is to look at the remaining memory access patterns with a recipe that examines cache behaviour directly.

---

## Pass 3: Sort Queries for Cache Locality (Memory Access)

### Profile Pass 2 with Memory Access

With vectorisation and parameter tuning in place, the remaining bottleneck is in how the HNSW graph is traversed. Open ATP and select **Recipes → Memory Access**. Choose `./build/search_pass2` as the target.

In the Memory Access summary, look at:
- **L1D cache hit rate** — the fraction of data loads satisfied by the L1 data cache
- **Average load latency** — the mean number of cycles each load instruction takes to complete
- **SPE sample locations** — which functions have the most cache-miss samples

<p align="center">
<img src="assets/pass2_memory_access.png" width="850" alt="Memory Access metrics after Pass 2 — L1D hit rate and load latency for graph traversal"/>
</p>

HNSW search is fundamentally a graph walk: starting from an entry point, the algorithm follows edges to neighboring nodes, evaluating distances at each hop and maintaining a priority queue of candidates. Each node in the graph stores:
- A list of neighbor IDs (pointers to other nodes)
- The node's embedding vector (128 × 4 = 512 bytes)

Nodes are allocated during index construction in insertion order. When search queries arrive in random order, each query enters the graph at an unrelated point in embedding space. Consecutive queries visit completely different subsets of the graph, so the nodes accessed by query N have no overlap with the nodes accessed by query N+1. This means every new query brings a fresh set of cache misses — the graph nodes from the previous query have been evicted.

The Memory Access recipe reveals this as:
- Lower-than-expected L1D hit rate
- Elevated average load latency
- SPE samples concentrated in the graph traversal and distance evaluation functions

**Complete diagnosis: L1D hit rate is lower than expected and load latency is elevated → consecutive queries traverse unrelated graph regions because they arrive in random order. The fix is to sort queries by similarity so consecutive queries traverse overlapping graph paths, keeping visited nodes warm in cache.**

### The fix

In `src/search_optimised.cpp`, the queries are sorted by their first dimension value before searching:

```cpp
std::vector<size_t> order(n_query);
std::iota(order.begin(), order.end(), 0);
std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return queries[a * dim] < queries[b * dim];
});
```

After sorting, consecutive queries have similar first-coordinate values, which is a rough proxy for spatial proximity in embedding space. Similar queries enter the HNSW graph at nearby entry points and traverse overlapping subsets of the graph. The nodes visited by query N are still in L1/L2 cache when query N+1 visits many of the same nodes.

> **Trade-off:** Sorting adds a small cost (sorting 10,000 indices is negligible), and results must be mapped back to original query indices for recall measurement. The cache locality improvement during the search phase more than compensates. In production, more sophisticated approaches exist — clustering queries into batches, or partitioning the index — but the simple sort demonstrates the principle.

### Run the Optimised Pipeline

```bash
./build/search_optimised
```

### Re-profile: did it work?

Open ATP and select **Recipes → Memory Access** again, this time targeting `./build/search_optimised`.

<p align="center">
<img src="assets/optimised_memory_access.png" width="850" alt="Memory Access metrics for optimised search — L1D hit rate improved, load latency reduced"/>
</p>

Compare cache metrics against Pass 2:

| Metric              | Pass 2    | Optimised (expected) |
|---------------------|-----------|----------------------|
| L1D hit rate        | ~85–90%   | ~92–97%              |
| Avg load latency    | Higher    | Lower                |

The rise in L1D hit rate and drop in average load latency confirm that the sorted query order improved cache locality during graph traversal. The three bottlenecks have been systematically identified and resolved.

---

## Summary

You started with a baseline image search pipeline using HNSWlib and used three different ATP recipes to diagnose and fix three distinct performance bottlenecks:

| Pass | ATP Recipe       | Problem                                    | Fix                                    |
|------|------------------|--------------------------------------------|----------------------------------------|
| 1    | Instruction Mix  | Scalar L2 distance (no SIMD)               | NEON intrinsics (4x float per instruction) |
| 2    | Topdown          | Over-provisioned M and ef_search → Backend Bound | Tuned parameters (M=16, ef_search=64) |
| 3    | Memory Access    | Random query order → cache-cold graph traversal | Sort queries by similarity             |

Each pass followed the same loop: profile to find the dominant bottleneck, apply a targeted fix, then re-profile to confirm the metric shifted as expected. The final optimised pipeline produces the same search results as the baseline (within recall tolerance).

### File reference

| File | Description |
|------|-------------|
| `scripts/generate_embeddings.py` | Generates the synthetic embedding dataset |
| `src/search_baseline.cpp` | Baseline — scalar distance, large params, random queries |
| `src/search_pass1.cpp` | After Pass 1 — NEON distance, other problems remain |
| `src/search_pass2.cpp` | After Pass 2 — NEON + tuned params, random queries remain |
| `src/search_optimised.cpp` | Fully optimised — all three fixes applied |
| `src/hnswlib/` | HNSWlib v0.8.0 header-only library |

### Key takeaway

The diagnostic method matters more than the specific fixes. Scalar distance computation, over-provisioned algorithmic parameters, and cache-unfriendly traversal patterns are common in graph-based search libraries, but they rarely announce themselves in source code. ATP's hardware-level recipes give you the evidence to identify each one and verify that your fix actually addressed the root cause — not just the symptom.
