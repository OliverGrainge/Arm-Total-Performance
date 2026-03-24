# Tutorial 6: Capstone — Optimising a CLIP Image Search Application with Arm Total Performance

In the previous tutorials you profiled small, self-contained workloads to learn individual ATP recipes. This capstone tutorial puts it all together: you will optimise the backend of a **real application** — a text-to-image search engine.

## What the application does

The application lets you type a text description — "a red sports car", "a cute puppy" — and instantly find the most visually matching images from a database of 50,000 photographs.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

### How it works

The key idea is **vector search**. A neural network called [CLIP](https://openai.com/research/clip) converts every image in the database into a list of 512 numbers (a "vector" or "embedding"). It does the same for the text query. Images that are visually similar to the text end up with similar vectors.

To find which image best matches a query, we measure the **L2 distance** between the query vector **q** and each image vector **x**:

$$d(\mathbf{q}, \mathbf{x}) = \sum_{i=1}^{512} (q_i - x_i)^2$$

The smaller the distance, the better the match. Searching for matching images becomes: *find the vectors in the database with the smallest distance to the query vector*.

With 50,000 images, computing this sum against every single vector (brute force) is too slow. Instead, the application uses [HNSWlib](https://github.com/nmslib/hnswlib), an open-source C++ library that builds a graph connecting similar vectors as neighbors. To answer a query, it walks the graph — hopping from neighbor to neighbor — to quickly find the closest matches without scanning the entire database. This is called **approximate nearest-neighbor (ANN) search**.

### What you will optimise

A single interactive query returns in milliseconds. But the backend that **builds the index** and **handles bulk query traffic** is where the real cost lives. When you need to re-index 50,000 images or serve 10,000 queries in a batch, every inefficiency adds up. That is the workload you will profile and optimise.

You will work through three optimisation passes using ATP. Each pass uses a different recipe to diagnose a different bottleneck:

1. **Instruction Mix** reveals that the distance function processes vectors one number at a time instead of using NEON SIMD instructions.
2. **Topdown** reveals that the graph search is doing far more work than necessary because its parameters are over-provisioned.
3. **Memory Access** reveals poor cache locality because queries arrive in random order, thrashing the CPU cache.

By the end of this tutorial, you will know how to:

1. Use the **Instruction Mix** recipe to detect scalar floating-point code that should be vectorised.
2. Use the **Topdown** recipe to identify Backend Bound caused by over-provisioned algorithmic parameters.
3. Use the **Memory Access** recipe to diagnose cache-unfriendly access patterns in pointer-chasing graph traversals.
4. Apply the profile → diagnose → fix → re-profile loop to a multi-component real-world workload.

## Before you begin

- An AWS Graviton 2/3 instance
- C++ compiler (g++ 9+ or clang++ 14+)
- CMake 3.16+
- Python 3.8+ with pip
- ATP installed and configured

## Terms used in this tutorial

- **CLIP**: Contrastive Language-Image Pre-training — a neural network that maps images and text into a shared vector space. Similar concepts have nearby vectors regardless of whether they originated as an image or text.
- **Vector / Embedding**: A fixed-length list of numbers (512 floats in this tutorial) representing an image or text query. Similar items have nearby vectors.
- **Vector search**: Finding the closest vectors in a database to a given query vector. This is how the application finds images that match a text description.
- **HNSW**: Hierarchical Navigable Small World — a graph-based algorithm that makes vector search fast by connecting similar vectors as neighbors, so you can find matches by walking the graph instead of scanning every vector.
- **Recall@K**: The fraction of true top-K nearest neighbors that the approximate search actually returns. A recall of 0.95 means 95% of the true neighbors are found.
- **M**: The maximum number of connections (edges) each node has in the HNSW graph. Larger M means more neighbors to evaluate per hop — higher recall but more work.
- **ef_search**: The size of the dynamic candidate list during search. Larger ef_search explores more candidates — higher recall but slower queries.
- **L2 distance**: The squared Euclidean distance between two vectors: `sum((a[i] - b[i])^2)`. This is the distance metric used throughout this tutorial.

---

## Architecture

The application has two layers:

**Dashboard (Python + Gradio)** — the interactive frontend. It loads the CLIP text encoder to convert typed queries into 512-dimensional vectors, searches an in-memory HNSWlib index, and displays the matching images.

**Backend (C++)** — the performance-critical layer. It handles two expensive operations:
1. **Index building**: inserting 50,000 image embeddings into an HNSW graph
2. **Batch search**: processing thousands of queries against the index

In production, the backend would be called every time the image database changes (re-indexing) and whenever a batch of user queries arrives. The C++ programs in this tutorial simulate this workload: each one builds the full index from 50,000 embeddings and then searches 10,000 queries. This generates enough sustained work (30–90 seconds on Graviton) for ATP to collect meaningful profiling samples.

### Try the dashboard

First, set up the data and run the dashboard to see the application in action.

#### Install Python dependencies

```bash
cd tutorial_6
pip install torch open-clip-torch Pillow numpy gradio hnswlib
```

#### Generate CLIP embeddings

```bash
python scripts/setup_data.py
```

This downloads CIFAR-100 (~170 MB), loads the CLIP ViT-B-32 model, and extracts 512-dimensional embeddings for all 60,000 images. The script creates these files in `data/`:

- `embeddings.bin` — 50,000 × 512 float32 vectors (~98 MB)
- `queries.bin` — 10,000 × 512 float32 vectors (~20 MB)
- `groundtruth.bin` — 10,000 × 10 int32 labels (brute-force exact top-10)
- `data_config.h` — C++ header with `#define` constants
- `images.npy`, `query_images.npy` — raw images for the dashboard
- `labels.npy`, `query_labels.npy`, `class_names.txt` — class metadata

#### Launch the dashboard

```bash
python dashboard/app.py
```

Open the URL printed in the terminal. Type a description — "a red sports car", "sunset over the ocean", "a cute puppy" — and the dashboard returns the 10 most similar images from the database.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

The search feels instant. But behind that single query is an HNSW index built from 50,000 embeddings, and at scale the backend must handle thousands of queries efficiently. The rest of this tutorial focuses on profiling and optimising that backend.

---

## Build the Backend

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

This produces four executables in `build/`, each representing a stage of optimisation:

| Executable | Description |
|------------|-------------|
| `image_search_baseline` | Scalar distance, over-provisioned parameters, random query order |
| `image_search_neon` | NEON distance, same parameters, random query order |
| `image_search_tuned` | NEON distance, tuned parameters, random query order |
| `image_search_optimised` | NEON distance, tuned parameters, sorted query order |

---

## Run the Baseline

```bash
./build/image_search_baseline
```

Expected output:

```
=== Baseline CLIP Image Search ===

Loading 50000 CLIP embeddings (dim=512)...
[1/2] Building index  (M=48, ef_construction=200)
      50000/50000
      65.3 s

[2/2] Searching  (ef_search=200, k=10, 10000 queries)
      9.7 s  (1030.9 queries/sec)
      Recall@10: 0.982

Total: 75.0 s
```

The timings will vary by instance type. Note the per-phase timings — both build and search are slower than they need to be. The baseline is deliberately sub-optimal in three ways, each of which will be diagnosed by a different ATP recipe.

---

## Profile the Baseline with ATP

### Step 1: CPU Cycle Hotspots — find where time is spent

Open ATP and select **Recipes → CPU Cycle Hotspots**. Choose `./build/image_search_baseline` as the target command. When it finishes, open the results.

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

Open ATP and select **Recipes → Instruction Mix**. Choose `./build/image_search_baseline` as the target.

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

The critical signal is the combination of **high scalar FP percentage** and **near-zero NEON percentage**. The L2 distance function is processing 512-dimensional float32 vectors one element at a time. Each iteration performs a subtract and a fused multiply-add on a single float — pure scalar floating point. With NEON, the same work could process 4 floats per instruction (128-bit registers) or even 16 floats per iteration with loop unrolling.

**Complete diagnosis: High scalar FP % and near-zero NEON % → the L2 distance function is entirely scalar. With 512 dimensions and millions of distance evaluations, this is the dominant bottleneck. The fix is to replace the scalar loop with NEON intrinsics.**

> **What to look for:** Compare the scalar FP % and NEON % in the baseline against the same metrics after the NEON fix. A drop in scalar FP % and a rise in NEON % confirms that the distance function is now vectorised.

---

## Pass 1: NEON Distance Function (Instruction Mix)

### The problem

Open `src/image_search_baseline.cpp` and note the `#define NO_MANUAL_VECTORIZATION` at the top. This tells HNSWlib to skip its x86 SIMD paths (SSE/AVX). On AArch64, HNSWlib has no built-in NEON path, so it falls back to the scalar `L2Sqr` function in `hnswlib/space_l2.h`:

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

This loop processes one float per iteration. With `dim=512`, each call executes 512 scalar subtracts, 512 scalar multiplies, and 512 scalar adds. The baseline is compiled with `-fno-tree-vectorize` to prevent the compiler from auto-vectorising this loop, ensuring a clear contrast.

### The fix

In `src/image_search_neon.cpp`, we define a custom `L2SpaceNeon` class that provides a NEON-accelerated distance function using Arm NEON intrinsics:

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

The inner loop is unrolled 4× to process 16 floats per iteration. With `dim=512`, this means 32 iterations of the main loop — 32 iterations replacing 512 scalar iterations.

HNSWlib's `SpaceInterface` allows plugging in a custom distance function. The `L2SpaceNeon` class simply returns `L2SqrNeon` as its distance function, and HNSWlib uses it for all distance evaluations during build and search.

### Run the NEON version

```bash
./build/image_search_neon
```

Compare the per-phase times against the baseline. Both build and search should be faster because every distance evaluation is now vectorised.

### Re-profile: did it work?

Open ATP and select **Recipes → Instruction Mix** again, this time targeting `./build/image_search_neon`.

<p align="center">
<img src="assets/neon_instruction_mix.png" width="850" alt="Instruction Mix after NEON fix — scalar FP % down, NEON % up"/>
</p>

Compare the instruction breakdown against the baseline:

| Category  | Baseline   | NEON (expected) |
|-----------|------------|-----------------|
| Scalar FP | 15–30%     | 2–8%            |
| NEON      | 0–2%       | 20–40%          |

The scalar FP percentage should drop dramatically and the NEON percentage should rise. This confirms that the distance function is now using SIMD instructions. The distance computation bottleneck is resolved, but the backend is still slower than it needs to be.

---

## Pass 2: Tune HNSW Parameters (Topdown)

### Profile the NEON version with Topdown

Now that the distance computation is vectorised, profile the NEON backend with a different recipe. Open ATP and select **Recipes → Topdown**. Choose `./build/image_search_neon` as the target.

In the Topdown summary, look at the four buckets:

<p align="center">
<img src="assets/neon_topdown.png" width="850" alt="Topdown summary after NEON fix — Backend Bound elevated"/>
</p>

You should see **Backend Bound** accounting for a significant share of slots during the search phase. The reason is the over-provisioned graph parameters:

- **M=48** means each node in the HNSW graph has up to 48 connections. At each hop during search, the algorithm evaluates the distance to up to 48 neighbors — far more than necessary for high recall.
- **ef_search=200** means the search maintains a candidate list of up to 200 entries. Each candidate insertion and extraction involves heap operations (`push_heap`, `pop_heap`) that access memory-resident priority queue data.

The combination creates excessive work per search query: too many distance evaluations (even though each one is now fast with NEON), too many heap operations, and too much data movement through the cache hierarchy. This shows up as Backend Bound because the CPU spends cycles waiting for data rather than doing useful arithmetic.

**Complete diagnosis: Backend Bound is elevated → the search phase performs excessive work due to over-provisioned parameters (M=48 creates too many edges, ef_search=200 creates too large a candidate queue). The fix is to reduce M and ef_search to values that maintain acceptable recall with less work per query.**

### The fix

In `src/image_search_tuned.cpp`, the HNSW parameters are tuned:

```
M:               48 → 16    (fewer edges per node)
ef_construction: 200 → 100  (build-time candidate list)
ef_search:       200 → 64   (search-time candidate list)
```

These values are typical for production vector search deployments. `M=16` provides enough connectivity for good recall on 512-dimensional data, and `ef_search=64` explores enough candidates to find the true top-10 with high probability.

> **Trade-off:** Reducing parameters lowers recall slightly. The key insight from Topdown is that the *hardware-level* cost of the over-provisioned parameters (Backend Bound) far exceeds the *algorithmic* benefit (marginal recall improvement). ATP gives you the evidence to make this trade-off quantitatively.

### Run the tuned version

```bash
./build/image_search_tuned
```

Both build and search should be significantly faster. The build is faster because `M=16` means fewer edges to maintain during insertion. The search is faster because each query evaluates fewer candidates.

### Re-profile: did it work?

Open ATP and select **Recipes → Topdown** again, this time targeting `./build/image_search_tuned`.

<p align="center">
<img src="assets/tuned_topdown.png" width="850" alt="Topdown summary after tuning — Backend Bound reduced, Retiring increased"/>
</p>

Compare the Topdown buckets against the NEON version:

| Bucket        | NEON   | Tuned (expected) |
|---------------|--------|--------------------|
| Backend Bound | High   | Lower              |
| Retiring      | Low    | Higher             |

The shift from Backend Bound to Retiring indicates that the CPU is spending more of its time doing useful work (distance computation, result extraction) and less time waiting for memory from oversized data structures. The search throughput (queries/sec) should increase substantially while recall remains above 0.95.

Two bottlenecks down. The vectorisation and parameter overheads are fixed. The next step is to look at the remaining memory access patterns with a recipe that examines cache behaviour directly.

---

## Pass 3: Sort Queries for Cache Locality (Memory Access)

### Profile the tuned version with Memory Access

With vectorisation and parameter tuning in place, the remaining bottleneck is in how the HNSW graph is traversed. Open ATP and select **Recipes → Memory Access**. Choose `./build/image_search_tuned` as the target.

In the Memory Access summary, look at:
- **L1D cache hit rate** — the fraction of data loads satisfied by the L1 data cache
- **Average load latency** — the mean number of cycles each load instruction takes to complete
- **SPE sample locations** — which functions have the most cache-miss samples

<p align="center">
<img src="assets/tuned_memory_access.png" width="850" alt="Memory Access metrics after tuning — L1D hit rate and load latency for graph traversal"/>
</p>

HNSW search is fundamentally a graph walk: starting from an entry point, the algorithm follows edges to neighboring nodes, evaluating distances at each hop and maintaining a priority queue of candidates. Each node in the graph stores:
- A list of neighbor IDs (pointers to other nodes)
- The node's embedding vector (512 × 4 = 2,048 bytes)

With 512-dimensional vectors, each node occupies about 2 KB. The L1 data cache on Graviton (64 KB) can hold roughly 32 vectors. Nodes are allocated during index construction in insertion order. When search queries arrive in random order, each query enters the graph at an unrelated point in embedding space. Consecutive queries visit completely different subsets of the graph, so the nodes accessed by query N have no overlap with the nodes accessed by query N+1. This means every new query brings a fresh set of cache misses — the graph nodes from the previous query have been evicted.

The Memory Access recipe reveals this as:
- Lower-than-expected L1D hit rate
- Elevated average load latency
- SPE samples concentrated in the graph traversal and distance evaluation functions

**Complete diagnosis: L1D hit rate is lower than expected and load latency is elevated → consecutive queries traverse unrelated graph regions because they arrive in random order. The fix is to sort queries by similarity so consecutive queries traverse overlapping graph paths, keeping visited nodes warm in cache.**

### The fix

In `src/image_search_optimised.cpp`, the queries are sorted by their first dimension value before searching:

```cpp
std::vector<size_t> order(n_query);
std::iota(order.begin(), order.end(), 0);
std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return queries[a * dim] < queries[b * dim];
});
```

After sorting, consecutive queries have similar first-coordinate values, which is a rough proxy for spatial proximity in embedding space. Similar queries enter the HNSW graph at nearby entry points and traverse overlapping subsets of the graph. The nodes visited by query N are still in L1/L2 cache when query N+1 visits many of the same nodes.

> **Trade-off:** Sorting adds a small cost (sorting 10,000 indices is negligible), and results must be mapped back to original query indices for recall measurement. The cache locality improvement during the search phase more than compensates. In production, more sophisticated approaches exist — clustering queries into batches, or partitioning the index — but the simple sort demonstrates the principle.

### Run the optimised backend

```bash
./build/image_search_optimised
```

### Re-profile: did it work?

Open ATP and select **Recipes → Memory Access** again, this time targeting `./build/image_search_optimised`.

<p align="center">
<img src="assets/optimised_memory_access.png" width="850" alt="Memory Access metrics for optimised backend — L1D hit rate improved, load latency reduced"/>
</p>

Compare cache metrics against the tuned version:

| Metric              | Tuned     | Optimised (expected) |
|---------------------|-----------|----------------------|
| L1D hit rate        | ~85–90%   | ~92–97%              |
| Avg load latency    | Higher    | Lower                |

The rise in L1D hit rate and drop in average load latency confirm that the sorted query order improved cache locality during graph traversal. The three bottlenecks have been systematically identified and resolved.

---

## Summary

You started with a real text-to-image search application — 50,000 CIFAR-100 images indexed by 512-dimensional CLIP embeddings in an HNSWlib graph — and used three different ATP recipes to diagnose and fix three distinct performance bottlenecks in the backend:

| Pass | ATP Recipe       | Problem                                    | Fix                                    |
|------|------------------|--------------------------------------------|----------------------------------------|
| 1    | Instruction Mix  | Scalar L2 distance (no SIMD) over 512 dims | NEON intrinsics (16 floats per iteration) |
| 2    | Topdown          | Over-provisioned M and ef_search → Backend Bound | Tuned parameters (M=16, ef_search=64) |
| 3    | Memory Access    | Random query order → cache-cold graph traversal | Sort queries by similarity             |

Each pass followed the same loop: profile to find the dominant bottleneck, apply a targeted fix, then re-profile to confirm the metric shifted as expected. The final optimised backend produces the same search results as the baseline (within recall tolerance) but at significantly higher throughput.

### File reference

| File | Description |
|------|-------------|
| `scripts/setup_data.py` | Downloads CIFAR-100 and generates CLIP embeddings |
| `scripts/generate_embeddings.py` | Alternative: generates synthetic embeddings (no CLIP needed) |
| `dashboard/app.py` | Gradio dashboard for interactive text-to-image search |
| `src/image_search_baseline.cpp` | Baseline backend — scalar distance, large params, random queries |
| `src/image_search_neon.cpp` | After Pass 1 — NEON distance, other problems remain |
| `src/image_search_tuned.cpp` | After Pass 2 — NEON + tuned params, random queries remain |
| `src/image_search_optimised.cpp` | Fully optimised — all three fixes applied |
| `src/hnswlib/` | HNSWlib v0.8.0 header-only library |
| `requirements.txt` | Python dependencies |

### Key takeaway

The diagnostic method matters more than the specific fixes. Scalar distance computation, over-provisioned algorithmic parameters, and cache-unfriendly traversal patterns are common in graph-based search libraries, but they rarely announce themselves in source code. ATP's hardware-level recipes give you the evidence to identify each one and verify that your fix actually addressed the root cause — not just the symptom.
