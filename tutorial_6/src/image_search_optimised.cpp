// Fully optimised CLIP image search — all three fixes applied.
//
// 1. NEON distance function (Instruction Mix fix)
// 2. Tuned HNSW parameters  (Topdown fix)
// 3. Queries sorted by first dimension for cache locality (Memory Access fix)
//
// With 768-dimensional embeddings each vector occupies 3,072 bytes, so the
// L1 data cache (typically 64 KB on Graviton) holds roughly 20 vectors.
// Sorting queries by their first coordinate groups similar queries together.
// Consecutive similar queries traverse overlapping graph paths, keeping the
// visited graph nodes warm in L1/L2 cache.

#define NO_MANUAL_VECTORIZATION
#include "hnswlib.h"
#include "data_config.h"

#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// NEON-accelerated L2 squared distance (same as Pass 1 and Pass 2)
// ---------------------------------------------------------------------------
static float L2SqrNeon(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* a = static_cast<const float*>(pVect1v);
    const float* b = static_cast<const float*>(pVect2v);
    size_t dim = *static_cast<const size_t*>(qty_ptr);

    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;

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
    for (; i + 3 < dim; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        sum = vfmaq_f32(sum, d, d);
    }
    float result = vaddvq_f32(sum);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

class L2SpaceNeon : public hnswlib::SpaceInterface<float> {
    size_t dim_;
    size_t data_size_;
 public:
    L2SpaceNeon(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {}
    size_t get_data_size() { return data_size_; }
    hnswlib::DISTFUNC<float> get_dist_func() { return L2SqrNeon; }
    void* get_dist_func_param() { return &dim_; }
    ~L2SpaceNeon() {}
};

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------
static std::vector<float> load_fvecs(const char* path, size_t n, size_t dim) {
    std::vector<float> v(n * dim);
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    in.read(reinterpret_cast<char*>(v.data()), n * dim * sizeof(float));
    return v;
}

static std::vector<int> load_ivecs(const char* path, size_t n, size_t k) {
    std::vector<int> v(n * k);
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    in.read(reinterpret_cast<char*>(v.data()), n * k * sizeof(int));
    return v;
}

// ---------------------------------------------------------------------------
// Recall calculation
// ---------------------------------------------------------------------------
static double compute_recall(const std::vector<int>& gt,
                             const std::vector<std::vector<std::pair<float, size_t>>>& results,
                             const std::vector<size_t>& order,
                             size_t k) {
    size_t n_queries = results.size();
    size_t hits = 0;
    for (size_t qi = 0; qi < n_queries; ++qi) {
        size_t orig = order[qi];  // map back to original query index
        for (size_t i = 0; i < results[qi].size() && i < k; ++i) {
            int id = static_cast<int>(results[qi][i].second);
            for (size_t j = 0; j < k; ++j) {
                if (gt[orig * k + j] == id) { ++hits; break; }
            }
        }
    }
    return static_cast<double>(hits) / (n_queries * k);
}

// ---------------------------------------------------------------------------
int main() {
    const size_t n_base  = DATA_N_BASE;
    const size_t n_query = DATA_N_QUERY;
    const size_t dim     = DATA_DIM;
    const size_t k       = DATA_K;

    std::cout << "=== Optimised CLIP Image Search ===\n\n";
    std::cout << "Loading " << n_base << " CLIP embeddings (dim=" << dim << ")...\n";
    auto base    = load_fvecs("data/embeddings.bin", n_base, dim);
    auto queries = load_fvecs("data/queries.bin",    n_query, dim);
    auto gt      = load_ivecs("data/groundtruth.bin", n_query, k);

    // --- Tuned parameters ---
    const int M = 16;
    const int ef_construction = 100;

    L2SpaceNeon space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n_base, M, ef_construction);

    std::cout << "[1/3] Building index  (M=" << M
              << ", ef_construction=" << ef_construction << ")\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_base; ++i) {
        index.addPoint(base.data() + i * dim, i);
        if ((i + 1) % 5000 == 0 || i + 1 == n_base)
            std::cout << "\r      " << (i + 1) << "/" << n_base << std::flush;
    }
    std::cout << "\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "      " << build_s << " s\n\n";

    // --- Sort queries by first dimension for cache locality ---
    std::cout << "[2/3] Sorting queries by first dimension\n";

    std::vector<size_t> order(n_query);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return queries[a * dim] < queries[b * dim];
    });

    auto t_sort_end = std::chrono::high_resolution_clock::now();
    double sort_s = std::chrono::duration<double>(t_sort_end - t1).count();
    std::cout << "      " << sort_s << " s\n\n";

    // --- Search in sorted order ---
    const int ef_search = 64;
    index.setEf(ef_search);

    std::cout << "[3/3] Searching  (ef_search=" << ef_search
              << ", k=" << k << ", " << n_query << " queries)\n";

    std::vector<std::vector<std::pair<float, size_t>>> results(n_query);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t qi = 0; qi < n_query; ++qi) {
        size_t orig = order[qi];
        auto pq = index.searchKnn(queries.data() + orig * dim, k);
        std::vector<std::pair<float, size_t>> res;
        while (!pq.empty()) {
            res.push_back({pq.top().first, pq.top().second});
            pq.pop();
        }
        results[qi] = res;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_s = std::chrono::duration<double>(t3 - t2).count();
    double qps = n_query / search_s;

    double recall = compute_recall(gt, results, order, k);

    std::cout << "      " << search_s << " s  (" << qps << " queries/sec)\n";
    std::cout << "      Recall@" << k << ": " << recall << "\n\n";
    std::cout << "Total: " << (build_s + sort_s + search_s) << " s\n";

    return 0;
}
