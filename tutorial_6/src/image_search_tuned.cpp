// Pass 2 — Tuned HNSW parameters.
//
// Fix: Reduce M, ef_construction, and ef_search to production values.
// The over-provisioned baseline parameters (M=48, ef_search=200) created a
// dense graph with large candidate queues, causing Backend Bound pressure
// from heap operations and excessive 768-dimensional distance evaluations.
//
// What changed vs Pass 1:
//   - M: 48 -> 16    (fewer edges per node — smaller graph, less work per hop)
//   - ef_construction: 200 -> 100
//   - ef_search: 200 -> 64  (smaller candidate queue — fewer distance evals)
// What is still sub-optimal:
//   - Random query order (no cache locality during graph traversal)

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
#include <vector>

// ---------------------------------------------------------------------------
// NEON-accelerated L2 squared distance (same as Pass 1)
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
                             size_t k) {
    size_t n_queries = results.size();
    size_t hits = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        for (size_t i = 0; i < results[q].size() && i < k; ++i) {
            int id = static_cast<int>(results[q][i].second);
            for (size_t j = 0; j < k; ++j) {
                if (gt[q * k + j] == id) { ++hits; break; }
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

    std::cout << "=== Pass 2: Tuned Parameters ===\n\n";
    std::cout << "Loading " << n_base << " CLIP embeddings (dim=" << dim << ")...\n";
    auto base    = load_fvecs("data/embeddings.bin", n_base, dim);
    auto queries = load_fvecs("data/queries.bin",    n_query, dim);
    auto gt      = load_ivecs("data/groundtruth.bin", n_query, k);

    // Tuned parameters — smaller graph, smaller candidate queue
    const int M = 16;
    const int ef_construction = 100;

    L2SpaceNeon space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n_base, M, ef_construction);

    std::cout << "[1/2] Building index  (M=" << M
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

    // Reduced ef_search
    const int ef_search = 64;
    index.setEf(ef_search);

    std::cout << "[2/2] Searching  (ef_search=" << ef_search
              << ", k=" << k << ", " << n_query << " queries)\n";

    std::vector<std::vector<std::pair<float, size_t>>> results(n_query);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < n_query; ++q) {
        auto pq = index.searchKnn(queries.data() + q * dim, k);
        std::vector<std::pair<float, size_t>> res;
        while (!pq.empty()) {
            res.push_back({pq.top().first, pq.top().second});
            pq.pop();
        }
        results[q] = res;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_s = std::chrono::duration<double>(t3 - t2).count();
    double qps = n_query / search_s;

    double recall = compute_recall(gt, results, k);

    std::cout << "      " << search_s << " s  (" << qps << " queries/sec)\n";
    std::cout << "      Recall@" << k << ": " << recall << "\n\n";
    std::cout << "Total: " << (build_s + search_s) << " s\n";

    return 0;
}
