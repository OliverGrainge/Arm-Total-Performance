// Baseline CLIP image search — scalar L2 distance, over-provisioned graph parameters,
// random query order.
//
// This is a similarity search over 50,000 CIFAR-100 images represented as
// 768-dimensional CLIP ViT-L/14 embeddings.  It is deliberately sub-optimal
// in three ways:
//
// 1. Uses HNSWlib's default scalar L2Sqr (no SIMD) — one float at a time.
//    Shows up as high scalar FP % and zero NEON in Instruction Mix.
// 2. Over-provisioned graph parameters (M=48, ef_construction=200, ef_search=200)
//    produce a dense graph with excessive neighbor evaluations per hop.
// 3. Queries arrive in random (file) order, so consecutive searches enter the
//    graph at unrelated points — poor temporal locality in graph node accesses.

#define NO_MANUAL_VECTORIZATION   // Force the scalar L2Sqr path in HNSWlib
#include "hnswlib.h"
#include "data_config.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

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

    // --- Load data ---
    std::cout << "=== Baseline CLIP Image Search ===\n\n";
    std::cout << "Loading " << n_base << " CLIP embeddings (dim=" << dim << ")...\n";
    auto base    = load_fvecs("data/embeddings.bin", n_base, dim);
    auto queries = load_fvecs("data/queries.bin",    n_query, dim);
    auto gt      = load_ivecs("data/groundtruth.bin", n_query, k);

    // --- Build index (over-provisioned parameters) ---
    const int M = 48;
    const int ef_construction = 200;

    hnswlib::L2Space space(dim);
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

    // --- Search (large ef_search, random query order) ---
    const int ef_search = 200;
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
