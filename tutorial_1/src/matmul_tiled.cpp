#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// 2D tiled version — all three loop dimensions (i, j, k) are blocked so
// that the working set fits in L2 cache.
//
// Graviton3: 64 KB L1d, 1 MB L2, ~32 MB LLC per core.
// TILE = 128 → each tile is 128×128×4 = 64 KB.
// Three tiles (A, B, C sub-blocks) = 192 KB — fits comfortably in L2
// but does NOT fit in L1d (64 KB).
//
// Compared to the naive version, LLC misses are largely eliminated
// because the tiles are re-used while resident in L2.  However L1d
// misses remain elevated because each tile exceeds the L1d capacity.
// The workload shifts from LLC-miss-dominated to L1-miss-dominated,
// which ATP will show as a reduction in Backend Memory Bound stalls.

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

int main(int argc, char* argv[]) {
    int M = 256;   // rows of A and C (reduced to limit runtime)
    int K = 8192;  // cols of A / rows of B
    int N = 2048;  // cols of B and C

    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) K = std::atoi(argv[2]);
    if (argc > 3) N = std::atoi(argv[3]);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(i % 97) * 0.01f;
    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>(i % 89) * 0.01f;

    auto start = std::chrono::high_resolution_clock::now();
    matmul_tiled(A.data(), B.data(), C.data(), M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * K * N) / (elapsed_ms * 1e6);

    std::cout << "2D-tiled matmul (" << M << "x" << K << " * " << K << "x" << N << ", tile=" << TILE << ")\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[M*N-1]=" << C[M * N - 1] << "\n";

    return 0;
}
