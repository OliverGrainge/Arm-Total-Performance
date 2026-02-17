#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// 2D tiled version — all three dimensions (i, j, k) are tiled so that
// the A, B, and C sub-blocks fit entirely in L1d cache.
//
// Graviton3 has 64 KB L1d and 1 MB L2 per core.
// A tile of 64×64 floats = 16 KB, so three tiles (A, B, C) = 48 KB
// which fits comfortably in L1d. This minimises both L1 and L2 misses,
// moving the workload from Memory Bound toward Retiring.

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

int main(int argc, char* argv[]) {
    int N = 4096;
    if (argc > 1) N = std::atoi(argv[1]);

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i % 97) * 0.01f;
        B[i] = static_cast<float>(i % 89) * 0.01f;
    }

    int reps = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end;
    do {
        matmul_tiled_2d(A.data(), B.data(), C.data(), N);
        ++reps;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(end - start).count() < 5.0);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N * reps) / (elapsed_ms * 1e6);

    std::cout << "2D-tiled matmul (" << N << "x" << N << ", tile=" << TILE << ", " << reps << " reps)\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";

    return 0;
}
