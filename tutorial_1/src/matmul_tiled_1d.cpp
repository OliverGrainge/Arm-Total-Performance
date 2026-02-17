#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// 1D tiled (k-strip) version — only the k dimension is tiled.
// Within each k-block the ikj loop order gives stride-1 access on B and C.
//
// Working set per k-block: the B strip is TILE rows × N columns.
// For TILE=64, N=4096: 64 × 4096 × 4 = 1 MB — fits in L2 (1 MB on
// Graviton3) but NOT in L1d (64 KB). This reduces L2 misses compared
// to the naive version, but L1 misses remain elevated.

constexpr int TILE = 64;

void matmul_tiled_1d(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int k0 = 0; k0 < N; k0 += TILE) {
        int k_end = std::min(k0 + TILE, N);
        for (int i = 0; i < N; ++i) {
            for (int k = k0; k < k_end; ++k) {
                float a_ik = A[i * N + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_ik * B[k * N + j];
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
        matmul_tiled_1d(A.data(), B.data(), C.data(), N);
        ++reps;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(end - start).count() < 5.0);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N * reps) / (elapsed_ms * 1e6);

    std::cout << "1D-tiled matmul (" << N << "x" << N << ", tile=" << TILE << ", " << reps << " reps)\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";

    return 0;
}
