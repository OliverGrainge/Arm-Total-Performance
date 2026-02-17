#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// Reordered ikj loop — the inner loop now iterates over j,
// accessing both B and C with stride-1 (row-major) patterns.
// This dramatically improves spatial locality and cache hit rates.

void matmul_ikj(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j]; // B and C both stride-1
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
        matmul_ikj(A.data(), B.data(), C.data(), N);
        ++reps;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(end - start).count() < 5.0);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N * reps) / (elapsed_ms * 1e6);

    std::cout << "Reordered (ikj) matmul (" << N << "x" << N << ", " << reps << " reps)\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";

    return 0;
}
