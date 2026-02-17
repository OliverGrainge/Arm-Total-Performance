#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// Naive ijk ordering — the inner loop accesses B[k*N+j] with stride N,
// jumping across rows on every iteration. For N=4096 each stride is 16 KB,
// far exceeding a cache line.  The full B matrix (64 MB) does not fit in
// the last-level cache (32 MB on Graviton3), so almost every B access
// results in an LLC miss and a trip to DRAM.  This makes the workload
// heavily Backend Bound → Memory Bound in the Top-Down model.

void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j]; // B access strides by N
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 4096;
    if (argc > 1) N = std::atoi(argv[1]);

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    // Initialise with deterministic values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i % 97) * 0.01f;
        B[i] = static_cast<float>(i % 89) * 0.01f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N) / (elapsed_ms * 1e6);

    std::cout << "Naive matmul (" << N << "x" << N << ")\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";

    return 0;
}
