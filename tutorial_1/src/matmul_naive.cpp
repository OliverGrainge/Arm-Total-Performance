#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B   (A is MxK, B is KxN, C is MxN)
// Naive ijk ordering — the inner loop accesses B[k*N+j] with stride N,
// jumping across rows on every iteration. For N=8192 each stride is 32 KB,
// far exceeding a cache line.  The full B matrix (256 MB) does not fit in
// the last-level cache (32 MB on Graviton3), so almost every B access
// results in an LLC miss and a trip to DRAM.  This makes the workload
// heavily Backend Bound → Memory Bound in the Top-Down model.
//
// M is kept small (512) to limit runtime while preserving the memory
// access profile on B — every row of A still sweeps the entire B matrix.

void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j]; // B access strides by N
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int M = 512;   // rows of A and C (reduced to limit runtime)
    int K = 8192;  // cols of A / rows of B
    int N = 8192;  // cols of B and C

    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) K = std::atoi(argv[2]);
    if (argc > 3) N = std::atoi(argv[3]);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    // Initialise with deterministic values
    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(i % 97) * 0.01f;
    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>(i % 89) * 0.01f;

    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(A.data(), B.data(), C.data(), M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * K * N) / (elapsed_ms * 1e6);

    std::cout << "Naive matmul (" << M << "x" << K << " * " << K << "x" << N << ")\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[M*N-1]=" << C[M * N - 1] << "\n";

    return 0;
}
