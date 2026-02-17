#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>

// Dense matrix multiplication: C = A * B
// Cache-tiled version — the matrix is divided into TILE x TILE blocks
// so that each block fits in L1/L2 cache. Within each tile the ikj
// ordering is used for stride-1 access.
//
// Graviton3 has 64 KB L1d and 1 MB L2 per core.
// A tile of 64x64 floats = 16 KB, so three tiles (A, B, C) = 48 KB
// which fits comfortably in L1d.

constexpr int TILE = 64;

void matmul_tiled(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, N * N * sizeof(float));

    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int i_end = std::min(i0 + TILE, N);
            int j_end = std::min(j0 + TILE, N);
            int width = j_end - j0;
            for (int k0 = 0; k0 < N; k0 += TILE) {
                int k_end = std::min(k0 + TILE, N);

                for (int i = i0; i < i_end; ++i) {
                    float* c_row = &C[i * N + j0];
                    for (int k = k0; k < k_end; ++k) {
                        float a_ik = A[i * N + k];
                        const float* b_row = &B[k * N + j0];
                        int jj = 0;
                        for (; jj + 8 <= width; jj += 8) {
                            c_row[jj + 0] += a_ik * b_row[jj + 0];
                            c_row[jj + 1] += a_ik * b_row[jj + 1];
                            c_row[jj + 2] += a_ik * b_row[jj + 2];
                            c_row[jj + 3] += a_ik * b_row[jj + 3];
                            c_row[jj + 4] += a_ik * b_row[jj + 4];
                            c_row[jj + 5] += a_ik * b_row[jj + 5];
                            c_row[jj + 6] += a_ik * b_row[jj + 6];
                            c_row[jj + 7] += a_ik * b_row[jj + 7];
                        }
                        for (; jj < width; ++jj) {
                            c_row[jj] += a_ik * b_row[jj];
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
        matmul_tiled(A.data(), B.data(), C.data(), N);
        ++reps;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(end - start).count() < 5.0);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N * reps) / (elapsed_ms * 1e6);

    std::cout << "Tiled matmul (" << N << "x" << N << ", tile=" << TILE << ", " << reps << " reps)\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";

    return 0;
}
