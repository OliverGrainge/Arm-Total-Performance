#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(__has_include)
#if __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define ATP_HAS_NEON_HEADER 1
#endif
#endif

// Dense matrix multiplication: C = A * B
// Tiled + explicit Arm NEON intrinsics in the inner loop.
// Falls back to scalar code when NEON is unavailable at compile time.

constexpr int TILE = 64;

void matmul_tiled_neon(const float* A, const float* B, float* C, int N) {
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
                        const float a_ik = A[i * N + k];
                        const float* b_row = &B[k * N + j0];

#if defined(ATP_HAS_NEON_HEADER) && defined(__ARM_NEON)
                        int jj = 0;
                        for (; jj + 4 <= width; jj += 4) {
                            float32x4_t c_vec = vld1q_f32(c_row + jj);
                            float32x4_t b_vec = vld1q_f32(b_row + jj);
#if defined(__aarch64__)
                            c_vec = vfmaq_n_f32(c_vec, b_vec, a_ik);
#else
                            c_vec = vmlaq_n_f32(c_vec, b_vec, a_ik);
#endif
                            vst1q_f32(c_row + jj, c_vec);
                        }
                        for (; jj < width; ++jj) {
                            c_row[jj] += a_ik * b_row[jj];
                        }
#else
                        for (int jj = 0; jj < width; ++jj) {
                            c_row[jj] += a_ik * b_row[jj];
                        }
#endif
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 1024;
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
        matmul_tiled_neon(A.data(), B.data(), C.data(), N);
        ++reps;
        end = std::chrono::high_resolution_clock::now();
    } while (std::chrono::duration<double>(end - start).count() < 5.0);

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * N * N * N * reps) / (elapsed_ms * 1e6);

    std::cout << "Tiled + NEON matmul (" << N << "x" << N << ", tile=" << TILE << ", " << reps << " reps)\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[N*N-1]=" << C[N * N - 1] << "\n";
#if !(defined(ATP_HAS_NEON_HEADER) && defined(__ARM_NEON))
    std::cout << "  Note: built without NEON support; scalar fallback path used.\n";
#endif

    return 0;
}
