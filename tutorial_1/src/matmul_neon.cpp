#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Dense matrix multiplication: C = A * B
// Register-blocked version with NEON intrinsics and B-tile packing.
//
// Two levels of tiling:
//   Outer tile: TILE = 64 → 64×64×4 = 16 KB per tile.
//     Three tiles (A, B, C) = 48 KB — fits in Graviton3 L1d (64 KB).
//   Inner micro-kernel: 4 rows × 4 columns (4×4 register block).
//
// B-tile packing:
//   Before the micro-kernel runs, the B tile is copied into a contiguous
//   micro-panel layout.  In the original code the innermost k-loop
//   accessed B[k*N+j] with stride N (one cache-line miss every few
//   iterations).  After packing, the k-loop reads B sequentially,
//   turning almost every access into an L1d hit.
//
// Each vfmaq_n_f32 performs 4 multiply-adds in a single instruction,
// giving 4× the work-per-instruction of the scalar tiled version.
// The four independent C accumulators also expose instruction-level
// parallelism, letting the out-of-order core overlap FMA latencies.
//
// Expected ATP profile: high Retiring %, low Backend Bound.

constexpr int TILE = 64;

// Pack B[k0:k_end][j0:j_end] into micro-panel format.
// Layout: for each 4-column micro-panel, all k rows are stored
// contiguously so the micro-kernel streams through them linearly.
static void pack_B_tile(const float* B, float* packed,
                        int k0, int k_end, int j0, int j_end, int N) {
    float* dst = packed;
    for (int j = j0; j < j_end; j += 4) {
        for (int k = k0; k < k_end; ++k) {
            vst1q_f32(dst, vld1q_f32(&B[k * N + j]));
            dst += 4;
        }
    }
}

void matmul_neon(const float* A, const float* B, float* C, int M, int K, int N) {
    std::memset(C, 0, M * N * sizeof(float));

    // Scratch buffer for one packed B tile (at most TILE × TILE floats)
    std::vector<float> packed_B(TILE * TILE);

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int i_end = std::min(i0 + TILE, M);
                int j_end = std::min(j0 + TILE, N);
                int k_end = std::min(k0 + TILE, K);
                int k_len = k_end - k0;

                // Pack B tile so micro-kernel reads are sequential
                pack_B_tile(B, packed_B.data(), k0, k_end, j0, j_end, N);

                // Process the tile in 4×4 micro-blocks
                for (int i = i0; i < i_end; i += 4) {
                    const float* bp = packed_B.data();
                    for (int j = j0; j < j_end; j += 4) {
                        // Load 4×4 block of C into NEON registers
                        float32x4_t c0 = vld1q_f32(&C[(i + 0) * N + j]);
                        float32x4_t c1 = vld1q_f32(&C[(i + 1) * N + j]);
                        float32x4_t c2 = vld1q_f32(&C[(i + 2) * N + j]);
                        float32x4_t c3 = vld1q_f32(&C[(i + 3) * N + j]);

                        const float* bp_k = bp;
                        for (int k = k0; k < k_end; ++k) {
                            // Packed B: sequential read of B[k][j:j+4]
                            float32x4_t b = vld1q_f32(bp_k);
                            bp_k += 4;
                            // Each vfmaq_n_f32: C_row += A[row][k] * B[k][j:j+4]
                            c0 = vfmaq_n_f32(c0, b, A[(i + 0) * K + k]);
                            c1 = vfmaq_n_f32(c1, b, A[(i + 1) * K + k]);
                            c2 = vfmaq_n_f32(c2, b, A[(i + 2) * K + k]);
                            c3 = vfmaq_n_f32(c3, b, A[(i + 3) * K + k]);
                        }

                        // Store the 4×4 result back
                        vst1q_f32(&C[(i + 0) * N + j], c0);
                        vst1q_f32(&C[(i + 1) * N + j], c1);
                        vst1q_f32(&C[(i + 2) * N + j], c2);
                        vst1q_f32(&C[(i + 3) * N + j], c3);
                        bp += k_len * 4;  // advance to next micro-panel
                    }
                }
            }
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

    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(i % 97) * 0.01f;
    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>(i % 89) * 0.01f;

    auto start = std::chrono::high_resolution_clock::now();
    matmul_neon(A.data(), B.data(), C.data(), M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * K * N) / (elapsed_ms * 1e6);

    std::cout << "NEON matmul (" << M << "x" << K << " * " << K << "x" << N << ", tile=" << TILE << ")\n";
    std::cout << "  Time:  " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";
    std::cout << "  Check:  C[0]=" << C[0] << " C[M*N-1]=" << C[M * N - 1] << "\n";

    return 0;
}
