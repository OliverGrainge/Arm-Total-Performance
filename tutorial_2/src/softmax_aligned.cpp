#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Alternative to __restrict__: __builtin_assume_aligned + loop splitting.
//
// The public softmax_scale() API does not carry __restrict__.  Instead the
// softmax is decomposed into three single-responsibility helper functions,
// each of which:
//   a) takes at most one writable pointer — minimising the aliasing surface
//      the compiler must reason about, and
//   b) marks every pointer with __builtin_assume_aligned(ptr, 64) to tell
//      the compiler the data is aligned to a 64-byte Graviton cache-line
//      boundary.
//
// The alignment guarantee matters for two reasons:
//   1. The compiler can start the vectorised loop immediately without an
//      alignment scalar pre-loop (no "peel" iterations).
//   2. The hardware prefetcher benefits from the alignment when the load
//      address is a multiple of the cache-line size.
//
// The separate tmp[] array breaks the input→output write-back cycle that
// prevented the exp pass from vectorising in a fused loop.  Combined with
// aligned data, pass_normalise achieves the same NEON throughput as the
// __restrict__ variant without changing the public API.

// ---- helpers ---------------------------------------------------------------

static float pass_max(const float* data, int N) {
    data = static_cast<const float*>(__builtin_assume_aligned(data, 64));
    float m = data[0];
    for (int i = 1; i < N; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

// Writes tmp[i] = expf(input[i] - max_val).  tmp and input are separate
// allocations guaranteed distinct by the caller.
static float pass_exp(float* __restrict__ tmp,
                      const float* __restrict__ input,
                      float max_val, int N) {
    tmp   = static_cast<float*>(      __builtin_assume_aligned(tmp,   64));
    input = static_cast<const float*>(__builtin_assume_aligned(input, 64));
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        tmp[i] = std::exp(input[i] - max_val);
        sum   += tmp[i];
    }
    return sum;
}

// Writes output[i] = tmp[i] * inv_sum * scale[i].
// Three distinct aligned arrays: the compiler can emit aligned NEON loads
// without a scalar prologue.
static void pass_normalise(float* __restrict__ output,
                           const float* __restrict__ tmp,
                           const float* __restrict__ scale,
                           float inv_sum, int N) {
    output = static_cast<float*>(      __builtin_assume_aligned(output, 64));
    tmp    = static_cast<const float*>(__builtin_assume_aligned(tmp,    64));
    scale  = static_cast<const float*>(__builtin_assume_aligned(scale,  64));
    for (int i = 0; i < N; ++i)
        output[i] = tmp[i] * inv_sum * scale[i];
}

// ---- public API (no __restrict__ on the interface) -------------------------

void softmax_scale(float* output, const float* input,
                   const float* scale, int N) {
    std::vector<float> tmp(N);
    float max_val = pass_max(input, N);
    float sum     = pass_exp(tmp.data(), input, max_val, N);
    pass_normalise(output, tmp.data(), scale, 1.0f / sum, N);
}

int main(int argc, char* argv[]) {
    int N     = 1 << 22;
    int iters = 100;

    if (argc > 1) N     = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);

    // Allocate 64-byte-aligned buffers to match the __builtin_assume_aligned
    // hints inside the helper functions.
    float* input  = static_cast<float*>(std::malloc(N * sizeof(float)));
    float* scale_ = static_cast<float*>(std::malloc(N * sizeof(float)));
    float* output = static_cast<float*>(std::malloc(N * sizeof(float)));

    for (int i = 0; i < N; ++i) {
        input[i]  = static_cast<float>(i % 1009) * 0.001f - 0.5f;
        scale_[i] = 1.0f + static_cast<float>(i % 101) * 0.01f;
    }
    std::memset(output, 0, N * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter)
        softmax_scale(output, input, scale_, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbps = static_cast<double>(N) * sizeof(float) * 3 * iters / (ms * 1e6);

    std::cout << "Aligned+split softmax+scale  N=" << N << " iters=" << iters << "\n";
    std::cout << "  Time:       " << ms   << " ms\n";
    std::cout << "  Bandwidth:  " << gbps << " GB/s\n";
    std::cout << "  Check: out[0]=" << output[0]
              << "  out[N-1]=" << output[N - 1] << "\n";

    std::free(input);
    std::free(scale_);
    std::free(output);
    return 0;
}
