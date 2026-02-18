#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// Softmax + element-wise scaling with __restrict__ on all pointer parameters.
//
// __restrict__ is a contract between the programmer and the compiler:
// "I guarantee these pointers do not alias for the duration of this call."
// The compiler no longer needs to insert alias-check guards or serialise
// loads and stores around every write.
//
// The normalise+scale pass:
//
//     out[i] *= inv_sum * scale[i];   // ATP hot line
//
// now vectorises cleanly.  ATP's Source Code Inspector will show 128-bit
// NEON LDR/STR instructions (LD1 { v0.4S }) on that line instead of the
// scalar 32-bit loads in the baseline.
//
// The expf pass remains scalar in all variants because expf() is a library
// function that does not auto-vectorise at -O2 without -ffast-math.

static float find_max(const float* __restrict__ data, int N) {
    float m = data[0];
    for (int i = 1; i < N; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

void softmax_scale(float* __restrict__ output,
                   const float* __restrict__ input,
                   const float* __restrict__ scale, int N) {
    float max_val = find_max(input, N);

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        output[i] = std::expf(input[i] - max_val);
        sum += output[i];
    }

    // With __restrict__ the compiler knows output[] and scale[] are disjoint.
    // It widens this loop to process four floats per instruction.
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; ++i)
        output[i] *= inv_sum * scale[i];
}

int main(int argc, char* argv[]) {
    int N     = 1 << 22;
    int iters = 100;

    if (argc > 1) N     = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);

    std::vector<float> input(N), scale(N), output(N);

    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(i % 1009) * 0.001f - 0.5f;
        scale[i] = 1.0f + static_cast<float>(i % 101) * 0.01f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter)
        softmax_scale(output.data(), input.data(), scale.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbps = static_cast<double>(N) * sizeof(float) * 3 * iters / (ms * 1e6);

    std::cout << "Restrict softmax+scale  N=" << N << " iters=" << iters << "\n";
    std::cout << "  Time:       " << ms   << " ms\n";
    std::cout << "  Bandwidth:  " << gbps << " GB/s\n";
    std::cout << "  Check: out[0]=" << output[0]
              << "  out[N-1]=" << output[N - 1] << "\n";
    return 0;
}
