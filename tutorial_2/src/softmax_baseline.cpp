#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// Softmax followed by element-wise scaling: out[i] = softmax(in)[i] * s[i]
//
// No pointer annotations.  The compiler cannot prove that 'output' does
// not alias 'input' or 'scale' — they are all float*, so type-based alias
// analysis offers no help.  The compiler must therefore treat every write
// through 'output' as a potential modification of 'input' or 'scale'.
//
// The consequence shows up most clearly in the normalise+scale pass:
//
//     out[i] *= inv_sum * scale[i];   // ATP hot line
//
// A write to out[i] could, in principle, corrupt scale[i+1], so the
// compiler serialises loads and stores rather than widening to NEON.
// ATP's Source Code Inspector will show 32-bit scalar LDR/STR instructions
// on that line in the baseline.

static float find_max(const float* data, int N) {
    float m = data[0];
    for (int i = 1; i < N; ++i)
        if (data[i] > m) m = data[i];
    return m;
}

void softmax_scale(float* output, const float* input,
                   const float* scale, int N) {
    // Pass 1: max (for numerical stability)
    float max_val = find_max(input, N);

    // Pass 2: exp(x - max) into output, accumulate sum.
    // Note: expf is a library call and does not auto-vectorise at -O2;
    //       this loop remains scalar in all three variants.
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // Pass 3: normalise and apply scale.
    // Without __restrict__, the compiler cannot prove output[] and scale[]
    // are disjoint.  The alias check prevents widening this into NEON ops.
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; ++i)
        output[i] *= inv_sum * scale[i];
}

int main(int argc, char* argv[]) {
    int N     = 1 << 22;  // 4 M floats (~16 MB)
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

    std::cout << "Baseline softmax+scale  N=" << N << " iters=" << iters << "\n";
    std::cout << "  Time:       " << ms   << " ms\n";
    std::cout << "  Bandwidth:  " << gbps << " GB/s\n";
    std::cout << "  Check: out[0]=" << output[0]
              << "  out[N-1]=" << output[N - 1] << "\n";
    return 0;
}
