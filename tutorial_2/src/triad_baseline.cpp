#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

// STREAM-style triad kernel:
//   out[i] = a[i] + alpha * b[i]
//
// Baseline version without pointer annotations. The compiler must assume that
// out, a, and b may alias, which can inhibit vectorization and keep this loop
// in scalar load/store form.
static void triad(float* out, const float* a, const float* b,
                  float alpha, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + alpha * b[i];
}

int main(int argc, char* argv[]) {
    int n = 1 << 23;   // 8M floats
    int iters = 200;
    float alpha = 0.75f;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);

    std::vector<float> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 1024) * 0.001f;
        b[i] = static_cast<float>((i * 3) % 2048) * 0.0005f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter)
        triad(out.data(), a.data(), b.data(), alpha, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double bytes = static_cast<double>(n) * sizeof(float) * 3.0 * iters;
    double gbps = bytes / (ms * 1e6);

    double check = 0.0;
    int sample = n < 1024 ? n : 1024;
    for (int i = 0; i < sample; ++i)
        check += out[i];

    std::cout << "Baseline triad  N=" << n << " iters=" << iters << "\n";
    std::cout << "  Time:       " << ms << " ms\n";
    std::cout << "  Bandwidth:  " << gbps << " GB/s\n";
    std::cout << "  Checksum:   " << check << "\n";
    return 0;
}
