#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Internal implementation uses restrict + alignment assumptions while keeping
// the public API unchanged.
static void triad_impl(float* __restrict__ out,
                       const float* __restrict__ a,
                       const float* __restrict__ b,
                       float alpha, int n) {
    out = static_cast<float*>(__builtin_assume_aligned(out, 64));
    a = static_cast<const float*>(__builtin_assume_aligned(a, 64));
    b = static_cast<const float*>(__builtin_assume_aligned(b, 64));

    for (int i = 0; i < n; ++i)
        out[i] = a[i] + alpha * b[i];
}

static void triad(float* out, const float* a, const float* b,
                  float alpha, int n) {
    triad_impl(out, a, b, alpha, n);
}

static float* alloc_aligned_floats(int n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, static_cast<size_t>(n) * sizeof(float)) != 0)
        return nullptr;
    return static_cast<float*>(p);
}

int main(int argc, char* argv[]) {
    int n = 1 << 23;
    int iters = 200;
    float alpha = 0.75f;

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);

    float* a = alloc_aligned_floats(n);
    float* b = alloc_aligned_floats(n);
    float* out = alloc_aligned_floats(n);
    if (!a || !b || !out) {
        std::cerr << "Aligned allocation failed\n";
        std::free(a);
        std::free(b);
        std::free(out);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 1024) * 0.001f;
        b[i] = static_cast<float>((i * 3) % 2048) * 0.0005f;
    }
    std::memset(out, 0, static_cast<size_t>(n) * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter)
        triad(out, a, b, alpha, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double bytes = static_cast<double>(n) * sizeof(float) * 3.0 * iters;
    double gbps = bytes / (ms * 1e6);

    double check = 0.0;
    int sample = n < 1024 ? n : 1024;
    for (int i = 0; i < sample; ++i)
        check += out[i];

    std::cout << "Aligned triad  N=" << n << " iters=" << iters << "\n";
    std::cout << "  Time:       " << ms << " ms\n";
    std::cout << "  Bandwidth:  " << gbps << " GB/s\n";
    std::cout << "  Checksum:   " << check << "\n";

    std::free(a);
    std::free(b);
    std::free(out);
    return 0;
}
