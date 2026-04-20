#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "base64_scalar.h"

#if defined(USE_X86_SIMD)
#include "base64_x86.h"
#define HAS_SIMD 1
#define SIMD_NAME "SSE"
#define base64_simd_encode base64_x86_encode
#define base64_simd_decode base64_x86_decode
#elif defined(USE_ARM_SIMD)
#include "base64_arm.h"
#define HAS_SIMD 1
#define SIMD_NAME "NEON"
#define base64_simd_encode base64_arm_encode
#define base64_simd_decode base64_arm_decode
#else
#define HAS_SIMD 0
#endif

static void fill_random(std::vector<uint8_t>& buf) {
    for (size_t i = 0; i < buf.size(); i++)
        buf[i] = (uint8_t)(rand() & 0xFF);
}

static double measure_encode(
    size_t (*fn)(const uint8_t*, size_t, char*),
    const uint8_t* input, size_t input_len, char* output, int iters)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fn(input, input_len, output);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

static double measure_decode(
    size_t (*fn)(const char*, size_t, uint8_t*),
    const char* input, size_t input_len, uint8_t* output, int iters)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fn(input, input_len, output);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

int main(int argc, char* argv[]) {
    size_t data_mb  = 10;
    int    iters    = 5;

    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--size") == 0 && a + 1 < argc)
            data_mb = (size_t)atoi(argv[++a]);
        else if (strcmp(argv[a], "--iters") == 0 && a + 1 < argc)
            iters = atoi(argv[++a]);
    }

    size_t data_size = data_mb * 1024 * 1024;
    size_t enc_size  = ((data_size + 2) / 3) * 4;

    printf("Base64 Benchmark  (%zu MB, %d iterations)\n\n", data_mb, iters);

    // Generate random input (simulating image data).
    srand(42);
    std::vector<uint8_t> input(data_size);
    fill_random(input);

    std::vector<char>    encoded(enc_size);
    std::vector<uint8_t> decoded(data_size);

    // ---- Scalar ----
    double sc_enc = measure_encode(base64_scalar_encode,
                                   input.data(), data_size,
                                   encoded.data(), iters);
    size_t enc_len = base64_scalar_encode(input.data(), data_size, encoded.data());

    double sc_dec = measure_decode(base64_scalar_decode,
                                   encoded.data(), enc_len,
                                   decoded.data(), iters);

    double sc_enc_tp = (double)data_size / (1024.0 * 1024.0) / sc_enc;
    double sc_dec_tp = (double)data_size / (1024.0 * 1024.0) / sc_dec;

    printf("%-12s  Encode: %8.1f MB/s   Decode: %8.1f MB/s\n",
           "Scalar", sc_enc_tp, sc_dec_tp);

    // Verify scalar round-trip.
    size_t dec_len = base64_scalar_decode(encoded.data(), enc_len, decoded.data());
    if (dec_len != data_size || memcmp(input.data(), decoded.data(), data_size) != 0) {
        printf("  ** scalar round-trip FAILED **\n");
        return 1;
    }

#if HAS_SIMD
    // ---- SIMD ----
    std::vector<char>    simd_enc(enc_size);
    std::vector<uint8_t> simd_dec(data_size);

    double sm_enc = measure_encode(base64_simd_encode,
                                   input.data(), data_size,
                                   simd_enc.data(), iters);
    size_t simd_enc_len = base64_simd_encode(input.data(), data_size, simd_enc.data());

    double sm_dec = measure_decode(base64_simd_decode,
                                   simd_enc.data(), simd_enc_len,
                                   simd_dec.data(), iters);

    double sm_enc_tp = (double)data_size / (1024.0 * 1024.0) / sm_enc;
    double sm_dec_tp = (double)data_size / (1024.0 * 1024.0) / sm_dec;

    printf("%-12s  Encode: %8.1f MB/s   Decode: %8.1f MB/s\n",
           SIMD_NAME, sm_enc_tp, sm_dec_tp);

    // Verify SIMD matches scalar output.
    if (simd_enc_len != enc_len ||
        memcmp(encoded.data(), simd_enc.data(), enc_len) != 0) {
        printf("  ** %s encode output MISMATCH vs scalar **\n", SIMD_NAME);
        return 1;
    }

    size_t simd_dec_len = base64_simd_decode(simd_enc.data(), simd_enc_len,
                                             simd_dec.data());
    if (simd_dec_len != data_size ||
        memcmp(input.data(), simd_dec.data(), data_size) != 0) {
        printf("  ** %s decode round-trip FAILED **\n", SIMD_NAME);
        return 1;
    }

    printf("\nSpeedup (%s / Scalar):  Encode %.2fx   Decode %.2fx\n",
           SIMD_NAME, sm_enc_tp / sc_enc_tp, sm_dec_tp / sc_dec_tp);
#endif

    printf("\nAll correctness checks passed.\n");
    return 0;
}
