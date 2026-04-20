#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base64_scalar.h"

#if defined(USE_X86_SIMD)
#include "base64_x86.h"
#define base64_simd_encode base64_x86_encode
#define base64_simd_decode base64_x86_decode
#define HAS_SIMD 1
#define SIMD_NAME "SSE"
#elif defined(USE_ARM_SIMD)
#include "base64_arm.h"
#define base64_simd_encode base64_arm_encode
#define base64_simd_decode base64_arm_decode
#define HAS_SIMD 1
#define SIMD_NAME "NEON"
#else
#define HAS_SIMD 0
#endif

static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { g_passed++; }
    else      { g_failed++; printf("FAIL: %s\n", msg); }
}

// RFC 4648 test vectors.
struct TestVec { const char* plain; const char* encoded; };
static const TestVec rfc_vectors[] = {
    {"",       ""},
    {"f",      "Zg=="},
    {"fo",     "Zm8="},
    {"foo",    "Zm9v"},
    {"foob",   "Zm9vYg=="},
    {"fooba",  "Zm9vYmE="},
    {"foobar", "Zm9vYmFy"},
};
static const int NUM_VECTORS = sizeof(rfc_vectors) / sizeof(rfc_vectors[0]);

static void test_scalar_known_vectors() {
    char enc_buf[64];
    uint8_t dec_buf[64];

    for (int t = 0; t < NUM_VECTORS; t++) {
        size_t plain_len = strlen(rfc_vectors[t].plain);
        size_t expect_len = strlen(rfc_vectors[t].encoded);

        // Encode
        size_t enc_len = base64_scalar_encode(
            (const uint8_t*)rfc_vectors[t].plain, plain_len, enc_buf);
        enc_buf[enc_len] = '\0';
        char msg[128];
        snprintf(msg, sizeof(msg), "scalar encode \"%s\" -> \"%s\" (got \"%s\")",
                 rfc_vectors[t].plain, rfc_vectors[t].encoded, enc_buf);
        check(enc_len == expect_len && strcmp(enc_buf, rfc_vectors[t].encoded) == 0, msg);

        // Decode
        size_t dec_len = base64_scalar_decode(
            rfc_vectors[t].encoded, expect_len, dec_buf);
        dec_buf[dec_len] = '\0';
        snprintf(msg, sizeof(msg), "scalar decode \"%s\" -> \"%s\" (got \"%.*s\")",
                 rfc_vectors[t].encoded, rfc_vectors[t].plain, (int)dec_len, dec_buf);
        check(dec_len == plain_len && memcmp(dec_buf, rfc_vectors[t].plain, plain_len) == 0, msg);
    }
}

static void test_scalar_roundtrip() {
    srand(123);
    bool all_ok = true;

    for (int size = 0; size <= 1024; size++) {
        uint8_t input[1024];
        for (int i = 0; i < size; i++) input[i] = (uint8_t)(rand() & 0xFF);

        char encoded[2048];
        uint8_t decoded[1024];

        size_t enc_len = base64_scalar_encode(input, (size_t)size, encoded);
        size_t dec_len = base64_scalar_decode(encoded, enc_len, decoded);

        if (dec_len != (size_t)size || memcmp(input, decoded, (size_t)size) != 0) {
            char msg[64];
            snprintf(msg, sizeof(msg), "scalar round-trip size=%d", size);
            check(false, msg);
            all_ok = false;
        }
    }
    if (all_ok) {
        printf("  scalar round-trip sizes 0..1024: OK\n");
        g_passed++;
    }
}

#if HAS_SIMD
static void test_simd_vs_scalar() {
    srand(456);
    bool all_ok = true;

    for (int size = 0; size <= 1024; size++) {
        uint8_t input[1024];
        for (int i = 0; i < size; i++) input[i] = (uint8_t)(rand() & 0xFF);

        char sc_enc[2048], sm_enc[2048];
        uint8_t sc_dec[1024], sm_dec[1024];

        size_t sc_enc_len = base64_scalar_encode(input, (size_t)size, sc_enc);
        size_t sm_enc_len = base64_simd_encode(input, (size_t)size, sm_enc);

        if (sc_enc_len != sm_enc_len ||
            memcmp(sc_enc, sm_enc, sc_enc_len) != 0) {
            char msg[64];
            snprintf(msg, sizeof(msg), "%s encode mismatch size=%d", SIMD_NAME, size);
            check(false, msg);
            all_ok = false;
            continue;
        }

        size_t sc_dec_len = base64_scalar_decode(sc_enc, sc_enc_len, sc_dec);
        size_t sm_dec_len = base64_simd_decode(sc_enc, sc_enc_len, sm_dec);

        if (sc_dec_len != sm_dec_len ||
            memcmp(sc_dec, sm_dec, sc_dec_len) != 0) {
            char msg[64];
            snprintf(msg, sizeof(msg), "%s decode mismatch size=%d", SIMD_NAME, size);
            check(false, msg);
            all_ok = false;
        }
    }
    if (all_ok) {
        printf("  %s vs scalar sizes 0..1024: OK\n", SIMD_NAME);
        g_passed++;
    }
}
#endif

int main() {
    printf("=== Base64 Correctness Tests ===\n\n");

    printf("Scalar known-vector tests:\n");
    test_scalar_known_vectors();

    printf("\nScalar round-trip tests:\n");
    test_scalar_roundtrip();

#if HAS_SIMD
    printf("\n%s vs scalar comparison tests:\n", SIMD_NAME);
    test_simd_vs_scalar();
#else
    printf("\n(SIMD tests skipped — no SIMD backend compiled)\n");
#endif

    printf("\n=== %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
