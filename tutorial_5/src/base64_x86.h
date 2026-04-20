#pragma once
#include <cstddef>
#include <cstdint>

// SSE2/SSSE3-accelerated base64 encode/decode.
// Processes 12 input bytes (encode) or 16 base64 characters (decode)
// per SIMD iteration. Falls back to scalar for the remainder.

size_t base64_x86_encode(const uint8_t* input, size_t input_len, char* output);
size_t base64_x86_decode(const char* input, size_t input_len, uint8_t* output);
