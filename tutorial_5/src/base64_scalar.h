#pragma once
#include <cstddef>
#include <cstdint>

// Portable scalar base64 encode/decode (no SIMD).
// These serve as the reference implementation and performance baseline.

// Encodes `input_len` bytes into base64. Writes to `output` (caller must
// allocate at least ((input_len + 2) / 3) * 4 bytes). Returns the number
// of base64 characters written (excluding any null terminator).
size_t base64_scalar_encode(const uint8_t* input, size_t input_len, char* output);

// Decodes `input_len` base64 characters back to bytes. Writes to `output`
// (caller must allocate at least (input_len / 4) * 3 bytes). Returns the
// number of decoded bytes written.
size_t base64_scalar_decode(const char* input, size_t input_len, uint8_t* output);
