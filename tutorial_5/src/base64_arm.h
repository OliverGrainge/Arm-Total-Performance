#pragma once
#include <cstddef>
#include <cstdint>

// NEON-accelerated base64 encode/decode for AArch64.
// This file starts as a scalar fallback stub. Use Codex with the Arm MCP
// Server to generate a real NEON implementation that matches the x86 SIMD
// version's performance.

size_t base64_arm_encode(const uint8_t* input, size_t input_len, char* output);
size_t base64_arm_decode(const char* input, size_t input_len, uint8_t* output);
