#include "base64_arm.h"
#include "base64_scalar.h"

#include <arm_neon.h>

namespace {

static inline uint8x16_t encode_indices(uint8x16_t input) {
    const uint8x16_t lhs_idx = {
        0, 0, 1, 0xFF,
        3, 3, 4, 0xFF,
        6, 6, 7, 0xFF,
        9, 9, 10, 0xFF
    };
    const uint8x16_t rhs_idx = {
        0xFF, 1, 2, 2,
        0xFF, 4, 5, 5,
        0xFF, 7, 8, 8,
        0xFF, 10, 11, 11
    };
    const int8x16_t lhs_shift = {
        -2, 4, 2, 0,
        -2, 4, 2, 0,
        -2, 4, 2, 0,
        -2, 4, 2, 0
    };
    const int8x16_t rhs_shift = {
        0, -4, -6, 0,
        0, -4, -6, 0,
        0, -4, -6, 0,
        0, -4, -6, 0
    };

    uint8x16_t lhs = vqtbl1q_u8(input, lhs_idx);
    uint8x16_t rhs = vqtbl1q_u8(input, rhs_idx);
    uint8x16_t indices = vorrq_u8(vshlq_u8(lhs, lhs_shift), vshlq_u8(rhs, rhs_shift));
    return vandq_u8(indices, vdupq_n_u8(0x3F));
}

static inline uint8x16_t encode_ascii(uint8x16_t indices) {
    uint8x16_t reduced = vqsubq_u8(indices, vdupq_n_u8(51));
    uint8x16_t less26 = vcgtq_s8(vdupq_n_s8(26), vreinterpretq_s8_u8(indices));
    reduced = vorrq_u8(vbicq_u8(reduced, less26), vandq_u8(less26, vdupq_n_u8(13)));

    const int8x16_t shift_lut = {
        71, -4, -4, -4, -4, -4, -4, -4,
        -4, -4, -4, -19, -16, 65, 0, 0
    };
    int8x16_t shifts = vqtbl1q_s8(shift_lut, reduced);
    return vreinterpretq_u8_s8(vaddq_s8(vreinterpretq_s8_u8(indices), shifts));
}

static inline uint8x16_t encode_simd(uint8x16_t input) {
    return encode_ascii(encode_indices(input));
}

static inline uint8x16_t decode_ascii_to_values(uint8x16_t input) {
    int8x16_t in_s8 = vreinterpretq_s8_u8(input);

    uint8x16_t mask_AZ = vandq_u8(
        vcgtq_s8(in_s8, vdupq_n_s8('A' - 1)),
        vcgtq_s8(vdupq_n_s8('Z' + 1), in_s8)
    );
    uint8x16_t mask_az = vandq_u8(
        vcgtq_s8(in_s8, vdupq_n_s8('a' - 1)),
        vcgtq_s8(vdupq_n_s8('z' + 1), in_s8)
    );
    uint8x16_t mask_09 = vandq_u8(
        vcgtq_s8(in_s8, vdupq_n_s8('0' - 1)),
        vcgtq_s8(vdupq_n_s8('9' + 1), in_s8)
    );
    uint8x16_t mask_plus = vceqq_u8(input, vdupq_n_u8('+'));
    uint8x16_t mask_slash = vceqq_u8(input, vdupq_n_u8('/'));

    int8x16_t offset = vandq_s8(vreinterpretq_s8_u8(mask_AZ), vdupq_n_s8(-65));
    offset = vorrq_s8(offset, vandq_s8(vreinterpretq_s8_u8(mask_az), vdupq_n_s8(-71)));
    offset = vorrq_s8(offset, vandq_s8(vreinterpretq_s8_u8(mask_09), vdupq_n_s8(4)));
    offset = vorrq_s8(offset, vandq_s8(vreinterpretq_s8_u8(mask_plus), vdupq_n_s8(19)));
    offset = vorrq_s8(offset, vandq_s8(vreinterpretq_s8_u8(mask_slash), vdupq_n_s8(16)));

    return vreinterpretq_u8_s8(vaddq_s8(in_s8, offset));
}

static inline uint8x16_t decode_merge(uint8x16_t values) {
    const uint8x16_t lhs_idx = {
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        12, 13, 14,
        0xFF, 0xFF, 0xFF, 0xFF
    };
    const uint8x16_t rhs_idx = {
        1, 2, 3,
        5, 6, 7,
        9, 10, 11,
        13, 14, 15,
        0xFF, 0xFF, 0xFF, 0xFF
    };
    const int8x16_t lhs_shift = {
        2, 4, 6,
        2, 4, 6,
        2, 4, 6,
        2, 4, 6,
        0, 0, 0, 0
    };
    const int8x16_t rhs_shift = {
        -4, -2, 0,
        -4, -2, 0,
        -4, -2, 0,
        -4, -2, 0,
        0, 0, 0, 0
    };

    uint8x16_t lhs = vqtbl1q_u8(values, lhs_idx);
    uint8x16_t rhs = vqtbl1q_u8(values, rhs_idx);
    return vorrq_u8(vshlq_u8(lhs, lhs_shift), vshlq_u8(rhs, rhs_shift));
}

static inline uint8x16_t decode_simd(uint8x16_t input) {
    return decode_merge(decode_ascii_to_values(input));
}

}  // namespace

size_t base64_arm_encode(const uint8_t* input, size_t input_len, char* output) {
    size_t i = 0;
    size_t j = 0;

    for (; i + 15 < input_len; i += 12, j += 16) {
        uint8x16_t in = vld1q_u8(input + i);
        uint8x16_t out = encode_simd(in);
        vst1q_u8(reinterpret_cast<uint8_t*>(output + j), out);
    }

    j += base64_scalar_encode(input + i, input_len - i, output + j);
    return j;
}

size_t base64_arm_decode(const char* input, size_t input_len, uint8_t* output) {
    size_t len = input_len;
    while (len > 0 && input[len - 1] == '=') {
        --len;
    }

    size_t i = 0;
    size_t j = 0;

    for (; i + 15 < len; i += 16, j += 12) {
        uint8x16_t in = vld1q_u8(reinterpret_cast<const uint8_t*>(input + i));
        uint8x16_t out = decode_simd(in);
        vst1q_u8(output + j, out);
    }

    j += base64_scalar_decode(input + i, len - i, output + j);
    return j;
}
