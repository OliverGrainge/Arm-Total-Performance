#include "base64_x86.h"
#include "base64_scalar.h"

#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3

// ---------------------------------------------------------------------------
// Encoding: 12 input bytes  -->  16 base64 characters per SIMD iteration
// ---------------------------------------------------------------------------
//
// The algorithm (Wojciech Mula, 2016) works in three stages:
//
//   1. RESHUFFLE – rearrange each group of 3 input bytes into a 32-bit lane
//      so that the subsequent multiply-shift trick can extract four 6-bit
//      indices from the 24 input bits.
//
//   2. SPLIT – use a multiply-and-mask trick to isolate the four 6-bit
//      fields inside each 32-bit lane.
//
//   3. LOOKUP – translate the 0-63 indices into ASCII base64 characters
//      using a comparison-based range map and a pshufb lookup table.
// ---------------------------------------------------------------------------

static inline __m128i encode_simd(__m128i input) {
    // Stage 1: Reshuffle input bytes.
    // For each group of 3 consecutive bytes (B0, B1, B2) we want the 32-bit
    // lane laid out as [B1, B0, B2, B1] (little-endian byte order).  This
    // arrangement lets the multiply trick in Stage 2 pull out the four 6-bit
    // fields with simple integer arithmetic.
    const __m128i shuffle_enc = _mm_setr_epi8(
         1,  0,  2,  1,     // bytes 0-2  -> lane 0
         4,  3,  5,  4,     // bytes 3-5  -> lane 1
         7,  6,  8,  7,     // bytes 6-8  -> lane 2
        10,  9, 11, 10      // bytes 9-11 -> lane 3
    );
    __m128i reshuffled = _mm_shuffle_epi8(input, shuffle_enc);

    // Stage 2: Extract 6-bit indices via multiply-and-shift.
    //
    // After the reshuffle each 32-bit lane holds 24 useful bits arranged so
    // that two complementary masks can isolate alternating 6-bit fields:
    //
    //   mask0 = 0x0FC0FC00  ->  selects bits for indices 0 and 2
    //   mask1 = 0x003F03F0  ->  selects bits for indices 1 and 3
    //
    // A 16-bit multiply then shifts each masked pair into its final byte
    // position:
    //
    //   mulhi_epu16(t0, 0x04000040)  is equivalent to  t0 >> {6, 10}
    //   mullo_epi16(t1, 0x01000010)  is equivalent to  t1 << {4, 8}
    //
    // OR-ing the two results places all four 6-bit indices in separate bytes.
    __m128i t0 = _mm_and_si128(reshuffled, _mm_set1_epi32(0x0FC0FC00));
    __m128i t1 = _mm_and_si128(reshuffled, _mm_set1_epi32(0x003F03F0));

    t0 = _mm_mulhi_epu16(t0, _mm_set1_epi32(0x04000040));
    t1 = _mm_mullo_epi16(t1, _mm_set1_epi32(0x01000010));

    __m128i indices = _mm_or_si128(t0, t1);

    // Stage 3: Translate 6-bit indices [0..63] to base64 ASCII.
    //
    // Build a "reduced" index that collapses the five ASCII ranges into a
    // small set of lookup keys:
    //
    //   index  0-25  (A-Z)  ->  reduced = 13   (forced via comparison)
    //   index 26-51  (a-z)  ->  reduced = 0    (saturating sub gives 0)
    //   index 52-61  (0-9)  ->  reduced = 1-10
    //   index 62     (+)    ->  reduced = 11
    //   index 63     (/)    ->  reduced = 12
    //
    // A single pshufb then maps each reduced value to the signed offset
    // that, when added to the original index, produces the correct ASCII
    // character.
    __m128i reduced = _mm_subs_epu8(indices, _mm_set1_epi8(51));

    // Distinguish 0-25 from 26-51 (both give reduced == 0 after sub).
    __m128i less26 = _mm_cmpgt_epi8(_mm_set1_epi8(26), indices);
    reduced = _mm_or_si128(
        _mm_andnot_si128(less26, reduced),
        _mm_and_si128(less26, _mm_set1_epi8(13))
    );

    //  reduced  ->  offset
    //  -------      ------
    //    0          +71   ('a' - 26)
    //    1..10      -4    ('0' - 52)
    //    11         -19   ('+' - 62)
    //    12         -16   ('/' - 63)
    //    13         +65   ('A' -  0)
    const __m128i shift_lut = _mm_setr_epi8(
        71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65, 0, 0
    );
    __m128i shifts = _mm_shuffle_epi8(shift_lut, reduced);

    return _mm_add_epi8(indices, shifts);
}

// ---------------------------------------------------------------------------
// Decoding: 16 base64 characters  -->  12 output bytes per SIMD iteration
// ---------------------------------------------------------------------------
//
//   1. REVERSE LOOKUP – map each ASCII character to its 6-bit value using
//      range comparisons (no lookup table in memory, fully in-register).
//
//   2. MERGE – pack four 6-bit values into three bytes using the
//      maddubs / madd multiply trick (the algebraic inverse of encoding).
//
//   3. RESHUFFLE – reorder the 32-bit lanes so the 12 decoded bytes are
//      contiguous.
// ---------------------------------------------------------------------------

static inline __m128i decode_ascii_to_values(__m128i input) {
    // Classify each byte into one of five base64 character ranges.
    __m128i mask_AZ = _mm_and_si128(
        _mm_cmpgt_epi8(input,               _mm_set1_epi8('A' - 1)),
        _mm_cmpgt_epi8(_mm_set1_epi8('Z' + 1), input)
    );
    __m128i mask_az = _mm_and_si128(
        _mm_cmpgt_epi8(input,               _mm_set1_epi8('a' - 1)),
        _mm_cmpgt_epi8(_mm_set1_epi8('z' + 1), input)
    );
    __m128i mask_09 = _mm_and_si128(
        _mm_cmpgt_epi8(input,               _mm_set1_epi8('0' - 1)),
        _mm_cmpgt_epi8(_mm_set1_epi8('9' + 1), input)
    );
    __m128i mask_plus  = _mm_cmpeq_epi8(input, _mm_set1_epi8('+'));
    __m128i mask_slash = _mm_cmpeq_epi8(input, _mm_set1_epi8('/'));

    // For each range, compute the signed offset such that  char + offset == index.
    //   'A'-'Z' (65-90)  -> index 0-25   : offset = -65
    //   'a'-'z' (97-122) -> index 26-51  : offset = -71
    //   '0'-'9' (48-57)  -> index 52-61  : offset =  +4
    //   '+'     (43)     -> index 62     : offset = +19
    //   '/'     (47)     -> index 63     : offset = +16
    __m128i offset = _mm_and_si128(mask_AZ, _mm_set1_epi8(-65));
    offset = _mm_or_si128(offset, _mm_and_si128(mask_az, _mm_set1_epi8(-71)));
    offset = _mm_or_si128(offset, _mm_and_si128(mask_09, _mm_set1_epi8(4)));
    offset = _mm_or_si128(offset, _mm_and_si128(mask_plus,  _mm_set1_epi8(19)));
    offset = _mm_or_si128(offset, _mm_and_si128(mask_slash, _mm_set1_epi8(16)));

    return _mm_add_epi8(input, offset);
}

static inline __m128i decode_simd(__m128i input) {
    __m128i values = decode_ascii_to_values(input);

    // Stage 2: Merge pairs of 6-bit values into 12-bit values.
    //
    // _mm_maddubs_epi16(a, b) treats `a` as unsigned bytes and `b` as signed
    // bytes, then for every consecutive pair (a[2i], a[2i+1]):
    //
    //   result_16[i] = a[2i] * b[2i]  +  a[2i+1] * b[2i+1]
    //
    // With coefficients [0x40, 0x01] = [64, 1]:
    //   pair(v0, v1) -> v0*64 + v1 = (v0 << 6) | v1   (a 12-bit value)
    __m128i merged_pairs = _mm_maddubs_epi16(
        values, _mm_set1_epi32(0x01400140)
    );

    // Stage 3: Merge pairs of 12-bit values into 24-bit values.
    //
    // _mm_madd_epi16(a, b) multiplies signed 16-bit pairs and sums adjacent
    // results into 32-bit lanes:
    //
    //   result_32[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
    //
    // With coefficients [0x1000, 0x0001] = [4096, 1]:
    //   pair(v01, v23) -> v01*4096 + v23 = (v01 << 12) | v23   (24 bits)
    __m128i merged_quads = _mm_madd_epi16(
        merged_pairs, _mm_set1_epi32(0x00011000)
    );

    // Stage 4: Reshuffle – extract the 3 useful bytes from each 32-bit lane.
    // In little-endian layout the 24-bit value occupies bytes [0..2] of each
    // lane in reversed order, so we pick byte offsets 2,1,0 for the first
    // group, 6,5,4 for the second, and so on.
    const __m128i shuffle_dec = _mm_setr_epi8(
         2,  1,  0,
         6,  5,  4,
        10,  9,  8,
        14, 13, 12,
        -1, -1, -1, -1   // unused – zeroed by pshufb
    );
    return _mm_shuffle_epi8(merged_quads, shuffle_dec);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

size_t base64_x86_encode(const uint8_t* input, size_t input_len, char* output) {
    size_t i = 0, j = 0;

    // SIMD path: process 12 bytes -> 16 base64 chars per iteration.
    for (; i + 15 < input_len; i += 12, j += 16) {
        __m128i in  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i out = encode_simd(in);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + j), out);
    }

    // Scalar tail for remaining bytes.
    j += base64_scalar_encode(input + i, input_len - i, output + j);
    return j;
}

size_t base64_x86_decode(const char* input, size_t input_len, uint8_t* output) {
    // Strip trailing padding before entering the SIMD loop.
    size_t len = input_len;
    while (len > 0 && input[len - 1] == '=') len--;

    size_t i = 0, j = 0;

    // SIMD path: process 16 base64 chars -> 12 bytes per iteration.
    for (; i + 15 < len; i += 16, j += 12) {
        __m128i in  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i out = decode_simd(in);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + j), out);
    }

    // Scalar tail for remaining characters.
    j += base64_scalar_decode(input + i, len - i, output + j);
    return j;
}
