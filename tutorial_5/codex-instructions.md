# Codex Task: Port x86 SIMD Base64 to ARM NEON

## Goal

Port the SSE2/SSSE3 base64 encode/decode implementation in `src/base64_x86.cpp`
to ARM NEON intrinsics. Write the result into `src/base64_arm.cpp`, replacing the
current scalar-fallback stubs.

## Context

The x86 implementation processes:
- **Encode:** 12 input bytes → 16 base64 characters per SIMD iteration
- **Decode:** 16 base64 characters → 12 output bytes per SIMD iteration

It uses a reshuffle → multiply-shift → lookup pipeline built on these SSE
intrinsics:

| x86 Intrinsic        | Purpose                           |
|-----------------------|-----------------------------------|
| `_mm_loadu_si128`     | Unaligned 128-bit load            |
| `_mm_storeu_si128`    | Unaligned 128-bit store           |
| `_mm_shuffle_epi8`    | Byte-level permutation (pshufb)   |
| `_mm_and_si128`       | Bitwise AND                       |
| `_mm_or_si128`        | Bitwise OR                        |
| `_mm_andnot_si128`    | Bitwise AND-NOT                   |
| `_mm_set1_epi8/32`    | Broadcast scalar to all lanes     |
| `_mm_setr_epi8`       | Set bytes in order                |
| `_mm_mulhi_epu16`     | Unsigned 16-bit multiply (high)   |
| `_mm_mullo_epi16`     | Signed 16-bit multiply (low)      |
| `_mm_maddubs_epi16`   | Multiply unsigned×signed + hadd   |
| `_mm_madd_epi16`      | Multiply signed 16-bit + hadd     |
| `_mm_subs_epu8`       | Saturating unsigned subtract      |
| `_mm_cmpgt_epi8`      | Signed byte greater-than compare  |
| `_mm_cmpeq_epi8`      | Byte equality compare             |
| `_mm_add_epi8`        | Byte add                          |

## Steps

1. **Use the Arm MCP Server `search_knowledge_base` tool** to look up the NEON
   equivalent for each x86 intrinsic listed above.

2. **Use the Arm MCP Server `migrate_ease_scan` tool** (with `scanner` set to
   `"cpp"`) to scan the workspace and confirm all x86-specific constructs that
   need porting.

3. **Port `encode_simd`** — translate the three encoding stages (reshuffle,
   split, lookup) to NEON. Pay special attention to:
   - `_mm_shuffle_epi8` → `vqtbl1q_u8` (the index semantics differ: NEON zeros
     the lane when the index byte has bit 7 set, which matches SSE pshufb).
   - The multiply-shift trick (`_mm_mulhi_epu16` / `_mm_mullo_epi16`): NEON has
     no direct `mulhi` — use a combination of `vmull_u16` (widening multiply)
     and `vshrn_n_u32` (narrowing shift), or rewrite using explicit shifts.

4. **Port `decode_simd`** — translate the reverse-lookup, merge, and reshuffle
   stages.  Pay special attention to:
   - `_mm_maddubs_epi16`: NEON has no fused multiply-add for bytes. Decompose
     into `vmull_u8` + `vpaddlq_u16` or equivalent shift-and-add sequences.
   - `_mm_madd_epi16`: similarly decompose into 16→32 multiply + pairwise add.

5. **Keep the public API identical** — `base64_arm_encode` and
   `base64_arm_decode` must have the same signature and behaviour as the x86
   versions (including scalar tail handling).

6. **Include `<arm_neon.h>`** at the top of the file.

## Verification

After writing the code, build and run the tests:

```bash
cmake -S . -B build && cmake --build build --parallel
./build/base64_test
```

All tests must pass (NEON output must match scalar output for every size 0–1024).
