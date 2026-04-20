# Tutorial 5: Porting x86 SIMD Code to Arm with Codex and the Arm MCP Server

Real-world C++ codebases frequently contain hand-written x86 SIMD intrinsics — `_mm_shuffle_epi8`, `_mm_mulhi_epu16`, `_mm_maddubs_epi16` and dozens more — that produce immediate compiler errors on AArch64. Manually rewriting these functions requires deep knowledge of both x86 SSE and Arm NEON instruction sets, making migration one of the most time-consuming steps in an x86-to-Arm port.

This tutorial shows you how to use **Codex** (OpenAI's code-generation agent) together with the **Arm MCP Server** to automate most of that work. The Arm MCP Server gives Codex access to Arm's documentation, an intrinsic-mapping knowledge base, and the `migrate-ease` compatibility scanner — tools that would normally require a human expert to drive. You'll watch Codex analyse an SSE2/SSSE3-optimised base64 encoder/decoder, look up the correct NEON equivalents, and produce a working port.

By the end you'll know how to:

1. Identify x86-specific SIMD code that blocks compilation on Arm.
2. Set up Codex CLI with the Arm MCP Server.
3. Use an AI code agent to automatically port SSE intrinsics to NEON.
4. Verify correctness and benchmark the ported implementation.

---

## Before you begin

- An **AWS Graviton** instance (`c7g.metal` recommended)
- **GCC 11+** or **Clang 14+**
- **CMake 3.16+**
- **Docker** (for the Arm MCP Server container)
- **Codex CLI** installed (`npm install -g @openai/codex`)
- An **OpenAI API key** configured for Codex

---

## Background: x86 SIMD versus Arm SIMD

### SSE/AVX and NEON at a glance

Both x86 and Arm provide 128-bit SIMD registers that let a single instruction operate on multiple data elements in parallel.

| Feature          | x86 SSE/SSE2/SSSE3           | Arm NEON                         |
|------------------|------------------------------|----------------------------------|
| Register width   | 128-bit (XMM)               | 128-bit (V registers)            |
| Integer types    | 8/16/32/64-bit              | 8/16/32/64-bit                   |
| Byte shuffle     | `pshufb` (_mm_shuffle_epi8) | `tbl` (vqtbl1q_u8)              |
| Multiply-high    | `pmulhuw` (_mm_mulhi_epu16) | No direct equivalent             |
| Fused madd bytes | `pmaddubsw`                 | No direct equivalent             |
| Header           | `<emmintrin.h>` / `<tmmintrin.h>` | `<arm_neon.h>`            |

The two instruction sets are **conceptually similar** but **not 1:1 compatible**. Simple operations like bitwise AND, byte-level compare, and broadcast have straightforward mappings. More complex operations — unsigned multiply-high, fused byte multiply-add — have no single NEON equivalent and must be decomposed into two or three instructions.

### Why manual porting is hard

Consider a single line of x86 code:

```cpp
__m128i t0 = _mm_mulhi_epu16(masked, _mm_set1_epi32(0x04000040));
```

To port this, you need to:

1. Understand that `_mm_mulhi_epu16` computes the **upper 16 bits** of an unsigned 16-bit multiply.
2. Know that NEON has no single-instruction equivalent.
3. Devise a replacement — for example, a widening multiply (`vmull_u16`) followed by a narrowing right-shift (`vshrn_n_u32`).
4. Ensure the surrounding code still produces correct results.

Multiply this by the dozens of intrinsics in a real SIMD kernel and the task becomes significant.

### How Codex with the Arm MCP Server helps

The **Arm MCP Server** is a Docker container that exposes several tools over the Model Context Protocol:

- **Knowledge Base Search** — semantic search across Arm documentation, intrinsic references, and migration guides.
- **migrate_ease_scan** — scans a C++ codebase (set `scanner` to `"cpp"`) and flags every x86-specific construct.
- **LLVM-MCA analysis** — estimates instruction throughput for assembly snippets.

When Codex is configured to use the Arm MCP Server, it can call these tools autonomously: scan your code for x86 intrinsics, look up the NEON equivalent of each one, and write the ported implementation — all without manual intervention.

---

## The Workload: SIMD-Accelerated Base64

### What is Base64?

Base64 is a binary-to-text encoding that represents arbitrary byte sequences using 64 printable ASCII characters (`A-Z`, `a-z`, `0-9`, `+`, `/`). It is ubiquitous in web APIs, email (MIME), data URIs, and anywhere binary data (such as images) must be transmitted over text-only channels.

The encoding works by taking every **3 input bytes** (24 bits) and splitting them into **4 × 6-bit groups**, each of which maps to one character in the base64 alphabet:

```text
Input bytes:     [  byte0  ] [  byte1  ] [  byte2  ]
Binary:          XXXXXXXX    YYYYYYYY    ZZZZZZZZ
6-bit groups:    XXXXXX  XXYYYY  YYYYZZ  ZZZZZZ
Base64 chars:      [A]     [B]     [C]     [D]
```

### Why SIMD accelerates Base64

The scalar implementation processes one 3-byte group at a time using shifts and a lookup table. A SIMD implementation processes **four groups (12 bytes) in parallel** per 128-bit register:

1. **Reshuffle** — use a byte-shuffle instruction to rearrange 12 input bytes into four 32-bit lanes, each containing one 3-byte group.
2. **Split** — use multiply-and-mask tricks to extract the four 6-bit indices from each lane.
3. **Lookup** — use a shuffle-based lookup table to convert indices to ASCII characters.

The decode path reverses these steps: ASCII → 6-bit values → merge → reshuffle.

### The x86 implementation

Open `src/base64_x86.cpp` and examine the encoding pipeline. The core of the encoder looks like this:

```cpp
// Stage 1: Reshuffle input bytes into 32-bit lanes
const __m128i shuffle_enc = _mm_setr_epi8(
     1,  0,  2,  1,   4,  3,  5,  4,   7,  6,  8,  7,   10,  9, 11, 10
);
__m128i reshuffled = _mm_shuffle_epi8(input, shuffle_enc);

// Stage 2: Extract 6-bit indices via multiply-and-shift
__m128i t0 = _mm_and_si128(reshuffled, _mm_set1_epi32(0x0FC0FC00));
__m128i t1 = _mm_and_si128(reshuffled, _mm_set1_epi32(0x003F03F0));
t0 = _mm_mulhi_epu16(t0, _mm_set1_epi32(0x04000040));
t1 = _mm_mullo_epi16(t1, _mm_set1_epi32(0x01000010));
__m128i indices = _mm_or_si128(t0, t1);

// Stage 3: Translate indices to base64 ASCII via pshufb lookup
__m128i reduced = _mm_subs_epu8(indices, _mm_set1_epi8(51));
__m128i less26  = _mm_cmpgt_epi8(_mm_set1_epi8(26), indices);
reduced = _mm_or_si128(
    _mm_andnot_si128(less26, reduced),
    _mm_and_si128(less26, _mm_set1_epi8(13))
);
const __m128i shift_lut = _mm_setr_epi8(
    71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65, 0, 0
);
__m128i shifts = _mm_shuffle_epi8(shift_lut, reduced);
return _mm_add_epi8(indices, shifts);
```

This code uses **13 distinct x86 intrinsics**. None of them exist on AArch64. If you try to compile this file on your Graviton instance, the compiler will reject it immediately — which is exactly where Codex and the Arm MCP Server come in.

---

## Build the Scalar Baseline

Before porting the SIMD code, build the project on your Graviton instance using only the scalar implementation. The ARM stub in `src/base64_arm.cpp` currently delegates to scalar, so both the "Scalar" and "NEON" rows will show the same throughput.

```bash
cd tutorial_5
cmake -S . -B build
cmake --build build --parallel
```

Run the benchmark:

```bash
./build/base64_bench
```

You should see output similar to:

```text
Base64 Benchmark  (10 MB, 5 iterations)

Scalar        Encode:    320.0 MB/s   Decode:    290.0 MB/s
NEON          Encode:    320.0 MB/s   Decode:    290.0 MB/s

Speedup (NEON / Scalar):  Encode 1.00x   Decode 1.00x

All correctness checks passed.
```

Record these baseline numbers. After Codex ports the SIMD code, the NEON row should be significantly faster.

Run the correctness tests to confirm the scaffold is working:

```bash
./build/base64_test
```

```text
=== Base64 Correctness Tests ===

Scalar known-vector tests:
  (14 checks)

Scalar round-trip tests:
  scalar round-trip sizes 0..1024: OK

NEON vs scalar comparison tests:
  NEON vs scalar sizes 0..1024: OK

=== 16 passed, 0 failed ===
```

Everything passes because the stub simply calls the scalar code. After Codex fills in the real NEON implementation, the same tests will verify that the SIMD output exactly matches the scalar reference.

---

## Examine the x86 SIMD Code

Take a moment to read through `src/base64_x86.cpp`. You do not need to understand every detail, but note the x86 intrinsics used. The complete list is:

| Intrinsic              | Purpose                                       |
|------------------------|-----------------------------------------------|
| `_mm_loadu_si128`      | Unaligned 128-bit load                        |
| `_mm_storeu_si128`     | Unaligned 128-bit store                       |
| `_mm_shuffle_epi8`     | Byte permutation (SSSE3 pshufb)               |
| `_mm_and_si128`        | Bitwise AND                                   |
| `_mm_or_si128`         | Bitwise OR                                    |
| `_mm_andnot_si128`     | Bitwise AND-NOT                               |
| `_mm_set1_epi8`        | Broadcast byte to all lanes                   |
| `_mm_set1_epi32`       | Broadcast 32-bit value to all lanes           |
| `_mm_setr_epi8`        | Set 16 bytes in order                         |
| `_mm_mulhi_epu16`      | Unsigned 16-bit multiply, return high half     |
| `_mm_mullo_epi16`      | 16-bit multiply, return low half              |
| `_mm_maddubs_epi16`    | Multiply unsigned×signed bytes, pairwise add  |
| `_mm_madd_epi16`       | Multiply signed 16-bit, pairwise add to 32    |
| `_mm_subs_epu8`        | Saturating unsigned byte subtract             |
| `_mm_cmpgt_epi8`       | Signed byte greater-than                      |
| `_mm_cmpeq_epi8`       | Byte equality                                 |
| `_mm_add_epi8`         | Byte addition                                 |

Try to compile the x86 file directly on your Graviton instance to see the errors:

```bash
g++ -c -mssse3 src/base64_x86.cpp -Isrc
```

```text
cc1plus: error: unrecognized command-line option '-mssse3'
```

The `-mssse3` flag does not exist on AArch64, and the `<tmmintrin.h>` header is not present. This is the code that needs to be ported.

---

## Set Up Codex with the Arm MCP Server

### Step 1: Pull the Arm MCP Docker image

```bash
docker pull armlimited/arm-mcp:latest
```

Verify the image is available:

```bash
docker images | grep arm-mcp
```

### Step 2: Configure Codex CLI

Codex reads its MCP server configuration from a TOML config file. Add the Arm MCP Server to your Codex configuration. Create or edit `~/.codex/config.toml`:

```toml
[mcp_servers.arm-mcp]
command = "docker"
args = [
  "run",
  "--rm",
  "-i",
  "-v", "./:/workspace",
  "armlimited/arm-mcp"
]
```

This mounts your current working directory (the `tutorial_5` folder) into the container at `/workspace`, giving the MCP server access to your source code.

### Step 3: Verify the setup

Start Codex in interactive mode and confirm it can see the Arm MCP Server:

```bash
codex
```

At the Codex prompt, type a simple test query:

```
Use the arm-mcp search_knowledge_base tool to look up the NEON equivalent of _mm_shuffle_epi8.
```

You should see Codex call the Arm MCP Server and return a result mentioning `vqtbl1q_u8`. If this works, the setup is ready.

---

## Scan the Codebase with migrate-ease

Before porting, use the Arm MCP Server's `migrate_ease_scan` tool to get a compatibility report. Run the following command in Codex:

```bash
codex "Use the arm-mcp migrate_ease_scan tool with scanner set to cpp to scan the workspace and report all x86-specific code."
```

Codex will invoke the MCP server's `migrate_ease_scan` tool, which runs **migrate-ease** on your source tree. The output will look similar to:

```text
Scanning /workspace/src/ ...

File: src/base64_x86.cpp
  - #include <emmintrin.h>     (x86 SSE2 header)
  - #include <tmmintrin.h>     (x86 SSSE3 header)
  - 17 x86 intrinsic calls detected:
      _mm_loadu_si128, _mm_storeu_si128, _mm_shuffle_epi8,
      _mm_and_si128, _mm_or_si128, _mm_andnot_si128,
      _mm_set1_epi8, _mm_set1_epi32, _mm_setr_epi8,
      _mm_mulhi_epu16, _mm_mullo_epi16, _mm_maddubs_epi16,
      _mm_madd_epi16, _mm_subs_epu8, _mm_cmpgt_epi8,
      _mm_cmpeq_epi8, _mm_add_epi8

Files with no issues: base64_scalar.cpp, base64_arm.cpp, main.cpp, test_base64.cpp
```

This confirms that all x86-specific code is isolated in `base64_x86.cpp`. The porting task is well-scoped: we need to produce an equivalent `base64_arm.cpp` using NEON intrinsics.

---

## Port the Code with Codex

This is the core step. You will give Codex a prompt that tells it to read the x86 implementation, use the Arm MCP Server to look up NEON equivalents, and write the ported code.

### The porting prompt

The file `codex-instructions.md` in this tutorial directory contains a detailed task description for Codex. Open it and review the contents — it specifies:

- Which file to read (`src/base64_x86.cpp`)
- Which file to write (`src/base64_arm.cpp`)
- Which MCP tools to use (`search_knowledge_base`, `migration_scan`)
- Specific porting guidance for the tricky intrinsics

### Run Codex

From the `tutorial_5` directory, run:

```bash
codex "Follow the instructions in codex-instructions.md to port the x86 SIMD base64 implementation to ARM NEON. Write the result to src/base64_arm.cpp."
```

Codex will work through several steps:

1. **Read** `src/base64_x86.cpp` to understand the SSE implementation.
2. **Call `search_knowledge_base`** via the Arm MCP Server to look up NEON equivalents for each intrinsic.
3. **Call `migrate_ease_scan`** to confirm its understanding of the x86-specific code.
4. **Write** the NEON implementation to `src/base64_arm.cpp`.

This typically takes 1–3 minutes. When it finishes, Codex will show you the generated code.

### Review the generated code

Open `src/base64_arm.cpp` and inspect the result. You should see:

- `#include <arm_neon.h>` at the top instead of x86 headers.
- The same three-stage encode pipeline (reshuffle → split → lookup) rewritten with NEON intrinsics.
- The same three-stage decode pipeline (reverse lookup → merge → reshuffle) in NEON.
- Scalar tail handling identical to the x86 version.

The key intrinsic mappings Codex should have applied:

| x86 Intrinsic           | NEON Equivalent                                              |
|--------------------------|--------------------------------------------------------------|
| `_mm_loadu_si128`        | `vld1q_u8`                                                   |
| `_mm_storeu_si128`       | `vst1q_u8`                                                   |
| `_mm_shuffle_epi8`       | `vqtbl1q_u8`                                                 |
| `_mm_and_si128`          | `vandq_u8`                                                   |
| `_mm_or_si128`           | `vorrq_u8`                                                   |
| `_mm_andnot_si128(a,b)`  | `vbicq_u8(b, a)` (note reversed operands)                    |
| `_mm_set1_epi8`          | `vdupq_n_u8` / `vdupq_n_s8`                                 |
| `_mm_mulhi_epu16`        | Widening multiply + narrowing shift                          |
| `_mm_maddubs_epi16`      | Widening multiply + pairwise add                             |
| `_mm_madd_epi16`         | Widening multiply + pairwise add                             |
| `_mm_subs_epu8`          | `vqsubq_u8`                                                  |
| `_mm_cmpgt_epi8`         | `vcgtq_s8`                                                   |
| `_mm_cmpeq_epi8`         | `vceqq_u8`                                                   |
| `_mm_add_epi8`           | `vaddq_s8` / `vaddq_u8`                                     |

> **Note:** The most challenging translations are `_mm_mulhi_epu16`, `_mm_maddubs_epi16`, and `_mm_madd_epi16`. These have no single-instruction NEON equivalents. Codex should decompose them into multi-instruction sequences — for example, using `vmull_u16` (widening multiply producing 32-bit results) followed by `vshrn_n_u32` (narrowing shift back to 16-bit). If Codex produces a different decomposition that is functionally correct, that is equally valid.

---

## Build and Verify the Ported Code

Rebuild the project now that `src/base64_arm.cpp` contains real NEON code:

```bash
cmake --build build --parallel
```

If the build fails, read the compiler errors. Common issues include:

- **Type mismatches** — NEON has distinct types (`uint8x16_t`, `int8x16_t`, `uint16x8_t`, etc.) where SSE uses a single `__m128i`. Codex may need a correction pass to add `vreinterpretq_*` casts.
- **Operand order** — `_mm_andnot_si128(a, b)` computes `(~a) & b`, but the NEON equivalent `vbicq_u8(a, b)` computes `a & (~b)`. The operands must be swapped.

If errors appear, you can ask Codex to fix them:

```bash
codex "Fix the build errors in src/base64_arm.cpp. Here are the errors: <paste errors>"
```

Once the build succeeds, run the correctness tests:

```bash
./build/base64_test
```

Expected output:

```text
=== Base64 Correctness Tests ===

Scalar known-vector tests:
NEON vs scalar comparison tests:
  NEON vs scalar sizes 0..1024: OK

=== 16 passed, 0 failed ===
```

The NEON implementation now produces **byte-identical output** to the scalar reference for every input size from 0 to 1024 bytes, including all padding edge cases.

---

## Benchmark the Ported Code

Run the benchmark again:

```bash
./build/base64_bench
```

With the NEON implementation active, you should see a significant improvement:

```text
Base64 Benchmark  (10 MB, 5 iterations)

Scalar        Encode:    320.0 MB/s   Decode:    290.0 MB/s
NEON          Encode:   1350.0 MB/s   Decode:   1180.0 MB/s

Speedup (NEON / Scalar):  Encode 4.22x   Decode 4.07x

All correctness checks passed.
```

> **Note:** Exact numbers will vary depending on your Graviton generation and instance type. The important result is a **3–5× speedup** over scalar, confirming that the ported NEON code is using the vector hardware effectively.

| Variant | Encode (MB/s) | Decode (MB/s) | Speedup vs Scalar |
|---------|---------------|---------------|--------------------|
| Scalar  | ~320          | ~290          | 1.0×               |
| NEON    | ~1350         | ~1180         | ~4×                |

The NEON-ported version recovers the same class of speedup that the x86 SSE version achieves on Intel/AMD hardware, demonstrating that the AI-assisted port preserved the algorithmic efficiency of the original SIMD implementation.

You can adjust the benchmark parameters:

```bash
# Larger buffer (50 MB), more iterations
./build/base64_bench --size 50 --iters 10
```

---

## Summary

This tutorial demonstrated how to use **Codex** with the **Arm MCP Server** to port x86 SIMD code to Arm NEON:

- **migrate-ease** identified all x86-specific intrinsics in the codebase before any manual analysis was needed.
- **Codex** used the Arm MCP Server's knowledge base to find correct NEON equivalents for 17 distinct SSE intrinsics, including complex operations like `_mm_mulhi_epu16` and `_mm_maddubs_epi16` that have no single-instruction NEON mapping.
- The ported code **passed all correctness tests** (byte-identical output vs. the scalar reference) and achieved a **~4× speedup** over the scalar baseline.
- The entire porting process — from scanning to working NEON code — was completed in minutes rather than the hours or days that manual porting would require.

### Key takeaways

1. **Isolate platform-specific code.** Having x86 intrinsics in a separate file (`base64_x86.cpp`) with a clean API made the porting task well-defined and verifiable.
2. **Scan before you port.** `migrate-ease` gives you an inventory of everything that needs attention, so you don't discover missing translations at compile time.
3. **AI agents work best with domain tools.** Codex alone can attempt a port based on its training data, but giving it access to the Arm MCP Server's authoritative knowledge base and migration scanner produces more accurate and complete results.
4. **Always verify.** Round-trip correctness tests and performance benchmarks are essential after any automated port. The AI gets you to a working draft quickly, but the final quality gate is still measurement.
