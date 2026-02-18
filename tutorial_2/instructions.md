# Tutorial 2: Memory Access Recipe

In this tutorial you will use ATP's **Source Code Inspector** to see how pointer aliasing annotations change the memory access pattern the compiler generates — and how to verify that change at the hardware level.

The workload is a softmax + vector-scaling kernel, the kind of post-attention computation that appears throughout transformer inference. You will work through three versions of the same kernel:

1. **Baseline** — no annotations; the compiler generates conservative scalar code.
2. **`__restrict__`** — aliasing ruled out explicitly; the normalise loop vectorises.
3. **`__builtin_assume_aligned` + loop splitting** — an alternative that achieves the same result without modifying the public API.

The diagnostic tool at the centre of the tutorial is ATP's **Source Code Inspector**: it maps hardware memory events back to the exact source lines that caused them, so you can confirm that your annotation actually changed the instructions the CPU is executing — not just the compiler's output.

**Prerequisites:** AWS Graviton 2/3 instance, GCC 9+ or Clang 14+, CMake 3.16+, ATP installed and configured.

---

## Background: Why Pointer Aliasing Blocks Vectorisation

### The aliasing problem

Consider the normalise+scale loop at the heart of this workload:

```cpp
float inv_sum = 1.0f / sum;
for (int i = 0; i < N; ++i)
    output[i] *= inv_sum * scale[i];
```

The compiler would like to widen this loop — process four floats at once using NEON multiply instructions. But it faces a problem: `output` and `scale` are both `float*`. Without more information, the compiler must assume they *could* point into overlapping memory regions.

If `output` and `scale` overlap, then a write to `output[i]` could change the value the next iteration reads from `scale[i+1]`. That dependency makes out-of-order execution and vectorisation unsafe unless the compiler can prove no overlap exists.

```
Worst-case the compiler must assume:
  output[0] written → could corrupt scale[1]
  output[1] written → could corrupt scale[2]
  ...
```

The compiler's response is to keep loads and stores scalar and in order.

### What `__restrict__` does

`__restrict__` (a GCC/Clang extension that models the C99 `restrict` keyword) is a contract: *"I, the programmer, guarantee these pointers do not alias."* Once the compiler has that guarantee it generates a straight vectorised loop:

```
Without __restrict__          With __restrict__
───────────────────────       ────────────────────────
LDR  s0, [x1]   ; scale[i]   LD1  { v0.4S }, [x1]  ; scale[i..i+3]
LDR  s1, [x0]   ; output[i]  LD1  { v1.4S }, [x0]  ; output[i..i+3]
FMUL s1, s1, s0              FMUL v1.4S, v1.4S, v0.4S
STR  s1, [x0]                ST1  { v1.4S }, [x0]
```

One instruction now processes four elements instead of one.

### ATP's Source Code Inspector

The Source Code Inspector is ATP's way of connecting hardware counters to specific lines of C++ source. After profiling, you open it from the **Functions** tab:

1. Select a hot function from the functions list.
2. Click the **Source** button (or **Source/Disassembly** toggle) at the top of the panel.
3. ATP annotates each source line with the hardware counter samples that hit instructions generated from that line.

For this tutorial you will use it to see:
- Which source lines generate memory traffic (load/store counts per line).
- Whether the instructions on the hot normalise line are scalar (`LDR s0`) or vector (`LD1 { v0.4S }`).
- How the access pattern changes between the three variants.

### ATP's recipes — which to use here

| Recipe | What it answers | When to use |
|---|---|---|
| **CPU Cycle Hotspots** | Where is time spent? | Start here — find the hot function |
| **Memory Access** | Which source lines generate which memory events? | Primary recipe for this tutorial |
| **Topdown** | Why is the function slow (category)? | Optional confirmation |

The **Memory Access** recipe uses Arm's Statistical Profiling Extension (SPE) to record load/store addresses alongside PC addresses. This is what feeds the Source Code Inspector with per-line memory data.

---

## 1. Build the Code

```bash
cd tutorial_2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces three executables: `softmax_baseline`, `softmax_restrict`, and `softmax_aligned`. All three compute the same result (`softmax(input) * scale`) over 4 M floats, repeated 100 times, so ATP gets enough samples.

Run each to confirm correctness before profiling — the `Check:` line should match across all three:

```bash
./softmax_baseline
./softmax_restrict
./softmax_aligned
```

---

## 2. Profile the Baseline — Learning the Source Code Inspector

### Step 1 — Run the Memory Access recipe

Open ATP and select **Recipes → Memory Access**. Choose `softmax_baseline` as the target:

<img src="assets/memory_access_recipe.png" width="850" alt="Selecting the Memory Access recipe in ATP"/>

ATP will launch the application and collect SPE (Statistical Profiling Extension) data for approximately 30 seconds while the softmax loop runs.

### Step 2 — Find the hot function

When the recipe completes, open the **Functions** tab. You will see a list of functions ranked by sample count. The top entry should be `softmax_scale` (or its inlined helper `pass_max` / the anonymous lambda from `std::expf`).

<img src="assets/baseline_functions.png" width="850" alt="Functions tab showing softmax_scale as the hot function"/>

### Step 3 — Open the Source Code Inspector

Click on `softmax_scale` to select it, then click the **Source** button in the top toolbar. ATP opens the Source Code Inspector pane showing the C++ source with hardware counter annotations alongside each line.

<img src="assets/baseline_source_inspector.png" width="850" alt="Source Code Inspector for the baseline softmax"/>

Look at the two hot loops:

**The exp pass (Pass 2):**
```cpp
output[i] = std::expf(input[i] - max_val);
```
This line will have the highest sample count. The instructions generated are scalar because `expf` is a library function and does not auto-vectorise at `-O2`.

**The normalise+scale pass (Pass 3):**
```cpp
output[i] *= inv_sum * scale[i];
```
Look at the disassembly ATP shows for this line. You will see scalar 32-bit load/store instructions:

```asm
ldr   s1, [x20, x8, lsl #2]   ; load scale[i]  — 32-bit scalar load
ldr   s0, [x19, x8, lsl #2]   ; load output[i] — 32-bit scalar load
fmul  s0, s0, s2               ; scalar multiply
str   s0, [x19, x8, lsl #2]   ; store output[i]
```

The `s` register prefix (`s0`, `s1`) is the key: this is the 32-bit lane of a NEON register, not a full 128-bit vector. The compiler is processing one float at a time.

**Diagnosis:** The normalise loop is scalar because the compiler cannot prove `output[]` and `scale[]` are disjoint. We have left 4× throughput on the table.

---

## 3. Fix 1: `__restrict__` — Tell the Compiler Pointers Don't Alias

The fix in `src/softmax_restrict.cpp` is a single keyword on each pointer parameter:

```cpp
void softmax_scale(float* __restrict__ output,
                   const float* __restrict__ input,
                   const float* __restrict__ scale, int N) {
```

No algorithmic change. No data movement. The compiler now knows it is safe to widen the inner loop.

### Re-profile: did vectorisation happen?

Run the Memory Access recipe on `softmax_restrict`. Open the Source Code Inspector for `softmax_scale` and navigate to the normalise loop.

<img src="assets/restrict_source_inspector.png" width="850" alt="Source Code Inspector for the restrict softmax"/>

The disassembly for `output[i] *= inv_sum * scale[i]` should now show 128-bit NEON instructions:

```asm
ld1   { v1.4s }, [x20]    ; load scale[i..i+3]  — 128-bit vector load
ld1   { v0.4s }, [x19]    ; load output[i..i+3] — 128-bit vector load
fmul  v0.4s, v0.4s, v1.4s ; vector multiply (4 floats at once)
st1   { v0.4s }, [x19]    ; store output[i..i+3]
```

The `v` register prefix and `.4s` lane specifier confirm that four floats are processed per instruction. The load/store count per source line in the Source Code Inspector will also drop by 4× — one hardware event per four elements.

Compare the bandwidth figures reported by each executable. The normalise pass improvement may be partially masked by the (still-scalar) `expf` loop, but the source inspector makes the vectorisation visible regardless of overall runtime.

---

## 4. Fix 2: `__builtin_assume_aligned` + Loop Splitting

`__restrict__` is effective but changes the public API — callers who pass overlapping buffers (even legitimately) now have undefined behaviour. A second approach avoids modifying the public signature.

The aligned variant in `src/softmax_aligned.cpp` uses two techniques together:

**Technique 1: Loop splitting.** The softmax is decomposed into three separate helper functions (`pass_max`, `pass_exp`, `pass_normalise`). Each function has at most one writable pointer, which limits the aliasing surface the compiler must consider within each function scope.

**Technique 2: `__builtin_assume_aligned`.** Each helper marks its pointer parameters as 64-byte aligned (Graviton's cache-line size):

```cpp
output = static_cast<float*>(__builtin_assume_aligned(output, 64));
scale  = static_cast<const float*>(__builtin_assume_aligned(scale, 64));
```

This gives the compiler two additional guarantees:
- It can emit the vectorised loop starting from byte 0 — no scalar prologue to align the pointer.
- The access pattern will always begin at a cache-line boundary, so the hardware prefetcher can track it accurately.

The public API is unchanged:

```cpp
void softmax_scale(float* output, const float* input,
                   const float* scale, int N)  // no __restrict__
```

### Re-profile: aligned loads visible in the inspector

Run the Memory Access recipe on `softmax_aligned`. In the Source Code Inspector, navigate to `pass_normalise` and look at the disassembly for:

```cpp
output[i] = tmp[i] * inv_sum * scale[i];
```

<img src="assets/aligned_source_inspector.png" width="850" alt="Source Code Inspector for the aligned+split softmax"/>

You should see 128-bit NEON loads as in the restrict variant, but now the load addresses in the SPE data will all be 64-byte aligned. ATP's memory access view may show the access pattern as a uniform stride with no alignment penalty events.

---

## 5. The Full Picture

Run all three and compare the output:

```bash
./softmax_baseline
./softmax_restrict
./softmax_aligned
```

Here is what ATP shows across the three variants:

| Variant | Normalise loop instruction | Effective elements/instruction | Alignment scalar pre-loop |
|---|---|---|---|
| **Baseline** | `ldr s0` (scalar 32-bit) | 1 | n/a |
| **Restrict** | `ld1 { v0.4s }` (NEON 128-bit) | 4 | yes (unaligned start possible) |
| **Aligned+split** | `ld1 { v0.4s }` (NEON 128-bit) | 4 | no (64-byte start guaranteed) |

And the key diagnostics at each step:

| Step | Source Code Inspector showed... | So we applied... |
|---|---|---|
| 1 | Scalar `ldr s0` on the normalise line; the compiler serialised loads due to aliasing | `__restrict__` — gave the compiler aliasing proof |
| 2 | NEON `ld1 { v0.4s }` on the normalise line; vectorised | `__builtin_assume_aligned` + split loops — same result, public API unchanged |

### Why the `expf` loop stays scalar

Both the restrict and aligned variants leave the `expf` loop scalar:

```cpp
output[i] = std::expf(input[i] - max_val);
```

`std::expf` is a C library function. At `-O2` the compiler cannot replace it with a NEON polynomial approximation without `-ffast-math` (which relaxes IEEE-754 rounding guarantees). This is expected and intentional — it isolates the aliasing effect on the normalise loop from an unrelated transcendental-function issue.

If you need a vectorised exp, Tutorial 3 shows how Arm Performance Libraries and KleidiAI provide optimised versions.

---

## Key Takeaways

1. **Pointer aliasing is a silent performance blocker.** The compiler cannot vectorise a loop if it cannot prove that a write through one pointer doesn't corrupt a read through another. The source code inspector reveals this with scalar load/store instructions on the hot line.

2. **`__restrict__` is the direct fix.** One keyword on each pointer parameter tells the compiler to generate the vectorised path unconditionally. Confirm the change in ATP — never assume the annotation had the intended effect.

3. **`__builtin_assume_aligned` + loop splitting is a complementary technique.** Splitting a fused kernel into single-responsibility functions reduces the aliasing surface within each function. Alignment hints allow the vectorised loop to start cleanly at a cache-line boundary. The public API stays clean.

4. **Use the Memory Access recipe + Source Code Inspector together.** The Memory Access recipe provides the SPE data; the Source Code Inspector connects that data to source lines and shows the assembly. The combination lets you confirm at the hardware level that your annotation changed the code path the CPU is executing.

5. **The exp loop is a different problem.** Aliasing annotations do not help a loop dominated by library function calls. Recognise the two bottleneck types and apply the right tool to each: annotations for aliasing, optimised libraries (Tutorial 3) for transcendental functions.
