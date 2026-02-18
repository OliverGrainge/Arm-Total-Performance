# Tutorial 2: Memory Access Recipe with Source Code Inspector

This tutorial demonstrates a clean memory-access optimization flow using a simple **STREAM-style triad** kernel:

```cpp
out[i] = a[i] + alpha * b[i]
```

You will profile three versions of the same operation:

1. **Baseline**: no pointer annotations (scalar baseline for comparison).
2. **`__restrict__`**: explicit no-alias contract to unlock vectorization.
3. **Aligned API-preserving variant**: internal `__builtin_assume_aligned(..., 64)` plus internal split implementation, while keeping the public function signature unchanged.

This is intentionally not ML-specific. It is memory-bound and gives very clear signals in ATP's **Memory Access** recipe and **Source Code Inspector**.

## Why this workload is better for Memory Access

- No transcendental math (`exp`, `log`, etc.) to hide memory effects.
- One hot source line maps directly to load/store behavior.
- Source changes (`__restrict__`, alignment assumptions) are easy to correlate with instruction changes in disassembly.

## Prerequisites

- AWS Graviton 2/3 instance
- GCC 9+ or Clang 14+
- CMake 3.16+
- Arm Total Performance installed and configured

---

## 1. Build

```bash
cd tutorial_2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Executables produced:

- `triad_baseline`
- `triad_restrict`
- `triad_aligned`

Run all three once to confirm they produce similar checksums:

```bash
./triad_baseline
./triad_restrict
./triad_aligned
```

---

## 2. Profile the Baseline

### Step 1: Run Memory Access recipe

In ATP, choose **Recipes -> Memory Access** and run it against `triad_baseline`.

### Step 2: Open the hot function

In **Functions** (or **Call Stack**), select the hot function (`triad`), then open source view:

- double-click the function, or
- right-click and select **View Source Code**.

If prompted, click **Specify Root Directory** and point ATP to this repository path.

### Step 3: Inspect hot source line and disassembly

Focus on:

```cpp
out[i] = a[i] + alpha * b[i];
```

In baseline, you should see scalar-style instructions associated with this line (single-element loads/stores) because vectorization is intentionally disabled for this binary.

---

## 3. Fix 1: `__restrict__`

Open `src/triad_restrict.cpp`. The only semantic change is pointer contracts:

```cpp
void triad(float* __restrict__ out,
           const float* __restrict__ a,
           const float* __restrict__ b,
           float alpha, int n)
```

Re-profile `triad_restrict` with Memory Access and inspect the same source line. You should now observe vector instructions (`v` registers, e.g. `ld1/st1/fmla` patterns), indicating widened memory operations.

What to confirm in Source Code Inspector:

- Same hot source line
- Lower instruction count pressure per element
- Vector load/store behavior replacing scalar access behavior

---

## 4. Fix 2: Preserve Public API + Assume Alignment

Open `src/triad_aligned.cpp`.

This variant keeps the public API unchanged and pushes optimization assumptions into an internal implementation:

- internal `triad_impl` uses `__restrict__`
- pointers are marked with `__builtin_assume_aligned(ptr, 64)`
- arrays are allocated with 64-byte alignment

Re-profile `triad_aligned` and inspect the same line in the inspector. You should again see vectorized memory ops, now with explicit alignment guarantees that can reduce alignment handling overhead.

---

## 5. What to Report in Tutorial 2

For your brief, report this exact mapping:

1. **Source change**: add `__restrict__` (and then alignment assumptions).
2. **Compiler output change**: scalar memory ops -> vector memory ops.
3. **Hardware/access view change**: Memory Access + Source Code Inspector show improved access pattern on the same source line.

Suggested evidence table:

| Variant | Source change | Inspector/disassembly signal | Runtime/BW trend |
|---|---|---|---|
| Baseline | none | scalar load/store pattern on hot line | lowest |
| Restrict | `__restrict__` pointers | vector load/store + fused multiply-add style pattern | improved |
| Aligned | internal restrict + 64B alignment assumptions | vector pattern maintained, cleaner aligned access behavior | similar or best |

---

## 6. Notes

- Baseline is intentionally compiled with vectorization disabled so the before/after difference is visible in ATP.
- Use large enough `N` and iterations (defaults are already tuned) to collect stable SPE samples.
- If Source Code Inspector cannot open files, ensure debug symbols are enabled (`-g`) and host source path matches the profiled binary.

---

## 7. ATP Checkpoints and Screenshot Placeholders

Use this section as a capture checklist while running ATP. Add screenshots under `tutorial_2/assets/` and paste short observations beneath each item.

### A. Baseline (`triad_baseline`)

- Checkpoint: Memory Access recipe configured for `triad_baseline`.
- Screenshot file: `assets/t2_baseline_recipe_setup.png`
- Note: confirm target path and recipe options.

- Checkpoint: Functions view with `triad` as hot function.
- Screenshot file: `assets/t2_baseline_functions.png`
- Note: record sample count / contribution for `triad`.

- Checkpoint: Source Code Inspector on `out[i] = a[i] + alpha * b[i];`.
- Screenshot file: `assets/t2_baseline_source.png`
- Note: capture scalar-style load/store pattern.

### B. Restrict (`triad_restrict`)

- Checkpoint: Memory Access recipe configured for `triad_restrict`.
- Screenshot file: `assets/t2_restrict_recipe_setup.png`
- Note: same run parameters as baseline.

- Checkpoint: Source Code Inspector on same source line.
- Screenshot file: `assets/t2_restrict_source.png`
- Note: capture vector instruction evidence (`v` registers / `ld1` / `st1` / `fmla` style pattern).

- Checkpoint: Disassembly snippet for hot line.
- Screenshot file: `assets/t2_restrict_disassembly.png`
- Note: compare against baseline scalar pattern.

### C. Aligned (`triad_aligned`)

- Checkpoint: Memory Access recipe configured for `triad_aligned`.
- Screenshot file: `assets/t2_aligned_recipe_setup.png`
- Note: same run parameters as baseline/restrict.

- Checkpoint: Source Code Inspector on same source line.
- Screenshot file: `assets/t2_aligned_source.png`
- Note: confirm vector pattern is maintained.

- Checkpoint: Memory access/address behavior view (if available in your ATP build).
- Screenshot file: `assets/t2_aligned_memory_view.png`
- Note: record any aligned/regular stride characteristics visible in UI.

### D. Final Comparison Table (fill in)

| Variant | Hot line instruction pattern | Key inspector observation | Time (ms) | Bandwidth (GB/s) |
|---|---|---|---:|---:|
| Baseline | [fill] | [fill] | [fill] | [fill] |
| Restrict | [fill] | [fill] | [fill] | [fill] |
| Aligned | [fill] | [fill] | [fill] | [fill] |

### E. Brief-ready conclusion template

Use this wording template in your write-up:

`Using ATP Memory Access with Source Code Inspector, we mapped a source-level aliasing/alignment change to a measurable change in memory instruction behavior on the same hot line. The baseline showed [scalar pattern], adding __restrict__ changed this to [vector pattern], and adding alignment assumptions preserved/improved this behavior with [alignment observation]. This correlated with runtime/bandwidth moving from [X] to [Y] to [Z].`
