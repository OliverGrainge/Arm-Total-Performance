# Tutorial 2: Memory Access Recipe - Source-Guided Optimization with `__restrict__`

The goal of this tutorial is to show how **Arm Total Performance (ATP)** is used to optimize a **memory-bound** C++ workload on AWS Graviton, not by guesswork, but by mapping profiler evidence directly to source code.

The method in this tutorial is straightforward: run ATP **Memory Access** to identify where memory latency is coming from and which source line is hot, open ATP **Source Code Inspector** on that line to inspect the generated instruction/access pattern, apply a source-level optimization (`__restrict__`) to remove aliasing uncertainty, and then re-run the same recipe to verify that the same line maps to a more efficient vectorized memory-access pattern.

The workload is a STREAM-style triad:

```cpp
out[i] = a[i] + alpha * b[i]
```

You will run one complete performance engineering loop in this tutorial: baseline profile, source-level diagnosis, optimization, and re-profile verification.


**Prerequisites:** AWS Graviton 2/3 instance, GCC 9+ or Clang 14+, CMake 3.16+, ATP installed/configured.

---

## Background: Why Memory Access + Source Inspector

The Memory Access recipe uses Arm SPE sampling to show where loads/stores are serviced (L1/L2/LLC/DRAM) and where latency accumulates. For memory-bound code, this gives faster diagnosis than guessing from wall-clock time alone.

Source Code Inspector closes the loop by showing what instruction pattern is attached to the hot source line. In this tutorial, that line is the triad update statement. You will use ATP to confirm the transition from scalar memory operations to vectorized memory operations after adding `__restrict__`.

---

## 1. Build the code

```bash
cd tutorial_2
mkdir -p build && cd build
cmake ..
make -j4
```

This builds two binaries:

- `triad_baseline` (`-O2 -g -fno-tree-vectorize`)
- `triad_restrict` (`-O2 -g -ftree-vectorize`)

Run both once to confirm correctness:

```bash
./triad_baseline
./triad_restrict
```

Checksums must match.

---

## 2. Profile baseline with Memory Access

Run baseline once in ATP to establish the memory behavior.

### Step 1: Configure and run the recipe

In ATP:

1. Open **Recipes**.
2. Select **Memory Access**.
3. Set workload to launch `triad_baseline`.
4. Confirm recipe status is ready.
5. Click **Run Recipe**.

Screenshot to capture:

<img src="assets/t2_baseline_recipe_setup.png" width="850" alt="Memory Access recipe setup for triad_baseline"/>

### Step 2: Identify the hot function/line

Open the Functions (or Call Stack) view and select `triad`. Then open source:

- Double-click function, or
- Right-click -> **View Source Code**.

If ATP asks, set the repository root directory.

Screenshot to capture:

<img src="assets/t2_baseline_functions.png" width="850" alt="Functions view showing triad hotspot in baseline"/>

### Step 3: Inspect baseline source mapping

Focus on the hot source line:

```cpp
out[i] = a[i] + alpha * b[i];
```

Baseline expectation:

- Scalar load/store pattern (`ldr s` / `str s` style mapping on AArch64).
- Higher instruction density per element.
- Memory-access-heavy behavior on that one source line.

Screenshot to capture:

<img src="assets/t2_baseline_source.png" width="850" alt="Source Code Inspector baseline triad line"/>

Diagnosis at this point: the hot memory behavior is concentrated on the triad line, and the baseline emits scalar per-element work.

---

## 3. Apply source optimization: `__restrict__`

Now inspect the optimized variant in `src/triad_restrict.cpp`:

```cpp
static void triad(float* __restrict__ out,
                  const float* __restrict__ a,
                  const float* __restrict__ b,
                  float alpha, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + alpha * b[i];
}
```

`__restrict__` tells the compiler these pointers do not alias. That enables safe auto-vectorization of the same source loop.

---

## 4. Re-profile restrict variant

Run the same Memory Access workflow on `triad_restrict`.

### Step 1: Run Memory Access on `triad_restrict`

Screenshot to capture:

<img src="assets/t2_restrict_recipe_setup.png" width="850" alt="Memory Access recipe setup for triad_restrict"/>

### Step 2: Inspect the same source line

Open `triad` in Source Code Inspector and compare the exact same line:

```cpp
out[i] = a[i] + alpha * b[i];
```

What to verify:

- Vector instruction mapping (`ldr q` / `str q`, `fmla v*.4s` style).
- Fewer instructions per processed element.
- Improved throughput and reduced runtime.

Screenshot to capture:

<img src="assets/t2_restrict_source.png" width="850" alt="Source Code Inspector restrict triad line"/>

Optional screenshot (if you open assembly-side details):

<img src="assets/t2_restrict_disassembly.png" width="850" alt="Disassembly evidence for vectorized restrict loop"/>

---

## 5. Compare results and report

Use your run outputs plus ATP observations in this table:

| Variant | Source change | Source Inspector instruction pattern | Time (ms) | Bandwidth (GB/s) |
|---|---|---|---:|---:|
| Baseline | none | scalar (`ldr s` / `str s`) | [fill] | [fill] |
| Restrict | `__restrict__` pointers | vector (`ldr q` / `fmla v*.4s` / `str q`) | [fill] | [fill] |

Expected trend on Graviton 3:

- Baseline around ~31 GB/s
- Restrict around ~58 GB/s
- Roughly 1.8-2.0x throughput improvement

Use this conclusion template:

> Using ATP Memory Access and Source Code Inspector, we mapped a source-level aliasing change (`__restrict__`) to a measurable change in memory instruction behavior on the same hot line. Baseline showed scalar load/store behavior, while restrict enabled vectorized NEON operations and higher effective memory throughput.

---

## 6. Required screenshots checklist

- [ ] `assets/t2_baseline_recipe_setup.png`
- [ ] `assets/t2_baseline_functions.png`
- [ ] `assets/t2_baseline_source.png`
- [ ] `assets/t2_restrict_recipe_setup.png`
- [ ] `assets/t2_restrict_source.png`
- [ ] `assets/t2_restrict_disassembly.png` (optional but recommended)

---

## Notes

- Keep `-g` enabled so ATP can resolve source locations.
- Keep problem size large enough (`N=8M`, `iters=200` defaults) for stable sampling.
- For this tutorial, the key validation is not only faster runtime, but visible mapping from source change -> instruction/access pattern change in ATP.
