# Tutorial 2: Memory Access + CPU Cycle Hotspots — Array-of-Structures vs Structure-of-Arrays

The goal of this tutorial is to show how **Arm Total Performance (ATP)** is used to diagnose and fix a **data layout** problem in a C++ workload on **AWS (Amazon Web Services) Graviton**, by mapping profiler evidence directly to source code.

The method is straightforward: run ATP **Memory Access** to identify where memory latency is coming from, then run ATP **CPU (central processing unit) Cycle Hotspots** to map hotspots to source lines with **View Source Code**. Apply a source-level restructuring (AoS → SoA) to eliminate cache line waste, and re-run both recipes to verify the improvement. You will run one complete performance engineering loop: baseline profile → source-level diagnosis → data layout fix → re-profile verification.

The workload is a particle physics position update:

```cpp
p[i].x += p[i].vx * dt;
p[i].y += p[i].vy * dt;
p[i].z += p[i].vz * dt;
```

run over 1,048,576 particles for 200 iterations. The same algorithm is implemented twice — once with an Array-of-Structures layout (`aos_baseline`) and once with a Structure-of-Arrays layout (`soa_optimized`). The data layout is the only variable.

**Prerequisites:** AWS Graviton 2/3 instance, GCC 9+ or Clang 14+, CMake 3.16+, ATP installed and configured.

## Terms used in this tutorial

- **CPU**: central processing unit.
- **AoS**: Array of Structures — one struct per particle, all fields contiguous in memory.
- **SoA**: Structure of Arrays — one array per field, all particles contiguous per field.
- **Cache line**: the minimum unit of data transferred between memory levels (64 bytes on Arm).
- **Cache line utilisation**: the fraction of each loaded cache line that contains useful data for the current computation.
- **SPE**: Arm Statistical Profiling Extension, the hardware sampling mechanism used by the Memory Access recipe.
- **Periodic Samples**: ATP's sampled execution counts shown in CPU Cycle Hotspots tables.

---

## Background: Cache line utilisation and why data layout matters

Every load from DRAM, L3, or L2 transfers a full **64-byte cache line**, regardless of how many bytes you actually use. Cache line utilisation measures how much of each transferred line was useful.

**AoS layout — low utilisation:**

```
ParticleAoS: [x y z vx vy vz | mass charge temp | pressure energy density | spin_x spin_y spin_z pad]
             |<--- 24 bytes used --->|<------------------ 40 bytes wasted ------------------>|
             |<-------------------------------- 64 bytes loaded -------------------------------->|
```

Each particle occupies exactly one cache line. The `update_positions` loop uses only `x`, `y`, `z`, `vx`, `vy`, `vz` — 24 bytes out of 64 loaded. The other 40 bytes (`mass`, `charge`, `temperature`, `pressure`, `energy`, `density`, `spin_*`) are evicted unused. Cache line utilisation: **37.5%**.

With 1,048,576 particles at 64 bytes each, the working set is **64 MB** — larger than the Graviton3 L3 cache (32 MB). Every iteration must pull most data from DRAM or LLC. The wasted bandwidth is also felt in the L1 cache: with 40 of every 64 loaded bytes being dead weight, useful data is displaced faster than necessary, lowering the L1C hit rate. ATP Memory Access will show a depressed L1C % Loads and elevated average access latency.

**SoA layout — full utilisation:**

```
particles.x:  [x0 x1 x2 x3 ... x15]   ← 16 floats per cache line, all used
particles.y:  [y0 y1 y2 y3 ... y15]   ← same
particles.z:  [z0 z1 z2 z3 ... z15]   ← same
particles.vx: [vx0 vx1 ... vx15]      ← same
...
```

Only the six arrays accessed by `update_positions` are ever loaded. Hot working set = 6 × 4 MB = **24 MB**. Every byte of every loaded cache line contains useful position or velocity data. Cache line utilisation: **100%**. The L1C hit rate rises and average latency falls — exactly what ATP Memory Access measures.

---

## 1. Build the code

```bash
cd tutorial_2
mkdir -p build && cd build
cmake ..
make -j4
```

This builds two binaries, both compiled with `-O2 -g`:

- `aos_baseline` — Array-of-Structures layout, 64-byte struct, 37.5% cache line utilisation in the hot loop
- `soa_optimized` — Structure-of-Arrays layout, 100% cache line utilisation in the hot loop

Run both to confirm the checksums match:

```bash
./aos_baseline
./soa_optimized
```

Both lines must print the same floating-point value (the label differs; the number must be identical).

---

## 2. Profile the AoS baseline with Memory Access

### Step 1: Configure and run the recipe

In ATP:

1. Open **Recipes**.
2. Select **Memory Access**.
3. Set the workload to launch `aos_baseline`.
4. Click **Run Recipe**.

![Memory Access recipe configured for aos_baseline](./assets/aos_recipe_setup.png)
*Memory Access recipe ready to run against `aos_baseline`.*

### Step 2: Read the Latency Breakdown table

After the run completes, open the **Latency Breakdown** tab. The table will look similar to this:

| Function | #SPE Samples | L1C % Loads | L1C Avg Latency | L1C Contrib (cyc) | L1C Contrib (%) | L2C % Loads | L2C Avg Latency | L2C Contrib (cyc) | L2C Contrib (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 63,990 | 68.12 | 28.89 | 19.68 | 97.83 | 1.01 | 43.06 | 0.44 | 2.17 |

> **Note:** `update_positions` does not appear as a separate row because the compiler inlines it into `main`. All samples are attributed to `main`.

The key signals in this profile:

- **L1C % Loads = 68.12%** — only 68 out of every 100 sampled loads resolved in L1 cache. The remaining ~32% missed L1C and went to L2C or further. This is the direct cost of the AoS layout: every particle access loads 64 bytes but only 24 bytes are used, so useful data is evicted from L1C sooner than necessary.
- **L1C Avg Latency = 28.89 cycles** — the average latency for loads attributed to L1C is elevated because cache pressure from the 64 MB working set causes frequent L1C evictions and refills.
- **L2C Avg Latency = 43.06 cycles** — the 1.01% of loads that miss L1C pay over 40 cycles on average.
- **L2C Contrib % = 2.17%** — despite being only 1% of loads, L2C accesses account for over 2% of total memory latency.

The profile is telling a clear story: the AoS loop loads far more data than it uses, reducing L1C effectiveness and raising average memory access latency across the board.

![AoS Latency Breakdown showing 63,990 SPE samples and 68.12% L1C hit rate](./assets/aos_latency_breakdown.png)
*AoS Latency Breakdown: 63,990 SPE samples on `main`, L1C % = 68.12%, L1C Avg Latency = 28.89 cycles, L2C Avg Latency = 43.06 cycles. Approximately 32% of loads miss L1C — a direct consequence of 37.5% cache line utilisation.*

### Step 3: Capture AoS memory baseline

Record the following values for later comparison:

- `#SPE Samples (main): 63,990`
- `L1C % Loads: 68.12%`
- `L1C Avg Latency: 28.89 cyc`
- `L2C % Loads: 1.01%`

---

## 3. Map the hotspot to source with CPU Cycle Hotspots (AoS)

### Step 1: Run CPU Cycle Hotspots on `aos_baseline`

In ATP:

1. Open **Recipes**.
2. Select **CPU Cycle Hotspots**.
3. Set the workload to launch `aos_baseline`.
4. Click **Run Recipe**.

![CPU Cycle Hotspots recipe configured for aos_baseline](./assets/aos_hotspots_setup.png)
*CPU Cycle Hotspots recipe ready to run against `aos_baseline`.*

### Step 2: Open source code

In the **Functions** table, locate `main`, then double-click it (or right-click → **View Source Code**).
If prompted, click **Specify Root Directory** and point to your local source tree.

The source view shows `aos_baseline.cpp`. Navigate to the `update_positions` function body (lines 19–22). The periodic sample counts on the three update lines are:

```
Line 19:    208  for (int i = 0; i < n; ++i) {
Line 20:  3,782      p[i].x += p[i].vx * dt;
Line 21:    268      p[i].y += p[i].vy * dt;
Line 22:    803      p[i].z += p[i].vz * dt;
```

Total on loop body: **5,061 periodic samples**. The `x` update line (line 20) dominates because the `x` field is at offset 0 in the struct — it is the first access per struct, which triggers the full cache line load.

![Source Code Inspector for aos_baseline showing periodic samples on the update loop lines](./assets/aos_source.png)
*Source Code Inspector (`aos_baseline.cpp`): the three position-update lines carry 3,782 + 268 + 803 = 4,853 samples, plus 208 on the loop control — 5,061 total. The `x` update dominates as the first field access per struct, which triggers the cache line fetch.*

---

## 4. The fix — restructure from AoS to SoA

Open `src/soa_optimized.cpp`. The algorithm is unchanged; only the data layout is different.

**AoS (baseline):** one struct per particle, all fields interleaved:

```cpp
struct ParticleAoS {
    float x, y, z;                   // used in hot loop
    float vx, vy, vz;                // used in hot loop
    float mass, charge, temperature; // not used — but loaded anyway
    float pressure, energy, density; // not used — but loaded anyway
    float spin_x, spin_y, spin_z;    // not used — but loaded anyway
    float pad;
};
std::vector<ParticleAoS> particles(N); // 64 MB working set
```

**SoA (optimised):** one array per field, access to each field is stride-1:

```cpp
struct ParticlesSoA {
    std::vector<float> x, y, z;       // used in hot loop — 3 × 4 MB
    std::vector<float> vx, vy, vz;    // used in hot loop — 3 × 4 MB
    std::vector<float> mass, charge, temperature; // separate, never loaded
    std::vector<float> pressure, energy, density; // separate, never loaded
    std::vector<float> spin_x, spin_y, spin_z;    // separate, never loaded
};
```

Why this fixes the problem:

1. **Only the hot arrays are loaded.** `update_positions` touches `x`, `y`, `z`, `vx`, `vy`, `vz`. In SoA these are six contiguous arrays totalling 24 MB.
2. **100% cache line utilisation.** Each 64-byte cache line of `particles.x` contains 16 consecutive `x` values, all of which are consumed before the line is evicted.
3. **Lower average memory latency.** With a smaller hot working set and full utilisation of every loaded byte, the L1C hit rate rises dramatically and average latency per load falls.

No compiler hints, vectorisation pragmas, or loop reordering are needed. The improvement comes entirely from restructuring the data.

---

## 5. Re-profile the SoA variant with Memory Access

Run the same Memory Access recipe with `soa_optimized` as the workload.

![Memory Access recipe configured for soa_optimized](./assets/soa_recipe_setup.png)
*Memory Access recipe ready to run against `soa_optimized`.*

### Step 1: Read the Latency Breakdown table

| Function | #SPE Samples | L1C % Loads | L1C Avg Latency | L1C Contrib (cyc) | L1C Contrib (%) | Potential Improvement (cyc) |
|---|---:|---:|---:|---:|---:|---:|
| main | 63,602 | 99.98 | 10.76 | 10.76 | 100 | 190,052 |

Compare each column against the AoS result:

- **L1C % Loads = 99.98%** — virtually every load now resolves in L1 cache, up from 68.12% in AoS. The SoA hot working set (24 MB) fits in L3 and is small enough for the hardware prefetcher to keep L1C populated continuously.
- **L1C Avg Latency = 10.76 cycles** — down from 28.89 cycles in AoS. With true L1C hits and no eviction pressure from unused data, the average load latency is nearly halved.
- **L1C Contrib % = 100%** — all measured memory latency now comes from L1C; L2C and DRAM are negligible.
- **Potential Improvement = 190,052 cycles** — ATP estimates the remaining optimisable memory latency. This number is meaningful for planning further work; it would be higher for AoS (visible if you scroll the Potential Improvement column into view in the AoS run).

![SoA Latency Breakdown showing 99.98% L1C hit rate and Potential Improvement of 190,052 cycles](./assets/soa_latency_breakdown.png)
*SoA Latency Breakdown: 63,602 SPE samples on `main`, L1C % = 99.98%, L1C Avg Latency = 10.76 cycles (down from 28.89), all latency contribution from L1C. The SoA layout has eliminated the L2C/LLC pressure seen in the AoS profile.*

### Step 2: Capture SoA memory numbers

- `#SPE Samples (main): 63,602`
- `L1C % Loads: 99.98%`
- `L1C Avg Latency: 10.76 cyc`
- `Potential Improvement: 190,052 cyc`

---

## 6. Re-map hotspot to source with CPU Cycle Hotspots (SoA)

Run **CPU Cycle Hotspots** with `soa_optimized`, open **View Source Code** for `main`, and navigate to the `update_positions` function body (lines 19–22 of `soa_optimized.cpp`):

```
Line 19:  210  for (int i = 0; i < n; ++i) {
Line 20:  756      p.x[i] += p.vx[i] * dt;
Line 21:  807      p.y[i] += p.vy[i] * dt;
Line 22:  927      p.z[i] += p.vz[i] * dt;
```

Total on loop body: **2,700 periodic samples**, down from 5,061 in `aos_baseline` — a reduction of approximately **47%** on the same source lines.

Note that the distribution across the three update lines is now more even (756, 807, 927) compared to the heavily skewed AoS distribution (3,782, 268, 803). In AoS, the `x` line dominated because it triggers the per-struct cache line fetch, carrying the latency for all subsequent fields. In SoA each array is independent, so the per-access cost is spread uniformly.

![Source Code Inspector for soa_optimized showing fewer periodic samples on the update loop lines](./assets/soa_source.png)
*Source Code Inspector (`soa_optimized.cpp`): the same update loop lines now carry 756 + 807 + 927 = 2,490 samples plus 210 on the loop control — 2,700 total, versus 5,061 in `aos_baseline`. The distribution is also more even: no single line dominates because each array access is independent.*

![CPU Cycle Hotspots recipe configured for soa_optimized](./assets/soa_hotspots_setup.png)
*CPU Cycle Hotspots recipe ready to run against `soa_optimized`.*

---

## 7. Compare results and interpret

| Metric | `aos_baseline` | `soa_optimized` | Change |
|---|---:|---:|---|
| **L1C % Loads** | 68.12% | 99.98% | +31.86 pp — nearly all loads now in L1C |
| **L1C Avg Latency** | 28.89 cyc | 10.76 cyc | −63% — latency per load nearly halved |
| **L2C % Loads** | 1.01% | ~0% | L2C pressure eliminated |
| **#SPE Samples (main)** | 63,990 | 63,602 | Similar — both ran to completion |
| **Hot loop periodic samples** | 5,061 | 2,700 | −47% on same source lines |
| **Potential Improvement** | (scroll to view) | 190,052 cyc | Lower = less remaining latency |

### Key interpretation

The SPE sample counts are similar between the two runs (63,990 vs 63,602). This is expected when both binaries run to completion and ATP collects samples throughout: the sample count reflects the runtime of the entire process, not just the hot loop. The primary signals of improvement are in the **cache tier breakdown** and the **CPU Cycle Hotspots source view**.

**L1C % Loads is the clearest memory-tier signal.** In AoS, 31.88% of loads missed L1C — every one of those misses paid 43+ cycle penalties in L2C (and further for LLC/DRAM on Graviton3 where the 64 MB working set exceeds L3). In SoA, 0.02% of loads miss L1C. The mechanism is exactly what the background section predicted: SoA eliminates the 40 bytes of dead weight per cache line, so the hardware prefetcher can keep up with the loop and the hot working set fits comfortably in L3.

**L1C Avg Latency shows the same story differently.** Even loads attributed to L1C took 28.89 cycles on average in AoS, versus 10.76 cycles in SoA. The AoS figure is elevated because cache pressure from the 64 MB working set causes frequent L1C evictions mid-loop, meaning some apparent "L1C" accesses actually stall while waiting for a line to be refetched.

**The source view makes the change attributable.** The same three update lines — the only lines that changed in meaning, not in code — drop from 5,061 to 2,700 periodic samples. The AoS source view reveals an additional pattern: the `x` update line (line 20) accounts for 3,782 of the 5,061 samples because it is the first field accessed per struct, triggering the full 64-byte cache line fetch and carrying the latency for every unused field. In SoA this pathology disappears: all three lines carry similar sample counts because each array access is independent.

> **Note on Graviton vs other hardware.** The screenshots were taken on a development machine with a large L3 cache. On Graviton3 (32 MB L3), the 64 MB AoS working set is larger than L3, so LLC and DRAM columns will show significant traffic in the AoS profile — the improvement will be more pronounced than shown here.

---

> **Conclusion:** ATP Memory Access + CPU Cycle Hotspots source view identified that `aos_baseline`'s data layout loaded 64 bytes per particle while using only 24 — a 37.5% cache line utilisation that manifested as a 68.12% L1C hit rate and 28.89-cycle average L1C latency. Restructuring to SoA raised L1C utilisation to 99.98% and lowered average L1C latency to 10.76 cycles, with no algorithmic change. CPU Cycle Hotspots confirmed the improvement on the exact source lines responsible, showing a 47% reduction in periodic samples on the update loop.

---

## 8. Required screenshots checklist

- [ ] `assets/aos_recipe_setup.png` — Memory Access recipe configured for `aos_baseline`
- [x] `assets/aos_latency_breakdown.png` — Latency Breakdown: 63,990 SPE samples, L1C % = 68.12%, L1C Avg Latency = 28.89 cyc, L2C % = 1.01%
- [ ] `assets/aos_hotspots_setup.png` — CPU Cycle Hotspots recipe configured for `aos_baseline`
- [x] `assets/aos_source.png` — View Source Code: lines 19–22 of `update_positions`, 5,061 total periodic samples, line 20 = 3,782
- [ ] `assets/soa_recipe_setup.png` — Memory Access recipe configured for `soa_optimized`
- [x] `assets/soa_latency_breakdown.png` — Latency Breakdown: 63,602 SPE samples, L1C % = 99.98%, L1C Avg Latency = 10.76 cyc, Potential Improvement = 190,052 cyc
- [ ] `assets/soa_hotspots_setup.png` — CPU Cycle Hotspots recipe configured for `soa_optimized`
- [x] `assets/soa_source.png` — View Source Code: same lines 19–22, 2,700 total periodic samples, even distribution across x/y/z

---

## Notes

- Keep `-g` enabled so ATP can resolve source locations to line numbers.
- Keep the problem size at `N = 1 << 20` and `iters = 200` for stable SPE sampling; smaller sizes may not produce enough samples for a reliable breakdown.
- Compare like-for-like runs: same Graviton instance, same CPU frequency policy, no other heavy workloads running concurrently.
- The key validation is combined evidence: higher `L1C % Loads` in Memory Access **and** fewer **Periodic Samples on the same source lines** in CPU Cycle Hotspots.
- If ATP does not resolve function names for `update_positions` (shows as `main` only), this is expected — the static function is inlined. Click `main` and navigate to the `update_positions` body in the source view.
- If ATP does not resolve source lines at all (shows `??`), ensure you point the source root to the `tutorial_2/src` directory and verify debug symbols are present (`file aos_baseline` should show `with debug_info`).
- On Graviton3 (32 MB L3), the AoS profile will additionally show significant LLC and DRAM % columns for `main`, because the 64 MB working set exceeds L3. The improvement from SoA will be more pronounced than on machines with larger caches.
