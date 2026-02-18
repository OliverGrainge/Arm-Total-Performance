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

With 1,048,576 particles at 64 bytes each, the working set is **64 MB** — larger than the typical Graviton3 L3 cache (32 MB). Every iteration must pull most data from DRAM or LLC. ATP Memory Access will show significant L2C/LLC/DRAM traffic.

**SoA layout — full utilisation:**

```
particles.x:  [x0 x1 x2 x3 ... x15]   ← 16 floats per cache line, all used
particles.y:  [y0 y1 y2 y3 ... y15]   ← same
particles.z:  [z0 z1 z2 z3 ... z15]   ← same
particles.vx: [vx0 vx1 ... vx15]      ← same
...
```

Only the six arrays accessed by `update_positions` are ever loaded. Hot working set = 6 × 4 MB = **24 MB** — fits in L3 on Graviton3. Every byte of every loaded cache line contains useful position or velocity data. Cache line utilisation: **100%**. ATP Memory Access will show a much higher L1C/L2C hit rate and lower DRAM traffic.

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

![Memory Access recipe configured for aos_baseline](assets/aos_recipe_setup.png)
*Memory Access recipe ready to run against `aos_baseline`.*

### Step 2: Read the Latency Breakdown table

After the run completes, open the **Latency Breakdown** tab. You should see a result similar to this structure (fill in your real numbers):

| Function | #SPE Samples | L1C % Loads | L2C % Loads | LLC % Loads | DRAM % Loads | Potential Improvement (cyc) |
|---|---:|---:|---:|---:|---:|---:|
| update_positions | XXX,XXX | XX% | XX% | XX% | XX% | X,XXX,XXX |

Key signals to look for:

- **High SPE sample count** — the hot loop is spending a large amount of time on memory operations.
- **Significant L2C / LLC / DRAM %** — cache misses are propagating beyond L1 because the 64 MB working set exceeds L3.
- **High Potential Improvement** — ATP estimates substantial cycles are lost to memory latency.

The cache tier distribution reflects what the background section predicted: every particle load brings in a full 64-byte cache line, but only 24 bytes are used. The remaining 40 bytes displace other useful data, causing thrashing in L2 and LLC.

![AoS Latency Breakdown showing high SPE sample count and significant L2/LLC/DRAM traffic](assets/aos_latency_breakdown.png)
*AoS Latency Breakdown: high SPE sample count on `update_positions`, elevated L2C/LLC/DRAM load percentages, high Potential Improvement — all consistent with 37.5% cache line utilisation.*

### Step 3: Capture AoS memory baseline

Record the following values for later comparison:

- `#SPE Samples (update_positions): ___________`
- `L1C % Loads: ___________`
- `DRAM % Loads: ___________`
- `Potential Improvement (cyc): ___________`

---

## 3. Map the hotspot to source with CPU Cycle Hotspots (AoS)

### Step 1: Run CPU Cycle Hotspots on `aos_baseline`

In ATP:

1. Open **Recipes**.
2. Select **CPU Cycle Hotspots**.
3. Set the workload to launch `aos_baseline`.
4. Click **Run Recipe**.

![CPU Cycle Hotspots recipe configured for aos_baseline](assets/aos_hotspots_setup.png)
*CPU Cycle Hotspots recipe ready to run against `aos_baseline`.*

### Step 2: Open source code

In the **Functions** table, locate `update_positions`, then double-click it (or right-click → **View Source Code**).
If prompted, click **Specify Root Directory** and point to your local source tree.

In `aos_baseline.cpp`, the source view will show the three hot lines of the update loop:

```cpp
p[i].x += p[i].vx * dt;
p[i].y += p[i].vy * dt;
p[i].z += p[i].vz * dt;
```

Record the **Periodic Samples** on these lines (example: `XXX,XXX` combined). Also record samples on the call site in `main`:

```cpp
update_positions(particles.data(), N, dt);
```

![Source Code Inspector for aos_baseline showing periodic samples on the update loop lines](assets/aos_source.png)
*Source Code Inspector: `update_positions` body carries high periodic sample counts, confirming this is the memory-bound hot loop.*

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

1. **Only the hot arrays are loaded.** `update_positions` touches `x`, `y`, `z`, `vx`, `vy`, `vz`. In SoA these are six contiguous arrays totalling 24 MB — within L3 on Graviton3.
2. **100% cache line utilisation.** Each 64-byte cache line of `particles.x` contains 16 consecutive `x` values, all of which are consumed before the line is evicted.
3. **Hardware prefetcher effectiveness.** Sequential stride-1 access across a single array is the pattern the hardware prefetcher is optimised for. With AoS the prefetcher must stride 64 bytes per useful 24 bytes; with SoA every fetch is directly used.

No compiler hints, vectorisation pragmas, or loop reordering are needed. The improvement comes entirely from restructuring the data.

---

## 5. Re-profile the SoA variant with Memory Access

Run the same Memory Access recipe with `soa_optimized` as the workload.

![Memory Access recipe configured for soa_optimized](assets/soa_recipe_setup.png)
*Memory Access recipe ready to run against `soa_optimized`.*

### Step 1: Read the Latency Breakdown table

| Function | #SPE Samples | L1C % Loads | L2C % Loads | LLC % Loads | DRAM % Loads | Potential Improvement (cyc) |
|---|---:|---:|---:|---:|---:|---:|
| update_positions | XXX,XXX | XX% | XX% | XX% | XX% | X,XXX,XXX |

Expected changes versus AoS:

- **SPE sample count drops substantially.** Fewer samples for the same workload means the loop completes in less wall-clock time — the primary throughput signal.
- **L1C % rises.** The 24 MB working set fits in L3; the hardware prefetcher keeps L1/L2 fed, so most loads resolve in L1 or L2.
- **DRAM % drops.** With a 24 MB hot working set instead of 64 MB, significantly less data must be fetched from DRAM each iteration.
- **Potential Improvement falls.** ATP estimates fewer cycles lost to memory latency, confirming reduced memory inefficiency.

![SoA Latency Breakdown showing fewer SPE samples, higher L1C%, lower DRAM%](assets/soa_latency_breakdown.png)
*SoA Latency Breakdown: substantially fewer SPE samples on `update_positions`, higher L1C %, lower DRAM %, lower Potential Improvement — consistent with 100% cache line utilisation and a 24 MB hot working set.*

### Step 2: Capture SoA memory numbers

Record the following for comparison:

- `#SPE Samples (update_positions): ___________`
- `L1C % Loads: ___________`
- `DRAM % Loads: ___________`
- `Potential Improvement (cyc): ___________`

---

## 6. Re-map hotspot to source with CPU Cycle Hotspots (SoA)

Run **CPU Cycle Hotspots** with `soa_optimized`, open **View Source Code** for `update_positions`, and navigate to the same hot loop:

```cpp
p.x[i] += p.vx[i] * dt;
p.y[i] += p.vy[i] * dt;
p.z[i] += p.vz[i] * dt;
```

The same source lines now carry **fewer periodic samples** than the AoS counterpart. Also check the call site in `main`:

```cpp
update_positions(particles, N, dt);
```

Record the new periodic sample count and compare with the AoS numbers from Section 3.

![Source Code Inspector for soa_optimized showing fewer periodic samples on the update loop lines](assets/soa_source.png)
*Source Code Inspector: the same update loop lines now carry fewer periodic samples, confirming reduced memory-bound stall time.*

---

## 7. Compare results and interpret

Fill in your real ATP numbers:

| Variant | Layout | Working set | Cache line utilisation | #SPE Samples | L1C % | DRAM % | Hot loop periodic samples | Potential Improvement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `aos_baseline` | AoS | 64 MB | 37.5% | _______ | __% | __% | _______ | _______ cyc |
| `soa_optimized` | SoA | 24 MB | 100%  | _______ | __% | __% | _______ | _______ cyc |

### Key interpretation

The AoS profile reveals the cost of carrying unused data through the cache hierarchy. Every access to a particle's position must also move `mass`, `charge`, `temperature`, and six other floats into the cache — even though `update_positions` never reads them. With a 64 MB working set larger than L3, those wasted bytes cause DRAM traffic on every iteration. ATP Memory Access makes this concrete: high SPE sample count, elevated DRAM %, high Potential Improvement.

The SoA profile shows what happens when you align data layout with access pattern. The hot arrays fit in L3, the hardware prefetcher works at 100% efficiency, and every byte moved through the memory hierarchy is consumed. The SPE sample count drops because the loop genuinely completes faster — not because of any compiler hint or algorithmic change. The Potential Improvement falls because there is less memory latency left to eliminate.

The CPU Cycle Hotspots source view makes the change attributable. The same source lines — the three position-update assignments — carry measurably fewer periodic samples in the SoA binary. Combined with the Memory Access run-level drop in SPE samples, you have two independent ATP measurements pointing at the same improvement: fewer total memory operations, and a lighter hot spot at the exact lines that changed.

This is the intended workflow for ATP Memory Access + CPU Cycle Hotspots:

1. **Memory Access** identifies the function and cache tier where latency accumulates.
2. **CPU Cycle Hotspots → View Source Code** maps that function to the specific source lines.
3. A source-level change (here: data layout restructuring) is applied.
4. Both recipes are re-run to confirm the improvement is real and attributable to the change.

---

> **Conclusion:** ATP Memory Access showed that `aos_baseline` suffered from low cache line utilisation (37.5%) due to the AoS data layout, forcing the 64 MB working set to repeatedly access L2/LLC/DRAM. Restructuring to SoA reduced the hot working set to 24 MB (fitting in L3) and raised cache line utilisation to 100%. Re-profiling with both recipes confirmed the improvement: substantially fewer SPE samples in Memory Access, lower DRAM %, and fewer periodic samples on the same hot source lines in CPU Cycle Hotspots.

---

## 8. Required screenshots checklist

- [ ] `assets/aos_recipe_setup.png` — Memory Access recipe configured for `aos_baseline`
- [ ] `assets/aos_latency_breakdown.png` — Latency Breakdown table: high SPE sample count, significant L2C/LLC/DRAM %, high Potential Improvement
- [ ] `assets/aos_hotspots_setup.png` — CPU Cycle Hotspots recipe configured for `aos_baseline`
- [ ] `assets/aos_source.png` — View Source Code showing high periodic samples on the `update_positions` loop lines
- [ ] `assets/soa_recipe_setup.png` — Memory Access recipe configured for `soa_optimized`
- [ ] `assets/soa_latency_breakdown.png` — Latency Breakdown table: fewer SPE samples, higher L1C %, lower DRAM %, lower Potential Improvement
- [ ] `assets/aos_hotspots_setup.png` — CPU Cycle Hotspots recipe configured for `soa_optimized`
- [ ] `assets/soa_source.png` — View Source Code showing fewer periodic samples on the same loop lines

---

## Notes

- Keep `-g` enabled so ATP can resolve source locations to line numbers.
- Keep the problem size at `N = 1 << 20` and `iters = 200` for stable SPE sampling; smaller sizes may not produce enough samples for a reliable breakdown.
- Compare like-for-like runs: same Graviton instance, same CPU frequency policy, no other heavy workloads running concurrently.
- The key validation is combined evidence: lower `#SPE Samples` in Memory Access **and** lower **Periodic Samples on the same source lines** in CPU Cycle Hotspots.
- If ATP does not resolve function names for `update_positions` (shows as `??`), ensure you point the source root to the `tutorial_2/src` directory and that debug symbols are present (`file aos_baseline` should show `with debug_info`).
