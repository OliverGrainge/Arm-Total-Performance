# Tutorial 7: Optimising a Redis Database with Arm Performix

## What is Redis?

[Redis](https://redis.io/) is one of the most popular in-memory databases in the world. It stores data as simple key-value pairs: you give it a name (the key) and some data (the value), and it keeps everything in RAM so it can return results in microseconds.

Redis is used everywhere:

- **Caching**: web applications store frequently accessed data in Redis so they do not have to query a slower database every time
- **Session storage**: websites keep user login sessions in Redis for fast access across multiple servers
- **Real-time analytics**: dashboards and monitoring systems use Redis to aggregate and query live metrics
- **Message queues**: Redis acts as a lightweight broker between services that need to pass data to each other

Because Redis is **single-threaded** (one CPU core handles every request), its performance depends entirely on how efficiently that single core can access memory. When the dataset is small, everything fits in the CPU's caches and responses are near-instant. But as the dataset grows into the hundreds of megabytes, the CPU starts spending more time *waiting for memory* than doing useful work.

In this tutorial you will set up a Redis server and populate it with a ~1 GB dataset of one million keys, then run a throughput benchmark to establish a performance baseline. From there you will attach Performix to the running server and use the CPU Microarchitecture Analysis recipe to identify performance bottlenecks, then examine the TLB Effectiveness table within the results to investigate further. Finally, you will apply a kernel-level optimisation, re-run the benchmark, and re-profile with Performix to confirm the improvement.

## How Redis handles a request

When a client asks Redis for a key, the following happens inside the server:

1. **Hash** the key name to find which bucket it belongs to
2. **Look up** the bucket in a large hash table spread across memory
3. **Read** the value data from wherever it is stored in memory
4. **Send** the result back to the client

Steps 2 and 3 are where performance matters. With a million keys and 1 GB of data, the hash table and values are scattered across many memory pages. Each lookup lands on a random page — the CPU cannot predict which data it will need next, so it cannot prefetch anything into its caches.

## Before you begin

- An **AWS Graviton 2/3** instance (e.g. `m7g.xlarge` with 4+ GB RAM)
- **Performix** installed and configured

## Terms used in this tutorial

| Term | What it means |
|------|---------------|
| **Page** | The operating system splits memory into small, equal-sized chunks called "pages". By default each page is 4 KB. |
| **Huge Pages** | A Linux feature that uses much bigger memory chunks (2 MB instead of 4 KB). Fewer pages means the CPU can keep track of more memory in its fast lookup table. |
| **TLB** | Translation Lookaside Buffer, a small fast lookup table in the CPU that remembers where recently used memory pages are stored. When the page you need is not in the TLB (a "TLB miss"), the CPU has to do a much slower search called a "page table walk". |
| **THP** | Transparent Huge Pages — a Linux kernel feature that automatically uses huge pages without the application needing to request them explicitly. |
| **Pipelining** | Sending multiple Redis commands in a batch without waiting for each response. This reduces network round trips and makes the benchmark focus on memory access rather than networking. |

---

## Step 1: Install Redis from source

The default `redis6` package on Amazon Linux 2023 does not include the benchmark tool and does not allow control over Transparent Huge Pages. Building `redis7` from source gives us both.

### Build and install

```bash
sudo dnf install -y gcc make
curl -O https://download.redis.io/releases/redis-7.2.7.tar.gz
tar xzf redis-7.2.7.tar.gz
cd redis-7.2.7
make -j$(nproc)
sudo make install
cd .. && rm -rf redis-7.2.7 redis-7.2.7.tar.gz
```

This installs `redis-server`, `redis-cli`, and `redis-benchmark` to `/usr/local/bin/`.

---

## Step 2: Set up the baseline environment

Before measuring anything, set up a clean baseline: Transparent Huge Pages disabled, Redis running with default 4 KB pages.

### Disable Transparent Huge Pages

Ensure THP is turned off so the baseline uses the default small pages:

```bash
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

### Start Redis

Start Redis with persistence disabled. We do not need Redis to save data to disk, and disabling persistence avoids `fork()` calls that would complicate profiling later:

```bash
redis-server --daemonize yes --save "" --appendonly no
```

> **Note:** You may see a warning about memory overcommit. This is safe to ignore for this tutorial. If you want to suppress it: `sudo sysctl vm.overcommit_memory=1`

Verify it is running:

```bash
redis-cli ping
```

You should see `PONG`.

### Load data

The load script uses `redis-benchmark` to populate Redis with approximately 1 million unique keys, each with a 1 KB value. This creates a dataset of roughly 1 GB in memory — large enough to put pressure on the CPU's memory system.

```bash
bash scripts/load_data.sh
```

You can verify the dataset size:

```bash
redis-cli info memory | grep used_memory_human
redis-cli dbsize
```

You should see approximately 1 GB of memory used and around 1 million keys.

---

## Step 3: Run the baseline benchmark

Measure the current performance. The benchmark script runs 5 million random GET operations with pipelining enabled (sending 32 commands per round trip). Pipelining maximises throughput and ensures the bottleneck is memory access inside the server rather than network round trips:

```bash
bash scripts/run_benchmark.sh
```

Expected output (timings vary by instance):

```
Summary:
  throughput summary: 777363.19 requests per second
  latency summary (msec):
          avg       min       p50       p95       p99       max
        1.859     0.456     1.879     2.215     2.295     2.911
```

Write down the **requests per second** number. This is your baseline.

---

## Step 4: Profile the baseline with Performix

The benchmark tells you *how fast* Redis is, but not *why* it is that speed. Performix answers the why — it shows where the CPU is spending its time and what is slowing it down.

### Start the workload

Run the infinite benchmark in one terminal — this keeps the workload running so you have time to attach the profiler:

```bash
bash scripts/run_benchmark_inf.sh
```

The script prints the Redis server PID on startup.

### Attach Performix to the Redis process

> **Important:** Attach Performix to the **redis-server** process, not the redis-benchmark client. Redis is single-threaded, so all the data access — hashing keys, looking up the hash table, reading values — happens in this one process. The benchmark client only sends commands and waits for responses.

The `run_benchmark_inf.sh` script prints the Redis server PID when it starts. Make a note of it, or look it up at any time with:

```bash
ps aux | grep redis-server
```

The PID is the number in the second column (e.g. `66484` in `... 66484 ... redis-server`).

To attach Performix:

1. In Performix, select the **CPU Microarchitecture** recipe
2. Open the dropdown menu next to the target selector and choose **Attach to Running Process**
3. Paste the Redis server PID into the process field
4. Click **Run Recipe**

Let the capture run for at least 30 seconds while the benchmark is active, then stop it. When you are done, stop the infinite benchmark with `Ctrl+C`.

### Analyse with CPU Microarchitecture Analysis

Once the capture completes, select the **CPU Microarchitecture Analysis** recipe to see where the CPU is spending its time. The CPU Microarchitecture Analysis model splits all CPU activity into four categories:

- **Frontend Bound** — the CPU is stalled fetching or decoding instructions
- **Backend Bound** — the CPU is stalled waiting for data (memory, caches)
- **Retiring** — the CPU is doing useful work
- **Bad Speculation** — the CPU did work that had to be thrown away (mispredicted branches)

<p align="center">
<img src="assets/baseline_topdown.png" width="850" alt="CPU Microarchitecture Analysis summary for Redis baseline, Backend Bound dominant at ~62%"/>
</p>

You should see that **Backend Bound** is the largest category at around **63%**. This means the CPU is spending most of its time *waiting for data to arrive from memory* rather than doing useful work. Only about 16% of the time is spent on actual computation (Retiring).

This is a strong signal: Redis's performance is limited by memory access, not by the complexity of its code. With a 1 GB dataset and random key lookups, each request is likely to access data that is not in the CPU's caches, resulting in expensive trips to main memory. This data cache pressure is the primary driver of the high Backend Bound percentage.

However, there is more to the memory story than data cache misses alone. Every memory access also requires the CPU to translate virtual addresses to physical ones — and that translation has its own performance cost. The **Data TLB Effectiveness** table lets us investigate whether address translation is adding unnecessary overhead on top of the underlying cache miss latency.

### Look at the TLB Effectiveness table

Within the CPU Microarchitecture Analysis results, locate the **Data TLB Effectiveness** table:

<p align="center">
<img src="assets/baseline_memory_access.png" width="250" alt="Data TLB Effectiveness — elevated DTLB MPKI and walk ratio"/>
</p>

Here are the key numbers and what they mean:

| Metric | Baseline value | What it tells you |
|--------|---------------|-------------------|
| **DTLB MPKI** | 1.52 | 1.52 data TLB misses per 1000 instructions — every ~650 instructions triggers a TLB miss |
| **L1 Data TLB MPKI** | 11.81 | 11.81 L1 DTLB misses per 1000 instructions — the first-level TLB is frequently unable to find the page |
| **DTLB Walk Ratio** | 0.43 | 0.43% of all TLB accesses trigger an expensive page table walk |
| **L1 Data TLB Miss Ratio** | 3.47 | 3.47% of L1 DTLB lookups miss — roughly 1 in 29 memory accesses |

#### Why these numbers matter

Every time Redis reads data from memory, the CPU must translate the virtual address to a physical one. This translation goes through a chain, each level slower than the last:

```
Memory access → L1 DTLB → (miss?) → L2 TLB → (miss?) → Page table walk
                  fast          slower           expensive
               (~1 cycle)    (~5-10 cycles)    (~10-100+ cycles)
```

While the dominant source of Backend Bound stalls is data cache misses from random access patterns, the TLB metrics show that address translation is adding extra latency on top of those cache misses. **3.5% of memory accesses miss the fast L1 DTLB**, and a significant portion of those miss the L2 TLB as well (L2 Unified TLB Miss Ratio: 13.14), triggering expensive page table walks. Each walk can cost 10–100+ cycles as the CPU reads multiple levels of page tables from memory — cycles that are spent purely on address translation, not on fetching the data itself.

In other words, when Redis already has to wait for data from main memory, TLB misses make that wait even longer by adding page table walk latency before the data fetch can even begin.

#### Why TLB pressure is high for this workload

The TLB pressure comes from a mismatch between the dataset size and the page size:

- Redis holds **~1 GB** of data (keys, values, and hash table entries) in memory
- The operating system splits this into **~250,000 small pages** (4 KB each)
- The L1 DTLB can only remember the locations of a **small number of pages** at a time (typically a few dozen on Arm Neoverse cores)
- Redis lookups are **random** — each request hashes to a different key, landing on a different page

Since the L1 DTLB can only track a few dozen pages but the dataset spans 250,000 pages, random lookups frequently land on a page the TLB does not know about. We cannot easily fix the random access pattern (that is inherent to how a key-value store works), but we *can* reduce the number of pages the TLB needs to track — and that is something we can change at the OS level without modifying Redis itself.

---

## Step 5: Optimisation — Transparent Huge Pages

The TLB Effectiveness table has shown that address translation overhead is adding unnecessary latency to an already memory-bound workload. While we cannot eliminate the data cache misses inherent to random key lookups, we *can* reduce the TLB miss rate by using **fewer, larger pages** so the TLB can cover more memory with the same number of entries.

### Why huge pages fix the problem

With the default 4 KB pages:

- 1 GB of data = **~250,000 pages**
- The L1 DTLB can only track **a few dozen** at a time
- Random lookups constantly miss the TLB → slow page table walks

With 2 MB huge pages:

- 1 GB of data = **~512 pages**
- Each TLB entry covers **512×** more memory
- The TLB can cover a much larger portion of the dataset
- Far fewer TLB misses → faster memory access

### Enable Transparent Huge Pages

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

The `defrag` setting tells the kernel to actively create contiguous 2 MB regions in memory rather than waiting passively. Without this, the kernel may not promote pages to huge pages quickly enough.

### Restart Redis with THP support

By default, Redis disables Transparent Huge Pages for its own process. It does this because THP can cause high memory usage when Redis forks for persistence (RDB/AOF saves). Since we have persistence disabled, there are no forks, so huge pages are safe. We tell Redis to allow them with `--disable-thp no`:

```bash
redis-cli shutdown
redis-server --daemonize yes --save "" --appendonly no --disable-thp no
```

> **Note:** You will see warnings about THP being enabled and memory overcommit. These are safe to ignore — we have disabled persistence, so the issues Redis warns about do not apply.

### Reload data

Reload the data so Redis allocates memory with huge pages from the start:

```bash
bash scripts/load_data.sh
```

### Verify huge pages are being used

This is an important check. If Redis is not actually using huge pages, the optimisation will have no effect:

```bash
sudo grep -e AnonHugePages /proc/$(redis-cli info server | grep process_id | cut -d: -f2 | tr -d '[:space:]')/smaps_rollup
```

You should see a large number close to the full dataset size (e.g. `AnonHugePages: 1112064 kB` for ~1 GB).

If it shows `0 kB`, check that:
- THP is enabled: `cat /sys/kernel/mm/transparent_hugepage/enabled` should show `[always]`
- Defrag is enabled: `cat /sys/kernel/mm/transparent_hugepage/defrag` should show `[always]`
- Redis was started with `--disable-thp no`
- Redis's THP flag is on: `cat /proc/$(redis-cli info server | grep process_id | cut -d: -f2 | tr -d '[:space:]')/status | grep THP` should show `THP_enabled: 1`

### Re-benchmark

```bash
bash scripts/run_benchmark.sh
```

Expected output:

```
Summary:
  throughput summary: 875043.69 requests per second
  latency summary (msec):
          avg       min       p50       p95       p99       max
        1.626     0.464     1.631     1.967     2.031     2.831
```

Compare this with your baseline: throughput increased from ~777K to ~875K requests per second — a **~13% improvement** — and average latency dropped from 1.86 ms to 1.63 ms.

---

## Step 6: Re-profile with Performix — confirming the fix

A benchmark number going up is good, but it does not tell you *why* it improved. Performix does. Repeat the profiling from Step 4: run the infinite benchmark, attach Performix to the `redis-server` process, and capture a new recording.

### CPU Microarchitecture Analysis comparison

<p align="center">
<img src="assets/hugepages_topdown.png" width="850" alt="CPU Microarchitecture Analysis after huge pages, Backend Bound reduced"/>
</p>

| Category      | Before  | After Huge Pages | What this means |
|---------------|---------|------------------|-----------------|
| Frontend Bound | 24.21% | 27.81%          | Slightly increased — now that memory stalls are reduced, frontend becomes a larger relative share |
| Backend Bound | 62.84%  | 58.19%           | Less time waiting for memory |
| Retiring      | 15.66%  | 15.88%           | Slightly more time doing useful work |

Backend Bound has dropped by nearly 5 percentage points — the CPU is spending less time stalled on memory. Notice that Frontend Bound increased slightly: this is normal. When you reduce one source of stalls (TLB-related memory latency), the remaining bottlenecks become a larger relative share of the total. The overall effect is positive, as confirmed by the 13% throughput improvement in the benchmark.

### Data TLB comparison

<p align="center">
<img src="assets/hugepages_memory_access.png" width="250" alt="Data TLB Effectiveness after huge pages, all metrics improved"/>
</p>

| Metric | Before | After Huge Pages | Change |
|--------|--------|------------------|--------|
| DTLB MPKI | 1.52 | 0.48 | **-68%** fewer TLB misses |
| L1 Data TLB MPKI | 11.81 | 10.87 | **-8%** fewer L1 DTLB misses |
| L2 Unified TLB MPKI | 1.65 | 0.55 | **-67%** fewer L2 TLB misses |
| DTLB Walk Ratio | 0.43 | 0.17 | **-60%** fewer page table walks |
| L1 Data TLB Miss Ratio | 3.47 | 3.19 | **-8%** fewer L1 misses per access |
| L2 Unified TLB Miss Ratio | 13.14 | 4.66 | **-65%** fewer L2 misses per access |

The most expensive metric — **DTLB MPKI** (the overall TLB miss rate that triggers page table walks) — dropped by 68%. The **L2 Unified TLB Miss Ratio** dropped by 65%, meaning far fewer L1 TLB misses escalate to expensive page table walks. The L1 Data TLB Miss Ratio only dropped by 8% — the L1 TLB is very small so it still misses frequently, but with huge pages the L2 TLB can catch most of those misses before they become costly walks.

With huge pages, each TLB entry covers 2 MB instead of 4 KB — 512 times more memory. The TLB can now track a much larger portion of the dataset, so random key lookups trigger far fewer expensive page table walks.

This is the value of profiling with Performix: you can trace the improvement from the high-level CPU Microarchitecture Analysis view (less Backend Bound) all the way down to the specific hardware counter (fewer DTLB walks) that explains it.

---

## Summary

| Step | What you did | What Performix showed |
|------|-------------|-----------------|
| **Baseline** | 1M keys, 1 KB values, default 4 KB pages | **Backend Bound at 63%** — the CPU spends most of its time waiting for memory, primarily due to data cache misses from random access patterns. Data TLB metrics show additional overhead from frequent TLB misses and page table walks (DTLB MPKI: 1.52, Walk Ratio: 0.43). |
| **Diagnosis** | Examined the CPU Microarchitecture Analysis results and TLB Effectiveness table | While data cache misses drive the majority of the memory stalls, **TLB misses add avoidable overhead**: ~250,000 small pages for a 1 GB dataset, but the L1 DTLB can only track a few dozen at a time. This is an actionable target we can improve at the OS level. |
| **Optimisation** | Enabled THP and `defrag`, restarted Redis with `--disable-thp no` | **DTLB misses drop by 68%**, page table walks drop by 60%. Backend Bound falls from 63% to 58%. Throughput improves by ~13%. |

### The profile → diagnose → fix → confirm loop

This tutorial followed the same workflow as the earlier tutorials:

1. **Profile** the workload with Performix to see where the CPU spends its time
2. **Diagnose** the root cause by examining the relevant recipe (Memory Access → Data TLB)
3. **Fix** the problem with a targeted change (huge pages to reduce TLB pressure)
4. **Confirm** the fix by re-profiling and checking that the specific metrics improved

The key insight is that the fix was not in the application code — Redis itself is already highly optimised. The primary bottleneck is data cache misses from random access patterns, which is inherent to how a key-value store works. But Performix revealed that **TLB misses were adding avoidable overhead** on top of those cache misses — overhead that could be reduced through the operating system's memory configuration. Without Performix, this secondary source of latency would have been very difficult to identify. The CPU Microarchitecture Analysis recipe pointed to a memory-bound workload, and the TLB Effectiveness table highlighted address translation as a specific, actionable area for improvement.

### File reference

| File | Description |
|------|-------------|
| `scripts/load_data.sh` | Populates Redis with ~1M keys (~1GB) |
| `scripts/run_benchmark.sh` | Runs a fixed GET benchmark and reports throughput |
| `scripts/run_benchmark_inf.sh` | Infinite benchmark loop for attaching a profiler |
