# Tutorial 7: Profiling and Optimising Redis on Arm with ATP

Redis is one of the most widely used in-memory data stores. It powers caches, session stores, and real-time analytics in applications around the world. Because all data lives in memory and a single thread handles every request, Redis performance depends heavily on how efficiently the CPU can access memory.

In this tutorial you will load a large dataset into Redis, benchmark it, use ATP to identify the bottleneck, and apply a kernel-level optimisation to improve throughput.

## How Redis works (the short version)

Redis stores all data in memory as key-value pairs. When a client asks for a key, Redis:

1. **Hashes** the key name to find which bucket it belongs to
2. **Looks up** the bucket in a large hash table spread across memory
3. **Reads** the value data from wherever it is stored in memory
4. **Sends** the result back to the client

With a small dataset this is extremely fast — everything fits in the CPU's caches. But when the dataset grows to hundreds of megabytes or more, the hash table and values are spread across many memory pages. Random lookups start hitting memory that is not in the cache, and the CPU spends more time waiting for data than doing useful work.

## Before you begin

- An **AWS Graviton 2/3** instance (e.g. `m7g.xlarge` with 4+ GB RAM)
- **ATP** installed and configured

## Terms used in this tutorial

| Term | What it means |
|------|---------------|
| **Page** | The operating system splits memory into small, equal-sized chunks called "pages". By default each page is 4 KB. |
| **Huge Pages** | A Linux feature that uses much bigger memory chunks (2 MB instead of 4 KB). Fewer pages means the CPU can keep track of more memory in its fast lookup table. |
| **TLB** | Translation Lookaside Buffer — a small, fast lookup table in the CPU that remembers where recently used memory pages are stored. When the page you need is not in the TLB (a "TLB miss"), the CPU has to do a much slower search called a "page table walk". |
| **THP** | Transparent Huge Pages — a Linux kernel feature that automatically uses huge pages without the application needing to request them explicitly. |
| **Pipelining** | Sending multiple Redis commands in a batch without waiting for each response. This reduces network round trips and makes the benchmark focus on memory access rather than networking. |

---

## Step 1: Install Redis from source

The default `redis6` package on Amazon Linux 2023 does not include the benchmark tool and does not allow control over Transparent Huge Pages. Building Redis 7 from source gives us both.

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

### Disable Transparent Huge Pages (baseline)

For the baseline measurement, ensure THP is disabled so Redis uses the default 4 KB pages:

```bash
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

### Start Redis

Start Redis with persistence disabled (we do not need to save data to disk):

```bash
redis-server --daemonize yes --save "" --appendonly no
```

> **Note:** You may see warnings about memory overcommit. These are safe to ignore for this tutorial. If you want to suppress them: `sudo sysctl vm.overcommit_memory=1`

Verify it is running:

```bash
redis-cli ping
```

You should see `PONG`.

---

## Step 2: Load data

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

Before making any changes, measure the current performance. The benchmark script runs 5 million random GET operations with pipelining enabled, which maximises throughput and ensures the bottleneck is memory access rather than network overhead:

```bash
bash scripts/run_benchmark.sh
```

Write down the **requests per second** number. This is your baseline.

---

## Step 4: Profile the baseline with ATP

### Start the workload

Run the infinite benchmark in one terminal:

```bash
bash scripts/run_benchmark_inf.sh
```

The script prints the Redis server PID on startup.

### Attach ATP to the Redis process

> **Important:** Attach ATP to the **redis-server** process, not the redis-benchmark client. The server is where all the data access happens.

Find the PID if you need it:

```bash
ps aux | grep redis-server
```

In ATP, select **Attach to Process** and enter the PID of the `redis-server` process. Start recording and let it capture for at least 30 seconds while the benchmark runs.

### Analyse with Topdown

Once the capture completes, select the **Topdown** recipe.

<p align="center">
<img src="assets/baseline_topdown.png" width="850" alt="Topdown summary for Redis baseline"/>
</p>

You should see that **Backend Bound** is the largest category (around 60%). This tells you the CPU is spending most of its time waiting for data to arrive from memory.

### Analyse with Memory Access

Now select the **Memory Access** recipe. The key metrics to look at are in the **Data TLB Effectiveness** panel:

<p align="center">
<img src="assets/baseline_memory_access.png" width="850" alt="Memory Access metrics for Redis baseline"/>
</p>

| Metric | What to look for |
|--------|-----------------|
| **DTLB MPKI** | Data TLB misses per 1000 instructions — how often the TLB cannot find the page |
| **L1 Data TLB MPKI** | L1 DTLB misses per 1000 instructions — misses at the first (fastest) TLB level |
| **DTLB Walk Ratio** | Percentage of TLB accesses that trigger an expensive page table walk |
| **L1 Data TLB Miss Ratio** | Percentage of L1 DTLB lookups that miss |

#### Understanding what the numbers mean

Every time Redis reads data from memory, the CPU must translate the virtual address to a physical one. This goes through a chain:

```
Memory access → L1 DTLB → (miss?) → L2 TLB → (miss?) → Page table walk
                  fast          slower           expensive
               (~1 cycle)    (~5-10 cycles)    (~10-100+ cycles)
```

With ~1 GB of data spread across roughly **250,000 small pages** (4 KB each), and the TLB only able to track about **48 pages** at a time, random key lookups cause frequent TLB misses. Each miss that is not caught by the L2 TLB triggers an expensive page table walk — the CPU must read multiple levels of page tables from memory to find the physical address.

Even a modest DTLB MPKI (e.g. 1–2 misses per 1000 instructions) adds up because each page table walk can cost 10–100+ cycles. With the CPU already 60% Backend Bound, every source of memory stall matters.

---

## Step 5: Optimisation — Transparent Huge Pages

### Why huge pages help

With the default 4 KB pages:

- 1 GB of data = **~250,000 pages**
- The TLB can remember **~48** at a time
- Random lookups constantly miss the TLB → slow page table walks

With 2 MB huge pages:

- 1 GB of data = **~512 pages**
- The TLB can cover a much larger portion of memory
- Far fewer TLB misses → faster memory access

### Enable Transparent Huge Pages

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

The `defrag` setting tells the kernel to actively create contiguous 2 MB regions in memory rather than waiting passively.

### Restart Redis with THP support

By default, Redis disables Transparent Huge Pages for itself because they can cause high memory usage when Redis forks for persistence (RDB/AOF saves). Since we have persistence disabled, we can safely tell Redis to allow them with `--disable-thp no`:

```bash
redis-cli shutdown
redis-server --daemonize yes --save "" --appendonly no --disable-thp no
```

Reload the data so Redis allocates memory with huge pages:

```bash
bash scripts/load_data.sh
```

### Verify huge pages are being used

```bash
sudo grep -e AnonHugePages /proc/$(redis-cli info server | grep process_id | cut -d: -f2 | tr -d '[:space:]')/smaps_rollup
```

You should see a large number — close to the full dataset size (e.g. `AnonHugePages: 1112064 kB` for ~1 GB). If it shows 0 kB, check that THP is enabled (`cat /sys/kernel/mm/transparent_hugepage/enabled` should show `[always]`) and that Redis was started with `--disable-thp no`.

### Re-benchmark

```bash
bash scripts/run_benchmark.sh
```

Compare this number with your baseline. You should see improved throughput.

---

## Step 6: Re-profile with ATP — confirming the fix

Repeat the profiling from Step 4: run the infinite benchmark, attach ATP to the redis-server process, and capture a new recording.

### Topdown comparison

<p align="center">
<img src="assets/hugepages_topdown.png" width="850" alt="Topdown after huge pages"/>
</p>

| Category      | Before       | After Huge Pages | What this means |
|---------------|--------------|------------------|-----------------|
| Backend Bound | High (~60%)  | Lower            | Less time waiting for memory |
| Retiring      | Low          | Higher           | More time doing useful work |

### Data TLB comparison

<p align="center">
<img src="assets/hugepages_memory_access.png" width="850" alt="Memory Access after huge pages"/>
</p>

| Metric | Before | After Huge Pages | What changed |
|--------|--------|------------------|--------------|
| DTLB MPKI | ~1.4 | ~0.5 | ~65% fewer TLB misses |
| L1 Data TLB MPKI | ~11 | ~4 | ~65% fewer L1 DTLB misses |
| DTLB Walk Ratio | ~0.39 | ~0.13 | ~67% fewer page table walks |
| L1 Data TLB Miss Ratio | ~3.2 | ~1.2 | ~62% fewer L1 misses per access |

With huge pages, each TLB entry covers 2 MB instead of 4 KB — 512 times more memory. The TLB can now track a much larger portion of the dataset, so random key lookups trigger far fewer expensive page table walks.

---

## Summary

You loaded a 1 GB dataset into Redis and used ATP to find and fix a performance problem:

| Step | What you did | What you learned |
|------|-------------|------------------|
| **Baseline** | ~1M keys with 1 KB values, default 4 KB pages, THP disabled | Redis is **Backend Bound** — the CPU waits for memory, with TLB misses contributing to the stall |
| **Huge pages** | Enabled THP, started Redis with `--disable-thp no` | TLB misses drop by ~65%, page table walks drop by ~67% |

### Why this matters

Redis is single-threaded — it cannot speed up by using more CPU cores. When the dataset is large, the only way to go faster is to make each memory access more efficient. Huge pages do exactly this by reducing the overhead of translating memory addresses.

This is a common pattern for in-memory workloads on Arm: the application code is already efficient, but the **operating system's memory configuration** is the bottleneck. ATP's Memory Access recipe gives you the evidence to find these problems and confirm that your changes worked.

### File reference

| File | Description |
|------|-------------|
| `scripts/load_data.sh` | Populates Redis with ~1M keys (~1GB) |
| `scripts/run_benchmark.sh` | Runs a fixed GET benchmark and reports throughput |
| `scripts/run_benchmark_inf.sh` | Infinite benchmark loop for attaching a profiler |
