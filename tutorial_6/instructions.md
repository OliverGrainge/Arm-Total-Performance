# Tutorial 6: Capstone — Optimising a CLIP Image Search Application with Arm Total Performance

In the previous tutorials you profiled small, self-contained workloads to learn individual ATP recipes. This capstone tutorial puts it all together: you will optimise the backend of a **real application** — a text-to-image search engine powered by [pgvector](https://github.com/pgvector/pgvector) on PostgreSQL.

## What the application does

The application lets you type a text description — "a red sports car", "a cute puppy" — and instantly find the most visually matching images from a database of 50,000 photographs.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

### How it works

The key idea is **vector search**. A neural network called [CLIP](https://openai.com/research/clip) converts every image in the database into a list of 512 numbers (a "vector" or "embedding"). It does the same for the text query. Images that are visually similar to the text end up with similar vectors.

To find which image best matches a query, we measure the **L2 (Euclidean) distance** between the query vector **q** and each image vector **x**:

$$d(\mathbf{q}, \mathbf{x}) = \sum_{i=1}^{512} (q_i - x_i)^2$$

The smaller the distance, the better the match. Searching for matching images becomes: *find the vectors in the database with the smallest distance to the query vector*.

With 50,000 images, comparing against every single vector would be too slow. Instead, the application uses [pgvector](https://github.com/pgvector/pgvector) — a PostgreSQL extension for vector similarity search. pgvector builds an **HNSW index** (Hierarchical Navigable Small World), which you can think of as a graph where similar vectors are connected as neighbors. Instead of checking all 50,000 vectors, a query walks this graph — hopping from neighbor to neighbor — to quickly zero in on the closest matches.

### Architecture

The application has two layers:

**Dashboard (Python + Gradio)** — the interactive frontend. It loads the CLIP text encoder to convert typed queries into 512-dimensional vectors, then sends SQL queries to PostgreSQL to search the pgvector index.

**PostgreSQL + pgvector** — the search backend. It stores the 50,000 image embeddings and serves nearest-neighbor queries using the HNSW index. This is the performance-critical component.

A single interactive query returns in milliseconds. But the backend that handles **bulk query traffic** is where the real cost lives. When you need to serve 10,000 queries in a batch — imagine a production service processing a stream of user searches — every inefficiency adds up. That is the workload you will profile and optimise.

## Before you begin

- An **AWS Graviton 2/3** instance (e.g. `m7g.xlarge` or larger)
- **PostgreSQL 14+** with **pgvector** extension
- **Python 3.8+** with pip
- **ATP** installed and configured

## Terms used in this tutorial

| Term | What it means |
|------|---------------|
| **CLIP** | A neural network that maps images and text into a shared space of numbers. Similar concepts get similar numbers, whether they started as an image or as text. |
| **Vector / Embedding** | A fixed-length list of numbers (512 floats here) that represents an image or text query. |
| **pgvector** | An open-source PostgreSQL extension that adds vector data types and similarity search. See [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector). |
| **HNSW** | Hierarchical Navigable Small World — a graph-based search algorithm. Think of it as a web of connections between similar vectors, so you can find nearest neighbors by walking the graph instead of scanning everything. |
| **`<->` operator** | The pgvector L2 distance operator. `ORDER BY embedding <-> query` sorts results by distance. |
| **ef_search** | Controls how thoroughly HNSW explores the graph during search. Higher = more accurate but slower. |
| **Page** | The operating system divides memory into fixed-size chunks called "pages". The default size is 4 KB (4,096 bytes). |
| **Huge Pages** | A Linux feature that uses 2 MB pages instead of the default 4 KB. This matters a lot for large memory regions, as explained in Step 5. |
| **TLB** | Translation Lookaside Buffer — a small, fast cache inside the CPU that remembers where recently used memory pages are physically located. Think of it as a quick-reference address book. When the address is not in the book (a "TLB miss"), the CPU must do a slow lookup called a "page table walk". |
| **shared_buffers** | PostgreSQL's main memory cache for table and index data. This is where your vector data lives in memory. |

---

## Step 1: Install PostgreSQL and pgvector

```bash
sudo dnf install -y postgresql16-server postgresql16-server-devel postgresql16-contrib
```

Initialise the database cluster (first time only) and start PostgreSQL:

```bash
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

Create a superuser for your Linux account:

```bash
sudo -u postgres createuser --superuser $USER
```

Install the pgvector extension from source:

```bash
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make PG_CONFIG=/usr/bin/pg_config
sudo make install PG_CONFIG=/usr/bin/pg_config
cd .. && rm -rf pgvector
```

---

## Step 2: Set up the data

### Generate CLIP embeddings and load into pgvector

```bash
python scripts/setup_data.py
```

This script:
1. Downloads the **CIFAR-100** dataset (~170 MB, 50,000 training images + 10,000 test images)
2. Loads the **CLIP ViT-B-32** model and extracts 512-dimensional embeddings for all images
3. Creates a PostgreSQL database called `clip_search`
4. Loads the 50,000 image embeddings into a `images` table with a pgvector HNSW index
5. Saves image arrays and query embeddings locally for the dashboard and benchmark

You can verify the database is set up correctly:

```bash
psql clip_search -c "SELECT COUNT(*) FROM images;"
```

You should see `50000`.

---

## Step 3: Try the dashboard

```bash
python dashboard/app.py
```

Open the URL printed in the terminal. Type a description — "a red sports car", "sunset over the ocean", "a cute puppy" — and the dashboard returns the 10 most similar images from the database.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

The search feels instant. But behind that single query is an HNSW index over 50,000 embeddings, and at scale the backend must handle thousands of queries efficiently. The rest of this tutorial focuses on profiling and optimising that backend.

---

## Step 4: Run the baseline benchmark

The benchmark script sends 10,000 nearest-neighbor queries to pgvector and reports throughput:

```bash
python scripts/benchmark.py
```

Expected output (timings vary by instance):

```
Loaded 10000 query embeddings (dim=512)

Benchmark: 10000 queries, k=10, ef_search=200
Database:  clip_search

   1000 / 10000  (142.3 queries/sec)
   2000 / 10000  (139.8 queries/sec)
   ...
  10000 / 10000  (141.5 queries/sec)

=============================================
  Total time:     70.67 s
  Throughput:     141.5 queries/sec
  Avg latency:    7.07 ms/query
  Queries:        10000
  ef_search:      200
=============================================
```

Record this baseline throughput. You will compare it against the optimised configuration.

---

## Step 5: Profile the baseline with ATP

While the benchmark runs, PostgreSQL does the real work: the `postgres` backend process walks the HNSW graph, computes distances between vectors, and reads data from shared memory. This is the process you will profile with ATP.

### Record the workload with ATP

Start the benchmark in one terminal:

```bash
python scripts/benchmark.py
```

In ATP, select **Attach to Process** and choose the `postgres` backend process connected to the `clip_search` database. Start recording and let it capture for at least 30 seconds while the benchmark runs.

### Analyse with Topdown

Once the capture completes, select the **Topdown** recipe to see the high-level breakdown.

In the Topdown summary, look at the four buckets:

<p align="center">
<img src="assets/baseline_topdown.png" width="850" alt="Topdown summary for baseline — Backend Bound elevated with Memory Bound component"/>
</p>

You should see **Backend Bound** accounting for a significant share of execution slots, with the **Memory Bound** sub-category being a major contributor. In plain terms: the CPU is spending a lot of its time *waiting for data from memory* rather than doing useful computation.

### Drill into memory: understanding TLB stalls

Now select the **Memory Access** recipe in ATP. Look at three key metrics:

- **DTLB walk cycles** — time the CPU spends looking up memory addresses after its quick-reference cache (the TLB) misses
- **L1D cache hit rate** — how often data is found in the CPU's fastest cache
- **Average load latency** — how many CPU cycles each memory read takes on average

<p align="center">
<img src="assets/baseline_memory_access.png" width="850" alt="Memory Access metrics — elevated TLB walk cycles"/>
</p>

The critical finding is **elevated DTLB walk cycles**. Here is what is happening and why it matters:

#### The address book analogy

To understand this bottleneck, think of memory like a library with thousands of bookshelves. The CPU needs to know the physical location of each "shelf" (memory page) it wants to read. It keeps a small, fast address book — the **TLB** — that maps page addresses. This address book can only hold about 48 entries.

Now consider the problem:

- PostgreSQL stores the HNSW index (all 50,000 vectors and their graph connections — about 100 MB of data) in its `shared_buffers` memory region.
- With the default configuration, this memory is divided into **4 KB pages**. That means ~25,000 pages to cover 100 MB.
- The TLB can only remember the locations of ~48 pages at a time.
- During an HNSW search, PostgreSQL hops between vectors that are scattered across this memory. Each hop likely lands on a *different* page.

Since the TLB can only track 48 pages but the search needs to access thousands of different pages, it constantly overflows. Every overflow (a "TLB miss") forces the CPU to do a slow, multi-step lookup called a **page table walk** — like having to look up an address in a filing cabinet instead of your quick-reference card.

#### The diagnosis

**Backend Bound -> Memory Bound -> TLB stalls.** The HNSW graph traversal accesses vectors spread across ~100 MB of shared memory. With 4 KB pages, the TLB cannot keep up. The fix: use **huge pages** (2 MB each), which cover 512x more memory per entry, so the TLB can track the entire memory region comfortably.

---

## Step 6: Optimisation 1 — PostgreSQL memory tuning

Before enabling huge pages, first ensure PostgreSQL is configured to make good use of memory. The default configuration is very conservative — it was designed to work on machines with as little as 256 MB of RAM.

### Check current settings

```bash
psql clip_search -c "SHOW shared_buffers;"
psql clip_search -c "SHOW work_mem;"
psql clip_search -c "SHOW effective_cache_size;"
```

The defaults are typically:
- `shared_buffers = 128MB`
- `work_mem = 4MB`
- `effective_cache_size = 4GB`

### Tune PostgreSQL memory

Edit the PostgreSQL configuration:

```bash
sudo nano /var/lib/pgsql/data/postgresql.conf
```

Set these four values:

```
shared_buffers = 512MB
work_mem = 128MB
effective_cache_size = 2GB
maintenance_work_mem = 256MB
```

**What each setting does and why we are changing it:**

| Setting | Default | New value | Why |
|---------|---------|-----------|-----|
| `shared_buffers` | 128 MB | 512 MB | This is PostgreSQL's own data cache. The pgvector index is ~100 MB, plus the table data. At 128 MB, there's barely room and data gets evicted. At 512 MB, the entire index stays comfortably in memory. |
| `work_mem` | 4 MB | 128 MB | Memory available per query for sorting and processing. The HNSW search builds a candidate list — more memory means it does not have to spill intermediate results to disk. |
| `effective_cache_size` | 4 GB | 2 GB | This does not allocate memory — it just tells PostgreSQL's query planner how much total cache (shared_buffers + OS file cache) is available, so it can make better decisions about whether to use indexes. |
| `maintenance_work_mem` | 64 MB | 256 MB | Memory for maintenance tasks like building indexes and VACUUM. Speeds up index rebuilds if you re-create the HNSW index. |

Restart PostgreSQL to apply changes:

```bash
sudo systemctl restart postgresql
```

### Re-benchmark

```bash
python scripts/benchmark.py
```

You should see a moderate improvement in throughput. The bigger `shared_buffers` ensures the index data stays in PostgreSQL's own cache instead of relying on the OS page cache, which has more overhead to access. But the underlying TLB problem still exists — the memory is still divided into small 4 KB pages.

---

## Step 7: Optimisation 2 — Huge pages

This is the key optimisation, directly targeting the TLB bottleneck that ATP revealed.

### Why huge pages fix the problem

Recall the address book analogy from Step 5. The TLB can hold ~48 entries. With standard 4 KB pages:

- 512 MB of shared_buffers = **131,072 pages** to track
- The TLB can hold **48** of them at a time
- Result: constant TLB misses, constant slow page table walks

With 2 MB huge pages:

- 512 MB of shared_buffers = **256 pages** to track
- The TLB can hold **48** of them at a time — that covers nearly the entire region
- Result: far fewer TLB misses, the CPU spends its time computing instead of waiting

### Step 7a: Reserve huge pages in the kernel

The kernel needs to set aside 2 MB pages in advance. We need enough for PostgreSQL's `shared_buffers` (512 MB) plus a small overhead:

```bash
# 512 MB / 2 MB = 256 pages, plus ~24 for PostgreSQL internals
echo 280 | sudo tee /proc/sys/vm/nr_hugepages
```

Verify the pages were reserved:

```bash
cat /proc/meminfo | grep HugePages
```

You should see `HugePages_Total: 280` and `HugePages_Free: 280` (or close to it).

### Step 7b: Tell PostgreSQL to use huge pages

Edit the configuration:

```bash
sudo nano /var/lib/pgsql/data/postgresql.conf
```

Add this line:

```
huge_pages = on
```

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

> **Troubleshooting:** If PostgreSQL fails to start, it could not get enough huge pages. This usually means the system could not allocate them because memory is fragmented. Try increasing `nr_hugepages` or freeing memory by stopping other services. Check logs with `sudo journalctl -u postgresql`.

### Step 7c: Make huge pages survive reboots

The `echo` command above only lasts until the next reboot. To make it permanent:

```bash
echo "vm.nr_hugepages = 280" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Re-benchmark

```bash
python scripts/benchmark.py
```

You should see a noticeable improvement in throughput. The TLB can now cover the shared memory region with far fewer entries, so the CPU spends less time on page table walks and more time on actual vector distance computation.

---

## Step 8: Re-profile with ATP — confirming the fix

Repeat the profiling process from Step 5 with huge pages enabled. Start the benchmark, attach ATP to the postgres backend, and capture a new recording.

### Topdown comparison

<p align="center">
<img src="assets/hugepages_topdown.png" width="850" alt="Topdown after huge pages — Backend Bound reduced"/>
</p>

| Bucket        | Baseline | After Huge Pages | What this means |
|---------------|----------|------------------|-----------------|
| Backend Bound | High     | Lower            | Less time waiting for memory |
| Retiring      | Low      | Higher           | More time doing useful work |

The shift from Backend Bound to Retiring is the confirmation: the CPU is now spending more of its time on actual computation (distance calculations, graph traversal) and less time stalled waiting for address translations.

### Memory Access comparison

<p align="center">
<img src="assets/hugepages_memory_access.png" width="850" alt="Memory Access after huge pages — TLB walk cycles reduced"/>
</p>

| Metric             | Baseline  | After Huge Pages | What changed |
|--------------------|-----------|------------------|--------------|
| DTLB walk cycles   | High      | Dramatically lower | Far fewer TLB misses — the "address book" can now cover the data |
| L1D hit rate       | ~85-90%   | ~92-97%          | With fewer TLB stalls, the caches work more effectively |
| Avg load latency   | Higher    | Lower            | Each memory read completes faster on average |

This confirms the diagnosis from Step 5: the TLB bottleneck was real, and huge pages resolved it.

---

## Summary

You started with a text-to-image search application — 50,000 images indexed by CLIP embeddings in pgvector — and used ATP to find and fix a memory bottleneck:

| Step | What you did | What ATP showed |
|------|-------------|-----------------|
| **Baseline** | Default PostgreSQL config, 4 KB pages | CPU is memory-bound, spending too much time on TLB misses (address lookups) |
| **Memory tuning** | Increased `shared_buffers` to 512 MB | Keeps the index in PostgreSQL's cache (moderate improvement, TLB issue remains) |
| **Huge pages** | Switched to 2 MB pages | TLB can now cover the data region — address lookup stalls disappear |

The optimisation followed the same loop as the earlier tutorials: **profile -> diagnose -> fix -> re-profile**. The difference is that this time the fix was not in source code but in **kernel and database configuration** — a common pattern when optimising production workloads on Arm.

### File reference

| File | Description |
|------|-------------|
| `scripts/setup_data.py` | Downloads CIFAR-100, generates CLIP embeddings, loads into pgvector |
| `scripts/benchmark.py` | Runs batch queries against pgvector and reports throughput |
| `dashboard/app.py` | Gradio dashboard for interactive text-to-image search |

### Key takeaway

Not all performance bottlenecks live in source code. When you deploy applications on Arm — especially database-backed services with large in-memory data structures — kernel-level configuration like huge pages can have a dramatic impact on throughput. ATP's Memory Access recipe gives you the evidence to identify these bottlenecks and verify that your configuration changes actually resolved them.
