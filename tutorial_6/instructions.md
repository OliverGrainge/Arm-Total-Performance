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

With 50,000 images, computing this sum against every single vector (brute force) is too slow. Instead, the application stores the embeddings in [pgvector](https://github.com/pgvector/pgvector), a PostgreSQL extension that adds vector similarity search. pgvector builds an **HNSW index** (Hierarchical Navigable Small World) — a graph connecting similar vectors as neighbors — so queries can find the closest matches by walking the graph rather than scanning every vector.

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

- **CLIP**: Contrastive Language-Image Pre-training — a neural network that maps images and text into a shared vector space. Similar concepts have nearby vectors regardless of whether they originated as an image or text.
- **Vector / Embedding**: A fixed-length list of numbers (512 floats in this tutorial) representing an image or text query.
- **pgvector**: An open-source PostgreSQL extension that adds vector data types and similarity search operators. See [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
- **HNSW**: Hierarchical Navigable Small World — a graph-based approximate nearest-neighbor algorithm. pgvector uses this as its primary index type.
- **`<->` operator**: The pgvector L2 distance operator. `ORDER BY embedding <-> query` sorts results by Euclidean distance.
- **ef_search**: The size of the dynamic candidate list during HNSW search. Larger values explore more candidates (higher recall, slower queries).
- **Huge Pages**: A Linux memory feature that uses 2 MB pages instead of the default 4 KB. Reduces TLB (Translation Lookaside Buffer) misses for workloads with large memory footprints.
- **TLB**: Translation Lookaside Buffer — a CPU cache that stores recent virtual-to-physical address translations. TLB misses are expensive because they require a page table walk.
- **shared_buffers**: PostgreSQL's main memory cache for table and index data. Stored in shared memory, allocated at startup.

---

## Step 1: Install PostgreSQL and pgvector

```bash
sudo apt update
sudo apt install -y postgresql postgresql-server-dev-16
```

Install the pgvector extension:

```bash
sudo apt install -y postgresql-16-pgvector
```

> **Note:** If your distribution does not package pgvector, you can build it from source:
> ```bash
> git clone https://github.com/pgvector/pgvector.git
> cd pgvector
> make
> sudo make install
> ```

Start PostgreSQL and create a user for your Linux account:

```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres createuser --superuser $USER
```

---

## Step 2: Set up the data

### Install Python dependencies

```bash
cd tutorial_6
pip install -r requirements.txt
```

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

While the benchmark runs, PostgreSQL does the real work: the `postgres` backend process traverses the HNSW graph, computes L2 distances over 512-dimensional vectors, and manages shared memory buffers. This is the process you will profile with ATP.

### Find the PostgreSQL backend PID

In one terminal, start the benchmark:

```bash
python scripts/benchmark.py
```

In a second terminal, find the backend process:

```bash
# The backend process is the one connected to clip_search
sudo -u postgres psql -c "SELECT pid, datname, state FROM pg_stat_activity WHERE datname = 'clip_search';"
```

Note the PID.

### Record with perf

With the benchmark still running, record the postgres backend:

```bash
sudo perf record -g -p <PID> -o perf_baseline.data -- sleep 30
```

This captures 30 seconds of profiling data from the postgres process handling the queries.

### Load in ATP

Open ATP and import `perf_baseline.data`. Select the **Topdown** recipe to see the high-level breakdown.

In the Topdown summary, look at the four buckets:

<p align="center">
<img src="assets/baseline_topdown.png" width="850" alt="Topdown summary for baseline — Backend Bound elevated with Memory Bound component"/>
</p>

You should see **Backend Bound** accounting for a significant share of execution slots, with the **Memory Bound** sub-category being a major contributor. This indicates the CPU is spending a large fraction of its time waiting for data from memory.

### Drill into memory: TLB stalls

Select the **Memory Access** recipe in ATP. Look at:

- **DTLB walk cycles** — time spent on page table walks after TLB misses
- **L1D cache hit rate** — the fraction of loads satisfied by L1 cache
- **Average load latency** — mean cycles per load

<p align="center">
<img src="assets/baseline_memory_access.png" width="850" alt="Memory Access metrics — elevated TLB walk cycles"/>
</p>

The critical signal is **elevated DTLB walk cycles**. Here is why:

PostgreSQL allocates its `shared_buffers` as a large shared memory segment. With the default configuration, this memory uses standard 4 KB pages. The pgvector HNSW index stores 50,000 × 512-dim vectors (~100 MB) plus the graph structure in shared buffers. During search, the backend follows graph edges, jumping between vectors that are spread across this memory region.

Each 512-dim vector occupies 2,048 bytes. With 4 KB pages, each vector spans roughly half a page. The L1 data TLB on Graviton caches a limited number of page translations (typically 48 entries for 4 KB pages). As the HNSW search hops between distant nodes in the graph, each hop accesses a different page, and the TLB cannot hold translations for all the pages being visited. The result is frequent TLB misses, each requiring an expensive page table walk.

**Diagnosis: Backend Bound → Memory Bound → TLB stalls. The HNSW graph traversal accesses vectors spread across ~100 MB of shared memory using 4 KB pages, causing frequent TLB misses. The fix is to use huge pages (2 MB), which reduces the number of TLB entries needed by 512×.**

---

## Step 6: Optimisation — PostgreSQL memory tuning

Before enabling huge pages, first ensure PostgreSQL is configured to make good use of memory. The default configuration is very conservative.

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
sudo nano /etc/postgresql/16/main/postgresql.conf
```

Set:

```
shared_buffers = 512MB
work_mem = 128MB
effective_cache_size = 2GB
maintenance_work_mem = 256MB
```

**Why these values:**
- `shared_buffers = 512MB` — large enough to hold the entire pgvector index (~100 MB vectors + graph structure) comfortably in shared memory, with room for the table data
- `work_mem = 128MB` — allows each query to sort and process candidates without spilling to disk
- `effective_cache_size = 2GB` — tells the query planner how much OS page cache is available
- `maintenance_work_mem = 256MB` — speeds up index rebuilds and VACUUM operations

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

### Re-benchmark

```bash
python scripts/benchmark.py
```

You should see a moderate improvement in throughput. The shared_buffers increase ensures the index data stays resident in PostgreSQL's shared memory rather than relying on OS page cache, reducing some overhead. But the TLB problem persists — the memory is still mapped with 4 KB pages.

---

## Step 7: Optimisation — Huge pages

This is the key kernel-level optimisation. Huge pages use 2 MB pages instead of 4 KB, reducing the number of TLB entries needed to cover the shared memory segment by 512×.

### Calculate the required number of huge pages

PostgreSQL's `shared_buffers` (512 MB) is the main consumer. Add some overhead for PostgreSQL internals:

```bash
# Required huge pages = shared_buffers / 2MB + overhead
# 512MB / 2MB = 256, plus ~20 for overhead
echo 280 | sudo tee /proc/sys/vm/nr_hugepages
```

Verify:

```bash
cat /proc/meminfo | grep HugePages
```

You should see `HugePages_Total: 280` and `HugePages_Free: 280` (or close to it).

### Configure PostgreSQL to use huge pages

Edit the configuration again:

```bash
sudo nano /etc/postgresql/16/main/postgresql.conf
```

Set:

```
huge_pages = on
```

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

> **Troubleshooting:** If PostgreSQL fails to start, it means there are not enough free huge pages. Increase `nr_hugepages` or free memory by stopping other services. Check logs with `sudo journalctl -u postgresql`.

### Make huge pages persistent across reboots

```bash
echo "vm.nr_hugepages = 280" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Re-benchmark

```bash
python scripts/benchmark.py
```

You should see a noticeable improvement in throughput. The TLB now needs far fewer entries to cover the same shared memory region, so page table walks are dramatically reduced.

---

## Step 8: Re-profile with ATP

Repeat the profiling process from Step 5 with huge pages enabled.

Start the benchmark, find the PID, and record:

```bash
sudo perf record -g -p <PID> -o perf_hugepages.data -- sleep 30
```

Load `perf_hugepages.data` in ATP and compare against the baseline.

### Topdown comparison

<p align="center">
<img src="assets/hugepages_topdown.png" width="850" alt="Topdown after huge pages — Backend Bound reduced"/>
</p>

| Bucket        | Baseline | After Huge Pages |
|---------------|----------|------------------|
| Backend Bound | High     | Lower            |
| Retiring      | Low      | Higher           |

The shift from Backend Bound to Retiring indicates the CPU is spending more time doing useful work (distance computation, graph traversal) and less time stalled on TLB misses.

### Memory Access comparison

<p align="center">
<img src="assets/hugepages_memory_access.png" width="850" alt="Memory Access after huge pages — TLB walk cycles reduced"/>
</p>

| Metric             | Baseline  | After Huge Pages |
|--------------------|-----------|------------------|
| DTLB walk cycles   | High      | Dramatically lower |
| L1D hit rate       | ~85–90%   | ~92–97%          |
| Avg load latency   | Higher    | Lower            |

The DTLB walk cycles should drop significantly. Each 2 MB huge page covers 512× more memory than a 4 KB page, so the TLB can now hold translations for the entire shared_buffers region in far fewer entries.

---

## Summary

You started with a real text-to-image search application — 50,000 CIFAR-100 images indexed by 512-dimensional CLIP embeddings in pgvector on PostgreSQL — and used ATP to diagnose and fix a memory bottleneck in the search backend:

| Step | What you did | ATP Recipe | What ATP revealed |
|------|-------------|------------|-------------------|
| Baseline | Default PostgreSQL config, 4 KB pages | Topdown + Memory Access | Backend Bound → Memory Bound → TLB stalls from 4 KB pages over ~100 MB of vector data |
| Memory tuning | `shared_buffers = 512MB`, `work_mem = 128MB` | — | Ensures index data stays in shared memory |
| Huge pages | `huge_pages = on`, `vm.nr_hugepages = 280` | Topdown + Memory Access | TLB walk cycles drop, Backend Bound reduces, Retiring increases |

The optimisation followed the same loop as the earlier tutorials: **profile → diagnose → fix → re-profile**. The difference is that this time the fix was not in source code but in **kernel and database configuration** — a common pattern when optimising production workloads on Arm.

### File reference

| File | Description |
|------|-------------|
| `scripts/setup_data.py` | Downloads CIFAR-100, generates CLIP embeddings, loads into pgvector |
| `scripts/benchmark.py` | Runs batch queries against pgvector and reports throughput |
| `dashboard/app.py` | Gradio dashboard for interactive text-to-image search |
| `requirements.txt` | Python dependencies |

### Key takeaway

Not all performance bottlenecks live in source code. When you deploy applications on Arm — especially database-backed services with large in-memory data structures — kernel-level configuration like huge pages can have a significant impact on throughput. ATP's Memory Access recipe gives you the evidence to identify TLB-related bottlenecks and verify that your kernel configuration changes actually resolved them.
