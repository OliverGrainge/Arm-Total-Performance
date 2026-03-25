# Tutorial 6: Capstone — Profiling a CLIP Image Search Application with Arm Total Performance

In the previous tutorials you profiled small programs to learn individual ATP recipes. This final tutorial brings everything together: you will profile a **real application** — a text-to-image search engine that uses [pgvector](https://github.com/pgvector/pgvector), a plugin for the PostgreSQL database — and use ATP's Topdown and Memory Access recipes to identify its performance bottleneck.

## What the application does

The application lets you type a text description — "a red sports car", "a cute puppy" — and instantly find the most visually matching images from a database of 50,000 photographs.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

### How it works

The key idea is **vector search**. An AI model called [CLIP](https://openai.com/research/clip) turns every image into a list of 512 numbers. This list of numbers is called a **vector**. CLIP also turns the text you type into a vector. When an image closely matches your text, their vectors will have similar numbers.

To find the best match, we compare vectors by looking at how different they are — we subtract each pair of numbers and add up the differences. The smaller the total difference, the better the match.

The application uses [pgvector](https://github.com/pgvector/pgvector) — a plugin for the PostgreSQL database that is designed for this kind of search. When you ask for the closest matches, pgvector scans through all 50,000 vectors, computes the distance to each one, and returns the best results. This **brute-force sequential scan** is simple and gives exact results, but it means the database must read through a large block of memory for every single query.

### Architecture

The application has two layers:

**Dashboard (Python + Gradio)** — the part you interact with. It takes the text you type, converts it into a vector using CLIP, and asks the database to find matching images.

**PostgreSQL + pgvector** — the search engine. It stores all 50,000 image vectors and finds the closest matches when asked. This is the part where performance matters most.

A single search returns in milliseconds. But when the system needs to handle **thousands of searches** — imagine many users searching at the same time — every bit of slowness adds up. That is the workload you will profile with ATP to understand where the CPU spends its time.

## Before you begin

- An **AWS Graviton 2/3** instance (e.g. `m7g.xlarge` or larger)
- **PostgreSQL 14+** with **pgvector** extension
- **Python 3.8+** with pip
- **ATP** installed and configured

Install the required Python packages:

```bash
pip install torch open-clip-torch Pillow numpy gradio psycopg2-binary
```

Install PostgreSQL and its development headers (needed to build the pgvector extension from source):

```bash
sudo dnf install -y postgresql16-server postgresql16-server-devel postgresql16-contrib
```

## Terms used in this tutorial

| Term | What it means |
|------|---------------|
| **Page** | The operating system splits memory into small, equal-sized chunks called "pages". By default each page is 4 KB — a very small piece of memory. |
| **TLB** | Translation Lookaside Buffer — a tiny, fast lookup table built into the CPU. It remembers where recently used memory pages are stored. Think of it like a short contacts list on your phone — it is quick to check, but can only hold a limited number of entries. When the page you need is not in the list (a "TLB miss"), the CPU has to do a much slower search to find it. |

---

## Step 1: Set up PostgreSQL and pgvector

PostgreSQL is the database that will store the 50,000 image vectors and answer search queries. pgvector is an extension that adds vector search capabilities to PostgreSQL — without it, PostgreSQL has no way to efficiently compare and rank vectors. It is not available as a pre-built package, so it needs to be compiled from source and installed into your PostgreSQL installation.

### Initialise and start PostgreSQL

Before PostgreSQL can be used for the first time, its on-disk storage (the "database cluster") needs to be initialised. This creates the system tables and configuration files that every PostgreSQL instance requires. You only need to do this once.

```bash
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

The `enable` command ensures PostgreSQL starts automatically if the instance is rebooted.

### Create a superuser for your Linux account

By default, PostgreSQL only has a built-in `postgres` user. Creating a superuser that matches your Linux username lets you run `psql` and other database commands without having to switch to the `postgres` account each time.

```bash
sudo -u postgres createuser --superuser $USER
```

### Build and install pgvector

pgvector is compiled against the PostgreSQL development headers that were installed in the "Before you begin" step. The build process produces a shared library and SQL files that PostgreSQL loads when you enable the extension in a database.

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
python3 scripts/setup_data.py
```

This script:
1. Downloads a dataset of 50,000 small photographs called **CIFAR-100** (~170 MB)
2. Uses the **CLIP** AI model to convert each image into a vector (a list of 512 numbers)
3. Creates a PostgreSQL database called `clip_search`
4. Loads all 50,000 image vectors into the database
5. Saves the images and some pre-made query vectors locally for the dashboard and benchmark

This may take around 5 minutes depending on your instance.

You can verify the database is set up correctly:

```bash
psql clip_search -c "SELECT COUNT(*) FROM images;"
```

You should see `50000`.

---

## Step 3: Try the dashboard

Start the dashboard:

```bash
python3 dashboard/app.py
```

Open `http://localhost:7860` in your browser.

> **Connecting remotely?** If the instance is not your local machine, you will need SSH port forwarding so your browser can reach the dashboard. Add the `-L` flag when connecting:
> ```bash
> ssh -L 7860:localhost:7860 user@<your-instance-ip>
> ```
> Then open `http://localhost:7860` in your local browser as normal. Type a description — "a red sports car", "sunset over the ocean", "a cute puppy" — and the dashboard returns the 10 most similar images from the database.

<p align="center">
<img src="assets/image_search.gif" width="700" alt="Dashboard demo — typing a text query and retrieving matching images"/>
</p>

The search feels instant for a single query. But behind the scenes, the database is searching through 50,000 vectors. When many users search at the same time, the database needs to handle thousands of these searches efficiently. The rest of this tutorial focuses on benchmarking this workload and using ATP to understand where the CPU spends its time.

---

## Step 4: Run the baseline benchmark

The benchmark script runs 10,000 image searches against the database and measures how fast they complete:

```bash
python3 scripts/benchmark.py
```

Expected output (timings vary by instance):

```
Loaded 10000 query embeddings (dim=512)

Benchmark: 10000 queries, k=10
Database:  clip_search

    1000 / 10000  (XX.X queries/sec)
    ...
   10000 / 10000  (XX.X queries/sec)

=============================================
  Total time:     XXX.XX s
  Throughput:     XX.X queries/sec
  Avg latency:    XX.XX ms/query
  Queries:        10000
=============================================
```

Without an index, pgvector performs a sequential scan through all 50,000 vectors for every query, so throughput will be lower than an indexed search. This is expected — the brute-force scan is the workload you will profile.

Write down this number (queries per second). This is the baseline you will investigate with ATP.

---

## Step 5: Profile the baseline with ATP

While the benchmark runs, PostgreSQL is doing all the heavy lifting: its worker process is scanning sequentially through all 50,000 vectors, computing distances, and reading data from memory. This is the process you will profile with ATP.

### Record the workload with ATP

Start the infinite benchmark in one terminal — this keeps the workload running so you have time to attach the profiler:

```bash
python3 scripts/benchmark_inf.py
```

> **Important:** Do not attach ATP to the Python script. The Python process only sends queries and waits for results — all the real work (searching vectors, traversing the index, reading memory) happens inside a **postgres worker process**. That is the process you need to profile.

To find it, open a second terminal and run:

```bash
ps aux | grep postgres
```

Look for a line like `postgres: <your-username> clip_search` — this is the backend worker handling your benchmark queries. Note its PID.

In ATP, select **Attach to Process** and enter the PID of the postgres backend process. Start recording and let it capture for at least 30 seconds while the benchmark runs. When you are done, stop the benchmark with `Ctrl+C`.

### Analyse with Topdown

Once the capture completes, select the **Topdown** recipe to see where the CPU is spending its time.

Look at the four categories in the Topdown summary:

<p align="center">
<img src="assets/baseline_topdown.png" width="850" alt="Topdown summary for baseline — Backend Bound elevated with Memory Bound component"/>
</p>

You should see that **Backend Bound** takes up a large portion, with **Memory Bound** being the biggest part of it. What this tells you: the CPU is spending a lot of its time *waiting for data to arrive from memory* instead of doing useful work.

### Drill into memory: understanding the slowdown

Now select the **Memory Access** recipe in ATP. Look at three key numbers:

- **DTLB walk cycles** — how much time the CPU wastes searching for memory locations when its quick lookup table (the TLB) does not have the answer
- **L1D cache hit rate** — how often the CPU finds the data it needs in its fastest storage
- **Average load latency** — how long each memory read takes on average

<p align="center">
<img src="assets/baseline_memory_access.png" width="850" alt="Memory Access metrics — elevated TLB walk cycles"/>
</p>

The most important finding is that **DTLB walk cycles are high**. Here is what that means:

#### The contacts list analogy

Imagine your phone has a contacts list that can only hold 48 entries. Every time you need to call someone who is not on the list, you have to dig through a huge filing cabinet to find their number — which is much slower.

The CPU has the same problem. It keeps a small, fast lookup table called the **TLB** that remembers where recently used memory pages are stored. This table can only hold about 48 entries.

Now here is the issue:

- PostgreSQL stores all 50,000 image vectors (about 100 MB of data) in memory.
- With the default settings, this memory is split into tiny **4 KB chunks** (pages). That means about **25,000 separate pages** to cover 100 MB.
- The TLB can only remember the locations of **48 pages** at a time.
- Each query scans sequentially through all 50,000 vectors. As the scan moves through memory, it crosses into a new page roughly every two vectors — far faster than the TLB can keep up.

Since the TLB can only track 48 pages but the scan crosses through thousands of pages per query, it constantly runs out of space. Every time this happens (a "TLB miss"), the CPU has to do a slow search to find the right memory location — like digging through that filing cabinet instead of checking your contacts list.

#### What this means

The CPU is slow because of **memory lookups, specifically TLB misses**. Each query scans through ~100 MB of data, which is split into too many tiny pages for the TLB to keep track of. A potential fix would be to use **huge pages** (2 MB each instead of 4 KB). Each huge page covers 512 times more memory, so the TLB could track the entire data region without running out of space.

---

## Summary

You set up a real text-to-image search application — 50,000 images in a pgvector database — and used ATP to understand where the CPU spends its time:

| Step | What you did | What you learned |
|------|-------------|------------------|
| **Setup** | Installed PostgreSQL, pgvector, and loaded 50,000 image vectors | The application works end-to-end with a dashboard and benchmark |
| **Baseline** | Measured throughput with the benchmark script | Established a performance baseline (~443 queries/sec) |
| **Topdown** | Used ATP's Topdown recipe on the postgres process | The CPU is **Backend Bound**, with **Memory Bound** as the largest component |
| **Memory Access** | Used ATP's Memory Access recipe | High **DTLB walk cycles** — the TLB cannot keep up with ~25,000 small pages as the scan moves through ~100 MB of vector data |

ATP's Topdown recipe pointed to a memory bottleneck, and the Memory Access recipe pinpointed TLB misses as the root cause. This is the kind of insight that would be very difficult to find without a profiling tool — the application code itself is not the problem, but the way the operating system manages memory for the database workload.

### File reference

| File | Description |
|------|-------------|
| `scripts/setup_data.py` | Downloads images, creates vectors with CLIP, loads them into the database |
| `scripts/benchmark.py` | Runs 10,000 search queries and measures speed |
| `scripts/benchmark_inf.py` | Infinite benchmark loop for attaching a profiler |
| `dashboard/app.py` | Interactive web interface for text-to-image search |
