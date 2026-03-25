#!/usr/bin/env python3
"""Benchmark pgvector image search throughput.

Runs batch nearest-neighbor queries against the pgvector database and reports
throughput (queries/sec) and average latency. Use this script to measure
performance before and after kernel optimisations.

Usage:
    python scripts/benchmark.py                      # default: 10000 queries
    python scripts/benchmark.py --n-queries 5000     # fewer queries
"""

import argparse
import os
import time

import numpy as np
import psycopg2


def run_benchmark(db_name, query_embeddings, n_queries, k):
    """Run batch kNN queries and report throughput."""
    queries = query_embeddings[:n_queries]
    print(f"Benchmark: {len(queries)} queries, k={k}")
    print(f"Database:  {db_name}\n")

    conn = psycopg2.connect(dbname=db_name)
    cur = conn.cursor()

    # Warm-up: run a few queries to warm the cache
    for i in range(min(10, len(queries))):
        emb_str = "[" + ",".join(f"{x:.6f}" for x in queries[i]) + "]"
        cur.execute(
            "SELECT id FROM images ORDER BY embedding <-> %s::vector LIMIT %s",
            (emb_str, k)
        )
        cur.fetchall()

    # Timed benchmark
    start = time.perf_counter()
    for i, query in enumerate(queries):
        emb_str = "[" + ",".join(f"{x:.6f}" for x in query) + "]"
        cur.execute(
            "SELECT id FROM images ORDER BY embedding <-> %s::vector LIMIT %s",
            (emb_str, k)
        )
        cur.fetchall()

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start
            qps = (i + 1) / elapsed
            print(f"  {i+1:>6d} / {len(queries)}  ({qps:.1f} queries/sec)")

    total_time = time.perf_counter() - start
    qps = len(queries) / total_time
    avg_latency_ms = total_time / len(queries) * 1000

    print(f"\n{'='*45}")
    print(f"  Total time:     {total_time:.2f} s")
    print(f"  Throughput:     {qps:.1f} queries/sec")
    print(f"  Avg latency:    {avg_latency_ms:.2f} ms/query")
    print(f"  Queries:        {len(queries)}")
    print(f"{'='*45}")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark pgvector image search")
    parser.add_argument("--db-name", default="clip_search", help="PostgreSQL database name")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Data directory")
    parser.add_argument("--n-queries", type=int, default=10000, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    args = parser.parse_args()

    # Load query embeddings
    query_path = os.path.join(args.data_dir, "query_embeddings.npy")
    if os.path.exists(query_path):
        query_embeddings = np.load(query_path)
    else:
        # Fall back to binary format
        bin_path = os.path.join(args.data_dir, "queries.bin")
        query_embeddings = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 512)

    print(f"Loaded {query_embeddings.shape[0]} query embeddings "
          f"(dim={query_embeddings.shape[1]})\n")

    run_benchmark(
        args.db_name, query_embeddings, args.n_queries, args.k
    )


if __name__ == "__main__":
    main()
