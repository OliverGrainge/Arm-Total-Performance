#!/usr/bin/env python3
"""Infinite benchmark loop for continuous pgvector profiling.

Runs kNN queries in an infinite loop so you can attach a profiler
(e.g. perf, ATP) at any time without worrying about the workload finishing.
Prints the process PID on startup so you can target it directly.

Usage:
    python scripts/benchmark_inf.py
"""

import argparse
import os
import time
import signal
import sys

import numpy as np
import psycopg2


def run_infinite_benchmark(db_name, query_embeddings, k):
    """Run kNN queries in an infinite loop."""
    print(f"PID: {os.getpid()}")
    print(f"Benchmark: infinite loop, k={k}")
    print(f"Database:  {db_name}")
    print("Press Ctrl+C to stop.\n")

    conn = psycopg2.connect(dbname=db_name)
    cur = conn.cursor()

    n_queries = len(query_embeddings)
    iteration = 0
    total_queries = 0
    start = time.perf_counter()

    while True:
        idx = total_queries % n_queries
        emb_str = "[" + ",".join(f"{x:.6f}" for x in query_embeddings[idx]) + "]"
        cur.execute(
            "SELECT id FROM images ORDER BY embedding <-> %s::vector LIMIT %s",
            (emb_str, k),
        )
        cur.fetchall()
        total_queries += 1

        if total_queries % 1000 == 0:
            elapsed = time.perf_counter() - start
            qps = total_queries / elapsed
            print(f"  {total_queries:>8d} queries  ({qps:.1f} queries/sec)")


def main():
    parser = argparse.ArgumentParser(description="Infinite pgvector benchmark loop")
    parser.add_argument("--db-name", default="clip_search", help="PostgreSQL database name")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Data directory")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    args = parser.parse_args()

    query_path = os.path.join(args.data_dir, "query_embeddings.npy")
    if os.path.exists(query_path):
        query_embeddings = np.load(query_path)
    else:
        bin_path = os.path.join(args.data_dir, "queries.bin")
        query_embeddings = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 512)

    print(f"Loaded {query_embeddings.shape[0]} query embeddings "
          f"(dim={query_embeddings.shape[1]})\n")

    run_infinite_benchmark(args.db_name, query_embeddings, args.k)


if __name__ == "__main__":
    main()
