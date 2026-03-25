#!/bin/bash
# Run a fixed GET benchmark against Redis.
# Uses pipelining (-P 32) to maximise throughput and reduce
# per-command networking overhead, so memory access dominates.

set -e

N_OPS=${1:-5000000}

echo "Running Redis GET benchmark (${N_OPS} operations, pipeline=32)..."
echo ""

redis-benchmark -t get -n "${N_OPS}" -r 1000000 -d 1024 -P 32
