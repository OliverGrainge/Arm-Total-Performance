#!/bin/bash
# Populate Redis with a large dataset for benchmarking.
# Creates ~1M unique keys with 1KB values (~1GB total in memory).

set -e

echo "Loading data into Redis..."
echo "Creating ~1M keys with 1KB values (~1GB in memory)"
echo ""

redis-benchmark -t set -n 2000000 -r 1000000 -d 1024 -P 32 -q

echo ""
echo "Redis memory usage:"
redis-cli info memory | grep -E "used_memory_human|used_memory_peak_human"
echo ""
echo "Total keys:"
redis-cli dbsize
