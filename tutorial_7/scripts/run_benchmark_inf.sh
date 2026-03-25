#!/bin/bash
# Infinite Redis GET benchmark for profiler attachment.
# Runs continuously until Ctrl+C.

set -e

echo "Redis server PID:"
redis6-cli info server | grep process_id
echo ""
echo "Running infinite GET benchmark (pipeline=32). Press Ctrl+C to stop."
echo ""

while true; do
    redis-benchmark -t get -n 1000000 -r 1000000 -d 1024 -P 32 -q
done
