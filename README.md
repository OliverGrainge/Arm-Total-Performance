# Performance Engineering on Arm

A set of hands-on tutorials for profiling and optimising C++ workloads on Arm using **Arm Performix**.

## Tutorials

| # | Title | Description |
|---|-------|-------------|
| 1 | Top-Down Performance Analysis | Profile and optimise a matrix multiplication using the top-down methodology |
| 2 | Optimising Memory-Bound Workloads | Diagnose cache bottlenecks in a particle simulation with the Memory Access recipe |
| 3 | Instruction Mix Optimisation | Use the Instruction Mix recipe to accelerate a GPT-2 inference workload |
| 4 | Optimising Redis | Profile a live Redis server and fix a TLB bottleneck at the OS level |

## Prerequisites

- **AWS Graviton metal instance** (e.g. `c7g.metal`) running Amazon Linux 2023
- **Arm Performix** — download and install from [Arm's website](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio). This is not included in the setup script and must be installed separately.

## Environment Setup

```bash
git clone <repo-url>
cd arm-total-performance
chmod +x setup.sh
./setup.sh
```

This installs system dependencies (gcc, cmake, etc.), creates a Python virtual environment at `.venv/`, and installs the required Python packages.

Activate the virtual environment before working with the tutorials:

```bash
source .venv/bin/activate
```

To build a tutorial (1-3):

```bash
cd tutorial_<n>
cmake -S . -B build
cmake --build build --parallel
```

Tutorial 4 has its own setup steps documented in its instructions.

## Getting Started

Each tutorial has an `instructions.md` with step-by-step guidance. Start with Tutorial 1 and work through them in order.
