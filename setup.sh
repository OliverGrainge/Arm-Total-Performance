#!/usr/bin/env bash
# ------------------------------------------------------------------
# setup.sh — One-shot environment setup for all tutorials
#
# Targets: Amazon Linux 2023 on AWS Graviton (aarch64)
# Covers:  Tutorials 1, 2, 3, and 4 (system deps + Python venv)
# Usage:   chmod +x setup.sh && ./setup.sh
#
# Note:    Tutorial 4 requires Redis, which is built from source
#          as part of that tutorial's instructions.
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "========================================="
echo " Arm-Total-Performance — Full Setup"
echo "========================================="

# ------------------------------------------------------------------
# 1. System packages (dnf — Amazon Linux 2023)
# ------------------------------------------------------------------
echo ""
echo "[1/3] Installing system packages..."

sudo dnf update -y

sudo dnf install -y \
    git \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    cmake \
    make \
    openssl-devel \
    libffi-devel

# ------------------------------------------------------------------
# 2. Python virtual environment + packages
# ------------------------------------------------------------------
echo ""
echo "[2/3] Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  -> Created venv at $VENV_DIR"
else
    echo "  -> venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# ------------------------------------------------------------------
# 3. Verify key tools
# ------------------------------------------------------------------
echo ""
echo "[3/3] Verifying installation..."
echo ""

check() {
    if command -v "$1" &> /dev/null; then
        echo "  [OK] $1 — $($1 --version 2>&1 | head -1)"
    else
        echo "  [MISSING] $1"
    fi
}

check git
check python3
check pip
check cmake
check g++

echo ""
python3 -c "
packages = ['numpy', 'matplotlib', 'PIL', 'torch', 'transformers', 'open_clip']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  [OK] {pkg}')
    except ImportError:
        print(f'  [MISSING] {pkg}')
"

echo ""
echo "========================================="
echo " Setup complete!"
echo ""
echo " To activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Notes:"
echo "  - Tutorials 1-3: cd into the tutorial directory and run:"
echo "      cmake -S . -B build && cmake --build build --parallel"
echo "  - Tutorial 4: Redis is built from source as part of the"
echo "    tutorial instructions."
echo "  - ATP must be installed separately."
echo "========================================="
