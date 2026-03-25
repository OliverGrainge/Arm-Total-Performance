#!/usr/bin/env bash
# ------------------------------------------------------------------
# setup.sh — One-shot environment setup for all tutorials
#
# Targets: Amazon Linux 2023 on AWS Graviton (aarch64)
# Covers:  Tutorials 1, 2, 3, and 6
# Usage:   chmod +x setup.sh && ./setup.sh
# ------------------------------------------------------------------

set -euo pipefail

echo "========================================="
echo " Arm-Total-Performance — Full Setup"
echo "========================================="

# ------------------------------------------------------------------
# 1. System packages (dnf — Amazon Linux 2023)
# ------------------------------------------------------------------
echo ""
echo "[1/5] Installing system packages..."

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
    libffi-devel \
    postgresql16-server \
    postgresql16-server-devel \
    postgresql16-contrib \
    nodejs \
    npm

# ------------------------------------------------------------------
# 2. PostgreSQL + pgvector (Tutorial 6)
# ------------------------------------------------------------------
echo ""
echo "[2/5] Setting up PostgreSQL..."

# Initialise the database cluster if not already done
if [ ! -f /var/lib/pgsql/data/PG_VERSION ]; then
    sudo postgresql-setup --initdb
fi

sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create a superuser role matching the current OS user (idempotent)
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$(whoami)'" | grep -q 1; then
    sudo -u postgres createuser --superuser "$(whoami)"
fi

# Install pgvector extension from source
echo ""
echo "  -> Building pgvector from source..."
PGVECTOR_DIR=$(mktemp -d)
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
pushd "$PGVECTOR_DIR" > /dev/null
make PG_CONFIG=/usr/bin/pg_config
sudo make install PG_CONFIG=/usr/bin/pg_config
popd > /dev/null
rm -rf "$PGVECTOR_DIR"

# Verify pgvector is available
echo "  -> Verifying pgvector installation..."
psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector" -c "DROP EXTENSION vector" 2>/dev/null \
    && echo "  [OK] pgvector extension is available" \
    || echo "  [WARNING] pgvector extension could not be loaded"

# ------------------------------------------------------------------
# 3. Python packages (all tutorials, single pip install)
# ------------------------------------------------------------------
echo ""
echo "[3/5] Installing Python packages..."

pip3 install --upgrade pip

pip3 install \
    numpy \
    matplotlib \
    Pillow \
    torch \
    transformers \
    open-clip-torch \
    'gradio==4.44.1' \
    'markupsafe>=2.0,<3' \
    'jinja2>=3.1' \
    'huggingface_hub<0.25' \
    psycopg2-binary

# ------------------------------------------------------------------
# 4. Node.js packages for Tutorial 6
# ------------------------------------------------------------------
echo ""
echo "[4/5] Installing Node.js packages for tutorial_6..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR/tutorial_6" > /dev/null
npm install
popd > /dev/null

# ------------------------------------------------------------------
# 5. Verify key tools
# ------------------------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."
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
check pip3
check cmake
check g++
check node
check npm
check psql

echo ""
python3 -c "
packages = [
    'numpy', 'matplotlib', 'PIL', 'torch',
    'transformers', 'open_clip', 'gradio', 'psycopg2'
]
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
echo " Notes:"
echo "  - Tutorials 1-3: cd into the tutorial directory and run:"
echo "      cmake -S . -B build && cmake --build build --parallel"
echo "  - Tutorial 6: load data into pgvector by running:"
echo "      python3 tutorial_6/scripts/setup_data.py"
echo "  - ATP must be installed separately."
echo "========================================="
