#!/usr/bin/env python3
"""
Visualise the galaxy particle simulation from Tutorial 2.

Reads a binary snapshot file produced by aos_baseline or soa_optimized
when run with the --visualize flag, then writes into assets/:
  assets/<stem>.gif  — animated GIF of all frames (differential rotation)

Run this script from the tutorial_2/ directory:
    python3 scripts/visualize.py                      # defaults to build/galaxy_aos.bin
    python3 scripts/visualize.py build/galaxy_soa.bin

Requirements: numpy, matplotlib, Pillow
    pip install numpy matplotlib Pillow
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary format (written by C++ --visualize)
#   int32  n_particles   — number of subsampled particles per frame
#   int32  n_frames      — total number of frames
#   per frame:
#     n_particles * float32  x
#     n_particles * float32  y
#     n_particles * float32  z
# ---------------------------------------------------------------------------

def read_snapshots(path):
    with open(path, 'rb') as f:
        n_particles, n_frames = struct.unpack('ii', f.read(8))
        frames = []
        for _ in range(n_frames):
            nbytes = n_particles * 4
            x = np.frombuffer(f.read(nbytes), dtype=np.float32).copy()
            y = np.frombuffer(f.read(nbytes), dtype=np.float32).copy()
            z = np.frombuffer(f.read(nbytes), dtype=np.float32).copy()
            frames.append((x, y, z))
    print(f"Loaded {n_frames} frames, {n_particles} particles each.")
    return frames


def make_gif(frames, out_path):
    """Render all frames as an animated GIF showing differential rotation."""
    # Compute per-particle radius from the initial frame for consistent colouring.
    x0, y0, _ = frames[0]
    r0 = np.hypot(x0, y0)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.axis('off')
    fig.tight_layout(pad=0)

    x, y, _ = frames[0]
    sc = ax.scatter(x, y, c=r0, cmap='plasma', s=0.3, alpha=0.3, linewidths=0,
                    vmin=r0.min(), vmax=r0.max())

    def update(frame_idx):
        xi, yi, _ = frames[frame_idx]
        sc.set_offsets(np.column_stack([xi, yi]))
        return (sc,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=True
    )
    ani.save(out_path, writer='pillow', fps=12, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    bin_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/galaxy_aos.bin")

    if not bin_path.exists():
        print(f"Error: file not found: {bin_path}")
        print("Run the binary with --visualize first, e.g.:")
        print("  cd build && ./aos_baseline --visualize")
        sys.exit(1)

    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    frames = read_snapshots(bin_path)

    stem = bin_path.stem                      # e.g. "galaxy_aos"
    make_gif(frames, assets / f"{stem}.gif")
