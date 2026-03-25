"""
Generate animated GIFs illustrating memory access patterns for:
  1. Naive matrix multiplication  (naive_memory_access.gif)
  2. Tiled matrix multiplication   (tiled_memory_access.gif)

Both use M=K=N=6 so total work (216 multiply-adds) is identical.
Every single memory access is shown — no sub-sampling.

Output is written to ../assets/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import io

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_BG        = "#1e2127"
C_CELL_DARK = "#2b3038"
C_CELL_EDGE = "#3e4450"
C_ACTIVE_A  = "#56b6c2"   # teal
C_ACTIVE_B  = "#e06c75"   # red
C_ACTIVE_C  = "#98c379"   # green
C_TRAIL_A   = "#1d474d"
C_TRAIL_B   = "#4d1f23"
C_TRAIL_C   = "#1f3d1f"
C_TILE_BOX  = "#e5c07b"   # tile boundary
C_TILE_FILL = "#2a3a50"
C_TEXT      = "#abb2bf"
C_LABEL     = "#e5c07b"

# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                facecolor=C_BG)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def draw_matrix(ax, rows, cols, title,
                highlight=None,
                trail=None, trail_color=None,
                hi_color="#ffffff",
                tile_boxes=None):       # list of (r0,c0,rlen,clen) dashed boxes
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # base fill
    ax.add_patch(plt.Rectangle((0, 0), cols, rows,
                                fc=C_CELL_DARK, ec="none", zorder=0))

    # trail
    if trail:
        seen = set()
        for (r, c) in trail:
            key = (r, c)
            if key not in seen:
                seen.add(key)
                ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                                           fc=trail_color, ec="none",
                                           zorder=1))

    # tile boxes (dashed outlines)
    if tile_boxes:
        for (r0, c0, rlen, clen) in tile_boxes:
            ax.add_patch(plt.Rectangle(
                (c0, rows - r0 - rlen), clen, rlen,
                fc=C_TILE_FILL, ec=C_TILE_BOX, lw=1.8,
                linestyle="--", zorder=2, alpha=0.55))

    # grid
    for r in range(rows + 1):
        ax.axhline(r, color=C_CELL_EDGE, lw=0.5, zorder=3)
    for c in range(cols + 1):
        ax.axvline(c, color=C_CELL_EDGE, lw=0.5, zorder=3)

    # active cell
    if highlight is not None:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                                   fc=hi_color, ec="white", lw=1.8,
                                   zorder=4))

    ax.set_title(title, color=C_LABEL, fontsize=10, fontweight="bold", pad=3)


def draw_mem_strip(ax, total, cur_idx, cur_color,
                   trail_indices=None, trail_color=None,
                   label=""):
    """Flat 1-D memory strip.  cur_idx is the address currently accessed."""
    ax.set_xlim(-0.5, total + 0.5)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.15), total, 0.7,
                                fc=C_CELL_DARK, ec=C_CELL_EDGE,
                                lw=0.4, zorder=0))

    # trail dots
    if trail_indices:
        seen = set()
        for idx in trail_indices:
            if idx not in seen:
                seen.add(idx)
                ax.add_patch(plt.Rectangle((idx, 0.15), 1, 0.7,
                                           fc=trail_color, ec="none",
                                           zorder=1))

    # current access
    ax.add_patch(plt.Rectangle((cur_idx, 0.15), 1, 0.7,
                               fc=cur_color, ec="white", lw=1.0,
                               zorder=2))

    if label:
        ax.text(-1.5, 0.5, label, color=C_TEXT, fontsize=8,
                va="center", ha="right", fontweight="bold")


def save_gif(frames, path, fps=10):
    dur = [int(1000 / fps)] * len(frames)
    dur[-1] = 1200
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   loop=0, duration=dur, optimize=False)
    print(f"  {len(frames)} frames  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# GIF 1 – Naive  (loop order: i → j → k,  every step)
# ══════════════════════════════════════════════════════════════════════════════
def make_naive_gif():
    M = K = N = 6
    print(f"Naive: {M}×{K}×{N} = {M*K*N} frames")

    trail_A, trail_B, trail_C = [], [], []
    frames = []

    def make_frame(i, j, k):
        fig = plt.figure(figsize=(11, 6), facecolor=C_BG)
        fig.patch.set_facecolor(C_BG)

        # ── grid layout ──────────────────────────────────────────────────────
        #   col 0: A (M rows, K cols)
        #   col 2: B (K rows, N cols)
        #   col 4: C (M rows, N cols)
        #   row 0: matrices   row 1: memory strips
        gs = fig.add_gridspec(
            2, 6,
            width_ratios=[K, 0.6, N, 0.6, N, 0.3],
            height_ratios=[max(M, K), 0.5],
            hspace=0.02, wspace=0.3,
            left=0.07, right=0.98, top=0.86, bottom=0.08,
        )
        ax_A  = fig.add_subplot(gs[0, 0])
        ax_B  = fig.add_subplot(gs[0, 2])
        ax_C  = fig.add_subplot(gs[0, 4])
        ax_mA = fig.add_subplot(gs[1, 0])
        ax_mB = fig.add_subplot(gs[1, 2])
        ax_mC = fig.add_subplot(gs[1, 4])

        # ── matrices ─────────────────────────────────────────────────────────
        draw_matrix(ax_A, M, K, f"A  [{M}×{K}]",
                    highlight=(i, k),
                    trail=trail_A, trail_color=C_TRAIL_A,
                    hi_color=C_ACTIVE_A)

        draw_matrix(ax_B, K, N, f"B  [{K}×{N}]",
                    highlight=(k, j),
                    trail=trail_B, trail_color=C_TRAIL_B,
                    hi_color=C_ACTIVE_B)

        draw_matrix(ax_C, M, N, f"C  [{M}×{N}]",
                    highlight=(i, j),
                    trail=trail_C, trail_color=C_TRAIL_C,
                    hi_color=C_ACTIVE_C)

        # ── memory strips ────────────────────────────────────────────────────
        # A[i,k] → flat index i*K + k  (increments by 1 each k-step)
        idx_a = i * K + k
        draw_mem_strip(ax_mA, M * K, idx_a, C_ACTIVE_A,
                       trail_indices=[r*K+c for r,c in trail_A],
                       trail_color=C_TRAIL_A, label="A")

        # B[k,j] → flat index k*N + j  (jumps by N each k-step ← the problem)
        idx_b = k * N + j
        draw_mem_strip(ax_mB, K * N, idx_b, C_ACTIVE_B,
                       trail_indices=[r*N+c for r,c in trail_B],
                       trail_color=C_TRAIL_B, label="B")

        idx_c = i * N + j
        draw_mem_strip(ax_mC, M * N, idx_c, C_ACTIVE_C,
                       trail_indices=[r*N+c for r,c in trail_C],
                       trail_color=C_TRAIL_C, label="C")

        # ── "×" and "+=" symbols between matrices ────────────────────────────
        posA = ax_A.get_position()
        posB = ax_B.get_position()
        posC = ax_C.get_position()
        mid_y = (posA.y0 + posA.y1) / 2

        fig.text((posA.x1 + posB.x0) / 2, mid_y, "×",
                 color=C_TEXT, fontsize=14, ha="center", va="center")
        fig.text((posB.x1 + posC.x0) / 2, mid_y, "→",
                 color=C_TEXT, fontsize=12, ha="center", va="center")

        # ── legend ───────────────────────────────────────────────────────────
        patches = [
            mpatches.Patch(color=C_ACTIVE_A,
                           label=f"A[{i},{k}]  addr {idx_a}  (sequential +1)"),
            mpatches.Patch(color=C_ACTIVE_B,
                           label=f"B[{k},{j}]  addr {idx_b}  (stride +{N} per k-step)"),
            mpatches.Patch(color=C_ACTIVE_C,
                           label=f"C[{i},{j}]  addr {idx_c}"),
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=8,
                   framealpha=0.3, facecolor=C_CELL_DARK,
                   edgecolor=C_CELL_EDGE, labelcolor=C_TEXT)

        step = len(frames) + 1
        fig.suptitle(
            f"Naive MatMul  ·  loop: for i → for j → for k\n"
            f"step {step}/{M*K*N}   i={i}  j={j}  k={k}   "
            f"C[{i},{j}] += A[{i},{k}] × B[{k},{j}]",
            color=C_TEXT, fontsize=10, y=0.97,
        )

        pil = fig_to_pil(fig)
        plt.close(fig)
        return pil

    # full i → j → k loop, every step
    for i in range(M):
        for j in range(N):
            for k in range(K):
                frames.append(make_frame(i, j, k))
                trail_A.append((i, k))
                trail_B.append((k, j))
                trail_C.append((i, j))

    save_gif(frames, os.path.join(ASSETS, "naive_memory_access.gif"), fps=10)


# ══════════════════════════════════════════════════════════════════════════════
# GIF 2 – Tiled  (loop order: i0 → j0 → k0 → i → k → j,  every step)
# ══════════════════════════════════════════════════════════════════════════════
def make_tiled_gif():
    M = K = N = 6
    TILE = 3
    print(f"Tiled: {M}×{K}×{N} TILE={TILE} = {M*K*N} frames")

    trail_A, trail_B, trail_C = [], [], []
    frames = []

    def make_frame(i0, j0, k0, i, k, j):
        i_end = min(i0 + TILE, M)
        j_end = min(j0 + TILE, N)
        k_end = min(k0 + TILE, K)

        fig = plt.figure(figsize=(11, 6), facecolor=C_BG)
        fig.patch.set_facecolor(C_BG)

        gs = fig.add_gridspec(
            2, 6,
            width_ratios=[K, 0.6, N, 0.6, N, 0.3],
            height_ratios=[max(M, K), 0.5],
            hspace=0.02, wspace=0.3,
            left=0.07, right=0.98, top=0.86, bottom=0.08,
        )
        ax_A  = fig.add_subplot(gs[0, 0])
        ax_B  = fig.add_subplot(gs[0, 2])
        ax_C  = fig.add_subplot(gs[0, 4])
        ax_mA = fig.add_subplot(gs[1, 0])
        ax_mB = fig.add_subplot(gs[1, 2])
        ax_mC = fig.add_subplot(gs[1, 4])

        # ── matrices ─────────────────────────────────────────────────────────
        draw_matrix(ax_A, M, K, f"A  [{M}×{K}]  (TILE={TILE})",
                    highlight=(i, k),
                    trail=trail_A, trail_color=C_TRAIL_A,
                    hi_color=C_ACTIVE_A,
                    tile_boxes=[(i0, k0, i_end - i0, k_end - k0)])

        draw_matrix(ax_B, K, N, f"B  [{K}×{N}]  (TILE={TILE})",
                    highlight=(k, j),
                    trail=trail_B, trail_color=C_TRAIL_B,
                    hi_color=C_ACTIVE_B,
                    tile_boxes=[(k0, j0, k_end - k0, j_end - j0)])

        draw_matrix(ax_C, M, N, f"C  [{M}×{N}]  (TILE={TILE})",
                    highlight=(i, j),
                    trail=trail_C, trail_color=C_TRAIL_C,
                    hi_color=C_ACTIVE_C,
                    tile_boxes=[(i0, j0, i_end - i0, j_end - j0)])

        # ── memory strips ────────────────────────────────────────────────────
        idx_a = i * K + k
        idx_b = k * N + j
        idx_c = i * N + j

        # Highlight the full tile footprint in memory as background
        tile_A_addrs = [ii*K + kk
                        for ii in range(i0, i_end)
                        for kk in range(k0, k_end)]
        # B tile accessed with inner j-loop: k0..k_end-1, j0..j_end-1
        # Within one (i,k) step, j sweeps j0→j_end → sequential in memory
        tile_B_addrs = [kk*N + jj
                        for kk in range(k0, k_end)
                        for jj in range(j0, j_end)]

        draw_mem_strip(ax_mA, M * K, idx_a, C_ACTIVE_A,
                       trail_indices=tile_A_addrs,
                       trail_color=C_TILE_FILL, label="A")

        draw_mem_strip(ax_mB, K * N, idx_b, C_ACTIVE_B,
                       trail_indices=tile_B_addrs,
                       trail_color=C_TILE_FILL, label="B")

        draw_mem_strip(ax_mC, M * N, idx_c, C_ACTIVE_C,
                       trail_indices=[ii*N+jj
                                      for ii in range(i0, i_end)
                                      for jj in range(j0, j_end)],
                       trail_color=C_TILE_FILL, label="C")

        # ── "×" and "→" ──────────────────────────────────────────────────────
        posA = ax_A.get_position()
        posB = ax_B.get_position()
        posC = ax_C.get_position()
        mid_y = (posA.y0 + posA.y1) / 2

        fig.text((posA.x1 + posB.x0) / 2, mid_y, "×",
                 color=C_TEXT, fontsize=14, ha="center", va="center")
        fig.text((posB.x1 + posC.x0) / 2, mid_y, "→",
                 color=C_TEXT, fontsize=12, ha="center", va="center")

        # ── legend ───────────────────────────────────────────────────────────
        patches = [
            mpatches.Patch(color=C_ACTIVE_A,
                           label=f"A[{i},{k}]  addr {idx_a}"),
            mpatches.Patch(color=C_ACTIVE_B,
                           label=f"B[{k},{j}]  addr {idx_b}  (inner j → +1 per step)"),
            mpatches.Patch(color=C_ACTIVE_C,
                           label=f"C[{i},{j}]  addr {idx_c}"),
            mpatches.Patch(color=C_TILE_FILL,
                           label=f"current tile footprint  ({TILE}×{TILE})"),
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=8,
                   framealpha=0.3, facecolor=C_CELL_DARK,
                   edgecolor=C_CELL_EDGE, labelcolor=C_TEXT)

        step = len(frames) + 1
        fig.suptitle(
            f"Tiled MatMul  ·  loop: for i0 → j0 → k0 → i → k → j  (TILE={TILE})\n"
            f"step {step}/{M*K*N}   tile(i0={i0},j0={j0},k0={k0})   "
            f"i={i}  k={k}  j={j}",
            color=C_TEXT, fontsize=10, y=0.97,
        )

        pil = fig_to_pil(fig)
        plt.close(fig)
        return pil

    # full tiled loop, every step
    for i0 in range(0, M, TILE):
        for j0 in range(0, N, TILE):
            for k0 in range(0, K, TILE):
                i_end = min(i0 + TILE, M)
                j_end = min(j0 + TILE, N)
                k_end = min(k0 + TILE, K)
                for i in range(i0, i_end):
                    for k in range(k0, k_end):
                        for j in range(j0, j_end):
                            frames.append(make_frame(i0, j0, k0, i, k, j))
                            trail_A.append((i, k))
                            trail_B.append((k, j))
                            trail_C.append((i, j))

    save_gif(frames, os.path.join(ASSETS, "tiled_memory_access.gif"), fps=10)


if __name__ == "__main__":
    print("Generating naive_memory_access.gif …")
    make_naive_gif()
    print("Generating tiled_memory_access.gif …")
    make_tiled_gif()
    print("Done.")
