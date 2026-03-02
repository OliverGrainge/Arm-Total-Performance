"""
Generate an animated GIF comparing AoS vs SoA memory access patterns.

Visual convention: left-to-right = contiguous addresses.

  AoS panel (top): rows = particles, cols = struct fields
    Reading across one row shows the contiguous 64-byte struct in memory.
    First 6 fields (x,y,z,vx,vy,vz) are hot; last 10 are cold/wasted.
    At step i the whole row for particle i is fetched — but only 6 cells used.

  SoA panel (bottom): rows = field arrays, cols = particle indices
    Reading across one row shows the contiguous array for that field.
    At step i, one cell per row is accessed; all loaded cells are used.

Output: ../assets/aos_vs_soa_memory.gif
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BG         = "#1e2127"
C_CELL_DARK  = "#2b3038"
C_CELL_EDGE  = "#3e4450"
C_HOT        = "#56b6c2"   # teal  — hot field, not yet visited
C_HOT_ACTIVE = "#98c379"   # green — hot field, currently accessed
C_COLD_DIM   = "#3a2020"   # very dark red — cold field, not yet visited
C_COLD_HI    = "#e06c75"   # red   — cold field loaded but wasted
C_TRAIL_HOT  = "#1d474d"   # dim teal — hot, previously accessed
C_TRAIL_COLD = "#2d1515"   # dim red  — cold, previously wasted
C_CACHELINE  = "#e5c07b"   # yellow — cache-line boundary
C_TEXT       = "#abb2bf"
C_LABEL      = "#e5c07b"

# ── Constants ──────────────────────────────────────────────────────────────────
N            = 16   # particles  (= 1 cache line of floats → clean SoA alignment)
HOT_FIELDS   = 6    # x y z vx vy vz
COLD_FIELDS  = 10   # mass chg tmp prs eng dns sx sy sz pad
TOTAL_FIELDS = 16   # HOT + COLD = one cache line (16 × 4 B = 64 B)

FIELD_NAMES = ["x", "y", "z", "vx", "vy", "vz",
               "mass", "chg", "tmp", "prs",
               "eng", "dns", "sx", "sy", "sz", "pad"]
HOT_NAMES   = FIELD_NAMES[:HOT_FIELDS]


# ── Helpers ────────────────────────────────────────────────────────────────────

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=C_BG)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def save_gif(frames, path, fps=5):
    dur = [int(1000 / fps)] * len(frames)
    dur[-1] = 2500
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   loop=0, duration=dur, optimize=False)
    print(f"  {len(frames)} frames  →  {path}")


# ── AoS panel ─────────────────────────────────────────────────────────────────

def draw_aos(ax, step):
    """
    rows = N particles  (Y-axis: p[0] at top, p[N-1] at bottom)
    cols = TOTAL_FIELDS  (X-axis: left = hot fields, right = cold/wasted)

    Reading left→right across any row is contiguous memory for that struct.
    Each row is exactly one 64-byte cache line.
    """
    rows, cols = N, TOTAL_FIELDS

    ax.set_xlim(-1.8, cols + 0.3)
    ax.set_ylim(-1.8, rows + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Array-of-Structures  (AoS)",
                 color=C_LABEL, fontsize=11, fontweight="bold", pad=4)

    # ── cells ─────────────────────────────────────────────────────────────────
    for p in range(rows):
        for f in range(cols):
            is_hot = f < HOT_FIELDS
            if p < step:
                fc = C_TRAIL_HOT  if is_hot else C_TRAIL_COLD
                ec, lw = C_CELL_EDGE, 0.4
            elif p == step:
                fc = C_HOT_ACTIVE if is_hot else C_COLD_HI
                ec, lw = "white", 1.5
            else:
                fc = C_CELL_DARK  if is_hot else C_COLD_DIM
                ec, lw = C_CELL_EDGE, 0.4

            ax.add_patch(plt.Rectangle((f, rows - 1 - p), 1, 1,
                                       fc=fc, ec=ec, lw=lw, zorder=3 if p == step else 2))

    # ── row labels (particle index) ────────────────────────────────────────────
    for p in range(rows):
        ax.text(-0.2, rows - 0.5 - p,
                f"p[{p}]",
                color=C_LABEL if p == step else C_TEXT,
                fontsize=6.5, ha="right", va="center",
                fontweight="bold" if p == step else "normal")

    # ── column labels (field names, top) ──────────────────────────────────────
    for f in range(cols):
        c = C_HOT if f < HOT_FIELDS else C_COLD_HI
        ax.text(f + 0.5, rows + 0.6, FIELD_NAMES[f],
                color=c, fontsize=6, ha="center", va="bottom",
                fontweight="bold", rotation=45)

    # ── hot / cold bracket labels ──────────────────────────────────────────────
    ax.text(HOT_FIELDS / 2,       rows + 2.1, "hot  (24 B used)",
            color=C_HOT,    fontsize=7, ha="center", va="center")
    ax.text((HOT_FIELDS + cols) / 2, rows + 2.1, "cold  (40 B wasted)",
            color=C_COLD_HI, fontsize=7, ha="center", va="center")

    # bracket lines under the labels
    for x0, x1, color in [
        (0,          HOT_FIELDS,   C_HOT),
        (HOT_FIELDS, TOTAL_FIELDS, C_COLD_HI),
    ]:
        ax.plot([x0, x1], [rows + 1.8, rows + 1.8], color=color, lw=1.2)
        ax.plot([x0, x0], [rows + 1.8, rows + 1.6], color=color, lw=1.2)
        ax.plot([x1, x1], [rows + 1.8, rows + 1.6], color=color, lw=1.2)

    # ── vertical divider between hot and cold ─────────────────────────────────
    ax.plot([HOT_FIELDS, HOT_FIELDS], [-0.05, rows + 0.05],
            color=C_CACHELINE, lw=1.2, linestyle=":", zorder=5)

    # ── cache-line box around current row ─────────────────────────────────────
    y = rows - 1 - step
    ax.add_patch(plt.Rectangle(
        (-0.08, y - 0.08), cols + 0.16, 1.16,
        fc="none", ec=C_CACHELINE, lw=2.0, zorder=6
    ))
    ax.text(cols + 0.2, y + 0.5, "64 B\ncache\nline",
            color=C_CACHELINE, fontsize=6.5, ha="left", va="center",
            linespacing=1.3)


# ── SoA panel ─────────────────────────────────────────────────────────────────

def draw_soa(ax, step):
    """
    rows = HOT_FIELDS  (Y-axis: one row per field array)
    cols = N particles  (X-axis: consecutive particle indices)

    Reading left→right across any row is contiguous memory for that array.
    For N=16 each row is exactly one 64-byte cache line.
    The first access loads the whole row; subsequent accesses are L1C hits.
    """
    rows, cols = HOT_FIELDS, N

    ax.set_xlim(-1.8, cols + 1.5)
    # Extend bottom so this panel is the same physical height as the AoS panel
    # (AoS ylim range ≈ 20.3; match that here so cells stay the same size).
    ax.set_ylim(-12.5, rows + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Structure-of-Arrays  (SoA)",
                 color=C_LABEL, fontsize=11, fontweight="bold", pad=4)

    # ── cells ─────────────────────────────────────────────────────────────────
    for f in range(rows):
        for p in range(cols):
            if p < step:
                fc = C_TRAIL_HOT
                ec, lw = C_CELL_EDGE, 0.4
            elif p == step:
                fc = C_HOT_ACTIVE
                ec, lw = "white", 1.5
            else:
                fc = C_CELL_DARK
                ec, lw = C_CELL_EDGE, 0.4

            ax.add_patch(plt.Rectangle((p, rows - 1 - f), 1, 1,
                                       fc=fc, ec=ec, lw=lw, zorder=3 if p == step else 2))

    # ── row labels (array name) ────────────────────────────────────────────────
    for f in range(rows):
        ax.text(-0.2, rows - 0.5 - f, HOT_NAMES[f],
                color=C_HOT, fontsize=8, ha="right", va="center",
                fontweight="bold")

    # ── cache-line boxes (one per row = entire array in one 64-byte load) ─────
    for f in range(rows):
        y = rows - 1 - f
        # bright on first access (step 0), dim afterwards (data already resident)
        ec  = C_CACHELINE if step == 0 else C_TRAIL_HOT
        lw  = 1.8         if step == 0 else 0.8
        alp = 0.9         if step == 0 else 0.35
        ax.add_patch(plt.Rectangle(
            (-0.08, y - 0.08), cols + 0.16, 1.16,
            fc="none", ec=ec, lw=lw, linestyle="--", zorder=5, alpha=alp
        ))

    # ── "all used" label at top ────────────────────────────────────────────────
    ax.text(cols / 2, rows + 1.0,
            "every cache line: 64 B loaded — 64 B used",
            color=C_HOT, fontsize=7, ha="center", va="center")

    ax.plot([0, cols], [rows + 0.7, rows + 0.7], color=C_HOT, lw=1.2)
    ax.plot([0,   0],  [rows + 0.7, rows + 0.5], color=C_HOT, lw=1.2)
    ax.plot([cols, cols], [rows + 0.7, rows + 0.5], color=C_HOT, lw=1.2)


# ── Main ──────────────────────────────────────────────────────────────────────

def make_gif():
    print(f"AoS vs SoA: {N} particles → {N} frames")
    frames = []

    for step in range(N):
        fig = plt.figure(figsize=(16, 8), facecolor=C_BG)
        fig.patch.set_facecolor(C_BG)

        # Side-by-side landscape layout.
        # Both panels have 16 data columns; equal width_ratios give matching
        # cell sizes. SoA ylim is extended so both panels are the same height.
        gs = fig.add_gridspec(
            1, 2,
            wspace=0.25,
            left=0.08, right=0.97, top=0.88, bottom=0.05,
        )
        ax_aos = fig.add_subplot(gs[0, 0])
        ax_soa = fig.add_subplot(gs[0, 1])

        draw_aos(ax_aos, step)
        draw_soa(ax_soa, step)

        fig.suptitle(
            f"Memory layout — particle  i = {step}  (step {step + 1} / {N})\n"
            f"p[i].x += p[i].vx * dt;   p[i].y += p[i].vy * dt;   p[i].z += p[i].vz * dt",
            color=C_TEXT, fontsize=9, y=0.97,
        )

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, os.path.join(ASSETS, "aos_vs_soa_memory.gif"))


if __name__ == "__main__":
    print("Generating aos_vs_soa_memory.gif …")
    make_gif()
    print("Done.")
