"""
Generate an animated GIF comparing memory access patterns for:
  Array-of-Structures (AoS) — the inefficient baseline
  Structure-of-Arrays (SoA) — the optimised layout

Demonstrates the cache waste in the particle position-update loop:
  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;

Visual layout (side-by-side, N=16 particles, 16 animation steps):
  Left  — AoS: fields on Y-axis (16 rows), particles on X-axis (16 cols)
          Each column = one particle = one 64-byte cache line
          Top 6 rows (x,y,z,vx,vy,vz) are HOT; bottom 10 are COLD/wasted
  Right — SoA: arrays on Y-axis (6 rows), particles on X-axis (16 cols)
          Each row = one field array = one 64-byte cache line (for N=16)
          Every cell is HOT; 100% cache-line utilisation

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

# ── Colour palette (matching tutorial_1) ──────────────────────────────────────
C_BG         = "#1e2127"
C_CELL_DARK  = "#2b3038"
C_CELL_EDGE  = "#3e4450"
C_HOT        = "#56b6c2"    # teal   — hot field (used)
C_HOT_ACTIVE = "#98c379"    # green  — hot field, currently being accessed
C_COLD_DIM   = "#3a2020"    # very dark red — cold field, not yet visited
C_COLD_HI    = "#e06c75"    # bright red  — cold field loaded but wasted
C_TRAIL_HOT  = "#1d474d"    # dim teal    — hot field, previously accessed
C_TRAIL_COLD = "#2d1515"    # dim red     — cold field, previously wasted
C_CACHELINE  = "#e5c07b"    # yellow      — cache-line boundary
C_MISS       = "#e06c75"    # red    — cache miss label
C_HIT        = "#98c379"    # green  — cache hit label
C_TEXT       = "#abb2bf"
C_LABEL      = "#e5c07b"

# ── Problem constants ──────────────────────────────────────────────────────────
N             = 16    # particles  (= CACHE_LINE_FLOATS for clean alignment)
HOT_FIELDS    = 6     # x, y, z, vx, vy, vz
COLD_FIELDS   = 10    # mass, chg, tmp, prs, eng, dns, sx, sy, sz, pad
TOTAL_FIELDS  = 16    # HOT + COLD  →  one cache line at 4 B/float
FLOAT_B       = 4     # bytes per float
CACHE_LINE_B  = 64    # bytes per cache line

FIELD_NAMES = ["x", "y", "z", "vx", "vy", "vz",
               "mass", "chg", "tmp", "prs",
               "eng", "dns", "sx", "sy", "sz", "pad"]
HOT_NAMES = FIELD_NAMES[:HOT_FIELDS]


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


# ── Panel drawing ──────────────────────────────────────────────────────────────

def draw_aos_panel(ax, step):
    """
    AoS grid: rows = TOTAL_FIELDS (Y-axis), cols = N particles (X-axis).
    Column i = all fields of particle i = one 64-byte cache line.
    Top HOT_FIELDS rows are used; bottom COLD_FIELDS rows are wasted.
    """
    rows = TOTAL_FIELDS
    cols = N

    # Extra padding for labels and annotations
    ax.set_xlim(-0.5, cols + 3.5)
    ax.set_ylim(-4.0, rows + 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Array-of-Structures  (AoS — Baseline)",
                 color=C_LABEL, fontsize=10, fontweight="bold", pad=6)

    # ── hot / cold divider line ───────────────────────────────────────────────
    divider_y = rows - HOT_FIELDS
    ax.plot([-0.1, cols + 0.1], [divider_y, divider_y],
            color=C_CACHELINE, lw=1.5, linestyle="--", zorder=6)

    ax.text(cols + 0.4, divider_y + 0.3,
            "↑ HOT\n  (used)", color=C_HOT, fontsize=7,
            ha="left", va="bottom", linespacing=1.3)
    ax.text(cols + 0.4, divider_y - 0.3,
            "↓ COLD\n  (wasted)", color=C_COLD_HI, fontsize=7,
            ha="left", va="top", linespacing=1.3)

    # ── draw cells ────────────────────────────────────────────────────────────
    for f in range(rows):
        is_hot = (f < HOT_FIELDS)
        for p in range(cols):
            if p < step:
                fc = C_TRAIL_HOT if is_hot else C_TRAIL_COLD
                ec = C_CELL_EDGE
                lw = 0.5
                zo = 2
            elif p == step:
                fc = C_HOT_ACTIVE if is_hot else C_COLD_HI
                ec = "white"
                lw = 1.5
                zo = 4
            else:
                fc = C_CELL_DARK if is_hot else C_COLD_DIM
                ec = C_CELL_EDGE
                lw = 0.5
                zo = 2

            y_pos = rows - 1 - f
            ax.add_patch(plt.Rectangle((p, y_pos), 1, 1,
                                       fc=fc, ec=ec, lw=lw, zorder=zo))

        # Row label (field name)
        is_hot_label_color = C_HOT if is_hot else C_COLD_HI
        ax.text(-0.3, rows - 0.5 - f, FIELD_NAMES[f],
                color=is_hot_label_color, fontsize=6.5,
                ha="right", va="center", fontweight="bold")

    # ── column header for current particle ───────────────────────────────────
    ax.text(step + 0.5, rows + 0.6, f"p[{step}]",
            color=C_LABEL, fontsize=7, ha="center", va="center",
            fontweight="bold")

    # ── cache-line highlight (full column = 64 B) ────────────────────────────
    ax.add_patch(plt.Rectangle(
        (step - 0.12, -0.12), 1.24, rows + 0.24,
        fc="none", ec=C_CACHELINE, lw=2.2, zorder=5
    ))

    # ── "MISS" annotation below column ───────────────────────────────────────
    ax.text(step + 0.5, -0.7, "MISS",
            color=C_MISS, fontsize=7, ha="center", va="center",
            fontweight="bold")

    # ── running cache-miss counter ────────────────────────────────────────────
    misses = step + 1
    ax.text(cols / 2, -2.0,
            f"Cache misses so far: {misses} / {N}  (1 miss per particle — every step!)",
            color=C_MISS, fontsize=8, ha="center", va="center",
            fontweight="bold")

    # ── cache-line utilisation stat ───────────────────────────────────────────
    used_B   = HOT_FIELDS * FLOAT_B
    loaded_B = TOTAL_FIELDS * FLOAT_B
    wasted_B = loaded_B - used_B
    pct      = 100 * used_B // loaded_B
    ax.text(cols / 2, -3.2,
            f"Each cache line: {loaded_B} B loaded  |  {used_B} B useful  |"
            f"  {wasted_B} B wasted  →  {pct}% utilisation",
            color=C_TEXT, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=C_CELL_DARK,
                      ec=C_COLD_HI, lw=1.2))


def draw_soa_panel(ax, step):
    """
    SoA grid: rows = HOT_FIELDS (Y-axis), cols = N particles (X-axis).
    Row f = contiguous array for field f.
    For N=16 each row is exactly one 64-byte cache line.
    First access (step 0) = 6 cache misses (one per array); steps 1-15 = hits.
    """
    rows = HOT_FIELDS
    cols = N

    ax.set_xlim(-0.5, cols + 3.5)
    ax.set_ylim(-4.0, rows + 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Structure-of-Arrays  (SoA — Optimised)",
                 color=C_LABEL, fontsize=10, fontweight="bold", pad=6)

    # ── draw cells ────────────────────────────────────────────────────────────
    for f in range(rows):
        for p in range(cols):
            if p < step:
                fc = C_TRAIL_HOT
                ec = C_CELL_EDGE
                lw = 0.5
                zo = 2
            elif p == step:
                fc = C_HOT_ACTIVE
                ec = "white"
                lw = 1.5
                zo = 4
            else:
                fc = C_CELL_DARK
                ec = C_CELL_EDGE
                lw = 0.5
                zo = 2

            ax.add_patch(plt.Rectangle((p, rows - 1 - f), 1, 1,
                                       fc=fc, ec=ec, lw=lw, zorder=zo))

        # Row label
        ax.text(-0.3, rows - 0.5 - f, HOT_NAMES[f],
                color=C_HOT, fontsize=7.5, ha="right", va="center",
                fontweight="bold")

    # ── cache-line boxes around each row ─────────────────────────────────────
    # For N=16 the entire row IS one cache line.
    # On step 0 they are being loaded (bright dashed); afterwards already resident.
    edge_color = C_CACHELINE if step == 0 else C_TRAIL_HOT
    edge_lw    = 2.0         if step == 0 else 1.0
    alpha      = 0.9         if step == 0 else 0.4
    for f in range(rows):
        ax.add_patch(plt.Rectangle(
            (-0.10, rows - 1 - f - 0.10), cols + 0.20, 1.20,
            fc="none", ec=edge_color, lw=edge_lw,
            linestyle="--", zorder=3, alpha=alpha
        ))

    # ── column header for current particle ───────────────────────────────────
    ax.text(step + 0.5, rows + 0.6, f"[{step}]",
            color=C_LABEL, fontsize=7, ha="center", va="center",
            fontweight="bold")

    # ── cache hit/miss annotation ─────────────────────────────────────────────
    if step == 0:
        label    = f"MISS ×{HOT_FIELDS}  (one per array)"
        lcolor   = C_MISS
    else:
        label    = f"L1C HIT ✓"
        lcolor   = C_HIT

    ax.text(step + 0.5, -0.7, label,
            color=lcolor, fontsize=7, ha="center", va="center",
            fontweight="bold")

    # ── running cache-miss summary ────────────────────────────────────────────
    # All 6 misses happen at step 0; afterwards none.
    total_misses = HOT_FIELDS
    hits_so_far  = step  # accesses after the first (step 0 is the miss)
    ax.text(cols / 2, -2.0,
            f"Cache misses: {total_misses} total  "
            f"(6 arrays × 1 miss each, all at first access)",
            color=C_HIT, fontsize=8, ha="center", va="center",
            fontweight="bold")

    # ── cache-line utilisation stat ────────────────────────────────────────────
    ax.text(cols / 2, -3.2,
            f"Each cache line: {CACHE_LINE_B} B loaded  |  {CACHE_LINE_B} B useful  |"
            f"  0 B wasted  →  100% utilisation",
            color=C_TEXT, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=C_CELL_DARK,
                      ec=C_HOT, lw=1.2))


# ── Main GIF builder ──────────────────────────────────────────────────────────

def make_comparison_gif():
    print(f"AoS vs SoA: {N} particles = {N} frames")
    frames = []

    for step in range(N):
        fig = plt.figure(figsize=(16, 7.5), facecolor=C_BG)
        fig.patch.set_facecolor(C_BG)

        # GridSpec: AoS panel | spacer | SoA panel
        # width_ratios reflect cell count: AoS=16 cols, SoA=16 cols, but AoS
        # is taller so give it slightly more horizontal space for balance.
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[TOTAL_FIELDS, 1.0, TOTAL_FIELDS],
            left=0.06, right=0.97, top=0.88, bottom=0.12,
            wspace=0.15,
        )
        ax_aos = fig.add_subplot(gs[0, 0])
        ax_soa = fig.add_subplot(gs[0, 2])

        draw_aos_panel(ax_aos, step)
        draw_soa_panel(ax_soa, step)

        # ── suptitle ────────────────────────────────────────────────────────
        loop_line = (
            f"p[i].x += p[i].vx * dt;   "
            f"p[i].y += p[i].vy * dt;   "
            f"p[i].z += p[i].vz * dt"
        )
        fig.suptitle(
            f"Memory Layout: AoS vs SoA  ·  hot loop: {loop_line}\n"
            f"step {step + 1}/{N}   processing particle  i = {step}",
            color=C_TEXT, fontsize=10, y=0.97,
        )

        # ── legend ──────────────────────────────────────────────────────────
        patches = [
            mpatches.Patch(color=C_HOT_ACTIVE,
                           label="Hot field (x, y, z, vx, vy, vz) — actively used"),
            mpatches.Patch(color=C_COLD_HI,
                           label="Cold field (mass, chg, …) — loaded but WASTED"),
            mpatches.Patch(color=C_TRAIL_HOT,
                           label="Hot field — previously accessed"),
            mpatches.Patch(color=C_TRAIL_COLD,
                           label="Cold field — previously wasted"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
                   framealpha=0.35, facecolor=C_CELL_DARK, edgecolor=C_CELL_EDGE,
                   labelcolor=C_TEXT, bbox_to_anchor=(0.5, 0.01))

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    save_gif(frames, os.path.join(ASSETS, "aos_vs_soa_memory.gif"), fps=5)


if __name__ == "__main__":
    print("Generating aos_vs_soa_memory.gif …")
    make_comparison_gif()
    print("Done.")
