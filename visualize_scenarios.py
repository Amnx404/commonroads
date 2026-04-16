"""
visualize_scenarios.py
======================
Render CommonRoad merge scenarios using MPRenderer and compare curvature designs.

3 main lanes (y = 0, 4, 8 m) + on-ramp (y = -10 → 0) with IDM traffic.
Produces:
  outputs/scenario_comparison.png  – grid: designs × (3 timestep snapshots + TTC plot)
  outputs/gif_<label>.gif           – animated traffic flow per design (20 s)
"""
from __future__ import annotations

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from PIL import Image
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

from simulation import run_simulation, SimResult
from compare_designs import (score_design, rank_designs,
                              TTC_TAU, TTC_CAP,
                              BG, PANEL_BG, FG, LANE_CLR, WARN, COLORS)

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── designs to compare (vary curvature, fixed merge_length) ──────────────────
VIS_DESIGNS = [
    dict(label="κ=0.2  straight ramp",  curvature=0.2, merge_length=120),
    dict(label="κ=0.5  standard curve", curvature=0.5, merge_length=120),
    dict(label="κ=0.8  S-curve ramp",   curvature=0.8, merge_length=120),
]

# Timesteps to snapshot: approach (t=10), active merge (t=70), post-merge (t=140)
KEY_STEPS  = [10, 70, 140]
PHASE_LBLS = ["Approach", "Merge", "Post-merge"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _plot_limits(sim: SimResult, x_pad: float = 30.0, y_pad: float = 6.0):
    """Bounding box for MPRenderer: tight around merge zone + ramp."""
    ml = sim.merge_length
    # x: show from a bit before start of ramp to a bit past merge end
    x_min = -x_pad
    x_max = ml + 150.0
    # y: cover lane 0 (y=0) down to ramp bottom (y≈-10) and up to lane 2 (y=8)
    y_min = -12.0 - y_pad
    y_max =  10.0 + y_pad
    return [x_min, x_max, y_min, y_max]


def _render_frame(scenario, t: int, ax, plot_limits):
    """Draw road + vehicles at timestep t using CommonRoad MPRenderer."""
    renderer = MPRenderer(ax=ax, plot_limits=plot_limits)
    dp = MPDrawParams()
    dp.time_begin = t
    dp.time_end   = t
    renderer.draw_scenario(scenario, draw_params=dp)
    renderer.render()
    # Overlay timestamp
    ax.text(0.02, 0.96, f"t = {t * 0.1:.1f} s",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="#222244", alpha=0.8))


def _fig_to_rgb(fig) -> np.ndarray:
    """Convert a matplotlib figure to an (H, W, 3) uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, facecolor=fig.get_facecolor())
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ── multi-panel comparison figure ────────────────────────────────────────────

def create_comparison_figure(designs_data: list, out_path: Path):
    """
    Grid figure: rows = designs, cols = 3 road snapshots + TTC time-series.
    designs_data: list of (label, SimResult, DesignScore)
    """
    n      = len(designs_data)
    n_snap = len(KEY_STEPS)
    n_cols = n_snap + 1   # snapshots + TTC plot

    fig = plt.figure(figsize=(5.5 * n_cols, 4.5 * n), facecolor=BG)
    fig.suptitle(
        "Highway merge: 3 lanes + on-ramp  ·  Traffic flow & safety comparison",
        color=FG, fontsize=13, y=0.995)

    gs = gridspec.GridSpec(n, n_cols, figure=fig,
                           hspace=0.38, wspace=0.22,
                           left=0.04, right=0.98, top=0.96, bottom=0.04)

    clrs = COLORS[:n]

    for row, (label, sim, score) in enumerate(designs_data):
        pl = _plot_limits(sim)

        # ── road snapshots ────────────────────────────────────────────────
        for col, (t, phase) in enumerate(zip(KEY_STEPS, PHASE_LBLS)):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#111122")
            _render_frame(sim.scenario, t, ax, pl)
            ax.set_title(phase, color=FG, fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(label, color=clrs[row], fontsize=9, labelpad=5,
                              fontweight="bold")
            for sp in ax.spines.values():
                sp.set_edgecolor(LANE_CLR)
            ax.tick_params(colors=FG, labelsize=6)

        # ── TTC time-series ───────────────────────────────────────────────
        ax_ttc = fig.add_subplot(gs[row, n_snap])
        ax_ttc.set_facecolor(PANEL_BG)
        ax_ttc.plot(score.times, score.ttc_series, color=clrs[row], lw=1.4)
        ax_ttc.fill_between(score.times, 0, score.ttc_series,
                            where=(score.ttc_series < TTC_TAU),
                            color=clrs[row], alpha=0.25)
        ax_ttc.axhline(TTC_TAU, color=WARN, lw=0.9, ls="--", alpha=0.85)
        ax_ttc.set_ylim(0, TTC_CAP + 0.5)
        ax_ttc.set_xlim(score.times[0], score.times[-1])
        ax_ttc.set_ylabel("Min TTC (s)", color=FG, fontsize=8)
        ax_ttc.set_xlabel("Time (s)",    color=FG, fontsize=8)
        ax_ttc.set_title("TTC over 20 s", color=FG, fontsize=9)
        ax_ttc.tick_params(colors=FG, labelsize=7)
        for sp in ax_ttc.spines.values():
            sp.set_edgecolor(LANE_CLR)
        # Summary annotation
        ax_ttc.text(0.97, 0.95,
                    f"min TTC = {score.min_ttc:.2f} s\n"
                    f"TIT     = {score.tit:.2f} s²\n"
                    f"unsafe  = {score.pct_unsafe:.1f}%",
                    transform=ax_ttc.transAxes, color=FG, fontsize=7.5,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG,
                              ec=LANE_CLR, alpha=0.9))

    fig.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure → {out_path}")


# ── animated GIF ──────────────────────────────────────────────────────────────

def create_scenario_gif(sim: SimResult, label: str, out_path: Path,
                        frame_step: int = 4, fps: int = 12):
    """
    Animated GIF of traffic flowing through the merge over 20 seconds.
    frame_step=4 → every 0.4 s real-time → 50 frames at 12 fps ≈ 4 s GIF.
    """
    import imageio.v2 as imageio

    pl       = _plot_limits(sim, x_pad=20, y_pad=4)
    n_steps  = len(sim.frames)
    ts       = list(range(0, n_steps, frame_step))

    print(f"    Rendering {len(ts)} frames for '{label}'…")
    frames = []
    for t in ts:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("#111122")
        ax.set_facecolor("#111122")
        ax.set_title(f"{label}   ·   t = {t * 0.1:.1f} s",
                     color="white", fontsize=10, pad=5)
        _render_frame(sim.scenario, t, ax, pl)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555577")
        fig.tight_layout(pad=0.5)
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(str(out_path), frames, fps=fps, loop=0)
    print(f"    GIF → {out_path}  ({len(frames)} frames)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Running IDM simulations (3 lanes + on-ramp, 20 s)…")
    designs_data = []

    for d in VIS_DESIGNS:
        print(f"  {d['label']}")
        sim   = run_simulation(merge_length=d["merge_length"],
                               curvature=d["curvature"])
        score = score_design(d["label"], sim)
        designs_data.append((d["label"], sim, score))

    ranked = rank_designs([s for _, _, s in designs_data])
    print("\n  Safety ranking:")
    for i, s in enumerate(ranked, 1):
        print(f"    {i}. {s.label:<30}  "
              f"min_TTC={s.min_ttc:.2f}s  "
              f"TIT={s.tit:.2f}s²  "
              f"unsafe={s.pct_unsafe:.1f}%")

    print("\nGenerating comparison figure…")
    create_comparison_figure(designs_data, OUT / "scenario_comparison.png")

    print("\nGenerating animated GIFs…")
    for d, (label, sim, _) in zip(VIS_DESIGNS, designs_data):
        slug = (label.replace("κ=", "k").replace(" ", "_")
                     .replace(".", "_").replace("/", "_")
                     .strip("_"))
        create_scenario_gif(sim, label, OUT / f"gif_{slug}.gif")

    print("\nDone.  Outputs in ./outputs/")


if __name__ == "__main__":
    main()
