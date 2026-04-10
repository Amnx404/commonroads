"""
visualize_scenarios.py
======================
Visual proof: CommonRoad MPRenderer showing road geometry + IDM cars for each
merge design, side-by-side with safety metrics.

Outputs
-------
outputs/scenario_comparison.png  – 4 designs × (3 snapshots + TTC chart) grid
outputs/gif_<design>.gif         – animated GIF per design (60 frames @ 10 fps)

Usage
-----
    python visualize_scenarios.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio

from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

from simulation import run_simulation, SimResult
from compare_designs import (
    score_design, rank_designs, DesignScore,
    DESIGNS, TTC_TAU, TTC_CAP,
    BG, PANEL_BG, FG, ACCENT, WARN, SAFE_CLR, COLORS, LANE_CLR,
)

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── key timesteps to snapshot ────────────────────────────────────────────────
KEY_STEPS  = [10, 60, 140]
KEY_LABELS = ["Approach", "Merging", "Post-merge"]
GIF_STRIDE = 5      # every 5th step → 60 frames
GIF_FPS    = 10


# ── camera windows ────────────────────────────────────────────────────────────
# Each snapshot uses a fixed view anchored to road geometry (not vehicles),
# so white-space artefacts and aspect distortion are avoided.
# Y is always [-5, 16]: main lane @ y=0, ramp @ y≈10.

Y_LO, Y_HI = -4, 14


def _camera(phase: str, merge_length: float) -> list[float]:
    """Fixed plot limits for each snapshot phase, keyed to the design's geometry."""
    ml = merge_length
    if phase == "Approach":
        # Show the full ramp (x=0 to ml) plus upstream run-up
        return [-80, ml + 80, Y_LO, Y_HI]
    elif phase == "Merging":
        # Zoom on the merge end — last 120 m of ramp + 130 m downstream
        return [ml - 120, ml + 130, Y_LO, Y_HI]
    else:  # Post-merge
        # Downstream flow after the merge end
        return [ml - 30, ml + 270, Y_LO, Y_HI]


def _gif_camera(merge_length: float) -> list[float]:
    """Fixed wide-angle camera for the GIF: full merge zone in view."""
    return [-80, merge_length + 200, Y_LO, Y_HI]


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_state(obs, t):
    if obs.initial_state.time_step == t:
        return obs.initial_state
    if obs.prediction:
        for s in obs.prediction.trajectory.state_list:
            if s.time_step == t:
                return s
    return None


def _make_draw_params(t: int) -> MPDrawParams:
    dp = MPDrawParams()
    dp.time_begin = t
    dp.time_end   = t
    dp.dynamic_obstacle.show_label        = False
    dp.dynamic_obstacle.draw_direction    = True
    dp.dynamic_obstacle.draw_icon         = False
    dp.dynamic_obstacle.draw_bounding_box = False
    return dp


def render_to_axes(scenario, t: int, ax: plt.Axes, plot_limits: list[float],
                   ttc_val: float | None = None, title: str = ""):
    """Render CommonRoad scenario at timestep t onto ax using MPRenderer."""
    ax.set_facecolor(PANEL_BG)
    renderer = MPRenderer(ax=ax, plot_limits=plot_limits)
    renderer.draw_scenario(scenario, draw_params=_make_draw_params(t))
    renderer.render()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(LANE_CLR)

    # Badge: time + TTC
    t_s   = t * 0.1
    badge = f"t = {t_s:.1f} s"
    if ttc_val is not None:
        clr    = WARN if ttc_val < TTC_TAU else SAFE_CLR
        badge += f"   TTC = {ttc_val:.1f} s"
    else:
        clr = FG
    ax.text(0.02, 0.97, badge, transform=ax.transAxes,
            color=clr, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL_BG, ec=LANE_CLR, alpha=0.85))
    if title:
        ax.set_title(title, color=FG, fontsize=7.5, pad=3)


def _style_ttc_ax(ax: plt.Axes):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=FG, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(LANE_CLR)


def plot_ttc_panel(ax: plt.Axes, score: DesignScore, rank: int, color: str):
    _style_ttc_ax(ax)
    ax.plot(score.times, score.ttc_series, color=color, lw=1.8)
    ax.fill_between(score.times, 0, score.ttc_series,
                    where=(score.ttc_series < TTC_TAU),
                    color=WARN, alpha=0.25)
    ax.axhline(TTC_TAU, color=WARN, lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0, TTC_CAP + 0.3)
    ax.set_ylabel("Min TTC (s)", color=FG, fontsize=7)
    ax.set_xlabel("Time (s)", color=FG, fontsize=7)
    info = (f"Rank #{rank}   min TTC = {score.min_ttc:.1f} s\n"
            f"TIT = {score.tit:.2f} s²   {score.pct_unsafe:.0f}% unsafe")
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            color=FG, fontsize=6.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=LANE_CLR, alpha=0.85))
    ax.set_title("Min TTC over time  (↑ = safer)", color=FG, fontsize=7.5, pad=3)


# ── comparison figure ─────────────────────────────────────────────────────────

def create_comparison_figure(
    designs_data: list[tuple[str, SimResult, DesignScore]],
    ranked: list[DesignScore],
    out_path: Path,
):
    """
    4 rows (designs) × 4 cols (approach | merge | post-merge | TTC chart).
    Each snapshot uses a geometry-anchored camera so road curvature is always visible.
    """
    n      = len(designs_data)
    n_snap = len(KEY_STEPS)
    n_cols = n_snap + 1

    fig = plt.figure(figsize=(24, 5 * n), facecolor=BG)
    fig.suptitle(
        "Merge Design Visual Comparison  —  CommonRoad MPRenderer + IDM Simulation",
        color=FG, fontsize=13, y=0.998, fontweight="bold",
    )

    rank_map = {s.label: i + 1 for i, s in enumerate(ranked)}

    outer_gs = gridspec.GridSpec(
        n, 1, figure=fig, hspace=0.22,
        left=0.01, right=0.99, top=0.975, bottom=0.02,
    )

    for row_idx, (label, sim, score) in enumerate(designs_data):
        rank  = rank_map[label]
        color = COLORS[row_idx % len(COLORS)]

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer_gs[row_idx],
            wspace=0.05, width_ratios=[1.4, 1, 1, 0.9],
        )

        # ── snapshot panels ────────────────────────────────────────────────
        for col_idx, (t, phase) in enumerate(zip(KEY_STEPS, KEY_LABELS)):
            ax   = fig.add_subplot(inner_gs[col_idx])
            lims = _camera(phase, sim.merge_length)
            ttc_val = float(score.ttc_series[t]) if t < len(score.ttc_series) else None
            render_to_axes(sim.scenario, t, ax, lims,
                           ttc_val=ttc_val, title=phase)

            # Row label pinned to first column
            if col_idx == 0:
                row_tag = (f"#{rank}  {label}\n"
                           f"L = {sim.merge_length:.0f} m   κ = {sim.curvature}")
                ax.text(0.02, 0.04, row_tag, transform=ax.transAxes,
                        color=color, fontsize=7.5, va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=color, alpha=0.88))

        # ── TTC panel ──────────────────────────────────────────────────────
        ax_ttc = fig.add_subplot(inner_gs[n_snap])
        plot_ttc_panel(ax_ttc, score, rank, color)

    fig.savefig(out_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison → {out_path}")


# ── animated GIF ──────────────────────────────────────────────────────────────

def create_scenario_gif(
    sim: SimResult,
    score: DesignScore,
    label: str,
    out_path: Path,
):
    """
    Full simulation as animated GIF.
    Left panel: road + vehicles (fixed camera).  Right panel: live TTC chart.
    """
    lims      = _gif_camera(sim.merge_length)
    timesteps = list(range(0, 300, GIF_STRIDE))
    frames    = []

    print(f"  GIF '{label}' ({len(timesteps)} frames)…", end="", flush=True)

    for t in timesteps:
        fig = plt.figure(figsize=(14, 3.5), facecolor=BG)
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.07,
                                left=0.01, right=0.99, top=0.88, bottom=0.08,
                                width_ratios=[3, 1])

        # ── road panel ────────────────────────────────────────────────────
        ax_road = fig.add_subplot(gs[0])
        ttc_val = float(score.ttc_series[t]) if t < len(score.ttc_series) else None
        render_to_axes(sim.scenario, t, ax_road, lims, ttc_val=ttc_val)
        title_clr = WARN if (ttc_val is not None and ttc_val < TTC_TAU) else FG
        ax_road.set_title(f"{label}   t = {t * 0.1:.1f} s",
                          color=title_clr, fontsize=9, pad=4)

        # ── live TTC chart ────────────────────────────────────────────────
        ax_ttc = fig.add_subplot(gs[1])
        _style_ttc_ax(ax_ttc)
        ts = score.times[:t + 1]
        tc = score.ttc_series[:t + 1]
        ax_ttc.plot(ts, tc, color=ACCENT, lw=1.5)
        if len(tc) > 0:
            ax_ttc.fill_between(ts, 0, tc, where=(tc < TTC_TAU), color=WARN, alpha=0.3)
        ax_ttc.axhline(TTC_TAU, color=WARN, lw=1, ls="--", alpha=0.7)
        ax_ttc.set_xlim(0, score.times[-1])
        ax_ttc.set_ylim(0, TTC_CAP + 0.3)
        ax_ttc.set_xlabel("Time (s)", color=FG, fontsize=7)
        ax_ttc.set_ylabel("Min TTC (s)", color=FG, fontsize=7)
        ax_ttc.set_title("Min TTC", color=FG, fontsize=8, pad=3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)
        print(".", end="", flush=True)

    imageio.mimsave(str(out_path), frames, fps=GIF_FPS, loop=0)
    print(f" → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Running simulations…")
    designs_data: list[tuple[str, SimResult, DesignScore]] = []

    for d in DESIGNS:
        label = d["label"]
        print(f"  {label}")
        sim   = run_simulation(merge_length=d["merge_length"], curvature=d["curvature"])
        score = score_design(label, sim)
        designs_data.append((label, sim, score))

    ranked = rank_designs([s for _, _, s in designs_data])

    print("\nGenerating comparison figure…")
    create_comparison_figure(designs_data, ranked, OUT / "scenario_comparison.png")

    print("\nGenerating animated GIFs…")
    for label, sim, score in designs_data:
        slug = (label.split("(")[0].strip()
                     .lower().replace(" ", "_").replace("-", "_"))
        create_scenario_gif(sim, score, label, OUT / f"gif_{slug}.gif")

    print("\n" + "=" * 68)
    print("  MERGE DESIGN SAFETY RANKING  (1 = safest)")
    print("=" * 68)
    print(f"  {'Design':<30} {'Rank':>4}  {'minTTC':>7}  {'TIT':>7}  {'%Unsafe':>8}")
    print("-" * 68)
    for i, s in enumerate(ranked, 1):
        print(f"  {s.label:<30} {i:>4}  {s.min_ttc:>7.2f}  {s.tit:>7.3f}  {s.pct_unsafe:>7.1f}%")
    print("=" * 68)
    print(f"\nOutputs → {OUT.resolve()}/")


if __name__ == "__main__":
    main()
