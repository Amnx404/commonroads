"""
run_headless.py
===============
Generate all outputs without a display (useful in CI / headless servers).

Outputs written to outputs/:
  merge_scene_t*.png  – road scene snapshots at key timesteps
  safety_timeseries.png – min TTC and DRAC over time
  sweep_length.png      – safety vs merge length
  sweep_curvature.png   – safety vs curvature
  USA_MERGE-1_1_T-1.xml – CommonRoad XML scenario
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from pathlib import Path

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from merge_scenario import bezier_merge_centerline
from safety_metrics import TTC_MAX, _get_state
from simulation import run_simulation, sweep_merge_length, sweep_curvature
from merge_scenario import get_planning_problem_set
from commonroad.scenario.scenario import Tag

UNSAFE_TTC = 3.0
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── dark style constants ────────────────────────────────────────────────────
BG       = "#1e1e2e"
PANEL_BG = "#2a2a3e"
FG       = "#cdd6f4"
ACCENT   = "#89b4fa"
WARN     = "#f38ba8"
SAFE_CLR = "#a6e3a1"
ROAD_CLR = "#45475a"
LANE_CLR = "#6c7086"
CAR_MAIN = "#89dceb"
CAR_RAMP = "#f9e2af"
MERGE_CLR= "#cba6f7"


def _ax_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(LANE_CLR)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def plot_scene(sim, t: int, ax):
    ax.clear()
    _ax_style(ax)
    ax.set_aspect("equal")
    ml = sim.merge_length
    y_ramp = 4.0 * 2.5

    road = mpatches.FancyBboxPatch(
        (-15, -3.5), ml + 315, 7,
        boxstyle="round,pad=0.5", facecolor=ROAD_CLR, edgecolor="none", zorder=0,
    )
    ax.add_patch(road)
    cl = bezier_merge_centerline(ml, y_ramp, 0.0, sim.curvature)
    ax.plot(cl[:, 0], cl[:, 1], "--", color=MERGE_CLR, lw=1.5, alpha=0.8, zorder=1)
    ax.axhline(0,  color=LANE_CLR, lw=1.0, ls="--", zorder=1)
    ax.axhline(-2, color=LANE_CLR, lw=0.5, ls="-",  zorder=1)
    ax.axhline(+2, color=LANE_CLR, lw=0.5, ls="-",  zorder=1)
    ax.axvline(ml, color=MERGE_CLR, lw=0.8, ls=":", alpha=0.5, zorder=1)

    frame = sim.frames[min(t, len(sim.frames) - 1)]
    for obs in sim.scenario.dynamic_obstacles:
        st = _get_state(obs, t)
        if st is None:
            continue
        pos = np.array(st.position)
        ori = float(st.orientation)
        color = CAR_RAMP if obs.obstacle_id >= 300 else CAR_MAIN
        from matplotlib.patches import FancyBboxPatch
        patch = FancyBboxPatch(
            (-4.7/2, -2.0/2), 4.7, 2.0,
            boxstyle="round,pad=0.3",
            facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3,
        )
        tfm = Affine2D().rotate(ori).translate(pos[0], pos[1]) + ax.transData
        patch.set_transform(tfm)
        ax.add_patch(patch)
        ax.text(pos[0], pos[1]+1.7, str(obs.obstacle_id),
                color="white", fontsize=6, ha="center", zorder=5)

    clr = SAFE_CLR if frame.min_ttc >= UNSAFE_TTC else WARN
    ax.text(0.99, 0.97, f"TTC {frame.min_ttc:.1f}s",
            transform=ax.transAxes, color=clr,
            fontsize=9, ha="right", va="top", fontweight="bold")
    ax.set_xlim(-15, ml + 315)
    ax.set_ylim(-6, 16)
    ax.set_title(f"Scene  t = {t * sim.scenario.dt:.1f} s", color=FG, fontsize=9)
    ax.set_xlabel("x (m)", color=FG, fontsize=7)
    ax.set_ylabel("y (m)", color=FG, fontsize=7)


def main():
    print("Building scenario (merge_length=120m, curvature=0.5)…")
    sim = run_simulation(merge_length=120.0, curvature=0.5)

    # ── Save CommonRoad XML ────────────────────────────────────────────────
    xml_path = OUT / "USA_MERGE-1_1_T-1.xml"
    pps = get_planning_problem_set()
    writer = CommonRoadFileWriter(
        scenario=sim.scenario,
        planning_problem_set=pps,
        author="merge_gui",
        affiliation="merge_safety_explorer",
        source="Procedural Bezier merge",
        tags={Tag.INTERSTATE},
    )
    writer.write_to_file(str(xml_path), overwrite_existing_file=OverwriteExistingFile.ALWAYS)
    print(f"  XML → {xml_path}")

    # ── Scene snapshots ────────────────────────────────────────────────────
    snapshot_steps = [0, 30, 60, 90, 120, 180]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), facecolor=BG)
    for ax, t in zip(axes.flat, snapshot_steps):
        plot_scene(sim, t, ax)
    fig.tight_layout()
    scene_path = OUT / "merge_scene_snapshots.png"
    fig.savefig(scene_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scene snapshots → {scene_path}")

    # ── Safety time-series ─────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 4), facecolor=BG)
    _ax_style(ax1)
    times   = sim.times
    min_ttc = sim.min_ttc_series
    max_dr  = sim.max_drac_series

    ax1.plot(times, min_ttc, color=ACCENT, lw=1.5, label="min TTC (s)")
    ax1.fill_between(times, 0, min_ttc,
                     where=(min_ttc < UNSAFE_TTC),
                     color=WARN, alpha=0.35, label=f"TTC < {UNSAFE_TTC}s (unsafe)")
    ax1.axhline(UNSAFE_TTC, color=WARN, lw=1, ls="--", alpha=0.8)
    ax1.set_ylabel("Min TTC (s)", color=FG, fontsize=8)
    ax1.set_ylim(0, TTC_MAX + 0.5)
    ax1.set_xlabel("Time (s)", color=FG, fontsize=8)
    ax1.set_title("Safety Metrics over Time", color=FG)

    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(times, max_dr, color=MERGE_CLR, lw=1.0, ls=":", alpha=0.8,
             label="max DRAC (m/s²)")
    ax2.axhline(3.4, color=MERGE_CLR, lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylabel("Max DRAC (m/s²)", color=MERGE_CLR, fontsize=8)
    ax2.tick_params(axis="y", colors=MERGE_CLR, labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor(LANE_CLR)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, n1+n2, facecolor=PANEL_BG, edgecolor=LANE_CLR,
               labelcolor=FG, fontsize=7)
    fig.tight_layout()
    ts_path = OUT / "safety_timeseries.png"
    fig.savefig(ts_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Time-series → {ts_path}")

    # ── Parameter sweeps ───────────────────────────────────────────────────
    print("Running merge-length sweep…")
    sw_len = sweep_merge_length()
    print("Running curvature sweep…")
    sw_cur = sweep_curvature()

    fig, (axL, axC) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax, data, xlabel in [
        (axL, sw_len, "Merge Length (m)"),
        (axC, sw_cur, "Curvature"),
    ]:
        _ax_style(ax)
        xs = data["params"]
        ax.plot(xs, data["mean_min_ttc"], "-o", color=ACCENT,
                lw=1.5, ms=6, label="Mean min TTC (s)")
        ax.axhline(UNSAFE_TTC, color=ACCENT, lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel("Mean min TTC (s)", color=FG, fontsize=8)
        ax.set_ylim(0, TTC_MAX + 0.5)
        ax.set_xlabel(xlabel, color=FG, fontsize=8)
        ax.set_title(f"Safety vs {xlabel}", color=FG)

        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL_BG)
        ax2.plot(xs, data["pct_unsafe"], "-s", color=WARN,
                 lw=1.5, ms=6, label="% unsafe frames")
        ax2.set_ylabel("% unsafe frames", color=WARN, fontsize=8)
        ax2.tick_params(axis="y", colors=WARN, labelsize=7)
        ax2.set_ylim(0, 105)
        for sp in ax2.spines.values():
            sp.set_edgecolor(LANE_CLR)
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, n1+n2, facecolor=PANEL_BG, edgecolor=LANE_CLR,
                  labelcolor=FG, fontsize=7)

    fig.tight_layout()
    sw_path = OUT / "parameter_sweeps.png"
    fig.savefig(sw_path, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parameter sweeps → {sw_path}")

    print("\nAll outputs written to outputs/")
    print(f"  min TTC = {sim.min_ttc_series.min():.2f}s")
    print(f"  unsafe frames = {(sim.min_ttc_series < UNSAFE_TTC).sum()} / {len(sim.frames)}")


if __name__ == "__main__":
    main()
