"""
generate_gifs.py
================
Render animated GIFs for review:

  outputs/gif_scene_default.gif      – default params (L=120m, κ=0.5)
  outputs/gif_scene_short_merge.gif  – short aggressive merge (L=60m, κ=0.2)
  outputs/gif_scene_long_smooth.gif  – long smooth merge  (L=220m, κ=0.8)
  outputs/gif_ttc_comparison.gif     – animated TTC+DRAC traces across 3 configs

Each GIF runs at 15 fps covering the full 30 s simulation.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D
from pathlib import Path

from merge_scenario import bezier_merge_centerline
from safety_metrics import TTC_MAX, _get_state
from simulation import run_simulation, SimResult

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
PANEL_BG  = "#2a2a3e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
WARN      = "#f38ba8"
SAFE_CLR  = "#a6e3a1"
ROAD_CLR  = "#45475a"
LANE_CLR  = "#6c7086"
CAR_MAIN  = "#89dceb"
CAR_RAMP  = "#f9e2af"
MERGE_CLR = "#cba6f7"
UNSAFE_TTC = 3.0

# subsample: render every Nth frame to keep file sizes sane
FRAME_SKIP = 3   # → 100 frames per 30 s sim at 15 fps ≈ 6-7 s GIF


def _ax_base(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(LANE_CLR)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def _draw_road(ax, sim: SimResult):
    """Draw static road elements (called once then cached as background)."""
    ml     = sim.merge_length
    y_ramp = 4.0 * 2.5   # LANE_WIDTH * 2.5 = 10 m

    road = mpatches.FancyBboxPatch(
        (-15, -4.5), ml + 315, 9,
        boxstyle="round,pad=0.5", facecolor=ROAD_CLR, edgecolor="none", zorder=0,
    )
    ax.add_patch(road)

    # Bezier ramp centreline
    cl = bezier_merge_centerline(ml, y_ramp, 0.0, sim.curvature)
    ax.plot(cl[:, 0], cl[:, 1], "--", color=MERGE_CLR, lw=1.2, alpha=0.7, zorder=1)

    # Ramp boundaries (offset ±2 m)
    from merge_scenario import _offset_centerline
    lb = _offset_centerline(cl, +2.0)
    rb = _offset_centerline(cl, -2.0)
    ax.plot(lb[:, 0], lb[:, 1], "-", color=LANE_CLR, lw=0.6, alpha=0.5, zorder=1)
    ax.plot(rb[:, 0], rb[:, 1], "-", color=LANE_CLR, lw=0.6, alpha=0.5, zorder=1)

    # Main lane markings
    ax.axhline(0,  color="#f5c842", lw=0.8, ls="--", zorder=1, alpha=0.6)  # centre
    ax.axhline(-2, color=LANE_CLR,  lw=0.5, ls="-",  zorder=1, alpha=0.5)
    ax.axhline(+2, color=LANE_CLR,  lw=0.5, ls="-",  zorder=1, alpha=0.5)

    # Merge point marker
    ax.axvline(ml, color=MERGE_CLR, lw=0.8, ls=":", alpha=0.5, zorder=1)
    ax.text(ml + 2, 12.5, "merge point", color=MERGE_CLR, fontsize=6, va="top")


def _draw_vehicles(ax, sim: SimResult, t: int) -> list:
    """Draw all vehicles at timestep t. Returns list of artists to remove later."""
    artists = []
    # Figure out the data→display scale so we can draw fixed-pixel-size cars
    # Use a display size of ~14px wide × 8px tall in data coords
    xl = ax.get_xlim(); yl = ax.get_ylim()
    fig_w, fig_h = ax.get_figure().get_size_inches()
    ax_w  = ax.get_position().width  * fig_w * ax.get_figure().dpi
    ax_h  = ax.get_position().height * fig_h * ax.get_figure().dpi
    x_scale = (xl[1] - xl[0]) / ax_w   # data units per pixel
    y_scale = (yl[1] - yl[0]) / ax_h

    # draw cars at a fixed visual size (pixels → data units)
    car_w_px, car_h_px = 18, 28   # width along x, height along y (pixels)
    cw = car_w_px * x_scale
    ch = car_h_px * y_scale

    for obs in sim.scenario.dynamic_obstacles:
        st = _get_state(obs, t)
        if st is None:
            continue
        pos = np.array(st.position)
        ori = float(st.orientation)
        color = CAR_RAMP if obs.obstacle_id >= 300 else CAR_MAIN

        patch = FancyBboxPatch(
            (-cw/2, -ch/2), cw, ch,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.95, zorder=3,
        )
        tfm = Affine2D().rotate(ori).translate(pos[0], pos[1]) + ax.transData
        patch.set_transform(tfm)
        ax.add_patch(patch)
        artists.append(patch)

        lbl = ax.text(pos[0], pos[1] + ch * 0.75, str(obs.obstacle_id),
                      color="white", fontsize=6, ha="center", zorder=5,
                      fontweight="bold")
        artists.append(lbl)

    return artists


# ─────────────────────────────────────────────────────────────────────────────
# GIF 1-3: Scene animation for a single config
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_gif(sim: SimResult, out_path: Path, title_tag: str):
    ml = sim.merge_length
    total_x = ml + 315.0

    fig, (ax_scene, ax_ttc) = plt.subplots(
        2, 1, figsize=(12, 6),
        facecolor=BG,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.38},
    )
    _ax_base(ax_scene)
    _ax_base(ax_ttc)

    # No equal-aspect — the road is long & narrow; let y stretch so cars are visible
    ax_scene.set_xlim(-15, total_x + 15)
    ax_scene.set_ylim(-8, 18)
    ax_scene.set_xlabel("x (m)", color=FG, fontsize=7)
    ax_scene.set_ylabel("y (m)", color=FG, fontsize=7)

    # Draw static road once
    _draw_road(ax_scene, sim)

    # Pre-plot the full TTC trace as static background
    times   = sim.times
    min_ttc = sim.min_ttc_series
    ax_ttc.plot(times, min_ttc, color=ACCENT, lw=1.2, alpha=0.4, zorder=1)
    ax_ttc.fill_between(times, 0, min_ttc,
                        where=(min_ttc < UNSAFE_TTC),
                        color=WARN, alpha=0.2, zorder=1)
    ax_ttc.axhline(UNSAFE_TTC, color=WARN, lw=0.8, ls="--", alpha=0.8)
    ax_ttc.set_ylim(0, TTC_MAX + 0.5)
    ax_ttc.set_ylabel("Min TTC (s)", color=FG, fontsize=7)
    ax_ttc.set_xlabel("Time (s)", color=FG, fontsize=7)
    ax_ttc.set_xlim(0, times[-1])

    # Dynamic elements: vehicle patches + time cursor + TTC readout
    vehicle_artists: list = []
    cursor_line     = ax_ttc.axvline(0, color=FG, lw=1, alpha=0.8, zorder=5)
    ttc_text        = ax_scene.text(
        0.98, 0.97, "", transform=ax_scene.transAxes,
        color=SAFE_CLR, fontsize=10, ha="right", va="top", fontweight="bold",
        zorder=10,
    )
    time_text = ax_scene.text(
        0.02, 0.97, "", transform=ax_scene.transAxes,
        color=FG, fontsize=8, ha="left", va="top", zorder=10,
    )
    title_text = ax_scene.set_title(
        f"Merge Safety  {title_tag}", color=FG, fontsize=10, pad=6,
    )

    frame_steps = list(range(0, len(sim.frames), FRAME_SKIP))

    def _update(fi):
        nonlocal vehicle_artists
        t = frame_steps[fi]

        # Remove old vehicle patches
        for a in vehicle_artists:
            a.remove()
        vehicle_artists = _draw_vehicles(ax_scene, sim, t)

        # Update cursor and readout
        ts   = sim.times[t]
        mttc = sim.min_ttc_series[t]
        cursor_line.set_xdata([ts, ts])
        clr = SAFE_CLR if mttc >= UNSAFE_TTC else WARN
        ttc_text.set_text(f"TTC {mttc:.1f}s")
        ttc_text.set_color(clr)
        time_text.set_text(f"t = {ts:.1f} s")

        return vehicle_artists + [cursor_line, ttc_text, time_text]

    anim = FuncAnimation(
        fig, _update,
        frames=len(frame_steps),
        interval=67,   # ~15 fps
        blit=False,
    )
    anim.save(str(out_path), writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GIF 4: Animated TTC comparison across 3 configs
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_gif(sims: list[tuple[str, SimResult]], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=BG, sharey=True)
    fig.suptitle("TTC Comparison: Merge Length & Curvature", color=FG, fontsize=11)

    colors = [ACCENT, SAFE_CLR, CAR_RAMP]
    cursors = []
    fills   = []

    for ax, (label, sim), col in zip(axes, sims, colors):
        _ax_base(ax)
        times   = sim.times
        min_ttc = sim.min_ttc_series

        ax.plot(times, min_ttc, color=col, lw=1.5, alpha=0.5, zorder=1)
        ax.fill_between(times, 0, min_ttc,
                        where=(min_ttc < UNSAFE_TTC),
                        color=WARN, alpha=0.25, zorder=1)
        ax.axhline(UNSAFE_TTC, color=WARN, lw=0.8, ls="--", alpha=0.8)
        ax.set_ylim(0, TTC_MAX + 0.5)
        ax.set_xlim(0, times[-1])
        ax.set_title(label, color=FG, fontsize=8)
        ax.set_xlabel("Time (s)", color=FG, fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Min TTC (s)", color=FG, fontsize=7)

        crs = ax.axvline(0, color=FG, lw=1.2, alpha=0.9, zorder=5)
        cursors.append(crs)

        # Running TTC fill (will be updated per frame)
        fill = ax.fill_between([0], [0], [0], color=col, alpha=0.6, zorder=2)
        fills.append((ax, fill, sim))

    pct_texts = [
        ax.text(0.97, 0.96, "", transform=ax.transAxes,
                color=WARN, fontsize=7, ha="right", va="top")
        for ax in axes
    ]

    n_frames = len(range(0, 300, FRAME_SKIP))
    frame_steps = list(range(0, 300, FRAME_SKIP))

    def _update(fi):
        t  = frame_steps[fi]
        ts = t * sims[0][1].scenario.dt

        for i, (crs, (ax, old_fill, sim), ptxt) in enumerate(
            zip(cursors, fills, pct_texts)
        ):
            crs.set_xdata([ts, ts])

            # Redraw fill up to current time
            old_fill.remove()
            times   = sim.times[:t+1]
            min_ttc = sim.min_ttc_series[:t+1]
            new_fill = ax.fill_between(times, 0, min_ttc,
                                       color=colors[i], alpha=0.55, zorder=2)
            fills[i] = (ax, new_fill, sim)

            # % unsafe so far
            unsafe = (sim.min_ttc_series[:t+1] < UNSAFE_TTC).sum()
            total  = max(t + 1, 1)
            ptxt.set_text(f"{100*unsafe/total:.0f}% unsafe")

        return cursors + [f for _, f, _ in fills] + pct_texts

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=67,
        blit=False,
    )
    fig.tight_layout()
    anim.save(str(out_path), writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    configs = [
        ("default",     dict(merge_length=120.0, curvature=0.5),  "L=120m  κ=0.5"),
        ("short_merge", dict(merge_length=60.0,  curvature=0.2),  "L=60m   κ=0.2  (aggressive)"),
        ("long_smooth", dict(merge_length=220.0, curvature=0.8),  "L=220m  κ=0.8  (gentle)"),
    ]

    sims = []
    print("Building scenarios…")
    for key, params, label in configs:
        print(f"  [{label}]")
        sim = run_simulation(**params)
        sims.append((label, sim))
        unsafe = (sim.min_ttc_series < UNSAFE_TTC).sum()
        print(f"    min TTC={sim.min_ttc_series.min():.2f}s  "
              f"unsafe frames={unsafe}/{len(sim.frames)}")

    print("\nRendering GIFs…")
    for (key, params, label), (_, sim) in zip(configs, sims):
        make_scene_gif(sim, OUT / f"gif_scene_{key}.gif", label)

    print("  Rendering comparison GIF…")
    make_comparison_gif(sims, OUT / "gif_ttc_comparison.gif")

    print("\nDone. GIFs written to outputs/:")
    for p in sorted(OUT.glob("*.gif")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
