"""
compare_designs.py
==================
Rank two (or more) merge designs by standardised safety metrics
drawn from the CommonRoad CriMe (Criticality Measures) literature.

Metrics implemented
-------------------
TTC   Time-to-Collision          [s]      lower = worse
THW   Time Headway               [s]      lower = worse
TIT   Time-Integrated TTC        [s²]     higher = worse  (∫ max(0, τ-TTC) dt)
DRAC  Decel Rate to Avoid Crash  [m/s²]   higher = worse
BTN   Brake Threat Number        [-]      >1 = physically impossible to brake
DST   Decel to Safety Time       [m/s²]   higher = worse

All metrics are computed pairwise across every vehicle pair at every
timestep and then aggregated into per-scenario scalar scores used for ranking.

Usage
-----
    python compare_designs.py                   # built-in 4-design sweep
    python compare_designs.py --no-plot         # print table only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from simulation import run_simulation, SimResult

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
A_MAX_BRAKE = 6.0     # maximum physical braking (m/s²) for BTN
TTC_TAU     = 3.0     # TIT threshold τ (s) — integrate below this
TTC_CAP     = 10.0    # cap for non-approaching pairs
DT          = 0.1     # scenario timestep

# ── palette ────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
PANEL_BG  = "#2a2a3e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
WARN      = "#f38ba8"
SAFE_CLR  = "#a6e3a1"
LANE_CLR  = "#6c7086"
COLORS    = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7", "#f38ba8", "#89dceb"]


# ── pairwise metrics at one timestep ──────────────────────────────────────

def _state(obs, t):
    if obs.initial_state.time_step == t:
        return obs.initial_state
    if obs.prediction:
        for s in obs.prediction.trajectory.state_list:
            if s.time_step == t:
                return s
    return None


def _pair_metrics(sa, sb) -> dict:
    """Compute all pairwise metrics for two states."""
    pa = np.array(sa.position)
    pb = np.array(sb.position)
    va = float(sa.velocity)
    vb = float(sb.velocity)
    oa = float(sa.orientation)
    ob = float(sb.orientation)

    delta = pb - pa
    dist  = float(np.linalg.norm(delta))
    gap   = max(dist - 4.7, 0.01)          # bumper-to-bumper

    # velocity vectors
    va_vec = va * np.array([np.cos(oa), np.sin(oa)])
    vb_vec = vb * np.array([np.cos(ob), np.sin(ob)])

    # closing speed (positive = approaching)
    unit = delta / max(dist, 1e-6)
    v_close = float(np.dot(va_vec - vb_vec, unit))

    # ── TTC ──────────────────────────────────────────────────────────────
    if v_close > 0.01:
        ttc = min(gap / v_close, TTC_CAP)
    else:
        ttc = TTC_CAP

    # ── THW ──────────────────────────────────────────────────────────────
    # THW = gap / ego_speed  (for the faster vehicle approaching the slower)
    v_ego = max(va, 0.01)
    thw   = gap / v_ego

    # ── DRAC ─────────────────────────────────────────────────────────────
    drac = (v_close ** 2) / (2.0 * gap) if v_close > 0.01 else 0.0

    # ── BTN ──────────────────────────────────────────────────────────────
    # BTN = DRAC / a_max_brake  (>1 means even max braking won't help)
    btn = drac / A_MAX_BRAKE

    # ── DST ──────────────────────────────────────────────────────────────
    # Decel needed to achieve THW = TTC_TAU:
    # a = v_close / (2 * TTC_TAU)  simplified
    dst = v_close / (2.0 * max(TTC_TAU, 0.1)) if v_close > 0.01 else 0.0

    return dict(ttc=ttc, thw=thw, drac=drac, btn=btn, dst=dst)


# ── per-scenario aggregation ───────────────────────────────────────────────

@dataclass
class DesignScore:
    label:       str
    merge_length: float
    curvature:   float

    # Per-frame min-TTC series (for plotting)
    ttc_series:  np.ndarray = field(default_factory=lambda: np.array([]))
    times:       np.ndarray = field(default_factory=lambda: np.array([]))

    # Aggregate scalars
    mean_ttc:    float = TTC_CAP
    min_ttc:     float = TTC_CAP
    tit:         float = 0.0       # ∫ max(0, τ - TTC) dt
    pct_unsafe:  float = 0.0       # % frames with TTC < τ
    max_drac:    float = 0.0
    mean_drac:   float = 0.0
    max_btn:     float = 0.0
    mean_dst:    float = 0.0

    # Composite rank score (lower = safer)
    rank_score:  float = 0.0


def score_design(label: str, sim: SimResult) -> DesignScore:
    obs   = list(sim.scenario.dynamic_obstacles)
    steps = len(sim.frames)

    ttc_per_frame  = []
    drac_per_frame = []
    btn_per_frame  = []
    dst_per_frame  = []

    for t in range(steps):
        states = {}
        for o in obs:
            s = _state(o, t)
            if s is not None:
                states[o.obstacle_id] = s

        ids = list(states.keys())
        frame_ttcs  = []
        frame_dracs = []
        frame_btns  = []
        frame_dsts  = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                m = _pair_metrics(states[ids[i]], states[ids[j]])
                frame_ttcs.append(m["ttc"])
                frame_dracs.append(m["drac"])
                frame_btns.append(m["btn"])
                frame_dsts.append(m["dst"])

        ttc_per_frame.append(min(frame_ttcs)  if frame_ttcs  else TTC_CAP)
        drac_per_frame.append(max(frame_dracs) if frame_dracs else 0.0)
        btn_per_frame.append(max(frame_btns)   if frame_btns  else 0.0)
        dst_per_frame.append(max(frame_dsts)   if frame_dsts  else 0.0)

    ttc_arr  = np.array(ttc_per_frame)
    drac_arr = np.array(drac_per_frame)
    btn_arr  = np.array(btn_per_frame)
    dst_arr  = np.array(dst_per_frame)

    # TIT = ∫ max(0, τ - TTC) dt
    tit = float(np.sum(np.maximum(0.0, TTC_TAU - ttc_arr)) * DT)

    ds = DesignScore(
        label=label,
        merge_length=sim.merge_length,
        curvature=sim.curvature,
        ttc_series=ttc_arr,
        times=sim.times,
        mean_ttc=float(np.mean(ttc_arr)),
        min_ttc=float(np.min(ttc_arr)),
        tit=tit,
        pct_unsafe=float(100 * np.mean(ttc_arr < TTC_TAU)),
        max_drac=float(np.max(drac_arr)),
        mean_drac=float(np.mean(drac_arr)),
        max_btn=float(np.max(btn_arr)),
        mean_dst=float(np.mean(dst_arr)),
    )
    return ds


def rank_designs(scores: list[DesignScore]) -> list[DesignScore]:
    """
    Composite rank score: normalised weighted sum of safety metrics.
    All terms oriented so higher = worse (lower rank_score = safer).
    """
    if len(scores) == 1:
        scores[0].rank_score = 0.0
        return scores

    def _norm(vals):
        v = np.array(vals, dtype=float)
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 1e-9 else np.zeros_like(v)

    weights = dict(tit=0.35, pct_unsafe=0.25, min_ttc_inv=0.20,
                   max_drac=0.10, max_btn=0.10)

    tit_n       = _norm([s.tit        for s in scores])
    pct_n       = _norm([s.pct_unsafe for s in scores])
    ttc_inv_n   = _norm([1/(s.min_ttc+0.01) for s in scores])
    drac_n      = _norm([s.max_drac   for s in scores])
    btn_n       = _norm([s.max_btn    for s in scores])

    for i, s in enumerate(scores):
        s.rank_score = (
            weights["tit"]        * tit_n[i]
            + weights["pct_unsafe"] * pct_n[i]
            + weights["min_ttc_inv"]* ttc_inv_n[i]
            + weights["max_drac"]   * drac_n[i]
            + weights["max_btn"]    * btn_n[i]
        )

    return sorted(scores, key=lambda s: s.rank_score)


# ── plotting ───────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(LANE_CLR)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def plot_comparison(ranked: list[DesignScore], out_path: Path):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("Merge Design Safety Comparison  (ranked safest → most dangerous)",
                 color=FG, fontsize=11, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_ttc   = fig.add_subplot(gs[0, :2])   # TTC time-series, wide
    ax_rank  = fig.add_subplot(gs[0, 2])    # rank bar
    ax_tit   = fig.add_subplot(gs[1, 0])    # TIT bar
    ax_drac  = fig.add_subplot(gs[1, 1])    # DRAC bar
    ax_radar = fig.add_subplot(gs[1, 2], polar=True)  # radar

    for ax in (ax_ttc, ax_rank, ax_tit, ax_drac):
        _ax_style(ax)

    labels = [s.label for s in ranked]
    clrs   = COLORS[:len(ranked)]

    # ── TTC time-series ───────────────────────────────────────────────────
    for s, c in zip(ranked, clrs):
        ax_ttc.plot(s.times, s.ttc_series, color=c, lw=1.5, label=s.label, alpha=0.9)
        ax_ttc.fill_between(s.times, 0, s.ttc_series,
                            where=(s.ttc_series < TTC_TAU),
                            color=c, alpha=0.15)
    ax_ttc.axhline(TTC_TAU, color=WARN, lw=1, ls="--", alpha=0.8)
    ax_ttc.set_ylim(0, TTC_CAP + 0.5)
    ax_ttc.set_ylabel("Min TTC (s)", color=FG, fontsize=8)
    ax_ttc.set_xlabel("Time (s)", color=FG, fontsize=8)
    ax_ttc.set_title("Min TTC over time  (dashed = 3 s unsafe threshold)", color=FG, fontsize=9)
    ax_ttc.legend(facecolor=PANEL_BG, edgecolor=LANE_CLR, labelcolor=FG,
                  fontsize=7, loc="upper right")

    # ── Rank bar ──────────────────────────────────────────────────────────
    scores = [s.rank_score for s in ranked]
    bars = ax_rank.barh(labels[::-1], scores[::-1], color=clrs[::-1],
                        edgecolor="white", linewidth=0.5)
    ax_rank.set_xlabel("Composite risk score (lower = safer)", color=FG, fontsize=8)
    ax_rank.set_title("Overall Rank", color=FG, fontsize=9)
    for bar, score in zip(bars, scores[::-1]):
        ax_rank.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{score:.3f}", va="center", color=FG, fontsize=7)

    # ── TIT bar ───────────────────────────────────────────────────────────
    tits = [s.tit for s in ranked]
    ax_tit.bar(labels, tits, color=clrs, edgecolor="white", linewidth=0.5)
    ax_tit.set_ylabel("TIT (s²)", color=FG, fontsize=8)
    ax_tit.set_title("Time-Integrated TTC\n(∫ max(0, τ−TTC) dt)", color=FG, fontsize=9)
    ax_tit.tick_params(axis="x", rotation=15, labelsize=7)

    # ── DRAC bar ──────────────────────────────────────────────────────────
    dracs = [s.max_drac for s in ranked]
    ax_drac.bar(labels, dracs, color=clrs, edgecolor="white", linewidth=0.5)
    ax_drac.axhline(3.4, color=WARN, lw=1, ls="--", alpha=0.7)
    ax_drac.set_ylabel("Max DRAC (m/s²)", color=FG, fontsize=8)
    ax_drac.set_title("Max Deceleration Rate\nto Avoid Crash  (3.4 = AASHTO limit)",
                      color=FG, fontsize=9)
    ax_drac.tick_params(axis="x", rotation=15, labelsize=7)

    # ── Radar chart ───────────────────────────────────────────────────────
    metric_names = ["TIT", "% Unsafe\nFrames", "Max DRAC", "Max BTN", "Mean DST"]
    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise each metric to 0-1 across designs
    def _norm_metric(vals):
        v = np.array(vals, dtype=float)
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 1e-9 else np.zeros_like(v)

    raw = np.array([
        _norm_metric([s.tit        for s in ranked]),
        _norm_metric([s.pct_unsafe for s in ranked]),
        _norm_metric([s.max_drac   for s in ranked]),
        _norm_metric([s.max_btn    for s in ranked]),
        _norm_metric([s.mean_dst   for s in ranked]),
    ]).T  # shape: (n_designs, n_metrics)

    ax_radar.set_facecolor(PANEL_BG)
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_names, color=FG, fontsize=7)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color=FG, fontsize=6)
    ax_radar.tick_params(colors=FG)
    for spine in ax_radar.spines.values():
        spine.set_color(LANE_CLR)
    ax_radar.grid(color=LANE_CLR, alpha=0.4)
    ax_radar.set_title("Risk Profile (1 = worst)", color=FG, fontsize=9, pad=15)

    for vals, c, lbl in zip(raw, clrs, labels):
        v = vals.tolist() + vals[:1].tolist()
        ax_radar.plot(angles, v, color=c, lw=1.5, label=lbl)
        ax_radar.fill(angles, v, color=c, alpha=0.1)

    ax_radar.legend(facecolor=PANEL_BG, edgecolor=LANE_CLR, labelcolor=FG,
                    fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path}")


# ── print table ───────────────────────────────────────────────────────────

def print_table(ranked: list[DesignScore]):
    col = 16
    hdr = f"{'Design':<{col}} {'Rank':>5}  {'Min TTC':>8}  {'TIT':>8}  {'%Unsafe':>8}  {'MaxDRAC':>8}  {'MaxBTN':>8}"
    print()
    print("=" * len(hdr))
    print("  MERGE DESIGN SAFETY RANKING  (1 = safest)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(ranked, 1):
        print(
            f"{s.label:<{col}} {i:>5}  "
            f"{s.min_ttc:>8.2f}  "
            f"{s.tit:>8.3f}  "
            f"{s.pct_unsafe:>7.1f}%  "
            f"{s.max_drac:>8.2f}  "
            f"{s.max_btn:>8.3f}"
        )
    print("=" * len(hdr))
    print()
    print("Metrics (CriMe-compatible definitions):")
    print("  Min TTC   : minimum time-to-collision across all pairs & timesteps [s]  (higher = safer)")
    print("  TIT       : time-integrated TTC = ∫ max(0, 3−TTC) dt  [s²]             (lower = safer)")
    print("  %Unsafe   : % of frames where any pair has TTC < 3 s                    (lower = safer)")
    print("  MaxDRAC   : max decel rate to avoid crash  [m/s²]  (AASHTO limit 3.4)  (lower = safer)")
    print("  MaxBTN    : max brake threat number = DRAC/a_max  (>1 = un-avoidable)  (lower = safer)")
    print()


# ── main ──────────────────────────────────────────────────────────────────

DESIGNS = [
    dict(label="Short-Straight   (L=60  κ=0.2)", merge_length=60,  curvature=0.2),
    dict(label="Standard-Curved  (L=120 κ=0.5)", merge_length=120, curvature=0.5),
    dict(label="Long-Straight    (L=180 κ=0.2)", merge_length=180, curvature=0.2),
    dict(label="Long-Curved      (L=220 κ=0.8)", merge_length=220, curvature=0.8),
]


def main(plot: bool = True):
    scores = []
    print("Running simulations…")
    for d in DESIGNS:
        label = d["label"]
        print(f"  {label}")
        sim = run_simulation(merge_length=d["merge_length"], curvature=d["curvature"])
        scores.append(score_design(label, sim))

    ranked = rank_designs(scores)
    print_table(ranked)

    if plot:
        plot_comparison(ranked, OUT / "design_comparison.png")

    return ranked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    main(plot=not args.no_plot)
