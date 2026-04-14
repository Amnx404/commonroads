"""
safety_metrics.py
=================
Compute per-timestep safety metrics from a CommonRoad scenario using the
CommonRoad obstacle/state API.  Metric definitions follow the CommonRoad
CriMe (Criticality Measures) standard.

Metrics
-------
TTC   Time-to-Collision          gap / closing_speed   [s]  higher = safer
THW   Time Headway               gap / ego_speed       [s]  higher = safer
DRAC  Decel Rate to Avoid Crash  cs² / (2·gap)        [m/s²]  lower = safer
BTN   Brake Threat Number        DRAC / B_MAX          [–]   <1 = avoidable
TIT   Time-Integrated TTC        ∫ max(0, τ − TTC) dt  [s²]  lower = safer
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from commonroad.scenario.scenario import Scenario

TTC_TAU    = 3.0    # unsafe TTC threshold (s)
TTC_MAX    = 10.0   # cap for non-approaching pairs (s)
DRAC_SAFE  = 3.4    # AASHTO safe decel threshold (m/s²)
B_MAX      = 6.0    # maximum braking decel for BTN (m/s²)
MIN_GAP    = 0.1    # m — avoid division-by-zero


@dataclass
class PairMetrics:
    id_a:          int
    id_b:          int
    gap:           float   # bumper-to-bumper distance (m)
    closing_speed: float   # positive = approaching (m/s)
    ttc:           float   # s
    drac:          float   # m/s²
    btn:           float   # dimensionless
    thw:           float   # s


@dataclass
class FrameMetrics:
    time_step: int
    time_s:    float
    pairs:     list[PairMetrics] = field(default_factory=list)

    @property
    def min_ttc(self) -> float:
        return min((p.ttc for p in self.pairs), default=TTC_MAX)

    @property
    def max_drac(self) -> float:
        return max((p.drac for p in self.pairs), default=0.0)

    @property
    def max_btn(self) -> float:
        return max((p.btn for p in self.pairs), default=0.0)

    @property
    def n_unsafe_pairs(self) -> int:
        return sum(1 for p in self.pairs if p.ttc < TTC_TAU)


def _get_state(obstacle, t: int):
    """Return the state for obstacle at timestep t, or None."""
    if obstacle.initial_state.time_step == t:
        return obstacle.initial_state
    if obstacle.prediction is None:
        return None
    for s in obstacle.prediction.trajectory.state_list:
        if s.time_step == t:
            return s
    return None


def _closing_speed(pos_a, vel_a, ori_a, pos_b, vel_b, ori_b) -> float:
    """Scalar closing speed of a approaching b (positive = closing)."""
    delta = np.asarray(pos_b) - np.asarray(pos_a)
    dist  = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return 0.0
    unit = delta / dist
    va = vel_a * np.array([np.cos(ori_a), np.sin(ori_a)])
    vb = vel_b * np.array([np.cos(ori_b), np.sin(ori_b)])
    return float(np.dot(va - vb, unit))


def compute_metrics(scenario: Scenario, car_length: float = 4.7) -> list[FrameMetrics]:
    """
    Iterate over all simulation timesteps and return one FrameMetrics
    per step containing pairwise TTC / DRAC / BTN / THW values.
    """
    obstacles = list(scenario.dynamic_obstacles)
    dt        = scenario.dt
    if not obstacles:
        n_steps = 1
    else:
        n_steps   = max(
            max(s.time_step for obs in obstacles
                for s in (list(obs.prediction.trajectory.state_list)
                          if obs.prediction else [])
                + [obs.initial_state]) + 1,
            1,
        )
    results: list[FrameMetrics] = []

    for t in range(n_steps):
        frame = FrameMetrics(time_step=t, time_s=t * dt)

        states: dict[int, object] = {}
        for obs in obstacles:
            s = _get_state(obs, t)
            if s is not None:
                states[obs.obstacle_id] = s

        ids = list(states.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sa = states[ids[i]]
                sb = states[ids[j]]

                pos_a = np.asarray(sa.position, dtype=float)
                pos_b = np.asarray(sb.position, dtype=float)
                centre_dist = float(np.linalg.norm(pos_b - pos_a))
                gap = max(centre_dist - car_length, MIN_GAP)

                cs = _closing_speed(
                    pos_a, float(sa.velocity), float(sa.orientation),
                    pos_b, float(sb.velocity), float(sb.orientation),
                )

                if cs > 1e-3:
                    ttc  = min(gap / cs, TTC_MAX)
                    drac = cs ** 2 / (2.0 * gap)
                else:
                    ttc  = TTC_MAX
                    drac = 0.0

                btn = drac / B_MAX
                thw = gap / max(float(sa.velocity), 0.1)

                frame.pairs.append(PairMetrics(
                    id_a=ids[i], id_b=ids[j],
                    gap=gap, closing_speed=cs,
                    ttc=ttc, drac=drac, btn=btn, thw=thw,
                ))

        results.append(frame)

    return results


@dataclass
class ScenarioSafety:
    """Aggregate safety summary for one scenario run."""
    curvature:    float
    mean_min_ttc: float   # mean of per-frame min-TTC
    pct_unsafe:   float   # % of frames with any pair TTC < TTC_TAU
    max_drac:     float   # worst observed DRAC  (m/s²)
    max_btn:      float   # worst observed BTN
    tit:          float   # Time-Integrated TTC  (Σ max(0, τ − TTC) · dt)
    min_gap_m:    float   # smallest bumper gap any pair, any step (m)
    merge_length: float = 0.0  # merge zone length (m); 0 if unused


def aggregate(
    curvature: float,
    frames: list[FrameMetrics],
    dt: float,
    merge_length: float = 0.0,
) -> ScenarioSafety:
    """Reduce per-frame metrics to a single ScenarioSafety record."""
    min_ttcs   = [f.min_ttc for f in frames]
    unsafe_cnt = sum(1 for f in frames if f.n_unsafe_pairs > 0)
    tit        = sum(max(0.0, TTC_TAU - f.min_ttc) * dt for f in frames)
    all_gaps   = [p.gap for f in frames for p in f.pairs]
    min_gap_m  = float(min(all_gaps)) if all_gaps else float("nan")

    return ScenarioSafety(
        curvature    = curvature,
        mean_min_ttc = float(np.mean(min_ttcs)),
        pct_unsafe   = 100.0 * unsafe_cnt / max(len(frames), 1),
        max_drac     = max((f.max_drac for f in frames), default=0.0),
        max_btn      = max((f.max_btn for f in frames), default=0.0),
        tit          = tit,
        min_gap_m    = min_gap_m,
        merge_length = merge_length,
    )
