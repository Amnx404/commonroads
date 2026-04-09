"""
safety_metrics.py
=================
Compute per-timestep safety metrics from a CommonRoad scenario.

Metrics
-------
TTC  (Time-to-Collision)
    For each pair of vehicles, project the closing speed along the
    line-of-sight.  TTC = gap / closing_speed if closing_speed > 0.
    Capped at TTC_MAX when vehicles are diverging.

DRAC (Deceleration Rate to Avoid Crash)
    DRAC = closing_speed² / (2 * gap)   [m/s²]
    A proxy for the required deceleration; values > 3.4 m/s² are
    considered unsafe (AASHTO standard).

PET  (Post-Encroachment Time)
    Approximated as the time gap between a merging vehicle arriving at
    the merge point and the nearest mainline vehicle passing it.

The module exposes :func:`compute_metrics` which returns a list of
:class:`FrameMetrics` (one per simulation step).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from commonroad.scenario.scenario import Scenario

TTC_MAX    = 10.0    # seconds  – cap for non-approaching pairs
DRAC_SAFE  = 3.4     # m/s²     – AASHTO "safe" deceleration threshold
MIN_GAP    = 0.1     # metres   – avoid division by zero


@dataclass
class PairMetrics:
    """Metrics for one vehicle-pair at one timestep."""
    id_a: int
    id_b: int
    gap: float           # centre-to-centre distance minus car lengths (m)
    closing_speed: float # positive = approaching (m/s)
    ttc: float           # seconds  (TTC_MAX if not converging)
    drac: float          # m/s²


@dataclass
class FrameMetrics:
    """All pair metrics at a single simulation timestep."""
    time_step: int
    time_s: float
    pairs: list[PairMetrics] = field(default_factory=list)

    @property
    def min_ttc(self) -> float:
        if not self.pairs:
            return TTC_MAX
        return min(p.ttc for p in self.pairs)

    @property
    def max_drac(self) -> float:
        if not self.pairs:
            return 0.0
        return max(p.drac for p in self.pairs)

    @property
    def n_unsafe_pairs(self) -> int:
        """Pairs with TTC < 3 s (common threshold)."""
        return sum(1 for p in self.pairs if p.ttc < 3.0)


def _get_state(obstacle, t: int):
    """Return the KSState for obstacle at timestep t, or None."""
    if obstacle.initial_state.time_step == t:
        return obstacle.initial_state
    if obstacle.prediction is None:
        return None
    for s in obstacle.prediction.trajectory.state_list:
        if s.time_step == t:
            return s
    return None


def _closing_speed(
    pos_a: np.ndarray,
    vel_a: float,
    ori_a: float,
    pos_b: np.ndarray,
    vel_b: float,
    ori_b: float,
) -> float:
    """
    Scalar closing speed along the line from a to b.
    Positive means the vehicles are approaching each other.
    """
    delta = pos_b - pos_a
    dist  = np.linalg.norm(delta)
    if dist < 1e-6:
        return 0.0
    unit = delta / dist

    v_a = vel_a * np.array([np.cos(ori_a), np.sin(ori_a)])
    v_b = vel_b * np.array([np.cos(ori_b), np.sin(ori_b)])

    # projection of relative velocity (a→b) onto the unit vector a→b
    # closing speed > 0 when a is catching up to b
    return float(np.dot(v_a - v_b, unit))


def compute_metrics(scenario: Scenario) -> list[FrameMetrics]:
    """
    Iterate over all simulation timesteps and return one FrameMetrics
    per step containing pairwise TTC / DRAC values.
    """
    obstacles = list(scenario.dynamic_obstacles)
    dt        = scenario.dt
    results   = []

    for t in range(int(scenario.dt * 0  ) , 300):   # up to 300 steps
        frame = FrameMetrics(time_step=t, time_s=t * dt)

        # Collect states at this step
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

                pos_a = np.array(sa.position)
                pos_b = np.array(sb.position)
                gap = max(float(np.linalg.norm(pos_b - pos_a)) - 4.7, MIN_GAP)

                cs = _closing_speed(
                    pos_a, float(sa.velocity), float(sa.orientation),
                    pos_b, float(sb.velocity), float(sb.orientation),
                )

                if cs > 1e-3:
                    ttc  = gap / cs
                    drac = cs ** 2 / (2.0 * gap)
                else:
                    ttc  = TTC_MAX
                    drac = 0.0

                ttc = min(ttc, TTC_MAX)

                frame.pairs.append(PairMetrics(
                    id_a=ids[i], id_b=ids[j],
                    gap=gap, closing_speed=cs,
                    ttc=ttc, drac=drac,
                ))

        results.append(frame)

    return results
