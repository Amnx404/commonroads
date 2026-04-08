"""
simulation.py
=============
Thin wrapper that builds a scenario, runs the metrics pass, and returns
everything the GUI needs in one call.

Also provides :func:`parameter_sweep` to evaluate safety over a grid of
(merge_length, curvature) values – used for the comparison plot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from merge_scenario import build_scenario, DT, STEPS
from safety_metrics import FrameMetrics, compute_metrics, aggregate_by_parameter
from commonroad.scenario.scenario import Scenario


@dataclass
class SimResult:
    scenario:     Scenario
    frames:       list[FrameMetrics]
    merge_length: float
    curvature:    float
    main_speed:   float
    ramp_speed:   float

    # Convenience time axis
    @property
    def times(self) -> np.ndarray:
        return np.array([f.time_s for f in self.frames])

    @property
    def min_ttc_series(self) -> np.ndarray:
        return np.array([f.min_ttc for f in self.frames])

    @property
    def max_drac_series(self) -> np.ndarray:
        return np.array([f.max_drac for f in self.frames])

    @property
    def n_unsafe_series(self) -> np.ndarray:
        return np.array([f.n_unsafe_pairs for f in self.frames])


def run_simulation(
    merge_length: float = 120.0,
    curvature: float = 0.5,
    n_main: int = 3,
    n_ramp: int = 2,
    main_speed: float = 28.0,
    ramp_speed: float = 22.0,
) -> SimResult:
    """Build scenario with given parameters and compute safety metrics."""
    scenario = build_scenario(
        merge_length=merge_length,
        curvature=curvature,
        n_main_vehicles=n_main,
        n_ramp_vehicles=n_ramp,
        main_speed=main_speed,
        ramp_speed=ramp_speed,
    )
    frames = compute_metrics(scenario)
    return SimResult(
        scenario=scenario,
        frames=frames,
        merge_length=merge_length,
        curvature=curvature,
        main_speed=main_speed,
        ramp_speed=ramp_speed,
    )


# ── parameter sweeps ─────────────────────────────────────────────────────────

def sweep_merge_length(
    lengths: list[float] | None = None,
    curvature: float = 0.5,
    n_main: int = 3,
    n_ramp: int = 2,
    main_speed: float = 28.0,
    ramp_speed: float = 22.0,
) -> dict:
    """Sweep merge length, return aggregated safety metrics."""
    if lengths is None:
        lengths = [60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 220.0]
    all_frames = [
        compute_metrics(
            build_scenario(
                merge_length=L, curvature=curvature,
                n_main_vehicles=n_main, n_ramp_vehicles=n_ramp,
                main_speed=main_speed, ramp_speed=ramp_speed,
            )
        )
        for L in lengths
    ]
    result = aggregate_by_parameter(lengths, all_frames)
    result["sweep"] = "merge_length"
    return result


def sweep_curvature(
    curvatures: list[float] | None = None,
    merge_length: float = 120.0,
    n_main: int = 3,
    n_ramp: int = 2,
    main_speed: float = 28.0,
    ramp_speed: float = 22.0,
) -> dict:
    """Sweep curvature parameter, return aggregated safety metrics."""
    if curvatures is None:
        curvatures = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    all_frames = [
        compute_metrics(
            build_scenario(
                merge_length=merge_length, curvature=c,
                n_main_vehicles=n_main, n_ramp_vehicles=n_ramp,
                main_speed=main_speed, ramp_speed=ramp_speed,
            )
        )
        for c in curvatures
    ]
    result = aggregate_by_parameter(curvatures, all_frames)
    result["sweep"] = "curvature"
    return result
