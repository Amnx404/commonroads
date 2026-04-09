"""
simulation.py
=============
Build a merge scenario and return a SimResult ready for safety scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from merge_scenario import build_scenario, DT, STEPS
from safety_metrics import FrameMetrics, compute_metrics
from commonroad.scenario.scenario import Scenario


@dataclass
class SimResult:
    scenario:     Scenario
    frames:       list[FrameMetrics]
    merge_length: float
    curvature:    float
    main_speed:   float
    ramp_speed:   float

    @property
    def times(self) -> np.ndarray:
        return np.array([f.time_s for f in self.frames])


def run_simulation(
    merge_length: float = 120.0,
    curvature:    float = 0.5,
    n_main:       int   = 3,
    n_ramp:       int   = 2,
    main_speed:   float = 28.0,
    ramp_speed:   float = 22.0,
) -> SimResult:
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
