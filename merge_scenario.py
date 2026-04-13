"""
merge_scenario.py
=================
Build a CommonRoad highway-merge scenario with a parametrically curved
on-ramp and IDM-based car-following.

The merge lane follows a cubic Bezier curve controlled by:
  * merge_length  – horizontal extent of the ramp (metres)
  * curvature     – 0 = straight diagonal, 1 = fully curved S-shape

Vehicles use the Intelligent Driver Model so they maintain safe gaps,
slow for leaders, and perform zip merges — no overlapping.
"""
from __future__ import annotations

import numpy as np
from commonroad.common.common_lanelet import LaneletType
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.state import InitialState, KSState
from commonroad.scenario.trajectory import Trajectory

from idm import RampMergeSimulator, build_main_path, build_ramp_path, CAR_LEN, S0, T_HEAD

DT:         float = 0.1
STEPS:      int   = 300
LANE_WIDTH: float = 4.0
CAR_WIDTH:  float = 2.0

# Short mainline past merge when no IDM vehicles (compact map for SUMO).
MAIN_DOWNSTREAM_SHORT_M: float = 320.0

ID_MAIN_A = 1
ID_RAMP   = 2
ID_MAIN_B = 3


# ── Bezier helpers ────────────────────────────────────────────────────────────

def _cubic_bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (
        (1 - t) ** 3 * np.array(p0)
        + 3 * (1 - t) ** 2 * t * np.array(p1)
        + 3 * (1 - t) * t ** 2 * np.array(p2)
        + t ** 3 * np.array(p3)
    )


def bezier_merge_centerline(
    merge_length: float,
    y_ramp: float,
    y_main: float,
    curvature: float,
    n: int = 200,
) -> np.ndarray:
    """
    Cubic Bezier centerline for the merging ramp.

    curvature=0  → nearly straight diagonal (aggressive angle)
    curvature=1  → smooth S-curve (gentle merge)
    """
    p0 = [0.0,                                     y_ramp]
    p1 = [merge_length * (0.1 + 0.4 * curvature),  y_ramp]
    p2 = [merge_length * (0.6 - 0.2 * curvature),  y_main]
    p3 = [merge_length,                             y_main]
    return _cubic_bezier(p0, p1, p2, p3, n)


def _offset_centerline(center: np.ndarray, offset: float) -> np.ndarray:
    dx = np.diff(center[:, 0])
    dy = np.diff(center[:, 1])
    norms = np.column_stack([-dy, dx])
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1e-9, lengths)
    norms /= lengths
    normal_avg = np.vstack([norms[0], (norms[:-1] + norms[1:]) / 2, norms[-1]])
    norm_len = np.linalg.norm(normal_avg, axis=1, keepdims=True)
    norm_len = np.where(norm_len == 0, 1e-9, norm_len)
    normal_avg /= norm_len
    return center + offset * normal_avg


# ── lanelet builders ──────────────────────────────────────────────────────────

def _make_straight_lanelet(lid, x0, x1, y_ctr, predecessor=None, successor=None,
                            lanelet_type=LaneletType.INTERSTATE) -> Lanelet:
    xs     = np.linspace(x0, x1, 200)
    center = np.column_stack([xs, np.full_like(xs, y_ctr)])
    left   = np.column_stack([xs, np.full_like(xs, y_ctr + LANE_WIDTH / 2)])
    right  = np.column_stack([xs, np.full_like(xs, y_ctr - LANE_WIDTH / 2)])
    return Lanelet(left_vertices=left, center_vertices=center, right_vertices=right,
                   lanelet_id=lid, predecessor=predecessor or [], successor=successor or [],
                   lanelet_type={lanelet_type})


def _make_bezier_lanelet(lid, merge_length, y_ramp, y_main, curvature,
                          x_offset=0.0, predecessor=None, successor=None) -> Lanelet:
    center = bezier_merge_centerline(merge_length, y_ramp, y_main, curvature)
    center[:, 0] += x_offset
    left  = _offset_centerline(center, +LANE_WIDTH / 2)
    right = _offset_centerline(center, -LANE_WIDTH / 2)
    return Lanelet(left_vertices=left, center_vertices=center, right_vertices=right,
                   lanelet_id=lid, predecessor=predecessor or [], successor=successor or [],
                   lanelet_type={LaneletType.ACCESS_RAMP})


# ── CommonRoad obstacle builder ───────────────────────────────────────────────

def _make_cr_vehicle(vid: int, positions: list, orientations: list,
                     velocities: list) -> DynamicObstacle:
    shape = Rectangle(length=CAR_LEN, width=CAR_WIDTH)
    states = [
        KSState(time_step=t,
                position=np.array(positions[t]),
                velocity=float(velocities[t]),
                orientation=float(orientations[t]),
                steering_angle=0.0)
        for t in range(len(positions))
    ]
    initial_state = InitialState(
        time_step=0,
        position=np.array(positions[0]),
        velocity=float(velocities[0]),
        orientation=float(orientations[0]),
        acceleration=0.0, yaw_rate=0.0, slip_angle=0.0,
    )
    trajectory = Trajectory(initial_time_step=1, state_list=states[1:])
    prediction = TrajectoryPrediction(trajectory=trajectory, shape=shape)
    return DynamicObstacle(
        obstacle_id=vid, obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape, initial_state=initial_state, prediction=prediction,
    )


# ── public API ────────────────────────────────────────────────────────────────

def build_scenario(
    merge_length:    float = 120.0,
    curvature:       float = 0.5,
    n_main_vehicles: int   = 3,
    n_ramp_vehicles: int   = 2,
    main_speed:      float = 28.0,
    ramp_speed:      float = 22.0,
) -> Scenario:
    """
    Build a CommonRoad scenario with IDM car-following.

    Vehicles maintain safe gaps automatically — no hard-coded trajectories.
    Ramp vehicles perform a zip merge into the first available gap.
    """
    y_main = 0.0
    y_ramp = LANE_WIDTH * 2.5   # 10 m lateral offset

    x_start     = 0.0
    x_merge_end = x_start + merge_length
    if n_main_vehicles > 0 or n_ramp_vehicles > 0:
        main_tail = max(1200.0, main_speed * STEPS * DT + 200.0)
    else:
        main_tail = MAIN_DOWNSTREAM_SHORT_M
    x_downstream = x_merge_end + main_tail

    if n_main_vehicles > 0 or n_ramp_vehicles > 0:
        ramp_path = build_ramp_path(merge_length, y_ramp, curvature)
        main_path = build_main_path(x_start - 300.0, x_downstream, y_main)

        sim = RampMergeSimulator(
            ramp_path=ramp_path,
            main_path=main_path,
            dt=DT,
            steps=STEPS,
            merge_x=x_merge_end,
        )

        main_spacing = main_speed * T_HEAD + S0 + CAR_LEN
        anchor_x = x_merge_end - main_spacing * 0.5
        for i in range(n_main_vehicles):
            x0 = anchor_x - i * main_spacing
            sim.add_main_vehicle(vid=200 + i, x0=x0, v0=main_speed, v_desired=main_speed)

        ramp_spacing = ramp_speed * T_HEAD + S0 + CAR_LEN
        for j in range(n_ramp_vehicles):
            ramp_s0 = j * ramp_spacing
            sim.add_ramp_vehicle(vid=300 + j, ramp_s0=ramp_s0, v0=ramp_speed, v_desired=ramp_speed)

        vehicle_states = sim.run()
    else:
        vehicle_states = []

    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID(country_id="DEU", map_name="MERGE", map_id=1),
        author="curvature_sweep",
        source="IDM Bezier merge — commonroad-io",
        tags={Tag.INTERSTATE},
    )
    lane_a = _make_straight_lanelet(
        ID_MAIN_A, x_start, x_merge_end, y_main, successor=[ID_MAIN_B])
    lane_ramp = _make_bezier_lanelet(
        ID_RAMP, merge_length, y_ramp, y_main, curvature,
        x_offset=x_start, successor=[ID_MAIN_B])
    lane_b = _make_straight_lanelet(
        ID_MAIN_B, x_merge_end, x_downstream, y_main,
        predecessor=[ID_MAIN_A, ID_RAMP])
    scenario.add_objects(
        LaneletNetwork.create_from_lanelet_list([lane_a, lane_ramp, lane_b]))

    for vs in vehicle_states:
        pos  = vs.positions
        oris = vs.orientations
        vels = [vs.v0]
        for t in range(1, len(pos)):
            dx = pos[t][0] - pos[t - 1][0]
            dy = pos[t][1] - pos[t - 1][1]
            vels.append(float(np.hypot(dx, dy) / DT))
        cr_obs = _make_cr_vehicle(vs.vid, pos, oris, vels)
        scenario.add_objects(cr_obs)

    return scenario


def merge_viewport_xy(
    merge_length: float,
    x_before: float = 35.0,
    x_after: float = 200.0,
    y_margin: float = 16.0,
) -> list[float]:
    y_main = 0.0
    y_ramp = LANE_WIDTH * 2.5
    return [
        -x_before,
        merge_length + x_after,
        min(y_main, y_ramp) - y_margin,
        max(y_main, y_ramp) + y_margin,
    ]


def get_planning_problem_set() -> PlanningProblemSet:
    return PlanningProblemSet()
