"""
merge_scenario.py
=================
Build a CommonRoad highway-merge scenario with three main lanes plus one
parametric Bezier on-ramp.

Road geometry
-------------
  Lane 0 (rightmost / merge lane)   y =  0 m
  Lane 1 (middle)                   y =  4 m
  Lane 2 (left / fastest)           y =  8 m
  On-ramp                           y = -10 m → curves up to y = 0

Traffic
-------
  n_main_vehicles vehicles per lane, spaced at IDM comfortable headway.
  n_ramp_vehicles on the ramp.

The on-ramp follows a cubic Bezier curve controlled by:
  merge_length  – horizontal extent of the ramp (metres)
  curvature     – 0 = straight diagonal, 1 = fully curved S-shape
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

# ── constants ──────────────────────────────────────────────────────────────────
DT:         float = 0.1
STEPS:      int   = 200     # 20 seconds of simulation
LANE_WIDTH: float = 4.0
CAR_WIDTH:  float = 2.0
N_LANES:    int   = 3

# Lane y-coordinates: index 0 = rightmost (merge lane)
Y_LANES = [LANE_WIDTH * i for i in range(N_LANES)]   # [0.0, 4.0, 8.0]
Y_RAMP  = -LANE_WIDTH * 2.5                           # -10 m (below lane 0)

# CommonRoad lanelet IDs
_LID_A    = [10, 11, 12]   # upstream lanes 0,1,2
_LID_RAMP = 20
_LID_B    = [30, 31, 32]   # downstream lanes 0,1,2


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
    p0 = [0.0,                                    y_ramp]
    p1 = [merge_length * (0.1 + 0.4 * curvature), y_ramp]
    p2 = [merge_length * (0.6 - 0.2 * curvature), y_main]
    p3 = [merge_length,                            y_main]
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

def _make_straight_lanelet(lid, x0, x1, y_ctr,
                            predecessor=None, successor=None,
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
    shape  = Rectangle(length=CAR_LEN, width=CAR_WIDTH)
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
    n_main_vehicles: int   = 6,     # per lane
    n_ramp_vehicles: int   = 3,
    main_speed:      float = 28.0,
    ramp_speed:      float = 22.0,
) -> Scenario:
    """
    Build a 3-lane + on-ramp CommonRoad scenario with IDM car-following.

    Three parallel main lanes (y = 0, 4, 8 m) run from x_upstream to
    x_downstream.  The on-ramp curves from y = -10 m into lane 0 (y = 0)
    at x = merge_length.
    """
    x_start      = 0.0
    x_merge_end  = x_start + merge_length
    x_upstream   = x_start - 400.0
    x_downstream = x_merge_end + max(600.0, main_speed * STEPS * DT + 200.0)

    # ── paths ─────────────────────────────────────────────────────────────
    ramp_path  = build_ramp_path(merge_length, Y_RAMP, Y_LANES[0], curvature)
    main_paths = [build_main_path(x_upstream, x_downstream, y) for y in Y_LANES]

    # ── IDM simulator ─────────────────────────────────────────────────────
    sim = RampMergeSimulator(
        main_paths=main_paths,
        ramp_path=ramp_path,
        dt=DT,
        steps=STEPS,
        merge_x=x_merge_end,
    )

    # Main lane vehicles: anchor first vehicle just upstream of merge, then
    # space at IDM comfortable headway distance.
    main_spacing = main_speed * T_HEAD + S0 + CAR_LEN  # ≈ 52 m
    anchor_x     = x_merge_end - main_spacing * 0.5

    for lane_id in range(N_LANES):
        for i in range(n_main_vehicles):
            x0  = anchor_x - i * main_spacing
            vid = 100 + lane_id * 10 + i
            sim.add_main_vehicle(vid=vid, lane_id=lane_id,
                                 x0=x0, v0=main_speed, v_desired=main_speed)

    # Ramp vehicles: start near the beginning of the ramp
    ramp_spacing = ramp_speed * T_HEAD + S0 + CAR_LEN  # ≈ 42 m
    for j in range(n_ramp_vehicles):
        ramp_s0 = j * ramp_spacing
        sim.add_ramp_vehicle(vid=200 + j, ramp_s0=ramp_s0,
                              v0=ramp_speed, v_desired=ramp_speed)

    # ── run IDM simulation ─────────────────────────────────────────────────
    vehicle_states = sim.run()

    # ── build CommonRoad road network ─────────────────────────────────────
    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID(country_id="DEU", map_name="MERGE", map_id=1),
        author="merge_gui",
        source="IDM Bezier merge",
        tags={Tag.INTERSTATE},
    )

    lanelets = []
    for i, y in enumerate(Y_LANES):
        pred_b = [_LID_A[i]] + ([_LID_RAMP] if i == 0 else [])
        la = _make_straight_lanelet(
            _LID_A[i], x_upstream, x_merge_end, y,
            successor=[_LID_B[i]])
        lb = _make_straight_lanelet(
            _LID_B[i], x_merge_end, x_downstream, y,
            predecessor=pred_b)
        lanelets.extend([la, lb])

    lane_ramp = _make_bezier_lanelet(
        _LID_RAMP, merge_length, Y_RAMP, Y_LANES[0], curvature,
        x_offset=x_start, successor=[_LID_B[0]])
    lanelets.append(lane_ramp)

    scenario.add_objects(LaneletNetwork.create_from_lanelet_list(lanelets))

    # ── add vehicles with IDM-computed trajectories ────────────────────────
    for vs in vehicle_states:
        pos  = vs.positions
        oris = vs.orientations
        vels = [vs.v0]
        for t in range(1, len(pos)):
            dx = pos[t][0] - pos[t-1][0]
            dy = pos[t][1] - pos[t-1][1]
            vels.append(float(np.hypot(dx, dy) / DT))

        cr_obs = _make_cr_vehicle(vs.vid, pos, oris, vels)
        scenario.add_objects(cr_obs)

    return scenario


def get_planning_problem_set() -> PlanningProblemSet:
    return PlanningProblemSet()
