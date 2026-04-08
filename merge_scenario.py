"""
merge_scenario.py
=================
Build a CommonRoad highway-merge scenario with a parametrically curved
on-ramp.  The merge lane follows a cubic Bezier curve whose shape is
controlled by two scalar parameters:

  * merge_length  – horizontal extent of the ramp (metres, default 120)
  * curvature     – 0 = straight diagonal, 1 = fully curved S-shape

Multiple vehicles are spawned on both the mainline and the ramp so that
several merge interactions (and a range of TTC values) occur within a
single 30-second run.
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

# ── simulation constants ────────────────────────────────────────────────────
DT: float = 0.1          # seconds per step
STEPS: int = 300          # 30 s total
LANE_WIDTH: float = 4.0   # metres
CAR_LENGTH: float = 4.7
CAR_WIDTH: float = 2.0

# lane IDs
ID_MAIN_A = 1   # mainline upstream
ID_RAMP    = 2   # on-ramp / merge lane
ID_MAIN_B = 3   # mainline downstream


# ── Bezier helpers ──────────────────────────────────────────────────────────

def _cubic_bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    """Return (n, 2) array of points on the cubic Bezier through p0..p3."""
    t = np.linspace(0.0, 1.0, n)
    t = t[:, None]
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
    Cubic Bezier centreline for the merge lane.

    Parameters
    ----------
    merge_length : float   horizontal length of the ramp section (m)
    y_ramp       : float   lateral position at the ramp start
    y_main       : float   lateral position at merge end (main lane centre)
    curvature    : float   0 = straight, 1 = fully curved (S-shape)
    n            : int     number of sample points

    Returns
    -------
    (n, 2) array  [x, y]
    """
    # Control points – tangent handles shift with `curvature`
    p0 = [0.0,          y_ramp]
    p1 = [merge_length * (0.1 + 0.4 * curvature), y_ramp]
    p2 = [merge_length * (0.6 - 0.2 * curvature), y_main]
    p3 = [merge_length,  y_main]
    return _cubic_bezier(p0, p1, p2, p3, n)


def _offset_centerline(
    center: np.ndarray, offset: float
) -> np.ndarray:
    """Perpendicular offset of a polyline by `offset` metres (left = +)."""
    dx = np.diff(center[:, 0])
    dy = np.diff(center[:, 1])
    # unit normals (rotate tangent 90° left)
    norms = np.column_stack([-dy, dx])
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1e-9, lengths)
    norms /= lengths
    # average adjacent normals for interior points
    normal_avg = np.vstack([norms[0], (norms[:-1] + norms[1:]) / 2, norms[-1]])
    norm_len = np.linalg.norm(normal_avg, axis=1, keepdims=True)
    norm_len = np.where(norm_len == 0, 1e-9, norm_len)
    normal_avg /= norm_len
    return center + offset * normal_avg


# ── lanelet builders ────────────────────────────────────────────────────────

def _make_straight_lanelet(
    lid: int,
    x0: float, x1: float,
    y_ctr: float,
    predecessor=None,
    successor=None,
    lanelet_type=LaneletType.INTERSTATE,
) -> Lanelet:
    xs = np.linspace(x0, x1, 200)
    center = np.column_stack([xs, np.full_like(xs, y_ctr)])
    left   = np.column_stack([xs, np.full_like(xs, y_ctr + LANE_WIDTH / 2)])
    right  = np.column_stack([xs, np.full_like(xs, y_ctr - LANE_WIDTH / 2)])
    return Lanelet(
        left_vertices=left,
        center_vertices=center,
        right_vertices=right,
        lanelet_id=lid,
        predecessor=predecessor or [],
        successor=successor or [],
        lanelet_type={lanelet_type},
    )


def _make_bezier_lanelet(
    lid: int,
    merge_length: float,
    y_ramp: float,
    y_main: float,
    curvature: float,
    x_offset: float = 0.0,
    predecessor=None,
    successor=None,
) -> Lanelet:
    center = bezier_merge_centerline(merge_length, y_ramp, y_main, curvature)
    center[:, 0] += x_offset
    left  = _offset_centerline(center, +LANE_WIDTH / 2)
    right = _offset_centerline(center, -LANE_WIDTH / 2)
    return Lanelet(
        left_vertices=left,
        center_vertices=center,
        right_vertices=right,
        lanelet_id=lid,
        predecessor=predecessor or [],
        successor=successor or [],
        lanelet_type={LaneletType.ACCESS_RAMP},
    )


# ── vehicle builder ─────────────────────────────────────────────────────────

def _make_vehicle(
    vid: int,
    positions: list[tuple[float, float]],
    velocity: float,
    orientation: float | list[float],
) -> DynamicObstacle:
    """
    orientation may be a single float (constant) or a list of per-step floats.
    """
    def _ori(t):
        if isinstance(orientation, (list, np.ndarray)):
            return float(orientation[t])
        return float(orientation)

    shape = Rectangle(length=CAR_LENGTH, width=CAR_WIDTH)
    states = [
        KSState(
            time_step=t,
            position=np.array([x, y]),
            velocity=velocity,
            orientation=_ori(t),
            steering_angle=0.0,
        )
        for t, (x, y) in enumerate(positions)
    ]
    initial_state = InitialState(
        time_step=0,
        position=np.array(positions[0]),
        velocity=velocity,
        orientation=_ori(0),
        acceleration=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
    )
    trajectory  = Trajectory(initial_time_step=1, state_list=states[1:])
    prediction  = TrajectoryPrediction(trajectory=trajectory, shape=shape)
    return DynamicObstacle(
        obstacle_id=vid,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=initial_state,
        prediction=prediction,
    )


# ── trajectory generators ───────────────────────────────────────────────────

def _straight_positions(
    x0: float, y: float, speed: float, steps: int
) -> list[tuple[float, float]]:
    return [(x0 + speed * DT * t, y) for t in range(steps)]


def _ramp_positions(
    x_offset: float,
    merge_length: float,
    y_ramp: float,
    y_main: float,
    curvature: float,
    speed: float,
    start_step: int = 0,
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Travel along the Bezier centreline at roughly constant speed, then
    continue straight on the main lane.

    Returns
    -------
    positions   : list of (x, y)
    orientations: list of heading angle (radians) at each step
    """
    center = bezier_merge_centerline(merge_length, y_ramp, y_main, curvature)
    center[:, 0] += x_offset

    diffs   = np.diff(center, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc     = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_arc = arc[-1]

    # Tangent angles along the curve
    tangent_angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    positions    : list[tuple[float, float]] = []
    orientations : list[float] = []

    for t in range(STEPS):
        s = speed * DT * (t - start_step)
        if t < start_step:
            positions.append((center[0, 0], center[0, 1]))
            orientations.append(float(tangent_angles[0]))
        elif s <= total_arc:
            idx  = np.searchsorted(arc, s, side="right") - 1
            idx  = min(idx, len(arc) - 2)
            frac = (s - arc[idx]) / (seg_len[idx] + 1e-9)
            xi   = center[idx, 0] + frac * diffs[idx, 0]
            yi   = center[idx, 1] + frac * diffs[idx, 1]
            positions.append((xi, yi))
            orientations.append(float(tangent_angles[idx]))
        else:
            overshoot = s - total_arc
            positions.append((center[-1, 0] + overshoot, y_main))
            orientations.append(0.0)   # straight after merge

    return positions, orientations


# ── public API ───────────────────────────────────────────────────────────────

def build_scenario(
    merge_length: float = 120.0,
    curvature: float = 0.5,
    n_main_vehicles: int = 3,
    n_ramp_vehicles: int = 2,
    main_speed: float = 28.0,   # ~100 km/h
    ramp_speed: float = 22.0,   # ~80 km/h
) -> Scenario:
    """
    Construct and return a CommonRoad Scenario for a highway merge.

    Parameters
    ----------
    merge_length    : float  – Bezier ramp length in metres
    curvature       : float  – 0 (straight) … 1 (curved)
    n_main_vehicles : int    – vehicles on the main lane
    n_ramp_vehicles : int    – vehicles entering via the ramp
    main_speed      : float  – mainline speed (m/s)
    ramp_speed      : float  – ramp vehicle speed (m/s)
    """
    scenario = Scenario(
        dt=DT,
        scenario_id=ScenarioID(country_id="DEU", map_name="MERGE", map_id=1),
        author="merge_gui",
        source="Procedural Bezier merge",
        tags={Tag.INTERSTATE},
    )

    x_upstream   = 0.0
    x_merge_end  = x_upstream + merge_length
    x_downstream = x_merge_end + 300.0

    y_main = 0.0
    y_ramp = LANE_WIDTH * 2.5   # ramp starts 10 m to the left

    # ── road network ─────────────────────────────────────────────────────
    lane_a = _make_straight_lanelet(
        ID_MAIN_A, x_upstream, x_merge_end, y_main, successor=[ID_MAIN_B]
    )
    lane_ramp = _make_bezier_lanelet(
        ID_RAMP, merge_length, y_ramp, y_main, curvature,
        x_offset=x_upstream, successor=[ID_MAIN_B]
    )
    lane_b = _make_straight_lanelet(
        ID_MAIN_B, x_merge_end, x_downstream, y_main,
        predecessor=[ID_MAIN_A, ID_RAMP]
    )
    network = LaneletNetwork.create_from_lanelet_list([lane_a, lane_ramp, lane_b])
    scenario.add_objects(network)

    # ── time the vehicles to actually interact at the merge point ───────
    # Approximate arc length of the Bezier ramp ≈ merge_length (close enough)
    # Ramp vehicle j arrives at merge point at t = (merge_length / ramp_speed) + j*3 s
    # Place main vehicles so they pass the merge point at the same time
    # (staggered by ±headway so some create conflicts, some don't)

    # Nominal arrival time of ramp vehicle 0 at merge point
    t_arrive_ramp0 = merge_length / ramp_speed   # seconds

    # Main-vehicle headway (seconds between consecutive vehicles)
    headway_s = 2.5

    for i in range(n_main_vehicles):
        # Offset each main vehicle's arrival time around t_arrive_ramp0
        # so we get both safe (large gap) and unsafe (small gap) interactions
        offset = (i - (n_main_vehicles - 1) / 2.0) * headway_s
        arrival_t = t_arrive_ramp0 + offset   # seconds
        # x0 such that x0 + main_speed * arrival_t = x_merge_end
        x0 = x_merge_end - main_speed * arrival_t
        positions = _straight_positions(x0, y_main, main_speed, STEPS)
        vehicle = _make_vehicle(200 + i, positions, main_speed, 0.0)
        scenario.add_objects(vehicle)

    # ── ramp vehicles (staggered by 3 s) ────────────────────────────────
    for j in range(n_ramp_vehicles):
        start_step = j * 30   # 3 s stagger
        positions, orientations = _ramp_positions(
            x_upstream, merge_length, y_ramp, y_main,
            curvature, ramp_speed, start_step
        )
        vehicle = _make_vehicle(300 + j, positions, ramp_speed, orientations)
        scenario.add_objects(vehicle)

    return scenario


def get_planning_problem_set() -> PlanningProblemSet:
    return PlanningProblemSet()
