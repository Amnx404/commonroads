"""
idm.py
======
Intelligent Driver Model (IDM) car-following simulation.

Supports multiple main lanes.  Each main vehicle follows only vehicles in the
same lane.  Ramp vehicles follow ramp-mates via arc-length and anticipate the
lane-0 main vehicles near the merge end (zip-merge logic).

Uses world-x (metres, increasing rightward) as the common spatial coordinate
for all leader-finding comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ── IDM parameters ────────────────────────────────────────────────────────────
A_MAX   = 1.5    # max acceleration      (m/s²)
B_COMF  = 2.0    # comfortable braking   (m/s²)
B_HARD  = 6.0    # emergency braking cap (m/s²)
T_HEAD  = 1.6    # safe time headway     (s)
S0      = 2.5    # minimum gap           (m)
DELTA   = 4.0    # acceleration exponent
CAR_LEN = 4.7    # vehicle length        (m)

MERGE_LOOKAHEAD = 80.0   # m before merge end that zip-merge logic activates


def idm_accel(v: float, v0: float, s: float, dv: float) -> float:
    """IDM acceleration (m/s²)."""
    s = max(s, 0.01)
    s_star = S0 + max(0.0, v * T_HEAD + v * dv / (2.0 * np.sqrt(A_MAX * B_COMF)))
    a = A_MAX * (1.0 - (v / max(v0, 0.01)) ** DELTA - (s_star / s) ** 2)
    return float(np.clip(a, -B_HARD, A_MAX))


# ── Path ──────────────────────────────────────────────────────────────────────

class _Path:
    """Arc-length parameterised path, with extrapolation past the end."""

    def __init__(self, pts: np.ndarray):
        diffs         = np.diff(pts, axis=0)
        seg_len       = np.linalg.norm(diffs, axis=1)
        self.arc      = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.total    = float(self.arc[-1])
        self.pts      = pts
        self.diffs    = diffs
        self.seg_len  = seg_len
        self.tangents = np.arctan2(diffs[:, 1], diffs[:, 0])

    def at(self, s: float) -> tuple[np.ndarray, float]:
        if s >= self.total:
            overshoot = s - self.total
            hdg = float(self.tangents[-1])
            xy  = self.pts[-1] + overshoot * np.array([np.cos(hdg), np.sin(hdg)])
            return xy, hdg
        s   = max(s, 0.0)
        idx = int(np.searchsorted(self.arc, s, side="right")) - 1
        idx = min(idx, len(self.arc) - 2)
        frac = (s - self.arc[idx]) / (self.seg_len[idx] + 1e-9)
        xy   = self.pts[idx] + frac * self.diffs[idx]
        return xy, float(self.tangents[idx])

    def s_at_x(self, x: float) -> float:
        """Arc-length at a given world-x (assumes monotone x along path)."""
        xs = self.pts[:, 0]
        idx = int(np.searchsorted(xs, x, side="right")) - 1
        idx = min(max(idx, 0), len(xs) - 2)
        frac = (x - xs[idx]) / (xs[idx+1] - xs[idx] + 1e-9)
        return float(self.arc[idx] + frac * (self.arc[idx+1] - self.arc[idx]))


# ── Vehicle state ─────────────────────────────────────────────────────────────

@dataclass
class VehicleState:
    vid:          int
    is_ramp:      bool
    lane_id:      int              # index into main_paths; 0 = rightmost/merge lane
    arc_s:        float            # arc-length along own path
    v:            float            # speed (m/s)
    v0:           float            # desired speed (m/s)
    merged:       bool = False     # True once ramp vehicle transitions to main lane

    positions:    list = field(default_factory=list)
    orientations: list = field(default_factory=list)


# ── Simulator ─────────────────────────────────────────────────────────────────

class RampMergeSimulator:
    """
    IDM simulation for a multi-lane highway merge.

    main_paths : list of _Path, one per main lane.
                 index 0 = rightmost lane (the one the ramp merges into).
    ramp_path  : _Path for the on-ramp (Bezier curve).
    """

    def __init__(
        self,
        main_paths:   list,        # list[_Path], index = lane_id
        ramp_path:    _Path,
        dt:           float = 0.1,
        steps:        int   = 200,
        merge_x:      float = 120.0,
    ):
        self.main_paths = main_paths
        self.ramp_path  = ramp_path
        self.dt         = dt
        self.steps      = steps
        self.merge_x    = merge_x
        self._vehicles: list[VehicleState] = []

    # ── add vehicles ──────────────────────────────────────────────────────

    def add_main_vehicle(self, vid: int, lane_id: int, x0: float,
                         v0: float, v_desired: float):
        """x0 = initial world-x; lane_id = index into main_paths."""
        arc_s = self.main_paths[lane_id].s_at_x(x0)
        vs = VehicleState(vid=vid, is_ramp=False, lane_id=lane_id,
                          arc_s=arc_s, v=v0, v0=v_desired, merged=True)
        self._vehicles.append(vs)

    def add_ramp_vehicle(self, vid: int, ramp_s0: float, v0: float, v_desired: float):
        """ramp_s0 = initial arc-length along the ramp.  Merges into lane 0."""
        vs = VehicleState(vid=vid, is_ramp=True, lane_id=0,
                          arc_s=ramp_s0, v=v0, v0=v_desired, merged=False)
        self._vehicles.append(vs)

    # ── path helpers ──────────────────────────────────────────────────────

    def _get_path(self, vs: VehicleState) -> _Path:
        if vs.is_ramp and not vs.merged:
            return self.ramp_path
        return self.main_paths[vs.lane_id]

    def _world_x(self, vs: VehicleState) -> float:
        xy, _ = self._get_path(vs).at(vs.arc_s)
        return float(xy[0])

    # ── position recording ────────────────────────────────────────────────

    def _record(self, vs: VehicleState):
        path = self._get_path(vs)
        xy, hdg = path.at(vs.arc_s)
        if not (vs.is_ramp and not vs.merged):
            hdg = 0.0          # main-lane vehicles always face east
        vs.positions.append((float(xy[0]), float(xy[1])))
        vs.orientations.append(float(hdg))

    # ── leader finding ────────────────────────────────────────────────────

    def _find_leader(self, ego: VehicleState) -> tuple[VehicleState | None, float, float]:
        """
        Return (leader, gap_m, dv).
        Main-lane vehicles only follow same-lane vehicles.
        Ramp vehicles follow ramp-mates and use zip-merge anticipation.
        """
        ego_x    = self._world_x(ego)
        best_leader = None
        best_gap    = np.inf

        for other in self._vehicles:
            if other.vid == ego.vid:
                continue

            # ── unmerged ramp vehicle ─────────────────────────────────────
            if other.is_ramp and not other.merged:
                if ego.is_ramp and not ego.merged:
                    # Same ramp: arc-length comparison
                    if other.arc_s <= ego.arc_s:
                        continue
                    g = other.arc_s - ego.arc_s - CAR_LEN
                    if g < best_gap:
                        best_gap    = g
                        best_leader = other
                # Main-lane ego ignores unmerged ramp vehicles
                continue

            # ── main-lane / merged vehicle ────────────────────────────────
            # Ramp ego (not merged) does not follow main vehicles here;
            # zip-merge anticipation is handled separately below.
            if ego.is_ramp and not ego.merged:
                continue

            # Same-lane check for main-lane ego
            if other.lane_id != ego.lane_id:
                continue

            other_x = self._world_x(other)
            if other_x <= ego_x:
                continue

            g = other_x - ego_x - CAR_LEN
            if g < best_gap:
                best_gap    = g
                best_leader = other

        # ── zip-merge anticipation ────────────────────────────────────────
        if ego.is_ramp and not ego.merged:
            dist_to_merge = self.ramp_path.total - ego.arc_s
            if dist_to_merge <= MERGE_LOOKAHEAD:
                t_arrive = dist_to_merge / max(ego.v, 0.1)
                for other in self._vehicles:
                    if other.is_ramp or other.lane_id != 0:
                        continue
                    other_x = self._world_x(other)
                    proj_x  = other_x + other.v * t_arrive
                    g_at_merge = proj_x - self.merge_x - CAR_LEN
                    if g_at_merge < 0:
                        effective_gap = dist_to_merge + max(g_at_merge, -CAR_LEN)
                        effective_gap = max(effective_gap, 0.5)
                        if effective_gap < best_gap:
                            best_gap    = effective_gap
                            best_leader = other

        if best_leader is None:
            return None, np.inf, 0.0

        dv = ego.v - best_leader.v
        return best_leader, max(best_gap, 0.01), dv

    # ── merge check ───────────────────────────────────────────────────────

    def _check_merge(self, vs: VehicleState):
        """Transition ramp vehicle to lane-0 main path at merge end."""
        if not vs.is_ramp or vs.merged:
            return
        if vs.arc_s >= self.ramp_path.total:
            vs.merged = True
            vs.arc_s  = self.main_paths[0].s_at_x(self.merge_x)

    # ── simulation step ───────────────────────────────────────────────────

    def _step(self):
        accels: dict[int, float] = {}
        for vs in self._vehicles:
            _, gap, dv = self._find_leader(vs)
            accels[vs.vid] = idm_accel(vs.v, vs.v0, gap, dv)

        for vs in self._vehicles:
            vs.v     = max(0.0, vs.v + accels[vs.vid] * self.dt)
            vs.arc_s += vs.v * self.dt
            self._check_merge(vs)

    # ── run ───────────────────────────────────────────────────────────────

    def run(self) -> list[VehicleState]:
        for vs in self._vehicles:
            self._record(vs)
        for _ in range(self.steps - 1):
            self._step()
            for vs in self._vehicles:
                self._record(vs)
        return self._vehicles


# ── factories ─────────────────────────────────────────────────────────────────

def build_ramp_path(merge_length: float, y_ramp: float,
                    y_main: float = 0.0, curvature: float = 0.5) -> _Path:
    from merge_scenario import bezier_merge_centerline
    pts = bezier_merge_centerline(merge_length, y_ramp, y_main, curvature, n=500)
    return _Path(pts)


def build_main_path(x_start: float, x_end: float, y: float = 0.0) -> _Path:
    xs  = np.linspace(x_start, x_end, 1000)
    pts = np.column_stack([xs, np.full_like(xs, y)])
    return _Path(pts)
