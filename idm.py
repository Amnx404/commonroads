"""
idm.py
======
Intelligent Driver Model (IDM) car-following simulation.

Uses world-x (metres, increasing rightward) as the common position
coordinate for all leader-finding comparisons, avoiding the arc-length
origin mismatch between the ramp path and the main-lane path.
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
    arc_s:        float          # arc-length along own path (ramp or main)
    v:            float          # speed (m/s)
    v0:           float          # desired speed (m/s)
    merged:       bool = False   # True once ramp vehicle finishes merging

    positions:    list = field(default_factory=list)
    orientations: list = field(default_factory=list)


# ── Simulator ─────────────────────────────────────────────────────────────────

class RampMergeSimulator:
    """
    IDM simulation for a highway merge.

    All spatial comparisons use world-x so ramp and main-lane
    vehicles share a single consistent coordinate.
    """

    def __init__(
        self,
        ramp_path:    _Path,
        main_path:    _Path,
        dt:           float = 0.1,
        steps:        int   = 300,
        merge_x:      float = 120.0,   # world-x of merge end
    ):
        self.ramp_path = ramp_path
        self.main_path = main_path
        self.dt        = dt
        self.steps     = steps
        self.merge_x   = merge_x   # world-x at which ramp meets main lane
        self._vehicles: list[VehicleState] = []

    # ── add vehicles ──────────────────────────────────────────────────────

    def add_main_vehicle(self, vid: int, x0: float, v0: float, v_desired: float):
        """x0 = initial world-x position."""
        arc_s = self.main_path.s_at_x(x0)
        vs = VehicleState(vid=vid, is_ramp=False, arc_s=arc_s, v=v0, v0=v_desired, merged=True)
        self._vehicles.append(vs)

    def add_ramp_vehicle(self, vid: int, ramp_s0: float, v0: float, v_desired: float):
        """ramp_s0 = initial arc-length position along the ramp path."""
        vs = VehicleState(vid=vid, is_ramp=True, arc_s=ramp_s0, v=v0, v0=v_desired, merged=False)
        self._vehicles.append(vs)

    # ── world-x helper ────────────────────────────────────────────────────

    def _world_x(self, vs: VehicleState) -> float:
        """Current world-x of a vehicle's front bumper centre."""
        if vs.is_ramp and not vs.merged:
            xy, _ = self.ramp_path.at(vs.arc_s)
            return float(xy[0])
        else:
            xy, _ = self.main_path.at(vs.arc_s)
            return float(xy[0])

    # ── position recording ────────────────────────────────────────────────

    def _record(self, vs: VehicleState):
        if vs.is_ramp and not vs.merged:
            xy, hdg = self.ramp_path.at(vs.arc_s)
        else:
            xy, hdg = self.main_path.at(vs.arc_s)
            hdg = 0.0
        vs.positions.append((float(xy[0]), float(xy[1])))
        vs.orientations.append(float(hdg))

    # ── leader finding ────────────────────────────────────────────────────

    def _find_leader(self, ego: VehicleState) -> tuple[VehicleState | None, float, float]:
        """
        Return (leader, gap_m, dv) for the most critical leader.
        gap_m is the bumper-to-bumper gap in world-x metres.
        dv is closing speed (positive = approaching).
        """
        ego_x = self._world_x(ego)

        best_leader = None
        best_gap    = np.inf

        for other in self._vehicles:
            if other.vid == ego.vid:
                continue

            # Ramp vehicle that hasn't yet merged only counts as a leader
            # for other ramp vehicles behind it on the same ramp.
            if other.is_ramp and not other.merged:
                if not (ego.is_ramp and not ego.merged):
                    continue   # don't follow unmerged ramp from main lane
                # Both on ramp: pure arc-length comparison
                if other.arc_s <= ego.arc_s:
                    continue
                g = other.arc_s - ego.arc_s - CAR_LEN
                if g < best_gap:
                    best_gap    = g
                    best_leader = other
                continue

            # Other is on main lane (or merged ramp)
            other_x = self._world_x(other)
            if other_x <= ego_x:
                continue   # behind ego

            g = other_x - ego_x - CAR_LEN
            if g < best_gap:
                best_gap    = g
                best_leader = other

        # ── zip-merge anticipation ────────────────────────────────────────
        # Ramp vehicle near merge end: also check the main-lane vehicle that
        # will be alongside / just behind at the merge point.
        if ego.is_ramp and not ego.merged:
            dist_to_merge = self.ramp_path.total - ego.arc_s
            if dist_to_merge <= MERGE_LOOKAHEAD:
                t_arrive = dist_to_merge / max(ego.v, 0.1)
                for other in self._vehicles:
                    if other.is_ramp or other.vid == ego.vid:
                        continue
                    other_x = self._world_x(other)
                    proj_x  = other_x + other.v * t_arrive
                    # Gap between projected other and merge point
                    g_at_merge = proj_x - self.merge_x - CAR_LEN
                    if g_at_merge < 0:
                        # Other will be at/behind merge end when ego arrives
                        # — ego must slow to let it clear first.
                        effective_gap = dist_to_merge + max(g_at_merge, -CAR_LEN)
                        effective_gap = max(effective_gap, 0.5)
                        if effective_gap < best_gap:
                            best_gap    = effective_gap
                            best_leader = other

        if best_leader is None:
            return None, np.inf, 0.0

        # Compute closing speed
        dv = ego.v - best_leader.v
        return best_leader, max(best_gap, 0.01), dv

    # ── merge check ───────────────────────────────────────────────────────

    def _check_merge(self, vs: VehicleState):
        """Transition ramp vehicle to main lane once it reaches the merge end."""
        if not vs.is_ramp or vs.merged:
            return
        if vs.arc_s >= self.ramp_path.total:
            vs.merged = True
            # Switch arc_s to main-path arc at merge_x
            vs.arc_s  = self.main_path.s_at_x(self.merge_x)

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

def build_ramp_path(merge_length: float, y_ramp: float, curvature: float) -> _Path:
    from merge_scenario import bezier_merge_centerline
    pts = bezier_merge_centerline(merge_length, y_ramp, 0.0, curvature, n=500)
    return _Path(pts)


def build_main_path(x_start: float, x_end: float, y: float = 0.0) -> _Path:
    xs  = np.linspace(x_start, x_end, 1000)
    pts = np.column_stack([xs, np.full_like(xs, y)])
    return _Path(pts)
