"""Crop CR scenario to merge-vicinity lanelets (around ACCESS_RAMP geometry).

Default mode is *bbox*: keep every lanelet that has at least one center vertex
inside an axis-aligned box around all ACCESS_RAMP lanelets, expanded by --pad-m.
Optional: also keep all lanelets that share a CommonRoad *intersection* with any
lanelet already selected (--expand-intersections).

Legacy *hop* mode: BFS up to --max-hop along predecessor/successor from ramp lanelets
(the old behaviour; easy to under/over-shoot).

Usage:
  uv run python tools/crop_merge_side_lanelets.py in.cr.xml out.cr.xml
  uv run python tools/crop_merge_side_lanelets.py in.cr.xml out.cr.xml --pad-m 200
  uv run python tools/crop_merge_side_lanelets.py in.cr.xml out.cr.xml --mode hop --max-hop 4
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from commonroad.common.common_lanelet import LaneletType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import ScenarioID


def patch_incoming_ids(cr_xml: Path) -> Path:
    patched = cr_xml.with_suffix(cr_xml.suffix + ".patched.xml")
    if patched.is_file():
        return patched
    tree = ET.parse(cr_xml)
    root = tree.getroot()
    for i, inc in enumerate(root.iter("incoming")):
        inc.set("id", str(100000 + i))
    tree.write(patched, encoding="utf-8", xml_declaration=True)
    return patched


def _ramp_vertex_bbox(ln, pad_m: float) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for la in ln.lanelets:
        if LaneletType.ACCESS_RAMP in la.lanelet_type:
            c = np.asarray(la.center_vertices, dtype=float)
            xs.extend(c[:, 0].tolist())
            ys.extend(c[:, 1].tolist())
    if not xs:
        raise ValueError("No ACCESS_RAMP lanelets in scenario.")
    return (
        float(min(xs) - pad_m),
        float(max(xs) + pad_m),
        float(min(ys) - pad_m),
        float(max(ys) + pad_m),
    )


def _lanelet_touches_bbox(la, bbox: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = bbox
    c = np.asarray(la.center_vertices, dtype=float)
    if c.size == 0:
        return False
    m = (c[:, 0] >= xmin) & (c[:, 0] <= xmax) & (c[:, 1] >= ymin) & (c[:, 1] <= ymax)
    return bool(np.any(m))


def merge_side_lanelet_ids_bbox(scenario, pad_m: float) -> set[int]:
    ln = scenario.lanelet_network
    bbox = _ramp_vertex_bbox(ln, pad_m)
    return {la.lanelet_id for la in ln.lanelets if _lanelet_touches_bbox(la, bbox)}


def merge_side_lanelet_ids_expand_intersections(ln, keep: set[int]) -> set[int]:
    out = set(keep)
    for inc in ln.intersections:
        touched: set[int] = set()
        for incoming in inc.incomings:
            touched |= set(incoming.incoming_lanelets)
        if touched & out:
            out |= touched
    return out


def merge_side_lanelet_ids_hop(scenario, max_hop: int) -> set[int]:
    ln = scenario.lanelet_network
    starts = [la.lanelet_id for la in ln.lanelets if LaneletType.ACCESS_RAMP in la.lanelet_type]
    if not starts:
        raise ValueError("No ACCESS_RAMP lanelets.")
    dist = {s: 0 for s in starts}
    q: deque[int] = deque(starts)
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_hop:
            continue
        la = ln.find_lanelet_by_id(u)
        for v in list(la.predecessor) + list(la.successor):
            nd = du + 1
            if v not in dist or dist[v] > nd:
                dist[v] = nd
                q.append(v)
    return set(dist.keys())


def crop_scenario(scenario, keep_ids: set[int]) -> None:
    ln = scenario.lanelet_network
    remove = [la for la in ln.lanelets if la.lanelet_id not in keep_ids]
    for la in remove:
        scenario.remove_lanelet(la, referenced_elements=True)
    scenario.remove_hanging_lanelet_members(remove)
    sid = scenario.scenario_id
    name = sid.map_name
    if name.endswith("MergeSide"):
        name = name[: -len("MergeSide")]
    name = name + "MergeSide"
    scenario.scenario_id = ScenarioID(
        cooperative=sid.cooperative,
        country_id=sid.country_id,
        map_name=name,
        map_id=sid.map_id,
        configuration_id=sid.configuration_id,
        obstacle_behavior=sid.obstacle_behavior,
        prediction_id=sid.prediction_id,
        scenario_version=sid.scenario_version,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop CommonRoad XML to merge vicinity.")
    ap.add_argument("input_xml", type=Path)
    ap.add_argument("output_xml", type=Path)
    ap.add_argument(
        "--mode",
        choices=("bbox", "hop"),
        default="bbox",
        help="bbox: ramp geometry + pad (default). hop: graph distance from ramp lanelets.",
    )
    ap.add_argument(
        "--pad-m",
        type=float,
        default=150.0,
        help="Metres to expand around ACCESS_RAMP centerline bbox (bbox mode only).",
    )
    ap.add_argument(
        "--expand-intersections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also keep lanelets in any intersection that touches the bbox selection (default: on).",
    )
    ap.add_argument("--max-hop", type=int, default=4, help="hop mode: max graph hops from ramp.")
    args = ap.parse_args()

    src = args.input_xml.resolve()
    if not src.is_file():
        raise SystemExit(f"Missing {src}")
    try:
        scenario, _ = CommonRoadFileReader(str(src)).open()
    except ValueError:
        scenario, _ = CommonRoadFileReader(str(patch_incoming_ids(src))).open()

    if args.mode == "bbox":
        keep = merge_side_lanelet_ids_bbox(scenario, args.pad_m)
        if args.expand_intersections:
            keep = merge_side_lanelet_ids_expand_intersections(scenario.lanelet_network, keep)
    else:
        keep = merge_side_lanelet_ids_hop(scenario, args.max_hop)

    crop_scenario(scenario, keep)
    out = args.output_xml.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    CommonRoadFileWriter(scenario, PlanningProblemSet()).write_to_file(
        str(out), OverwriteExistingFile.ALWAYS
    )
    print(f"mode={args.mode} kept={len(keep)} lanelets")
    print(sorted(keep))
    print(out)


if __name__ == "__main__":
    main()
