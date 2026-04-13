"""Run: python cr_map.py visualise ... | python cr_map.py crop-box ..."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import ScenarioID
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer


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


def load_scenario(xml_path: Path):
    try:
        return CommonRoadFileReader(str(xml_path)).open()
    except ValueError:
        return CommonRoadFileReader(str(patch_incoming_ids(xml_path))).open()


def axis_aligned_bbox_from_corners(x1, y1, x2, y2):
    xmin, xmax = (float(x1), float(x2)) if float(x1) <= float(x2) else (float(x2), float(x1))
    ymin, ymax = (float(y1), float(y2)) if float(y1) <= float(y2) else (float(y2), float(y1))
    return xmin, xmax, ymin, ymax


def lanelet_ids_in_bbox(lanelet_network, bbox):
    xmin, xmax, ymin, ymax = bbox
    keep = set()
    for la in lanelet_network.lanelets:
        c = np.asarray(la.center_vertices, dtype=float)
        if c.size == 0:
            continue
        if np.any((c[:, 0] >= xmin) & (c[:, 0] <= xmax) & (c[:, 1] >= ymin) & (c[:, 1] <= ymax)):
            keep.add(la.lanelet_id)
    return keep


def crop_scenario_to_lanelets(scenario, keep_ids, name_suffix):
    remove = [la for la in scenario.lanelet_network.lanelets if la.lanelet_id not in keep_ids]
    for la in remove:
        scenario.remove_lanelet(la, referenced_elements=True)
    scenario.remove_hanging_lanelet_members(remove)
    sid = scenario.scenario_id
    base = sid.map_name[:-len(name_suffix)] if sid.map_name.endswith(name_suffix) else sid.map_name
    scenario.scenario_id = ScenarioID(
        cooperative=sid.cooperative,
        country_id=sid.country_id,
        map_name=base + name_suffix,
        map_id=sid.map_id,
        configuration_id=sid.configuration_id,
        obstacle_behavior=sid.obstacle_behavior,
        prediction_id=sid.prediction_id,
        scenario_version=sid.scenario_version,
    )


def _frac(lim, fraction):
    if not (0 < fraction <= 1.0):
        return lim
    x0, x1, y0, y1 = lim
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    wx, wy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return [cx - wx / 2, cx + wx / 2, cy - wy / 2, cy + wy / 2]


def plot_limits_for_network(ln, margin, view_fraction):
    xs, ys = [], []
    for ll in ln.lanelets:
        c = np.asarray(ll.center_vertices)
        xs.extend(c[:, 0].tolist())
        ys.extend(c[:, 1].tolist())
    if not xs:
        return [-1.0, 1.0, -1.0, 1.0]
    lim = [float(min(xs) - margin), float(max(xs) + margin), float(min(ys) - margin), float(max(ys) - margin)]
    return _frac(lim, view_fraction) if view_fraction is not None else lim


def plot_limits_for_bbox(bbox, margin):
    a, b, c, d = bbox
    return [a - margin, b + margin, c - margin, d + margin]


def draw_lanelets(ax, scenario, limits, vs):
    r = MPRenderer(plot_limits=limits, ax=ax)
    p = MPDrawParams()
    scenario.lanelet_network.draw(r, draw_params=p)
    r.render()
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect(vs, adjustable="box")
    ax.set_axis_off()


def draw_full(ax, scenario, limits, vs, t):
    r = MPRenderer(plot_limits=limits, ax=ax)
    p = MPDrawParams()
    p.time_begin = t
    scenario.draw(r, draw_params=p)
    r.render()
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect(vs, adjustable="box")
    ax.set_axis_off()


def cmd_visualise(args):
    p = args.cr_xml.resolve()
    if not p.is_file():
        raise SystemExit("not found: " + str(p))
    sc, _ = load_scenario(p)
    sid = str(getattr(sc, "scenario_id", "scenario")).replace("/", "_")
    if args.limits_box:
        lim = plot_limits_for_bbox(axis_aligned_bbox_from_corners(*args.limits_box), args.margin)
    else:
        lim = plot_limits_for_network(sc.lanelet_network, args.margin, args.view_fraction)
    out = args.output or (p.parent / (p.stem.replace(".cr", "") + "_map.png"))
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    if args.lanes_only:
        draw_lanelets(ax, sc, lim, args.vertical_scale)
        ax.set_title(sid + " — lanelets", fontsize=11)
    else:
        draw_full(ax, sc, lim, args.vertical_scale, args.time_step)
        ax.set_title(sid + " — t=" + str(args.time_step), fontsize=11)
    if args.labels:
        for la in sc.lanelet_network.lanelets:
            c = np.asarray(la.center_vertices)
            m = c[len(c) // 2]
            ax.text(float(m[0]), float(m[1]), str(la.lanelet_id), fontsize=6, ha="center", va="center", clip_on=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(out)
    plt.close(fig) if not args.show else plt.show()
    return 0


def parse_bbox(args):
    if args.box:
        return axis_aligned_bbox_from_corners(*args.box)
    if args.top_right and args.bottom_left:
        return axis_aligned_bbox_from_corners(*args.top_right, *args.bottom_left)
    if args.top_right and args.bottom_right:
        trx, try_ = args.top_right
        brx, bry = args.bottom_right
        if abs(trx - brx) > 1e-3:
            raise SystemExit("top-right and bottom-right need same x; use --box for general rect")
        if args.left_x is None:
            raise SystemExit("need --left-x with top-right + bottom-right")
        xmin, xmax = sorted((float(args.left_x), float(trx)))
        ymin, ymax = sorted((float(bry), float(try_)))
        return xmin, xmax, ymin, ymax
    raise SystemExit("need --box or top-right+bottom-left or top-right+bottom-right+left-x")


def cmd_crop(args):
    src = args.input_xml.resolve()
    if not src.is_file():
        raise SystemExit("missing " + str(src))
    bbox = parse_bbox(args)
    try:
        sc, _ = CommonRoadFileReader(str(src)).open()
    except ValueError:
        sc, _ = CommonRoadFileReader(str(patch_incoming_ids(src))).open()
    keep = lanelet_ids_in_bbox(sc.lanelet_network, bbox)
    if not keep:
        raise SystemExit("empty crop")
    crop_scenario_to_lanelets(sc, keep, args.name_suffix)
    out = args.output_xml.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    CommonRoadFileWriter(sc, PlanningProblemSet()).write_to_file(str(out), OverwriteExistingFile.ALWAYS)
    print(bbox, len(keep), out)
    return 0


def main(argv=None):
    r = argparse.ArgumentParser()
    s = r.add_subparsers(dest="cmd", required=True)
    v = s.add_parser("visualise")
    v.add_argument("cr_xml", type=Path)
    v.add_argument("-o", "--output", type=Path, default=None)
    v.add_argument("--lanes-only", action="store_true")
    v.add_argument("--time-step", type=int, default=0)
    v.add_argument("--figsize", type=float, nargs=2, default=(16.0, 6.0))
    v.add_argument("--dpi", type=int, default=150)
    v.add_argument("--margin", type=float, default=12.0)
    v.add_argument("--view-fraction", type=float, default=None)
    v.add_argument("--limits-box", type=float, nargs=4, default=None)
    v.add_argument("--vertical-scale", type=float, default=10.0)
    v.add_argument("--show", action="store_true")
    v.add_argument("--labels", action="store_true")
    c = s.add_parser("crop-box")
    c.add_argument("input_xml", type=Path)
    c.add_argument("output_xml", type=Path)
    c.add_argument("--box", type=float, nargs=4, default=None)
    c.add_argument("--top-right", type=float, nargs=2, default=None)
    c.add_argument("--bottom-left", type=float, nargs=2, default=None)
    c.add_argument("--bottom-right", type=float, nargs=2, default=None)
    c.add_argument("--left-x", type=float, default=None)
    c.add_argument("--name-suffix", default="CropBox")
    a = r.parse_args(argv)
    return cmd_visualise(a) if a.cmd == "visualise" else cmd_crop(a)


if __name__ == "__main__":
    raise SystemExit(main())
