"""Interactive crop for CommonRoad XML (GUI)."""
from __future__ import annotations
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import ScenarioID
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer

def patch_incoming_ids(cr_xml):
    patched = cr_xml.with_suffix(cr_xml.suffix + ".patched.xml")
    if patched.is_file():
        return patched
    tree = ET.parse(cr_xml)
    root = tree.getroot()
    for i, inc in enumerate(root.iter("incoming")):
        inc.set("id", str(100000 + i))
    tree.write(patched, encoding="utf-8", xml_declaration=True)
    return patched

def load_scenario(xml_path):
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
        cooperative=sid.cooperative, country_id=sid.country_id, map_name=base + name_suffix,
        map_id=sid.map_id, configuration_id=sid.configuration_id, obstacle_behavior=sid.obstacle_behavior,
        prediction_id=sid.prediction_id, scenario_version=sid.scenario_version)

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
    lim = [float(min(xs) - margin), float(max(xs) + margin), float(min(ys) - margin), float(max(ys) + margin)]
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

def write_crop(src, out, bbox, name_suffix):
    try:
        sc, _ = CommonRoadFileReader(str(src)).open()
    except ValueError:
        sc, _ = CommonRoadFileReader(str(patch_incoming_ids(src))).open()
    keep = lanelet_ids_in_bbox(sc.lanelet_network, bbox)
    if not keep:
        raise SystemExit("No lanelets in box.")
    crop_scenario_to_lanelets(sc, keep, name_suffix)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    CommonRoadFileWriter(sc, PlanningProblemSet()).write_to_file(str(out), OverwriteExistingFile.ALWAYS)
    print("Wrote", out, len(keep), "lanelets")

def pick_clicks(sc, limits, figsize, dpi, vs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_lanelets(ax, sc, limits, vs)
    sid = str(getattr(sc, "scenario_id", "scenario")).replace("/", "_")
    ax.set_title(sid + " — click TWO corners", fontsize=11)
    fig.text(0.5, 0.02, "Two clicks = diagonal corners of crop rectangle.", ha="center", fontsize=9)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)
    pts = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(pts) < 2:
        raise SystemExit("Need 2 clicks")
    return axis_aligned_bbox_from_corners(pts[0][0], pts[0][1], pts[1][0], pts[1][1])

def pick_drag(sc, limits, figsize, dpi, vs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_lanelets(ax, sc, limits, vs)
    sid = str(getattr(sc, "scenario_id", "scenario")).replace("/", "_")
    ax.set_title(sid + " — drag box, Enter to confirm", fontsize=11)
    state = {"bbox": None}

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        state["bbox"] = axis_aligned_bbox_from_corners(
            eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=1, minspany=1,
                      spancoords="data", interactive=True)

    def on_key(event):
        if event.key in ("enter", "return") and state["bbox"] is not None:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(0.5, 0.02, "Drag with left mouse, then press Enter.", ha="center", fontsize=9)
    fig.tight_layout()
    plt.show(block=True)
    bbox = state["bbox"]
    if bbox is None:
        raise SystemExit("No selection")
    return bbox

def main():
    p = argparse.ArgumentParser(description="Interactive crop (clicks or drag).")
    p.add_argument("input_xml", type=Path)
    p.add_argument("output_xml", type=Path, nargs="?", default=None)
    p.add_argument("--mode", choices=("clicks", "drag"), default="clicks")
    p.add_argument("--figsize", type=float, nargs=2, default=(16.0, 6.0))
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--margin", type=float, default=12.0)
    p.add_argument("--view-fraction", type=float, default=None)
    p.add_argument("--vertical-scale", type=float, default=10.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--name-suffix", default="CropBox")
    args = p.parse_args()
    src = args.input_xml.resolve()
    if not src.is_file():
        raise SystemExit("not found")
    sc, _ = load_scenario(src)
    lim = plot_limits_for_network(sc.lanelet_network, args.margin, args.view_fraction)
    fs = tuple(args.figsize)
    bbox = pick_clicks(sc, lim, fs, args.dpi, args.vertical_scale) if args.mode == "clicks" else pick_drag(sc, lim, fs, args.dpi, args.vertical_scale)
    xmin, xmax, ymin, ymax = bbox
    out_s = str(args.output_xml) if args.output_xml else "OUT.cr.xml"
    print("uv run python tools/cr_map.py crop-box", '"' + str(src) + '"', out_s, "--box", xmin, ymin, xmax, ymax)
    if args.preview:
        fig, ax = plt.subplots(figsize=fs, dpi=args.dpi)
        draw_lanelets(ax, sc, plot_limits_for_bbox(bbox, args.margin), args.vertical_scale)
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "r-", lw=2)
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)
    if args.output_xml is not None:
        write_crop(src, args.output_xml, bbox, args.name_suffix)

if __name__ == "__main__":
    main()
