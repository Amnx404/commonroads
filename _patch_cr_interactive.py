# One-off patcher — run: python _patch_cr_interactive.py
from pathlib import Path

p = Path(__file__).resolve().parent / "tools" / "cr_map.py"
t = p.read_text(encoding="utf-8")

header_old = '''"""CommonRoad: visualise (PNG) and crop-box (axis-aligned map box)."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np'''

header_new = '''"""CommonRoad: visualise (PNG), crop-box, and crop-interactive."""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if "crop-interactive" in sys.argv:
    import matplotlib as _mpl
    _mpl.use(os.environ.get("MPLBACKEND", "TkAgg"))

import matplotlib.pyplot as plt
import numpy as np'''

if header_old not in t:
    raise SystemExit("header block not found")
t = t.replace(header_old, header_new, 1)

insert_before_build = '''

def _write_crop_from_bbox(src: Path, out: Path, bbox, name_suffix: str) -> int:
    try:
        sc, _ = CommonRoadFileReader(str(src)).open()
    except ValueError:
        sc, _ = CommonRoadFileReader(str(patch_incoming_ids(src))).open()
    keep = lanelet_ids_in_bbox(sc.lanelet_network, bbox)
    if not keep:
        raise SystemExit("no lanelets in selected box")
    crop_scenario_to_lanelets(sc, keep, name_suffix)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    CommonRoadFileWriter(sc, PlanningProblemSet()).write_to_file(str(out), OverwriteExistingFile.ALWAYS)
    print("bbox xmin,xmax,ymin,ymax =", bbox)
    print("kept", len(keep), "lanelets ->", out)
    return 0


def cmd_crop_interactive(args):
    src = args.input_xml.resolve()
    if not src.is_file():
        raise SystemExit("missing " + str(src))
    sc, _ = load_scenario(src)
    sid = str(getattr(sc, "scenario_id", "scenario")).replace("/", "_")
    lim = plot_limits_for_network(sc.lanelet_network, args.margin, args.view_fraction)
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    draw_lanelets(ax, sc, lim, args.vertical_scale)
    ax.set_title(sid + " — click TWO map corners of the crop box", fontsize=11)
    fig.suptitle(
        "Mode: "
        + args.mode
        + (" — drag a rectangle, then press Enter" if args.mode == "drag" else " — two clicks (any diagonal)"),
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout()
    plt.show(block=True)

    bbox = None
    if args.mode == "clicks":
        fig2, ax2 = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
        draw_lanelets(ax2, sc, lim, args.vertical_scale)
        ax2.set_title("Click exactly 2 corners (close window after 2 clicks if it stays open)", fontsize=10)
        fig2.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)
        pts = plt.ginput(2, timeout=0, show_clicks=True)
        plt.close(fig2)
        if len(pts) < 2:
            raise SystemExit("need two clicks")
        bbox = axis_aligned_bbox_from_corners(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
    else:
        from matplotlib.widgets import RectangleSelector

        state = {"bbox": None}

        def onselect(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            bbox = axis_aligned_bbox_from_corners(
                eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
            )
            state["bbox"] = bbox

        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="data",
            interactive=True,
        )
        plt.show(block=True)
        ext = getattr(selector, "extents", None)
        if ext is not None:
            x1, x2, y1, y2 = ext
            bbox = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        elif state["bbox"] is not None:
            bbox = state["bbox"]
        if bbox is None:
            raise SystemExit("no rectangle selected")

    if args.preview:
        fig3, ax3 = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
        draw_lanelets(ax3, sc, plot_limits_for_bbox(bbox, args.margin), args.vertical_scale)
        xmin, xmax, ymin, ymax = bbox
        ax3.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "r-", lw=2)
        ax3.set_title("Crop preview — close to continue", fontsize=11)
        fig3.tight_layout()
        plt.show(block=True)

    print(
        "Non-interactive equivalent:\\n  uv run python tools/cr_map.py crop-box",
        str(src),
        str(args.output_xml or "out.cr.xml"),
        "--box",
        bbox[0],
        bbox[2],
        bbox[1],
        bbox[3],
    )
    if args.output_xml is not None:
        return _write_crop_from_bbox(src, args.output_xml, bbox, args.name_suffix)
    return 0


'''

marker = "def build_parser():"
if insert_before_build.strip() in t:
    raise SystemExit("already patched")
if marker not in t:
    raise SystemExit("build_parser marker missing")
t = t.replace(marker, insert_before_build + marker, 1)

# extend build_parser
old_sub = '    c.set_defaults(_fn=cmd_crop_box)\n    return r'
new_sub = '''    c.set_defaults(_fn=cmd_crop_box)
    i = s.add_parser(
        "crop-interactive",
        help="Pick crop box on map (two clicks or drag); optional write XML.",
    )
    i.add_argument("input_xml", type=Path)
    i.add_argument(
        "output_xml",
        type=Path,
        nargs="?",
        default=None,
        help="If set, write cropped scenario here after selection.",
    )
    i.add_argument(
        "--mode",
        choices=("clicks", "drag"),
        default="clicks",
        help="clicks: two mouse clicks for corners. drag: drag rectangle, Enter when done.",
    )
    i.add_argument("--figsize", type=float, nargs=2, default=(16.0, 6.0))
    i.add_argument("--dpi", type=int, default=120)
    i.add_argument("--margin", type=float, default=12.0)
    i.add_argument("--view-fraction", type=float, default=None)
    i.add_argument("--vertical-scale", type=float, default=10.0)
    i.add_argument(
        "--preview",
        action="store_true",
        help="Show bbox outline on lane map before saving.",
    )
    i.add_argument("--name-suffix", default="CropBox")
    i.set_defaults(_fn=cmd_crop_interactive)
    return r'''

if old_sub not in t:
    raise SystemExit("subparser tail not found")
t = t.replace(old_sub, new_sub, 1)

p.write_text(t, encoding="utf-8")
print("patched", p)
