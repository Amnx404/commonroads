"""Single entry: SUMO + CommonRoad pipeline driven by a YAML config."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad.common.util import Interval

import sumo as _sumo_pkg
os.environ["SUMO_HOME"] = str(Path(_sumo_pkg.__file__).resolve().parent)

from commonroad_sumo import NonInteractiveSumoSimulation, SumoProject
from commonroad_sumo.cr2sumo.traffic_generator.random_trips_traffic_generator import (
    RandomTripsTrafficGenerator,
    RandomTripsTrafficGeneratorConfig,
)

from safety_metrics import (
    B_MAX,
    DRAC_SAFE,
    ScenarioSafety,
    TTC_TAU,
    aggregate,
    compute_metrics,
)


def default_config() -> dict[str, Any]:
    return {
        "scenario": {
            "kind": "folder",
            "cr_xml": None,
            "patch_incoming": True,
            "toy_map_generation": {
                "enabled": False,
                "output_cr_xml": None,
                "ramp_merge_start_x": 2800.0,
                "ramp_taper_length": 480.0,
                "ramp_outward_curve": 0.0,
            },
        },
        "simulation": {
            "n_runs": 1,
            "parallel_workers": None,
        },
        "sumo": {
            "mode": "bundled",
            "steps": 300,
        },
        "output": {
            "dir": "outputs/run",
            "group": None,
            "name": None,
            "subfolder": None,
            "clean": False,
            "timestamp_folder": True,
            "include_steps_in_folder_name": True,
        },
        "plot": {
            "figsize_in": [16.0, 6.0],
            "dpi": 150,
            "vertical_scale": 10.0,
            "view_fraction": 0.35,
            "gif_target_frames": 72,
            "gif_fps": 12.0,
            "gif_dpi": 85,
        },
        "artifacts": {
            "lanes_png": True,
            "three_times_png": True,
            "gif": True,
        },
        "metrics": {
            "enabled": True,
            "csv_name": "metrics.csv",
            "plot_name": "metrics.png",
            "runs_metrics_xml": "runs_metrics.xml",
            "summary_stats_file": "summary_stats.csv",
        },
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return deep_merge(default_config(), raw)


def _resolve_path(cfg_dir: Path, p: str | Path | None) -> Path | None:
    if p is None:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = (cfg_dir / pp).resolve()
    return pp


def _ensure_dir(p: Path, clean: bool) -> None:
    if clean and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _run_leaf_name(cfg: dict[str, Any], *, base: str, steps: int) -> str:
    """Unique per-invocation folder: {base}_{N}steps_{YYYYMMDD_HHMMSS} by default."""
    out = cfg["output"]
    parts: list[str] = [str(base).strip() or "run"]
    if out.get("include_steps_in_folder_name", True):
        parts.append(f"{int(steps)}steps")
    if out.get("timestamp_folder", True):
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def _output_run_root(cfg: dict[str, Any], cfg_path: Path) -> Path:
    """Resolved output root: output.dir, optionally nested under output.group."""
    out_root = _resolve_path(cfg_path.parent, cfg["output"]["dir"])
    assert out_root is not None
    grp = cfg["output"].get("group")
    if grp:
        out_root = out_root / str(grp).strip()
    return out_root


def patch_incoming_ids(cr_xml: Path) -> Path:
    patched = cr_xml.with_suffix(cr_xml.suffix + ".patched.xml")
    if patched.is_file():
        return patched
    tree = ET.parse(cr_xml)
    root = tree.getroot()
    base = 100000
    for i, inc in enumerate(root.iter("incoming")):
        inc.set("id", str(base + i))
    tree.write(patched, encoding="utf-8", xml_declaration=True)
    return patched


def _format_coord(v: float) -> str:
    txt = f"{float(v):.6f}"
    txt = txt.rstrip("0").rstrip(".")
    if "." not in txt:
        txt += ".0"
    return txt


def _lanelet_by_id(root: ET.Element, lanelet_id: int) -> ET.Element:
    for lanelet in root.findall("lanelet"):
        if lanelet.get("id") == str(lanelet_id):
            return lanelet
    raise ValueError(f"lanelet id={lanelet_id} not found in source map")


def _set_lanelet_bound_points(
    lanelet: ET.Element,
    bound_tag: str,
    points: list[tuple[float, float]],
) -> None:
    bound = lanelet.find(bound_tag)
    if bound is None:
        raise ValueError(f"lanelet id={lanelet.get('id')} missing {bound_tag}")
    for pt in list(bound.findall("point")):
        bound.remove(pt)
    for x, y in points:
        pt = ET.SubElement(bound, "point")
        x_el = ET.SubElement(pt, "x")
        y_el = ET.SubElement(pt, "y")
        x_el.text = _format_coord(x)
        y_el.text = _format_coord(y)


def _toy_ramp_profile(t: float) -> float:
    # Skewed bump for pre-merge shaping: peak earlier, gentle return near join.
    # 0 at both ends, normalized to 1 at t=1/3.
    return (27.0 / 4.0) * t * (1.0 - t) * (1.0 - t)


def _generate_toy_map_variant(
    *,
    src_xml: Path,
    out_xml: Path,
    ramp_merge_start_x: float,
    ramp_taper_length: float,
    ramp_outward_curve: float,
) -> None:
    aux_lane_begin_x = 1200.0
    through_end_x = 4000.0
    max_abs_curve_m = 100
    min_taper_len = 20.0
    if not (aux_lane_begin_x + 20.0 <= ramp_merge_start_x <= through_end_x - 40.0):
        raise ValueError(
            "scenario.toy_map_generation.ramp_merge_start_x must be in "
            f"[{aux_lane_begin_x + 20.0}, {through_end_x - 40.0}]"
        )
    if float(ramp_taper_length) < min_taper_len:
        raise ValueError(
            "scenario.toy_map_generation.ramp_taper_length must be >= "
            f"{min_taper_len}"
        )
    ramp_merge_end_x = ramp_merge_start_x + float(ramp_taper_length)
    if ramp_merge_end_x > through_end_x - 20.0:
        raise ValueError(
            "scenario.toy_map_generation.ramp_merge_start_x + ramp_taper_length must be <= "
            f"{through_end_x - 20.0}"
        )
    if abs(float(ramp_outward_curve)) > max_abs_curve_m:
        raise ValueError(
            "scenario.toy_map_generation.ramp_outward_curve is in meters and must satisfy "
            f"|value| <= {max_abs_curve_m}. Typical values are 0.0 to 1.5."
        )

    tree = ET.parse(src_xml)
    root = tree.getroot()

    ll2 = _lanelet_by_id(root, 2)
    ll5 = _lanelet_by_id(root, 5)
    ll7 = _lanelet_by_id(root, 7)
    ll8 = _lanelet_by_id(root, 8)
    ll9 = _lanelet_by_id(root, 9)
    ll10 = _lanelet_by_id(root, 10)
    ll11 = _lanelet_by_id(root, 11)
    ll3 = _lanelet_by_id(root, 3)
    ll6 = _lanelet_by_id(root, 6)

    # Lane 7 ONLY: ramp_outward_curve applied here and nowhere else.
    # The base profile curves the ramp toward the road; outward_curve bulges it further outward.
    x7 = [0.0, 240.0, 480.0, 720.0, 960.0, 1200.0]
    base_left7_y  = [-6.8, -6.6, -6.2, -5.6, -4.8, -2.0]
    base_right7_y = [-10.8, -10.6, -10.2, -9.6, -8.8, -6.0]
    left7  = [(x, y - float(ramp_outward_curve) * _toy_ramp_profile(t))
              for x, y, t in zip(x7, base_left7_y,  [i/5.0 for i in range(6)])]
    right7 = [(x, y - float(ramp_outward_curve) * _toy_ramp_profile(t))
              for x, y, t in zip(x7, base_right7_y, [i/5.0 for i in range(6)])]
    _set_lanelet_bound_points(ll7, "leftBound", left7)
    _set_lanelet_bound_points(ll7, "rightBound", right7)

    # Move lanelet split points — no curvature on any of these.
    _set_lanelet_bound_points(ll2, "leftBound", [(1200.0, 6.0), (ramp_merge_start_x, 6.0)])
    _set_lanelet_bound_points(ll2, "rightBound", [(1200.0, 2.0), (ramp_merge_start_x, 2.0)])
    _set_lanelet_bound_points(ll5, "leftBound", [(1200.0, 2.0), (ramp_merge_start_x, 2.0)])
    _set_lanelet_bound_points(ll5, "rightBound", [(1200.0, -2.0), (ramp_merge_start_x, -2.0)])
    # Lanelet 8: straight pre-merge connector, no curvature.
    _set_lanelet_bound_points(ll8, "leftBound",  [(1200.0, -2.0), (ramp_merge_start_x, -2.0)])
    _set_lanelet_bound_points(ll8, "rightBound", [(1200.0, -6.0), (ramp_merge_start_x, -6.0)])

    # Lanelet 9: clean monotonic taper into mainline, no curvature.
    taper_len = ramp_merge_end_x - ramp_merge_start_x
    taper_samples = [0.0, 0.5, 0.8, 1.0]
    left9 = []
    right9 = []
    for s in taper_samples:
        x = ramp_merge_start_x + taper_len * s
        left9.append((x, -2.0 + 4.0 * s))
        right9.append((x, -6.0 + 4.0 * s))
    _set_lanelet_bound_points(ll9, "leftBound", left9)
    _set_lanelet_bound_points(ll9, "rightBound", right9)

    # Through-lane split/join points follow the zipper window.
    _set_lanelet_bound_points(ll10, "leftBound", [(ramp_merge_start_x, 6.0), (ramp_merge_end_x, 6.0)])
    _set_lanelet_bound_points(ll10, "rightBound", [(ramp_merge_start_x, 2.0), (ramp_merge_end_x, 2.0)])
    _set_lanelet_bound_points(ll11, "leftBound", [(ramp_merge_start_x, 2.0), (ramp_merge_end_x, 2.0)])
    _set_lanelet_bound_points(ll11, "rightBound", [(ramp_merge_start_x, -2.0), (ramp_merge_end_x, -2.0)])
    _set_lanelet_bound_points(ll3, "leftBound", [(ramp_merge_end_x, 6.0), (4000.0, 6.0)])
    _set_lanelet_bound_points(ll3, "rightBound", [(ramp_merge_end_x, 2.0), (4000.0, 2.0)])
    _set_lanelet_bound_points(ll6, "leftBound", [(ramp_merge_end_x, 2.0), (4000.0, 2.0)])
    _set_lanelet_bound_points(ll6, "rightBound", [(ramp_merge_end_x, -2.0), (4000.0, -2.0)])

    ET.indent(tree, space="  ")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def _maybe_generate_toy_map(
    cfg: dict[str, Any],
    cfg_path: Path,
    xml_path: Path,
    *,
    combo_override: dict | None = None,
    out_xml_override: Path | None = None,
) -> Path:
    scen = cfg.get("scenario") or {}
    gen = scen.get("toy_map_generation")
    if not isinstance(gen, dict) or not bool(gen.get("enabled", False)):
        return xml_path

    # Merge per-combo overrides on top of gen dict
    effective = dict(gen)
    if combo_override:
        effective.update(combo_override)

    if out_xml_override is not None:
        out_xml = out_xml_override
    else:
        out_xml = _resolve_path(cfg_path.parent, effective.get("output_cr_xml"))
    if out_xml is None:
        name = xml_path.name
        if name.endswith(".cr.xml"):
            out_name = f"{name[:-len('.cr.xml')]}.generated.cr.xml"
        else:
            out_name = f"{xml_path.stem}.generated.xml"
        out_xml = xml_path.parent / out_name

    ramp_merge_start_x = float(effective.get("ramp_merge_start_x", 2800.0))
    if "ramp_taper_length" in effective:
        ramp_taper_length = float(effective.get("ramp_taper_length", 480.0))
    else:
        ramp_taper_length = float(effective.get("ramp_merge_end_x", 3280.0)) - ramp_merge_start_x

    _generate_toy_map_variant(
        src_xml=xml_path,
        out_xml=out_xml,
        ramp_merge_start_x=ramp_merge_start_x,
        ramp_taper_length=ramp_taper_length,
        ramp_outward_curve=float(effective.get("ramp_outward_curve", 0.0)),
    )
    print(f"Generated toy-map variant: {out_xml.resolve()}")
    return out_xml


def _scenario_base_names_for_sumo_cfg(cr_xml: Path) -> list[str]:
    name = cr_xml.name
    keys: list[str] = []
    if name.endswith(".cr.xml.patched.xml"):
        keys.append(name[: -len(".cr.xml.patched.xml")])
    elif name.endswith(".cr.xml"):
        keys.append(cr_xml.stem[: -3] if cr_xml.stem.endswith(".cr") else cr_xml.stem)
    keys.append(cr_xml.stem)
    out: list[str] = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def sumo_project_name_from_cfg(cfg: Path) -> str:
    if cfg.name.endswith(".sumo.cfg"):
        return cfg.name[: -len(".sumo.cfg")]
    return cfg.stem


def resolve_sumo_cfg_near_cr(cr_xml: Path) -> Path:
    folder = cr_xml.parent
    cfgs = sorted(folder.glob("*.sumo.cfg"))
    if not cfgs:
        raise FileNotFoundError(
            f"No *.sumo.cfg in {folder}. Use sumo.mode: random or add a SUMO bundle."
        )
    if len(cfgs) == 1:
        return cfgs[0]
    for key in _scenario_base_names_for_sumo_cfg(cr_xml):
        for c in cfgs:
            if sumo_project_name_from_cfg(c) == key:
                return c
    return cfgs[0]


def _crop_limits_to_fraction(limits: list[float], fraction: float) -> list[float]:
    if not (0 < fraction <= 1.0):
        return limits
    x0, x1, y0, y1 = limits
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    wx, wy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return [cx - wx / 2, cx + wx / 2, cy - wy / 2, cy + wy / 2]


def _plot_limits_lanelets(
    lanelet_network,
    margin: float,
    view_fraction: float | None,
) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for ll in lanelet_network.lanelets:
        c = np.asarray(ll.center_vertices)
        xs.extend(c[:, 0].tolist())
        ys.extend(c[:, 1].tolist())
    limits = [
        float(min(xs) - margin),
        float(max(xs) + margin),
        float(min(ys) - margin),
        float(max(ys) + margin),
    ]
    if view_fraction is not None:
        limits = _crop_limits_to_fraction(limits, view_fraction)
    return limits


def _draw_lane_network_only(ax, scenario, limits: list[float], vscale: float) -> None:
    renderer = MPRenderer(plot_limits=limits, ax=ax)
    params = MPDrawParams()
    scenario.lanelet_network.draw(renderer, draw_params=params)
    renderer.render()
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect(vscale, adjustable="box")


def _draw_scenario_at_step(ax, scenario, limits: list[float], t: int, vscale: float) -> None:
    dt = scenario.dt
    renderer = MPRenderer(plot_limits=limits, ax=ax)
    params = MPDrawParams()
    params.time_begin = t
    scenario.draw(renderer, draw_params=params)
    renderer.render()
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect(vscale, adjustable="box")
    ax.set_title(f"step {t} ({t * dt:.1f} s)")


def _gif_frame_indices(n_steps: int, target_frames: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    step = max(1, (n_steps + target_frames - 1) // target_frames)
    frames = list(range(0, n_steps, step))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    return frames


def _plot_metrics_single(results: list[ScenarioSafety], out_path: Path, figsize: tuple[float, float]) -> Path:
    labels = [f"run {i}" for i in range(len(results))]
    min_ttcs = [r.mean_min_ttc for r in results]
    pct_uns = [r.pct_unsafe for r in results]
    max_dracs = [r.max_drac for r in results]
    max_btns = [r.max_btn for r in results]
    tits = [r.tit for r in results]
    min_gaps = [r.min_gap_m for r in results]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    x = np.arange(len(results))

    ax = axes[0, 0]
    ax.plot(x, min_ttcs, "o-", color="steelblue", lw=2, ms=5)
    ax.axhline(TTC_TAU, color="crimson", ls="--", lw=1)
    ax.set_title("Mean min-TTC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("s")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, pct_uns, color="tomato", alpha=0.85, edgecolor="darkred")
    ax.set_title(f"% steps TTC < {TTC_TAU}s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("%")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 2]
    ax.fill_between(x, tits, alpha=0.35, color="purple")
    ax.plot(x, tits, "D-", color="purple", lw=2, ms=4)
    ax.set_title("TIT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("s²")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, max_dracs, "s-", color="darkorange", lw=2, ms=5)
    ax.axhline(DRAC_SAFE, color="crimson", ls="--", lw=1)
    ax.set_title("Max DRAC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("m/s²")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, max_btns, "^-", color="teal", lw=2, ms=5)
    ax.axhline(1.0, color="crimson", ls="--", lw=1)
    ax.set_title(f"Max BTN (Bmax={B_MAX})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("-")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(x, min_gaps, "h-", color="forestgreen", lw=2, ms=6)
    ax.axhline(0.0, color="crimson", ls=":", lw=1.5)
    ax.set_title("Min bumper gap")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()



def _save_csv_single(
    results: list[ScenarioSafety],
    out_path: Path,
    *,
    seeds: list[int] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [-1] * len(results)
    elif len(seeds) != len(results):
        raise ValueError("seeds length must match results")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "random_seed",
                "mean_min_ttc_s",
                "pct_unsafe",
                "max_drac_ms2",
                "max_btn",
                "tit_s2",
                "min_gap_m",
            ]
        )
        for i, r in enumerate(results):
            w.writerow(
                [
                    str(i),
                    str(seeds[i]),
                    f"{r.mean_min_ttc:.4f}",
                    f"{r.pct_unsafe:.2f}",
                    f"{r.max_drac:.4f}",
                    f"{r.max_btn:.4f}",
                    f"{r.tit:.4f}",
                    f"{r.min_gap_m:.4f}",
                ]
            )
    return out_path.resolve()


def _run_sumo(
    scenario,
    *,
    mode: str,
    cr_xml_path: Path | None,
    steps: int,
    sumo: dict[str, Any] | None = None,
):
    sumo = sumo or {}
    mode = mode.lower()
    if mode == "random":
        sim_duration_s = steps * scenario.dt
        seed = int(sumo.get("random_seed", 1234))
        max_veh = int(sumo.get("max_veh_per_km", 25))
        density_scale = float(sumo.get("random_density_scale", 1.0))
        traffic_cfg = RandomTripsTrafficGeneratorConfig(
            random_seed=seed,
            max_veh_per_km=max(1, int(max_veh * density_scale)),
            departure_interval_vehicles=Interval(0, sim_duration_s),
        )
        traffic_gen = RandomTripsTrafficGenerator(traffic_cfg)
        sim = NonInteractiveSumoSimulation.from_scenario(scenario, traffic_gen)
    elif mode == "bundled":
        if cr_xml_path is None:
            raise ValueError("bundled mode requires scenario.cr_xml on disk")
        sumo_cfg = resolve_sumo_cfg_near_cr(cr_xml_path)
        project = SumoProject(sumo_project_name_from_cfg(sumo_cfg), sumo_cfg.parent)
        sim = NonInteractiveSumoSimulation(scenario, project)
    else:
        raise ValueError(f"Unknown sumo.mode: {mode!r}")
    result = sim.run(simulation_steps=steps)
    sim_scenario = result.scenario if hasattr(result, "scenario") else result.get_scenario()
    return sim_scenario



def _repeat_worker_cap(n_runs: int, sim_cfg: dict[str, Any]) -> int:
    if n_runs <= 1:
        return 1
    raw = sim_cfg.get("parallel_workers")
    if raw is not None:
        return max(1, min(int(raw), n_runs - 1))
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu, n_runs - 1))


def _write_runs_metrics_xml(
    path: Path,
    *,
    scenario_xml: Path,
    steps: int,
    mode: str,
    seeds: list[int],
    safeties: list[ScenarioSafety],
) -> Path:
    root = ET.Element("SafetyRuns")
    root.set("version", "1")
    src = ET.SubElement(root, "SourceScenario")
    src.set("path", str(scenario_xml))
    src.set("sumo_mode", mode)
    src.set("steps", str(steps))
    src.set("n_runs", str(len(safeties)))
    for i, (seed, s) in enumerate(zip(seeds, safeties)):
        run_el = ET.SubElement(root, "Run")
        run_el.set("index", str(i))
        run_el.set("random_seed", str(seed))
        for tag, val in (
            ("mean_min_ttc_s", s.mean_min_ttc),
            ("pct_unsafe", s.pct_unsafe),
            ("max_drac_ms2", s.max_drac),
            ("max_btn", s.max_btn),
            ("tit_s2", s.tit),
            ("min_gap_m", s.min_gap_m),
        ):
            el = ET.SubElement(run_el, tag)
            el.text = f"{val:.6g}"
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path.resolve()


def _write_summary_stats_csv(path: Path, safeties: list[ScenarioSafety]) -> Path:
    if not safeties:
        return path
    keys = [
        ("mean_min_ttc_s", "mean_min_ttc"),
        ("pct_unsafe", "pct_unsafe"),
        ("max_drac_ms2", "max_drac"),
        ("max_btn", "max_btn"),
        ("tit_s2", "tit"),
        ("min_gap_m", "min_gap_m"),
    ]
    rows: list[dict[str, Any]] = []
    for csv_name, attr in keys:
        vals = np.array([getattr(s, attr) for s in safeties], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "metric": csv_name,
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "p95": float(np.percentile(finite, 95)),
                "n": int(finite.size),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "std", "min", "max", "p95", "n"],
        )
        w.writeheader()
        w.writerows(rows)
    return path.resolve()

def _artifacts(
    sim_scenario,
    out_dir: Path,
    title_prefix: str,
    steps: int,
    plot_cfg: dict[str, Any],
    art: dict[str, Any],
    limits: list[float],
    *,
    fast: bool = False,
) -> None:
    if fast:
        return
    vscale = float(plot_cfg["vertical_scale"])
    figsize = tuple(float(x) for x in plot_cfg["figsize_in"])
    dpi = int(plot_cfg["dpi"])
    gif_frames = int(plot_cfg["gif_target_frames"])
    gif_fps = float(plot_cfg["gif_fps"])
    gif_dpi = int(plot_cfg["gif_dpi"])

    if art["lanes_png"]:
        fig_l, ax_l = plt.subplots(figsize=figsize)
        _draw_lane_network_only(ax_l, sim_scenario, limits, vscale)
        ax_l.set_title(f"{title_prefix} — lanes")
        plt.tight_layout()
        fig_l.savefig(out_dir / "lanes.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_l)

    if art["three_times_png"]:
        time_steps = sorted({0, max(steps // 2, 0), max(steps - 1, 0)})
        fig, axes = plt.subplots(len(time_steps), 1, figsize=figsize, sharex=True)
        if len(time_steps) == 1:
            axes = [axes]
        for ax, t in zip(axes, time_steps):
            _draw_scenario_at_step(ax, sim_scenario, limits, t, vscale)
        fig.suptitle(f"{title_prefix} — three times", fontsize=11, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / "three_times.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if art["gif"]:
        anim_frames = _gif_frame_indices(steps, gif_frames)
        fig_g, ax_g = plt.subplots(figsize=figsize)

        def _update(ti: int) -> None:
            ax_g.clear()
            _draw_scenario_at_step(ax_g, sim_scenario, limits, ti, vscale)

        anim = mpl_animation.FuncAnimation(
            fig_g,
            _update,
            frames=anim_frames,
            interval=max(1, int(1000.0 / gif_fps)),
            blit=False,
        )
        gif_path = out_dir / "simulation.gif"
        anim.save(str(gif_path), writer=mpl_animation.PillowWriter(fps=gif_fps), dpi=gif_dpi)
        plt.close(fig_g)


_TOY_MAP_SWEEP_KEYS = ("ramp_merge_start_x", "ramp_taper_length", "ramp_outward_curve")


def _parse_multi_val(raw) -> list[float]:
    """Accept a scalar, a list, or a comma-separated string and return a list of floats."""
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, list):
        return [float(v) for v in raw]
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    return [float(s) for s in parts]


def _toy_map_param_combos(gen: dict) -> list[dict]:
    """Return list of single-value dicts covering the cartesian product of all sweep keys."""
    axes: list[tuple[str, list[float]]] = []
    for key in _TOY_MAP_SWEEP_KEYS:
        if key in gen:
            axes.append((key, _parse_multi_val(gen[key])))
    if not axes:
        return [{}]
    keys, value_lists = zip(*axes)
    combos = []
    for vals in itertools.product(*value_lists):
        combos.append(dict(zip(keys, vals)))
    return combos


def _combo_tag(combo: dict) -> str:
    """Short filesystem-safe label for one parameter combo."""
    abbrev = {"ramp_merge_start_x": "start", "ramp_taper_length": "taper", "ramp_outward_curve": "curve"}
    parts = []
    for key in _TOY_MAP_SWEEP_KEYS:
        if key in combo:
            label = abbrev.get(key, key)
            val = combo[key]
            txt = f"{val:.4g}".replace(".", "p")
            parts.append(f"{label}{txt}")
    return "_".join(parts)


def _folder_case(cfg: dict[str, Any], cfg_path: Path, *, combo_override: dict | None = None, name_suffix: str = "") -> None:
    scen = cfg["scenario"]
    xml_path = _resolve_path(cfg_path.parent, scen["cr_xml"])
    if xml_path is None or not xml_path.is_file():
        raise FileNotFoundError(f"scenario.cr_xml must exist: {xml_path}")

    # Per-combo generated XML sits beside source with a combo-specific name
    gen_out_override: Path | None = None
    if combo_override:
        src_xml = xml_path
        tag = _combo_tag(combo_override)
        if src_xml.name.endswith(".cr.xml"):
            gen_out_override = src_xml.parent / f"{src_xml.name[:-len('.cr.xml')]}.{tag}.cr.xml"
        else:
            gen_out_override = src_xml.parent / f"{src_xml.stem}.{tag}.xml"
    xml_path = _maybe_generate_toy_map(cfg, cfg_path, xml_path, combo_override=combo_override, out_xml_override=gen_out_override)

    if scen.get("patch_incoming", True):
        xml_path = patch_incoming_ids(xml_path)

    scenario_probe, _ = CommonRoadFileReader(str(xml_path)).open()
    scenario_id = str(getattr(scenario_probe, "scenario_id", "scenario")).replace("/", "_")

    sim_cfg = cfg.get("simulation") or {}
    n_runs = int(sim_cfg.get("n_runs", 1))
    if n_runs < 1:
        raise ValueError("simulation.n_runs must be >= 1")
    if n_runs > 1 and not cfg["metrics"].get("enabled", True):
        raise ValueError("simulation.n_runs > 1 requires metrics.enabled (for combined outputs)")

    sumo_cfg = cfg["sumo"]
    steps = int(sumo_cfg["steps"])
    mode = str(sumo_cfg["mode"])
    base_seed = int(sumo_cfg.get("random_seed", 1234))

    def _run_one_sim(seed: int):
        sc, _ = CommonRoadFileReader(str(xml_path)).open()
        sm = dict(sumo_cfg)
        sm["random_seed"] = int(seed)
        return _run_sumo(sc, mode=mode, cr_xml_path=xml_path, steps=steps, sumo=sm)

    sim_scenario = _run_one_sim(base_seed)

    out_root = _output_run_root(cfg, cfg_path)
    out_cfg = cfg["output"]
    name_base = out_cfg.get("name") or out_cfg.get("subfolder") or scenario_id
    combo_part = ("_" + name_suffix) if name_suffix else ""
    leaf = _run_leaf_name(cfg, base=str(name_base) + combo_part, steps=steps)
    out_dir = out_root / leaf
    _ensure_dir(out_dir, bool(out_cfg.get("clean")))
    print(f"Output directory: {out_dir.resolve()}")

    plot_cfg = cfg["plot"]
    vf = plot_cfg.get("view_fraction")
    vf_f = float(vf) if vf is not None else None
    limits = _plot_limits_lanelets(
        sim_scenario.lanelet_network,
        margin=12.0,
        view_fraction=vf_f,
    )

    _artifacts(
        sim_scenario,
        out_dir,
        title_prefix=scenario_id,
        steps=steps,
        plot_cfg=plot_cfg,
        art=cfg["artifacts"],
        limits=limits,
        fast=False,
    )

    safeties: list[ScenarioSafety] = []
    seeds: list[int] = []
    seeds.append(base_seed)

    if cfg["metrics"]["enabled"]:
        frames = compute_metrics(sim_scenario, car_length=4.7)
        safeties.append(aggregate(0.0, frames, sim_scenario.dt, merge_length=0.0))

    if n_runs > 1:
        from repeat_worker import run_repeat_job

        payloads: list[dict[str, Any]] = []
        for i in range(1, n_runs):
            payloads.append(
                {
                    "xml_path": str(xml_path.resolve()),
                    "patch_incoming": bool(scen.get("patch_incoming", True)),
                    "mode": mode,
                    "steps": steps,
                    "sumo": dict(sumo_cfg),
                    "random_seed": base_seed + i,
                }
            )
        workers = _repeat_worker_cap(n_runs, sim_cfg)
        if workers <= 1:
            extra = [run_repeat_job(pl) for pl in tqdm(payloads, desc="Extra runs")]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                extra = list(
                    tqdm(
                        ex.map(run_repeat_job, payloads),
                        total=len(payloads),
                        desc="Extra runs",
                    )
                )
        safeties.extend(extra)
        seeds.extend(base_seed + i for i in range(1, n_runs))

    if cfg["metrics"]["enabled"]:
        mcfg = cfg["metrics"]
        _save_csv_single(safeties, out_dir / mcfg["csv_name"], seeds=seeds)
        _plot_metrics_single(
            safeties,
            out_dir / mcfg["plot_name"],
            figsize=tuple(float(x) for x in plot_cfg["figsize_in"]),
        )
        _write_runs_metrics_xml(
            out_dir / mcfg["runs_metrics_xml"],
            scenario_xml=xml_path,
            steps=steps,
            mode=mode,
            seeds=seeds,
            safeties=safeties,
        )
        _write_summary_stats_csv(out_dir / mcfg["summary_stats_file"], safeties)





def _folder_case_sweep(cfg: dict[str, Any], cfg_path: Path) -> None:
    """Run _folder_case once per combo of multi-valued toy_map_generation params."""
    gen = (cfg.get("scenario") or {}).get("toy_map_generation") or {}
    combos = _toy_map_param_combos(gen) if gen.get("enabled") else [{}]
    if len(combos) <= 1:
        _folder_case(cfg, cfg_path, combo_override=combos[0] if combos else None)
        return
    print(f"Sweep: {len(combos)} combinations")
    for i, combo in enumerate(combos):
        tag = _combo_tag(combo)
        print(f"  [{i+1}/{len(combos)}] {tag}")
        _folder_case(cfg, cfg_path, combo_override=combo, name_suffix=tag)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run SUMO + CommonRoad from YAML config.")
    p.add_argument("config", type=Path, help="Path to config YAML")
    args = p.parse_args(argv)
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)

    kind = str(cfg["scenario"]["kind"]).lower()
    if kind == "folder":
        _folder_case_sweep(cfg, cfg_path)
    else:
        raise ValueError(
            "Only scenario.kind: folder is supported (synthetic merge sweep was removed)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



