"""Single entry: SUMO + CommonRoad pipeline driven by a YAML config."""

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_sumo import NonInteractiveSumoSimulation, SumoProject

from scaled_random_traffic import ScaledRandomTrafficGenerator
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
        },
        "synthetic": {
            "merge_lengths": [80.0, 120.0, 160.0],
            "curvatures": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95],
            "n_main_vehicles": 0,
            "n_ramp_vehicles": 0,
            "main_speed": 28.0,
            "ramp_speed": 22.0,
        },
        "sumo": {
            "mode": "bundled",
            "steps": 300,
            "random_density_scale": 1.0,
            "random_seed": 1234,
            "random_map_matching_delta": 10,
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
            "aggregate_synthetic": True,
            "aggregate_plot_name": "accident_metrics_vs_curvature.png",
            "aggregate_csv_name": "curvature_sweep_metrics.csv",
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
    labels = [str(i) for i in range(len(results))]
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


def _plot_metrics_synthetic_multi(
    results: list[ScenarioSafety], out_path: Path, figsize: tuple[float, float]
) -> Path:
    by_l: dict[float, list[ScenarioSafety]] = defaultdict(list)
    for r in results:
        by_l[r.merge_length].append(r)
    lengths = sorted(by_l.keys())
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("SUMO safety vs curvature (one curve per L)", fontsize=11, y=1.02)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(lengths), 1)))

    def _series(ax, ykey: str) -> None:
        for i, L in enumerate(lengths):
            series = sorted(by_l[L], key=lambda x: x.curvature)
            kappas = [s.curvature for s in series]
            ys = [getattr(s, ykey) for s in series]
            ax.plot(
                kappas,
                ys,
                "o-",
                color=cmap[i % 10],
                lw=2,
                ms=4,
                label=f"L={L:.0f}m",
            )
        ax.legend(fontsize=7, loc="best")

    ax = axes[0, 0]
    _series(ax, "mean_min_ttc")
    ax.axhline(TTC_TAU, color="crimson", ls="--", lw=1)
    ax.set_xlabel("kappa")
    ax.set_ylabel("mean min-TTC")
    ax.set_title("Mean min-TTC")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _series(ax, "pct_unsafe")
    ax.set_xlabel("kappa")
    ax.set_ylabel("%")
    ax.set_title(f"% TTC < {TTC_TAU}s")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    _series(ax, "tit")
    ax.set_xlabel("kappa")
    ax.set_ylabel("TIT")
    ax.set_title("TIT")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _series(ax, "max_drac")
    ax.axhline(DRAC_SAFE, color="crimson", ls="--", lw=1)
    ax.set_xlabel("kappa")
    ax.set_ylabel("max DRAC")
    ax.set_title("Max DRAC")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    _series(ax, "max_btn")
    ax.axhline(1.0, color="crimson", ls="--", lw=1)
    ax.set_xlabel("kappa")
    ax.set_ylabel("max BTN")
    ax.set_title(f"Max BTN Bmax={B_MAX}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    _series(ax, "min_gap_m")
    ax.axhline(0.0, color="crimson", ls=":", lw=1.5)
    ax.set_xlabel("kappa")
    ax.set_ylabel("min gap m")
    ax.set_title("Min gap")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def _save_csv_single(results: list[ScenarioSafety], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
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
                    f"{r.mean_min_ttc:.4f}",
                    f"{r.pct_unsafe:.2f}",
                    f"{r.max_drac:.4f}",
                    f"{r.max_btn:.4f}",
                    f"{r.tit:.4f}",
                    f"{r.min_gap_m:.4f}",
                ]
            )
    return out_path.resolve()


def _save_csv_synthetic(results: list[ScenarioSafety], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: (r.merge_length, r.curvature))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "merge_length_m",
                "curvature",
                "mean_min_ttc_s",
                "pct_unsafe",
                "max_drac_ms2",
                "max_btn",
                "tit_s2",
                "min_gap_m",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    f"{r.merge_length:.3f}",
                    f"{r.curvature:.3f}",
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
        gen = ScaledRandomTrafficGenerator(
            map_matching_delta=int(sumo.get("random_map_matching_delta", 10)),
            seed=int(sumo.get("random_seed", 1234)),
            density_scale=float(sumo.get("random_density_scale", 1.0)),
        )
        sim = NonInteractiveSumoSimulation.from_scenario(scenario, gen)
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


def _artifacts(
    sim_scenario,
    out_dir: Path,
    title_prefix: str,
    steps: int,
    plot_cfg: dict[str, Any],
    art: dict[str, Any],
    limits: list[float],
) -> None:
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


def _folder_case(cfg: dict[str, Any], cfg_path: Path) -> None:
    scen = cfg["scenario"]
    xml_path = _resolve_path(cfg_path.parent, scen["cr_xml"])
    if xml_path is None or not xml_path.is_file():
        raise FileNotFoundError(f"scenario.cr_xml must exist: {xml_path}")

    if scen.get("patch_incoming", True):
        xml_path = patch_incoming_ids(xml_path)

    scenario, _ = CommonRoadFileReader(str(xml_path)).open()
    scenario_id = str(getattr(scenario, "scenario_id", "scenario")).replace("/", "_")

    sumo_cfg = cfg["sumo"]
    steps = int(sumo_cfg["steps"])
    mode = str(sumo_cfg["mode"])

    sim_scenario = _run_sumo(
        scenario, mode=mode, cr_xml_path=xml_path, steps=steps, sumo=sumo_cfg
    )

    out_root = _output_run_root(cfg, cfg_path)
    out_cfg = cfg["output"]
    name_base = out_cfg.get("name") or out_cfg.get("subfolder") or scenario_id
    leaf = _run_leaf_name(cfg, base=str(name_base), steps=steps)
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
    )

    if cfg["metrics"]["enabled"]:
        frames = compute_metrics(sim_scenario, car_length=4.7)
        safety = aggregate(0.0, frames, sim_scenario.dt, merge_length=0.0)
        mcfg = cfg["metrics"]
        _save_csv_single([safety], out_dir / mcfg["csv_name"])
        _plot_metrics_single(
            [safety], out_dir / mcfg["plot_name"], figsize=tuple(float(x) for x in plot_cfg["figsize_in"])
        )


def _synthetic_case(cfg: dict[str, Any], cfg_path: Path) -> None:
    from merge_scenario import build_scenario, merge_viewport_xy

    syn = cfg["synthetic"]
    sumo_cfg = cfg["sumo"]
    steps = int(sumo_cfg["steps"])
    mode = str(sumo_cfg["mode"])
    if mode != "random":
        raise ValueError("synthetic scenarios require sumo.mode: random (SUMO generates traffic).")

    out_root_parent = _output_run_root(cfg, cfg_path)
    out_cfg = cfg["output"]
    name_base = out_cfg.get("name") or out_cfg.get("subfolder") or "synthetic_sweep"
    leaf = _run_leaf_name(cfg, base=str(name_base), steps=steps)
    out_root = out_root_parent / leaf
    _ensure_dir(out_root, bool(out_cfg.get("clean")))
    print(f"Output directory: {out_root.resolve()}")

    plot_cfg = cfg["plot"]
    results: list[ScenarioSafety] = []

    for merge_length in syn["merge_lengths"]:
        for kappa in syn["curvatures"]:
            tag = f"merge_L{float(merge_length):g}_k{float(kappa):g}"
            run_dir = out_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            base = build_scenario(
                merge_length=float(merge_length),
                curvature=float(kappa),
                n_main_vehicles=int(syn["n_main_vehicles"]),
                n_ramp_vehicles=int(syn["n_ramp_vehicles"]),
                main_speed=float(syn["main_speed"]),
                ramp_speed=float(syn["ramp_speed"]),
            )
            sim_scenario = _run_sumo(
                base, mode="random", cr_xml_path=None, steps=steps, sumo=sumo_cfg
            )
            limits = merge_viewport_xy(float(merge_length))

            _artifacts(
                sim_scenario,
                run_dir,
                title_prefix=tag,
                steps=steps,
                plot_cfg=plot_cfg,
                art=cfg["artifacts"],
                limits=limits,
            )

            if cfg["metrics"]["enabled"]:
                frames = compute_metrics(sim_scenario, car_length=4.7)
                safety = aggregate(
                    float(kappa),
                    frames,
                    sim_scenario.dt,
                    merge_length=float(merge_length),
                )
                results.append(safety)
                mcfg = cfg["metrics"]
                _save_csv_synthetic([safety], run_dir / mcfg["csv_name"])
                _plot_metrics_single(
                    [safety],
                    run_dir / mcfg["plot_name"],
                    figsize=tuple(float(x) for x in plot_cfg["figsize_in"]),
                )

    if cfg["metrics"]["enabled"] and cfg["metrics"].get("aggregate_synthetic") and len(results) > 1:
        mcfg = cfg["metrics"]
        _plot_metrics_synthetic_multi(
            results,
            out_root / mcfg["aggregate_plot_name"],
            figsize=tuple(float(x) for x in plot_cfg["figsize_in"]),
        )
        _save_csv_synthetic(results, out_root / mcfg["aggregate_csv_name"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run SUMO + CommonRoad from YAML config.")
    p.add_argument("config", type=Path, help="Path to config YAML")
    args = p.parse_args(argv)
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)

    kind = str(cfg["scenario"]["kind"]).lower()
    if kind == "folder":
        _folder_case(cfg, cfg_path)
    elif kind == "synthetic":
        _synthetic_case(cfg, cfg_path)
    else:
        raise ValueError(f"Unknown scenario.kind: {cfg['scenario']['kind']!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
