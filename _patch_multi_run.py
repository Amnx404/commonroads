from pathlib import Path

p = Path("run.py")
t = p.read_text(encoding="utf-8")

needle = '''        "scenario": {
            "kind": "folder",
            "cr_xml": None,
            "patch_incoming": True,
        },
        "sumo": {'''
insert = '''        "scenario": {
            "kind": "folder",
            "cr_xml": None,
            "patch_incoming": True,
        },
        "simulation": {
            "n_runs": 1,
            "parallel_workers": None,
        },
        "sumo": {'''
if needle not in t:
    raise SystemExit("needle1 missing")
t = t.replace(needle, insert, 1)

needle2 = '''        "metrics": {
            "enabled": True,
            "csv_name": "metrics.csv",
            "plot_name": "metrics.png",
        },
    }'''
repl2 = '''        "metrics": {
            "enabled": True,
            "csv_name": "metrics.csv",
            "plot_name": "metrics.png",
            "runs_metrics_xml": "runs_metrics.xml",
            "summary_stats_file": "summary_stats.csv",
        },
    }'''
if needle2 not in t:
    raise SystemExit("needle2 missing")
t = t.replace(needle2, repl2, 1)

old_art = '''def _artifacts(
    sim_scenario,
    out_dir: Path,
    title_prefix: str,
    steps: int,
    plot_cfg: dict[str, Any],
    art: dict[str, Any],
    limits: list[float],
) -> None:
    vscale = float(plot_cfg["vertical_scale"])'''
new_art = '''def _artifacts(
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
    vscale = float(plot_cfg["vertical_scale"])'''
if old_art not in t:
    raise SystemExit("artifacts sig missing")
t = t.replace(old_art, new_art, 1)

helpers = '''

def _repeat_worker_cap(n_runs: int, sim_cfg: dict[str, Any]) -> int:
    """Max parallel workers for extra repeats (not counting the first full run)."""
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
    """Across-run mean/std/min/max/p95 for safety scalars."""
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
        vals = __import__("numpy").array([getattr(s, attr) for s in safeties], dtype=float)
        finite = vals[__import__("numpy").isfinite(vals)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "metric": csv_name,
                "mean": float(__import__("numpy").mean(finite)),
                "std": float(__import__("numpy").std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "min": float(__import__("numpy").min(finite)),
                "max": float(__import__("numpy").max(finite)),
                "p95": float(__import__("numpy").percentile(finite, 95)),
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

'''

marker = "\ndef _artifacts("
if "_repeat_worker_cap" not in t:
    idx = t.find(marker)
    if idx == -1:
        raise SystemExit("marker artifacts not found")
    t = t[:idx] + helpers + t[idx + 1 :]

p.write_text(t, encoding="utf-8")
print("patch1 ok")
