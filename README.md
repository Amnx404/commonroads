# CommonRoad: SUMO simulation from YAML + map tools

## Setup

```bash
uv sync
```

On **Windows**, if `commonroad_sumo` raises `UnicodeDecodeError` when using random traffic, force UTF-8 **before** starting Python:

```powershell
$env:PYTHONUTF8 = "1"
```

## Run (SUMO + CommonRoad)

Takes a single YAML config (paths inside it are relative to the config file directory).

```bash
uv run python run.py <config.yaml>
```

**Examples**

```bash
uv run python run.py config.example.yaml
uv run python run.py config.frankfurt_shorter.yaml
```

Copy `config.example.yaml`, set `scenario.cr_xml`, and choose `sumo.mode` (`random` or `bundled`). Other keys (output layout, plots, `sumo.random_density_scale`, etc.) are optional; see `config.example.yaml` and `config.frankfurt_shorter.yaml`.

### Multi-run replicates (`simulation.n_runs`)

- Run **1** is full fidelity: lane PNG, three-times PNG, GIF (per `artifacts`), then safety metrics.
- Runs **2 … N** only re-run SUMO + metrics (no PNG/GIF) and are executed **in parallel** when `simulation.parallel_workers` is not `1` (default `null` → up to `min(8, CPU cores, N−1)` workers).
- Each run uses `sumo.random_seed + (run_index)` so random traffic differs between replicates.
- **`n_runs` > 1 requires `metrics.enabled: true`** (combined outputs need per-run metrics).

Outputs in the same run folder:

| File | Content |
|------|---------|
| `metrics.csv` | One row per run + `random_seed` column |
| `metrics.png` | Panel plot across runs |
| `runs_metrics.xml` | All per-run safety scalars in one XML document |
| `summary_stats.csv` | Across-run mean, std, min, max, p95 for each safety metric |

Progress for extra runs uses **tqdm** (`Extra runs` bar).

---

## Map: `visualise`

```bash
uv run python tools/cr_map.py visualise <scenario.cr.xml> [options]
```

| Option | Meaning |
|--------|---------|
| `-o`, `--output` `<path>` | Output PNG (default: `<xml_stem>_map.png` beside the XML) |
| `--lanes-only` | Lanelet network only (no dynamic obstacles) |
| `--time-step` `<int>` | Time step to draw (default: `0`) |
| `--figsize` `<w> <h>` | Figure size in inches (default: `16.0` `6.0`) |
| `--dpi` `<int>` | Raster DPI (default: `150`) |
| `--margin` `<float>` | Metres padding around lanelet bbox (default: `12.0`) |
| `--view-fraction` `<float>` | Zoom to central fraction of bbox (e.g. `0.35`); omit for full extent |
| `--limits-box` `<X1> <Y1> <X2> <Y2>` | Fixed axis-aligned limits in map metres |
| `--vertical-scale` `<float>` | Matplotlib y-aspect scale (default: `10.0`) |
| `--show` | Open interactive window |
| `--labels` | Draw lanelet id labels |

**Examples**

```bash
uv run python tools/cr_map.py visualise scenarios/foo/scenario.cr.xml --lanes-only -o map.png
uv run python tools/cr_map.py visualise scenarios/foo/scenario.cr.xml --view-fraction 0.35 --show
```

---

## Map: `crop-box`

Axis-aligned crop in map metres.

```bash
uv run python tools/cr_map.py crop-box <input.cr.xml> <output.cr.xml> [bbox options]
```

**Bounding box (choose one)**

| Option | Meaning |
|--------|---------|
| `--box` `<X1> <Y1> <X2> <Y2>` | Two diagonal corners |
| `--top-right` `<X> <Y>` and `--bottom-left` `<X> <Y>` | Opposite corners |
| `--top-right` `<X> <Y>` and `--bottom-right` `<X> <Y>` and `--left-x` `<X>` | Vertical strip (same x on the right) |

| Option | Meaning |
|--------|---------|
| `--name-suffix` `<str>` | Appended to scenario map name (default: `CropBox`) |

**Example**

```bash
uv run python tools/cr_map.py crop-box in.cr.xml out.cr.xml --box 0 0 500 300 --name-suffix MyCrop
```

---

## Map: `crop-interactive`

Forwards to `tools/crop_interactive.py`:

```bash
uv run python tools/cr_map.py crop-interactive <input.cr.xml> [<output.cr.xml>] [options]
```

Or:

```bash
uv run python tools/crop_interactive.py <input.cr.xml> [<output.cr.xml>] [options]
```

| Option | Meaning |
|--------|---------|
| `<output.cr.xml>` | Optional; if omitted, prints a `crop-box` command line only (no XML written) |
| `--mode` `clicks` or `drag` | Two-click corners vs drag rectangle + **Enter** (default: `clicks`) |
| `--figsize` `<w> <h>` | Inches (default: `16.0` `6.0`) |
| `--dpi` `<int>` | Default: `120` |
| `--margin` `<float>` | View padding in m (default: `12.0`) |
| `--view-fraction` `<float>` | Optional zoom on the pick view |
| `--vertical-scale` `<float>` | Default: `10.0` |
| `--preview` | Second figure: red crop rectangle outline |
| `--name-suffix` `<str>` | When writing output (default: `CropBox`) |

**Examples**

```bash
uv run python tools/cr_map.py crop-interactive map.cr.xml cropped.cr.xml --mode clicks
uv run python tools/cr_map.py crop-interactive map.cr.xml --mode drag --preview
```

Use a GUI Matplotlib backend if needed (e.g. set `MPLBACKEND=TkAgg`).
