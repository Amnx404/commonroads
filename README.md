# CommonRoad merge / SUMO experiments

Single entry point: `run.py` reads a `YAML config` (paths, figure size, SUMO steps, plot viewport, synthetic sweep parameters). Copy `config.example.yaml` and edit.

## Run

```bash
uv sync
uv run python run.py config.example.yaml
```

## Outputs

Each run writes under `output.dir` (and optional `output.group`) into a new folder named {name}_{N}steps_{YYYYMMDD_HHMMSS}. The base name is `output.name`, else `output.subfolder`, else the scenario id (folder maps) or `synthetic_sweep`. Toggle `output.timestamp_folder` or `output.include_steps_in_folder_name` if you need a stable path. The script prints the resolved output directory when it starts writing.

## Map preview and crop

```bash
uv run python tools/cr_map.py visualise scenarios/.../scenario.cr.xml --lanes-only -o preview.png
uv run python tools/cr_map.py visualise scenario.cr.xml --view-fraction 0.35
uv run python tools/cr_map.py visualise scenario.cr.xml --limits-box x1 y1 x2 y2 --lanes-only -o zoom.png
uv run python tools/cr_map.py crop-box in.cr.xml out.cr.xml --box x1 y1 x2 y2
uv run python tools/cr_map.py crop-interactive in.cr.xml out.cr.xml --mode clicks
uv run python tools/cr_map.py crop-interactive in.cr.xml --mode drag --preview
```

`--box` and `--limits-box`: two corners in map metres; crop/viewport is axis-aligned min/max. For a vertical strip on the right, use `--top-right` and `--bottom-right` with the same x plus `--left-x`. `tools/visualise_map.py` forwards to `cr_map visualise`. Interactive crop needs a GUI matplotlib backend (`MPLBACKEND=TkAgg` if needed).

## Layout

- `run.py` — config-driven pipeline (bundled SUMO folder or synthetic merge).
- `merge_scenario.py` — parametric merge geometry + IDM vehicles (optional).
- `idm.py` — Intelligent Driver Model for merge scenarios.
- `safety_metrics.py` — TTC / DRAC / BTN / TIT metrics from CommonRoad states.
- `tools/cr_map.py` — visualise, crop-box, crop-interactive.
- `tools/crop_interactive.py` — GUI pick of crop rectangle (also invoked via `cr_map crop-interactive`).

## References

- CommonRoad overview: https://commonroad.in.tum.de/static/media/LA17.8f9d9dc3.pdf
- Competition materials: https://commonroad.in.tum.de/static/media/callForSubmissions_22.dafe9c44.pdf
