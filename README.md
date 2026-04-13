# CommonRoad: SUMO simulation from YAML + map tools

## Run (SUMO + CommonRoad)

```bash
uv sync
$env:PYTHONUTF8='1'   # Windows: avoids UnicodeDecodeError in commonroad_sumo
uv run python run.py config.example.yaml
```

Copy `config.example.yaml`, set `scenario.cr_xml` to your `.cr.xml`, choose `sumo.mode` (`random` or `bundled`). See `config.frankfurt_shorter.yaml` for a filled-in example.

## Map: visualise and interactive crop

```bash
uv run python tools/cr_map.py visualise path/to/scenario.cr.xml --lanes-only -o map.png
uv run python tools/cr_map.py crop-interactive path/to/in.cr.xml path/to/out.cr.xml --mode clicks
```

`crop-interactive` supports `--mode drag` and `--preview`. It delegates to `tools/crop_interactive.py`.
