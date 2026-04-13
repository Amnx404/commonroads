from pathlib import Path

p = Path(__file__).resolve().parent / "README.md"
t = p.read_text(encoding="utf-8")
# Collapse duplicate ## Outputs into one section
start = t.find("## Outputs")
if start == -1:
    raise SystemExit(0)
rest = t[start + len("## Outputs") :]
next_h2 = rest.find("\n## ")
if next_h2 == -1:
    raise SystemExit("unexpected README format")

# find second ## Outputs if present
second = rest.find("\n## Outputs", 0, next_h2)
if second != -1:
    before = t[: start + len("## Outputs")]
    after_layout = t.find("\n## Layout", start)
    body = (
        "\n\nEach run writes under `output.dir` (and optional `output.group`) into a new folder "
        "named `{name}_{N}steps_{YYYYMMDD_HHMMSS}`. The base `name` is `output.name`, else "
        "`output.subfolder`, else the scenario id (folder maps) or `synthetic_sweep`. Toggle "
        "`output.timestamp_folder` or `output.include_steps_in_folder_name` if you need a "
        "stable path. The script prints the resolved output directory when it starts writing.\n"
    )
    t = before + body + t[after_layout:]
    p.write_text(t, encoding="utf-8")
