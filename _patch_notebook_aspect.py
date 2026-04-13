import json

p = r"D:/commonroads/merge.ipynb"
with open(p, encoding="utf-8") as f:
    nb = json.load(f)

n = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]
    if not isinstance(src, list):
        continue
    text = "".join(src)
    if "VERTICAL_SCALE" not in text or "_draw_dual_frame" not in text:
        continue
    out = []
    for line in src:
        if 'ax.set_aspect("equal")' in line:
            out.append(
                line.replace(
                    'ax.set_aspect("equal")',
                    'ax.set_aspect(VERTICAL_SCALE, adjustable="box")',
                )
            )
            n += 1
        elif "figsize=(14, 5.5)" in line and (
            "fig_static" in line or "fig_a" in line
        ):
            out.append(line.replace("figsize=(14, 5.5)", "figsize=(14, 8.5)"))
            n += 1
        else:
            out.append(line)
    cell["source"] = out

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("lines changed:", n)
