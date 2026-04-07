# CommonRoad merge scenario demo

This repository now contains a minimal script that:

1. Builds a **synthetic merge road layout** using `commonroad-io` lanelets.
2. Adds two moving vehicles (mainline + on-ramp merge vehicle).
3. Exports a valid CommonRoad XML scenario.
4. Renders a GIF simulation of the merge.

## Online references used

I used CommonRoad web resources to ground the scenario format and ecosystem context:

- CommonRoad overview / benchmark paper: https://commonroad.in.tum.de/static/media/LA17.8f9d9dc3.pdf
- CommonRoad competition call (describes scenario database coverage and format usage): https://commonroad.in.tum.de/static/media/callForSubmissions_22.dafe9c44.pdf

## Run

```bash
python -m pip install commonroad-io matplotlib numpy scipy
python generate_merge_simulation.py
```

## Outputs

The script writes these at runtime (they are ignored in git and not committed):

- `outputs/USA_MERGE-1_1_T-1.xml` (CommonRoad scenario)
- `outputs/merge_simulation.gif` (working merge simulation)
