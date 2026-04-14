"""One-off SUMO + safety aggregate for multiprocessing (no matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_repeat_job(payload: dict[str, Any]):
    """Load scenario from disk, run SUMO, return aggregate safety (pickle-friendly)."""
    from commonroad.common.file_reader import CommonRoadFileReader

    import run as R
    from safety_metrics import aggregate, compute_metrics

    xml = Path(payload["xml_path"])
    if payload.get("patch_incoming"):
        xml = R.patch_incoming_ids(xml)
    scenario, _ = CommonRoadFileReader(str(xml)).open()
    sumo = dict(payload["sumo"])
    sumo["random_seed"] = int(payload["random_seed"])
    sim_scenario = R._run_sumo(
        scenario,
        mode=str(payload["mode"]),
        cr_xml_path=xml,
        steps=int(payload["steps"]),
        sumo=sumo,
    )
    frames = compute_metrics(sim_scenario, car_length=4.7)
    return aggregate(0.0, frames, sim_scenario.dt, merge_length=0.0)
