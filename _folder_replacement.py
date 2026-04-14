def _folder_case(cfg: dict[str, Any], cfg_path: Path) -> None:
    scen = cfg["scenario"]
    xml_path = _resolve_path(cfg_path.parent, scen["cr_xml"])
    if xml_path is None or not xml_path.is_file():
        raise FileNotFoundError(f"scenario.cr_xml must exist: {xml_path}")

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

    def _run_one_sim(seed: int):
        sc, _ = CommonRoadFileReader(str(xml_path)).open()
        sm = dict(sumo_cfg)
        sm["random_seed"] = int(seed)
        return _run_sumo(sc, mode=mode, cr_xml_path=xml_path, steps=steps, sumo=sm)

    safeties: list[ScenarioSafety] = []
    seeds: list[int] = []

    sim_scenario = _run_one_sim(base_seed)
    seeds.append(base_seed)
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
