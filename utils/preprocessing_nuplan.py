# utils/preprocessing_nuplan.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np

# nuPlan devkit imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.training.preprocessing.features.agents import Agents

#
def load_nuplan_scene_as_batch(
    data_root: str | None = None,
    map_root: str | None = None,
    db_files: list[str] | None = None,
    scenario_name_contains: str = "",   # pick by substring, leave "" to take the first
    T_past: int = 8,
    T_future: int = 12,
    min_agents: int = 3,
    stride: int = 1,      # take every 'stride' frame (e.g., 2 → downsample)
):
    """
    Returns:
      batch: {"past": (1, A, T_past, 2), "future": (1, A, T_future, 2)}
      agent_ids: list[str]
    Assumes nuPlan devkit is installed and NUPLAN_DATA_ROOT / maps exist.
    """

    # Resolve paths from env if not provided
    data_root = data_root or os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root  = map_root  or os.environ.get("NUPLAN_MAP_ROOT",  os.path.join(data_root, "maps"))
    assert data_root and Path(data_root).exists(), "Set NUPLAN_DATA_ROOT to your nuPlan data folder."
    assert map_root and Path(map_root).exists(),   "Set NUPLAN_MAP_ROOT to your nuPlan maps folder."

    # If db_files not specified, pick the default sqlite(s) under nuplan-vX.Y
    if db_files is None:
        # Common structure: {data_root}/nuplan-v1.1/splits/mini/mini.db, or train/val dbs.
        # Try to find ANY .db in subtree; you can hardcode if you know your file.
        db_files = [str(p) for p in Path(data_root).rglob("*.db")]
        if not db_files:
            raise FileNotFoundError("No .db files found under NUPLAN_DATA_ROOT. Download a mini/train/val split first.")

    # Build scenarios
    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        db_files=db_files,
        verbose=False,
    )

    # Keep the filter permissive; we’ll select by name substring (optional)
    scenario_filter = ScenarioFilter(
        scenario_types=None,    # all types
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=0.0,
        ego_displacement_minimum_m=0.0,
        ego_start_speed_minimum=0.0,
        ego_stop_speed_threshold=0.0,
    )

    scenarios = builder.get_scenarios(scenario_filter)
    if not scenarios:
        raise RuntimeError("No nuPlan scenarios found with the given paths/filter.")
    if scenario_name_contains:
        scenarios = [s for s in scenarios if scenario_name_contains.lower() in s.scenario_name.lower()] or scenarios

    scenario = scenarios[0]  # take first match

    # Gather per-timestep tracked objects; nuPlan scenarios have fixed sampling (e.g., 20 Hz)
    n_iters = scenario.get_number_of_iterations()
    # We need T_past + 1 (current) + T_future frames in a row; choose the earliest feasible window
    window_len = T_past + 1 + T_future
    if n_iters < window_len * stride:
        raise ValueError(f"Scenario too short: {n_iters} < required {window_len*stride} frames with stride={stride}")

    # Anchor the window so that index 'anchor' is the "current" frame
    anchor = T_past * stride
    indices = list(range(anchor - T_past*stride, anchor + (T_future+1)*stride, stride))
    # indices length = T_past + 1 + T_future

    # Build tracks: dict[agent_token] -> list[(t_idx_in_window, x, y)]
    tracks: dict[str, list[tuple[int, float, float]]] = {}

    for local_t, it in enumerate(indices):
        # Tracked objects at iteration 'it'
        objs = scenario.get_tracked_objects_at_iteration(it)
        # Each obj has 3D box (x, y, yaw, size, etc.) and (often) velocity
        for obj in objs.tracked_objects:
            tok = getattr(obj, "track_token", None) or getattr(obj, "token", None)
            if tok is None:
                continue
            # center.xy in global frame [m]
            try:
                x = obj.box.center.x
                y = obj.box.center.y
            except Exception:
                # some versions expose as tuple/array
                x, y = float(obj.box.center[0]), float(obj.box.center[1])
            tracks.setdefault(tok, []).append((local_t, x, y))

    # Keep only agents with a complete window
    A_past, A_fut, kept_ids = [], [], []
    for tok, seq in tracks.items():
        if len(seq) < (T_past + 1 + T_future):
            continue
        seq = sorted(seq, key=lambda z: z[0])  # sort by local t
        # Extract past (0..T_past-1), skip current (T_past), future (T_past+1..)
        past_xy   = np.array([[x, y] for t, x, y in seq[0:T_past]], dtype=float)
        future_xy = np.array([[x, y] for t, x, y in seq[T_past+1:T_past+1+T_future]], dtype=float)
        if past_xy.shape[0] == T_past and future_xy.shape[0] == T_future:
            A_past.append(past_xy)
            A_fut.append(future_xy)
            kept_ids.append(tok)

    if len(A_past) < min_agents:
        raise ValueError(f"Only {len(A_past)} valid agents; need >= {min_agents}.")

    past   = np.stack(A_past, axis=0)     # (A, T_past, 2)
    future = np.stack(A_fut,  axis=0)     # (A, T_future, 2)

    batch = {"past": past[None, ...], "future": future[None, ...]}  # (1, A, T, 2)
    return batch, kept_ids
