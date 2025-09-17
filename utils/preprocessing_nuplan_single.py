# utils/preprocessing_nuplan_single.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

def _is_vehicle(obj) -> bool:
    """
    Returns True only for non-ego vehicle objects.
    We check tracked_object_type for robustness across devkit versions.
    """
    t = getattr(obj, "tracked_object_type", None)
    if t is None:
        return False
    # t can be an Enum or string-like; be lenient:
    name = getattr(t, "name", None) or str(t)
    return "VEHICLE" in name.upper()

# This function returns a batch with a single scenario and single agent, and the agent's 
# past and future positions, we are going to predict the future based on the past.
# That data consists of:
#   past:   (1, 1, T_past, 2) -- (batch, agent, time, (x,y)) that mean for the
#   last T_past frames we know the positions of the agent(car)
#   future: (1, 1, T_future, 2) -- (batch, agent, time, (x,y)) that mean for the next T_future frames
#   we want to predict the positions of the agent(car)
def load_nuplan_single_vehicle(
    data_root: str | None = None,
    map_root: str | None = None,
    db_files: list[str] | None = None,
    scenario_name_contains: str = "",   # "" -> first scenario
    T_past: int = 8,
    T_future: int = 12,
    stride: int = 1,
    use_ego: bool = True,               # True = ego car, False = first non-ego vehicle
):
    """
    Returns:
      batch: {"past": (1, 1, T_past, 2), "future": (1, 1, T_future, 2)}
      agent_id: str  ("ego" or vehicle token)
    Only CAR/VEHICLE agents are considered (if use_ego=False).
    """
    data_root = data_root or os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root  = map_root  or os.environ.get("NUPLAN_MAP_ROOT", os.path.join(data_root, "maps"))
    assert data_root and Path(data_root).exists(), "Set NUPLAN_DATA_ROOT to your nuPlan data folder."
    assert map_root and Path(map_root).exists(),   "Set NUPLAN_MAP_ROOT to your nuPlan maps folder."

    if db_files is None:
        db_files = [str(p) for p in Path(data_root).rglob("*.db")]
        if not db_files:
            raise FileNotFoundError("No .db files found under NUPLAN_DATA_ROOT.")

    builder = NuPlanScenarioBuilder(data_root=data_root, map_root=map_root, db_files=db_files, verbose=False)
    scenario_filter = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=None,
        timestamp_threshold_s=0.0, ego_displacement_minimum_m=0.0,
        ego_start_speed_minimum=0.0, ego_stop_speed_threshold=0.0,
    )
    scenarios = builder.get_scenarios(scenario_filter)
    if not scenarios:
        raise RuntimeError("No scenarios found.")
    if scenario_name_contains:
        scenarios = [s for s in scenarios if scenario_name_contains.lower() in s.scenario_name.lower()] or scenarios

    scenario = scenarios[0]

    needed = (T_past + 1 + T_future) * stride
    n_iters = scenario.get_number_of_iterations()
    if n_iters < needed:
        raise ValueError(f"Scenario too short: {n_iters} < {needed} frames with stride={stride}")

    anchor = T_past * stride
    idxs = list(range(anchor - T_past*stride, anchor + (T_future+1)*stride, stride))  # Tp + 1 + Tf

    xs, ys = [], []
    agent_id = "ego"

    if use_ego:
        # Ego car only (always a vehicle)
        for it in idxs:
            ego = scenario.get_ego_state_at_iteration(it)
            # robust position extraction
            for attr_seq in (("center","x","y"), ("rear_axle","x","y")):
                try:
                    obj = getattr(ego, attr_seq[0])
                    x = float(getattr(obj, attr_seq[1]))
                    y = float(getattr(obj, attr_seq[2]))
                    break
                except Exception:
                    x = y = None
            if x is None:
                pos = getattr(ego, "pose", None)
                if pos is None:
                    raise RuntimeError("Cannot extract ego position.")
                x, y = float(pos.x), float(pos.y)
            xs.append(x); ys.append(y)
    else:
        # First non-ego VEHICLE with a complete window
        # We'll gather vehicles at each timestep keyed by token, then pick any with full coverage.
        tracks = {}
        for local_t, it in enumerate(idxs):
            objs = scenario.get_tracked_objects_at_iteration(it)
            for obj in objs.tracked_objects:
                if not _is_vehicle(obj):
                    continue
                tok = getattr(obj, "track_token", None) or getattr(obj, "token", None)
                if tok is None:
                    continue
                cx, cy = float(obj.box.center.x), float(obj.box.center.y)
                tracks.setdefault(tok, []).append((local_t, cx, cy))
        # select first with complete window
        sel = None
        for tok, seq in tracks.items():
            if len(seq) >= (T_past + 1 + T_future):
                seq = sorted(seq, key=lambda z: z[0])
                sel = (tok, seq)
                break
        if sel is None:
            raise RuntimeError("No vehicle with a full window in this scenario.")
        agent_id, seq = sel
        xs = [p[1] for p in seq]
        ys = [p[2] for p in seq]

    xs = np.asarray(xs); ys = np.asarray(ys)
    past_xy   = np.stack([xs[:T_past], ys[:T_past]], axis=-1)                       # (Tp,2)
    future_xy = np.stack([xs[T_past+1:T_past+1+T_future], ys[T_past+1:T_past+1+T_future]], axis=-1)  # (Tf,2)

    past   = past_xy[None, None, ...]     # (1,1,Tp,2)
    future = future_xy[None, None, ...]   # (1,1,Tf,2)
    return {"past": past, "future": future}, agent_id
