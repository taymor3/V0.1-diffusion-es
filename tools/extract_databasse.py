# === Minimal per-family export using the nuPlan devkit (mirrors your example) ===
import os, json, math
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from tqdm import tqdm

# nuPlan devkit (same style as your snippet)
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# ---- config (envs like your example) ----
DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT", "/data/nuplan-v1.1/trainval")
MAP_ROOT  = os.getenv("NUPLAN_MAPS_ROOT", "/data/nuplan-v1.1/maps")
MAP_VER   = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
SENSOR_ROOT = os.getenv("NUPLAN_SENSOR_ROOT", None)
DB_FILES = os.getenv("NUPLAN_DB_FILES", None)   # dir with .db files or a single .db (optional)
from pathlib import Path
if DB_FILES and os.path.isdir(DB_FILES):
    DB_FILES = [str(p) for p in sorted(Path(DB_FILES).glob("*.db"))[:2]]  # <-- try 1–2 files first
OUT_DIR = Path("./cache/family_dbs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# reuse your families from Cell 3; otherwise define them here
# Family name -> list of scenario-type tags
FAMILY_MAP = {
    "turn_left": [
        "starting_left_turn",
    ],
    "turn_right": [
        "starting_right_turn",
    ],
    "straight_through_intersection": [
        "traversing_intersection",
        "traversing_traffic_light_intersection",
        "on_intersection",
        "on_all_way_stop_intersection",
        "on_traffic_light_intersection",
        "starting_straight_stop_sign_intersection_traversal",
        "starting_straight_traffic_light_intersection_traversal",
        "on_stopline_stop_sign",
    ],
    "lane_change_left": [
        "changing_lane_to_left",
    ],
    "lane_change_right": [
        "changing_lane_to_right",
    ],
    "stop_for_red_light": [
        "on_stopline_traffic_light",
        "stationary_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "stopping_at_traffic_light_with_lead",
        "stopping_at_traffic_light_without_lead",
    ],
    "go_on_green": [
        "accelerating_at_traffic_light",
        "accelerating_at_traffic_light_with_lead",
        "accelerating_at_traffic_light_without_lead",
        "starting_straight_traffic_light_intersection_traversal",
    ],
    "yield_to_pedestrian": [
        "near_pedestrian_on_crosswalk",
        "near_pedestrian_on_crosswalk_with_ego",
        "stationary_at_crosswalk",
        "stopping_at_crosswalk",
        "traversing_crosswalk",
        "waiting_for_pedestrian_to_cross",
        "on_stopline_crosswalk",
        "behind_pedestrian_on_driveable",
        "near_pedestrian_at_pickup_dropoff",
    ],
    "cut_in": [
        "near_long_vehicle",
        "near_high_speed_vehicle",
        "near_multiple_vehicles",
    ],
    "car_following": [
        "stationary_in_traffic",
        "stationary",
        "following_lane_with_lead",
        "following_lane_with_slow_lead",
        "following_lane_without_lead",
        "stopping_with_lead",
        "behind_bike",
    ],
    "pickup_dropoff": [
        "on_pickup_dropoff",
        "traversing_pickup_dropoff",
        "behind_pedestrian_on_pickup_dropoff",
        "near_pedestrian_at_pickup_dropoff",
        "on_carpark",
    ],
    "construction_zone": [
        "near_construction_zone_sign",
        "near_trafficcone_on_driveable",
        "near_barrier_on_driveable",
        "traversing_narrow_lane",
    ],
    "speed_events": [
        "high_magnitude_speed",
        "medium_magnitude_speed",
        "low_magnitude_speed",
        "high_magnitude_jerk",
        "high_lateral_acceleration",
    ],
}


# If other parts of your notebook still expect FAMILY_TYPES as a list of family names:
FAMILY_TYPES = list(FAMILY_MAP.keys())


# optional: restrict to training logs if file is present (same as your example)
log_names = None
if os.path.exists("./nuplan_train.json"):
    with open("./nuplan_train.json", "r", encoding="utf-8") as f:
        log_names = json.load(f)

# ---- builder (same constructor shape as your example) ----
builder = NuPlanScenarioBuilder(
    data_root=DATA_ROOT,
    map_root=MAP_ROOT,
    sensor_root=SENSOR_ROOT,  # can be None
    db_files=DB_FILES,        # can be None; builder will scan DATA_ROOT
    map_version=MAP_VER,
)

# a simple worker (process pool can be flaky in notebooks; set to False if needed)
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
worker = SingleMachineParallelExecutor(use_process_pool=False)
# ---- a tiny helper to match the ScenarioFilter positional signature from your example ----
def get_filter_parameters(
    scenario_types=None,
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None
    # NOTE: order matters; this matches the ctor your example uses
    return (
        scenario_types, scenario_tokens, log_names, map_names,
        num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s,
        ego_displacement_minimum_m, expand_scenarios, remove_invalid_goals,
        shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance
    )

def _f(x):
    try: return float(x)
    except: return None

def _pose_size_vel(obj) -> Dict[str, Any]:
    x = y = yaw = None
    c = getattr(obj, "center", None)
    if c is not None:
        x = _f(getattr(c, "x", None)); y = _f(getattr(c, "y", None)); yaw = _f(getattr(c, "heading", None))
    yaw = _f(getattr(obj, "yaw", yaw))
    length = width = height = None
    box = getattr(obj, "box", None)
    if box is not None:
        length = _f(getattr(box, "length", None)); width = _f(getattr(box, "width", None)); height = _f(getattr(box, "height", None))
    vx = vy = speed = None
    vel = getattr(obj, "velocity", None)
    if vel is not None:
        vx = _f(getattr(vel, "x", None)); vy = _f(getattr(vel, "y", None))
        if vx is not None and vy is not None: speed = math.hypot(vx, vy)
    return dict(x=x, y=y, yaw=yaw, length=length, width=width, height=height, vx=vx, vy=vy, speed=speed)

def _obj_id(o):
    for k in ("track_token", "token", "track_id", "instance_token"):
        if hasattr(o, k): return str(getattr(o, k))
    return None

def _obj_type(o):
    t = getattr(o, "tracked_object_type", None)
    return getattr(t, "name", str(t))
def _time_s_from(s, i):
    """Robust time (seconds) for iteration i across nuPlan versions."""
    # Preferred (if available)
    if hasattr(s, "get_time_point_at_iteration"):
        tp = s.get_time_point_at_iteration(i)
        if hasattr(tp, "time_s"):   return float(tp.time_s)
        if hasattr(tp, "time_us"):  return float(tp.time_us) / 1e6
        if hasattr(tp, "time_ns"):  return float(tp.time_ns) / 1e9

    # Fallback: via ego state's time_point
    ego = s.get_ego_state_at_iteration(i)
    tp = getattr(ego, "time_point", None)
    if tp is not None:
        if hasattr(tp, "time_s"):   return float(tp.time_s)
        if hasattr(tp, "time_us"):  return float(tp.time_us) / 1e6
        if hasattr(tp, "time_ns"):  return float(tp.time_ns) / 1e9
    return None

def export_family(family: str, limit_total: Optional[int] = None):
    # build the filter EXACTLY like your example (positional), but with scenario_types=[family]
    scenario_types = FAMILY_MAP[family]            # list[str]

    filt = ScenarioFilter(*get_filter_parameters(
        scenario_types=scenario_types,
        num_scenarios_per_type=None,
        limit_total_scenarios=limit_total,
        shuffle=True,
        scenario_tokens=None,
        log_names=log_names,
    ))

    scenarios = builder.get_scenarios(filt, worker)
    rows: List[Dict[str, Any]] = []

    for s in tqdm(scenarios, desc=family, leave=False):
        meta = {
            "family": family,
            "scenario_type": getattr(s, "scenario_type", family),
            "scenario_name": getattr(s, "scenario_name", None),
            "log_name": getattr(s, "log_name", None),
            "map_name": getattr(s, "map_name", None),
        }
        n = int(s.get_number_of_iterations()) if hasattr(s, "get_number_of_iterations") \
            else int(getattr(s, "number_of_iterations", 0))

        for i in range(n):
            time_s = _time_s_from(s, i)

            tracked = s.get_tracked_objects_at_iteration(i)
            objs = getattr(tracked, "tracked_objects", None) or []
            for o in objs:
                tname = _obj_type(o)
                if tname not in (TrackedObjectType.VEHICLE.name, TrackedObjectType.PEDESTRIAN.name):
                    continue
                rows.append({
                    **meta,
                    "iteration": i,
                    "time_s": time_s,
                    "agent_id": _obj_id(o),
                    "agent_type": tname,
                    **_pose_size_vel(o),
                })

    if not rows:
        print(f"[{family}] no rows; skipping.")
        return

    df = pd.DataFrame(rows)
    out_fp = OUT_DIR / f"{family}.parquet"
    try:
        df.to_parquet(out_fp, index=False)
    except Exception:
        out_fp = OUT_DIR / f"{family}.csv.gz"
        df.to_csv(out_fp, index=False, compression="gzip")
    print(f"[{family}] wrote {len(df):,} rows -> {out_fp}")
    
def get_db_files(number_of_scenarios =2):
    # ---- run for all families ----
    for fam in FAMILY_TYPES:
        export_family(fam, number_of_scenarios)  # set to a small int to smoke-test
