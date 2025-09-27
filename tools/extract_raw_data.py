# tools/extract_states.py
import os, glob, math, sqlite3, numpy as np, pandas as pd
from pathlib import Path

# ---------- paths (reuse your setup) ----------
NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT", "/home/taimor/data1/nuplan-v1.1/splits/mini")
OUT_DIR = Path("processed/nuplan_parquet"); 
OUT_DIR.mkdir()

# ---------- helpers ----------
def yaw_from_quaternion(qw, qx, qy, qz):
    # standard ZYX yaw from quaternion
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def add_derivatives(df, id_col, t_col, x_col, y_col, vx_col=None, vy_col=None):
    # timestamps in nuPlan are integers (microseconds). Convert to seconds.
    df = df.sort_values([id_col, t_col]).copy()
    dt = df.groupby(id_col)[t_col].diff().astype("float64") / 1e6
    # velocity
    if vx_col is None or vy_col is None:
        dx = df.groupby(id_col)[x_col].diff()
        dy = df.groupby(id_col)[y_col].diff()
        df["velocity_x"] = (dx / dt).replace([np.inf, -np.inf], np.nan)
        df["velocity_y"] = (dy / dt).replace([np.inf, -np.inf], np.nan)
    else:
        df["velocity_x"] = df[vx_col]
        df["velocity_y"] = df[vy_col]
    df["speed"] = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)
    # acceleration
    dvx = df.groupby(id_col)["velocity_x"].diff()
    dvy = df.groupby(id_col)["velocity_y"].diff()
    df["acceleration_x"] = (dvx / dt).replace([np.inf, -np.inf], np.nan)
    df["acceleration_y"] = (dvy / dt).replace([np.inf, -np.inf], np.nan)
    df["acceleration"] = np.sqrt(df["acceleration_x"]**2 + df["acceleration_y"]**2)
    # yaw rate from heading
    dpsi = df.groupby(id_col)["heading"].diff()
    # unwrap heading before diff to avoid 2π jumps
    df["heading_unwrapped"] = df.groupby(id_col)["heading"].transform(np.unwrap)
    dpsi = df.groupby(id_col)["heading_unwrapped"].diff()
    df["yaw_rate"] = (dpsi / dt).replace([np.inf, -np.inf], np.nan)
    return df.drop(columns=["heading_unwrapped"])
def process_one_db(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    db_stem = Path(db_path).stem
    db_name = Path(db_path).name

    def has_col(table, col):
        cols = pd.read_sql_query(f"PRAGMA table_info({table})", con)["name"].tolist()
        return col in cols

    # ---- Ego (log_token may or may not exist) ----
    ego_cols = [
        "timestamp","x AS center_x","y AS center_y","z",
        "qw","qx","qy","qz",
        "acceleration_x","acceleration_y","acceleration_z",
        "angular_rate_x","angular_rate_y","angular_rate_z"
    ]
    if has_col("ego_pose", "log_token"):
        ego_cols = ["log_token"] + ego_cols
    ego = pd.read_sql_query(f"SELECT {', '.join('ep.'+c if ' AS ' not in c else 'ep.'+c for c in ego_cols)} FROM ego_pose ep", con)

    if not ego.empty:
        ego["heading"] = np.vectorize(yaw_from_quaternion)(ego.qw, ego.qx, ego.qy, ego.qz)
        ego["agent_id"] = "ego"
        ego["agent_type"] = "EGO"
        ego["steering_angle"] = np.nan
        ego.rename(columns={"angular_rate_z": "yaw_rate"}, inplace=True)
        ego = add_derivatives(ego, id_col="agent_id", t_col="timestamp",
                              x_col="center_x", y_col="center_y")
        # if log_token missing, fill from file stem
        if "log_token" not in ego.columns:
            ego["log_token"] = db_stem
        ego = ego[[
            "log_token","timestamp","agent_id","agent_type",
            "center_x","center_y","heading",
            "velocity_x","velocity_y","speed",
            "acceleration_x","acceleration_y","acceleration",
            "yaw_rate","steering_angle"
        ]]
        ego["source_db"] = db_name

    # ---- Agents: timestamp from lidar_pc; vx/vy may not exist ----
    vxvy = ", lb.vx, lb.vy" if has_col("lidar_box","vx") and has_col("lidar_box","vy") else ""
    agents = pd.read_sql_query(f"""
        SELECT 
            lb.track_token AS agent_id,
            lpc.timestamp,
            lb.x AS center_x,
            lb.y AS center_y,
            lb.yaw AS heading
            {vxvy},
            tr.category_token,
            COALESCE(cat.name, 'UNKNOWN') AS agent_type
        FROM lidar_box AS lb
        JOIN lidar_pc AS lpc ON lpc.token = lb.lidar_pc_token
        LEFT JOIN track AS tr ON tr.token = lb.track_token
        LEFT JOIN category AS cat ON cat.token = tr.category_token
    """, con)

    if not agents.empty:
        agents["agent_type"] = agents["agent_type"].str.upper()
        agents = add_derivatives(
            agents, id_col="agent_id", t_col="timestamp",
            x_col="center_x", y_col="center_y",
            vx_col="vx" if "vx" in agents.columns else None,
            vy_col="vy" if "vy" in agents.columns else None
        )
        agents["steering_angle"] = np.nan
        agents["log_token"] = ego["log_token"].iloc[0] if not ego.empty else db_stem
        agents["source_db"] = db_name
        agents = agents[[
            "log_token","timestamp","agent_id","agent_type",
            "center_x","center_y","heading",
            "velocity_x","velocity_y","speed",
            "acceleration_x","acceleration_y","acceleration",
            "yaw_rate","steering_angle","source_db"
        ]]

    # ---- unify ----
    if (ego is None or ego.empty) and (agents is None or agents.empty):
        con.close()
        return pd.DataFrame()

    frames = []
    if not ego.empty: frames.append(ego)
    if not agents.empty: frames.append(agents)
    df = pd.concat(frames, ignore_index=True, sort=False)

    con.close()
    return df

def main():
    db_files = sorted(glob.glob(os.path.join(NUPLAN_DATA_ROOT, "*.db")))
    if not db_files:
        raise FileNotFoundError(f"No .db under {NUPLAN_DATA_ROOT}")
    frames = []
    for db in db_files[:3]:
        print(f"→ processing {db}")
        frames.append(process_one_db(db))
    all_df = pd.concat(frames, ignore_index=True).sort_values(["source_db","timestamp","agent_id"])
    # parquet for speed / downstream ML (pyarrow required)
    out_path = OUT_DIR / "nuplan_states.parquet"
    all_df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} with {len(all_df):,} rows")

if __name__ == "__main__":
    main()
