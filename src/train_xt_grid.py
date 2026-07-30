"""
Classic 16×12 Markov Expected Threat (xT) trainer
Reads raw StatsBomb event JSONs and produces:
  - models/xt_grid_16x12.npy
  - data/processed/actions_xt_grid.parquet
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EVENTS_DIR = ROOT / "data" / "raw" / "events"
OUT_GRID = ROOT / "models" / "xt_grid_16x12.npy"
OUT_ACTIONS = ROOT / "data" / "processed" / "actions_xt_grid.parquet"

# ------------------------------------------------------------------
# Grid settings (classic Karun Singh)
# ------------------------------------------------------------------
N_X = 16
N_Y = 12
PITCH_X = 120.0
PITCH_Y = 80.0

# Set-piece patterns to exclude
SET_PIECE_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Kick Off",
    "From Keeper",
    "From Counter",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_events(match_id: int) -> list:
    path = EVENTS_DIR / f"{match_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bin_coords(x: np.ndarray, y: np.ndarray):
    x_bin = np.clip((x / PITCH_X * N_X).astype(int), 0, N_X - 1)
    y_bin = np.clip((y / PITCH_Y * N_Y).astype(int), 0, N_Y - 1)
    return x_bin, y_bin


# ------------------------------------------------------------------
# 1. Collect actions (open-play only)
# ------------------------------------------------------------------
print("Loading events …")
rows = []
event_files = sorted(EVENTS_DIR.glob("*.json"))
print(f"Found {len(event_files)} match files")

for ef in event_files:
    match_id = int(ef.stem)
    events = load_events(match_id)

    for e in events:
        # --- open-play filter ---
        pattern = e.get("play_pattern", {}).get("name")
        if pattern in SET_PIECE_PATTERNS:
            continue

        t = e.get("type", {}).get("name")
        if t not in ("Pass", "Carry", "Shot"):
            continue

        loc = e.get("location")
        if not loc or len(loc) < 2:
            continue

        start_x, start_y = loc[0], loc[1]

        end = None
        if t == "Pass":
            end = e.get("pass", {}).get("end_location")
        elif t == "Carry":
            end = e.get("carry", {}).get("end_location")
        elif t == "Shot":
            end = e.get("shot", {}).get("end_location") or loc

        if not end or len(end) < 2:
            continue

        end_x, end_y = end[0], end[1]

        outcome = None
        if t == "Pass":
            outcome = e.get("pass", {}).get("outcome", {}).get("name")
        elif t == "Shot":
            outcome = e.get("shot", {}).get("outcome", {}).get("name")

        rows.append({
            "match_id": match_id,
            "index": e.get("index"),
            "type": t,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "outcome": outcome,
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "minute": e.get("minute"),
            "possession": e.get("possession"),
            "play_pattern": pattern,
        })

df = pd.DataFrame(rows)
print(f"Total open-play actions loaded: {len(df):,}")

# ------------------------------------------------------------------
# 2. Successful moves + shots
# ------------------------------------------------------------------
is_shot = df["type"] == "Shot"
is_move = df["type"].isin(["Pass", "Carry"])

successful_move = is_move & (
    (df["type"] == "Carry") |
    (df["outcome"].isna())
)

df_moves = df[successful_move].copy()
df_shots = df[is_shot].copy()

print(f"Successful moves: {len(df_moves):,}")
print(f"Shots:            {len(df_shots):,}")

# ------------------------------------------------------------------
# 3. Bin onto 16×12 grid
# ------------------------------------------------------------------
sx, sy = bin_coords(df_moves["start_x"].values, df_moves["start_y"].values)
ex, ey = bin_coords(df_moves["end_x"].values,   df_moves["end_y"].values)

df_moves["start_bin"] = sx * N_Y + sy
df_moves["end_bin"]   = ex * N_Y + ey

shot_x, shot_y = bin_coords(df_shots["start_x"].values, df_shots["start_y"].values)
df_shots["bin"] = shot_x * N_Y + shot_y
df_shots["is_goal"] = df_shots["outcome"] == "Goal"

# ------------------------------------------------------------------
# 4. Zone probabilities (with light smoothing)
# ------------------------------------------------------------------
n_zones = N_X * N_Y
eps = 1e-6

move_start_count = np.bincount(df_moves["start_bin"], minlength=n_zones) + eps
shot_count       = np.bincount(df_shots["bin"],       minlength=n_zones) + eps
goal_count       = np.bincount(
    df_shots.loc[df_shots["is_goal"], "bin"], minlength=n_zones
)

total_start = move_start_count + shot_count

move_prob = move_start_count / total_start
shot_prob = shot_count / total_start
goal_prob = goal_count / shot_count          # still can be zero in some zones

# ------------------------------------------------------------------
# 5. Transition matrix
# ------------------------------------------------------------------
T = np.zeros((n_zones, n_zones))
from_bins = df_moves["start_bin"].values
to_bins   = df_moves["end_bin"].values

for f, t in zip(from_bins, to_bins):
    T[f, t] += 1.0

row_sums = T.sum(axis=1, keepdims=True)
T = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)

# ------------------------------------------------------------------
# 6. Value iteration (more iterations + tighter tolerance)
# ------------------------------------------------------------------
print("Solving xT by value iteration …")
xT = shot_prob * goal_prob

for iteration in range(100):
    xT_new = shot_prob * goal_prob + move_prob * (T @ xT)
    diff = np.abs(xT_new - xT).max()
    xT = xT_new
    if diff < 1e-8:
        print(f"Converged after {iteration + 1} iterations (max diff = {diff:.2e})")
        break
else:
    print(f"Reached max iterations (last max diff = {diff:.2e})")

xT_grid = xT.reshape(N_X, N_Y)
print("xT surface shape:", xT_grid.shape)
print(f"xT range: {xT_grid.min():.4f} → {xT_grid.max():.4f}")

# ------------------------------------------------------------------
# 7. Value every successful move
# ------------------------------------------------------------------
df_moves["xt_start"] = xT[df_moves["start_bin"].values]
df_moves["xt_end"]   = xT[df_moves["end_bin"].values]
df_moves["xt_value"] = df_moves["xt_end"] - df_moves["xt_start"]

# ------------------------------------------------------------------
# 8. Save
# ------------------------------------------------------------------
OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_GRID, xT_grid)
print(f"Saved grid → {OUT_GRID}")

OUT_ACTIONS.parent.mkdir(parents=True, exist_ok=True)
df_moves.to_parquet(OUT_ACTIONS, index=False)
print(f"Saved valued actions → {OUT_ACTIONS}")

print("\nTop 10 highest-value open-play actions:")
print(
    df_moves.nlargest(10, "xt_value")[
        ["minute", "player", "team", "start_x", "start_y", "end_x", "end_y", "xt_value"]
    ].to_string(index=False)
)