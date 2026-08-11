"""
Build an action-level dataset from raw StatsBomb event files.

For every Pass and Carry, we record:
  - start (x, y) and end (x, y) on a 120 x 80 pitch
  - whether the possession that contains this action ends in a goal
    within a short horizon (this is the training signal for xT)

Pitch coordinate system (StatsBomb): x in [0, 120], y in [0, 80].
Attacking direction is always left -> right for the team in possession,
so x=120 is the opponent's goal line.
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SHOT_HORIZON = 10  # how many events forward within the same possession we look for a shot


def load_match_events(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_match(events, match_id):
    """Return a list of action dicts for one match."""
    # Index events by their position in the list, and group indices by possession id.
    possession_to_indices = {}
    for i, e in enumerate(events):
        p = e.get("possession")
        possession_to_indices.setdefault(p, []).append(i)

    # For each possession, determine if it ends in a goal (shot outcome == 'Goal')
    possession_goal = {}
    for p, idxs in possession_to_indices.items():
        scored = False
        for i in idxs:
            e = events[i]
            if e["type"]["name"] == "Shot":
                outcome = e.get("shot", {}).get("outcome", {}).get("name")
                if outcome == "Goal":
                    scored = True
                    break
        possession_goal[p] = scored

    rows = []
    for i, e in enumerate(events):
        etype = e["type"]["name"]
        if etype not in ("Pass", "Carry"):
            continue

        loc = e.get("location")
        if etype == "Pass":
            sub = e.get("pass", {})
            end = sub.get("end_location")
            outcome = sub.get("outcome", {}).get("name")  # None means completed
            completed = outcome is None
            body_part = sub.get("body_part", {}).get("name")
            under_pressure = e.get("under_pressure", False)
        else:  # Carry
            sub = e.get("carry", {})
            end = sub.get("end_location")
            completed = True
            body_part = None
            under_pressure = e.get("under_pressure", False)

        if not loc or not end:
            continue

        possession = e.get("possession")
        led_to_goal = int(possession_goal.get(possession, False))

        rows.append({
            "match_id": match_id,
            "event_id": e.get("id"),
            "index": e.get("index"),
            "period": e.get("period"),
            "minute": e.get("minute"),
            "possession": possession,
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "position": e.get("position", {}).get("name"),
            "type": etype,
            "start_x": loc[0],
            "start_y": loc[1],
            "end_x": end[0],
            "end_y": end[1],
            "completed": completed,
            "under_pressure": bool(under_pressure),
            "led_to_goal": led_to_goal,
        })

    return rows


def build(limit=None, out_name="actions.parquet"):
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    if limit:
        files = files[:limit]

    all_rows = []
    for path in tqdm(files, desc="Parsing matches"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        events = load_match_events(path)
        all_rows.extend(process_match(events, match_id))

    df = pd.DataFrame(all_rows)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} actions from {len(files)} matches -> {out_path}")
    print(f"Goal-possession rate: {df['led_to_goal'].mean():.4f}")
    print(df["type"].value_counts())
    return df


if __name__ == "__main__":
    build()