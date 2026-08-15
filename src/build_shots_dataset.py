import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def process_match_shots(events, match_id):
    rows = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        shot = e.get("shot", {})
        loc = e.get("location")
        if not loc:
            continue
        outcome = shot.get("outcome", {}).get("name")
        rows.append({
            "match_id": match_id,
            "event_id": e.get("id"),
            "minute": e.get("minute"),
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "x": loc[0], "y": loc[1],
            "body_part": shot.get("body_part", {}).get("name"),
            "technique": shot.get("technique", {}).get("name"),
            "shot_type": shot.get("type", {}).get("name"),
            "under_pressure": bool(e.get("under_pressure", False)),
            "first_time": bool(shot.get("first_time", False)),
            "key_pass_id": shot.get("key_pass_id"),
            "statsbomb_xg": shot.get("statsbomb_xg"),
            "is_goal": int(outcome == "Goal"),
        })
    return rows


def build(out_name="shots.parquet"):
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Parsing shots"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match_shots(events, match_id))

    df = pd.DataFrame(all_rows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} shots from {len(files)} matches -> {out_path}")
    return df


if __name__ == "__main__":
    build()