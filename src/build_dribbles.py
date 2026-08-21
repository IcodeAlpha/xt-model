"""
Dribble / take-on dataset. StatsBomb tags every Dribble event with a clean
Complete/Incomplete outcome directly on the dribbling player -- no
resolution needed, unlike aerials.
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def process_match(events, match_id):
    rows = []
    for e in events:
        if e["type"]["name"] != "Dribble":
            continue
        loc = e.get("location")
        if not loc:
            continue
        outcome = e.get("dribble", {}).get("outcome", {}).get("name")
        rows.append({
            "match_id": match_id, "minute": e.get("minute"),
            "player": e.get("player", {}).get("name"), "team": e.get("team", {}).get("name"),
            "x": loc[0], "y": loc[1],
            "under_pressure": bool(e.get("under_pressure", False)),
            "completed": int(outcome == "Complete"),
        })
    return rows


def build(out_name="dribbles.parquet"):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matches", "43_106.json"),
              encoding="utf-8") as f:
        wc2022_ids = {m["match_id"] for m in json.load(f)}

    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    files = [p for p in files if int(os.path.basename(p).replace(".json", "")) in wc2022_ids]

    all_rows = []
    for path in tqdm(files, desc="Finding dribbles"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match(events, match_id))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} dribble attempts from {len(files)} matches -> {out_path}")
    print(f"Success rate: {df['completed'].mean():.4f}")
    return df


if __name__ == "__main__":
    build()