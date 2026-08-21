"""
Counter-pressing value: did applying pressure actually win the ball back?

StatsBomb tags every Pressure event (a player closing down an opponent on
the ball) but doesn't say whether it worked. We detect success ourselves:
a pressure "succeeds" if the pressuring team's possession_team appears on
any event within 5 seconds of the press (i.e. they regained the ball
shortly after, rather than the opponent playing through/around it).

This is the same logic real pressing metrics ("pressures leading to
regains") use in practice.
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

from features import angle_to_goal

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

REGAIN_WINDOW_SECONDS = 5.0


def _time_seconds(e):
    return (e.get("minute") or 0) * 60 + (e.get("second") or 0)


def process_match(events, match_id):
    rows = []
    n = len(events)
    for i, e in enumerate(events):
        if e["type"]["name"] != "Pressure":
            continue
        loc = e.get("location")
        if not loc:
            continue

        team = e.get("team", {}).get("name")
        period = e.get("period")
        t0 = _time_seconds(e)

        success = False
        for j in range(i + 1, min(i + 60, n)):  # cap lookahead for speed
            e2 = events[j]
            if e2.get("period") != period:
                break
            if _time_seconds(e2) - t0 > REGAIN_WINDOW_SECONDS:
                break
            if e2.get("possession_team", {}).get("name") == team:
                success = True
                break

        rows.append({
            "match_id": match_id,
            "minute": e.get("minute"),
            "player": e.get("player", {}).get("name"),
            "team": team,
            "x": loc[0], "y": loc[1],
            "duration": e.get("duration"),
            "success": int(success),
        })
    return rows


def build(out_name="pressures.parquet"):
    # Only the 64 official World Cup 2022 matches -- data/raw/events also
    # contains a few other matches (2019 CL Final, Euro 2024 Final, 1970 WC
    # Final) downloaded separately for sample CSV testing. Those must NOT
    # leak into this project's trained models/rankings.
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matches", "43_106.json"),
              encoding="utf-8") as f:
        wc2022_ids = {m["match_id"] for m in json.load(f)}

    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    files = [p for p in files if int(os.path.basename(p).replace(".json", "")) in wc2022_ids]

    all_rows = []
    for path in tqdm(files, desc="Detecting pressure regains"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match(events, match_id))

    df = pd.DataFrame(all_rows)
    # mirror x: pressure happens in the pressuring player's own attacking frame,
    # so x near 120 means deep in the opponent's half -- the danger of a HIGH
    # press. Mirror so "dist to goal" reads distance to the OPPONENT's goal
    # (the goal a successful high press threatens), consistent with how
    # build_defensive.py mirrors for the goal being defended.
    df["dist_opp_goal"] = ((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2) ** 0.5
    df["angle_opp_goal"] = angle_to_goal(df["x"].values, df["y"].values)
    df["duration"] = df["duration"].fillna(df["duration"].median())

    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} pressures from {len(files)} matches -> {out_path}")
    print(f"Success rate (regain within {REGAIN_WINDOW_SECONDS}s): {df['success'].mean():.4f}")
    return df


if __name__ == "__main__":
    build()