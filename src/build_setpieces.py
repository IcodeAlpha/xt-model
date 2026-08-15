"""
Set-piece deliveries (corners and free kicks) as their own dataset.

These behave differently from open-play passes -- delivery quality depends
on trajectory/height and how many attackers are in the box, not the same
distance/angle-to-goal relationship that drives open-play xT. Worth a
dedicated model rather than reusing the open-play xT model, the way
Phase 3's defensive metric did.

Target: did this delivery directly lead to a shot (StatsBomb's
shot_assist / goal_assist flags -- see Phase 2's build_xa.py for why both
need checking, not just shot_assist).
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

from features import angle_to_goal

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SET_PIECE_TYPES = {"Corner", "Free Kick"}


def process_match(events, match_id):
    rows = []
    for e in events:
        if e["type"]["name"] != "Pass":
            continue
        p = e.get("pass", {})
        pass_type = p.get("type", {}).get("name")
        if pass_type not in SET_PIECE_TYPES:
            continue

        loc = e.get("location")
        end = p.get("end_location")
        if not loc or not end:
            continue

        outcome = p.get("outcome", {}).get("name")
        completed = outcome is None
        led_to_shot = bool(p.get("shot_assist") or p.get("goal_assist"))
        led_to_goal = bool(p.get("goal_assist"))

        rows.append({
            "match_id": match_id,
            "minute": e.get("minute"),
            "player": e.get("player", {}).get("name"),
            "team": e.get("team", {}).get("name"),
            "set_piece_type": pass_type,
            "start_x": loc[0], "start_y": loc[1],
            "end_x": end[0], "end_y": end[1],
            "height": p.get("height", {}).get("name"),
            "body_part": p.get("body_part", {}).get("name"),
            "under_pressure": bool(e.get("under_pressure", False)),
            "completed": completed,
            "led_to_shot": int(led_to_shot),
            "led_to_goal": int(led_to_goal),
        })
    return rows


def build(out_name="setpieces.parquet"):
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Finding set pieces"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match(events, match_id))

    df = pd.DataFrame(all_rows)
    df["end_dist_goal"] = ((120 - df["end_x"]) ** 2 + (40 - df["end_y"]) ** 2) ** 0.5
    df["end_angle_goal"] = angle_to_goal(df["end_x"].values, df["end_y"].values)
    df["is_corner"] = (df["set_piece_type"] == "Corner").astype(int)
    df["is_high"] = (df["height"] == "High Pass").astype(int)
    df["is_ground"] = (df["height"] == "Ground Pass").astype(int)
    df["under_pressure_int"] = df["under_pressure"].astype(int)

    # StatsBomb's "Free Kick" pass type covers EVERY free kick, including
    # mundane short restarts taken from deep/midfield after a foul -- those
    # aren't attacking deliveries and dominate the dataset (most free kicks
    # are not taken near the box). Mixing them in makes "led to shot" almost
    # perfectly separable by end-location alone (deep restarts trivially
    # can't reach the box), which inflates AUC without teaching anything
    # about delivery quality. Corners are always genuine attacking
    # deliveries by definition; for free kicks we keep only ones taken from
    # the attacking third (start_x >= 80), which is the standard definition
    # analysts use for "attacking free kick."
    df["is_attacking_delivery"] = df["is_corner"].astype(bool) | (df["start_x"] >= 80)

    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} set-piece deliveries from {len(files)} matches -> {out_path}")
    print(f"Attacking deliveries (corners + final-third free kicks): {df['is_attacking_delivery'].sum():,}")
    print(f"Led to shot rate (attacking only): "
          f"{df[df['is_attacking_delivery']]['led_to_shot'].mean():.4f}")
    print(df["set_piece_type"].value_counts())
    return df


if __name__ == "__main__":
    build()