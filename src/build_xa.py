"""
Expected Assists (xA): for every pass that StatsBomb tags as directly
assisting a shot (pass.shot_assist == True, linked via assisted_shot_id),
credit the passer with the trained xG model's predicted probability for
that resulting shot.

This differs from real assists in the same way xG differs from goals: it
values the QUALITY of the chance created, regardless of whether the shot
was actually scored. A player who sets up three great chances that all get
saved should still rank highly -- that's the point of the metric.
"""
import json
import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import angle_to_goal
from train_xg import XG_FEATURES

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def process_match_key_passes(events, match_id):
    # index all shots in this match by their event id
    shots_by_id = {}
    for e in events:
        if e["type"]["name"] == "Shot":
            shots_by_id[e["id"]] = e

    rows = []
    for e in events:
        if e["type"]["name"] != "Pass":
            continue
        p = e.get("pass", {})
        # StatsBomb uses these as MUTUALLY EXCLUSIVE flags: shot_assist for
        # passes leading to a non-scoring shot, goal_assist for passes
        # leading directly to a goal. Checking only shot_assist silently
        # excludes every pass that actually produced a goal.
        if not (p.get("shot_assist") or p.get("goal_assist")):
            continue
        shot_id = p.get("assisted_shot_id")
        shot_event = shots_by_id.get(shot_id)
        if shot_event is None:
            continue

        shot = shot_event.get("shot", {})
        loc = shot_event.get("location")
        if not loc:
            continue

        rows.append({
            "match_id": match_id,
            "minute": e.get("minute"),
            "player": e.get("player", {}).get("name"),
            "team": e.get("team", {}).get("name"),
            "x": loc[0], "y": loc[1],
            "body_part": shot.get("body_part", {}).get("name"),
            "technique": shot.get("technique", {}).get("name"),
            "first_time": bool(shot.get("first_time", False)),
            "under_pressure": bool(shot_event.get("under_pressure", False)),
            "resulted_in_goal": int(shot.get("outcome", {}).get("name") == "Goal"),
            "statsbomb_xg": shot.get("statsbomb_xg"),
        })
    return rows


def build():
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Finding key passes"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match_key_passes(events, match_id))

    df = pd.DataFrame(all_rows)
    print(f"{len(df):,} key passes (shot-assisting passes) found across {len(files)} matches")

    df["dist_goal"] = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)
    df["angle_goal"] = angle_to_goal(df["x"].values, df["y"].values)
    df["is_foot"] = df["body_part"].isin(["Right Foot", "Left Foot"]).astype(int)
    df["is_head"] = (df["body_part"] == "Head").astype(int)
    df["first_time_int"] = df["first_time"].astype(int)
    df["under_pressure_int"] = df["under_pressure"].astype(int)

    model = joblib.load(os.path.join(MODELS_DIR, "xg_model.pkl"))
    df["xa_value"] = model.predict_proba(df[XG_FEATURES])[:, 1]

    out_path = os.path.join(PROCESSED_DIR, "key_passes.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total xA across the tournament: {df['xa_value'].sum():.2f}  "
          f"(vs {df['resulted_in_goal'].sum()} actual assists that were scored)")
    return df


def build_rankings(df):
    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(xa=("xa_value", "sum"), key_passes=("xa_value", "count"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(total_xa=("xa", "sum"), key_passes=("key_passes", "sum"),
             total_minutes=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["xa_per90"] = tournament["total_xa"] / tournament["total_minutes"].clip(lower=1) * 90

    out_path = os.path.join(PROCESSED_DIR, "player_xa_tournament.csv")
    tournament.sort_values("total_xa", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote xA rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    df = build()
    tournament = build_rankings(df)

    print(f"\n=== Top 20 by TOTAL Expected Assists ===")
    print(tournament.sort_values("total_xa", ascending=False).head(20)
          [["player", "team", "total_xa", "key_passes", "total_minutes", "matches_played"]]
          .to_string(index=False))