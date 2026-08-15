"""
Defensive Value: how much attacking danger did a defensive action deny?

Rather than training a brand-new classifier (there's no clean ground-truth
label for "how dangerous was the situation this tackle prevented" -- it's a
counterfactual), this reuses the ALREADY-TRAINED xT surface model from
train_baseline.py. The intuition: our xT model already learned "how
dangerous is a completed pass/carry ending at this pitch location." A
successful defensive action at a given spot denies exactly that amount of
danger to the opponent.

Coordinate note: StatsBomb records each event's (x, y) in the ACTING
player's own attacking-direction frame -- so for a defender, x near 0 means
"near MY OWN goal," the most dangerous place to concede. We mirror
x -> 120 - x before scoring, so the trained model (which learned danger
relative to x=120 = opponent's goal) reads it as "how dangerous would this
spot be for an attacker," which is exactly the danger the defender denied.

Included action types (successful only):
  - Interception (outcome: Won / Success In Play / Success Out)
  - Tackle (Duel type=Tackle, same success outcomes)
  - Block (excluding "offensive" blocks, which aren't defensive actions)
  - Clearance (presence of the event = success, no outcome field)
  - Ball Recovery (recovery_failure == False)
"""
import json
import os
import glob
import joblib
import pandas as pd
from tqdm import tqdm

from features import angle_to_goal
from train_baseline import END_STATE_FEATURES

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

SUCCESS_OUTCOMES = {"Won", "Success In Play", "Success Out"}


def process_match_defensive(events, match_id):
    rows = []
    for e in events:
        t = e["type"]["name"]
        loc = e.get("location")
        if not loc:
            continue

        is_defensive_success = False

        if t == "Interception":
            outcome = e.get("interception", {}).get("outcome", {}).get("name")
            is_defensive_success = outcome in SUCCESS_OUTCOMES

        elif t == "Duel":
            d = e.get("duel", {})
            if d.get("type", {}).get("name") == "Tackle":
                outcome = d.get("outcome", {}).get("name")
                is_defensive_success = outcome in SUCCESS_OUTCOMES

        elif t == "Block":
            is_defensive_success = not e.get("block", {}).get("offensive", False)

        elif t == "Clearance":
            is_defensive_success = True  # presence of the event = success

        elif t == "Ball Recovery":
            is_defensive_success = not e.get("ball_recovery", {}).get("recovery_failure", False)

        if not is_defensive_success:
            continue

        rows.append({
            "match_id": match_id,
            "minute": e.get("minute"),
            "type": t,
            "player": e.get("player", {}).get("name"),
            "team": e.get("team", {}).get("name"),
            "x": loc[0], "y": loc[1],
        })
    return rows


def build():
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Finding defensive actions"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match_defensive(events, match_id))

    df = pd.DataFrame(all_rows)
    print(f"{len(df):,} successful defensive actions across {len(files)} matches")
    print(df["type"].value_counts())

    # Mirror x so the trained xT model reads "distance to the goal being defended"
    mirror_x = 120 - df["x"]
    dist_goal = ((120 - mirror_x) ** 2 + (40 - df["y"]) ** 2) ** 0.5
    angle_goal = angle_to_goal(mirror_x.values, df["y"].values)

    X = pd.DataFrame({
        "end_x": mirror_x, "end_y": df["y"],
        "end_dist_goal": dist_goal, "end_angle_goal": angle_goal,
        "is_pass": 1, "is_carry": 0, "completed_int": 1, "under_pressure_int": 0,
    })[END_STATE_FEATURES]

    model = joblib.load(os.path.join(MODELS_DIR, "lgbm_baseline.pkl"))
    df["danger_denied"] = model.predict_proba(X)[:, 1]

    out_path = os.path.join(PROCESSED_DIR, "defensive_actions.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total danger denied across the tournament: {df['danger_denied'].sum():.2f}")
    return df


def build_rankings(df):
    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(danger_denied=("danger_denied", "sum"), defensive_actions=("danger_denied", "count"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(total_danger_denied=("danger_denied", "sum"),
             defensive_actions=("defensive_actions", "sum"),
             total_minutes=("minutes_played", "sum"),
             matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["denied_per90"] = (
        tournament["total_danger_denied"] / tournament["total_minutes"].clip(lower=1) * 90
    )

    out_path = os.path.join(PROCESSED_DIR, "player_defensive_tournament.csv")
    tournament.sort_values("total_danger_denied", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote defensive rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    df = build()
    tournament = build_rankings(df)

    print(f"\n=== Top 20 by TOTAL danger denied ===")
    print(tournament.sort_values("total_danger_denied", ascending=False).head(20)
          [["player", "team", "total_danger_denied", "defensive_actions", "total_minutes", "matches_played"]]
          .to_string(index=False))