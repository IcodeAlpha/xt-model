"""
Rank every player in the dataset by threat added (xT), both as a raw
tournament total and normalized per 90 minutes played (the fair way to
compare players who played different amounts of football).

Uses the LightGBM model for scoring -- its raw probabilities are well
calibrated (see README), unlike the deep model's pos_weight-skewed outputs,
which makes it the right choice when the actual xt_added *magnitude* matters
(as it does here), not just its use for ranking within a single play.
"""
import os
import joblib
import numpy as np
import pandas as pd

from features import add_features
from train_baseline import END_STATE_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MIN_MINUTES_FOR_PER90 = 90  # below this, per-90 rates are too noisy to trust


def score_actions(df, model):
    start_cols = {
        "end_x": df["start_x"], "end_y": df["start_y"],
        "end_dist_goal": df["start_dist_goal"], "end_angle_goal": df["start_angle_goal"],
        "is_pass": df["is_pass"], "is_carry": df["is_carry"],
        "completed_int": df["completed_int"], "under_pressure_int": df["under_pressure_int"],
    }
    end_cols = {
        "end_x": df["end_x"], "end_y": df["end_y"],
        "end_dist_goal": df["end_dist_goal"], "end_angle_goal": df["end_angle_goal"],
        "is_pass": df["is_pass"], "is_carry": df["is_carry"],
        "completed_int": df["completed_int"], "under_pressure_int": df["under_pressure_int"],
    }
    X_start = pd.DataFrame(start_cols)[END_STATE_FEATURES]
    X_end = pd.DataFrame(end_cols)[END_STATE_FEATURES]
    p_start = model.predict_proba(X_start)[:, 1]
    p_end = model.predict_proba(X_end)[:, 1]
    return p_end - p_start


def build_rankings():
    actions = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    actions = actions[actions["completed"] == True].reset_index(drop=True)  # noqa: E712
    actions = add_features(actions)

    model = joblib.load(os.path.join(MODELS_DIR, "lgbm_baseline.pkl"))
    actions["xt_added"] = score_actions(actions, model)

    # Per-player, per-match aggregation first (so "each single one, from each match"
    # is genuinely available), then rolled up to a tournament total.
    per_match = (
        actions.groupby(["match_id", "player", "team", "position"], dropna=False)
        .agg(xt_added=("xt_added", "sum"), actions=("xt_added", "count"))
        .reset_index()
    )

    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)  # fallback if a name didn't match
    per_match["xt_per90"] = per_match["xt_added"] / per_match["minutes_played"].clip(lower=1) * 90

    per_match_path = os.path.join(PROCESSED_DIR, "player_xt_per_match.csv")
    per_match.sort_values(["match_id", "xt_added"], ascending=[True, False]).to_csv(per_match_path, index=False)
    print(f"Wrote per-match player rankings ({len(per_match):,} rows) -> {per_match_path}")

    # Tournament-wide rollup: every player, all matches combined
    tournament = (
        per_match.groupby(["player", "team"])
        .agg(
            total_xt_added=("xt_added", "sum"),
            total_actions=("actions", "sum"),
            total_minutes=("minutes_played", "sum"),
            matches_played=("match_id", "nunique"),
        )
        .reset_index()
    )
    tournament["xt_per90"] = tournament["total_xt_added"] / tournament["total_minutes"].clip(lower=1) * 90

    tournament_path = os.path.join(PROCESSED_DIR, "player_xt_tournament.csv")
    tournament.sort_values("total_xt_added", ascending=False).to_csv(tournament_path, index=False)
    print(f"Wrote tournament-wide player rankings ({len(tournament):,} players) -> {tournament_path}")

    return per_match, tournament


if __name__ == "__main__":
    per_match, tournament = build_rankings()

    print(f"\n=== Top 25 by TOTAL threat added (whole tournament, all {tournament.shape[0]} players) ===")
    print(tournament.sort_values("total_xt_added", ascending=False).head(25)
          [["player", "team", "total_xt_added", "total_minutes", "matches_played", "xt_per90"]]
          .to_string(index=False))

    qualified = tournament[tournament["total_minutes"] >= MIN_MINUTES_FOR_PER90]
    print(f"\n=== Top 25 by xT PER 90 (min {MIN_MINUTES_FOR_PER90} minutes played, "
          f"{len(qualified)}/{len(tournament)} players qualify) ===")
    print(qualified.sort_values("xt_per90", ascending=False).head(25)
          [["player", "team", "xt_per90", "total_xt_added", "total_minutes", "matches_played"]]
          .to_string(index=False))