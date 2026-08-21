"""
Rank players by dribble value added: actual successful take-ons minus the
model's expected completions, given location. Rewards genuinely difficult
dribbles completed, not just high-volume dribblers in easy space.
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MIN_ATTEMPTS = 5


def build_rankings():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "dribbles_scored.parquet"))
    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(attempts=("completed", "count"), actual_completions=("completed", "sum"),
             expected_completions=("expected_completion", "sum"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(attempts=("attempts", "sum"), actual_completions=("actual_completions", "sum"),
             expected_completions=("expected_completions", "sum"),
             total_minutes=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["dribble_value_added"] = tournament["actual_completions"] - tournament["expected_completions"]
    tournament["success_rate"] = tournament["actual_completions"] / tournament["attempts"].clip(lower=1)

    out_path = os.path.join(PROCESSED_DIR, "player_dribble_tournament.csv")
    tournament.sort_values("dribble_value_added", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote dribble rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    qualified = tournament[tournament["attempts"] >= MIN_ATTEMPTS]
    print(f"\n=== Top 20 by dribble value added, min {MIN_ATTEMPTS} attempts "
          f"({len(qualified)}/{len(tournament)} qualify) ===")
    print(qualified.sort_values("dribble_value_added", ascending=False).head(20)
          [["player", "team", "dribble_value_added", "attempts", "actual_completions",
            "success_rate", "matches_played"]]
          .to_string(index=False))