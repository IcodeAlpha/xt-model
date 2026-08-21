"""
Rank players by press-resistance: on passes taken UNDER PRESSURE only,
actual completions minus the model's location-based expected completions.
Positive means a player retains the ball under pressure more often than
their pass geometry alone would predict -- composure, not just easy passing.
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MIN_PRESSURED_PASSES = 10  # below this, the rate is too noisy to trust


def build_rankings():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "passes_completion_scored.parquet"))
    df = df[df["under_pressure_int"] == 1].reset_index(drop=True)

    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(pressured_passes=("completed_int", "count"),
             actual_completions=("completed_int", "sum"),
             expected_completions=("expected_completion", "sum"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(pressured_passes=("pressured_passes", "sum"),
             actual_completions=("actual_completions", "sum"),
             expected_completions=("expected_completions", "sum"),
             total_minutes=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["press_resistance"] = tournament["actual_completions"] - tournament["expected_completions"]
    tournament["completion_rate_under_pressure"] = (
        tournament["actual_completions"] / tournament["pressured_passes"].clip(lower=1)
    )

    out_path = os.path.join(PROCESSED_DIR, "player_press_resistance_tournament.csv")
    tournament.sort_values("press_resistance", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote press-resistance rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    qualified = tournament[tournament["pressured_passes"] >= MIN_PRESSURED_PASSES]
    print(f"\n=== Top 20 by press-resistance, min {MIN_PRESSURED_PASSES} pressured passes "
          f"({len(qualified)}/{len(tournament)} qualify) ===")
    print(qualified.sort_values("press_resistance", ascending=False).head(20)
          [["player", "team", "press_resistance", "pressured_passes",
            "completion_rate_under_pressure", "matches_played"]]
          .to_string(index=False))