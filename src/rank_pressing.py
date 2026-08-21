"""
Rank players by pressing value added: actual successful pressures minus the
model's expected success count, summed per player. Positive means a player
wins the ball back more often than the model expects given where/how they
pressed; negative means they apply a lot of pressure that doesn't convert.
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def build_rankings():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "pressures_scored.parquet"))
    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(pressures=("success", "count"), actual_regains=("success", "sum"),
             expected_regains=("expected_success", "sum"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(pressures=("pressures", "sum"), actual_regains=("actual_regains", "sum"),
             expected_regains=("expected_regains", "sum"),
             total_minutes=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["pressing_value_added"] = tournament["actual_regains"] - tournament["expected_regains"]
    tournament["pressing_value_per90"] = (
        tournament["pressing_value_added"] / tournament["total_minutes"].clip(lower=1) * 90
    )

    out_path = os.path.join(PROCESSED_DIR, "player_pressing_tournament.csv")
    tournament.sort_values("pressing_value_added", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote pressing rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    print(f"\n=== Top 20 by pressing value added (actual regains - expected) ===")
    print(tournament.sort_values("pressing_value_added", ascending=False).head(20)
          [["player", "team", "pressing_value_added", "pressures", "actual_regains",
            "expected_regains", "matches_played"]]
          .to_string(index=False))