"""
Rank players by aerial duels won above expected (location-based baseline).
Also reports simple win rate for context, since the model's location signal
is weak -- raw win rate among contested duels is arguably the more
intuitive number here, with value-added as the "controlled for difficulty"
version.
"""
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MIN_DUELS = 5


def build_rankings():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "aerials_scored.parquet"))
    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))

    per_match = (
        df.groupby(["match_id", "player", "team"])
        .agg(duels=("won", "count"), actual_wins=("won", "sum"),
             expected_wins=("expected_win_prob", "sum"))
        .reset_index()
    )
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(duels=("duels", "sum"), actual_wins=("actual_wins", "sum"),
             expected_wins=("expected_wins", "sum"),
             total_minutes=("minutes_played", "sum"), matches_played=("match_id", "nunique"))
        .reset_index()
    )
    tournament["aerial_value_added"] = tournament["actual_wins"] - tournament["expected_wins"]
    tournament["win_rate"] = tournament["actual_wins"] / tournament["duels"].clip(lower=1)

    out_path = os.path.join(PROCESSED_DIR, "player_aerial_tournament.csv")
    tournament.sort_values("aerial_value_added", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote aerial rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    qualified = tournament[tournament["duels"] >= MIN_DUELS]
    print(f"\n=== Top 20 by aerial value added, min {MIN_DUELS} duels "
          f"({len(qualified)}/{len(tournament)} qualify) ===")
    print(qualified.sort_values("aerial_value_added", ascending=False).head(20)
          [["player", "team", "aerial_value_added", "duels", "actual_wins", "win_rate", "matches_played"]]
          .to_string(index=False))