"""
Progressive actions: a widely-used analytics metric independent of the xT
model entirely. A completed pass or carry counts as PROGRESSIVE if it either:

  (a) moves the ball at least 25% closer to the opponent's goal
      (using straight-line distance to goal), or
  (b) ends inside the opponent's penalty box, regardless of distance covered
      (a short cutback into the box is progressive even if the % isn't huge)

This is the standard definition used across public football analytics
(fbref/Opta-style), not something specific to this project's xT model --
it's a useful sanity-check metric precisely because it's independent of any
model's judgment calls.
"""
import os
import pandas as pd

from features import add_features

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PROGRESS_THRESHOLD = 0.25  # min fraction of distance-to-goal closed
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102.0, 18.0, 62.0  # opponent's penalty box


def flag_progressive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    in_box = (df["end_x"] >= BOX_X_MIN) & (df["end_y"].between(BOX_Y_MIN, BOX_Y_MAX))
    closed_enough = df["progress_frac"] >= PROGRESS_THRESHOLD
    df["is_progressive"] = closed_enough | in_box
    return df


def build_rankings():
    actions = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    actions = actions[actions["completed"] == True].reset_index(drop=True)  # noqa: E712
    actions = add_features(actions)
    actions = flag_progressive(actions)

    per_match = (
        actions.groupby(["match_id", "player", "team"], dropna=False)
        .agg(
            progressive_actions=("is_progressive", "sum"),
            progressive_passes=("is_progressive", lambda s: (s & (actions.loc[s.index, "type"] == "Pass")).sum()),
            progressive_carries=("is_progressive", lambda s: (s & (actions.loc[s.index, "type"] == "Carry")).sum()),
            total_actions=("is_progressive", "count"),
        )
        .reset_index()
    )

    minutes = pd.read_parquet(os.path.join(PROCESSED_DIR, "player_minutes.parquet"))
    per_match = per_match.merge(minutes[["match_id", "player", "minutes_played"]],
                                 on=["match_id", "player"], how="left")
    per_match["minutes_played"] = per_match["minutes_played"].fillna(90)

    tournament = (
        per_match.groupby(["player", "team"])
        .agg(
            progressive_actions=("progressive_actions", "sum"),
            progressive_passes=("progressive_passes", "sum"),
            progressive_carries=("progressive_carries", "sum"),
            total_minutes=("minutes_played", "sum"),
            matches_played=("match_id", "nunique"),
        )
        .reset_index()
    )
    tournament["progressive_per90"] = (
        tournament["progressive_actions"] / tournament["total_minutes"].clip(lower=1) * 90
    )

    out_path = os.path.join(PROCESSED_DIR, "player_progressive.csv")
    tournament.sort_values("progressive_actions", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote progressive-actions rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    print(f"\n=== Top 20 by TOTAL progressive actions ===")
    print(tournament.sort_values("progressive_actions", ascending=False).head(20)
          [["player", "team", "progressive_actions", "progressive_passes",
            "progressive_carries", "total_minutes", "matches_played"]]
          .to_string(index=False))

    qualified = tournament[tournament["total_minutes"] >= 300]
    print(f"\n=== Top 20 by progressive actions PER 90 (min 300 minutes) ===")
    print(qualified.sort_values("progressive_per90", ascending=False).head(20)
          [["player", "team", "progressive_per90", "progressive_actions", "total_minutes"]]
          .to_string(index=False))