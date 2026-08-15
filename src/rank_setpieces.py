"""
Rank players by set-piece delivery value: for every completed attacking
corner/free-kick delivery, score it with the dedicated set-piece model
(train_setpiece.py) and sum per player.
"""
import os
import joblib
import pandas as pd

from train_setpiece import SETPIECE_FEATURES

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_rankings():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "setpieces.parquet"))
    df = df[(df["completed"] == True) & (df["is_attacking_delivery"] == True)].reset_index(drop=True)  # noqa: E712

    model = joblib.load(os.path.join(MODELS_DIR, "setpiece_model.pkl"))
    df["delivery_value"] = model.predict_proba(df[SETPIECE_FEATURES])[:, 1]

    tournament = (
        df.groupby(["player", "team"])
        .agg(total_delivery_value=("delivery_value", "sum"),
             deliveries=("delivery_value", "count"),
             actual_shots_created=("led_to_shot", "sum"),
             actual_goals=("led_to_goal", "sum"))
        .reset_index()
    )
    tournament["value_per_delivery"] = tournament["total_delivery_value"] / tournament["deliveries"]

    out_path = os.path.join(PROCESSED_DIR, "player_setpiece_tournament.csv")
    tournament.sort_values("total_delivery_value", ascending=False).to_csv(out_path, index=False)
    print(f"Wrote set-piece rankings ({len(tournament):,} players) -> {out_path}")
    return tournament


if __name__ == "__main__":
    tournament = build_rankings()
    print(f"\n=== Top 20 by TOTAL set-piece delivery value ===")
    print(tournament.sort_values("total_delivery_value", ascending=False).head(20)
          [["player", "team", "total_delivery_value", "deliveries",
            "actual_shots_created", "actual_goals"]]
          .to_string(index=False))