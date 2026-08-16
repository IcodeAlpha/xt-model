"""
Game-state-aware xT: does the SAME pass carry different value depending on
score and time remaining? Retrains the LightGBM baseline with score_diff
and minutes_remaining added to the feature set, and compares CV AUC against
the original location-only model to see whether game state adds real
predictive signal or just noise.
"""
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from features import add_features, TARGET
from train_baseline import END_STATE_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

GAMESTATE_FEATURES = END_STATE_FEATURES + ["score_diff", "minutes_remaining"]


def load_data():
    actions = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    actions = actions[actions["completed"] == True].reset_index(drop=True)  # noqa: E712
    actions = add_features(actions)

    gamestate = pd.read_parquet(os.path.join(PROCESSED_DIR, "gamestate.parquet"))
    df = actions.merge(gamestate, on=["match_id", "index"], how="inner")
    print(f"{len(df):,} actions merged with game state "
          f"(started with {len(actions):,} completed actions)")
    return df


def cross_validate(df, features, n_folds=5, seed=42, label=""):
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)
    folds = np.array_split(match_ids, n_folds)

    aucs = []
    for i in range(n_folds):
        test_ids = set(folds[i])
        train_ids = set(match_ids) - test_ids
        tr = df[df["match_id"].isin(train_ids)]
        te = df[df["match_id"].isin(test_ids)]

        m = LGBMClassifier(n_estimators=300, learning_rate=0.02, max_depth=3, num_leaves=7,
                            min_child_samples=300, subsample=0.7, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0,
                            random_state=seed, verbose=-1)
        m.fit(tr[features], tr[TARGET])
        p = m.predict_proba(te[features])[:, 1]
        auc = roc_auc_score(te[TARGET], p)
        aucs.append(auc)

    aucs = np.array(aucs)
    print(f"[{label}] CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = load_data()

    print("\n--- Baseline (location only, for comparison) ---")
    baseline_aucs = cross_validate(df, END_STATE_FEATURES, label="location-only")

    print("\n--- With game state (score_diff, minutes_remaining) ---")
    gamestate_aucs = cross_validate(df, GAMESTATE_FEATURES, label="with-gamestate")

    diff = gamestate_aucs.mean() - baseline_aucs.mean()
    print(f"\nDifference: {diff:+.4f} "
          f"({'game state helps' if diff > 0.005 else 'no meaningful difference' if abs(diff) <= 0.005 else 'game state hurts'})")

    # Fit final model on all data, save it
    model = LGBMClassifier(n_estimators=300, learning_rate=0.02, max_depth=3, num_leaves=7,
                            min_child_samples=300, subsample=0.7, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0,
                            random_state=42, verbose=-1)
    model.fit(df[GAMESTATE_FEATURES], df[TARGET])
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "lgbm_gamestate.pkl"))
    print("Saved model to models/lgbm_gamestate.pkl")

    # Feature importance -- how much does the model actually lean on game state?
    importances = pd.Series(model.feature_importances_, index=GAMESTATE_FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    # Concrete illustration: same pass (halfway line -> edge of box), different game states
    print("\n--- Same pass, different game states ---")
    example = pd.DataFrame({
        "end_x": [110]*4, "end_y": [40]*4,
        "end_dist_goal": [np.sqrt((120-110)**2+0**2)]*4,
        "end_angle_goal": [0.6]*4,
        "is_pass": [1]*4, "is_carry": [0]*4, "completed_int": [1]*4, "under_pressure_int": [0]*4,
        "score_diff": [0, 0, -1, -1],
        "minutes_remaining": [80, 5, 80, 5],
    })
    example["p_score"] = model.predict_proba(example[GAMESTATE_FEATURES])[:, 1]
    example["scenario"] = ["0-0, early", "0-0, injury time", "trailing, early", "trailing, injury time"]
    print(example[["scenario", "p_score"]].to_string(index=False))


if __name__ == "__main__":
    main()