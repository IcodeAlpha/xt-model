"""
Press-resistance: how much better does a player complete passes UNDER
PRESSURE than a location-based model expects?

Trains a single model predicting P(pass completes) from location/distance
context (deliberately NOT including under_pressure as a feature -- the
model should learn general pass difficulty by location, then we compare
actual vs expected specifically on the under-pressure subset). This isolates
"composure when closed down" as its own skill, separate from the general xT
value of the pass itself.
"""
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from features import add_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Deliberately location/distance only -- NOT under_pressure_int -- so the
# model reflects "how hard is this pass by geometry alone," and we can then
# see how much pressure specifically knocks players off that baseline.
COMPLETION_FEATURES = ["start_x", "start_y", "end_x", "end_y",
                        "start_dist_goal", "end_dist_goal", "move_dist"]


def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = df[df["type"] == "Pass"].reset_index(drop=True)  # carries have no completion outcome
    df = add_features(df)
    return df


def cross_validate(df, n_folds=5, seed=42):
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

        m = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=4, num_leaves=15,
                            min_child_samples=100, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[COMPLETION_FEATURES], tr["completed_int"])
        p = m.predict_proba(te[COMPLETION_FEATURES])[:, 1]
        auc = roc_auc_score(te["completed_int"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = load_data()
    print(f"{len(df):,} passes, completion rate {df['completed_int'].mean():.4f}")
    print(f"Completion rate under pressure: {df[df['under_pressure_int']==1]['completed_int'].mean():.4f}")
    print(f"Completion rate NOT under pressure: {df[df['under_pressure_int']==0]['completed_int'].mean():.4f}")

    print("\nRunning 5-fold cross-validation...")
    cross_validate(df)

    model = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=4, num_leaves=15,
                            min_child_samples=100, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[COMPLETION_FEATURES], df["completed_int"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "completion_model.pkl"))
    print("Saved model to models/completion_model.pkl")

    df["expected_completion"] = model.predict_proba(df[COMPLETION_FEATURES])[:, 1]
    out_path = os.path.join(PROCESSED_DIR, "passes_completion_scored.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote scored passes -> {out_path}")
    return df


if __name__ == "__main__":
    main()