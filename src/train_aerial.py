"""
Aerial duel win probability, location-weighted. Since we don't know a
row's attacking direction, use distance to the NEAREST goal (a proxy for
"is this happening in a crowded box," which is where most aerials cluster
around set pieces) rather than a directional distance-to-goal.
"""
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

AERIAL_FEATURES = ["x", "y", "dist_nearest_goal", "dist_from_center_y"]


def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "aerials.parquet"))
    dist_x0 = np.sqrt(df["x"] ** 2 + (40 - df["y"]) ** 2)
    dist_x120 = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)
    df["dist_nearest_goal"] = np.minimum(dist_x0, dist_x120)
    df["dist_from_center_y"] = (df["y"] - 40).abs()
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

        m = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=50, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[AERIAL_FEATURES], tr["won"])
        p = m.predict_proba(te[AERIAL_FEATURES])[:, 1]
        auc = roc_auc_score(te["won"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = load_data()
    print(f"{len(df):,} aerial duel sides, win rate {df['won'].mean():.4f} (should be ~0.5 by construction)")

    print("Running 5-fold cross-validation...")
    cross_validate(df)

    model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=50, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[AERIAL_FEATURES], df["won"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "aerial_model.pkl"))
    print("Saved model to models/aerial_model.pkl")

    df["expected_win_prob"] = model.predict_proba(df[AERIAL_FEATURES])[:, 1]
    out_path = os.path.join(PROCESSED_DIR, "aerials_scored.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote scored aerials -> {out_path}")
    return df


if __name__ == "__main__":
    main()