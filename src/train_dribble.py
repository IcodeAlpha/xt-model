"""
Dribble success model: predicts P(this take-on succeeds) from location and
pressure context. Player value = actual completions minus expected
completions -- rewards genuinely difficult dribbles completed, not just
high-volume dribblers operating in easy space.
"""
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from features import angle_to_goal

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DRIBBLE_FEATURES = ["x", "y", "dist_goal", "angle_goal", "under_pressure_int"]


def load_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "dribbles.parquet"))
    df["dist_goal"] = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)
    df["angle_goal"] = angle_to_goal(df["x"].values, df["y"].values)
    df["under_pressure_int"] = df["under_pressure"].astype(int)
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
        if te["completed"].nunique() < 2:
            continue

        m = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=40, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[DRIBBLE_FEATURES], tr["completed"])
        p = m.predict_proba(te[DRIBBLE_FEATURES])[:, 1]
        auc = roc_auc_score(te["completed"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = load_data()
    print(f"{len(df):,} dribble attempts, success rate {df['completed'].mean():.4f}")
    print(f"Success rate under pressure: {df[df['under_pressure_int']==1]['completed'].mean():.4f}")
    print(f"Success rate NOT under pressure: {df[df['under_pressure_int']==0]['completed'].mean():.4f}")

    print("\nRunning 5-fold cross-validation...")
    cross_validate(df)

    model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=40, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[DRIBBLE_FEATURES], df["completed"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "dribble_model.pkl"))
    print("Saved model to models/dribble_model.pkl")

    df["expected_completion"] = model.predict_proba(df[DRIBBLE_FEATURES])[:, 1]
    out_path = os.path.join(PROCESSED_DIR, "dribbles_scored.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote scored dribbles -> {out_path}")
    return df


if __name__ == "__main__":
    main()