"""
Standalone Expected Goals (xG) model -- a single-task LightGBM classifier,
independent of the multi-task xT/xG model from earlier. This is what powers
the xA (Expected Assists) metric: xA = sum of predicted xG for every shot a
player's pass directly assisted.

Compared against StatsBomb's own professional xG as a benchmark.
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

XG_FEATURES = ["x", "y", "dist_goal", "angle_goal", "is_foot", "is_head",
               "first_time_int", "under_pressure_int"]


def load_xg_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "shots.parquet"))
    df["dist_goal"] = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)
    df["angle_goal"] = angle_to_goal(df["x"].values, df["y"].values)
    df["is_foot"] = df["body_part"].isin(["Right Foot", "Left Foot"]).astype(int)
    df["is_head"] = (df["body_part"] == "Head").astype(int)
    df["first_time_int"] = df["first_time"].astype(int)
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
        tr, te = df[df["match_id"].isin(train_ids)], df[df["match_id"].isin(test_ids)]
        if te["is_goal"].nunique() < 2:
            continue

        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=30, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[XG_FEATURES], tr["is_goal"])
        p = m.predict_proba(te[XG_FEATURES])[:, 1]
        auc = roc_auc_score(te["is_goal"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr)} test={len(te)})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = load_xg_data()
    print(f"{len(df):,} shots, goal rate {df['is_goal'].mean():.4f}")

    print("Running 5-fold cross-validation...")
    cross_validate(df)

    # Fit final model on all data for production use
    model = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=30, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[XG_FEATURES], df["is_goal"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "xg_model.pkl"))
    print("Saved model to models/xg_model.pkl")

    # Sanity check against StatsBomb's own xG
    df["our_xg"] = model.predict_proba(df[XG_FEATURES])[:, 1]
    corr = df[["our_xg", "statsbomb_xg"]].corr().iloc[0, 1]
    print(f"Correlation with StatsBomb's own xG: {corr:.4f}")
    print(f"Our mean predicted xG: {df['our_xg'].mean():.4f}  vs  StatsBomb mean xG: {df['statsbomb_xg'].mean():.4f}")


if __name__ == "__main__":
    main()