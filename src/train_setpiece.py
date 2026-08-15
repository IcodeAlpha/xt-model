"""
Set-piece delivery quality model: a dedicated LightGBM classifier predicting
P(this corner/free-kick delivery directly leads to a shot), trained
separately from the open-play xT model since set pieces behave differently
(trajectory/height matters more than the usual distance/angle relationship).
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

SETPIECE_FEATURES = ["end_x", "end_y", "end_dist_goal", "end_angle_goal",
                      "is_corner", "is_high", "is_ground", "under_pressure_int"]


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
        if te["led_to_shot"].nunique() < 2 or len(tr) < 50:
            continue

        m = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=30, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[SETPIECE_FEATURES], tr["led_to_shot"])
        p = m.predict_proba(te[SETPIECE_FEATURES])[:, 1]
        auc = roc_auc_score(te["led_to_shot"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr)} test={len(te)})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "setpieces.parquet"))
    df = df[(df["completed"] == True) & (df["is_attacking_delivery"] == True)].reset_index(drop=True)  # noqa: E712
    print(f"{len(df):,} completed ATTACKING set-piece deliveries "
          f"(corners + final-third free kicks), led-to-shot rate {df['led_to_shot'].mean():.4f}")

    print("Running 5-fold cross-validation...")
    cross_validate(df)

    model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=30, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[SETPIECE_FEATURES], df["led_to_shot"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "setpiece_model.pkl"))
    print("Saved model to models/setpiece_model.pkl")

    df["delivery_value"] = model.predict_proba(df[SETPIECE_FEATURES])[:, 1]

    out_path = os.path.join(PROCESSED_DIR, "setpieces_scored.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote scored deliveries -> {out_path}")
    return df


if __name__ == "__main__":
    main()