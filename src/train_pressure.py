"""
Counter-pressing value model: predicts P(this pressure leads to a regain
within 5 seconds), trained on location + duration. Player value = actual
regains minus expected regains (from the model) -- rewards players who win
the ball back in situations the model says are genuinely hard to regain in,
not just players who apply the most pressure in general.
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

PRESSURE_FEATURES = ["x", "y", "dist_opp_goal", "angle_opp_goal", "duration"]


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

        m = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=100, reg_lambda=2.0, random_state=seed, verbose=-1)
        m.fit(tr[PRESSURE_FEATURES], tr["success"])
        p = m.predict_proba(te[PRESSURE_FEATURES])[:, 1]
        auc = roc_auc_score(te["success"], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "pressures.parquet"))
    print(f"{len(df):,} pressures, success rate {df['success'].mean():.4f}")

    print("Running 5-fold cross-validation...")
    cross_validate(df)

    model = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=3, num_leaves=7,
                            min_child_samples=100, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(df[PRESSURE_FEATURES], df["success"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "pressure_model.pkl"))
    print("Saved model to models/pressure_model.pkl")

    df["expected_success"] = model.predict_proba(df[PRESSURE_FEATURES])[:, 1]
    out_path = os.path.join(PROCESSED_DIR, "pressures_scored.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote scored pressures -> {out_path}")
    return df


if __name__ == "__main__":
    main()