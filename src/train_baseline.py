"""
Baseline model: LightGBM classifier predicting P(possession ends in goal | action end-state).

This mirrors how professional analytics vendors (e.g. StatsBomb's own xT/OBV models)
typically approach this -- a gradient-boosted tree over hand-crafted spatial features.
We train it on the END state of the action, then also score the START state,
so xT_value = P(end) - P(start), matching the classic Karun Singh xT formulation.
"""
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from features import add_features, train_val_test_split_by_match, TARGET

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Features describing the END location + context -- this is what the model scores.
END_STATE_FEATURES = [
    "end_x", "end_y", "end_dist_goal", "end_angle_goal",
    "is_pass", "is_carry", "completed_int", "under_pressure_int",
]


def cross_validate(df, n_folds=5, seed=42):
    """5-fold CV split by match_id. With only 64 matches, a single 70/15/15
    split is noisy -- CV gives a much more honest estimate of generalization."""
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

        m = LGBMClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=3, num_leaves=7,
            min_child_samples=300, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0,
            random_state=seed, verbose=-1,
        )
        m.fit(tr[END_STATE_FEATURES], tr[TARGET])
        p = m.predict_proba(te[END_STATE_FEATURES])[:, 1]
        auc = roc_auc_score(te[TARGET], p)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = add_features(df)
    # only keep completed actions -- an incomplete pass doesn't actually deliver the ball
    df = df[df["completed"] == True].reset_index(drop=True)  # noqa: E712

    print("Running 5-fold cross-validation (match-level split)...")
    cross_validate(df)
    print()

    train_df, val_df, test_df = train_val_test_split_by_match(df)
    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    X_train, y_train = train_df[END_STATE_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[END_STATE_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[END_STATE_FEATURES], test_df[TARGET]

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=7,
        min_child_samples=300,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        ll = log_loss(y, p)
        print(f"[{name}] AUC={auc:.4f}  logloss={ll:.4f}  base_rate={y.mean():.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "lgbm_baseline.pkl"))
    print("Saved model to models/lgbm_baseline.pkl")

    return model, (train_df, val_df, test_df)


if __name__ == "__main__":
    main()
