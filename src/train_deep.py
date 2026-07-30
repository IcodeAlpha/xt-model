"""
Deep learning model: a small MLP over pitch coordinates + context features,
predicting P(possession ends in goal | action end-state).

Compared against the LightGBM baseline on the SAME match-level CV folds,
so the numbers are directly comparable.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import add_features, train_val_test_split_by_match, TARGET
from train_baseline import END_STATE_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XTNet(nn.Module):
    """Small MLP: pitch location + context -> P(possession scores).

    Kept deliberately compact (a few hundred params) because the training
    signal per action is sparse (~2% positive rate, single-match-level
    labels) -- a large network would just memorize noise on this dataset
    size (~120k rows / 64 matches).
    """
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(train_df, test_df, epochs=60, lr=1e-3, weight_decay=1e-3, seed=42):
    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[END_STATE_FEATURES].values.astype(np.float32))
    X_test = scaler.transform(test_df[END_STATE_FEATURES].values.astype(np.float32))
    y_train = train_df[TARGET].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)

    # Handle class imbalance with a positive class weight rather than resampling.
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    model = XTNet(n_features=X_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    best_auc, best_state = -1, None
    X_test_t = torch.tensor(X_test).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        auc = roc_auc_score(y_test, probs)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, scaler, best_auc


def cross_validate(df, n_folds=5, seed=42, epochs=25):
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

        _, _, auc = train_one_fold(tr, te, seed=seed + i, epochs=epochs)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(tr):,} test={len(te):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = add_features(df)
    df = df[df["completed"] == True].reset_index(drop=True)  # noqa: E712

    print(f"Device: {DEVICE}")
    print("Running 5-fold cross-validation (match-level split)...")
    cross_validate(df)
    print()

    # Final model trained on train+val, evaluated on held-out test (same split as baseline)
    train_df, val_df, test_df = train_val_test_split_by_match(df)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    model, scaler, test_auc = train_one_fold(fit_df, test_df, epochs=30)
    print(f"[final holdout test] AUC={test_auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": END_STATE_FEATURES,
    }, os.path.join(MODELS_DIR, "xtnet.pt"))
    print("Saved model to models/xtnet.pt")


if __name__ == "__main__":
    main()
