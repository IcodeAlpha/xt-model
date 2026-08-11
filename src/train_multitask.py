"""
Multi-task deep learning model.

Idea: xT (possession value on passes/carries) and xG (shot quality) are both
fundamentally about "how dangerous is the ball at this pitch location" --
they should share a representation of pitch geometry, even though they're
trained on different event types with different task-specific features.

Architecture:
  shared trunk: (x, y, dist_to_goal, angle_to_goal) -> hidden representation
  xT head:      shared repr + [is_pass, is_carry, completed, under_pressure] -> P(possession scores)
  xG head:      shared repr + [body_part_*, first_time, under_pressure]      -> P(shot scores)

Trained jointly: each step draws one batch from each task and combines the
losses. This lets the xG task (which has direct, dense supervision -- a shot's
outcome is immediate) regularize the pitch-geometry representation that the
xT task (whose signal is sparse and several actions removed from any goal)
also relies on.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import add_features, angle_to_goal, train_val_test_split_by_match, TARGET
from train_baseline import END_STATE_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHARED_FEATURES = ["x", "y", "dist_goal", "angle_goal"]
XT_EXTRA = ["is_pass", "is_carry", "completed_int", "under_pressure_int"]
XG_EXTRA = ["is_foot", "is_head", "first_time_int", "under_pressure_int"]


def load_xt_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = add_features(df)
    df = df[df["completed"] == True].reset_index(drop=True)  # noqa: E712
    df = df.rename(columns={"end_x": "x", "end_y": "y", "end_dist_goal": "dist_goal",
                             "end_angle_goal": "angle_goal"})
    return df


def load_xg_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "shots.parquet"))
    df["dist_goal"] = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)
    df["angle_goal"] = angle_to_goal(df["x"].values, df["y"].values)
    df["is_foot"] = df["body_part"].isin(["Right Foot", "Left Foot"]).astype(int)
    df["is_head"] = (df["body_part"] == "Head").astype(int)
    df["first_time_int"] = df["first_time"].astype(int)
    df["under_pressure_int"] = df["under_pressure"].astype(int)
    return df


class MultiTaskXTXG(nn.Module):
    def __init__(self, n_shared=4, n_xt_extra=4, n_xg_extra=4, hidden=32, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_shared, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.xt_head = nn.Sequential(
            nn.Linear(hidden + n_xt_extra, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.xg_head = nn.Sequential(
            nn.Linear(hidden + n_xg_extra, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward_xt(self, shared_x, extra_x):
        h = self.trunk(shared_x)
        return self.xt_head(torch.cat([h, extra_x], dim=-1)).squeeze(-1)

    def forward_xg(self, shared_x, extra_x):
        h = self.trunk(shared_x)
        return self.xg_head(torch.cat([h, extra_x], dim=-1)).squeeze(-1)


def _prep(df, shared_cols, extra_cols, target_col, shared_scaler=None, extra_scaler=None, fit=False):
    shared = df[shared_cols].values.astype(np.float32)
    extra = df[extra_cols].values.astype(np.float32)
    if fit:
        shared_scaler = StandardScaler().fit(shared)
        extra_scaler = StandardScaler().fit(extra)
    shared = shared_scaler.transform(shared).astype(np.float32)
    extra = extra_scaler.transform(extra).astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    return shared, extra, y, shared_scaler, extra_scaler


def train_one_fold(xt_train, xt_test, xg_train, xg_test, epochs=40, lr=1e-3, wd=1e-3, seed=42):
    torch.manual_seed(seed)

    xt_shared_tr, xt_extra_tr, xt_y_tr, xt_s_sc, xt_e_sc = _prep(
        xt_train, SHARED_FEATURES, XT_EXTRA, TARGET, fit=True)
    xt_shared_te, xt_extra_te, xt_y_te, _, _ = _prep(
        xt_test, SHARED_FEATURES, XT_EXTRA, TARGET, xt_s_sc, xt_e_sc)

    xg_shared_tr, xg_extra_tr, xg_y_tr, xg_s_sc, xg_e_sc = _prep(
        xg_train, SHARED_FEATURES, XG_EXTRA, "is_goal", fit=True)
    xg_shared_te, xg_extra_te, xg_y_te, _, _ = _prep(
        xg_test, SHARED_FEATURES, XG_EXTRA, "is_goal", xg_s_sc, xg_e_sc)

    model = MultiTaskXTXG().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    xt_pos_weight = torch.tensor([(xt_y_tr == 0).sum() / max((xt_y_tr == 1).sum(), 1)]).to(DEVICE)
    xg_pos_weight = torch.tensor([(xg_y_tr == 0).sum() / max((xg_y_tr == 1).sum(), 1)]).to(DEVICE)
    xt_loss_fn = nn.BCEWithLogitsLoss(pos_weight=xt_pos_weight)
    xg_loss_fn = nn.BCEWithLogitsLoss(pos_weight=xg_pos_weight)

    xt_shared_tr_t = torch.tensor(xt_shared_tr).to(DEVICE)
    xt_extra_tr_t = torch.tensor(xt_extra_tr).to(DEVICE)
    xt_y_tr_t = torch.tensor(xt_y_tr).to(DEVICE)
    xg_shared_tr_t = torch.tensor(xg_shared_tr).to(DEVICE)
    xg_extra_tr_t = torch.tensor(xg_extra_tr).to(DEVICE)
    xg_y_tr_t = torch.tensor(xg_y_tr).to(DEVICE)

    n_xt, n_xg = len(xt_y_tr), len(xg_y_tr)
    batch_xt, batch_xg = 512, min(64, max(n_xg, 1))

    best_xt_auc, best_xg_auc, best_state = -1, -1, None

    for epoch in range(epochs):
        model.train()
        perm_xt = torch.randperm(n_xt)
        n_batches = max(n_xt // batch_xt, 1)
        for b in range(n_batches):
            idx_xt = perm_xt[b * batch_xt:(b + 1) * batch_xt]
            idx_xg = torch.randint(0, n_xg, (batch_xg,))

            opt.zero_grad()
            xt_logits = model.forward_xt(xt_shared_tr_t[idx_xt], xt_extra_tr_t[idx_xt])
            xg_logits = model.forward_xg(xg_shared_tr_t[idx_xg], xg_extra_tr_t[idx_xg])
            loss = xt_loss_fn(xt_logits, xt_y_tr_t[idx_xt]) + xg_loss_fn(xg_logits, xg_y_tr_t[idx_xg])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xt_probs = torch.sigmoid(model.forward_xt(
                torch.tensor(xt_shared_te).to(DEVICE), torch.tensor(xt_extra_te).to(DEVICE))).cpu().numpy()
            xg_probs = torch.sigmoid(model.forward_xg(
                torch.tensor(xg_shared_te).to(DEVICE), torch.tensor(xg_extra_te).to(DEVICE))).cpu().numpy()
        xt_auc = roc_auc_score(xt_y_te, xt_probs)
        xg_auc = roc_auc_score(xg_y_te, xg_probs) if len(np.unique(xg_y_te)) > 1 else float("nan")

        if xt_auc + (xg_auc if xg_auc == xg_auc else 0) > best_xt_auc + (best_xg_auc if best_xg_auc == best_xg_auc else 0):
            best_xt_auc, best_xg_auc = xt_auc, xg_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    scalers = {"xt_shared": xt_s_sc, "xt_extra": xt_e_sc, "xg_shared": xg_s_sc, "xg_extra": xg_e_sc}
    return model, scalers, best_xt_auc, best_xg_auc


def cross_validate(xt_df, xg_df, n_folds=5, seed=42, epochs=25):
    rng = np.random.default_rng(seed)
    match_ids = xt_df["match_id"].unique()
    rng.shuffle(match_ids)
    folds = np.array_split(match_ids, n_folds)

    xt_aucs, xg_aucs = [], []
    for i in range(n_folds):
        test_ids = set(folds[i])
        train_ids = set(match_ids) - test_ids

        xt_tr = xt_df[xt_df["match_id"].isin(train_ids)]
        xt_te = xt_df[xt_df["match_id"].isin(test_ids)]
        xg_tr = xg_df[xg_df["match_id"].isin(train_ids)]
        xg_te = xg_df[xg_df["match_id"].isin(test_ids)]

        if xg_te["is_goal"].nunique() < 2 or len(xg_tr) < 20:
            continue  # skip folds with too few shots/goals to evaluate meaningfully

        _, _, xt_auc, xg_auc = train_one_fold(xt_tr, xt_te, xg_tr, xg_te, epochs=epochs, seed=seed + i)
        xt_aucs.append(xt_auc)
        xg_aucs.append(xg_auc)
        print(f"  fold {i+1}/{n_folds}: xT AUC={xt_auc:.4f}  xG AUC={xg_auc:.4f}  "
              f"(xt_train={len(xt_tr):,} xg_train={len(xg_tr):,} xg_test={len(xg_te):,})")

    xt_aucs, xg_aucs = np.array(xt_aucs), np.array(xg_aucs)
    print(f"CV xT AUC: {xt_aucs.mean():.4f} +/- {xt_aucs.std():.4f}")
    print(f"CV xG AUC: {xg_aucs.mean():.4f} +/- {xg_aucs.std():.4f}")
    return xt_aucs, xg_aucs


def main():
    xt_df = load_xt_data()
    xg_df = load_xg_data()
    print(f"xT actions: {len(xt_df):,}   xG shots: {len(xg_df):,}")
    print(f"Device: {DEVICE}")

    print("\nRunning 5-fold cross-validation (match-level split, joint training)...")
    cross_validate(xt_df, xg_df)

    # Sanity check against StatsBomb's own professional xG model
    corr = None
    if xg_df["statsbomb_xg"].notna().any():
        corr = xg_df[["statsbomb_xg"]].assign(is_goal=xg_df["is_goal"]).corr().iloc[0, 1]
    print(f"\nStatsBomb xG vs actual goal correlation (reference/benchmark): {corr:.4f}" if corr else "")


if __name__ == "__main__":
    main()