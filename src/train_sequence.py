"""
Sequence model: instead of scoring a single action in isolation, feed the
LSTM the last K actions of the possession (as feature vectors) and let it
learn momentum/buildup patterns -- e.g. "three quick one-touch passes toward
goal" carries different signal than the same final action reached slowly.

Uses proper variable-length sequence packing (nn.utils.rnn.pack_padded_sequence)
rather than naive zero-padding, so the LSTM never "sees" padding as real actions.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import add_features, TARGET

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_FEATURES = ["start_x", "start_y", "end_x", "end_y", "start_dist_goal", "end_dist_goal",
                 "dx", "dy", "is_pass", "is_carry", "completed_int", "under_pressure_int"]
MAX_LEN = 6  # last up to 6 actions of the possession


def build_possession_sequences(df):
    """For each action, gather the trailing window of actions within the same
    possession (including itself), most-recent-last. Returns a list of
    (sequence_array, length, label, match_id) tuples."""
    df = df.sort_values(["match_id", "possession", "index"]).reset_index(drop=True)

    sequences = []
    for (match_id, possession), block in df.groupby(["match_id", "possession"], sort=False):
        feats = block[SEQ_FEATURES].values.astype(np.float32)
        labels = block["led_to_goal"].values
        for i in range(len(block)):
            lo = max(0, i - MAX_LEN + 1)
            seq = feats[lo:i + 1]
            sequences.append((seq, len(seq), labels[i], match_id))
    return sequences


class XTLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, padded, lengths):
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1]).squeeze(-1)


def make_batch(sequences, scaler, device):
    lengths_list = [l for _, l, _, _ in sequences]
    labels = torch.tensor([lb for _, _, lb, _ in sequences], dtype=torch.float32)

    # Transform all rows across all sequences in one call, then split back --
    # calling scaler.transform per-sequence in a Python loop dominated runtime.
    concat = np.concatenate([s for s, _, _, _ in sequences], axis=0)
    concat_scaled = scaler.transform(concat).astype(np.float32)
    splits = np.cumsum(lengths_list)[:-1]
    per_seq = np.split(concat_scaled, splits)

    seqs = [torch.from_numpy(s) for s in per_seq]
    lengths = torch.tensor(lengths_list)
    padded = pad_sequence(seqs, batch_first=True).float().to(device)
    return padded, lengths, labels.to(device)


def train_one_fold(train_seqs, test_seqs, epochs=15, lr=1e-3, wd=1e-3, seed=42, batch_size=2048):
    torch.manual_seed(seed)

    all_train_feats = np.concatenate([s for s, _, _, _ in train_seqs], axis=0)
    scaler = StandardScaler().fit(all_train_feats)

    y_train = np.array([lb for _, _, lb, _ in train_seqs])
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(DEVICE)

    model = XTLSTM(n_features=len(SEQ_FEATURES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(train_seqs)
    best_auc, best_state = -1, None

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            batch = [train_seqs[i] for i in idx]
            padded, lengths, labels = make_batch(batch, scaler, DEVICE)
            opt.zero_grad()
            logits = model(padded, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for b in range(0, len(test_seqs), 1024):
                batch = test_seqs[b:b + 1024]
                padded, lengths, labels = make_batch(batch, scaler, DEVICE)
                probs = torch.sigmoid(model(padded, lengths)).cpu().numpy()
                all_probs.append(probs)
                all_y.append(labels.cpu().numpy())
        all_probs, all_y = np.concatenate(all_probs), np.concatenate(all_y)
        auc = roc_auc_score(all_y, all_probs)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, scaler, best_auc


def cross_validate(sequences, n_folds=5, seed=42, epochs=20):
    rng = np.random.default_rng(seed)
    match_ids = np.unique([m for _, _, _, m in sequences])
    rng.shuffle(match_ids)
    folds = np.array_split(match_ids, n_folds)

    aucs = []
    for i in range(n_folds):
        test_ids = set(folds[i])
        train_ids = set(match_ids) - test_ids
        train_seqs = [s for s in sequences if s[3] in train_ids]
        test_seqs = [s for s in sequences if s[3] in test_ids]

        _, _, auc = train_one_fold(train_seqs, test_seqs, epochs=epochs, seed=seed + i)
        aucs.append(auc)
        print(f"  fold {i+1}/{n_folds}: AUC={auc:.4f}  (train={len(train_seqs):,} test={len(test_seqs):,})")

    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return aucs


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = add_features(df)
    df = df[df["completed"] == True].reset_index(drop=True)  # noqa: E712

    print("Building possession sequences...")
    sequences = build_possession_sequences(df)
    print(f"{len(sequences):,} sequences (max length {MAX_LEN})")
    print(f"Device: {DEVICE}")

    print("\nRunning 5-fold cross-validation (match-level split)...")
    cross_validate(sequences)
    print("\nFor comparison: single-action MLP CV AUC was 0.618 +/- 0.038 (train_deep.py)")
    print("                LightGBM baseline CV AUC was 0.602 +/- 0.035 (train_baseline.py)")


if __name__ == "__main__":
    main()