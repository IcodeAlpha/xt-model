"""
Where does the game-state AUC gain come from?

train_gamestate.py reports location-only CV AUC 0.60 -> 0.68 with score_diff and
minutes_remaining added. That is a big jump for two simple features, so this script
checks it.

SUSPECT: build_gamestate.py computes
    minutes_remaining = full_time - minute
where full_time is the minute of the match's own "Half End" events. So the feature uses
HINDSIGHT: it already "knows" how much stoppage time there will be (93 to 103 min) and
whether the match goes to extra time (121 to 124 min, only 5 of 64 matches). A first-half
pass at minute 10 gets ~112 minutes remaining in an extra-time match but ~87 otherwise, so
the model can partly tell WHICH KIND OF MATCH it is in (knockout, extra time, long
stoppages), which is not game state and is not knowable when the pass is played.

TEST: compare against a clock-only version that cannot see the future:
    minutes_left_90 = max(0, 90 - minute)

Reading the results:
  * If the "no hindsight" rows keep most of the gain  -> game state really helps.
  * If they lose most of it and "match_length only" alone gains a lot
                                                       -> the gain was hindsight leakage;
                                                          switch the model to minutes_left_90.

Uses the same match-level 5-fold split and model settings as train_gamestate.py.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from features import TARGET
from train_baseline import END_STATE_FEATURES

N_FOLDS = 5
SEED = 42
N_PERM_REPEATS = 5


def make_model(seed=SEED):
    # identical settings to train_gamestate.py
    return LGBMClassifier(n_estimators=300, learning_rate=0.02, max_depth=3, num_leaves=7,
                          min_child_samples=300, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0,
                          random_state=seed, verbose=-1)


def get_folds(df, n_folds=N_FOLDS, seed=SEED):
    """Match-level folds, same recipe as train_gamestate.cross_validate."""
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)
    return np.array_split(match_ids, n_folds)


def add_clock_features(df):
    df = df.copy()
    # what a live model could actually know: the clock, capped at 90
    df["minutes_left_90"] = np.clip(90 - df["minute"], 0, None)
    # the hindsight part: total match length, recovered as remaining + elapsed
    df["match_length"] = df["minutes_remaining"] + df["minute"]
    return df


def feature_sets():
    loc = list(END_STATE_FEATURES)
    return {
        "location only (baseline)":                   loc,
        "+ score_diff":                               loc + ["score_diff"],
        "+ minutes_remaining (as shipped)":           loc + ["minutes_remaining"],
        "+ minutes_left_90 (clock only)":             loc + ["minutes_left_90"],
        "+ both, as shipped":                         loc + ["score_diff", "minutes_remaining"],
        "+ both, NO hindsight":                       loc + ["score_diff", "minutes_left_90"],
        "+ match_length only (hindsight alone)":      loc + ["match_length"],
    }


def rate_tables(df):
    print("Goal-possession rate by context (share of actions in possessions that end in a goal)\n")
    sd = pd.cut(df["score_diff"], [-99, -2, -1, 0, 1, 99],
                labels=["down 2+", "down 1", "level", "up 1", "up 2+"])
    print(df.groupby(sd, observed=True)[TARGET].agg(rate="mean", actions="size").round(4).to_string())
    print()
    ml = pd.cut(df["match_length"], [0, 100, 110, 200],
                labels=["<=100 min (regulation)", "101-110", ">110 (extra time)"])
    print(df.groupby(ml, observed=True)[TARGET].agg(rate="mean", actions="size").round(4).to_string())
    print()


def run_ablation(df):
    folds = get_folds(df)
    results = {}
    for name, feats in feature_sets().items():
        aucs = []
        for i in range(N_FOLDS):
            test_ids = set(folds[i])
            te = df[df["match_id"].isin(test_ids)]
            tr = df[~df["match_id"].isin(test_ids)]
            m = make_model()
            m.fit(tr[feats], tr[TARGET])
            aucs.append(roc_auc_score(te[TARGET], m.predict_proba(te[feats])[:, 1]))
        results[name] = np.array(aucs)

    base = results["location only (baseline)"]
    print(f"{'feature set':<42}{'CV AUC':>8}{'sd':>7}{'vs base':>9}{'folds better':>14}")
    for name, a in results.items():
        better = int((a > base).sum())
        print(f"{name:<42}{a.mean():>8.4f}{a.std():>7.4f}{a.mean() - base.mean():>+9.4f}"
              f"{better:>9}/{N_FOLDS}")
    return results


def permutation_drop(df, feats, probe, seed=SEED):
    """AUC lost when `probe` features are shuffled on the held-out fold (bigger = relied on more)."""
    rng = np.random.default_rng(seed)
    folds = get_folds(df)
    drops = {p: [] for p in probe}
    for i in range(N_FOLDS):
        test_ids = set(folds[i])
        te = df[df["match_id"].isin(test_ids)].copy()
        tr = df[~df["match_id"].isin(test_ids)]
        m = make_model()
        m.fit(tr[feats], tr[TARGET])
        base = roc_auc_score(te[TARGET], m.predict_proba(te[feats])[:, 1])
        for p in probe:
            d = []
            for _ in range(N_PERM_REPEATS):
                shuffled = te.copy()
                shuffled[p] = rng.permutation(shuffled[p].to_numpy())
                d.append(base - roc_auc_score(te[TARGET], m.predict_proba(shuffled[feats])[:, 1]))
            drops[p].append(np.mean(d))
    return {p: float(np.mean(v)) for p, v in drops.items()}


def verdict(results):
    base = results["location only (baseline)"].mean()
    shipped = results["+ both, as shipped"].mean() - base
    clean = results["+ both, NO hindsight"].mean() - base
    hind = results["+ match_length only (hindsight alone)"].mean() - base
    print(f"\nGain as shipped: {shipped:+.4f}   Gain with no hindsight: {clean:+.4f}   "
          f"Hindsight feature alone: {hind:+.4f}")
    if shipped <= 0.005:
        print("-> No meaningful gain to explain.")
    elif clean >= 0.7 * shipped:
        print("-> Most of the gain survives without hindsight: game state genuinely helps.")
    elif clean <= 0.3 * shipped:
        print("-> Most of the gain disappears without hindsight: it was the model learning "
              "match length (stoppage/extra time), not game state. Use minutes_left_90.")
    else:
        print("-> Mixed: part of the gain is real, part is hindsight. Use minutes_left_90 and "
              "report the smaller number.")


def main(df=None):
    if df is None:
        from train_gamestate import load_data
        df = load_data()
    df = add_clock_features(df)

    print(f"\n{len(df):,} actions, {df['match_id'].nunique()} matches\n")
    rate_tables(df)

    print("=== Ablation (same match-level folds for every row) ===")
    results = run_ablation(df)
    verdict(results)

    print("\n=== Permutation importance: AUC lost when a feature is shuffled in held-out matches ===")
    shipped = list(END_STATE_FEATURES) + ["score_diff", "minutes_remaining"]
    clean = list(END_STATE_FEATURES) + ["score_diff", "minutes_left_90"]
    for label, feats in (("as shipped", shipped), ("no hindsight", clean)):
        drops = permutation_drop(df, feats, ["score_diff", feats[-1]])
        print(f"  model {label}: " + ", ".join(f"{k} {v:+.4f}" for k, v in drops.items()))
    return results


if __name__ == "__main__":