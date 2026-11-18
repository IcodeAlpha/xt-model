"""
Uncertainty + reliability for the player xT ranking.

Two questions the raw leaderboard cannot answer:

1. HOW MUCH SHOULD WE TRUST A PLAYER'S NUMBER?
   Each player only has 1-7 matches. We bootstrap over a player's own matches
   (resample with replacement) and recompute xT per 90 to get a 90% interval.
   Wide interval = the ranking position is mostly luck of which matches they played.

2. IS THE METRIC A STABLE PLAYER TRAIT?
   Split each player's matches randomly into two halves, compute xT/90 in each,
   and correlate across players. Repeat many times. High correlation = the metric
   measures something that persists; ~0 = it is noise. We apply the
   Spearman-Brown correction because each half has only half the data.

Caveats:
  * The bootstrap only captures match-to-match variation for that player. It does not
    capture model error in xT itself or opponent strength.
  * A decent split-half correlation can come from ROLE (midfielders simply touch the ball
    more) rather than skill. It shows the metric is stable, not that it measures ability.

Input : data/processed/player_xt_per_match.csv  (written by rank_players.py)
Output: data/processed/player_xt_tournament_ci.csv
"""
import os

import numpy as np
import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
IN_PATH = os.path.join(PROC, "player_xt_per_match.csv")
OUT_PATH = os.path.join(PROC, "player_xt_tournament_ci.csv")

MIN_MINUTES = 180      # players below this are too noisy to rank at all
N_BOOT = 2000
N_SPLITS = 500
RELIABILITY_MIN_MATCHES = 4
SEED = 7


def per90(xt, minutes):
    return 90.0 * np.sum(xt) / np.sum(minutes)


def bootstrap_intervals(df, rng):
    rows = []
    for (player, team), g in df.groupby(["player", "team"]):
        xt = g["xt_added"].to_numpy()
        mins = g["minutes_played"].to_numpy()
        if mins.sum() < MIN_MINUTES:
            continue
        n = len(g)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boots = 90.0 * xt[idx].sum(axis=1) / mins[idx].sum(axis=1)
        rows.append({
            "player": player, "team": team, "matches": n,
            "minutes": int(mins.sum()),
            "xt_per90": per90(xt, mins),
            "ci_low": np.percentile(boots, 5),
            "ci_high": np.percentile(boots, 95),
        })
    out = pd.DataFrame(rows)
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    out["rank"] = out["xt_per90"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def split_half_reliability(df, rng):
    games = df.groupby("player").filter(lambda g: len(g) >= RELIABILITY_MIN_MATCHES)
    players = {p: (g["xt_added"].to_numpy(), g["minutes_played"].to_numpy())
               for p, g in games.groupby("player")}
    corrs = []
    for _ in range(N_SPLITS):
        a, b = [], []
        for xt, mins in players.values():
            perm = rng.permutation(len(xt))
            h = len(xt) // 2
            i, j = perm[:h], perm[h:2 * h]
            a.append(per90(xt[i], mins[i]))
            b.append(per90(xt[j], mins[j]))
        r = pd.Series(a).corr(pd.Series(b), method="spearman")
        corrs.append(r)
    r = float(np.mean(corrs))
    sb = 2 * r / (1 + r)          # Spearman-Brown: estimate for the full-length metric
    return len(players), r, sb, float(np.std(corrs))


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(IN_PATH)
    df = df[df["minutes_played"] > 0]

    ci = bootstrap_intervals(df, rng)
    ci.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(ci)} players (>= {MIN_MINUTES} min) -> {OUT_PATH}\n")

    top = ci.head(10)
    print("Top 10 by xT/90 with 90% bootstrap interval:")
    for _, r in top.iterrows():
        print(f"  {int(r['rank']):>3}. {r['player'][:28]:<28} {r['xt_per90']:.3f}  "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  ({int(r['matches'])} matches)")

    # How many players are statistically separable from the #1 spot?
    best_low = ci.iloc[0]["ci_low"]
    overlap = int((ci["ci_high"] >= best_low).sum())
    print(f"\nPlayers whose interval overlaps the #1 player's: {overlap} of {len(ci)}")
    print(f"Median interval width: {ci['ci_width'].median():.3f} "
          f"(median xT/90 = {ci['xt_per90'].median():.3f})")

    n, r, sb, sd = split_half_reliability(df, rng)
    print(f"\nSplit-half reliability ({n} players with >= {RELIABILITY_MIN_MATCHES} matches, "
          f"{N_SPLITS} random splits):")
    print(f"  Spearman r between halves = {r:.2f} (sd {sd:.2f}); "
          f"Spearman-Brown corrected = {sb:.2f}")


if __name__ == "__main__":
    main()