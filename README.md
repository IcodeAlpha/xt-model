# Expected Threat (xT): Classical ML vs. Deep Learning on Real Match Data

A from-scratch implementation of **possession-value modeling** in football analytics —
the same family of model used by clubs and analytics vendors to answer "how much did
that pass actually help create a goal-scoring opportunity?"

Two models are trained and evaluated head-to-head on identical cross-validation folds:

1. **LightGBM** (gradient-boosted trees) — the classical, industry-standard approach
2. **A small PyTorch MLP** — a deep learning approach, trained on the same features

## Data

Real professional match data from **[StatsBomb Open Data](https://github.com/statsbomb/open-data)**
(free, no API key required). This project uses the complete **FIFA World Cup 2022** —
all 64 matches, event-by-event.

- 122,279 completed passes and carries extracted from raw event streams
- Every action labeled with whether the possession it belongs to ended in a goal
- Base rate: ~1.9% of possessions end in a goal — a genuinely hard, imbalanced problem

## Method

For every pass/carry we know the start location, end location, and whether that
possession eventually produced a goal. We train a classifier to predict:

```
P(goal | this action's end location, action type, pressure, completion)
```

The **value of an action** is then the model's probability at the end location minus
its probability at the start location — the classic Karun Singh xT formulation,
applied here at the individual-action level rather than a discretized 12x8 grid.

### Why this is genuinely hard (and why that matters)

A single pass is usually several actions removed from any shot. Predicting "will this
possession score" from one action's coordinates is a weak, noisy signal by nature —
which is exactly why match-level cross-validation (not a random row-level split) is
used throughout. A row-level split leaks information between actions in the same
possession/match and produces badly inflated, meaningless AUC scores. Splitting by
match_id is the only honest way to evaluate this.

## Results (5-fold match-level cross-validation)

| Model | CV AUC | Held-out test AUC |
|---|---|---|
| LightGBM (baseline) | 0.602 ± 0.035 | 0.559 |
| PyTorch MLP (deep learning) | **0.618 ± 0.038** | 0.559 |

Both models land in the range reported by published possession-value models on
similar-sized datasets. The neural net edges out the tree-based baseline on average,
though the gap is within the fold-to-fold variance — with only 64 matches, that's an
honest result, not a case for over-claiming "deep learning wins."

## Figures

- `figures/xt_surface_comparison.png` — the learned pitch-value surface for both
  models side by side. Both correctly learn that value rises sharply near the
  opponent's goal and the middle channel, without ever being told the rules of football.
- `figures/pass_value_map_3869685.png` — every completed pass from the **2022 World
  Cup Final (Argentina 3–3 France)**, colored by threat added. The single highest-value
  pass found (Griezmann, 67th minute) is a line-breaking ball from deep in France's own
  half out to the edge of Argentina's box — a sanity check that the model is picking up
  real tactical signal, not noise.

## Project structure

```
src/
  download_data.py    # pulls raw StatsBomb event JSON for a competition/season
  build_dataset.py    # parses events into action-level rows with possession/goal labels
  features.py         # feature engineering + match-level train/val/test split
  train_baseline.py   # LightGBM model + 5-fold CV
  train_deep.py        # PyTorch MLP model + 5-fold CV (same protocol as baseline)
  visualize.py         # pitch-value heatmaps and match pass-value overlays
models/                # trained model artifacts
figures/               # rendered PNGs
data/processed/        # actions.parquet -- the action-level dataset used for training
```

## Reproducing this

```bash
pip install lightgbm torch pandas numpy matplotlib scikit-learn tqdm requests pyarrow

cd src
python download_data.py     # ~190MB, all 64 World Cup 2022 matches
python build_dataset.py     # builds data/processed/actions.parquet
python train_baseline.py    # LightGBM + CV
python train_deep.py        # PyTorch MLP + CV
python visualize.py         # renders figures/
```

Swap the competition/season IDs in `download_data.py` to run this on any other
competition in StatsBomb's open data catalog (e.g. Euro 2020, Champions League seasons,
FA Women's Super League).

## Natural extensions

- **Add tracking data** (if available) to model off-ball positioning and pressure
  properly, instead of using StatsBomb's "under_pressure" flag as a proxy
- **Sequence modeling**: feed the last N actions of a possession into a small
  transformer instead of scoring actions independently — captures momentum/buildup
- **Player-level aggregation**: sum xT-added per 90 minutes to rank players on
  progressive/creative contribution, independent of goals and assists
- **Shots/xG as a second head**: multi-task training with shot-conversion probability
  alongside possession-value, sharing the pitch-location representation
