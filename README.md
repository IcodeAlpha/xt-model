# Expected Threat (xT): Possession Value and Player Rankings from Real Match Data

A from-scratch implementation of **possession-value modeling** in football analytics,
extended into a full player-evaluation toolkit. The core question: *how much did that
pass, carry, tackle or corner actually help (or stop) a goal-scoring chance?*

Everything is built on real professional data from **[StatsBomb Open Data](https://github.com/statsbomb/open-data)**
(free, no API key): the complete **FIFA World Cup 2022**, all 64 matches, event by event.

The project has three layers:

1. **Core xT models**: LightGBM vs. a PyTorch MLP, plus a classic grid-based xT, a multi-task model and a sequence (LSTM) model.
2. **Player metrics**: threat added, progressive actions, expected assists, defensive value, set pieces, pressing, press resistance, aerials and dribbles.
3. **Delivery**: a FastAPI service and a browser leaderboard dashboard.

## Core idea

For every pass/carry we know the start location, end location, and whether the possession
it belonged to ended in a goal. We train a classifier to predict
`P(goal | where this action ended, action type, pressure)`.

The **value of an action** is the model's probability at the end location minus its
probability at the start location (the Karun Singh xT formulation), applied per action
rather than on a coarse grid.

### Why match-level cross-validation

A single pass is usually several actions removed from any shot, so the signal is weak and
noisy. Splitting rows randomly would leak information between actions of the same match and
inflate AUC. All evaluation here splits **by `match_id`**.

## Core model results (5-fold match-level CV)

| Model | CV AUC | Held-out test AUC |
|---|---|---|
| LightGBM (baseline) | 0.602 ± 0.035 | 0.559 |
| PyTorch MLP | 0.618 ± 0.038 | 0.559 |

The gap is within fold-to-fold variance; with 64 matches this is not evidence that deep
learning wins. Note that the MLP picks its best epoch using the evaluation fold's AUC, which
slightly flatters its CV number.

Other core models:

- `train_xt_grid.py`: classic 16x12 Markov xT (open play only), solved by value iteration. No ML.
- `train_gamestate.py`: adds score difference and minutes remaining to test whether game state changes a pass's value.
- `train_multitask.py`: one network with a shared pitch-geometry trunk and two heads (xT and xG).
- `train_sequence.py`: an LSTM over the last 6 actions of a possession.

## Player metrics

Most metrics use the same recipe: **actual minus expected**, so players are credited for
beating what an average player would do in the same situation.

| Metric | Script(s) | What it measures |
|---|---|---|
| Threat added (xT) | `rank_players.py` | Sum of xT over every completed pass/carry, total and per 90 |
| Progressive actions | `rank_progressive.py` | Passes/carries closing >=25% of distance to goal or ending in the box (rule-based, model-free) |
| Expected assists (xA) | `train_xg.py`, `build_xa.py` | Quality of the shots a player's passes create, via a trained xG model |
| Defensive value | `build_defensive.py` | Danger denied by successful tackles, interceptions, blocks, clearances, recoveries |
| Set pieces | `build_setpieces.py`, `train_setpiece.py`, `rank_setpieces.py` | Corner and attacking free-kick delivery quality |
| Pressing | `build_pressure.py`, `train_pressure.py`, `rank_pressing.py` | Regains within 5 s of a press, minus expected |
| Press resistance | `train_completion.py`, `rank_press_resistance.py` | Pass completion under pressure, minus expected |
| Aerials | `build_aerials.py`, `train_aerial.py`, `rank_aerials.py` | Aerial duel wins minus expected (winners recovered via related events) |
| Dribbles | `build_dribbles.py`, `train_dribble.py`, `rank_dribbles.py` | Take-on completions minus expected |

Per-90 figures use minutes computed from lineups and substitutions (`build_player_minutes.py`).

## Project structure

```
src/
  download_data.py        # pulls raw StatsBomb event JSON
  build_dataset.py        # action-level rows (passes/carries) with goal labels
  features.py             # geometry features + match-level splits
  build_player_minutes.py # minutes per player per match
  build_shots_dataset.py  # shots table for xG
  train_baseline.py       # LightGBM xT + CV
  train_deep.py           # PyTorch MLP xT + CV
  train_xt_grid.py        # classic 16x12 grid xT
  train_gamestate.py      # xT with score/time context
  train_multitask.py      # joint xT + xG network
  train_sequence.py       # LSTM over possession sequences
  train_xg.py, build_xa.py
  build_defensive.py
  build_setpieces.py, train_setpiece.py, rank_setpieces.py
  build_pressure.py, train_pressure.py, rank_pressing.py
  train_completion.py, rank_press_resistance.py
  build_aerials.py, train_aerial.py, rank_aerials.py
  build_dribbles.py, train_dribble.py, rank_dribbles.py
  rank_players.py, rank_progressive.py
  visualize.py            # pitch heatmaps and match pass maps
  api.py                  # FastAPI service
  xt_leaderboard.html     # browser dashboard (talks to the API)
  run_all.py              # runs the whole pipeline in dependency order
models/                   # trained model artifacts
figures/                  # rendered PNGs
data/raw/                 # StatsBomb JSON (downloaded)
data/processed/           # parquet tables and per-player CSV rankings
```

## Reproducing this

```bash
pip install lightgbm torch pandas numpy matplotlib scikit-learn tqdm requests pyarrow fastapi uvicorn python-multipart

cd src
python run_all.py                   # everything except optional heavy steps; skips finished steps
python run_all.py --with-optional   # also deep model, multi-task, sequence model, figures
python run_all.py --list            # show steps in order
python run_all.py --only build_xa   # run one step
python run_all.py --force           # rerun everything
```

`download_data.py` fetches ~190 MB. Swap the competition/season IDs in it to run on any
other StatsBomb open competition (Euro 2020, Champions League seasons, FA WSL, ...).

## API and dashboard

```bash
cd src
uvicorn api:app --reload
```

- Swagger UI at `http://127.0.0.1:8000/docs`
- `POST /predict/xt`: score one pass/carry (LightGBM, deep model, and game-state model if score/time given)
- `GET /health`: simple status check
- `GET /players/{tournament, progressive, xa, defensive, setpieces, pressing, pressresistance, aerials, dribbles}`
- `POST /rank/csv`: upload a CSV with `start_x, start_y, end_x, end_y` (optional `player, team, is_carry, under_pressure, minutes_played`) and get a ranking; uploads are saved to `data/uploads/`
- `GET /uploads`, `GET /uploads/{id}`

Open `src/xt_leaderboard.html` in a browser while the API runs. CORS is wide open for local
use; restrict `allow_origins` before any public deployment.

## Figures

- `figures/xt_surface_comparison.png`: learned pitch-value surface, both models side by side.
- `figures/pass_value_map_3869685.png`: every completed pass of the 2022 Final colored by threat added.

## Known limitations

- **Small sample.** 64 matches means 1 to 7 games per player; rankings are noisy, especially per-90 for low-minute players.
- **Weak label.** Every action in a possession that ends in a goal gets the same label, however far it was from the goal.
- **Aerials** have a ~50% win rate by construction (each duel yields one winner and one loser row), so the model learns little beyond location.
- **Red cards** are not modelled in minutes played.
- **Pressing success** is a 5-second possession-regain heuristic, not a ground-truth label.