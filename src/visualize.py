"""
Visualizations:
  1. The learned xT value surface across the pitch (both models), as a heatmap.
  2. A real match's passes, colored by threat added (xT_end - xT_start).
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.collections import LineCollection

from features import add_features, END_FEATURES
from train_baseline import END_STATE_FEATURES
from train_deep import XTNet, DEVICE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")


def draw_pitch(ax):
    """Minimal top-down pitch outline, StatsBomb 120x80 coordinate system."""
    ax.set_facecolor("#0b3d0b")
    ax.plot([0, 0, 120, 120, 0], [0, 80, 80, 0, 0], color="white", lw=1.2)
    ax.plot([60, 60], [0, 80], color="white", lw=1)
    centre_circle = plt.Circle((60, 40), 9.15, color="white", fill=False, lw=1)
    ax.add_patch(centre_circle)
    # penalty boxes
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color="white", lw=1)
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color="white", lw=1)
    # six-yard boxes
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color="white", lw=1)
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color="white", lw=1)
    # goals
    ax.plot([0, -2, -2, 0], [36, 36, 44, 44], color="white", lw=1)
    ax.plot([120, 122, 122, 120], [36, 36, 44, 44], color="white", lw=1)
    ax.set_xlim(-3, 123)
    ax.set_ylim(-3, 83)
    ax.set_aspect("equal")
    ax.axis("off")


def build_grid_features(n=60):
    """Build a dense grid of (x, y) points covering the pitch, in END_STATE_FEATURES order."""
    xs = np.linspace(0, 120, n)
    ys = np.linspace(0, 80, n)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"end_x": xx.ravel(), "end_y": yy.ravel()})
    grid["end_dist_goal"] = np.sqrt((120 - grid["end_x"]) ** 2 + (40 - grid["end_y"]) ** 2)

    # reuse angle_to_goal from features.py
    from features import angle_to_goal
    grid["end_angle_goal"] = angle_to_goal(grid["end_x"].values, grid["end_y"].values)

    # assume a "typical" completed ground pass, not under pressure, for the surface plot
    grid["is_pass"] = 1
    grid["is_carry"] = 0
    grid["completed_int"] = 1
    grid["under_pressure_int"] = 0
    return xx, yy, grid[END_STATE_FEATURES]


def score_lgbm(grid_X):
    model = joblib.load(os.path.join(MODELS_DIR, "lgbm_baseline.pkl"))
    return model.predict_proba(grid_X)[:, 1]


def score_deep(grid_X):
    ckpt = torch.load(os.path.join(MODELS_DIR, "xtnet.pt"), map_location=DEVICE, weights_only=False)
    model = XTNet(n_features=len(ckpt["features"])).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = grid_X[ckpt["features"]].values.astype(np.float32)
    X = (X - ckpt["scaler_mean"].astype(np.float32)) / ckpt["scaler_scale"].astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X.astype(np.float32)).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def plot_surfaces():
    os.makedirs(FIG_DIR, exist_ok=True)
    xx, yy, grid_X = build_grid_features(n=80)

    p_lgbm = score_lgbm(grid_X).reshape(xx.shape)
    p_deep = score_deep(grid_X).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, surface, title in [
        (axes[0], p_lgbm, "LightGBM: P(goal | pass ends here)"),
        (axes[1], p_deep, "Neural net: P(goal | pass ends here)"),
    ]:
        draw_pitch(ax)
        im = ax.pcolormesh(xx, yy, surface, cmap="inferno", alpha=0.85, shading="auto",
                            vmin=0, vmax=max(p_lgbm.max(), p_deep.max()))
        ax.set_title(title, color="black", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle("Learned pitch-value surface — attacking left to right (World Cup 2022, StatsBomb open data)",
                 fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "xt_surface_comparison.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)
    return p_lgbm, p_deep


def plot_match_pass_values(match_id, team_name=None, exclude_positions=("Goalkeeper",)):
    """Overlay a real match's passes colored by learned threat added (xT delta).

    Goalkeeper actions are excluded from the leaderboard by default: keeper
    goal kicks and clearances often travel far downfield and get scored as
    "high value" purely by end location, which isn't a meaningful read of
    buildup play (a position-only xT model can't tell a controlled progressive
    pass from a hopeful long ball that happened to land forward).
    """
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "actions.parquet"))
    df = df[(df["match_id"] == match_id) & (df["type"] == "Pass") & (df["completed"] == True)]
    if team_name:
        df = df[df["team"] == team_name]
    df = add_features(df)

    ckpt = torch.load(os.path.join(MODELS_DIR, "xtnet.pt"), map_location=DEVICE, weights_only=False)
    model = XTNet(n_features=len(ckpt["features"])).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def score(sub_df, prefix):
        cols = {
            "end_x": sub_df[f"{prefix}_x"], "end_y": sub_df[f"{prefix}_y"],
            "end_dist_goal": sub_df[f"{prefix}_dist_goal"],
            "end_angle_goal": sub_df[f"{prefix}_angle_goal"],
            "is_pass": 1, "is_carry": 0, "completed_int": 1, "under_pressure_int": 0,
        }
        X = pd.DataFrame(cols)[ckpt["features"]].values.astype(np.float32)
        X = (X - ckpt["scaler_mean"].astype(np.float32)) / ckpt["scaler_scale"].astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X.astype(np.float32)).to(DEVICE))).cpu().numpy()
        return probs

    df["p_start"] = score(df, "start")
    df["p_end"] = score(df, "end")
    df["xt_added"] = df["p_end"] - df["p_start"]

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax)

    segments = np.stack([df[["start_x", "start_y"]].values, df[["end_x", "end_y"]].values], axis=1)
    vmax = df["xt_added"].abs().quantile(0.98)
    lc = LineCollection(segments, cmap="coolwarm", array=df["xt_added"].values,
                         linewidths=1.6, alpha=0.85, clim=(-vmax, vmax))
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Threat added by pass (xT delta)")

    top = df.nlargest(3, "xt_added")
    for _, row in top.iterrows():
        ax.annotate(f"{row['player']}", (row["end_x"], row["end_y"]),
                    color="white", fontsize=8, ha="center", va="bottom")

    title = f"Pass threat map — match {match_id}"
    if team_name:
        title += f" ({team_name})"
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"pass_value_map_{match_id}{'_' + team_name if team_name else ''}.png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)

    leaderboard = df.sort_values("xt_added", ascending=False)[
        ["minute", "player", "team", "position", "start_x", "start_y", "end_x", "end_y", "xt_added"]
    ]
    if exclude_positions:
        outfield = leaderboard[~leaderboard["position"].isin(exclude_positions)]
    else:
        outfield = leaderboard
    return leaderboard, outfield


if __name__ == "__main__":
    plot_surfaces()

    # Argentina vs France, World Cup 2022 Final -- 3-3, decided on penalties
    FINAL_MATCH_ID = 3869685
    all_passes, outfield_passes = plot_match_pass_values(match_id=FINAL_MATCH_ID)
    print("\nTop 10 (all positions, including goalkeepers):")
    print(all_passes.head(10).to_string(index=False))
    print("\nTop 10 (outfield players only):")
    print(outfield_passes.head(10).to_string(index=False))