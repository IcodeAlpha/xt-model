"""
Feature engineering for the Expected Threat (xT) model.

Pitch is 120 (x) x 80 (y). Goal is centered on the x=120 line, y=40,
with goal mouth width of ~7.32m mapped to roughly y in [36, 44].
"""
import numpy as np
import pandas as pd

GOAL_X, GOAL_Y = 120.0, 40.0


def dist_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def angle_to_goal(x, y):
    # angle subtended by the goal mouth from (x, y) -- classic shot-angle feature
    import numpy as np
    post1 = np.array([GOAL_X, 36.0])
    post2 = np.array([GOAL_X, 44.0])
    p = np.stack([x, y], axis=-1)
    v1 = post1 - p
    v2 = post2 - p
    dot = (v1 * v2).sum(-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos_a = np.clip(dot / (n1 * n2 + 1e-9), -1, 1)
    return np.arccos(cos_a)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_dist_goal"] = dist_to_goal(df["start_x"], df["start_y"])
    df["end_dist_goal"] = dist_to_goal(df["end_x"], df["end_y"])
    df["start_angle_goal"] = angle_to_goal(df["start_x"].values, df["start_y"].values)
    df["end_angle_goal"] = angle_to_goal(df["end_x"].values, df["end_y"].values)
    df["dx"] = df["end_x"] - df["start_x"]
    df["dy"] = df["end_y"] - df["start_y"]
    df["move_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["is_pass"] = (df["type"] == "Pass").astype(int)
    df["is_carry"] = (df["type"] == "Carry").astype(int)
    df["completed_int"] = df["completed"].astype(int)
    df["under_pressure_int"] = df["under_pressure"].astype(int)
    # progressiveness: fraction of remaining distance to goal covered
    df["progress_frac"] = (df["start_dist_goal"] - df["end_dist_goal"]) / (
        df["start_dist_goal"] + 1e-6
    )
    return df


START_FEATURES = ["start_x", "start_y", "start_dist_goal", "start_angle_goal"]
END_FEATURES = ["end_x", "end_y", "end_dist_goal", "end_angle_goal"]
CONTEXT_FEATURES = [
    "dx", "dy", "move_dist", "is_pass", "is_carry",
    "completed_int", "under_pressure_int", "progress_frac",
]

ALL_FEATURES = START_FEATURES + END_FEATURES + CONTEXT_FEATURES
TARGET = "led_to_goal"


def train_val_test_split_by_match(df: pd.DataFrame, seed=42):
    """Split by match_id (not by row!) so we never leak the same match across splits."""
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)

    n = len(match_ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_ids = set(match_ids[:n_train])
    val_ids = set(match_ids[n_train:n_train + n_val])
    test_ids = set(match_ids[n_train + n_val:])

    train_df = df[df["match_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["match_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["match_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df
