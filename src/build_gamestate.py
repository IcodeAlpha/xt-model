"""
Compute game state (score differential, minutes remaining) at the moment of
every pass/carry action, so we can test whether the SAME pass is more or
less valuable depending on context -- e.g. a raw distance-to-goal pass late
while trailing plausibly carries more urgency/risk-taking than the
identical pass at 0-0 in minute 10.

Score tracking: goals come from Shot events with outcome == "Goal", plus
"Own Goal For" events (which credit the benefiting team directly -- see
build_gamestate.py's own investigation: "Own Goal Against" is a paired,
redundant event for the conceding team and must NOT also be counted, or
every own goal gets double-counted).
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def compute_gamestate(events, match_id):
    full_time = max(
        (e["minute"] for e in events if e["type"]["name"] == "Half End" and e["period"] <= 4),
        default=90,
    )

    teams = set()
    for e in events:
        t = e.get("team", {}).get("name")
        if t:
            teams.add(t)
    score = {t: 0 for t in teams}

    rows = []
    for e in events:
        etype = e["type"]["name"]
        team = e.get("team", {}).get("name")

        # Record game state BEFORE applying this event's own goal (so a
        # scoring action's game-state reflects the state it was taken in,
        # not after it changed the score).
        if etype in ("Pass", "Carry"):
            opponent_score = sum(v for t, v in score.items() if t != team)
            own_score = score.get(team, 0)
            rows.append({
                "match_id": match_id,
                "index": e.get("index"),
                "score_diff": own_score - opponent_score,
                "minutes_remaining": max(0, full_time - (e.get("minute") or 0)),
            })

        # Now apply any goal this event caused, so subsequent actions see the updated score.
        if etype == "Shot":
            outcome = e.get("shot", {}).get("outcome", {}).get("name")
            if outcome == "Goal" and team:
                score[team] = score.get(team, 0) + 1
        elif etype == "Own Goal For" and team:
            score[team] = score.get(team, 0) + 1

    return rows


def build(out_name="gamestate.parquet"):
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Computing game state"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(compute_gamestate(events, match_id))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} action-level game states -> {out_path}")
    print(df["score_diff"].value_counts().sort_index())
    return df


if __name__ == "__main__":
    build()