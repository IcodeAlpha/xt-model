"""
Aerial duel dominance: StatsBomb only tags the LOSING side of an aerial
duel directly (Duel event, type "Aerial Lost"). The winner isn't tagged as
a duel at all -- they're recoverable via related_events, which link to
whatever the winning player did next (almost always a Pass, Clearance, or
Ball Receipt from the opposing team at essentially the same moment/location).

We resolve both sides of every detected aerial duel this way, so "win rate"
reflects genuine contested aerials, not just a count of losses.
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def process_match(events, match_id):
    by_id = {e["id"]: e for e in events}
    rows = []

    for e in events:
        if e["type"]["name"] != "Duel":
            continue
        if e.get("duel", {}).get("type", {}).get("name") != "Aerial Lost":
            continue

        loc = e.get("location")
        loser_team = e.get("team", {}).get("name")
        if not loc or not loser_team:
            continue

        # Find the winner: the first related event belonging to a DIFFERENT team.
        winner = None
        for rel_id in e.get("related_events", []):
            rel = by_id.get(rel_id)
            if rel and rel.get("team", {}).get("name") and rel["team"]["name"] != loser_team:
                winner = rel
                break

        # Loser's row
        rows.append({
            "match_id": match_id, "minute": e.get("minute"),
            "player": e.get("player", {}).get("name"), "team": loser_team,
            "x": loc[0], "y": loc[1], "won": 0,
        })
        # Winner's row, if resolved -- use the winner's own location if given, else the duel's location
        if winner is not None and winner.get("player", {}).get("name"):
            win_loc = winner.get("location", loc)
            rows.append({
                "match_id": match_id, "minute": e.get("minute"),
                "player": winner["player"]["name"], "team": winner["team"]["name"],
                "x": win_loc[0], "y": win_loc[1], "won": 1,
            })

    return rows


def build(out_name="aerials.parquet"):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "matches", "43_106.json"),
              encoding="utf-8") as f:
        wc2022_ids = {m["match_id"] for m in json.load(f)}

    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    files = [p for p in files if int(os.path.basename(p).replace(".json", "")) in wc2022_ids]

    all_rows = []
    for path in tqdm(files, desc="Resolving aerial duels"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(process_match(events, match_id))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} aerial duel sides from {len(files)} matches -> {out_path}")
    n_duels = df["won"].value_counts()
    print(f"Losses tagged directly: {n_duels.get(0,0)}  |  Winners resolved via related_events: {n_duels.get(1,0)}")
    unresolved = n_duels.get(0, 0) - n_duels.get(1, 0)
    print(f"Unresolved (no opposing related event found): {unresolved} "
          f"({unresolved / max(n_duels.get(0,1),1):.1%} of duels)")
    return df


if __name__ == "__main__":
    build()