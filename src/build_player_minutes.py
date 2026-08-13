"""
Compute minutes played per player per match, needed to rank players by
xT-added PER 90 MINUTES rather than raw totals (which would just reward
players who happened to play more matches/minutes).

Logic:
  - Starting XI event -> player starts at minute 0
  - Substitution event -> outgoing player's minutes end at sub minute,
                           incoming player (`substitution.replacement`) starts then
  - Full-time minute = max "Half End" minute across periods 1-4
    (period 5 is a penalty shootout, not playing time, so it's excluded)
  - A player who starts and is never subbed off plays until full-time minute
  - Red cards (Bad Behaviour / sent off) aren't in this open-data schema as a
    distinct stoppage event in all seasons, so we don't model early red-card
    exits here -- documented as a known simplification below.
"""
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "events")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def compute_match_minutes(events, match_id):
    full_time = max(
        (e["minute"] for e in events if e["type"]["name"] == "Half End" and e["period"] <= 4),
        default=90,
    )

    # player_id -> {"name":..., "team":..., "on": minute, "off": minute or None}
    spells = {}

    for e in events:
        if e["type"]["name"] == "Starting XI":
            team_name = e["team"]["name"]
            for entry in e.get("tactics", {}).get("lineup", []):
                p = entry["player"]
                spells.setdefault(p["id"], []).append(
                    {"name": p["name"], "team": team_name, "on": 0, "off": None}
                )

        elif e["type"]["name"] == "Substitution":
            team_name = e["team"]["name"]
            off_player = e["player"]
            sub = e.get("substitution", {})
            on_player = sub.get("replacement")
            minute = e["minute"]

            # close the outgoing player's spell
            if off_player["id"] in spells:
                for spell in spells[off_player["id"]]:
                    if spell["off"] is None:
                        spell["off"] = minute
                        break

            # open a new spell for the incoming player
            if on_player:
                spells.setdefault(on_player["id"], []).append(
                    {"name": on_player["name"], "team": team_name, "on": minute, "off": None}
                )

    rows = []
    for player_id, player_spells in spells.items():
        total_minutes = 0
        for spell in player_spells:
            end = spell["off"] if spell["off"] is not None else full_time
            total_minutes += max(0, end - spell["on"])
        rows.append({
            "match_id": match_id,
            "player": player_spells[0]["name"],
            "team": player_spells[0]["team"],
            "minutes_played": total_minutes,
        })
    return rows


def build(out_name="player_minutes.parquet"):
    files = sorted(glob.glob(os.path.join(RAW_EVENTS_DIR, "*.json")))
    all_rows = []
    for path in tqdm(files, desc="Computing minutes"):
        match_id = int(os.path.basename(path).replace(".json", ""))
        with open(path, encoding="utf-8") as f:
            events = json.load(f)
        all_rows.extend(compute_match_minutes(events, match_id))

    df = pd.DataFrame(all_rows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, out_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} player-match minute records from {len(files)} matches -> {out_path}")
    return df


if __name__ == "__main__":
    build()