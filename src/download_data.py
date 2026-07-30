"""
Download StatsBomb open-data event files for a given competition/season.
Source: https://github.com/statsbomb/open-data
"""
import json
import os
import time
import requests
from tqdm import tqdm

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_events(competition_id: int, season_id: int):
    matches_dir = os.path.join(RAW_DIR, "matches")
    os.makedirs(matches_dir, exist_ok=True)

    matches_path = os.path.join(matches_dir, f"{competition_id}_{season_id}.json")

    # Download matches list if missing
    if not os.path.exists(matches_path):
        url = f"{BASE}/matches/{competition_id}/{season_id}.json"
        print(f"Downloading matches from {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(matches_path, "wb") as f:
            f.write(resp.content)

    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    events_dir = os.path.join(RAW_DIR, "events")
    os.makedirs(events_dir, exist_ok=True)

    match_ids = [m["match_id"] for m in matches]
    print(f"Downloading events for {len(match_ids)} matches...")

    for match_id in tqdm(match_ids):
        out_path = os.path.join(events_dir, f"{match_id}.json")
        if os.path.exists(out_path):
            continue
        url = f"{BASE}/events/{match_id}.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        time.sleep(0.05)  # be polite to raw.githubusercontent.com

    return match_ids


if __name__ == "__main__":
    # FIFA World Cup 2022
    download_events(competition_id=43, season_id=106)
    print("Done.")