"""
Runs the entire xT project pipeline end to end, in the correct dependency
order, with clear progress logging.

Usage:
    python run_all.py                  # run everything not already done
    python run_all.py --force          # rerun every step, ignoring existing outputs
    python run_all.py --only build_xa  # run just one step (and check its deps exist)
    python run_all.py --list           # show the full step list and exit
    python run_all.py --with-optional  # also run the optional/heavier steps
                                        # (deep learning, multi-task, sequence model,
                                        #  figures) if those scripts are present

Each step is skipped automatically if its output file(s) already exist --
delete a file (or use --force) to force that step to rerun. Steps run in
order and the pipeline stops on the first failure, since everything after
depends on what came before.
"""
import argparse
import os
import subprocess
import sys
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
DATA = os.path.join(ROOT_DIR, "data", "processed")
RAW_MATCHES = os.path.join(ROOT_DIR, "data", "raw", "matches", "43_106.json")
MODELS = os.path.join(ROOT_DIR, "models")

# Each step: (name, script, [output files that mean "already done"], optional)
# Order matters -- this is real dependency order, not alphabetical.
STEPS = [
    ("download_data",          "download_data.py",          [RAW_MATCHES],                                  False),
    ("build_dataset",          "build_dataset.py",          [f"{DATA}/actions.parquet"],                    False),
    ("build_player_minutes",   "build_player_minutes.py",   [f"{DATA}/player_minutes.parquet"],             False),
    ("build_shots_dataset",    "build_shots_dataset.py",    [f"{DATA}/shots.parquet"],                      False),
    ("train_baseline",         "train_baseline.py",         [f"{MODELS}/lgbm_baseline.pkl"],                False),
    ("train_xg",                "train_xg.py",               [f"{MODELS}/xg_model.pkl"],                     False),
    ("build_xa",                "build_xa.py",                [f"{DATA}/player_xa_tournament.csv"],           False),
    ("build_defensive",         "build_defensive.py",         [f"{DATA}/player_defensive_tournament.csv"],    False),
    ("build_setpieces",         "build_setpieces.py",         [f"{DATA}/setpieces.parquet"],                  False),
    ("train_setpiece",          "train_setpiece.py",          [f"{MODELS}/setpiece_model.pkl"],               False),
    ("rank_setpieces",          "rank_setpieces.py",          [f"{DATA}/player_setpiece_tournament.csv"],     False),
    ("build_gamestate",         "build_gamestate.py",         [f"{DATA}/gamestate.parquet"],                  False),
    ("train_gamestate",         "train_gamestate.py",         [f"{MODELS}/lgbm_gamestate.pkl"],               False),
    ("build_pressure",          "build_pressure.py",          [f"{DATA}/pressures.parquet"],                  False),
    ("train_pressure",          "train_pressure.py",          [f"{MODELS}/pressure_model.pkl"],               False),
    ("rank_pressing",           "rank_pressing.py",           [f"{DATA}/player_pressing_tournament.csv"],     False),
    ("train_completion",        "train_completion.py",        [f"{MODELS}/completion_model.pkl"],             False),
    ("rank_press_resistance",   "rank_press_resistance.py",   [f"{DATA}/player_press_resistance_tournament.csv"], False),
    ("build_aerials",           "build_aerials.py",           [f"{DATA}/aerials.parquet"],                    False),
    ("train_aerial",            "train_aerial.py",            [f"{MODELS}/aerial_model.pkl"],                 False),
    ("rank_aerials",             "rank_aerials.py",             [f"{DATA}/player_aerial_tournament.csv"],       False),
    ("build_dribbles",           "build_dribbles.py",           [f"{DATA}/dribbles.parquet"],                   False),
    ("train_dribble",            "train_dribble.py",            [f"{MODELS}/dribble_model.pkl"],                False),
    ("rank_dribbles",             "rank_dribbles.py",             [f"{DATA}/player_dribble_tournament.csv"],      False),
    ("rank_players",              "rank_players.py",              [f"{DATA}/player_xt_tournament.csv"],           False),
    ("rank_progressive",          "rank_progressive.py",          [f"{DATA}/player_progressive.csv"],             False),
    # Optional: heavier / require torch / not needed for the API or leaderboard,
    # only run with --with-optional and only if the script is actually present
    # (these were built in earlier phases and may not be in every checkout).
    ("train_deep",                "train_deep.py",                [f"{MODELS}/xtnet.pt"],                         True),
    ("train_multitask",           "train_multitask.py",           [],                                              True),
    ("train_sequence",            "train_sequence.py",            [],                                              True),
    ("visualize",                  "visualize.py",                  [f"{ROOT_DIR}/figures/xt_surface_comparison.png"], True),
]


def outputs_exist(paths):
    return len(paths) > 0 and all(os.path.exists(p) for p in paths)


def run_step(name, script, force):
    script_path = os.path.join(SRC_DIR, script)
    if not os.path.exists(script_path):
        print(f"  [SKIP] {script} not found in src/ -- skipping this optional step.")
        return "skipped-missing"

    print(f"\n{'='*70}\n  {name}  ({script})\n{'='*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=SRC_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode} after {elapsed:.1f}s")
        return "failed"

    print(f"[OK] {script} finished in {elapsed:.1f}s")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Rerun every step, ignoring existing outputs.")
    parser.add_argument("--only", metavar="STEP", help="Run only this one step by name (see --list).")
    parser.add_argument("--list", action="store_true", help="List all steps and exit.")
    parser.add_argument("--with-optional", action="store_true",
                         help="Also run optional steps (deep learning, multi-task, sequence, figures).")
    args = parser.parse_args()

    if args.list:
        print("Pipeline steps, in order:\n")
        for name, script, _, optional in STEPS:
            tag = " (optional)" if optional else ""
            print(f"  {name:<24} {script}{tag}")
        return

    steps_to_run = STEPS
    if args.only:
        matching = [s for s in STEPS if s[0] == args.only]
        if not matching:
            print(f"Unknown step '{args.only}'. Run with --list to see valid step names.")
            sys.exit(1)
        steps_to_run = matching
    elif not args.with_optional:
        steps_to_run = [s for s in STEPS if not s[3]]

    print(f"Running {len(steps_to_run)} step(s)...")
    t_start = time.time()
    n_ok, n_skipped, n_failed = 0, 0, 0

    for name, script, check_files, optional in steps_to_run:
        if not args.force and outputs_exist(check_files):
            print(f"\n[SKIP] {name}: output already exists ({check_files[0]})")
            n_skipped += 1
            continue

        status = run_step(name, script, args.force)
        if status == "ok":
            n_ok += 1
        elif status == "skipped-missing":
            n_skipped += 1
        else:
            n_failed += 1
            print(f"\nStopping pipeline: '{name}' failed. Fix the error above and rerun --")
            print(f"progress so far is preserved, so rerunning run_all.py will pick up where this left off.")
            break

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {elapsed/60:.1f} min. {n_ok} ran, {n_skipped} skipped, {n_failed} failed.")
    print(f"{'='*70}")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()