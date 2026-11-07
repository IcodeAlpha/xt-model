"""
FastAPI service wrapping the trained xT (possession-value) and xG (shot-value)
models, with an interactive Swagger UI for testing.

Run:
    uvicorn api:app --reload

Then open:
    http://127.0.0.1:8000/docs      <- Swagger UI (try it out interactively)
    http://127.0.0.1:8000/redoc     <- alternative docs UI
"""
import os
import io
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import angle_to_goal, add_features
from train_baseline import END_STATE_FEATURES

try:
    import torch
    from train_deep import XTNet, DEVICE
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")


def _save_upload(filename, rows_scored, players):
    """Persist a scored upload to disk as JSON, so it survives page reloads
    and server restarts -- no database needed, just a file per upload."""
    import json as json_module
    import time
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    upload_id = f"{int(time.time() * 1000)}"
    record = {
        "id": upload_id,
        "filename": filename,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows_scored": rows_scored,
        "players": players,
    }
    with open(os.path.join(UPLOADS_DIR, f"{upload_id}.json"), "w") as f:
        json_module.dump(record, f)
    return upload_id


def _list_uploads():
    """Metadata only (no player data) -- keeps the list endpoint light."""
    import json as json_module
    if not os.path.exists(UPLOADS_DIR):
        return []
    records = []
    for fname in sorted(os.listdir(UPLOADS_DIR), reverse=True):  # newest first
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(UPLOADS_DIR, fname)) as f:
            r = json_module.load(f)
        records.append({
            "id": r["id"], "filename": r["filename"], "uploaded_at": r["uploaded_at"],
            "rows_scored": r["rows_scored"], "player_count": len(r["players"]),
        })
    return records

app = FastAPI(
    title="Expected Threat (xT) API",
    description=(
        "Score football passes/carries by how much they increase a team's "
        "chance of scoring, using models trained on 2022 World Cup StatsBomb "
        "open data. See README for methodology and cross-validated results."
    ),
    version="1.0.0",
)

# Allow the standalone HTML dashboard (opened via file:// or any local server)
# to call this API directly from the browser. Fine for local/dev use --
# tighten allow_origins before deploying this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
_lgbm_model = None
_deep_model = None
_deep_ckpt = None
_gamestate_model = None


@app.on_event("startup")
def load_models():
    global _lgbm_model, _deep_model, _deep_ckpt, _gamestate_model

    lgbm_path = os.path.join(MODELS_DIR, "lgbm_baseline.pkl")
    if os.path.exists(lgbm_path):
        _lgbm_model = joblib.load(lgbm_path)

    gamestate_path = os.path.join(MODELS_DIR, "lgbm_gamestate.pkl")
    if os.path.exists(gamestate_path):
        _gamestate_model = joblib.load(gamestate_path)

    deep_path = os.path.join(MODELS_DIR, "xtnet.pt")
    if TORCH_AVAILABLE and os.path.exists(deep_path):
        ckpt = torch.load(deep_path, map_location=DEVICE, weights_only=False)
        model = XTNet(n_features=len(ckpt["features"])).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _deep_model = model
        _deep_ckpt = ckpt


# ---------------------------------------------------------------------------
# Request / response schemas -- these drive the Swagger UI form fields
# ---------------------------------------------------------------------------
class PassInput(BaseModel):
    start_x: float = Field(..., ge=0, le=120, description="Start X on a 120-wide pitch (0 = own goal line)")
    start_y: float = Field(..., ge=0, le=80, description="Start Y on an 80-wide pitch")
    end_x: float = Field(..., ge=0, le=120, description="End X (120 = opponent's goal line)")
    end_y: float = Field(..., ge=0, le=80, description="End Y")
    is_carry: bool = Field(False, description="True for a carry, False for a pass")
    under_pressure: bool = Field(False, description="Was the player under pressure?")
    score_diff: int | None = Field(None, description="Goal difference for the team in possession (e.g. -1 if trailing by one). Omit to use the location-only model.")
    minutes_remaining: float | None = Field(None, description="Minutes left in the match. Omit to use the location-only model.")

    class Config:
        json_schema_extra = {
            "example": {
                "start_x": 80.0, "start_y": 40.0,
                "end_x": 110.0, "end_y": 40.0,
                "is_carry": False, "under_pressure": False,
                "score_diff": None, "minutes_remaining": None,
            }
        }


class XTResult(BaseModel):
    model: str
    p_start: float = Field(..., description="P(possession scores) at the start location")
    p_end: float = Field(..., description="P(possession scores) at the end location")
    xt_added: float = Field(..., description="Threat added by this action (p_end - p_start)")


class XTResponse(BaseModel):
    lgbm: XTResult | None = None
    deep: XTResult | None = None
    gamestate: XTResult | None = None


def _build_row(x, y, is_carry, under_pressure):
    dist_goal = float(np.sqrt((120 - x) ** 2 + (40 - y) ** 2))
    angle_goal = float(angle_to_goal(np.array([x]), np.array([y]))[0])
    return {
        "end_x": x, "end_y": y,
        "end_dist_goal": dist_goal, "end_angle_goal": angle_goal,
        "is_pass": 0 if is_carry else 1, "is_carry": 1 if is_carry else 0,
        "completed_int": 1, "under_pressure_int": int(under_pressure),
    }


def _score_lgbm(row):
    import pandas as pd
    X = pd.DataFrame([row])[END_STATE_FEATURES]
    return float(_lgbm_model.predict_proba(X)[0, 1])


def _score_gamestate(row, score_diff, minutes_remaining):
    import pandas as pd
    from train_gamestate import GAMESTATE_FEATURES
    full_row = {**row, "score_diff": score_diff, "minutes_remaining": minutes_remaining}
    X = pd.DataFrame([full_row])[GAMESTATE_FEATURES]
    return float(_gamestate_model.predict_proba(X)[0, 1])


def _score_deep(row):
    x = np.array([[row[f] for f in _deep_ckpt["features"]]], dtype=np.float32)
    x = (x - _deep_ckpt["scaler_mean"].astype(np.float32)) / _deep_ckpt["scaler_scale"].astype(np.float32)
    with torch.no_grad():
        logit = _deep_model(torch.tensor(x.astype(np.float32)).to(DEVICE))
        prob = torch.sigmoid(logit).item()
    return prob


def _parse_bool_column(series, column_name):
    """Parse common CSV boolean forms without treating the string "False" as true."""
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n", ""}

    def parse(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        raise HTTPException(
            status_code=400,
            detail=f"Invalid boolean value in '{column_name}': {value!r}. Use true/false or 1/0.",
        )

    return series.map(parse)


@app.post("/predict/xt", response_model=XTResponse, tags=["xT"])
def predict_xt(action: PassInput):
    """Score a pass or carry: returns the threat added (xT delta) by both
    the LightGBM baseline and the PyTorch deep learning model, so you can
    compare them side by side on the same input."""
    if _lgbm_model is None and _deep_model is None:
        raise HTTPException(status_code=503, detail="No trained models found in models/. Train them first.")

    start_row = _build_row(action.start_x, action.start_y, action.is_carry, action.under_pressure)
    end_row = _build_row(action.end_x, action.end_y, action.is_carry, action.under_pressure)

    response = {}
    if _lgbm_model is not None:
        p_start, p_end = _score_lgbm(start_row), _score_lgbm(end_row)
        response["lgbm"] = XTResult(model="lgbm_baseline", p_start=p_start, p_end=p_end,
                                     xt_added=p_end - p_start)
    if _deep_model is not None:
        p_start, p_end = _score_deep(start_row), _score_deep(end_row)
        response["deep"] = XTResult(model="xtnet_mlp", p_start=p_start, p_end=p_end,
                                     xt_added=p_end - p_start)

    if (_gamestate_model is not None and action.score_diff is not None
            and action.minutes_remaining is not None):
        p_start = _score_gamestate(start_row, action.score_diff, action.minutes_remaining)
        p_end = _score_gamestate(end_row, action.score_diff, action.minutes_remaining)
        response["gamestate"] = XTResult(model="lgbm_gamestate", p_start=p_start, p_end=p_end,
                                          xt_added=p_end - p_start)
    return response


@app.get("/health", tags=["meta"])
def health():
    """Simple health check -- also reports which models loaded successfully."""
    return {
        "status": "ok",
        "lgbm_loaded": _lgbm_model is not None,
        "deep_loaded": _deep_model is not None,
        "torch_available": TORCH_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Full tournament player rankings -- backs the leaderboard UI's "load all
# players" button, instead of the UI having to embed a hardcoded snapshot.
# ---------------------------------------------------------------------------
@app.get("/players/tournament", tags=["players"])
def players_tournament():
    """Every player ranked from the precomputed World Cup 2022 dataset
    (built by rank_players.py). Returns the same numbers as
    data/processed/player_xt_tournament.csv, as JSON."""
    path = os.path.join(PROCESSED_DIR, "player_xt_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_xt_tournament.csv not found -- run rank_players.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("total_xt_added", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/progressive", tags=["players"])
def players_progressive():
    """Progressive actions per player -- a rule-based metric independent of
    the xT model (see rank_progressive.py for the definition)."""
    path = os.path.join(PROCESSED_DIR, "player_progressive.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_progressive.csv not found -- run rank_progressive.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("progressive_actions", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/xa", tags=["players"])
def players_xa():
    """Expected Assists per player -- sum of predicted xG for every shot
    a player's pass directly led to (see build_xa.py)."""
    path = os.path.join(PROCESSED_DIR, "player_xa_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_xa_tournament.csv not found -- run build_xa.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("total_xa", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/defensive", tags=["players"])
def players_defensive():
    """Danger denied per player -- successful defensive actions scored by
    reusing the trained xT surface model at the mirrored location (see
    build_defensive.py for the methodology)."""
    path = os.path.join(PROCESSED_DIR, "player_defensive_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_defensive_tournament.csv not found -- run build_defensive.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("total_danger_denied", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/setpieces", tags=["players"])
def players_setpieces():
    """Set-piece delivery value per player -- corners and attacking-third
    free kicks scored by a dedicated model (see train_setpiece.py)."""
    path = os.path.join(PROCESSED_DIR, "player_setpiece_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_setpiece_tournament.csv not found -- run rank_setpieces.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("total_delivery_value", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/pressing", tags=["players"])
def players_pressing():
    """Pressing value added per player -- actual regains minus expected
    regains from a dedicated pressure-success model (see train_pressure.py,
    rank_pressing.py)."""
    path = os.path.join(PROCESSED_DIR, "player_pressing_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_pressing_tournament.csv not found -- run rank_pressing.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("pressing_value_added", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/pressresistance", tags=["players"])
def players_press_resistance():
    """Press-resistance per player -- completions above expected specifically
    on passes taken under pressure (see train_completion.py,
    rank_press_resistance.py)."""
    path = os.path.join(PROCESSED_DIR, "player_press_resistance_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_press_resistance_tournament.csv not found -- run rank_press_resistance.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("press_resistance", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/aerials", tags=["players"])
def players_aerials():
    """Aerial duel value added per player -- wins above expected from a
    location-based aerial win probability model (see train_aerial.py,
    rank_aerials.py). Winner side resolved via related_events, since
    StatsBomb only tags the loser directly."""
    path = os.path.join(PROCESSED_DIR, "player_aerial_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_aerial_tournament.csv not found -- run rank_aerials.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("aerial_value_added", ascending=False)
    return df.to_dict(orient="records")


@app.get("/players/dribbles", tags=["players"])
def players_dribbles():
    """Dribble/take-on value added per player -- completions above expected
    from a location-based dribble success model (see train_dribble.py,
    rank_dribbles.py)."""
    path = os.path.join(PROCESSED_DIR, "player_dribble_tournament.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                             detail="player_dribble_tournament.csv not found -- run rank_dribbles.py first.")
    df = pd.read_csv(path)
    df = df.sort_values("dribble_value_added", ascending=False)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Score completed passes/crosses/carries from a CSV of actions from ANY match/competition,
# including a full match event log produced by live_match_logger.html.
#
# Expected CSV columns:
#   required: start_x, start_y, end_x, end_y
#   optional: action (Pass, Cross, Carry), completed, player, team, is_carry (true/false),
#             under_pressure (true/false), minutes_played (per player, for per-90)
#
# This is what "run the model on different data" means in practice: export
# actions in this shape (e.g. from build_dataset.py's actions.parquet on a
# different StatsBomb competition, converted to CSV) and upload it here.
# ---------------------------------------------------------------------------
@app.post("/rank/csv", tags=["players"])
async def rank_from_csv(file: UploadFile = File(...)):
    """Upload action rows and rank players using completed passes, crosses and carries.

    Full event logs may include other action types and unsuccessful moves;
    those rows are retained by the logger but excluded from xT scoring.
    """
    if _lgbm_model is None:
        raise HTTPException(status_code=503, detail="LightGBM model not loaded.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required = {"start_x", "start_y", "end_x", "end_y"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400,
                             detail=f"CSV is missing required columns: {sorted(missing)}")

    if "action" in df.columns:
        action_types = df["action"].astype("string").str.strip().str.lower()
        completed = (
            _parse_bool_column(df["completed"], "completed")
            if "completed" in df.columns else pd.Series(True, index=df.index)
        )
        df = df.loc[action_types.isin({"pass", "cross", "carry"}) & completed].copy()
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no completed passes, crosses or carries to score.",
            )
        df["is_carry"] = action_types.loc[df.index].eq("carry")
    else:
        df["is_carry"] = _parse_bool_column(df["is_carry"], "is_carry") if "is_carry" in df.columns else False
    df["under_pressure"] = (
        _parse_bool_column(df["under_pressure"], "under_pressure")
        if "under_pressure" in df.columns else False
    )
    df["player"] = df["player"] if "player" in df.columns else "Unknown"
    df["team"] = df["team"] if "team" in df.columns else "Unknown"

    def score_side(prefix):
        x = df[f"{prefix}_x"].values
        y = df[f"{prefix}_y"].values
        dist_goal = np.sqrt((120 - x) ** 2 + (40 - y) ** 2)
        ang_goal = angle_to_goal(x, y)
        X = pd.DataFrame({
            "end_x": x, "end_y": y, "end_dist_goal": dist_goal, "end_angle_goal": ang_goal,
            "is_pass": (~df["is_carry"]).astype(int), "is_carry": df["is_carry"].astype(int),
            "completed_int": 1, "under_pressure_int": df["under_pressure"].astype(int),
        })[END_STATE_FEATURES]
        return _lgbm_model.predict_proba(X)[:, 1]

    df["xt_added"] = score_side("end") - score_side("start")

    grouped = (
        df.groupby(["player", "team"])
        .agg(total_xt_added=("xt_added", "sum"), total_actions=("xt_added", "count"))
        .reset_index()
        .sort_values("total_xt_added", ascending=False)
    )

    if "minutes_played" in df.columns:
        minutes = df.groupby("player")["minutes_played"].max()
        grouped["total_minutes"] = grouped["player"].map(minutes)
        grouped["xt_per90"] = grouped["total_xt_added"] / grouped["total_minutes"].clip(lower=1) * 90

    players_records = grouped.to_dict(orient="records")
    upload_id = _save_upload(file.filename, len(df), players_records)

    return {
        "upload_id": upload_id,
        "rows_scored": len(df),
        "players": players_records,
    }


@app.get("/uploads", tags=["uploads"])
def list_uploads():
    """List every past CSV upload (metadata only, not the full player data --
    use /uploads/{id} to fetch a specific one's results). Newest first."""
    return _list_uploads()


@app.get("/uploads/{upload_id}", tags=["uploads"])
def get_upload(upload_id: str):
    """Fetch a specific past upload's full scored results."""
    import json as json_module
    path = os.path.join(UPLOADS_DIR, f"{upload_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found.")
    with open(path) as f:
        return json_module.load(f)