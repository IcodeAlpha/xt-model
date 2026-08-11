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
import numpy as np
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import angle_to_goal
from train_baseline import END_STATE_FEATURES
from train_deep import XTNet, DEVICE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(
    title="Expected Threat (xT) API",
    description=(
        "Score football passes/carries by how much they increase a team's "
        "chance of scoring, using models trained on 2022 World Cup StatsBomb "
        "open data. See README for methodology and cross-validated results."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
_lgbm_model = None
_deep_model = None
_deep_ckpt = None


@app.on_event("startup")
def load_models():
    global _lgbm_model, _deep_model, _deep_ckpt

    lgbm_path = os.path.join(MODELS_DIR, "lgbm_baseline.pkl")
    if os.path.exists(lgbm_path):
        _lgbm_model = joblib.load(lgbm_path)

    deep_path = os.path.join(MODELS_DIR, "xtnet.pt")
    if os.path.exists(deep_path):
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

    class Config:
        json_schema_extra = {
            "example": {
                "start_x": 80.0, "start_y": 40.0,
                "end_x": 110.0, "end_y": 40.0,
                "is_carry": False, "under_pressure": False,
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


def _score_deep(row):
    x = np.array([[row[f] for f in _deep_ckpt["features"]]], dtype=np.float32)
    x = (x - _deep_ckpt["scaler_mean"].astype(np.float32)) / _deep_ckpt["scaler_scale"].astype(np.float32)
    with torch.no_grad():
        logit = _deep_model(torch.tensor(x.astype(np.float32)).to(DEVICE))
        prob = torch.sigmoid(logit).item()
    return prob


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
    return response


@app.get("/health", tags=["meta"])
def health():
    """Simple health check -- also reports which models loaded successfully."""
    return {
        "status": "ok",
        "lgbm_loaded": _lgbm_model is not None,
        "deep_loaded": _deep_model is not None,
    }