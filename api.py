import os
import hopsworks
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AQI Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

_project = None
_models_cache = {}

def get_project():
    global _project
    if _project is None:
        _project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    return _project

def get_model(horizon_name):
    """Model download karke uska type detect karta hai (sklearn ya NN), aur predict-ready object return karta hai"""
    if horizon_name not in _models_cache:
        project = get_project()
        mr = project.get_model_registry()
        model_obj = mr.get_model(name=f"aqi_predictor_{horizon_name}")
        model_dir = model_obj.download()

        # Model type check karo
        model_type_path = f"{model_dir}/model_type.pkl"
        if os.path.exists(model_type_path):
            model_type_info = joblib.load(model_type_path)
            model_type = model_type_info.get("type", "sklearn")
        else:
            model_type = "sklearn"  # purane models ke liye fallback

        if model_type == "nn":
            import tensorflow as tf
            nn_model = tf.keras.models.load_model(f"{model_dir}/nn_model.keras")
            scaler = joblib.load(f"{model_dir}/scaler.pkl")
            _models_cache[horizon_name] = {"type": "nn", "model": nn_model, "scaler": scaler}
        else:
            sklearn_model = joblib.load(f"{model_dir}/model.pkl")
            _models_cache[horizon_name] = {"type": "sklearn", "model": sklearn_model}

    return _models_cache[horizon_name]

def predict_with_model(model_entry, X):
    """Model type ke hisaab se sahi predict method use karta hai"""
    if model_entry["type"] == "nn":
        X_scaled = model_entry["scaler"].transform(X)
        pred = model_entry["model"].predict(X_scaled, verbose=0).flatten()[0]
    else:
        pred = model_entry["model"].predict(X)[0]
    return float(pred)

def get_latest_features():
    project = get_project()
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df = aqi_fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()

    return df

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "o3", "so2",
    "aqi_lag_1", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change_rate"
]

@app.get("/")
def root():
    return {"status": "AQI Predictor API is running"}

@app.get("/current")
def current_aqi():
    df = get_latest_features()
    latest = df.iloc[-1]
    return {
        "aqi": int(latest["aqi"]),
        "pm2_5": float(latest["pm2_5"]),
        "pm10": float(latest["pm10"]),
        "co": float(latest["co"]),
        "no2": float(latest["no2"]),
        "o3": float(latest["o3"]),
        "so2": float(latest["so2"]),
        "datetime_utc": str(latest["datetime_utc"]),
    }

@app.get("/predict/{horizon}")
def predict(horizon: str):
    if horizon not in ["24h", "48h", "72h"]:
        raise HTTPException(status_code=400, detail="Horizon must be 24h, 48h, or 72h")

    df = get_latest_features()
    latest = df.iloc[-1]
    X = latest[FEATURE_COLS].to_frame().T.astype(float)

    if X.isnull().values.any():
        raise HTTPException(status_code=503, detail="Not enough historical data yet")

    try:
        model_entry = get_model(horizon)
        pred = round(predict_with_model(model_entry, X))
        return {"horizon": horizon, "predicted_aqi": pred, "model_type": model_entry["type"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")