import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- CONFIG (Secrets se aayenge) ----------
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

# ---------- STEP 1: Feature Store se data nikalo ----------
def fetch_training_data():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df = aqi_fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, project, mr

# ---------- STEP 2: Feature Engineering ----------
def prepare_data(df):
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()

    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    return df

# ---------- STEP 3: Ek horizon ke liye train karo ----------
def train_for_horizon(df, target_col, horizon_name):
    feature_cols = [
        "hour", "day", "month", "day_of_week",
        "pm2_5", "pm10", "co", "no2", "o3", "so2",
        "aqi_lag_1", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change_rate"
    ]

    data = df.dropna(subset=feature_cols + [target_col])

    if len(data) < 8:
        print(f"WARNING: {horizon_name} - sirf {len(data)} usable rows hain, skip kar rahe hain")
        return None, None

    X = data[feature_cols]
    y = data[target_col]

    split_idx = max(1, int(len(data) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    best_model, best_name, best_rmse = None, None, float("inf")

    print(f"\n--- {horizon_name} Forecast ---")
    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")

        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name

    print(f"Best for {horizon_name}: {best_name}")
    return best_model, best_name

# ---------- STEP 4: Model Registry mein save karo ----------
def save_model_to_registry(model, mr, horizon_name):
    if model is None:
        return

    model_dir = f"model_{horizon_name}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")

    mr_model = mr.python.create_model(
        name=f"aqi_predictor_{horizon_name}",
        description=f"AQI prediction model for {horizon_name} horizon"
    )
    mr_model.save(model_dir)
    print(f"Saved: {horizon_name} model to Model Registry")

# ---------- RUN ----------
if __name__ == "__main__":
    df, project, mr = fetch_training_data()
    print(f"Total rows fetched: {len(df)}")

    df = prepare_data(df)

    horizons = {
        "24h": "target_24h",
        "48h": "target_48h",
        "72h": "target_72h",
    }

    for horizon_name, target_col in horizons.items():
        model, model_name = train_for_horizon(df, target_col, horizon_name)
        save_model_to_registry(model, mr, horizon_name)