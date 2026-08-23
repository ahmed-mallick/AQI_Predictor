import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

# ---------- CONFIG ----------
HOPSWORKS_API_KEY = "wk2WQDxtwVmRTrM6.Jel2VY74EDLPU3vdGGJ7VVSG9fJEzaL6NpiWhb5AfdlpxcE5mvMGQ0QP5sdBuIGQ"
WEATHER_FG_NAME = "karachi_weather_features"
WEATHER_FG_VERSION = 1

# ---------- STEP 1: Dono Feature Groups Se Data Fetch Aur Join Karo ----------
def fetch_training_data():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df_pollution = aqi_fg.read()

    weather_fg = fs.get_feature_group(name=WEATHER_FG_NAME, version=WEATHER_FG_VERSION)
    df_weather = weather_fg.read()

    # timestamp pe join karo
    df = pd.merge(df_pollution, df_weather, on="timestamp", how="inner", suffixes=("", "_weather"))
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df, project, mr

# ---------- STEP 2: Feature Engineering (Timeline Fix + Weather Included) ----------
def prepare_data(df):
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)

    numeric_cols = ["pm2_5", "pm10", "co", "no2", "o3", "so2", "aqi",
                     "temperature", "humidity", "pressure", "wind_speed", "clouds"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=6)

    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_rolling_mean_3"] = df["aqi"].rolling(window=3).mean()
    df["aqi_change_rate"] = df["aqi"].diff()

    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    df = df.reset_index().rename(columns={"index": "datetime_utc"})
    return df

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "o3", "so2",
    "aqi_lag_1", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change_rate",
    "temperature", "humidity", "pressure", "wind_speed", "clouds",  # 👈 NAYE WEATHER FEATURES
]

# ---------- STEP 3: Neural Network ----------
def build_nn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ---------- STEP 4: Train aur Compare ----------
def train_for_horizon(df, target_col, horizon_name):
    data = df.dropna(subset=FEATURE_COLS + [target_col])

    if len(data) < 10:
        print(f"WARNING: {horizon_name} - sirf {len(data)} usable rows hain, skip kar rahe hain")
        return None, None, None

    X = data[FEATURE_COLS]
    y = data[target_col]

    split_idx = max(1, int(len(data) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model, best_name, best_rmse = None, None, float("inf")
    best_is_nn = False

    print(f"\n--- {horizon_name} Forecast ---")
    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    sklearn_models = {
        "Ridge (Statistical)": Ridge(alpha=1.0),
        "RandomForest (ML)": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }

    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")

        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name
            best_is_nn = False

    nn_model = build_nn_model(X_train_scaled.shape[1])
    nn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.1 if len(X_train) > 10 else 0,
        epochs=100, batch_size=32, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )
    nn_preds = nn_model.predict(X_test_scaled, verbose=0).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
    mae = mean_absolute_error(y_test, nn_preds)
    r2 = r2_score(y_test, nn_preds) if len(y_test) > 1 else float("nan")

    print(f"Neural Network (Deep Learning) -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

    if rmse < best_rmse:
        best_rmse, best_model, best_name = rmse, (nn_model, scaler), "Neural Network (Deep Learning)"
        best_is_nn = True

    print(f"Best for {horizon_name}: {best_name}")
    return best_model, best_name, best_is_nn

# ---------- STEP 5: Model Registry Mein Save ----------
def save_model_to_registry(model, mr, horizon_name, is_nn):
    if model is None:
        return

    model_dir = f"model_{horizon_name}"
    os.makedirs(model_dir, exist_ok=True)

    if is_nn:
        nn_model, scaler = model
        nn_model.save(f"{model_dir}/nn_model.keras")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
        joblib.dump({"type": "nn"}, f"{model_dir}/model_type.pkl")
    else:
        joblib.dump(model, f"{model_dir}/model.pkl")
        joblib.dump({"type": "sklearn"}, f"{model_dir}/model_type.pkl")

    mr_model = mr.python.create_model(
        name=f"aqi_predictor_{horizon_name}",
        description=f"Best AQI prediction model for {horizon_name} horizon (with weather features)"
    )
    mr_model.save(model_dir)
    print(f"Saved: {horizon_name} model to Model Registry")

# ---------- RUN ----------
if __name__ == "__main__":
    df, project, mr = fetch_training_data()
    print(f"Total rows fetched: {len(df)}")

    df = prepare_data(df)

    horizons = {"24h": "target_24h", "48h": "target_48h", "72h": "target_72h"}

    for horizon_name, target_col in horizons.items():
        model, model_name, is_nn = train_for_horizon(df, target_col, horizon_name)
        save_model_to_registry(model, mr, horizon_name, is_nn)