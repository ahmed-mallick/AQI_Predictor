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

# ---------- STEP 3: Neural Network banao ----------
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

# ---------- STEP 4: Train aur Compare (3 models) ----------
def train_for_horizon(df, target_col, horizon_name):
    feature_cols = [
        "hour", "day", "month", "day_of_week",
        "pm2_5", "pm10", "co", "no2", "o3", "so2",
        "aqi_lag_1", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change_rate"
    ]

    data = df.dropna(subset=feature_cols + [target_col])

    if len(data) < 10:
        print(f"WARNING: {horizon_name} - sirf {len(data)} usable rows hain, skip kar rahe hain")
        return None, None, None

    X = data[feature_cols]
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
        "RandomForest (ML)": RandomForestRegressor(n_estimators=100, random_state=42),
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
        epochs=100, batch_size=8, verbose=0,
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

# ---------- STEP 5: Model Registry mein save karo ----------
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
        description=f"Best AQI prediction model for {horizon_name} horizon"
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
        model, model_name, is_nn = train_for_horizon(df, target_col, horizon_name)
        save_model_to_registry(model, mr, horizon_name, is_nn)