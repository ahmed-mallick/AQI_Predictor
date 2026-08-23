import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import hopsworks

# TensorFlow sirf tab chahiye jab kisi horizon ka best model Neural Network ho.
# Isay lazy-import karte hain taake agar TF is machine par load na ho paye
# (jaisa Windows DLL issues mein hota hai), sklearn-based horizons phir bhi kaam karein.
_tf_keras = None
_tf_import_error = None


def get_keras():
    global _tf_keras, _tf_import_error
    if _tf_keras is None and _tf_import_error is None:
        try:
            from tensorflow import keras as _keras
            _tf_keras = _keras
        except Exception as e:
            _tf_import_error = e
    return _tf_keras

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

HAZARD_THRESHOLD = 150
AQI_FG_NAME = "karachi_aqi_features"
AQI_FG_VERSION = 1
WEATHER_FG_NAME = "karachi_weather_features"
WEATHER_FG_VERSION = 1

# Must match training_pipeline.py exactly, or predictions will silently use wrong columns.
FEATURE_COLS = [
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "co", "no2", "o3", "so2",
    "aqi_lag_1", "aqi_lag_3", "aqi_rolling_mean_3", "aqi_change_rate",
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
]

HORIZONS = {"24h": ("Tomorrow", 24), "48h": ("Day After", 48), "72h": ("In 3 Days", 72)}

# ---------- CUSTOM CSS (unchanged from original) ----------
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .main-header h1 { font-size: 2.6rem; margin-bottom: 0.2rem; }
    .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
    .current-aqi-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 20px; padding: 2.5rem; text-align: center; color: white; margin-bottom: 2rem;
    }
    .current-aqi-value { font-size: 5rem; font-weight: 800; line-height: 1; margin: 0.5rem 0; }
    .current-aqi-label { font-size: 1.3rem; font-weight: 600; opacity: 0.95; }
    .current-aqi-time { font-size: 0.85rem; opacity: 0.6; margin-top: 0.8rem; }
    .forecast-card { border-radius: 16px; padding: 1.5rem; text-align: center; color: white; height: 100%; }
    .forecast-day { font-size: 0.9rem; opacity: 0.85; font-weight: 600; }
    .forecast-value { font-size: 2.8rem; font-weight: 800; margin: 0.4rem 0; }
    .forecast-category { font-size: 1rem; font-weight: 500; }
    .alert-banner {
        background: #dc2626; color: white; padding: 1rem 1.5rem; border-radius: 12px;
        font-weight: 600; text-align: center; margin-bottom: 1.5rem;
    }
    .footer-note {
        text-align: center; color: #999; font-size: 0.8rem; margin-top: 3rem;
        padding-top: 1.5rem; border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def aqi_category(aqi):
    if aqi <= 50: return "Good", "#22c55e", "🟢"
    elif aqi <= 100: return "Moderate", "#eab308", "🟡"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#f97316", "🟠"
    elif aqi <= 200: return "Unhealthy", "#ef4444", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "#a855f7", "🟣"
    else: return "Hazardous", "#7f1d1d", "🟤"


# ---------- HOPSWORKS CONNECTION ----------
@st.cache_resource(ttl=3600)
def get_project():
    api_key = st.secrets["HOPSWORKS_API_KEY"]
    return hopsworks.login(api_key_value=api_key)


# ---------- DATA FETCH (mirrors training_pipeline.py) ----------
@st.cache_data(ttl=1800)
def fetch_raw_data():
    project = get_project()
    fs = project.get_feature_store()

    aqi_fg = fs.get_feature_group(name=AQI_FG_NAME, version=AQI_FG_VERSION)
    df_pollution = aqi_fg.read()

    weather_fg = fs.get_feature_group(name=WEATHER_FG_NAME, version=WEATHER_FG_VERSION)
    df_weather = weather_fg.read()

    df = pd.merge(df_pollution, df_weather, on="timestamp", how="inner", suffixes=("", "_weather"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def prepare_features(df):
    """Same feature engineering as training_pipeline.prepare_data, minus the target columns."""
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

    df = df.reset_index().rename(columns={"index": "datetime_utc"})
    return df


# ---------- MODEL LOADING (from Hopsworks Model Registry) ----------
@st.cache_resource(ttl=3600)
def load_model(horizon_name):
    project = get_project()
    mr = project.get_model_registry()

    models = mr.get_models(name=f"aqi_predictor_{horizon_name}")
    if not models:
        return None
    model_meta = max(models, key=lambda m: m.version)
    model_dir = model_meta.download()

    model_type = joblib.load(os.path.join(model_dir, "model_type.pkl"))
    if model_type["type"] == "nn":
        keras = get_keras()
        if keras is None:
            # TensorFlow load nahi ho saka is machine par — is horizon ko skip kar dein
            # bajaye poori app crash karne ke.
            return None
        nn_model = keras.models.load_model(os.path.join(model_dir, "nn_model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        return ("nn", nn_model, scaler)
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        return ("sklearn", model, None)


def predict_horizon(horizon_name, feature_row):
    bundle = load_model(horizon_name)
    if bundle is None:
        return None
    kind, model, scaler = bundle
    X = feature_row[FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
    if kind == "nn":
        X = scaler.transform(X)
        pred = model.predict(X, verbose=0).flatten()[0]
    else:
        pred = model.predict(X)[0]
    return round(float(pred))


# ---------- HEADER ----------
st.markdown("""
<div class="main-header"><h1>🌫️ Karachi AQI Predictor</h1></div>
<p class="subtitle">3-day Air Quality Index forecast, powered by a serverless ML pipeline</p>
""", unsafe_allow_html=True)

with st.spinner("Loading latest data from Hopsworks..."):
    try:
        raw_df = fetch_raw_data()
        df = prepare_features(raw_df)
    except Exception as e:
        st.error(f"Could not load AQI data from Hopsworks. ({e})")
        st.stop()

valid_df = df.dropna(subset=FEATURE_COLS + ["aqi"])
if valid_df.empty:
    st.error("Not enough data yet to show predictions. Please run the feature pipeline a few more times.")
    st.stop()

latest_row = valid_df.iloc[-1]
latest_aqi = int(latest_row["aqi"])
latest_time = pd.to_datetime(latest_row["datetime_utc"])
category, color, emoji = aqi_category(latest_aqi)

if latest_aqi > HAZARD_THRESHOLD:
    st.markdown(
        f'<div class="alert-banner">⚠️ ALERT: Current AQI ({latest_aqi}) is at hazardous levels. Take precautions!</div>',
        unsafe_allow_html=True,
    )

# ---------- CURRENT AQI CARD ----------
st.markdown(f"""
<div class="current-aqi-card" style="background: linear-gradient(135deg, {color}dd, {color}99);">
    <div class="current-aqi-label">{emoji} {category}</div>
    <div class="current-aqi-value">{latest_aqi}</div>
    <div class="current-aqi-label" style="font-size:1rem; opacity:0.8;">Current Air Quality Index</div>
    <div class="current-aqi-time">Last updated: {latest_time.strftime('%d %b %Y, %H:%M UTC')}</div>
</div>
""", unsafe_allow_html=True)

# ---------- FORECAST ----------
st.markdown("### 📅 3-Day Forecast")

predictions = {}
hazard_horizons = []

cols = st.columns(3)
for i, (horizon_name, (label, hours_ahead)) in enumerate(HORIZONS.items()):
    try:
        pred = predict_horizon(horizon_name, latest_row)
    except Exception:
        pred = None

    if pred is None:
        with cols[i]:
            st.info(f"{label} forecast will be available once more data is collected.")
        continue

    predictions[horizon_name] = pred
    if pred > HAZARD_THRESHOLD:
        hazard_horizons.append(label)
    cat, col_color, em = aqi_category(pred)

    with cols[i]:
        st.markdown(f"""
        <div class="forecast-card" style="background: linear-gradient(135deg, {col_color}dd, {col_color}99);">
            <div class="forecast-day">{label} ({horizon_name})</div>
            <div class="forecast-value">{pred}</div>
            <div class="forecast-category">{em} {cat}</div>
        </div>
        """, unsafe_allow_html=True)

if hazard_horizons:
    st.markdown(
        f'<div class="alert-banner">🚨 Forecast alert: hazardous AQI expected on {", ".join(hazard_horizons)}. '
        f'Plan outdoor activity accordingly.</div>',
        unsafe_allow_html=True,
    )

# ---------- HISTORICAL vs PREDICTED CHART ----------
st.markdown("### 📈 Historical vs Predicted AQI")

try:
    hist_df = valid_df.tail(72)[["datetime_utc", "aqi"]].rename(columns={"datetime_utc": "Time", "aqi": "AQI"})
    hist_df["Series"] = "Historical"

    if predictions:
        future_rows = []
        for horizon_name, (_, hours_ahead) in HORIZONS.items():
            if horizon_name in predictions:
                future_rows.append({
                    "Time": latest_time + pd.Timedelta(hours=hours_ahead),
                    "AQI": predictions[horizon_name],
                    "Series": "Predicted",
                })
        pred_df = pd.DataFrame(future_rows)
        bridge = pd.DataFrame([{"Time": latest_time, "AQI": latest_aqi, "Series": "Predicted"}])
        chart_df = pd.concat([hist_df, bridge, pred_df], ignore_index=True)
    else:
        chart_df = hist_df

    hazard_rule = alt.Chart(pd.DataFrame({"y": [HAZARD_THRESHOLD]})).mark_rule(
        color="#dc2626", strokeDash=[6, 4]
    ).encode(y="y")

    line_chart = alt.Chart(chart_df).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("Time:T", title=None),
        y=alt.Y("AQI:Q", title="AQI"),
        color=alt.Color("Series:N", scale=alt.Scale(
            domain=["Historical", "Predicted"], range=["#3b82f6", "#f97316"])),
        strokeDash=alt.condition(alt.datum.Series == "Predicted", alt.value([5, 5]), alt.value([0])),
        tooltip=["Time:T", "AQI:Q", "Series:N"],
    ).properties(height=340)

    st.altair_chart((line_chart + hazard_rule).interactive(), use_container_width=True)
    st.caption(f"Dashed red line marks the hazardous AQI threshold ({HAZARD_THRESHOLD}).")
except Exception as e:
    st.info(f"Historical trend chart unavailable: {e}")

# ---------- POLLUTANT BREAKDOWN ----------
st.markdown("### 🧪 Current Pollutant Levels")
p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("PM2.5", f"{latest_row['pm2_5']:.1f}")
p2.metric("PM10", f"{latest_row['pm10']:.1f}")
p3.metric("CO", f"{latest_row['co']:.1f}")
p4.metric("NO2", f"{latest_row['no2']:.2f}")
p5.metric("O3", f"{latest_row['o3']:.1f}")
p6.metric("SO2", f"{latest_row['so2']:.2f}")

st.markdown("""
<div class="footer-note">
    Data source: OpenWeather Air Pollution API &nbsp;|&nbsp; Model Registry: Hopsworks &nbsp;|&nbsp; Auto-updated hourly
</div>
""", unsafe_allow_html=True)