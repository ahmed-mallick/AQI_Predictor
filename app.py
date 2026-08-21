import os
import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

API_URL = os.environ.get("API_URL", "https://aqi-predictor-r4zk.onrender.com")
HAZARD_THRESHOLD = 150

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


@st.cache_data(ttl=1800)
def get_current_aqi():
    r = requests.get(f"{API_URL}/current", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800)
def get_history(hours=72):
    r = requests.get(f"{API_URL}/history", params={"hours": hours}, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800)
def get_predictions_all():
    """One round trip for all 3 horizons instead of 3 separate calls."""
    r = requests.get(f"{API_URL}/predict_all", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- HEADER ----------
st.markdown("""
<div class="main-header"><h1>🌫️ Karachi AQI Predictor</h1></div>
<p class="subtitle">3-day Air Quality Index forecast, powered by a serverless ML pipeline</p>
""", unsafe_allow_html=True)

with st.spinner("Loading latest data..."):
    try:
        current = get_current_aqi()
    except Exception as e:
        st.error(f"Could not reach the prediction API. Make sure it's running. ({e})")
        st.stop()

latest_aqi = current["aqi"]
latest_time = pd.to_datetime(current["datetime_utc"])
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

horizons = {"24h": ("Tomorrow", 1), "48h": ("Day After", 2), "72h": ("In 3 Days", 3)}
predictions = {}
hazard_horizons = []

try:
    all_preds = get_predictions_all()
except Exception:
    all_preds = {}

cols = st.columns(3)
for i, (horizon_name, (label, days)) in enumerate(horizons.items()):
    entry = all_preds.get(horizon_name, {})
    if "predicted_aqi" not in entry:
        with cols[i]:
            st.info(f"{label} forecast will be available once more data is collected.")
        continue

    pred = entry["predicted_aqi"]
    predictions[horizon_name] = pred
    if entry.get("is_hazardous"):
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

# Forward-looking hazard alert — distinct from the "current AQI is hazardous" banner above.
if hazard_horizons:
    st.markdown(
        f'<div class="alert-banner">🚨 Forecast alert: hazardous AQI expected on {", ".join(hazard_horizons)}. '
        f'Plan outdoor activity accordingly.</div>',
        unsafe_allow_html=True,
    )

# ---------- HISTORICAL vs PREDICTED CHART ----------
st.markdown("### 📈 Historical vs Predicted AQI")

try:
    hist = get_history(hours=72)
    hist_df = pd.DataFrame({
        "Time": pd.to_datetime(hist["timestamps"]),
        "AQI": hist["aqi"],
        "Series": "Historical",
    })

    if predictions:
        future_rows = []
        for horizon_name, hours_ahead in [("24h", 24), ("48h", 48), ("72h", 72)]:
            if horizon_name in predictions:
                future_rows.append({
                    "Time": latest_time + pd.Timedelta(hours=hours_ahead),
                    "AQI": predictions[horizon_name],
                    "Series": "Predicted",
                })
        pred_df = pd.DataFrame(future_rows)
        # Bridge point so the predicted line connects visually to "now".
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
p1.metric("PM2.5", f"{current['pm2_5']:.1f}")
p2.metric("PM10", f"{current['pm10']:.1f}")
p3.metric("CO", f"{current['co']:.1f}")
p4.metric("NO2", f"{current['no2']:.2f}")
p5.metric("O3", f"{current['o3']:.1f}")
p6.metric("SO2", f"{current['so2']:.2f}")

st.markdown("""
<div class="footer-note">
    Data source: OpenWeather Air Pollution API &nbsp;|&nbsp; Model Registry: Hopsworks &nbsp;|&nbsp; Auto-updated hourly
</div>
""", unsafe_allow_html=True)