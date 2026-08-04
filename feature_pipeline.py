import os
import requests
import pandas as pd
from datetime import datetime, timezone
import hopsworks

# ---------- CONFIG (Secrets se aayenge) ----------
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
LAT = 24.8607
LON = 67.0011

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

def calculate_aqi(concentration, breakpoints):
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
            return round(aqi)
    return 500

def fetch_raw_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def build_features():
    raw = fetch_raw_data()
    entry = raw["list"][0]
    components = entry["components"]
    timestamp = entry["dt"]

    pm25 = components["pm2_5"]
    pm10 = components["pm10"]

    aqi_pm25 = calculate_aqi(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = calculate_aqi(pm10, PM10_BREAKPOINTS)
    final_aqi = max(aqi_pm25, aqi_pm10)

    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    features = {
        "timestamp": timestamp,
        "datetime_utc": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "pm2_5": float(pm25),
        "pm10": float(pm10),
        "co": float(components["co"]),
        "no2": float(components["no2"]),
        "o3": float(components["o3"]),
        "so2": float(components["so2"]),
        "aqi": int(final_aqi),
    }
    return features

def push_to_hopsworks(features):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    df = pd.DataFrame([features])

    aqi_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=1,
        description="Hourly AQI and pollutant features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    aqi_fg.insert(df)
    print("Data successfully pushed to Hopsworks Feature Store!")

if __name__ == "__main__":
    features = build_features()
    print("Fetched features:", features)
    push_to_hopsworks(features)
