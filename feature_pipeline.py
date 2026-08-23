import requests
import pandas as pd
from datetime import datetime, timezone
import hopsworks
import os
# ---------- CONFIG ----------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
LAT = 24.8607
LON = 67.0011

# 👇 APNI WEATHER FEATURE GROUP KI DETAILS YAHAN CONFIRM KARO
WEATHER_FG_NAME = "karachi_weather_features"
WEATHER_FG_VERSION = 1

# ---------- EPA AQI BREAKPOINTS ----------
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
]

def calculate_aqi(concentration, breakpoints):
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low)
    return 500

def fetch_pollution_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_weather_data():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def build_pollution_features(raw_pollution):
    entry = raw_pollution["list"][0]
    components = entry["components"]
    timestamp = entry["dt"]

    pm25 = components["pm2_5"]
    pm10 = components["pm10"]

    aqi_pm25 = calculate_aqi(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = calculate_aqi(pm10, PM10_BREAKPOINTS)
    final_aqi = max(aqi_pm25, aqi_pm10)

    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    return {
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

def build_weather_features(raw_weather, timestamp):
    # Weather API ka apna alag 'dt' hota hai, lekin hum pollution wala timestamp use karenge
    # taake dono feature groups ka key match ho aur join asaan ho
    main = raw_weather.get("main", {})
    wind = raw_weather.get("wind", {})
    clouds = raw_weather.get("clouds", {})

    return {
        "timestamp": timestamp,
        "temperature": float(main.get("temp", 0)),
        "humidity": float(main.get("humidity", 0)),
        "pressure": float(main.get("pressure", 0)),
        "wind_speed": float(wind.get("speed", 0)),
        "clouds": float(clouds.get("all", 0)),
    }

def push_to_hopsworks(pollution_features, weather_features):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # ---------- Pollution Feature Group ----------
    pollution_df = pd.DataFrame([pollution_features])
    aqi_fg = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=1,
        description="Hourly AQI and pollutant features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    aqi_fg.insert(pollution_df)
    print("Pollution data pushed to Hopsworks!")

    # ---------- Weather Feature Group ----------
    weather_df = pd.DataFrame([weather_features])
    weather_fg = fs.get_or_create_feature_group(
        name=WEATHER_FG_NAME,
        version=WEATHER_FG_VERSION,
        description="Hourly weather features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    weather_fg.insert(weather_df)
    print("Weather data pushed to Hopsworks!")

if __name__ == "__main__":
    raw_pollution = fetch_pollution_data()
    pollution_features = build_pollution_features(raw_pollution)
    print("Fetched pollution features:", pollution_features)

    raw_weather = fetch_weather_data()
    weather_features = build_weather_features(raw_weather, pollution_features["timestamp"])
    print("Fetched weather features:", weather_features)

    push_to_hopsworks(pollution_features, weather_features)