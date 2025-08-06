import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class WeatherDataCollector:
    def __init__(self):
        self.historical_url = "https://archive-api.open-meteo.com/v1/era5"
        self.current_url = "https://api.open-meteo.com/v1/forecast"
        self.izmir_coords = {"latitude": 38.4192, "longitude": 27.1287}
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_historical_training_data(self, years_back=3):
        """Fetch historical data for ML model training"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        params = {
            "latitude": self.izmir_coords["latitude"],
            "longitude": self.izmir_coords["longitude"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code"
            ],
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max"
            ],
            "timezone": "Europe/Istanbul"
        }
        
        print(f"Fetching {years_back} years of historical data for model training...")
        response = requests.get(self.historical_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process hourly data
            hourly_df = pd.DataFrame({
                'datetime': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'precipitation': data['hourly']['precipitation'],
                'pressure': data['hourly']['surface_pressure'],
                'cloud_cover': data['hourly']['cloud_cover'],
                'wind_speed': data['hourly']['wind_speed_10m'],
                'wind_direction': data['hourly']['wind_direction_10m'],
                'weather_code': data['hourly']['weather_code']
            })
            
            # Process daily data
            daily_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'weather_code': data['daily']['weather_code'],
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'precipitation_sum': data['daily']['precipitation_sum'],
                'wind_speed_max': data['daily']['wind_speed_10m_max']
            })
            
            # Save training data
            hourly_df.to_csv(f"{self.data_dir}/historical_hourly.csv", index=False)
            daily_df.to_csv(f"{self.data_dir}/historical_daily.csv", index=False)
            
            print(f"✅ Training data collected: {len(hourly_df)} hourly records, {len(daily_df)} daily records")
            return hourly_df, daily_df
        else:
            raise Exception(f"Failed to fetch historical data: {response.status_code}")
    
    def fetch_current_and_forecast(self, forecast_days=7):
        """Fetch current weather and forecast data"""
        params = {
            "latitude": self.izmir_coords["latitude"],
            "longitude": self.izmir_coords["longitude"],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code"
            ],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m", 
                "precipitation",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code"
            ],
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max"
            ],
            "forecast_days": forecast_days,
            "timezone": "Europe/Istanbul"
        }
        
        print(f"Fetching current weather and {forecast_days}-day forecast...")
        response = requests.get(self.current_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Current weather
            current = data['current']
            current_weather = {
                'datetime': pd.to_datetime(current['time']),
                'temperature': current['temperature_2m'],
                'humidity': current['relative_humidity_2m'],
                'precipitation': current['precipitation'],
                'pressure': current['surface_pressure'],
                'cloud_cover': current['cloud_cover'],
                'wind_speed': current['wind_speed_10m'],
                'wind_direction': current['wind_direction_10m'],
                'weather_code': current['weather_code']
            }
            
            # Forecast data
            forecast_df = pd.DataFrame({
                'datetime': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'precipitation': data['hourly']['precipitation'],
                'pressure': data['hourly']['surface_pressure'],
                'cloud_cover': data['hourly']['cloud_cover'],
                'wind_speed': data['hourly']['wind_speed_10m'],
                'wind_direction': data['hourly']['wind_direction_10m'],
                'weather_code': data['hourly']['weather_code']
            })
            
            daily_forecast_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'weather_code': data['daily']['weather_code'],
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'precipitation_sum': data['daily']['precipitation_sum'],
                'wind_speed_max': data['daily']['wind_speed_10m_max']
            })
            
            # Save current data
            forecast_df.to_csv(f"{self.data_dir}/current_forecast.csv", index=False)
            daily_forecast_df.to_csv(f"{self.data_dir}/daily_forecast.csv", index=False)
            
            import json
            with open(f"{self.data_dir}/current_weather.json", 'w') as f:
                json.dump({k: str(v) if isinstance(v, pd.Timestamp) else v 
                          for k, v in current_weather.items()}, f, indent=2)
            
            print(f"✅ Current: {current_weather['temperature']:.1f}°C, {self.get_weather_condition(current_weather['weather_code'])}")
            return current_weather, forecast_df, daily_forecast_df
        else:
            raise Exception(f"Failed to fetch current data: {response.status_code}")
    
    def get_weather_condition(self, weather_code):
        """Convert weather codes to readable conditions"""
        weather_conditions = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Drizzle: Light", 53: "Drizzle: Moderate", 55: "Drizzle: Dense",
            56: "Freezing Drizzle: Light", 57: "Freezing Drizzle: Dense",
            61: "Rain: Slight", 63: "Rain: Moderate", 65: "Rain: Heavy",
            66: "Freezing Rain: Light", 67: "Freezing Rain: Heavy",
            71: "Snow fall: Slight", 73: "Snow fall: Moderate", 75: "Snow fall: Heavy",
            77: "Snow grains",
            80: "Rain showers: Slight", 81: "Rain showers: Moderate", 82: "Rain showers: Violent",
            85: "Snow showers: Slight", 86: "Snow showers: Heavy",
            95: "Thunderstorm: Slight or moderate",
            96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return weather_conditions.get(weather_code, "Unknown")
    
    def categorize_weather(self, weather_code):
        """Categorize weather into simple types"""
        if weather_code in [0, 1]:
            return "Sunny"
        elif weather_code in [2, 3]:
            return "Partly Cloudy"
        elif weather_code in [45, 48]:
            return "Foggy"
        elif weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
            return "Rainy"
        elif weather_code in [71, 73, 75, 77, 85, 86]:
            return "Snowy"
        elif weather_code in [95, 96, 99]:
            return "Stormy"
        else:
            return "Other"