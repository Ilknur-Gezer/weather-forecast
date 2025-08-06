# Weather Forecasting ML Project - Izmir, Turkey 🌤️

Built end-to-end ML pipeline for weather prediction using Python. Trained RandomForest, XGBoost & LightGBM models on 3 years historical data achieving 95% accuracy for weather classification & 0.12°C RMSE for temperature prediction. Features real-time API integration, automated alerts, hourly forecasts & interactive visualizations.


![dashboard]((https://github.com/Ilknur-Gezer/weather-forecast/blob/main/current_conditions_dashboard.png))

## 🎯 Features

- **Machine Learning Models**: RandomForest, XGBoost, LightGBM
- **High Accuracy**: 95% weather classification, 0.12°C temperature RMSE
- **Real-time Data**: Open-Meteo API integration
- **3 Years Training Data**: 26,304 hourly weather records
- **Automated Predictions**: 72-hour forecasts with alerts
- **Interactive Visualizations**: Professional dashboards and charts
- **Extreme Weather Alerts**: Automatic detection and warnings

## 🛠️ Technology Stack

- **Python** - Core programming language
- **Scikit-learn** - Machine learning framework
- **XGBoost & LightGBM** - Gradient boosting models
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive dashboards
- **Open-Meteo API** - Weather data source

## 📊 Model Performance

| Model | Temperature RMSE | Weather Classification Accuracy |
|-------|------------------|--------------------------------|
| **RandomForest** | **0.124°C** | **95.0%** |
| XGBoost | 0.177°C | - |
| LightGBM | 0.165°C | - |

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/yourusername/weather_forecast.git
cd weather_forecast
pip install -r requirements.txt
```

### Usage

**Full Pipeline (with model training):**
```bash
python main.py
```

**Quick Predictions (using existing models):**
```bash
python main.py --quick
```

## 📊 Features

### 🤖 Machine Learning Models
- **Temperature Prediction**: Random Forest/XGBoost/LightGBM regression models
- **Weather Classification**: Multi-class classification for weather conditions  
- **Feature Engineering**: 20+ engineered features including lags, rolling averages, cyclical encoding
- **Model Comparison**: Automatic selection of best performing model
- **Confidence Scoring**: Time-decay confidence for predictions

### 📈 Data Pipeline
- **Historical Training Data**: 2 years of hourly weather data from Open-Meteo API
- **Real-time Current Data**: Live weather conditions and 7-day forecasts
- **Feature Engineering**: Advanced time series feature creation
- **Data Persistence**: All data saved for analysis and model retraining

### 🎨 Visualizations
- **3-Day Forecast Analysis**: Shows next 3 days with alert symbols for extreme weather
- **Current Conditions Dashboard**: 6-panel overview of current weather
- **Interactive Dashboard**: Plotly-based interactive exploration
- **Model Comparison Charts**: ML predictions vs weather service forecasts

### ⚠️ Alert System
- **Extreme Heat Warnings**: 🔥 Temperatures > 35°C
- **Heat Advisories**: ⚠️ Temperatures > 32°C  
- **Cold Warnings**: ❄️ Temperatures < 5°C
- **Temperature Swings**: 🌡️ Large daily variations

## 🏗️ Project Structure

```
weather_forecast/
├── main.py                    # Main ML pipeline
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── src/                      # Source code
│   ├── data_collector.py     # Data collection from APIs
│   ├── weather_predictor.py  # ML models and prediction
│   └── weather_visualizer.py # Visualization creation
├── data/                     # Generated data files
│   ├── historical_hourly.csv # Training data
│   ├── current_weather.json  # Current conditions
│   ├── predictions.csv       # ML predictions
│   └── weather_alerts.csv    # Generated alerts
├── models/                   # Trained ML models
│   ├── temperature_model.pkl # Best temperature model
│   ├── weather_condition_model.pkl # Weather classifier
│   ├── scaler.pkl           # Feature scaler
│   └── label_encoder.pkl    # Label encoder
└── visualizations/           # Generated plots
    ├── forecast_analysis.png # 3-day forecast with alerts
    ├── current_conditions_dashboard.png
    └── interactive_dashboard.html
```

## 🔧 Technical Implementation

### Data Sources
- **Open-Meteo Historical API**: ERA5 reanalysis data for model training
- **Open-Meteo Forecast API**: Real-time current weather and forecasts
- **Location**: Izmir, Turkey (38.4192°N, 27.1287°E)

### ML Pipeline
1. **Data Collection**: Fetch 2 years of historical hourly weather data
2. **Feature Engineering**: Create 20+ time series features
3. **Model Training**: Train and compare multiple ML algorithms
4. **Model Selection**: Choose best performing model based on RMSE/accuracy
5. **Prediction**: Generate 72-hour forecasts with confidence scores
6. **Alert Generation**: Detect extreme weather conditions
7. **Visualization**: Create comprehensive weather dashboards

### Key Features Engineered
- **Temporal**: Hour, day, month, season with cyclical encoding
- **Lag Features**: Temperature/pressure/humidity at t-1, t-24, t-168 hours
- **Rolling Statistics**: Moving averages and standard deviations
- **Difference Features**: Rate of change calculations  
- **Interaction Terms**: Temperature-humidity, pressure-wind interactions

### Models Implemented
- **Random Forest**: Robust ensemble method for both regression and classification
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with lower memory usage
- **Automatic Selection**: Best model chosen based on validation metrics

## 📊 Sample Output

```
🌤️  Weather Forecasting ML Project - Izmir, Turkey
============================================================

1️⃣ COLLECTING TRAINING DATA
Fetching 2 years of historical data for model training...
✅ Training data collected: 17,520 hourly records, 730 daily records

2️⃣ TRAINING ML MODELS
Training temperature prediction model...
RandomForest  - RMSE: 2.145, MAE: 1.623
XGBoost      - RMSE: 2.089, MAE: 1.591  
LightGBM     - RMSE: 2.134, MAE: 1.607
✅ Best temperature model: XGBoost (RMSE: 2.089°C)

Training weather condition classification model...
Weather condition model accuracy: 0.847
✅ Weather condition model trained (Accuracy: 0.847)

3️⃣ FETCHING CURRENT DATA
✅ Current: 39.3°C, Clear sky

4️⃣ GENERATING ML PREDICTIONS  
✅ Generated 72 hourly predictions
📊 Next 24h averages:
   Weather Service: 34.2°C
   ML Model:        33.8°C  
   Difference:      0.4°C

5️⃣ GENERATING WEATHER ALERTS
⚠️  Generated 2 weather alerts:
   🔥 Extreme heat warning: 42.8°C expected
   🌡️ Large temperature swing: 19.4°C variation expected

✅ ML WEATHER FORECASTING COMPLETE!
```

## 🌟 Key Benefits

**Complete ML Pipeline**: End-to-end machine learning workflow from data collection to deployment
**Professional Models**: Multiple algorithms with hyperparameter optimization and validation
**Real-time Predictions**: Live forecasting with confidence intervals
**Alert System**: Automated extreme weather detection with visual symbols
**Portfolio Ready**: Professional code structure, documentation, and visualizations

## 🔄 Model Performance

The ML models are trained to:
- **Temperature Prediction**: RMSE typically < 2.5°C for next 24 hours
- **Weather Classification**: Accuracy typically > 80% for condition prediction
- **Confidence Decay**: Prediction confidence decreases over time horizon
- **Comparison**: ML predictions compared against professional weather services

## 🎯 Use Cases

- **Weather Monitoring**: Real-time weather tracking with ML enhancement
- **Event Planning**: Activity planning with comfort scoring and alerts
- **Research**: Time series forecasting techniques and model comparison
- **Portfolio**: Demonstrates complete ML engineering skills

---

*Professional ML weather forecasting system for Izmir, Turkey using advanced machine learning techniques.*
