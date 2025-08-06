#!/usr/bin/env python3
"""
Weather Forecasting ML Project - Izmir, Turkey
Complete ML pipeline with model training, prediction, and visualization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import WeatherDataCollector
from weather_predictor import WeatherPredictor
from weather_visualizer import WeatherVisualizer

def main():
    print("🌤️  Weather Forecasting ML Project - Izmir, Turkey")
    print("=" * 60)
    print("📅 Building complete ML pipeline for weather prediction")
    print("🎯 Training models, generating forecasts, and creating visualizations")
    print("=" * 60)
    
    # Initialize components
    collector = WeatherDataCollector()
    predictor = WeatherPredictor()
    visualizer = WeatherVisualizer()
    
    try:
        # Step 1: Collect training data
        print("\\n1️⃣ COLLECTING TRAINING DATA")
        print("-" * 30)
        historical_hourly, historical_daily = collector.fetch_historical_training_data(years_back=3)
        
        # Step 2: Train ML models
        print("\\n2️⃣ TRAINING ML MODELS")
        print("-" * 30)
        
        # Train temperature prediction model
        temp_model, temp_results = predictor.train_temperature_model(historical_hourly)
        
        # Train weather condition classification model  
        condition_model, condition_accuracy = predictor.train_weather_condition_model(historical_hourly)
        
        # Step 3: Get current weather and forecast
        print("\\n3️⃣ FETCHING CURRENT DATA")
        print("-" * 30)
        current_weather, forecast_df, daily_forecast_df = collector.fetch_current_and_forecast(forecast_days=7)
        
        # Step 4: Generate ML predictions
        print("\\n4️⃣ GENERATING ML PREDICTIONS")
        print("-" * 30)
        predictions_df = predictor.predict_next_hours(current_weather, forecast_df, hours_ahead=72)
        
        if predictions_df is not None and len(predictions_df) > 0:
            print(f"✅ Generated {len(predictions_df)} hourly predictions")
            
            # Show comparison for next 24 hours
            next_24h = predictions_df.head(24)
            if 'forecast_temperature' in next_24h.columns:
                forecast_avg = next_24h['forecast_temperature'].mean()
                ml_avg = next_24h['predicted_temperature'].mean()
                print(f"📊 Next 24h averages:")
                print(f"   Weather Service: {forecast_avg:.1f}°C")
                print(f"   ML Model:        {ml_avg:.1f}°C")
                print(f"   Difference:      {abs(ml_avg - forecast_avg):.1f}°C")
            else:
                ml_avg = next_24h['predicted_temperature'].mean()
                ml_range = f"{next_24h['predicted_temperature'].min():.1f}°C - {next_24h['predicted_temperature'].max():.1f}°C"
                print(f"📊 Next 24h ML predictions:")
                print(f"   Average: {ml_avg:.1f}°C")
                print(f"   Range: {ml_range}")
        else:
            print("❌ No predictions generated - check model training")
        
        # Step 5: Generate weather alerts
        print("\\n5️⃣ GENERATING WEATHER ALERTS")
        print("-" * 30)
        alerts = predictor.generate_weather_alerts(predictions_df, current_weather)
        
        if alerts:
            print(f"⚠️  Generated {len(alerts)} weather alerts:")
            for alert in alerts:
                print(f"   {alert['symbol']} {alert['message']}")
        else:
            print("✅ No weather alerts - conditions look normal")
        
        # Step 6: Create visualizations
        print("\\n6️⃣ CREATING VISUALIZATIONS")
        print("-" * 30)
        viz_results = visualizer.create_all_visualizations(
            current_weather, forecast_df, daily_forecast_df, predictions_df, alerts
        )
        
        # Step 7: Summary and results
        print("\\n" + "=" * 60)
        print("✅ ML WEATHER FORECASTING COMPLETE!")
        print("=" * 60)
        
        print(f"\\n📊 CURRENT CONDITIONS:")
        print(f"   🌡️  Temperature: {current_weather['temperature']:.1f}°C")
        print(f"   🌤️  Condition: {collector.get_weather_condition(current_weather['weather_code'])}")
        print(f"   💧 Humidity: {current_weather['humidity']:.0f}%")
        print(f"   🌬️  Wind: {current_weather['wind_speed']:.1f} km/h")
        
        print(f"\\n🤖 TRAINED ML MODELS:")
        print(f"   📈 Temperature Model: Saved to models/temperature_model.pkl")
        print(f"   🌦️  Weather Condition Model: Saved to models/weather_condition_model.pkl")
        print(f"   📏 Feature Scaler: Saved to models/scaler.pkl")
        print(f"   🏷️  Label Encoder: Saved to models/label_encoder.pkl")
        
        print(f"\\n🔮 PREDICTIONS GENERATED:")
        print(f"   ⏰ Forecast horizon: 72 hours")
        print(f"   📊 ML predictions: {len(predictions_df) if predictions_df is not None else 0} hours")
        print(f"   ⚠️  Weather alerts: {len(alerts)} alerts")
        
        print(f"\\n📈 VISUALIZATIONS CREATED:")
        for plot in viz_results['static_plots']:
            print(f"   🖼️  {plot}")
        for plot in viz_results['interactive_plots']:
            print(f"   🌐 {plot}")
        
        print(f"\\n📁 KEY FILES GENERATED:")
        files_to_check = [
            "models/temperature_model.pkl",
            "models/weather_condition_model.pkl", 
            "models/scaler.pkl",
            "data/historical_hourly.csv",
            "data/predictions.csv",
            "data/current_weather.json"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path}")
        
        print(f"\\n🎯 NEXT STEPS:")
        print(f"   1. Open visualizations/interactive_dashboard.html for detailed exploration")
        print(f"   2. Check visualizations/forecast_analysis.png for 3-day forecast with alerts")
        print(f"   3. Review model performance and predictions")
        print(f"   4. Models are saved and can be reused for future predictions")
        
        if alerts:
            print(f"\\n⚠️  IMPORTANT WEATHER ALERTS:")
            for alert in alerts:
                print(f"   {alert['symbol']} {alert['message']}")
        
    except Exception as e:
        print(f"\\n❌ Error during ML pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def quick_prediction():
    """Quick prediction using existing models"""
    print("⚡ Quick Weather Prediction - Using Existing Models")
    print("-" * 50)
    
    try:
        collector = WeatherDataCollector()
        predictor = WeatherPredictor()
        
        # Get current weather
        current_weather, forecast_df, daily_forecast_df = collector.fetch_current_and_forecast(forecast_days=3)
        
        # Load existing models and predict
        predictions_df = predictor.predict_next_hours(current_weather, forecast_df, hours_ahead=24)
        
        if predictions_df is not None:
            print(f"🌡️  Current: {current_weather['temperature']:.1f}°C")
            print(f"📊 Next 24h ML prediction range: {predictions_df['predicted_temperature'].min():.1f}°C - {predictions_df['predicted_temperature'].max():.1f}°C")
            
            # Generate alerts
            alerts = predictor.generate_weather_alerts(predictions_df, current_weather)
            if alerts:
                for alert in alerts:
                    print(f"⚠️  {alert['message']}")
        else:
            print("❌ Models not found. Please run full training first: python main.py")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_prediction()
    else:
        exit_code = main()
        sys.exit(exit_code)