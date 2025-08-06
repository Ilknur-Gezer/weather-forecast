import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    def __init__(self):
        self.models_dir = "models"
        self.data_dir = "data"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model storage
        self.temperature_model = None
        self.weather_condition_model = None
        self.scaler = None
        self.label_encoder = None
    
    def prepare_features(self, df):
        """Create features for ML models"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['season'] = (df['datetime'].dt.month % 12) // 3 + 1
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (previous time steps)
        df['temp_lag_1'] = df['temperature'].shift(1)
        df['temp_lag_24'] = df['temperature'].shift(24)  # Previous day same hour
        df['temp_lag_168'] = df['temperature'].shift(168)  # Previous week same hour
        df['pressure_lag_1'] = df['pressure'].shift(1)
        df['humidity_lag_1'] = df['humidity'].shift(1)
        
        # Rolling statistics
        df['temp_rolling_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()
        df['temp_rolling_24'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        df['temp_rolling_std_24'] = df['temperature'].rolling(window=24, min_periods=1).std()
        df['pressure_rolling_12'] = df['pressure'].rolling(window=12, min_periods=1).mean()
        df['humidity_rolling_6'] = df['humidity'].rolling(window=6, min_periods=1).mean()
        
        # Difference features (rate of change)
        df['temp_diff_1h'] = df['temperature'].diff(1)
        df['pressure_diff_1h'] = df['pressure'].diff(1)
        df['humidity_diff_1h'] = df['humidity'].diff(1)
        
        # Weather interactions
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['pressure_wind_interaction'] = df['pressure'] * df['wind_speed'] / 1000
        
        # Remove first week of data due to lag features
        df = df.iloc[168:].copy()
        
        return df
    
    def train_temperature_model(self, historical_df):
        """Train ML model for temperature prediction"""
        print("Training temperature prediction model...")
        
        # Prepare features
        df = self.prepare_features(historical_df)
        
        feature_columns = [
            'hour', 'day', 'month', 'day_of_year', 'season', 'is_weekend',
            'humidity', 'pressure', 'wind_speed', 'cloud_cover',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'temp_lag_1', 'temp_lag_24', 'temp_lag_168',
            'pressure_lag_1', 'humidity_lag_1',
            'temp_rolling_3', 'temp_rolling_24', 'temp_rolling_std_24',
            'pressure_rolling_12', 'humidity_rolling_6',
            'temp_diff_1h', 'pressure_diff_1h', 'humidity_diff_1h',
            'temp_humidity_interaction', 'pressure_wind_interaction'
        ]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['temperature']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        }
        
        best_model = None
        best_score = float('inf')
        best_name = ""
        
        results = {}
        for name, model in models.items():
            if name in ['RandomForest']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
            
            print(f"{name:12} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            
            if mse < best_score:
                best_score = mse
                best_model = model
                best_name = name
        
        self.temperature_model = best_model
        
        # Save model and scaler
        joblib.dump(best_model, f"{self.models_dir}/temperature_model.pkl")
        joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
        
        # Save feature columns
        joblib.dump(feature_columns, f"{self.models_dir}/feature_columns.pkl")
        
        print(f"✅ Best temperature model: {best_name} (RMSE: {np.sqrt(best_score):.3f}°C)")
        return best_model, results
    
    def train_weather_condition_model(self, historical_df):
        """Train ML model for weather condition classification"""
        print("Training weather condition classification model...")
        
        from data_collector import WeatherDataCollector
        collector = WeatherDataCollector()
        
        # Prepare features
        df = self.prepare_features(historical_df)
        df = df.dropna()
        
        # Add weather categories
        df['weather_category'] = df['weather_code'].apply(collector.categorize_weather)
        
        # Select features for classification (simpler set)
        feature_columns = [
            'hour', 'month', 'season', 'temperature', 'humidity', 'pressure',
            'wind_speed', 'cloud_cover', 'temp_rolling_24', 'pressure_rolling_12',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        X = df[feature_columns]
        y = df['weather_category']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Filter out classes with too few samples and reset indices
        class_counts = pd.Series(y_encoded).value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        
        # Create aligned mask
        mask = np.isin(y_encoded, valid_classes)
        X_filtered = X[mask].reset_index(drop=True)
        y_filtered = y_encoded[mask]
        
        print(f"Using {len(valid_classes)} weather classes with sufficient samples")
        
        # Split data
        if len(np.unique(y_filtered)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Weather condition model accuracy: {accuracy:.3f}")
        
        # Print classification report
        unique_classes_in_test = np.unique(y_test)
        class_names_in_test = [self.label_encoder.classes_[i] for i in unique_classes_in_test]
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names_in_test, zero_division=0))
        
        self.weather_condition_model = model
        
        # Save model and encoder
        joblib.dump(model, f"{self.models_dir}/weather_condition_model.pkl")
        joblib.dump(self.label_encoder, f"{self.models_dir}/label_encoder.pkl")
        joblib.dump(feature_columns, f"{self.models_dir}/condition_feature_columns.pkl")
        
        print(f"✅ Weather condition model trained (Accuracy: {accuracy:.3f})")
        return model, accuracy
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.temperature_model = joblib.load(f"{self.models_dir}/temperature_model.pkl")
            self.weather_condition_model = joblib.load(f"{self.models_dir}/weather_condition_model.pkl")
            self.scaler = joblib.load(f"{self.models_dir}/scaler.pkl")
            self.label_encoder = joblib.load(f"{self.models_dir}/label_encoder.pkl")
            self.feature_columns = joblib.load(f"{self.models_dir}/feature_columns.pkl")
            self.condition_feature_columns = joblib.load(f"{self.models_dir}/condition_feature_columns.pkl")
            print("✅ Models loaded successfully")
            return True
        except FileNotFoundError:
            print("❌ Models not found. Please train models first.")
            return False
    
    def predict_next_hours(self, current_data, forecast_df, hours_ahead=72):
        """Predict weather for next N hours"""
        if not self.load_models():
            return None
        
        predictions = []
        
        # Combine current + forecast data for feature creation
        combined_df = forecast_df.copy()
        
        # Prepare features for the forecast data
        features_df = self.prepare_features(combined_df)
        
        # Make predictions for next hours_ahead hours
        for i in range(min(hours_ahead, len(features_df))):
            row = features_df.iloc[i:i+1]
            
            if row[self.feature_columns].isnull().any().any():
                continue
            
            # Temperature prediction
            if isinstance(self.temperature_model, RandomForestRegressor):
                temp_pred = self.temperature_model.predict(row[self.feature_columns])[0]
            else:
                temp_pred = self.temperature_model.predict(self.scaler.transform(row[self.feature_columns]))[0]
            
            # Weather condition prediction
            condition_features = row[self.condition_feature_columns].fillna(row[self.condition_feature_columns].mean())
            condition_pred_encoded = self.weather_condition_model.predict(condition_features)[0]
            condition_pred = self.label_encoder.inverse_transform([condition_pred_encoded])[0]
            
            predictions.append({
                'datetime': row['datetime'].iloc[0],
                'predicted_temperature': temp_pred,
                'predicted_condition': condition_pred,
                'forecast_temperature': row['temperature'].iloc[0],  # Original forecast for comparison
                'temperature_confidence': max(0.6, 0.95 - i * 0.005)  # Decreasing confidence over time
            })
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(f"{self.data_dir}/predictions.csv", index=False)
        
        return predictions_df
    
    def generate_weather_alerts(self, predictions_df, current_weather):
        """Generate weather alerts based on predictions"""
        alerts = []
        
        if predictions_df is None or len(predictions_df) == 0:
            return alerts
        
        # Temperature alerts
        max_temp = predictions_df['predicted_temperature'].max()
        min_temp = predictions_df['predicted_temperature'].min()
        current_temp = current_weather['temperature']
        
        # Extreme heat alert
        if max_temp > 35:
            alerts.append({
                'type': 'EXTREME_HEAT',
                'severity': 'HIGH',
                'message': f'Extreme heat warning: {max_temp:.1f}°C expected',
                'symbol': '🔥',
                'when': predictions_df.loc[predictions_df['predicted_temperature'].idxmax(), 'datetime']
            })
        elif max_temp > 32:
            alerts.append({
                'type': 'HEAT_ADVISORY',
                'severity': 'MEDIUM',
                'message': f'High temperature advisory: {max_temp:.1f}°C expected',
                'symbol': '⚠️',
                'when': predictions_df.loc[predictions_df['predicted_temperature'].idxmax(), 'datetime']
            })
        
        # Cold alert
        if min_temp < 5:
            alerts.append({
                'type': 'COLD_WARNING',
                'severity': 'MEDIUM',
                'message': f'Cold warning: {min_temp:.1f}°C expected',
                'symbol': '❄️',
                'when': predictions_df.loc[predictions_df['predicted_temperature'].idxmin(), 'datetime']
            })
        
        # Rapid temperature change
        temp_diff = abs(max_temp - min_temp)
        if temp_diff > 15:
            alerts.append({
                'type': 'TEMP_SWING',
                'severity': 'MEDIUM',
                'message': f'Large temperature swing: {temp_diff:.1f}°C variation expected',
                'symbol': '🌡️',
                'when': datetime.now()
            })
        
        # Save alerts
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_csv(f"{self.data_dir}/weather_alerts.csv", index=False)
        
        return alerts