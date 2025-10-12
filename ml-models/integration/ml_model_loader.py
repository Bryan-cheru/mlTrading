"""
ES ML Model Integration for NinjaTrader
Connects trained ML models with the NinjaTrader AddOn system
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ESMLModelLoader:
    """
    Loads and provides predictions from trained ML models
    For integration with NinjaTrader AddOn
    """
    
    def __init__(self, model_path="models/es_ml_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_loaded = False
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model and components"""
        try:
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", self.model_path)
            
            if os.path.exists(full_path):
                # Load the complete model with scaler and feature info
                model_data = joblib.load(full_path)
                
                if isinstance(model_data, dict):
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_columns = model_data['feature_columns']
                else:
                    # Legacy model format
                    self.model = model_data
                    print("⚠️ Warning: Using legacy model format without scaler")
                
                self.is_loaded = True
                print(f"✅ ML model loaded successfully from {full_path}")
                print(f"🔧 Model type: {type(self.model).__name__}")
                
                if self.feature_columns:
                    print(f"📊 Features: {len(self.feature_columns)}")
                
                return True
            else:
                print(f"❌ Model file not found: {full_path}")
                print("🔧 System will use technical analysis mode")
                return False
                
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            print("🔧 System will use technical analysis mode")
            return False
    
    def engineer_features_for_prediction(self, price_data):
        """
        Engineer features from real-time price data for prediction
        Compatible with NinjaTrader bar data format
        
        Args:
            price_data: Dictionary with OHLCV data or pandas DataFrame
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(price_data, dict):
                # Handle single bar data
                df = pd.DataFrame([price_data])
            elif hasattr(price_data, 'to_dict'):
                # Handle NinjaTrader Bars object
                df = pd.DataFrame([{
                    'Open': price_data.GetOpen(-1),
                    'High': price_data.GetHigh(-1), 
                    'Low': price_data.GetLow(-1),
                    'Close': price_data.GetClose(-1),
                    'Volume': price_data.GetVolume(-1)
                }])
            else:
                df = price_data.copy()
            
            # Ensure we have minimum required data
            if len(df) < 50:
                print("⚠️ Insufficient data for ML prediction (need 50+ bars)")
                return None
            
            # Calculate the same features used in training
            features = {}
            
            # Current price info
            current_close = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            # Price ratios and changes
            features['High_Low_Ratio'] = df['High'].iloc[-1] / df['Low'].iloc[-1]
            features['Open_Close_Ratio'] = df['Open'].iloc[-1] / df['Close'].iloc[-1]
            
            # Volume features
            vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
            features['Volume_MA'] = vol_ma if not pd.isna(vol_ma) else current_volume
            features['Volume_Ratio'] = current_volume / features['Volume_MA'] if features['Volume_MA'] > 0 else 1.0
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                sma = df['Close'].rolling(period).mean().iloc[-1]
                if not pd.isna(sma):
                    features[f'SMA_{period}'] = sma
                    features[f'Price_vs_SMA_{period}'] = current_close / sma - 1
                else:
                    features[f'SMA_{period}'] = current_close
                    features[f'Price_vs_SMA_{period}'] = 0.0
            
            # SMA ratios
            if features['SMA_50'] > 0:
                features['SMA_Ratio_20_50'] = features['SMA_20'] / features['SMA_50']
            else:
                features['SMA_Ratio_20_50'] = 1.0
            
            # Volatility (rolling standard deviation)
            for period in [10, 20]:
                vol = df['Close'].pct_change().rolling(period).std().iloc[-1]
                features[f'Volatility_{period}'] = vol if not pd.isna(vol) else 0.01
            
            # Bollinger Bands
            bb_period = 20
            sma_20 = features['SMA_20']
            std_20 = df['Close'].rolling(bb_period).std().iloc[-1]
            if not pd.isna(std_20):
                features['BB_Upper'] = sma_20 + (2 * std_20)
                features['BB_Lower'] = sma_20 - (2 * std_20)
                features['BB_Position'] = (current_close - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'])
            else:
                features['BB_Upper'] = sma_20 * 1.02
                features['BB_Lower'] = sma_20 * 0.98
                features['BB_Position'] = 0.5
            
            # MACD (simplified)
            ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]
            if not pd.isna(ema_12) and not pd.isna(ema_26):
                features['MACD'] = ema_12 - ema_26
                features['MACD_Signal'] = features['MACD'] * 0.9  # Simplified signal line
            else:
                features['MACD'] = 0.0
                features['MACD_Signal'] = 0.0
            
            # RSI (simplified)
            returns = df['Close'].pct_change().dropna()
            if len(returns) >= 14:
                gains = returns.where(returns > 0, 0).rolling(14).mean().iloc[-1]
                losses = -returns.where(returns < 0, 0).rolling(14).mean().iloc[-1]
                if losses > 0:
                    rs = gains / losses
                    features['RSI'] = 100 - (100 / (1 + rs))
                else:
                    features['RSI'] = 100
            else:
                features['RSI'] = 50
            
            # Time-based features
            now = datetime.now()
            features['Hour'] = now.hour
            features['Day_of_Week'] = now.weekday()
            features['Is_Market_Hours'] = 1 if 9 <= now.hour <= 16 else 0
            
            # Momentum
            if len(df) >= 10:
                price_10_ago = df['Close'].iloc[-10]
                features['Momentum_10'] = (current_close - price_10_ago) / price_10_ago
            else:
                features['Momentum_10'] = 0.0
            
            return features
            
        except Exception as e:
            print(f"❌ Error engineering features: {e}")
            return None
    
    def predict_signal(self, price_data):
        """
        Generate ML trading signal from current market data
        
        Returns:
            dict: {
                'signal': 'BUY'/'SELL'/'HOLD',
                'confidence': float (0-1),
                'probabilities': [sell_prob, hold_prob, buy_prob],
                'ml_enabled': bool
            }
        """
        if not self.is_loaded or self.model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': [0.33, 0.34, 0.33],
                'ml_enabled': False
            }
        
        try:
            # Engineer features
            features = self.engineer_features_for_prediction(price_data)
            if features is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'probabilities': [0.33, 0.34, 0.33],
                    'ml_enabled': False
                }
            
            # Create feature vector (match training feature order)
            if self.feature_columns:
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(features.get(col, 0.0))
            else:
                # Use available features
                feature_vector = list(features.values())
            
            # Convert to numpy array
            X = np.array(feature_vector).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Map to signal classes: [SELL=0, HOLD=1, BUY=2]
            signal_map = ['SELL', 'HOLD', 'BUY']
            predicted_class = np.argmax(probabilities)
            signal = signal_map[predicted_class]
            confidence = probabilities[predicted_class]
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'ml_enabled': True
            }
            
        except Exception as e:
            print(f"❌ Error generating ML prediction: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': [0.33, 0.34, 0.33],
                'ml_enabled': False
            }

# Global model loader instance
_ml_model = None

def get_ml_model():
    """Get the global ML model instance"""
    global _ml_model
    if _ml_model is None:
        _ml_model = ESMLModelLoader()
    return _ml_model

def predict_ml_signal(price_data):
    """
    Convenience function for getting ML predictions
    Can be called directly from NinjaTrader AddOn
    """
    model = get_ml_model()
    return model.predict_signal(price_data)

if __name__ == "__main__":
    # Test the ML model loader
    print("🧪 Testing ML Model Loader...")
    
    model = ESMLModelLoader()
    
    if model.is_loaded:
        print("✅ Model loaded successfully!")
        
        # Test with dummy data
        test_data = {
            'Open': 4500.0,
            'High': 4505.0,
            'Low': 4495.0,
            'Close': 4502.0,
            'Volume': 1000
        }
        
        result = model.predict_signal(test_data)
        print(f"🎯 Test prediction: {result}")
    else:
        print("❌ Model loading failed")