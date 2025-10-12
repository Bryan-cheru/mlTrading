"""
Transformer Model Loader for NinjaTrader Integration
Loads and provides predictions from trained transformer models
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from ml_models.advanced.transformer_model import TradingTransformer

class ESTransformerLoader:
    """
    Loads transformer model for real-time predictions
    """
    
    def __init__(self, model_path="models/es_transformer_complete.joblib"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_config = None
        self.seq_length = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained transformer model"""
        try:
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", self.model_path)
            
            if os.path.exists(full_path):
                # Load the complete model
                checkpoint = joblib.load(full_path)
                
                # Extract components
                self.model_config = checkpoint['model_config']
                self.scaler = checkpoint['scaler']
                self.feature_columns = checkpoint['feature_columns']
                self.seq_length = self.model_config['seq_length']
                
                # Rebuild model
                self.model = TradingTransformer(
                    input_dim=self.model_config['input_dim'],
                    d_model=self.model_config['d_model'],
                    nhead=self.model_config['nhead'],
                    num_layers=self.model_config['num_layers'],
                    seq_length=self.model_config['seq_length'],
                    num_classes=3,
                    dropout=0.1
                ).to(self.device)
                
                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                self.is_loaded = True
                print(f"✅ Transformer model loaded successfully")
                print(f"🔧 Model: {self.model_config['num_layers']} layers, {self.model_config['d_model']} dim")
                print(f"📊 Features: {len(self.feature_columns)}")
                print(f"⚡ Device: {self.device}")
                
                return True
            else:
                print(f"❌ Transformer model not found: {full_path}")
                print("🔧 System will use Random Forest model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading transformer model: {e}")
            print("🔧 System will use Random Forest model")
            return False
    
    def engineer_features_for_prediction(self, historical_data):
        """
        Engineer features from historical price data for real-time prediction
        
        Args:
            historical_data: DataFrame with OHLCV data (minimum seq_length rows)
        """
        try:
            df = historical_data.copy()
            
            if len(df) < self.seq_length:
                print(f"⚠️ Insufficient data for transformer prediction (need {self.seq_length}+ bars)")
                return None
            
            # Take only the last seq_length bars
            df = df.tail(self.seq_length).copy()
            
            # Calculate all features that were used in training
            
            # Basic OHLCV (already present)
            # Ensure these columns exist
            required_base = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_base:
                if col not in df.columns:
                    print(f"❌ Missing required column: {col}")
                    return None
            
            # Price ratios and changes
            df['Returns'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Moving averages and ratios
            for period in [5, 10, 20, 50]:
                col_sma = f'SMA_{period}'
                col_ratio = f'Price_SMA_Ratio_{period}'
                
                df[col_sma] = df['Close'].rolling(period).mean()
                df[col_ratio] = df['Close'] / df[col_sma]
            
            # Technical indicators
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands position
            bb_sma = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Volatility
            df['Volatility'] = df['Returns'].rolling(20).std()
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Time features
            df['Hour'] = df.index.hour
            df['Day_of_Week'] = df.index.dayofweek
            df['Is_Market_Hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 16)).astype(int)
            
            # Select only the features used in training
            feature_matrix = df[self.feature_columns].values
            
            # Handle NaN values (fill with median or forward fill)
            feature_df = pd.DataFrame(feature_matrix, columns=self.feature_columns)
            feature_df = feature_df.fillna(method='ffill').fillna(feature_df.median())
            
            return feature_df.values
            
        except Exception as e:
            print(f"❌ Error engineering features: {e}")
            return None
    
    def predict_signal(self, historical_data):
        """
        Generate trading signal from historical market data
        
        Args:
            historical_data: DataFrame with OHLCV data or dictionary with market data
            
        Returns:
            dict: {
                'signal': 'BUY'/'SELL'/'HOLD',
                'confidence': float (0-1),
                'probabilities': [sell_prob, hold_prob, buy_prob],
                'transformer_enabled': bool
            }
        """
        if not self.is_loaded or self.model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': [0.33, 0.34, 0.33],
                'transformer_enabled': False
            }
        
        try:
            # Convert dictionary to DataFrame if needed
            if isinstance(historical_data, dict):
                if 'historical_closes' in historical_data:
                    # Create DataFrame from historical data
                    closes = historical_data['historical_closes']
                    volumes = historical_data.get('historical_volumes', [1000] * len(closes))
                    
                    df_data = []
                    for i, (close, volume) in enumerate(zip(closes, volumes)):
                        df_data.append({
                            'Open': close * 0.999,
                            'High': close * 1.001,
                            'Low': close * 0.999,
                            'Close': close,
                            'Volume': volume
                        })
                    
                    historical_data = pd.DataFrame(df_data)
                    # Add simple datetime index
                    historical_data.index = pd.date_range(
                        start=datetime.now() - pd.Timedelta(hours=len(df_data)),
                        periods=len(df_data),
                        freq='H'
                    )
                else:
                    # Single bar data - can't use transformer
                    return {
                        'signal': 'HOLD',
                        'confidence': 0.0,
                        'probabilities': [0.33, 0.34, 0.33],
                        'transformer_enabled': False
                    }
            
            # Engineer features
            features = self.engineer_features_for_prediction(historical_data)
            if features is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'probabilities': [0.33, 0.34, 0.33],
                    'transformer_enabled': False
                }
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Convert to tensor and add batch dimension
            sequence_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                signals, confidence = self.model(sequence_tensor)
                
                # Convert to probabilities
                probabilities = torch.softmax(signals, dim=1).cpu().numpy()[0]
                
                # Get predicted class and confidence
                predicted_class = np.argmax(probabilities)
                max_prob = probabilities[predicted_class]
                model_confidence = confidence.cpu().item()
                
                # Combine prediction confidence with model confidence
                final_confidence = max_prob * model_confidence
                
                # Map to signal names
                signal_map = ['SELL', 'HOLD', 'BUY']
                signal = signal_map[predicted_class]
                
                return {
                    'signal': signal,
                    'confidence': float(final_confidence),
                    'probabilities': probabilities.tolist(),
                    'transformer_enabled': True
                }
                
        except Exception as e:
            print(f"❌ Error generating transformer prediction: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'probabilities': [0.33, 0.34, 0.33],
                'transformer_enabled': False
            }

# Global transformer instance
_transformer_model = None

def get_transformer_model():
    """Get the global transformer model instance"""
    global _transformer_model
    if _transformer_model is None:
        _transformer_model = ESTransformerLoader()
    return _transformer_model

def predict_transformer_signal(historical_data):
    """
    Convenience function for getting transformer predictions
    Can be called directly from NinjaTrader AddOn
    """
    model = get_transformer_model()
    return model.predict_signal(historical_data)

if __name__ == "__main__":
    # Test the transformer loader
    print("🧪 Testing Transformer Model Loader...")
    
    loader = ESTransformerLoader()
    
    if loader.is_loaded:
        print("✅ Transformer loaded successfully!")
        
        # Test with dummy historical data
        test_data = pd.DataFrame({
            'Open': np.random.normal(4500, 10, 100),
            'High': np.random.normal(4505, 10, 100),
            'Low': np.random.normal(4495, 10, 100),
            'Close': np.random.normal(4500, 10, 100),
            'Volume': np.random.normal(1000, 100, 100)
        })
        test_data.index = pd.date_range(start='2025-01-01', periods=100, freq='H')
        
        result = loader.predict_signal(test_data)
        print(f"🎯 Test prediction: {result}")
    else:
        print("❌ Transformer loading failed")