"""
ES ML Model Integration
Connects trained ML models with NinjaTrader AddOn
Real-time feature engineering and prediction
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class ESMLPredictor:
    """
    Real-time ML prediction engine for ES futures
    Integrates with NinjaTrader AddOn for live trading
    """
    
    def __init__(self, model_path="models/es_ml_model.joblib"):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.is_loaded = False
        self.price_history = []
        self.volume_history = []
        
        # Try to load model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained ML model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_loaded = True
            print(f"✅ ML model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"⚠️ Could not load ML model: {e}")
            print("🔄 Falling back to technical analysis only")
            return False
    
    def update_price_data(self, price, volume=0, timestamp=None):
        """
        Update price history for feature calculation
        Called from NinjaTrader AddOn with each new bar
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        self.price_history.append({
            'timestamp': timestamp,
            'close': price,
            'volume': volume
        })
        
        # Keep only last 100 bars for efficiency
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
    
    def calculate_features(self):
        """
        Calculate real-time features from price history
        Must match the features used in training
        """
        if len(self.price_history) < 50:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.price_history)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        # Calculate same features as in training
        features = {}
        
        try:
            # Price-based features
            features['Returns_1'] = df['close'].pct_change(1).iloc[-1]
            features['Returns_5'] = df['close'].pct_change(5).iloc[-1]
            features['Returns_10'] = df['close'].pct_change(10).iloc[-1]
            
            # Moving averages
            features['SMA_10'] = df['close'].rolling(10).mean().iloc[-1]
            features['SMA_20'] = df['close'].rolling(20).mean().iloc[-1]
            features['SMA_50'] = df['close'].rolling(50).mean().iloc[-1]
            
            if features['SMA_20'] != 0:
                features['SMA_Ratio_10_20'] = features['SMA_10'] / features['SMA_20']
            else:
                features['SMA_Ratio_10_20'] = 1.0
                
            if features['SMA_50'] != 0:
                features['SMA_Ratio_20_50'] = features['SMA_20'] / features['SMA_50']
            else:
                features['SMA_Ratio_20_50'] = 1.0
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            features['RSI'] = rsi_series.iloc[-1] if not rsi_series.empty else 50.0
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            bb_upper = bb_middle + (2 * bb_std)
            bb_lower = bb_middle - (2 * bb_std)
            
            features['BB_Middle'] = bb_middle.iloc[-1]
            features['BB_Upper'] = bb_upper.iloc[-1]
            features['BB_Lower'] = bb_lower.iloc[-1]
            
            bb_range = features['BB_Upper'] - features['BB_Lower']
            if bb_range != 0:
                features['BB_Position'] = (df['close'].iloc[-1] - features['BB_Lower']) / bb_range
            else:
                features['BB_Position'] = 0.5
            
            # Volatility
            features['Volatility_10'] = df['close'].pct_change().rolling(10).std().iloc[-1]
            features['Volatility_20'] = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Volume features (if available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                volume_ma = df['volume'].rolling(20).mean()
                features['Volume_MA'] = volume_ma.iloc[-1]
                if features['Volume_MA'] != 0:
                    features['Volume_Ratio'] = df['volume'].iloc[-1] / features['Volume_MA']
                else:
                    features['Volume_Ratio'] = 1.0
            else:
                features['Volume_MA'] = 0.0
                features['Volume_Ratio'] = 1.0
            
            # Momentum
            if len(df) >= 5:
                features['Momentum_5'] = df['close'].iloc[-1] / df['close'].iloc[-6] - 1
            else:
                features['Momentum_5'] = 0.0
                
            if len(df) >= 10:
                features['Momentum_10'] = df['close'].iloc[-1] / df['close'].iloc[-11] - 1
            else:
                features['Momentum_10'] = 0.0
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            
            features['MACD'] = macd.iloc[-1]
            features['MACD_Signal'] = macd_signal.iloc[-1]
            features['MACD_Histogram'] = features['MACD'] - features['MACD_Signal']
            
            # Support/Resistance
            high_20 = df['close'].rolling(20).max()  # Using close as proxy for high
            low_20 = df['close'].rolling(20).min()   # Using close as proxy for low
            
            features['High_20'] = high_20.iloc[-1]
            features['Low_20'] = low_20.iloc[-1]
            
            price_range = features['High_20'] - features['Low_20']
            if price_range != 0:
                features['Price_Position'] = (df['close'].iloc[-1] - features['Low_20']) / price_range
            else:
                features['Price_Position'] = 0.5
            
            # Time-based features
            current_time = df.index[-1]
            features['Hour'] = current_time.hour
            features['DayOfWeek'] = current_time.dayofweek
            features['IsMarketOpen'] = 1 if 9 <= current_time.hour <= 16 else 0
            
            # Fill any NaN values
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            print(f"❌ Error calculating features: {e}")
            return None
    
    def predict_signal(self, price, volume=0, timestamp=None):
        """
        Generate ML-based trading signal
        Returns (signal, confidence, features_dict)
        """
        # Update price history
        self.update_price_data(price, volume, timestamp)
        
        # Calculate features
        features = self.calculate_features()
        if features is None:
            return "HOLD", 0.0, {}
        
        # Use ML model if available
        if self.is_loaded and self.model is not None:
            try:
                # Ensure all required features are present
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(features.get(col, 0.0))
                
                # Scale features
                X_scaled = self.scaler.transform([feature_vector])
                
                # Make prediction
                prediction = self.model.predict(X_scaled)[0]
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
                
                # Convert to signal
                signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                signal = signal_map.get(prediction, 'HOLD')
                
                return signal, confidence, features
                
            except Exception as e:
                print(f"❌ ML prediction error: {e}")
                # Fall back to technical analysis
                return self.technical_analysis_fallback(features)
        
        else:
            # Use technical analysis as fallback
            return self.technical_analysis_fallback(features)
    
    def technical_analysis_fallback(self, features):
        """
        Fallback technical analysis when ML model is not available
        """
        signals = []
        
        # SMA Signal
        if features.get('SMA_Ratio_20_50', 1.0) > 1.02:  # 20 SMA > 50 SMA by 2%
            signals.append('BUY')
        elif features.get('SMA_Ratio_20_50', 1.0) < 0.98:  # 20 SMA < 50 SMA by 2%
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        # RSI Signal
        rsi = features.get('RSI', 50)
        if rsi < 30:
            signals.append('BUY')
        elif rsi > 70:
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        # Bollinger Bands Signal
        bb_position = features.get('BB_Position', 0.5)
        if bb_position <= 0.1:  # Near lower band
            signals.append('BUY')
        elif bb_position >= 0.9:  # Near upper band
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        # Momentum Signal
        momentum = features.get('Momentum_5', 0)
        if momentum > 0.002:  # Positive momentum > 0.2%
            signals.append('BUY')
        elif momentum < -0.002:  # Negative momentum < -0.2%
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        # Calculate consensus
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        total_signals = len(signals)
        
        if buy_count > sell_count:
            consensus = 'BUY'
            confidence = buy_count / total_signals
        elif sell_count > buy_count:
            consensus = 'SELL'
            confidence = sell_count / total_signals
        else:
            consensus = 'HOLD'
            confidence = 0.5
        
        return consensus, confidence, features
    
    def get_model_status(self):
        """Get status of ML model"""
        return {
            'ml_model_loaded': self.is_loaded,
            'feature_count': len(self.feature_columns),
            'price_history_length': len(self.price_history),
            'model_type': 'RandomForest' if self.is_loaded else 'Technical Analysis'
        }


# Example usage for testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ESMLPredictor()
    
    # Test with sample data
    import random
    base_price = 5800.0
    
    print("🧪 Testing ML predictor with sample data...")
    
    for i in range(60):  # 60 bars of data
        # Simulate price movement
        price = base_price + random.uniform(-10, 10)
        volume = random.randint(1000, 5000)
        
        signal, confidence, features = predictor.predict_signal(price, volume)
        
        if i >= 50:  # Only print last 10 predictions
            print(f"Bar {i+1}: Price={price:.2f}, Signal={signal}, Confidence={confidence:.3f}")
    
    # Print model status
    status = predictor.get_model_status()
    print(f"\n📊 Model Status: {status}")