"""
Real-time Feature Engineering for ES Trading
Optimized for low-latency production use
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
    """Market data container"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class FeatureSet:
    """Feature set container"""
    timestamp: pd.Timestamp
    features: Dict[str, float]
    target: Optional[float] = None

class RealTimeFeatureEngine:
    """
    Real-time feature engineering for ES futures trading
    Optimized for <10ms latency requirements
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        self.price_buffer = deque(maxlen=lookback_periods)
        self.volume_buffer = deque(maxlen=lookback_periods)
        self.feature_cache = {}
        
        # Pre-allocate numpy arrays for performance
        self.price_array = np.zeros(lookback_periods)
        self.volume_array = np.zeros(lookback_periods)
        
        print(f"FeatureEngine initialized with {lookback_periods} period lookback")
    
    def update_buffers(self, market_data: MarketData):
        """Update rolling buffers with new market data"""
        self.price_buffer.append({
            'timestamp': market_data.timestamp,
            'open': market_data.open,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close
        })
        self.volume_buffer.append(market_data.volume)
        
        # Update numpy arrays for TA-Lib
        if len(self.price_buffer) >= 2:
            prices = [bar['close'] for bar in self.price_buffer]
            volumes = list(self.volume_buffer)
            
            self.price_array[:len(prices)] = prices
            self.volume_array[:len(volumes)] = volumes
    
    def compute_price_features(self) -> Dict[str, float]:
        """Compute price-based features"""
        if len(self.price_buffer) < 20:
            return {}
        
        features = {}
        prices = [bar['close'] for bar in self.price_buffer]
        highs = [bar['high'] for bar in self.price_buffer]
        lows = [bar['low'] for bar in self.price_buffer]
        opens = [bar['open'] for bar in self.price_buffer]
        
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price
        
        # Basic price features
        features['price'] = current_price
        features['returns'] = (current_price - prev_price) / prev_price if prev_price != 0 else 0
        features['log_returns'] = np.log(current_price / prev_price) if prev_price > 0 else 0
        
        # Price ratios
        features['high_low_ratio'] = highs[-1] / lows[-1] if lows[-1] != 0 else 1
        features['close_open_ratio'] = current_price / opens[-1] if opens[-1] != 0 else 1
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(prices) >= window:
                ma = np.mean(prices[-window:])
                features[f'sma_{window}'] = ma
                features[f'price_to_sma_{window}'] = current_price / ma if ma != 0 else 1
                
                # Standard deviation
                std = np.std(prices[-window:])
                features[f'std_{window}'] = std
                features[f'bollinger_upper_{window}'] = ma + 2 * std
                features[f'bollinger_lower_{window}'] = ma - 2 * std
        
        return features
    
    def compute_technical_indicators(self) -> Dict[str, float]:
        """Compute technical indicators using TA-Lib"""
        if len(self.price_buffer) < 20:
            return {}
        
        features = {}
        
        try:
            prices = np.array([bar['close'] for bar in self.price_buffer], dtype=np.float64)
            highs = np.array([bar['high'] for bar in self.price_buffer], dtype=np.float64)
            lows = np.array([bar['low'] for bar in self.price_buffer], dtype=np.float64)
            volumes = np.array(list(self.volume_buffer), dtype=np.float64)
            
            # RSI
            for period in [14, 21]:
                if len(prices) >= period:
                    rsi = talib.RSI(prices, timeperiod=period)
                    if not np.isnan(rsi[-1]):
                        features[f'rsi_{period}'] = rsi[-1]
            
            # MACD
            if len(prices) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(prices)
                if not np.isnan(macd[-1]):
                    features['macd'] = macd[-1]
                    features['macd_signal'] = macd_signal[-1]
                    features['macd_histogram'] = macd_hist[-1]
            
            # Bollinger Bands
            if len(prices) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                if not np.isnan(bb_upper[-1]):
                    features['bb_upper'] = bb_upper[-1]
                    features['bb_middle'] = bb_middle[-1]
                    features['bb_lower'] = bb_lower[-1]
                    features['bb_position'] = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Stochastic Oscillator
            if len(prices) >= 14:
                slowk, slowd = talib.STOCH(highs, lows, prices)
                if not np.isnan(slowk[-1]):
                    features['stoch_k'] = slowk[-1]
                    features['stoch_d'] = slowd[-1]
            
            # ATR (Average True Range)
            if len(prices) >= 14:
                atr = talib.ATR(highs, lows, prices)
                if not np.isnan(atr[-1]):
                    features['atr'] = atr[-1]
                    features['atr_ratio'] = atr[-1] / prices[-1] if prices[-1] != 0 else 0
            
            # Williams %R
            if len(prices) >= 14:
                willr = talib.WILLR(highs, lows, prices)
                if not np.isnan(willr[-1]):
                    features['williams_r'] = willr[-1]
            
            # CCI (Commodity Channel Index)
            if len(prices) >= 14:
                cci = talib.CCI(highs, lows, prices)
                if not np.isnan(cci[-1]):
                    features['cci'] = cci[-1]
                    
        except Exception as e:
            print(f"Error computing technical indicators: {e}")
            
        return features
    
    def compute_volume_features(self) -> Dict[str, float]:
        """Compute volume-based features"""
        if len(self.volume_buffer) < 10:
            return {}
        
        features = {}
        volumes = list(self.volume_buffer)
        current_volume = volumes[-1]
        
        # Volume moving averages
        for window in [5, 10, 20]:
            if len(volumes) >= window:
                vol_ma = np.mean(volumes[-window:])
                features[f'volume_sma_{window}'] = vol_ma
                features[f'volume_ratio_{window}'] = current_volume / vol_ma if vol_ma != 0 else 1
        
        # Volume trend
        if len(volumes) >= 5:
            recent_vol = np.mean(volumes[-5:])
            older_vol = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else recent_vol
            features['volume_trend'] = recent_vol / older_vol if older_vol != 0 else 1
        
        return features
    
    def compute_volatility_features(self) -> Dict[str, float]:
        """Compute volatility-based features"""
        if len(self.price_buffer) < 20:
            return {}
        
        features = {}
        returns = []
        prices = [bar['close'] for bar in self.price_buffer]
        
        # Calculate returns
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 10:
            return features
        
        # Realized volatility
        for window in [10, 20, 50]:
            if len(returns) >= window:
                vol = np.std(returns[-window:]) * np.sqrt(252 * 24)  # Annualized hourly vol
                features[f'volatility_{window}'] = vol
                
                # Volatility ratio
                if window == 20 and len(returns) >= 50:
                    long_vol = np.std(returns[-50:]) * np.sqrt(252 * 24)
                    features['volatility_ratio'] = vol / long_vol if long_vol != 0 else 1
        
        # Volatility trend
        if len(returns) >= 20:
            recent_vol = np.std(returns[-10:])
            older_vol = np.std(returns[-20:-10])
            features['volatility_trend'] = recent_vol / older_vol if older_vol != 0 else 1
        
        return features
    
    def compute_time_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Compute time-based features"""
        features = {}
        
        # Hour of day (normalized)
        features['hour'] = timestamp.hour / 23.0
        features['minute'] = timestamp.minute / 59.0
        
        # Day of week (normalized)
        features['day_of_week'] = timestamp.dayofweek / 6.0
        
        # Month (normalized)
        features['month'] = (timestamp.month - 1) / 11.0
        
        # Market session indicators
        hour = timestamp.hour
        features['pre_market'] = 1.0 if 4 <= hour < 9 else 0.0
        features['market_hours'] = 1.0 if 9 <= hour < 16 else 0.0
        features['after_hours'] = 1.0 if 16 <= hour < 20 else 0.0
        features['overnight'] = 1.0 if hour >= 20 or hour < 4 else 0.0
        
        return features
    
    def compute_momentum_features(self) -> Dict[str, float]:
        """Compute momentum-based features"""
        if len(self.price_buffer) < 10:
            return {}
        
        features = {}
        prices = [bar['close'] for bar in self.price_buffer]
        
        # Rate of change
        for period in [5, 10, 20]:
            if len(prices) > period:
                roc = (prices[-1] - prices[-period-1]) / prices[-period-1] if prices[-period-1] != 0 else 0
                features[f'roc_{period}'] = roc
        
        # Price momentum
        if len(prices) >= 10:
            momentum = prices[-1] - prices[-10]
            features['momentum_10'] = momentum
            
            # Momentum acceleration
            if len(prices) >= 20:
                prev_momentum = prices[-10] - prices[-20]
                features['momentum_acceleration'] = momentum - prev_momentum
        
        return features
    
    def generate_features(self, market_data: MarketData) -> FeatureSet:
        """
        Generate complete feature set for given market data
        Optimized for real-time performance
        """
        # Update buffers
        self.update_buffers(market_data)
        
        # Compute all feature categories
        all_features = {}
        
        # Price features
        all_features.update(self.compute_price_features())
        
        # Technical indicators
        all_features.update(self.compute_technical_indicators())
        
        # Volume features
        all_features.update(self.compute_volume_features())
        
        # Volatility features
        all_features.update(self.compute_volatility_features())
        
        # Time features
        all_features.update(self.compute_time_features(market_data.timestamp))
        
        # Momentum features
        all_features.update(self.compute_momentum_features())
        
        # Clean features (remove NaN values)
        clean_features = {}
        for key, value in all_features.items():
            if not np.isnan(value) and not np.isinf(value):
                clean_features[key] = float(value)
        
        return FeatureSet(
            timestamp=market_data.timestamp,
            features=clean_features
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        # Generate dummy features to get all possible names
        dummy_data = MarketData(
            timestamp=pd.Timestamp.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000
        )
        
        # Fill buffer with dummy data
        for _ in range(self.lookback_periods):
            self.update_buffers(dummy_data)
        
        feature_set = self.generate_features(dummy_data)
        return list(feature_set.features.keys())

def main():
    """Test feature engineering"""
    print("Testing Real-Time Feature Engineering")
    print("="*50)
    
    engine = RealTimeFeatureEngine()
    
    # Generate test data
    base_price = 4500.0
    for i in range(100):
        price_change = np.random.normal(0, 0.001)
        current_price = base_price * (1 + price_change)
        
        market_data = MarketData(
            timestamp=pd.Timestamp.now() + pd.Timedelta(minutes=i),
            open=current_price * 0.999,
            high=current_price * 1.001,
            low=current_price * 0.998,
            close=current_price,
            volume=np.random.randint(1000, 5000)
        )
        
        features = engine.generate_features(market_data)
        
        if i % 20 == 0:
            print(f"Step {i}: Generated {len(features.features)} features")
            if features.features:
                sample_features = dict(list(features.features.items())[:5])
                print(f"Sample features: {sample_features}")
    
    print(f"\nFinal feature count: {len(features.features)}")
    print("Feature engineering test completed!")

if __name__ == "__main__":
    main()