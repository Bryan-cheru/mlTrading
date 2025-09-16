"""
Feature Engineering Engine
Technical indicators, statistical features, and market microstructure features
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import talib
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Container for engineered features"""
    timestamp: pd.Timestamp
    symbol: str
    features: Dict[str, float]
    labels: Optional[Dict[str, float]] = None

class TechnicalIndicators:
    """
    Technical indicator calculations using TA-Lib and custom implementations
    """
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period).mean().values
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        try:
            return talib.RSI(prices, timeperiod=period)
        except:
            # Fallback implementation
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).values
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(prices, period)
        rolling_std = pd.Series(prices).rolling(window=period).std().values
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence)"""
        try:
            macd_line, signal_line, histogram = talib.MACD(prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
            return macd_line, signal_line, histogram
        except:
            # Fallback implementation
            ema_fast = TechnicalIndicators.ema(prices, fast_period)
            ema_slow = TechnicalIndicators.ema(prices, slow_period)
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalIndicators.ema(macd_line, signal_period)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return slowk, slowd
        except:
            # Fallback implementation
            lowest_low = pd.Series(low).rolling(window=k_period).min()
            highest_high = pd.Series(high).rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent.values, d_percent.values
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        try:
            return talib.ATR(high, low, close, timeperiod=period)
        except:
            # Fallback implementation
            high_low = high - low
            high_close_prev = np.abs(high - np.roll(close, 1))
            low_close_prev = np.abs(low - np.roll(close, 1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            return pd.Series(true_range).rolling(window=period).mean().values

class StatisticalFeatures:
    """
    Statistical and mathematical features
    """
    
    @staticmethod
    def returns(prices: np.ndarray, periods: List[int] = [1, 5, 10, 20]) -> Dict[str, np.ndarray]:
        """Calculate returns over multiple periods"""
        features = {}
        price_series = pd.Series(prices)
        
        for period in periods:
            returns = price_series.pct_change(periods=period)
            features[f'return_{period}d'] = returns.values
            features[f'log_return_{period}d'] = np.log(price_series / price_series.shift(period)).values
        
        return features
    
    @staticmethod
    def volatility(prices: np.ndarray, windows: List[int] = [5, 10, 20, 50]) -> Dict[str, np.ndarray]:
        """Calculate volatility over multiple windows"""
        features = {}
        returns = pd.Series(prices).pct_change()
        
        for window in windows:
            vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
            features[f'volatility_{window}d'] = vol.values
        
        return features
    
    @staticmethod
    def momentum_features(prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Momentum-based features"""
        price_series = pd.Series(prices)
        volume_series = pd.Series(volumes)
        
        features = {}
        
        # Price momentum
        for period in [5, 10, 20]:
            momentum = (price_series - price_series.shift(period)) / price_series.shift(period)
            features[f'price_momentum_{period}d'] = momentum.values
        
        # Volume momentum
        for period in [5, 10, 20]:
            vol_momentum = (volume_series - volume_series.shift(period)) / volume_series.shift(period)
            features[f'volume_momentum_{period}d'] = vol_momentum.values
        
        # Price-volume correlation
        for window in [10, 20, 50]:
            corr = price_series.rolling(window=window).corr(volume_series)
            features[f'price_volume_corr_{window}d'] = corr.values
        
        return features
    
    @staticmethod
    def regime_features(prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Market regime detection features"""
        price_series = pd.Series(prices)
        returns = price_series.pct_change()
        
        features = {}
        
        # Trend strength
        for window in [10, 20, 50]:
            trend_strength = returns.rolling(window=window).mean() / returns.rolling(window=window).std()
            features[f'trend_strength_{window}d'] = trend_strength.values
        
        # Volatility regime
        for window in [20, 50]:
            vol_regime = returns.rolling(window=window).std() / returns.rolling(window=window*2).std()
            features[f'volatility_regime_{window}d'] = vol_regime.values
        
        # Volume regime
        volume_series = pd.Series(volumes)
        for window in [20, 50]:
            vol_ma = volume_series.rolling(window=window).mean()
            vol_regime = volume_series / vol_ma
            features[f'volume_regime_{window}d'] = vol_regime.values
        
        return features

class FeatureEngineering:
    """
    Main feature engineering class that combines all feature types
    """
    
    def __init__(self):
        self.technical = TechnicalIndicators()
        self.statistical = StatisticalFeatures()
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for a given OHLCV dataframe
        
        Args:
            df: DataFrame with columns ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract OHLCV arrays
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        # Initialize feature dictionary
        features = {}
        
        # Technical indicators
        try:
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                features[f'sma_{period}'] = self.technical.sma(close_prices, period)
                features[f'ema_{period}'] = self.technical.ema(close_prices, period)
            
            # RSI
            features['rsi_14'] = self.technical.rsi(close_prices, 14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical.bollinger_bands(close_prices, 20, 2)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            macd_line, signal_line, histogram = self.technical.macd(close_prices)
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            
            # Stochastic
            stoch_k, stoch_d = self.technical.stochastic(high_prices, low_prices, close_prices)
            features['stochastic_k'] = stoch_k
            features['stochastic_d'] = stoch_d
            
            # ATR
            features['atr_14'] = self.technical.atr(high_prices, low_prices, close_prices, 14)
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
        
        # Statistical features
        try:
            # Returns
            return_features = self.statistical.returns(close_prices)
            features.update(return_features)
            
            # Volatility
            vol_features = self.statistical.volatility(close_prices)
            features.update(vol_features)
            
            # Momentum
            momentum_features = self.statistical.momentum_features(close_prices, volumes)
            features.update(momentum_features)
            
            # Regime features
            regime_features = self.statistical.regime_features(close_prices, volumes)
            features.update(regime_features)
            
        except Exception as e:
            logger.warning(f"Error calculating statistical features: {e}")
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        
        # Combine with original data
        result_df = pd.concat([df, feature_df], axis=1)
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        logger.info(f"Engineered {len(self.feature_names)} features")
        
        return result_df
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 60) -> pd.DataFrame:
        """
        Create prediction labels for supervised learning
        
        Args:
            df: DataFrame with price data
            prediction_horizon: Minutes ahead to predict
        
        Returns:
            DataFrame with labels added
        """
        if 'close' not in df.columns:
            return df
        
        # Forward returns (our prediction target)
        df['label_return'] = df['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        
        # Classification labels
        df['label_direction'] = np.where(df['label_return'] > 0, 1, 0)  # Up/Down
        df['label_magnitude'] = np.where(np.abs(df['label_return']) > 0.01, 1, 0)  # Significant move
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame, normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix for ML models
        
        Args:
            df: DataFrame with engineered features
            normalize: Whether to normalize features
        
        Returns:
            Feature matrix and feature names
        """
        if not self.feature_names:
            # Extract feature columns (exclude original OHLCV and metadata)
            exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']
            label_cols = [col for col in df.columns if col.startswith('label_')]
            exclude_cols.extend(label_cols)
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        # Extract feature matrix
        feature_matrix = df[self.feature_names].values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if requested
        if normalize:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix, self.feature_names

def main():
    """
    Example usage of feature engineering
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    # Simulate price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 1000)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Engineer features
    fe = FeatureEngineering()
    df_with_features = fe.engineer_features(df)
    df_with_labels = fe.create_labels(df_with_features)
    
    print(f"Original shape: {df.shape}")
    print(f"With features: {df_with_features.shape}")
    print(f"Feature names: {fe.feature_names[:10]}...")  # Show first 10
    
    # Get feature matrix
    X, feature_names = fe.get_feature_matrix(df_with_features)
    print(f"Feature matrix shape: {X.shape}")

if __name__ == "__main__":
    main()
