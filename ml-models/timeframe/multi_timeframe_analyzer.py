"""
Multi-Timeframe Analysis Module for Institutional Trading
==================================================

This module implements advanced multi-timeframe analysis capabilities:
- Hierarchical feature extraction across multiple timeframes
- Cross-timeframe signal confirmation and divergence detection
- Time series aggregation and alignment
- Institutional-grade timeframe correlation analysis

Supports: 1min, 5min, 15min, 1hr, 4hr, daily timeframes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, use fallback if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not available, using fallback calculations")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Advanced multi-timeframe analysis for institutional trading
    
    Features:
    - Hierarchical time series aggregation
    - Cross-timeframe signal alignment
    - Momentum divergence detection  
    - Institutional pattern recognition
    """
    
    def __init__(self, base_timeframe: str = '1min'):
        """
        Initialize the multi-timeframe analyzer
        
        Args:
            base_timeframe: Base timeframe for analysis ('1min', '5min', etc.)
        """
        self.base_timeframe = base_timeframe
        self.timeframes = ['1min', '5min', '15min', '1H', '4H', '1D']
        self.aggregation_rules = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'last': 'last',
            'bid': 'last',
            'ask': 'last'
        }
        
        # Timeframe multipliers for feature alignment
        self.tf_multipliers = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '1H': 60,
            '4H': 240,
            '1D': 1440
        }
        
        self.scalers = {}
        logger.info(f"🕐 Multi-timeframe analyzer initialized with base timeframe: {base_timeframe}")
    
    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to target timeframe with proper OHLCV aggregation
        
        Args:
            data: DataFrame with OHLCV data
            target_timeframe: Target timeframe ('5min', '1H', etc.)
        
        Returns:
            Resampled DataFrame
        """
        try:
            # Ensure we have a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    logger.warning("No timestamp column found, using existing index")
            
            # Create aggregation rules for available columns
            agg_rules = {col: rule for col, rule in self.aggregation_rules.items() 
                        if col in data.columns}
            
            if not agg_rules:
                logger.error("No valid columns found for aggregation")
                return pd.DataFrame()
            
            # Resample data
            resampled = data.resample(target_timeframe).agg(agg_rules)
            
            # Drop rows with NaN in OHLC
            if 'close' in resampled.columns:
                resampled = resampled.dropna(subset=['close'])
            
            logger.debug(f"📊 Resampled {len(data)} -> {len(resampled)} bars for {target_timeframe}")
            return resampled
            
        except Exception as e:
            logger.error(f"❌ Error resampling data to {target_timeframe}: {e}")
            return pd.DataFrame()
    
    def extract_timeframe_features(self, data: pd.DataFrame, timeframe: str) -> Dict[str, float]:
        """
        Extract features specific to a timeframe
        
        Args:
            data: OHLCV data for the timeframe
            timeframe: Timeframe identifier
        
        Returns:
            Dictionary of timeframe-specific features
        """
        features = {}
        
        if len(data) < 20:  # Need minimum data for calculations
            return {}
        
        # Price features
        close = data['close']
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        volume = data['volume'] if 'volume' in data.columns else pd.Series(1000, index=data.index)
        
        # Trend features
        features[f'{timeframe}_sma_20'] = close.rolling(20).mean().iloc[-1]
        features[f'{timeframe}_sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        features[f'{timeframe}_ema_12'] = close.ewm(span=12).mean().iloc[-1]
        features[f'{timeframe}_ema_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # Momentum features
        features[f'{timeframe}_rsi'] = self._calculate_rsi(close, 14)
        features[f'{timeframe}_macd'], features[f'{timeframe}_macd_signal'] = self._calculate_macd(close)
        features[f'{timeframe}_atr'] = self._calculate_atr(high, low, close, 14)
        
        # Volume features
        features[f'{timeframe}_volume_sma'] = volume.rolling(20).mean().iloc[-1]
        features[f'{timeframe}_volume_ratio'] = volume.iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1e-10)
        
        # Volatility features
        returns = close.pct_change().dropna()
        if len(returns) > 1:
            features[f'{timeframe}_volatility'] = returns.std()
            features[f'{timeframe}_skewness'] = returns.skew()
            features[f'{timeframe}_kurtosis'] = returns.kurtosis()
        
        # Support/Resistance levels
        features[f'{timeframe}_support'] = low.rolling(20).min().iloc[-1]
        features[f'{timeframe}_resistance'] = high.rolling(20).max().iloc[-1]
        features[f'{timeframe}_support_strength'] = self._calculate_support_strength(close, low)
        features[f'{timeframe}_resistance_strength'] = self._calculate_resistance_strength(close, high)
        
        # Momentum divergence
        features[f'{timeframe}_price_momentum'] = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
        features[f'{timeframe}_volume_momentum'] = (volume.iloc[-1] - volume.rolling(20).mean().iloc[-1]) / (volume.rolling(20).mean().iloc[-1] + 1e-10)
        
        # Handle NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if HAS_TALIB:
            try:
                rsi = talib.RSI(prices.values, timeperiod=period)
                return rsi[-1] if not np.isnan(rsi[-1]) else 50.0
            except:
                pass
        
        # Fallback calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        if HAS_TALIB:
            try:
                macd, signal, _ = talib.MACD(prices.values)
                return (macd[-1] if not np.isnan(macd[-1]) else 0.0,
                       signal[-1] if not np.isnan(signal[-1]) else 0.0)
            except:
                pass
        
        # Fallback calculation
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
               signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0)
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if HAS_TALIB:
            try:
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                return atr[-1] if not np.isnan(atr[-1]) else 0.0
            except:
                pass
        
        # Fallback calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_support_strength(self, close: pd.Series, low: pd.Series) -> float:
        """Calculate support level strength"""
        if len(close) < 20:
            return 0.0
        
        try:
            support_level = low.rolling(20).min().iloc[-1]
            recent_lows = low.tail(20)
            touches = sum(1 for price in recent_lows if abs(price - support_level) / (support_level + 1e-10) < 0.001)
            return min(touches / 20.0, 1.0)  # Normalize to [0, 1]
        except:
            return 0.0
    
    def _calculate_resistance_strength(self, close: pd.Series, high: pd.Series) -> float:
        """Calculate resistance level strength"""
        if len(close) < 20:
            return 0.0
        
        try:
            resistance_level = high.rolling(20).max().iloc[-1]
            recent_highs = high.tail(20)
            touches = sum(1 for price in recent_highs if abs(price - resistance_level) / (resistance_level + 1e-10) < 0.001)
            return min(touches / 20.0, 1.0)  # Normalize to [0, 1]
        except:
            return 0.0
    
    def detect_divergences(self, timeframe_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Detect divergences between timeframes
        
        Args:
            timeframe_features: Features for each timeframe
        
        Returns:
            Divergence features
        """
        divergences = {}
        
        # Check price momentum divergences
        short_tf = '1min'
        medium_tf = '15min'
        long_tf = '1H'
        
        if all(tf in timeframe_features for tf in [short_tf, medium_tf, long_tf]):
            # Price momentum divergences
            short_momentum = timeframe_features[short_tf].get(f'{short_tf}_price_momentum', 0)
            medium_momentum = timeframe_features[medium_tf].get(f'{medium_tf}_price_momentum', 0)
            long_momentum = timeframe_features[long_tf].get(f'{long_tf}_price_momentum', 0)
            
            divergences['short_medium_divergence'] = short_momentum - medium_momentum
            divergences['medium_long_divergence'] = medium_momentum - long_momentum
            divergences['short_long_divergence'] = short_momentum - long_momentum
            
            # RSI divergences
            short_rsi = timeframe_features[short_tf].get(f'{short_tf}_rsi', 50)
            medium_rsi = timeframe_features[medium_tf].get(f'{medium_tf}_rsi', 50)
            long_rsi = timeframe_features[long_tf].get(f'{long_tf}_rsi', 50)
            
            divergences['rsi_short_medium_div'] = short_rsi - medium_rsi
            divergences['rsi_medium_long_div'] = medium_rsi - long_rsi
            
            # Volume divergences
            short_vol = timeframe_features[short_tf].get(f'{short_tf}_volume_momentum', 0)
            medium_vol = timeframe_features[medium_tf].get(f'{medium_tf}_volume_momentum', 0)
            
            divergences['volume_divergence'] = short_vol - medium_vol
            
            # Trend alignment score
            short_trend = 1 if short_momentum > 0 else -1
            medium_trend = 1 if medium_momentum > 0 else -1
            long_trend = 1 if long_momentum > 0 else -1
            
            divergences['trend_alignment'] = (short_trend + medium_trend + long_trend) / 3.0
        
        return divergences
    
    def analyze_multi_timeframe(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Main multi-timeframe analysis function
        
        Args:
            data: Base timeframe OHLCV data
        
        Returns:
            Combined multi-timeframe features
        """
        logger.info(f"🔄 Starting multi-timeframe analysis with {len(data)} base periods")
        
        all_features = {}
        timeframe_features = {}
        
        for timeframe in self.timeframes:
            try:
                # Skip if target timeframe is shorter than base
                if (self.tf_multipliers.get(timeframe, 1) < 
                    self.tf_multipliers.get(self.base_timeframe, 1)):
                    continue
                
                # Resample data
                if timeframe == self.base_timeframe:
                    tf_data = data.copy()
                else:
                    tf_data = self.resample_data(data, timeframe)
                
                if len(tf_data) < 10:  # Need minimum data
                    logger.warning(f"⚠️  Insufficient data for {timeframe}: {len(tf_data)} periods")
                    continue
                
                # Extract features for this timeframe
                tf_features = self.extract_timeframe_features(tf_data, timeframe)
                timeframe_features[timeframe] = tf_features
                all_features.update(tf_features)
                
                logger.debug(f"✅ Generated {len(tf_features)} features for {timeframe}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {timeframe}: {e}")
                continue
        
        # Detect cross-timeframe divergences
        try:
            divergence_features = self.detect_divergences(timeframe_features)
            all_features.update(divergence_features)
            logger.debug(f"✅ Generated {len(divergence_features)} divergence features")
        except Exception as e:
            logger.error(f"❌ Error detecting divergences: {e}")
        
        # Calculate timeframe strength indicators
        try:
            strength_features = self._calculate_timeframe_strength(timeframe_features)
            all_features.update(strength_features)
        except Exception as e:
            logger.error(f"❌ Error calculating timeframe strength: {e}")
        
        logger.info(f"🎯 Multi-timeframe analysis complete: {len(all_features)} total features")
        return all_features
    
    def _calculate_timeframe_strength(self, timeframe_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate relative strength across timeframes"""
        strength_features = {}
        
        # RSI strength comparison
        rsi_values = []
        for tf in self.timeframes:
            if tf in timeframe_features:
                rsi = timeframe_features[tf].get(f'{tf}_rsi', 50)
                rsi_values.append(rsi)
        
        if rsi_values:
            strength_features['rsi_cross_tf_mean'] = np.mean(rsi_values)
            strength_features['rsi_cross_tf_std'] = np.std(rsi_values)
            strength_features['rsi_consistency'] = 1.0 - (np.std(rsi_values) / 50.0)  # Lower std = higher consistency
        
        # Volatility consistency
        vol_values = []
        for tf in self.timeframes:
            if tf in timeframe_features:
                vol = timeframe_features[tf].get(f'{tf}_volatility', 0)
                vol_values.append(vol)
        
        if vol_values:
            strength_features['volatility_cross_tf_mean'] = np.mean(vol_values)
            strength_features['volatility_trend'] = np.corrcoef(range(len(vol_values)), vol_values)[0, 1] if len(vol_values) > 1 else 0
        
        return strength_features

# Testing and Usage Example
if __name__ == "__main__":
    logger.info("🧪 Testing Multi-Timeframe Analyzer...")
    
    # Create comprehensive sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=2000, freq='1min')  # More data for proper resampling
    
    # Generate realistic intraday price data with trend and noise
    base_price = 4400
    trend = np.linspace(0, 50, 2000)  # Upward trend
    noise = np.cumsum(np.random.normal(0, 0.5, 2000))
    prices = base_price + trend + noise
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.2, 2000),
        'high': prices + np.abs(np.random.normal(0, 1.0, 2000)),
        'low': prices - np.abs(np.random.normal(0, 1.0, 2000)),
        'close': prices,
        'volume': np.random.randint(500, 2000, 2000).astype(float),
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    # Test the analyzer
    analyzer = MultiTimeframeAnalyzer(base_timeframe='1min')
    
    try:
        # Analyze multi-timeframe features
        features = analyzer.analyze_multi_timeframe(sample_data)
        
        logger.info(f"✅ Multi-timeframe analysis generated {len(features)} features")
        
        # Show sample features
        sample_features = dict(list(features.items())[:10])
        logger.info(f"📊 Sample features: {sample_features}")
        
        # Test resampling
        for tf in ['5min', '15min', '1H']:
            resampled = analyzer.resample_data(sample_data, tf)
            logger.info(f"📈 {tf} resampling: {len(sample_data)} -> {len(resampled)} periods")
        
        logger.info("🎉 Multi-timeframe analyzer testing complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in multi-timeframe testing: {e}")
        import traceback
        traceback.print_exc()
