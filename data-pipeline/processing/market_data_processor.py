"""
Market Data Processor
Processes real market data from NinjaTrader 8 and other sources
Calculates technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

# Technical indicators
try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volume import VolumeSMAIndicator, OnBalanceVolumeIndicator
except ImportError:
    logging.warning("TA library not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "ta"])
    import ta

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """
    Processes market data and calculates features for ML models
    Optimized for real-time processing with NinjaTrader 8 data
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.last_calculation_time = {}
        
    def calculate_features(self, bars_1m: pd.DataFrame, bars_5m: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive features from market data
        
        Args:
            bars_1m: 1-minute OHLCV data
            bars_5m: 5-minute OHLCV data (optional)
            
        Returns:
            Dictionary of calculated features
        """
        if bars_1m.empty:
            return {}
        
        try:
            features = {}
            
            # Basic price features
            features.update(self._calculate_price_features(bars_1m))
            
            # Technical indicators
            features.update(self._calculate_technical_indicators(bars_1m))
            
            # Volume features
            features.update(self._calculate_volume_features(bars_1m))
            
            # Statistical features
            features.update(self._calculate_statistical_features(bars_1m))
            
            # Market microstructure features
            features.update(self._calculate_microstructure_features(bars_1m))
            
            # Multi-timeframe features (if 5m data available)
            if bars_5m is not None and not bars_5m.empty:
                features.update(self._calculate_multi_timeframe_features(bars_1m, bars_5m))
            
            # Regime detection features
            features.update(self._calculate_regime_features(bars_1m))
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return {}
    
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict:
        """Calculate basic price-based features"""
        features = {}
        
        if len(df) < 2:
            return features
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # Returns
        features['return_1'] = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0
        features['return_5'] = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else 0
        features['return_15'] = (close.iloc[-1] - close.iloc[-16]) / close.iloc[-16] if len(close) >= 16 else 0
        
        # Price levels
        features['price_close'] = close.iloc[-1]
        features['price_high'] = high.iloc[-1]
        features['price_low'] = low.iloc[-1]
        features['price_open'] = open_price.iloc[-1]
        
        # Range features
        features['true_range'] = max(
            high.iloc[-1] - low.iloc[-1],
            abs(high.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0,
            abs(low.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0
        )
        
        features['body_size'] = abs(close.iloc[-1] - open_price.iloc[-1])
        features['upper_shadow'] = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        features['lower_shadow'] = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        return features
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        features = {}
        
        if len(df) < 20:
            return features
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        try:
            # Moving averages
            sma_20 = SMAIndicator(close=close, window=20).sma_indicator()
            ema_12 = EMAIndicator(close=close, window=12).ema_indicator()
            ema_26 = EMAIndicator(close=close, window=26).ema_indicator()
            
            features['sma_20'] = sma_20.iloc[-1] if not sma_20.empty else 0
            features['ema_12'] = ema_12.iloc[-1] if not ema_12.empty else 0
            features['ema_26'] = ema_26.iloc[-1] if not ema_26.empty else 0
            
            # Price relative to moving averages
            current_price = close.iloc[-1]
            features['price_vs_sma20'] = (current_price - features['sma_20']) / features['sma_20'] if features['sma_20'] != 0 else 0
            features['price_vs_ema12'] = (current_price - features['ema_12']) / features['ema_12'] if features['ema_12'] != 0 else 0
            
            # MACD
            if len(df) >= 26:
                macd_line = MACD(close=close).macd()
                macd_signal = MACD(close=close).macd_signal()
                macd_diff = MACD(close=close).macd_diff()
                
                features['macd_line'] = macd_line.iloc[-1] if not macd_line.empty else 0
                features['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0
                features['macd_histogram'] = macd_diff.iloc[-1] if not macd_diff.empty else 0
            
            # RSI
            if len(df) >= 14:
                rsi = RSIIndicator(close=close, window=14).rsi()
                features['rsi_14'] = rsi.iloc[-1] if not rsi.empty else 50
            
            # Bollinger Bands
            if len(df) >= 20:
                bb = BollingerBands(close=close, window=20, window_dev=2)
                bb_upper = bb.bollinger_hband()
                bb_lower = bb.bollinger_lband()
                bb_middle = bb.bollinger_mavg()
                
                features['bb_upper'] = bb_upper.iloc[-1] if not bb_upper.empty else current_price
                features['bb_lower'] = bb_lower.iloc[-1] if not bb_lower.empty else current_price
                features['bb_middle'] = bb_middle.iloc[-1] if not bb_middle.empty else current_price
                
                # BB position
                bb_width = features['bb_upper'] - features['bb_lower']
                features['bb_position'] = (current_price - features['bb_lower']) / bb_width if bb_width != 0 else 0.5
            
            # ATR
            if len(df) >= 14:
                atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
                features['atr_14'] = atr.iloc[-1] if not atr.empty else 0
            
            # Stochastic
            if len(df) >= 14:
                stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
                features['stoch_k'] = stoch.stoch().iloc[-1] if not stoch.stoch().empty else 50
                features['stoch_d'] = stoch.stoch_signal().iloc[-1] if not stoch.stoch_signal().empty else 50
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based features"""
        features = {}
        
        if len(df) < 5:
            return features
        
        volume = df['volume']
        close = df['close']
        
        try:
            # Volume statistics
            features['volume_current'] = volume.iloc[-1]
            features['volume_avg_20'] = volume.tail(20).mean() if len(volume) >= 20 else volume.mean()
            features['volume_ratio'] = volume.iloc[-1] / features['volume_avg_20'] if features['volume_avg_20'] != 0 else 1
            
            # Volume trend
            features['volume_trend_5'] = volume.tail(5).corr(pd.Series(range(5))) if len(volume) >= 5 else 0
            
            # VWAP
            if len(df) >= 10:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (typical_price * volume).cumsum() / volume.cumsum()
                features['vwap'] = vwap.iloc[-1]
                features['price_vs_vwap'] = (close.iloc[-1] - features['vwap']) / features['vwap']
            
            # On Balance Volume
            if len(df) >= 10:
                obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
                features['obv'] = obv.iloc[-1] if not obv.empty else 0
                
                # OBV trend
                if len(obv) >= 5:
                    features['obv_trend_5'] = obv.tail(5).corr(pd.Series(range(5)))
        
        except Exception as e:
            logger.error(f"Error calculating volume features: {e}")
        
        return features
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical features"""
        features = {}
        
        if len(df) < 10:
            return features
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        try:
            # Volatility
            features['volatility_10'] = returns.tail(10).std() * np.sqrt(252 * 24 * 60) if len(returns) >= 10 else 0
            features['volatility_20'] = returns.tail(20).std() * np.sqrt(252 * 24 * 60) if len(returns) >= 20 else 0
            
            # Skewness and Kurtosis
            if len(returns) >= 20:
                features['skewness_20'] = returns.tail(20).skew()
                features['kurtosis_20'] = returns.tail(20).kurtosis()
            
            # Rolling correlations (if we have enough data)
            if len(returns) >= 50:
                # Autocorrelation
                features['autocorr_1'] = returns.tail(50).autocorr(lag=1)
                features['autocorr_5'] = returns.tail(50).autocorr(lag=5)
            
            # Z-score (price relative to recent mean)
            if len(close) >= 20:
                mean_20 = close.tail(20).mean()
                std_20 = close.tail(20).std()
                features['zscore_20'] = (close.iloc[-1] - mean_20) / std_20 if std_20 != 0 else 0
        
        except Exception as e:
            logger.error(f"Error calculating statistical features: {e}")
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Calculate market microstructure features"""
        features = {}
        
        if len(df) < 5:
            return features
        
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            open_price = df['open']
            
            # Price impact measures
            features['high_low_ratio'] = high.iloc[-1] / low.iloc[-1] if low.iloc[-1] != 0 else 1
            features['close_open_ratio'] = close.iloc[-1] / open_price.iloc[-1] if open_price.iloc[-1] != 0 else 1
            
            # Gap features
            if len(df) >= 2:
                features['gap'] = (open_price.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if close.iloc[-2] != 0 else 0
            
            # Doji pattern detection
            body_size = abs(close.iloc[-1] - open_price.iloc[-1])
            range_size = high.iloc[-1] - low.iloc[-1]
            features['doji_ratio'] = body_size / range_size if range_size != 0 else 0
            
            # Hammer/Shooting star patterns
            if range_size != 0:
                upper_shadow = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
                lower_shadow = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
                
                features['upper_shadow_ratio'] = upper_shadow / range_size
                features['lower_shadow_ratio'] = lower_shadow / range_size
        
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
        
        return features
    
    def _calculate_multi_timeframe_features(self, bars_1m: pd.DataFrame, bars_5m: pd.DataFrame) -> Dict:
        """Calculate multi-timeframe features"""
        features = {}
        
        try:
            if len(bars_5m) < 5:
                return features
            
            # 5-minute timeframe features
            close_5m = bars_5m['close']
            
            # Trend alignment
            if len(close_5m) >= 3:
                trend_5m = 1 if close_5m.iloc[-1] > close_5m.iloc[-3] else -1
                trend_1m = 1 if bars_1m['close'].iloc[-1] > bars_1m['close'].iloc[-5] else -1
                
                features['trend_alignment'] = 1 if trend_5m == trend_1m else 0
            
            # Price relative to 5m levels
            current_price = bars_1m['close'].iloc[-1]
            features['price_vs_5m_high'] = (current_price - bars_5m['high'].iloc[-1]) / bars_5m['high'].iloc[-1]
            features['price_vs_5m_low'] = (current_price - bars_5m['low'].iloc[-1]) / bars_5m['low'].iloc[-1]
            
            # 5-minute momentum
            if len(close_5m) >= 2:
                features['momentum_5m'] = (close_5m.iloc[-1] - close_5m.iloc[-2]) / close_5m.iloc[-2]
        
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe features: {e}")
        
        return features
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> Dict:
        """Calculate regime detection features"""
        features = {}
        
        if len(df) < 50:
            return features
        
        try:
            close = df['close']
            volume = df['volume']
            
            # Volatility regime
            returns = close.pct_change().dropna()
            if len(returns) >= 20:
                recent_vol = returns.tail(10).std()
                historical_vol = returns.tail(50).std()
                features['vol_regime'] = recent_vol / historical_vol if historical_vol != 0 else 1
            
            # Volume regime
            recent_vol_avg = volume.tail(10).mean()
            historical_vol_avg = volume.tail(50).mean()
            features['volume_regime'] = recent_vol_avg / historical_vol_avg if historical_vol_avg != 0 else 1
            
            # Trend strength
            if len(close) >= 20:
                high_20 = close.tail(20).max()
                low_20 = close.tail(20).min()
                current_price = close.iloc[-1]
                
                # Position in range
                features['range_position'] = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
        
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
        
        return features
