"""
Real-Time Feature Store
High-performance feature computation and storage for live trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging
from collections import deque
import redis
import json
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:
    """Real-time feature vector with metadata"""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    confidence: float = 1.0
    latency_ms: float = 0.0
    source: str = "realtime"

@dataclass
class FeatureConfig:
    """Configuration for feature computation"""
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    window_size: int = 20
    update_frequency_ms: int = 100
    enabled: bool = True

class RealTimeFeatureStore:
    """
    High-performance real-time feature store for institutional trading
    
    Features:
    - Sub-millisecond feature computation
    - Incremental updates with rolling windows
    - Redis caching for ultra-low latency access
    - Multi-threaded feature computation
    - Dependency management for complex features
    """
    
    def __init__(self, redis_config: Dict = None, max_history: int = 10000):
        self.redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 0}
        self.max_history = max_history
        
        # Data storage
        self.price_history = {}  # symbol -> deque of prices
        self.volume_history = {}  # symbol -> deque of volumes
        self.feature_cache = {}  # symbol -> latest features
        
        # Feature computation
        self.feature_configs = {}
        self.computation_threads = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Redis connection for ultra-fast access
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            self.redis_client.ping()
            logger.info("Connected to Redis for feature caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
            self.redis_client = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize built-in features
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize built-in financial features"""
        
        # Price-based features
        self.register_feature(FeatureConfig(
            name="sma_20",
            function=self._compute_sma,
            window_size=20
        ))
        
        self.register_feature(FeatureConfig(
            name="ema_12",
            function=self._compute_ema,
            window_size=12
        ))
        
        self.register_feature(FeatureConfig(
            name="rsi_14",
            function=self._compute_rsi,
            window_size=14
        ))
        
        self.register_feature(FeatureConfig(
            name="bollinger_position",
            function=self._compute_bollinger_position,
            window_size=20
        ))
        
        # Volatility features
        self.register_feature(FeatureConfig(
            name="volatility_20",
            function=self._compute_volatility,
            window_size=20
        ))
        
        # Momentum features
        self.register_feature(FeatureConfig(
            name="momentum_10",
            function=self._compute_momentum,
            window_size=10
        ))
        
        # Volume features
        self.register_feature(FeatureConfig(
            name="volume_sma_20",
            function=self._compute_volume_sma,
            window_size=20
        ))
        
        # Cross-asset features (for multiple instruments)
        self.register_feature(FeatureConfig(
            name="correlation_spy_5d",
            function=self._compute_correlation,
            dependencies=["SPY"],
            window_size=120  # 5 days of 1-minute bars
        ))
    
    def register_feature(self, config: FeatureConfig):
        """Register a new feature for computation"""
        with self.lock:
            self.feature_configs[config.name] = config
            logger.info(f"Registered feature: {config.name}")
    
    def update_market_data(self, symbol: str, price: float, volume: int, timestamp: datetime = None):
        """
        Update market data and trigger feature computation
        
        This is the main entry point for real-time data
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        start_time = time.perf_counter()
        
        with self.lock:
            # Initialize symbol data if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.max_history)
                self.volume_history[symbol] = deque(maxlen=self.max_history)
                self.feature_cache[symbol] = {}
            
            # Add new data
            self.price_history[symbol].append((timestamp, price))
            self.volume_history[symbol].append((timestamp, volume))
        
        # Compute features asynchronously for better performance
        self.executor.submit(self._compute_features_async, symbol, timestamp, start_time)
    
    def _compute_features_async(self, symbol: str, timestamp: datetime, start_time: float):
        """Compute features asynchronously"""
        try:
            features = {}
            
            # Compute all enabled features
            for feature_name, config in self.feature_configs.items():
                if not config.enabled:
                    continue
                
                try:
                    value = config.function(symbol, config)
                    if value is not None:
                        features[feature_name] = float(value)
                except Exception as e:
                    logger.warning(f"Feature computation failed for {feature_name}: {e}")
            
            # Update cache
            with self.lock:
                self.feature_cache[symbol] = features
            
            # Cache in Redis for ultra-fast access
            if self.redis_client:
                try:
                    self.redis_client.hset(
                        f"features:{symbol}",
                        mapping={k: str(v) for k, v in features.items()}
                    )
                    self.redis_client.expire(f"features:{symbol}", 300)  # 5 minute expiry
                except Exception as e:
                    logger.warning(f"Redis caching failed: {e}")
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(f"Features computed for {symbol} in {latency_ms:.2f}ms: {len(features)} features")
            
        except Exception as e:
            logger.error(f"Async feature computation failed for {symbol}: {e}")
    
    def get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest features for a symbol"""
        
        # Try Redis first for ultra-low latency
        if self.redis_client:
            try:
                redis_features = self.redis_client.hgetall(f"features:{symbol}")
                if redis_features:
                    return {k.decode(): float(v.decode()) for k, v in redis_features.items()}
            except Exception as e:
                logger.warning(f"Redis feature retrieval failed: {e}")
        
        # Fallback to memory cache
        with self.lock:
            return self.feature_cache.get(symbol, {}).copy()
    
    def get_feature_vector(self, symbol: str) -> Optional[FeatureVector]:
        """Get complete feature vector with metadata"""
        features = self.get_features(symbol)
        if not features:
            return None
        
        return FeatureVector(
            timestamp=datetime.now(),
            symbol=symbol,
            features=features,
            confidence=1.0,
            source="realtime"
        )
    
    # Feature computation methods
    def _compute_sma(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Simple Moving Average"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])[-config.window_size:]]
        if len(prices) < config.window_size:
            return None
        
        return np.mean(prices)
    
    def _compute_ema(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Exponential Moving Average"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])[-config.window_size:]]
        if len(prices) < 2:
            return None
        
        alpha = 2.0 / (config.window_size + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _compute_rsi(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Relative Strength Index"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])[-config.window_size-1:]]
        if len(prices) < config.window_size + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_bollinger_position(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Position within Bollinger Bands"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])[-config.window_size:]]
        if len(prices) < config.window_size:
            return None
        
        sma = np.mean(prices)
        std = np.std(prices)
        current_price = prices[-1]
        
        if std == 0:
            return 0.5
        
        # Position between lower and upper band (0 = lower band, 1 = upper band)
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    def _compute_volatility(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Price volatility (standard deviation of returns)"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])[-config.window_size:]]
        if len(prices) < 2:
            return None
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def _compute_momentum(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Price momentum"""
        if symbol not in self.price_history:
            return None
        
        prices = [p[1] for p in list(self.price_history[symbol])]
        if len(prices) < config.window_size + 1:
            return None
        
        current_price = prices[-1]
        past_price = prices[-config.window_size-1]
        
        return (current_price - past_price) / past_price
    
    def _compute_volume_sma(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Volume Simple Moving Average"""
        if symbol not in self.volume_history:
            return None
        
        volumes = [v[1] for v in list(self.volume_history[symbol])[-config.window_size:]]
        if len(volumes) < config.window_size:
            return None
        
        return np.mean(volumes)
    
    def _compute_correlation(self, symbol: str, config: FeatureConfig) -> Optional[float]:
        """Correlation with reference instrument"""
        # This would require data from multiple symbols
        # Placeholder implementation
        return 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (would be updated by ML model)"""
        # This would be updated by the ML model during training
        return {
            "sma_20": 0.15,
            "rsi_14": 0.12,
            "volatility_20": 0.10,
            "momentum_10": 0.08,
            "bollinger_position": 0.07,
            "ema_12": 0.06,
            "volume_sma_20": 0.05
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if self.redis_client:
            self.redis_client.close()
        logger.info("Feature store cleanup completed")
