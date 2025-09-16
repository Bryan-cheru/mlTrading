"""
Real-time Data Manager
Manages multiple data sources and provides unified interface
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .ninjatrader_connector import NinjaTraderDataProvider, MarketData, BarData
from ..processing.market_data_processor import MarketDataProcessor

logger = logging.getLogger(__name__)

class RealTimeDataManager:
    """
    Manages real-time data from multiple sources including NinjaTrader 8
    Provides unified interface for the ML trading system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.ninjatrader = NinjaTraderDataProvider()
        self.data_processor = MarketDataProcessor()
        
        # Data storage
        self.current_prices = {}
        self.market_data_cache = {}
        self.bar_cache = {}
        
        # Subscriptions
        self.data_subscribers = []
        self.is_running = False
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.data_lock = threading.Lock()
        
    async def start(self):
        """Start the real-time data manager"""
        try:
            # Connect to NinjaTrader
            if not self.ninjatrader.connect():
                raise ConnectionError("Failed to connect to NinjaTrader 8")
            
            # Subscribe to instruments
            instruments = self.config.get('instruments', [])
            timeframes = self.config.get('timeframes', ['1 Minute', '5 Minute'])
            
            for instrument in instruments:
                self.ninjatrader.add_instrument(instrument, timeframes)
                logger.info(f"Subscribed to {instrument}")
            
            self.is_running = True
            logger.info("Real-time data manager started successfully")
            
            # Start data processing loop
            await self._data_processing_loop()
            
        except Exception as e:
            logger.error(f"Failed to start data manager: {e}")
            raise
    
    def stop(self):
        """Stop the data manager"""
        self.is_running = False
        self.ninjatrader.disconnect()
        self.executor.shutdown(wait=True)
        logger.info("Real-time data manager stopped")
    
    def subscribe_to_data(self, callback: Callable):
        """Subscribe to processed data updates"""
        self.data_subscribers.append(callback)
    
    def get_current_price(self, instrument: str) -> Optional[float]:
        """Get current price for instrument"""
        with self.data_lock:
            return self.current_prices.get(instrument)
    
    def get_latest_bars(self, instrument: str, timeframe: str = "1 Minute", 
                       periods: int = 100) -> pd.DataFrame:
        """Get latest bar data"""
        return self.ninjatrader.get_bars_dataframe(instrument, timeframe, periods)
    
    def get_market_data_features(self, instrument: str) -> Dict:
        """Get processed market data features"""
        with self.data_lock:
            return self.market_data_cache.get(instrument, {})
    
    async def _data_processing_loop(self):
        """Main data processing loop"""
        while self.is_running:
            try:
                # Process data for each instrument
                instruments = self.config.get('instruments', [])
                
                for instrument in instruments:
                    await self._process_instrument_data(instrument)
                
                # Notify subscribers
                await self._notify_subscribers()
                
                # Sleep briefly to prevent overwhelming the system
                await asyncio.sleep(0.1)  # 100ms update frequency
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_instrument_data(self, instrument: str):
        """Process data for a specific instrument"""
        try:
            # Get current price
            current_price = self.ninjatrader.get_current_price(instrument)
            if current_price:
                with self.data_lock:
                    self.current_prices[instrument] = current_price
            
            # Get recent bars for feature calculation
            bars_1m = self.ninjatrader.get_bars_dataframe(instrument, "1 Minute", 200)
            bars_5m = self.ninjatrader.get_bars_dataframe(instrument, "5 Minute", 100)
            
            if not bars_1m.empty:
                # Calculate features using the data processor
                features = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.data_processor.calculate_features,
                    bars_1m, bars_5m
                )
                
                with self.data_lock:
                    self.market_data_cache[instrument] = {
                        'timestamp': datetime.now(),
                        'current_price': current_price,
                        'features': features,
                        'bars_1m_count': len(bars_1m),
                        'bars_5m_count': len(bars_5m)
                    }
                
        except Exception as e:
            logger.error(f"Error processing data for {instrument}: {e}")
    
    async def _notify_subscribers(self):
        """Notify all data subscribers"""
        if not self.data_subscribers:
            return
        
        with self.data_lock:
            data_snapshot = {
                'timestamp': datetime.now(),
                'prices': self.current_prices.copy(),
                'market_data': self.market_data_cache.copy()
            }
        
        for callback in self.data_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_snapshot)
                else:
                    callback(data_snapshot)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")


class DataValidator:
    """
    Validates incoming market data for quality and consistency
    """
    
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        
    def validate_market_data(self, data: MarketData) -> bool:
        """Validate market data point"""
        try:
            # Basic sanity checks
            if data.bid <= 0 or data.ask <= 0 or data.last <= 0:
                return False
            
            if data.bid >= data.ask:
                return False
            
            if data.volume < 0:
                return False
            
            # Check for extreme price movements (circuit breaker)
            if data.instrument in self.price_history:
                last_price = self.price_history[data.instrument][-1] if self.price_history[data.instrument] else None
                
                if last_price and abs(data.last - last_price) / last_price > 0.1:  # 10% move
                    logger.warning(f"Large price movement detected for {data.instrument}: {last_price} -> {data.last}")
                    # Don't reject, but log for investigation
            
            # Update history
            if data.instrument not in self.price_history:
                self.price_history[data.instrument] = []
                self.volume_history[data.instrument] = []
            
            self.price_history[data.instrument].append(data.last)
            self.volume_history[data.instrument].append(data.volume)
            
            # Keep only last 1000 points
            if len(self.price_history[data.instrument]) > 1000:
                self.price_history[data.instrument] = self.price_history[data.instrument][-1000:]
                self.volume_history[data.instrument] = self.volume_history[data.instrument][-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            return False
    
    def validate_bar_data(self, data: BarData) -> bool:
        """Validate bar data"""
        try:
            # OHLC consistency checks
            if not (data.low <= data.open <= data.high and 
                   data.low <= data.close <= data.high and
                   data.low <= data.high):
                return False
            
            if data.volume < 0:
                return False
            
            # All prices should be positive
            if any(price <= 0 for price in [data.open, data.high, data.low, data.close]):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating bar data: {e}")
            return False


# Configuration for different market types
FUTURES_CONFIG = {
    'instruments': ['ES', 'NQ', 'YM', 'RTY'],  # E-mini futures
    'timeframes': ['1 Minute', '5 Minute', '15 Minute'],
    'trading_hours': {
        'start': '09:30',
        'end': '16:00',
        'timezone': 'US/Eastern'
    }
}

FOREX_CONFIG = {
    'instruments': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
    'timeframes': ['1 Minute', '5 Minute', '15 Minute', '1 Hour'],
    'trading_hours': {
        'start': '17:00',  # Sunday
        'end': '17:00',    # Friday
        'timezone': 'US/Eastern'
    }
}

CRYPTO_CONFIG = {
    'instruments': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
    'timeframes': ['1 Minute', '5 Minute', '15 Minute', '1 Hour'],
    'trading_hours': {
        'start': '00:00',
        'end': '23:59',
        'timezone': 'UTC'
    }
}
