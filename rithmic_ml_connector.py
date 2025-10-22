"""
ML-Compatible Rithmic Data Connector
Integrates with existing Rithmic infrastructure for ML training
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional, Callable, Any

# Add project paths
sys.path.append('.')
sys.path.append('./data-pipeline')

try:
    # Try different import paths for the rithmic connector
    try:
        from data_pipeline.ingestion.rithmic_connector import RithmicConnector, RithmicTick
    except ImportError:
        # Alternative path
        sys.path.append('./data-pipeline/ingestion')
        from rithmic_connector import RithmicConnector, RithmicTick
    RITHMIC_IMPORT_SUCCESS = True
except ImportError:
    # Create mock classes if import fails
    RITHMIC_IMPORT_SUCCESS = False
    
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class RithmicTick:
        symbol: str
        timestamp: datetime
        price: float
        size: int
        bid: float
        ask: float
        bid_size: int
        ask_size: int
        volume: int
        is_bid: bool
        exchange: str = "CME"
    
    class RithmicConnector:
        async def connect(self):
            return False

logger = logging.getLogger(__name__)

class RithmicDataConnector:
    """ML-compatible wrapper for Rithmic data collection"""
    
    def __init__(self):
        self.is_connected = False
        self.collected_ticks = []
        
    async def connect(self) -> bool:
        """Try to connect to real Rithmic"""
        if RITHMIC_IMPORT_SUCCESS:
            try:
                self.rithmic_connector = RithmicConnector()
                return await self.rithmic_connector.connect()
            except Exception as e:
                logger.error(f"❌ Failed to initialize Rithmic connector: {e}")
                return False
        else:
            logger.warning("⚠️ Rithmic connector not available")
            return False
    
    async def subscribe_market_data(self, symbol: str, callback: Callable = None) -> bool:
        """Subscribe to real-time market data"""
        if not hasattr(self, 'rithmic_connector') or not self.rithmic_connector:
            logger.error("❌ Rithmic connector not initialized")
            return False
            
        # Internal callback to collect data
        def data_collector_callback(tick: RithmicTick):
            self.collected_ticks.append(tick)
            if callback:
                callback(tick)
        
        return await self.rithmic_connector.subscribe_market_data(symbol, data_collector_callback)
    
    def get_collected_data(self) -> List[RithmicTick]:
        """Get all collected tick data"""
        return self.collected_ticks.copy()
    
    def clear_data(self):
        """Clear collected data"""
        self.collected_ticks = []

# Create a synthetic data generator for testing when Rithmic isn't available
class MockRithmicDataConnector:
    """Mock connector for testing ML training pipeline"""
    
    def __init__(self):
        self.is_connected = False
        self.collected_ticks = []
        
    async def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        logger.info("📡 Mock Rithmic connector - using realistic ES simulation")
        return True
    
    async def subscribe_market_data(self, symbol: str, callback: Callable = None) -> bool:
        """Generate realistic ES futures data"""
        if not self.is_connected:
            return False
            
        logger.info(f"📊 Generating realistic {symbol} market data...")
        
        # Generate realistic ES futures data
        base_price = 4500.0  # ES typical range
        current_price = base_price
        timestamp = datetime.now()
        
        # Generate 300 realistic ticks (5 minutes at ~1 tick/second)
        for i in range(300):
            # Realistic price movement (ES moves in 0.25 increments)
            price_change = np.random.choice([-0.25, 0, 0.25], p=[0.3, 0.4, 0.3])
            current_price += price_change
            
            # Realistic volume
            volume = np.random.choice([1, 2, 3, 4, 5, 10], p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
            
            # Create realistic tick
            tick = RithmicTick(
                symbol=symbol,
                timestamp=timestamp + timedelta(seconds=i),
                price=current_price,
                size=volume,
                bid=current_price - 0.25,
                ask=current_price + 0.25,
                bid_size=5,
                ask_size=5,
                volume=volume,
                is_bid=np.random.choice([True, False])
            )
            
            self.collected_ticks.append(tick)
            
            if callback:
                callback(tick)
                
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        logger.info(f"✅ Generated {len(self.collected_ticks)} realistic {symbol} ticks")
        return True
    
    def get_collected_data(self) -> List[RithmicTick]:
        """Get collected mock data"""
        return self.collected_ticks.copy()
    
    def clear_data(self):
        """Clear collected data"""
        self.collected_ticks = []

# Smart connector that tries real Rithmic first, falls back to mock
class SmartRithmicConnector:
    """Intelligent connector that uses real Rithmic if available, mock otherwise"""
    
    def __init__(self):
        self.connector = None
        self.using_real_data = False
        
    async def initialize(self):
        """Initialize the best available connector"""
        # Try real Rithmic first
        try:
            real_connector = RithmicDataConnector()
            if await real_connector.connect():
                self.connector = real_connector
                self.using_real_data = True
                logger.info("✅ Using REAL Rithmic market data")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Real Rithmic unavailable: {e}")
        
        # Fallback to mock
        self.connector = MockRithmicDataConnector()
        await self.connector.connect()
        self.using_real_data = False
        logger.info("📡 Using realistic market simulation (Rithmic unavailable)")
        return True
    
    async def collect_data(self, symbol: str, duration_minutes: int = 5) -> pd.DataFrame:
        """Collect market data for ML training"""
        if not self.connector:
            await self.initialize()
        
        # Clear previous data
        self.connector.clear_data()
        
        # Start data collection
        logger.info(f"📊 Collecting {symbol} data for {duration_minutes} minutes...")
        
        await self.connector.subscribe_market_data(symbol)
        
        # Wait for collection (mock generates immediately, real waits)
        if not self.using_real_data:
            # Mock generates all data immediately
            pass
        else:
            # Real data - wait for collection period
            await asyncio.sleep(duration_minutes * 60)
        
        # Get collected data
        ticks = self.connector.get_collected_data()
        
        if not ticks:
            logger.error("❌ No data collected")
            return None
        
        # Convert to DataFrame
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'price': tick.price,
                'size': tick.size,
                'bid': tick.bid,
                'ask': tick.ask,
                'volume': tick.volume
            })
        
        df = pd.DataFrame(data)
        
        logger.info(f"✅ Collected {len(df)} ticks")
        logger.info(f"📈 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        logger.info(f"📊 Data source: {'REAL Rithmic R|API' if self.using_real_data else 'Realistic Simulation'}")
        
        return df