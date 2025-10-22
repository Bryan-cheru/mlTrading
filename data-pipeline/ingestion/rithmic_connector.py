"""
Professional Rithmic Data Connector with R | API Integration
Institutional-grade market data for ES futures trading using Rithmic R | API
"""

import asyncio
import os
import sys
import time
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rithmic R | API Configuration - Professional SDK Integration
RITHMIC_SDK_PATH = os.path.join(project_root, "13.6.0.0", "win10", "lib_472")
RITHMIC_AVAILABLE = os.path.exists(RITHMIC_SDK_PATH) and os.path.exists(os.path.join(RITHMIC_SDK_PATH, "rapiplus.dll"))

# Rithmic R | API initialization - Use Professional Connector
rithmic_api = None
if RITHMIC_AVAILABLE:
    try:
        from .rithmic_professional_connector import ProfessionalRithmicConnector
        rithmic_api = ProfessionalRithmicConnector()
        logger.info(f"✅ Rithmic R | API Professional SDK initialized: {RITHMIC_SDK_PATH}")
    except ImportError as e:
        logger.warning(f"⚠️ Rithmic Professional SDK not found - using simulation mode: {e}")
        RITHMIC_AVAILABLE = False
    except Exception as e:
        logger.error(f"❌ Failed to initialize Rithmic Professional SDK: {e}")
        RITHMIC_AVAILABLE = False

@dataclass
class RithmicTick:
    """Real-time market tick data from Rithmic R | API"""
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
    session: str = "regular"

@dataclass
class RithmicOrderBook:
    """Order book depth from Rithmic R | API"""
    symbol: str
    timestamp: datetime
    bids: List[tuple]  # (price, size) pairs
    asks: List[tuple]  # (price, size) pairs
    exchange: str = "CME"

class RithmicConnector:
    """
    Professional Rithmic data connector using R | API
    Provides institutional-grade tick-by-tick data and order book depth
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'server': 'rithmic_paper_trading',
            'username': os.getenv('RITHMIC_USERNAME', ''),
            'password': os.getenv('RITHMIC_PASSWORD', ''),
            'system_name': 'paper_trading',
            'instruments': ['ESZ4', 'ESH5'],
            'market_data_exchange': 'CME',
            'heartbeat_interval': 30
        }
        
        # Connection state
        self.is_connected = False
        self.is_subscribed = False
        self.rithmic_client = rithmic_api
        
        # Data storage
        self.tick_data: Dict[str, List[RithmicTick]] = {}
        self.order_books: Dict[str, RithmicOrderBook] = {}
        self.market_data_queue = queue.Queue(maxsize=10000)
        
        # Callbacks for real-time processing
        self.tick_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        
        # Performance tracking
        self.tick_count = 0
        self.last_tick_time = None
        
        logger.info("🔌 Rithmic R | API Connector initialized")
    
    async def connect(self) -> bool:
        """Connect to Rithmic using R | API"""
        try:
            if not RITHMIC_AVAILABLE or not self.rithmic_client:
                logger.error("❌ Rithmic R | API not available")
                return await self._simulate_connection()
            
            logger.info("🔌 Connecting to Rithmic R | API...")
            
            # Connect to R | API gateway
            success = await self.rithmic_client.connect(
                username=self.config['username'],
                password=self.config['password'],
                system_name=self.config['system_name']
            )
            
            if success:
                self.is_connected = True
                logger.info("✅ Connected to Rithmic R | API successfully")
                return True
            else:
                logger.error("❌ Failed to connect to Rithmic R | API")
                return await self._simulate_connection()
                
        except Exception as e:
            logger.error(f"❌ Rithmic R | API connection error: {e}")
            return await self._simulate_connection()
    
    async def subscribe_market_data(self, symbol: str, callback: Callable = None) -> bool:
        """Subscribe to real-time market data for a symbol"""
        try:
            if not self.is_connected:
                logger.error("❌ Not connected to Rithmic R | API")
                return False
            
            if callback:
                self.tick_callbacks.append(callback)
            
            if RITHMIC_AVAILABLE and self.rithmic_client:
                # Subscribe using real R | API
                success = await self.rithmic_client.subscribe_market_data(
                    symbol=symbol,
                    exchange=self.config['market_data_exchange']
                )
                
                if success:
                    # Register callback for this symbol
                    self.rithmic_client.register_callback(symbol, self._on_rithmic_tick)
                    logger.info(f"✅ Subscribed to {symbol} via R | API")
                    return True
                else:
                    logger.error(f"❌ R | API subscription failed for {symbol}")
                    return False
            else:
                # Fallback to simulation
                return await self._simulate_subscription(symbol, callback)
                
        except Exception as e:
            logger.error(f"❌ Error subscribing to {symbol}: {e}")
            return False
    
    def _on_rithmic_tick(self, tick_data: Dict):
        """Handle tick data from Rithmic R | API"""
        try:
            # Convert R | API tick to RithmicTick
            tick = RithmicTick(
                symbol=tick_data['symbol'],
                timestamp=tick_data['timestamp'],
                price=tick_data['price'],
                size=tick_data['size'],
                bid=tick_data.get('bid', 0.0),
                ask=tick_data.get('ask', 0.0),
                bid_size=tick_data.get('bid_size', 0),
                ask_size=tick_data.get('ask_size', 0),
                volume=tick_data['volume'],
                is_bid=tick_data.get('is_bid', False)
            )
            
            # Process tick data
            self._process_tick_data(tick)
            
        except Exception as e:
            logger.error(f"❌ Error processing R | API tick: {e}")
    
    def _process_tick_data(self, tick: RithmicTick):
        """Process incoming tick data from Rithmic R | API"""
        try:
            # Store tick data
            if tick.symbol not in self.tick_data:
                self.tick_data[tick.symbol] = []
            
            self.tick_data[tick.symbol].append(tick)
            
            # Keep only recent ticks (last 1000)
            if len(self.tick_data[tick.symbol]) > 1000:
                self.tick_data[tick.symbol] = self.tick_data[tick.symbol][-1000:]
            
            # Update performance metrics
            self.tick_count += 1
            self.last_tick_time = tick.timestamp
            
            # Call registered callbacks
            tick_dict = {
                'symbol': tick.symbol,
                'timestamp': tick.timestamp,
                'price': tick.price,
                'size': tick.size,
                'volume': tick.volume,
                'bid': tick.bid,
                'ask': tick.ask
            }
            
            for callback in self.tick_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(tick_dict))
                    else:
                        callback(tick_dict)
                except Exception as e:
                    logger.error(f"❌ Error in tick callback: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")
    
    async def _simulate_connection(self) -> bool:
        """Simulate Rithmic connection for development/testing"""
        logger.info("🔄 Using simulated Rithmic connection for development")
        
        await asyncio.sleep(1)
        self.is_connected = True
        logger.info("✅ Simulated Rithmic connection established")
        return True
    
    async def _simulate_subscription(self, symbol: str, callback: Callable = None) -> bool:
        """Simulate market data subscription"""
        logger.info(f"🔄 Simulating market data subscription for {symbol}")
        
        if callback:
            self.tick_callbacks.append(callback)
        
        # Start simulated data generator
        asyncio.create_task(self._generate_simulated_data(symbol))
        return True
    
    async def _generate_simulated_data(self, symbol: str):
        """Generate simulated tick data for testing"""
        base_price = 5000.0 if 'ES' in symbol else 17000.0
        
        while self.is_connected:
            try:
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.25)
                base_price += price_change
                
                # Generate simulated tick
                tick = RithmicTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=round(base_price, 2),
                    size=np.random.randint(1, 10),
                    bid=round(base_price - 0.25, 2),
                    ask=round(base_price + 0.25, 2),
                    bid_size=np.random.randint(5, 50),
                    ask_size=np.random.randint(5, 50),
                    volume=np.random.randint(1, 10),
                    is_bid=np.random.choice([True, False])
                )
                
                # Process simulated tick
                self._process_tick_data(tick)
                
                # Random delay between ticks (50-200ms)
                await asyncio.sleep(np.random.uniform(0.05, 0.2))
                
            except Exception as e:
                logger.error(f"❌ Error generating simulated data: {e}")
                await asyncio.sleep(1)
    
    def get_order_book(self, symbol: str) -> Optional[RithmicOrderBook]:
        """Get current order book for symbol"""
        return self.order_books.get(symbol)
    
    def get_latest_tick(self, symbol: str) -> Optional[RithmicTick]:
        """Get latest tick for symbol"""
        if symbol in self.tick_data and self.tick_data[symbol]:
            return self.tick_data[symbol][-1]
        return None
    
    def get_performance_stats(self) -> Dict:
        """Get connector performance statistics"""
        return {
            'connected': self.is_connected,
            'tick_count': self.tick_count,
            'last_tick_time': self.last_tick_time,
            'subscribed_symbols': list(self.tick_data.keys()),
            'using_real_api': RITHMIC_AVAILABLE,
            'api_client': type(self.rithmic_client).__name__ if self.rithmic_client else None
        }
    
    async def disconnect(self):
        """Disconnect from Rithmic R | API and cleanup resources"""
        try:
            if self.is_connected:
                logger.info("🔌 Disconnecting from Rithmic R | API...")
                
                if RITHMIC_AVAILABLE and self.rithmic_client:
                    await self.rithmic_client.disconnect()
                
                self.is_connected = False
                self.is_subscribed = False
                
                # Clear callbacks
                self.tick_callbacks.clear()
                self.orderbook_callbacks.clear()
                
                logger.info("✅ Disconnected from Rithmic R | API successfully")
                
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}")
    
    def get_api_info(self) -> Dict:
        """Get information about the Rithmic R | API integration"""
        return {
            'api_available': RITHMIC_AVAILABLE,
            'sdk_path': RITHMIC_SDK_PATH,
            'sdk_exists': os.path.exists(RITHMIC_SDK_PATH),
            'connection_status': self.is_connected,
            'subscription_status': self.is_subscribed,
            'client_type': type(self.rithmic_client).__name__ if self.rithmic_client else None,
            'config': {k: v for k, v in self.config.items() if k not in ['username', 'password']}
        }
    
    async def health_check(self) -> Dict:
        """Perform comprehensive health check of Rithmic R | API connection"""
        health_status = {
            'overall_status': 'healthy',
            'connection': self.is_connected,
            'subscription': self.is_subscribed,
            'tick_rate': 0,
            'data_quality': 'good',
            'api_integration': RITHMIC_AVAILABLE,
            'errors': []
        }
        
        try:
            # Check tick rate (ticks per second)
            if self.tick_count > 0 and self.last_tick_time:
                time_diff = (datetime.now() - self.last_tick_time).total_seconds()
                if time_diff > 0:
                    health_status['tick_rate'] = round(self.tick_count / time_diff, 2)
            
            # Check data quality
            recent_ticks = 0
            for symbol_ticks in self.tick_data.values():
                recent_ticks += len([t for t in symbol_ticks 
                                   if (datetime.now() - t.timestamp).total_seconds() < 60])
            
            if recent_ticks < 10:
                health_status['data_quality'] = 'poor'
                health_status['errors'].append('Low recent tick count')
            
            # Check connection status
            if not self.is_connected:
                health_status['overall_status'] = 'disconnected'
                health_status['errors'].append('Not connected to Rithmic R | API')
            
            # Check API availability
            if not RITHMIC_AVAILABLE:
                health_status['errors'].append('Real R | API not available - using simulation')
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['errors'].append(f'Health check error: {str(e)}')
        
        return health_status

# Professional usage example
if __name__ == "__main__":
    async def main():
        # Initialize Rithmic connector with R | API
        connector = RithmicConnector({
            'instruments': ['ESZ4'],
            'username': os.getenv('RITHMIC_USERNAME'),
            'password': os.getenv('RITHMIC_PASSWORD'),
            'system_name': 'paper_trading'
        })
        
        # Register callback for tick processing
        def process_tick(tick_data: Dict):
            print(f"📊 {tick_data['symbol']}: ${tick_data['price']:.2f} @ {tick_data['timestamp']}")
        
        # Connect and subscribe
        await connector.connect()
        
        # Check API information
        api_info = connector.get_api_info()
        print(f"🔧 R | API Info: {api_info}")
        
        # Subscribe to market data
        await connector.subscribe_market_data('ESZ4', process_tick)
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Health check
        health = await connector.health_check()
        print(f"💊 Health Status: {health}")
        
        # Performance stats
        stats = connector.get_performance_stats()
        print(f"📈 Performance: {stats}")
        
        await connector.disconnect()
    
    asyncio.run(main())