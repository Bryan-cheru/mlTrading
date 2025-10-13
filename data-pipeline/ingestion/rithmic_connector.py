"""
Rithmic Data Connector for ES Trading System
Professional-grade market data integration for institutional trading
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import json
import queue
import numpy as np
import pandas as pd

# Rithmic API imports (you'll need to install Rithmic SDK)
try:
    # Placeholder - replace with actual Rithmic imports
    # from rithmic import RithmicAPI, MarketDataRequest, OrderBookRequest
    pass
except ImportError:
    logging.warning("Rithmic SDK not installed - using simulation mode")

logger = logging.getLogger(__name__)

@dataclass
class RithmicTick:
    """Individual tick data from Rithmic"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    volume: int
    is_bid: bool  # True if trade at bid, False if at ask

@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: int
    orders: int

@dataclass
class RithmicOrderBook:
    """Complete order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
class RithmicConnector:
    """
    Professional Rithmic data connector for ES trading
    Provides tick-by-tick data, order book, and market microstructure
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'server': 'rithmic_paper_trading',  # or production server
            'username': '',  # Set from environment
            'password': '',  # Set from environment
            'instruments': ['ESZ4', 'ESH5'],  # ES futures contracts
            'market_data_exchange': 'CME',
            'heartbeat_interval': 30
        }
        
        # Connection state
        self.is_connected = False
        self.is_subscribed = False
        
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
        self.latency_samples = []
        
        logger.info("🎯 Rithmic Connector initialized for ES trading")
    
    async def connect(self) -> bool:
        """Connect to Rithmic servers"""
        try:
            logger.info("🔌 Connecting to Rithmic...")
            
            # Initialize Rithmic API (placeholder)
            # self.api = RithmicAPI()
            # await self.api.connect(
            #     server=self.config['server'],
            #     username=self.config['username'],
            #     password=self.config['password']
            # )
            
            # Simulation mode for development
            logger.info("📊 Running in simulation mode - implement actual Rithmic API")
            self.is_connected = True
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            logger.info("✅ Connected to Rithmic successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Rithmic: {e}")
            return False
    
    async def subscribe_market_data(self, symbols: List[str] = None) -> bool:
        """Subscribe to real-time market data for ES futures"""
        try:
            if not self.is_connected:
                logger.error("❌ Not connected to Rithmic")
                return False
            
            symbols = symbols or self.config['instruments']
            
            for symbol in symbols:
                logger.info(f"📈 Subscribing to market data: {symbol}")
                
                # Subscribe to tick data
                # await self.api.subscribe_market_data(
                #     symbol=symbol,
                #     exchange=self.config['market_data_exchange'],
                #     callback=self._on_tick_data
                # )
                
                # Subscribe to order book
                # await self.api.subscribe_market_depth(
                #     symbol=symbol,
                #     exchange=self.config['market_data_exchange'],
                #     levels=10,  # 10 levels of market depth
                #     callback=self._on_order_book_update
                # )
                
                # Initialize data storage
                self.tick_data[symbol] = []
                
            self.is_subscribed = True
            logger.info(f"✅ Subscribed to {len(symbols)} instruments")
            
            # Start simulation data for development
            asyncio.create_task(self._simulate_market_data())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to market data: {e}")
            return False
    
    def _on_tick_data(self, tick_data: Dict):
        """Handle incoming tick data from Rithmic"""
        try:
            # Convert Rithmic tick data to our format
            tick = RithmicTick(
                symbol=tick_data['symbol'],
                timestamp=datetime.fromtimestamp(tick_data['timestamp']),
                price=tick_data['last_price'],
                size=tick_data['last_size'],
                bid=tick_data['bid'],
                ask=tick_data['ask'],
                bid_size=tick_data['bid_size'],
                ask_size=tick_data['ask_size'],
                volume=tick_data['volume'],
                is_bid=tick_data['last_price'] <= tick_data['bid']
            )
            
            # Store tick data
            symbol = tick.symbol
            if symbol not in self.tick_data:
                self.tick_data[symbol] = []
            
            self.tick_data[symbol].append(tick)
            
            # Keep only last 10000 ticks per symbol
            if len(self.tick_data[symbol]) > 10000:
                self.tick_data[symbol] = self.tick_data[symbol][-10000:]
            
            # Calculate latency
            current_time = datetime.now()
            if self.last_tick_time:
                latency = (current_time - tick.timestamp).total_seconds() * 1000
                self.latency_samples.append(latency)
                
                # Keep only last 1000 latency samples
                if len(self.latency_samples) > 1000:
                    self.latency_samples = self.latency_samples[-1000:]
            
            self.last_tick_time = current_time
            self.tick_count += 1
            
            # Queue for processing
            try:
                self.market_data_queue.put_nowait(tick)
            except queue.Full:
                logger.warning("⚠️ Market data queue full, dropping tick")
            
            # Call registered callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"❌ Error in tick callback: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")
    
    def _on_order_book_update(self, book_data: Dict):
        """Handle order book updates from Rithmic"""
        try:
            symbol = book_data['symbol']
            
            # Convert to our order book format
            bids = [
                OrderBookLevel(price=level['price'], size=level['size'], orders=level.get('orders', 0))
                for level in book_data['bids']
            ]
            
            asks = [
                OrderBookLevel(price=level['price'], size=level['size'], orders=level.get('orders', 0))
                for level in book_data['asks']
            ]
            
            order_book = RithmicOrderBook(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(book_data['timestamp']),
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price)
            )
            
            self.order_books[symbol] = order_book
            
            # Call registered callbacks
            for callback in self.orderbook_callbacks:
                try:
                    callback(order_book)
                except Exception as e:
                    logger.error(f"❌ Error in order book callback: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing order book update: {e}")
    
    def get_latest_tick(self, symbol: str) -> Optional[RithmicTick]:
        """Get the latest tick for a symbol"""
        if symbol in self.tick_data and self.tick_data[symbol]:
            return self.tick_data[symbol][-1]
        return None
    
    def get_latest_order_book(self, symbol: str) -> Optional[RithmicOrderBook]:
        """Get the latest order book for a symbol"""
        return self.order_books.get(symbol)
    
    def get_market_features(self, symbol: str, lookback_ticks: int = 100) -> Dict[str, float]:
        """
        Extract market microstructure features from recent tick data
        This is where the magic happens for your ML model!
        """
        try:
            if symbol not in self.tick_data or len(self.tick_data[symbol]) < lookback_ticks:
                return {}
            
            recent_ticks = self.tick_data[symbol][-lookback_ticks:]
            order_book = self.order_books.get(symbol)
            
            features = {}
            
            # Basic price features
            prices = [tick.price for tick in recent_ticks]
            features['current_price'] = prices[-1]
            features['price_change'] = prices[-1] - prices[0]
            features['price_volatility'] = np.std(prices)
            
            # Order flow features
            buy_volume = sum(tick.size for tick in recent_ticks if not tick.is_bid)
            sell_volume = sum(tick.size for tick in recent_ticks if tick.is_bid)
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                features['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
                features['buy_ratio'] = buy_volume / total_volume
            else:
                features['order_flow_imbalance'] = 0.0
                features['buy_ratio'] = 0.5
            
            # Bid-ask spread features
            if recent_ticks:
                latest_tick = recent_ticks[-1]
                features['bid_ask_spread'] = latest_tick.ask - latest_tick.bid
                features['bid_ask_midpoint'] = (latest_tick.bid + latest_tick.ask) / 2
                features['price_vs_midpoint'] = latest_tick.price - features['bid_ask_midpoint']
            
            # Order book features (if available)
            if order_book and order_book.bids and order_book.asks:
                top_bid = order_book.bids[0]
                top_ask = order_book.asks[0]
                
                features['top_bid_size'] = top_bid.size
                features['top_ask_size'] = top_ask.size
                features['book_imbalance'] = (top_bid.size - top_ask.size) / (top_bid.size + top_ask.size)
                
                # Depth analysis
                bid_depth = sum(level.size for level in order_book.bids[:5])
                ask_depth = sum(level.size for level in order_book.asks[:5])
                features['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            # Volume profile features
            features['total_volume'] = total_volume
            features['avg_trade_size'] = total_volume / len(recent_ticks) if recent_ticks else 0
            features['large_trade_ratio'] = sum(1 for tick in recent_ticks if tick.size > 10) / len(recent_ticks)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting market features: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get connector performance statistics"""
        avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0
        return {
            'total_ticks': self.tick_count,
            'avg_latency_ms': avg_latency,
            'is_connected': self.is_connected,
            'is_subscribed': self.is_subscribed,
            'queue_size': self.market_data_queue.qsize()
        }
    
    def register_tick_callback(self, callback: Callable):
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
    
    def register_orderbook_callback(self, callback: Callable):
        """Register callback for order book updates"""
        self.orderbook_callbacks.append(callback)
    
    async def _heartbeat_loop(self):
        """Maintain connection heartbeat"""
        while self.is_connected:
            try:
                # Send heartbeat to Rithmic
                # await self.api.send_heartbeat()
                await asyncio.sleep(self.config['heartbeat_interval'])
            except Exception as e:
                logger.error(f"❌ Heartbeat failed: {e}")
                break
    
    async def _simulate_market_data(self):
        """Simulate market data for development/testing"""
        logger.info("📊 Starting market data simulation...")
        
        base_price = 4400.0  # ES base price
        
        while self.is_subscribed:
            try:
                # Generate realistic tick data
                price_change = np.random.normal(0, 0.25)  # 0.25 point std
                current_price = base_price + price_change
                
                # Create simulated tick
                tick_data = {
                    'symbol': 'ESZ4',
                    'timestamp': time.time(),
                    'last_price': current_price,
                    'last_size': np.random.randint(1, 20),
                    'bid': current_price - 0.25,
                    'ask': current_price + 0.25,
                    'bid_size': np.random.randint(5, 50),
                    'ask_size': np.random.randint(5, 50),
                    'volume': np.random.randint(1, 20)
                }
                
                self._on_tick_data(tick_data)
                
                # Update base price slowly
                base_price += np.random.normal(0, 0.01)
                
                # Realistic tick frequency (100-1000 ticks per second for ES)
                await asyncio.sleep(0.01)  # 100ms between ticks for simulation
                
            except Exception as e:
                logger.error(f"❌ Error in simulation: {e}")
                await asyncio.sleep(1)
    
    async def disconnect(self):
        """Disconnect from Rithmic"""
        try:
            self.is_connected = False
            self.is_subscribed = False
            
            # if hasattr(self, 'api'):
            #     await self.api.disconnect()
            
            logger.info("✅ Disconnected from Rithmic")
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {e}")

# Usage example for your ES trading system
if __name__ == "__main__":
    async def main():
        # Initialize Rithmic connector
        connector = RithmicConnector({
            'instruments': ['ESZ4'],  # December 2024 ES futures
            'server': 'rithmic_paper_trading'
        })
        
        # Register callback for tick processing
        def process_tick(tick: RithmicTick):
            features = connector.get_market_features(tick.symbol)
            print(f"📊 {tick.symbol}: ${tick.price:.2f}, Features: {len(features)}")
        
        connector.register_tick_callback(process_tick)
        
        # Connect and subscribe
        await connector.connect()
        await connector.subscribe_market_data()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Show performance stats
        stats = connector.get_performance_stats()
        print(f"📈 Performance: {stats}")
        
        await connector.disconnect()
    
    asyncio.run(main())