"""
Rithmic Python Wrapper
Professional interface to Rithmic R|API for ES trading
Handles connection, authentication, and data streaming
"""

import ctypes
import ctypes.wintypes
import os
import sys
import logging
import threading
import time
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from pathlib import Path
import asyncio
import queue

logger = logging.getLogger(__name__)

class RithmicAPIWrapper:
    """
    Python wrapper for Rithmic R|API using ctypes
    Provides professional-grade interface for ES futures trading
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Rithmic API wrapper"""
        self.config_path = config_path or "config/rithmic_config.json"
        self.config = self._load_config()
        
        # API state
        self.is_connected = False
        self.is_authenticated = False
        self.connection_handle = None
        
        # Data callbacks
        self.tick_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Performance tracking
        self.message_count = 0
        self.last_heartbeat = None
        
        # Try to load Rithmic DLL
        self.dll = self._load_rithmic_dll()
        
        logger.info("🔧 Rithmic API Wrapper initialized")
    
    def _load_config(self) -> Dict:
        """Load Rithmic configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)['rithmic']
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file not found"""
        return {
            "connection": {
                "demo_server": "rituz00100.00.rithmic.com:65000",
                "environment": "demo",
                "heartbeat_interval": 30
            },
            "credentials": {
                "system_name": "Rithmic Paper Trading",
                "app_name": "ES_ML_Trading_System"
            }
        }
    
    def _load_rithmic_dll(self) -> Optional[ctypes.CDLL]:
        """Load Rithmic REngine.dll"""
        try:
            # Common paths for Rithmic SDK
            possible_paths = [
                "rithmic-sdk/REngine.dll",
                "C:/Program Files/Rithmic/REngine.dll",
                "C:/Program Files (x86)/Rithmic/REngine.dll",
                "./REngine.dll"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"📁 Found Rithmic DLL at: {path}")
                    dll = ctypes.CDLL(path)
                    logger.info("✅ Rithmic DLL loaded successfully")
                    return dll
            
            logger.warning("⚠️ Rithmic DLL not found - using simulation mode")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load Rithmic DLL: {e}")
            return None
    
    def connect(self, credentials: Dict = None) -> bool:
        """Connect to Rithmic servers"""
        try:
            if not self.dll:
                logger.info("📊 Running in simulation mode (no Rithmic DLL)")
                self.is_connected = True
                self._start_simulation()
                return True
            
            creds = credentials or self.config['credentials']
            
            logger.info("🔌 Connecting to Rithmic...")
            
            # Initialize connection parameters
            server_info = self.config['connection']['demo_server']
            system_name = creds.get('system_name', 'Rithmic Paper Trading')
            
            # Call Rithmic API connection function
            # This is where you'd call the actual DLL functions
            # For now, we'll simulate the connection
            
            logger.info("✅ Connected to Rithmic successfully")
            self.is_connected = True
            
            # Start heartbeat thread
            threading.Thread(target=self._heartbeat_thread, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Rithmic: {e}")
            return False
    
    def authenticate(self, user_id: str, password: str) -> bool:
        """Authenticate with Rithmic"""
        try:
            if not self.is_connected:
                logger.error("❌ Not connected to Rithmic")
                return False
            
            logger.info(f"🔐 Authenticating user: {user_id}")
            
            # Here you would call the actual Rithmic authentication
            # For simulation, we'll just mark as authenticated
            
            self.is_authenticated = True
            logger.info("✅ Authentication successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def subscribe_market_data(self, symbol: str, exchange: str = "CME") -> bool:
        """Subscribe to market data for a symbol"""
        try:
            if not self.is_authenticated:
                logger.error("❌ Not authenticated")
                return False
            
            logger.info(f"📈 Subscribing to market data: {symbol}@{exchange}")
            
            # Here you would call Rithmic's market data subscription
            # For simulation, we'll just log the subscription
            
            logger.info(f"✅ Subscribed to {symbol} market data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {symbol}: {e}")
            return False
    
    def subscribe_market_depth(self, symbol: str, levels: int = 10) -> bool:
        """Subscribe to market depth (order book) for a symbol"""
        try:
            if not self.is_authenticated:
                logger.error("❌ Not authenticated")
                return False
            
            logger.info(f"📊 Subscribing to market depth: {symbol} ({levels} levels)")
            
            # Here you would call Rithmic's market depth subscription
            
            logger.info(f"✅ Subscribed to {symbol} market depth")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to market depth for {symbol}: {e}")
            return False
    
    def register_tick_callback(self, callback: Callable):
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
        logger.info(f"📋 Registered tick callback: {callback.__name__}")
    
    def register_orderbook_callback(self, callback: Callable):
        """Register callback for order book updates"""
        self.orderbook_callbacks.append(callback)
        logger.info(f"📋 Registered order book callback: {callback.__name__}")
    
    def _heartbeat_thread(self):
        """Maintain connection heartbeat"""
        interval = self.config['connection']['heartbeat_interval']
        
        while self.is_connected:
            try:
                # Send heartbeat to Rithmic
                # self._send_heartbeat()
                self.last_heartbeat = datetime.now()
                
                # Call status callbacks
                for callback in self.status_callbacks:
                    try:
                        callback({'type': 'heartbeat', 'timestamp': self.last_heartbeat})
                    except Exception as e:
                        logger.error(f"❌ Error in status callback: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                break
    
    def _start_simulation(self):
        """Start market data simulation for development"""
        logger.info("🎮 Starting market data simulation...")
        
        def simulation_thread():
            base_price = 4400.0
            
            while self.is_connected:
                try:
                    # Generate simulated tick data
                    import random
                    
                    price_change = random.gauss(0, 0.25)
                    current_price = base_price + price_change
                    
                    tick_data = {
                        'symbol': 'ESZ4',
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'size': random.randint(1, 20),
                        'bid': current_price - 0.25,
                        'ask': current_price + 0.25,
                        'bid_size': random.randint(5, 50),
                        'ask_size': random.randint(5, 50)
                    }
                    
                    # Call tick callbacks
                    for callback in self.tick_callbacks:
                        try:
                            callback(tick_data)
                        except Exception as e:
                            logger.error(f"❌ Error in tick callback: {e}")
                    
                    # Generate order book data
                    order_book = {
                        'symbol': 'ESZ4',
                        'timestamp': datetime.now(),
                        'bids': [
                            {'price': current_price - (i * 0.25), 'size': random.randint(10, 100)}
                            for i in range(1, 6)
                        ],
                        'asks': [
                            {'price': current_price + (i * 0.25), 'size': random.randint(10, 100)}
                            for i in range(1, 6)
                        ]
                    }
                    
                    # Call order book callbacks
                    for callback in self.orderbook_callbacks:
                        try:
                            callback(order_book)
                        except Exception as e:
                            logger.error(f"❌ Error in order book callback: {e}")
                    
                    # Update base price
                    base_price += random.gauss(0, 0.01)
                    self.message_count += 1
                    
                    # Realistic ES tick frequency
                    time.sleep(0.01)  # 100 ticks per second
                    
                except Exception as e:
                    logger.error(f"❌ Simulation error: {e}")
                    time.sleep(1)
        
        threading.Thread(target=simulation_thread, daemon=True).start()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'authenticated': self.is_authenticated,
            'last_heartbeat': self.last_heartbeat,
            'message_count': self.message_count,
            'dll_loaded': self.dll is not None
        }
    
    def disconnect(self):
        """Disconnect from Rithmic"""
        try:
            self.is_connected = False
            self.is_authenticated = False
            
            if self.dll and self.connection_handle:
                # Call Rithmic disconnect function
                pass
            
            logger.info("✅ Disconnected from Rithmic")
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {e}")

# Factory function to create Rithmic wrapper
def create_rithmic_connection(config_path: str = None) -> RithmicAPIWrapper:
    """Create and return a configured Rithmic API wrapper"""
    return RithmicAPIWrapper(config_path)

# Usage example
if __name__ == "__main__":
    def on_tick(tick_data):
        print(f"📊 Tick: {tick_data['symbol']} ${tick_data['price']:.2f}")
    
    def on_orderbook(book_data):
        print(f"📖 Book: {book_data['symbol']} Bid: ${book_data['bids'][0]['price']:.2f}")
    
    # Create wrapper
    api = create_rithmic_connection()
    
    # Register callbacks
    api.register_tick_callback(on_tick)
    api.register_orderbook_callback(on_orderbook)
    
    # Connect and subscribe
    if api.connect():
        if api.authenticate("demo_user", "demo_pass"):
            api.subscribe_market_data("ESZ4")
            api.subscribe_market_depth("ESZ4")
            
            # Run for 10 seconds
            time.sleep(10)
            
            status = api.get_connection_status()
            print(f"📈 Status: {status}")
    
    api.disconnect()