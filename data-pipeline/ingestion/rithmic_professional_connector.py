"""
Professional Rithmic R|API Integration for Institutional Trading
Uses the official Rithmic SDK 13.6.0.0 with proper callbacks and connection management
"""

import os
import sys
import time
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue
import clr

# Add Rithmic SDK path
RITHMIC_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "13.6.0.0", "win10", "lib_472")
sys.path.append(RITHMIC_SDK_PATH)

# Add reference to Rithmic DLL
try:
    clr.AddReference(os.path.join(RITHMIC_SDK_PATH, "rapiplus.dll"))
    from com.omnesys.rapi import *
    from com.omnesys.omne.om import *
    RITHMIC_AVAILABLE = True
    logging.info("✅ Rithmic R|API Professional SDK loaded successfully")
except Exception as e:
    RITHMIC_AVAILABLE = False
    logging.error(f"❌ Failed to load Rithmic SDK: {e}")

logger = logging.getLogger(__name__)

@dataclass
class RithmicMarketData:
    """Professional market data from Rithmic R|API"""
    symbol: str
    timestamp: datetime
    last_price: float
    last_size: int
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    volume: int
    open_price: float
    high_price: float
    low_price: float
    settlement_price: float
    
@dataclass
class RithmicOrderBookLevel:
    """Order book level data"""
    price: float
    size: int
    num_orders: int

@dataclass
class RithmicOrderBook:
    """Complete order book data"""
    symbol: str
    timestamp: datetime
    bids: List[RithmicOrderBookLevel]
    asks: List[RithmicOrderBookLevel]

class RithmicCallbacks(RCallbacks):
    """
    Professional Rithmic callback implementation
    Following official SDK patterns from SampleMD.cs
    """
    
    def __init__(self, connector):
        super().__init__()
        self.connector = connector
        self.login_status = "NotLoggedIn"
        self.logged_into_md = False
        self.got_price_incr_info = False
        self.received_agreement_list = False
        
    def Alert(self, alert_info):
        """Handle system alerts"""
        try:
            message = str(alert_info)
            logger.info(f"🔔 Rithmic Alert: {message}")
            
            # Handle critical alerts
            if "login" in message.lower():
                self.connector._handle_login_alert(message)
            elif "disconnect" in message.lower():
                self.connector._handle_disconnect_alert(message)
                
        except Exception as e:
            logger.error(f"❌ Error in Alert callback: {e}")
    
    def AgreementList(self, agreement_list):
        """Handle agreement list"""
        try:
            self.received_agreement_list = True
            logger.info("📋 Received Rithmic agreement list")
            
            # Auto-accept agreements (configure as needed)
            if hasattr(self.connector, 'auto_accept_agreements') and self.connector.auto_accept_agreements:
                # Implementation for auto-accepting agreements
                pass
                
        except Exception as e:
            logger.error(f"❌ Error in AgreementList callback: {e}")
    
    def LoginComplete(self, login_complete):
        """Handle login completion"""
        try:
            if login_complete and login_complete.IsSuccess():
                self.login_status = "LoggedIn"
                logger.info("✅ Rithmic login completed successfully")
                self.connector._on_login_success()
            else:
                self.login_status = "LoginFailed"
                error_msg = str(login_complete) if login_complete else "Unknown error"
                logger.error(f"❌ Rithmic login failed: {error_msg}")
                self.connector._on_login_failure(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Error in LoginComplete callback: {e}")
    
    def MdLoginComplete(self, md_login_complete):
        """Handle market data login completion"""
        try:
            if md_login_complete and md_login_complete.IsSuccess():
                self.logged_into_md = True
                logger.info("✅ Rithmic market data login completed")
                self.connector._on_md_login_success()
            else:
                error_msg = str(md_login_complete) if md_login_complete else "Unknown error"
                logger.error(f"❌ Rithmic market data login failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Error in MdLoginComplete callback: {e}")
    
    def LastTrade(self, last_trade):
        """Handle last trade updates"""
        try:
            if last_trade:
                market_data = RithmicMarketData(
                    symbol=str(last_trade.Exchange) + " " + str(last_trade.Ticker),
                    timestamp=datetime.now(),
                    last_price=float(last_trade.Price) if last_trade.Price else 0.0,
                    last_size=int(last_trade.Size) if last_trade.Size else 0,
                    bid_price=0.0,  # Will be updated by BidQuote
                    ask_price=0.0,  # Will be updated by AskQuote
                    bid_size=0,
                    ask_size=0,
                    volume=int(last_trade.TotalVolume) if last_trade.TotalVolume else 0,
                    open_price=float(last_trade.OpeningPrice) if last_trade.OpeningPrice else 0.0,
                    high_price=float(last_trade.HighPrice) if last_trade.HighPrice else 0.0,
                    low_price=float(last_trade.LowPrice) if last_trade.LowPrice else 0.0,
                    settlement_price=float(last_trade.SettlementPrice) if last_trade.SettlementPrice else 0.0
                )
                
                # Send to connector's data handlers
                self.connector._process_market_data(market_data)
                
        except Exception as e:
            logger.error(f"❌ Error in LastTrade callback: {e}")
    
    def BidQuote(self, bid_quote):
        """Handle bid quote updates"""
        try:
            if bid_quote:
                symbol = str(bid_quote.Exchange) + " " + str(bid_quote.Ticker)
                self.connector._update_bid_quote(symbol, float(bid_quote.Price), int(bid_quote.Size))
                
        except Exception as e:
            logger.error(f"❌ Error in BidQuote callback: {e}")
    
    def AskQuote(self, ask_quote):
        """Handle ask quote updates"""
        try:
            if ask_quote:
                symbol = str(ask_quote.Exchange) + " " + str(ask_quote.Ticker)
                self.connector._update_ask_quote(symbol, float(ask_quote.Price), int(ask_quote.Size))
                
        except Exception as e:
            logger.error(f"❌ Error in AskQuote callback: {e}")

class ProfessionalRithmicConnector:
    """
    Professional Rithmic R|API Connector for Institutional Trading
    Implements proper connection management and data streaming
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.engine = None
        self.callbacks = None
        self.is_connected = False
        self.is_md_logged_in = False
        
        # Data management
        self.market_data_cache = {}
        self.order_book_cache = {}
        self.data_callbacks = []
        self.tick_handlers = []
        
        # Connection settings
        self.user_id = ""
        self.password = ""
        self.system_name = ""
        self.auto_accept_agreements = True
        
        # Performance tracking
        self.message_count = 0
        self.last_heartbeat = datetime.now()
        
        logger.info("🏛️ Professional Rithmic Connector initialized")
    
    def _load_config(self, config_file: str = None) -> Dict:
        """Load Rithmic configuration"""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "..", "..", "config", "rithmic_config.json")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get("rithmic", {})
        except Exception as e:
            logger.warning(f"⚠️ Could not load config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "connection": {
                "demo_server": "rituz00100.00.rithmic.com:65000~rituz00100.00.rithmic.com:65001",
                "live_server": "rituz00100.00.rithmic.com:65000~rituz00100.00.rithmic.com:65001",
                "heartbeat_interval": 30,
                "connection_timeout": 10,
                "environment": "demo"
            },
            "credentials": {
                "system_name": "Rithmic Paper Trading",
                "app_name": "ES_ML_Trading_System",
                "app_version": "1.0.0"
            }
        }
    
    async def connect(self, user_id: str, password: str, system_name: str = None) -> bool:
        """Connect to Rithmic R|API Professional"""
        if not RITHMIC_AVAILABLE:
            logger.error("❌ Rithmic SDK not available")
            return False
        
        try:
            self.user_id = user_id
            self.password = password
            self.system_name = system_name or self.config.get("credentials", {}).get("system_name", "Rithmic Paper Trading")
            
            logger.info("🔌 Connecting to Rithmic R|API Professional...")
            
            # Initialize REngine
            self.engine = REngine()
            
            # Set up callbacks
            self.callbacks = RithmicCallbacks(self)
            self.engine.setCallbacks(self.callbacks)
            
            # Get server connection string
            server_config = self.config.get("connection", {})
            environment = server_config.get("environment", "demo")
            server_string = server_config.get(f"{environment}_server", server_config.get("demo_server"))
            
            # Login to repository
            login_params = LoginParams()
            login_params.User = self.user_id
            login_params.Password = self.password
            login_params.System = self.system_name
            login_params.AppName = self.config.get("credentials", {}).get("app_name", "ES_ML_Trading_System")
            login_params.AppVersion = self.config.get("credentials", {}).get("app_version", "1.0.0")
            
            # Connect
            result = self.engine.login(login_params, server_string)
            
            if result == 0:  # Success
                logger.info("✅ Rithmic repository login initiated")
                
                # Wait for login completion
                timeout = 30  # 30 seconds timeout
                start_time = time.time()
                
                while (self.callbacks.login_status != "LoggedIn" and 
                       self.callbacks.login_status != "LoginFailed" and 
                       (time.time() - start_time) < timeout):
                    await asyncio.sleep(0.1)
                
                if self.callbacks.login_status == "LoggedIn":
                    self.is_connected = True
                    logger.info("✅ Connected to Rithmic R|API successfully")
                    
                    # Now login to market data
                    await self._login_market_data()
                    
                    return True
                else:
                    logger.error("❌ Rithmic login timeout or failed")
                    return False
            else:
                logger.error(f"❌ Rithmic login failed with code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Rithmic connection error: {e}")
            return False
    
    async def _login_market_data(self) -> bool:
        """Login to market data server"""
        try:
            logger.info("📊 Logging into Rithmic market data...")
            
            # Market data login
            md_login_params = MdLoginParams()
            md_login_params.User = self.user_id
            md_login_params.Password = self.password
            
            result = self.engine.loginToMd(md_login_params)
            
            if result == 0:
                # Wait for MD login completion
                timeout = 30
                start_time = time.time()
                
                while (not self.callbacks.logged_into_md and 
                       (time.time() - start_time) < timeout):
                    await asyncio.sleep(0.1)
                
                if self.callbacks.logged_into_md:
                    self.is_md_logged_in = True
                    logger.info("✅ Market data login successful")
                    return True
                else:
                    logger.error("❌ Market data login timeout")
                    return False
            else:
                logger.error(f"❌ Market data login failed with code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market data login error: {e}")
            return False
    
    async def subscribe_market_data(self, symbol: str, exchange: str = "CME") -> bool:
        """Subscribe to market data for a symbol"""
        if not self.is_md_logged_in:
            logger.error("❌ Not logged into market data")
            return False
        
        try:
            logger.info(f"📈 Subscribing to market data: {exchange} {symbol}")
            
            # Create subscription request
            subscribe_params = MdSubscribeParams()
            subscribe_params.Exchange = exchange
            subscribe_params.Ticker = symbol
            subscribe_params.UpdateBits = 127  # All updates
            
            result = self.engine.subscribe(subscribe_params)
            
            if result == 0:
                logger.info(f"✅ Subscribed to {exchange} {symbol}")
                return True
            else:
                logger.error(f"❌ Subscription failed for {symbol} with code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Subscription error for {symbol}: {e}")
            return False
    
    def add_tick_handler(self, callback: Callable[[RithmicMarketData], None]):
        """Add callback for real-time tick data"""
        self.tick_handlers.append(callback)
    
    def _process_market_data(self, market_data: RithmicMarketData):
        """Process incoming market data"""
        try:
            # Update cache
            self.market_data_cache[market_data.symbol] = market_data
            self.message_count += 1
            
            # Call all registered handlers
            for handler in self.tick_handlers:
                try:
                    handler(market_data)
                except Exception as e:
                    logger.error(f"❌ Error in tick handler: {e}")
            
            # Performance logging
            if self.message_count % 1000 == 0:
                logger.debug(f"📊 Processed {self.message_count} market data messages")
                
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
    
    def _update_bid_quote(self, symbol: str, price: float, size: int):
        """Update bid quote in cache"""
        if symbol in self.market_data_cache:
            self.market_data_cache[symbol].bid_price = price
            self.market_data_cache[symbol].bid_size = size
    
    def _update_ask_quote(self, symbol: str, price: float, size: int):
        """Update ask quote in cache"""
        if symbol in self.market_data_cache:
            self.market_data_cache[symbol].ask_price = price
            self.market_data_cache[symbol].ask_size = size
    
    def _on_login_success(self):
        """Handle successful login"""
        logger.info("✅ Rithmic repository login successful")
    
    def _on_login_failure(self, error_msg: str):
        """Handle login failure"""
        logger.error(f"❌ Rithmic login failed: {error_msg}")
    
    def _on_md_login_success(self):
        """Handle successful market data login"""
        logger.info("✅ Rithmic market data login successful")
    
    def _handle_login_alert(self, message: str):
        """Handle login-related alerts"""
        logger.info(f"🔔 Login alert: {message}")
    
    def _handle_disconnect_alert(self, message: str):
        """Handle disconnect alerts"""
        logger.warning(f"⚠️ Disconnect alert: {message}")
        self.is_connected = False
        self.is_md_logged_in = False
    
    async def disconnect(self):
        """Disconnect from Rithmic"""
        try:
            if self.engine:
                self.engine.shutdown()
                logger.info("✅ Disconnected from Rithmic")
                
            self.is_connected = False
            self.is_md_logged_in = False
            
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")
    
    def get_latest_data(self, symbol: str) -> Optional[RithmicMarketData]:
        """Get latest market data for symbol"""
        return self.market_data_cache.get(symbol)
    
    def get_connection_status(self) -> Dict:
        """Get connection status"""
        return {
            'connected': self.is_connected,
            'md_logged_in': self.is_md_logged_in,
            'message_count': self.message_count,
            'last_heartbeat': self.last_heartbeat,
            'subscribed_symbols': list(self.market_data_cache.keys())
        }

# Example usage for institutional trading
async def main():
    """Example usage of Professional Rithmic Connector"""
    connector = ProfessionalRithmicConnector()
    
    # Add tick handler for institutional processing
    def process_institutional_tick(market_data: RithmicMarketData):
        print(f"📊 {market_data.symbol}: {market_data.last_price} @ {market_data.timestamp}")
    
    connector.add_tick_handler(process_institutional_tick)
    
    # Load credentials (replace with your actual credentials)
    user_id = "your_user_id"
    password = "your_password"
    system_name = "your_system_name"
    
    try:
        # Connect to Rithmic
        if await connector.connect(user_id, password, system_name):
            
            # Subscribe to ES futures
            await connector.subscribe_market_data("ESZ4", "CME")
            await connector.subscribe_market_data("NQZ4", "CME")
            
            # Keep running to receive data
            logger.info("🔄 Receiving market data... Press Ctrl+C to stop")
            
            while True:
                await asyncio.sleep(1)
                
                # Print status every 30 seconds
                status = connector.get_connection_status()
                if status['message_count'] % 100 == 0:
                    logger.info(f"📊 Status: {status}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        await connector.disconnect()

if __name__ == "__main__":
    asyncio.run(main())