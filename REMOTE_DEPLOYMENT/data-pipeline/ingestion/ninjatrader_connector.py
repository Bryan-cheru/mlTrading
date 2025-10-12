"""
NinjaTrader 8 Integration Module
Connects to NinjaTrader 8 for real-time data and order execution
"""

import socket
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure compatible with NinjaTrader"""
    instrument: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int

@dataclass
class BarData:
    """OHLCV bar data from NinjaTrader"""
    instrument: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    time_frame: str  # "1 Minute", "5 Minute", etc.

class NinjaTraderConnector:
    """
    NinjaTrader 8 connector using ATI (Automated Trading Interface)
    Supports real-time data streaming and order execution
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 36973):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self.data_callbacks = {}
        self.order_callbacks = {}
        self.subscriptions = set()
        self.running = False
        
    def connect(self) -> bool:
        """Connect to NinjaTrader 8 ATI"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            logger.info(f"Connected to NinjaTrader 8 at {self.host}:{self.port}")
            
            # Start message listener thread
            self.running = True
            self.listener_thread = threading.Thread(target=self._message_listener)
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to NinjaTrader: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.is_connected = False
        logger.info("Disconnected from NinjaTrader")
    
    def subscribe_market_data(self, instrument: str, callback: Callable[[MarketData], None]):
        """Subscribe to real-time market data for an instrument"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        # NinjaTrader ATI command format
        command = f"SUBSCRIBE;{instrument};BID|ASK|LAST|VOLUME\r\n"
        self.socket.send(command.encode())
        
        self.data_callbacks[instrument] = callback
        self.subscriptions.add(instrument)
        logger.info(f"Subscribed to market data for {instrument}")
    
    def subscribe_bars(self, instrument: str, timeframe: str, callback: Callable[[BarData], None]):
        """Subscribe to bar data (OHLCV)"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        command = f"SUBSCRIBE_BARS;{instrument};{timeframe}\r\n"
        self.socket.send(command.encode())
        
        bar_key = f"{instrument}_{timeframe}"
        self.data_callbacks[bar_key] = callback
        logger.info(f"Subscribed to {timeframe} bars for {instrument}")
    
    def get_historical_data(self, instrument: str, timeframe: str, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Request historical data from NinjaTrader"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        # Format dates for NinjaTrader
        start_str = start_date.strftime("%Y%m%d %H:%M:%S")
        end_str = end_date.strftime("%Y%m%d %H:%M:%S")
        
        command = f"HISTORICAL;{instrument};{timeframe};{start_str};{end_str}\r\n"
        self.socket.send(command.encode())
        
        # Wait for response (simplified - in production use proper async handling)
        time.sleep(2)
        
        # Return placeholder DataFrame - in real implementation, collect actual data
        return pd.DataFrame()
    
    def place_order(self, instrument: str, action: str, quantity: int, 
                   order_type: str = "MARKET", price: float = None) -> str:
        """Place order through NinjaTrader"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        order_id = f"ORDER_{int(time.time())}"
        
        if order_type == "MARKET":
            command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};MARKET\r\n"
        elif order_type == "LIMIT":
            command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};LIMIT;{price}\r\n"
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        self.socket.send(command.encode())
        logger.info(f"Placed {action} order for {quantity} {instrument}")
        
        return order_id
    
    def cancel_order(self, order_id: str):
        """Cancel an existing order"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        command = f"CANCEL_ORDER;{order_id}\r\n"
        self.socket.send(command.encode())
        logger.info(f"Cancelled order {order_id}")
    
    def get_position(self, instrument: str) -> Dict:
        """Get current position for instrument"""
        if not self.is_connected:
            raise ConnectionError("Not connected to NinjaTrader")
        
        command = f"GET_POSITION;{instrument}\r\n"
        self.socket.send(command.encode())
        
        # Return placeholder - in real implementation, parse response
        return {"instrument": instrument, "quantity": 0, "average_price": 0.0}
    
    def _message_listener(self):
        """Listen for incoming messages from NinjaTrader"""
        buffer = ""
        
        while self.running and self.is_connected:
            try:
                data = self.socket.recv(4096).decode()
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\r\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    if line:
                        self._process_message(line)
                        
            except Exception as e:
                logger.error(f"Error in message listener: {e}")
                break
    
    def _process_message(self, message: str):
        """Process incoming message from NinjaTrader"""
        parts = message.split(';')
        
        if not parts:
            return
        
        msg_type = parts[0]
        
        if msg_type == "MARKET_DATA" and len(parts) >= 7:
            # Parse market data: MARKET_DATA;INSTRUMENT;BID;ASK;LAST;VOLUME;TIMESTAMP
            instrument = parts[1]
            bid = float(parts[2])
            ask = float(parts[3])
            last = float(parts[4])
            volume = int(parts[5])
            timestamp = datetime.now()  # In real implementation, parse timestamp
            
            market_data = MarketData(
                instrument=instrument,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                bid_size=100,  # Default sizes
                ask_size=100
            )
            
            if instrument in self.data_callbacks:
                self.data_callbacks[instrument](market_data)
        
        elif msg_type == "BAR_DATA" and len(parts) >= 8:
            # Parse bar data: BAR_DATA;INSTRUMENT;TIMEFRAME;OPEN;HIGH;LOW;CLOSE;VOLUME;TIMESTAMP
            instrument = parts[1]
            timeframe = parts[2]
            open_price = float(parts[3])
            high = float(parts[4])
            low = float(parts[5])
            close = float(parts[6])
            volume = int(parts[7])
            timestamp = datetime.now()
            
            bar_data = BarData(
                instrument=instrument,
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                time_frame=timeframe
            )
            
            bar_key = f"{instrument}_{timeframe}"
            if bar_key in self.data_callbacks:
                self.data_callbacks[bar_key](bar_data)
        
        elif msg_type == "ORDER_UPDATE":
            # Handle order updates
            if len(parts) >= 4:
                order_id = parts[1]
                status = parts[2]
                message = parts[3] if len(parts) > 3 else ""
                logger.info(f"Order {order_id} status: {status} - {message}")


class NinjaTraderDataProvider:
    """
    High-level data provider that manages NinjaTrader connections
    and provides clean interface for the ML system
    """
    
    def __init__(self):
        self.connector = NinjaTraderConnector()
        self.current_data = {}
        self.bar_data = {}
        
    def connect(self) -> bool:
        """Connect to NinjaTrader"""
        return self.connector.connect()
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        self.connector.disconnect()
    
    def add_instrument(self, instrument: str, timeframes: List[str] = ["1 Minute"]):
        """Add instrument for data collection"""
        # Subscribe to real-time data
        self.connector.subscribe_market_data(
            instrument, 
            lambda data: self._update_market_data(data)
        )
        
        # Subscribe to bar data for each timeframe
        for tf in timeframes:
            self.connector.subscribe_bars(
                instrument, 
                tf, 
                lambda data: self._update_bar_data(data)
            )
        
        logger.info(f"Added instrument {instrument} with timeframes {timeframes}")
    
    def _update_market_data(self, data: MarketData):
        """Update current market data"""
        self.current_data[data.instrument] = data
    
    def _update_bar_data(self, data: BarData):
        """Update bar data"""
        key = f"{data.instrument}_{data.time_frame}"
        if key not in self.bar_data:
            self.bar_data[key] = []
        self.bar_data[key].append(data)
        
        # Keep only last 1000 bars to manage memory
        if len(self.bar_data[key]) > 1000:
            self.bar_data[key] = self.bar_data[key][-1000:]
    
    def get_current_price(self, instrument: str) -> Optional[float]:
        """Get current price for instrument"""
        if instrument in self.current_data:
            return self.current_data[instrument].last
        return None
    
    def get_bars_dataframe(self, instrument: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Get bar data as pandas DataFrame"""
        key = f"{instrument}_{timeframe}"
        
        if key not in self.bar_data or not self.bar_data[key]:
            return pd.DataFrame()
        
        bars = self.bar_data[key][-periods:]
        
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            for bar in bars
        ])
        
        df.set_index('timestamp', inplace=True)
        return df
