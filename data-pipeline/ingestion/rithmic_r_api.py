"""
Rithmic R | API Socket Client
Professional implementation following Rithmic R | API protocol
"""

import socket
import json
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class RithmicRAPIClient:
    """
    Professional Rithmic R | API socket client
    Implements the actual Rithmic API protocol for institutional trading
    """
    
    def __init__(self, sdk_path: str):
        self.sdk_path = sdk_path
        self.host = "127.0.0.1"  # Default Rithmic R | API Gateway
        self.port = 65000        # Default R | API port
        
        # Connection management
        self.socket = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.response_handlers = {}
        self.subscription_callbacks = {}
        
        # R | API session management
        self.session_id = None
        self.login_token = None
        self.request_id = 1
        
        # Background threads
        self.reader_thread = None
        self.heartbeat_thread = None
        
    async def connect(self, username: str, password: str, system_name: str = "paper_trading"):
        """Connect to Rithmic R | API Gateway"""
        try:
            logger.info(f"🔌 Connecting to Rithmic R | API at {self.host}:{self.port}")
            
            # Create socket connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Start message reader thread
            self.reader_thread = threading.Thread(target=self._message_reader, daemon=True)
            self.reader_thread.start()
            
            # Login to R | API
            login_success = await self._login(username, password, system_name)
            
            if login_success:
                self.is_connected = True
                
                # Start heartbeat thread
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self.heartbeat_thread.start()
                
                logger.info("✅ Connected to Rithmic R | API successfully")
                return True
            else:
                logger.error("❌ Rithmic R | API login failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Rithmic R | API connection error: {e}")
            return False
    
    async def _login(self, username: str, password: str, system_name: str) -> bool:
        """Perform R | API login sequence"""
        try:
            # R | API Login Request (template)
            login_request = {
                "template_id": 10,  # Login Request
                "user_msg": [
                    username,
                    password,
                    system_name,
                    "1.0.0"  # API version
                ]
            }
            
            # Send login request
            response = await self._send_request(login_request)
            
            if response and response.get("template_id") == 11:  # Login Response
                if response.get("rpcode") == "0":  # Success
                    self.session_id = response.get("user_msg", [None])[0]
                    logger.info("✅ R | API login successful")
                    return True
                else:
                    logger.error(f"❌ R | API login failed: {response.get('user_msg')}")
                    return False
            else:
                logger.error("❌ Invalid R | API login response")
                return False
                
        except Exception as e:
            logger.error(f"❌ R | API login error: {e}")
            return False
    
    async def subscribe_market_data(self, symbol: str, exchange: str = "CME") -> bool:
        """Subscribe to real-time market data"""
        try:
            # R | API Market Data Request
            md_request = {
                "template_id": 100,  # Market Data Request
                "user_msg": [
                    symbol,
                    exchange,
                    "1"  # Subscribe flag
                ]
            }
            
            response = await self._send_request(md_request)
            
            if response and response.get("rpcode") == "0":
                logger.info(f"✅ Subscribed to {symbol} market data")
                return True
            else:
                logger.error(f"❌ Market data subscription failed for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market data subscription error: {e}")
            return False
    
    async def _send_request(self, request: Dict) -> Optional[Dict]:
        """Send request to R | API and wait for response"""
        try:
            if not self.socket:
                return None
            
            # Add request ID
            request["rq_id"] = str(self.request_id)
            self.request_id += 1
            
            # Serialize message
            message = json.dumps(request) + "\n"
            
            # Send message
            self.socket.send(message.encode('utf-8'))
            
            # Wait for response (simplified - production would use proper request/response matching)
            time.sleep(0.1)
            
            try:
                response = self.message_queue.get(timeout=5)
                return response
            except queue.Empty:
                logger.warning("⚠️ R | API request timeout")
                return None
                
        except Exception as e:
            logger.error(f"❌ R | API send error: {e}")
            return None
    
    def _message_reader(self):
        """Background thread to read messages from R | API"""
        buffer = ""
        
        while self.is_connected:
            try:
                if not self.socket:
                    break
                
                # Read data
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (line-separated)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            self._handle_message(message)
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Invalid R | API message: {line}")
                
            except Exception as e:
                logger.error(f"❌ R | API reader error: {e}")
                break
    
    def _handle_message(self, message: Dict):
        """Handle incoming R | API message"""
        try:
            template_id = message.get("template_id")
            
            if template_id == 101:  # Market Data Update
                self._handle_market_data(message)
            elif template_id == 11:  # Login Response
                self.message_queue.put(message)
            elif template_id == 102:  # Market Data Response
                self.message_queue.put(message)
            else:
                # Generic response
                self.message_queue.put(message)
                
        except Exception as e:
            logger.error(f"❌ Message handling error: {e}")
    
    def _handle_market_data(self, message: Dict):
        """Handle real-time market data updates"""
        try:
            user_msg = message.get("user_msg", [])
            if len(user_msg) >= 6:
                symbol = user_msg[0]
                last_price = float(user_msg[1]) if user_msg[1] else 0.0
                last_size = int(user_msg[2]) if user_msg[2] else 0
                bid_price = float(user_msg[3]) if user_msg[3] else 0.0
                ask_price = float(user_msg[4]) if user_msg[4] else 0.0
                volume = int(user_msg[5]) if user_msg[5] else 0
                
                # Create tick data
                tick_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'price': last_price,
                    'size': last_size,
                    'bid': bid_price,
                    'ask': ask_price,
                    'volume': volume
                }
                
                # Call callbacks
                for callback in self.subscription_callbacks.get(symbol, []):
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"❌ Callback error: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Market data processing error: {e}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain connection"""
        while self.is_connected:
            try:
                if self.socket:
                    heartbeat = {
                        "template_id": 18,  # Heartbeat
                        "user_msg": []
                    }
                    message = json.dumps(heartbeat) + "\n"
                    self.socket.send(message.encode('utf-8'))
                
                time.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                break
    
    def register_callback(self, symbol: str, callback: Callable):
        """Register callback for market data updates"""
        if symbol not in self.subscription_callbacks:
            self.subscription_callbacks[symbol] = []
        self.subscription_callbacks[symbol].append(callback)
    
    async def disconnect(self):
        """Disconnect from R | API"""
        try:
            self.is_connected = False
            
            if self.socket:
                # Send logout message
                logout = {
                    "template_id": 12,  # Logout Request
                    "user_msg": []
                }
                message = json.dumps(logout) + "\n"
                self.socket.send(message.encode('utf-8'))
                
                time.sleep(1)
                self.socket.close()
                self.socket = None
            
            logger.info("✅ Disconnected from Rithmic R | API")
            
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")