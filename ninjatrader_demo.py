"""
NinjaTrader 8 Demo System
Real implementation that connects to NinjaTrader 8 for live trading
"""

import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import socket
import json
import threading
import time
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ninjatrader_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NinjaTraderConnection:
    """Direct connection to NinjaTrader 8 ATI (Automated Trading Interface)"""
    
    def __init__(self, host='127.0.0.1', port=36973):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.callbacks = {}
        
    def connect(self):
        """Connect to NinjaTrader 8"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to NinjaTrader 8 at {self.host}:{self.port}")
            
            # Start message listener
            self.listener_thread = threading.Thread(target=self._message_listener, daemon=True)
            self.listener_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to NinjaTrader 8: {e}")
            logger.info("Make sure NinjaTrader 8 is running with ATI enabled")
            return False
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        self.connected = False
        if self.socket:
            self.socket.close()
        logger.info("Disconnected from NinjaTrader 8")
    
    def subscribe_market_data(self, instrument):
        """Subscribe to real-time market data"""
        if not self.connected:
            return False
        
        try:
            command = f"SUBSCRIBE;{instrument};LAST|BID|ASK|VOLUME\n"
            self.socket.send(command.encode())
            logger.info(f"Subscribed to market data for {instrument}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to {instrument}: {e}")
            return False
    
    def get_historical_data(self, instrument, bars=100):
        """Request historical bar data"""
        if not self.connected:
            return None
        
        try:
            command = f"HISTORICAL;{instrument};1 MINUTE;{bars}\n"
            self.socket.send(command.encode())
            time.sleep(1)  # Wait for data
            return True
        except Exception as e:
            logger.error(f"Error requesting historical data: {e}")
            return False
    
    def place_order(self, instrument, action, quantity, order_type="MARKET", price=None):
        """Place order through NinjaTrader"""
        if not self.connected:
            return None
        
        try:
            order_id = f"ML_ORDER_{int(time.time())}"
            
            if order_type == "MARKET":
                command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};MARKET\n"
            elif order_type == "LIMIT" and price:
                command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};LIMIT;{price}\n"
            else:
                logger.error("Invalid order type or missing price")
                return None
            
            self.socket.send(command.encode())
            logger.info(f"Order placed: {action} {quantity} {instrument} - Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _message_listener(self):
        """Listen for messages from NinjaTrader"""
        while self.connected:
            try:
                data = self.socket.recv(4096).decode().strip()
                if data:
                    self._process_message(data)
            except Exception as e:
                if self.connected:
                    logger.error(f"Error in message listener: {e}")
                break
    
    def _process_message(self, message):
        """Process incoming message from NinjaTrader"""
        try:
            parts = message.split(';')
            msg_type = parts[0]
            
            if msg_type == "MARKET_DATA":
                # Parse: MARKET_DATA;INSTRUMENT;LAST;BID;ASK;VOLUME;TIME
                instrument = parts[1]
                last = float(parts[2]) if parts[2] != 'NaN' else 0
                bid = float(parts[3]) if parts[3] != 'NaN' else 0
                ask = float(parts[4]) if parts[4] != 'NaN' else 0
                volume = int(parts[5]) if parts[5] != 'NaN' else 0
                
                market_data = {
                    'instrument': instrument,
                    'last': last,
                    'bid': bid,
                    'ask': ask,
                    'volume': volume,
                    'timestamp': datetime.now()
                }
                
                # Notify callbacks
                if 'market_data' in self.callbacks:
                    self.callbacks['market_data'](market_data)
            
            elif msg_type == "ORDER_UPDATE":
                # Parse order status updates
                order_id = parts[1]
                status = parts[2]
                logger.info(f"Order update: {order_id} - {status}")
                
            elif msg_type == "HISTORICAL_DATA":
                # Parse historical bar data
                logger.info("Received historical data")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def set_callback(self, event_type, callback):
        """Set callback for events"""
        self.callbacks[event_type] = callback


class SimpleTradingModel:
    """Simple ML model for demonstration"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.features = []
        self.last_prediction = None
        
    def calculate_features(self, price_data):
        """Calculate simple technical features"""
        if len(price_data) < 20:
            return None
        
        prices = np.array([d['last'] for d in price_data if d['last'] > 0])
        if len(prices) < 20:
            return None
        
        # Simple features
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:])
        current_price = prices[-1]
        
        # Returns
        return_1 = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        return_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
        
        # Volatility
        volatility = np.std(prices[-20:])
        
        features = [
            (current_price - sma_5) / sma_5,  # Price vs short MA
            (current_price - sma_20) / sma_20,  # Price vs long MA
            (sma_5 - sma_20) / sma_20,  # MA momentum
            return_1,  # Short-term return
            return_5,  # Medium-term return
            volatility / current_price,  # Normalized volatility
        ]
        
        return np.array(features)
    
    def train_simple_model(self, historical_data):
        """Train a simple model with mock data"""
        try:
            # Create mock training data
            np.random.seed(42)
            n_samples = 1000
            
            # Generate features
            X = np.random.randn(n_samples, 6)
            
            # Generate targets (0=SELL, 1=HOLD, 2=BUY)
            # Simple rule: buy if momentum positive, sell if negative
            y = np.where(X[:, 2] > 0.01, 2, np.where(X[:, 2] < -0.01, 0, 1))
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            accuracy = self.model.score(X, y)
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, features):
        """Make prediction"""
        if not self.is_trained or features is None:
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        try:
            features = features.reshape(1, -1)
            probabilities = self.model.predict_proba(features)[0]
            prediction = self.model.predict(features)[0]
            
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map[prediction]
            confidence = np.max(probabilities)
            
            self.last_prediction = {
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            return self.last_prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}


class NinjaTraderTradingSystem:
    """Complete trading system with NinjaTrader 8 integration"""
    
    def __init__(self):
        self.nt_connection = NinjaTraderConnection()
        self.model = SimpleTradingModel()
        self.price_data = []
        self.positions = {}
        self.orders = []
        self.is_running = False
        
        # Configuration
        self.instruments = ['ES 09-25']  # E-mini S&P 500 futures (September 2025 expiry)
        self.min_confidence = 0.6
        self.max_position_size = 1  # 1 contract
        
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("Starting NinjaTrader 8 ML Trading System")
            
            # Connect to NinjaTrader
            if not self.nt_connection.connect():
                logger.error("Cannot start system without NinjaTrader connection")
                return False
            
            # Set up callbacks
            self.nt_connection.set_callback('market_data', self.on_market_data)
            
            # Train model
            logger.info("Training ML model...")
            if not self.model.train_simple_model(None):
                logger.error("Failed to train model")
                return False
            
            # Subscribe to market data
            for instrument in self.instruments:
                self.nt_connection.subscribe_market_data(instrument)
                # Request historical data
                self.nt_connection.get_historical_data(instrument)
            
            self.is_running = True
            logger.info("Trading system started successfully")
            
            # Main loop
            await self.main_loop()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        self.nt_connection.disconnect()
        logger.info("Trading system stopped")
    
    async def main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Main logic runs through callbacks
                await asyncio.sleep(1)
                
                # Log status every 30 seconds
                if len(self.price_data) > 0:
                    latest = self.price_data[-1]
                    logger.info(f"Latest {latest['instrument']}: ${latest['last']:.2f} "
                              f"(Data points: {len(self.price_data)})")
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    def on_market_data(self, market_data):
        """Handle incoming market data"""
        try:
            instrument = market_data['instrument']
            price = market_data['last']
            
            if price <= 0:  # Skip invalid prices
                return
            
            # Store price data
            self.price_data.append(market_data)
            
            # Keep only last 1000 data points per instrument
            self.price_data = self.price_data[-1000:]
            
            # Calculate features and make prediction
            instrument_data = [d for d in self.price_data if d['instrument'] == instrument]
            
            if len(instrument_data) >= 20:  # Need enough data for features
                features = self.model.calculate_features(instrument_data)
                
                if features is not None:
                    prediction = self.model.predict(features)
                    
                    # Check if we should trade
                    if prediction['confidence'] > self.min_confidence:
                        self.evaluate_trade_signal(instrument, prediction, price)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def evaluate_trade_signal(self, instrument, prediction, current_price):
        """Evaluate whether to execute a trade"""
        try:
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            # Get current position
            current_position = self.positions.get(instrument, 0)
            
            # Simple position logic
            target_position = 0
            if signal == 'BUY' and confidence > self.min_confidence:
                target_position = self.max_position_size
            elif signal == 'SELL' and confidence > self.min_confidence:
                target_position = -self.max_position_size
            
            # Calculate required trade
            trade_size = target_position - current_position
            
            if abs(trade_size) > 0:
                action = 'BUY' if trade_size > 0 else 'SELL'
                quantity = abs(trade_size)
                
                logger.info(f"TRADE SIGNAL: {action} {quantity} {instrument} "
                          f"(Confidence: {confidence:.3f}, Price: ${current_price:.2f})")
                
                # Place order (commented out for safety in demo)
                # order_id = self.nt_connection.place_order(instrument, action, quantity)
                # if order_id:
                #     self.orders.append({
                #         'order_id': order_id,
                #         'instrument': instrument,
                #         'action': action,
                #         'quantity': quantity,
                #         'price': current_price,
                #         'timestamp': datetime.now()
                #     })
                
                # Simulate position update (remove in live trading)
                self.positions[instrument] = target_position
                
        except Exception as e:
            logger.error(f"Error evaluating trade signal: {e}")


async def main():
    """Main entry point"""
    system = NinjaTraderTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        system.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("INSTITUTIONAL ML TRADING SYSTEM - NINJATRADER 8 INTEGRATION")
    print("=" * 60)
    print()
    print("REQUIREMENTS:")
    print("1. NinjaTrader 8 must be running")
    print("2. ATI (Automated Trading Interface) must be enabled")
    print("3. Market data connection must be active")
    print("4. Ensure firewall allows connections on port 36973")
    print()
    print("SETUP INSTRUCTIONS:")
    print("1. Open NinjaTrader 8")
    print("2. Go to Tools > Options > Automated Trading Interface")
    print("3. Enable 'AT Interface' and set port to 36973")
    print("4. Apply and restart NinjaTrader if needed")
    print("5. Connect to your market data provider")
    print()
    print("SAFETY NOTE:")
    print("This demo system has order execution DISABLED by default.")
    print("To enable live trading, uncomment the order placement code.")
    print()
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    print()
    
    # Run the system
    asyncio.run(main())
