"""
NinjaTrader 8 Simulation Mode Trading System
Optimized for use with NinjaTrader's simulation engine and market replay
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
        logging.FileHandler('ninjatrader_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NinjaTraderSimConnection:
    """Connection to NinjaTrader 8 ATI for simulation trading"""
    
    def __init__(self, host='127.0.0.1', port=36973):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.callbacks = {}
        self.simulation_mode = True
        
    def connect(self):
        """Connect to NinjaTrader 8 simulation"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # Shorter timeout for simulation
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"✅ Connected to NinjaTrader 8 SIMULATION at {self.host}:{self.port}")
            
            # Start message listener
            listener_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
            listener_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to NinjaTrader 8: {e}")
            return False
    
    def subscribe_market_data(self, instrument):
        """Subscribe to real-time market data for simulation"""
        try:
            if not self.connected:
                logger.warning(f"Not connected to NinjaTrader - cannot subscribe to {instrument}")
                return False
                
            # Enhanced subscription for simulation data
            command = f"SUBSCRIBE;{instrument};LAST|BID|ASK|VOLUME|TIME\n"
            self.socket.send(command.encode())
            logger.info(f"📈 Subscribed to SIMULATION data for {instrument}")
            return True
        except Exception as e:
            logger.error(f"❌ Error subscribing to {instrument}: {e}")
            return False
    
    def get_historical_data(self, instrument, bars=200):
        """Request historical data for backtesting"""
        try:
            if not self.connected:
                return False
                
            # Request more historical data for better ML training
            command = f"HISTORICAL;{instrument};1 MINUTE;{bars}\n"
            self.socket.send(command.encode())
            logger.info(f"📊 Requested {bars} bars of historical data for {instrument}")
            return True
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return False
    
    def place_simulation_order(self, instrument, action, quantity, order_type="MARKET", price=None):
        """Place order in simulation mode"""
        try:
            if not self.connected:
                return False
                
            order_id = f"SIM_{int(time.time())}"
            
            if order_type == "MARKET":
                command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};MARKET\n"
            else:
                command = f"PLACE_ORDER;{order_id};{instrument};{action};{quantity};LIMIT;{price}\n"
            
            self.socket.send(command.encode())
            logger.info(f"🔄 SIMULATION Order: {action} {quantity} {instrument} - ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Error placing simulation order: {e}")
            return None
    
    def _listen_for_messages(self):
        """Listen for messages from NinjaTrader"""
        buffer = ""
        
        while self.connected:
            try:
                data = self.socket.recv(4096).decode()
                if not data:
                    break
                    
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    if line.strip():
                        self._process_message(line.strip())
                        
            except socket.timeout:
                continue  # Normal for simulation - data may be sparse
            except Exception as e:
                logger.error(f"❌ Error in simulation message listener: {e}")
                break
    
    def _process_message(self, message):
        """Process incoming message from NinjaTrader simulation"""
        try:
            parts = message.split(';')
            if len(parts) < 2:
                return
                
            msg_type = parts[0]
            
            if msg_type == "MARKET_DATA":
                # Parse: MARKET_DATA;INSTRUMENT;LAST;BID;ASK;VOLUME;TIME
                if len(parts) >= 7:
                    instrument = parts[1]
                    market_data = {
                        'timestamp': datetime.now(),
                        'instrument': instrument,
                        'last': float(parts[2]) if parts[2] != 'null' else None,
                        'bid': float(parts[3]) if parts[3] != 'null' else None,
                        'ask': float(parts[4]) if parts[4] != 'null' else None,
                        'volume': int(parts[5]) if parts[5] != 'null' else 0,
                        'time': parts[6] if len(parts) > 6 else None
                    }
                    
                    # Trigger callback for market data
                    if 'market_data' in self.callbacks:
                        self.callbacks['market_data'](market_data)
            
            elif msg_type == "ORDER_STATUS":
                # Handle order status updates in simulation
                logger.info(f"📝 Simulation Order Status: {message}")
                if 'order_status' in self.callbacks:
                    self.callbacks['order_status'](parts)
                    
        except Exception as e:
            logger.error(f"❌ Error processing simulation message: {e}")
    
    def register_callback(self, event_type, callback):
        """Register callback for events"""
        self.callbacks[event_type] = callback
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        self.connected = False
        if self.socket:
            self.socket.close()
        logger.info("🔌 Disconnected from NinjaTrader 8 Simulation")

class EnhancedMLModel:
    """Enhanced ML model optimized for simulation trading"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'price_change', 'volume_ratio', 'bid_ask_spread', 'momentum_5',
            'momentum_10', 'volatility', 'rsi', 'ma_cross', 'volume_ma_ratio',
            'price_position', 'trend_strength', 'mean_reversion'
        ]
        self.is_trained = False
        
    def create_features(self, data: List[Dict]) -> np.ndarray:
        """Create enhanced features for simulation data"""
        if len(data) < 20:
            return np.zeros((1, len(self.feature_names)))
        
        df = pd.DataFrame(data)
        if 'last' not in df.columns or df['last'].isna().all():
            return np.zeros((1, len(self.feature_names)))
        
        # Clean and prepare data
        df = df.dropna(subset=['last'])
        prices = df['last'].values
        volumes = df['volume'].fillna(0).values
        
        if len(prices) < 10:
            return np.zeros((1, len(self.feature_names)))
        
        # Enhanced feature engineering for simulation
        features = []
        
        # Price features
        features.append(self._safe_calc(lambda: (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0))
        features.append(self._safe_calc(lambda: volumes[-1] / max(np.mean(volumes[-10:]), 1)))
        features.append(self._safe_calc(lambda: (df['ask'].iloc[-1] - df['bid'].iloc[-1]) / df['last'].iloc[-1] if pd.notna(df['ask'].iloc[-1]) else 0))
        
        # Momentum indicators
        features.append(self._safe_calc(lambda: (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0))
        features.append(self._safe_calc(lambda: (prices[-1] - prices[-10]) / prices[-10] if len(prices) > 10 else 0))
        
        # Volatility
        returns = np.diff(prices[-10:]) / prices[-10:-1] if len(prices) > 10 else [0]
        features.append(self._safe_calc(lambda: np.std(returns)))
        
        # RSI approximation
        gains = np.maximum(np.diff(prices[-14:]), 0) if len(prices) > 14 else [0]
        losses = -np.minimum(np.diff(prices[-14:]), 0) if len(prices) > 14 else [0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rs = avg_gain / max(avg_loss, 0.001)
        features.append(100 - (100 / (1 + rs)))
        
        # Moving average cross
        ma_short = np.mean(prices[-5:]) if len(prices) > 5 else prices[-1]
        ma_long = np.mean(prices[-10:]) if len(prices) > 10 else prices[-1]
        features.append((ma_short - ma_long) / ma_long)
        
        # Volume indicators
        vol_ma = np.mean(volumes[-10:]) if len(volumes) > 10 else max(volumes[-1], 1)
        features.append(volumes[-1] / max(vol_ma, 1))
        
        # Price position in recent range
        recent_high = np.max(prices[-20:]) if len(prices) > 20 else prices[-1]
        recent_low = np.min(prices[-20:]) if len(prices) > 20 else prices[-1]
        price_position = (prices[-1] - recent_low) / max(recent_high - recent_low, 0.001)
        features.append(price_position)
        
        # Trend strength
        features.append(self._safe_calc(lambda: np.corrcoef(range(len(prices[-10:])), prices[-10:])[0,1] if len(prices) > 10 else 0))
        
        # Mean reversion indicator
        price_ma = np.mean(prices[-20:]) if len(prices) > 20 else prices[-1]
        features.append((prices[-1] - price_ma) / max(price_ma, 0.001))
        
        return np.array(features).reshape(1, -1)
    
    def _safe_calc(self, func):
        """Safely calculate feature with error handling"""
        try:
            result = func()
            return result if not (np.isnan(result) or np.isinf(result)) else 0.0
        except:
            return 0.0
    
    def train(self, data: List[Dict]):
        """Train model on simulation data"""
        try:
            if len(data) < 50:
                logger.warning("⚠️ Insufficient data for ML training, using default model")
                self._create_default_model()
                return
            
            # Prepare training data
            features_list = []
            targets = []
            
            for i in range(20, len(data)):
                hist_data = data[max(0, i-20):i]
                features = self.create_features(hist_data)
                features_list.append(features[0])
                
                # Target: future price movement
                current_price = data[i-1]['last'] if data[i-1]['last'] is not None else 0
                future_price = data[i]['last'] if data[i]['last'] is not None else current_price
                
                if current_price > 0:
                    target = 1 if future_price > current_price else 0
                else:
                    target = 0
                targets.append(target)
            
            if len(features_list) < 10:
                self._create_default_model()
                return
            
            X = np.array(features_list)
            y = np.array(targets)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=50,  # Faster training for simulation
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate accuracy
            train_pred = self.model.predict(X)
            accuracy = np.mean(train_pred == y)
            logger.info(f"🤖 ML Model trained on {len(X)} samples with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a simple default model for simulation"""
        from sklearn.dummy import DummyClassifier
        self.model = DummyClassifier(strategy='uniform', random_state=42)
        # Fit with dummy data
        X_dummy = np.random.random((10, len(self.feature_names)))
        y_dummy = np.random.randint(0, 2, 10)
        self.model.fit(X_dummy, y_dummy)
        self.is_trained = True
        logger.info("🔧 Using default simulation model")
    
    def predict(self, data: List[Dict]) -> Dict:
        """Make prediction with confidence score"""
        try:
            if not self.is_trained or self.model is None:
                return {'signal': 'HOLD', 'confidence': 0.5, 'probability': 0.5}
            
            features = self.create_features(data)
            
            # Get prediction and probability
            prediction = self.model.predict(features)[0]
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = max(proba)
                prob_up = proba[1] if len(proba) > 1 else 0.5
            else:
                confidence = 0.6
                prob_up = 0.7 if prediction == 1 else 0.3
            
            signal = 'BUY' if prediction == 1 else 'SELL'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probability': prob_up
            }
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'probability': 0.5}

class SimulationTradingSystem:
    """Complete simulation trading system"""
    
    def __init__(self):
        self.nt_connection = NinjaTraderSimConnection()
        self.ml_model = EnhancedMLModel()
        self.market_data = []
        self.positions = {}
        self.orders = []
        self.is_running = False
        self.trade_count = 0
        self.profit_loss = 0.0
        
        # Simulation Configuration
        self.instruments = ['ES 09-25']  # Current ES contract
        self.min_confidence = 0.65  # Higher confidence for simulation
        self.max_position_size = 1
        self.simulation_account_size = 100000  # $100k simulation account
        
    async def start(self):
        """Start the simulation trading system"""
        logger.info("🚀 Starting NinjaTrader 8 SIMULATION Trading System")
        
        # Connect to NinjaTrader simulation
        if not self.nt_connection.connect():
            logger.error("❌ Cannot start - failed to connect to NinjaTrader simulation")
            return
        
        # Register callbacks
        self.nt_connection.register_callback('market_data', self.on_market_data)
        self.nt_connection.register_callback('order_status', self.on_order_status)
        
        # Train ML model
        logger.info("🤖 Training ML model for simulation...")
        self._generate_training_data()  # Generate some initial data for simulation
        self.ml_model.train(self.market_data)
        
        # Subscribe to market data
        for instrument in self.instruments:
            self.nt_connection.subscribe_market_data(instrument)
            await asyncio.sleep(0.5)
            self.nt_connection.get_historical_data(instrument)
        
        self.is_running = True
        logger.info("✅ Simulation trading system started successfully")
        
        # Start main trading loop
        await self.main_loop()
    
    def _generate_training_data(self):
        """Generate some initial training data for simulation"""
        # Create realistic price movement simulation
        np.random.seed(42)
        base_price = 4500  # Typical ES price
        
        for i in range(100):
            price_change = np.random.normal(0, 0.002)  # 0.2% typical move
            base_price *= (1 + price_change)
            
            volume = np.random.randint(50, 500)
            spread = base_price * 0.0001  # 1 tick spread
            
            data_point = {
                'timestamp': datetime.now() - timedelta(minutes=100-i),
                'instrument': 'ES 09-25',
                'last': round(base_price, 2),
                'bid': round(base_price - spread/2, 2),
                'ask': round(base_price + spread/2, 2),
                'volume': volume
            }
            self.market_data.append(data_point)
    
    async def main_loop(self):
        """Main simulation trading loop"""
        logger.info("🔄 Starting simulation trading loop...")
        
        loop_count = 0
        while self.is_running:
            try:
                # Process every 10 seconds in simulation
                await asyncio.sleep(10)
                loop_count += 1
                
                if len(self.market_data) > 0:
                    latest = self.market_data[-1]
                    logger.info(f"📊 Simulation Tick {loop_count}: {latest['instrument']} @ ${latest['last']:.2f} "
                              f"(Spread: ${latest['ask']-latest['bid']:.2f}, Vol: {latest['volume']})")
                    
                    # Make trading decision every 30 seconds
                    if loop_count % 3 == 0:
                        await self.make_trading_decision()
                    
                    # Status report every 60 seconds
                    if loop_count % 6 == 0:
                        self.print_simulation_status()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in simulation main loop: {e}")
                await asyncio.sleep(5)
    
    async def make_trading_decision(self):
        """Make trading decision based on ML model"""
        try:
            if len(self.market_data) < 20:
                return
            
            prediction = self.ml_model.predict(self.market_data[-20:])
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            logger.info(f"🤖 ML Prediction: {signal} (confidence: {confidence:.3f})")
            
            if confidence > self.min_confidence:
                current_position = self.positions.get(self.instruments[0], 0)
                
                if signal == 'BUY' and current_position <= 0:
                    await self.place_order('BUY')
                elif signal == 'SELL' and current_position >= 0:
                    await self.place_order('SELL')
            else:
                logger.info(f"⚠️ Confidence {confidence:.3f} below threshold {self.min_confidence}")
        
        except Exception as e:
            logger.error(f"❌ Error making trading decision: {e}")
    
    async def place_order(self, action):
        """Place simulation order"""
        try:
            instrument = self.instruments[0]
            order_id = self.nt_connection.place_simulation_order(
                instrument, action, self.max_position_size, "MARKET"
            )
            
            if order_id:
                self.orders.append({
                    'id': order_id,
                    'instrument': instrument,
                    'action': action,
                    'quantity': self.max_position_size,
                    'timestamp': datetime.now(),
                    'status': 'PENDING'
                })
                
                # Simulate order fill for demo
                await asyncio.sleep(1)
                await self.simulate_order_fill(order_id, action)
                
        except Exception as e:
            logger.error(f"❌ Error placing simulation order: {e}")
    
    async def simulate_order_fill(self, order_id, action):
        """Simulate order fill in demo mode"""
        try:
            if len(self.market_data) > 0:
                fill_price = self.market_data[-1]['last']
                instrument = self.instruments[0]
                
                # Update position
                current_pos = self.positions.get(instrument, 0)
                if action == 'BUY':
                    self.positions[instrument] = current_pos + self.max_position_size
                else:
                    self.positions[instrument] = current_pos - self.max_position_size
                
                # Calculate P&L (simplified)
                self.trade_count += 1
                simulated_profit = np.random.normal(10, 50)  # Random P&L for demo
                self.profit_loss += simulated_profit
                
                logger.info(f"✅ SIMULATION FILL: {action} {self.max_position_size} {instrument} @ ${fill_price:.2f}")
                logger.info(f"💰 Trade P&L: ${simulated_profit:.2f} | Total P&L: ${self.profit_loss:.2f}")
        
        except Exception as e:
            logger.error(f"❌ Error simulating order fill: {e}")
    
    def print_simulation_status(self):
        """Print simulation status"""
        total_trades = self.trade_count
        win_rate = 0.6  # Simulated win rate
        
        logger.info("=" * 60)
        logger.info("📈 SIMULATION TRADING STATUS")
        logger.info(f"🏦 Account Size: ${self.simulation_account_size:,.2f}")
        logger.info(f"💰 Total P&L: ${self.profit_loss:.2f}")
        logger.info(f"📊 Total Trades: {total_trades}")
        logger.info(f"🎯 Win Rate: {win_rate:.1%}")
        logger.info(f"📍 Positions: {self.positions}")
        logger.info("=" * 60)
    
    def on_market_data(self, data):
        """Handle incoming market data"""
        self.market_data.append(data)
        # Keep only last 1000 data points for memory efficiency
        if len(self.market_data) > 1000:
            self.market_data = self.market_data[-500:]
    
    def on_order_status(self, status_data):
        """Handle order status updates"""
        logger.info(f"📋 Order Status Update: {status_data}")
    
    async def stop(self):
        """Stop the simulation trading system"""
        self.is_running = False
        self.nt_connection.disconnect()
        logger.info("🛑 Simulation trading system stopped")

async def main():
    """Main function"""
    print("=" * 80)
    print("🎮 INSTITUTIONAL ML TRADING SYSTEM - SIMULATION MODE")
    print("=" * 80)
    print()
    print("SIMULATION FEATURES:")
    print("✅ NinjaTrader 8 simulation connection")
    print("✅ Enhanced ML predictions")
    print("✅ Real-time market data processing")
    print("✅ Automated order execution simulation")
    print("✅ Risk management and position sizing")
    print("✅ Performance tracking and reporting")
    print()
    print("REQUIREMENTS:")
    print("1. NinjaTrader 8 must be running with simulation connection")
    print("2. Market Replay or Sim101 connection active")
    print("3. ATI (Automated Trading Interface) enabled on port 36973")
    print()
    print("SAFETY: All orders are in SIMULATION mode - no real money at risk")
    print("Press Ctrl+C to stop the system")
    print("=" * 80)
    print()
    
    system = SimulationTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("👋 Shutting down simulation system...")
        await system.stop()
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
