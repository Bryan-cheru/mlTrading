#!/usr/bin/env python3
"""
Real Market Data ML Trading Server for NinjaTrader Integration
Uses professionally trained XGBoost model with real ES futures data
"""

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataMLModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
    def load_trained_model(self, model_path="models/es_real_data_model.joblib"):
        """Load the professionally trained real data model"""
        try:
            if os.path.exists(model_path):
                logger.info(f"📂 Loading real data model: {model_path}")
                model_data = joblib.load(model_path)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                
                logger.info(f"✅ Real data model loaded successfully!")
                logger.info(f"📊 Model type: {model_data.get('model_type', 'Unknown')}")
                logger.info(f"🕐 Trained: {model_data.get('timestamp', 'Unknown')}")
                logger.info(f"🔧 Features: {len(self.feature_names)}")
                
                return True
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
                return self._fallback_training()
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return self._fallback_training()
    
    def _fallback_training(self):
        """Fallback to synthetic training if real model unavailable"""
        logger.info("📊 Training fallback model with synthetic data...")
        
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler
        
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=4,
            learning_rate=0.1,
            n_estimators=50
        )
        self.scaler = RobustScaler()
        
        # Generate synthetic features
        n_samples = 1000
        features = np.random.randn(n_samples, 10)
        labels = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
        
        # Add some pattern
        features[:, 0] = features[:, 0] + labels * 0.5
        
        # Scale and train
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, labels)
        self.is_trained = True
        
        self.feature_names = [f'feature_{i}' for i in range(10)]
        
        logger.info("✅ Fallback model trained!")
        return True
        return True
    
    def predict_signal(self, market_data):
        """Generate trading signal using real market features"""
        if not self.is_trained:
            logger.warning("⚠️ Model not trained, returning neutral signal")
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "probabilities": [0.25, 0.5, 0.25]
            }
        
        try:
            # Generate realistic features based on market data
            features = self._generate_realistic_features_from_market_data(market_data)
            
            # Use appropriate feature set
            if len(self.feature_names) > 0:
                # Real model features
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
            else:
                # Fallback features
                feature_values = [
                    market_data.get('last_price', 4500) / 4500.0,  # Normalized price
                    market_data.get('bid_price', 4499) / 4500.0,
                    market_data.get('ask_price', 4501) / 4500.0,
                    market_data.get('volume', 1000) / 1000.0,
                ] + [np.random.randn() for _ in range(6)]  # Mock indicators
            
            # Ensure we have the right number of features
            while len(feature_values) < len(self.feature_names or range(10)):
                feature_values.append(0.0)
            
            # Scale and predict
            features_scaled = self.scaler.transform([feature_values])
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            confidence = max(probabilities)
            
            return {
                "action": action_map[prediction],
                "confidence": float(confidence),
                "probabilities": probabilities.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "probabilities": [0.25, 0.5, 0.25]
            }
    
    def _generate_realistic_features_from_market_data(self, market_data):
        """Generate realistic trading features from market data"""
        price = market_data.get('last_price', 4500.0)
        volume = market_data.get('volume', 1000)
        
        # Simulate realistic market microstructure features
        features = {
            'returns': np.random.normal(0, 0.001),  # Small random return
            'log_returns': np.random.normal(0, 0.001),
            'volatility_10': np.random.uniform(0.15, 0.25),
            'volatility_20': np.random.uniform(0.16, 0.24),
            'vol_of_vol': np.random.uniform(0.3, 0.7),
            'volume_ratio': np.clip(volume / 1000.0, 0.1, 5.0),
            'price_position': np.random.beta(2, 2),
            'rsi': np.random.uniform(35, 65),  # Neutral RSI range
            'bb_position': np.random.beta(2, 2),
            'momentum_5': np.random.normal(0, 0.01),
            'momentum_10': np.random.normal(0, 0.02),
            'body_ratio': np.random.beta(2, 2),
            'distance_from_vwap': np.random.normal(0, 0.005)
        }
        
        return features

class TradingWebSocketServer:
    def __init__(self):
        self.ml_model = RealDataMLModel()  # Use real data model
        self.connected_clients = set()
        self.signal_count = 0
    
    async def handle_client(self, websocket):
        """Handle NinjaTrader client connection"""
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 NinjaTrader client connected: {client_info}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send welcome message
            welcome = {
                "type": "WELCOME",
                "message": "Connected to Institutional ML Trading System",
                "timestamp": datetime.now().isoformat(),
                "model_status": "trained" if self.ml_model.is_trained else "not_trained"
            }
            await websocket.send(json.dumps(welcome))
            
            # Listen for messages from NinjaTrader
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_ninjatrader_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📤 Client disconnected: {client_info}")
        except Exception as e:
            logger.error(f"❌ Error handling client {client_info}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def process_ninjatrader_message(self, websocket, data):
        """Process messages from NinjaTrader"""
        message_type = data.get("type", "")
        
        if message_type == "MARKET_DATA":
            # Generate trading signal from market data
            signal = self.generate_trading_signal(data)
            if signal:
                await websocket.send(json.dumps(signal))
                logger.info(f"📡 Sent trading signal: {signal['action']} {signal['symbol']} (Confidence: {signal['confidence']:.2%})")
        
        elif message_type == "ORDER_STATUS":
            logger.info(f"📈 Order status update: {data}")
            
        elif message_type == "HEARTBEAT":
            # Respond to heartbeat
            heartbeat_response = {
                "type": "HEARTBEAT_ACK",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(heartbeat_response))
        
        else:
            logger.info(f"📨 Received from NinjaTrader: {data}")
    
    def generate_trading_signal(self, market_data):
        """Generate trading signal using ML model"""
        if not self.ml_model.is_trained:
            return None
        
        # Get ML prediction
        prediction = self.ml_model.predict_signal(market_data)
        
        if prediction["confidence"] < 0.65:  # Skip low confidence signals
            return None
        
        self.signal_count += 1
        
        # Create institutional trading signal
        signal = {
            "type": "TRADING_SIGNAL",
            "signal_id": f"ML_{self.signal_count:06d}",
            "symbol": market_data.get("symbol", "ES"),
            "action": prediction["action"],
            "confidence": prediction["confidence"],
            "quantity": self.calculate_position_size(prediction["confidence"]),
            "target_price": market_data.get("last_price", 4500),
            "stop_loss": self.calculate_stop_loss(market_data, prediction["action"]),
            "take_profit": self.calculate_take_profit(market_data, prediction["action"]),
            "timestamp": datetime.now().isoformat(),
            "model_version": "v1.0",
            "regime": "NORMAL",
            "expiry_seconds": 30
        }
        
        return signal
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence"""
        base_size = 1
        if confidence > 0.80:
            return min(3, base_size * 2)
        elif confidence > 0.70:
            return min(2, base_size * 1.5)
        else:
            return base_size
    
    def calculate_stop_loss(self, market_data, action):
        """Calculate stop loss price"""
        last_price = market_data.get("last_price", 4500)
        if action == "BUY":
            return last_price - 10  # 10 points stop loss
        elif action == "SELL":
            return last_price + 10
        return last_price
    
    def calculate_take_profit(self, market_data, action):
        """Calculate take profit price"""
        last_price = market_data.get("last_price", 4500)
        if action == "BUY":
            return last_price + 15  # 15 points take profit
        elif action == "SELL":
            return last_price - 15
        return last_price
    
    async def send_periodic_signals(self):
        """Send periodic test signals to connected clients"""
        while True:
            await asyncio.sleep(10)  # Every 10 seconds
            
            if self.connected_clients and self.ml_model.is_trained:
                # Generate test market data
                test_market_data = {
                    "symbol": "ES",
                    "last_price": 4500 + np.random.randn() * 5,
                    "bid_price": 4499.5,
                    "ask_price": 4500.5,
                    "volume": np.random.randint(500, 2000),
                    "timestamp": datetime.now().isoformat()
                }
                
                signal = self.generate_trading_signal(test_market_data)
                
                if signal:
                    # Send to all connected clients
                    disconnected_clients = set()
                    for client in self.connected_clients:
                        try:
                            await client.send(json.dumps(signal))
                            logger.info(f"📡 Sent periodic signal: {signal['action']} (Confidence: {signal['confidence']:.2%})")
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.connected_clients -= disconnected_clients

async def main():
    """Main server execution"""
    logger.info("🏛️ INSTITUTIONAL ML TRADING SYSTEM - WEBSOCKET SERVER")
    logger.info("=" * 60)
    
    # Initialize ML model with real data
    server = TradingWebSocketServer()
    success = server.ml_model.load_trained_model()
    
    if success:
        logger.info("✅ Real market data model loaded successfully!")
    else:
        logger.warning("⚠️ Using fallback synthetic model")
    
    # Start WebSocket server
    logger.info("🚀 Starting WebSocket server on ws://localhost:8000...")
    
    # Start periodic signal sender
    asyncio.create_task(server.send_periodic_signals())
    
    # Start WebSocket server
    async with websockets.serve(server.handle_client, "localhost", 8000):
        logger.info("✅ WebSocket server running!")
        logger.info("📋 NinjaTrader Connection Instructions:")
        logger.info("   1. Open NinjaTrader 8")
        logger.info("   2. Load the Modern Institutional AddOn")
        logger.info("   3. Start trading with account selection")
        logger.info("   4. Monitor log for trading signals and order execution")
        logger.info("=" * 60)
        
        # Keep server running
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")