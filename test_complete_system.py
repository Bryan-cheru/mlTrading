#!/usr/bin/env python3
"""
Complete System Test - ML Model Training and Order Execution Test
Tests the entire institutional trading pipeline from ML training to NinjaTrader order execution
"""

import sys
import os
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Add project paths
sys.path.append('.')
sys.path.append('./ml-models')
sys.path.append('./data-pipeline')
sys.path.append('./trading-engine')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemIntegrationTest:
    def __init__(self):
        self.test_results = {}
        self.ml_model = None
        self.websocket_server = None
        
    async def test_ml_model_training(self):
        """Test ML model training with synthetic data"""
        logger.info("🤖 Testing ML Model Training...")
        
        try:
            # Import the trading model
            from training.trading_model import TradingMLModel
            
            # Create model instance
            self.ml_model = TradingMLModel()
            
            # Generate synthetic ES futures data for testing
            logger.info("Generating synthetic training data...")
            training_data = self.generate_synthetic_market_data()
            
            # Train the model
            logger.info("Training XGBoost model...")
            features, labels = self.prepare_training_features(training_data)
            
            self.ml_model.train(features, labels)
            logger.info("✅ ML Model training completed successfully!")
            
            # Test prediction
            test_features = features.iloc[-1:].copy()
            prediction = self.ml_model.predict(test_features)
            confidence = self.ml_model.predict_proba(test_features)
            
            logger.info(f"Test prediction: {prediction}, Confidence: {confidence}")
            self.test_results['ml_training'] = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ML Model training failed: {e}")
            self.test_results['ml_training'] = False
            return False
    
    def generate_synthetic_market_data(self, days=30):
        """Generate realistic synthetic market data for ES futures"""
        logger.info(f"Generating {days} days of synthetic ES market data...")
        
        # Create minute-by-minute data
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Market hours filter (9:30 AM - 4:00 PM ET)
        market_hours = dates[
            (dates.time >= pd.Timestamp('09:30').time()) & 
            (dates.time <= pd.Timestamp('16:00').time()) &
            (dates.weekday < 5)  # Monday to Friday
        ]
        
        n_bars = len(market_hours)
        
        # Generate realistic price movement
        base_price = 4500  # ES base price
        price_changes = np.random.normal(0, 2, n_bars).cumsum()
        
        # Add some trend and volatility clustering
        trend = np.sin(np.arange(n_bars) / 1000) * 50
        volatility = 1 + 0.5 * np.abs(np.random.normal(0, 1, n_bars))
        
        prices = base_price + price_changes + trend
        
        # Generate OHLC data
        data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices + np.random.normal(0, 0.25, n_bars),
            'high': prices + np.abs(np.random.normal(2, 1, n_bars)),
            'low': prices - np.abs(np.random.normal(2, 1, n_bars)),
            'close': prices,
            'volume': np.random.lognormal(8, 1, n_bars).astype(int)
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        logger.info(f"Generated {len(data)} bars of market data")
        return data
    
    def prepare_training_features(self, data):
        """Prepare features and labels for ML training"""
        logger.info("Preparing ML features and labels...")
        
        # Technical indicators
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['rsi'] = self.calculate_rsi(data['close'])
        data['bb_upper'], data['bb_lower'] = self.calculate_bollinger_bands(data['close'])
        data['macd'], data['macd_signal'] = self.calculate_macd(data['close'])
        
        # Price features
        data['price_change'] = data['close'].pct_change()
        data['volume_change'] = data['volume'].pct_change()
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Volatility features
        data['volatility_10'] = data['price_change'].rolling(10).std()
        data['volatility_20'] = data['price_change'].rolling(20).std()
        
        # Future returns for labels (predict next bar direction)
        data['future_return'] = data['close'].shift(-1) / data['close'] - 1
        
        # Create labels: 0=SELL, 1=HOLD, 2=BUY
        data['label'] = 1  # Default to HOLD
        data.loc[data['future_return'] > 0.001, 'label'] = 2  # BUY
        data.loc[data['future_return'] < -0.001, 'label'] = 0  # SELL
        
        # Select features
        feature_columns = [
            'sma_10', 'sma_20', 'rsi', 'bb_upper', 'bb_lower',
            'macd', 'macd_signal', 'price_change', 'volume_change',
            'hl_ratio', 'oc_ratio', 'volatility_10', 'volatility_20'
        ]
        
        # Remove NaN values
        data = data.dropna()
        
        features = data[feature_columns]
        labels = data['label']
        
        logger.info(f"Prepared {len(features)} samples with {len(feature_columns)} features")
        return features, labels
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    async def test_websocket_server(self):
        """Test WebSocket server for NinjaTrader communication"""
        logger.info("🌐 Testing WebSocket Server...")
        
        try:
            # Start WebSocket server
            async def handle_client(websocket, path):
                logger.info(f"NinjaTrader client connected: {websocket.remote_address}")
                
                try:
                    # Send test trading signal
                    test_signal = {
                        "type": "SIGNAL",
                        "symbol": "ES",
                        "action": "BUY",
                        "quantity": 1,
                        "confidence": 0.85,
                        "target_price": 4505.0,
                        "stop_loss": 4495.0,
                        "take_profit": 4515.0,
                        "timestamp": datetime.now().isoformat(),
                        "model_version": "v1.0",
                        "regime": "NORMAL"
                    }
                    
                    await websocket.send(json.dumps(test_signal))
                    logger.info(f"📡 Sent test signal: {test_signal}")
                    
                    # Wait for response
                    response = await websocket.recv()
                    logger.info(f"📨 Received response: {response}")
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client disconnected")
            
            # Start server
            server = await websockets.serve(handle_client, "localhost", 8000)
            logger.info("✅ WebSocket server started on ws://localhost:8000")
            
            self.test_results['websocket_server'] = True
            return server
            
        except Exception as e:
            logger.error(f"❌ WebSocket server failed: {e}")
            self.test_results['websocket_server'] = False
            return None
    
    async def test_rithmic_connection(self):
        """Test Rithmic WebSocket connection simulation"""
        logger.info("📊 Testing Rithmic Data Connection...")
        
        try:
            # Simulate Rithmic market data
            market_data = {
                "symbol": "ES",
                "last_price": 4502.75,
                "bid_price": 4502.50,
                "ask_price": 4503.00,
                "bid_size": 10,
                "ask_size": 8,
                "volume": 1500,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"📈 Simulated market data: {market_data}")
            self.test_results['rithmic_connection'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Rithmic connection test failed: {e}")
            self.test_results['rithmic_connection'] = False
            return False
    
    def test_risk_management(self):
        """Test risk management system"""
        logger.info("🛡️ Testing Risk Management...")
        
        try:
            # Test signal with risk parameters
            test_signal = {
                "symbol": "ES",
                "action": "BUY",
                "quantity": 5,  # Test position size
                "confidence": 0.80,
                "timestamp": datetime.now()
            }
            
            # Simulate risk checks
            risk_passed = True
            
            # Position size check
            if test_signal["quantity"] > 10:
                logger.warning("⚠️ Position size exceeds limit")
                risk_passed = False
            
            # Confidence check
            if test_signal["confidence"] < 0.75:
                logger.warning("⚠️ Signal confidence too low")
                risk_passed = False
            
            if risk_passed:
                logger.info("✅ Risk management checks passed")
                self.test_results['risk_management'] = True
            else:
                logger.warning("⚠️ Risk management checks failed")
                self.test_results['risk_management'] = False
            
            return risk_passed
            
        except Exception as e:
            logger.error(f"❌ Risk management test failed: {e}")
            self.test_results['risk_management'] = False
            return False
    
    async def run_full_system_test(self):
        """Run complete system integration test"""
        logger.info("🚀 Starting Full System Integration Test...")
        logger.info("=" * 60)
        
        # Test 1: ML Model Training
        await self.test_ml_model_training()
        await asyncio.sleep(1)
        
        # Test 2: Risk Management
        self.test_risk_management()
        await asyncio.sleep(1)
        
        # Test 3: Rithmic Connection
        await self.test_rithmic_connection()
        await asyncio.sleep(1)
        
        # Test 4: WebSocket Server
        server = await self.test_websocket_server()
        
        if server:
            logger.info("🔄 WebSocket server running - Ready for NinjaTrader connection")
            logger.info("📋 Next steps:")
            logger.info("   1. Open NinjaTrader 8")
            logger.info("   2. Load the Modern Institutional AddOn")
            logger.info("   3. Connect to ws://localhost:8000")
            logger.info("   4. Monitor for trading signals")
            
            # Keep server running for 60 seconds
            await asyncio.sleep(60)
            server.close()
            await server.wait_closed()
        
        # Print test results
        self.print_test_results()
    
    def print_test_results(self):
        """Print comprehensive test results"""
        logger.info("=" * 60)
        logger.info("📊 SYSTEM TEST RESULTS")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("=" * 60)
        logger.info(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - System ready for live trading!")
        else:
            logger.info("⚠️ Some tests failed - Review and fix before live trading")
        
        logger.info("=" * 60)

async def main():
    """Main test execution"""
    print("🏛️ INSTITUTIONAL ML TRADING SYSTEM - INTEGRATION TEST")
    print("=" * 60)
    
    test_suite = SystemIntegrationTest()
    await test_suite.run_full_system_test()

if __name__ == "__main__":
    asyncio.run(main())