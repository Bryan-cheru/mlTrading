"""
Standalone Demo System - No NinjaTrader Required
Demonstrates the complete ML trading system with synthetic data
"""

import time
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticMarketData:
    """Generate realistic synthetic market data"""
    
    def __init__(self, symbol: str, initial_price: float):
        self.symbol = symbol
        self.price = initial_price
        self.trend = 0.0
        self.volatility = 0.02
        
    def generate_tick(self):
        """Generate next market tick"""
        # Random walk with trend
        change = np.random.normal(self.trend, self.volatility)
        self.price *= (1 + change/100)
        
        # Add some trend shifts
        if random.random() < 0.01:  # 1% chance of trend change
            self.trend = np.random.normal(0, 0.5)
        
        volume = random.randint(50, 500)
        spread = self.price * 0.0001  # 1 tick spread
        
        return {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'price': round(self.price, 2),
            'bid': round(self.price - spread/2, 2),
            'ask': round(self.price + spread/2, 2),
            'volume': volume
        }

class StandaloneFeatureEngine:
    """Feature engineering for demo system"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.price_history = {}
        self.features_cache = {}
    
    def update(self, symbol: str, tick: Dict):
        """Update with new market tick"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': tick['price'],
            'volume': tick['volume'],
            'timestamp': tick['timestamp']
        })
        
        # Keep only recent data
        if len(self.price_history[symbol]) > self.window_size * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.window_size:]
        
        # Calculate features
        self._calculate_features(symbol)
    
    def _calculate_features(self, symbol: str):
        """Calculate technical features"""
        data = self.price_history[symbol]
        if len(data) < 10:
            return
        
        prices = [d['price'] for d in data]
        volumes = [d['volume'] for d in data]
        
        features = {}
        
        # Price features
        features['price'] = prices[-1]
        features['sma_10'] = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # Returns
        if len(prices) > 1:
            features['return_1'] = (prices[-1] - prices[-2]) / prices[-2]
        else:
            features['return_1'] = 0
        
        # Volatility
        if len(prices) > 10:
            returns = np.diff(prices[-10:]) / np.array(prices[-10:-1])
            features['volatility'] = np.std(returns)
        else:
            features['volatility'] = 0
        
        # Volume features
        features['volume'] = volumes[-1]
        features['volume_sma'] = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        
        # Momentum
        if len(prices) >= 5:
            features['momentum'] = (prices[-1] - prices[-5]) / prices[-5]
        else:
            features['momentum'] = 0
        
        # Trend strength
        if len(prices) >= 20:
            slope = np.polyfit(range(20), prices[-20:], 1)[0]
            features['trend_strength'] = slope / prices[-1]
        else:
            features['trend_strength'] = 0
        
        # RSI approximation
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, 15):
                change = prices[-i] - prices[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-10)
            features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50
        
        self.features_cache[symbol] = features
    
    def get_features(self, symbol: str) -> Dict:
        """Get current features for symbol"""
        return self.features_cache.get(symbol, {})

class StandaloneMLModel:
    """Simple ML model for demo"""
    
    def __init__(self):
        self.is_trained = False
        self.thresholds = {
            'buy': {'rsi_low': 30, 'momentum_high': 0.001, 'trend_strong': 0.0005},
            'sell': {'rsi_high': 70, 'momentum_low': -0.001, 'trend_weak': -0.0005}
        }
    
    def train(self, features_history: List[Dict]):
        """Train model (simplified)"""
        logger.info("Training ML model on historical features...")
        # In real system, this would train XGBoost model
        self.is_trained = True
        logger.info("Model training completed")
    
    def predict(self, features: Dict) -> Dict:
        """Generate trading signal"""
        if not features or not self.is_trained:
            return {'signal': 0, 'confidence': 0.0}
        
        signal = 0
        confidence = 0.5
        
        # Simple rule-based signals (in real system, use trained model)
        rsi = features.get('rsi', 50)
        momentum = features.get('momentum', 0)
        trend = features.get('trend_strength', 0)
        
        # Buy signals
        if (rsi < 30 and momentum > 0.001 and trend > 0):
            signal = 1
            confidence = min(0.9, 0.6 + abs(trend) * 100)
        
        # Sell signals
        elif (rsi > 70 and momentum < -0.001 and trend < 0):
            signal = -1
            confidence = min(0.9, 0.6 + abs(trend) * 100)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'features_used': len(features)
        }

class StandalonePortfolio:
    """Portfolio management for demo"""
    
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.trades = []
        self.current_prices = {}
    
    def update_price(self, symbol: str, price: float):
        """Update current price"""
        self.current_prices[symbol] = price
    
    def execute_trade(self, symbol: str, signal: int, confidence: float, price: float):
        """Execute trade based on signal"""
        if confidence < 0.6:  # Minimum confidence threshold
            return False
        
        # Calculate position size based on confidence
        max_position_value = self.get_total_equity() * 0.1  # 10% max per position
        base_contracts = int(max_position_value / (price * 50))  # Assuming $50 per point
        position_size = int(base_contracts * confidence)
        
        if signal == 1:  # Buy
            if position_size * price * 50 <= self.cash:
                self.positions[symbol] = self.positions.get(symbol, 0) + position_size
                self.cash -= position_size * price * 50
                
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': position_size,
                    'price': price,
                    'timestamp': datetime.now(),
                    'confidence': confidence
                }
                self.trades.append(trade)
                logger.info(f"🟢 BUY {position_size} {symbol} @ ${price:.2f} (Confidence: {confidence:.1%})")
                return True
        
        elif signal == -1:  # Sell
            current_position = self.positions.get(symbol, 0)
            if position_size <= current_position:
                self.positions[symbol] -= position_size
                self.cash += position_size * price * 50
                
                trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position_size,
                    'price': price,
                    'timestamp': datetime.now(),
                    'confidence': confidence
                }
                self.trades.append(trade)
                logger.info(f"🔴 SELL {position_size} {symbol} @ ${price:.2f} (Confidence: {confidence:.1%})")
                return True
        
        return False
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio value"""
        equity = self.cash
        for symbol, quantity in self.positions.items():
            if symbol in self.current_prices:
                equity += quantity * self.current_prices[symbol] * 50
        return equity
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        total_equity = self.get_total_equity()
        total_pnl = total_equity - self.initial_cash
        
        return {
            'total_equity': total_equity,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'pnl_percentage': (total_pnl / self.initial_cash) * 100,
            'positions': dict(self.positions),
            'num_trades': len(self.trades),
            'num_positions': len([p for p in self.positions.values() if p != 0])
        }

class StandaloneTradingSystem:
    """Complete standalone trading system"""
    
    def __init__(self):
        self.instruments = ['ES', 'NQ', 'YM']
        self.market_data = {}
        self.feature_engine = StandaloneFeatureEngine()
        self.ml_model = StandaloneMLModel()
        self.portfolio = StandalonePortfolio()
        
        # Initialize synthetic data generators
        self.data_generators = {
            'ES': SyntheticMarketData('ES', 4500.0),
            'NQ': SyntheticMarketData('NQ', 15000.0),
            'YM': SyntheticMarketData('YM', 34000.0)
        }
        
        self.running = False
        self.loop_count = 0
        self.signals_generated = 0
        self.trades_executed = 0
        
        logger.info("Standalone ML Trading System initialized")
    
    def initialize(self):
        """Initialize system"""
        logger.info("Initializing trading system...")
        
        # Generate some historical data for training
        historical_features = []
        for _ in range(100):
            for symbol in self.instruments:
                tick = self.data_generators[symbol].generate_tick()
                self.feature_engine.update(symbol, tick)
                features = self.feature_engine.get_features(symbol)
                if features:
                    historical_features.append(features)
        
        # Train model
        self.ml_model.train(historical_features)
        logger.info("System initialization complete")
    
    def process_market_data(self):
        """Process market data for all instruments"""
        for symbol in self.instruments:
            # Generate new market tick
            tick = self.data_generators[symbol].generate_tick()
            
            # Update features
            self.feature_engine.update(symbol, tick)
            self.portfolio.update_price(symbol, tick['price'])
            
            # Get features
            features = self.feature_engine.get_features(symbol)
            if not features:
                continue
            
            # Generate ML prediction
            prediction = self.ml_model.predict(features)
            
            if prediction['signal'] != 0:
                self.signals_generated += 1
                logger.info(f"📊 Signal for {symbol}: {prediction['signal']} (Confidence: {prediction['confidence']:.1%})")
                
                # Execute trade
                success = self.portfolio.execute_trade(
                    symbol, 
                    prediction['signal'], 
                    prediction['confidence'], 
                    tick['price']
                )
                
                if success:
                    self.trades_executed += 1
    
    def print_status(self):
        """Print system status"""
        performance = self.portfolio.get_performance_summary()
        
        print("\n" + "="*80)
        print("📊 TRADING SYSTEM STATUS")
        print("="*80)
        print(f"⏱️  Runtime: {self.loop_count} cycles")
        print(f"💰 Total Equity: ${performance['total_equity']:,.2f}")
        print(f"💵 Cash: ${performance['cash']:,.2f}")
        print(f"📈 P&L: ${performance['total_pnl']:,.2f} ({performance['pnl_percentage']:+.2f}%)")
        print(f"📍 Positions: {performance['num_positions']}")
        print(f"📡 Signals Generated: {self.signals_generated}")
        print(f"✅ Trades Executed: {self.trades_executed}")
        
        if performance['positions']:
            print(f"📋 Current Positions:")
            for symbol, quantity in performance['positions'].items():
                if quantity != 0:
                    price = self.portfolio.current_prices.get(symbol, 0)
                    value = quantity * price * 50
                    print(f"   {symbol}: {quantity} contracts @ ${price:.2f} = ${value:,.2f}")
        
        print("="*80)
    
    async def run(self):
        """Run the trading system"""
        self.initialize()
        
        print("\n" + "="*80)
        print("🚀 INSTITUTIONAL ML TRADING SYSTEM - DEMO MODE")
        print("="*80)
        print("✅ Using synthetic market data")
        print("✅ ML model trained and ready")
        print("✅ Portfolio management active")
        print("✅ Risk controls enabled")
        print("\n⚡ System Features:")
        print("   • Real-time feature engineering")
        print("   • ML-based signal generation")
        print("   • Dynamic position sizing")
        print("   • Performance monitoring")
        print("\n🎯 Demo will run for 100 cycles...")
        print("="*80)
        
        self.running = True
        
        try:
            while self.running and self.loop_count < 100:
                self.loop_count += 1
                
                # Process market data
                self.process_market_data()
                
                # Print status every 20 cycles
                if self.loop_count % 20 == 0:
                    self.print_status()
                
                # Brief pause
                await asyncio.sleep(0.5)
            
            # Final summary
            print("\n🏁 DEMO COMPLETED!")
            self.print_status()
            
            performance = self.portfolio.get_performance_summary()
            print(f"\n📊 FINAL RESULTS:")
            print(f"Return: {performance['pnl_percentage']:+.2f}%")
            print(f"Total Trades: {self.trades_executed}")
            if self.trades_executed > 0:
                win_rate = self.trades_executed / self.signals_generated * 100
                print(f"Signal-to-Trade Rate: {win_rate:.1f}%")
            
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        finally:
            self.running = False

async def main():
    """Main demo function"""
    system = StandaloneTradingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
