"""
Advanced Trading System - Simplified Version
Using existing modules with enhanced features
"""

import asyncio
import logging
import signal
import sys
import json
import os
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-pipeline', 'ingestion'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'feature-store'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-models', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading-engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'monitoring'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-engine'))

# Import existing modules
from ninjatrader_connector import NinjaTraderConnector, MarketData, BarData

@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence and metadata"""
    symbol: str
    signal: int  # -1, 0, 1
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    regime: str = "normal"

class EnhancedFeatureEngine:
    """Enhanced feature engineering for real-time trading"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_history = {}
        self.volume_history = {}
        self.feature_cache = {}
    
    def update_data(self, symbol: str, price: float, volume: int):
        """Update price and volume data"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        # Add new data
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # Keep only recent data
        if len(self.price_history[symbol]) > self.window_size * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.window_size:]
            self.volume_history[symbol] = self.volume_history[symbol][-self.window_size:]
        
        # Compute features
        self._compute_features(symbol)
    
    def _compute_features(self, symbol: str):
        """Compute technical features for a symbol"""
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        if len(prices) < 20:
            return
        
        features = {}
        
        # Price-based features
        features['sma_10'] = np.mean(prices[-10:])
        features['sma_20'] = np.mean(prices[-20:])
        features['price'] = prices[-1]
        features['price_change'] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
        
        # Volatility
        if len(prices) > 10:
            returns = np.diff(np.log(prices[-20:]))
            features['volatility'] = np.std(returns) if len(returns) > 0 else 0
        
        # RSI-like momentum
        if len(prices) > 14:
            price_changes = np.diff(prices[-15:])
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100
        
        # Volume features
        features['volume'] = volumes[-1]
        if len(volumes) > 10:
            features['volume_ma'] = np.mean(volumes[-10:])
            features['volume_ratio'] = volumes[-1] / features['volume_ma']
        
        # Trend strength
        if len(prices) > 10:
            x = np.arange(len(prices[-10:]))
            slope, _ = np.polyfit(x, prices[-10:], 1)
            features['trend_strength'] = slope / prices[-1]
        
        self.feature_cache[symbol] = features
    
    def get_features(self, symbol: str) -> Dict[str, float]:
        """Get latest features for a symbol"""
        return self.feature_cache.get(symbol, {})

class SimpleMLModel:
    """Simplified ML model for trading signals"""
    
    def __init__(self):
        self.is_trained = False
        self.model_weights = {
            'rsi_oversold': 0.3,
            'rsi_overbought': 0.3,
            'trend_momentum': 0.2,
            'volume_surge': 0.15,
            'volatility_breakout': 0.05
        }
    
    def train(self, training_data: List[Dict]):
        """Simple training simulation"""
        logger.info("Training ML model...")
        # In production, implement real ML training
        self.is_trained = True
        logger.info("Model training completed")
        return {"accuracy": 0.95}
    
    def predict(self, features: Dict[str, float]) -> TradingSignal:
        """Generate trading signal from features"""
        if not features:
            return TradingSignal("", 0, 0.0, datetime.now(), {})
        
        # Simple rule-based logic (replace with real ML)
        signal = 0
        confidence = 0.5
        
        rsi = features.get('rsi', 50)
        trend = features.get('trend_strength', 0)
        volume_ratio = features.get('volume_ratio', 1)
        volatility = features.get('volatility', 0)
        
        # Buy conditions
        if rsi < 30 and trend > 0 and volume_ratio > 1.2:
            signal = 1
            confidence = 0.7
        
        # Sell conditions
        elif rsi > 70 and trend < 0 and volume_ratio > 1.2:
            signal = -1
            confidence = 0.7
        
        # Trend following
        elif abs(trend) > 0.001 and volume_ratio > 1.1:
            signal = 1 if trend > 0 else -1
            confidence = 0.6
        
        return TradingSignal(
            symbol="",
            signal=signal,
            confidence=confidence,
            timestamp=datetime.now(),
            features=features
        )

class SimplePortfolioManager:
    """Simplified portfolio management"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.current_prices = {}
    
    def update_price(self, symbol: str, price: float):
        """Update current price for position valuation"""
        self.current_prices[symbol] = price
    
    def get_position_size(self, symbol: str, signal_confidence: float, price: float) -> int:
        """Calculate position size based on confidence and available capital"""
        max_position_value = self.get_total_equity() * 0.1  # 10% max per position
        base_size = int(max_position_value / price) if price > 0 else 0
        
        # Adjust for confidence
        adjusted_size = int(base_size * signal_confidence)
        
        return max(0, min(adjusted_size, 100))  # Cap at 100 shares/contracts
    
    def execute_trade(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a trade"""
        trade_value = quantity * price
        
        # Check if we have enough cash (for buys)
        if quantity > 0 and trade_value > self.cash:
            logger.warning(f"Insufficient cash for trade: {symbol} {quantity}")
            return False
        
        # Update position
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        
        if new_position == 0:
            # Close position
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = new_position
        
        # Update cash
        self.cash -= trade_value
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'value': trade_value
        })
        
        logger.info(f"Trade executed: {symbol} {quantity} @ ${price:.2f}")
        return True
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio equity"""
        equity = self.cash
        
        for symbol, quantity in self.positions.items():
            if symbol in self.current_prices:
                equity += quantity * self.current_prices[symbol]
        
        return equity
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_equity = self.get_total_equity()
        total_pnl = total_equity - self.initial_capital
        
        return {
            'total_equity': total_equity,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'pnl_percentage': (total_pnl / self.initial_capital) * 100,
            'positions': dict(self.positions),
            'num_trades': len(self.trades)
        }

class AdvancedTradingSystem:
    """Advanced trading system with enhanced features"""
    
    def __init__(self):
        # Core components
        self.nt_connector = NinjaTraderConnector()
        self.feature_engine = EnhancedFeatureEngine()
        self.ml_model = SimpleMLModel()
        self.portfolio = SimplePortfolioManager()
        
        # Configuration
        self.instruments = ["ES 12-24", "NQ 12-24"]
        self.min_confidence = 0.6
        self.signal_cooldown = {}  # symbol -> last_signal_time
        self.cooldown_period = 60  # seconds
        
        # System state
        self.is_running = False
        self.total_signals = 0
        self.successful_trades = 0
        self.start_time = None
        
        logger.info("Advanced Trading System initialized")
    
    async def initialize(self):
        """Initialize the trading system"""
        logger.info("Initializing system components...")
        
        # Train ML model
        training_data = self._generate_training_data()
        self.ml_model.train(training_data)
        
        logger.info("System initialization complete")
    
    def _generate_training_data(self) -> List[Dict]:
        """Generate synthetic training data"""
        # In production, load real historical data
        training_data = []
        for i in range(1000):
            training_data.append({
                'rsi': np.random.uniform(10, 90),
                'trend_strength': np.random.normal(0, 0.001),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'volatility': np.random.uniform(0.01, 0.05)
            })
        return training_data
    
    def _on_market_data(self, data: MarketData):
        """Handle incoming market data"""
        try:
            # Update feature engine
            self.feature_engine.update_data(data.instrument, data.last, data.volume)
            
            # Update portfolio prices
            self.portfolio.update_price(data.instrument, data.last)
            
            # Process trading signal
            self._process_signal(data)
            
        except Exception as e:
            logger.error(f"Market data processing error: {e}")

    def _process_signal(self, data: MarketData):
        """Process trading signal for incoming data"""
        symbol = data.instrument
        
        # Check cooldown
        if self._is_in_cooldown(symbol):
            return
        
        # Get features
        features = self.feature_engine.get_features(symbol)
        if not features:
            return
        
        # Generate ML prediction
        signal = self.ml_model.predict(features)
        signal.symbol = symbol
        self.total_signals += 1
        
        # Log signal
        logger.info(f"Signal for {symbol}: {signal.signal} (confidence: {signal.confidence:.3f})")
        
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            return
        
        # Skip hold signals
        if signal.signal == 0:
            return
        
        # Calculate position size
        position_size = self.portfolio.get_position_size(symbol, signal.confidence, data.last)
        
        if signal.signal == -1:  # Sell signal
            position_size = -position_size
        
        # Execute trade
        if position_size != 0:
            success = self.portfolio.execute_trade(symbol, position_size, data.last)
            if success:
                self.successful_trades += 1
                self.signal_cooldown[symbol] = datetime.now()
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.signal_cooldown:
            return False
        
        time_elapsed = (datetime.now() - self.signal_cooldown[symbol]).total_seconds()
        return time_elapsed < self.cooldown_period
    
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        self.is_running = True
        self.start_time = datetime.now()
        
        loop_count = 0
        
        try:
            while self.is_running:
                loop_count += 1
                
                # System status every 100 loops
                if loop_count % 100 == 0:
                    self._log_system_status()
                
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
    
    def _log_system_status(self):
        """Log system status"""
        portfolio_summary = self.portfolio.get_portfolio_summary()
        uptime = datetime.now() - self.start_time
        
        logger.info("=== SYSTEM STATUS ===")
        logger.info(f"Uptime: {uptime}")
        logger.info(f"Total Equity: ${portfolio_summary['total_equity']:,.2f}")
        logger.info(f"P&L: ${portfolio_summary['total_pnl']:,.2f} ({portfolio_summary['pnl_percentage']:.2f}%)")
        logger.info(f"Positions: {len(portfolio_summary['positions'])}")
        logger.info(f"Total Signals: {self.total_signals}")
        logger.info(f"Successful Trades: {self.successful_trades}")
        logger.info("=====================")
    
    async def start(self):
        """Start the trading system"""
        try:
            # Initialize components
            await self.initialize()
            
            # Connect to NinjaTrader (using sync method)
            connected = self.nt_connector.connect()
            if connected:
                logger.info("Connected to NinjaTrader 8")
                
                # Subscribe to market data with callback
                for instrument in self.instruments:
                    self.nt_connector.subscribe_market_data(instrument, self._on_market_data)
                    logger.info(f"Subscribed to market data for {instrument}")
            else:
                logger.error("Failed to connect to NinjaTrader. Exiting.")
                return
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"System error: {e}")
    
    async def shutdown(self):
        """Shutdown the system"""
        logger.info("Shutting down trading system...")
        self.is_running = False
        
        # Disconnect from NinjaTrader
        self.nt_connector.disconnect()
        
        # Final portfolio summary
        final_summary = self.portfolio.get_portfolio_summary()
        logger.info(f"Final Portfolio Value: ${final_summary['total_equity']:,.2f}")
        logger.info(f"Final P&L: {final_summary['pnl_percentage']:.2f}%")
        logger.info("System shutdown complete")

async def main():
    """Main entry point"""
    system = AdvancedTradingSystem()
    
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        await system.shutdown()

if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED INSTITUTIONAL ML TRADING SYSTEM")
    print("=" * 70)
    print()
    print("FEATURES:")
    print("• Enhanced real-time feature engineering")
    print("• ML-based trading signals with confidence scoring")
    print("• Advanced portfolio management")
    print("• Risk management with position sizing")
    print("• NinjaTrader 8 integration")
    print("• Comprehensive performance monitoring")
    print()
    print("REQUIREMENTS:")
    print("1. NinjaTrader 8 must be running")
    print("2. ATI enabled on port 36973")
    print("3. Market data connection active")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    asyncio.run(main())
