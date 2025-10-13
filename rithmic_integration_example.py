"""
Integrate Rithmic with Existing Trading System
Quick integration script to connect Rithmic data to your ML pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing components
from ml_models.training.trading_model import TradingMLModel
from feature_store.realtime.feature_engineering import RealTimeFeatureEngine
from trading_engine.ninjatrader_executor import NinjaTraderExecutor

# Import new Rithmic connector
from data_pipeline.ingestion.rithmic_connector import RithmicConnector

class RithmicMLTradingSystem:
    """
    Enhanced trading system with Rithmic data integration
    Combines your existing ML pipeline with professional market data
    """
    
    def __init__(self):
        # Initialize components
        self.rithmic = RithmicConnector()
        self.ml_model = TradingMLModel()
        self.feature_engine = RealTimeFeatureEngine()
        self.executor = NinjaTraderExecutor()
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        
        print("🚀 Enhanced ES Trading System with Rithmic Data")
    
    async def start_trading(self):
        """Start the enhanced trading system"""
        
        # Connect to Rithmic
        if await self.rithmic.connect():
            print("✅ Connected to Rithmic")
            
            # Subscribe to ES data
            await self.rithmic.subscribe_market_data(['ESZ4'])
            
            # Register callback for real-time processing
            self.rithmic.register_tick_callback(self.process_tick)
            
            print("📊 Receiving real-time ES data...")
            print("🤖 ML model ready for signal generation")
            
        else:
            print("❌ Failed to connect to Rithmic")
    
    def process_tick(self, tick_data):
        """Process each tick for trading signals"""
        try:
            # Extract market microstructure features
            features = self.rithmic.get_market_features(
                tick_data['symbol'], 
                lookback_ticks=100
            )
            
            # Enhanced features with Rithmic data
            enhanced_features = {
                **features,
                'current_price': tick_data['price'],
                'volume': tick_data['size'],
                'spread': tick_data['ask'] - tick_data['bid'],
                'timestamp': tick_data['timestamp']
            }
            
            # Generate ML prediction
            prediction = self.ml_model.predict(enhanced_features)
            
            # Check signal strength
            if prediction['confidence'] > 0.75:  # High confidence only
                self.execute_signal(prediction, tick_data)
            
            self.signals_generated += 1
            
        except Exception as e:
            print(f"❌ Error processing tick: {e}")
    
    def execute_signal(self, prediction, tick_data):
        """Execute trading signal"""
        try:
            signal_type = prediction['signal']
            confidence = prediction['confidence']
            current_price = tick_data['price']
            
            if signal_type in ['BUY', 'SELL']:
                # Execute through NinjaTrader
                order_result = self.executor.submit_order(
                    symbol='ESZ4',
                    action=signal_type,
                    quantity=1,  # Start small
                    price=current_price,
                    order_type='LIMIT'
                )
                
                if order_result:
                    self.trades_executed += 1
                    print(f"🎯 {signal_type} signal executed: ${current_price:.2f} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error executing signal: {e}")
    
    def get_performance_stats(self):
        """Get trading performance statistics"""
        return {
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'execution_rate': self.trades_executed / max(1, self.signals_generated),
            'rithmic_stats': self.rithmic.get_performance_stats()
        }

# Quick start function
async def start_enhanced_trading():
    """Start the enhanced trading system"""
    system = RithmicMLTradingSystem()
    await system.start_trading()
    
    # Run for demonstration
    import asyncio
    await asyncio.sleep(60)  # Run for 1 minute
    
    # Show stats
    stats = system.get_performance_stats()
    print(f"\n📈 Performance Summary:")
    print(f"Signals Generated: {stats['signals_generated']}")
    print(f"Trades Executed: {stats['trades_executed']}")
    print(f"Execution Rate: {stats['execution_rate']:.2%}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_enhanced_trading())