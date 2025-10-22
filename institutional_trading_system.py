"""
Mathematical ML Trading System - Clean Implementation
Single consolidated system using mathematical functions instead of technical indicators
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import logging

from data_pipeline.ingestion.rithmic_connector import RithmicConnector
from data_pipeline.ingestion.ninjatrader_connector import NinjaTraderConnector  
from ml_models.training.trading_model import TradingMLModel
from feature_store.realtime.mathematical_features import MathematicalFeatureEngine, MarketData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mathematical_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MathematicalMLTradingSystem:
    """
    Consolidated mathematical ML trading system
    Uses mathematical functions instead of technical indicators
    """
    
    def __init__(self):
        # Core components
        self.rithmic_connector = RithmicConnector()
        self.ninjatrader_connector = NinjaTraderConnector()
        self.ml_model = TradingMLModel()
        self.mathematical_engine = MathematicalFeatureEngine(lookback_periods=300)
        
        # System state
        self.is_running = False
        self.is_trading_enabled = True
        self.daily_pnl = 0.0
        self.total_signals = 0
        self.executed_trades = 0
        
        logger.info("🧮 Mathematical ML Trading System initialized")
    
    async def start_system(self):
        """Start the complete trading system"""
        try:
            logger.info("🚀 Starting Mathematical ML Trading System")
            
            # Connect to data sources
            await self.rithmic_connector.connect()
            await self.ninjatrader_connector.connect()
            
            # Subscribe to ES futures data
            await self.rithmic_connector.subscribe_market_data('ESZ4', self.process_market_data)
            
            self.is_running = True
            logger.info("✅ System started successfully")
            
            # Main trading loop
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            await self.shutdown()
    
    async def process_market_data(self, tick_data):
        """Process market data through mathematical feature generation"""
        try:
            # Convert to MarketData format
            market_data = MarketData(
                timestamp=pd.Timestamp(tick_data['timestamp']),
                open=tick_data.get('open', tick_data['price']),
                high=tick_data.get('high', tick_data['price']),
                low=tick_data.get('low', tick_data['price']),
                close=tick_data['price'],
                volume=tick_data.get('volume', 0)
            )
            
            # Generate mathematical features
            feature_set = self.mathematical_engine.generate_features(market_data)
            
            if len(feature_set.features) < 20:
                return
            
            # Generate ML prediction using mathematical features
            ml_features = self.ml_model.prepare_features(feature_set.features)
            prediction = self.ml_model.predict(ml_features)
            
            # Mathematical signal validation
            validated_signal = self.validate_signal_mathematically(prediction, feature_set.features)
            
            if validated_signal['confidence'] > 0.70:
                await self.execute_trade(validated_signal, feature_set.features)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def validate_signal_mathematically(self, ml_prediction, mathematical_features):
        """Validate ML prediction using mathematical criteria"""
        
        signal = {
            'action': ml_prediction.get('action', 'HOLD'),
            'confidence': ml_prediction.get('confidence', 0.0),
            'mathematical_validation': {}
        }
        
        # Z-score validation (replace RSI validation)
        z_score_20 = mathematical_features.get('z_score_price_20', 0)
        if abs(z_score_20) > 2.0:  # Statistically significant
            signal['mathematical_validation']['significant_zscore'] = True
            signal['confidence'] *= 1.2
        
        # VaR-based risk assessment
        var_95 = mathematical_features.get('var_95', 0)
        if abs(var_95) > 0.03:  # High risk regime
            signal['confidence'] *= 0.7
            signal['mathematical_validation']['high_risk'] = True
        
        # Entropy-based uncertainty filter
        shannon_entropy = mathematical_features.get('shannon_entropy', 0)
        if shannon_entropy > 3.5:  # High uncertainty
            signal['confidence'] *= 0.8
            signal['mathematical_validation']['high_uncertainty'] = True
        
        # Kelly criterion validation
        kelly_fraction = mathematical_features.get('kelly_fraction', 0)
        if abs(kelly_fraction) > 0.1:
            signal['mathematical_validation']['favorable_kelly'] = True
            signal['confidence'] *= 1.1
        
        # Final confidence adjustment
        signal['confidence'] = min(signal['confidence'], 1.0)
        
        return signal
    
    async def execute_trade(self, signal, mathematical_features):
        """Execute trade through NinjaTrader"""
        try:
            logger.info(f"🎯 Executing Mathematical Signal:")
            logger.info(f"   Action: {signal['action']}")
            logger.info(f"   Confidence: {signal['confidence']:.3f}")
            logger.info(f"   Validations: {signal['mathematical_validation']}")
            
            # Calculate position size using Kelly criterion
            kelly_fraction = mathematical_features.get('kelly_fraction', 0)
            base_size = 1
            position_size = max(1, min(5, int(base_size * abs(kelly_fraction) * 10))) if kelly_fraction != 0 else base_size
            
            # Execute through NinjaTrader
            order_result = await self.ninjatrader_connector.place_order(
                symbol='ESZ4',
                action=signal['action'],
                quantity=position_size,
                order_type='MARKET'
            )
            
            if order_result['status'] == 'filled':
                self.executed_trades += 1
                logger.info(f"✅ Trade executed: {order_result}")
            
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            logger.info("🔄 Shutting down system...")
            self.is_running = False
            
            if self.rithmic_connector:
                await self.rithmic_connector.disconnect()
            
            if self.ninjatrader_connector:
                await self.ninjatrader_connector.disconnect()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

async def main():
    """Main entry point"""
    system = MathematicalMLTradingSystem()
    
    try:
        await system.start_system()
    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown requested")
    except Exception as e:
        logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())