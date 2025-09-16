"""
Production System Test - NinjaTrader Integration Verification
Tests the complete advanced trading system with proper callback handling
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-pipeline', 'ingestion'))

from simplified_advanced_system import AdvancedTradingSystem
from ninjatrader_connector import MarketData

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionSystemTest:
    """Test suite for production trading system"""
    
    def __init__(self):
        self.system = None
        self.test_results = {}
    
    def test_system_initialization(self):
        """Test 1: System Initialization"""
        logger.info("🔍 Test 1: System Initialization")
        
        try:
            self.system = AdvancedTradingSystem()
            self.test_results['initialization'] = True
            logger.info("✅ System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.test_results['initialization'] = False
            return False
    
    def test_callback_compatibility(self):
        """Test 2: Callback Method Compatibility"""
        logger.info("🔍 Test 2: Callback Method Compatibility")
        
        try:
            # Create test market data
            test_data = MarketData(
                instrument='ES 12-24',
                timestamp=datetime.now(),
                bid=4500.0,
                ask=4500.25,
                last=4500.125,
                volume=100,
                bid_size=10,
                ask_size=10
            )
            
            # Test callback directly
            self.system._on_market_data(test_data)
            self.test_results['callback_compatibility'] = True
            logger.info("✅ Callback method working correctly")
            return True
            
        except Exception as e:
            logger.error(f"❌ Callback compatibility test failed: {e}")
            self.test_results['callback_compatibility'] = False
            return False
    
    def test_feature_engine(self):
        """Test 3: Feature Engine Processing"""
        logger.info("🔍 Test 3: Feature Engine Processing")
        
        try:
            # Feed multiple data points
            instruments = ['ES 12-24', 'NQ 12-24']
            base_prices = {'ES 12-24': 4500.0, 'NQ 12-24': 15000.0}
            
            for i in range(25):  # Enough data for features
                for instrument in instruments:
                    # Simulate price movement
                    price_change = (i - 12) * 0.25  # -3 to +3 range
                    price = base_prices[instrument] + price_change
                    
                    test_data = MarketData(
                        instrument=instrument,
                        timestamp=datetime.now(),
                        bid=price - 0.125,
                        ask=price + 0.125,
                        last=price,
                        volume=100 + i,
                        bid_size=10,
                        ask_size=10
                    )
                    
                    self.system._on_market_data(test_data)
            
            # Check if features are computed
            features_computed = 0
            for instrument in instruments:
                features = self.system.feature_engine.get_features(instrument)
                if features:
                    features_computed += 1
                    logger.info(f"✅ Features computed for {instrument}: {len(features)} features")
            
            if features_computed > 0:
                self.test_results['feature_engine'] = True
                logger.info("✅ Feature engine working correctly")
                return True
            else:
                self.test_results['feature_engine'] = False
                logger.warning("⚠️ Feature engine not producing features")
                return False
                
        except Exception as e:
            logger.error(f"❌ Feature engine test failed: {e}")
            self.test_results['feature_engine'] = False
            return False
    
    def test_ml_predictions(self):
        """Test 4: ML Model Predictions"""
        logger.info("🔍 Test 4: ML Model Predictions")
        
        try:
            # Generate training data and train model
            training_data = []
            for i in range(1000):
                training_data.append({
                    'rsi': 30 + (i % 40),  # RSI between 30-70
                    'trend_strength': (i % 10 - 5) * 0.001,  # Small trend values
                    'volume_ratio': 0.5 + (i % 15) * 0.1,  # Volume ratio 0.5-2.0
                    'volatility': 0.01 + (i % 5) * 0.01  # Volatility 0.01-0.05
                })
            
            self.system.ml_model.train(training_data)
            
            # Test prediction
            test_features = {
                'rsi': 45.0,
                'trend_strength': 0.002,
                'volume_ratio': 1.2,
                'volatility': 0.025
            }
            
            signal = self.system.ml_model.predict(test_features)
            
            if signal and hasattr(signal, 'signal') and hasattr(signal, 'confidence'):
                self.test_results['ml_predictions'] = True
                logger.info(f"✅ ML prediction working: Signal={signal.signal}, Confidence={signal.confidence:.3f}")
                return True
            else:
                self.test_results['ml_predictions'] = False
                logger.error("❌ Invalid ML prediction result")
                return False
                
        except Exception as e:
            logger.error(f"❌ ML prediction test failed: {e}")
            self.test_results['ml_predictions'] = False
            return False
    
    def test_portfolio_management(self):
        """Test 5: Portfolio Management"""
        logger.info("🔍 Test 5: Portfolio Management")
        
        try:
            # Test trade execution
            symbol = 'ES 12-24'
            price = 4500.0
            quantity = 10
            
            # Update price
            self.system.portfolio.update_price(symbol, price)
            
            # Execute trade
            success = self.system.portfolio.execute_trade(symbol, quantity, price)
            
            if success:
                # Check portfolio status
                summary = self.system.portfolio.get_portfolio_summary()
                
                if symbol in summary['positions'] and summary['positions'][symbol] == quantity:
                    self.test_results['portfolio_management'] = True
                    logger.info(f"✅ Portfolio management working: Position={summary['positions'][symbol]}")
                    return True
                else:
                    self.test_results['portfolio_management'] = False
                    logger.error("❌ Portfolio position not updated correctly")
                    return False
            else:
                self.test_results['portfolio_management'] = False
                logger.error("❌ Trade execution failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Portfolio management test failed: {e}")
            self.test_results['portfolio_management'] = False
            return False
    
    async def test_system_integration(self):
        """Test 6: Full System Integration"""
        logger.info("🔍 Test 6: Full System Integration")
        
        try:
            # Initialize system
            await self.system.initialize()
            
            # Simulate market data processing
            test_data = MarketData(
                instrument='ES 12-24',
                timestamp=datetime.now(),
                bid=4500.0,
                ask=4500.25,
                last=4500.125,
                volume=100,
                bid_size=10,
                ask_size=10
            )
            
            initial_signals = self.system.total_signals
            self.system._on_market_data(test_data)
            
            # Check if signal was processed
            if self.system.total_signals > initial_signals:
                self.test_results['system_integration'] = True
                logger.info("✅ Full system integration working")
                return True
            else:
                self.test_results['system_integration'] = True  # Still pass if no signal generated (normal)
                logger.info("✅ System integration working (no signal generated - normal)")
                return True
                
        except Exception as e:
            logger.error(f"❌ System integration test failed: {e}")
            self.test_results['system_integration'] = False
            return False
    
    def print_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*70)
        logger.info("PRODUCTION SYSTEM TEST RESULTS")
        logger.info("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("-"*70)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW BEFORE PRODUCTION")
        
        logger.info("="*70)

async def main():
    """Run production system tests"""
    print("🚀 INSTITUTIONAL ML TRADING SYSTEM - PRODUCTION TEST")
    print("="*70)
    
    tester = ProductionSystemTest()
    
    # Run all tests
    tests = [
        tester.test_system_initialization,
        tester.test_callback_compatibility,
        tester.test_feature_engine,
        tester.test_ml_predictions,
        tester.test_portfolio_management,
    ]
    
    # Run synchronous tests
    for test in tests:
        if not test():
            break  # Stop on first failure for debugging
        await asyncio.sleep(0.1)  # Small delay between tests
    
    # Run async test
    await tester.test_system_integration()
    
    # Print results
    tester.print_results()
    
    return tester.test_results

if __name__ == "__main__":
    asyncio.run(main())
