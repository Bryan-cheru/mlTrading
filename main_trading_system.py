"""
Main Trading System
Orchestrates the entire institutional ML trading system with NinjaTrader 8 integration
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.ingestion.realtime_data_manager import RealTimeDataManager, FUTURES_CONFIG
from data_pipeline.real_time_ingestion import RealTimeDataProcessor, RealTimeSignal
from data_pipeline.live_market_data import LiveMarketDataManager
from ml_models.training.trading_model import TradingMLModel, SignalGenerator
from trading_engine.ninjatrader_executor import NinjaTraderExecutor, FUTURES_EXECUTION_CONFIG
from monitoring.performance_monitor import PerformanceMonitor
from risk_engine.risk_manager import AdvancedRiskManager

class InstitutionalTradingSystem:
    """
    Main institutional ML trading system with NinjaTrader 8 integration
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        
        # Core components
        self.data_manager = None
        self.live_data_processor = None  # New live data processor
        self.ml_model = None
        self.signal_generator = None
        self.executor = None
        self.performance_monitor = None
        self.risk_manager = None
        
        # System state
        self.is_running = False
        self.start_time = None
        self.total_trades = 0
        self.daily_pnl = 0.0
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            'instruments': ['ES', 'NQ'],  # E-mini S&P 500 and NASDAQ futures
            'timeframes': ['1 Minute', '5 Minute'],
            'trading_hours': {
                'start': '09:30',
                'end': '16:00',
                'timezone': 'US/Eastern'
            },
            'model': {
                'retrain_interval_hours': 24,
                'min_confidence': 0.65,
                'max_position_size': 0.05  # 5% of account
            },
            'risk': {
                'max_daily_loss': 0.02,  # 2% daily loss limit
                'max_position_size': 5,  # 5 contracts
                'stop_loss_pct': 0.01,   # 1% stop loss
                'take_profit_pct': 0.02  # 2% take profit
            },
            'execution': {
                'account_balance': 100000,  # $100K account
                'use_ninjatrader': True,
                'simulation_mode': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return default_config
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Institutional ML Trading System...")
            
            # Initialize live data processor (Phase 3 enhancement)
            logger.info("📡 Initializing Live Market Data Processor...")
            self.live_data_processor = RealTimeDataProcessor()
            if not await self.live_data_processor.initialize_system():
                logger.error("❌ Failed to initialize live data processor")
                return False
            
            # Add signal callback
            self.live_data_processor.add_signal_callback(self._on_live_signal_received)
            
            # Initialize legacy data manager for compatibility
            try:
                self.data_manager = RealTimeDataManager(self.config)
                logger.info("✅ Legacy data manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Legacy data manager failed: {e}")
                self.data_manager = None
            
            # Initialize ML model
            try:
                self.ml_model = TradingMLModel()
                logger.info("✅ ML model initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML model initialization failed: {e}")
                self.ml_model = None
            
            # Initialize signal generator
            try:
                self.signal_generator = SignalGenerator(self.config['risk'])
                logger.info("✅ Signal generator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Signal generator initialization failed: {e}")
                self.signal_generator = None
            
            # Initialize NinjaTrader executor
            if self.config['execution']['use_ninjatrader']:
                try:
                    self.executor = NinjaTraderExecutor(FUTURES_EXECUTION_CONFIG)
                    if not await self.executor.connect():
                        logger.error("❌ Failed to connect to NinjaTrader")
                        # Don't fail initialization, continue in simulation mode
                        self.executor = None
                    else:
                        logger.info("✅ NinjaTrader executor connected")
                except Exception as e:
                    logger.warning(f"⚠️ NinjaTrader executor failed: {e}")
                    self.executor = None
            
            # Initialize performance monitor
            try:
                self.performance_monitor = PerformanceMonitor()
                logger.info("✅ Performance monitor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Performance monitor failed: {e}")
                self.performance_monitor = None
            
            # Initialize risk manager
            try:
                self.risk_manager = AdvancedRiskManager(self.config['risk'])
                logger.info("✅ Risk manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Risk manager failed: {e}")
                self.risk_manager = None
            
            # Set up callbacks
            self._setup_callbacks()
            
            logger.info("✅ System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            return False
    
    def _setup_callbacks(self):
        """Set up event callbacks between components"""
        try:
            # Legacy data callbacks
            if self.data_manager:
                self.data_manager.subscribe_to_data(self._on_market_data_update)
            
            # Order execution callbacks
            if self.executor:
                self.executor.add_order_callback(self._on_order_update)
                self.executor.add_position_callback(self._on_position_update)
            
        except Exception as e:
            logger.error(f"❌ Error setting up callbacks: {e}")
    
    def _on_live_signal_received(self, signal: RealTimeSignal):
        """Handle live trading signals from Phase 3 data processor"""
        try:
            logger.info(
                f"🎯 LIVE SIGNAL: {signal.signal_type} {signal.symbol} @ {signal.price:.2f} "
                f"(Confidence: {signal.confidence:.2f})"
            )
            
            # Execute the signal
            asyncio.create_task(self._execute_live_signal(signal))
            
        except Exception as e:
            logger.error(f"❌ Error handling live signal: {e}")
    
    async def _execute_live_signal(self, signal: RealTimeSignal):
        """Execute a live trading signal"""
        try:
            # Apply additional risk checks
            if self.risk_manager:
                risk_check = self.risk_manager.check_signal_risk({
                    'symbol': signal.symbol,
                    'signal': signal.signal_type,
                    'price': signal.price,
                    'confidence': signal.confidence,
                    'position_size': signal.position_size
                })
                
                if not risk_check.get('approved', False):
                    logger.warning(f"🛡️ Signal rejected by risk manager: {risk_check.get('reason', 'Unknown')}")
                    return
            
            # Execute through NinjaTrader if available
            if self.executor:
                order_result = await self.executor.place_order(
                    symbol=signal.symbol,
                    action=signal.signal_type,
                    quantity=int(signal.position_size),
                    price=signal.price,
                    order_type='MARKET'
                )
                
                if order_result.get('success', False):
                    logger.info(f"✅ Order executed: {order_result}")
                    
                    # Update performance tracking
                    if self.performance_monitor:
                        self.performance_monitor.record_trade({
                            'symbol': signal.symbol,
                            'action': signal.signal_type,
                            'price': signal.price,
                            'quantity': signal.position_size,
                            'timestamp': signal.timestamp
                        })
                else:
                    logger.error(f"❌ Order execution failed: {order_result}")
            else:
                # Simulation mode
                logger.info(f"📝 SIMULATION: {signal.signal_type} {signal.symbol} @ {signal.price:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error executing live signal: {e}")
    
    async def start(self):
        """Start the trading system"""
        try:
            if not await self.initialize():
                logger.error("❌ Failed to initialize system")
                return
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("🚀 Starting Institutional ML Trading System...")
            
            # Start live data processor (Phase 3)
            if self.live_data_processor:
                symbols = self.config['instruments']
                await self.live_data_processor.start_real_time_processing(symbols)
                logger.info(f"📡 Live data processing started for: {symbols}")
            
            # Start legacy data manager for compatibility
            if self.data_manager:
                asyncio.create_task(self.data_manager.start())
                logger.info("✅ Legacy data manager started")
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading system"""
        try:
            logger.info("🛑 Stopping trading system...")
            
            self.is_running = False
            
            # Stop live data processor
            if self.live_data_processor:
                await self.live_data_processor.stop_processing()
                logger.info("📡 Live data processor stopped")
            
            # Stop legacy data manager
            if self.data_manager:
                self.data_manager.stop()
                logger.info("✅ Legacy data manager stopped")
            
            # Disconnect executor
            if self.executor:
                self.executor.disconnect()
                logger.info("🔌 NinjaTrader executor disconnected")
            
            # Log final performance
            self._log_final_performance()
            
            logger.info("✅ Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")
    
    async def _main_trading_loop(self):
        """Main trading logic loop"""
        logger.info("🔄 Starting main trading loop...")
        
        last_model_retrain = datetime.now()
        last_status_update = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Print status every 30 seconds
                if (current_time - last_status_update).total_seconds() >= 30:
                    await self._print_system_status()
                    last_status_update = current_time
                
                # Check if we should retrain the model
                if (current_time - last_model_retrain).total_seconds() > \
                   self.config['model']['retrain_interval_hours'] * 3600:
                    await self._retrain_model()
                    last_model_retrain = datetime.now()
                
                # Main trading logic now runs through live data callbacks
                await asyncio.sleep(1)  # Prevent busy loop
                
            except Exception as e:
                logger.error(f"❌ Error in main trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _print_system_status(self):
        """Print current system status"""
        try:
            if self.live_data_processor:
                status = self.live_data_processor.get_system_status()
                
                logger.info(
                    f"📊 SYSTEM STATUS: "
                    f"Processing: {'✅' if status['processing_active'] else '❌'} | "
                    f"Symbols: {len(status['subscribed_symbols'])} | "
                    f"Ticks: {status['stats']['ticks_processed']} | "
                    f"Bars: {status['stats']['bars_processed']} | "
                    f"Signals: {status['stats']['signals_generated']}"
                )
                
                if 'latency_stats' in status:
                    latency = status['latency_stats']
                    logger.info(
                        f"⚡ PERFORMANCE: "
                        f"Avg Latency: {latency['avg_ms']:.1f}ms | "
                        f"Max: {latency['max_ms']:.1f}ms | "
                        f"P95: {latency['p95_ms']:.1f}ms"
                    )
            
            # Runtime statistics
            if self.start_time:
                runtime = datetime.now() - self.start_time
                logger.info(f"⏱️ RUNTIME: {str(runtime).split('.')[0]} | Total Trades: {self.total_trades}")
            
        except Exception as e:
            logger.error(f"❌ Error printing system status: {e}")
    
    async def _on_market_data_update(self, data_snapshot: Dict):
        """Handle market data updates"""
        try:
            current_time = datetime.now()
            
            # Check if we're in trading hours
            if not self._is_trading_time(current_time):
                return
            
            # Process each instrument
            for instrument in self.config['instruments']:
                if instrument in data_snapshot['market_data']:
                    market_data = data_snapshot['market_data'][instrument]
                    current_price = data_snapshot['prices'].get(instrument)
                    
                    if current_price and market_data.get('features'):
                        await self._process_trading_signal(
                            instrument, 
                            market_data['features'], 
                            current_price
                        )
            
        except Exception as e:
            logger.error(f"Error processing market data update: {e}")
    
    async def _process_trading_signal(self, instrument: str, features: Dict, current_price: float):
        """Process trading signal for an instrument"""
        try:
            # Skip if model not trained
            if not self.ml_model.is_trained:
                logger.warning("ML model not trained yet")
                return
            
            # Get ML prediction
            prediction = self.ml_model.predict(features)
            
            # Generate trading signal with risk management
            trading_signal = self.signal_generator.generate_signal(
                prediction, 
                current_price, 
                self.config['execution']['account_balance']
            )
            
            # Additional risk checks
            if not self.risk_manager.check_signal_risk(trading_signal, instrument):
                logger.info(f"Signal rejected by risk manager for {instrument}")
                return
            
            # Execute trade if signal is actionable
            if trading_signal['signal'] in ['BUY', 'SELL'] and trading_signal['position_size'] > 0:
                await self._execute_trade(instrument, trading_signal)
            
            # Log signal for analysis
            logger.info(f"Signal for {instrument}: {trading_signal['signal']} "
                       f"(Confidence: {trading_signal['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing trading signal for {instrument}: {e}")
    
    async def _execute_trade(self, instrument: str, signal: Dict):
        """Execute a trade through NinjaTrader"""
        try:
            if not self.executor:
                logger.warning("No executor available for trade execution")
                return
            
            # Place the order
            order_id = await self.executor.place_order(
                instrument=instrument,
                action=signal['signal'],
                quantity=signal['position_size'],
                order_type='MARKET'  # Use market orders for now
            )
            
            if order_id:
                logger.info(f"Trade executed: {signal['signal']} {signal['position_size']} {instrument} "
                           f"at {signal['entry_price']:.2f}")
                self.total_trades += 1
                
                # Update performance tracking
                self.performance_stats['total_trades'] += 1
            else:
                logger.warning(f"Failed to execute trade for {instrument}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _on_order_update(self, order):
        """Handle order status updates"""
        try:
            logger.info(f"Order update: {order.order_id} - {order.status.value}")
            
            if self.performance_monitor:
                self.performance_monitor.record_order(order)
                
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    def _on_position_update(self, position):
        """Handle position updates"""
        try:
            logger.info(f"Position update: {position.instrument} - "
                       f"Qty: {position.quantity}, PnL: {position.realized_pnl:.2f}")
            
            # Update daily P&L
            self.daily_pnl += position.realized_pnl
            
            if self.performance_monitor:
                self.performance_monitor.record_position(position)
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def _retrain_model(self):
        """Retrain the ML model with recent data"""
        try:
            logger.info("Starting model retraining...")
            
            # Get historical data for training
            training_data = await self._prepare_training_data()
            
            if len(training_data) > 100:  # Minimum samples for training
                metrics = self.ml_model.train(training_data)
                logger.info(f"Model retrained. Accuracy: {metrics.get('accuracy', 0):.4f}")
            else:
                logger.warning("Insufficient data for model retraining")
                
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
    
    async def _prepare_training_data(self) -> 'pd.DataFrame':
        """Prepare training data from recent market activity"""
        # This would typically involve:
        # 1. Collecting recent bar data
        # 2. Calculating features
        # 3. Labeling data based on future returns
        # 4. Creating DataFrame for training
        
        # Placeholder implementation
        import pandas as pd
        return pd.DataFrame()
    
    def _is_trading_time(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        try:
            # Simplified trading hours check
            # In production, this would consider:
            # - Market holidays
            # - Different sessions (pre-market, regular, after-hours)
            # - Timezone conversions
            
            hour = current_time.hour
            minute = current_time.minute
            
            # Regular trading hours: 9:30 AM - 4:00 PM ET
            start_time = 9 * 60 + 30  # 9:30 AM in minutes
            end_time = 16 * 60        # 4:00 PM in minutes
            current_minutes = hour * 60 + minute
            
            return start_time <= current_minutes <= end_time
            
        except Exception as e:
            logger.error(f"Error checking trading time: {e}")
            return False
    
    def _log_final_performance(self):
        """Log final performance statistics"""
        try:
            runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            logger.info("=" * 50)
            logger.info("FINAL PERFORMANCE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Total Trades: {self.total_trades}")
            logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
            logger.info(f"Win Rate: {self._calculate_win_rate():.2f}%")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error logging final performance: {e}")
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from performance stats"""
        total = self.performance_stats['winning_trades'] + self.performance_stats['losing_trades']
        if total == 0:
            return 0.0
        return (self.performance_stats['winning_trades'] / total) * 100


async def main():
    """Main entry point"""
    # Create trading system
    trading_system = InstitutionalTradingSystem()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(trading_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the trading system
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await trading_system.stop()


if __name__ == "__main__":
    # Run the trading system
    asyncio.run(main())
