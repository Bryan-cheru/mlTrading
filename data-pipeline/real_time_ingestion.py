"""
Real-Time Data Ingestion Engine
Integrates live market data into the institutional trading system
Features: Real-time processing, data validation, feature engineering, signal generation
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our live market data system
from live_market_data import (
    LiveMarketDataManager, 
    AlphaVantageProvider, 
    YahooFinanceProvider, 
    NinjaTraderProvider,
    MarketTick, 
    MarketBar,
    validate_market_data
)

# Import existing system components
try:
    from feature_store.realtime.realtime_features import RealtimeFeatureEngine
    from ml_models.inference.trading_model import TradingModel
    from trading_engine.signal_generator import SignalGenerator
    from risk_engine.risk_manager import AdvancedRiskManager
    from monitoring.performance_monitor import PerformanceMonitor
except ImportError as e:
    logging.warning(f"⚠️ Could not import some components: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RealTimeSignal:
    """Real-time trading signal generated from live data"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float
    features: Dict[str, float]
    model_prediction: float
    risk_score: float
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class RealTimeDataProcessor:
    """Processes real-time market data and generates trading signals"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.data_manager = LiveMarketDataManager()
        self.feature_engine = None
        self.trading_model = None
        self.signal_generator = None
        self.risk_manager = None
        self.performance_monitor = None
        
        # Data storage
        self.tick_buffer = queue.Queue(maxsize=10000)
        self.bar_buffer = queue.Queue(maxsize=1000)
        self.latest_bars = {}  # symbol -> latest bar
        self.latest_ticks = {}  # symbol -> latest tick
        
        # Processing state
        self.processing_active = False
        self.subscribed_symbols = set()
        self.signal_callbacks = []
        
        # Performance tracking
        self.stats = {
            'ticks_processed': 0,
            'bars_processed': 0,
            'signals_generated': 0,
            'last_update': datetime.now(),
            'processing_latency_ms': [],
            'data_quality_score': 1.0
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.processing_thread = None
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration"""
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../config/live_market_data_config.json'
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)['live_market_data']
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'symbols': {
                'futures': ['ES', 'NQ', 'YM', 'RTY']
            },
            'timeframes': ['1m', '5m', '15m'],
            'data_quality': {
                'max_price_change_percent': 10.0,
                'stale_data_threshold_seconds': 300
            },
            'streaming': {
                'max_latency_ms': 50,
                'buffer_size': 10000
            }
        }
    
    async def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Real-Time Data Ingestion Engine...")
            
            # Initialize market data providers
            await self._setup_data_providers()
            
            # Initialize processing components
            await self._setup_processing_components()
            
            # Setup data callbacks
            self._setup_data_callbacks()
            
            logger.info("✅ Real-Time Data Ingestion Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {str(e)}")
            return False
    
    async def _setup_data_providers(self) -> None:
        """Setup market data providers"""
        
        # Add Yahoo Finance (free tier)
        yahoo_provider = YahooFinanceProvider()
        self.data_manager.add_provider("YahooFinance", yahoo_provider)
        
        # Add NinjaTrader (simulated for now)
        ninja_provider = NinjaTraderProvider()
        self.data_manager.add_provider("NinjaTrader", ninja_provider)
        
        # Add Alpha Vantage if API key is available
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_vantage_key:
            alpha_provider = AlphaVantageProvider(alpha_vantage_key)
            self.data_manager.add_provider("AlphaVantage", alpha_provider)
            logger.info("✅ Alpha Vantage provider added")
        else:
            logger.info("ℹ️ Alpha Vantage API key not found, using free providers only")
        
        # Connect to all providers
        connections = await self.data_manager.connect_all_providers()
        connected_count = sum(connections.values())
        
        if connected_count == 0:
            raise Exception("No market data providers connected successfully")
        
        logger.info(f"📡 {connected_count}/{len(connections)} providers connected")
    
    async def _setup_processing_components(self) -> None:
        """Setup ML and trading components"""
        try:
            # Initialize feature engine
            self.feature_engine = RealtimeFeatureEngine()
            logger.info("✅ Feature engine initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Feature engine not available: {str(e)}")
            self.feature_engine = None
        
        try:
            # Initialize trading model
            self.trading_model = TradingModel()
            logger.info("✅ Trading model initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Trading model not available: {str(e)}")
            self.trading_model = None
        
        try:
            # Initialize signal generator
            self.signal_generator = SignalGenerator()
            logger.info("✅ Signal generator initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Signal generator not available: {str(e)}")
            self.signal_generator = None
        
        try:
            # Initialize risk manager
            self.risk_manager = AdvancedRiskManager()
            logger.info("✅ Risk manager initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Risk manager not available: {str(e)}")
            self.risk_manager = None
        
        try:
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            logger.info("✅ Performance monitor initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Performance monitor not available: {str(e)}")
            self.performance_monitor = None
    
    def _setup_data_callbacks(self) -> None:
        """Setup callbacks for real-time data"""
        
        def on_tick_received(tick: MarketTick):
            """Handle incoming tick data"""
            try:
                # Store in buffer for processing
                if not self.tick_buffer.full():
                    self.tick_buffer.put(tick, block=False)
                    self.latest_ticks[tick.symbol] = tick
                    self.stats['ticks_processed'] += 1
                else:
                    logger.warning("⚠️ Tick buffer is full, dropping data")
                    
            except Exception as e:
                logger.error(f"❌ Error processing tick: {str(e)}")
        
        def on_bar_received(bar: MarketBar):
            """Handle incoming bar data"""
            try:
                # Validate data quality
                if not validate_market_data(bar):
                    logger.warning(f"⚠️ Invalid bar data for {bar.symbol}")
                    return
                
                # Store in buffer for processing
                if not self.bar_buffer.full():
                    self.bar_buffer.put(bar, block=False)
                    self.latest_bars[bar.symbol] = bar
                    self.stats['bars_processed'] += 1
                    
                    # Trigger signal generation for this bar
                    asyncio.create_task(self._process_bar_for_signals(bar))
                    
                else:
                    logger.warning("⚠️ Bar buffer is full, dropping data")
                    
            except Exception as e:
                logger.error(f"❌ Error processing bar: {str(e)}")
        
        # Register callbacks
        self.data_manager.add_tick_callback(on_tick_received)
        self.data_manager.add_bar_callback(on_bar_received)
        
        logger.info("✅ Data callbacks registered")
    
    async def _process_bar_for_signals(self, bar: MarketBar) -> None:
        """Process a bar and generate trading signals"""
        start_time = time.time()
        
        try:
            # Skip if required components are not available
            if not all([self.feature_engine, self.trading_model, self.signal_generator]):
                logger.debug("⚠️ Components not available for signal generation")
                return
            
            # Get historical data for feature calculation
            historical_bars = await self.data_manager.get_aggregated_historical_data(
                bar.symbol, bar.timeframe, hours_back=24
            )
            
            if len(historical_bars) < 50:  # Need minimum history
                logger.debug(f"⚠️ Insufficient historical data for {bar.symbol}")
                return
            
            # Convert to DataFrame for processing
            bar_data = []
            for b in historical_bars:
                bar_data.append({
                    'timestamp': b.timestamp,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume
                })
            
            df = pd.DataFrame(bar_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Generate features
            features = self.feature_engine.calculate_features(df)
            
            if not features:
                logger.debug(f"⚠️ No features generated for {bar.symbol}")
                return
            
            # Get model prediction
            prediction = self.trading_model.predict(features)
            
            # Generate signal
            signal_data = self.signal_generator.generate_signal(
                bar.symbol, bar.close, prediction, features
            )
            
            if signal_data['signal'] != 'HOLD':
                # Apply risk management
                if self.risk_manager:
                    risk_result = self.risk_manager.check_signal_risk(signal_data)
                    if not risk_result['approved']:
                        logger.info(f"🛡️ Signal rejected by risk manager: {risk_result['reason']}")
                        return
                
                # Create real-time signal
                signal = RealTimeSignal(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    signal_type=signal_data['signal'],
                    confidence=signal_data.get('confidence', 0.5),
                    price=bar.close,
                    features=features,
                    model_prediction=prediction,
                    risk_score=signal_data.get('risk_score', 0.5),
                    position_size=signal_data.get('position_size', 1),
                    stop_loss=signal_data.get('stop_loss'),
                    take_profit=signal_data.get('take_profit')
                )
                
                # Notify signal callbacks
                for callback in self.signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"❌ Signal callback error: {str(e)}")
                
                self.stats['signals_generated'] += 1
                
                # Log signal
                logger.info(
                    f"🎯 SIGNAL: {signal.signal_type} {signal.symbol} @ {signal.price:.2f} "
                    f"(Confidence: {signal.confidence:.2f}, Risk: {signal.risk_score:.2f})"
                )
            
            # Track processing latency
            processing_time = (time.time() - start_time) * 1000
            self.stats['processing_latency_ms'].append(processing_time)
            
            # Keep only last 1000 latency measurements
            if len(self.stats['processing_latency_ms']) > 1000:
                self.stats['processing_latency_ms'] = self.stats['processing_latency_ms'][-1000:]
            
            # Check latency requirement (<50ms)
            if processing_time > self.config['streaming']['max_latency_ms']:
                logger.warning(f"⚠️ High processing latency: {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Error processing bar for signals: {str(e)}")
    
    def add_signal_callback(self, callback) -> None:
        """Add callback for trading signals"""
        self.signal_callbacks.append(callback)
        logger.info("✅ Signal callback added")
    
    async def start_real_time_processing(self, symbols: List[str]) -> None:
        """Start real-time data processing"""
        try:
            self.subscribed_symbols = set(symbols)
            self.processing_active = True
            
            # Start real-time data streaming
            await self.data_manager.start_real_time_streaming(symbols)
            
            # Start background processing thread
            self.processing_thread = threading.Thread(
                target=self._background_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info(f"🚀 Real-time processing started for: {symbols}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time processing: {str(e)}")
            raise
    
    def _background_processing_loop(self) -> None:
        """Background thread for data processing"""
        logger.info("🔄 Background processing loop started")
        
        while self.processing_active:
            try:
                # Process performance monitoring
                if self.performance_monitor:
                    self._update_performance_stats()
                
                # Process any queued data (if needed)
                self._process_queued_data()
                
                # Update system stats
                self.stats['last_update'] = datetime.now()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"❌ Background processing error: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _process_queued_data(self) -> None:
        """Process any queued data that needs attention"""
        # This could include batch processing, cleanup, etc.
        pass
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics"""
        try:
            # Calculate average latency
            if self.stats['processing_latency_ms']:
                avg_latency = np.mean(self.stats['processing_latency_ms'])
                max_latency = np.max(self.stats['processing_latency_ms'])
                
                # Update performance monitor if available
                if hasattr(self.performance_monitor, 'update_latency_stats'):
                    self.performance_monitor.update_latency_stats(avg_latency, max_latency)
            
            # Calculate data quality score
            total_data_points = self.stats['ticks_processed'] + self.stats['bars_processed']
            if total_data_points > 0:
                # Simple quality metric based on processing success
                self.stats['data_quality_score'] = min(1.0, total_data_points / (total_data_points + 1))
            
        except Exception as e:
            logger.error(f"❌ Error updating performance stats: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'processing_active': self.processing_active,
            'subscribed_symbols': list(self.subscribed_symbols),
            'connected_providers': len(self.data_manager.providers),
            'stats': self.stats.copy(),
            'latest_data': {
                'ticks': {symbol: tick.timestamp.isoformat() 
                         for symbol, tick in self.latest_ticks.items()},
                'bars': {symbol: bar.timestamp.isoformat() 
                        for symbol, bar in self.latest_bars.items()}
            }
        }
        
        # Add latency statistics
        if self.stats['processing_latency_ms']:
            status['latency_stats'] = {
                'avg_ms': np.mean(self.stats['processing_latency_ms']),
                'max_ms': np.max(self.stats['processing_latency_ms']),
                'min_ms': np.min(self.stats['processing_latency_ms']),
                'p95_ms': np.percentile(self.stats['processing_latency_ms'], 95)
            }
        
        return status
    
    async def stop_processing(self) -> None:
        """Stop real-time processing"""
        try:
            logger.info("🛑 Stopping real-time processing...")
            
            self.processing_active = False
            
            # Stop data streaming
            await self.data_manager.disconnect_all_providers()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=10)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Real-time processing stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {str(e)}")

# Example usage and testing
async def main():
    """Example usage of real-time data ingestion"""
    
    # Initialize processor
    processor = RealTimeDataProcessor()
    
    # Add signal callback
    def on_signal_generated(signal: RealTimeSignal):
        print(f"🎯 NEW SIGNAL: {signal.signal_type} {signal.symbol} @ {signal.price:.2f}")
        print(f"   Confidence: {signal.confidence:.2f}, Risk Score: {signal.risk_score:.2f}")
    
    processor.add_signal_callback(on_signal_generated)
    
    try:
        # Initialize system
        success = await processor.initialize_system()
        if not success:
            logger.error("❌ Failed to initialize system")
            return
        
        # Start processing
        symbols = ['ES', 'NQ', 'YM', 'RTY']
        await processor.start_real_time_processing(symbols)
        
        # Run for demo period
        logger.info("⏳ Running real-time processing for 60 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 60:
            # Print status every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                status = processor.get_system_status()
                logger.info(
                    f"📊 Status: {status['stats']['ticks_processed']} ticks, "
                    f"{status['stats']['bars_processed']} bars, "
                    f"{status['stats']['signals_generated']} signals"
                )
                
                if 'latency_stats' in status:
                    logger.info(f"⚡ Latency: {status['latency_stats']['avg_ms']:.1f}ms avg")
            
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Runtime error: {str(e)}")
    finally:
        # Cleanup
        await processor.stop_processing()
        logger.info("✅ Real-time data ingestion demo completed")

if __name__ == "__main__":
    asyncio.run(main())
