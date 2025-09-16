"""
Phase 3 Live Market Data Integration Demo
Demonstrates the new real-time market data system with professional feeds
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.live_market_data import (
    LiveMarketDataManager,
    AlphaVantageProvider,
    YahooFinanceProvider,
    NinjaTraderProvider,
    MarketTick,
    MarketBar
)
from data_pipeline.real_time_ingestion import RealTimeDataProcessor, RealTimeSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDataDemo:
    """Demonstration of Phase 3 live market data capabilities"""
    
    def __init__(self):
        self.data_processor = None
        self.signals_received = []
        self.ticks_received = []
        self.bars_received = []
        
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of live market data features"""
        
        logger.info("🚀 Starting Phase 3 Live Market Data Integration Demo")
        logger.info("=" * 60)
        
        try:
            # Test 1: Direct provider connections
            await self._test_provider_connections()
            
            # Test 2: Historical data aggregation
            await self._test_historical_data_aggregation()
            
            # Test 3: Real-time data processing
            await self._test_real_time_processing()
            
            # Test 4: Signal generation
            await self._test_signal_generation()
            
            # Test 5: Performance metrics
            await self._test_performance_metrics()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
        finally:
            logger.info("✅ Phase 3 Live Market Data Demo completed")
    
    async def _test_provider_connections(self):
        """Test connections to different market data providers"""
        logger.info("\n📡 TEST 1: Provider Connections")
        logger.info("-" * 40)
        
        # Initialize data manager
        data_manager = LiveMarketDataManager()
        
        # Add providers
        data_manager.add_provider("YahooFinance", YahooFinanceProvider())
        data_manager.add_provider("NinjaTrader", NinjaTraderProvider())
        
        # Test Alpha Vantage if API key available
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key:
            data_manager.add_provider("AlphaVantage", AlphaVantageProvider(alpha_key))
            logger.info("🔑 Alpha Vantage API key found")
        else:
            logger.info("ℹ️ Alpha Vantage API key not found (using free providers)")
        
        # Connect to all providers
        connections = await data_manager.connect_all_providers()
        
        # Report results
        for provider, connected in connections.items():
            status = "✅ Connected" if connected else "❌ Failed"
            logger.info(f"   {provider}: {status}")
        
        await data_manager.disconnect_all_providers()
        
        connected_count = sum(connections.values())
        logger.info(f"📊 Result: {connected_count}/{len(connections)} providers connected")
    
    async def _test_historical_data_aggregation(self):
        """Test historical data retrieval and aggregation"""
        logger.info("\n📈 TEST 2: Historical Data Aggregation")
        logger.info("-" * 40)
        
        data_manager = LiveMarketDataManager()
        data_manager.add_provider("YahooFinance", YahooFinanceProvider())
        data_manager.add_provider("NinjaTrader", NinjaTraderProvider())
        
        await data_manager.connect_all_providers()
        
        # Test different symbols and timeframes
        test_cases = [
            ("ES", "5m"),
            ("NQ", "1m"),
            ("AAPL", "15m")
        ]
        
        for symbol, timeframe in test_cases:
            logger.info(f"📊 Fetching {symbol} data ({timeframe})...")
            
            bars = await data_manager.get_aggregated_historical_data(
                symbol, timeframe, hours_back=4
            )
            
            if bars:
                latest_bar = bars[-1]
                logger.info(
                    f"   ✅ {len(bars)} bars | Latest: {latest_bar.close:.2f} "
                    f"@ {latest_bar.timestamp.strftime('%H:%M:%S')} [{latest_bar.source}]"
                )
            else:
                logger.warning(f"   ⚠️ No data received for {symbol}")
        
        await data_manager.disconnect_all_providers()
    
    async def _test_real_time_processing(self):
        """Test real-time data processing system"""
        logger.info("\n🔴 TEST 3: Real-Time Data Processing")
        logger.info("-" * 40)
        
        # Initialize processor
        self.data_processor = RealTimeDataProcessor()
        
        # Setup callbacks
        def on_tick(tick):
            self.ticks_received.append(tick)
            if len(self.ticks_received) <= 5:  # Log first 5 only
                logger.info(f"   📊 TICK: {tick.symbol} @ {tick.last:.2f} [{tick.source}]")
        
        def on_bar(bar):
            self.bars_received.append(bar)
            if len(self.bars_received) <= 5:  # Log first 5 only
                logger.info(f"   📈 BAR: {bar.symbol} OHLC({bar.open:.2f}, {bar.high:.2f}, {bar.low:.2f}, {bar.close:.2f}) [{bar.source}]")
        
        # Initialize system
        success = await self.data_processor.initialize_system()
        if not success:
            logger.error("❌ Failed to initialize real-time processor")
            return
        
        # Start processing
        symbols = ['ES', 'NQ']
        await self.data_processor.start_real_time_processing(symbols)
        
        # Run for 30 seconds
        logger.info("⏳ Collecting real-time data for 30 seconds...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 30:
            await asyncio.sleep(1)
        
        # Report results
        logger.info(f"📊 Results: {len(self.ticks_received)} ticks, {len(self.bars_received)} bars received")
        
        await self.data_processor.stop_processing()
    
    async def _test_signal_generation(self):
        """Test trading signal generation"""
        logger.info("\n🎯 TEST 4: Signal Generation")
        logger.info("-" * 40)
        
        if not self.data_processor:
            logger.warning("⚠️ Data processor not available, skipping signal test")
            return
        
        # Setup signal callback
        def on_signal(signal):
            self.signals_received.append(signal)
            logger.info(
                f"   🎯 SIGNAL: {signal.signal_type} {signal.symbol} @ {signal.price:.2f} "
                f"(Confidence: {signal.confidence:.2f}, Risk: {signal.risk_score:.2f})"
            )
        
        # Re-initialize for signal testing
        self.data_processor = RealTimeDataProcessor()
        await self.data_processor.initialize_system()
        self.data_processor.add_signal_callback(on_signal)
        
        # Start processing
        symbols = ['ES', 'NQ']
        await self.data_processor.start_real_time_processing(symbols)
        
        # Run for 45 seconds to allow signal generation
        logger.info("⏳ Monitoring for trading signals for 45 seconds...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 45:
            await asyncio.sleep(1)
        
        # Report results
        logger.info(f"📊 Results: {len(self.signals_received)} signals generated")
        
        if self.signals_received:
            signal_types = {}
            for signal in self.signals_received:
                signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
            
            logger.info(f"   Signal breakdown: {signal_types}")
        
        await self.data_processor.stop_processing()
    
    async def _test_performance_metrics(self):
        """Test performance monitoring and metrics"""
        logger.info("\n⚡ TEST 5: Performance Metrics")
        logger.info("-" * 40)
        
        if not self.data_processor:
            logger.warning("⚠️ Data processor not available, skipping performance test")
            return
        
        # Get system status
        status = self.data_processor.get_system_status()
        
        logger.info("📊 System Performance Summary:")
        logger.info(f"   Ticks Processed: {status['stats']['ticks_processed']:,}")
        logger.info(f"   Bars Processed: {status['stats']['bars_processed']:,}")
        logger.info(f"   Signals Generated: {status['stats']['signals_generated']:,}")
        logger.info(f"   Data Quality Score: {status['stats']['data_quality_score']:.3f}")
        
        if 'latency_stats' in status:
            latency = status['latency_stats']
            logger.info(f"   Average Latency: {latency['avg_ms']:.2f}ms")
            logger.info(f"   Maximum Latency: {latency['max_ms']:.2f}ms")
            logger.info(f"   95th Percentile: {latency['p95_ms']:.2f}ms")
            
            # Check if we meet institutional requirements (<50ms)
            if latency['p95_ms'] < 50:
                logger.info("   ✅ Latency meets institutional requirements (<50ms)")
            else:
                logger.warning(f"   ⚠️ Latency exceeds requirements: {latency['p95_ms']:.1f}ms")
        
        # Data quality analysis
        total_data_points = len(self.ticks_received) + len(self.bars_received)
        logger.info(f"   Total Data Points: {total_data_points:,}")
        
        if total_data_points > 0:
            data_rate = total_data_points / 60  # per minute
            logger.info(f"   Data Throughput: {data_rate:.1f} points/minute")

async def main():
    """Run the live market data demo"""
    demo = LiveDataDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    # Set environment variables for demo (optional)
    # os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_api_key_here'
    
    print("🚀 Phase 3: Live Market Data Integration Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("✅ Multi-provider market data connections")
    print("✅ Real-time data streaming with <50ms latency")
    print("✅ Historical data aggregation")
    print("✅ Automated signal generation")
    print("✅ Performance monitoring and quality control")
    print("=" * 50)
    
    asyncio.run(main())
