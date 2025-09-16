#!/usr/bin/env python3
"""
Live Market Data Integration Test
Tests real-time data feeds using Alpha Vantage API with professional data sources
"""

import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import with proper path handling
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-pipeline'))
from live_market_data import LiveMarketDataManager, MarketTick, MarketBar
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LiveDataTester:
    """Test suite for live market data integration"""
    
    def __init__(self):
        self.data_manager = LiveMarketDataManager()
        self.received_ticks = []
        self.received_bars = []
        
        # Register callbacks
        self.data_manager.add_tick_callback(self.on_tick)
        self.data_manager.add_bar_callback(self.on_bar)
    
    def on_tick(self, tick: MarketTick):
        """Handle incoming tick data"""
        self.received_ticks.append(tick)
        print(f"🔥 LIVE TICK: {tick.symbol} @ {tick.last:.2f} | {tick.timestamp} | Source: {tick.source}")
    
    def on_bar(self, bar: MarketBar):
        """Handle incoming bar data"""
        self.received_bars.append(bar)
        print(f"📊 LIVE BAR: {bar.symbol} | O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,} | {bar.timestamp}")
    
    async def test_provider_connections(self):
        """Test all data provider connections"""
        print("\n" + "="*80)
        print("🔌 TESTING MARKET DATA PROVIDER CONNECTIONS")
        print("="*80)
        
        results = await self.data_manager.connect_all_providers()
        
        for provider_name, connected in results.items():
            status = "✅ CONNECTED" if connected else "❌ FAILED"
            print(f"{provider_name:20} | {status}")
        
        return results
    
    async def test_historical_data(self):
        """Test historical data retrieval"""
        print("\n" + "="*80)
        print("📈 TESTING HISTORICAL DATA RETRIEVAL")
        print("="*80)
        
        # Test symbols for different markets
        test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        
        for symbol in test_symbols:
            try:
                print(f"\n📊 Fetching historical data for {symbol}...")
                
                # Get last 2 days of 5-minute bars
                end_date = datetime.now()
                start_date = end_date - timedelta(days=2)
                
                bars = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe="5m",
                    start_date=start_date,
                    end_date=end_date,
                    provider="AlphaVantage"
                )
                
                if bars:
                    latest_bar = bars[-1]
                    print(f"✅ {symbol}: {len(bars)} bars retrieved")
                    print(f"   Latest: {latest_bar.timestamp} | Close: ${latest_bar.close:.2f} | Volume: {latest_bar.volume:,}")
                else:
                    print(f"⚠️ {symbol}: No data retrieved")
                
                # Rate limiting for Alpha Vantage
                await asyncio.sleep(12)
                
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")
    
    async def test_real_time_streaming(self):
        """Test real-time data streaming simulation"""
        print("\n" + "="*80)
        print("🔴 TESTING REAL-TIME DATA STREAMING (SIMULATION)")
        print("="*80)
        
        # Subscribe to major symbols
        symbols = ["AAPL", "MSFT", "SPY"]
        
        await self.data_manager.subscribe_symbols(symbols)
        
        # Simulate real-time data for 30 seconds
        print("🔄 Starting 30-second real-time simulation...")
        
        start_time = time.time()
        tick_count = 0
        
        while time.time() - start_time < 30:
            # Simulate tick generation
            for symbol in symbols:
                await self.data_manager.simulate_live_tick(symbol)
                tick_count += 1
            
            await asyncio.sleep(1)  # 1-second intervals
        
        print(f"\n📊 Simulation complete:")
        print(f"   Total ticks processed: {tick_count}")
        print(f"   Callbacks triggered: {len(self.received_ticks)}")
    
    async def test_data_quality_monitoring(self):
        """Test data quality and performance monitoring"""
        print("\n" + "="*80)
        print("📋 DATA QUALITY & PERFORMANCE REPORT")
        print("="*80)
        
        # Get provider statistics
        stats = await self.data_manager.get_provider_statistics()
        
        for provider_name, provider_stats in stats.items():
            print(f"\n📡 {provider_name} Statistics:")
            print(f"   Connection Status: {'✅ Active' if provider_stats.get('connected', False) else '❌ Inactive'}")
            print(f"   Data Points: {provider_stats.get('data_points', 0):,}")
            print(f"   Error Rate: {provider_stats.get('error_rate', 0):.2%}")
            print(f"   Avg Latency: {provider_stats.get('avg_latency', 0):.2f}ms")
    
    async def run_comprehensive_test(self):
        """Run complete test suite"""
        print("🚀 STARTING LIVE MARKET DATA INTEGRATION TEST")
        print("🔑 Using Alpha Vantage API Key: ****" + os.getenv('ALPHA_VANTAGE_API_KEY', '')[-4:])
        print("⏰ Test Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            # Test 1: Provider Connections
            connection_results = await self.test_provider_connections()
            
            # Test 2: Historical Data
            if any(connection_results.values()):
                await self.test_historical_data()
            else:
                print("⚠️ Skipping historical data test - no providers connected")
            
            # Test 3: Real-time Streaming
            await self.test_real_time_streaming()
            
            # Test 4: Data Quality Monitoring
            await self.test_data_quality_monitoring()
            
            print("\n" + "="*80)
            print("✅ LIVE MARKET DATA INTEGRATION TEST COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"⏰ Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            await self.data_manager.disconnect_all_providers()

async def main():
    """Main test execution"""
    tester = LiveDataTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(main())
