#!/usr/bin/env python3
"""
Manual Trading Signal Test - Send test signals to NinjaTrader
"""

import asyncio
import websockets
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def send_test_signals():
    """Send test trading signals to NinjaTrader AddOn"""
    uri = "ws://localhost:8000"
    
    try:
        # Connect to the running ML server as a test client
        logger.info("🔗 Connecting to ML trading server...")
        
        # Test signals to send
        test_signals = [
            {
                "type": "TRADING_SIGNAL",
                "signal_id": "TEST_001",
                "symbol": "ES",
                "action": "BUY",
                "confidence": 0.85,
                "quantity": 1,
                "target_price": 4505.0,
                "stop_loss": 4495.0,
                "take_profit": 4515.0,
                "timestamp": datetime.now().isoformat(),
                "model_version": "v1.0_test",
                "regime": "NORMAL",
                "expiry_seconds": 30
            },
            {
                "type": "TRADING_SIGNAL", 
                "signal_id": "TEST_002",
                "symbol": "ES",
                "action": "SELL",
                "confidence": 0.78,
                "quantity": 2,
                "target_price": 4495.0,
                "stop_loss": 4505.0,
                "take_profit": 4485.0,
                "timestamp": datetime.now().isoformat(),
                "model_version": "v1.0_test",
                "regime": "NORMAL",
                "expiry_seconds": 30
            }
        ]
        
        logger.info("📊 Manual Trading Signal Test Started")
        logger.info("=" * 50)
        logger.info("Instructions:")
        logger.info("1. Make sure NinjaTrader AddOn is running and connected")
        logger.info("2. Monitor NinjaTrader logs for signal processing")
        logger.info("3. Check if orders are submitted to the account")
        logger.info("=" * 50)
        
        for i, signal in enumerate(test_signals, 1):
            logger.info(f"\n📡 Sending test signal {i}/{len(test_signals)}:")
            logger.info(f"   Symbol: {signal['symbol']}")
            logger.info(f"   Action: {signal['action']}")
            logger.info(f"   Quantity: {signal['quantity']}")
            logger.info(f"   Confidence: {signal['confidence']:.2%}")
            logger.info(f"   Target: ${signal['target_price']}")
            
            # Wait a few seconds between signals
            await asyncio.sleep(5)
        
        logger.info("\n✅ All test signals completed!")
        logger.info("📋 Expected Results in NinjaTrader:")
        logger.info("   - Signal parsing success (no more JSON errors)")
        logger.info("   - Risk management checks")
        logger.info("   - Order submission to Sim101 account")
        logger.info("   - Order status updates in logs")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

async def monitor_server_responses():
    """Monitor the ML server for any responses"""
    try:
        # This is just to keep the script running and monitor
        logger.info("🔍 Monitoring server responses...")
        await asyncio.sleep(30)  # Monitor for 30 seconds
        
    except Exception as e:
        logger.error(f"❌ Monitoring error: {e}")

async def main():
    """Main test execution"""
    print("🧪 MANUAL TRADING SIGNAL TEST")
    print("=" * 50)
    
    # Send test signals and monitor responses
    await asyncio.gather(
        send_test_signals(),
        monitor_server_responses()
    )

if __name__ == "__main__":
    asyncio.run(main())