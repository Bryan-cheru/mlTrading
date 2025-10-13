#!/usr/bin/env python3
"""
Test Rithmic Connection
Verify SDK installation and connectivity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')

from rithmic_wrapper import RithmicAPIWrapper
import time

def test_rithmic_connection():
    print("🧪 TESTING RITHMIC CONNECTION")
    print("=" * 40)
    
    # Create API wrapper
    api = RithmicAPIWrapper()
    
    # Test basic functionality
    print("📋 Connection Status:")
    status = api.get_connection_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test callbacks
    def on_tick(tick_data):
        print(f"📊 Received tick: {tick_data['symbol']} ${tick_data['price']:.2f}")
    
    def on_orderbook(book_data):
        bids = book_data['bids']
        asks = book_data['asks']
        if bids and asks:
            spread = asks[0]['price'] - bids[0]['price']
            print(f"📖 Spread: ${spread:.2f}")
    
    # Register callbacks
    api.register_tick_callback(on_tick)
    api.register_orderbook_callback(on_orderbook)
    
    # Test connection
    print("\n🔌 Testing connection...")
    if api.connect():
        print("✅ Connection successful")
        
        # Test authentication (with dummy credentials)
        if api.authenticate("test_user", "test_pass"):
            print("✅ Authentication successful")
            
            # Test subscriptions
            if api.subscribe_market_data("ESZ4"):
                print("✅ Market data subscription successful")
            
            if api.subscribe_market_depth("ESZ4"):
                print("✅ Market depth subscription successful")
            
            # Run for 5 seconds to see data
            print("\n📊 Receiving market data for 5 seconds...")
            time.sleep(5)
            
        else:
            print("❌ Authentication failed")
    else:
        print("❌ Connection failed")
    
    # Cleanup
    api.disconnect()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_rithmic_connection()
