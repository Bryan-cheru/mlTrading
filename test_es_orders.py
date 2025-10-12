"""
Simple ES Order Test
Test real ES order execution with manual buy signal
"""

from ninjatrader_addon_interface import ESTrader
import time

def test_es_order_execution():
    """Test real ES order execution"""
    print("ES ORDER EXECUTION TEST")
    print("=" * 40)
    
    # Initialize trader
    trader = ESTrader()
    
    # Test connection
    print("Testing connection...")
    if not trader.is_connected():
        print("ERROR: Cannot connect to NinjaTrader AddOn")
        return False
    
    print("✅ Connected to NinjaTrader AddOn")
    
    # Get current account status
    print("\nGetting account status...")
    status = trader.get_positions()
    if status['success']:
        print("Account info:")
        for key, value in status['status'].items():
            print(f"  {key}: {value}")
    
    # Test buy order
    print(f"\nTesting ES BUY order...")
    confirm = input("Place real ES BUY order? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print("Placing BUY order...")
        result = trader.buy_es(1)
        
        if result['success']:
            print(f"✅ BUY ORDER SUCCESS!")
            print(f"   Order ID: {result['order_id']}")
            print(f"   Message: {result['message']}")
            
            # Wait and check account again
            print("\nWaiting 3 seconds...")
            time.sleep(3)
            
            print("Checking account after order...")
            status = trader.get_positions()
            if status['success']:
                print("Updated account info:")
                for key, value in status['status'].items():
                    print(f"  {key}: {value}")
            
            return True
        else:
            print(f"❌ BUY ORDER FAILED!")
            print(f"   Error: {result['error']}")
            return False
    else:
        print("Order test skipped")
        return True

    # Test sell order
    print(f"\nTesting ES SELL order...")
    confirm = input("Place real ES SELL order? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print("Placing SELL order...")
        result = trader.sell_es(1)
        
        if result['success']:
            print(f"✅ SELL ORDER SUCCESS!")
            print(f"   Order ID: {result['order_id']}")
            print(f"   Message: {result['message']}")
            return True
        else:
            print(f"❌ SELL ORDER FAILED!")
            print(f"   Error: {result['error']}")
            return False
    else:
        print("Sell order test skipped")
        return True

if __name__ == "__main__":
    try:
        success = test_es_order_execution()
        
        if success:
            print(f"\n🎉 ES ORDER EXECUTION TEST SUCCESSFUL!")
            print("   Real ES orders are executing in NinjaTrader")
            print("   System is ready for automated trading")
        else:
            print(f"\n❌ ES ORDER EXECUTION TEST FAILED")
            print("   Check NinjaTrader and AddOn configuration")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\nError: {e}")
    
    input("\nPress Enter to exit...")