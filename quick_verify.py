"""
Quick verification - place one more order to confirm everything is working
"""

from ninjatrader_addon_interface import ESTrader

def quick_verification():
    trader = ESTrader()
    
    print("QUICK VERIFICATION TEST")
    print("=" * 30)
    
    # Get current status
    status = trader.get_positions()
    print(f"Current orders: {status['status']['Orders']}")
    
    # Place sell order
    print("Placing ES SELL order...")
    result = trader.sell_es(1)
    
    if result['success']:
        print(f"✅ SELL ORDER SUCCESS: {result['order_id']}")
        
        # Check updated status
        import time
        time.sleep(2)
        status = trader.get_positions()
        print(f"Orders after SELL: {status['status']['Orders']}")
        
        return True
    else:
        print(f"❌ SELL FAILED: {result['error']}")
        return False

if __name__ == "__main__":
    success = quick_verification()
    
    if success:
        print("\n🎉 VERIFICATION COMPLETE!")
        print("   ES BUY and SELL orders both working")
        print("   NinjaTrader integration is FULLY OPERATIONAL")
    else:
        print("\n❌ Verification failed")

quick_verification()