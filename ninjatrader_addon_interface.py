"""
NinjaTrader AddOn Interface
Communicates with ESOrderExecutor AddOn for real order execution
"""

import socket
import time
from datetime import datetime
import logging
from typing import Optional, Dict, Any

class NinjaTraderAddOnInterface:
    """Interface to communicate with NinjaTrader ESOrderExecutor AddOn"""
    
    def __init__(self, host='localhost', port=36974):
        self.host = host
        self.port = port
        self.timeout = 10
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command to NinjaTrader AddOn and return response"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                
                # Send command
                sock.send((command + '\n').encode())
                
                # Receive response
                response = sock.recv(4096).decode().strip()
                
                self.logger.info(f"Sent: {command}")
                self.logger.info(f"Received: {response}")
                
                return response
                
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return None
    
    def place_es_order(self, order_id: str, side: str, quantity: int, 
                      order_type: str = "MARKET") -> Dict[str, Any]:
        """
        Place ES futures order
        
        Args:
            order_id: Unique order identifier
            side: "BUY" or "SELL"
            quantity: Number of contracts
            order_type: "MARKET", "LIMIT", etc.
        
        Returns:
            Dictionary with success status and message
        """
        command = f"PLACE_ORDER|{order_id}|ES|{side}|{quantity}|{order_type}"
        response = self.send_command(command)
        
        if response:
            if "SUCCESS" in response:
                return {
                    'success': True,
                    'message': response,
                    'order_id': order_id
                }
            else:
                return {
                    'success': False,
                    'error': response,
                    'order_id': order_id
                }
        else:
            return {
                'success': False,
                'error': 'No response from NinjaTrader AddOn',
                'order_id': order_id
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order by ID"""
        command = f"CANCEL_ORDER|{order_id}|DUMMY|DUMMY|1|DUMMY"
        response = self.send_command(command)
        
        if response and "SUCCESS" in response:
            return {'success': True, 'message': response}
        else:
            return {'success': False, 'error': response or 'Cancel failed'}
    
    def get_account_status(self) -> Dict[str, Any]:
        """Get account status and positions"""
        command = "STATUS|DUMMY_ID|DUMMY|DUMMY|1|DUMMY"
        response = self.send_command(command)
        
        if response and "ACCOUNT_STATUS" in response:
            # Parse response
            parts = response.split('|')
            status = {}
            
            for part in parts[1:]:  # Skip ACCOUNT_STATUS
                if ':' in part:
                    key, value = part.split(':', 1)
                    status[key] = value
            
            return {'success': True, 'status': status}
        else:
            return {'success': False, 'error': response or 'Status request failed'}
    
    def test_connection(self) -> bool:
        """Test if AddOn is responding"""
        try:
            result = self.get_account_status()
            return result['success']
        except:
            return False


class ESTrader:
    """High-level ES trading interface using the AddOn"""
    
    def __init__(self):
        self.addon = NinjaTraderAddOnInterface()
        self.order_counter = 0
        
    def generate_order_id(self, prefix: str = "ES") -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        timestamp = int(time.time())
        return f"{prefix}_{timestamp}_{self.order_counter}"
    
    def buy_es(self, quantity: int = 1) -> Dict[str, Any]:
        """Place ES buy order"""
        order_id = self.generate_order_id("ES_BUY")
        return self.addon.place_es_order(order_id, "BUY", quantity, "MARKET")
    
    def sell_es(self, quantity: int = 1) -> Dict[str, Any]:
        """Place ES sell order"""
        order_id = self.generate_order_id("ES_SELL")
        return self.addon.place_es_order(order_id, "SELL", quantity, "MARKET")
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self.addon.get_account_status()
    
    def is_connected(self) -> bool:
        """Check if connected to NinjaTrader"""
        return self.addon.test_connection()


def test_addon_integration():
    """Test the AddOn integration"""
    print("🧪 TESTING NINJATRADER ADDON INTEGRATION")
    print("=" * 60)
    
    trader = ESTrader()
    
    # Test 1: Connection
    print("📡 Testing connection...")
    if trader.is_connected():
        print("✅ Connected to NinjaTrader AddOn")
        print("   ESOrderExecutor is running and responding")
    else:
        print("❌ Cannot connect to AddOn")
        print("   ESOrderExecutor AddOn is not responding on port 36974")
        print("\n🔧 Setup Required:")
        print("   1. Open NinjaTrader 8")
        print("   2. Press F11 (NinjaScript Editor)")
        print("   3. Compile ESOrderExecutor AddOn (F5)")
        print("   4. Enable in Tools → AddOns")
        print("   5. Restart NinjaTrader")
        print("   6. Check Output window for startup messages")
        print("\n📚 See: NINJATRADER_SETUP_GUIDE.md for detailed steps")
        return False
    
    # Test 2: Account status
    print("\n📊 Getting account status...")
    status = trader.get_positions()
    if status['success']:
        print("✅ Account status retrieved:")
        for key, value in status['status'].items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Failed to get status: {status['error']}")
    
    # Test 3: Place order
    print(f"\n🎯 Placing test ES buy order...")
    confirm = input("❓ Place real ES buy order? (y/n): ").strip().lower()
    
    if confirm == 'y':
        result = trader.buy_es(1)
        
        if result['success']:
            print(f"✅ ORDER SUCCESSFUL!")
            print(f"   {result['message']}")
            print(f"   Order ID: {result['order_id']}")
            print(f"   Check NinjaTrader Orders tab")
            return True
        else:
            print(f"❌ ORDER FAILED!")
            print(f"   Error: {result['error']}")
            return False
    else:
        print("⏸️ Order test skipped")
        return True


if __name__ == "__main__":
    try:
        success = test_addon_integration()
        
        if success:
            print(f"\n🎉 ADDON INTEGRATION SUCCESSFUL!")
            print("   NinjaTrader AddOn is working correctly")
            print("   Can proceed with automated ES trading")
        else:
            print(f"\n❌ ADDON INTEGRATION FAILED")
            print("   Check AddOn installation and configuration")
            
    except KeyboardInterrupt:
        print("\n\n⏸️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\n⏸️ Press Enter to exit...")
