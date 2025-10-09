"""
Simple AddOn Connection Test
Debug the NinjaTrader AddOn connection issue
"""

import socket
import time

def test_addon_connection():
    """Test basic connection to AddOn"""
    print("🔧 DEBUGGING NINJATRADER ADDON CONNECTION")
    print("=" * 60)
    
    try:
        # Test connection
        print("📡 Testing connection to port 36974...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        result = sock.connect_ex(('localhost', 36974))
        
        if result == 0:
            print("✅ Socket connection successful!")
            
            # Test simple command
            print("📤 Sending simple test command...")
            test_command = "TEST\n"
            sock.send(test_command.encode())
            
            time.sleep(2)
            
            try:
                response = sock.recv(1024).decode()
                print(f"📥 Response: '{response}'")
                
                if response:
                    print("✅ AddOn is responding!")
                else:
                    print("❌ No response from AddOn")
                    
            except Exception as e:
                print(f"❌ Error receiving response: {e}")
            
            sock.close()
            
        else:
            print("❌ Cannot connect to port 36974")
            print("   AddOn may not be running")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print(f"\n🔍 TROUBLESHOOTING STEPS:")
    print(f"   1. Check if NinjaTrader 8 is running")
    print(f"   2. Verify AddOn is enabled in Tools → AddOns")
    print(f"   3. Check NinjaTrader Output window for AddOn startup messages")
    print(f"   4. Try restarting NinjaTrader")

def test_port_availability():
    """Test if port 36974 is available"""
    print(f"\n🔍 TESTING PORT AVAILABILITY")
    print("=" * 40)
    
    # Test if port is in use
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.bind(('localhost', 36974))
        test_sock.close()
        print("✅ Port 36974 is available")
    except OSError:
        print("⚠️ Port 36974 is in use (good - AddOn might be running)")

def test_ati_connection():
    """Test ATI connection for comparison"""
    print(f"\n📊 TESTING ATI CONNECTION (for comparison)")
    print("=" * 50)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        if sock.connect_ex(('localhost', 36973)) == 0:
            print("✅ ATI connection working on port 36973")
            
            sock.send("ASK\n".encode())
            time.sleep(1)
            
            response = sock.recv(1024).decode()
            print(f"📥 ATI response length: {len(response)} characters")
            
            sock.close()
        else:
            print("❌ ATI not responding on port 36973")
            
    except Exception as e:
        print(f"❌ ATI test error: {e}")

if __name__ == "__main__":
    test_port_availability()
    test_addon_connection()
    test_ati_connection()
    
    input("\n⏸️ Press Enter to exit...")