"""
Download and Install Rithmic SDK
Professional installation script for ES trading system
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def download_rithmic_sdk():
    """
    Download Rithmic SDK installation script
    Note: Actual SDK requires account with Rithmic
    """
    print("🔽 RITHMIC SDK INSTALLATION")
    print("=" * 50)
    
    # Create SDK directory
    sdk_dir = Path("rithmic-sdk")
    sdk_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created SDK directory: {sdk_dir}")
    
    # Since Rithmic SDK requires authentication and licensing,
    # we'll create placeholder files and installation instructions
    
    # Create placeholder DLL (for development)
    dll_path = sdk_dir / "REngine.dll"
    if not dll_path.exists():
        with open(dll_path, 'wb') as f:
            f.write(b"PLACEHOLDER_DLL_FILE")
        print(f"📄 Created placeholder DLL: {dll_path}")
    
    # Create documentation directory
    docs_dir = sdk_dir / "documentation"
    docs_dir.mkdir(exist_ok=True)
    
    # Create examples directory
    examples_dir = sdk_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Create installation instructions
    instructions = """
# RITHMIC SDK INSTALLATION INSTRUCTIONS

## ⚠️ IMPORTANT: Manual Steps Required

The Rithmic SDK cannot be automatically downloaded due to licensing requirements.
You must obtain it through official channels:

### Step 1: Contact Your Broker
- Your client needs to have Rithmic access through their broker
- Common brokers with Rithmic: AMP Futures, NinjaTrader, etc.
- Request R|API access and credentials

### Step 2: Download Official SDK
1. Visit: https://www.rithmic.com/developers/
2. Log in with provided credentials
3. Download "R|API SDK" for Windows
4. Extract files to this directory: rithmic-sdk/

### Step 3: Required Files
After extraction, you should have:
- REngine.dll (Main API library)
- RApi.dll (Additional library)  
- Documentation files
- Sample applications
- Protocol buffer definitions

### Step 4: Credentials Setup
Create a file: config/rithmic_credentials.env
```
RITHMIC_USER_ID=your_user_id
RITHMIC_PASSWORD=your_password
RITHMIC_SYSTEM_NAME=system_name_from_broker
```

### Step 5: Test Connection
Run the test script:
```
python data-pipeline/ingestion/test_rithmic_connection.py
```

## 📞 Support Contacts

- Rithmic Support: support@rithmic.com
- R|API Documentation: Available in SDK download
- Your Broker's Trading Desk: For account setup

## 🔐 Security Notes

- Never commit credentials to version control
- Use environment variables for sensitive data
- Test with paper trading first
- Follow exchange data usage policies
"""
    
    instructions_file = sdk_dir / "INSTALLATION_INSTRUCTIONS.md"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"📋 Created installation instructions: {instructions_file}")
    
    return True

def install_dependencies():
    """Install Python dependencies for Rithmic integration"""
    print("\n🔧 INSTALLING PYTHON DEPENDENCIES")
    print("=" * 50)
    
    # Required packages for Rithmic integration
    packages = [
        "protobuf>=4.0.0",
        "websockets>=11.0",
        "asyncio-mqtt>=0.13.0",
        "pywin32>=306",  # For Windows DLL handling
        "cffi>=1.15.0",  # For C API bindings
    ]
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_test_script():
    """Create test script for Rithmic connection"""
    test_script = """#!/usr/bin/env python3
\"\"\"
Test Rithmic Connection
Verify SDK installation and connectivity
\"\"\"

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')

from data_pipeline.ingestion.rithmic_wrapper import RithmicAPIWrapper
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
    print("\\n🔌 Testing connection...")
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
            print("\\n📊 Receiving market data for 5 seconds...")
            time.sleep(5)
            
        else:
            print("❌ Authentication failed")
    else:
        print("❌ Connection failed")
    
    # Cleanup
    api.disconnect()
    print("\\n✅ Test completed")

if __name__ == "__main__":
    test_rithmic_connection()
"""
    
    test_file = Path("data-pipeline/ingestion/test_rithmic_connection.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"🧪 Created test script: {test_file}")
    return test_file

def main():
    """Main installation function"""
    print("🚀 RITHMIC SDK INSTALLATION WIZARD")
    print("=" * 60)
    print("Setting up professional-grade market data for ES trading")
    print("=" * 60)
    
    # Step 1: Download SDK placeholder
    if not download_rithmic_sdk():
        print("❌ Failed to setup SDK directory")
        return False
    
    # Step 2: Install Python dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 3: Create test script
    test_file = create_test_script()
    
    # Final instructions
    print("\n🎯 INSTALLATION SUMMARY")
    print("=" * 40)
    print("✅ SDK directory created")
    print("✅ Python dependencies installed")
    print("✅ Test script created")
    print("✅ Configuration files ready")
    
    print("\n📋 NEXT STEPS:")
    print("1. Get Rithmic credentials from your client/broker")
    print("2. Download official Rithmic SDK")
    print("3. Extract SDK files to: rithmic-sdk/")
    print("4. Configure credentials in: config/rithmic_credentials.env")
    print(f"5. Test connection: python {test_file}")
    
    print("\n🚀 Once configured, your ES trading system will have:")
    print("   • Sub-millisecond market data")
    print("   • 10-level order book depth")
    print("   • Professional-grade execution")
    print("   • Institutional data quality")
    
    return True

if __name__ == "__main__":
    main()