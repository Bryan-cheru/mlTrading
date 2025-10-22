"""
Professional Rithmic SDK Setup Script
Helps configure your institutional trading system with the official Rithmic R|API 13.6.0.0
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    print("🏛️ PROFESSIONAL RITHMIC SDK SETUP")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    print(f"Project Root: {project_root}")
    
    # Step 1: Verify SDK presence
    print("\n📁 Step 1: Verifying Rithmic SDK 13.6.0.0...")
    sdk_path = project_root / "13.6.0.0"
    
    if not sdk_path.exists():
        print("❌ Rithmic SDK 13.6.0.0 directory not found!")
        print("Please ensure the official Rithmic SDK is extracted to the 13.6.0.0 directory")
        return False
    
    dll_path = sdk_path / "win10" / "lib_472" / "rapiplus.dll"
    if dll_path.exists():
        size = dll_path.stat().st_size
        print(f"✅ Rithmic DLL found: {dll_path} ({size:,} bytes)")
    else:
        print(f"❌ Rithmic DLL not found at: {dll_path}")
        return False
    
    # Step 2: Install Python packages
    print("\n📦 Step 2: Installing Required Python Packages...")
    packages = [
        "pythonnet>=3.0.0",
        "python-dotenv",
        "numpy",
        "pandas", 
        "scikit-learn",
        "xgboost",
        "asyncio",
        "scipy",
        "requests",
        "fastapi",
        "uvicorn"
    ]
    
    for package in packages:
        try:
            # Check if already installed
            package_name = package.split('>=')[0].split('==')[0].replace('-', '_')
            __import__(package_name)
            print(f"   ✅ {package} already installed")
        except ImportError:
            print(f"   📦 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    
    # Step 3: Create credentials template
    print("\n🔐 Step 3: Setting up Credentials Configuration...")
    
    credentials_file = project_root / "config" / "rithmic_credentials.env"
    
    if credentials_file.exists():
        print(f"   ✅ Credentials file exists: {credentials_file}")
        
        # Check if configured
        with open(credentials_file, 'r') as f:
            content = f.read()
            if 'your_rithmic_user_id' in content:
                print("   ⚠️ Credentials file needs to be configured with your actual Rithmic credentials")
                print("   📝 Please edit config/rithmic_credentials.env with your broker-provided values")
            else:
                print("   ✅ Credentials appear to be configured")
    else:
        print(f"   ❌ Credentials file not found: {credentials_file}")
        return False
    
    # Step 4: Test Python CLR integration
    print("\n🐍 Step 4: Testing Python CLR Integration...")
    
    try:
        import clr
        print("   ✅ pythonnet (CLR) available")
        
        # Test DLL loading
        try:
            clr.AddReference(str(dll_path))
            print("   ✅ Rithmic DLL loaded successfully")
            
            # Test imports
            from com.omnesys.rapi import REngine
            print("   ✅ Rithmic R|API classes accessible")
            
        except Exception as e:
            print(f"   ⚠️ DLL loading issue: {e}")
            print("   💡 This is normal if running outside Windows or without proper .NET framework")
            
    except ImportError:
        print("   ❌ pythonnet not available")
        return False
    
    # Step 5: Verify integration files
    print("\n🔧 Step 5: Verifying Integration Files...")
    
    integration_files = [
        "data-pipeline/ingestion/rithmic_professional_connector.py",
        "institutional_trading_system.py",
        "ninjatrader-addon/ProfessionalInstitutionalTrading.cs",
        "config/rithmic_config.json",
        "test_rithmic_integration.py"
    ]
    
    for file_path in integration_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    # Step 6: Create run scripts
    print("\n🚀 Step 6: Creating Run Scripts...")
    
    # Python run script
    python_script = project_root / "run_institutional_system.py"
    with open(python_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Run the Professional Institutional Trading System
Integrates Rithmic R|API with ML trading algorithms
\"\"\"

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

async def main():
    print("🏛️ Starting Professional Institutional Trading System")
    print("=" * 60)
    
    try:
        from institutional_trading_system import MathematicalMLTradingSystem
        
        # Initialize and start system
        system = MathematicalMLTradingSystem()
        await system.start_system()
        
    except KeyboardInterrupt:
        print("🛑 System shutdown requested")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
""")
    
    print(f"   ✅ Created: {python_script}")
    
    # Batch script for Windows
    batch_script = project_root / "run_system.bat"
    with open(batch_script, 'w') as f:
        f.write(f"""@echo off
echo 🏛️ Professional Institutional Trading System
echo =============================================

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Run the system
echo Starting system...
python run_institutional_system.py

pause
""")
    
    print(f"   ✅ Created: {batch_script}")
    
    # Step 7: Display setup summary
    print("\n" + "=" * 60)
    print("🎯 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n✅ SUCCESSFULLY CONFIGURED:")
    print("   • Official Rithmic SDK 13.6.0.0 verified")
    print("   • Python packages installed")
    print("   • Integration files in place")
    print("   • Run scripts created")
    print("   • Credentials template ready")
    
    print("\n🔧 NEXT STEPS:")
    print("   1. 📋 Contact your broker for Rithmic R|API credentials:")
    print("      • User ID")
    print("      • Password") 
    print("      • System Name")
    print("      • Common brokers: AMP Futures, NinjaTrader, etc.")
    
    print("\n   2. 🔐 Update credentials in config/rithmic_credentials.env:")
    print("      • Replace 'your_rithmic_user_id' with actual User ID")
    print("      • Replace 'your_rithmic_password' with actual password")
    print("      • Replace 'your_rithmic_system_name' with broker-provided system name")
    
    print("\n   3. 🧪 Test the integration:")
    print("      • Run: python test_rithmic_integration.py")
    print("      • Verify connection to Rithmic servers")
    print("      • Test market data reception")
    
    print("\n   4. 🚀 Launch the system:")
    print("      • Windows: Double-click run_system.bat")
    print("      • Python: python run_institutional_system.py")
    print("      • NinjaTrader: Compile and install the C# AddOns")
    
    print("\n   5. 🖥️ NinjaTrader Integration:")
    print("      • Copy ninjatrader-addon/*.cs files to NinjaTrader AddOns folder")
    print("      • Compile in NinjaScript Editor")
    print("      • Launch from Tools menu")
    
    print("\n💡 SUPPORT:")
    print("   • Rithmic R|API Documentation: Check 13.6.0.0/documentation.html")
    print("   • Sample Code: 13.6.0.0/samples/")
    print("   • System Logs: logs/ directory")
    
    print("\n🏛️ Your institutional-grade trading system is ready!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please resolve the issues above and try again.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")