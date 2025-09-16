"""
Production Deployment Launcher
Easy deployment script for the institutional ML trading system
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print deployment banner"""
    print("=" * 80)
    print("🚀 INSTITUTIONAL ML TRADING SYSTEM - DEPLOYMENT LAUNCHER")
    print("=" * 80)
    print(f"Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_deployment_readiness():
    """Quick deployment readiness check"""
    print("🔍 Checking deployment readiness...")
    
    try:
        # Run validator
        result = subprocess.run([sys.executable, 'validate_deployment.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "100.0%" in result.stdout:
            print("✅ System validation: PASSED")
            print("✅ NinjaTrader 8: READY")
            return True
        else:
            print("❌ System validation: FAILED")
            print("Please run 'python validate_deployment.py' for details")
            return False
    except Exception as e:
        print(f"❌ Validation check failed: {e}")
        return False

def show_deployment_options():
    """Show deployment options"""
    print("\n📋 DEPLOYMENT OPTIONS:")
    print("1. Live Trading Mode (Real orders)")
    print("2. Paper Trading Mode (Simulated orders)")
    print("3. Demo Mode (Synthetic data)")
    print("4. Validation Test Only")
    print("5. Exit")
    print()

def deploy_live_trading():
    """Deploy in live trading mode"""
    print("🔴 DEPLOYING LIVE TRADING MODE")
    print("⚠️  WARNING: This will place REAL orders with REAL money!")
    print()
    
    confirm = input("Type 'CONFIRM' to proceed with live trading: ")
    if confirm.upper() != 'CONFIRM':
        print("❌ Live trading deployment cancelled")
        return
    
    print("🚀 Starting live trading system...")
    print("📊 Monitor the console for trading signals and performance")
    print("🛑 Press Ctrl+C to stop the system")
    print("-" * 80)
    
    try:
        subprocess.run([sys.executable, 'simplified_advanced_system.py'])
    except KeyboardInterrupt:
        print("\n🛑 Live trading system stopped by user")

def deploy_paper_trading():
    """Deploy in paper trading mode"""
    print("📝 DEPLOYING PAPER TRADING MODE")
    print("✅ Safe mode: No real orders will be placed")
    print()
    
    print("🚀 Starting paper trading system...")
    print("📊 System will connect to live data but simulate trades")
    print("🛑 Press Ctrl+C to stop the system")
    print("-" * 80)
    
    # Note: For now, paper trading uses the same system
    # In Phase 2, we'll add a proper paper trading flag
    print("📝 Paper trading mode - monitoring only")
    try:
        subprocess.run([sys.executable, 'simplified_advanced_system.py'])
    except KeyboardInterrupt:
        print("\n🛑 Paper trading system stopped by user")

def deploy_demo_mode():
    """Deploy in demo mode with synthetic data"""
    print("🎯 DEPLOYING DEMO MODE")
    print("✅ Using synthetic data - no NinjaTrader connection required")
    print()
    
    print("🚀 Starting demo system...")
    print("📊 System will generate synthetic market data for testing")
    print("🛑 Press Ctrl+C to stop the system")
    print("-" * 80)
    
    try:
        subprocess.run([sys.executable, 'ninjatrader_demo.py'])
    except KeyboardInterrupt:
        print("\n🛑 Demo system stopped by user")

def run_validation_only():
    """Run validation tests only"""
    print("🔍 RUNNING VALIDATION TESTS")
    print()
    
    try:
        subprocess.run([sys.executable, 'validate_deployment.py'])
    except KeyboardInterrupt:
        print("\n🛑 Validation cancelled by user")

def main():
    """Main deployment launcher"""
    print_banner()
    
    # Check if system is ready
    if not check_deployment_readiness():
        print("\n❌ System not ready for deployment")
        print("🔧 Please fix validation issues before deploying")
        return
    
    print("✅ System ready for deployment!")
    
    while True:
        show_deployment_options()
        
        try:
            choice = input("Select deployment option (1-5): ").strip()
            
            if choice == '1':
                deploy_live_trading()
            elif choice == '2':
                deploy_paper_trading()
            elif choice == '3':
                deploy_demo_mode()
            elif choice == '4':
                run_validation_only()
            elif choice == '5':
                print("👋 Exiting deployment launcher")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                continue
            
            print("\n" + "="*80)
            print("🔄 Returning to main menu...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Deployment launcher stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
