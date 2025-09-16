#!/usr/bin/env python3
"""
Institutional Trading System Status Check
Verifies all components are properly installed and functional
"""

import sys
import os
import importlib
import subprocess
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_imports():
    """Check critical package imports"""
    print("🔍 Checking Critical Package Imports...")
    
    critical_packages = [
        ('streamlit', 'Streamlit Dashboard'),
        ('plotly', 'Advanced Visualizations'),
        ('pandas', 'Data Processing'),
        ('numpy', 'Numerical Computing'),
        ('xgboost', 'XGBoost ML'),
        ('lightgbm', 'LightGBM ML'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch Deep Learning'),
    ]
    
    results = []
    for package, description in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package:12} - {description}")
            results.append(True)
        except ImportError:
            print(f"❌ {package:12} - {description} (MISSING)")
            results.append(False)
    
    return all(results)

def check_system_components():
    """Check system component availability"""
    print("\n🏗️ Checking System Components...")
    
    components = [
        ('data-pipeline/ingestion', 'Data Ingestion'),
        ('ml-models/training', 'ML Training'),
        ('trading-engine', 'Trading Engine'),
        ('monitoring', 'Performance Monitoring'),
        ('risk-engine', 'Risk Management'),
        ('feature-store', 'Feature Store'),
    ]
    
    results = []
    for path, description in components:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            print(f"✅ {path:25} - {description}")
            results.append(True)
        else:
            print(f"❌ {path:25} - {description} (MISSING)")
            results.append(False)
    
    return all(results)

def check_dashboard_status():
    """Check if dashboard is running"""
    print("\n📊 Checking Dashboard Status...")
    
    try:
        import requests
        response = requests.get("http://localhost:8502", timeout=5)
        if response.status_code == 200:
            print("✅ Performance Dashboard - Running on http://localhost:8502")
            return True
        else:
            print(f"⚠️ Performance Dashboard - Response Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Performance Dashboard - Not accessible ({str(e)})")
        return False

def check_institutional_features():
    """Check institutional-grade features"""
    print("\n🏛️ Checking Institutional Features...")
    
    features = [
        ('monitoring/performance_dashboard.py', 'Enhanced Dashboard'),
        ('monitoring/performance_analytics.py', 'Performance Analytics'),
        ('risk-engine/advanced_risk_manager.py', 'Advanced Risk Management'),
        ('trading-engine/multi_instrument_system.py', 'Multi-Instrument Trading'),
        ('ml-models/training/institutional_trading_system.py', 'Institutional ML System'),
    ]
    
    results = []
    for path, description in features:
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            print(f"✅ {description}")
            results.append(True)
        else:
            print(f"❌ {description} (MISSING)")
            results.append(False)
    
    return all(results)

def main():
    """Main status check"""
    print("=" * 60)
    print("🏛️ INSTITUTIONAL ML TRADING SYSTEM STATUS CHECK")
    print("=" * 60)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project Root: {project_root}")
    print("=" * 60)
    
    # Run all checks
    imports_ok = check_imports()
    components_ok = check_system_components()
    dashboard_ok = check_dashboard_status()
    institutional_ok = check_institutional_features()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SYSTEM STATUS SUMMARY")
    print("=" * 60)
    
    checks = [
        (imports_ok, "Critical Package Imports"),
        (components_ok, "System Components"),
        (dashboard_ok, "Performance Dashboard"),
        (institutional_ok, "Institutional Features"),
    ]
    
    all_passed = True
    for passed, description in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} - {description}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 SYSTEM STATUS: ✅ ALL SYSTEMS OPERATIONAL")
        print("🚀 Ready for institutional trading operations!")
    else:
        print("⚠️ SYSTEM STATUS: ❌ ISSUES DETECTED")
        print("🔧 Please address the failed checks above.")
    
    print("=" * 60)
    print(f"🌐 Dashboard URL: http://localhost:8502")
    print(f"📊 Enhanced Features: Institutional-Grade Analytics")
    print(f"🛡️ Risk Management: Advanced Multi-Method VaR")
    print(f"📈 Multi-Instrument: ES/NQ/YM/RTY Futures Support")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    main()
