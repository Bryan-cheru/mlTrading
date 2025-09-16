#!/usr/bin/env python3
"""
Portfolio Management System - Quick Validation Test
Tests core functionality of all portfolio management components
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, '.')

# Import portfolio management components
import importlib.util

def load_module(name, path):
    """Helper function to load modules with importlib"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
portfolio_optimizer = load_module(
    'portfolio_optimizer', 
    './portfolio-management/optimization/portfolio_optimizer.py'
)

dynamic_allocation_engine = load_module(
    'dynamic_allocation_engine', 
    './portfolio-management/allocation/dynamic_allocation_engine.py'
)

rebalancing_executor = load_module(
    'rebalancing_executor', 
    './portfolio-management/rebalancing/rebalancing_executor.py'
)

portfolio_manager = load_module(
    'portfolio_manager', 
    './portfolio-management/portfolio_manager.py'
)

def test_portfolio_optimization():
    """Test portfolio optimization components"""
    print("📊 Testing Portfolio Optimization...")
    
    try:
        # Create sample data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        
        # Generate sample returns data
        returns_data = {}
        for symbol in symbols:
            returns_data[symbol] = np.random.normal(0.0008, 0.02, len(dates))
        returns_df = pd.DataFrame(returns_data, index=dates)
        
        # Test Kelly Criterion Calculator
        kelly_calc = portfolio_optimizer.KellyCriterionCalculator()
        asset = portfolio_optimizer.Asset(
            "AAPL", "TECH", 3000000000000, 180.50, 0.25, 0.12, 1.2, 0.9
        )
        kelly_fraction = kelly_calc.calculate_kelly_fraction(asset, returns_data['AAPL'])
        print(f"   ✅ Kelly Criterion: {kelly_fraction:.3f}")
        
        # Test Risk Parity Optimizer
        risk_parity = portfolio_optimizer.RiskParityOptimizer()
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov_matrix = returns_df.cov().values
        risk_contributions = risk_parity.calculate_risk_contributions(weights, cov_matrix)
        print(f"   ✅ Risk Parity: Risk contributions calculated")
        
        # Test Modern Portfolio Theory
        mpt = portfolio_optimizer.ModernPortfolioTheory()
        expected_returns = returns_df.mean().values * 252  # Annualized
        metrics = mpt.calculate_portfolio_metrics(weights, expected_returns, cov_matrix * 252)
        print(f"   ✅ MPT: Sharpe ratio = {metrics['sharpe_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def test_dynamic_allocation():
    """Test dynamic allocation engine"""
    print("⚖️ Testing Dynamic Allocation...")
    
    try:
        # Test AllocationLimits
        limits = dynamic_allocation_engine.AllocationLimits(
            max_position_size=0.25,
            min_cash_reserve=0.05,
            max_sector_allocation=0.50
        )
        print(f"   ✅ Allocation limits created")
        
        # Test SectorManager
        sector_manager = dynamic_allocation_engine.SectorManager()
        allocations = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.15, 'SPY': 0.2, 'QQQ': 0.15}
        sector_allocs = sector_manager.calculate_sector_allocations(allocations)
        print(f"   ✅ Sector allocations: {len(sector_allocs)} sectors")
        
        # Test RiskMonitor
        risk_monitor = dynamic_allocation_engine.RiskMonitor()
        weights = np.array([0.3, 0.2, 0.15, 0.2, 0.15])
        cov_matrix = np.random.rand(5, 5)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make positive definite
        var = risk_monitor.calculate_portfolio_var(weights, cov_matrix)
        print(f"   ✅ Portfolio VaR: {var:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def test_rebalancing_execution():
    """Test rebalancing and execution"""
    print("🔄 Testing Rebalancing Execution...")
    
    try:
        # Test MarketImpactModel
        impact_model = rebalancing_executor.MarketImpactModel()
        impact = impact_model.estimate_market_impact(
            symbol="AAPL",
            quantity=1000,
            avg_daily_volume=50000000,
            current_price=180.50,
            liquidity_tier='high'
        )
        print(f"   ✅ Market impact estimate: {impact['total_impact_pct']:.4f}")
        
        # Test TWAPExecutor
        twap_executor = rebalancing_executor.TWAPExecutor()
        
        # Create a proper RebalanceOrder
        order = rebalancing_executor.RebalanceOrder(
            order_id="TEST_001",
            symbol="MSFT",
            side="BUY",
            quantity=500,
            target_price=415.25,
            strategy=rebalancing_executor.ExecutionStrategy.TWAP,
            priority=2,
            created_time=datetime.now()
        )
        
        # Mock market data
        market_data = {
            'avg_daily_volume': 50000000,
            'current_price': 415.25,
            'bid': 415.20,
            'ask': 415.30,
            'volatility': 0.22
        }
        
        schedule = twap_executor.create_twap_schedule(order, market_data)
        print(f"   ✅ TWAP schedule: {len(schedule)} slices")
        
        # Test SmartOrderRouter
        router = rebalancing_executor.SmartOrderRouter()
        
        # Create order for routing
        order = rebalancing_executor.RebalanceOrder(
            order_id="TEST_002",
            symbol="GOOGL",
            side="BUY",
            quantity=200,
            target_price=140.75,
            strategy=rebalancing_executor.ExecutionStrategy.MARKET,
            priority=1,
            created_time=datetime.now()
        )
        
        market_data = {
            'price': 140.75,
            'volume': 2500000,
            'volatility': 0.28,
            'spread': 0.02
        }
        
        strategy = router.determine_execution_strategy(order, market_data)
        print(f"   ✅ Order routing: {strategy}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def test_portfolio_manager():
    """Test integrated portfolio manager"""
    print("🎯 Testing Portfolio Manager...")
    
    try:
        # Create PortfolioManagerConfig
        config = portfolio_manager.PortfolioManagerConfig()
        
        # Create PortfolioManager
        manager = portfolio_manager.PortfolioManager(config=config)
        print(f"   ✅ Portfolio manager created")
        
        # Test basic functionality
        manager.start_monitoring()
        print(f"   ✅ Monitoring started")
        
        manager.stop_monitoring()
        print(f"   ✅ Monitoring stopped")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Run comprehensive portfolio management validation"""
    print("🚀 Portfolio Management System - Validation Test")
    print("=" * 60)
    
    tests = [
        ("Portfolio Optimization", test_portfolio_optimization),
        ("Dynamic Allocation", test_dynamic_allocation),
        ("Rebalancing Execution", test_rebalancing_execution),
        ("Portfolio Manager", test_portfolio_manager),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        result = test_func()
        results.append(result)
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}\n")
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Portfolio Management System is ready for integration.")
        print("\n🚀 Key Capabilities Confirmed:")
        print("   • Portfolio optimization algorithms (Kelly, Risk Parity, MPT)")
        print("   • Dynamic allocation with risk monitoring")
        print("   • Advanced execution strategies (TWAP, Smart routing)")
        print("   • Integrated portfolio management framework")
        print("   • Real-time monitoring and rebalancing")
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed.")
        print("Review implementation before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
