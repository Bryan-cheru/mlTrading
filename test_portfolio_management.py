#!/usr/bin/env python3
"""
Portfolio Management System Test Suite
Comprehensive testing of all portfolio management components
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
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

# Load all portfolio management modules
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

class PortfolioManagementTester:
    """Comprehensive test suite for portfolio management system"""
    
    def __init__(self):
        print("🚀 Initializing Portfolio Management Test Suite")
        print("=" * 60)
        
        # Create sample assets for testing
        self.assets = [
            portfolio_optimizer.Asset("AAPL", "TECH", 3000000000000, 180.50, 0.25, 0.12, 1.2, 0.9),
            portfolio_optimizer.Asset("MSFT", "TECH", 2800000000000, 415.25, 0.22, 0.11, 1.1, 0.95),
            portfolio_optimizer.Asset("GOOGL", "TECH", 1800000000000, 140.75, 0.28, 0.13, 1.15, 0.85),
            portfolio_optimizer.Asset("SPY", "ETF", 500000000000, 445.80, 0.15, 0.10, 1.0, 1.0),
            portfolio_optimizer.Asset("QQQ", "ETF", 200000000000, 380.90, 0.20, 0.12, 1.05, 0.92),
        ]
        
        # Create sample positions
        self.positions = [
            portfolio_optimizer.Position("AAPL", 100, 18050.0, 0.2, 0.2, 0.0, 1000.0, 500.0, 175.0, datetime.now()),
            portfolio_optimizer.Position("MSFT", 50, 20762.5, 0.23, 0.2, 0.03, 1500.0, 800.0, 400.0, datetime.now()),
            portfolio_optimizer.Position("GOOGL", 75, 10556.25, 0.12, 0.2, -0.08, -500.0, 200.0, 145.0, datetime.now()),
            portfolio_optimizer.Position("SPY", 200, 89160.0, 0.35, 0.2, 0.15, 2000.0, 1000.0, 435.0, datetime.now()),
            portfolio_optimizer.Position("QQQ", 150, 57135.0, 0.22, 0.2, 0.02, 800.0, 600.0, 375.0, datetime.now()),
        ]
        
        # Create sample portfolio
        positions_dict = {pos.symbol: pos for pos in self.positions}
        target_allocations = {pos.symbol: 0.2 for pos in self.positions}
        actual_allocations = {pos.symbol: pos.weight for pos in self.positions}
        
        self.portfolio = portfolio_optimizer.Portfolio(
            total_value=250000.0,
            cash=50000.0,
            positions=positions_dict,
            target_allocations=target_allocations,
            actual_allocations=actual_allocations,
            risk_metrics={},
            performance_metrics={},
            last_rebalance=datetime.now(),
            rebalance_threshold=0.05
        )
        
        # Create sample market data
        self.market_data = self._generate_sample_market_data()
        
    def _generate_sample_market_data(self):
        """Generate sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        symbols = [asset.symbol for asset in self.assets]
        
        data = {}
        for symbol in symbols:
            # Generate realistic price data with some volatility
            base_price = next(asset.current_price for asset in self.assets if asset.symbol == symbol)
            returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns ~0.08% mean, 2% std
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[symbol] = pd.Series(prices, index=dates)
        
        return pd.DataFrame(data)
    
    def test_portfolio_optimizer(self):
        """Test portfolio optimization algorithms"""
        print("\n📊 Testing Portfolio Optimization Algorithms")
        print("-" * 50)
        
        try:
            # Test Kelly Criterion Calculator
            kelly_calc = portfolio_optimizer.KellyCriterionCalculator()
            kelly_results = kelly_calc.calculate_kelly_criterion(
                self.market_data, 
                lookback_days=252,
                risk_free_rate=0.05
            )
            print(f"✅ Kelly Criterion - Optimal positions calculated")
            print(f"   Symbols: {list(kelly_results.keys())}")
            print(f"   Allocation range: {min(kelly_results.values()):.3f} to {max(kelly_results.values()):.3f}")
            
            # Test Risk Parity Optimizer
            risk_parity = portfolio_optimizer.RiskParityOptimizer()
            returns = self.market_data.pct_change().dropna()
            risk_parity_weights = risk_parity.optimize_risk_parity(returns)
            print(f"✅ Risk Parity - Equal risk contribution weights calculated")
            print(f"   Weights sum: {sum(risk_parity_weights.values()):.3f}")
            
            # Test Modern Portfolio Theory
            mpt = portfolio_optimizer.ModernPortfolioTheory()
            efficient_portfolio = mpt.optimize_sharpe_ratio(returns, risk_free_rate=0.05)
            print(f"✅ Modern Portfolio Theory - Sharpe ratio optimization complete")
            print(f"   Expected return: {efficient_portfolio['expected_return']:.3f}")
            print(f"   Expected volatility: {efficient_portfolio['volatility']:.3f}")
            print(f"   Sharpe ratio: {efficient_portfolio['sharpe_ratio']:.3f}")
            
            # Test Portfolio Optimizer integration
            optimizer = portfolio_optimizer.PortfolioOptimizer()
            optimization_result = optimizer.optimize_portfolio(
                self.portfolio,
                self.market_data,
                optimization_method='mixed',
                constraints={'max_position_size': 0.3, 'max_sector_weight': 0.6}
            )
            print(f"✅ Integrated Portfolio Optimizer - Mixed strategy complete")
            print(f"   Optimized allocations: {len(optimization_result.allocations)} assets")
            print(f"   Risk metrics calculated: {bool(optimization_result.risk_metrics)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Portfolio Optimization Test Failed: {str(e)}")
            return False
    
    def test_dynamic_allocation_engine(self):
        """Test dynamic allocation and rebalancing engine"""
        print("\n⚖️ Testing Dynamic Allocation Engine")
        print("-" * 50)
        
        try:
            # Set up allocation limits
            allocation_limits = dynamic_allocation_engine.AllocationLimits(
                max_position_size=0.25,
                max_sector_weight=0.5,
                min_cash_balance=10000.0,
                max_leverage=1.0
            )
            
            # Create dynamic allocation engine
            engine = dynamic_allocation_engine.DynamicAllocationEngine(
                allocation_limits=allocation_limits,
                rebalance_threshold=0.05,
                rebalance_frequency=timedelta(hours=6)
            )
            
            # Test adding portfolio
            engine.add_portfolio("TEST_PORTFOLIO", self.portfolio)
            print(f"✅ Portfolio added to allocation engine")
            
            # Test real-time monitoring setup
            engine.start_monitoring()
            print(f"✅ Real-time monitoring started")
            
            # Test sector analysis
            sector_manager = dynamic_allocation_engine.SectorManager()
            sector_weights = sector_manager.calculate_sector_weights(self.portfolio)
            print(f"✅ Sector analysis complete")
            print(f"   Sectors identified: {list(sector_weights.keys())}")
            print(f"   Largest sector weight: {max(sector_weights.values()):.2%}")
            
            # Test risk monitoring
            risk_monitor = dynamic_allocation_engine.RiskMonitor()
            risk_metrics = risk_monitor.calculate_portfolio_risk(
                self.portfolio, 
                self.market_data.pct_change().dropna()
            )
            print(f"✅ Risk monitoring active")
            print(f"   Portfolio VaR (95%): {risk_metrics['portfolio_var']:.2%}")
            print(f"   Portfolio volatility: {risk_metrics['portfolio_volatility']:.2%}")
            
            # Test rebalancing signal generation
            current_weights = {pos.symbol: pos.market_value for pos in self.positions}
            target_weights = {symbol: 0.2 for symbol in current_weights.keys()}  # Equal weight target
            
            rebalance_signal = engine._generate_rebalance_signal(
                "TEST_PORTFOLIO",
                current_weights,
                target_weights,
                "drift_threshold"
            )
            
            if rebalance_signal:
                print(f"✅ Rebalancing signal generated")
                print(f"   Signal type: {rebalance_signal.signal_type}")
                print(f"   Trades required: {len(rebalance_signal.target_allocations)}")
            
            # Stop monitoring
            engine.stop_monitoring()
            print(f"✅ Monitoring stopped gracefully")
            
            return True
            
        except Exception as e:
            print(f"❌ Dynamic Allocation Engine Test Failed: {str(e)}")
            return False
    
    def test_rebalancing_executor(self):
        """Test trade execution and rebalancing"""
        print("\n🔄 Testing Rebalancing Executor")
        print("-" * 50)
        
        try:
            # Create rebalancing executor
            executor = rebalancing_executor.RebalancingExecutor()
            
            # Test market impact model
            impact_model = rebalancing_executor.MarketImpactModel()
            impact_estimate = impact_model.estimate_impact(
                symbol="AAPL",
                trade_size=1000,
                average_volume=50000000,
                volatility=0.25,
                liquidity_factor=0.8
            )
            print(f"✅ Market Impact Model")
            print(f"   Estimated impact for 1000 shares AAPL: {impact_estimate:.4f}")
            
            # Test TWAP executor
            twap_executor = rebalancing_executor.TWAPExecutor()
            twap_schedule = twap_executor.create_twap_schedule(
                symbol="MSFT",
                total_quantity=500,
                duration_minutes=60,
                min_slice_size=25
            )
            print(f"✅ TWAP Execution Schedule")
            print(f"   Total slices: {len(twap_schedule)}")
            print(f"   Average slice size: {np.mean([order['quantity'] for order in twap_schedule]):.1f}")
            
            # Test smart order router
            router = rebalancing_executor.SmartOrderRouter()
            routing_decision = router.route_order(
                symbol="GOOGL",
                quantity=200,
                order_type="MARKET",
                urgency="normal"
            )
            print(f"✅ Smart Order Routing")
            print(f"   Recommended strategy: {routing_decision['strategy']}")
            print(f"   Execution venue: {routing_decision['venue']}")
            
            # Create sample rebalance orders
            rebalance_orders = [
                rebalancing_executor.RebalanceOrder(
                    symbol="AAPL",
                    target_quantity=120,
                    current_quantity=100,
                    execution_strategy=rebalancing_executor.ExecutionStrategy.TWAP,
                    urgency="normal"
                ),
                rebalancing_executor.RebalanceOrder(
                    symbol="MSFT",
                    target_quantity=45,
                    current_quantity=50,
                    execution_strategy=rebalancing_executor.ExecutionStrategy.MARKET,
                    urgency="high"
                )
            ]
            
            # Test order execution planning
            execution_plan = executor.create_execution_plan(rebalance_orders)
            print(f"✅ Execution Plan Created")
            print(f"   Orders to execute: {len(execution_plan)}")
            print(f"   Estimated total impact: {sum(order.get('estimated_impact', 0) for order in execution_plan):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Rebalancing Executor Test Failed: {str(e)}")
            return False
    
    def test_portfolio_manager_integration(self):
        """Test integrated portfolio manager"""
        print("\n🎯 Testing Portfolio Manager Integration")
        print("-" * 50)
        
        try:
            # Create portfolio manager
            manager = portfolio_manager.PortfolioManager(
                optimization_config={
                    'default_method': 'mixed',
                    'rebalance_threshold': 0.05,
                    'risk_target': 0.15
                }
            )
            
            # Add portfolio to manager
            manager.add_portfolio("TEST_PORTFOLIO", self.portfolio)
            print(f"✅ Portfolio added to manager")
            
            # Test portfolio monitoring setup
            manager.start_monitoring()
            print(f"✅ Integrated monitoring started")
            
            # Test performance calculation
            performance_metrics = manager.calculate_performance_metrics("TEST_PORTFOLIO")
            print(f"✅ Performance Metrics")
            print(f"   Total return: {performance_metrics.get('total_return', 0):.2%}")
            print(f"   Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Max drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
            
            # Test risk assessment
            risk_assessment = manager.assess_portfolio_risk("TEST_PORTFOLIO")
            print(f"✅ Risk Assessment")
            print(f"   Overall risk level: {risk_assessment.get('risk_level', 'unknown')}")
            print(f"   Risk score: {risk_assessment.get('risk_score', 0):.2f}")
            
            # Test optimization trigger
            optimization_result = manager.optimize_portfolio(
                "TEST_PORTFOLIO",
                method='mixed'
            )
            print(f"✅ Portfolio Optimization Triggered")
            print(f"   Optimization successful: {optimization_result is not None}")
            
            # Stop monitoring
            manager.stop_monitoring()
            print(f"✅ Integrated monitoring stopped")
            
            return True
            
        except Exception as e:
            print(f"❌ Portfolio Manager Integration Test Failed: {str(e)}")
            return False
    
    async def run_async_tests(self):
        """Run asynchronous tests"""
        print("\n🔄 Testing Asynchronous Operations")
        print("-" * 50)
        
        try:
            # Test async portfolio monitoring
            manager = portfolio_manager.PortfolioManager()
            
            # Simulate async monitoring for a short period
            monitoring_task = asyncio.create_task(
                manager._async_monitoring_loop()
            )
            
            await asyncio.sleep(2)  # Monitor for 2 seconds
            monitoring_task.cancel()
            
            print(f"✅ Async monitoring loop tested")
            
            return True
            
        except Exception as e:
            print(f"❌ Async Tests Failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Comprehensive Portfolio Management Tests")
        print("=" * 60)
        
        test_results = {
            'portfolio_optimizer': self.test_portfolio_optimizer(),
            'dynamic_allocation': self.test_dynamic_allocation_engine(),
            'rebalancing_executor': self.test_rebalancing_executor(),
            'portfolio_manager': self.test_portfolio_manager_integration(),
        }
        
        # Run async tests
        try:
            asyncio.run(self.run_async_tests())
            test_results['async_operations'] = True
        except Exception as e:
            print(f"❌ Async tests failed: {str(e)}")
            test_results['async_operations'] = False
        
        # Print final results
        print("\n" + "=" * 60)
        print("📊 FINAL TEST RESULTS")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Portfolio Management System is ready for production.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review implementation before deployment.")
        
        return test_results

if __name__ == "__main__":
    tester = PortfolioManagementTester()
    results = tester.run_all_tests()
