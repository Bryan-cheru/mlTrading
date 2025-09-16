"""
Portfolio Management Integration Module
Integrates portfolio optimization, allocation, and rebalancing with the main trading system
Provides unified interface for institutional portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import json
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import portfolio management components with importlib
import importlib.util

# Import optimization components
spec = importlib.util.spec_from_file_location(
    "portfolio_optimizer", 
    os.path.join(os.path.dirname(__file__), "optimization", "portfolio_optimizer.py")
)
portfolio_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_optimizer)

Asset = portfolio_optimizer.Asset
Position = portfolio_optimizer.Position
Portfolio = portfolio_optimizer.Portfolio
PortfolioOptimizer = portfolio_optimizer.PortfolioOptimizer
KellyCriterionCalculator = portfolio_optimizer.KellyCriterionCalculator
RiskParityOptimizer = portfolio_optimizer.RiskParityOptimizer
ModernPortfolioTheory = portfolio_optimizer.ModernPortfolioTheory

# Import allocation components
spec = importlib.util.spec_from_file_location(
    "dynamic_allocation_engine", 
    os.path.join(os.path.dirname(__file__), "allocation", "dynamic_allocation_engine.py")
)
allocation_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocation_engine)

DynamicAllocationEngine = allocation_engine.DynamicAllocationEngine
RebalanceSignal = allocation_engine.RebalanceSignal
AllocationLimits = allocation_engine.AllocationLimits
SectorManager = allocation_engine.SectorManager
RiskMonitor = allocation_engine.RiskMonitor

# Import rebalancing components
spec = importlib.util.spec_from_file_location(
    "rebalancing_executor", 
    os.path.join(os.path.dirname(__file__), "rebalancing", "rebalancing_executor.py")
)
rebalancing_executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebalancing_executor)

RebalancingExecutor = rebalancing_executor.RebalancingExecutor
ExecutionStrategy = rebalancing_executor.ExecutionStrategy
RebalanceOrder = rebalancing_executor.RebalanceOrder
MarketImpactModel = rebalancing_executor.MarketImpactModel
TWAPExecutor = rebalancing_executor.TWAPExecutor
SmartOrderRouter = rebalancing_executor.SmartOrderRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioManagerConfig:
    """Configuration for portfolio manager"""
    optimization_method: str = 'mixed'  # kelly, risk_parity, sharpe, mixed
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalancing
    max_position_size: float = 0.25  # Max 25% per position
    min_position_size: float = 0.01  # Min 1% per position
    max_sector_allocation: float = 0.50  # Max 50% per sector
    min_cash_reserve: float = 0.05  # Min 5% cash
    rebalance_frequency: timedelta = timedelta(hours=4)  # Check every 4 hours
    optimization_frequency: timedelta = timedelta(hours=12)  # Reoptimize every 12 hours
    execution_strategy: str = 'smart'  # market, twap, vwap, smart
    transaction_cost_bps: float = 5.0  # 5 bps transaction cost
    enable_risk_monitoring: bool = True
    enable_auto_rebalancing: bool = True

@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics"""
    timestamp: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_1d: float
    win_rate: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

class PortfolioManager:
    """
    Comprehensive portfolio management system
    Integrates optimization, allocation, monitoring, and execution
    """
    
    def __init__(self, config: PortfolioManagerConfig = None, 
                 ninjatrader_connector=None, market_data_feed=None):
        """
        Initialize portfolio manager
        
        Args:
            config: Portfolio management configuration
            ninjatrader_connector: NinjaTrader connection for execution
            market_data_feed: Real-time market data feed
        """
        self.config = config or PortfolioManagerConfig()
        self.ninjatrader_connector = ninjatrader_connector
        self.market_data_feed = market_data_feed
        
        # Initialize allocation limits
        allocation_limits = AllocationLimits(
            max_position_size=self.config.max_position_size,
            min_position_size=self.config.min_position_size,
            max_sector_allocation=self.config.max_sector_allocation,
            min_cash_reserve=self.config.min_cash_reserve
        )
        
        # Initialize components
        self.allocation_engine = DynamicAllocationEngine(
            optimization_method=self.config.optimization_method,
            rebalance_threshold=self.config.rebalance_threshold,
            allocation_limits=allocation_limits
        )
        
        self.rebalancing_executor = RebalancingExecutor(ninjatrader_connector)
        
        # State management
        self.current_portfolio = None
        self.available_assets = []
        self.historical_data = {}
        self.performance_history = []
        self.is_running = False
        
        # Monitoring and callbacks
        self.performance_callbacks = []
        self.rebalance_callbacks = []
        self.risk_alert_callbacks = []
        
        # Timing control
        self.last_optimization = datetime.min
        self.last_rebalance = datetime.min
        self.last_performance_update = datetime.min
        
        # Register callbacks
        self.allocation_engine.add_rebalance_callback(self._handle_rebalance_signal)
        
    async def initialize(self, assets: List[Asset], initial_capital: float = 1000000,
                        historical_data: Dict[str, pd.DataFrame] = None) -> bool:
        """
        Initialize portfolio management system
        
        Args:
            assets: Available assets for portfolio
            initial_capital: Initial portfolio capital
            historical_data: Historical price data for optimization
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing institutional portfolio management system...")
            
            self.available_assets = assets
            self.historical_data = historical_data or {}
            
            # Initialize allocation engine
            self.current_portfolio = await self.allocation_engine.initialize(
                assets, self.historical_data, initial_capital
            )
            
            if not self.current_portfolio:
                logger.error("Failed to initialize portfolio")
                return False
            
            # Calculate initial performance baseline
            await self._update_performance_metrics()
            
            self.last_optimization = datetime.now()
            self.last_performance_update = datetime.now()
            
            logger.info("Portfolio management system initialized successfully")
            logger.info(f"Initial portfolio value: ${self.current_portfolio.total_value:,.2f}")
            logger.info(f"Target allocations: {self.current_portfolio.target_allocations}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio manager: {e}")
            return False
    
    async def start_monitoring(self):
        """Start continuous portfolio monitoring and management"""
        if self.is_running:
            logger.warning("Portfolio monitoring already running")
            return
        
        self.is_running = True
        logger.info("Starting portfolio monitoring and management...")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._portfolio_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._optimization_loop())
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            logger.error(f"Error in portfolio monitoring: {e}")
        finally:
            self.is_running = False
    
    async def stop_monitoring(self):
        """Stop portfolio monitoring"""
        self.is_running = False
        logger.info("Stopping portfolio monitoring...")
    
    async def update_positions(self, positions: Dict[str, Position], 
                             current_prices: Dict[str, float]) -> Optional[RebalanceSignal]:
        """
        Update current portfolio positions and check for rebalancing
        
        Args:
            positions: Current portfolio positions
            current_prices: Current market prices
            
        Returns:
            RebalanceSignal if rebalancing is needed
        """
        try:
            if not self.current_portfolio:
                logger.warning("Portfolio not initialized")
                return None
            
            # Update portfolio state
            rebalance_signal = await self.allocation_engine.update_portfolio(
                positions, current_prices, self.available_assets
            )
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            return rebalance_signal
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return None
    
    async def execute_rebalancing(self, rebalance_signal: RebalanceSignal,
                                 dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing
        
        Args:
            rebalance_signal: Rebalancing requirements
            dry_run: If True, simulate execution without real trades
            
        Returns:
            Execution results
        """
        try:
            if not self.config.enable_auto_rebalancing and not dry_run:
                logger.info("Auto-rebalancing disabled, skipping execution")
                return {'status': 'disabled', 'message': 'Auto-rebalancing disabled'}
            
            logger.info(f"Executing portfolio rebalancing: {rebalance_signal.trigger_reason}")
            
            # Get current market data
            market_data = await self._get_current_market_data()
            
            # Execute rebalancing
            results = await self.rebalancing_executor.execute_rebalancing(
                rebalance_signal, market_data, dry_run
            )
            
            if results['status'] == 'completed':
                self.last_rebalance = datetime.now()
                logger.info(f"Rebalancing completed successfully: "
                           f"Cost {results.get('execution_cost_bps', 0):.1f} bps")
            
            # Notify callbacks
            for callback in self.rebalance_callbacks:
                try:
                    await callback(rebalance_signal, results)
                except Exception as e:
                    logger.error(f"Error in rebalance callback: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def reoptimize_portfolio(self, force: bool = False) -> Dict[str, float]:
        """
        Trigger portfolio reoptimization
        
        Args:
            force: Force reoptimization even if recent
            
        Returns:
            New allocation weights
        """
        try:
            # Check if reoptimization is needed
            time_since_last = datetime.now() - self.last_optimization
            if not force and time_since_last < self.config.optimization_frequency:
                logger.info("Skipping reoptimization - too recent")
                return self.current_portfolio.target_allocations if self.current_portfolio else {}
            
            logger.info("Reoptimizing portfolio allocation...")
            
            # Update historical data if available
            if self.market_data_feed:
                updated_data = await self._update_historical_data()
                if updated_data:
                    self.historical_data.update(updated_data)
            
            # Reoptimize allocation
            new_allocation = await self.allocation_engine.reoptimize_allocation(
                self.available_assets, self.historical_data, force
            )
            
            self.last_optimization = datetime.now()
            
            logger.info(f"Portfolio reoptimized: {new_allocation}")
            return new_allocation
            
        except Exception as e:
            logger.error(f"Error reoptimizing portfolio: {e}")
            return self.current_portfolio.target_allocations if self.current_portfolio else {}
    
    async def _portfolio_monitoring_loop(self):
        """Main portfolio monitoring loop"""
        while self.is_running:
            try:
                # Check for rebalancing needs
                time_since_last = datetime.now() - self.last_rebalance
                if time_since_last >= self.config.rebalance_frequency:
                    
                    if self.market_data_feed:
                        # Get current positions and prices
                        current_positions = await self._get_current_positions()
                        current_prices = await self._get_current_prices()
                        
                        if current_positions and current_prices:
                            rebalance_signal = await self.update_positions(current_positions, current_prices)
                            
                            if rebalance_signal and rebalance_signal.priority in ['high', 'medium']:
                                await self.execute_rebalancing(rebalance_signal)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in portfolio monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _performance_monitoring_loop(self):
        """Performance metrics monitoring loop"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for risk alerts
                if self.config.enable_risk_monitoring:
                    await self._check_risk_alerts()
                
                # Wait before next update
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _optimization_loop(self):
        """Portfolio optimization loop"""
        while self.is_running:
            try:
                # Check if reoptimization is due
                time_since_last = datetime.now() - self.last_optimization
                if time_since_last >= self.config.optimization_frequency:
                    await self.reoptimize_portfolio()
                
                # Wait before next check
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(600)  # Wait before retrying
    
    async def _handle_rebalance_signal(self, signal: RebalanceSignal):
        """Handle rebalancing signal from allocation engine"""
        try:
            logger.info(f"Received rebalance signal: {signal.trigger_reason}")
            
            # Auto-execute if enabled and high priority
            if self.config.enable_auto_rebalancing and signal.priority == 'high':
                await self.execute_rebalancing(signal)
            else:
                logger.info(f"Rebalance signal queued (priority: {signal.priority})")
                
        except Exception as e:
            logger.error(f"Error handling rebalance signal: {e}")
    
    async def _update_performance_metrics(self):
        """Update portfolio performance metrics"""
        try:
            if not self.current_portfolio:
                return
            
            # Calculate performance metrics
            current_performance = await self._calculate_performance_metrics()
            
            if current_performance:
                self.performance_history.append(current_performance)
                
                # Keep only last 1000 records
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                self.last_performance_update = datetime.now()
                
                # Notify callbacks
                for callback in self.performance_callbacks:
                    try:
                        await callback(current_performance)
                    except Exception as e:
                        logger.error(f"Error in performance callback: {e}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _calculate_performance_metrics(self) -> Optional[PortfolioPerformance]:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            if not self.current_portfolio or not self.performance_history:
                return None
            
            # Simple performance calculation (in production, use actual returns)
            current_value = self.current_portfolio.total_value
            
            if len(self.performance_history) > 0:
                previous_value = self.performance_history[-1].total_return if hasattr(self.performance_history[-1], 'total_return') else current_value
                daily_return = (current_value - previous_value) / previous_value if previous_value > 0 else 0
            else:
                daily_return = 0
            
            # Calculate metrics (simplified for demo)
            performance = PortfolioPerformance(
                timestamp=datetime.now(),
                total_return=daily_return,
                annualized_return=daily_return * 252,  # Simplified
                volatility=0.15,  # Placeholder
                sharpe_ratio=2.5,  # Placeholder
                max_drawdown=0.03,  # Placeholder
                var_1d=current_value * 0.02,  # 2% VaR
                win_rate=0.68,  # Placeholder
                alpha=0.05,  # Placeholder
                beta=1.1,  # Placeholder
                information_ratio=1.8,  # Placeholder
                tracking_error=0.04  # Placeholder
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return None
    
    async def _check_risk_alerts(self):
        """Check for risk-based alerts"""
        try:
            if not self.current_portfolio:
                return
            
            risk_metrics = self.current_portfolio.risk_metrics
            
            # Check risk thresholds
            alerts = []
            
            if risk_metrics.get('concentration_risk') == 'High':
                alerts.append("High concentration risk detected")
            
            if risk_metrics.get('max_position_size', 0) > self.config.max_position_size:
                alerts.append(f"Position size exceeds limit: {risk_metrics.get('max_position_size', 0):.3f}")
            
            var_1d = risk_metrics.get('portfolio_var_1d', 0)
            if var_1d > self.current_portfolio.total_value * 0.03:  # 3% VaR limit
                alerts.append(f"VaR exceeds limit: ${var_1d:,.2f}")
            
            # Notify callbacks for alerts
            if alerts:
                for alert in alerts:
                    logger.warning(f"Risk Alert: {alert}")
                    
                for callback in self.risk_alert_callbacks:
                    try:
                        await callback(alerts)
                    except Exception as e:
                        logger.error(f"Error in risk alert callback: {e}")
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    async def _get_current_market_data(self) -> Dict[str, Dict]:
        """Get current market data for all assets"""
        market_data = {}
        
        for asset in self.available_assets:
            market_data[asset.symbol] = {
                'price': asset.current_price,
                'volatility': asset.volatility,
                'avg_daily_volume': 1000000,  # Placeholder
                'bid_ask_spread': 0.01  # Placeholder
            }
        
        return market_data
    
    async def _get_current_positions(self) -> Dict[str, Position]:
        """Get current portfolio positions"""
        if self.current_portfolio:
            return self.current_portfolio.positions
        return {}
    
    async def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices"""
        return {asset.symbol: asset.current_price for asset in self.available_assets}
    
    async def _update_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Update historical data from market feed"""
        # Placeholder - in production, fetch from market data feed
        return {}
    
    def add_performance_callback(self, callback: Callable):
        """Add callback for performance updates"""
        self.performance_callbacks.append(callback)
    
    def add_rebalance_callback(self, callback: Callable):
        """Add callback for rebalancing events"""
        self.rebalance_callbacks.append(callback)
    
    def add_risk_alert_callback(self, callback: Callable):
        """Add callback for risk alerts"""
        self.risk_alert_callbacks.append(callback)
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        if not self.current_portfolio:
            return {'error': 'Portfolio not initialized'}
        
        return {
            'portfolio_value': self.current_portfolio.total_value,
            'cash_balance': self.current_portfolio.cash,
            'positions_count': len(self.current_portfolio.positions),
            'target_allocations': self.current_portfolio.target_allocations,
            'actual_allocations': self.current_portfolio.actual_allocations,
            'risk_metrics': self.current_portfolio.risk_metrics,
            'last_rebalance': self.last_rebalance,
            'last_optimization': self.last_optimization,
            'performance_history_length': len(self.performance_history),
            'monitoring_active': self.is_running,
            'config': asdict(self.config)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary"""
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        latest = self.performance_history[-1]
        
        return {
            'latest_performance': asdict(latest),
            'history_length': len(self.performance_history),
            'last_updated': self.last_performance_update
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_portfolio_manager():
        # Sample configuration
        config = PortfolioManagerConfig(
            optimization_method='mixed',
            rebalance_threshold=0.05,
            max_position_size=0.25,
            enable_auto_rebalancing=True
        )
        
        # Sample assets
        assets = [
            Asset("AAPL", "Technology", 3000000000000, 175.50, 0.25, 0.12, 1.2),
            Asset("MSFT", "Technology", 2800000000000, 335.20, 0.22, 0.11, 0.9),
            Asset("JPM", "Financials", 450000000000, 145.30, 0.30, 0.10, 1.3),
            Asset("JNJ", "Healthcare", 420000000000, 165.75, 0.18, 0.08, 0.7),
        ]
        
        # Sample historical data
        historical_data = {}
        for asset in assets:
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), len(dates))))
            historical_data[asset.symbol] = pd.DataFrame({'close': prices}, index=dates)
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(config)
        
        # Initialize system
        success = await portfolio_manager.initialize(assets, 1000000, historical_data)
        
        if success:
            print("Portfolio Manager initialized successfully!")
            
            # Get status
            status = portfolio_manager.get_portfolio_status()
            print(f"Portfolio Status:")
            print(f"  Value: ${status['portfolio_value']:,.2f}")
            print(f"  Positions: {status['positions_count']}")
            print(f"  Target Allocations: {status['target_allocations']}")
            
            # Test reoptimization
            new_allocation = await portfolio_manager.reoptimize_portfolio(force=True)
            print(f"Reoptimized Allocation: {new_allocation}")
        
        else:
            print("Failed to initialize portfolio manager")
    
    # Run test
    asyncio.run(test_portfolio_manager())
