"""
Dynamic Portfolio Allocation Engine
Real-time portfolio management with automatic rebalancing and risk monitoring
Integrates with existing trading system for seamless portfolio optimization
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our optimization components
import importlib.util
spec = importlib.util.spec_from_file_location(
    "portfolio_optimizer", 
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "optimization", "portfolio_optimizer.py")
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RebalanceSignal:
    """Signal to rebalance portfolio"""
    timestamp: datetime
    trigger_reason: str
    current_allocations: Dict[str, float]
    target_allocations: Dict[str, float]
    rebalance_trades: Dict[str, float]  # symbol -> shares to buy/sell
    expected_cost: float
    priority: str  # 'low', 'medium', 'high'

@dataclass
class AllocationLimits:
    """Portfolio allocation constraints"""
    max_position_size: float = 0.25  # Max 25% per position
    min_position_size: float = 0.01  # Min 1% per position
    max_sector_allocation: float = 0.50  # Max 50% per sector
    max_correlation_group: float = 0.40  # Max 40% in highly correlated assets
    max_single_stock: float = 0.15  # Max 15% in single stock
    min_cash_reserve: float = 0.05  # Min 5% cash reserve

class SectorManager:
    """
    Manages sector-based diversification and allocation limits
    """
    
    def __init__(self):
        # Standard GICS sectors
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],
            'Consumer_Discretionary': ['HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
            'Communication': ['VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'TMUS'],
            'Industrials': ['BA', 'CAT', 'UPS', 'HON', 'LMT', 'MMM', 'GE'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'XEL'],
            'Real_Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'EXR'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW'],
            'Consumer_Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'KMB']
        }
        
        # Reverse mapping: symbol -> sector
        self.symbol_to_sector = {}
        for sector, symbols in self.sectors.items():
            for symbol in symbols:
                self.symbol_to_sector[symbol] = sector
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        return self.symbol_to_sector.get(symbol, 'Other')
    
    def calculate_sector_allocations(self, portfolio_allocations: Dict[str, float]) -> Dict[str, float]:
        """Calculate current sector allocations"""
        sector_allocations = {}
        
        for symbol, weight in portfolio_allocations.items():
            sector = self.get_sector(symbol)
            sector_allocations[sector] = sector_allocations.get(sector, 0) + weight
        
        return sector_allocations
    
    def check_sector_limits(self, allocations: Dict[str, float], 
                          limits: AllocationLimits) -> List[str]:
        """Check if sector allocations exceed limits"""
        violations = []
        sector_allocations = self.calculate_sector_allocations(allocations)
        
        for sector, allocation in sector_allocations.items():
            if allocation > limits.max_sector_allocation:
                violations.append(f"Sector {sector} allocation {allocation:.3f} exceeds limit {limits.max_sector_allocation:.3f}")
        
        return violations

class RiskMonitor:
    """
    Real-time portfolio risk monitoring
    """
    
    def __init__(self, var_confidence: float = 0.05):
        self.var_confidence = var_confidence  # 5% VaR
        self.monitoring_active = False
        self.risk_alerts = []
        
    def calculate_portfolio_var(self, weights: np.ndarray, cov_matrix: np.ndarray, 
                               confidence: float = None) -> float:
        """Calculate Value-at-Risk for portfolio"""
        if confidence is None:
            confidence = self.var_confidence
            
        try:
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            var = -scipy_stats.norm.ppf(confidence) * portfolio_vol
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_component_var(self, weights: np.ndarray, cov_matrix: np.ndarray, 
                               confidence: float = None) -> np.ndarray:
        """Calculate component VaR for each position"""
        if confidence is None:
            confidence = self.var_confidence
            
        try:
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return np.zeros_like(weights)
                
            # Marginal VaR
            marginal_var = -scipy_stats.norm.ppf(confidence) * np.dot(cov_matrix, weights) / portfolio_vol
            
            # Component VaR
            component_var = weights * marginal_var
            
            return component_var
        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return np.zeros_like(weights)
    
    def monitor_portfolio_risk(self, portfolio: Portfolio, assets: List[Asset]) -> Dict[str, Any]:
        """Comprehensive portfolio risk monitoring"""
        try:
            # Prepare data
            symbols = list(portfolio.positions.keys())
            weights = np.array([portfolio.actual_allocations.get(symbol, 0) for symbol in symbols])
            
            # Simple covariance estimation (in production, use historical data)
            volatilities = np.array([next(a.volatility for a in assets if a.symbol == symbol) for symbol in symbols])
            correlation = np.eye(len(symbols)) * 0.7 + np.ones((len(symbols), len(symbols))) * 0.3
            cov_matrix = np.outer(volatilities, volatilities) * correlation
            
            # Calculate risk metrics
            portfolio_var = self.calculate_portfolio_var(weights, cov_matrix)
            component_var = self.calculate_component_var(weights, cov_matrix)
            
            # Concentration risk
            herfindahl_index = np.sum(weights ** 2)
            concentration_risk = "High" if herfindahl_index > 0.2 else "Medium" if herfindahl_index > 0.1 else "Low"
            
            # Liquidity risk (simplified)
            liquidity_scores = np.array([next(a.liquidity_score for a in assets if a.symbol == symbol) for symbol in symbols])
            weighted_liquidity = np.sum(weights * liquidity_scores)
            
            risk_report = {
                'timestamp': datetime.now(),
                'portfolio_var_1d': portfolio_var,
                'component_var': {symbol: var for symbol, var in zip(symbols, component_var)},
                'concentration_risk': concentration_risk,
                'herfindahl_index': herfindahl_index,
                'weighted_liquidity': weighted_liquidity,
                'max_position_size': np.max(weights),
                'num_positions': len([w for w in weights if w > 0.01])  # Positions > 1%
            }
            
            return risk_report
            
        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

class RebalancingEngine:
    """
    Automated portfolio rebalancing engine
    """
    
    def __init__(self, rebalance_threshold: float = 0.05, 
                 max_turnover: float = 0.20, transaction_cost: float = 0.001):
        """
        Initialize rebalancing engine
        
        Args:
            rebalance_threshold: Trigger rebalancing when deviation exceeds this
            max_turnover: Maximum portfolio turnover per rebalancing
            transaction_cost: Estimated transaction cost (bps)
        """
        self.rebalance_threshold = rebalance_threshold
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.last_rebalance = datetime.now()
        self.min_rebalance_interval = timedelta(hours=1)  # Min 1 hour between rebalances
        
    def check_rebalance_triggers(self, portfolio: Portfolio) -> Optional[RebalanceSignal]:
        """Check if portfolio needs rebalancing"""
        try:
            # Time-based check
            if datetime.now() - self.last_rebalance < self.min_rebalance_interval:
                return None
            
            # Calculate deviations
            deviations = {}
            max_deviation = 0
            total_deviation = 0
            
            for symbol in portfolio.target_allocations:
                target = portfolio.target_allocations[symbol]
                actual = portfolio.actual_allocations.get(symbol, 0)
                deviation = abs(actual - target)
                deviations[symbol] = deviation
                max_deviation = max(max_deviation, deviation)
                total_deviation += deviation
            
            # Trigger conditions
            trigger_reason = None
            priority = 'low'
            
            if max_deviation > self.rebalance_threshold:
                trigger_reason = f"Max deviation {max_deviation:.3f} exceeds threshold {self.rebalance_threshold:.3f}"
                priority = 'high' if max_deviation > 0.10 else 'medium'
            elif total_deviation > 0.20:
                trigger_reason = f"Total deviation {total_deviation:.3f} indicates drift"
                priority = 'medium'
            
            # Risk-based triggers
            if portfolio.risk_metrics.get('concentration_risk') == 'High':
                trigger_reason = "High concentration risk detected"
                priority = 'high'
            
            if trigger_reason:
                # Calculate rebalancing trades
                rebalance_trades = self._calculate_rebalancing_trades(portfolio)
                estimated_cost = self._estimate_transaction_costs(rebalance_trades, portfolio)
                
                # Cost-benefit analysis
                if estimated_cost > portfolio.total_value * 0.005:  # More than 0.5% of portfolio
                    logger.warning(f"Rebalancing cost {estimated_cost:.2f} may be too high")
                    if priority != 'high':
                        return None  # Skip if not high priority
                
                return RebalanceSignal(
                    timestamp=datetime.now(),
                    trigger_reason=trigger_reason,
                    current_allocations=portfolio.actual_allocations.copy(),
                    target_allocations=portfolio.target_allocations.copy(),
                    rebalance_trades=rebalance_trades,
                    expected_cost=estimated_cost,
                    priority=priority
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking rebalance triggers: {e}")
            return None
    
    def _calculate_rebalancing_trades(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate required trades for rebalancing"""
        trades = {}
        
        for symbol in portfolio.target_allocations:
            target_weight = portfolio.target_allocations[symbol]
            current_weight = portfolio.actual_allocations.get(symbol, 0)
            
            target_value = target_weight * portfolio.total_value
            current_value = current_weight * portfolio.total_value
            
            trade_value = target_value - current_value
            
            if symbol in portfolio.positions:
                current_price = portfolio.positions[symbol].market_value / portfolio.positions[symbol].shares
                trade_shares = trade_value / current_price
                
                if abs(trade_shares) > 1:  # Only trade if significant
                    trades[symbol] = trade_shares
        
        return trades
    
    def _estimate_transaction_costs(self, trades: Dict[str, float], 
                                  portfolio: Portfolio) -> float:
        """Estimate transaction costs for rebalancing"""
        total_cost = 0
        
        for symbol, shares in trades.items():
            if symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                trade_value = abs(shares) * (position.market_value / position.shares)
                cost = trade_value * self.transaction_cost
                total_cost += cost
        
        return total_cost

class DynamicAllocationEngine:
    """
    Main dynamic portfolio allocation engine
    Coordinates optimization, monitoring, and rebalancing
    """
    
    def __init__(self, 
                 optimization_method: str = 'mixed',
                 rebalance_threshold: float = 0.05,
                 allocation_limits: Optional[AllocationLimits] = None):
        """
        Initialize dynamic allocation engine
        
        Args:
            optimization_method: Portfolio optimization strategy
            rebalance_threshold: Deviation threshold for rebalancing
            allocation_limits: Portfolio allocation constraints
        """
        self.optimization_method = optimization_method
        self.allocation_limits = allocation_limits or AllocationLimits()
        
        # Initialize components
        self.optimizer = PortfolioOptimizer(optimization_method)
        self.sector_manager = SectorManager()
        self.risk_monitor = RiskMonitor()
        self.rebalancing_engine = RebalancingEngine(rebalance_threshold)
        
        # State tracking
        self.current_portfolio = None
        self.optimization_history = []
        self.rebalance_history = []
        self.is_running = False
        
        # Callbacks
        self.rebalance_callbacks = []
        self.risk_alert_callbacks = []
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self, initial_assets: List[Asset], 
                        historical_data: Dict[str, pd.DataFrame],
                        initial_capital: float = 1000000) -> Portfolio:
        """Initialize portfolio with optimal allocation"""
        try:
            logger.info("Initializing dynamic portfolio allocation engine...")
            
            # Calculate optimal initial allocation
            optimal_allocation = self.optimizer.optimize_portfolio(initial_assets, historical_data)
            
            # Apply constraints
            optimal_allocation = self._apply_allocation_limits(optimal_allocation, initial_assets)
            
            # Create initial portfolio
            self.current_portfolio = Portfolio(
                total_value=initial_capital,
                cash=initial_capital * self.allocation_limits.min_cash_reserve,
                positions={},
                target_allocations=optimal_allocation,
                actual_allocations={symbol: 0.0 for symbol in optimal_allocation},
                risk_metrics={},
                performance_metrics={},
                last_rebalance=datetime.now(),
                rebalance_threshold=self.rebalancing_engine.rebalance_threshold
            )
            
            # Initial risk assessment
            risk_report = self.risk_monitor.monitor_portfolio_risk(self.current_portfolio, initial_assets)
            self.current_portfolio.risk_metrics = risk_report
            
            logger.info("Dynamic allocation engine initialized successfully")
            logger.info(f"Initial allocation: {optimal_allocation}")
            
            return self.current_portfolio
            
        except Exception as e:
            logger.error(f"Error initializing allocation engine: {e}")
            raise
    
    async def update_portfolio(self, current_positions: Dict[str, Position],
                              current_prices: Dict[str, float],
                              assets: List[Asset]) -> Optional[RebalanceSignal]:
        """
        Update portfolio state and check for rebalancing needs
        
        Args:
            current_positions: Current portfolio positions
            current_prices: Current market prices
            assets: Available assets
            
        Returns:
            RebalanceSignal if rebalancing is needed
        """
        try:
            if not self.current_portfolio:
                logger.warning("Portfolio not initialized")
                return None
            
            # Update portfolio positions
            self.current_portfolio.positions = current_positions
            
            # Calculate current allocations
            total_value = sum(pos.market_value for pos in current_positions.values())
            self.current_portfolio.total_value = total_value
            
            actual_allocations = {}
            for symbol, position in current_positions.items():
                actual_allocations[symbol] = position.market_value / total_value if total_value > 0 else 0
            
            self.current_portfolio.actual_allocations = actual_allocations
            
            # Update risk metrics
            risk_report = self.risk_monitor.monitor_portfolio_risk(self.current_portfolio, assets)
            self.current_portfolio.risk_metrics = risk_report
            
            # Check for rebalancing needs
            rebalance_signal = self.rebalancing_engine.check_rebalance_triggers(self.current_portfolio)
            
            if rebalance_signal:
                logger.info(f"Rebalancing signal generated: {rebalance_signal.trigger_reason}")
                self.rebalance_history.append(rebalance_signal)
                
                # Notify callbacks
                for callback in self.rebalance_callbacks:
                    try:
                        await callback(rebalance_signal)
                    except Exception as e:
                        logger.error(f"Error in rebalance callback: {e}")
            
            return rebalance_signal
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return None
    
    async def reoptimize_allocation(self, assets: List[Asset], 
                                   historical_data: Dict[str, pd.DataFrame],
                                   force_reoptimization: bool = False) -> Dict[str, float]:
        """
        Reoptimize portfolio allocation based on current market conditions
        
        Args:
            assets: Available assets
            historical_data: Updated historical data
            force_reoptimization: Force reoptimization even if recent
            
        Returns:
            New optimal allocation
        """
        try:
            # Check if reoptimization is needed
            last_optimization = self.optimization_history[-1]['timestamp'] if self.optimization_history else datetime.min
            time_since_last = datetime.now() - last_optimization
            
            if not force_reoptimization and time_since_last < timedelta(hours=4):
                logger.info("Skipping reoptimization - too recent")
                return self.current_portfolio.target_allocations
            
            logger.info("Reoptimizing portfolio allocation...")
            
            # Calculate new optimal allocation
            new_allocation = self.optimizer.optimize_portfolio(assets, historical_data, self.current_portfolio)
            
            # Apply constraints
            new_allocation = self._apply_allocation_limits(new_allocation, assets)
            
            # Update target allocations
            if self.current_portfolio:
                self.current_portfolio.target_allocations = new_allocation
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'allocation': new_allocation.copy(),
                'method': self.optimization_method
            })
            
            logger.info(f"Portfolio reoptimized: {new_allocation}")
            
            return new_allocation
            
        except Exception as e:
            logger.error(f"Error reoptimizing allocation: {e}")
            return self.current_portfolio.target_allocations if self.current_portfolio else {}
    
    def _apply_allocation_limits(self, allocation: Dict[str, float], 
                               assets: List[Asset]) -> Dict[str, float]:
        """Apply allocation constraints and limits"""
        try:
            # Apply position size limits
            for symbol in allocation:
                allocation[symbol] = max(self.allocation_limits.min_position_size,
                                       min(allocation[symbol], self.allocation_limits.max_position_size))
            
            # Check sector limits
            violations = self.sector_manager.check_sector_limits(allocation, self.allocation_limits)
            
            if violations:
                logger.warning(f"Sector limit violations: {violations}")
                # Reduce allocations in violating sectors
                sector_allocations = self.sector_manager.calculate_sector_allocations(allocation)
                
                for violation in violations:
                    # Simple reduction (in production, use optimization)
                    for symbol in allocation:
                        sector = self.sector_manager.get_sector(symbol)
                        if sector in violation:
                            allocation[symbol] *= 0.8  # Reduce by 20%
            
            # Renormalize to account for cash reserve
            total_allocation = sum(allocation.values())
            max_invested = 1.0 - self.allocation_limits.min_cash_reserve
            
            if total_allocation > max_invested:
                scaling_factor = max_invested / total_allocation
                for symbol in allocation:
                    allocation[symbol] *= scaling_factor
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error applying allocation limits: {e}")
            return allocation
    
    def add_rebalance_callback(self, callback: Callable):
        """Add callback for rebalancing signals"""
        self.rebalance_callbacks.append(callback)
    
    def add_risk_alert_callback(self, callback: Callable):
        """Add callback for risk alerts"""
        self.risk_alert_callbacks.append(callback)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        if not self.current_portfolio:
            return {'error': 'Portfolio not initialized'}
        
        return {
            'total_value': self.current_portfolio.total_value,
            'cash': self.current_portfolio.cash,
            'num_positions': len(self.current_portfolio.positions),
            'target_allocations': self.current_portfolio.target_allocations,
            'actual_allocations': self.current_portfolio.actual_allocations,
            'risk_metrics': self.current_portfolio.risk_metrics,
            'last_rebalance': self.current_portfolio.last_rebalance,
            'rebalance_history_count': len(self.rebalance_history),
            'optimization_history_count': len(self.optimization_history)
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_dynamic_allocation():
        # Sample assets
        assets = [
            Asset("AAPL", "Technology", 3000000000000, 175.50, 0.25, 0.12, 1.2),
            Asset("MSFT", "Technology", 2800000000000, 335.20, 0.22, 0.11, 0.9),
            Asset("JPM", "Financials", 450000000000, 145.30, 0.30, 0.10, 1.3),
            Asset("JNJ", "Healthcare", 420000000000, 165.75, 0.18, 0.08, 0.7),
            Asset("XOM", "Energy", 350000000000, 95.20, 0.35, 0.15, 1.4),
        ]
        
        # Sample historical data
        historical_data = {}
        for asset in assets:
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), len(dates))))
            historical_data[asset.symbol] = pd.DataFrame({'close': prices}, index=dates)
        
        # Initialize allocation engine
        engine = DynamicAllocationEngine(optimization_method='mixed')
        
        # Initialize portfolio
        portfolio = await engine.initialize(assets, historical_data, initial_capital=1000000)
        
        print("Initial Portfolio:")
        print(f"Total Value: ${portfolio.total_value:,.2f}")
        print("Target Allocations:")
        for symbol, weight in portfolio.target_allocations.items():
            print(f"  {symbol}: {weight:.4f} ({weight*100:.2f}%)")
        
        # Simulate some position changes
        current_positions = {}
        for symbol, target_weight in portfolio.target_allocations.items():
            asset = next(a for a in assets if a.symbol == symbol)
            shares = (target_weight * portfolio.total_value) / asset.current_price
            current_positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                market_value=shares * asset.current_price,
                weight=target_weight,
                target_weight=target_weight,
                deviation=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_price=asset.current_price,
                entry_date=datetime.now()
            )
        
        # Update portfolio
        rebalance_signal = await engine.update_portfolio(current_positions, 
                                                        {asset.symbol: asset.current_price for asset in assets},
                                                        assets)
        
        if rebalance_signal:
            print(f"\nRebalancing needed: {rebalance_signal.trigger_reason}")
        else:
            print("\nNo rebalancing needed")
        
        # Get portfolio summary
        summary = engine.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Cash: ${summary['cash']:,.2f}")
        print(f"Positions: {summary['num_positions']}")
        print(f"Risk Metrics: {summary['risk_metrics']}")
    
    # Run test
    asyncio.run(test_dynamic_allocation())
