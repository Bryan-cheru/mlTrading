"""
Enhanced Portfolio Management System
Institutional-grade portfolio optimization with dynamic allocation, Kelly Criterion, and risk parity
Features: Real-time optimization, correlation analysis, sector diversification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Asset:
    """Individual asset in the portfolio"""
    symbol: str
    sector: str
    market_cap: float
    current_price: float
    volatility: float
    expected_return: float
    beta: float
    liquidity_score: float = 1.0  # 0-1, higher = more liquid

@dataclass
class Position:
    """Current portfolio position"""
    symbol: str
    shares: float
    market_value: float
    weight: float
    target_weight: float
    deviation: float
    unrealized_pnl: float
    realized_pnl: float
    entry_price: float
    entry_date: datetime

@dataclass
class Portfolio:
    """Complete portfolio state"""
    total_value: float
    cash: float
    positions: Dict[str, Position]
    target_allocations: Dict[str, float]
    actual_allocations: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_rebalance: datetime
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance

class KellyCriterionCalculator:
    """
    Kelly Criterion position sizing for optimal bet sizing
    Maximizes long-term growth rate while managing risk
    """
    
    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        
    def calculate_kelly_fraction(self, asset: Asset, historical_returns: np.ndarray) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing
        
        Args:
            asset: Asset information
            historical_returns: Historical return data
            
        Returns:
            Kelly fraction (0-1, where 1 = 100% allocation)
        """
        try:
            if len(historical_returns) < 30:
                logger.warning(f"Insufficient data for Kelly calculation: {asset.symbol}")
                return 0.1  # Conservative default
            
            # Calculate win rate and average win/loss
            positive_returns = historical_returns[historical_returns > 0]
            negative_returns = historical_returns[historical_returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                return 0.1  # Conservative default
            
            win_rate = len(positive_returns) / len(historical_returns)
            avg_win = np.mean(positive_returns)
            avg_loss = abs(np.mean(negative_returns))
            
            # Kelly fraction = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-p
            if avg_loss == 0:
                return 0.1
                
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety constraints
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% per position
            
            # Risk adjustment based on volatility
            vol_adjustment = 1 - min(asset.volatility / 0.5, 0.8)  # Reduce for high vol
            kelly_fraction *= vol_adjustment
            
            logger.info(f"Kelly fraction for {asset.symbol}: {kelly_fraction:.4f} "
                       f"(win_rate: {win_rate:.3f}, b: {b:.3f})")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction for {asset.symbol}: {e}")
            return 0.1

class RiskParityOptimizer:
    """
    Risk Parity portfolio optimization
    Allocates capital so each position contributes equally to portfolio risk
    """
    
    def __init__(self):
        self.tolerance = 1e-8
        self.max_iterations = 1000
        
    def calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset"""
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        if portfolio_vol == 0:
            return np.zeros_like(weights)
        
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib
    
    def risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Objective function for risk parity optimization"""
        risk_contrib = self.calculate_risk_contributions(weights, cov_matrix)
        target_contrib = 1.0 / len(weights)  # Equal risk contribution
        
        # Sum of squared deviations from equal risk contribution
        objective = np.sum((risk_contrib - target_contrib) ** 2)
        return objective
    
    def optimize_weights(self, cov_matrix: np.ndarray, bounds: List[Tuple] = None) -> np.ndarray:
        """
        Optimize portfolio weights for risk parity
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            bounds: Weight bounds for each asset
            
        Returns:
            Optimal weights array
        """
        try:
            n_assets = cov_matrix.shape[0]
            
            # Default bounds: 0-40% per asset
            if bounds is None:
                bounds = [(0.01, 0.4)] * n_assets
            
            # Initial guess: equal weights
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints: weights sum to 1
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Optimize
            result = minimize(
                self.risk_parity_objective,
                initial_weights,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                weights = result.x
                risk_contrib = self.calculate_risk_contributions(weights, cov_matrix)
                logger.info(f"Risk parity optimization successful. Risk contributions: {risk_contrib}")
                return weights
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return initial_weights
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            n_assets = cov_matrix.shape[0]
            return np.array([1.0 / n_assets] * n_assets)

class ModernPortfolioTheory:
    """
    Modern Portfolio Theory optimization
    Efficient frontier calculation and mean-variance optimization
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def calculate_portfolio_metrics(self, weights: np.ndarray, expected_returns: np.ndarray, 
                                  cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_sharpe_ratio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                            bounds: List[Tuple] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            bounds: Weight bounds for each asset
            
        Returns:
            Tuple of (optimal_weights, portfolio_metrics)
        """
        try:
            n_assets = len(expected_returns)
            
            # Default bounds: 0-30% per asset
            if bounds is None:
                bounds = [(0.0, 0.3)] * n_assets
            
            # Objective function: negative Sharpe ratio (for minimization)
            def negative_sharpe(weights):
                metrics = self.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
                return -metrics['sharpe_ratio']
            
            # Constraints: weights sum to 1
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(optimal_weights, expected_returns, cov_matrix)
                logger.info(f"Sharpe optimization successful. Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
                return optimal_weights, metrics
            else:
                logger.warning("Sharpe optimization failed, using equal weights")
                equal_weights = np.array([1.0 / n_assets] * n_assets)
                metrics = self.calculate_portfolio_metrics(equal_weights, expected_returns, cov_matrix)
                return equal_weights, metrics
                
        except Exception as e:
            logger.error(f"Error in Sharpe optimization: {e}")
            n_assets = len(expected_returns)
            equal_weights = np.array([1.0 / n_assets] * n_assets)
            metrics = self.calculate_portfolio_metrics(equal_weights, expected_returns, cov_matrix)
            return equal_weights, metrics

class CorrelationAnalyzer:
    """
    Correlation analysis and diversification metrics
    """
    
    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        
    def calculate_rolling_correlation(self, returns_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Calculate rolling correlation matrix"""
        return returns_df.rolling(window=window).corr()
    
    def calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Calculate portfolio diversification ratio
        DR = (sum of weighted individual volatilities) / portfolio volatility
        Higher values indicate better diversification
        """
        try:
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.sum(weights * individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return 1.0
                
            diversification_ratio = weighted_avg_vol / portfolio_vol
            return diversification_ratio
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0
    
    def detect_correlation_clusters(self, correlation_matrix: np.ndarray, 
                                  asset_symbols: List[str], threshold: float = 0.7) -> List[List[str]]:
        """Detect groups of highly correlated assets"""
        try:
            clusters = []
            processed = set()
            
            for i, symbol_i in enumerate(asset_symbols):
                if symbol_i in processed:
                    continue
                    
                cluster = [symbol_i]
                processed.add(symbol_i)
                
                for j, symbol_j in enumerate(asset_symbols):
                    if i != j and symbol_j not in processed:
                        if abs(correlation_matrix[i, j]) >= threshold:
                            cluster.append(symbol_j)
                            processed.add(symbol_j)
                
                if len(cluster) > 1:
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting correlation clusters: {e}")
            return []

class PortfolioOptimizer:
    """
    Main portfolio optimization engine combining multiple strategies
    """
    
    def __init__(self, optimization_method: str = 'mixed'):
        """
        Initialize portfolio optimizer
        
        Args:
            optimization_method: 'kelly', 'risk_parity', 'sharpe', 'mixed'
        """
        self.optimization_method = optimization_method
        self.kelly_calculator = KellyCriterionCalculator()
        self.risk_parity_optimizer = RiskParityOptimizer()
        self.mpt_optimizer = ModernPortfolioTheory()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Optimization parameters
        self.max_position_size = 0.25  # Max 25% per position
        self.min_position_size = 0.01  # Min 1% per position
        self.correlation_threshold = 0.8  # High correlation threshold
        
    def prepare_optimization_data(self, assets: List[Asset], 
                                historical_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for optimization
        
        Returns:
            Tuple of (expected_returns, covariance_matrix, historical_returns_matrix)
        """
        try:
            symbols = [asset.symbol for asset in assets]
            
            # Align historical data
            returns_df = pd.DataFrame()
            for symbol in symbols:
                if symbol in historical_data and not historical_data[symbol].empty:
                    returns = historical_data[symbol]['close'].pct_change().dropna()
                    returns_df[symbol] = returns
                else:
                    # Use expected return as fallback
                    asset = next(a for a in assets if a.symbol == symbol)
                    synthetic_returns = np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), 252)
                    returns_df[symbol] = synthetic_returns
            
            # Calculate expected returns (annualized)
            expected_returns = returns_df.mean() * 252
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov() * 252
            
            return expected_returns.values, cov_matrix.values, returns_df.values
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            n_assets = len(assets)
            return (np.array([0.1] * n_assets), 
                   np.eye(n_assets) * 0.04, 
                   np.random.normal(0, 0.02, (252, n_assets)))
    
    def optimize_portfolio(self, assets: List[Asset], 
                         historical_data: Dict[str, pd.DataFrame],
                         current_portfolio: Optional[Portfolio] = None) -> Dict[str, float]:
        """
        Optimize portfolio allocation using selected method
        
        Args:
            assets: List of available assets
            historical_data: Historical price/return data
            current_portfolio: Current portfolio state (for transaction cost consideration)
            
        Returns:
            Optimal allocation weights dictionary
        """
        try:
            logger.info(f"Starting portfolio optimization using {self.optimization_method} method")
            
            # Prepare optimization data
            expected_returns, cov_matrix, historical_returns = self.prepare_optimization_data(assets, historical_data)
            symbols = [asset.symbol for asset in assets]
            
            # Apply optimization method
            if self.optimization_method == 'kelly':
                weights = self._optimize_kelly(assets, historical_returns)
            elif self.optimization_method == 'risk_parity':
                weights = self.risk_parity_optimizer.optimize_weights(cov_matrix)
            elif self.optimization_method == 'sharpe':
                weights, _ = self.mpt_optimizer.optimize_sharpe_ratio(expected_returns, cov_matrix)
            elif self.optimization_method == 'mixed':
                weights = self._optimize_mixed(assets, expected_returns, cov_matrix, historical_returns)
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            # Apply constraints and adjustments
            weights = self._apply_constraints(weights, assets, cov_matrix)
            
            # Convert to dictionary
            allocation = {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            # Calculate portfolio metrics
            metrics = self.mpt_optimizer.calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            diversification_ratio = self.correlation_analyzer.calculate_diversification_ratio(weights, cov_matrix)
            
            logger.info(f"Portfolio optimization complete:")
            logger.info(f"  Expected Return: {metrics['return']:.4f}")
            logger.info(f"  Volatility: {metrics['volatility']:.4f}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"  Diversification Ratio: {diversification_ratio:.4f}")
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Return equal weights as fallback
            n_assets = len(assets)
            return {asset.symbol: 1.0/n_assets for asset in assets}
    
    def _optimize_kelly(self, assets: List[Asset], historical_returns: np.ndarray) -> np.ndarray:
        """Optimize using Kelly Criterion"""
        weights = []
        for i, asset in enumerate(assets):
            asset_returns = historical_returns[:, i]
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(asset, asset_returns)
            weights.append(kelly_fraction)
        
        # Normalize to sum to 1
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.array([1.0/len(assets)] * len(assets))
        
        return weights
    
    def _optimize_mixed(self, assets: List[Asset], expected_returns: np.ndarray, 
                       cov_matrix: np.ndarray, historical_returns: np.ndarray) -> np.ndarray:
        """Optimize using mixed approach combining multiple methods"""
        
        # Get weights from different methods
        kelly_weights = self._optimize_kelly(assets, historical_returns)
        risk_parity_weights = self.risk_parity_optimizer.optimize_weights(cov_matrix)
        sharpe_weights, _ = self.mpt_optimizer.optimize_sharpe_ratio(expected_returns, cov_matrix)
        
        # Weighted combination
        mixed_weights = (0.4 * kelly_weights + 
                        0.3 * risk_parity_weights + 
                        0.3 * sharpe_weights)
        
        # Normalize
        mixed_weights = mixed_weights / np.sum(mixed_weights)
        
        return mixed_weights
    
    def _apply_constraints(self, weights: np.ndarray, assets: List[Asset], 
                          cov_matrix: np.ndarray) -> np.ndarray:
        """Apply position size and correlation constraints"""
        
        # Apply position size limits
        weights = np.clip(weights, self.min_position_size, self.max_position_size)
        
        # Check for high correlations and adjust
        correlation_matrix = np.corrcoef(cov_matrix)
        symbols = [asset.symbol for asset in assets]
        clusters = self.correlation_analyzer.detect_correlation_clusters(
            correlation_matrix, symbols, self.correlation_threshold
        )
        
        # Reduce weights in highly correlated clusters
        for cluster in clusters:
            cluster_indices = [i for i, asset in enumerate(assets) if asset.symbol in cluster]
            if len(cluster_indices) > 1:
                # Reduce total weight of cluster
                cluster_weight = np.sum(weights[cluster_indices])
                max_cluster_weight = 0.4  # Max 40% in correlated assets
                
                if cluster_weight > max_cluster_weight:
                    reduction_factor = max_cluster_weight / cluster_weight
                    for idx in cluster_indices:
                        weights[idx] *= reduction_factor
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights

# Example usage and testing
if __name__ == "__main__":
    # Example assets
    assets = [
        Asset("AAPL", "Technology", 3000000000000, 175.50, 0.25, 0.12, 1.2),
        Asset("MSFT", "Technology", 2800000000000, 335.20, 0.22, 0.11, 0.9),
        Asset("GOOGL", "Technology", 1700000000000, 135.80, 0.28, 0.13, 1.1),
        Asset("JPM", "Finance", 450000000000, 145.30, 0.30, 0.10, 1.3),
        Asset("JNJ", "Healthcare", 420000000000, 165.75, 0.18, 0.08, 0.7),
    ]
    
    # Create sample historical data
    historical_data = {}
    for asset in assets:
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), len(dates))))
        historical_data[asset.symbol] = pd.DataFrame({'close': prices}, index=dates)
    
    # Test portfolio optimization
    optimizer = PortfolioOptimizer(optimization_method='mixed')
    optimal_allocation = optimizer.optimize_portfolio(assets, historical_data)
    
    print("Optimal Portfolio Allocation:")
    for symbol, weight in optimal_allocation.items():
        print(f"  {symbol}: {weight:.4f} ({weight*100:.2f}%)")
