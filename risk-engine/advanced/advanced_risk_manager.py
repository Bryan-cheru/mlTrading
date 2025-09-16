"""
Advanced Risk Management System for Institutional Trading
========================================================

This module implements sophisticated risk management techniques used by institutional traders:
- Dynamic position sizing using Kelly Criterion
- Value at Risk (VaR) calculations with multiple methodologies  
- Portfolio optimization with modern portfolio theory
- Real-time risk monitoring and automated controls
- Advanced stop-loss and take-profit mechanisms
- Correlation-based risk adjustments
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    """
    Institutional-grade risk management system
    
    Features:
    - Kelly Criterion position sizing
    - Multiple VaR methodologies (Historical, Parametric, Monte Carlo)
    - Portfolio optimization
    - Dynamic risk controls
    - Correlation analysis
    """
    
    def __init__(self, initial_capital: float = 100000, max_position_size: float = 0.1, 
                 var_confidence: float = 0.05, risk_free_rate: float = 0.02):
        """
        Initialize the advanced risk management system
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            var_confidence: VaR confidence level (0.05 = 95% VaR)
            risk_free_rate: Risk-free rate for Sharpe calculations
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.var_confidence = var_confidence
        self.risk_free_rate = risk_free_rate
        
        # Risk tracking
        self.positions = {}  # Current positions
        self.returns_history = []
        self.pnl_history = []
        self.drawdown_history = []
        self.var_history = []
        
        # Risk limits
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_drawdown = 0.10    # 10% max drawdown
        self.correlation_threshold = 0.7  # Correlation limit for position sizing
        
        # Performance metrics
        self.metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'calmar_ratio': 0.0
        }
        
        logger.info(f"🛡️  Advanced Risk Manager initialized")
        logger.info(f"💰 Initial capital: ${initial_capital:,.2f}")
        logger.info(f"📊 Max position size: {max_position_size:.1%}")
        logger.info(f"⚠️  VaR confidence: {(1-var_confidence):.1%}")
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float, 
                                 confidence: float = 1.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive value)
            confidence: Confidence in the prediction (0-1)
        
        Returns:
            Optimal position size as fraction of capital
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss  # Odds ratio
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply confidence adjustment
            kelly_fraction *= confidence
            
            # Apply safety margin (typically use 25-50% of full Kelly)
            kelly_fraction *= 0.25  # Conservative Kelly
            
            # Cap at maximum position size
            kelly_fraction = min(kelly_fraction, self.max_position_size)
            kelly_fraction = max(kelly_fraction, 0.0)  # No negative positions
            
            logger.debug(f"📊 Kelly Criterion: {kelly_fraction:.3f} "
                        f"(win_rate: {win_rate:.3f}, ratio: {b:.3f}, confidence: {confidence:.3f})")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"❌ Error calculating Kelly criterion: {e}")
            return 0.01  # Conservative fallback
    
    def calculate_var(self, returns: np.ndarray, method: str = 'historical', 
                     time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methodologies
        
        Args:
            returns: Array of historical returns
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            time_horizon: Time horizon in days
        
        Returns:
            Dictionary with VaR values and statistics
        """
        if len(returns) < 30:
            return {'var': 0.0, 'expected_shortfall': 0.0, 'method': method}
        
        # Scale for time horizon
        scaling_factor = np.sqrt(time_horizon)
        
        try:
            if method == 'historical':
                # Historical simulation
                sorted_returns = np.sort(returns)
                var_index = int(np.floor(len(sorted_returns) * self.var_confidence))
                var = -sorted_returns[var_index] * scaling_factor
                
                # Expected Shortfall (Conditional VaR)
                es = -np.mean(sorted_returns[:var_index]) * scaling_factor
                
            elif method == 'parametric':
                # Assume normal distribution
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                # VaR using normal distribution
                var = -(mean_return + stats.norm.ppf(self.var_confidence) * std_return) * scaling_factor
                
                # Expected Shortfall for normal distribution
                es = -(mean_return + std_return * stats.norm.pdf(stats.norm.ppf(self.var_confidence)) / self.var_confidence) * scaling_factor
                
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                n_simulations = 10000
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Generate random scenarios
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                
                # Calculate VaR and ES
                sorted_sim = np.sort(simulated_returns)
                var_index = int(np.floor(n_simulations * self.var_confidence))
                var = -sorted_sim[var_index] * scaling_factor
                es = -np.mean(sorted_sim[:var_index]) * scaling_factor
                
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            result = {
                'var': max(var, 0.0),
                'expected_shortfall': max(es, 0.0),
                'method': method,
                'confidence': 1 - self.var_confidence,
                'time_horizon': time_horizon,
                'observations': len(returns)
            }
            
            self.var_history.append(result['var'])
            logger.debug(f"📊 {method.title()} VaR: {result['var']:.4f}, ES: {result['expected_shortfall']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating VaR: {e}")
            return {'var': 0.05, 'expected_shortfall': 0.075, 'method': method}
    
    def optimize_portfolio(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                          current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize portfolio using Modern Portfolio Theory
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            current_positions: Current position sizes
        
        Returns:
            Optimal position weights
        """
        try:
            n_assets = len(expected_returns)
            
            if n_assets == 0:
                return {}
            
            # Objective function: minimize portfolio variance for given return
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            ]
            
            # Bounds for each weight (between 0 and max_position_size)
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(portfolio_variance, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                
                # Convert to dictionary
                asset_names = list(current_positions.keys())[:n_assets]
                optimal_positions = dict(zip(asset_names, optimal_weights))
                
                # Calculate portfolio statistics
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_variance = result.fun
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                logger.info(f"📈 Portfolio optimization successful")
                logger.info(f"🎯 Expected return: {portfolio_return:.4f}")
                logger.info(f"📊 Portfolio volatility: {portfolio_volatility:.4f}")
                
                return optimal_positions
                
            else:
                logger.warning("⚠️  Portfolio optimization failed, using current positions")
                return current_positions
                
        except Exception as e:
            logger.error(f"❌ Error in portfolio optimization: {e}")
            return current_positions
    
    def calculate_position_size(self, signal_strength: float, historical_performance: Dict[str, float],
                               current_volatility: float, correlation_adjustment: float = 1.0) -> float:
        """
        Calculate optimal position size combining multiple risk factors
        
        Args:
            signal_strength: Strength of trading signal (0-1)
            historical_performance: Dict with win_rate, avg_win, avg_loss
            current_volatility: Current market volatility
            correlation_adjustment: Adjustment for portfolio correlation
        
        Returns:
            Position size as fraction of capital
        """
        try:
            # Base position size using Kelly Criterion
            kelly_size = self.calculate_kelly_criterion(
                historical_performance.get('win_rate', 0.5),
                historical_performance.get('avg_win', 0.01),
                historical_performance.get('avg_loss', 0.01),
                signal_strength
            )
            
            # Volatility adjustment
            baseline_vol = 0.01  # 1% baseline daily volatility
            vol_adjustment = baseline_vol / (current_volatility + 1e-10)
            vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Limit adjustment
            
            # Apply adjustments
            adjusted_size = kelly_size * vol_adjustment * correlation_adjustment
            
            # Final safety checks
            adjusted_size = min(adjusted_size, self.max_position_size)
            adjusted_size = max(adjusted_size, 0.0)
            
            # Check against daily loss limit
            max_loss_size = self.max_daily_loss / (current_volatility + 1e-10)
            adjusted_size = min(adjusted_size, max_loss_size)
            
            logger.debug(f"📊 Position sizing: Kelly={kelly_size:.3f}, "
                        f"Vol_adj={vol_adjustment:.3f}, Final={adjusted_size:.3f}")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.01  # Conservative fallback
    
    def update_portfolio_metrics(self, current_positions: Dict[str, float], 
                                current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update portfolio performance metrics
        
        Args:
            current_positions: Current position sizes
            current_prices: Current asset prices
        
        Returns:
            Updated performance metrics
        """
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            for asset, position in current_positions.items():
                if asset in current_prices:
                    portfolio_value += position * current_prices[asset]
            
            # Update capital
            self.current_capital = portfolio_value
            
            # Calculate returns
            total_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            # Update return history
            if len(self.pnl_history) > 0:
                period_return = (portfolio_value - self.pnl_history[-1]) / self.pnl_history[-1]
                self.returns_history.append(period_return)
            
            self.pnl_history.append(portfolio_value)
            
            # Calculate drawdown
            peak = max(self.pnl_history) if self.pnl_history else portfolio_value
            drawdown = (peak - portfolio_value) / peak
            self.drawdown_history.append(drawdown)
            
            # Calculate metrics if we have enough data
            if len(self.returns_history) >= 30:
                returns_array = np.array(self.returns_history)
                
                # Sharpe ratio
                excess_returns = returns_array - (self.risk_free_rate / 252)  # Daily risk-free rate
                sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
                
                # Sortino ratio (downside deviation)
                negative_returns = returns_array[returns_array < 0]
                downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 1e-10
                sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
                
                # Maximum drawdown
                max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
                
                # VaR
                var_result = self.calculate_var(returns_array)
                var_95 = var_result['var']
                
                # Calmar ratio
                annual_return = total_return * (252 / len(self.returns_history))
                calmar_ratio = annual_return / (max_drawdown + 1e-10)
                
                # Update metrics
                self.metrics.update({
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'calmar_ratio': calmar_ratio,
                    'current_drawdown': drawdown,
                    'portfolio_value': portfolio_value
                })
            
            logger.debug(f"💰 Portfolio value: ${portfolio_value:,.2f}, "
                        f"Return: {total_return:.3%}, Drawdown: {drawdown:.3%}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio metrics: {e}")
            return self.metrics
    
    def check_risk_limits(self, proposed_position: Dict[str, float]) -> Dict[str, Union[bool, str]]:
        """
        Check if proposed position violates risk limits
        
        Args:
            proposed_position: Proposed position sizes
        
        Returns:
            Risk check results
        """
        violations = []
        
        try:
            # Check individual position sizes
            for asset, size in proposed_position.items():
                if abs(size) > self.max_position_size:
                    violations.append(f"Position size limit exceeded for {asset}: {size:.3f}")
            
            # Check total position size
            total_exposure = sum(abs(size) for size in proposed_position.values())
            if total_exposure > 1.0:
                violations.append(f"Total exposure exceeds 100%: {total_exposure:.3f}")
            
            # Check current drawdown
            current_drawdown = self.metrics.get('current_drawdown', 0.0)
            if current_drawdown > self.max_drawdown:
                violations.append(f"Maximum drawdown exceeded: {current_drawdown:.3%}")
            
            # Check daily loss
            if len(self.returns_history) > 0:
                last_return = self.returns_history[-1]
                if last_return < -self.max_daily_loss:
                    violations.append(f"Daily loss limit exceeded: {last_return:.3%}")
            
            # Check VaR
            current_var = self.metrics.get('var_95', 0.0)
            if current_var > 0.05:  # 5% VaR limit
                violations.append(f"VaR limit exceeded: {current_var:.3%}")
            
            result = {
                'approved': len(violations) == 0,
                'violations': violations,
                'total_exposure': total_exposure,
                'current_drawdown': current_drawdown
            }
            
            if violations:
                logger.warning(f"⚠️  Risk limit violations: {'; '.join(violations)}")
            else:
                logger.debug("✅ Risk limits check passed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return {'approved': False, 'violations': [f"Risk check error: {str(e)}"]}
    
    def calculate_stop_loss_take_profit(self, entry_price: float, position_size: float,
                                       volatility: float, risk_reward_ratio: float = 2.0) -> Dict[str, float]:
        """
        Calculate dynamic stop-loss and take-profit levels
        
        Args:
            entry_price: Entry price
            position_size: Position size (positive for long, negative for short)
            volatility: Current volatility estimate
            risk_reward_ratio: Reward-to-risk ratio
        
        Returns:
            Stop-loss and take-profit levels
        """
        try:
            # ATR-based stop loss (2x volatility)
            atr_multiplier = 2.0
            stop_distance = volatility * atr_multiplier
            
            if position_size > 0:  # Long position
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * risk_reward_ratio)
            else:  # Short position
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * risk_reward_ratio)
            
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(take_profit - entry_price)
            
            result = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_per_share': risk_per_share,
                'reward_per_share': reward_per_share,
                'actual_risk_reward': reward_per_share / (risk_per_share + 1e-10)
            }
            
            logger.debug(f"🎯 Stop/Target: Entry=${entry_price:.2f}, "
                        f"Stop=${stop_loss:.2f}, Target=${take_profit:.2f}, "
                        f"R:R={result['actual_risk_reward']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating stop/target: {e}")
            return {
                'stop_loss': entry_price * 0.98 if position_size > 0 else entry_price * 1.02,
                'take_profit': entry_price * 1.04 if position_size > 0 else entry_price * 0.96,
                'risk_per_share': entry_price * 0.02,
                'reward_per_share': entry_price * 0.04,
                'actual_risk_reward': 2.0
            }
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get comprehensive risk summary"""
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_return': self.metrics.get('total_return', 0.0)
            },
            'risk_metrics': {
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': self.metrics.get('sortino_ratio', 0.0),
                'max_drawdown': self.metrics.get('max_drawdown', 0.0),
                'current_drawdown': self.metrics.get('current_drawdown', 0.0),
                'var_95': self.metrics.get('var_95', 0.0),
                'calmar_ratio': self.metrics.get('calmar_ratio', 0.0)
            },
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'var_confidence': 1 - self.var_confidence
            },
            'positions': self.positions,
            'history_length': len(self.returns_history)
        }

# Testing and Usage Example
if __name__ == "__main__":
    logger.info("🧪 Testing Advanced Risk Management System...")
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(
        initial_capital=100000,
        max_position_size=0.15,
        var_confidence=0.05,
        risk_free_rate=0.02
    )
    
    try:
        # Test Kelly Criterion
        logger.info("📊 Testing Kelly Criterion...")
        kelly_size = risk_manager.calculate_kelly_criterion(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.015,
            confidence=0.8
        )
        logger.info(f"✅ Kelly position size: {kelly_size:.3%}")
        
        # Test VaR calculations
        logger.info("📊 Testing VaR calculations...")
        
        # Generate sample returns
        np.random.seed(42)
        sample_returns = np.random.normal(0.0005, 0.02, 252)  # 1 year of daily returns
        
        # Test different VaR methods
        for method in ['historical', 'parametric', 'monte_carlo']:
            var_result = risk_manager.calculate_var(sample_returns, method=method)
            logger.info(f"✅ {method.title()} VaR: {var_result['var']:.3%}, "
                       f"ES: {var_result['expected_shortfall']:.3%}")
        
        # Test position sizing
        logger.info("📊 Testing position sizing...")
        historical_perf = {
            'win_rate': 0.55,
            'avg_win': 0.025,
            'avg_loss': 0.018
        }
        
        position_size = risk_manager.calculate_position_size(
            signal_strength=0.75,
            historical_performance=historical_perf,
            current_volatility=0.02,
            correlation_adjustment=0.9
        )
        logger.info(f"✅ Recommended position size: {position_size:.3%}")
        
        # Test stop-loss/take-profit
        logger.info("📊 Testing stop-loss/take-profit...")
        entry_price = 4500
        sl_tp = risk_manager.calculate_stop_loss_take_profit(
            entry_price=entry_price,
            position_size=0.1,
            volatility=50,
            risk_reward_ratio=2.5
        )
        logger.info(f"✅ Entry: ${entry_price}, Stop: ${sl_tp['stop_loss']:.2f}, "
                   f"Target: ${sl_tp['take_profit']:.2f}")
        
        # Test portfolio metrics
        logger.info("📊 Testing portfolio metrics...")
        test_positions = {'ES': 0.08, 'NQ': 0.05}
        test_prices = {'ES': 4500, 'NQ': 15000}
        
        for i in range(10):
            # Simulate price movements
            test_prices['ES'] += np.random.normal(0, 25)
            test_prices['NQ'] += np.random.normal(0, 100)
            
            metrics = risk_manager.update_portfolio_metrics(test_positions, test_prices)
        
        logger.info(f"✅ Portfolio metrics updated: {len(risk_manager.returns_history)} periods")
        
        # Test risk limits
        logger.info("📊 Testing risk limits...")
        risky_position = {'ES': 0.25, 'NQ': 0.20}  # Exceeds limits
        risk_check = risk_manager.check_risk_limits(risky_position)
        logger.info(f"✅ Risk check: Approved={risk_check['approved']}, "
                   f"Violations={len(risk_check['violations'])}")
        
        # Get risk summary
        logger.info("📊 Getting risk summary...")
        risk_summary = risk_manager.get_risk_summary()
        logger.info(f"✅ Risk summary generated with {risk_summary['history_length']} observations")
        
        # Display key metrics
        capital = risk_summary['capital']
        metrics = risk_summary['risk_metrics']
        
        logger.info("📈 RISK MANAGEMENT SUMMARY")
        logger.info(f"💰 Portfolio Value: ${capital['current']:,.2f} "
                   f"(Return: {capital['total_return']:.2%})")
        logger.info(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"⚠️  VaR (95%): {metrics['var_95']:.2%}")
        
        logger.info("🎉 Advanced Risk Management System testing complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in risk management testing: {e}")
        import traceback
        traceback.print_exc()
