"""
Multi-Instrument Trading System for Institutional ML Trading
Manages trading across multiple futures contracts (ES, NQ, YM, RTY) with correlation analysis,
pair trading strategies, portfolio optimization, and risk management across instruments.
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import sys
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.extend([
    project_root,
    os.path.join(project_root, 'ml-models', 'integration'),
    os.path.join(project_root, 'risk-engine', 'advanced'),
    os.path.join(project_root, 'monitoring')
])

try:
    from institutional_trading_system import InstitutionalMLTradingSystem
    from advanced_risk_manager import AdvancedRiskManager
    from performance_dashboard import PerformanceAnalytics
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstrumentType(Enum):
    """Supported futures instruments"""
    ES = "ES"  # S&P 500 E-mini
    NQ = "NQ"  # NASDAQ 100 E-mini  
    YM = "YM"  # Dow Jones E-mini
    RTY = "RTY"  # Russell 2000 E-mini

@dataclass
class InstrumentConfig:
    """Configuration for individual trading instruments"""
    symbol: str
    description: str
    tick_size: float
    tick_value: float
    margin_req: float
    session_hours: Tuple[int, int]  # (start_hour, end_hour)
    volatility_factor: float = 1.0
    correlation_threshold: float = 0.7

@dataclass
class Signal:
    """Trading signal for an instrument"""
    instrument: str
    signal_type: str  # 'LONG', 'SHORT', 'CLOSE'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    reasoning: str
    risk_score: float = 0.0

@dataclass
class Position:
    """Position tracking for multi-instrument portfolio"""
    instrument: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    stop_loss: float
    take_profit: float

class CorrelationAnalyzer:
    """
    Analyzes correlations between instruments for risk management and pair trading
    """
    
    def __init__(self, lookback_period: int = 252):
        """
        Initialize correlation analyzer
        
        Args:
            lookback_period: Number of periods for correlation calculation
        """
        self.lookback_period = lookback_period
        self.price_history: Dict[str, List[float]] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        logger.info(f"📊 Correlation Analyzer initialized (lookback: {lookback_period})")
    
    def update_prices(self, instrument: str, price: float, timestamp: datetime = None):
        """Update price history for correlation calculations"""
        if instrument not in self.price_history:
            self.price_history[instrument] = []
        
        self.price_history[instrument].append(price)
        
        # Keep only recent data
        if len(self.price_history[instrument]) > self.lookback_period:
            self.price_history[instrument].pop(0)
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between all instruments
        
        Returns:
            DataFrame with correlation matrix
        """
        if len(self.price_history) < 2:
            return pd.DataFrame()
        
        # Ensure all instruments have sufficient data
        min_length = min(len(prices) for prices in self.price_history.values())
        if min_length < 30:  # Need at least 30 data points
            logger.warning(f"⚠️ Insufficient data for correlation analysis: {min_length} points")
            return pd.DataFrame()
        
        # Create DataFrame with aligned data
        data = {}
        for instrument, prices in self.price_history.items():
            data[instrument] = prices[-min_length:]  # Use last min_length points
        
        df = pd.DataFrame(data)
        
        # Calculate returns for correlation
        returns_df = df.pct_change().dropna()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        logger.info(f"📊 Updated correlation matrix: {list(self.correlation_matrix.columns)}")
        return self.correlation_matrix
    
    def get_correlation(self, instrument1: str, instrument2: str) -> float:
        """Get correlation between two specific instruments"""
        if self.correlation_matrix.empty:
            self.calculate_correlations()
        
        try:
            return self.correlation_matrix.loc[instrument1, instrument2]
        except (KeyError, IndexError):
            return 0.0
    
    def find_pairs(self, min_correlation: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated instrument pairs for pair trading
        
        Args:
            min_correlation: Minimum correlation for pair identification
            
        Returns:
            List of tuples (instrument1, instrument2, correlation)
        """
        if self.correlation_matrix.empty:
            self.calculate_correlations()
        
        pairs = []
        instruments = list(self.correlation_matrix.columns)
        
        for i, inst1 in enumerate(instruments):
            for inst2 in instruments[i+1:]:
                corr = abs(self.correlation_matrix.loc[inst1, inst2])
                if corr >= min_correlation:
                    pairs.append((inst1, inst2, corr))
        
        # Sort by correlation strength
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"🔍 Found {len(pairs)} correlated pairs (min correlation: {min_correlation})")
        return pairs

class PortfolioOptimizer:
    """
    Optimizes portfolio allocation across multiple instruments using Modern Portfolio Theory
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.returns_history: Dict[str, List[float]] = {}
        logger.info(f"📈 Portfolio Optimizer initialized (risk-free rate: {risk_free_rate*100:.1f}%)")
    
    def update_returns(self, instrument: str, return_pct: float):
        """Update returns history for optimization"""
        if instrument not in self.returns_history:
            self.returns_history[instrument] = []
        
        self.returns_history[instrument].append(return_pct)
        
        # Keep reasonable history
        if len(self.returns_history[instrument]) > 252:  # ~1 year
            self.returns_history[instrument].pop(0)
    
    def calculate_optimal_weights(self, instruments: List[str], 
                                target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using mean-variance optimization
        
        Args:
            instruments: List of instruments to include in optimization
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            
        Returns:
            Dictionary of optimal weights {instrument: weight}
        """
        # Filter instruments with sufficient data
        valid_instruments = []
        returns_data = []
        
        for instrument in instruments:
            if (instrument in self.returns_history and 
                len(self.returns_history[instrument]) >= 30):
                valid_instruments.append(instrument)
                returns_data.append(self.returns_history[instrument][-60:])  # Last 60 periods
        
        if len(valid_instruments) < 2:
            logger.warning("⚠️ Insufficient data for portfolio optimization")
            return {inst: 1.0/len(instruments) for inst in instruments}  # Equal weights
        
        # Create returns matrix
        min_length = min(len(returns) for returns in returns_data)
        returns_matrix = np.array([returns[-min_length:] for returns in returns_data]).T
        
        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Number of assets
        n_assets = len(valid_instruments)
        
        # Optimization constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only portfolio
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        try:
            if target_return is None:
                # Maximize Sharpe ratio
                def negative_sharpe(weights):
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    if portfolio_volatility == 0:
                        return -float('inf')
                    return -(portfolio_return - self.risk_free_rate/252) / portfolio_volatility
                
                result = minimize(negative_sharpe, x0, method='SLSQP', 
                                bounds=bounds, constraints=constraints)
            else:
                # Minimize risk for target return
                target_constraint = {'type': 'eq', 
                                   'fun': lambda x: np.dot(x, expected_returns) - target_return}
                
                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                result = minimize(portfolio_variance, x0, method='SLSQP',
                                bounds=bounds, constraints=[constraints, target_constraint])
            
            if result.success:
                optimal_weights = dict(zip(valid_instruments, result.x))
                
                # Add zero weights for instruments not included
                for instrument in instruments:
                    if instrument not in optimal_weights:
                        optimal_weights[instrument] = 0.0
                
                logger.info(f"📊 Optimal weights calculated: {optimal_weights}")
                return optimal_weights
            else:
                logger.warning("⚠️ Portfolio optimization failed, using equal weights")
                return {inst: 1.0/len(instruments) for inst in instruments}
                
        except Exception as e:
            logger.error(f"❌ Portfolio optimization error: {e}")
            return {inst: 1.0/len(instruments) for inst in instruments}

class MultiInstrumentTradingSystem:
    """
    Main multi-instrument trading system coordinating all components
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize multi-instrument trading system
        
        Args:
            initial_capital: Starting portfolio capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize components
        self.correlation_analyzer = CorrelationAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.performance_analytics = PerformanceAnalytics()
        
        # Initialize risk manager if available
        try:
            self.risk_manager = AdvancedRiskManager(
                initial_capital=initial_capital,
                max_position_size=0.15,  # 15% max per instrument
                var_confidence=0.95
            )
        except:
            self.risk_manager = None
            logger.warning("⚠️ Advanced Risk Manager not available")
        
        # Trading system for each instrument
        self.trading_systems: Dict[str, Any] = {}
        
        # Instrument configurations
        self.instruments = {
            InstrumentType.ES.value: InstrumentConfig(
                symbol="ES", description="S&P 500 E-mini",
                tick_size=0.25, tick_value=12.50, margin_req=14000,
                session_hours=(9, 16), volatility_factor=1.0
            ),
            InstrumentType.NQ.value: InstrumentConfig(
                symbol="NQ", description="NASDAQ 100 E-mini",
                tick_size=0.25, tick_value=5.00, margin_req=18000,
                session_hours=(9, 16), volatility_factor=1.2
            ),
            InstrumentType.YM.value: InstrumentConfig(
                symbol="YM", description="Dow Jones E-mini",
                tick_size=1.0, tick_value=5.00, margin_req=8000,
                session_hours=(9, 16), volatility_factor=0.8
            ),
            InstrumentType.RTY.value: InstrumentConfig(
                symbol="RTY", description="Russell 2000 E-mini",
                tick_size=0.10, tick_value=5.00, margin_req=7000,
                session_hours=(9, 16), volatility_factor=1.5
            )
        }
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.pending_signals: List[Signal] = []
        self.market_data: Dict[str, Dict] = {}
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_pnl: List[Dict] = []
        
        logger.info(f"🏦 Multi-Instrument Trading System initialized")
        logger.info(f"💰 Initial capital: ${initial_capital:,.2f}")
        logger.info(f"📈 Instruments: {list(self.instruments.keys())}")
    
    def initialize_trading_systems(self):
        """Initialize individual trading systems for each instrument"""
        for instrument_code in self.instruments.keys():
            try:
                # Create individual trading system for each instrument
                system = InstitutionalMLTradingSystem(
                    base_timeframe='1min',
                    confidence_threshold=0.65
                )
                self.trading_systems[instrument_code] = system
                logger.info(f"✅ Trading system initialized for {instrument_code}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize trading system for {instrument_code}: {e}")
    
    def update_market_data(self, instrument: str, data: Dict):
        """
        Update market data for an instrument
        
        Args:
            instrument: Instrument symbol
            data: Market data dictionary with OHLCV
        """
        self.market_data[instrument] = data
        
        # Update correlation analyzer
        if 'close' in data:
            self.correlation_analyzer.update_prices(instrument, data['close'])
        
        # Update portfolio optimizer with returns
        if instrument in self.market_data:
            prev_data = self.market_data.get(f"{instrument}_prev")
            if prev_data and 'close' in prev_data and 'close' in data:
                return_pct = ((data['close'] - prev_data['close']) / prev_data['close']) * 100
                self.portfolio_optimizer.update_returns(instrument, return_pct)
            
            # Store previous data
            self.market_data[f"{instrument}_prev"] = data.copy()
    
    async def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals for all instruments
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Generate signals for each instrument
        for instrument, system in self.trading_systems.items():
            try:
                if instrument not in self.market_data:
                    continue
                
                # Create sample market data for signal generation
                data = self.market_data[instrument]
                
                # Generate signal using institutional trading system
                # This is a simplified example - in practice, you'd use the full system
                confidence = np.random.uniform(0.4, 0.9)
                
                if confidence > 0.65:  # Confidence threshold
                    signal_type = np.random.choice(['LONG', 'SHORT'])
                    
                    signal = Signal(
                        instrument=instrument,
                        signal_type=signal_type,
                        confidence=confidence,
                        entry_price=data.get('close', 4400),
                        stop_loss=data.get('close', 4400) * (0.995 if signal_type == 'LONG' else 1.005),
                        take_profit=data.get('close', 4400) * (1.010 if signal_type == 'LONG' else 0.990),
                        timestamp=datetime.now(),
                        reasoning=f"ML model confidence: {confidence:.2f}",
                        risk_score=1.0 - confidence
                    )
                    
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"❌ Error generating signal for {instrument}: {e}")
        
        logger.info(f"🎯 Generated {len(signals)} signals")
        return signals
    
    def check_correlations(self, new_signal: Signal) -> bool:
        """
        Check if new signal conflicts with existing positions due to high correlation
        
        Args:
            new_signal: New signal to evaluate
            
        Returns:
            True if signal is acceptable, False if rejected due to correlation
        """
        correlations = self.correlation_analyzer.calculate_correlations()
        
        if correlations.empty:
            return True  # No correlation data yet
        
        for instrument, position in self.positions.items():
            if instrument == new_signal.instrument:
                continue
            
            try:
                correlation = abs(correlations.loc[new_signal.instrument, instrument])
                
                # Check if highly correlated positions in same direction
                if (correlation > 0.8 and 
                    ((new_signal.signal_type == 'LONG' and position.quantity > 0) or
                     (new_signal.signal_type == 'SHORT' and position.quantity < 0))):
                    
                    logger.warning(f"⚠️ Signal rejected: High correlation ({correlation:.2f}) "
                                 f"between {new_signal.instrument} and {instrument}")
                    return False
                    
            except (KeyError, IndexError):
                continue
        
        return True
    
    def calculate_position_size(self, signal: Signal) -> int:
        """
        Calculate optimal position size for a signal considering portfolio allocation
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size in contracts
        """
        instrument_config = self.instruments[signal.instrument]
        
        # Get optimal portfolio weights
        active_instruments = list(self.instruments.keys())
        optimal_weights = self.portfolio_optimizer.calculate_optimal_weights(active_instruments)
        
        # Calculate base position size
        instrument_allocation = optimal_weights.get(signal.instrument, 0.25)  # Default 25%
        allocated_capital = self.current_capital * instrument_allocation
        
        # Consider margin requirements
        max_contracts_by_margin = int(allocated_capital / instrument_config.margin_req)
        
        # Apply confidence-based sizing
        confidence_factor = min(signal.confidence, 0.9)  # Cap at 90%
        base_size = max(1, int(max_contracts_by_margin * confidence_factor))
        
        # Risk management limits
        if self.risk_manager:
            # Apply Kelly Criterion if enough data
            try:
                kelly_size = self.risk_manager.calculate_kelly_criterion(0.6, 100, -80)  # Sample values
                kelly_contracts = int((self.current_capital * kelly_size / 100) / instrument_config.margin_req)
                base_size = min(base_size, kelly_contracts)
            except:
                pass
        
        # Final position size
        position_size = max(1, min(base_size, 5))  # Between 1 and 5 contracts
        
        logger.info(f"📊 Position size for {signal.instrument}: {position_size} contracts "
                   f"(allocation: {instrument_allocation:.1%}, confidence: {signal.confidence:.2f})")
        
        return position_size
    
    async def execute_signal(self, signal: Signal) -> bool:
        """
        Execute a trading signal
        
        Args:
            signal: Signal to execute
            
        Returns:
            True if executed successfully, False otherwise
        """
        try:
            # Check correlations
            if not self.check_correlations(signal):
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            if signal.signal_type == 'SHORT':
                position_size = -position_size
            
            # Risk check
            if self.risk_manager:
                risk_check = self.risk_manager.check_risk_limits({
                    signal.instrument: abs(position_size) * 0.15  # Assume 15% allocation
                })
                
                if not risk_check['approved']:
                    logger.warning(f"⚠️ Signal rejected by risk manager: {risk_check['violations']}")
                    return False
            
            # Execute position
            position = Position(
                instrument=signal.instrument,
                quantity=position_size,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                unrealized_pnl=0.0,
                entry_time=signal.timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            self.positions[signal.instrument] = position
            
            logger.info(f"✅ Executed {signal.signal_type} signal for {signal.instrument}: "
                       f"{position_size} contracts @ ${signal.entry_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing signal for {signal.instrument}: {e}")
            return False
    
    def update_positions(self):
        """Update positions with current market prices and check exit conditions"""
        for instrument, position in list(self.positions.items()):
            if instrument in self.market_data:
                current_price = self.market_data[instrument].get('close', position.current_price)
                position.current_price = current_price
                
                # Calculate unrealized P&L
                price_diff = current_price - position.entry_price
                if position.quantity < 0:  # Short position
                    price_diff = -price_diff
                
                position.unrealized_pnl = price_diff * abs(position.quantity) * \
                                        self.instruments[instrument].tick_value / \
                                        self.instruments[instrument].tick_size
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if position.quantity > 0:  # Long position
                    if current_price <= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif current_price >= position.take_profit:
                        should_exit = True
                        exit_reason = "Take profit"
                else:  # Short position
                    if current_price >= position.stop_loss:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif current_price <= position.take_profit:
                        should_exit = True
                        exit_reason = "Take profit"
                
                if should_exit:
                    self.close_position(instrument, exit_reason)
    
    def close_position(self, instrument: str, reason: str = "Manual"):
        """
        Close a position and record the trade
        
        Args:
            instrument: Instrument to close
            reason: Reason for closing
        """
        if instrument not in self.positions:
            return
        
        position = self.positions[instrument]
        
        # Record completed trade
        trade_record = {
            'symbol': instrument,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'quantity': abs(position.quantity),
            'side': 'LONG' if position.quantity > 0 else 'SHORT',
            'pnl': position.unrealized_pnl,
            'exit_reason': reason
        }
        
        self.trade_history.append(trade_record)
        self.performance_analytics.add_trade(trade_record)
        
        # Update capital
        self.current_capital += position.unrealized_pnl
        
        # Remove position
        del self.positions[instrument]
        
        logger.info(f"🔄 Closed {instrument} position: P&L ${position.unrealized_pnl:.2f} ({reason})")
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_equity = self.current_capital + total_unrealized
        
        # Calculate correlations
        correlations = self.correlation_analyzer.calculate_correlations()
        
        # Get performance metrics
        performance = self.performance_analytics.calculate_performance_metrics()
        
        summary = {
            'current_capital': self.current_capital,
            'unrealized_pnl': total_unrealized,
            'total_equity': total_equity,
            'total_return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'active_positions': len(self.positions),
            'instruments_traded': list(self.positions.keys()),
            'correlation_matrix': correlations.to_dict() if not correlations.empty else {},
            'performance_metrics': performance,
            'position_details': {inst: asdict(pos) for inst, pos in self.positions.items()}
        }
        
        return summary
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Update positions
            self.update_positions()
            
            # Generate new signals
            signals = await self.generate_signals()
            
            # Execute signals
            for signal in signals:
                await self.execute_signal(signal)
            
            # Log portfolio status
            summary = self.get_portfolio_summary()
            logger.info(f"💼 Portfolio: ${summary['total_equity']:,.2f} "
                       f"({summary['total_return']:+.2f}%) - "
                       f"{summary['active_positions']} positions")
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")

async def main():
    """Main function to test the multi-instrument trading system"""
    logger.info("🧪 Testing Multi-Instrument Trading System...")
    
    # Initialize system
    system = MultiInstrumentTradingSystem(initial_capital=250000)
    system.initialize_trading_systems()
    
    # Simulate market data updates
    instruments = ['ES', 'NQ', 'YM', 'RTY']
    base_prices = {'ES': 4400, 'NQ': 15000, 'YM': 34000, 'RTY': 2000}
    
    logger.info("📊 Simulating trading cycles...")
    
    for cycle in range(10):  # 10 trading cycles
        logger.info(f"🔄 Trading cycle {cycle + 1}/10")
        
        # Update market data for all instruments
        for instrument in instruments:
            # Simulate price movement
            base_price = base_prices[instrument]
            price_change = np.random.normal(0, base_price * 0.005)  # 0.5% volatility
            current_price = base_price + price_change
            base_prices[instrument] = current_price
            
            market_data = {
                'open': current_price * 0.999,
                'high': current_price * 1.001,
                'low': current_price * 0.999,
                'close': current_price,
                'volume': np.random.randint(1000, 5000)
            }
            
            system.update_market_data(instrument, market_data)
        
        # Run trading cycle
        await system.run_trading_cycle()
        
        # Brief pause
        await asyncio.sleep(1)
    
    # Final portfolio summary
    summary = system.get_portfolio_summary()
    
    logger.info("🎯 FINAL PORTFOLIO SUMMARY")
    logger.info(f"💰 Initial Capital: ${system.initial_capital:,.2f}")
    logger.info(f"💰 Current Capital: ${summary['current_capital']:,.2f}")
    logger.info(f"📈 Total Equity: ${summary['total_equity']:,.2f}")
    logger.info(f"📊 Total Return: {summary['total_return']:+.2f}%")
    logger.info(f"🎯 Active Positions: {summary['active_positions']}")
    logger.info(f"📋 Instruments: {summary['instruments_traded']}")
    
    # Performance metrics
    perf = summary['performance_metrics']
    logger.info(f"📊 Total Trades: {perf['total_trades']}")
    logger.info(f"🎯 Win Rate: {perf['win_rate']:.1f}%")
    logger.info(f"📈 Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    
    logger.info("🎉 Multi-Instrument Trading System testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
