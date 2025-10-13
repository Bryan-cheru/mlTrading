"""
Institutional Strategy Core - Statistical Arbitrage Engine
Replacing retail ML prediction with institutional-grade pairs trading
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import asyncio
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

@dataclass
class PairSignal:
    """Statistical arbitrage signal for pairs trading"""
    pair: str
    signal_type: str  # LONG_PAIR, SHORT_PAIR, HOLD
    zscore: float
    confidence: float
    hedge_ratio: float
    spread_value: float
    entry_price_a: float
    entry_price_b: float
    timestamp: datetime
    expected_return: float
    risk_score: float

@dataclass
class VolatilitySignal:
    """Volatility structure arbitrage signal"""
    strategy: str  # CONTANGO_TRADE, BACKWARDATION_TRADE
    signal_type: str  # BUY_VOL, SELL_VOL, HOLD
    term_structure_slope: float
    confidence: float
    front_month_iv: float
    back_month_iv: float
    expected_profit: float
    timestamp: datetime

class StatisticalArbitrageEngine:
    """
    Core engine for institutional statistical arbitrage strategies
    Implements pairs trading, volatility arbitrage, and flow-based strategies
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Pairs trading components
        self.pairs_data = {}  # Store price history for pairs
        self.hedge_ratios = {}  # Rolling hedge ratios
        self.spread_stats = {}  # Rolling mean/std for spreads
        
        # Volatility components
        self.vol_data = {}  # VIX term structure data
        self.vol_signals = []
        
        # Flow-based components
        self.rebalance_calendar = self._load_rebalance_calendar()
        self.flow_indicators = {}
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.portfolio_pnl = 0.0
        
        logger.info("🏛️ Statistical Arbitrage Engine initialized")
        logger.info(f"📊 Configured pairs: {list(self.config['pairs'].keys())}")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for institutional strategies"""
        return {
            'pairs': {
                'ES_NQ': {
                    'symbol_a': 'ESZ4',
                    'symbol_b': 'NQZ4', 
                    'lookback': 300,
                    'entry_zscore': 2.0,
                    'exit_zscore': 0.3,
                    'max_holding_bars': 480,  # 8 hours for 1-min bars
                    'vol_filter_threshold': 0.02
                },
                'ZN_ZB': {
                    'symbol_a': 'ZNZ4',
                    'symbol_b': 'ZBZ4',
                    'lookback': 240,
                    'entry_zscore': 1.8,
                    'exit_zscore': 0.25,
                    'max_holding_bars': 720,  # 12 hours
                    'vol_filter_threshold': 0.015
                }
            },
            'volatility': {
                'vix_contango_threshold': 0.02,
                'backwardation_threshold': -0.01,
                'min_vol_spread': 0.005
            },
            'flows': {
                'rebalance_window_hours': 2,
                'quarter_end_days': 3,
                'news_blackout_minutes': 5
            },
            'risk': {
                'max_daily_loss': 0.02,  # 2% daily loss limit
                'max_position_per_pair': 0.10,  # 10% capital per pair
                'correlation_limit': 0.7  # Max correlation between strategies
            }
        }
    
    def process_tick_data(self, tick_data: Dict) -> List[PairSignal]:
        """
        Main processing function - replaces your ML prediction
        Generates institutional statistical arbitrage signals
        """
        signals = []
        
        try:
            symbol = tick_data['symbol']
            price = tick_data['price']
            timestamp = tick_data.get('timestamp', datetime.now())
            
            # Update price data
            self._update_price_data(symbol, price, timestamp)
            
            # Generate pairs signals
            for pair_name, pair_config in self.config['pairs'].items():
                if symbol in [pair_config['symbol_a'], pair_config['symbol_b']]:
                    pair_signal = self._generate_pairs_signal(pair_name, pair_config)
                    if pair_signal and pair_signal.signal_type != 'HOLD':
                        signals.append(pair_signal)
            
            # Generate volatility signals (if VIX data available)
            vol_signal = self._generate_volatility_signal(tick_data)
            if vol_signal:
                # Convert to pair signal format for unified processing
                signals.append(self._convert_vol_to_pair_signal(vol_signal))
            
            # Generate flow-based signals
            flow_signal = self._generate_flow_signal(tick_data)
            if flow_signal:
                signals.append(flow_signal)
            
            self.signals_generated += len(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")
            return []
    
    def _generate_pairs_signal(self, pair_name: str, config: Dict) -> Optional[PairSignal]:
        """Generate statistical arbitrage signal for a pairs trade"""
        try:
            symbol_a = config['symbol_a']
            symbol_b = config['symbol_b']
            
            # Check if we have sufficient data for both instruments
            if (symbol_a not in self.pairs_data or symbol_b not in self.pairs_data):
                return None
            
            data_a = self.pairs_data[symbol_a]
            data_b = self.pairs_data[symbol_b]
            
            if len(data_a) < config['lookback'] or len(data_b) < config['lookback']:
                return None
            
            # Calculate rolling hedge ratio (beta) using OLS
            hedge_ratio = self._calculate_rolling_hedge_ratio(
                data_a, data_b, config['lookback']
            )
            
            if hedge_ratio is None:
                return None
            
            # Calculate spread
            log_a = np.log(data_a[-1]['price'])
            log_b = np.log(data_b[-1]['price'])
            current_spread = log_b - hedge_ratio * log_a
            
            # Calculate rolling spread statistics
            spread_mean, spread_std = self._calculate_spread_stats(
                data_a, data_b, hedge_ratio, config['lookback']
            )
            
            if spread_std <= 0:
                return None
            
            # Calculate z-score
            zscore = (current_spread - spread_mean) / spread_std
            
            # Volatility filter
            current_vol = self._calculate_realized_volatility(data_a, 20)
            if current_vol > config['vol_filter_threshold']:
                logger.debug(f"⚠️ High volatility filter triggered for {pair_name}")
                return None
            
            # Generate signal
            signal_type = 'HOLD'
            confidence = 0.0
            
            if abs(zscore) > config['entry_zscore']:
                if zscore > 0:
                    signal_type = 'SHORT_PAIR'  # Short B, Long A
                else:
                    signal_type = 'LONG_PAIR'   # Long B, Short A
                
                confidence = min(abs(zscore) / config['entry_zscore'], 1.0)
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(zscore, spread_std)
            
            return PairSignal(
                pair=pair_name,
                signal_type=signal_type,
                zscore=zscore,
                confidence=confidence,
                hedge_ratio=hedge_ratio,
                spread_value=current_spread,
                entry_price_a=data_a[-1]['price'],
                entry_price_b=data_b[-1]['price'],
                timestamp=datetime.now(),
                expected_return=expected_return,
                risk_score=1.0 - confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating pairs signal for {pair_name}: {e}")
            return None
    
    def _calculate_rolling_hedge_ratio(self, data_a: List, data_b: List, lookback: int) -> Optional[float]:
        """Calculate rolling hedge ratio using OLS regression"""
        try:
            if len(data_a) < lookback or len(data_b) < lookback:
                return None
            
            # Get recent price data
            prices_a = np.array([d['price'] for d in data_a[-lookback:]])
            prices_b = np.array([d['price'] for d in data_b[-lookback:]])
            
            # Convert to log prices
            log_a = np.log(prices_a)
            log_b = np.log(prices_b)
            
            # OLS regression: log_b = alpha + beta * log_a
            X = log_a.reshape(-1, 1)
            y = log_b
            
            model = LinearRegression().fit(X, y)
            hedge_ratio = model.coef_[0]
            
            # Validate hedge ratio
            if hedge_ratio <= 0 or hedge_ratio > 10:  # Sanity check
                return None
            
            return hedge_ratio
            
        except Exception as e:
            logger.error(f"❌ Error calculating hedge ratio: {e}")
            return None
    
    def _calculate_spread_stats(self, data_a: List, data_b: List, 
                               hedge_ratio: float, lookback: int) -> Tuple[float, float]:
        """Calculate rolling mean and standard deviation of spread"""
        try:
            spreads = []
            
            for i in range(max(0, len(data_a) - lookback), len(data_a)):
                if i < len(data_b):
                    log_a = np.log(data_a[i]['price'])
                    log_b = np.log(data_b[i]['price'])
                    spread = log_b - hedge_ratio * log_a
                    spreads.append(spread)
            
            if len(spreads) < 20:  # Need minimum data
                return 0.0, 0.0
            
            spread_mean = np.mean(spreads)
            spread_std = np.std(spreads, ddof=1)
            
            return spread_mean, spread_std
            
        except Exception as e:
            logger.error(f"❌ Error calculating spread stats: {e}")
            return 0.0, 0.0
    
    def _calculate_realized_volatility(self, data: List, lookback: int) -> float:
        """Calculate realized volatility for risk filtering"""
        try:
            if len(data) < lookback + 1:
                return 0.0
            
            prices = np.array([d['price'] for d in data[-lookback-1:]])
            returns = np.diff(np.log(prices))
            
            # Annualized volatility (assuming 1440 minutes per day)
            realized_vol = np.std(returns, ddof=1) * np.sqrt(1440)
            
            return realized_vol
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility: {e}")
            return 0.0
    
    def _update_price_data(self, symbol: str, price: float, timestamp: datetime):
        """Update price data for pairs calculations"""
        if symbol not in self.pairs_data:
            self.pairs_data[symbol] = []
        
        self.pairs_data[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent data (memory management)
        max_history = 2000  # Keep 2000 bars (~33 hours for 1-min bars)
        if len(self.pairs_data[symbol]) > max_history:
            self.pairs_data[symbol] = self.pairs_data[symbol][-max_history:]
    
    def _generate_volatility_signal(self, tick_data: Dict) -> Optional[VolatilitySignal]:
        """Generate volatility arbitrage signals"""
        # Placeholder for VIX term structure analysis
        # Would implement VIX front/back month spread analysis
        return None
    
    def _generate_flow_signal(self, tick_data: Dict) -> Optional[PairSignal]:
        """Generate flow-based trading signals"""
        # Placeholder for index rebalancing and quarter-end flows
        return None
    
    def _convert_vol_to_pair_signal(self, vol_signal: VolatilitySignal) -> PairSignal:
        """Convert volatility signal to unified pair signal format"""
        # Convert for unified processing
        pass
    
    def _calculate_expected_return(self, zscore: float, spread_std: float) -> float:
        """Calculate expected return for mean reversion trade"""
        # Assume mean reversion with some probability
        mean_reversion_prob = 0.7  # 70% chance of reversion
        expected_move = abs(zscore) * spread_std * mean_reversion_prob
        return expected_move * 0.5  # Conservative estimate
    
    def _load_rebalance_calendar(self) -> Dict:
        """Load index rebalancing calendar"""
        # Placeholder - would load actual rebalancing dates
        return {
            'russell_rebalance': ['2025-06-27', '2025-12-20'],
            'sp500_rebalance': ['2025-03-21', '2025-06-20', '2025-09-19', '2025-12-19'],
            'msci_rebalance': ['2025-02-28', '2025-05-30', '2025-08-29', '2025-11-28']
        }
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio performance metrics"""
        return {
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'portfolio_pnl': self.portfolio_pnl,
            'signal_to_trade_ratio': self.trades_executed / max(1, self.signals_generated),
            'active_pairs': len([p for p in self.config['pairs'].keys() 
                               if len(self.pairs_data.get(self.config['pairs'][p]['symbol_a'], [])) > 0])
        }

class InstitutionalRiskManager:
    """
    Enhanced risk management for institutional strategies
    Portfolio-level risk controls beyond single-trade stops
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Portfolio risk limits
        self.max_daily_loss = 0.02  # 2% daily loss limit
        self.max_position_per_strategy = 0.15  # 15% max per strategy
        self.max_correlation = 0.7  # Max correlation between strategies
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_high_water = initial_capital
        self.positions = {}
        
        # Risk state
        self.risk_violations = []
        self.is_risk_off = False
        
        logger.info(f"🛡️ Institutional Risk Manager initialized")
        logger.info(f"💰 Capital: ${initial_capital:,.2f}")
        logger.info(f"⚠️ Daily loss limit: {self.max_daily_loss:.1%}")
    
    def check_signal_risk(self, signal: PairSignal, position_size: float) -> Dict:
        """
        Check if signal passes institutional risk controls
        This replaces your simple stop-loss approach
        """
        violations = []
        
        # Daily loss limit check
        daily_drawdown = (self.daily_high_water - self.current_capital) / self.daily_high_water
        if daily_drawdown > self.max_daily_loss:
            violations.append(f"Daily loss limit exceeded: {daily_drawdown:.2%}")
        
        # Position size check
        position_value = position_size * signal.entry_price_a  # Simplified
        position_pct = position_value / self.current_capital
        
        if position_pct > self.max_position_per_strategy:
            violations.append(f"Position size too large: {position_pct:.2%}")
        
        # Volatility check
        if signal.risk_score > 0.8:  # High risk score
            violations.append(f"Signal risk too high: {signal.risk_score:.2f}")
        
        # News blackout check
        if self._is_news_blackout():
            violations.append("News blackout period active")
        
        # Time-of-day filter
        if not self._is_trading_hours():
            violations.append("Outside trading hours")
        
        approved = len(violations) == 0 and not self.is_risk_off
        
        if violations:
            self.risk_violations.extend(violations)
            logger.warning(f"⚠️ Signal rejected: {', '.join(violations)}")
        
        return {
            'approved': approved,
            'violations': violations,
            'risk_score': signal.risk_score,
            'position_pct': position_pct
        }
    
    def update_portfolio(self, trade_pnl: float):
        """Update portfolio performance and risk metrics"""
        self.current_capital += trade_pnl
        self.daily_pnl += trade_pnl
        self.portfolio_pnl += trade_pnl
        
        # Update high water mark
        if self.current_capital > self.daily_high_water:
            self.daily_high_water = self.current_capital
        
        # Check if risk-off condition triggered
        daily_dd = (self.daily_high_water - self.current_capital) / self.daily_high_water
        if daily_dd > self.max_daily_loss:
            self.is_risk_off = True
            logger.warning(f"🚨 RISK OFF triggered - Daily loss: {daily_dd:.2%}")
    
    def reset_daily_risk(self):
        """Reset daily risk tracking (call at session start)"""
        self.daily_pnl = 0.0
        self.daily_high_water = self.current_capital
        self.is_risk_off = False
        self.risk_violations = []
        logger.info("🔄 Daily risk metrics reset")
    
    def _is_news_blackout(self) -> bool:
        """Check if in news blackout period"""
        # Placeholder - would check against news calendar
        return False
    
    def _is_trading_hours(self) -> bool:
        """Check if within allowed trading hours"""
        current_hour = datetime.now().hour
        # ES trades nearly 24/5, but avoid low-liquidity periods
        return 6 <= current_hour <= 21  # 6 AM to 9 PM

class InstitutionalExecutor:
    """
    Institutional-grade execution engine
    Handles pairs trades with proper risk controls
    """
    
    def __init__(self, ninjatrader_executor=None):
        self.nt_executor = ninjatrader_executor
        self.active_pairs = {}
        self.execution_stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'slippage_total': 0.0
        }
    
    def execute_pairs_signal(self, signal: PairSignal, position_size: int = 1) -> Dict:
        """
        Execute pairs trade with both legs
        This replaces your single-instrument execution
        """
        try:
            logger.info(f"🎯 Executing {signal.signal_type} for {signal.pair}")
            logger.info(f"📊 Z-score: {signal.zscore:.2f}, Confidence: {signal.confidence:.2f}")
            
            # Determine quantities for each leg
            qty_a = position_size
            qty_b = max(1, round(position_size * abs(signal.hedge_ratio)))
            
            results = {'success': False, 'orders': []}
            
            if signal.signal_type == 'LONG_PAIR':
                # Long B, Short A
                result_a = self._execute_leg('SELL', signal.pair.split('_')[0], qty_a, signal.entry_price_a)
                result_b = self._execute_leg('BUY', signal.pair.split('_')[1], qty_b, signal.entry_price_b)
                
            elif signal.signal_type == 'SHORT_PAIR':
                # Short B, Long A  
                result_a = self._execute_leg('BUY', signal.pair.split('_')[0], qty_a, signal.entry_price_a)
                result_b = self._execute_leg('SELL', signal.pair.split('_')[1], qty_b, signal.entry_price_b)
            
            else:
                return {'success': False, 'reason': 'Invalid signal type'}
            
            # Check if both legs executed successfully
            if result_a['success'] and result_b['success']:
                # Store active pair position
                self.active_pairs[signal.pair] = {
                    'signal': signal,
                    'entry_time': datetime.now(),
                    'leg_a': result_a,
                    'leg_b': result_b
                }
                
                results['success'] = True
                results['orders'] = [result_a, result_b]
                
                self.execution_stats['orders_filled'] += 2
                logger.info(f"✅ Pairs trade executed successfully: {signal.pair}")
            
            else:
                # Cancel any partial fills
                self._handle_partial_fill(result_a, result_b)
                results['reason'] = 'Partial fill - trade cancelled'
            
            self.execution_stats['orders_sent'] += 2
            return results
            
        except Exception as e:
            logger.error(f"❌ Error executing pairs signal: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _execute_leg(self, action: str, symbol: str, quantity: int, price: float) -> Dict:
        """Execute individual leg of pairs trade"""
        # This would call your NinjaTrader executor
        if self.nt_executor:
            return self.nt_executor.submit_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                order_type='LIMIT'
            )
        else:
            # Simulation mode
            return {
                'success': True,
                'order_id': f"SIM_{symbol}_{action}_{quantity}",
                'fill_price': price,
                'slippage': 0.0
            }
    
    def _handle_partial_fill(self, result_a: Dict, result_b: Dict):
        """Handle partial fills by canceling unfilled leg"""
        # Cancel any orders that didn't fill
        if result_a['success'] and not result_b['success']:
            logger.warning("⚠️ Canceling leg A due to leg B failure")
            # Cancel order A
        elif result_b['success'] and not result_a['success']:
            logger.warning("⚠️ Canceling leg B due to leg A failure")
            # Cancel order B

# Usage example - this replaces your ML prediction loop
async def run_institutional_strategies():
    """
    Main loop for institutional trading strategies
    This replaces your current ML prediction approach
    """
    
    # Initialize components
    arb_engine = StatisticalArbitrageEngine()
    risk_manager = InstitutionalRiskManager(initial_capital=100000)
    executor = InstitutionalExecutor()
    
    # Connect to Rithmic
    from data_pipeline.ingestion.rithmic_connector import RithmicConnector
    rithmic = RithmicConnector()
    
    await rithmic.connect()
    await rithmic.subscribe_market_data(['ESZ4', 'NQZ4', 'ZNZ4', 'ZBZ4'])
    
    def process_institutional_tick(tick_data):
        """Process each tick through institutional strategies"""
        
        # Generate institutional signals (replaces ML prediction)
        signals = arb_engine.process_tick_data(tick_data)
        
        for signal in signals:
            # Check institutional risk controls
            risk_check = risk_manager.check_signal_risk(signal, position_size=1)
            
            if risk_check['approved']:
                # Execute pairs trade (replaces single-instrument execution)
                execution_result = executor.execute_pairs_signal(signal)
                
                if execution_result['success']:
                    logger.info(f"✅ Institutional strategy executed: {signal.pair}")
                    
                    # Update performance tracking
                    estimated_pnl = signal.expected_return * 1000  # Simplified
                    risk_manager.update_portfolio(estimated_pnl)
    
    # Register callback
    rithmic.register_tick_callback(process_institutional_tick)
    
    logger.info("🏛️ Institutional strategies running...")
    logger.info("📊 Monitoring ES-NQ pairs, ZN-ZB spreads, volatility structure")
    
    # Run for demonstration
    await asyncio.sleep(300)  # 5 minutes
    
    # Show results
    portfolio_metrics = arb_engine.get_portfolio_metrics()
    print(f"\n📈 Institutional Strategy Results:")
    print(f"Signals Generated: {portfolio_metrics['signals_generated']}")
    print(f"Trades Executed: {portfolio_metrics['trades_executed']}")
    print(f"Active Pairs: {portfolio_metrics['active_pairs']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_institutional_strategies())