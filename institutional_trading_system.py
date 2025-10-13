"""
Enhanced ES Trading System - Institutional Strategy Integration
Combining your existing ML pipeline with institutional-grade strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.ingestion.rithmic_connector import RithmicConnector
from ml_models.training.trading_model import TradingMLModel
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InstitutionalTradingSystem:
    """
    Enhanced trading system implementing institutional strategies:
    1. Statistical Arbitrage (Pairs Trading)
    2. Volatility Structure Trading
    3. Flow-Based Strategies
    """
    
    def __init__(self):
        # Data sources
        self.rithmic = RithmicConnector()
        self.ml_model = TradingMLModel()
        
        # Strategy components
        self.pairs_engine = PairsArbitrageEngine()
        self.vol_engine = VolatilityArbitrageEngine()
        self.flow_engine = FlowBasedEngine()
        
        # Performance tracking
        self.portfolio_metrics = {
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'strategies_active': 0
        }
        
        logger.info("🏛️ Institutional Trading System initialized")
    
    async def start_institutional_trading(self):
        """Start all institutional strategies"""
        
        # Connect to Rithmic for professional data
        await self.rithmic.connect()
        await self.rithmic.subscribe_market_data(['ESZ4', 'NQZ4', 'ZNZ4', 'ZBZ4'])
        
        # Register callbacks for each strategy
        self.rithmic.register_tick_callback(self.process_institutional_signals)
        
        logger.info("🎯 Institutional strategies active:")
        logger.info("   • ES-NQ Pairs Arbitrage")
        logger.info("   • ZN-ZB Rates Arbitrage") 
        logger.info("   • VIX Term Structure")
        logger.info("   • Index Rebalancing Flows")
    
    def process_institutional_signals(self, tick_data):
        """Process tick data through institutional strategy filters"""
        
        # 1. Pairs Arbitrage Signals
        pairs_signal = self.pairs_engine.evaluate_spread(tick_data)
        
        # 2. Volatility Structure Signals  
        vol_signal = self.vol_engine.evaluate_term_structure(tick_data)
        
        # 3. Flow-Based Signals
        flow_signal = self.flow_engine.evaluate_institutional_flows(tick_data)
        
        # 4. Combined ML Enhancement
        ml_features = self.extract_institutional_features(tick_data)
        ml_signal = self.ml_model.predict(ml_features)
        
        # Portfolio-level signal combination
        combined_signal = self.combine_institutional_signals(
            pairs_signal, vol_signal, flow_signal, ml_signal
        )
        
        if combined_signal['strength'] > 0.75:  # High confidence only
            self.execute_institutional_strategy(combined_signal)
    
    def extract_institutional_features(self, tick_data):
        """Extract features for institutional strategies"""
        
        # Get Rithmic microstructure features
        rithmic_features = self.rithmic.get_market_features(
            tick_data['symbol'], lookback_ticks=100
        )
        
        # Add institutional-specific features
        institutional_features = {
            **rithmic_features,
            
            # Pairs trading features
            'es_nq_spread': self.pairs_engine.get_current_spread('ES', 'NQ'),
            'es_nq_zscore': self.pairs_engine.get_zscore('ES', 'NQ'),
            'zn_zb_spread': self.pairs_engine.get_current_spread('ZN', 'ZB'),
            
            # Volatility features
            'vix_contango': self.vol_engine.get_contango_level(),
            'vol_term_slope': self.vol_engine.get_term_structure_slope(),
            
            # Flow features
            'rebalance_window': self.flow_engine.is_rebalance_window(),
            'quarter_end_effect': self.flow_engine.get_quarter_end_strength(),
            'institutional_flow': self.flow_engine.get_flow_pressure(),
            
            # Market microstructure (from Rithmic)
            'order_flow_imbalance': rithmic_features.get('order_flow_imbalance', 0),
            'bid_ask_pressure': rithmic_features.get('bid_ask_pressure', 0),
            'market_maker_activity': rithmic_features.get('market_maker_activity', 0)
        }
        
        return institutional_features

class PairsArbitrageEngine:
    """Statistical arbitrage engine for futures pairs"""
    
    def __init__(self):
        self.lookback = 300  # Rolling window
        self.entry_zscore = 2.0
        self.exit_zscore = 0.3
        
        # Store spread data
        self.spreads = {}
        self.hedge_ratios = {}
        
    def evaluate_spread(self, tick_data):
        """Evaluate pairs trading opportunities"""
        symbol = tick_data['symbol']
        
        if symbol in ['ESZ4', 'NQZ4']:
            return self._evaluate_es_nq_spread(tick_data)
        elif symbol in ['ZNZ4', 'ZBZ4']:
            return self._evaluate_zn_zb_spread(tick_data)
        
        return {'signal': 'HOLD', 'strength': 0.0}
    
    def _evaluate_es_nq_spread(self, tick_data):
        """Evaluate ES-NQ pairs opportunity"""
        
        # Get current prices (would need both instruments)
        # This is simplified - in reality you'd have both prices
        
        # Calculate rolling hedge ratio (beta)
        beta = self._calculate_rolling_beta('ES', 'NQ')
        
        # Calculate spread and z-score
        spread = self._calculate_spread('ES', 'NQ', beta)
        zscore = self._calculate_zscore(spread)
        
        # Generate signal
        if abs(zscore) > self.entry_zscore:
            signal_type = 'SHORT_PAIR' if zscore > 0 else 'LONG_PAIR'
            return {
                'signal': signal_type,
                'strength': min(abs(zscore) / self.entry_zscore, 1.0),
                'pair': 'ES-NQ',
                'zscore': zscore,
                'beta': beta
            }
        
        return {'signal': 'HOLD', 'strength': 0.0}

class VolatilityArbitrageEngine:
    """Volatility structure arbitrage engine"""
    
    def __init__(self):
        self.vix_data = {}
        self.vol_thresholds = {
            'contango_entry': 0.02,  # 2% contango
            'backwardation_entry': -0.01  # 1% backwardation
        }
    
    def evaluate_term_structure(self, tick_data):
        """Evaluate volatility term structure opportunities"""
        
        # Calculate VIX term structure slope
        contango_level = self.get_contango_level()
        
        if contango_level > self.vol_thresholds['contango_entry']:
            return {
                'signal': 'SELL_VOL_FRONT',
                'strength': 0.8,
                'strategy': 'contango_trade',
                'expected_return': contango_level * 0.1
            }
        elif contango_level < self.vol_thresholds['backwardation_entry']:
            return {
                'signal': 'BUY_VOL_FRONT', 
                'strength': 0.7,
                'strategy': 'backwardation_trade',
                'expected_return': abs(contango_level) * 0.15
            }
        
        return {'signal': 'HOLD', 'strength': 0.0}

class FlowBasedEngine:
    """Flow-based strategy engine"""
    
    def __init__(self):
        self.rebalance_calendar = self._load_rebalance_calendar()
        self.quarter_end_windows = self._define_quarter_end_windows()
    
    def evaluate_institutional_flows(self, tick_data):
        """Evaluate flow-based opportunities"""
        
        current_time = datetime.now()
        
        # Check for index rebalancing
        if self.is_rebalance_window():
            flow_pressure = self.get_flow_pressure()
            return {
                'signal': 'FADE_FLOW' if flow_pressure > 0.5 else 'RIDE_FLOW',
                'strength': 0.6,
                'strategy': 'index_rebalance',
                'flow_direction': 'buy' if flow_pressure > 0 else 'sell'
            }
        
        # Check for quarter-end effects
        qe_strength = self.get_quarter_end_strength()
        if qe_strength > 0.3:
            return {
                'signal': 'QUARTER_END_TRADE',
                'strength': qe_strength,
                'strategy': 'quarter_end_effect'
            }
        
        return {'signal': 'HOLD', 'strength': 0.0}

# Integration example
async def main():
    """Example of running institutional strategies"""
    
    system = InstitutionalTradingSystem()
    await system.start_institutional_trading()
    
    # Run for demonstration
    import asyncio
    await asyncio.sleep(300)  # 5 minutes
    
    # Report results
    metrics = system.portfolio_metrics
    print(f"📊 Institutional Trading Results:")
    print(f"Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())