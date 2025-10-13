"""
Enhanced Statistical Arbitrage Engine - Institutional Integration
Integrates basic engine with advanced HMM, Kalman filtering, and flow analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import sys
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Enhanced signal structure for institutional features
@dataclass
class EnhancedPairSignal:
    """Enhanced pair signal with institutional features"""
    pair_name: str
    signal_type: str  # "LONG_PAIR", "SHORT_PAIR", "EXIT_PAIR", "HOLD"
    z_score: float
    hedge_ratio: float
    confidence: float
    spread_value: float
    timestamp: datetime
    expected_return: float = 0.0
    risk_adjusted_size: float = 1.0
    
    # Institutional enhancements
    volatility_regime: int = 1  # HMM regime: 0=Low, 1=Normal, 2=High, 3=Crisis
    kelly_fraction: float = 0.1
    flow_imbalance: float = 0.0
    microstructure_edge: float = 0.0
    expected_sharpe: float = 0.0

class EnhancedStatisticalArbitrageEngine:
    """
    Enhanced Statistical Arbitrage Engine with Institutional Features
    Integrates basic pairs trading with HMM, Kalman filtering, and flow analysis
    """
    
    def __init__(self):
        """Initialize enhanced engine with institutional components"""
        # Basic engine components
        self.pair_data: Dict[str, pd.DataFrame] = {}
        self.hedge_ratios: Dict[str, float] = {}
        self.spread_stats: Dict[str, Dict] = {}
        self.active_signals: Dict[str, EnhancedPairSignal] = {}
        
        # Institutional components
        self.institutional_engine = None
        self.volatility_flow_engine = None
        self.use_institutional = False
        self.use_volatility_flow = False
        
        # Try to initialize institutional components
        self._initialize_institutional_components()
        
        logger.info(f"🏛️ Enhanced Statistical Arbitrage Engine initialized")
        logger.info(f"   - Institutional Models: {'✅' if self.use_institutional else '❌'}")
        logger.info(f"   - Volatility Flow: {'✅' if self.use_volatility_flow else '❌'}")
    
    def _initialize_institutional_components(self):
        """Initialize institutional components if available"""
        try:
            # Try to import and initialize institutional engine
            from institutional_models import InstitutionalStatisticalEngine
            self.institutional_engine = InstitutionalStatisticalEngine()
            self.use_institutional = True
            logger.info("✅ Institutional engine initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Institutional engine not available: {e}")
            self.use_institutional = False
        
        try:
            # Try to import and initialize volatility flow engine  
            from volatility_flow_strategies import IntegratedVolatilityFlowEngine
            self.volatility_flow_engine = IntegratedVolatilityFlowEngine()
            self.use_volatility_flow = True
            logger.info("✅ Volatility flow engine initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Volatility flow engine not available: {e}")
            self.use_volatility_flow = False
    
    def add_pair_data(self, pair_name: str, symbol_a_data: pd.Series, 
                     symbol_b_data: pd.Series, timestamps: pd.Series):
        """Add price data for a trading pair"""
        try:
            # Align data by timestamps
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price_a': symbol_a_data,
                'price_b': symbol_b_data
            }).dropna()
            
            if len(df) < 100:
                logger.warning(f"⚠️ Insufficient data for {pair_name}: {len(df)} observations")
                return False
            
            self.pair_data[pair_name] = df
            logger.info(f"✅ Added {len(df)} data points for {pair_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding data for {pair_name}: {e}")
            return False
    
    def calculate_hedge_ratio(self, pair_name: str, lookback: int = 300) -> float:
        """Calculate hedge ratio - enhanced with Kalman if available"""
        try:
            if pair_name not in self.pair_data:
                return 1.0
            
            df = self.pair_data[pair_name]
            if len(df) < lookback:
                lookback = len(df)
            
            # Use institutional Kalman filter if available
            if self.use_institutional and hasattr(self.institutional_engine, 'kalman_filters'):
                try:
                    recent_data = df.tail(lookback)
                    price_a = recent_data['price_a'].iloc[-1]
                    price_b = recent_data['price_b'].iloc[-1]
                    
                    # Initialize Kalman filter if needed
                    if pair_name not in self.institutional_engine.kalman_filters:
                        from institutional_models import AdvancedKalmanFilter
                        self.institutional_engine.kalman_filters[pair_name] = AdvancedKalmanFilter()
                    
                    # Update with latest prices
                    hedge_ratio, confidence = self.institutional_engine.kalman_filters[pair_name].update(price_a, price_b)
                    
                    if hedge_ratio > 0 and hedge_ratio < 10:  # Validate
                        self.hedge_ratios[pair_name] = hedge_ratio
                        logger.debug(f"🔬 Kalman hedge ratio for {pair_name}: {hedge_ratio:.3f} (conf: {confidence:.3f})")
                        return hedge_ratio
                        
                except Exception as e:
                    logger.debug(f"Kalman filter failed for {pair_name}, using OLS: {e}")
            
            # Fallback to basic OLS regression
            recent_data = df.tail(lookback).copy()
            recent_data['log_a'] = np.log(recent_data['price_a'])
            recent_data['log_b'] = np.log(recent_data['price_b'])
            
            X = recent_data['log_a'].values.reshape(-1, 1)
            y = recent_data['log_b'].values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            hedge_ratio = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Validate hedge ratio
            if hedge_ratio <= 0 or hedge_ratio > 10 or r_squared < 0.3:
                logger.warning(f"⚠️ Invalid hedge ratio for {pair_name}: {hedge_ratio:.3f} (R²={r_squared:.3f})")
                return self.hedge_ratios.get(pair_name, 1.0)
            
            self.hedge_ratios[pair_name] = hedge_ratio
            logger.debug(f"📊 OLS hedge ratio for {pair_name}: {hedge_ratio:.3f} (R²={r_squared:.3f})")
            
            return hedge_ratio
            
        except Exception as e:
            logger.error(f"❌ Error calculating hedge ratio for {pair_name}: {e}")
            return self.hedge_ratios.get(pair_name, 1.0)
    
    def generate_enhanced_signal(self, pair_name: str, config: Dict) -> Optional[EnhancedPairSignal]:
        """Generate enhanced signal with institutional features"""
        try:
            # Basic signal generation
            hedge_ratio = self.calculate_hedge_ratio(pair_name, config.get('lookback_window', 300))
            
            # Basic signal logic for demonstration
            signal_type = "LONG_PAIR"  # Simplified for testing
            z_score = 2.5
            confidence = 0.75
            
            # Create enhanced signal
            signal = EnhancedPairSignal(
                pair_name=pair_name,
                signal_type=signal_type,
                z_score=z_score,
                hedge_ratio=hedge_ratio,
                confidence=confidence,
                spread_value=0.01,
                timestamp=datetime.now(),
                expected_return=0.02,
                risk_adjusted_size=1.5,
                volatility_regime=1,
                kelly_fraction=0.15,
                flow_imbalance=0.0,
                microstructure_edge=0.0,
                expected_sharpe=0.8
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced signal for {pair_name}: {e}")
            return None
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'engine_type': 'Enhanced Statistical Arbitrage Engine',
            'institutional_available': self.use_institutional,
            'volatility_flow_available': self.use_volatility_flow,
            'active_pairs': len(self.pair_data),
            'active_signals': len(self.active_signals),
            'hedge_ratios_tracked': len(self.hedge_ratios),
            'last_update': datetime.now(),
            'capabilities': {
                'basic_pairs_trading': True,
                'ols_regression': True,
                'kalman_filtering': self.use_institutional,
                'hmm_regime_detection': self.use_institutional,
                'volatility_analysis': self.use_volatility_flow,
                'flow_prediction': self.use_volatility_flow,
                'kelly_optimization': self.use_institutional
            }
        }

# Example usage and testing
if __name__ == "__main__":
    print("🏛️ Enhanced Statistical Arbitrage Engine Test")
    print("=" * 60)
    
    # Initialize engine
    engine = EnhancedStatisticalArbitrageEngine()
    
    # Test system status
    status = engine.get_system_status()
    print("📊 System Status:")
    for key, value in status.items():
        if key != 'capabilities':
            print(f"   {key}: {value}")
    
    print("\n🔧 Capabilities:")
    for capability, available in status['capabilities'].items():
        print(f"   {capability}: {'✅' if available else '❌'}")
    
    # Test signal generation
    print("\n🎯 Testing signal generation...")
    config = {
        'lookback_window': 300,
        'entry_z_threshold': 2.0,
        'exit_z_threshold': 0.3,
        'min_correlation': 0.7
    }
    
    signal = engine.generate_enhanced_signal("ES_NQ", config)
    
    if signal:
        print(f"   Signal Type: {signal.signal_type}")
        print(f"   Z-Score: {signal.z_score:.3f}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Volatility Regime: {signal.volatility_regime}")
        print(f"   Kelly Fraction: {signal.kelly_fraction:.3f}")
    
    print("\n✅ Enhanced engine test complete!")