"""
Institutional Statistical Models - Advanced Implementation
Based on Leonard Baum's HMM models with James Ax algebraic improvements
Renaissance Technologies & Two Sigma inspired approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class InstitutionalSignal:
    """Advanced institutional signal with multiple edge types"""
    pair_name: str
    signal_type: str  # "MEAN_REVERSION", "VOLATILITY_ARBITRAGE", "FLOW_BASED"
    strategy_type: str  # "PAIRS", "VIX_STRUCTURE", "INDEX_REBALANCE", "CALENDAR_SPREAD"
    entry_signal: str  # "LONG_PAIR", "SHORT_PAIR", "VOLATILITY_LONG", "VOLATILITY_SHORT"
    confidence: float
    expected_sharpe: float
    kelly_fraction: float
    z_score: float
    hedge_ratio: float
    volatility_regime: int  # HMM state: 0=Low, 1=Normal, 2=High, 3=Crisis
    flow_imbalance: float
    microstructure_edge: float
    timestamp: datetime

@dataclass
class MarketRegime:
    """Market regime identification using HMM"""
    regime_id: int
    regime_name: str
    volatility_level: float
    correlation_level: float
    mean_reversion_strength: float
    confidence: float

class AdvancedKalmanFilter:
    """
    Advanced Kalman Filter for dynamic hedge ratio estimation
    Based on state-space models used by Renaissance Technologies
    """
    
    def __init__(self):
        """Initialize Kalman filter"""
        # State: [hedge_ratio, hedge_ratio_velocity]
        self.state = np.array([1.0, 0.0])  # Initial hedge ratio = 1.0
        
        # State covariance matrix
        self.P = np.eye(2) * 0.1
        
        # Process noise covariance (how much hedge ratio can change)
        self.Q = np.array([[0.001, 0.0001], 
                          [0.0001, 0.0001]])
        
        # Measurement noise covariance
        self.R = 0.01
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])
        
        # Measurement matrix (we observe the hedge ratio)
        self.H = np.array([1.0, 0.0])
        
        self.is_initialized = False
        
        logger.debug("🔬 Advanced Kalman Filter initialized")
    
    def update(self, price_a: float, price_b: float) -> Tuple[float, float]:
        """Update filter with new price observations"""
        try:
            if not self.is_initialized:
                # Initialize with simple regression estimate
                self.state[0] = price_b / price_a if price_a > 0 else 1.0
                self.is_initialized = True
                return self.state[0], 1.0
            
            # Prediction step
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F.T + self.Q
            
            # Measurement (observed hedge ratio from current prices)
            if price_a > 0:
                observed_ratio = price_b / price_a
                
                # Innovation (residual)
                y = observed_ratio - (self.H @ self.state)
                
                # Innovation covariance
                S = self.H @ self.P @ self.H.T + self.R
                
                # Kalman gain
                K = self.P @ self.H.T / S
                
                # Update step
                self.state = self.state + K * y
                self.P = self.P - np.outer(K, self.H @ self.P)
                
                # Calculate confidence based on trace of covariance matrix
                confidence = 1.0 / (1.0 + np.trace(self.P))
                
                return self.state[0], confidence
            else:
                return self.state[0], 0.5
            
        except Exception as e:
            logger.error(f"❌ Kalman filter update error: {e}")
            return 1.0, 0.0

class HiddenMarkovRegime:
    """
    Hidden Markov Model for market regime detection
    Following Leonard Baum's original framework
    """
    
    def __init__(self, n_regimes: int = 3):
        """Initialize HMM with market regimes"""
        self.n_regimes = n_regimes
        self.regime_names = ["Low Vol", "Normal", "High Vol", "Crisis"][:n_regimes]
        
        # Initialize with simple model (would be trained on historical data)
        self.transition_matrix = self._initialize_transition_matrix()
        self.emission_params = self._initialize_emission_params()
        
        self.current_regime = 1  # Start in normal regime
        self.regime_probabilities = np.array([0.1, 0.8, 0.1])[:n_regimes]
        self.is_fitted = False
        
        logger.info(f"🎯 HMM initialized with {n_regimes} regimes")
    
    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize transition matrix with regime persistence"""
        if self.n_regimes == 3:
            # Low Vol -> Normal -> High Vol (with persistence)
            return np.array([
                [0.85, 0.14, 0.01],  # Low Vol stays low, occasionally -> Normal
                [0.10, 0.80, 0.10],  # Normal can go either way
                [0.05, 0.15, 0.80]   # High Vol stays high, occasionally -> Normal
            ])
        else:
            # Default: equal transitions with persistence
            matrix = np.eye(self.n_regimes) * 0.7
            off_diag = 0.3 / (self.n_regimes - 1)
            matrix[matrix == 0] = off_diag
            return matrix
    
    def _initialize_emission_params(self) -> Dict:
        """Initialize emission parameters for each regime"""
        if self.n_regimes == 3:
            return {
                'volatility_means': [0.01, 0.02, 0.04],     # Low, Normal, High volatility
                'volatility_stds': [0.002, 0.005, 0.01],
                'correlation_means': [0.8, 0.6, 0.3],       # High, Medium, Low correlation
                'correlation_stds': [0.1, 0.15, 0.2]
            }
        else:
            # Default parameters
            vol_means = np.linspace(0.01, 0.05, self.n_regimes)
            vol_stds = np.linspace(0.002, 0.015, self.n_regimes)
            return {
                'volatility_means': vol_means.tolist(),
                'volatility_stds': vol_stds.tolist(),
                'correlation_means': [0.7] * self.n_regimes,
                'correlation_stds': [0.15] * self.n_regimes
            }
    
    def predict_regime(self, market_features: np.ndarray) -> MarketRegime:
        """Predict current market regime"""
        try:
            # Simple regime detection based on volatility
            # In production, this would use full Baum-Welch algorithm
            
            volatility = market_features[0] if len(market_features) > 0 else 0.02
            
            # Classify regime based on volatility thresholds
            if volatility < 0.015:
                regime_id = 0  # Low volatility
            elif volatility < 0.035:
                regime_id = 1  # Normal volatility  
            else:
                regime_id = 2  # High volatility
            
            # Ensure regime_id is within bounds
            regime_id = min(regime_id, self.n_regimes - 1)
            
            regime = MarketRegime(
                regime_id=regime_id,
                regime_name=self.regime_names[regime_id],
                volatility_level=volatility,
                correlation_level=0.6,  # Default
                mean_reversion_strength=1.0 / (1.0 + volatility * 50),  # Inverse relationship
                confidence=0.75
            )
            
            self.current_regime = regime_id
            return regime
            
        except Exception as e:
            logger.error(f"❌ Regime prediction error: {e}")
            # Return default normal regime
            return MarketRegime(
                regime_id=1,
                regime_name="Normal",
                volatility_level=0.02,
                correlation_level=0.6,
                mean_reversion_strength=0.7,
                confidence=0.5
            )

class VolatilityStructureAnalyzer:
    """
    Advanced volatility surface analysis
    VIX term structure and volatility arbitrage opportunities
    """
    
    def __init__(self):
        """Initialize volatility analyzer"""
        self.vix_history = pd.DataFrame()
        self.term_structure_history = []
        
        logger.info("📊 Volatility Structure Analyzer initialized")
    
    def analyze_vix_structure(self, vx_front: float, vx_back: float, vix_spot: float) -> Dict:
        """Analyze VIX term structure for contango/backwardation"""
        try:
            # Calculate term structure slope
            term_slope = (vx_back - vx_front) / vx_front
            
            # Calculate VIX basis (futures vs spot)
            front_basis = (vx_front - vix_spot) / vix_spot
            back_basis = (vx_back - vix_spot) / vix_spot
            
            # Classify market structure
            if term_slope > 0.05:
                structure = "STRONG_CONTANGO"
                signal_strength = min(term_slope * 10, 1.0)
            elif term_slope > 0.01:
                structure = "CONTANGO"
                signal_strength = term_slope * 20
            elif term_slope < -0.05:
                structure = "STRONG_BACKWARDATION"
                signal_strength = min(-term_slope * 10, 1.0)
            elif term_slope < -0.01:
                structure = "BACKWARDATION"
                signal_strength = -term_slope * 20
            else:
                structure = "FLAT"
                signal_strength = 0.0
            
            # Generate trading signal
            trade_signal = "HOLD"
            if structure in ["STRONG_CONTANGO", "CONTANGO"] and signal_strength > 0.3:
                trade_signal = "SHORT_VIX_FUTURES"  # Sell contango
            elif structure in ["STRONG_BACKWARDATION", "BACKWARDATION"] and signal_strength > 0.3:
                trade_signal = "LONG_VIX_FUTURES"   # Buy backwardation
            
            analysis = {
                'term_slope': term_slope,
                'front_basis': front_basis,
                'back_basis': back_basis,
                'structure': structure,
                'signal_strength': signal_strength,
                'trade_signal': trade_signal,
                'expected_return': signal_strength * 0.05,  # 5% max expected return
                'confidence': min(signal_strength * 1.5, 1.0)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ VIX structure analysis error: {e}")
            return {'trade_signal': 'HOLD', 'confidence': 0.0}

class FlowBasedAnalyzer:
    """
    Institutional flow analysis and prediction
    Index rebalancing, end-of-quarter flows, cross-market arbitrage
    """
    
    def __init__(self):
        """Initialize flow analyzer"""
        self.rebalance_calendar = self._build_rebalance_calendar()
        self.flow_history = pd.DataFrame()
        
        logger.info("🌊 Flow-Based Analyzer initialized")
    
    def _build_rebalance_calendar(self) -> pd.DataFrame:
        """Build calendar of known institutional rebalancing dates"""
        # Simplified calendar - in production would be comprehensive
        dates = pd.date_range('2024-01-01', '2025-12-31', freq='ME')  # Month end
        
        calendar = pd.DataFrame({
            'date': dates,
            'event_type': 'MONTH_END_REBALANCE',
            'expected_flow_intensity': 0.6
        })
        
        # Add quarterly events (higher intensity)
        quarterly_dates = dates[dates.month.isin([3, 6, 9, 12])]
        calendar.loc[calendar['date'].isin(quarterly_dates), 'expected_flow_intensity'] = 0.9
        
        return calendar
    
    def analyze_flow_imbalance(self, volume_profile: np.ndarray, 
                             price_impact: np.ndarray) -> Dict:
        """Analyze institutional flow imbalance"""
        try:
            # Volume-weighted price impact
            vwap_impact = np.average(price_impact, weights=volume_profile)
            
            # Flow direction analysis
            cumulative_impact = np.cumsum(price_impact * volume_profile)
            flow_persistence = self._calculate_flow_persistence(cumulative_impact)
            
            # Institutional vs retail flow detection
            large_block_ratio = np.sum(volume_profile > np.percentile(volume_profile, 90)) / len(volume_profile)
            
            # Flow prediction
            expected_continuation = self._predict_flow_continuation(
                vwap_impact, flow_persistence, large_block_ratio
            )
            
            analysis = {
                'vwap_impact': vwap_impact,
                'flow_persistence': flow_persistence,
                'large_block_ratio': large_block_ratio,
                'expected_continuation': expected_continuation,
                'institutional_flow_detected': large_block_ratio > 0.3 and abs(vwap_impact) > 0.001,
                'flow_signal_strength': min(large_block_ratio * abs(vwap_impact) * 1000, 1.0)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Flow analysis error: {e}")
            return {'institutional_flow_detected': False, 'flow_signal_strength': 0.0}
    
    def _calculate_flow_persistence(self, cumulative_impact: np.ndarray) -> float:
        """Calculate how persistent institutional flows are"""
        if len(cumulative_impact) < 2:
            return 0.0
        
        # Measure trend strength using linear regression
        x = np.arange(len(cumulative_impact))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative_impact)
        
        return abs(r_value)  # R-squared measures persistence
    
    def _predict_flow_continuation(self, vwap_impact: float, 
                                 persistence: float, large_block_ratio: float) -> float:
        """Predict likelihood of flow continuation"""
        # Simple prediction model - in production would use ML
        continuation_score = (
            abs(vwap_impact) * 100 * 0.4 +    # Price impact weight
            persistence * 0.3 +                # Persistence weight
            large_block_ratio * 0.3            # Institution weight
        )
        
        # Apply sign based on impact direction
        return np.sign(vwap_impact) * min(continuation_score, 1.0)

class KellyOptimalSizing:
    """
    Kelly Criterion implementation for optimal position sizing
    Based on expected returns and win probabilities
    """
    
    def __init__(self):
        """Initialize Kelly sizer"""
        self.max_kelly_fraction = 0.25  # Cap Kelly at 25% (fractional Kelly)
        self.min_kelly_fraction = 0.01  # Minimum position size
        
        logger.info("💰 Kelly Optimal Sizing initialized")
    
    def calculate_kelly_fraction(self, expected_return: float, win_probability: float,
                               avg_win: float, avg_loss: float, 
                               strategy_volatility: float) -> float:
        """Calculate optimal Kelly fraction"""
        try:
            if win_probability <= 0 or win_probability >= 1:
                return self.min_kelly_fraction
            
            if avg_win <= 0 or avg_loss <= 0:
                return self.min_kelly_fraction
            
            # Standard Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_prob, q = 1-win_prob
            b = avg_win / avg_loss
            p = win_probability
            q = 1 - win_probability
            
            kelly_fraction = (b * p - q) / b
            
            # Adjust for volatility (reduce size in high vol regimes)
            vol_adjustment = 1.0 / (1.0 + strategy_volatility * 10)
            kelly_fraction *= vol_adjustment
            
            # Apply bounds
            kelly_fraction = max(kelly_fraction, self.min_kelly_fraction)
            kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"❌ Kelly calculation error: {e}")
            return self.min_kelly_fraction

class InstitutionalStatisticalEngine:
    """
    Master class combining all advanced institutional approaches
    Integrates HMM, Kalman filtering, volatility analysis, and flow detection
    """
    
    def __init__(self):
        """Initialize institutional engine"""
        self.kalman_filters = {}  # Per pair
        self.hmm_regime = HiddenMarkovRegime()
        self.volatility_analyzer = VolatilityStructureAnalyzer()
        self.flow_analyzer = FlowBasedAnalyzer()
        self.kelly_sizer = KellyOptimalSizing()
        
        self.market_features_history = []
        self.institutional_signals = {}
        
        logger.info("🏛️ Institutional Statistical Engine initialized")
    
    def process_market_update(self, market_data: Dict) -> List[InstitutionalSignal]:
        """Process market update and generate institutional signals"""
        try:
            signals = []
            
            # Extract market features for regime detection
            market_features = self._extract_market_features(market_data)
            regime = self.hmm_regime.predict_regime(market_features)
            
            # Update Kalman filters for all pairs
            for pair_name, pair_data in market_data.items():
                if isinstance(pair_data, dict) and 'price_a' in pair_data and 'price_b' in pair_data:
                    
                    # Initialize Kalman filter if needed
                    if pair_name not in self.kalman_filters:
                        self.kalman_filters[pair_name] = AdvancedKalmanFilter()
                    
                    # Update hedge ratio
                    hedge_ratio, confidence = self.kalman_filters[pair_name].update(
                        pair_data['price_a'], pair_data['price_b']
                    )
                    
                    # Generate mean reversion signal
                    signal = self._generate_mean_reversion_signal(
                        pair_name, pair_data, hedge_ratio, regime
                    )
                    
                    if signal:
                        signals.append(signal)
            
            # Generate volatility signals if VIX data available
            if 'vix_data' in market_data:
                vol_signals = self._generate_volatility_signals(market_data['vix_data'], regime)
                signals.extend(vol_signals)
            
            # Generate flow signals if flow data available
            if 'flow_data' in market_data:
                flow_signals = self._generate_flow_signals(market_data['flow_data'], regime)
                signals.extend(flow_signals)
            
            logger.debug(f"Generated {len(signals)} institutional signals")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Market update processing error: {e}")
            return []
    
    def _extract_market_features(self, market_data: Dict) -> np.ndarray:
        """Extract features for regime detection"""
        try:
            # Extract volatility, correlation, volume features
            volatility = 0.02  # Default
            correlation = 0.6  # Default
            volume_imbalance = 0.0  # Default
            
            # Try to extract real features if available
            if 'market_stats' in market_data:
                stats = market_data['market_stats']
                volatility = stats.get('volatility', 0.02)
                correlation = stats.get('correlation', 0.6)
                volume_imbalance = stats.get('volume_imbalance', 0.0)
            
            features = np.array([volatility, correlation, volume_imbalance, 0.0])  # 4 features
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {e}")
            return np.array([0.02, 0.6, 0.0, 0.0])
    
    def _generate_mean_reversion_signal(self, pair_name: str, pair_data: Dict,
                                      hedge_ratio: float, regime: MarketRegime) -> Optional[InstitutionalSignal]:
        """Generate sophisticated mean reversion signal"""
        try:
            # Calculate spread and z-score (simplified)
            price_a = pair_data['price_a']
            price_b = pair_data['price_b']
            
            # Simple spread calculation
            spread = np.log(price_b) - hedge_ratio * np.log(price_a)
            z_score = spread * 50  # Simplified z-score
            
            # Generate signal if z-score is significant
            if abs(z_score) > 2.0:
                kelly_fraction = self.kelly_sizer.calculate_kelly_fraction(
                    expected_return=0.02,
                    win_probability=0.6,
                    avg_win=0.015,
                    avg_loss=0.01,
                    strategy_volatility=regime.volatility_level
                )
                
                entry_signal = "LONG_PAIR" if z_score < 0 else "SHORT_PAIR"
                
                signal = InstitutionalSignal(
                    pair_name=pair_name,
                    signal_type="MEAN_REVERSION",
                    strategy_type="PAIRS",
                    entry_signal=entry_signal,
                    confidence=min(abs(z_score) / 4.0, 1.0),
                    expected_sharpe=1.2,
                    kelly_fraction=kelly_fraction,
                    z_score=z_score,
                    hedge_ratio=hedge_ratio,
                    volatility_regime=regime.regime_id,
                    flow_imbalance=0.0,
                    microstructure_edge=0.1,
                    timestamp=datetime.now()
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Mean reversion signal error: {e}")
            return None
    
    def _generate_volatility_signals(self, vix_data: Dict, 
                                   regime: MarketRegime) -> List[InstitutionalSignal]:
        """Generate volatility structure signals"""
        signals = []
        
        try:
            if 'vx_front' in vix_data and 'vx_back' in vix_data:
                vol_analysis = self.volatility_analyzer.analyze_vix_structure(
                    vix_data['vx_front'],
                    vix_data['vx_back'],
                    vix_data.get('vix_spot', 20.0)
                )
                
                if vol_analysis.get('trade_signal') != 'HOLD':
                    kelly_fraction = self.kelly_sizer.calculate_kelly_fraction(
                        expected_return=0.05,
                        win_probability=0.55,
                        avg_win=0.03,
                        avg_loss=0.02,
                        strategy_volatility=regime.volatility_level * 2
                    )
                    
                    signal = InstitutionalSignal(
                        pair_name="VIX_STRUCTURE",
                        signal_type="VOLATILITY_ARBITRAGE",
                        strategy_type="VIX_STRUCTURE",
                        entry_signal=vol_analysis['trade_signal'],
                        confidence=vol_analysis['confidence'],
                        expected_sharpe=1.0,
                        kelly_fraction=kelly_fraction,
                        z_score=vol_analysis['term_slope'] * 100,
                        hedge_ratio=1.0,
                        volatility_regime=regime.regime_id,
                        flow_imbalance=0.0,
                        microstructure_edge=vol_analysis['signal_strength'],
                        timestamp=datetime.now()
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Volatility signal error: {e}")
            return []
    
    def _generate_flow_signals(self, flow_data: Dict, 
                             regime: MarketRegime) -> List[InstitutionalSignal]:
        """Generate flow-based signals"""
        signals = []
        
        try:
            if 'volume_profile' in flow_data and 'price_impact' in flow_data:
                flow_analysis = self.flow_analyzer.analyze_flow_imbalance(
                    flow_data['volume_profile'],
                    flow_data['price_impact']
                )
                
                if flow_analysis.get('institutional_flow_detected'):
                    kelly_fraction = self.kelly_sizer.calculate_kelly_fraction(
                        expected_return=0.01,
                        win_probability=0.65,
                        avg_win=0.008,
                        avg_loss=0.005,
                        strategy_volatility=regime.volatility_level
                    )
                    
                    entry_signal = "LONG_FLOW" if flow_analysis['expected_continuation'] > 0 else "SHORT_FLOW"
                    
                    signal = InstitutionalSignal(
                        pair_name="INSTITUTIONAL_FLOW",
                        signal_type="FLOW_BASED",
                        strategy_type="INDEX_REBALANCE",
                        entry_signal=entry_signal,
                        confidence=flow_analysis['flow_signal_strength'],
                        expected_sharpe=1.5,
                        kelly_fraction=kelly_fraction,
                        z_score=0.0,
                        hedge_ratio=1.0,
                        volatility_regime=regime.regime_id,
                        flow_imbalance=flow_analysis['expected_continuation'],
                        microstructure_edge=flow_analysis['large_block_ratio'],
                        timestamp=datetime.now()
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Flow signal error: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    print("🏛️ Institutional Statistical Models Test")
    print("=" * 60)
    
    # Test individual components
    print("🔬 Testing Kalman Filter...")
    kalman = AdvancedKalmanFilter()
    for i in range(10):
        price_a = 4500 + np.random.normal(0, 5)
        price_b = 15000 + 3.3 * (price_a - 4500) + np.random.normal(0, 10)
        hedge_ratio, confidence = kalman.update(price_a, price_b)
        print(f"   Step {i+1}: Hedge Ratio = {hedge_ratio:.3f}, Confidence = {confidence:.3f}")
    
    print("\n🎯 Testing HMM Regime Detection...")
    hmm = HiddenMarkovRegime()
    for vol in [0.01, 0.025, 0.05]:
        features = np.array([vol, 0.6, 0.0, 0.0])
        regime = hmm.predict_regime(features)
        print(f"   Volatility {vol:.3f} -> Regime: {regime.regime_name} (ID: {regime.regime_id})")
    
    print("\n📊 Testing Volatility Analysis...")
    vol_analyzer = VolatilityStructureAnalyzer()
    vix_analysis = vol_analyzer.analyze_vix_structure(18.5, 20.2, 17.8)
    print(f"   VIX Structure: {vix_analysis['structure']}")
    print(f"   Trade Signal: {vix_analysis['trade_signal']}")
    print(f"   Confidence: {vix_analysis['confidence']:.3f}")
    
    print("\n💰 Testing Kelly Sizing...")
    kelly = KellyOptimalSizing()
    kelly_fraction = kelly.calculate_kelly_fraction(0.02, 0.6, 0.015, 0.01, 0.02)
    print(f"   Kelly Fraction: {kelly_fraction:.3f}")
    
    print("\n🏛️ Testing Complete Institutional Engine...")
    engine = InstitutionalStatisticalEngine()
    
    # Simulate market data
    market_data = {
        'ES_NQ': {
            'price_a': 4520.50,
            'price_b': 15890.25
        },
        'market_stats': {
            'volatility': 0.025,
            'correlation': 0.75,
            'volume_imbalance': 0.1
        }
    }
    
    signals = engine.process_market_update(market_data)
    
    print(f"\n📈 Generated {len(signals)} institutional signals:")
    for signal in signals:
        print(f"   {signal.strategy_type}: {signal.entry_signal}")
        print(f"      Confidence: {signal.confidence:.3f}")
        print(f"      Kelly Fraction: {signal.kelly_fraction:.3f}")
        print(f"      Expected Sharpe: {signal.expected_sharpe:.2f}")
        print(f"      Volatility Regime: {signal.volatility_regime}")
    
    print("\n✅ Institutional models test complete!")