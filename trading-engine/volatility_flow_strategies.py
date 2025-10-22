"""
Volatility Flow Strategies - Advanced Implementation
VIX term structure analysis and institutional flow prediction
Based on volatility surface arbitrage and flow-based trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class VolatilitySignal:
    """Volatility arbitrage signal"""
    signal_type: str  # "VIX_CONTANGO", "VIX_BACKWARDATION", "TERM_STRUCTURE"
    strategy_type: str  # "VOLATILITY_ARBITRAGE", "CALENDAR_SPREAD"
    entry_direction: str  # "LONG", "SHORT", "SPREAD"
    confidence: float
    expected_return: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    term_structure_slope: float
    volatility_regime: int
    timestamp: datetime

@dataclass  
class FlowSignal:
    """Institutional flow prediction signal"""
    signal_type: str  # "INDEX_REBALANCE", "QUARTER_END", "OPTIONS_EXPIRY", "EARNINGS"
    flow_direction: str  # "INSTITUTIONAL_BUYING", "INSTITUTIONAL_SELLING"
    flow_intensity: float  # 0.0 to 1.0
    expected_duration_hours: float
    price_impact_estimate: float
    volume_concentration: float  # How concentrated the flow will be
    dark_pool_activity: float  # Expected dark pool usage
    timestamp: datetime

class AdvancedVolatilityEngine:
    """
    Advanced volatility surface analysis and trading
    VIX term structure, volatility arbitrage, calendar spreads
    """
    
    def __init__(self):
        """Initialize volatility engine"""
        self.vix_history = pd.DataFrame()
        self.term_structure_models = {}
        self.volatility_surfaces = {}
        
        # Volatility thresholds
        self.contango_threshold = 0.02  # 2% monthly slope for strong contango
        self.backwardation_threshold = -0.02  # 2% monthly slope for strong backwardation
        
        logger.info("📈 Advanced Volatility Engine initialized")
    
    def analyze_vix_term_structure(self, futures_prices: Dict[str, float], 
                                 spot_vix: float) -> VolatilitySignal:
        """Analyze VIX futures term structure for arbitrage opportunities"""
        try:
            if len(futures_prices) < 2:
                return self._create_hold_signal()
            
            # Sort futures by expiration (assume VX1, VX2, VX3... format)
            sorted_futures = sorted(futures_prices.items())
            
            if len(sorted_futures) < 2:
                return self._create_hold_signal()
            
            front_month = sorted_futures[0][1]
            second_month = sorted_futures[1][1]
            
            # Calculate term structure slope (monthly)
            monthly_slope = (second_month - front_month) / front_month
            
            # Calculate VIX basis (futures premium/discount to spot)
            front_basis = (front_month - spot_vix) / spot_vix if spot_vix > 0 else 0
            
            # Analyze market structure
            structure_analysis = self._analyze_market_structure(
                monthly_slope, front_basis, spot_vix
            )
            
            # Generate volatility signal
            signal = VolatilitySignal(
                signal_type=structure_analysis['signal_type'],
                strategy_type="VOLATILITY_ARBITRAGE",
                entry_direction=structure_analysis['entry_direction'],
                confidence=structure_analysis['confidence'],
                expected_return=structure_analysis['expected_return'],
                risk_level=structure_analysis['risk_level'],
                term_structure_slope=monthly_slope,
                volatility_regime=self._classify_volatility_regime(spot_vix),
                timestamp=datetime.now()
            )
            
            logger.debug(f"📊 VIX analysis: {structure_analysis['signal_type']} "
                        f"(slope: {monthly_slope:.3f}, confidence: {structure_analysis['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ VIX term structure analysis error: {e}")
            return self._create_hold_signal()
    
    def _analyze_market_structure(self, monthly_slope: float, front_basis: float, 
                                spot_vix: float) -> Dict:
        """Analyze VIX market structure for trading opportunities"""
        
        # Strong contango (sell volatility)
        if monthly_slope > self.contango_threshold:
            return {
                'signal_type': 'VIX_CONTANGO',
                'entry_direction': 'SHORT',
                'confidence': min(monthly_slope / self.contango_threshold, 1.0),
                'expected_return': monthly_slope * 0.5,  # 50% capture of slope
                'risk_level': 'MEDIUM'
            }
        
        # Strong backwardation (buy volatility)
        elif monthly_slope < self.backwardation_threshold:
            return {
                'signal_type': 'VIX_BACKWARDATION', 
                'entry_direction': 'LONG',
                'confidence': min(abs(monthly_slope) / abs(self.backwardation_threshold), 1.0),
                'expected_return': abs(monthly_slope) * 0.4,  # 40% capture (more risk)
                'risk_level': 'HIGH'
            }
        
        # Mild contango (calendar spreads)
        elif 0.005 < monthly_slope <= self.contango_threshold:
            return {
                'signal_type': 'MILD_CONTANGO',
                'entry_direction': 'SPREAD',
                'confidence': (monthly_slope - 0.005) / (self.contango_threshold - 0.005),
                'expected_return': monthly_slope * 0.3,
                'risk_level': 'LOW'
            }
        
        # Flat or unclear structure
        else:
            return {
                'signal_type': 'HOLD',
                'entry_direction': 'HOLD',
                'confidence': 0.0,
                'expected_return': 0.0,
                'risk_level': 'LOW'
            }
    
    def _classify_volatility_regime(self, spot_vix: float) -> int:
        """Classify current volatility regime"""
        if spot_vix < 15:
            return 0  # Low volatility regime
        elif spot_vix < 25:
            return 1  # Normal volatility regime
        elif spot_vix < 35:
            return 2  # High volatility regime
        else:
            return 3  # Crisis/extreme volatility regime
    
    def _create_hold_signal(self) -> VolatilitySignal:
        """Create default HOLD signal"""
        return VolatilitySignal(
            signal_type="HOLD",
            strategy_type="VOLATILITY_ARBITRAGE",
            entry_direction="HOLD",
            confidence=0.0,
            expected_return=0.0,
            risk_level="LOW",
            term_structure_slope=0.0,
            volatility_regime=1,
            timestamp=datetime.now()
        )
    
    def analyze_options_volatility_surface(self, options_data: Dict) -> List[VolatilitySignal]:
        """Analyze options volatility surface for mispricings"""
        signals = []
        
        try:
            # This would analyze implied vs realized volatility
            # For now, return simplified analysis
            
            if 'implied_vol' in options_data and 'realized_vol' in options_data:
                iv = options_data['implied_vol']
                rv = options_data['realized_vol']
                
                vol_spread = iv - rv
                
                if vol_spread > 0.05:  # IV 5% above RV
                    signal = VolatilitySignal(
                        signal_type="IV_OVERPRICED",
                        strategy_type="VOL_SURFACE_ARB",
                        entry_direction="SHORT",
                        confidence=min(vol_spread * 10, 1.0),
                        expected_return=vol_spread * 0.3,
                        risk_level="MEDIUM",
                        term_structure_slope=0.0,
                        volatility_regime=1,
                        timestamp=datetime.now()
                    )
                    signals.append(signal)
                
                elif vol_spread < -0.03:  # IV 3% below RV
                    signal = VolatilitySignal(
                        signal_type="IV_UNDERPRICED",
                        strategy_type="VOL_SURFACE_ARB", 
                        entry_direction="LONG",
                        confidence=min(abs(vol_spread) * 15, 1.0),
                        expected_return=abs(vol_spread) * 0.4,
                        risk_level="HIGH",
                        term_structure_slope=0.0,
                        volatility_regime=1,
                        timestamp=datetime.now()
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Options volatility surface analysis error: {e}")
            return []

class InstitutionalFlowPredictor:
    """
    Advanced institutional flow prediction engine
    Detects and predicts large institutional trading flows
    """
    
    def __init__(self):
        """Initialize flow predictor"""
        self.flow_history = pd.DataFrame()
        self.rebalance_calendar = self._build_rebalance_calendar()
        self.flow_models = {}
        
        # Flow detection thresholds
        self.large_block_threshold = 10000  # Shares for large blocks
        self.institutional_volume_ratio = 0.15  # 15% of volume
        
        logger.info("🌊 Institutional Flow Predictor initialized")
    
    def predict_institutional_flows(self, market_data: Dict) -> List[FlowSignal]:
        """
        Predict institutional flows based on multiple signals
        """
        try:
            signals = []
            
            # Index rebalancing flows
            rebalance_signal = self._detect_rebalancing_flows(market_data)
            if rebalance_signal:
                signals.append(rebalance_signal)
            
            # Quarter-end flows  
            quarter_end_signal = self._detect_quarter_end_flows(market_data)
            if quarter_end_signal:
                signals.append(quarter_end_signal)
                
            # Options expiration flows
            options_flow_signal = self._detect_options_flows(market_data)
            if options_flow_signal:
                signals.append(options_flow_signal)
                
            # Earnings-related flows
            earnings_signal = self._detect_earnings_flows(market_data)
            if earnings_signal:
                signals.append(earnings_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Flow prediction error: {e}")
            return []
    
    def _build_rebalance_calendar(self) -> pd.DataFrame:
        """Build calendar of known rebalancing events"""
        # Russell recon (late June), MSCI rebalancing, etc.
        events = [
            {'date': '2025-06-27', 'event': 'Russell Reconstitution', 'intensity': 0.9},
            {'date': '2025-12-20', 'event': 'MSCI Rebalancing', 'intensity': 0.7},
        ]
        
        # Add monthly rebalancing (end of month)
        monthly_dates = pd.date_range('2024-01-01', '2025-12-31', freq='ME')
        for date in monthly_dates:
            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'Monthly Rebalancing',
                'intensity': 0.4
            })
        
        return pd.DataFrame(events)
    
    def _detect_rebalancing_flows(self, market_data: Dict) -> Optional[FlowSignal]:
        """Detect index rebalancing flows"""
        try:
            current_date = datetime.now().date()
            
            # Check if we're in a rebalancing window
            rebalance_events = self.rebalance_calendar[
                pd.to_datetime(self.rebalance_calendar['date']).dt.date == current_date
            ]
            
            if len(rebalance_events) > 0:
                event = rebalance_events.iloc[0]
                
                # Analyze volume patterns for rebalancing
                volume_analysis = self._analyze_volume_patterns(market_data)
                
                if volume_analysis['institutional_activity'] > 0.3:
                    
                    return FlowSignal(
                        signal_type="INDEX_REBALANCE",
                        flow_direction="INSTITUTIONAL_BUYING" if volume_analysis['estimated_impact'] > 0 else "INSTITUTIONAL_SELLING",
                        flow_intensity=event['intensity'],
                        expected_duration_hours=4.0,  # Typically concentrated near close
                        price_impact_estimate=volume_analysis['estimated_impact'],
                        volume_concentration=0.8,  # Very concentrated
                        dark_pool_activity=0.6,  # High dark pool usage
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Rebalancing flow detection error: {e}")
            return None
    
    def _detect_quarter_end_flows(self, market_data: Dict) -> Optional[FlowSignal]:
        """Detect quarter-end institutional flows"""
        try:
            current_date = datetime.now()
            
            # Check if we're within 3 days of quarter end
            quarter_ends = [
                datetime(current_date.year, 3, 31),
                datetime(current_date.year, 6, 30), 
                datetime(current_date.year, 9, 30),
                datetime(current_date.year, 12, 31)
            ]
            
            days_to_quarter_end = min([abs((qe - current_date).days) for qe in quarter_ends])
            
            if days_to_quarter_end <= 3:
                # Analyze flow patterns
                flow_analysis = self._analyze_quarter_end_patterns(market_data)
                
                if flow_analysis['window_dressing_detected']:
                    
                    return FlowSignal(
                        signal_type="QUARTER_END",
                        flow_direction=flow_analysis['flow_direction'],
                        flow_intensity=0.7,
                        expected_duration_hours=6.0,  # Spread over last few days
                        price_impact_estimate=flow_analysis['price_impact'],
                        volume_concentration=0.5,  # Moderately concentrated
                        dark_pool_activity=0.4,  # Moderate dark pool usage
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Quarter-end flow detection error: {e}")
            return None
    
    def _detect_options_flows(self, market_data: Dict) -> Optional[FlowSignal]:
        """Detect options expiration related flows"""
        try:
            # Check for options expiration (typically 3rd Friday)
            current_date = datetime.now().date()
            
            if self._is_options_expiration_week(current_date):
                
                # Analyze gamma and delta hedging flows
                options_analysis = self._analyze_options_flows(market_data)
                
                if options_analysis['hedging_flow_detected']:
                    
                    return FlowSignal(
                        signal_type="OPTIONS_EXPIRY",
                        flow_direction=options_analysis['flow_direction'],
                        flow_intensity=options_analysis['gamma_exposure'],
                        expected_duration_hours=2.0,  # Concentrated around expiry
                        price_impact_estimate=options_analysis['delta_impact'],
                        volume_concentration=0.9,  # Very concentrated
                        dark_pool_activity=0.2,  # Lower dark pool for hedging
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Options flow detection error: {e}")
            return None
    
    def _detect_earnings_flows(self, market_data: Dict) -> Optional[FlowSignal]:
        """Detect earnings-related institutional flows"""
        try:
            # This would integrate with earnings calendar
            # For now, simplified detection based on volume patterns
            
            volume_analysis = self._analyze_volume_patterns(market_data)
            
            if volume_analysis['institutional_activity'] > 0.5:
                earnings_patterns = self._analyze_earnings_patterns(market_data)
                
                if earnings_patterns['pre_earnings_positioning']:
                    
                    return FlowSignal(
                        signal_type="EARNINGS",
                        flow_direction=earnings_patterns['positioning_direction'],
                        flow_intensity=earnings_patterns['flow_intensity'],
                        expected_duration_hours=12.0,  # Pre-earnings positioning
                        price_impact_estimate=earnings_patterns['volatility_impact'],
                        volume_concentration=0.6,
                        dark_pool_activity=0.7,  # High dark pool for positioning
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Earnings flow detection error: {e}")
            return None
    
    def _analyze_volume_patterns(self, market_data: Dict) -> Dict:
        """Analyze volume patterns for institutional activity"""
        # Simplified volume analysis
        volume = market_data.get('volume', 1000000)
        avg_volume = market_data.get('avg_volume', 800000)
        
        volume_ratio = volume / avg_volume
        large_block_ratio = min(volume_ratio / 2.0, 1.0)
        
        return {
            'institutional_activity': large_block_ratio,
            'estimated_impact': volume_ratio * 0.001,  # 0.1% per 2x volume
            'concentration': min(volume_ratio, 2.0) / 2.0,
            'dark_pool_ratio': 0.3  # Default
        }
    
    def _analyze_quarter_end_patterns(self, market_data: Dict) -> Dict:
        """Analyze quarter-end flow patterns"""
        return {
            'window_dressing_detected': True,  # Simplified
            'flow_direction': 'INSTITUTIONAL_BUYING',
            'price_impact': 0.002
        }
    
    def _is_options_expiration_week(self, current_date) -> bool:
        """Check if current week contains options expiration"""
        # Third Friday of the month
        year = current_date.year
        month = current_date.month
        
        # Find third Friday
        first_day = datetime(year, month, 1).date()
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        # Check if within expiration week
        days_to_expiry = abs((third_friday - current_date).days)
        return days_to_expiry <= 4
    
    def _analyze_options_flows(self, market_data: Dict) -> Dict:
        """Analyze options-related hedging flows"""
        return {
            'hedging_flow_detected': True,  # Simplified
            'flow_direction': 'INSTITUTIONAL_SELLING',
            'gamma_exposure': 0.5,
            'delta_impact': 0.001
        }
    
    def _analyze_earnings_patterns(self, market_data: Dict) -> Dict:
        """Analyze earnings-related flow patterns"""
        return {
            'pre_earnings_positioning': False,  # Simplified
            'positioning_direction': 'INSTITUTIONAL_BUYING',
            'flow_intensity': 0.4,
            'volatility_impact': 0.003
        }

class IntegratedVolatilityFlowEngine:
    """
    Integrated engine combining volatility and flow strategies
    Provides comprehensive institutional signal generation
    """
    
    def __init__(self):
        """Initialize integrated engine"""
        self.volatility_engine = AdvancedVolatilityEngine()
        self.flow_predictor = InstitutionalFlowPredictor()
        
        self.combined_signals = []
        self.strategy_correlations = np.eye(3)  # Vol, Flow, Pairs
        
        logger.info("🔥 Integrated Volatility-Flow Engine initialized")
    
    def generate_institutional_signals(self, market_data: Dict) -> Dict:
        """Generate comprehensive institutional signals"""
        try:
            signals = {
                'volatility_signals': [],
                'flow_signals': [],
                'combined_score': 0.0,
                'risk_adjusted_allocation': {}
            }
            
            # Generate volatility signals
            if 'vix_data' in market_data:
                vol_signal = self.volatility_engine.analyze_vix_term_structure(
                    market_data['vix_data'].get('futures_prices', {}),
                    market_data['vix_data'].get('spot', 20.0)
                )
                
                if vol_signal.confidence > 0.3:
                    signals['volatility_signals'].append(vol_signal)
            
            # Generate flow signals
            flow_signals = self.flow_predictor.predict_institutional_flows(market_data)
            signals['flow_signals'] = flow_signals
            
            # Calculate combined score
            vol_score = sum([s.confidence * s.expected_return for s in signals['volatility_signals']])
            flow_score = sum([s.flow_intensity * s.price_impact_estimate * 100 for s in flow_signals])
            
            signals['combined_score'] = vol_score + flow_score
            
            # Risk-adjusted allocation
            if signals['combined_score'] > 0:
                signals['risk_adjusted_allocation'] = self._calculate_risk_allocation(
                    signals['volatility_signals'], flow_signals
                )
            
            logger.debug(f"🔥 Generated institutional signals: "
                        f"{len(signals['volatility_signals'])} vol, {len(flow_signals)} flow")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Institutional signal generation error: {e}")
            return {'volatility_signals': [], 'flow_signals': [], 'combined_score': 0.0}
    
    def _calculate_risk_allocation(self, vol_signals: List[VolatilitySignal], 
                                 flow_signals: List[FlowSignal]) -> Dict:
        """Calculate risk-adjusted position allocation"""
        try:
            allocation = {}
            
            # Volatility strategy allocation
            for vol_signal in vol_signals:
                if vol_signal.risk_level == "LOW":
                    base_allocation = 0.3
                elif vol_signal.risk_level == "MEDIUM":
                    base_allocation = 0.2
                else:  # HIGH
                    base_allocation = 0.1
                
                # Adjust by confidence
                adjusted_allocation = base_allocation * vol_signal.confidence
                allocation[f"VOL_{vol_signal.signal_type}"] = adjusted_allocation
            
            # Flow strategy allocation
            for flow_signal in flow_signals:
                # Flow strategies generally lower risk due to predictable nature
                base_allocation = 0.25
                
                # Adjust by intensity and duration
                time_decay = max(0.5, 1.0 - flow_signal.expected_duration_hours / 24.0)
                adjusted_allocation = base_allocation * flow_signal.flow_intensity * time_decay
                
                allocation[f"FLOW_{flow_signal.signal_type}"] = adjusted_allocation
            
            # Normalize allocations to sum to 1.0
            total_allocation = sum(allocation.values())
            if total_allocation > 0:
                allocation = {k: v / total_allocation for k, v in allocation.items()}
            
            return allocation
            
        except Exception as e:
            logger.error(f"❌ Risk allocation calculation error: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    print("🌊 Volatility Flow Strategies Test")
    print("=" * 60)
    
    # Test volatility analysis
    print("📈 Testing VIX Term Structure Analysis...")
    vol_engine = AdvancedVolatilityEngine()
    
    # Test contango scenario
    vix_futures = {'VX1': 18.5, 'VX2': 20.2, 'VX3': 21.1}
    vol_signal = vol_engine.analyze_vix_term_structure(vix_futures, 17.8)
    print(f"   Structure: {vol_signal.signal_type}")
    print(f"   Direction: {vol_signal.entry_direction}")
    print(f"   Confidence: {vol_signal.confidence:.3f}")
    print(f"   Expected Return: {vol_signal.expected_return:.3f}")
    
    # Test flow prediction
    print("\n🌊 Testing Institutional Flow Prediction...")
    flow_predictor = InstitutionalFlowPredictor()
    
    market_data = {
        'volume': 1500000,
        'avg_volume': 1000000,
        'price_change': 0.005
    }
    
    flow_signals = flow_predictor.predict_institutional_flows(market_data)
    print(f"   Detected {len(flow_signals)} flow signals")
    
    for signal in flow_signals:
        print(f"   {signal.signal_type}: {signal.flow_direction}")
        print(f"      Intensity: {signal.flow_intensity:.2f}")
        print(f"      Duration: {signal.expected_duration_hours:.1f}h")
    
    # Test integrated engine
    print("\n🔥 Testing Integrated Volatility-Flow Engine...")
    engine = IntegratedVolatilityFlowEngine()
    
    # Simulate market data
    market_data = {
        'vix_data': {
            'futures_prices': {
                'VX1': 18.5,
                'VX2': 20.2,
                'VX3': 21.1
            },
            'spot': 17.8
        },
        'volume': 1500000,
        'avg_volume': 1000000,
        'price_change': 0.005
    }
    
    # Generate signals
    signals = engine.generate_institutional_signals(market_data)
    
    print("Volatility Signals:")
    for signal in signals['volatility_signals']:
        print(f"  {signal.signal_type}: {signal.entry_direction} (Conf: {signal.confidence:.2f})")
    
    print("\nFlow Signals:")
    for signal in signals['flow_signals']:
        print(f"  {signal.signal_type}: {signal.flow_direction} (Int: {signal.flow_intensity:.2f})")
    
    print(f"\nCombined Score: {signals['combined_score']:.3f}")
    
    if signals['risk_adjusted_allocation']:
        print("\nRisk-Adjusted Allocation:")
        for strategy, allocation in signals['risk_adjusted_allocation'].items():
            print(f"  {strategy}: {allocation:.1%}")
    
    print("\n✅ Volatility flow strategies test complete!")