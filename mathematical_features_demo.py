"""
Mathematical Features Demo
Demonstrates the replacement of technical indicators with mathematical functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

from feature_store.realtime.mathematical_features import MathematicalFeatureEngine, MarketData
from ml_models.training.trading_model import TradingMLModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathematicalFeaturesDemo:
    """
    Demonstration of mathematical feature generation
    Shows how mathematical functions replace technical indicators
    """
    
    def __init__(self):
        self.mathematical_engine = MathematicalFeatureEngine(lookback_periods=200)
        self.ml_model = TradingMLModel()
        
        # Generate synthetic ES data for demonstration
        self.synthetic_data = self._generate_synthetic_es_data()
        
        logger.info("🧮 Mathematical Features Demo initialized")
    
    def _generate_synthetic_es_data(self, n_points=500):
        """Generate synthetic ES futures data for demonstration"""
        
        # Start with realistic ES price around 5000
        base_price = 5000.0
        
        # Generate realistic price movements
        np.random.seed(42)
        returns = np.random.normal(0, 0.002, n_points)  # ~0.2% daily vol
        
        # Add some trend and mean reversion
        for i in range(1, len(returns)):
            # Mean reversion component
            returns[i] += -0.1 * returns[i-1]
            
            # Trend component (small)
            if i > 100:
                returns[i] += 0.0001
        
        # Convert to prices
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC data
        data = []
        base_time = datetime.now() - timedelta(minutes=n_points)
        
        for i, price in enumerate(prices):
            # Generate realistic OHLC around close price
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            open_price = prices[i-1] if i > 0 else price
            
            # Generate volume
            volume = int(np.random.normal(1000, 200))
            
            market_data = MarketData(
                timestamp=base_time + timedelta(minutes=i),
                open=open_price,
                high=max(high, price, open_price),
                low=min(low, price, open_price),
                close=price,
                volume=max(volume, 100)
            )
            
            data.append(market_data)
        
        return data
    
    def demonstrate_mathematical_vs_technical(self):
        """
        Compare mathematical functions vs technical indicators
        Show how mathematical features provide more rigorous analysis
        """
        
        logger.info("🔬 Demonstrating Mathematical vs Technical Indicators")
        logger.info("=" * 80)
        
        # Process some sample data
        feature_comparisons = []
        
        for i, market_data in enumerate(self.synthetic_data[-50:]):  # Last 50 points
            
            # Generate mathematical features
            feature_set = self.mathematical_engine.generate_features(market_data)
            features = feature_set.features
            
            if len(features) > 20:  # Ensure we have sufficient features
                
                comparison = {
                    'timestamp': market_data.timestamp,
                    'price': market_data.close,
                    
                    # === MATHEMATICAL REPLACEMENTS ===
                    
                    # Z-Score (replaces RSI)
                    'z_score_20': features.get('z_score_price_20', 0),
                    'z_score_50': features.get('z_score_price_50', 0),
                    'rsi_equivalent': self._z_score_to_rsi(features.get('z_score_price_20', 0)),
                    
                    # Correlation (replaces MACD)
                    'autocorr_1': features.get('autocorr_lag_1', 0),
                    'autocorr_5': features.get('autocorr_lag_5', 0),
                    'macd_equivalent': features.get('autocorr_lag_1', 0) * 100,
                    
                    # Quantiles (replace Bollinger Bands)
                    'quantile_5': features.get('quantile_5', 0),
                    'quantile_95': features.get('quantile_95', 0),
                    'bollinger_equivalent': features.get('return_percentile_rank', 0.5),
                    
                    # GARCH Volatility (replaces ATR)
                    'garch_vol': features.get('ewm_volatility_alpha_10', 0),
                    'atr_equivalent': features.get('ewm_volatility_alpha_10', 0) * market_data.close,
                    
                    # Information Theory
                    'shannon_entropy': features.get('shannon_entropy', 0),
                    'market_uncertainty': features.get('relative_entropy', 0),
                    
                    # Probability Functions
                    'var_95': features.get('var_95', 0),
                    'expected_shortfall': features.get('expected_shortfall_95', 0),
                    
                    # Optimization
                    'sharpe_ratio': features.get('sharpe_ratio', 0),
                    'kelly_fraction': features.get('kelly_fraction', 0)
                }
                
                feature_comparisons.append(comparison)
        
        # Display comparison
        if feature_comparisons:
            latest = feature_comparisons[-1]
            
            logger.info("📊 LATEST MATHEMATICAL ANALYSIS:")
            logger.info(f"Price: ${latest['price']:.2f}")
            logger.info("")
            
            logger.info("🧮 MATHEMATICAL FUNCTIONS (NEW):")
            logger.info(f"  Z-Score (20): {latest['z_score_20']:.3f}")
            logger.info(f"  Z-Score (50): {latest['z_score_50']:.3f}")
            logger.info(f"  Autocorr (1): {latest['autocorr_1']:.3f}")
            logger.info(f"  GARCH Vol: {latest['garch_vol']:.4f}")
            logger.info(f"  Shannon Entropy: {latest['shannon_entropy']:.3f}")
            logger.info(f"  VaR (95%): {latest['var_95']:.4f}")
            logger.info(f"  Kelly Fraction: {latest['kelly_fraction']:.3f}")
            logger.info("")
            
            logger.info("📈 TECHNICAL INDICATOR EQUIVALENTS (OLD):")
            logger.info(f"  RSI Equivalent: {latest['rsi_equivalent']:.1f}")
            logger.info(f"  MACD Equivalent: {latest['macd_equivalent']:.1f}")
            logger.info(f"  Bollinger Position: {latest['bollinger_equivalent']:.3f}")
            logger.info(f"  ATR Equivalent: ${latest['atr_equivalent']:.2f}")
            
        return feature_comparisons
    
    def demonstrate_mathematical_signal_generation(self):
        """Demonstrate signal generation using mathematical features"""
        
        logger.info("\n🎯 MATHEMATICAL SIGNAL GENERATION")
        logger.info("=" * 80)
        
        signals_generated = []
        
        for market_data in self.synthetic_data[-10:]:  # Last 10 points
            
            # Generate mathematical features
            feature_set = self.mathematical_engine.generate_features(market_data)
            features = feature_set.features
            
            if len(features) > 20:
                
                # Use ML model with mathematical features
                ml_features = self.ml_model.prepare_features(features)
                
                # Generate trading signal based on mathematical analysis
                signal = self._generate_mathematical_signal(features)
                
                signals_generated.append({
                    'timestamp': market_data.timestamp,
                    'price': market_data.close,
                    'signal': signal,
                    'key_features': {
                        'z_score': features.get('z_score_price_20', 0),
                        'var_95': features.get('var_95', 0),
                        'entropy': features.get('shannon_entropy', 0),
                        'kelly': features.get('kelly_fraction', 0)
                    }
                })
        
        # Display signals
        for signal_data in signals_generated[-5:]:  # Last 5 signals
            logger.info(f"Time: {signal_data['timestamp'].strftime('%H:%M:%S')}")
            logger.info(f"Price: ${signal_data['price']:.2f}")
            logger.info(f"Signal: {signal_data['signal']['action']} (Confidence: {signal_data['signal']['confidence']:.2f})")
            logger.info(f"Mathematical Basis:")
            logger.info(f"  Z-Score: {signal_data['key_features']['z_score']:.3f}")
            logger.info(f"  VaR (95%): {signal_data['key_features']['var_95']:.4f}")
            logger.info(f"  Entropy: {signal_data['key_features']['entropy']:.3f}")
            logger.info(f"  Kelly Fraction: {signal_data['key_features']['kelly']:.3f}")
            logger.info("-" * 40)
        
        return signals_generated
    
    def _generate_mathematical_signal(self, features):
        """Generate trading signal using mathematical criteria"""
        
        # Extract key mathematical features
        z_score = features.get('z_score_price_20', 0)
        var_95 = features.get('var_95', 0)
        entropy = features.get('shannon_entropy', 0)
        autocorr = features.get('autocorr_lag_1', 0)
        kelly = features.get('kelly_fraction', 0)
        sharpe = features.get('sharpe_ratio', 0)
        
        # Mathematical signal logic
        signal_strength = 0.0
        action = 'HOLD'
        confidence = 0.0
        
        # Z-Score based signals (replace RSI)
        if z_score > 2.0:  # Statistically high
            action = 'SELL'
            signal_strength += abs(z_score) / 2.0
        elif z_score < -2.0:  # Statistically low
            action = 'BUY'
            signal_strength += abs(z_score) / 2.0
        
        # Momentum confirmation (autocorrelation)
        if action == 'BUY' and autocorr > 0.2:
            signal_strength *= 1.2
        elif action == 'SELL' and autocorr < -0.2:
            signal_strength *= 1.2
        
        # Risk adjustment (VaR)
        if abs(var_95) > 0.03:  # High risk
            signal_strength *= 0.7
        
        # Uncertainty filter (entropy)
        if entropy > 3.5:  # High uncertainty
            signal_strength *= 0.8
        
        # Kelly criterion validation
        if kelly > 0.1 and action == 'BUY':
            signal_strength *= 1.1
        elif kelly < -0.1 and action == 'SELL':
            signal_strength *= 1.1
        
        # Final confidence calculation
        confidence = min(signal_strength, 1.0)
        
        return {
            'action': action,
            'confidence': confidence,
            'mathematical_basis': {
                'z_score': z_score,
                'var_95': var_95,
                'entropy': entropy,
                'kelly_fraction': kelly,
                'autocorr': autocorr
            }
        }
    
    def _z_score_to_rsi(self, z_score):
        """Convert Z-score to RSI-like scale for comparison"""
        # Map Z-score to 0-100 scale like RSI
        # Z-score of +2 = RSI ~70, Z-score of -2 = RSI ~30
        return 50 + (z_score / 4.0) * 50
    
    def demonstrate_feature_advantages(self):
        """Demonstrate advantages of mathematical features over technical indicators"""
        
        logger.info("\n✅ ADVANTAGES OF MATHEMATICAL FEATURES")
        logger.info("=" * 80)
        
        advantages = [
            "1. STATISTICAL RIGOR",
            "   • Z-scores provide exact standard deviation measurements",
            "   • P-values enable hypothesis testing",
            "   • Confidence intervals give probability bounds",
            "",
            "2. MATHEMATICAL FOUNDATION", 
            "   • Based on proven statistical theory",
            "   • No arbitrary parameters (like RSI 70/30 levels)",
            "   • Adaptive to market conditions",
            "",
            "3. RISK QUANTIFICATION",
            "   • VaR provides precise risk measurements", 
            "   • Expected Shortfall quantifies tail risk",
            "   • Kelly Criterion optimizes position sizing",
            "",
            "4. INFORMATION THEORY",
            "   • Shannon entropy measures market uncertainty",
            "   • Mutual information captures non-linear relationships",
            "   • Complexity measures identify regime changes",
            "",
            "5. TIME SERIES ANALYSIS",
            "   • GARCH models dynamic volatility",
            "   • Autocorrelation quantifies momentum",
            "   • Mean reversion coefficients measure stability",
            "",
            "6. OPTIMIZATION INTEGRATION",
            "   • Sharpe ratio for risk-adjusted returns",
            "   • Portfolio theory integration",
            "   • Mathematical optimization compatibility"
        ]
        
        for advantage in advantages:
            logger.info(advantage)

async def main():
    """Run the mathematical features demonstration"""
    
    demo = MathematicalFeaturesDemo()
    
    logger.info("🚀 MATHEMATICAL FEATURES DEMONSTRATION")
    logger.info("Replacing Technical Indicators with Mathematical Functions")
    logger.info("=" * 80)
    
    # 1. Compare mathematical vs technical features
    comparisons = demo.demonstrate_mathematical_vs_technical()
    
    # 2. Show signal generation
    signals = demo.demonstrate_mathematical_signal_generation()
    
    # 3. Explain advantages
    demo.demonstrate_feature_advantages()
    
    logger.info("\n🎯 SUMMARY:")
    logger.info(f"✅ Generated {len(comparisons)} mathematical feature sets")
    logger.info(f"✅ Created {len(signals)} mathematically-based signals")
    logger.info("✅ Mathematical functions provide institutional-grade analysis")
    logger.info("✅ Statistically rigorous alternative to technical indicators")

if __name__ == "__main__":
    asyncio.run(main())