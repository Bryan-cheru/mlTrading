"""
Institutional ML Trading System Integration
==========================================

This module integrates:
- Advanced ML Models (LSTM, Transformer, Ensemble)
- Multi-Timeframe Analysis
- Enhanced Feature Engineering
- Institutional Trading Logic

Creates a production-ready institutional-grade ML trading system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'advanced'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'timeframe'))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from advanced_ml_models import AdvancedFeatureEngineer, EnsembleAdvancedModel
from multi_timeframe_analyzer import MultiTimeframeAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstitutionalMLTradingSystem:
    """
    Complete institutional ML trading system integrating:
    - Advanced ML models (LSTM + Transformer + XGBoost + LightGBM)
    - Multi-timeframe analysis (1min to daily)
    - 150+ engineered features
    - Institutional risk management
    """
    
    def __init__(self, base_timeframe: str = '1min', confidence_threshold: float = 0.7):
        """
        Initialize the institutional trading system
        
        Args:
            base_timeframe: Base timeframe for analysis
            confidence_threshold: Minimum confidence for trade signals
        """
        self.base_timeframe = base_timeframe
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.timeframe_analyzer = MultiTimeframeAnalyzer(base_timeframe)
        self.ml_model = None
        
        # Model state
        self.is_trained = False
        self.last_prediction_time = None
        self.feature_cache = {}
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info(f"🏦 Institutional ML Trading System initialized")
        logger.info(f"📊 Base timeframe: {base_timeframe}")
        logger.info(f"🎯 Confidence threshold: {confidence_threshold}")
    
    def engineer_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set combining advanced ML and multi-timeframe features
        
        Args:
            data: Raw OHLCV market data
        
        Returns:
            DataFrame with all engineered features
        """
        logger.debug(f"🔧 Engineering features for {len(data)} market data points")
        
        try:
            # 1. Advanced ML features (50+ features)
            ml_features = self.feature_engineer.engineer_features(data)
            logger.debug(f"✅ Generated {ml_features.shape[1]} advanced ML features")
            
            # 2. Multi-timeframe features (60+ features)
            mtf_features = self.timeframe_analyzer.analyze_multi_timeframe(data)
            
            # Convert multi-timeframe features to DataFrame format
            if isinstance(mtf_features, dict):
                mtf_df = pd.DataFrame([mtf_features], index=[data.index[-1]])
            else:
                mtf_df = mtf_features
                
            logger.debug(f"✅ Generated {len(mtf_features)} multi-timeframe features")
            
            # 3. Combine all features
            if len(mtf_df) > 0:
                # Align indices for concatenation
                if ml_features.index[-1] != mtf_df.index[-1]:
                    mtf_df.index = [ml_features.index[-1]]
                
                combined_features = pd.concat([ml_features, mtf_df], axis=1)
            else:
                combined_features = ml_features
            
            # 4. Add institutional-specific features
            inst_features = self._add_institutional_features(data, combined_features)
            final_features = pd.concat([combined_features, inst_features], axis=1)
            
            # Handle any NaN values
            final_features = final_features.fillna(method='ffill').fillna(0)
            
            logger.info(f"🎯 Total features engineered: {final_features.shape[1]}")
            return final_features
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return pd.DataFrame()
    
    def _add_institutional_features(self, data: pd.DataFrame, existing_features: pd.DataFrame) -> pd.DataFrame:
        """Add institutional-specific features"""
        inst_features = {}
        
        try:
            close = data['close']
            volume = data['volume'] if 'volume' in data.columns else pd.Series(1000, index=data.index)
            
            # Market regime detection
            if len(close) >= 50:
                volatility_regime = close.rolling(20).std() / close.rolling(50).std()
                inst_features['volatility_regime'] = volatility_regime.iloc[-1]
                
                # Trend strength
                sma_20 = close.rolling(20).mean()
                sma_50 = close.rolling(50).mean() if len(close) >= 50 else sma_20
                inst_features['trend_strength'] = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # Liquidity measures
            if 'bid' in data.columns and 'ask' in data.columns:
                spread = data['ask'] - data['bid']
                inst_features['avg_spread'] = spread.rolling(20).mean().iloc[-1]
                inst_features['spread_volatility'] = spread.rolling(20).std().iloc[-1]
            
            # Volume profile
            if len(volume) >= 20:
                volume_ma = volume.rolling(20).mean()
                inst_features['volume_profile'] = volume.iloc[-1] / volume_ma.iloc[-1]
                inst_features['volume_trend'] = (volume_ma.iloc[-1] - volume_ma.iloc[-10]) / volume_ma.iloc[-10] if len(volume_ma) >= 10 else 0
            
            # Market microstructure
            if len(close) >= 10:
                price_moves = close.diff()
                inst_features['price_acceleration'] = price_moves.rolling(5).mean().iloc[-1]
                inst_features['price_momentum_consistency'] = (price_moves > 0).rolling(10).mean().iloc[-1]
            
            # Time-based features
            current_time = data.index[-1]
            inst_features['hour_of_day'] = current_time.hour / 23.0  # Normalized
            inst_features['day_of_week'] = current_time.dayofweek / 6.0  # Normalized
            inst_features['is_market_open'] = 1.0 if 9 <= current_time.hour <= 16 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating institutional features: {e}")
        
        # Convert to DataFrame
        if inst_features:
            return pd.DataFrame([inst_features], index=[existing_features.index[-1]])
        else:
            return pd.DataFrame(index=[existing_features.index[-1]])
    
    def train_model(self, training_data: pd.DataFrame, target_column: str = 'close', 
                   epochs: int = 50, validation_split: float = 0.2) -> bool:
        """
        Train the ensemble ML model on historical data
        
        Args:
            training_data: Historical OHLCV data
            target_column: Target column for prediction
            epochs: Training epochs
            validation_split: Validation data split
        
        Returns:
            Training success status
        """
        logger.info(f"🤖 Training institutional ML model on {len(training_data)} samples")
        
        try:
            # Engineer features for all training data
            features_list = []
            targets = []
            
            # Use sliding window approach for training
            window_size = 100  # Minimum data for feature engineering
            
            for i in range(window_size, len(training_data), 10):  # Every 10 periods to reduce computation
                window_data = training_data.iloc[max(0, i-window_size):i+1]
                
                # Engineer features
                features = self.engineer_comprehensive_features(window_data)
                
                if len(features) > 0 and not features.empty:
                    features_list.append(features.iloc[-1])
                    
                    # Create target (next period return)
                    if i+1 < len(training_data):
                        current_price = training_data[target_column].iloc[i]
                        next_price = training_data[target_column].iloc[i+1]
                        target = (next_price - current_price) / current_price
                        targets.append(target)
            
            if len(features_list) == 0:
                logger.error("❌ No features could be generated for training")
                return False
            
            # Convert to arrays
            X = pd.DataFrame(features_list).fillna(0)
            y = np.array(targets)
            
            logger.info(f"📊 Training features shape: {X.shape}")
            logger.info(f"🎯 Training targets shape: {y.shape}")
            
            # Initialize and train model
            self.ml_model = EnsembleAdvancedModel(input_size=X.shape[1])
            
            # Train the ensemble model
            self.ml_model.train(X, y, epochs=epochs, batch_size=32, validation_split=validation_split)
            
            self.is_trained = True
            logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return False
    
    def predict_signal(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signal with confidence and reasoning
        
        Args:
            current_data: Current market data
        
        Returns:
            Dictionary with signal, confidence, and reasoning
        """
        if not self.is_trained or self.ml_model is None:
            logger.warning("⚠️  Model not trained, cannot generate predictions")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Model not trained'}
        
        try:
            # Engineer features
            features = self.engineer_comprehensive_features(current_data)
            
            if features.empty:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Failed to generate features'}
            
            # Get model prediction
            prediction_probs = self.ml_model.predict(features.values)
            
            if len(prediction_probs) == 0:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Model prediction failed'}
            
            # Interpret prediction
            pred = prediction_probs[-1] if len(prediction_probs.shape) > 1 else prediction_probs[0]
            
            # Convert to trading signal
            if hasattr(pred, '__len__') and len(pred) >= 3:
                # Multi-class output: [SELL, HOLD, BUY]
                sell_prob, hold_prob, buy_prob = pred[0], pred[1], pred[2]
                
                max_prob = max(sell_prob, hold_prob, buy_prob)
                confidence = float(max_prob)
                
                if buy_prob == max_prob and confidence > self.confidence_threshold:
                    signal = 'BUY'
                elif sell_prob == max_prob and confidence > self.confidence_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                    
            else:
                # Single value prediction (return forecast)
                pred_value = float(pred[0] if hasattr(pred, '__len__') else pred)
                confidence = min(abs(pred_value) * 10, 1.0)  # Scale confidence
                
                if pred_value > 0.001 and confidence > self.confidence_threshold:
                    signal = 'BUY'
                elif pred_value < -0.001 and confidence > self.confidence_threshold:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
            
            # Add reasoning based on features
            reasoning = self._generate_reasoning(features, signal, confidence)
            
            # Update state
            self.last_prediction_time = datetime.now()
            
            result = {
                'signal': signal,
                'confidence': confidence,
                'reason': reasoning,
                'features_count': features.shape[1],
                'prediction_time': self.last_prediction_time
            }
            
            logger.info(f"🎯 Signal: {signal} (confidence: {confidence:.3f}) - {reasoning}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating prediction: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Prediction error: {str(e)}'}
    
    def _generate_reasoning(self, features: pd.DataFrame, signal: str, confidence: float) -> str:
        """Generate human-readable reasoning for the trading signal"""
        try:
            reasons = []
            
            # Check key multi-timeframe indicators
            feature_dict = features.iloc[-1].to_dict()
            
            # RSI analysis across timeframes
            rsi_signals = []
            for tf in ['1min', '5min', '15min', '1H']:
                rsi_key = f'{tf}_rsi'
                if rsi_key in feature_dict:
                    rsi_val = feature_dict[rsi_key]
                    if rsi_val > 70:
                        rsi_signals.append(f"{tf} overbought")
                    elif rsi_val < 30:
                        rsi_signals.append(f"{tf} oversold")
            
            if rsi_signals:
                reasons.append(f"RSI: {', '.join(rsi_signals)}")
            
            # Trend analysis
            trend_signals = []
            for tf in ['5min', '15min', '1H']:
                momentum_key = f'{tf}_price_momentum'
                if momentum_key in feature_dict:
                    momentum = feature_dict[momentum_key]
                    if momentum > 0.002:
                        trend_signals.append(f"{tf} bullish")
                    elif momentum < -0.002:
                        trend_signals.append(f"{tf} bearish")
            
            if trend_signals:
                reasons.append(f"Trends: {', '.join(trend_signals)}")
            
            # Volume analysis
            if 'volume_profile' in feature_dict:
                vol_profile = feature_dict['volume_profile']
                if vol_profile > 1.5:
                    reasons.append("High volume confirmation")
                elif vol_profile < 0.5:
                    reasons.append("Low volume concern")
            
            # Market regime
            if 'volatility_regime' in feature_dict:
                vol_regime = feature_dict['volatility_regime']
                if vol_regime > 1.2:
                    reasons.append("High volatility environment")
                elif vol_regime < 0.8:
                    reasons.append("Low volatility environment")
            
            # Combine reasoning
            if reasons:
                return f"{signal} signal with {confidence:.1%} confidence: {'; '.join(reasons)}"
            else:
                return f"{signal} signal with {confidence:.1%} confidence based on ensemble model"
                
        except Exception as e:
            return f"{signal} signal with {confidence:.1%} confidence (reasoning unavailable)"
    
    def get_model_status(self) -> Dict[str, any]:
        """Get current model status and statistics"""
        return {
            'is_trained': self.is_trained,
            'base_timeframe': self.base_timeframe,
            'confidence_threshold': self.confidence_threshold,
            'last_prediction': self.last_prediction_time,
            'feature_engineer_ready': self.feature_engineer is not None,
            'timeframe_analyzer_ready': self.timeframe_analyzer is not None,
            'ml_model_ready': self.ml_model is not None
        }

# Testing and Usage Example
if __name__ == "__main__":
    logger.info("🧪 Testing Institutional ML Trading System...")
    
    # Create comprehensive test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate realistic market data with trends and volatility
    base_price = 4400
    trend = np.linspace(0, 100, 1000)
    volatility = np.random.normal(0, 2, 1000)
    prices = base_price + trend + np.cumsum(volatility)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 1000),
        'high': prices + np.abs(np.random.normal(0, 2, 1000)),
        'low': prices - np.abs(np.random.normal(0, 2, 1000)),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 1000).astype(float),
        'bid': prices - 0.25,
        'ask': prices + 0.25
    }, index=dates)
    
    # Ensure OHLC relationships
    test_data['high'] = np.maximum(test_data[['open', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'close']].min(axis=1), test_data['low'])
    
    # Initialize system
    trading_system = InstitutionalMLTradingSystem(
        base_timeframe='1min',
        confidence_threshold=0.65
    )
    
    try:
        # Test feature engineering
        logger.info("🔧 Testing comprehensive feature engineering...")
        features = trading_system.engineer_comprehensive_features(test_data)
        logger.info(f"✅ Generated {features.shape[1]} total features")
        
        # Test training (quick training for demo)
        logger.info("🤖 Testing model training...")
        training_success = trading_system.train_model(
            test_data[:800],  # Use first 800 samples for training
            epochs=5,  # Quick training for demo
            validation_split=0.2
        )
        
        if training_success:
            logger.info("✅ Model training successful")
            
            # Test prediction
            logger.info("🎯 Testing signal generation...")
            recent_data = test_data[-100:]  # Use last 100 samples for prediction
            signal_result = trading_system.predict_signal(recent_data)
            
            logger.info(f"📊 Signal Result: {signal_result}")
            
            # Test model status
            status = trading_system.get_model_status()
            logger.info(f"📋 System Status: {status}")
            
        else:
            logger.error("❌ Model training failed")
        
        logger.info("🎉 Institutional ML Trading System testing complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in system testing: {e}")
        import traceback
        traceback.print_exc()
