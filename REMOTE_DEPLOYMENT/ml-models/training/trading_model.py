"""
ML Trading Model
XGBoost-based model for real-time trading decisions
Optimized for institutional-grade performance
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os

logger = logging.getLogger(__name__)

class TradingMLModel:
    """
    XGBoost-based machine learning model for trading decisions
    Designed for real-time inference with NinjaTrader 8 data
    """
    
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or self._get_default_config()
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = []
        self.is_trained = False
        self.last_prediction_time = None
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
    def _get_default_config(self) -> Dict:
        """Get default XGBoost configuration optimized for trading"""
        return {
            'objective': 'multi:softprob',  # Multi-class classification (BUY, SELL, HOLD)
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster for large datasets
            'early_stopping_rounds': 20,
            'eval_metric': 'mlogloss'
        }
    
    def prepare_features(self, market_features: Dict, historical_data: pd.DataFrame = None) -> np.ndarray:
        """
        Prepare features for model input
        
        Args:
            market_features: Real-time features from market data processor
            historical_data: Historical bar data for additional features
            
        Returns:
            Numpy array of processed features
        """
        try:
            # Core features from real-time data
            feature_vector = []
            
            # Price features
            feature_vector.extend([
                market_features.get('return_1', 0),
                market_features.get('return_5', 0),
                market_features.get('return_15', 0),
                market_features.get('price_vs_sma20', 0),
                market_features.get('price_vs_ema12', 0),
                market_features.get('price_vs_vwap', 0),
            ])
            
            # Technical indicators
            feature_vector.extend([
                market_features.get('rsi_14', 50),
                market_features.get('macd_line', 0),
                market_features.get('macd_histogram', 0),
                market_features.get('bb_position', 0.5),
                market_features.get('stoch_k', 50),
                market_features.get('stoch_d', 50),
            ])
            
            # Volume features
            feature_vector.extend([
                market_features.get('volume_ratio', 1),
                market_features.get('volume_trend_5', 0),
                market_features.get('obv_trend_5', 0),
            ])
            
            # Volatility features
            feature_vector.extend([
                market_features.get('atr_14', 0),
                market_features.get('volatility_10', 0),
                market_features.get('volatility_20', 0),
            ])
            
            # Statistical features
            feature_vector.extend([
                market_features.get('skewness_20', 0),
                market_features.get('kurtosis_20', 0),
                market_features.get('zscore_20', 0),
                market_features.get('autocorr_1', 0),
            ])
            
            # Microstructure features
            feature_vector.extend([
                market_features.get('high_low_ratio', 1),
                market_features.get('gap', 0),
                market_features.get('doji_ratio', 0),
                market_features.get('upper_shadow_ratio', 0),
                market_features.get('lower_shadow_ratio', 0),
            ])
            
            # Regime features
            feature_vector.extend([
                market_features.get('vol_regime', 1),
                market_features.get('volume_regime', 1),
                market_features.get('range_position', 0.5),
                market_features.get('trend_alignment', 0),
            ])
            
            # Time-based features
            now = datetime.now()
            feature_vector.extend([
                now.hour / 24.0,  # Hour of day normalized
                now.minute / 60.0,  # Minute of hour normalized
                now.weekday() / 6.0,  # Day of week normalized
            ])
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return zeros if error occurs
            return np.zeros((1, 30))
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'target') -> Dict:
        """
        Train the XGBoost model
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Training metrics
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            X = training_data.drop(columns=[target_column, 'timestamp'], errors='ignore')
            y = training_data[target_column]
            
            self.feature_names = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Create and train model
            self.model = xgb.XGBClassifier(**self.model_config)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')
            
            # Final training on all data
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_scaled, y)],
                verbose=False
            )
            
            self.is_trained = True
            
            # Calculate metrics
            y_pred = self.model.predict(X_scaled)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X),
                'features': len(self.feature_names)
            }
            
            self.performance_metrics = metrics
            logger.info(f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {}
    
    def predict(self, market_features: Dict) -> Dict:
        """
        Make real-time prediction
        
        Args:
            market_features: Real-time market features
            
        Returns:
            Prediction result with probabilities and signal
        """
        if not self.is_trained:
            logger.warning("Model not trained. Cannot make prediction.")
            return {'signal': 'HOLD', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.33]}
        
        try:
            # Prepare features
            X = self.prepare_features(market_features)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            # Map prediction to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map.get(prediction, 'HOLD')
            
            confidence = np.max(probabilities)
            
            result = {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'timestamp': datetime.now(),
                'features_used': len(self.feature_names) if self.feature_names else 0
            }
            
            # Store prediction for performance tracking
            self.prediction_history.append(result)
            self.last_prediction_time = datetime.now()
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'probabilities': [0.33, 0.33, 0.33]}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained or not self.feature_names:
            return {}
        
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.model_config,
                'performance_metrics': self.performance_metrics,
                'training_date': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_config = model_data['config']
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def update_model(self, new_data: pd.DataFrame, target_column: str = 'target'):
        """
        Update model with new data (incremental learning)
        """
        if not self.is_trained:
            logger.warning("Model not trained. Use train() method first.")
            return
        
        try:
            # Prepare new data
            X_new = new_data.drop(columns=[target_column, 'timestamp'], errors='ignore')
            y_new = new_data[target_column]
            
            # Scale new features
            X_new_scaled = self.scaler.transform(X_new)
            
            # Update model with new data
            self.model.fit(
                X_new_scaled, y_new,
                xgb_model=self.model.get_booster(),
                eval_set=[(X_new_scaled, y_new)],
                verbose=False
            )
            
            logger.info(f"Model updated with {len(new_data)} new samples")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    def get_model_status(self) -> Dict:
        """Get current model status and performance"""
        return {
            'is_trained': self.is_trained,
            'last_prediction_time': self.last_prediction_time,
            'prediction_count': len(self.prediction_history),
            'performance_metrics': self.performance_metrics,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_config': self.model_config
        }


class SignalGenerator:
    """
    Generates trading signals based on ML model predictions
    Includes risk management and position sizing
    """
    
    def __init__(self, risk_config: Dict = None):
        self.risk_config = risk_config or {
            'min_confidence': 0.65,  # Minimum confidence for signal
            'max_position_size': 0.1,  # Maximum 10% position
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.04,  # 4% take profit
            'max_daily_loss': 0.05,  # Maximum 5% daily loss
        }
        
        self.daily_pnl = 0.0
        self.open_positions = {}
        
    def generate_signal(self, prediction: Dict, current_price: float, 
                       account_balance: float) -> Dict:
        """
        Generate trading signal with risk management
        
        Args:
            prediction: ML model prediction
            current_price: Current market price
            account_balance: Current account balance
            
        Returns:
            Trading signal with position size and risk parameters
        """
        try:
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            # Check confidence threshold
            if confidence < self.risk_config['min_confidence']:
                signal = 'HOLD'
            
            # Check daily loss limit
            if self.daily_pnl < -self.risk_config['max_daily_loss'] * account_balance:
                signal = 'HOLD'
                logger.warning("Daily loss limit reached. No new positions.")
            
            # Calculate position size
            position_size = 0
            if signal in ['BUY', 'SELL']:
                max_risk = self.risk_config['max_position_size'] * account_balance
                position_size = int(max_risk / current_price)
            
            # Calculate risk parameters
            stop_loss = None
            take_profit = None
            
            if signal == 'BUY':
                stop_loss = current_price * (1 - self.risk_config['stop_loss_pct'])
                take_profit = current_price * (1 + self.risk_config['take_profit_pct'])
            elif signal == 'SELL':
                stop_loss = current_price * (1 + self.risk_config['stop_loss_pct'])
                take_profit = current_price * (1 - self.risk_config['take_profit_pct'])
            
            return {
                'signal': signal,
                'confidence': confidence,
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(),
                'risk_reward_ratio': self.risk_config['take_profit_pct'] / self.risk_config['stop_loss_pct']
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'position_size': 0,
                'timestamp': datetime.now()
            }
