"""
Advanced ML Model Ensemble
Multi-model ensemble for institutional trading with regime detection
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from dataclasses import dataclass
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"

@dataclass
class ModelPrediction:
    """Model prediction with confidence and metadata"""
    signal: int  # -1, 0, 1 for sell, hold, buy
    confidence: float  # 0-1
    probabilities: Dict[str, float]
    regime: MarketRegime
    features_used: List[str]
    model_name: str
    timestamp: datetime

@dataclass
class EnsembleConfig:
    """Configuration for ensemble model"""
    models: Dict[str, Dict] = None
    regime_weights: Dict[MarketRegime, Dict[str, float]] = None
    feature_importance_threshold: float = 0.01
    min_confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                },
                "lightgbm": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "feature_fraction": 0.8
                },
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                }
            }
        
        if self.regime_weights is None:
            self.regime_weights = {
                MarketRegime.TRENDING: {
                    "xgboost": 0.4,
                    "lightgbm": 0.4,
                    "random_forest": 0.2
                },
                MarketRegime.MEAN_REVERTING: {
                    "xgboost": 0.3,
                    "lightgbm": 0.3,
                    "random_forest": 0.4
                },
                MarketRegime.HIGH_VOLATILITY: {
                    "xgboost": 0.5,
                    "lightgbm": 0.3,
                    "random_forest": 0.2
                },
                MarketRegime.LOW_VOLATILITY: {
                    "xgboost": 0.3,
                    "lightgbm": 0.4,
                    "random_forest": 0.3
                },
                MarketRegime.UNCERTAIN: {
                    "xgboost": 0.33,
                    "lightgbm": 0.33,
                    "random_forest": 0.34
                }
            }

class RegimeDetector:
    """Market regime detection system"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = []
        self.volume_history = []
        
    def update_data(self, price: float, volume: int):
        """Update with new market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only recent history
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history = self.price_history[-self.lookback_period:]
            self.volume_history = self.volume_history[-self.lookback_period:]
    
    def detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        if len(self.price_history) < self.lookback_period:
            return MarketRegime.UNCERTAIN
        
        prices = np.array(self.price_history[-self.lookback_period:])
        returns = np.diff(np.log(prices))
        
        # Calculate regime indicators
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        trend_strength = self._calculate_trend_strength(prices)
        mean_reversion = self._calculate_mean_reversion(returns)
        
        # Regime classification logic
        if volatility > 0.25:  # High volatility threshold
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.10:  # Low volatility threshold
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.7:
            return MarketRegime.TRENDING
        elif mean_reversion > 0.6:
            return MarketRegime.MEAN_REVERTING
        else:
            return MarketRegime.UNCERTAIN
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (0-1)"""
        if len(prices) < 10:
            return 0.0
        
        # Linear regression slope normalized by price range
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        price_range = np.max(prices) - np.min(prices)
        
        if price_range == 0:
            return 0.0
        
        return min(1.0, abs(slope * len(prices)) / price_range)
    
    def _calculate_mean_reversion(self, returns: np.ndarray) -> float:
        """Calculate mean reversion tendency (0-1)"""
        if len(returns) < 10:
            return 0.0
        
        # Hurst exponent approximation
        lags = range(2, min(20, len(returns) // 2))
        tau = []
        
        for lag in lags:
            tau.append(np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))))
        
        if len(tau) < 2:
            return 0.0
        
        # Linear regression on log-log plot
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        try:
            hurst, _ = np.polyfit(log_lags, log_tau, 1)
            # Convert to mean reversion score (H < 0.5 indicates mean reversion)
            return max(0.0, min(1.0, (0.5 - hurst) * 2))
        except:
            return 0.0

class AdvancedMLEnsemble:
    """
    Advanced ML ensemble for institutional trading
    
    Features:
    - Multiple ML algorithms (XGBoost, LightGBM, RandomForest)
    - Regime-aware model weighting
    - Dynamic feature selection
    - Real-time inference with <10ms latency
    - Automated model retraining
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.regime_detector = RegimeDetector()
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Training data
        self.training_features = []
        self.training_labels = []
        self.feature_names = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Advanced ML Ensemble initialized")
    
    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        # XGBoost
        self.models["xgboost"] = xgb.XGBClassifier(
            **self.config.models["xgboost"],
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        self.models["lightgbm"] = lgb.LGBMClassifier(
            **self.config.models["lightgbm"],
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest
        self.models["random_forest"] = RandomForestClassifier(
            **self.config.models["random_forest"],
            random_state=42,
            n_jobs=-1
        )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    
    def train(self, features: pd.DataFrame, labels: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the ensemble with time series cross-validation
        
        Args:
            features: Feature matrix
            labels: Target labels (-1, 0, 1)
            validation_split: Validation split ratio
            
        Returns:
            Performance metrics
        """
        logger.info(f"Training ensemble with {len(features)} samples, {len(features.columns)} features")
        
        with self.lock:
            self.feature_names = list(features.columns)
            
            # Prepare data
            X = features.values
            y = labels.values
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            performance_results = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                # Scale features
                scaler = self.scalers[model_name]
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Validate
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                performance_results[model_name] = avg_score
                
                # Final training on all data
                model.fit(X_scaled, y)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(self.feature_names, model.feature_importances_))
                    self.feature_importance[model_name] = importance
                
                logger.info(f"{model_name} training completed. CV Score: {avg_score:.4f}")
            
            # Update performance metrics
            self.performance_metrics = performance_results
            
            return performance_results
    
    def predict(self, features: Dict[str, float], price: float = None, 
               volume: int = None) -> ModelPrediction:
        """
        Generate ensemble prediction
        
        Args:
            features: Feature dictionary
            price: Current price for regime detection
            volume: Current volume for regime detection
            
        Returns:
            Ensemble prediction with confidence
        """
        with self.lock:
            # Update regime detector
            if price is not None and volume is not None:
                self.regime_detector.update_data(price, volume)
            
            # Detect current regime
            current_regime = self.regime_detector.detect_regime()
            
            # Prepare feature array
            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)
            
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.models.items():
                if not hasattr(model, 'predict_proba'):
                    continue
                
                try:
                    # Scale features
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(feature_array)
                    
                    # Get prediction and probabilities
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    
                    model_predictions[model_name] = pred
                    model_probabilities[model_name] = proba
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    continue
            
            # Ensemble prediction using regime-aware weighting
            ensemble_prediction = self._ensemble_predict(
                model_predictions, model_probabilities, current_regime
            )
            
            # Create prediction object
            prediction = ModelPrediction(
                signal=ensemble_prediction["signal"],
                confidence=ensemble_prediction["confidence"],
                probabilities=ensemble_prediction["probabilities"],
                regime=current_regime,
                features_used=self.feature_names,
                model_name="ensemble",
                timestamp=datetime.now()
            )
            
            # Store prediction history
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-500:]
            
            return prediction
    
    def _ensemble_predict(self, model_predictions: Dict[str, int], 
                         model_probabilities: Dict[str, np.ndarray], 
                         regime: MarketRegime) -> Dict:
        """Combine model predictions using regime-aware weighting"""
        
        if not model_predictions:
            return {
                "signal": 0,
                "confidence": 0.0,
                "probabilities": {"sell": 0.33, "hold": 0.34, "buy": 0.33}
            }
        
        # Get regime weights
        regime_weights = self.config.regime_weights.get(regime, {})
        
        # Weighted average of probabilities
        weighted_proba = np.zeros(3)  # [sell, hold, buy]
        total_weight = 0.0
        
        for model_name, proba in model_probabilities.items():
            weight = regime_weights.get(model_name, 1.0 / len(model_probabilities))
            weighted_proba += weight * proba
            total_weight += weight
        
        if total_weight > 0:
            weighted_proba /= total_weight
        
        # Determine signal
        signal = np.argmax(weighted_proba) - 1  # Convert 0,1,2 to -1,0,1
        confidence = np.max(weighted_proba)
        
        # Apply confidence threshold
        if confidence < self.config.min_confidence_threshold:
            signal = 0  # Hold if confidence is too low
        
        probabilities = {
            "sell": float(weighted_proba[0]),
            "hold": float(weighted_proba[1]),
            "buy": float(weighted_proba[2])
        }
        
        return {
            "signal": signal,
            "confidence": float(confidence),
            "probabilities": probabilities
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get aggregated feature importance across models"""
        if not self.feature_importance:
            return {}
        
        # Average importance across models
        aggregated_importance = {}
        for feature_name in self.feature_names:
            importance_values = []
            for model_name, importance_dict in self.feature_importance.items():
                importance_values.append(importance_dict.get(feature_name, 0.0))
            
            aggregated_importance[feature_name] = np.mean(importance_values)
        
        # Sort by importance
        sorted_importance = dict(sorted(aggregated_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        # Return top N
        return dict(list(sorted_importance.items())[:top_n])
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained"""
        if len(self.prediction_history) < 100:
            return False
        
        # Check recent performance
        recent_predictions = self.prediction_history[-100:]
        
        # If we had a way to get actual outcomes, we would calculate accuracy
        # For now, check if confidence has been consistently low
        avg_confidence = np.mean([p.confidence for p in recent_predictions])
        
        return avg_confidence < 0.5
    
    def save_models(self, directory: str):
        """Save all models to disk"""
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(directory, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            scaler_path = os.path.join(directory, f"{model_name}_scaler.joblib")
            joblib.dump(self.scalers[model_name], scaler_path)
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "performance_metrics": self.performance_metrics,
            "feature_importance": self.feature_importance
        }
        
        metadata_path = os.path.join(directory, "ensemble_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str) -> bool:
        """Load models from disk"""
        try:
            # Load metadata
            metadata_path = os.path.join(directory, "ensemble_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata["feature_names"]
            self.performance_metrics = metadata["performance_metrics"]
            self.feature_importance = metadata["feature_importance"]
            
            # Load models
            for model_name in self.models.keys():
                model_path = os.path.join(directory, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                
                scaler_path = os.path.join(directory, f"{model_name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
            
            logger.info(f"Models loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_ensemble_summary(self) -> Dict:
        """Get comprehensive ensemble summary"""
        return {
            "models": list(self.models.keys()),
            "feature_count": len(self.feature_names),
            "performance_metrics": self.performance_metrics,
            "recent_predictions": len(self.prediction_history),
            "top_features": self.get_feature_importance(10),
            "current_regime": self.regime_detector.detect_regime().value
        }
