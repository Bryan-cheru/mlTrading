"""
XGBoost Model Training and Validation
Institutional-grade ML model with proper validation and monitoring
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    total_return: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'sharpe_ratio': self.sharpe_ratio,
            'hit_rate': self.hit_rate,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return
        }

@dataclass
class ModelInfo:
    """Container for model metadata"""
    model_id: str
    symbol: str
    model_type: str
    training_start: datetime
    training_end: datetime
    feature_count: int
    training_samples: int
    validation_samples: int
    hyperparameters: Dict[str, Any]
    metrics: ModelMetrics

class XGBoostTrainer:
    """
    XGBoost model trainer with institutional-grade validation and monitoring
    """
    
    def __init__(self, model_params: Optional[Dict] = None):
        self.model_params = model_params or {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.model_info: Optional[ModelInfo] = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'label_return', 
                    test_size: float = 0.2, validation_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper time series split
        
        Args:
            df: DataFrame with features and labels
            target_col: Name of target column
            test_size: Fraction of data for testing
            validation_size: Fraction of remaining data for validation
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Remove rows with missing targets
        df_clean = df.dropna(subset=[target_col]).copy()
        
        if df_clean.empty:
            raise ValueError("No valid data after removing missing targets")
        
        # Sort by timestamp for proper time series split
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
        
        # Identify feature columns
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']
        label_cols = [col for col in df_clean.columns if col.startswith('label_')]
        exclude_cols.extend(label_cols)
        
        self.feature_names = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Extract features and target
        X = df_clean[self.feature_names].values
        y = df_clean[target_col].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time series split (chronological order)
        n_samples = len(X)
        test_split_idx = int(n_samples * (1 - test_size))
        val_split_idx = int(test_split_idx * (1 - validation_size))
        
        X_train = X[:val_split_idx]
        X_val = X[val_split_idx:test_split_idx]
        X_test = X[test_split_idx:]
        
        y_train = y[:val_split_idx]
        y_val = y[val_split_idx:test_split_idx]
        y_test = y[test_split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            cv_folds: int = 3) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using time series cross-validation
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              tune_hyperparameters: bool = False) -> xgb.XGBRegressor:
        """
        Train XGBoost model with early stopping
        """
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            best_params = self.hyperparameter_tuning(X_train, y_train)
            self.model_params.update(best_params)
        
        # Create model
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                symbol: str = "UNKNOWN") -> ModelMetrics:
        """
        Comprehensive model evaluation with financial metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Financial metrics
        hit_rate = np.mean((y_pred > 0) == (y_test > 0))
        
        # Simulate trading performance
        positions = np.where(y_pred > 0, 1, -1)  # Long/Short based on prediction
        strategy_returns = positions * y_test
        
        # Sharpe ratio
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Total return
        total_return = cumulative_returns[-1] - 1
        
        metrics = ModelMetrics(
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            total_return=total_return
        )
        
        logger.info(f"Model evaluation for {symbol}:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  Hit Rate: {hit_rate:.3f}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.3f}")
        logger.info(f"  Total Return: {total_return:.3f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_dir: Path, symbol: str) -> str:
        """
        Save trained model and metadata
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{symbol}_{timestamp}"
        
        # Save model
        model_path = model_dir / f"{model_id}_model.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_dir / f"{model_id}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'symbol': symbol,
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'model_info': self.model_info.to_dict() if self.model_info else None
        }
        
        metadata_path = model_dir / f"{model_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_id}")
        return model_id
    
    def load_model(self, model_dir: Path, model_id: str) -> None:
        """
        Load trained model and metadata
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / f"{model_id}_model.joblib"
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = model_dir / f"{model_id}_scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = model_dir / f"{model_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.model_params = metadata['model_params']
        
        logger.info(f"Model loaded: {model_id}")

class ModelManager:
    """
    Manages multiple models for different symbols
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, XGBoostTrainer] = {}
    
    def train_symbol_model(self, symbol: str, df: pd.DataFrame, 
                          tune_hyperparameters: bool = False) -> ModelMetrics:
        """
        Train a model for a specific symbol
        """
        logger.info(f"Training model for {symbol}")
        
        # Create trainer
        trainer = XGBoostTrainer()
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
        
        # Train model
        trainer.train(X_train, y_train, X_val, y_val, tune_hyperparameters)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test, symbol)
        
        # Save model
        model_id = trainer.save_model(self.model_dir, symbol)
        
        # Store trainer
        self.models[symbol] = trainer
        
        return metrics
    
    def predict(self, symbol: str, features: np.ndarray) -> np.ndarray:
        """
        Make predictions for a symbol
        """
        if symbol not in self.models:
            raise ValueError(f"No model trained for {symbol}")
        
        trainer = self.models[symbol]
        if trainer.model is None:
            raise ValueError(f"Model for {symbol} not trained")
        
        # Scale features
        features_scaled = trainer.scaler.transform(features)
        
        # Predict
        predictions = trainer.model.predict(features_scaled)
        
        return predictions

def main():
    """
    Example usage of XGBoost trainer
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with features and labels
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic feature data
    X = np.random.randn(n_samples, n_features)
    
    # Generate synthetic target (returns)
    y = np.random.randn(n_samples) * 0.02  # 2% volatility
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['label_return'] = y
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    df['symbol'] = 'TEST'
    
    # Train model
    trainer = XGBoostTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train with hyperparameter tuning
    trainer.train(X_train, y_train, X_val, y_val, tune_hyperparameters=False)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test, 'TEST')
    
    # Feature importance
    importance_df = trainer.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()
