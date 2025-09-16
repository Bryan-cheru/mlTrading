"""
Institutional ML Trading System - Demo Script
Simplified version that demonstrates the complete pipeline with historical data
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from config.settings import settings
sys.path.append(str(project_root / "data-pipeline"))
sys.path.append(str(project_root / "feature-store"))
sys.path.append(str(project_root / "ml-models"))

from ingestion.market_data import MarketDataCollector, DataValidator
from feature_engineering import FeatureEngineering
from training.xgboost_trainer import ModelManager, XGBoostTrainer

# Configure logging without Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.logs_dir / 'trading_system_demo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemDemo:
    """
    Simplified demo of the institutional ML trading system
    """
    
    def __init__(self):
        self.settings = settings
        self.data_collector = None
        self.feature_engineer = FeatureEngineering()
        self.model_manager = ModelManager(self.settings.models_dir)
        self.validator = DataValidator()
        
    def collect_historical_data_for_demo(self, symbols: list, days: int = 30) -> pd.DataFrame:
        """
        Collect historical data for demonstration
        """
        logger.info(f"Collecting {days} days of historical data for demo")
        
        self.data_collector = MarketDataCollector(symbols)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            
            historical_data = self.data_collector.get_historical_data(
                symbol, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if historical_data:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': dp.timestamp,
                        'symbol': dp.symbol,
                        'open': dp.open_price,
                        'high': dp.high_price,
                        'low': dp.low_price,
                        'close': dp.close_price,
                        'volume': dp.volume,
                        'source': dp.source
                    }
                    for dp in historical_data
                ])
                
                all_data.append(df)
                logger.info(f"Collected {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data collected for {symbol}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total historical data: {len(combined_df)} records")
            return combined_df
        else:
            logger.error("No historical data collected")
            return pd.DataFrame()
    
    def run_complete_demo(self):
        """
        Run complete demo of the trading system
        """
        logger.info("INSTITUTIONAL ML TRADING SYSTEM - DEMO")
        logger.info("=" * 60)
        
        try:
            # Use fewer symbols for demo
            demo_symbols = ["AAPL", "SPY", "BTC-USD"]
            
            # Step 1: Collect historical data
            logger.info("Step 1: Collecting historical market data...")
            market_data = self.collect_historical_data_for_demo(demo_symbols, days=60)
            
            if market_data.empty:
                logger.error("No market data collected - demo cannot continue")
                return
            
            # Step 2: Engineer features
            logger.info("Step 2: Engineering features...")
            features_df = self.engineer_features(market_data)
            
            if features_df.empty:
                logger.error("No features engineered - demo cannot continue")
                return
            
            # Step 3: Train models
            logger.info("Step 3: Training ML models...")
            training_results = self.train_models(features_df)
            
            if not training_results:
                logger.error("No models trained - demo cannot continue")
                return
            
            # Step 4: Generate predictions
            logger.info("Step 4: Generating predictions...")
            predictions_df = self.generate_predictions(features_df)
            
            # Display results
            self.display_results(training_results, predictions_df)
            
            # Save results
            self.save_demo_results(training_results, predictions_df, features_df)
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw market data
        """
        logger.info("Engineering features...")
        
        if df.empty:
            logger.error("No data to engineer features from")
            return df
        
        # Group by symbol and engineer features for each
        results = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) > 50:  # Need minimum data for feature engineering
                logger.info(f"Engineering features for {symbol}")
                
                # Engineer features
                features_df = self.feature_engineer.engineer_features(symbol_df)
                
                # Create labels for supervised learning
                features_with_labels = self.feature_engineer.create_labels(
                    features_df, 
                    prediction_horizon=1  # Predict 1 period ahead for demo
                )
                
                results.append(features_with_labels)
                logger.info(f"Engineered {len(self.feature_engineer.feature_names)} features for {symbol}")
            else:
                logger.warning(f"Insufficient data for {symbol} ({len(symbol_df)} records)")
        
        if results:
            combined_features = pd.concat(results, ignore_index=True)
            logger.info(f"Total engineered dataset: {combined_features.shape}")
            return combined_features
        else:
            logger.error("No features engineered")
            return pd.DataFrame()
    
    def train_models(self, df: pd.DataFrame) -> dict:
        """
        Train ML models for each symbol
        """
        logger.info("Training ML models...")
        
        if df.empty:
            logger.error("No data to train models")
            return {}
        
        results = {}
        
        # Train model for each symbol
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Need sufficient data with labels
            valid_data = symbol_df.dropna(subset=['label_return'])
            
            if len(valid_data) > 100:  # Minimum training data
                logger.info(f"Training model for {symbol}")
                
                try:
                    # Train model
                    metrics = self.model_manager.train_symbol_model(
                        symbol, 
                        valid_data, 
                        tune_hyperparameters=False  # Set to True for better performance
                    )
                    
                    results[symbol] = metrics
                    logger.info(f"Model trained for {symbol} - Sharpe: {metrics.sharpe_ratio:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training model for {symbol}: {e}")
            else:
                logger.warning(f"Insufficient training data for {symbol} ({len(valid_data)} records)")
        
        return results
    
    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using trained models
        """
        logger.info("Generating predictions...")
        
        if df.empty:
            return df
        
        # Get latest data for each symbol
        latest_data = df.groupby('symbol').tail(1).copy()
        
        predictions = []
        for _, row in latest_data.iterrows():
            symbol = row['symbol']
            
            try:
                # Extract features
                feature_matrix, feature_names = self.feature_engineer.get_feature_matrix(
                    pd.DataFrame([row])
                )
                
                if len(feature_matrix) > 0 and symbol in self.model_manager.models:
                    # Make prediction
                    prediction = self.model_manager.predict(symbol, feature_matrix)[0]
                    
                    predictions.append({
                        'symbol': symbol,
                        'timestamp': row['timestamp'],
                        'current_price': row['close'],
                        'predicted_return': prediction,
                        'signal': 'BUY' if prediction > 0.005 else 'SELL' if prediction < -0.005 else 'HOLD'
                    })
                    
                    logger.info(f"{symbol}: {prediction:.4f} ({predictions[-1]['signal']})")
                
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")
        
        if predictions:
            return pd.DataFrame(predictions)
        else:
            return pd.DataFrame()
    
    def display_results(self, training_results: dict, predictions_df: pd.DataFrame):
        """
        Display demo results
        """
        logger.info("Demo completed successfully!")
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE SUMMARY:")
        
        for symbol, metrics in training_results.items():
            logger.info(f"  {symbol}:")
            logger.info(f"    Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            logger.info(f"    Hit Rate: {metrics.hit_rate:.3f}")
            logger.info(f"    Max Drawdown: {metrics.max_drawdown:.3f}")
            logger.info(f"    Total Return: {metrics.total_return:.3f}")
        
        if not predictions_df.empty:
            logger.info("LATEST PREDICTIONS:")
            for _, pred in predictions_df.iterrows():
                logger.info(f"  {pred['symbol']}: {pred['signal']} (Return: {pred['predicted_return']:.4f})")
    
    def save_demo_results(self, training_results: dict, predictions_df: pd.DataFrame, features_df: pd.DataFrame):
        """
        Save demo results
        """
        results_dir = self.settings.data_dir / "demo_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        if not predictions_df.empty:
            predictions_df.to_csv(results_dir / f"demo_predictions_{timestamp}.csv", index=False)
        
        # Save feature data sample
        if not features_df.empty:
            sample_features = features_df.sample(min(1000, len(features_df)))
            sample_features.to_csv(results_dir / f"demo_features_sample_{timestamp}.csv", index=False)
        
        # Save performance metrics
        metrics_summary = []
        for symbol, metrics in training_results.items():
            metrics_summary.append({
                'symbol': symbol,
                'sharpe_ratio': metrics.sharpe_ratio,
                'hit_rate': metrics.hit_rate,
                'max_drawdown': metrics.max_drawdown,
                'total_return': metrics.total_return,
                'mse': metrics.mse,
                'mae': metrics.mae,
                'r2': metrics.r2
            })
        
        if metrics_summary:
            metrics_df = pd.DataFrame(metrics_summary)
            metrics_df.to_csv(results_dir / f"demo_metrics_{timestamp}.csv", index=False)
        
        logger.info(f"Demo results saved to {results_dir}")

def main():
    """
    Main entry point for demo
    """
    print("INSTITUTIONAL ML TRADING SYSTEM - DEMO")
    print("Starting automated trading pipeline demonstration...")
    print("=" * 60)
    
    # Initialize and run demo
    demo = TradingSystemDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
