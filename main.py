"""
Institutional ML Trading System - Main Execution Script
This script demonstrates the complete pipeline from data ingestion to model training
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.logs_dir / 'trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemPipeline:
    """
    Main pipeline orchestrator for the institutional ML trading system
    """
    
    def __init__(self):
        self.settings = settings
        self.data_collector = None
        self.feature_engineer = FeatureEngineering()
        self.model_manager = ModelManager(self.settings.models_dir)
        self.validator = DataValidator()
        
    async def collect_market_data(self) -> pd.DataFrame:
        """
        Step 1: Collect market data for all instruments
        """
        logger.info("🚀 Starting market data collection...")
        
        # Create data collector
        self.data_collector = MarketDataCollector(
            instruments=self.settings.market_data.instruments,
            alpha_vantage_key=self.settings.market_data.alpha_vantage_api_key
        )
        
        # Collect real-time data
        async with self.data_collector as collector:
            market_data = await collector.collect_realtime_data()
        
        # Convert to standardized DataFrame
        all_data = []
        for symbol, data_points in market_data.items():
            if data_points:
                # Clean data
                clean_data = self.validator.clean_data(data_points)
                
                if clean_data:
                    # Convert to DataFrame format
                    symbol_df = pd.DataFrame([
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
                        for dp in clean_data
                    ])
                    all_data.append(symbol_df)
                    logger.info(f"✅ Collected {len(clean_data)} data points for {symbol}")
                else:
                    logger.warning(f"⚠️ No clean data for {symbol}")
            else:
                logger.warning(f"⚠️ No data collected for {symbol}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"📊 Total market data collected: {len(combined_df)} records")
            return combined_df
        else:
            logger.error("❌ No market data collected")
            return pd.DataFrame()
    
    def collect_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Collect historical data for training
        """
        logger.info(f"📈 Collecting {days} days of historical data for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if self.data_collector is None:
            self.data_collector = MarketDataCollector([symbol])
        
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
            
            logger.info(f"✅ Collected {len(df)} historical records for {symbol}")
            return df
        else:
            logger.error(f"❌ No historical data collected for {symbol}")
            return pd.DataFrame()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Engineer features from raw market data
        """
        logger.info("🔧 Engineering features...")
        
        if df.empty:
            logger.error("❌ No data to engineer features from")
            return df
        
        # Group by symbol and engineer features for each
        results = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) > 50:  # Need minimum data for feature engineering
                logger.info(f"🛠️ Engineering features for {symbol}")
                
                # Engineer features
                features_df = self.feature_engineer.engineer_features(symbol_df)
                
                # Create labels for supervised learning
                features_with_labels = self.feature_engineer.create_labels(
                    features_df, 
                    prediction_horizon=self.settings.ml.prediction_horizon_minutes
                )
                
                results.append(features_with_labels)
                logger.info(f"✅ Engineered {len(self.feature_engineer.feature_names)} features for {symbol}")
            else:
                logger.warning(f"⚠️ Insufficient data for {symbol} ({len(symbol_df)} records)")
        
        if results:
            combined_features = pd.concat(results, ignore_index=True)
            logger.info(f"🎯 Total engineered dataset: {combined_features.shape}")
            return combined_features
        else:
            logger.error("❌ No features engineered")
            return pd.DataFrame()
    
    def train_models(self, df: pd.DataFrame) -> dict:
        """
        Step 3: Train ML models for each symbol
        """
        logger.info("🤖 Training ML models...")
        
        if df.empty:
            logger.error("❌ No data to train models")
            return {}
        
        results = {}
        
        # Train model for each symbol
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Need sufficient data with labels
            valid_data = symbol_df.dropna(subset=['label_return'])
            
            if len(valid_data) > 100:  # Minimum training data
                logger.info(f"🎓 Training model for {symbol}")
                
                try:
                    # Train model
                    metrics = self.model_manager.train_symbol_model(
                        symbol, 
                        valid_data, 
                        tune_hyperparameters=False  # Set to True for better performance
                    )
                    
                    results[symbol] = metrics
                    logger.info(f"✅ Model trained for {symbol} - Sharpe: {metrics.sharpe_ratio:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training model for {symbol}: {e}")
            else:
                logger.warning(f"⚠️ Insufficient training data for {symbol} ({len(valid_data)} records)")
        
        return results
    
    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Generate predictions using trained models
        """
        logger.info("🔮 Generating predictions...")
        
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
                    
                    logger.info(f"📊 {symbol}: {prediction:.4f} ({predictions[-1]['signal']})")
                
            except Exception as e:
                logger.error(f"❌ Error generating prediction for {symbol}: {e}")
        
        if predictions:
            return pd.DataFrame(predictions)
        else:
            return pd.DataFrame()
    
    async def run_full_pipeline(self):
        """
        Run the complete trading system pipeline
        """
        logger.info("🎬 Starting Institutional ML Trading System Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Collect market data
            market_data = await self.collect_market_data()
            
            if market_data.empty:
                logger.error("❌ Pipeline failed: No market data collected")
                return
            
            # Step 2: Collect additional historical data for training
            logger.info("📚 Collecting historical data for model training...")
            historical_datasets = []
            
            for symbol in self.settings.market_data.instruments[:3]:  # Limit to first 3 for demo
                hist_data = self.collect_historical_data(symbol, days=7)  # Last 7 days
                if not hist_data.empty:
                    historical_datasets.append(hist_data)
            
            if historical_datasets:
                historical_data = pd.concat(historical_datasets, ignore_index=True)
                combined_data = pd.concat([historical_data, market_data], ignore_index=True)
            else:
                combined_data = market_data
            
            # Step 3: Engineer features
            features_df = self.engineer_features(combined_data)
            
            if features_df.empty:
                logger.error("❌ Pipeline failed: No features engineered")
                return
            
            # Step 4: Train models
            training_results = self.train_models(features_df)
            
            if not training_results:
                logger.error("❌ Pipeline failed: No models trained")
                return
            
            # Step 5: Generate predictions
            predictions_df = self.generate_predictions(features_df)
            
            # Display results
            logger.info("🎉 Pipeline completed successfully!")
            logger.info("=" * 60)
            logger.info("📈 MODEL PERFORMANCE SUMMARY:")
            
            for symbol, metrics in training_results.items():
                logger.info(f"  {symbol}:")
                logger.info(f"    Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
                logger.info(f"    Hit Rate: {metrics.hit_rate:.3f}")
                logger.info(f"    Max Drawdown: {metrics.max_drawdown:.3f}")
                logger.info(f"    Total Return: {metrics.total_return:.3f}")
            
            if not predictions_df.empty:
                logger.info("🔮 LATEST PREDICTIONS:")
                for _, pred in predictions_df.iterrows():
                    logger.info(f"  {pred['symbol']}: {pred['signal']} (Return: {pred['predicted_return']:.4f})")
            
            # Save results
            results_dir = self.settings.data_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not predictions_df.empty:
                predictions_df.to_csv(results_dir / f"predictions_{timestamp}.csv", index=False)
            
            logger.info(f"💾 Results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e}")
            raise

async def main():
    """
    Main entry point
    """
    print("🏛️  INSTITUTIONAL ML TRADING SYSTEM")
    print("🚀 Starting automated trading pipeline...")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = TradingSystemPipeline()
    await pipeline.run_full_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
