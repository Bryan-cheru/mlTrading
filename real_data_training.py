#!/usr/bin/env python3
"""
Real Market Data ML Training - Professional Implementation
Uses actual ES futures data instead of synthetic data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RealMarketDataTraining:
    def __init__(self):
        self.symbols = {
            'ES': 'ES=F',  # E-mini S&P 500 futures
            'NQ': 'NQ=F',  # E-mini NASDAQ futures  
            'YM': 'YM=F',  # E-mini Dow futures
            'RTY': 'RTY=F' # E-mini Russell 2000 futures
        }
    
    def download_real_futures_data(self, symbol='ES', period='2y'):
        """Download real futures market data"""
        logger.info(f"Downloading real {symbol} futures data...")
        
        try:
            ticker = self.symbols.get(symbol, 'ES=F')
            
            # Download real market data
            data = yf.download(
                ticker,
                period=period,  # 2 years of data
                interval='1m',   # 1-minute bars
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data found for {ticker}, trying alternative...")
                # Try SPY as proxy for ES
                data = yf.download('SPY', period=period, interval='1m', progress=False)
            
            # Clean and prepare data
            data = data.dropna()
            data.reset_index(inplace=True)
            
            # Rename columns to match our format
            data.columns = data.columns.str.lower()
            if 'datetime' in data.columns:
                data.rename(columns={'datetime': 'timestamp'}, inplace=True)
            elif 'date' in data.columns:
                data.rename(columns={'date': 'timestamp'}, inplace=True)
            
            logger.info(f"✅ Downloaded {len(data)} real market bars for {symbol}")
            logger.info(f"Data range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to download real data: {e}")
            return None
    
    def create_realistic_features(self, data):
        """Create features from real market data"""
        logger.info("Creating features from real market microstructure...")
        
        # Real technical indicators
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Volatility measures
        data['returns'] = data['close'].pct_change()
        data['volatility_20'] = data['returns'].rolling(20).std() * np.sqrt(252 * 390)  # Annualized
        data['atr'] = self.calculate_atr(data)
        
        # Market microstructure
        data['spread'] = data['high'] - data['low']
        data['body_size'] = np.abs(data['close'] - data['open'])
        data['upper_wick'] = data['high'] - np.maximum(data['open'], data['close'])
        data['lower_wick'] = np.minimum(data['open'], data['close']) - data['low']
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Time-based features (real market hours matter)
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['minute'] = pd.to_datetime(data['timestamp']).dt.minute
        data['is_open'] = ((data['hour'] >= 9) & (data['hour'] < 16)).astype(int)  # Market hours
        data['is_close'] = ((data['hour'] == 15) & (data['minute'] >= 50)).astype(int)  # Close proximity
        
        # Real regime detection
        data['volatility_regime'] = (data['volatility_20'] > data['volatility_20'].rolling(100).quantile(0.8)).astype(int)
        data['trend_strength'] = np.abs(data['sma_10'] - data['sma_20']) / data['close']
        
        return data
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def create_realistic_labels(self, data, horizon=5):
        """Create labels based on real forward returns"""
        # Future return over next N minutes
        data['future_return'] = data['close'].shift(-horizon) / data['close'] - 1
        
        # Dynamic thresholds based on real volatility
        volatility = data['returns'].rolling(100).std()
        buy_threshold = volatility * 1.5   # 1.5x recent volatility
        sell_threshold = -volatility * 1.5
        
        # Labels: 0=SELL, 1=HOLD, 2=BUY
        data['label'] = 1  # Default HOLD
        data.loc[data['future_return'] > buy_threshold, 'label'] = 2   # BUY
        data.loc[data['future_return'] < sell_threshold, 'label'] = 0  # SELL
        
        return data

def train_with_real_data():
    """Train ML model with real market data"""
    logger.info("🏛️ INSTITUTIONAL ML TRAINING - REAL MARKET DATA")
    logger.info("=" * 60)
    
    trainer = RealMarketDataTraining()
    
    # Download real ES futures data
    data = trainer.download_real_futures_data('ES', period='1y')
    
    if data is not None:
        # Create realistic features
        data = trainer.create_realistic_features(data)
        data = trainer.create_realistic_labels(data)
        
        # Remove NaN values
        data = data.dropna()
        
        logger.info(f"✅ Training dataset ready: {len(data)} samples")
        logger.info(f"📊 Label distribution:")
        logger.info(f"   SELL: {(data['label'] == 0).sum()}")
        logger.info(f"   HOLD: {(data['label'] == 1).sum()}")
        logger.info(f"   BUY:  {(data['label'] == 2).sum()}")
        
        return data
    else:
        logger.error("❌ Failed to get real market data")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    real_data = train_with_real_data()