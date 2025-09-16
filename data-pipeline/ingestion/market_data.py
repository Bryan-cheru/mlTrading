"""
Market Data Ingestion Module
Real-time and historical market data collection from multiple sources
"""
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Standardized market data point structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: str

class MarketDataCollector:
    """
    Unified market data collector supporting multiple data sources
    """
    
    def __init__(self, instruments: List[str], alpha_vantage_key: str = ""):
        self.instruments = instruments
        self.alpha_vantage_key = alpha_vantage_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_yahoo_finance_data(self, symbol: str, period: str = "1d") -> List[MarketDataPoint]:
        """
        Fetch data from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1m")
            
            data_points = []
            for timestamp, row in hist.iterrows():
                if not pd.isna(row['Close']):
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume']),
                        source="yahoo_finance"
                    )
                    data_points.append(data_point)
            
            logger.info(f"Collected {len(data_points)} data points for {symbol} from Yahoo Finance")
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return []
    
    async def get_alpha_vantage_data(self, symbol: str) -> List[MarketDataPoint]:
        """
        Fetch data from Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided")
            return []
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "1min",
                "apikey": self.alpha_vantage_key,
                "outputsize": "compact"
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if "Time Series (1min)" not in data:
                    logger.error(f"Invalid Alpha Vantage response for {symbol}: {data}")
                    return []
                
                time_series = data["Time Series (1min)"]
                data_points = []
                
                for timestamp_str, ohlcv in time_series.items():
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(ohlcv["1. open"]),
                        high_price=float(ohlcv["2. high"]),
                        low_price=float(ohlcv["3. low"]),
                        close_price=float(ohlcv["4. close"]),
                        volume=int(ohlcv["5. volume"]),
                        source="alpha_vantage"
                    )
                    data_points.append(data_point)
                
                logger.info(f"Collected {len(data_points)} data points for {symbol} from Alpha Vantage")
                return data_points
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return []
    
    async def collect_realtime_data(self) -> Dict[str, List[MarketDataPoint]]:
        """
        Collect real-time data for all instruments from available sources
        """
        all_data = {}
        
        # Collect data for each instrument
        for symbol in self.instruments:
            logger.info(f"Collecting data for {symbol}")
            
            # Try Yahoo Finance first (free and reliable)
            yahoo_data = await self.get_yahoo_finance_data(symbol, period="1d")
            if yahoo_data:
                all_data[symbol] = yahoo_data
            else:
                # Fallback to Alpha Vantage if available
                alpha_data = await self.get_alpha_vantage_data(symbol)
                if alpha_data:
                    all_data[symbol] = alpha_data
                else:
                    logger.warning(f"No data collected for {symbol}")
        
        return all_data
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[MarketDataPoint]:
        """
        Get historical data for backtesting and training
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval="1h")
            
            data_points = []
            for timestamp, row in hist.iterrows():
                if not pd.isna(row['Close']):
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume']),
                        source="yahoo_finance_historical"
                    )
                    data_points.append(data_point)
            
            logger.info(f"Collected {len(data_points)} historical data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

class DataValidator:
    """
    Validates incoming market data for quality and consistency
    """
    
    @staticmethod
    def validate_data_point(data_point: MarketDataPoint) -> bool:
        """
        Validate a single market data point
        """
        checks = [
            data_point.open_price > 0,
            data_point.high_price > 0,
            data_point.low_price > 0,
            data_point.close_price > 0,
            data_point.volume >= 0,
            data_point.high_price >= data_point.low_price,
            data_point.high_price >= max(data_point.open_price, data_point.close_price),
            data_point.low_price <= min(data_point.open_price, data_point.close_price),
        ]
        
        return all(checks)
    
    @staticmethod
    def detect_outliers(data_points: List[MarketDataPoint], z_threshold: float = 3.0) -> List[bool]:
        """
        Detect outliers using z-score method
        """
        if len(data_points) < 10:  # Need minimum data points for outlier detection
            return [False] * len(data_points)
        
        prices = [dp.close_price for dp in data_points]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        
        if len(returns) == 0:
            return [False] * len(data_points)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        outliers = [False]  # First point can't be an outlier
        for i, ret in enumerate(returns):
            z_score = abs(ret - mean_return) / std_return if std_return > 0 else 0
            outliers.append(z_score > z_threshold)
        
        return outliers
    
    def clean_data(self, data_points: List[MarketDataPoint]) -> List[MarketDataPoint]:
        """
        Clean and validate market data
        """
        if not data_points:
            return []
        
        # Basic validation
        valid_points = [dp for dp in data_points if self.validate_data_point(dp)]
        logger.info(f"Filtered {len(data_points) - len(valid_points)} invalid data points")
        
        # Outlier detection
        outlier_flags = self.detect_outliers(valid_points)
        clean_points = [dp for dp, is_outlier in zip(valid_points, outlier_flags) if not is_outlier]
        logger.info(f"Removed {len(valid_points) - len(clean_points)} outlier data points")
        
        return clean_points

async def main():
    """
    Example usage of the market data collector
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example instruments
    instruments = ["AAPL", "MSFT", "SPY", "BTC-USD"]
    
    # Create data collector
    async with MarketDataCollector(instruments) as collector:
        # Collect real-time data
        data = await collector.collect_realtime_data()
        
        # Validate and clean data
        validator = DataValidator()
        
        for symbol, data_points in data.items():
            clean_data = validator.clean_data(data_points)
            print(f"{symbol}: {len(clean_data)} clean data points")
            
            if clean_data:
                latest = clean_data[-1]
                print(f"  Latest: {latest.close_price} at {latest.timestamp}")

if __name__ == "__main__":
    asyncio.run(main())
