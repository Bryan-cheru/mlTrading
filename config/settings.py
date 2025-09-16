"""
Institutional ML Trading System - Configuration Settings
"""
import os
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_database: str = "trading_data"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

@dataclass
class MarketDataConfig:
    """Market data configuration settings"""
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    yahoo_finance_enabled: bool = True
    update_frequency_seconds: int = 60
    instruments: List[str] = None
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",  # Tech stocks
                "SPY", "QQQ", "IWM",  # ETFs
                "EURUSD", "GBPUSD", "USDJPY",  # Forex pairs
                "BTC-USD", "ETH-USD"  # Crypto
            ]

@dataclass
class MLConfig:
    """Machine learning configuration settings"""
    model_type: str = "xgboost"
    feature_window_days: int = 30
    prediction_horizon_minutes: int = 60
    retrain_frequency_hours: int = 24
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # XGBoost specific parameters
    xgb_params: Dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }

@dataclass
class RiskConfig:
    """Risk management configuration settings"""
    max_position_size: float = 0.05  # 5% of portfolio
    max_drawdown_threshold: float = 0.05  # 5% max drawdown
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_correlation_threshold: float = 0.7
    var_confidence_level: float = 0.95

@dataclass
class TradingConfig:
    """Trading execution configuration"""
    dry_run: bool = True  # Start in paper trading mode
    initial_capital: float = 100000.0  # $100k initial capital
    commission_rate: float = 0.001  # 0.1% commission
    slippage_bps: float = 5.0  # 5 basis points slippage
    max_orders_per_minute: int = 10

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    prometheus_port: int = 8000
    log_level: str = "INFO"
    alert_webhook_url: str = ""
    performance_check_interval_minutes: int = 15
    
class Settings:
    """Main settings class that consolidates all configuration"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.market_data = MarketDataConfig()
        self.ml = MLConfig()
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.monitoring = MonitoringConfig()
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    @property
    def performance_targets(self) -> Dict[str, float]:
        """Performance targets from project requirements"""
        return {
            "target_sharpe_ratio": 2.0,
            "max_drawdown": 0.05,
            "win_rate": 0.55,
            "annual_return_target": 0.20,
            "max_inference_latency_ms": 10,
            "uptime_target": 0.999
        }

# Global settings instance
settings = Settings()
