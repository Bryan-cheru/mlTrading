# Institutional ML Trading System - AI Agent Instructions

## Architecture Overview

This is an **institutional-grade ML trading system** with **NinjaTrader 8 integration**. The system follows a 4-tier architecture:

1. **Data Layer**: `data-pipeline/` - Real-time market data ingestion via NinjaTrader 8 socket connections
2. **ML Pipeline**: `ml-models/` - XGBoost-based models with <10ms inference latency requirements  
3. **Trading Layer**: `trading-engine/` - Order execution through NinjaTrader 8 API
4. **Monitoring**: `monitoring/`, `risk-engine/` - Real-time performance tracking and risk management

## Critical Integration Patterns

### NinjaTrader 8 Socket Communication
- **All NinjaTrader connections use TCP sockets on port 36973**
- Use `ninjatrader_connector.py` for market data subscriptions and order execution
- Message format: JSON with specific NinjaTrader command structures
- **Always handle connection timeouts** - NinjaTrader connections are fragile

### Real-Time Data Flow
```
NinjaTrader → ninjatrader_connector.py → realtime_data_manager.py → feature_engineering.py → trading_model.py → ninjatrader_executor.py
```

### Performance Requirements
- **Model inference latency: <10ms** (use pre-computed features when possible)
- **Target Sharpe ratio: >2.0, Max drawdown: <5%**
- Use `RobustScaler` not `StandardScaler` for financial data robustness

## Project-Specific Conventions

### Module Structure
- Use `sys.path.append()` pattern in main files for relative imports
- All modules have dedicated `__init__.py` files for clean imports
- Configuration centralized in `config/settings.py` and `config/system_config.json`

### Data Handling
- **Always use `pd.Timestamp` for datetime handling** (financial data standard)
- Market data uses `MarketData` and `BarData` dataclasses for type safety
- Features stored as `Dict[str, float]` in `FeatureSet` containers

### Risk Management Integration
- **Every trading signal must pass `risk_manager.check_signal_risk()`**
- Use `AdvancedRiskManager` for institutional-grade risk controls
- Log all risk violations to `violations` list for compliance

## Development Workflows

### Running the System
```bash
# Activate virtual environment first
& "venv/Scripts/Activate.ps1"

# Main system (requires NinjaTrader 8 running)
python main_trading_system.py

# Demo mode (synthetic data for testing)
python ninjatrader_demo.py
```

### Python Environment
- **Always use the venv**: `"C:/Users/Brian Cheruiyot/Desktop/InstitutionalMLTrading/venv/Scripts/python.exe"`
- Compatible with Python 3.13+ and NinjaTrader 8 .NET integration

### Configuration Management
- System config in `config/system_config.json` for NinjaTrader settings
- ML parameters in `config/settings.py` dataclasses
- **Never hardcode API keys** - use environment variables

## Key Components to Understand

### `main_trading_system.py`
- **Central orchestrator** - initializes all components and manages main trading loop
- Handles graceful shutdown with signal handlers
- Uses ThreadPoolExecutor for concurrent operations

### `data-pipeline/ingestion/ninjatrader_connector.py`
- **Critical for NinjaTrader integration** - all market data flows through here
- Implements retry logic and connection management
- Uses threading for non-blocking socket operations

### `ml-models/training/trading_model.py`
- XGBoost model with financial-specific preprocessing
- **Uses TimeSeriesSplit for backtesting** (never use random splits with time series)
- Implements real-time prediction caching for performance

### `feature-store/feature_engineering.py`
- Technical indicators using TA-Lib library
- **All features must be computed incrementally** for real-time performance
- Uses sliding windows for efficient computation

## Financial Domain Knowledge

### Futures Trading Focus
- Primary instruments: ES (E-mini S&P 500), NQ (NASDAQ), etc.
- **Tick sizes and values are instrument-specific** - reference `system_config.json`
- Margin requirements enforced in risk management

### Performance Metrics
- Use Sharpe ratio, max drawdown, win rate for model evaluation
- **Never use standard ML metrics alone** - financial performance is different
- All performance tracking in `PerformanceMonitor` class

## Testing and Debugging

### Error Handling Patterns
- Socket connection errors are expected - implement retry logic
- **Always log to both file and console** using the configured logger
- Use `try/except` blocks around all NinjaTrader API calls

### Common Issues
- **NinjaTrader connection timeouts**: Check if NT8 is running and accepting connections
- **Import path issues**: Use the `sys.path.append()` pattern consistently
- **Data latency**: Pre-compute features where possible for <10ms inference

## Production-Ready Implementation Requirements

### No Synthetic or Simulated Data
- **NEVER create or use synthetic data** in any implementation
- **NEVER use simulated market data** for testing or development
- **Always integrate with real market data sources** (NinjaTrader 8, live feeds)
- **Production-ready only** - all code must work with live market conditions
- Use historical real data for backtesting, never generated/synthetic datasets

### Real Data Sources Only
- **NinjaTrader 8**: Primary real-time data and execution platform
- **Live market feeds**: Alpha Vantage, Interactive Brokers, TD Ameritrade APIs
- **Historical real data**: Yahoo Finance, Quandl for backtesting
- **Real news feeds**: Bloomberg, Reuters APIs for sentiment analysis

## External Dependencies

- **NinjaTrader 8**: Required for live trading, uses .NET integration
- **XGBoost**: Primary ML framework, optimized for financial time series
- **TA-Lib**: Technical analysis library for indicators
- **Real market data**: Only real market data - never synthetic or simulated

When working on this codebase, prioritize real-time performance, NinjaTrader compatibility, and institutional-grade risk management over generic ML best practices. All implementations must be production-ready and work with live market data.
