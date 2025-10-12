# ES Trading System - Clean Production Codebase

## Quick Start
```bash
# Setup (one-time)
setup_client.bat

# Launch system
start_system.bat
```

## Core Files

### Trading System
- `enhanced_trading_ui.py` - Main trading dashboard with training progress
- `launch_production.py` - Production server launcher
- `production_server.py` - Web/mobile API server

### ML Pipeline
- `gpu_transformer_trainer.py` - GPU-optimized transformer training
- `train_ultimate_transformer.py` - High-performance training coordinator
- `ml-models/training/trading_model.py` - Core ML models

### NinjaTrader Integration
- `ninjatrader-addon/ESMLTradingSystemMain.cs` - Live trading AddOn
- `ninjatrader_addon_interface.py` - Python interface

### Data & Risk
- `data-pipeline/` - Market data ingestion
- `feature-store/` - Real-time feature engineering  
- `risk-engine/` - Risk management
- `trading-engine/` - Order execution

### Utilities
- `create_mobile_app.py` - Mobile dashboard generator
- `config/` - System configuration
- `models/` - Trained model storage

## Architecture
```
NinjaTrader ← → Python ML ← → Web API ← → Mobile Dashboard
     ↓              ↓           ↓            ↓
Live Trading   AI Signals   Database    Remote Access
```

## Usage
1. Development: Use enhanced UI for training and testing
2. Production: Use production server for live trading
3. Mobile: Access via web dashboard for monitoring