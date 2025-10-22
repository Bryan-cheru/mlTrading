# System Architecture Diagram

## 📊 Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INSTITUTIONAL ML TRADING SYSTEM                           │
│                         Production Architecture                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA SOURCES                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │   Rithmic    │         │ NinjaTrader  │         │   Market     │       │
│   │   R | API    │◄───────►│   Platform   │◄───────►│   Exchanges  │       │
│   │   (ES, NQ)   │         │   (Live)     │         │   (CME)      │       │
│   └──────┬───────┘         └──────┬───────┘         └──────────────┘       │
│          │                        │                                          │
│          │ Real-time              │ Market Data                             │
│          │ Tick Data              │ + Execution                             │
│          ▼                        ▼                                          │
└──────────────────────────────────────────────────────────────────────────────┘
           │                        │
           │                        │
           ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: DATA PIPELINE                                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  data_pipeline/ingestion/                                  │            │
│   │                                                              │            │
│   │  ┌───────────────────┐    ┌────────────────────┐          │            │
│   │  │ rithmic_connector │    │ninjatrader_connector│          │            │
│   │  │                   │    │                     │          │            │
│   │  │ - WebSocket       │    │ - TCP Socket        │          │            │
│   │  │ - Protocol Buffer │    │ - Command Protocol  │          │            │
│   │  │ - SSL/TLS         │    │ - Order Execution   │          │            │
│   │  └─────────┬─────────┘    └──────────┬──────────┘          │            │
│   │            │                           │                     │            │
│   │            └───────────┬───────────────┘                     │            │
│   │                        ▼                                     │            │
│   │            ┌─────────────────────┐                          │            │
│   │            │  Market Data Queue   │                          │            │
│   │            │  (Thread-Safe)       │                          │            │
│   │            └──────────┬───────────┘                          │            │
│   └───────────────────────┼──────────────────────────────────────┘            │
│                           │                                                   │
│                           ▼                                                   │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  data_pipeline/processing/                                 │            │
│   │                                                              │            │
│   │  ┌──────────────────────┐    ┌──────────────────────┐     │            │
│   │  │ market_data_processor│    │ feature_engineering  │     │            │
│   │  │                      │───►│                      │     │            │
│   │  │ - Tick aggregation   │    │ - Technical indicators│     │            │
│   │  │ - OHLC bars          │    │ - Volume analysis    │     │            │
│   │  │ - Data cleaning      │    │ - Volatility metrics │     │            │
│   │  └──────────────────────┘    └──────────┬───────────┘     │            │
│   └───────────────────────────────────────────┼─────────────────┘            │
│                                               │                               │
└───────────────────────────────────────────────┼───────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: MACHINE LEARNING PIPELINE                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  feature_store/                                             │            │
│   │                                                              │            │
│   │  ┌──────────────────────────────────────────────┐          │            │
│   │  │  Feature Engineering                          │          │            │
│   │  │                                               │          │            │
│   │  │  • returns, log_returns                      │          │            │
│   │  │  • volume_ratio, volume_sma                  │          │            │
│   │  │  • realized_vol_10, realized_vol_20          │          │            │
│   │  │  • RSI, SMA, EMA, MACD                       │          │            │
│   │  │  • body_ratio, upper/lower_shadow            │          │            │
│   │  │  • momentum_5, momentum_10                   │          │            │
│   │  │  • bollinger_position, distance_from_vwap    │          │            │
│   │  └────────────────────┬─────────────────────────┘          │            │
│   └───────────────────────┼────────────────────────────────────┘            │
│                           │                                                   │
│                           ▼                                                   │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  ml_models/                                                 │            │
│   │                                                              │            │
│   │  ┌──────────────────┐           ┌──────────────────┐      │            │
│   │  │   Training       │           │   Inference      │      │            │
│   │  │                  │           │                  │      │            │
│   │  │ XGBoost Model:   │           │ Real-time        │      │            │
│   │  │ • 3-class (B/S/H)│──────────►│ Prediction       │      │            │
│   │  │ • 200 trees      │           │ • <100ms latency │      │            │
│   │  │ • RobustScaler   │           │ • Confidence     │      │            │
│   │  │ • TimeSeriesSplit│           │ • Signal gen     │      │            │
│   │  └──────────────────┘           └────────┬─────────┘      │            │
│   └───────────────────────────────────────────┼─────────────────┘            │
│                                               │                               │
│                                               ▼                               │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  ml_trading_server.py (WebSocket Server)                   │            │
│   │                                                              │            │
│   │  ┌─────────────────────────────────────────────┐           │            │
│   │  │  Signal Generation & Distribution            │           │            │
│   │  │                                              │           │            │
│   │  │  Port: 8000                                 │           │            │
│   │  │  Protocol: WebSocket (ws://)                │           │            │
│   │  │  Frequency: 10-second intervals             │           │            │
│   │  │                                              │           │            │
│   │  │  Message Format:                            │           │            │
│   │  │  {                                           │           │            │
│   │  │    "type": "TRADING_SIGNAL",                │           │            │
│   │  │    "action": "BUY",                         │           │            │
│   │  │    "confidence": 0.76,                      │           │            │
│   │  │    "symbol": "ES",                          │           │            │
│   │  │    "features": {...}                        │           │            │
│   │  │  }                                           │           │            │
│   │  └─────────────────────┬───────────────────────┘           │            │
│   └────────────────────────┼──────────────────────────────────┘            │
│                            │                                                  │
└────────────────────────────┼──────────────────────────────────────────────────┘
                             │
                             │ WebSocket Connection
                             │ (JSON Messages)
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: TRADING ENGINE (NinjaTrader 8 AddOn)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  ModernInstitutionalAddOn.cs (C# AddOn)                    │            │
│   │                                                              │            │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────┐ │            │
│   │  │ WebSocket Client │  │  Signal Processor │  │ UI Tab   │ │            │
│   │  │                  │─►│                   │  │          │ │            │
│   │  │ - Async connect  │  │ - JSON parsing    │  │ - Charts │ │            │
│   │  │ - Auto-reconnect │  │ - Signal queue    │  │ - Metrics│ │            │
│   │  │ - Message buffer │  │ - Validation      │  │ - Controls│ │            │
│   │  └──────────────────┘  └─────────┬─────────┘  └──────────┘ │            │
│   │                                   │                          │            │
│   │                                   ▼                          │            │
│   │  ┌────────────────────────────────────────────────────┐    │            │
│   │  │  EnhancedRiskManager                               │    │            │
│   │  │                                                     │    │            │
│   │  │  ┌──────────────────┐  ┌───────────────────────┐  │    │            │
│   │  │  │ Circuit Breaker  │  │ Position Limits       │  │    │            │
│   │  │  │                  │  │                       │  │    │            │
│   │  │  │ - PnL monitoring │  │ - Max position: 2    │  │    │            │
│   │  │  │ - Trade velocity │  │ - Daily trades: 20   │  │    │            │
│   │  │  │ - Cool-down      │  │ - Daily loss: 5%     │  │    │            │
│   │  │  │ - Auto-halt      │  │ - Confidence: 70%    │  │    │            │
│   │  │  └──────────────────┘  └───────────────────────┘  │    │            │
│   │  └──────────────────────────┬─────────────────────────┘    │            │
│   │                             │                               │            │
│   │                             ▼                               │            │
│   │  ┌────────────────────────────────────────────────────┐    │            │
│   │  │  ModernInstitutionalTradingEngine                  │    │            │
│   │  │                                                     │    │            │
│   │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │    │            │
│   │  │  │Order Executor│  │Position Track│  │PnL Track│  │    │            │
│   │  │  │              │  │              │  │         │  │    │            │
│   │  │  │ - Market ord │  │ - Real-time  │  │ - Live  │  │    │            │
│   │  │  │ - Limit ord  │  │ - Multi-inst │  │ - Daily │  │    │            │
│   │  │  │ - Stop ord   │  │ - Exit logic │  │ - Total │  │    │            │
│   │  │  └──────────────┘  └──────────────┘  └─────────┘  │    │            │
│   │  └────────────────────────────────────────────────────┘    │            │
│   └───────────────────────────┬────────────────────────────────┘            │
│                               │                                              │
│                               ▼                                              │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  NinjaTrader 8 Order API                                   │            │
│   │  - SubmitOrderAsync()                                      │            │
│   │  - CancelOrderAsync()                                      │            │
│   │  - Account.GetAccountValue()                               │            │
│   └─────────────────────────┬──────────────────────────────────┘            │
│                             │                                                │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              │
                              │ Live Orders
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: EXECUTION & MONITORING                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │  Live Market     │◄───┤  NinjaTrader 8   ├───►│  Brokerage       │     │
│   │  Execution       │    │  Platform        │    │  (Simulation)    │     │
│   │                  │    │                  │    │                  │     │
│   │  - ES Futures    │    │  - Sim101        │    │  - Fill confirm  │     │
│   │  - Real fills    │    │  - Real-time P&L │    │  - Position mgmt │     │
│   │  - Low latency   │    │  - Charts        │    │  - Account data  │     │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘     │
│                                                                               │
│   ┌────────────────────────────────────────────────────────────┐            │
│   │  monitoring/ (Performance Tracking)                         │            │
│   │                                                              │            │
│   │  ┌──────────────────────┐    ┌──────────────────────┐     │            │
│   │  │ Performance Monitor  │    │  Risk Monitor        │     │            │
│   │  │                      │    │                      │     │            │
│   │  │ - Sharpe ratio       │    │ - VaR calculation   │     │            │
│   │  │ - Max drawdown       │    │ - Exposure tracking │     │            │
│   │  │ - Win rate           │    │ - Correlation check │     │            │
│   │  │ - Profit factor      │    │ - Limit monitoring  │     │            │
│   │  └──────────────────────┘    └──────────────────────┘     │            │
│   └────────────────────────────────────────────────────────────┘            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  DATA FLOW DIAGRAM                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Market Data ──► Rithmic ──► Data Pipeline ──► Feature Store ──►            │
│                                                                               │
│  ──► ML Model ──► Signals ──► WebSocket ──► NinjaTrader AddOn ──►           │
│                                                                               │
│  ──► Risk Check ──► Order Execution ──► Live Market ──► P&L Tracking         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  KEY PERFORMANCE METRICS                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Latency Requirements:                                                       │
│  • Market Data → ML Model: <50ms                                            │
│  • ML Model Inference: <100ms                                               │
│  • Signal → Order Execution: <200ms                                         │
│  • Total End-to-End: <500ms                                                 │
│                                                                               │
│  Risk Limits:                                                                │
│  • Max Position Size: 2 contracts                                           │
│  • Daily Trade Limit: 20 trades                                             │
│  • Daily Loss Limit: 5% of capital                                          │
│  • Min Signal Confidence: 70%                                               │
│  • Stop Loss: 1% per trade                                                  │
│  • Take Profit: 2% per trade                                                │
│                                                                               │
│  Target Performance:                                                         │
│  • Sharpe Ratio: >2.0                                                       │
│  • Max Drawdown: <5%                                                        │
│  • Win Rate: >55%                                                           │
│  • Profit Factor: >1.5                                                      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  TECHNOLOGY STACK                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Languages:                                                                  │
│  • Python 3.13+ (ML, data pipeline, server)                                │
│  • C# .NET 4.7.2 (NinjaTrader AddOn)                                       │
│                                                                               │
│  Key Libraries:                                                              │
│  • XGBoost 2.0+ (Machine Learning)                                          │
│  • scikit-learn 1.4+ (Preprocessing)                                        │
│  • pandas 2.0+ (Data manipulation)                                          │
│  • websockets 11.0+ (Real-time communication)                               │
│  • Newtonsoft.Json (C# JSON parsing)                                        │
│                                                                               │
│  Platforms:                                                                  │
│  • NinjaTrader 8 (Trading platform)                                         │
│  • Rithmic R|API 13.6+ (Market data)                                       │
│  • Windows 10/11 (Operating system)                                         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

```

## 📈 System State Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   SYSTEM STATES                              │
└──────────────────────────────────────────────────────────────┘

         ┌─────────────┐
         │  STARTING   │
         └──────┬──────┘
                │
                │ Initialize Components
                ▼
         ┌─────────────┐
         │ CONNECTING  │
         └──────┬──────┘
                │
                │ WebSocket + Rithmic
                ▼
         ┌─────────────┐
         │  CONNECTED  │
         └──────┬──────┘
                │
                │ Load ML Model
                ▼
         ┌─────────────┐
         │   READY     │◄──────────┐
         └──────┬──────┘           │
                │                  │
                │ Start Trading    │ Pause
                ▼                  │
         ┌─────────────┐           │
         │   TRADING   ├───────────┘
         └──────┬──────┘
                │
                │ Risk Violation
                ▼
         ┌─────────────┐
         │ CIRCUIT     │
         │ BREAKER     │
         └──────┬──────┘
                │
                │ Cool-down Period
                ▼
         ┌─────────────┐
         │  RECOVERY   │
         └──────┬──────┘
                │
                │ Manual Reset / Auto Resume
                └────────►[READY]

    Emergency Stop ──► [STOPPED]
                         │
                         │ Cleanup
                         ▼
                    [SHUTDOWN]
```

## 🔄 Trade Execution Flow

```
┌──────────────────────────────────────────────────────────────┐
│              TRADE EXECUTION SEQUENCE                        │
└──────────────────────────────────────────────────────────────┘

1. Market Data Received
   ↓
2. Feature Calculation (13+ features)
   ↓
3. ML Model Prediction
   ├─ Action: BUY/SELL/HOLD
   └─ Confidence: 0.00-1.00
   ↓
4. Signal Validation
   ├─ Confidence >= 0.70?
   ├─ Daily trades < 20?
   ├─ Position size < 2?
   └─ Circuit breaker OK?
   ↓
5. Risk Check
   ├─ VaR within limits?
   ├─ Correlation check
   ├─ Exposure check
   └─ Stop loss calculation
   ↓
6. Order Submission
   ├─ Market/Limit/Stop order
   ├─ Account: Sim101
   └─ Instrument: ES
   ↓
7. Order Confirmation
   ├─ Fill price
   ├─ Fill time
   └─ Order ID
   ↓
8. Position Tracking
   ├─ Update position
   ├─ Calculate P&L
   └─ Monitor exits
   ↓
9. Performance Logging
   ├─ Trade record
   ├─ Metrics update
   └─ Dashboard refresh

Total Time: <500ms (target)
```

---

**Architecture Version**: 1.0  
**Last Updated**: October 22, 2025  
**Status**: Production-Ready (with minor fixes needed)
