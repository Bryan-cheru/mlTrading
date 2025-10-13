"""
Python Controller for Institutional NinjaTrader Integration
Connects institutional statistical arbitrage strategy with NinjaTrader execution
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from data_pipeline.ingestion.rithmic_connector import RithmicConnector
from trading_engine.ninjatrader_executor import NinjaTraderExecutor
from risk_engine.advanced_risk_manager import AdvancedRiskManager
from feature_store.realtime.institutional_strategy_core import StatisticalArbitrageEngine
from monitoring.performance_monitor import PerformanceMonitor
from config.settings import SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/institutional_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PositionUpdate:
    """Position update from NinjaTrader"""
    symbol: str
    quantity: int
    price: float
    pnl: float
    timestamp: datetime

@dataclass
class ExecutionReport:
    """Trade execution report"""
    pair_name: str
    signal_type: str
    symbol_a: str
    symbol_b: str
    quantity_a: int
    quantity_b: int
    price_a: float
    price_b: float
    z_score: float
    confidence: float
    timestamp: datetime

class InstitutionalController:
    """
    Main controller for institutional statistical arbitrage system
    Bridges Python ML/stats with NinjaTrader execution
    """
    
    def __init__(self, config_path: str = "config/system_config.json"):
        """Initialize institutional trading controller"""
        self.config = self._load_config(config_path)
        
        # === CORE COMPONENTS ===
        self.rithmic_connector = None
        self.ninjatrader_executor = None
        self.arbitrage_engine = None
        self.risk_manager = None
        self.performance_monitor = None
        
        # === STATE TRACKING ===
        self.active_pairs: Dict[str, dict] = {}
        self.market_data: Dict[str, dict] = {}
        self.last_update_time: Dict[str, datetime] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # === CONTROL FLAGS ===
        self.is_running = False
        self.is_trading_enabled = True
        self.last_risk_check = datetime.now()
        
        # === PERFORMANCE TRACKING ===
        self.daily_pnl = 0.0
        self.total_signals = 0
        self.executed_trades = 0
        self.risk_violations = []
        
        logger.info("🏛️ Institutional Controller initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Add institutional-specific settings
            config.update({
                "pairs_config": {
                    "ES_NQ": {
                        "symbols": ["ES 12-25", "NQ 12-25"],
                        "lookback": 300,
                        "entry_z_score": 2.0,
                        "exit_z_score": 0.3,
                        "max_holding_minutes": 480,
                        "vol_filter": 0.02
                    },
                    "ZN_ZB": {
                        "symbols": ["ZN 12-25", "ZB 12-25"],
                        "lookback": 240,
                        "entry_z_score": 1.8,
                        "exit_z_score": 0.25,
                        "max_holding_minutes": 720,
                        "vol_filter": 0.015
                    }
                },
                "risk_limits": {
                    "max_daily_loss_pct": 2.0,
                    "max_concurrent_pairs": 4,
                    "position_size": 1,
                    "confidence_threshold": 0.6
                },
                "trading_hours": {
                    "start_hour": 6,
                    "end_hour": 21,
                    "enable_news_blackout": True
                }
            })
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("🔧 Initializing institutional trading system...")
            
            # === DATA CONNECTION ===
            self.rithmic_connector = RithmicConnector(
                username=self.config.get("rithmic_username"),
                password=self.config.get("rithmic_password"),
                gateway=self.config.get("rithmic_gateway", "Rithmic Paper Trading")
            )
            
            # === TRADING EXECUTION ===
            self.ninjatrader_executor = NinjaTraderExecutor(
                host=self.config.get("ninjatrader_host", "localhost"),
                port=self.config.get("ninjatrader_port", 36973)
            )
            
            # === STATISTICAL ARBITRAGE ENGINE ===
            self.arbitrage_engine = StatisticalArbitrageEngine()
            
            # === RISK MANAGEMENT ===
            self.risk_manager = AdvancedRiskManager(
                max_position_size=self.config["risk_limits"]["position_size"] * 10,
                max_daily_loss=self.config["risk_limits"]["max_daily_loss_pct"] / 100.0,
                volatility_threshold=0.02
            )
            
            # === PERFORMANCE MONITORING ===
            self.performance_monitor = PerformanceMonitor()
            
            # === CONNECT TO DATA SOURCES ===
            await self.rithmic_connector.connect()
            await self.ninjatrader_executor.connect()
            
            # === SUBSCRIBE TO PAIRS DATA ===
            await self._subscribe_to_pairs()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _subscribe_to_pairs(self):
        """Subscribe to market data for all trading pairs"""
        try:
            for pair_name, config in self.config["pairs_config"].items():
                symbols = config["symbols"]
                
                for symbol in symbols:
                    await self.rithmic_connector.subscribe_market_data(
                        symbol=symbol,
                        callback=self._on_market_data_update
                    )
                    
                    # Initialize tracking
                    self.market_data[symbol] = {
                        "price": 0.0,
                        "volume": 0,
                        "bid": 0.0,
                        "ask": 0.0,
                        "last_update": datetime.now()
                    }
                    
                logger.info(f"📊 Subscribed to {pair_name}: {symbols}")
                
        except Exception as e:
            logger.error(f"❌ Subscription failed: {e}")
            raise
    
    async def _on_market_data_update(self, symbol: str, data: dict):
        """Handle real-time market data updates"""
        try:
            # === UPDATE MARKET DATA ===
            self.market_data[symbol].update({
                "price": data.get("last_price", 0.0),
                "volume": data.get("volume", 0),
                "bid": data.get("bid", 0.0),
                "ask": data.get("ask", 0.0),
                "last_update": datetime.now()
            })
            
            self.last_update_time[symbol] = datetime.now()
            
            # === PROCESS PAIR SIGNALS ===
            await self._process_pair_updates(symbol)
            
        except Exception as e:
            logger.error(f"❌ Market data update error for {symbol}: {e}")
    
    async def _process_pair_updates(self, updated_symbol: str):
        """Process statistical arbitrage signals when pair data updates"""
        try:
            # === FIND PAIRS CONTAINING THIS SYMBOL ===
            relevant_pairs = []
            for pair_name, config in self.config["pairs_config"].items():
                if updated_symbol in config["symbols"]:
                    relevant_pairs.append(pair_name)
            
            # === PROCESS EACH RELEVANT PAIR ===
            for pair_name in relevant_pairs:
                await self._analyze_pair(pair_name)
                
        except Exception as e:
            logger.error(f"❌ Pair processing error: {e}")
    
    async def _analyze_pair(self, pair_name: str):
        """Analyze statistical arbitrage opportunity for a pair"""
        try:
            config = self.config["pairs_config"][pair_name]
            symbols = config["symbols"]
            
            # === CHECK DATA FRESHNESS ===
            if not self._has_fresh_data(symbols):
                return
            
            # === PREPARE DATA FOR ANALYSIS ===
            prices_a = self.market_data[symbols[0]]["price"]
            prices_b = self.market_data[symbols[1]]["price"]
            
            if prices_a <= 0 or prices_b <= 0:
                return
            
            # === GENERATE STATISTICAL SIGNAL ===
            feature_set = {
                f"{symbols[0]}_price": prices_a,
                f"{symbols[1]}_price": prices_b,
                "timestamp": datetime.now().timestamp()
            }
            
            signal = self.arbitrage_engine.generate_pair_signal(
                pair_name=pair_name,
                feature_set=feature_set,
                config=config
            )
            
            if signal and signal.signal_type != "HOLD":
                self.total_signals += 1
                
                # === RISK CHECK ===
                if await self._check_signal_risk(signal):
                    await self._execute_pair_signal(signal)
            
        except Exception as e:
            logger.error(f"❌ Pair analysis error for {pair_name}: {e}")
    
    def _has_fresh_data(self, symbols: List[str]) -> bool:
        """Check if all symbols have fresh data"""
        cutoff_time = datetime.now() - timedelta(seconds=30)
        
        for symbol in symbols:
            if symbol not in self.last_update_time:
                return False
            if self.last_update_time[symbol] < cutoff_time:
                return False
        
        return True
    
    async def _check_signal_risk(self, signal) -> bool:
        """Comprehensive risk check for trading signals"""
        try:
            self.risk_violations.clear()
            
            # === TRADING HOURS CHECK ===
            if not self._is_within_trading_hours():
                self.risk_violations.append("Outside trading hours")
                return False
            
            # === NEWS BLACKOUT CHECK ===
            if self._is_news_blackout():
                self.risk_violations.append("News blackout period")
                return False
            
            # === CONFIDENCE CHECK ===
            if signal.confidence < self.config["risk_limits"]["confidence_threshold"]:
                self.risk_violations.append(f"Low confidence: {signal.confidence:.2f}")
                return False
            
            # === MAX POSITIONS CHECK ===
            if len(self.active_pairs) >= self.config["risk_limits"]["max_concurrent_pairs"]:
                self.risk_violations.append("Maximum concurrent positions reached")
                return False
            
            # === DAILY LOSS CHECK ===
            max_loss_pct = self.config["risk_limits"]["max_daily_loss_pct"] / 100.0
            if abs(self.daily_pnl) > max_loss_pct:
                self.risk_violations.append(f"Daily loss limit exceeded: {self.daily_pnl:.2%}")
                self.is_trading_enabled = False
                return False
            
            # === ADVANCED RISK MANAGER ===
            risk_approved = self.risk_manager.check_signal_risk(signal, self.market_data)
            if not risk_approved:
                self.risk_violations.extend(self.risk_manager.violations)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk check error: {e}")
            return False
    
    def _is_within_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        current_hour = datetime.now().hour
        start_hour = self.config["trading_hours"]["start_hour"]
        end_hour = self.config["trading_hours"]["end_hour"]
        
        return start_hour <= current_hour <= end_hour
    
    def _is_news_blackout(self) -> bool:
        """Check if current time is during news blackout"""
        if not self.config["trading_hours"]["enable_news_blackout"]:
            return False
        
        # === MAJOR NEWS TIMES (would load from external calendar) ===
        blackout_times = [
            (8, 30),   # 8:30 AM ET - Economic data
            (10, 0),   # 10:00 AM ET - Economic data
            (14, 0),   # 2:00 PM ET - FOMC
        ]
        
        current_time = datetime.now()
        current_hour_min = (current_time.hour, current_time.minute)
        
        for blackout_hour, blackout_min in blackout_times:
            if (current_hour_min[0] == blackout_hour and 
                abs(current_hour_min[1] - blackout_min) <= 5):
                return True
        
        return False
    
    async def _execute_pair_signal(self, signal):
        """Execute statistical arbitrage signal through NinjaTrader"""
        try:
            logger.info(f"🎯 Executing {signal.signal_type} for {signal.pair_name}")
            logger.info(f"📊 Z-Score: {signal.z_score:.2f}, Confidence: {signal.confidence:.2f}")
            
            config = self.config["pairs_config"][signal.pair_name]
            symbols = config["symbols"]
            position_size = self.config["risk_limits"]["position_size"]
            
            # === CALCULATE HEDGE QUANTITIES ===
            qty_a = position_size
            qty_b = max(1, int(round(position_size * abs(signal.hedge_ratio))))
            
            if signal.signal_type == "LONG_PAIR":
                # Long B, Short A
                await self._execute_pair_trades(
                    symbol_a=symbols[0], qty_a=-qty_a,  # Short A
                    symbol_b=symbols[1], qty_b=qty_b,   # Long B
                    signal=signal
                )
            
            elif signal.signal_type == "SHORT_PAIR":
                # Short B, Long A
                await self._execute_pair_trades(
                    symbol_a=symbols[0], qty_a=qty_a,   # Long A
                    symbol_b=symbols[1], qty_b=-qty_b,  # Short B
                    signal=signal
                )
            
            elif signal.signal_type == "EXIT_PAIR":
                await self._exit_pair_position(signal.pair_name)
            
            self.executed_trades += 1
            
        except Exception as e:
            logger.error(f"❌ Execution error for {signal.pair_name}: {e}")
    
    async def _execute_pair_trades(self, symbol_a: str, qty_a: int, 
                                  symbol_b: str, qty_b: int, signal):
        """Execute the actual pair trades"""
        try:
            # === EXECUTE TRADES SIMULTANEOUSLY ===
            tasks = []
            
            if qty_a != 0:
                tasks.append(self.ninjatrader_executor.place_order(
                    symbol=symbol_a,
                    quantity=abs(qty_a),
                    side="BUY" if qty_a > 0 else "SELL",
                    order_type="MARKET"
                ))
            
            if qty_b != 0:
                tasks.append(self.ninjatrader_executor.place_order(
                    symbol=symbol_b,
                    quantity=abs(qty_b),
                    side="BUY" if qty_b > 0 else "SELL",
                    order_type="MARKET"
                ))
            
            # === EXECUTE SIMULTANEOUSLY ===
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # === TRACK POSITION ===
                self.active_pairs[signal.pair_name] = {
                    "signal_type": signal.signal_type,
                    "entry_time": datetime.now(),
                    "z_score_entry": signal.z_score,
                    "symbols": [symbol_a, symbol_b],
                    "quantities": [qty_a, qty_b],
                    "entry_prices": [
                        self.market_data[symbol_a]["price"],
                        self.market_data[symbol_b]["price"]
                    ]
                }
                
                # === LOG EXECUTION ===
                execution_report = ExecutionReport(
                    pair_name=signal.pair_name,
                    signal_type=signal.signal_type,
                    symbol_a=symbol_a,
                    symbol_b=symbol_b,
                    quantity_a=qty_a,
                    quantity_b=qty_b,
                    price_a=self.market_data[symbol_a]["price"],
                    price_b=self.market_data[symbol_b]["price"],
                    z_score=signal.z_score,
                    confidence=signal.confidence,
                    timestamp=datetime.now()
                )
                
                self.execution_reports.append(execution_report)
                
                logger.info(f"✅ Pair executed: {symbol_a}({qty_a}) {symbol_b}({qty_b})")
            
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            raise
    
    async def _exit_pair_position(self, pair_name: str):
        """Exit an active pair position"""
        try:
            if pair_name not in self.active_pairs:
                logger.warning(f"⚠️ No active position for {pair_name}")
                return
            
            position = self.active_pairs[pair_name]
            symbols = position["symbols"]
            quantities = position["quantities"]
            
            # === FLATTEN POSITIONS ===
            tasks = []
            
            for symbol, qty in zip(symbols, quantities):
                if qty != 0:
                    # Reverse the position
                    exit_side = "SELL" if qty > 0 else "BUY"
                    tasks.append(self.ninjatrader_executor.place_order(
                        symbol=symbol,
                        quantity=abs(qty),
                        side=exit_side,
                        order_type="MARKET"
                    ))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # === CALCULATE P&L ===
            holding_time = datetime.now() - position["entry_time"]
            logger.info(f"🔄 {pair_name} position closed after {holding_time.total_seconds()/60:.0f} minutes")
            
            # === REMOVE FROM ACTIVE PAIRS ===
            del self.active_pairs[pair_name]
            
        except Exception as e:
            logger.error(f"❌ Exit error for {pair_name}: {e}")
    
    async def run(self):
        """Main trading loop"""
        try:
            logger.info("🚀 Starting institutional trading system")
            self.is_running = True
            
            while self.is_running:
                try:
                    # === MONITOR ACTIVE POSITIONS ===
                    await self._monitor_active_positions()
                    
                    # === PERIODIC RISK CHECK ===
                    if datetime.now() - self.last_risk_check > timedelta(minutes=5):
                        await self._periodic_risk_check()
                        self.last_risk_check = datetime.now()
                    
                    # === UPDATE PERFORMANCE METRICS ===
                    self._update_performance_metrics()
                    
                    # === LOG STATUS ===
                    if datetime.now().minute % 10 == 0:  # Every 10 minutes
                        self._log_system_status()
                    
                    await asyncio.sleep(1)  # 1-second loop
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Critical error in trading system: {e}")
        finally:
            await self.shutdown()
    
    async def _monitor_active_positions(self):
        """Monitor active pairs for time-based exits"""
        try:
            positions_to_exit = []
            
            for pair_name, position in self.active_pairs.items():
                config = self.config["pairs_config"][pair_name]
                holding_time = datetime.now() - position["entry_time"]
                max_holding_minutes = config["max_holding_minutes"]
                
                if holding_time.total_seconds() / 60 > max_holding_minutes:
                    positions_to_exit.append(pair_name)
                    logger.info(f"⏰ Time-based exit for {pair_name}")
            
            # === EXECUTE TIME-BASED EXITS ===
            for pair_name in positions_to_exit:
                await self._exit_pair_position(pair_name)
                
        except Exception as e:
            logger.error(f"❌ Position monitoring error: {e}")
    
    async def _periodic_risk_check(self):
        """Periodic risk management checks"""
        try:
            # === CHECK SYSTEM HEALTH ===
            data_freshness_issues = []
            stale_cutoff = datetime.now() - timedelta(minutes=2)
            
            for symbol, last_update in self.last_update_time.items():
                if last_update < stale_cutoff:
                    data_freshness_issues.append(symbol)
            
            if data_freshness_issues:
                logger.warning(f"⚠️ Stale data for: {data_freshness_issues}")
            
            # === RISK MANAGER UPDATE ===
            risk_status = self.risk_manager.get_risk_status()
            if risk_status["violations"]:
                logger.warning(f"⚠️ Risk violations: {risk_status['violations']}")
                
        except Exception as e:
            logger.error(f"❌ Periodic risk check error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            # Calculate current P&L from active positions
            current_pnl = 0.0
            
            for pair_name, position in self.active_pairs.items():
                symbols = position["symbols"]
                quantities = position["quantities"]
                entry_prices = position["entry_prices"]
                
                for i, (symbol, qty, entry_price) in enumerate(zip(symbols, quantities, entry_prices)):
                    if symbol in self.market_data:
                        current_price = self.market_data[symbol]["price"]
                        if qty != 0 and current_price > 0:
                            position_pnl = qty * (current_price - entry_price)
                            current_pnl += position_pnl
            
            self.daily_pnl = current_pnl
            
        except Exception as e:
            logger.error(f"❌ Performance metrics error: {e}")
    
    def _log_system_status(self):
        """Log current system status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "active_pairs": len(self.active_pairs),
                "daily_pnl": self.daily_pnl,
                "total_signals": self.total_signals,
                "executed_trades": self.executed_trades,
                "trading_enabled": self.is_trading_enabled,
                "risk_violations": len(self.risk_violations)
            }
            
            logger.info(f"📊 System Status: {json.dumps(status, indent=2)}")
            
        except Exception as e:
            logger.error(f"❌ Status logging error: {e}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            logger.info("🔄 Shutting down institutional trading system...")
            
            self.is_running = False
            
            # === CLOSE ALL POSITIONS ===
            if self.active_pairs:
                logger.info(f"🔄 Closing {len(self.active_pairs)} active positions...")
                for pair_name in list(self.active_pairs.keys()):
                    await self._exit_pair_position(pair_name)
            
            # === DISCONNECT FROM DATA SOURCES ===
            if self.rithmic_connector:
                await self.rithmic_connector.disconnect()
            
            if self.ninjatrader_executor:
                await self.ninjatrader_executor.disconnect()
            
            # === SAVE EXECUTION REPORTS ===
            if self.execution_reports:
                reports_df = pd.DataFrame([asdict(report) for report in self.execution_reports])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                reports_df.to_csv(f"logs/execution_reports_{timestamp}.csv", index=False)
                logger.info(f"💾 Saved {len(self.execution_reports)} execution reports")
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

async def main():
    """Main entry point for institutional trading system"""
    controller = InstitutionalController()
    
    try:
        await controller.initialize()
        await controller.run()
    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown requested")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        await controller.shutdown()

if __name__ == "__main__":
    asyncio.run(main())