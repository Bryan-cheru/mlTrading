"""
Complete ES Trading System with Real NinjaTrader Integration
Uses ATI for market data and AddOn for order execution
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import sqlite3
import logging
from typing import Dict, Any, Optional
import socket
from ninjatrader_addon_interface import ESTrader
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ESDataManager:
    """Handles ES market data collection via Yahoo Finance"""
    
    def __init__(self):
        self.symbol = "ES=F"  # ES futures
        self.logger = logging.getLogger(__name__)
    
    def get_current_data(self, period="5d", interval="15m") -> pd.DataFrame:
        """Get current ES market data"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.error("No data received from Yahoo Finance")
                return pd.DataFrame()
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower


class ESSignalGenerator:
    """Generates trading signals for ES futures"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on technical analysis
        
        Returns:
            Dict with signal info: {'action': 'BUY'/'SELL'/'HOLD', 'confidence': float, 'reason': str}
        """
        if data.empty or len(data) < 50:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = []
        reasons = []
        
        # Signal 1: SMA Crossover
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals.append('BUY')
            reasons.append('SMA20 crossed above SMA50')
        elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals.append('SELL')
            reasons.append('SMA20 crossed below SMA50')
        
        # Signal 2: RSI Conditions
        if latest['RSI'] < 30:
            signals.append('BUY')
            reasons.append(f'RSI oversold ({latest["RSI"]:.1f})')
        elif latest['RSI'] > 70:
            signals.append('SELL')
            reasons.append(f'RSI overbought ({latest["RSI"]:.1f})')
        
        # Signal 3: Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            signals.append('BUY')
            reasons.append('Price below lower Bollinger Band')
        elif latest['Close'] > latest['BB_Upper']:
            signals.append('SELL')
            reasons.append('Price above upper Bollinger Band')
        
        # Signal 4: Price momentum
        price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        if price_change > 0.5:  # 0.5% gain
            signals.append('BUY')
            reasons.append(f'Strong upward momentum ({price_change:.2f}%)')
        elif price_change < -0.5:  # 0.5% loss
            signals.append('SELL')
            reasons.append(f'Strong downward momentum ({price_change:.2f}%)')
        
        # Combine signals
        if not signals:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No clear signal'}
        
        # Calculate confidence based on signal agreement
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        total_signals = len(signals)
        
        if buy_signals > sell_signals:
            confidence = buy_signals / total_signals
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': '; '.join(reasons),
                'price': latest['Close'],
                'signals_count': f'{buy_signals}/{total_signals}'
            }
        elif sell_signals > buy_signals:
            confidence = sell_signals / total_signals
            return {
                'action': 'SELL', 
                'confidence': confidence,
                'reason': '; '.join(reasons),
                'price': latest['Close'],
                'signals_count': f'{sell_signals}/{total_signals}'
            }
        else:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conflicting signals'}


class ESRiskManager:
    """Risk management for ES trading"""
    
    def __init__(self, max_position_size=2, max_daily_trades=5):
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.daily_trades = 0
        self.current_position = 0
        self.last_trade_date = None
        self.logger = logging.getLogger(__name__)
    
    def reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
    
    def can_trade(self, action: str, quantity: int = 1) -> Dict[str, Any]:
        """Check if trade is allowed"""
        self.reset_daily_counters()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return {
                'allowed': False,
                'reason': f'Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})'
            }
        
        # Check position size limits
        new_position = self.current_position
        if action == 'BUY':
            new_position += quantity
        elif action == 'SELL':
            new_position -= quantity
        
        if abs(new_position) > self.max_position_size:
            return {
                'allowed': False,
                'reason': f'Position size limit exceeded (would be {new_position}, max {self.max_position_size})'
            }
        
        return {'allowed': True, 'reason': 'Trade approved'}
    
    def record_trade(self, action: str, quantity: int = 1):
        """Record completed trade"""
        self.reset_daily_counters()
        self.daily_trades += 1
        
        if action == 'BUY':
            self.current_position += quantity
        elif action == 'SELL':
            self.current_position -= quantity
        
        self.logger.info(f"Trade recorded: {action} {quantity}, new position: {self.current_position}")


class ATIDataInterface:
    """Interface to NinjaTrader ATI for market data"""
    
    def __init__(self, host='localhost', port=36973):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
    
    def get_account_info(self) -> Optional[str]:
        """Get account information from ATI"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            if sock.connect_ex((self.host, self.port)) == 0:
                sock.send("ASK\n".encode())
                time.sleep(1)
                response = sock.recv(4096).decode()
                sock.close()
                return response
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"ATI data request failed: {e}")
            return None


class CompleteTradingSystem:
    """Complete ES trading system with real NinjaTrader integration"""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('es_trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = ESDataManager()
        self.signal_generator = ESSignalGenerator()
        self.risk_manager = ESRiskManager()
        self.ati_data = ATIDataInterface()
        self.ninja_trader = ESTrader()
        
        # Trading state
        self.is_running = False
        self.trades_today = []
        
        # Database for trade logging
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for trade logging"""
        try:
            conn = sqlite3.connect('es_trades.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price REAL,
                    order_id TEXT,
                    signal_confidence REAL,
                    signal_reason TEXT,
                    status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """Log trade to database"""
        try:
            conn = sqlite3.connect('es_trades.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (timestamp, action, quantity, price, order_id, 
                                 signal_confidence, signal_reason, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                trade_info.get('action'),
                trade_info.get('quantity', 1),
                trade_info.get('price', 0),
                trade_info.get('order_id'),
                trade_info.get('confidence', 0),
                trade_info.get('reason'),
                trade_info.get('status')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Trade logging failed: {e}")
    
    def check_system_health(self) -> Dict[str, bool]:
        """Check if all system components are healthy"""
        health = {}
        
        # Check data connection
        try:
            data = self.data_manager.get_current_data()
            health['data'] = not data.empty
        except:
            health['data'] = False
        
        # Check ATI connection
        health['ati'] = self.ati_data.get_account_info() is not None
        
        # Check NinjaTrader AddOn
        health['ninjatrader'] = self.ninja_trader.is_connected()
        
        return health
    
    def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute trade based on signal"""
        if signal['action'] == 'HOLD':
            return True
        
        # Check risk management
        risk_check = self.risk_manager.can_trade(signal['action'])
        if not risk_check['allowed']:
            self.logger.warning(f"Trade blocked by risk management: {risk_check['reason']}")
            return False
        
        # Execute order via NinjaTrader AddOn
        try:
            if signal['action'] == 'BUY':
                result = self.ninja_trader.buy_es(1)
            else:  # SELL
                result = self.ninja_trader.sell_es(1)
            
            if result['success']:
                self.logger.info(f"✅ Order executed: {result['message']}")
                
                # Record trade
                self.risk_manager.record_trade(signal['action'])
                
                # Log to database
                trade_info = {
                    'action': signal['action'],
                    'quantity': 1,
                    'price': signal.get('price', 0),
                    'order_id': result['order_id'],
                    'confidence': signal['confidence'],
                    'reason': signal['reason'],
                    'status': 'EXECUTED'
                }
                self.log_trade(trade_info)
                
                return True
            else:
                self.logger.error(f"❌ Order failed: {result['error']}")
                
                # Log failed trade
                trade_info = {
                    'action': signal['action'],
                    'confidence': signal['confidence'],
                    'reason': signal['reason'],
                    'status': 'FAILED: ' + result['error']
                }
                self.log_trade(trade_info)
                
                return False
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        self.logger.info("🔄 Starting trading cycle...")
        
        # Check system health
        health = self.check_system_health()
        self.logger.info(f"System health: {health}")
        
        if not all(health.values()):
            self.logger.warning("⚠️ System health issues detected")
            if not health['data']:
                self.logger.error("❌ Market data unavailable")
            if not health['ati']:
                self.logger.warning("⚠️ ATI connection lost")
            if not health['ninjatrader']:
                self.logger.error("❌ NinjaTrader AddOn not responding")
                return False
        
        # Get market data
        data = self.data_manager.get_current_data()
        if data.empty:
            self.logger.error("❌ No market data available")
            return False
        
        # Generate signal
        signal = self.signal_generator.generate_signal(data)
        self.logger.info(f"📊 Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.logger.info(f"📋 Reason: {signal['reason']}")
        
        # Execute trade if signal is strong enough
        if signal['confidence'] >= 0.7:  # 70% confidence threshold
            self.logger.info(f"🎯 Strong signal detected, executing trade...")
            success = self.execute_trade(signal)
            if success:
                self.logger.info("✅ Trade executed successfully")
            else:
                self.logger.error("❌ Trade execution failed")
            return success
        else:
            self.logger.info("⏸️ Signal confidence too low, holding position")
            return True
    
    def start_automated_trading(self, interval_minutes=15):
        """Start automated trading with specified interval"""
        self.logger.info("🚀 STARTING AUTOMATED ES TRADING SYSTEM")
        self.logger.info("=" * 60)
        
        # System health check
        health = self.check_system_health()
        self.logger.info(f"Initial system health: {health}")
        
        if not health['ninjatrader']:
            self.logger.error("❌ NinjaTrader AddOn not responding")
            self.logger.error("   Please ensure ESOrderExecutor AddOn is running")
            return False
        
        self.is_running = True
        cycle_count = 0
        
        try:
            while self.is_running:
                cycle_count += 1
                self.logger.info(f"\n📊 Trading Cycle #{cycle_count}")
                self.logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                try:
                    self.run_trading_cycle()
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                
                # Wait for next cycle
                self.logger.info(f"⏳ Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏸️ Trading system stopped by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.is_running = False
    
    def stop_trading(self):
        """Stop automated trading"""
        self.is_running = False
        self.logger.info("🛑 Trading system stopping...")


def main():
    """Main execution"""
    print("🚀 ES FUTURES AUTOMATED TRADING SYSTEM")
    print("=" * 60)
    print("✅ Data: Yahoo Finance + ATI")
    print("✅ Orders: NinjaTrader AddOn")
    print("✅ Risk Management: Position limits")
    print("✅ AI Signals: Technical analysis")
    
    # Initialize system
    trading_system = CompleteTradingSystem()
    
    # Check system readiness
    print(f"\n🔍 SYSTEM HEALTH CHECK")
    health = trading_system.check_system_health()
    
    print(f"   📊 Market Data: {'✅' if health['data'] else '❌'}")
    print(f"   🔌 ATI Connection: {'✅' if health['ati'] else '⚠️'}")
    print(f"   🎯 NinjaTrader: {'✅' if health['ninjatrader'] else '❌'}")
    
    if not health['ninjatrader']:
        print(f"\n❌ CRITICAL: NinjaTrader AddOn not responding")
        print(f"   Please install and start ESOrderExecutor AddOn")
        print(f"   See ADDON_INSTALLATION_GUIDE.md for instructions")
        return
    
    if not health['data']:
        print(f"\n❌ WARNING: Market data unavailable")
        print(f"   System will attempt to continue with available data")
    
    # Get user choice
    print(f"\n🎯 TRADING OPTIONS:")
    print(f"   1. Run single trading cycle (test)")
    print(f"   2. Start automated trading (15-min intervals)")
    print(f"   3. Check system status only")
    
    choice = input(f"\n❓ Select option (1-3): ").strip()
    
    try:
        if choice == '1':
            print(f"\n🧪 RUNNING SINGLE TRADING CYCLE")
            success = trading_system.run_trading_cycle()
            if success:
                print(f"✅ Trading cycle completed successfully")
            else:
                print(f"❌ Trading cycle failed")
                
        elif choice == '2':
            print(f"\n🚀 STARTING AUTOMATED TRADING")
            print(f"   Interval: 15 minutes")
            print(f"   Press Ctrl+C to stop")
            
            confirm = input(f"\n❓ Start automated trading? (y/n): ").strip().lower()
            if confirm == 'y':
                trading_system.start_automated_trading(15)
            else:
                print(f"⏸️ Automated trading cancelled")
                
        elif choice == '3':
            print(f"\n📊 SYSTEM STATUS:")
            health = trading_system.check_system_health()
            for component, status in health.items():
                print(f"   {component}: {'✅ Working' if status else '❌ Failed'}")
                
            # Get account status
            if health['ninjatrader']:
                positions = trading_system.ninja_trader.get_positions()
                if positions['success']:
                    print(f"\n💰 ACCOUNT STATUS:")
                    for key, value in positions['status'].items():
                        print(f"   {key}: {value}")
        else:
            print(f"❌ Invalid option")
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️ System interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print(f"\n📊 TRADING SESSION COMPLETE")
    print(f"   Check es_trading_system.log for detailed logs")
    print(f"   Check es_trades.db for trade history")


if __name__ == "__main__":
    main()
