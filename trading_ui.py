#!/usr/bin/env python3
"""
ES Trading System - Professional Trading Interface
Real-time trading dashboard with live data, signals, and performance monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import yfinance as yf

# Import our trading system components
from complete_es_trading_system import CompleteTradingSystem
from ninjatrader_addon_interface import ESTrader

class TradingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("ES Trading System - Professional Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # System components
        self.trading_system = None
        self.es_trader = None
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Data storage
        self.current_data = {}
        self.signal_history = []
        self.trade_history = []
        self.performance_data = {
            'pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'current_position': 0,
            'daily_trades': 0
        }
        
        # Create UI components
        self.create_widgets()
        self.setup_database()
        
        # Start data update thread
        self.start_data_thread()

    def create_widgets(self):
        """Create all UI components"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create sections
        self.create_header(main_frame)
        self.create_left_panel(main_frame)
        self.create_center_panel(main_frame)
        self.create_right_panel(main_frame)
        self.create_bottom_panel(main_frame)

    def create_header(self, parent):
        """Create header with system status and controls"""
        
        header_frame = ttk.LabelFrame(parent, text="System Control", padding="10")
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # System status
        status_frame = ttk.Frame(header_frame)
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(status_frame, text="System Status:", font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=(0, 10))
        self.status_label = ttk.Label(status_frame, text="STOPPED", foreground='red', font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(status_frame, text="Time:", font=('Arial', 10)).grid(row=0, column=2, padx=(0, 5))
        self.time_label = ttk.Label(status_frame, text="", font=('Arial', 10))
        self.time_label.grid(row=0, column=3, padx=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.start_button = ttk.Button(button_frame, text="Start System", command=self.start_system)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop System", command=self.stop_system, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        ttk.Button(button_frame, text="Reset Data", command=self.reset_data).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(button_frame, text="Export Report", command=self.export_report).grid(row=0, column=3)

    def create_left_panel(self, parent):
        """Create left panel with market data and signals"""
        
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        
        # Market Data Section
        market_frame = ttk.LabelFrame(left_frame, text="Market Data - ES Futures", padding="10")
        market_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        market_frame.columnconfigure(1, weight=1)
        
        # Price display
        price_data = [
            ("Current Price:", "price", "$0.00"),
            ("Change:", "change", "$0.00"),
            ("Change %:", "change_pct", "0.00%"),
            ("Volume:", "volume", "0"),
            ("Open:", "open", "$0.00"),
            ("High:", "high", "$0.00"),
            ("Low:", "low", "$0.00"),
            ("Previous Close:", "prev_close", "$0.00")
        ]
        
        self.market_labels = {}
        for i, (label, key, default) in enumerate(price_data):
            ttk.Label(market_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.market_labels[key] = ttk.Label(market_frame, text=default, font=('Arial', 10))
            self.market_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Signal Section
        signal_frame = ttk.LabelFrame(left_frame, text="Current Signals", padding="10")
        signal_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        signal_frame.columnconfigure(1, weight=1)
        
        # Signal indicators
        signal_data = [
            ("SMA Signal:", "sma_signal", "HOLD"),
            ("RSI Signal:", "rsi_signal", "HOLD"),
            ("Bollinger Signal:", "bb_signal", "HOLD"),
            ("Momentum Signal:", "momentum_signal", "HOLD"),
            ("Consensus:", "consensus", "HOLD"),
            ("Confidence:", "confidence", "0%")
        ]
        
        self.signal_labels = {}
        for i, (label, key, default) in enumerate(signal_data):
            ttk.Label(signal_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.signal_labels[key] = ttk.Label(signal_frame, text=default, font=('Arial', 10))
            self.signal_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Position Section
        position_frame = ttk.LabelFrame(left_frame, text="Current Position", padding="10")
        position_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        position_frame.columnconfigure(1, weight=1)
        position_frame.rowconfigure(4, weight=1)
        
        # Position data
        position_data = [
            ("Position:", "position", "0 contracts"),
            ("Entry Price:", "entry_price", "$0.00"),
            ("Current P&L:", "unrealized_pnl", "$0.00"),
            ("Daily Trades:", "daily_trades", "0/5")
        ]
        
        self.position_labels = {}
        for i, (label, key, default) in enumerate(position_data):
            ttk.Label(position_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.position_labels[key] = ttk.Label(position_frame, text=default, font=('Arial', 10))
            self.position_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)

    def create_center_panel(self, parent):
        """Create center panel with price chart"""
        
        center_frame = ttk.LabelFrame(parent, text="ES Futures Price Chart", padding="10")
        center_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('ES Futures - 15 Minute Chart', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price ($)')
        self.ax.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=center_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chart controls
        chart_controls = ttk.Frame(center_frame)
        chart_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(chart_controls, text="Refresh Chart", command=self.update_chart).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(chart_controls, text="1 Day", command=lambda: self.change_timeframe('1d')).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(chart_controls, text="5 Days", command=lambda: self.change_timeframe('5d')).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(chart_controls, text="1 Month", command=lambda: self.change_timeframe('1mo')).grid(row=0, column=3)

    def create_right_panel(self, parent):
        """Create right panel with performance metrics"""
        
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        
        # Performance Section
        perf_frame = ttk.LabelFrame(right_frame, text="Performance Metrics", padding="10")
        perf_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        perf_frame.columnconfigure(1, weight=1)
        
        # Performance data
        perf_data = [
            ("Total P&L:", "total_pnl", "$0.00"),
            ("Today's P&L:", "daily_pnl", "$0.00"),
            ("Total Trades:", "total_trades", "0"),
            ("Winning Trades:", "winning_trades", "0"),
            ("Win Rate:", "win_rate", "0%"),
            ("Average Win:", "avg_win", "$0.00"),
            ("Average Loss:", "avg_loss", "$0.00"),
            ("Profit Factor:", "profit_factor", "0.00")
        ]
        
        self.perf_labels = {}
        for i, (label, key, default) in enumerate(perf_data):
            ttk.Label(perf_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.perf_labels[key] = ttk.Label(perf_frame, text=default, font=('Arial', 10))
            self.perf_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Account Section
        account_frame = ttk.LabelFrame(right_frame, text="Account Information", padding="10")
        account_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        account_frame.columnconfigure(1, weight=1)
        
        # Account data
        account_data = [
            ("Account:", "account_name", "Sim101"),
            ("Balance:", "balance", "$0.00"),
            ("Equity:", "equity", "$0.00"),
            ("Margin Used:", "margin_used", "$0.00"),
            ("Buying Power:", "buying_power", "$0.00")
        ]
        
        self.account_labels = {}
        for i, (label, key, default) in enumerate(account_data):
            ttk.Label(account_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.account_labels[key] = ttk.Label(account_frame, text=default, font=('Arial', 10))
            self.account_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)
        
        # Risk Section
        risk_frame = ttk.LabelFrame(right_frame, text="Risk Management", padding="10")
        risk_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        risk_frame.columnconfigure(1, weight=1)
        
        # Risk data
        risk_data = [
            ("Max Position:", "max_position", "2 contracts"),
            ("Max Daily Trades:", "max_daily_trades", "5 trades"),
            ("Min Confidence:", "min_confidence", "70%"),
            ("Risk Per Trade:", "risk_per_trade", "$500"),
            ("Daily Risk Limit:", "daily_risk", "$1,000")
        ]
        
        self.risk_labels = {}
        for i, (label, key, default) in enumerate(risk_data):
            ttk.Label(risk_frame, text=label, font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.risk_labels[key] = ttk.Label(risk_frame, text=default, font=('Arial', 10))
            self.risk_labels[key].grid(row=i, column=1, sticky=tk.E, pady=2)

    def create_bottom_panel(self, parent):
        """Create bottom panel with logs and trade history"""
        
        bottom_frame = ttk.LabelFrame(parent, text="System Activity", padding="10")
        bottom_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(bottom_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Activity Log Tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Activity Log")
        
        # Create scrolled text for logs
        log_scroll_frame = ttk.Frame(log_frame)
        log_scroll_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll_frame.columnconfigure(0, weight=1)
        log_scroll_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_scroll_frame, height=8, font=('Consolas', 10))
        log_scrollbar = ttk.Scrollbar(log_scroll_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Trade History Tab
        trade_frame = ttk.Frame(notebook)
        notebook.add(trade_frame, text="Trade History")
        
        # Create treeview for trade history
        trade_columns = ('Time', 'Action', 'Quantity', 'Price', 'P&L', 'Signal')
        self.trade_tree = ttk.Treeview(trade_frame, columns=trade_columns, show='headings', height=8)
        
        for col in trade_columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        trade_scrollbar = ttk.Scrollbar(trade_frame, orient="vertical", command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=trade_scrollbar.set)
        
        self.trade_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        trade_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        trade_frame.columnconfigure(0, weight=1)
        trade_frame.rowconfigure(0, weight=1)

    def setup_database(self):
        """Initialize database for trade logging"""
        try:
            conn = sqlite3.connect('es_trading_ui.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price REAL,
                    pnl REAL,
                    signal TEXT,
                    confidence REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    total_pnl REAL,
                    daily_pnl REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    current_position INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            self.log_message("Database initialized successfully")
        except Exception as e:
            self.log_message(f"Database setup error: {e}")

    def start_system(self):
        """Start the trading system"""
        try:
            self.log_message("Initializing trading system...")
            
            # Initialize trading system
            self.trading_system = CompleteTradingSystem()
            self.es_trader = ESTrader()
            
            # Test NinjaTrader connection
            account_status = self.es_trader.get_account_status()
            if account_status and 'error' not in account_status.lower():
                self.log_message(f"NinjaTrader connected: {account_status}")
            else:
                self.log_message("Warning: NinjaTrader connection issue")
            
            self.is_running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="RUNNING", foreground='green')
            
            self.log_message("Trading system started successfully!")
            self.log_message("System will check for signals every 15 minutes")
            
        except Exception as e:
            self.log_message(f"Error starting system: {e}")
            messagebox.showerror("Error", f"Failed to start trading system: {e}")

    def stop_system(self):
        """Stop the trading system"""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="STOPPED", foreground='red')
        self.log_message("Trading system stopped")

    def start_data_thread(self):
        """Start background thread for data updates"""
        def update_loop():
            while True:
                try:
                    if self.is_running:
                        self.update_market_data()
                        self.update_signals()
                        self.update_performance()
                        self.check_trading_signals()
                    
                    # Update time display
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.root.after(0, lambda: self.time_label.config(text=current_time))
                    
                    time.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"Data update error: {e}"))
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

    def update_market_data(self):
        """Update market data display"""
        try:
            # Get ES futures data
            es_data = yf.download("ES=F", period="1d", interval="15m", progress=False)
            
            if not es_data.empty:
                latest = es_data.iloc[-1]
                previous = es_data.iloc[-2] if len(es_data) > 1 else latest
                
                current_price = latest['Close']
                change = current_price - previous['Close']
                change_pct = (change / previous['Close']) * 100
                
                # Update market data labels
                market_updates = {
                    'price': f"${current_price:.2f}",
                    'change': f"${change:+.2f}",
                    'change_pct': f"{change_pct:+.2f}%",
                    'volume': f"{int(latest['Volume']):,}",
                    'open': f"${latest['Open']:.2f}",
                    'high': f"${latest['High']:.2f}",
                    'low': f"${latest['Low']:.2f}",
                    'prev_close': f"${previous['Close']:.2f}"
                }
                
                # Update UI in main thread
                for key, value in market_updates.items():
                    self.root.after(0, lambda k=key, v=value: self.market_labels[k].config(text=v))
                
                # Update change color
                color = 'green' if change >= 0 else 'red'
                self.root.after(0, lambda: self.market_labels['change'].config(foreground=color))
                self.root.after(0, lambda: self.market_labels['change_pct'].config(foreground=color))
                
                # Store current data for chart
                self.current_data = {
                    'price': current_price,
                    'data': es_data,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Market data error: {e}"))

    def update_signals(self):
        """Update signal display"""
        try:
            if self.trading_system and 'data' in self.current_data:
                data = self.current_data['data']
                signal_result = self.trading_system.signal_generator.generate_signal(data)
                
                if signal_result:
                    signals = signal_result.get('individual_signals', {})
                    
                    signal_updates = {
                        'sma_signal': signals.get('sma', 'HOLD'),
                        'rsi_signal': signals.get('rsi', 'HOLD'),
                        'bb_signal': signals.get('bollinger', 'HOLD'),
                        'momentum_signal': signals.get('momentum', 'HOLD'),
                        'consensus': signal_result.get('signal', 'HOLD'),
                        'confidence': f"{signal_result.get('confidence', 0)*100:.1f}%"
                    }
                    
                    # Update signal labels with colors
                    for key, value in signal_updates.items():
                        if key != 'confidence':
                            color = 'green' if value == 'BUY' else ('red' if value == 'SELL' else 'orange')
                        else:
                            confidence_val = float(value.replace('%', ''))
                            color = 'green' if confidence_val >= 70 else ('orange' if confidence_val >= 50 else 'red')
                        
                        self.root.after(0, lambda k=key, v=value, c=color: (
                            self.signal_labels[k].config(text=v, foreground=c)
                        ))
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Signal update error: {e}"))

    def update_performance(self):
        """Update performance metrics"""
        try:
            # Get performance data from database
            conn = sqlite3.connect('es_trading_ui.db')
            
            # Calculate total P&L
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
            result = cursor.fetchone()
            total_pnl = result[0] if result[0] else 0.0
            
            # Calculate today's P&L
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("SELECT SUM(pnl) FROM trades WHERE DATE(timestamp) = ? AND pnl IS NOT NULL", (today,))
            result = cursor.fetchone()
            daily_pnl = result[0] if result[0] else 0.0
            
            # Get trade statistics
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl > 0")
            result = cursor.fetchone()
            avg_win = result[0] if result[0] else 0.0
            
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl < 0")
            result = cursor.fetchone()
            avg_loss = abs(result[0]) if result[0] else 0.0
            
            conn.close()
            
            # Calculate derived metrics
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            profit_factor = (avg_win / avg_loss) if avg_loss > 0 else 0
            
            # Update performance labels
            perf_updates = {
                'total_pnl': f"${total_pnl:.2f}",
                'daily_pnl': f"${daily_pnl:.2f}",
                'total_trades': str(total_trades),
                'winning_trades': str(winning_trades),
                'win_rate': f"{win_rate:.1f}%",
                'avg_win': f"${avg_win:.2f}",
                'avg_loss': f"${avg_loss:.2f}",
                'profit_factor': f"{profit_factor:.2f}"
            }
            
            for key, value in perf_updates.items():
                self.root.after(0, lambda k=key, v=value: self.perf_labels[k].config(text=v))
            
            # Update P&L colors
            total_color = 'green' if total_pnl >= 0 else 'red'
            daily_color = 'green' if daily_pnl >= 0 else 'red'
            self.root.after(0, lambda: self.perf_labels['total_pnl'].config(foreground=total_color))
            self.root.after(0, lambda: self.perf_labels['daily_pnl'].config(foreground=daily_color))
            
            # Update account information
            if self.es_trader:
                account_status = self.es_trader.get_account_status()
                if account_status and '$' in account_status:
                    try:
                        balance = account_status.split('$')[1].split()[0]
                        self.root.after(0, lambda: self.account_labels['balance'].config(text=f"${balance}"))
                        self.root.after(0, lambda: self.account_labels['equity'].config(text=f"${balance}"))
                        
                        # Calculate buying power (simplified)
                        buying_power = float(balance.replace(',', '')) * 0.8  # 80% buying power
                        self.root.after(0, lambda: self.account_labels['buying_power'].config(text=f"${buying_power:,.2f}"))
                    except:
                        pass
                
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Performance update error: {e}"))

    def check_trading_signals(self):
        """Check for trading signals and execute trades"""
        try:
            if not self.is_running or not self.trading_system:
                return
            
            # Check if it's time to trade (every 15 minutes)
            now = datetime.now()
            if now.minute % 15 != 0:
                return
            
            # Get current data
            if 'data' not in self.current_data:
                return
                
            data = self.current_data['data']
            
            # Generate signal
            signal_result = self.trading_system.signal_generator.generate_signal(data)
            if not signal_result:
                return
            
            signal = signal_result.get('signal', 'HOLD')
            confidence = signal_result.get('confidence', 0)
            
            self.log_message(f"Signal generated: {signal} (confidence: {confidence:.1%})")
            
            # Check risk management
            current_position = self.get_current_position()
            daily_trades = self.get_daily_trade_count()
            
            risk_check = self.trading_system.risk_manager.check_signal_risk(
                signal, current_position, daily_trades, confidence
            )
            
            if not risk_check['approved']:
                self.log_message(f"Signal rejected: {risk_check['reason']}")
                return
            
            # Execute trade
            if signal in ['BUY', 'SELL']:
                self.execute_trade(signal, 1, confidence)
                
        except Exception as e:
            self.log_message(f"Signal check error: {e}")

    def execute_trade(self, action: str, quantity: int, confidence: float):
        """Execute a trade"""
        try:
            current_price = self.current_data.get('price', 0)
            
            if action == 'BUY':
                result = self.es_trader.buy_es(quantity)
            else:
                result = self.es_trader.sell_es(quantity)
            
            self.log_message(f"Trade executed: {action} {quantity} ES @ ${current_price:.2f}")
            self.log_message(f"NinjaTrader response: {result}")
            
            # Log trade to database
            self.log_trade(action, quantity, current_price, confidence)
            
            # Update trade history display
            self.update_trade_history()
            
        except Exception as e:
            self.log_message(f"Trade execution error: {e}")

    def log_trade(self, action: str, quantity: int, price: float, confidence: float):
        """Log trade to database"""
        try:
            conn = sqlite3.connect('es_trading_ui.db')
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO trades (timestamp, action, quantity, price, pnl, signal, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, action, quantity, price, None, action, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_message(f"Trade logging error: {e}")

    def update_trade_history(self):
        """Update trade history display"""
        try:
            # Clear existing items
            for item in self.trade_tree.get_children():
                self.trade_tree.delete(item)
            
            # Get recent trades
            conn = sqlite3.connect('es_trading_ui.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, action, quantity, price, pnl, signal
                FROM trades
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            
            trades = cursor.fetchall()
            conn.close()
            
            for trade in trades:
                timestamp, action, quantity, price, pnl, signal = trade
                
                # Format timestamp
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%H:%M:%S')
                
                # Format values
                pnl_str = f"${pnl:.2f}" if pnl else "Pending"
                
                self.trade_tree.insert('', 0, values=(
                    time_str, action, quantity, f"${price:.2f}", pnl_str, signal
                ))
                
        except Exception as e:
            self.log_message(f"Trade history update error: {e}")

    def get_current_position(self) -> int:
        """Get current position from database"""
        try:
            conn = sqlite3.connect('es_trading_ui.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END)
                FROM trades
            ''')
            result = cursor.fetchone()
            conn.close()
            return result[0] if result[0] else 0
        except:
            return 0

    def get_daily_trade_count(self) -> int:
        """Get today's trade count"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            conn = sqlite3.connect('es_trading_ui.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?', (today,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result[0] else 0
        except:
            return 0

    def update_chart(self):
        """Update price chart"""
        try:
            if 'data' not in self.current_data:
                # Get initial data
                es_data = yf.download("ES=F", period="1d", interval="15m", progress=False)
            else:
                es_data = self.current_data['data']
            
            if es_data.empty:
                return
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot price data
            times = es_data.index
            prices = es_data['Close']
            
            self.ax.plot(times, prices, 'b-', linewidth=2, label='ES Futures')
            
            # Add moving averages
            if len(es_data) >= 20:
                sma_20 = es_data['Close'].rolling(window=20).mean()
                self.ax.plot(times, sma_20, 'r--', alpha=0.7, label='SMA 20')
            
            if len(es_data) >= 50:
                sma_50 = es_data['Close'].rolling(window=50).mean()
                self.ax.plot(times, sma_50, 'g--', alpha=0.7, label='SMA 50')
            
            # Format chart
            self.ax.set_title('ES Futures - 15 Minute Chart', fontsize=14, fontweight='bold')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Price ($)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Rotate x-axis labels
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Tight layout
            self.fig.tight_layout()
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Chart update error: {e}")

    def change_timeframe(self, period: str):
        """Change chart timeframe"""
        try:
            self.log_message(f"Updating chart to {period} timeframe...")
            
            # Get data for new timeframe
            es_data = yf.download("ES=F", period=period, interval="15m", progress=False)
            
            if not es_data.empty:
                self.current_data['data'] = es_data
                self.update_chart()
                self.log_message(f"Chart updated to {period} timeframe")
            
        except Exception as e:
            self.log_message(f"Timeframe change error: {e}")

    def log_message(self, message: str):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        try:
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
        except:
            pass  # In case UI is not ready

    def reset_data(self):
        """Reset all data and performance metrics"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all data? This cannot be undone."):
            try:
                # Clear database
                conn = sqlite3.connect('es_trading_ui.db')
                cursor = conn.cursor()
                cursor.execute('DELETE FROM trades')
                cursor.execute('DELETE FROM performance')
                conn.commit()
                conn.close()
                
                # Clear UI displays
                self.trade_tree.delete(*self.trade_tree.get_children())
                self.log_text.delete(1.0, tk.END)
                
                self.log_message("All data has been reset")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset data: {e}")

    def export_report(self):
        """Export trading report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"es_trading_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("ES Trading System - Performance Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Performance summary
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 20 + "\n")
                for key, label in [('total_pnl', 'Total P&L'), ('daily_pnl', "Today's P&L"), 
                                 ('total_trades', 'Total Trades'), ('win_rate', 'Win Rate')]:
                    f.write(f"{label}: {self.perf_labels[key].cget('text')}\n")
                
                f.write("\n")
                
                # Trade history
                f.write("TRADE HISTORY\n")
                f.write("-" * 15 + "\n")
                f.write("Time\t\tAction\tQty\tPrice\t\tP&L\n")
                
                conn = sqlite3.connect('es_trading_ui.db')
                cursor = conn.cursor()
                cursor.execute('SELECT timestamp, action, quantity, price, pnl FROM trades ORDER BY timestamp')
                trades = cursor.fetchall()
                conn.close()
                
                for trade in trades:
                    timestamp, action, quantity, price, pnl = trade
                    dt = datetime.fromisoformat(timestamp)
                    pnl_str = f"${pnl:.2f}" if pnl else "Pending"
                    f.write(f"{dt.strftime('%H:%M:%S')}\t{action}\t{quantity}\t${price:.2f}\t\t{pnl_str}\n")
            
            self.log_message(f"Report exported to {filename}")
            messagebox.showinfo("Export Complete", f"Report exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")


def main():
    """Main function to run the trading dashboard"""
    print("Starting ES Trading System UI...")
    
    # Create main window
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create dashboard
    dashboard = TradingDashboard(root)
    
    # Add initial log message
    dashboard.log_message("ES Trading System UI initialized")
    dashboard.log_message("Click 'Start System' to begin automated trading")
    
    # Start the GUI
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down trading system...")


if __name__ == "__main__":
    main()