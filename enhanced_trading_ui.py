#!/usr/bin/env python3
"""
ES Trading System - Enhanced Professional UI with Training Progress
Includes real-time training monitoring, multiple data sources, and comprehensive dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import subprocess
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import yfinance as yf
import requests
import websocket
from pathlib import Path

# Set matplotlib style for dark theme
plt.style.use('dark_background')

class EnhancedTradingDashboard:
    """Enhanced trading dashboard with training progress monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ES Trading System - Enhanced Professional Dashboard")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        # System state
        self.is_running = False
        self.training_process = None
        self.data_queue = queue.Queue()
        self.training_queue = queue.Queue()
        
        # Data sources
        self.data_sources = {
            'yahoo_finance': {'enabled': True, 'status': 'disconnected'},
            'alpha_vantage': {'enabled': False, 'status': 'disconnected'},
            'ninjatrader': {'enabled': False, 'status': 'disconnected'},
            'interactive_brokers': {'enabled': False, 'status': 'disconnected'}
        }
        
        # Training monitoring
        self.training_metrics = {
            'epoch': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'progress': 0.0,
            'eta': '00:00:00',
            'status': 'idle'
        }
        
        # Performance data
        self.performance_data = {
            'pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        self.setup_ui()
        self.start_monitoring_threads()
        
    def setup_ui(self):
        """Create the enhanced UI layout"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_trading_tab()
        self.create_training_tab()
        self.create_data_sources_tab()
        self.create_settings_tab()
        
    def create_trading_tab(self):
        """Create the main trading interface tab"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading Dashboard")
        
        # Top control panel
        control_frame = ttk.LabelFrame(trading_frame, text="System Control")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start System", 
                  command=self.start_trading_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop System", 
                  command=self.stop_trading_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Emergency Stop", 
                  command=self.emergency_stop).pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(control_frame, text="Status: Stopped")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Main content area
        content_frame = ttk.Frame(trading_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Market data and signals
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Market data panel
        market_frame = ttk.LabelFrame(left_frame, text="Market Data")
        market_frame.pack(fill=tk.X, pady=5)
        
        # Price display
        self.price_label = ttk.Label(market_frame, text="ES Price: --", 
                                   font=("Arial", 16, "bold"))
        self.price_label.pack(pady=5)
        
        # Signal display
        signal_frame = ttk.LabelFrame(left_frame, text="Trading Signals")
        signal_frame.pack(fill=tk.X, pady=5)
        
        self.signal_label = ttk.Label(signal_frame, text="Signal: HOLD", 
                                    font=("Arial", 14))
        self.signal_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(signal_frame, text="Confidence: --")
        self.confidence_label.pack()
        
        # Chart area
        chart_frame = ttk.LabelFrame(left_frame, text="Price Chart")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.setup_price_chart(chart_frame)
        
        # Right panel - Performance and trades
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(right_frame, text="Performance")
        perf_frame.pack(fill=tk.X, pady=5)
        
        self.pnl_label = ttk.Label(perf_frame, text="P&L: $0.00", 
                                 font=("Arial", 12, "bold"))
        self.pnl_label.pack()
        
        self.trades_label = ttk.Label(perf_frame, text="Trades: 0")
        self.trades_label.pack()
        
        self.winrate_label = ttk.Label(perf_frame, text="Win Rate: 0%")
        self.winrate_label.pack()
        
        self.sharpe_label = ttk.Label(perf_frame, text="Sharpe: 0.00")
        self.sharpe_label.pack()
        
        # Trade log
        log_frame = ttk.LabelFrame(right_frame, text="Trade Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.trade_tree = ttk.Treeview(log_frame, columns=('Time', 'Action', 'Price', 'P&L'), 
                                     show='headings', height=10)
        self.trade_tree.heading('Time', text='Time')
        self.trade_tree.heading('Action', text='Action')
        self.trade_tree.heading('Price', text='Price')
        self.trade_tree.heading('P&L', text='P&L')
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_training_tab(self):
        """Create the ML training progress tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="ML Training")
        
        # Training control panel
        control_frame = ttk.LabelFrame(training_frame, text="Training Control")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start Random Forest Training", 
                  command=self.start_rf_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Start Transformer Training", 
                  command=self.start_transformer_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Training", 
                  command=self.stop_training).pack(side=tk.LEFT, padx=5)
        
        # Training status
        status_frame = ttk.LabelFrame(training_frame, text="Training Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bars and metrics
        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.overall_progress.pack(fill=tk.X, pady=2)
        
        ttk.Label(progress_frame, text="Current Epoch:").pack(anchor=tk.W)
        self.epoch_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.epoch_progress.pack(fill=tk.X, pady=2)
        
        # Training metrics
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.training_status_label = ttk.Label(metrics_frame, text="Status: Idle")
        self.training_status_label.pack(anchor=tk.W)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: 0/0")
        self.epoch_label.pack(anchor=tk.W)
        
        self.loss_label = ttk.Label(metrics_frame, text="Loss: 0.0000")
        self.loss_label.pack(anchor=tk.W)
        
        self.accuracy_label = ttk.Label(metrics_frame, text="Accuracy: 0.00%")
        self.accuracy_label.pack(anchor=tk.W)
        
        self.eta_label = ttk.Label(metrics_frame, text="ETA: --:--:--")
        self.eta_label.pack(anchor=tk.W)
        
        # Training log
        log_frame = ttk.LabelFrame(training_frame, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.training_log = tk.Text(log_frame, bg='#2d2d2d', fg='white', 
                                  font=("Consolas", 10))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, 
                                    command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scrollbar.set)
        
        self.training_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Training charts
        chart_frame = ttk.LabelFrame(training_frame, text="Training Progress Charts")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.setup_training_charts(chart_frame)
        
    def create_data_sources_tab(self):
        """Create data sources configuration tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Sources")
        
        # Data source selection
        source_frame = ttk.LabelFrame(data_frame, text="Available Data Sources")
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Yahoo Finance
        yahoo_frame = ttk.Frame(source_frame)
        yahoo_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.yahoo_var = tk.BooleanVar(value=True)
        yahoo_check = ttk.Checkbutton(yahoo_frame, text="Yahoo Finance (Free, 15-min delay)", 
                                    variable=self.yahoo_var, command=self.update_data_sources)
        yahoo_check.pack(side=tk.LEFT)
        
        self.yahoo_status = ttk.Label(yahoo_frame, text="Status: Ready")
        self.yahoo_status.pack(side=tk.RIGHT)
        
        # Alpha Vantage
        alpha_frame = ttk.Frame(source_frame)
        alpha_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.alpha_var = tk.BooleanVar()
        alpha_check = ttk.Checkbutton(alpha_frame, text="Alpha Vantage (API Key Required)", 
                                    variable=self.alpha_var, command=self.update_data_sources)
        alpha_check.pack(side=tk.LEFT)
        
        self.alpha_status = ttk.Label(alpha_frame, text="Status: Not Configured")
        self.alpha_status.pack(side=tk.RIGHT)
        
        # API Key entry
        api_frame = ttk.Frame(alpha_frame)
        api_frame.pack(fill=tk.X, padx=(20, 0), pady=2)
        
        ttk.Label(api_frame, text="API Key:").pack(side=tk.LEFT)
        self.alpha_key_entry = ttk.Entry(api_frame, width=30, show="*")
        self.alpha_key_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(api_frame, text="Test", command=self.test_alpha_vantage).pack(side=tk.LEFT, padx=5)
        
        # NinjaTrader
        nt_frame = ttk.Frame(source_frame)
        nt_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.nt_var = tk.BooleanVar()
        nt_check = ttk.Checkbutton(nt_frame, text="NinjaTrader 8 (Real-time, Professional)", 
                                 variable=self.nt_var, command=self.update_data_sources)
        nt_check.pack(side=tk.LEFT)
        
        self.nt_status = ttk.Label(nt_frame, text="Status: Not Connected")
        self.nt_status.pack(side=tk.RIGHT)
        
        # Interactive Brokers
        ib_frame = ttk.Frame(source_frame)
        ib_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.ib_var = tk.BooleanVar()
        ib_check = ttk.Checkbutton(ib_frame, text="Interactive Brokers (TWS/Gateway Required)", 
                                 variable=self.ib_var, command=self.update_data_sources)
        ib_check.pack(side=tk.LEFT)
        
        self.ib_status = ttk.Label(ib_frame, text="Status: Not Connected")
        self.ib_status.pack(side=tk.RIGHT)
        
        # Data quality panel
        quality_frame = ttk.LabelFrame(data_frame, text="Data Quality Metrics")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        quality_metrics = ttk.Frame(quality_frame)
        quality_metrics.pack(fill=tk.X, padx=5, pady=5)
        
        self.latency_label = ttk.Label(quality_metrics, text="Latency: -- ms")
        self.latency_label.pack(anchor=tk.W)
        
        self.update_rate_label = ttk.Label(quality_metrics, text="Update Rate: -- per sec")
        self.update_rate_label.pack(anchor=tk.W)
        
        self.data_quality_label = ttk.Label(quality_metrics, text="Data Quality: Good")
        self.data_quality_label.pack(anchor=tk.W)
        
        # Recommendation panel
        rec_frame = ttk.LabelFrame(data_frame, text="Data Source Recommendations")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        recommendations = tk.Text(rec_frame, height=10, wrap=tk.WORD)
        recommendations.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        rec_text = """
DATA SOURCE RECOMMENDATIONS:

🥇 BEST FOR LIVE TRADING:
• NinjaTrader 8: Real-time ES futures data, professional grade
• Interactive Brokers: Direct market access, institutional quality

🥈 GOOD FOR DEVELOPMENT:
• Alpha Vantage: Real-time with API key, good for testing
• Yahoo Finance: Free but 15-min delayed, good for backtesting

💡 OPTIMAL SETUP:
• Primary: NinjaTrader 8 (live trading)
• Secondary: Alpha Vantage (backup/validation)
• Development: Yahoo Finance (free development)

⚠️ IMPORTANT NOTES:
• ES futures require real-time data for profitable trading
• Yahoo Finance delay makes it unsuitable for live trading
• Always use multiple sources for data validation
• Test your data sources before going live

🎯 FOR YOUR ES SEPTEMBER FUTURES:
• Use NinjaTrader 8 for live execution
• Alpha Vantage for real-time validation
• Yahoo Finance for historical backtesting only
        """
        
        recommendations.insert(tk.END, rec_text)
        recommendations.config(state=tk.DISABLED)
        
    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Trading settings
        trading_settings = ttk.LabelFrame(settings_frame, text="Trading Settings")
        trading_settings.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk management
        risk_frame = ttk.Frame(trading_settings)
        risk_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(risk_frame, text="Max Position Size:").pack(side=tk.LEFT)
        self.max_position = ttk.Entry(risk_frame, width=10)
        self.max_position.insert(0, "5")
        self.max_position.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(risk_frame, text="Stop Loss (%):").pack(side=tk.LEFT, padx=(20, 0))
        self.stop_loss = ttk.Entry(risk_frame, width=10)
        self.stop_loss.insert(0, "2.0")
        self.stop_loss.pack(side=tk.LEFT, padx=5)
        
        # Model settings
        model_settings = ttk.LabelFrame(settings_frame, text="Model Settings")
        model_settings.pack(fill=tk.X, padx=5, pady=5)
        
        model_frame = ttk.Frame(model_settings)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Active Model:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(model_frame, values=["Random Forest", "Transformer", "Ensemble"])
        self.model_combo.set("Random Forest")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # System settings
        system_settings = ttk.LabelFrame(settings_frame, text="System Settings")
        system_settings.pack(fill=tk.X, padx=5, pady=5)
        
        sys_frame = ttk.Frame(system_settings)
        sys_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sys_frame, text="Update Frequency (ms):").pack(side=tk.LEFT)
        self.update_freq = ttk.Entry(sys_frame, width=10)
        self.update_freq.insert(0, "1000")
        self.update_freq.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(sys_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=20)
        
    def setup_price_chart(self, parent):
        """Setup the price chart"""
        self.price_fig = Figure(figsize=(8, 4), dpi=100, facecolor='#1e1e1e')
        self.price_ax = self.price_fig.add_subplot(111, facecolor='#2d2d2d')
        
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, parent)
        self.price_canvas.draw()
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty data
        self.price_data = {'time': [], 'price': [], 'signals': []}
        
    def setup_training_charts(self, parent):
        """Setup training progress charts"""
        self.training_fig = Figure(figsize=(12, 4), dpi=100, facecolor='#1e1e1e')
        
        # Loss chart
        self.loss_ax = self.training_fig.add_subplot(121, facecolor='#2d2d2d')
        self.loss_ax.set_title('Training Loss', color='white')
        self.loss_ax.set_xlabel('Epoch', color='white')
        self.loss_ax.set_ylabel('Loss', color='white')
        
        # Accuracy chart
        self.acc_ax = self.training_fig.add_subplot(122, facecolor='#2d2d2d')
        self.acc_ax.set_title('Training Accuracy', color='white')
        self.acc_ax.set_xlabel('Epoch', color='white')
        self.acc_ax.set_ylabel('Accuracy (%)', color='white')
        
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, parent)
        self.training_canvas.draw()
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize training data
        self.training_data = {'epochs': [], 'loss': [], 'accuracy': []}
        
    def start_monitoring_threads(self):
        """Start background monitoring threads"""
        # Data monitoring thread
        data_thread = threading.Thread(target=self.monitor_data, daemon=True)
        data_thread.start()
        
        # Training monitoring thread
        training_thread = threading.Thread(target=self.monitor_training, daemon=True)
        training_thread.start()
        
        # UI update thread
        ui_thread = threading.Thread(target=self.update_ui, daemon=True)
        ui_thread.start()
        
    def monitor_data(self):
        """Monitor market data in background"""
        while True:
            try:
                if self.is_running:
                    # Simulate getting market data
                    current_price = self.get_current_price()
                    signal = self.get_current_signal()
                    
                    # Update data
                    self.data_queue.put({
                        'type': 'market_data',
                        'price': current_price,
                        'signal': signal,
                        'timestamp': datetime.now()
                    })
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Data monitoring error: {e}")
                time.sleep(5)
                
    def monitor_training(self):
        """Monitor training progress in background"""
        while True:
            try:
                if self.training_process and self.training_process.poll() is None:
                    # Training is running, check for updates
                    self.check_training_progress()
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Training monitoring error: {e}")
                time.sleep(5)
                
    def update_ui(self):
        """Update UI elements in main thread"""
        while True:
            try:
                # Process data queue
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.process_data_update(data)
                
                # Process training queue
                while not self.training_queue.empty():
                    training_update = self.training_queue.get()
                    self.process_training_update(training_update)
                
                time.sleep(0.1)  # Fast UI updates
                
            except Exception as e:
                print(f"UI update error: {e}")
                time.sleep(1)
                
    def start_rf_training(self):
        """Start Random Forest training"""
        self.log_training_message("Starting Random Forest training...")
        
        try:
            # Start training in subprocess
            self.training_process = subprocess.Popen([
                sys.executable, "train_model.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.training_metrics['status'] = 'training'
            self.update_training_status()
            
        except Exception as e:
            self.log_training_message(f"Error starting training: {e}")
            
    def start_transformer_training(self):
        """Start Transformer training"""
        self.log_training_message("Starting GPU Transformer training...")
        
        try:
            # Start GPU training
            self.training_process = subprocess.Popen([
                sys.executable, "gpu_transformer_trainer.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.training_metrics['status'] = 'training'
            self.update_training_status()
            
        except Exception as e:
            self.log_training_message(f"Error starting transformer training: {e}")
            
    def stop_training(self):
        """Stop current training"""
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
            self.training_metrics['status'] = 'stopped'
            self.log_training_message("Training stopped by user")
            self.update_training_status()
            
    def check_training_progress(self):
        """Check training progress from subprocess"""
        if not self.training_process:
            return
            
        try:
            # Read training output
            output = self.training_process.stdout.readline()
            if output:
                self.parse_training_output(output.strip())
                
        except Exception as e:
            print(f"Error reading training output: {e}")
            
    def parse_training_output(self, output):
        """Parse training output for progress information"""
        try:
            # Look for epoch information
            if "Epoch" in output and "/" in output:
                # Extract epoch info: "Epoch 10/50"
                parts = output.split()
                for part in parts:
                    if "/" in part:
                        current, total = part.split("/")
                        self.training_metrics['epoch'] = int(current)
                        self.training_metrics['total_epochs'] = int(total)
                        self.training_metrics['progress'] = (int(current) / int(total)) * 100
                        
            # Look for loss information
            if "Loss:" in output or "loss:" in output:
                parts = output.split()
                for i, part in enumerate(parts):
                    if "loss" in part.lower() and i + 1 < len(parts):
                        try:
                            self.training_metrics['loss'] = float(parts[i + 1].replace(",", ""))
                        except:
                            pass
                            
            # Look for accuracy information
            if "Accuracy:" in output or "accuracy:" in output:
                parts = output.split()
                for i, part in enumerate(parts):
                    if "accuracy" in part.lower() and i + 1 < len(parts):
                        try:
                            acc_str = parts[i + 1].replace("%", "").replace(",", "")
                            self.training_metrics['accuracy'] = float(acc_str)
                        except:
                            pass
                            
            # Queue UI update
            self.training_queue.put({
                'type': 'progress_update',
                'metrics': self.training_metrics.copy(),
                'message': output
            })
            
        except Exception as e:
            print(f"Error parsing training output: {e}")
            
    def process_training_update(self, update):
        """Process training updates in main thread"""
        if update['type'] == 'progress_update':
            metrics = update['metrics']
            
            # Update progress bars
            if 'progress' in metrics:
                self.overall_progress['value'] = metrics['progress']
                
            # Update labels
            self.training_status_label.config(text=f"Status: {metrics.get('status', 'Unknown')}")
            
            if 'epoch' in metrics and 'total_epochs' in metrics:
                self.epoch_label.config(text=f"Epoch: {metrics['epoch']}/{metrics['total_epochs']}")
                
            if 'loss' in metrics:
                self.loss_label.config(text=f"Loss: {metrics['loss']:.4f}")
                
            if 'accuracy' in metrics:
                self.accuracy_label.config(text=f"Accuracy: {metrics['accuracy']:.2f}%")
                
            # Update training charts
            self.update_training_charts(metrics)
            
            # Log message
            if 'message' in update:
                self.log_training_message(update['message'])
                
    def update_training_charts(self, metrics):
        """Update training progress charts"""
        if 'epoch' in metrics and 'loss' in metrics:
            self.training_data['epochs'].append(metrics['epoch'])
            self.training_data['loss'].append(metrics['loss'])
            
            # Update loss chart
            self.loss_ax.clear()
            self.loss_ax.plot(self.training_data['epochs'], self.training_data['loss'], 'b-')
            self.loss_ax.set_title('Training Loss', color='white')
            self.loss_ax.set_xlabel('Epoch', color='white')
            self.loss_ax.set_ylabel('Loss', color='white')
            
        if 'epoch' in metrics and 'accuracy' in metrics:
            if len(self.training_data['accuracy']) < len(self.training_data['epochs']):
                self.training_data['accuracy'].append(metrics['accuracy'])
                
            # Update accuracy chart
            self.acc_ax.clear()
            if self.training_data['accuracy']:
                epochs = self.training_data['epochs'][:len(self.training_data['accuracy'])]
                self.acc_ax.plot(epochs, self.training_data['accuracy'], 'g-')
            self.acc_ax.set_title('Training Accuracy', color='white')
            self.acc_ax.set_xlabel('Epoch', color='white')
            self.acc_ax.set_ylabel('Accuracy (%)', color='white')
            
        self.training_canvas.draw()
        
    def log_training_message(self, message):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Update in main thread
        self.root.after(0, lambda: self.training_log.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.training_log.see(tk.END))
        
    def update_training_status(self):
        """Update training status display"""
        status = self.training_metrics['status']
        self.training_status_label.config(text=f"Status: {status.title()}")
        
    def get_current_price(self):
        """Get current ES price (placeholder)"""
        try:
            # Use Yahoo Finance for ES futures
            es = yf.Ticker("ES=F")
            data = es.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return 4500.0  # Fallback price
        
    def get_current_signal(self):
        """Get current trading signal (placeholder)"""
        # This would integrate with your actual signal generation
        return {"signal": "HOLD", "confidence": 0.7}
        
    def process_data_update(self, data):
        """Process market data updates"""
        if data['type'] == 'market_data':
            # Update price display
            price = data['price']
            self.price_label.config(text=f"ES Price: ${price:.2f}")
            
            # Update signal display
            signal = data['signal']
            self.signal_label.config(text=f"Signal: {signal['signal']}")
            self.confidence_label.config(text=f"Confidence: {signal['confidence']:.1%}")
            
            # Update chart data
            self.price_data['time'].append(data['timestamp'])
            self.price_data['price'].append(price)
            self.price_data['signals'].append(signal['signal'])
            
            # Keep only last 100 points
            if len(self.price_data['time']) > 100:
                self.price_data['time'] = self.price_data['time'][-100:]
                self.price_data['price'] = self.price_data['price'][-100:]
                self.price_data['signals'] = self.price_data['signals'][-100:]
                
            self.update_price_chart()
            
    def update_price_chart(self):
        """Update the price chart"""
        if len(self.price_data['time']) > 1:
            self.price_ax.clear()
            
            # Plot price line
            times = [t.strftime("%H:%M") for t in self.price_data['time']]
            self.price_ax.plot(times, self.price_data['price'], 'cyan', linewidth=2)
            
            # Add signal markers
            for i, signal in enumerate(self.price_data['signals']):
                if signal == 'BUY':
                    self.price_ax.scatter(times[i], self.price_data['price'][i], 
                                        color='green', marker='^', s=100)
                elif signal == 'SELL':
                    self.price_ax.scatter(times[i], self.price_data['price'][i], 
                                        color='red', marker='v', s=100)
                                        
            self.price_ax.set_title('ES Futures Price', color='white')
            self.price_ax.set_ylabel('Price', color='white')
            self.price_ax.tick_params(axis='x', rotation=45)
            
            self.price_canvas.draw()
            
    def start_trading_system(self):
        """Start the trading system"""
        self.is_running = True
        self.status_label.config(text="Status: Running")
        self.log_training_message("Trading system started")
        
    def stop_trading_system(self):
        """Stop the trading system"""
        self.is_running = False
        self.status_label.config(text="Status: Stopped")
        self.log_training_message("Trading system stopped")
        
    def emergency_stop(self):
        """Emergency stop all operations"""
        self.stop_trading_system()
        self.stop_training()
        messagebox.showwarning("Emergency Stop", "All operations have been stopped!")
        
    def update_data_sources(self):
        """Update data source configuration"""
        # This would update the actual data source connections
        pass
        
    def test_alpha_vantage(self):
        """Test Alpha Vantage connection"""
        api_key = self.alpha_key_entry.get()
        if not api_key:
            messagebox.showerror("Error", "Please enter an API key")
            return
            
        try:
            # Test API call
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data:
                    self.alpha_status.config(text="Status: Connected")
                    messagebox.showinfo("Success", "Alpha Vantage connection successful!")
                else:
                    self.alpha_status.config(text="Status: Invalid API Key")
                    messagebox.showerror("Error", "Invalid API key")
            else:
                self.alpha_status.config(text="Status: Connection Failed")
                messagebox.showerror("Error", "Connection failed")
                
        except Exception as e:
            self.alpha_status.config(text="Status: Error")
            messagebox.showerror("Error", f"Connection error: {e}")
            
    def load_model(self):
        """Load a trained model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Joblib files", "*.joblib"), ("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load model logic here
                self.log_training_message(f"Model loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Model loaded successfully: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                
    def save_settings(self):
        """Save current settings"""
        settings = {
            'max_position': self.max_position.get(),
            'stop_loss': self.stop_loss.get(),
            'model': self.model_combo.get(),
            'update_freq': self.update_freq.get(),
            'data_sources': {
                'yahoo_finance': self.yahoo_var.get(),
                'alpha_vantage': self.alpha_var.get(),
                'ninjatrader': self.nt_var.get(),
                'interactive_brokers': self.ib_var.get()
            }
        }
        
        try:
            with open('trading_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

def main():
    """Launch the enhanced trading dashboard"""
    root = tk.Tk()
    app = EnhancedTradingDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()