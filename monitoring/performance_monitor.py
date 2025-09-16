"""
Performance Monitor
Tracks and analyzes trading performance in real-time
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    trade_id: str
    instrument: str
    action: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    status: str = "OPEN"

class PerformanceMonitor:
    """
    Real-time performance monitoring and analysis
    """
    
    def __init__(self):
        self.trades = []
        self.orders = []
        self.positions = []
        self.daily_stats = {}
        self.start_time = datetime.now()
        
    def record_order(self, order):
        """Record order execution"""
        self.orders.append({
            'order_id': order.order_id,
            'instrument': order.instrument,
            'action': order.action,
            'quantity': order.quantity,
            'price': order.average_fill_price,
            'timestamp': order.fill_timestamp or order.timestamp,
            'status': order.status.value
        })
    
    def record_position(self, position):
        """Record position update"""
        self.positions.append({
            'instrument': position.instrument,
            'quantity': position.quantity,
            'average_price': position.average_price,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'timestamp': position.timestamp
        })
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame([
            {
                'pnl': trade.pnl,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time
            }
            for trade in self.trades if trade.status == "CLOSED"
        ])
        
        if df.empty:
            return {}
        
        total_pnl = df['pnl'].sum()
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        total_trades = len(df)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
        }
