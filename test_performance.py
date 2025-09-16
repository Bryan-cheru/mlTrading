"""
Test script for Performance Analytics Dashboard
"""

import sys
import os
sys.path.append('monitoring')
from performance_dashboard import PerformanceAnalytics
import datetime
import numpy as np

def test_performance_analytics():
    """Test the performance analytics engine"""
    print("📊 Testing Performance Analytics...")
    
    # Initialize analytics
    analytics = PerformanceAnalytics()
    
    # Add sample trades
    base_time = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for i in range(10):
        trade = {
            'symbol': 'ES',
            'entry_time': base_time + datetime.timedelta(days=i*2),
            'exit_time': base_time + datetime.timedelta(days=i*2 + 1),
            'entry_price': 4400 + np.random.normal(0, 20),
            'exit_price': 4400 + np.random.normal(5, 25),
            'quantity': 1,
            'side': 'LONG',
            'pnl': np.random.normal(50, 100)
        }
        analytics.add_trade(trade)
    
    # Calculate metrics
    metrics = analytics.calculate_performance_metrics()
    
    print("📊 Performance Analytics Test Results:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print("✅ Performance Analytics testing complete!")
    
    return True

if __name__ == "__main__":
    test_performance_analytics()
