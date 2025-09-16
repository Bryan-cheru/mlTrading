"""
Advanced Risk Manager
Institutional-grade risk management for the trading system
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    """
    Advanced risk management with multiple layers of protection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.violations = []
        
    def check_signal_risk(self, signal: Dict, instrument: str) -> bool:
        """
        Check if trading signal passes risk management
        """
        try:
            # Check confidence threshold
            if signal['confidence'] < self.config.get('min_confidence', 0.65):
                logger.info(f"Signal rejected: Low confidence {signal['confidence']:.3f}")
                return False
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.get('max_daily_loss', 0.02):
                logger.warning("Daily loss limit reached")
                return False
            
            # Check position size
            if signal['position_size'] > self.config.get('max_position_size', 5):
                logger.warning(f"Position size too large: {signal['position_size']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        
        # Update drawdown tracking
        if self.daily_pnl > self.peak_equity:
            self.peak_equity = self.daily_pnl
        
        current_drawdown = (self.peak_equity - self.daily_pnl) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
