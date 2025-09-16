"""
Portfolio Manager
Advanced portfolio management for institutional trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import threading
import json

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderStatus(Enum):
    PENDING = "Pending"
    FILLED = "Filled"
    PARTIALLY_FILLED = "PartiallyFilled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Is this a long position"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Is this a short position"""
        return self.quantity < 0

@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0

@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    symbol: str
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    side: str  # "BUY" or "SELL"

class PortfolioManager:
    """
    Advanced portfolio management system
    
    Features:
    - Real-time position tracking
    - Risk-adjusted position sizing
    - Portfolio optimization
    - Performance attribution
    - Risk metrics calculation
    """
    
    def __init__(self, initial_capital: float = 100000.0, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_history = []
        self.performance_metrics = {}
        
        # Risk management
        self.max_position_size = 0.1  # 10% max per position
        self.max_sector_exposure = 0.3  # 30% max per sector
        self.stop_loss_pct = 0.02  # 2% stop loss
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, volatility: float) -> int:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        
        Args:
            symbol: Trading instrument
            signal_strength: ML model confidence (0-1)
            current_price: Current market price
            volatility: Historical volatility
            
        Returns:
            Position size in shares/contracts
        """
        with self.lock:
            # Base position size using Kelly Criterion
            win_rate = 0.55  # Historical win rate
            avg_win = 0.015  # Average win percentage
            avg_loss = 0.012  # Average loss percentage
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Adjust for signal strength
            adjusted_fraction = kelly_fraction * signal_strength
            
            # Adjust for volatility (reduce size for high volatility)
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)
            adjusted_fraction *= volatility_adjustment
            
            # Calculate dollar amount
            available_capital = self.get_available_capital()
            target_dollar_amount = available_capital * adjusted_fraction
            
            # Apply maximum position size limit
            max_dollar_amount = self.get_total_equity() * self.max_position_size
            target_dollar_amount = min(target_dollar_amount, max_dollar_amount)
            
            # Convert to shares/contracts
            if current_price > 0:
                position_size = int(target_dollar_amount / current_price)
            else:
                position_size = 0
            
            logger.debug(f"Position size calculation for {symbol}: {position_size} units "
                        f"(${target_dollar_amount:,.2f}, signal: {signal_strength:.3f})")
            
            return position_size
    
    def submit_order(self, symbol: str, quantity: int, order_type: OrderType, 
                    price: Optional[float] = None, stop_price: Optional[float] = None) -> str:
        """Submit a new order"""
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price
        )
        
        with self.lock:
            self.orders[order_id] = order
        
        logger.info(f"Order submitted: {order_id} - {symbol} {quantity} {order_type.value}")
        return order_id
    
    def fill_order(self, order_id: str, filled_price: float, filled_quantity: int = None, 
                  commission: float = 1.0) -> bool:
        """Fill an order (simulated execution)"""
        with self.lock:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            order = self.orders[order_id]
            if filled_quantity is None:
                filled_quantity = order.quantity
            
            # Update order
            order.filled_quantity += filled_quantity
            order.filled_price = filled_price
            order.commission = commission
            
            if order.filled_quantity >= abs(order.quantity):
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Update positions
            self._update_position(order.symbol, filled_quantity, filled_price)
            
            # Record trade
            trade = Trade(
                trade_id=f"TRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=order.symbol,
                quantity=filled_quantity,
                price=filled_price,
                timestamp=datetime.now(),
                commission=commission,
                side="BUY" if filled_quantity > 0 else "SELL"
            )
            self.trades.append(trade)
            
            # Update cash
            trade_value = filled_quantity * filled_price
            self.cash -= trade_value + commission
            
            logger.info(f"Order filled: {order_id} - {filled_quantity} @ ${filled_price:.2f}")
            return True
    
    def _update_position(self, symbol: str, quantity: int, price: float):
        """Update position for a symbol"""
        if symbol not in self.positions:
            if quantity != 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price
                )
        else:
            position = self.positions[symbol]
            
            # Calculate realized PnL for position reduction
            if (position.quantity > 0 and quantity < 0) or (position.quantity < 0 and quantity > 0):
                closing_quantity = min(abs(quantity), abs(position.quantity))
                if position.quantity > 0:
                    realized_pnl = closing_quantity * (price - position.entry_price)
                else:
                    realized_pnl = closing_quantity * (position.entry_price - price)
                
                position.realized_pnl += realized_pnl
            
            # Update position
            new_quantity = position.quantity + quantity
            if new_quantity == 0:
                # Position closed
                del self.positions[symbol]
            else:
                # Update average entry price for additions
                if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
                    total_cost = position.quantity * position.entry_price + quantity * price
                    position.entry_price = total_cost / new_quantity
                
                position.quantity = new_quantity
                position.current_price = price
    
    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions"""
        with self.lock:
            for symbol, price in price_data.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.current_price = price
                    
                    # Calculate unrealized PnL
                    if position.is_long:
                        position.unrealized_pnl = position.quantity * (price - position.entry_price)
                    else:
                        position.unrealized_pnl = position.quantity * (position.entry_price - price)
    
    def get_total_equity(self) -> float:
        """Calculate total portfolio equity"""
        with self.lock:
            equity = self.cash
            for position in self.positions.values():
                equity += position.market_value
            return equity
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        # Reserve some cash for margin and running costs
        return max(0, self.cash * 0.9)
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        with self.lock:
            total_equity = self.get_total_equity()
            total_pnl = total_equity - self.initial_capital
            
            summary = {
                "total_equity": total_equity,
                "cash": self.cash,
                "total_pnl": total_pnl,
                "pnl_percentage": (total_pnl / self.initial_capital) * 100,
                "num_positions": len(self.positions),
                "num_trades": len(self.trades),
                "positions": []
            }
            
            for position in self.positions.values():
                summary["positions"].append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl
                })
            
            return summary
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        # Convert to pandas for easier calculation
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for trade in self.trades[-100:] if self._trade_pnl(trade) > 0)
        total_recent_trades = min(100, len(self.trades))
        win_rate = winning_trades / total_recent_trades if total_recent_trades > 0 else 0
        
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0
        }
    
    def _trade_pnl(self, trade: Trade) -> float:
        """Calculate PnL for a single trade (simplified)"""
        # This would need more sophisticated logic to match trades
        return 0.0
    
    def should_reduce_risk(self) -> bool:
        """Determine if risk should be reduced"""
        metrics = self.calculate_risk_metrics()
        
        # Reduce risk if:
        # 1. Current drawdown > 3%
        # 2. Win rate < 45%
        # 3. Too many positions
        
        current_dd = abs(metrics.get("current_drawdown", 0))
        win_rate = metrics.get("win_rate", 1.0)
        
        return (current_dd > 0.03 or 
                win_rate < 0.45 or 
                len(self.positions) > self.max_positions * 0.8)
    
    def update_equity_curve(self):
        """Update equity curve for performance tracking"""
        current_equity = self.get_total_equity()
        self.equity_curve.append(current_equity)
        
        # Keep only recent history to manage memory
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-5000:]
    
    def get_position_limits(self) -> Dict[str, int]:
        """Get position limits for risk management"""
        return {
            "max_positions": self.max_positions,
            "max_position_value": int(self.get_total_equity() * self.max_position_size),
            "available_capital": int(self.get_available_capital())
        }
