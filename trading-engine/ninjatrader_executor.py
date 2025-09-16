"""
NinjaTrader 8 Execution Engine
Handles order execution and position management through NinjaTrader 8
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

from ..data-pipeline.ingestion.ninjatrader_connector import NinjaTraderConnector

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    order_id: str
    instrument: str
    action: str  # BUY or SELL
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    timestamp: datetime = None
    fill_timestamp: Optional[datetime] = None

@dataclass
class Position:
    instrument: str
    quantity: int  # Positive for long, negative for short
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = None

class NinjaTraderExecutor:
    """
    Executes trades through NinjaTrader 8 with institutional-grade risk management
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'host': '127.0.0.1',
            'port': 36973,
            'account_id': 'SIM101',  # Default simulation account
            'max_position_size': 10,
            'max_daily_trades': 100,
            'enable_risk_checks': True
        }
        
        self.connector = NinjaTraderConnector(
            host=self.config['host'], 
            port=self.config['port']
        )
        
        # Order and position tracking
        self.orders = {}
        self.positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # Risk management
        self.risk_manager = RiskManager(self.config)
        
        # Event callbacks
        self.order_callbacks = []
        self.position_callbacks = []
        
        # Connection status
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to NinjaTrader 8"""
        try:
            success = self.connector.connect()
            if success:
                self.is_connected = True
                logger.info("Connected to NinjaTrader 8 for trade execution")
                
                # Register for order and position updates
                self.connector.order_callbacks['executor'] = self._handle_order_update
                
                return True
            else:
                logger.error("Failed to connect to NinjaTrader 8")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to NinjaTrader: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        self.connector.disconnect()
        self.is_connected = False
        logger.info("Disconnected from NinjaTrader 8")
    
    async def place_order(self, instrument: str, action: str, quantity: int,
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Optional[str]:
        """
        Place order through NinjaTrader
        
        Args:
            instrument: Trading instrument (e.g., 'ES', 'NQ')
            action: 'BUY' or 'SELL'
            quantity: Number of contracts/shares
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("Not connected to NinjaTrader")
            return None
        
        try:
            # Risk checks
            if self.config['enable_risk_checks']:
                risk_result = self.risk_manager.check_order_risk(
                    instrument, action, quantity, self.positions, self.daily_trades
                )
                
                if not risk_result['approved']:
                    logger.warning(f"Order rejected by risk manager: {risk_result['reason']}")
                    return None
            
            # Generate order ID
            order_id = f"ORDER_{instrument}_{int(time.time())}"
            
            # Create order object
            order = Order(
                order_id=order_id,
                instrument=instrument,
                action=action,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                timestamp=datetime.now()
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Send order to NinjaTrader
            nt_order_id = await self._send_order_to_ninjatrader(order)
            
            if nt_order_id:
                logger.info(f"Order placed: {action} {quantity} {instrument} - Order ID: {order_id}")
                self.daily_trades += 1
                return order_id
            else:
                # Remove failed order
                del self.orders[order_id]
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def _send_order_to_ninjatrader(self, order: Order) -> Optional[str]:
        """Send order to NinjaTrader through connector"""
        try:
            if order.order_type == OrderType.MARKET:
                return self.connector.place_order(
                    order.instrument,
                    order.action,
                    order.quantity,
                    "MARKET"
                )
            elif order.order_type == OrderType.LIMIT:
                return self.connector.place_order(
                    order.instrument,
                    order.action,
                    order.quantity,
                    "LIMIT",
                    order.price
                )
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending order to NinjaTrader: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        try:
            self.connector.cancel_order(order_id)
            self.orders[order_id].status = OrderStatus.CANCELLED
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def _handle_order_update(self, order_update: Dict):
        """Handle order status updates from NinjaTrader"""
        try:
            order_id = order_update.get('order_id')
            status = order_update.get('status')
            filled_qty = order_update.get('filled_quantity', 0)
            fill_price = order_update.get('fill_price', 0.0)
            
            if order_id in self.orders:
                order = self.orders[order_id]
                
                # Update order status
                if status == 'FILLED':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_fill_price = fill_price
                    order.fill_timestamp = datetime.now()
                    
                    # Update position
                    self._update_position(order)
                    
                elif status == 'PARTIALLY_FILLED':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = filled_qty
                    order.average_fill_price = fill_price
                    
                elif status == 'REJECTED':
                    order.status = OrderStatus.REJECTED
                    
                elif status == 'CANCELLED':
                    order.status = OrderStatus.CANCELLED
                
                # Notify callbacks
                for callback in self.order_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"Error in order callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    def _update_position(self, order: Order):
        """Update position based on filled order"""
        try:
            instrument = order.instrument
            
            if instrument not in self.positions:
                self.positions[instrument] = Position(
                    instrument=instrument,
                    quantity=0,
                    average_price=0.0,
                    timestamp=datetime.now()
                )
            
            position = self.positions[instrument]
            
            # Calculate new position
            if order.action == 'BUY':
                new_quantity = position.quantity + order.filled_quantity
                if position.quantity == 0:
                    position.average_price = order.average_fill_price
                else:
                    # Weighted average price
                    total_cost = (position.quantity * position.average_price + 
                                order.filled_quantity * order.average_fill_price)
                    position.average_price = total_cost / new_quantity
                position.quantity = new_quantity
                
            elif order.action == 'SELL':
                # Calculate realized P&L for closing positions
                if position.quantity > 0:  # Closing long position
                    closing_qty = min(position.quantity, order.filled_quantity)
                    realized_pnl = closing_qty * (order.average_fill_price - position.average_price)
                    position.realized_pnl += realized_pnl
                    self.daily_pnl += realized_pnl
                
                position.quantity -= order.filled_quantity
                
                # If position goes negative, we're now short
                if position.quantity < 0:
                    position.average_price = order.average_fill_price
            
            position.timestamp = datetime.now()
            
            # Notify position callbacks
            for callback in self.position_callbacks:
                try:
                    callback(position)
                except Exception as e:
                    logger.error(f"Error in position callback: {e}")
                    
            logger.info(f"Position updated: {instrument} - Quantity: {position.quantity}, Avg Price: {position.average_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def get_position(self, instrument: str) -> Optional[Position]:
        """Get current position for instrument"""
        return self.positions.get(instrument)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        return self.daily_pnl
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add callback for order updates"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[Position], None]):
        """Add callback for position updates"""
        self.position_callbacks.append(callback)


class RiskManager:
    """
    Risk management for order execution
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def check_order_risk(self, instrument: str, action: str, quantity: int,
                        current_positions: Dict[str, Position],
                        daily_trades: int) -> Dict:
        """
        Check if order passes risk management rules
        
        Returns:
            Dict with 'approved' (bool) and 'reason' (str)
        """
        try:
            # Check daily trade limit
            if daily_trades >= self.config.get('max_daily_trades', 100):
                return {
                    'approved': False,
                    'reason': 'Daily trade limit exceeded'
                }
            
            # Check position size limit
            current_qty = 0
            if instrument in current_positions:
                current_qty = current_positions[instrument].quantity
            
            new_qty = current_qty + (quantity if action == 'BUY' else -quantity)
            max_position = self.config.get('max_position_size', 10)
            
            if abs(new_qty) > max_position:
                return {
                    'approved': False,
                    'reason': f'Position size would exceed limit: {abs(new_qty)} > {max_position}'
                }
            
            # Check quantity is positive
            if quantity <= 0:
                return {
                    'approved': False,
                    'reason': 'Order quantity must be positive'
                }
            
            return {
                'approved': True,
                'reason': 'Order approved'
            }
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {
                'approved': False,
                'reason': f'Risk check error: {e}'
            }


# Example usage and configuration
FUTURES_EXECUTION_CONFIG = {
    'host': '127.0.0.1',
    'port': 36973,
    'account_id': 'SIM101',
    'max_position_size': 5,  # 5 contracts max
    'max_daily_trades': 50,
    'enable_risk_checks': True,
    'instruments': {
        'ES': {'tick_size': 0.25, 'tick_value': 12.50},
        'NQ': {'tick_size': 0.25, 'tick_value': 5.00},
        'YM': {'tick_size': 1.0, 'tick_value': 5.00},
        'RTY': {'tick_size': 0.1, 'tick_value': 5.00}
    }
}
