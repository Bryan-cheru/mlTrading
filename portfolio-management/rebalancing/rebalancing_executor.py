"""
Real-Time Portfolio Rebalancing System
Automated rebalancing with NinjaTrader 8 integration and smart execution algorithms
Features: TWAP execution, market impact minimization, transaction cost optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our portfolio components
import importlib.util

# Import allocation engine
spec = importlib.util.spec_from_file_location(
    "dynamic_allocation_engine", 
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "allocation", "dynamic_allocation_engine.py")
)
allocation_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocation_engine)

RebalanceSignal = allocation_engine.RebalanceSignal
Portfolio = allocation_engine.Portfolio
Position = allocation_engine.Position
Asset = allocation_engine.Asset
DynamicAllocationEngine = allocation_engine.DynamicAllocationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionStrategy(Enum):
    """Order execution strategies"""
    MARKET = "MARKET"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "IMPL_SHORTFALL"
    SMART_ORDER = "SMART_ORDER"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class RebalanceOrder:
    """Individual rebalancing order"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    target_price: Optional[float]
    strategy: ExecutionStrategy
    priority: int  # 1=highest, 5=lowest
    created_time: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    execution_cost: float = 0.0
    market_impact: float = 0.0

@dataclass
class ExecutionBatch:
    """Batch of orders for coordinated execution"""
    batch_id: str
    orders: List[RebalanceOrder]
    total_value: float
    estimated_impact: float
    execution_timeframe: timedelta
    created_time: datetime
    completion_time: Optional[datetime] = None

class MarketImpactModel:
    """
    Market impact estimation for optimal execution sizing
    """
    
    def __init__(self):
        # Market impact parameters (calibrated to US equities)
        self.temporary_impact_coeff = 0.5
        self.permanent_impact_coeff = 0.1
        self.liquidity_adjustment = {
            'high': 0.5,    # Large cap, high volume
            'medium': 1.0,  # Mid cap, medium volume
            'low': 2.0      # Small cap, low volume
        }
        
    def estimate_market_impact(self, symbol: str, quantity: float, 
                             avg_daily_volume: float, current_price: float,
                             liquidity_tier: str = 'medium') -> Dict[str, float]:
        """
        Estimate market impact for a trade
        
        Args:
            symbol: Stock symbol
            quantity: Shares to trade (signed: positive=buy, negative=sell)
            avg_daily_volume: Average daily volume
            current_price: Current market price
            liquidity_tier: 'high', 'medium', 'low'
            
        Returns:
            Dictionary with impact estimates
        """
        try:
            if avg_daily_volume <= 0:
                return {'temporary_impact': 0.01, 'permanent_impact': 0.005, 'total_cost': quantity * current_price * 0.015}
            
            # Participation rate (fraction of daily volume)
            participation_rate = abs(quantity) / avg_daily_volume
            
            # Liquidity adjustment
            liquidity_mult = self.liquidity_adjustment.get(liquidity_tier, 1.0)
            
            # Temporary impact (recovers after trade)
            temporary_impact = self.temporary_impact_coeff * np.sqrt(participation_rate) * liquidity_mult
            
            # Permanent impact (price moves permanently)
            permanent_impact = self.permanent_impact_coeff * participation_rate * liquidity_mult
            
            # Total impact as percentage of price
            total_impact_pct = temporary_impact + permanent_impact
            
            # Cost in dollars
            trade_value = abs(quantity) * current_price
            impact_cost = trade_value * total_impact_pct
            
            return {
                'temporary_impact': temporary_impact,
                'permanent_impact': permanent_impact,
                'total_impact_pct': total_impact_pct,
                'participation_rate': participation_rate,
                'impact_cost': impact_cost,
                'trade_value': trade_value
            }
            
        except Exception as e:
            logger.error(f"Error estimating market impact for {symbol}: {e}")
            return {'temporary_impact': 0.01, 'permanent_impact': 0.005, 'total_cost': 0.0}
    
    def optimize_execution_schedule(self, orders: List[RebalanceOrder],
                                  market_data: Dict[str, Dict]) -> List[ExecutionBatch]:
        """
        Optimize execution schedule to minimize market impact
        
        Args:
            orders: List of rebalancing orders
            market_data: Market data including volume, volatility, etc.
            
        Returns:
            List of execution batches scheduled over time
        """
        try:
            # Sort orders by priority and market impact
            sorted_orders = sorted(orders, key=lambda x: (x.priority, -abs(x.quantity)))
            
            batches = []
            current_batch = []
            batch_value = 0
            max_batch_value = 10000000  # $10M max per batch
            
            for order in sorted_orders:
                order_value = abs(order.quantity) * market_data.get(order.symbol, {}).get('price', 100)
                
                # Start new batch if current would be too large
                if batch_value + order_value > max_batch_value and current_batch:
                    batch = ExecutionBatch(
                        batch_id=f"batch_{len(batches)+1}_{datetime.now().strftime('%H%M%S')}",
                        orders=current_batch.copy(),
                        total_value=batch_value,
                        estimated_impact=self._estimate_batch_impact(current_batch, market_data),
                        execution_timeframe=timedelta(minutes=15),
                        created_time=datetime.now()
                    )
                    batches.append(batch)
                    current_batch = []
                    batch_value = 0
                
                current_batch.append(order)
                batch_value += order_value
            
            # Add final batch
            if current_batch:
                batch = ExecutionBatch(
                    batch_id=f"batch_{len(batches)+1}_{datetime.now().strftime('%H%M%S')}",
                    orders=current_batch.copy(),
                    total_value=batch_value,
                    estimated_impact=self._estimate_batch_impact(current_batch, market_data),
                    execution_timeframe=timedelta(minutes=15),
                    created_time=datetime.now()
                )
                batches.append(batch)
            
            logger.info(f"Optimized execution into {len(batches)} batches")
            return batches
            
        except Exception as e:
            logger.error(f"Error optimizing execution schedule: {e}")
            # Fallback: single batch
            return [ExecutionBatch(
                batch_id="fallback_batch",
                orders=orders,
                total_value=sum(abs(o.quantity) * 100 for o in orders),
                estimated_impact=0.01,
                execution_timeframe=timedelta(minutes=30),
                created_time=datetime.now()
            )]
    
    def _estimate_batch_impact(self, orders: List[RebalanceOrder], 
                              market_data: Dict[str, Dict]) -> float:
        """Estimate total market impact for a batch of orders"""
        total_impact = 0
        for order in orders:
            symbol_data = market_data.get(order.symbol, {})
            avg_volume = symbol_data.get('avg_daily_volume', 1000000)
            price = symbol_data.get('price', 100)
            
            impact = self.estimate_market_impact(
                order.symbol, order.quantity, avg_volume, price
            )
            total_impact += impact['impact_cost']
        
        return total_impact

class TWAPExecutor:
    """
    Time-Weighted Average Price execution algorithm
    """
    
    def __init__(self, execution_horizon: timedelta = timedelta(minutes=15)):
        self.execution_horizon = execution_horizon
        self.min_slice_size = 100  # Minimum shares per slice
        self.max_slices = 20  # Maximum number of slices
        
    def create_twap_schedule(self, order: RebalanceOrder, 
                           market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create TWAP execution schedule
        
        Args:
            order: Rebalancing order to execute
            market_data: Current market data
            
        Returns:
            List of execution slices with timing
        """
        try:
            total_quantity = abs(order.quantity)
            
            # Determine number of slices
            avg_volume = market_data.get('avg_daily_volume', 1000000)
            participation_limit = 0.05  # Max 5% of daily volume per slice
            max_slice_by_volume = avg_volume * participation_limit
            
            optimal_slices = min(
                self.max_slices,
                max(2, int(total_quantity / max(self.min_slice_size, max_slice_by_volume)))
            )
            
            # Create execution slices
            slice_size = total_quantity / optimal_slices
            slice_interval = self.execution_horizon / optimal_slices
            
            schedule = []
            remaining_quantity = total_quantity
            
            for i in range(optimal_slices):
                # Adjust last slice to handle rounding
                if i == optimal_slices - 1:
                    current_slice_size = remaining_quantity
                else:
                    current_slice_size = min(slice_size, remaining_quantity)
                
                # Add some randomization to avoid gaming
                randomization = np.random.uniform(0.8, 1.2)
                current_slice_size *= randomization
                current_slice_size = max(self.min_slice_size, current_slice_size)
                current_slice_size = min(current_slice_size, remaining_quantity)
                
                execution_time = datetime.now() + (slice_interval * i)
                
                schedule.append({
                    'slice_id': f"{order.order_id}_slice_{i+1}",
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': current_slice_size if order.quantity > 0 else -current_slice_size,
                    'execution_time': execution_time,
                    'strategy': 'MARKET',  # Individual slices use market orders
                    'parent_order_id': order.order_id
                })
                
                remaining_quantity -= current_slice_size
                
                if remaining_quantity <= 0:
                    break
            
            logger.info(f"Created TWAP schedule for {order.symbol}: {len(schedule)} slices over {self.execution_horizon}")
            return schedule
            
        except Exception as e:
            logger.error(f"Error creating TWAP schedule for {order.order_id}: {e}")
            # Fallback: single market order
            return [{
                'slice_id': f"{order.order_id}_single",
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'execution_time': datetime.now(),
                'strategy': 'MARKET',
                'parent_order_id': order.order_id
            }]

class SmartOrderRouter:
    """
    Smart order routing for optimal execution across strategies
    """
    
    def __init__(self):
        self.strategy_thresholds = {
            'market_order_max': 5000,      # Max $5k for market orders
            'twap_min_value': 10000,       # Min $10k for TWAP
            'large_order_threshold': 100000 # $100k+ gets special handling
        }
        
    def determine_execution_strategy(self, order: RebalanceOrder, 
                                   market_data: Dict[str, Any]) -> ExecutionStrategy:
        """
        Determine optimal execution strategy for an order
        
        Args:
            order: Rebalancing order
            market_data: Current market conditions
            
        Returns:
            Optimal execution strategy
        """
        try:
            # Calculate order value
            price = market_data.get('price', order.target_price or 100)
            order_value = abs(order.quantity) * price
            
            # Get market conditions
            volatility = market_data.get('volatility', 0.2)
            avg_volume = market_data.get('avg_daily_volume', 1000000)
            bid_ask_spread = market_data.get('bid_ask_spread', 0.01)
            
            # Participation rate
            participation_rate = abs(order.quantity) / avg_volume if avg_volume > 0 else 0
            
            logger.info(f"Order analysis for {order.symbol}: "
                       f"Value=${order_value:,.0f}, Participation={participation_rate:.4f}, "
                       f"Volatility={volatility:.3f}")
            
            # Strategy selection logic
            if order_value <= self.strategy_thresholds['market_order_max']:
                return ExecutionStrategy.MARKET
            
            elif order_value >= self.strategy_thresholds['large_order_threshold']:
                if participation_rate > 0.05:  # High market impact
                    return ExecutionStrategy.IMPLEMENTATION_SHORTFALL
                else:
                    return ExecutionStrategy.VWAP
            
            elif order_value >= self.strategy_thresholds['twap_min_value']:
                if volatility > 0.3:  # High volatility
                    return ExecutionStrategy.TWAP
                else:
                    return ExecutionStrategy.SMART_ORDER
            
            else:
                return ExecutionStrategy.SMART_ORDER
                
        except Exception as e:
            logger.error(f"Error determining execution strategy: {e}")
            return ExecutionStrategy.MARKET  # Safe fallback
    
    def optimize_order_parameters(self, order: RebalanceOrder, 
                                 strategy: ExecutionStrategy,
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize order parameters for chosen strategy
        
        Returns:
            Dictionary with optimized parameters
        """
        try:
            price = market_data.get('price', order.target_price or 100)
            volatility = market_data.get('volatility', 0.2)
            
            if strategy == ExecutionStrategy.MARKET:
                return {
                    'order_type': 'MARKET',
                    'time_in_force': 'IOC',  # Immediate or Cancel
                    'execution_horizon': timedelta(seconds=30)
                }
            
            elif strategy == ExecutionStrategy.TWAP:
                return {
                    'order_type': 'LIMIT',
                    'limit_price': price * (1.001 if order.quantity > 0 else 0.999),
                    'execution_horizon': timedelta(minutes=15),
                    'slice_count': min(20, max(3, abs(order.quantity) // 500))
                }
            
            elif strategy == ExecutionStrategy.VWAP:
                return {
                    'order_type': 'LIMIT',
                    'limit_price': price * (1.002 if order.quantity > 0 else 0.998),
                    'execution_horizon': timedelta(minutes=30),
                    'volume_participation': 0.05  # 5% of volume
                }
            
            elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
                return {
                    'order_type': 'LIMIT',
                    'limit_price': price * (1.0005 if order.quantity > 0 else 0.9995),
                    'execution_horizon': timedelta(hours=2),
                    'urgency': 'low',
                    'adapt_to_market': True
                }
            
            else:  # SMART_ORDER
                return {
                    'order_type': 'ADAPTIVE',
                    'execution_horizon': timedelta(minutes=10),
                    'max_participation': 0.03,
                    'urgency': 'medium'
                }
                
        except Exception as e:
            logger.error(f"Error optimizing order parameters: {e}")
            return {'order_type': 'MARKET'}

class RebalancingExecutor:
    """
    Main rebalancing execution engine
    Coordinates order creation, optimization, and execution
    """
    
    def __init__(self, ninjatrader_connector=None):
        """
        Initialize rebalancing executor
        
        Args:
            ninjatrader_connector: NinjaTrader connection for order execution
        """
        self.ninjatrader_connector = ninjatrader_connector
        
        # Initialize components
        self.market_impact_model = MarketImpactModel()
        self.twap_executor = TWAPExecutor()
        self.smart_router = SmartOrderRouter()
        
        # State tracking
        self.active_orders = {}
        self.execution_history = []
        self.performance_metrics = {}
        
        # Execution parameters
        self.max_concurrent_orders = 10
        self.order_timeout = timedelta(minutes=30)
        
        # Event callbacks
        self.execution_callbacks = []
        
    async def execute_rebalancing(self, rebalance_signal: RebalanceSignal,
                                 market_data: Dict[str, Dict],
                                 dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing
        
        Args:
            rebalance_signal: Rebalancing signal with trade requirements
            market_data: Current market data for all symbols
            dry_run: If True, simulate execution without placing real orders
            
        Returns:
            Execution results and metrics
        """
        try:
            logger.info(f"Starting rebalancing execution: {rebalance_signal.trigger_reason}")
            
            # Create rebalancing orders
            orders = self._create_rebalancing_orders(rebalance_signal, market_data)
            
            if not orders:
                logger.warning("No orders created for rebalancing")
                return {'status': 'no_orders', 'orders': []}
            
            # Optimize execution strategy for each order
            for order in orders:
                symbol_data = market_data.get(order.symbol, {})
                strategy = self.smart_router.determine_execution_strategy(order, symbol_data)
                order.strategy = strategy
                
                # Optimize parameters
                order_params = self.smart_router.optimize_order_parameters(order, strategy, symbol_data)
                order.target_price = order_params.get('limit_price')
            
            # Create execution batches
            batches = self.market_impact_model.optimize_execution_schedule(orders, market_data)
            
            if dry_run:
                return self._simulate_execution(batches, market_data)
            
            # Execute batches
            execution_results = []
            for batch in batches:
                batch_result = await self._execute_batch(batch, market_data)
                execution_results.append(batch_result)
                
                # Wait between batches to minimize impact
                if batch != batches[-1]:  # Not the last batch
                    await asyncio.sleep(60)  # 1 minute between batches
            
            # Compile results
            total_executed_value = sum(r.get('executed_value', 0) for r in execution_results)
            total_execution_cost = sum(r.get('execution_cost', 0) for r in execution_results)
            
            results = {
                'status': 'completed',
                'orders_created': len(orders),
                'batches_executed': len(execution_results),
                'total_executed_value': total_executed_value,
                'total_execution_cost': total_execution_cost,
                'execution_cost_bps': (total_execution_cost / total_executed_value * 10000) if total_executed_value > 0 else 0,
                'batch_results': execution_results,
                'timestamp': datetime.now()
            }
            
            # Record execution
            self.execution_history.append({
                'rebalance_signal': rebalance_signal,
                'results': results,
                'orders': [asdict(order) for order in orders]
            })
            
            logger.info(f"Rebalancing execution completed: "
                       f"{results['orders_created']} orders, "
                       f"${total_executed_value:,.2f} executed, "
                       f"{results['execution_cost_bps']:.1f} bps cost")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _create_rebalancing_orders(self, rebalance_signal: RebalanceSignal,
                                  market_data: Dict[str, Dict]) -> List[RebalanceOrder]:
        """Create individual orders from rebalancing signal"""
        orders = []
        order_counter = 1
        
        for symbol, trade_shares in rebalance_signal.rebalance_trades.items():
            if abs(trade_shares) < 1:  # Skip tiny trades
                continue
            
            symbol_data = market_data.get(symbol, {})
            current_price = symbol_data.get('price', 100)
            
            # Determine priority based on deviation size
            current_weight = rebalance_signal.current_allocations.get(symbol, 0)
            target_weight = rebalance_signal.target_allocations.get(symbol, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > 0.10:
                priority = 1  # High priority
            elif deviation > 0.05:
                priority = 2  # Medium priority
            else:
                priority = 3  # Low priority
            
            order = RebalanceOrder(
                order_id=f"rebal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order_counter:03d}",
                symbol=symbol,
                side="BUY" if trade_shares > 0 else "SELL",
                quantity=abs(trade_shares),
                target_price=current_price,
                strategy=ExecutionStrategy.SMART_ORDER,  # Will be optimized later
                priority=priority,
                created_time=datetime.now()
            )
            
            orders.append(order)
            order_counter += 1
        
        return orders
    
    async def _execute_batch(self, batch: ExecutionBatch, 
                           market_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute a batch of orders"""
        try:
            logger.info(f"Executing batch {batch.batch_id} with {len(batch.orders)} orders")
            
            batch_results = []
            
            for order in batch.orders:
                if order.strategy == ExecutionStrategy.TWAP:
                    # Execute TWAP strategy
                    result = await self._execute_twap_order(order, market_data)
                else:
                    # Execute as single order
                    result = await self._execute_single_order(order, market_data)
                
                batch_results.append(result)
                
                # Brief pause between orders in batch
                await asyncio.sleep(1)
            
            # Calculate batch metrics
            executed_value = sum(r.get('executed_value', 0) for r in batch_results)
            execution_cost = sum(r.get('execution_cost', 0) for r in batch_results)
            
            batch.completion_time = datetime.now()
            
            return {
                'batch_id': batch.batch_id,
                'orders_executed': len(batch_results),
                'executed_value': executed_value,
                'execution_cost': execution_cost,
                'order_results': batch_results,
                'completion_time': batch.completion_time
            }
            
        except Exception as e:
            logger.error(f"Error executing batch {batch.batch_id}: {e}")
            return {
                'batch_id': batch.batch_id,
                'error': str(e),
                'orders_executed': 0,
                'executed_value': 0,
                'execution_cost': 0
            }
    
    async def _execute_single_order(self, order: RebalanceOrder, 
                                   market_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute a single order"""
        try:
            logger.info(f"Executing order {order.order_id}: {order.side} {order.quantity} {order.symbol}")
            
            if self.ninjatrader_connector and self.ninjatrader_connector.is_connected:
                # Execute through NinjaTrader
                order_id = self.ninjatrader_connector.place_order(
                    instrument=order.symbol,
                    action=order.side,
                    quantity=int(order.quantity),
                    order_type="MARKET" if order.strategy == ExecutionStrategy.MARKET else "LIMIT",
                    price=order.target_price
                )
                
                # Wait for fill (simplified - in production, use event-driven)
                await asyncio.sleep(2)
                
                # Simulate fill for demo
                fill_price = order.target_price * (1 + np.random.normal(0, 0.001))
                execution_cost = abs(order.quantity) * abs(fill_price - order.target_price)
                
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = fill_price
                order.execution_cost = execution_cost
                
            else:
                # Simulate execution for demo
                logger.info(f"Simulating execution for {order.order_id}")
                fill_price = order.target_price * (1 + np.random.normal(0, 0.002))
                execution_cost = abs(order.quantity) * 0.01  # $0.01 per share
                
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = fill_price
                order.execution_cost = execution_cost
            
            executed_value = order.filled_quantity * order.average_fill_price
            
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'average_fill_price': order.average_fill_price,
                'executed_value': executed_value,
                'execution_cost': order.execution_cost
            }
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': 'REJECTED',
                'error': str(e),
                'executed_value': 0,
                'execution_cost': 0
            }
    
    async def _execute_twap_order(self, order: RebalanceOrder, 
                                 market_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute order using TWAP strategy"""
        try:
            logger.info(f"Executing TWAP order {order.order_id}")
            
            symbol_data = market_data.get(order.symbol, {})
            schedule = self.twap_executor.create_twap_schedule(order, symbol_data)
            
            slice_results = []
            total_filled = 0
            total_cost = 0
            
            for slice_info in schedule:
                # Wait until slice execution time
                wait_time = (slice_info['execution_time'] - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(min(wait_time, 30))  # Max 30 second wait
                
                # Execute slice
                slice_order = RebalanceOrder(
                    order_id=slice_info['slice_id'],
                    symbol=slice_info['symbol'],
                    side=slice_info['side'],
                    quantity=abs(slice_info['quantity']),
                    target_price=order.target_price,
                    strategy=ExecutionStrategy.MARKET,
                    priority=order.priority,
                    created_time=datetime.now()
                )
                
                slice_result = await self._execute_single_order(slice_order, market_data)
                slice_results.append(slice_result)
                
                total_filled += slice_result.get('filled_quantity', 0)
                total_cost += slice_result.get('execution_cost', 0)
            
            # Update parent order
            order.status = OrderStatus.FILLED
            order.filled_quantity = total_filled
            order.execution_cost = total_cost
            if total_filled > 0:
                order.average_fill_price = sum(r.get('executed_value', 0) for r in slice_results) / total_filled
            
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': order.status.value,
                'strategy': 'TWAP',
                'slices_executed': len(slice_results),
                'filled_quantity': total_filled,
                'average_fill_price': order.average_fill_price,
                'executed_value': total_filled * order.average_fill_price,
                'execution_cost': total_cost,
                'slice_results': slice_results
            }
            
        except Exception as e:
            logger.error(f"Error executing TWAP order {order.order_id}: {e}")
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'status': 'REJECTED',
                'error': str(e),
                'executed_value': 0,
                'execution_cost': 0
            }
    
    def _simulate_execution(self, batches: List[ExecutionBatch], 
                          market_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Simulate execution for dry run"""
        logger.info("Simulating rebalancing execution (dry run)")
        
        total_orders = sum(len(batch.orders) for batch in batches)
        total_value = sum(batch.total_value for batch in batches)
        estimated_cost = sum(batch.estimated_impact for batch in batches)
        
        return {
            'status': 'simulated',
            'batches': len(batches),
            'total_orders': total_orders,
            'total_value': total_value,
            'estimated_cost': estimated_cost,
            'estimated_cost_bps': (estimated_cost / total_value * 10000) if total_value > 0 else 0,
            'simulation': True
        }
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        if not self.execution_history:
            return {'error': 'No execution history available'}
        
        # Calculate metrics from execution history
        total_executions = len(self.execution_history)
        total_value = sum(e['results'].get('total_executed_value', 0) for e in self.execution_history)
        total_cost = sum(e['results'].get('total_execution_cost', 0) for e in self.execution_history)
        
        avg_cost_bps = (total_cost / total_value * 10000) if total_value > 0 else 0
        
        return {
            'total_executions': total_executions,
            'total_executed_value': total_value,
            'total_execution_cost': total_cost,
            'average_cost_bps': avg_cost_bps,
            'last_execution': self.execution_history[-1]['results']['timestamp'] if self.execution_history else None
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_rebalancing_executor():
        # Create sample rebalancing signal
        rebalance_signal = RebalanceSignal(
            timestamp=datetime.now(),
            trigger_reason="Test rebalancing",
            current_allocations={'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.2, 'JPM': 0.15, 'JNJ': 0.1},
            target_allocations={'AAPL': 0.25, 'MSFT': 0.3, 'GOOGL': 0.2, 'JPM': 0.15, 'JNJ': 0.1},
            rebalance_trades={'AAPL': -500, 'MSFT': 400, 'GOOGL': 0, 'JPM': 0, 'JNJ': 0},
            expected_cost=1500.0,
            priority='medium'
        )
        
        # Sample market data
        market_data = {
            'AAPL': {'price': 175.50, 'avg_daily_volume': 50000000, 'volatility': 0.25, 'bid_ask_spread': 0.01},
            'MSFT': {'price': 335.20, 'avg_daily_volume': 30000000, 'volatility': 0.22, 'bid_ask_spread': 0.02},
            'GOOGL': {'price': 135.80, 'avg_daily_volume': 25000000, 'volatility': 0.28, 'bid_ask_spread': 0.03}
        }
        
        # Initialize executor
        executor = RebalancingExecutor()
        
        # Execute rebalancing (dry run)
        results = await executor.execute_rebalancing(rebalance_signal, market_data, dry_run=True)
        
        print("Rebalancing Execution Results:")
        print(f"Status: {results['status']}")
        print(f"Total Orders: {results.get('total_orders', 0)}")
        print(f"Total Value: ${results.get('total_value', 0):,.2f}")
        print(f"Estimated Cost: ${results.get('estimated_cost', 0):.2f}")
        print(f"Cost (bps): {results.get('estimated_cost_bps', 0):.1f}")
        
        # Get execution metrics
        metrics = executor.get_execution_metrics()
        print(f"\nExecution Metrics: {metrics}")
    
    # Run test
    asyncio.run(test_rebalancing_executor())
