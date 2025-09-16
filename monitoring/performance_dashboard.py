"""
Enhanced Performance Analytics Dashboard with Real-time Visualization
Provides comprehensive performance tracking, risk analytics, and visualization for institutional trading systems.
Features: Sharpe/Sortino ratios, drawdown analysis, real-time P&L tracking, trade analytics, risk heatmaps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.extend([project_root, os.path.join(project_root, 'risk-engine', 'advanced')])

try:
    from advanced_risk_manager import AdvancedRiskManager
except ImportError:
    st.error("⚠️ Advanced Risk Manager not found. Please ensure the risk engine is properly installed.")
    AdvancedRiskManager = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """
    Advanced performance analytics engine for institutional trading systems.
    Tracks P&L, risk metrics, trade statistics, and generates comprehensive reports.
    """
    
    def __init__(self):
        """Initialize performance analytics with default settings"""
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.risk_manager = AdvancedRiskManager() if AdvancedRiskManager else None
        logger.info("📊 Performance Analytics initialized")
    
    def add_trade(self, trade_data: Dict) -> None:
        """
        Add a completed trade to the performance tracking system
        
        Args:
            trade_data: Dictionary containing trade information
                       Required: symbol, entry_time, exit_time, entry_price, exit_price, 
                                quantity, side (LONG/SHORT), pnl
        """
        required_fields = ['symbol', 'entry_time', 'exit_time', 'entry_price', 
                          'exit_price', 'quantity', 'side', 'pnl']
        
        if not all(field in trade_data for field in required_fields):
            missing = [f for f in required_fields if f not in trade_data]
            logger.error(f"❌ Missing required trade fields: {missing}")
            return
        
        # Calculate additional metrics
        trade_data['duration'] = (trade_data['exit_time'] - trade_data['entry_time']).total_seconds() / 3600  # hours
        trade_data['return_pct'] = (trade_data['pnl'] / (trade_data['entry_price'] * trade_data['quantity'])) * 100
        trade_data['trade_id'] = len(self.trades) + 1
        
        self.trades.append(trade_data)
        logger.info(f"📈 Trade added: {trade_data['symbol']} P&L: ${trade_data['pnl']:.2f}")
    
    def update_portfolio_snapshot(self, portfolio_value: float, positions: Dict, timestamp: datetime = None) -> None:
        """
        Update portfolio snapshot for performance tracking
        
        Args:
            portfolio_value: Current total portfolio value
            positions: Dictionary of current positions {symbol: quantity}
            timestamp: Snapshot timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        snapshot = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'positions': positions.copy(),
            'total_exposure': sum(abs(qty) for qty in positions.values())
        }
        
        self.portfolio_history.append(snapshot)
        logger.info(f"📊 Portfolio snapshot: ${portfolio_value:,.2f}")
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary containing all performance statistics
        """
        if not self.trades:
            return self._empty_metrics()
        
        df_trades = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Return metrics
        returns = df_trades['return_pct'].values
        avg_return = np.mean(returns)
        return_std = np.std(returns) if len(returns) > 1 else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = (avg_return / return_std) if return_std > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
        sortino_ratio = (avg_return / downside_std) if downside_std > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Duration metrics
        avg_trade_duration = df_trades['duration'].mean()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_return': avg_return,
            'return_volatility': return_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': avg_trade_duration,
            'best_trade': df_trades['pnl'].max(),
            'worst_trade': df_trades['pnl'].min(),
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary when no trades exist"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'avg_return': 0, 'return_volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'max_drawdown': 0, 'avg_trade_duration': 0, 'best_trade': 0, 'worst_trade': 0,
            'total_return': 0
        }
    
    def generate_trade_analysis(self) -> pd.DataFrame:
        """
        Generate detailed trade analysis DataFrame
        
        Returns:
            DataFrame with trade statistics by symbol, time period, etc.
        """
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Symbol analysis
        symbol_stats = df.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean', 'std'],
            'return_pct': ['mean', 'std'],
            'duration': 'mean'
        }).round(2)
        
        # Flatten column names
        symbol_stats.columns = ['_'.join(col).strip() for col in symbol_stats.columns]
        symbol_stats = symbol_stats.reset_index()
        
        # Add win rates per symbol
        win_rates = df.groupby('symbol').apply(
            lambda x: (x['pnl'] > 0).mean() * 100
        ).reset_index(name='win_rate')
        
        symbol_stats = symbol_stats.merge(win_rates, on='symbol')
        
        return symbol_stats
    
    def create_equity_curve(self) -> go.Figure:
        """
        Create interactive equity curve visualization
        
        Returns:
            Plotly figure showing portfolio equity progression
        """
        if not self.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.trades)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = 100000 + df['cumulative_pnl']  # Assuming $100k starting capital
        
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=df['equity'],
            mode='lines',
            name='Portfolio Equity',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add drawdown fill
        running_max = df['equity'].expanding().max()
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=running_max,
            mode='lines',
            name='Running High',
            line=dict(color='#A23B72', width=1, dash='dash'),
            hovertemplate='Date: %{x}<br>High: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_returns_distribution(self) -> go.Figure:
        """
        Create returns distribution histogram
        
        Returns:
            Plotly figure showing trade returns distribution
        """
        if not self.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.trades)
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df['return_pct'],
            nbinsx=30,
            name='Returns Distribution',
            marker_color='#F18F01',
            opacity=0.7
        ))
        
        # Add vertical lines for mean and std
        mean_return = df['return_pct'].mean()
        std_return = df['return_pct'].std()
        
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_return:.2f}%")
        fig.add_vline(x=mean_return + std_return, line_dash="dot", line_color="orange",
                     annotation_text=f"+1σ: {mean_return + std_return:.2f}%")
        fig.add_vline(x=mean_return - std_return, line_dash="dot", line_color="orange",
                     annotation_text=f"-1σ: {mean_return - std_return:.2f}%")
        
        fig.update_layout(
            title='Trade Returns Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_monthly_heatmap(self) -> go.Figure:
        """
        Create monthly returns heatmap
        
        Returns:
            Plotly figure showing monthly performance heatmap
        """
        if not self.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.trades)
        df['year'] = df['exit_time'].dt.year
        df['month'] = df['exit_time'].dt.month
        
        # Calculate monthly returns
        monthly_returns = df.groupby(['year', 'month'])['return_pct'].sum().reset_index()
        
        # Create pivot table for heatmap
        heatmap_data = monthly_returns.pivot(index='year', columns='month', values='return_pct')
        
        # Month names for x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[month_names[i-1] for i in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=heatmap_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        return fig

class PerformanceDashboard:
    """
    Streamlit-based interactive performance dashboard
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.analytics = PerformanceAnalytics()
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample trading data for demonstration"""
        # Generate sample trades
        np.random.seed(42)
        base_time = datetime.now() - timedelta(days=90)
        
        symbols = ['ES', 'NQ', 'YM', 'RTY']
        sides = ['LONG', 'SHORT']
        
        for i in range(50):  # 50 sample trades
            entry_time = base_time + timedelta(hours=i*4 + np.random.randint(0, 4))
            exit_time = entry_time + timedelta(hours=np.random.randint(1, 48))
            
            symbol = np.random.choice(symbols)
            side = np.random.choice(sides)
            quantity = np.random.randint(1, 5)
            entry_price = 4400 + np.random.normal(0, 50)
            
            # Simulate realistic returns with slight positive bias
            return_pct = np.random.normal(0.3, 2.5)  # 0.3% average with 2.5% volatility
            exit_price = entry_price * (1 + return_pct/100)
            
            pnl = (exit_price - entry_price) * quantity * (1 if side == 'LONG' else -1)
            
            trade_data = {
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'side': side,
                'pnl': pnl
            }
            
            self.analytics.add_trade(trade_data)
        
        logger.info("📊 Sample trading data loaded successfully")
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Institutional ML Trading - Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Institutional ML Trading Performance Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🔧 Dashboard Controls")
            
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", date.today() - timedelta(days=90))
            with col2:
                end_date = st.date_input("End Date", date.today())
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            # Risk settings (if risk manager available)
            if self.analytics.risk_manager:
                st.subheader("🛡️ Risk Settings")
                var_confidence = st.slider("VaR Confidence (%)", 90, 99, 95)
                max_position = st.slider("Max Position Size (%)", 5, 25, 15)
        
        # Main dashboard content
        self._render_overview()
        self._render_performance_metrics()
        self._render_charts()
        self._render_trade_analysis()
        self._render_risk_metrics()
    
    def _render_overview(self):
        """Render institutional-grade overview section with key metrics"""
        # Main header with system status
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.title("🏦 Institutional ML Trading Dashboard")
        with col2:
            st.metric("🟢 System Status", "LIVE", delta="Active")
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("⏰ Time", current_time)
        
        st.markdown("---")
        
        # Performance Overview
        st.header("📈 Performance Overview")
        
        metrics = self.analytics.calculate_performance_metrics()
        
        # Top row - Primary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            pnl_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
            st.metric(
                label="💰 Total P&L",
                value=f"${metrics['total_pnl']:,.2f}",
                delta=f"{metrics['total_return']:.2f}%" if metrics['total_return'] != 0 else None,
                delta_color=pnl_color
            )
        
        with col2:
            win_rate_color = "normal" if metrics['win_rate'] >= 50 else "inverse"
            st.metric(
                label="🎯 Win Rate",
                value=f"{metrics['win_rate']:.1f}%",
                delta=f"{metrics['winning_trades']}/{metrics['total_trades']} trades",
                delta_color=win_rate_color
            )
        
        with col3:
            sharpe_color = "normal" if metrics['sharpe_ratio'] > 0 else "inverse"
            st.metric(
                label="📊 Sharpe Ratio",
                value=f"{metrics['sharpe_ratio']:.3f}",
                delta="Risk-adjusted return",
                delta_color=sharpe_color
            )
        
        with col4:
            dd_color = "inverse" if abs(metrics['max_drawdown']) > 5 else "normal"
            st.metric(
                label="📉 Max Drawdown",
                value=f"{metrics['max_drawdown']:.2f}%",
                delta="Peak-to-trough",
                delta_color=dd_color
            )
        
        with col5:
            pf_color = "normal" if metrics['profit_factor'] > 1.5 else "inverse"
            st.metric(
                label="⚖️ Profit Factor",
                value=f"{metrics['profit_factor']:.2f}",
                delta="Win/Loss ratio",
                delta_color=pf_color
            )
        
        # Second row - Additional key metrics
        st.markdown("### 📊 Additional Performance Metrics")
        col6, col7, col8, col9, col10 = st.columns(5)
        
        with col6:
            st.metric(
                label="📈 Sortino Ratio",
                value=f"{metrics['sortino_ratio']:.3f}",
                delta="Downside risk-adjusted"
            )
        
        with col7:
            st.metric(
                label="🎲 Return Volatility",
                value=f"{metrics['return_volatility']:.2f}%",
                delta="Risk measure"
            )
        
        with col8:
            st.metric(
                label="🏆 Best Trade",
                value=f"${metrics['best_trade']:,.2f}",
                delta="Single trade high"
            )
        
        with col9:
            st.metric(
                label="💥 Worst Trade",
                value=f"${metrics['worst_trade']:,.2f}",
                delta="Single trade low"
            )
        
        with col10:
            avg_duration_days = metrics['avg_trade_duration'] / 24
            st.metric(
                label="⏱️ Avg Duration",
                value=f"{avg_duration_days:.1f} days",
                delta="Trade holding period"
            )
        
        # Performance rating
        st.markdown("### 🎯 Performance Rating")
        rating_col1, rating_col2, rating_col3 = st.columns(3)
        
        with rating_col1:
            # Overall performance score
            score = 0
            if metrics['sharpe_ratio'] > 1.0: score += 25
            if metrics['win_rate'] > 55: score += 25
            if abs(metrics['max_drawdown']) < 10: score += 25
            if metrics['profit_factor'] > 1.5: score += 25
            
            if score >= 75:
                st.success(f"🌟 EXCELLENT ({score}/100)")
            elif score >= 50:
                st.info(f"📈 GOOD ({score}/100)")
            elif score >= 25:
                st.warning(f"⚠️ MODERATE ({score}/100)")
            else:
                st.error(f"🔴 NEEDS IMPROVEMENT ({score}/100)")
        
        with rating_col2:
            # Risk assessment
            if abs(metrics['max_drawdown']) < 5:
                st.success("🛡️ LOW RISK")
            elif abs(metrics['max_drawdown']) < 10:
                st.info("📊 MODERATE RISK")
            else:
                st.error("⚠️ HIGH RISK")
        
        with rating_col3:
            # Consistency rating
            if metrics['sharpe_ratio'] > 1.5 and metrics['win_rate'] > 60:
                st.success("🎯 HIGHLY CONSISTENT")
            elif metrics['sharpe_ratio'] > 1.0 and metrics['win_rate'] > 50:
                st.info("📈 CONSISTENT")
            else:
                st.warning("📊 VARIABLE PERFORMANCE")
    
    def _render_performance_metrics(self):
        """Render detailed performance metrics"""
        st.header("📊 Detailed Performance Metrics")
        
        metrics = self.analytics.calculate_performance_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Trading Statistics")
            st.write(f"**Total Trades:** {metrics['total_trades']}")
            st.write(f"**Winning Trades:** {metrics['winning_trades']}")
            st.write(f"**Losing Trades:** {metrics['losing_trades']}")
            st.write(f"**Average Win:** ${metrics['avg_win']:,.2f}")
            st.write(f"**Average Loss:** ${metrics['avg_loss']:,.2f}")
            st.write(f"**Best Trade:** ${metrics['best_trade']:,.2f}")
            st.write(f"**Worst Trade:** ${metrics['worst_trade']:,.2f}")
        
        with col2:
            st.subheader("📊 Risk Metrics")
            st.write(f"**Average Return:** {metrics['avg_return']:.2f}%")
            st.write(f"**Return Volatility:** {metrics['return_volatility']:.2f}%")
            st.write(f"**Sortino Ratio:** {metrics['sortino_ratio']:.2f}")
            st.write(f"**Average Trade Duration:** {metrics['avg_trade_duration']:.1f} hours")
            
            # Risk assessment
            if metrics['sharpe_ratio'] > 1.5:
                st.success("🟢 Excellent risk-adjusted performance")
            elif metrics['sharpe_ratio'] > 1.0:
                st.info("🔵 Good risk-adjusted performance")
            elif metrics['sharpe_ratio'] > 0.5:
                st.warning("🟡 Moderate risk-adjusted performance")
            else:
                st.error("🔴 Poor risk-adjusted performance")
    
    def _render_charts(self):
        """Render enhanced performance charts with institutional standards"""
        st.header("📈 Performance Analytics")
        
        # Chart selection tabs
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
            "📈 Equity & Drawdown", "📊 Returns Analysis", "🔥 Performance Heatmaps", "⚖️ Risk Analytics"
        ])
        
        with chart_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Equity Curve")
                equity_fig = self.create_equity_curve()
                st.plotly_chart(equity_fig, width='stretch')
            
            with col2:
                st.subheader("Drawdown Analysis")
                drawdown_fig = self.create_drawdown_chart()
                st.plotly_chart(drawdown_fig, width='stretch')
        
        with chart_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Returns Distribution")
                returns_fig = self.analytics.create_returns_distribution()
                st.plotly_chart(returns_fig, width='stretch')
            
            with col2:
                st.subheader("Rolling Performance")
                rolling_fig = self.create_rolling_metrics()
                st.plotly_chart(rolling_fig, width='stretch')
        
        with chart_tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Monthly Returns Heatmap")
                heatmap_fig = self.analytics.create_monthly_heatmap()
                st.plotly_chart(heatmap_fig, width='stretch')
            
            with col2:
                st.subheader("Trade Performance by Hour")
                hourly_fig = self.create_hourly_performance()
                st.plotly_chart(hourly_fig, width='stretch')
        
        with chart_tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk-Return Scatter")
                risk_return_fig = self.create_risk_return_scatter()
                st.plotly_chart(risk_return_fig, width='stretch')
            
            with col2:
                st.subheader("VaR Analysis")
                var_fig = self.create_var_analysis()
                st.plotly_chart(var_fig, width='stretch')
    
    def create_drawdown_chart(self) -> go.Figure:
        """Create drawdown analysis chart"""
        if not self.analytics.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.analytics.trades)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = 100000 + df['cumulative_pnl']
        
        # Calculate drawdown
        running_max = df['equity'].expanding().max()
        drawdown = ((df['equity'] - running_max) / running_max) * 100
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Equity Curve', 'Drawdown %'),
                           vertical_spacing=0.1)
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#2E86AB', width=2)
        ), row=1, col=1)
        
        # Drawdown
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='#A23B72', width=1),
            fillcolor='rgba(162, 59, 114, 0.3)'
        ), row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_rolling_metrics(self) -> go.Figure:
        """Create rolling performance metrics chart"""
        if not self.analytics.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.analytics.trades)
        
        # Calculate rolling metrics (10-trade window)
        window = min(10, len(df))
        rolling_returns = df['return_pct'].rolling(window=window).mean()
        rolling_volatility = df['return_pct'].rolling(window=window).std()
        rolling_sharpe = rolling_returns / rolling_volatility
        
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Rolling Returns', 'Rolling Volatility', 'Rolling Sharpe'),
                           vertical_spacing=0.08)
        
        # Rolling returns
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=rolling_returns,
            mode='lines',
            name='Rolling Returns',
            line=dict(color='#2E86AB', width=2)
        ), row=1, col=1)
        
        # Rolling volatility
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=rolling_volatility,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='#F18F01', width=2)
        ), row=2, col=1)
        
        # Rolling Sharpe
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='#A23B72', width=2)
        ), row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def create_hourly_performance(self) -> go.Figure:
        """Create hourly performance analysis"""
        if not self.analytics.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.analytics.trades)
        df['hour'] = df['entry_time'].dt.hour
        
        hourly_stats = df.groupby('hour').agg({
            'pnl': ['sum', 'count', 'mean'],
            'return_pct': 'mean'
        }).round(2)
        
        hourly_stats.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'avg_return']
        hourly_stats = hourly_stats.reset_index()
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('P&L by Hour', 'Trade Count by Hour'),
                           vertical_spacing=0.1)
        
        # P&L by hour
        fig.add_trace(go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['total_pnl'],
            name='Total P&L',
            marker_color='#2E86AB'
        ), row=1, col=1)
        
        # Trade count by hour
        fig.add_trace(go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['trade_count'],
            name='Trade Count',
            marker_color='#F18F01'
        ), row=2, col=1)
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="# Trades", row=2, col=1)
        
        return fig
    
    def create_risk_return_scatter(self) -> go.Figure:
        """Create risk-return scatter plot by instrument"""
        if not self.analytics.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.analytics.trades)
        
        symbol_stats = df.groupby('symbol').agg({
            'return_pct': ['mean', 'std'],
            'pnl': 'count'
        }).round(3)
        
        symbol_stats.columns = ['avg_return', 'volatility', 'trade_count']
        symbol_stats = symbol_stats.reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=symbol_stats['volatility'],
            y=symbol_stats['avg_return'],
            mode='markers+text',
            text=symbol_stats['symbol'],
            textposition="top center",
            marker=dict(
                size=symbol_stats['trade_count'] * 2,
                color=symbol_stats['avg_return'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Avg Return %")
            ),
            name='Instruments'
        ))
        
        fig.update_layout(
            title='Risk-Return Profile by Instrument',
            xaxis_title='Volatility (%)',
            yaxis_title='Average Return (%)',
            height=400
        )
        
        return fig
    
    def create_var_analysis(self) -> go.Figure:
        """Create VaR analysis visualization"""
        if not self.analytics.trades or not self.analytics.risk_manager:
            fig = go.Figure()
            fig.add_annotation(text="No risk data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Sample returns for VaR calculation
        df = pd.DataFrame(self.analytics.trades)
        returns = df['return_pct'].values / 100  # Convert to decimal
        
        try:
            hist_var, hist_es = self.analytics.risk_manager.calculate_historical_var(returns)
            param_var, param_es = self.analytics.risk_manager.calculate_parametric_var(returns)
            mc_var, mc_es = self.analytics.risk_manager.calculate_monte_carlo_var(returns)
            
            var_methods = ['Historical', 'Parametric', 'Monte Carlo']
            var_values = [hist_var, param_var, mc_var]
            es_values = [hist_es, param_es, mc_es]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=var_methods,
                y=var_values,
                name='VaR (95%)',
                marker_color='#A23B72'
            ))
            
            fig.add_trace(go.Bar(
                x=var_methods,
                y=es_values,
                name='Expected Shortfall',
                marker_color='#F18F01'
            ))
            
            fig.update_layout(
                title='Value at Risk Analysis (95% Confidence)',
                yaxis_title='Risk (%)',
                height=400,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"VaR calculation error: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def _render_trade_analysis(self):
        """Render enhanced trade analysis with institutional standards"""
        st.header("🎯 Trade Analysis & Intelligence")
        
        # Trade analysis tabs
        trade_tab1, trade_tab2, trade_tab3, trade_tab4 = st.tabs([
            "📋 Trade Summary", "📊 Performance Breakdown", "⏱️ Duration Analysis", "🎯 Pattern Recognition"
        ])
        
        with trade_tab1:
            self._render_trade_summary_table()
        
        with trade_tab2:
            self._render_performance_breakdown()
        
        with trade_tab3:
            self._render_duration_analysis()
        
        with trade_tab4:
            self._render_pattern_analysis()
    
    def create_equity_curve(self) -> go.Figure:
        """Create equity curve visualization"""
        if not self.analytics.trades:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        df = pd.DataFrame(self.analytics.trades)
        df = df.sort_values('exit_time')
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = 100000 + df['cumulative_pnl']  # Starting with $100,000
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['exit_time'],
            y=df['equity'],
            mode='lines',
            name='Portfolio Equity',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Add starting point
        if len(df) > 0:
            fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                         annotation_text="Starting Capital ($100,000)")
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _render_trade_summary_table(self):
        """Render enhanced trade summary table"""
        if not self.analytics.trades:
            st.warning("📊 No trade data available for analysis")
            return
        
        # Trade summary filters
        col1, col2, col3, col4 = st.columns(4)
        
        df = pd.DataFrame(self.analytics.trades)
        
        with col1:
            symbols = ['All'] + list(df['symbol'].unique())
            selected_symbol = st.selectbox("🎯 Instrument", symbols)
        
        with col2:
            trade_types = ['All', 'Long', 'Short']
            selected_type = st.selectbox("📈 Trade Type", trade_types)
        
        with col3:
            status_options = ['All', 'Profitable', 'Losing']
            selected_status = st.selectbox("💰 Outcome", status_options)
        
        with col4:
            date_range = st.selectbox("📅 Period", ['All Time', 'Last 30 Days', 'Last 7 Days', 'Today'])
        
        # Filter data
        filtered_df = df.copy()
        
        if selected_symbol != 'All':
            filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
        
        if selected_type != 'All':
            trade_direction = 'long' if selected_type == 'Long' else 'short'
            filtered_df = filtered_df[filtered_df['side'] == trade_direction]
        
        if selected_status != 'All':
            if selected_status == 'Profitable':
                filtered_df = filtered_df[filtered_df['pnl'] > 0]
            else:
                filtered_df = filtered_df[filtered_df['pnl'] <= 0]
        
        # Apply date filter
        if date_range != 'All Time':
            now = datetime.now()
            if date_range == 'Today':
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_range == 'Last 7 Days':
                start_date = now - timedelta(days=7)
            else:  # Last 30 Days
                start_date = now - timedelta(days=30)
            
            filtered_df = filtered_df[filtered_df['exit_time'] >= start_date]
        
        # Display filtered results
        st.subheader(f"📊 Trade Results ({len(filtered_df)} trades)")
        
        if len(filtered_df) > 0:
            # Prepare display dataframe
            display_df = filtered_df.copy()
            display_df = display_df.sort_values('exit_time', ascending=False)
            
            # Format columns for display
            display_df['Entry Time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Exit Time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Symbol'] = display_df['symbol']
            display_df['Side'] = display_df['side'].str.title()
            display_df['Quantity'] = display_df['quantity']
            display_df['Entry Price'] = display_df['entry_price'].round(2)
            display_df['Exit Price'] = display_df['exit_price'].round(2)
            display_df['P&L'] = display_df['pnl'].round(2)
            display_df['Return %'] = display_df['return_pct'].round(2)
            display_df['Duration (min)'] = ((display_df['exit_time'] - display_df['entry_time']).dt.total_seconds() / 60).round(1)
            
            # Select columns for display
            display_columns = ['Entry Time', 'Exit Time', 'Symbol', 'Side', 'Quantity', 
                              'Entry Price', 'Exit Price', 'P&L', 'Return %', 'Duration (min)']
            
            # Color-code P&L
            def color_pnl(val):
                if pd.isna(val):
                    return ''
                color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#6c757d'
                return f'color: {color}; font-weight: bold'
            
            styled_df = display_df[display_columns].style.applymap(color_pnl, subset=['P&L', 'Return %'])
            
            st.dataframe(styled_df, width='stretch', height=400)
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = filtered_df['pnl'].sum()
                pnl_color = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
                st.metric("Total P&L", f"${total_pnl:,.2f}", delta=None, delta_color="off")
                st.write(f"{pnl_color} Overall Performance")
            
            with col2:
                win_rate = (filtered_df['pnl'] > 0).mean() * 100
                color = "🟢" if win_rate > 60 else "🟡" if win_rate > 40 else "🔴"
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.write(f"{color} Success Rate")
            
            with col3:
                avg_return = filtered_df['return_pct'].mean()
                return_color = "🟢" if avg_return > 0 else "🔴" if avg_return < 0 else "⚪"
                st.metric("Avg Return", f"{avg_return:.2f}%")
                st.write(f"{return_color} Average Performance")
            
            with col4:
                avg_duration = (filtered_df['exit_time'] - filtered_df['entry_time']).dt.total_seconds().mean() / 60
                duration_color = "🟢" if avg_duration < 60 else "🟡" if avg_duration < 240 else "🔴"
                st.metric("Avg Duration", f"{avg_duration:.1f} min")
                st.write(f"{duration_color} Trade Length")
        
        else:
            st.info("📊 No trades match the selected filters")
    
    def _render_performance_breakdown(self):
        """Render performance breakdown analysis"""
        if not self.analytics.trades:
            st.warning("📊 No trade data available")
            return
        
        df = pd.DataFrame(self.analytics.trades)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Performance by Instrument")
            symbol_stats = df.groupby('symbol').agg({
                'pnl': ['sum', 'count', 'mean'],
                'return_pct': 'mean'
            }).round(2)
            
            symbol_stats.columns = ['Total P&L', 'Trade Count', 'Avg P&L', 'Avg Return %']
            st.dataframe(symbol_stats, width='stretch')
        
        with col2:
            st.subheader("🎯 Performance by Direction")
            side_stats = df.groupby('side').agg({
                'pnl': ['sum', 'count', 'mean'],
                'return_pct': 'mean'
            }).round(2)
            
            side_stats.columns = ['Total P&L', 'Trade Count', 'Avg P&L', 'Avg Return %']
            st.dataframe(side_stats, width='stretch')
        
        # Performance distribution
        st.subheader("📊 P&L Distribution Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Winning trades
            winning_trades = df[df['pnl'] > 0]
            if len(winning_trades) > 0:
                st.success("🎯 Winning Trades")
                st.write(f"Count: {len(winning_trades)}")
                st.write(f"Total P&L: ${winning_trades['pnl'].sum():,.2f}")
                st.write(f"Avg P&L: ${winning_trades['pnl'].mean():,.2f}")
                st.write(f"Best Trade: ${winning_trades['pnl'].max():,.2f}")
        
        with col2:
            # Losing trades
            losing_trades = df[df['pnl'] < 0]
            if len(losing_trades) > 0:
                st.error("🎯 Losing Trades")
                st.write(f"Count: {len(losing_trades)}")
                st.write(f"Total P&L: ${losing_trades['pnl'].sum():,.2f}")
                st.write(f"Avg P&L: ${losing_trades['pnl'].mean():,.2f}")
                st.write(f"Worst Trade: ${losing_trades['pnl'].min():,.2f}")
        
        with col3:
            # Breakeven trades
            breakeven_trades = df[df['pnl'] == 0]
            if len(breakeven_trades) > 0:
                st.info("⚪ Breakeven Trades")
                st.write(f"Count: {len(breakeven_trades)}")
                st.write(f"Percentage: {(len(breakeven_trades)/len(df)*100):.1f}%")
    
    def _render_duration_analysis(self):
        """Render trade duration analysis"""
        if not self.analytics.trades:
            st.warning("📊 No trade data available")
            return
        
        df = pd.DataFrame(self.analytics.trades)
        df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        df['duration_hours'] = df['duration_minutes'] / 60
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Duration Statistics")
            
            duration_stats = {
                'Average Duration': f"{df['duration_minutes'].mean():.1f} minutes",
                'Median Duration': f"{df['duration_minutes'].median():.1f} minutes",
                'Shortest Trade': f"{df['duration_minutes'].min():.1f} minutes",
                'Longest Trade': f"{df['duration_minutes'].max():.1f} minutes",
                'Std Deviation': f"{df['duration_minutes'].std():.1f} minutes"
            }
            
            for stat, value in duration_stats.items():
                st.write(f"**{stat}:** {value}")
        
        with col2:
            st.subheader("📊 Duration vs Performance")
            
            # Categorize by duration
            df['duration_category'] = pd.cut(df['duration_minutes'], 
                                           bins=[0, 5, 15, 60, float('inf')],
                                           labels=['< 5 min', '5-15 min', '15-60 min', '> 60 min'])
            
            duration_perf = df.groupby('duration_category').agg({
                'pnl': ['mean', 'sum', 'count'],
                'return_pct': 'mean'
            }).round(2)
            
            duration_perf.columns = ['Avg P&L', 'Total P&L', 'Count', 'Avg Return %']
            st.dataframe(duration_perf, width='stretch')
        
        # Duration distribution chart
        st.subheader("📈 Trade Duration Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['duration_minutes'],
            nbinsx=20,
            marker_color='#2E86AB',
            opacity=0.7,
            name='Duration Distribution'
        ))
        
        fig.update_layout(
            title='Trade Duration Distribution',
            xaxis_title='Duration (minutes)',
            yaxis_title='Number of Trades',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def _render_pattern_analysis(self):
        """Render pattern recognition analysis"""
        if not self.analytics.trades:
            st.warning("📊 No trade data available")
            return
        
        df = pd.DataFrame(self.analytics.trades)
        
        st.subheader("🎯 Trading Pattern Recognition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🕐 Trading Hours Analysis**")
            
            df['hour'] = df['entry_time'].dt.hour
            hourly_performance = df.groupby('hour').agg({
                'pnl': 'mean',
                'return_pct': 'mean'
            }).round(2)
            
            best_hour = hourly_performance['pnl'].idxmax()
            worst_hour = hourly_performance['pnl'].idxmin()
            
            st.write(f"🟢 **Best Hour:** {best_hour}:00 (Avg P&L: ${hourly_performance.loc[best_hour, 'pnl']:.2f})")
            st.write(f"🔴 **Worst Hour:** {worst_hour}:00 (Avg P&L: ${hourly_performance.loc[worst_hour, 'pnl']:.2f})")
            
        with col2:
            st.write("**📅 Day of Week Analysis**")
            
            df['day_of_week'] = df['entry_time'].dt.day_name()
            daily_performance = df.groupby('day_of_week').agg({
                'pnl': 'mean',
                'return_pct': 'mean'
            }).round(2)
            
            best_day = daily_performance['pnl'].idxmax()
            worst_day = daily_performance['pnl'].idxmin()
            
            st.write(f"🟢 **Best Day:** {best_day} (Avg P&L: ${daily_performance.loc[best_day, 'pnl']:.2f})")
            st.write(f"🔴 **Worst Day:** {worst_day} (Avg P&L: ${daily_performance.loc[worst_day, 'pnl']:.2f})")
        
        # Streak analysis
        st.subheader("🔥 Winning/Losing Streaks")
        
        # Calculate streaks
        df = df.sort_values('exit_time')
        df['win'] = df['pnl'] > 0
        df['streak_id'] = (df['win'] != df['win'].shift()).cumsum()
        
        streaks = df.groupby('streak_id').agg({
            'win': 'first',
            'pnl': 'sum'
        })
        
        winning_streaks = streaks[streaks['win']]['pnl'].value_counts().sort_index(ascending=False)
        losing_streaks = streaks[~streaks['win']]['pnl'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(winning_streaks) > 0:
                max_win_streak = len(streaks[streaks['win']].groupby(level=0).first())
                st.success(f"🔥 **Longest Winning Streak:** {max_win_streak} trades")
            else:
                st.write("No winning streaks recorded")
        
        with col2:
            if len(losing_streaks) > 0:
                max_lose_streak = len(streaks[~streaks['win']].groupby(level=0).first())
                st.error(f"❄️ **Longest Losing Streak:** {max_lose_streak} trades")
            else:
                st.write("No losing streaks recorded")
        
        # Performance consistency
        st.subheader("📊 Performance Consistency")
        
        consistency_metrics = {
            'Win Rate Stability': f"{(df['pnl'] > 0).rolling(10).mean().std():.3f}",
            'Return Volatility': f"{df['return_pct'].std():.2f}%",
            'P&L Volatility': f"${df['pnl'].std():.2f}",
            'Sharpe Ratio': f"{(df['return_pct'].mean() / df['return_pct'].std()):.2f}" if df['return_pct'].std() > 0 else "N/A"
        }
        
        for metric, value in consistency_metrics.items():
            st.write(f"**{metric}:** {value}")
    
    def _render_risk_metrics(self):
        """Render risk management metrics"""
        st.header("🛡️ Risk Management")
        
        if self.analytics.risk_manager:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Risk Metrics")
                
                # Sample portfolio data for risk calculation
                sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
                
                try:
                    # Calculate VaR
                    hist_var, hist_es = self.analytics.risk_manager.calculate_historical_var(sample_returns)
                    param_var, param_es = self.analytics.risk_manager.calculate_parametric_var(sample_returns)
                    mc_var, mc_es = self.analytics.risk_manager.calculate_monte_carlo_var(sample_returns)
                    
                    st.write(f"**Historical VaR (95%):** {hist_var:.2f}%")
                    st.write(f"**Parametric VaR (95%):** {param_var:.2f}%")
                    st.write(f"**Monte Carlo VaR (95%):** {mc_var:.2f}%")
                    st.write(f"**Expected Shortfall:** {hist_es:.2f}%")
                    
                except Exception as e:
                    st.error(f"Risk calculation error: {e}")
            
            with col2:
                st.subheader("Position Sizing")
                
                # Sample Kelly Criterion calculation
                try:
                    win_rate = 0.55
                    avg_win = 100
                    avg_loss = -80
                    kelly_size = self.analytics.risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                    
                    st.write(f"**Kelly Optimal Size:** {kelly_size:.1f}%")
                    st.write(f"**Current Win Rate:** {win_rate*100:.1f}%")
                    st.write(f"**Risk/Reward Ratio:** {abs(avg_win/avg_loss):.2f}")
                    
                    # Position size recommendation
                    if kelly_size > 10:
                        st.warning("🟡 High Kelly size - consider risk management")
                    elif kelly_size > 5:
                        st.info("🔵 Moderate Kelly size - reasonable position")
                    else:
                        st.success("🟢 Conservative Kelly size - low risk")
                        
                except Exception as e:
                    st.error(f"Kelly calculation error: {e}")
        else:
            st.warning("⚠️ Risk manager not available. Install advanced risk management module for full functionality.")

def main():
    """Main function to run the dashboard"""
    dashboard = PerformanceDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
