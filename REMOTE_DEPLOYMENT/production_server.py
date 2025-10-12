"""
Production-Ready ES Trading System Architecture
Hybrid approach combining NinjaTrader reliability with modern interfaces
"""

import asyncio
import websockets
import json
import sqlite3
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import logging

class ProductionESTradingServer:
    """
    Production-ready trading server that bridges:
    - NinjaTrader AddOn (live trading)
    - Web interface (monitoring/control)
    - Mobile app (alerts/status)
    - Database (logging/analytics)
    """
    
    def __init__(self):
        self.app = FastAPI(title="ES Trading System API")
        self.setup_database()
        self.setup_routes()
        self.active_connections = []
        
        # Trading state
        self.system_status = "STOPPED"
        self.current_position = 0
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.latest_signals = {}
        
    def setup_database(self):
        """Setup production database"""
        self.db = sqlite3.connect('production_trading.db', check_same_thread=False)
        
        # Create tables
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                action TEXT,
                quantity INTEGER,
                price REAL,
                pnl REAL,
                confidence REAL,
                signal_type TEXT
            )
        ''')
        
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                signal TEXT,
                confidence REAL,
                features TEXT,
                market_data TEXT
            )
        ''')
        
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY,
                date DATE,
                total_pnl REAL,
                daily_pnl REAL,
                trades_count INTEGER,
                win_rate REAL,
                max_drawdown REAL
            )
        ''')
        
        self.db.commit()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(self.get_dashboard_html())
        
        @self.app.get("/api/status")
        async def get_status():
            return {
                "system_status": self.system_status,
                "current_position": self.current_position,
                "daily_pnl": self.daily_pnl,
                "trade_count": self.trade_count,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/performance")
        async def get_performance():
            cursor = self.db.execute('''
                SELECT * FROM performance 
                ORDER BY date DESC 
                LIMIT 30
            ''')
            data = cursor.fetchall()
            return {"performance": data}
        
        @self.app.get("/api/trades")
        async def get_trades():
            cursor = self.db.execute('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            data = cursor.fetchall()
            return {"trades": data}
        
        @self.app.post("/api/control/{action}")
        async def control_system(action: str):
            if action == "start":
                self.system_status = "RUNNING"
                await self.broadcast_update({"type": "status", "status": "RUNNING"})
                return {"success": True, "message": "System started"}
            elif action == "stop":
                self.system_status = "STOPPED"
                await self.broadcast_update({"type": "status", "status": "STOPPED"})
                return {"success": True, "message": "System stopped"}
            else:
                return {"success": False, "message": "Invalid action"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle client messages
                    await websocket.receive_text()
            except:
                self.active_connections.remove(websocket)
    
    def log_trade(self, action, quantity, price, pnl, confidence, signal_type):
        """Log trade to database"""
        self.db.execute('''
            INSERT INTO trades (timestamp, action, quantity, price, pnl, confidence, signal_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), action, quantity, price, pnl, confidence, signal_type))
        self.db.commit()
    
    def log_signal(self, signal, confidence, features, market_data):
        """Log ML signal to database"""
        self.db.execute('''
            INSERT INTO signals (timestamp, signal, confidence, features, market_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), signal, confidence, json.dumps(features), json.dumps(market_data)))
        self.db.commit()
    
    async def broadcast_update(self, data):
        """Broadcast updates to all connected clients"""
        if self.active_connections:
            message = json.dumps(data)
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except:
                    self.active_connections.remove(connection)
    
    def get_dashboard_html(self):
        """Generate modern dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>ES Trading System - Production Dashboard</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            background: #0a0a0a; 
            color: #ffffff; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .card { 
            background: #1a1a1a; 
            border: 1px solid #333; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
        }
        .status-running { color: #4CAF50; }
        .status-stopped { color: #f44336; }
        .metric { 
            display: inline-block; 
            margin: 10px 20px; 
            text-align: center; 
        }
        .metric-value { 
            font-size: 2em; 
            font-weight: bold; 
        }
        .metric-label { 
            color: #999; 
            font-size: 0.9em; 
        }
        .controls { 
            text-align: center; 
            margin: 20px; 
        }
        button { 
            background: #667eea; 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 6px; 
            cursor: pointer; 
            margin: 0 10px; 
            font-size: 16px; 
        }
        button:hover { 
            background: #5a6fd8; 
        }
        .chart-container { 
            height: 300px; 
            background: #111; 
            border-radius: 10px; 
            margin: 20px 0; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ES ML Trading System - Production Dashboard</h1>
            <p>Institutional-grade automated trading with machine learning</p>
        </div>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <div class="metric-value status-stopped" id="system-status">STOPPED</div>
                <div class="metric-label">System Status</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="current-position">0</div>
                <div class="metric-label">Position (Contracts)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="daily-pnl">$0.00</div>
                <div class="metric-label">Daily P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="trade-count">0</div>
                <div class="metric-label">Trades Today</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="controlSystem('start')">▶️ Start System</button>
            <button onclick="controlSystem('stop')">⏹️ Stop System</button>
            <button onclick="refreshData()">🔄 Refresh</button>
        </div>
        
        <div class="card">
            <h2>Real-time Performance Chart</h2>
            <div class="chart-container">
                <p>📊 Real-time P&L chart will appear here</p>
                <p>💡 Connect to TradingView or similar charting library</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Trades</h2>
            <div id="trades-list">
                <p>Loading trades...</p>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'status') {
                document.getElementById('system-status').textContent = data.status;
                document.getElementById('system-status').className = 
                    data.status === 'RUNNING' ? 'metric-value status-running' : 'metric-value status-stopped';
            }
        };
        
        async function controlSystem(action) {
            const response = await fetch(`/api/control/${action}`, { method: 'POST' });
            const result = await response.json();
            if (result.success) {
                refreshData();
            }
        }
        
        async function refreshData() {
            // Fetch current status
            const statusResponse = await fetch('/api/status');
            const status = await statusResponse.json();
            
            document.getElementById('system-status').textContent = status.system_status;
            document.getElementById('current-position').textContent = status.current_position;
            document.getElementById('daily-pnl').textContent = `$${status.daily_pnl.toFixed(2)}`;
            document.getElementById('trade-count').textContent = status.trade_count;
            
            // Fetch trades
            const tradesResponse = await fetch('/api/trades');
            const trades = await tradesResponse.json();
            
            const tradesList = document.getElementById('trades-list');
            tradesList.innerHTML = trades.trades.map(trade => 
                `<div style="padding: 10px; border-bottom: 1px solid #333;">
                    ${trade[1]} - ${trade[2]} ${trade[3]} @ $${trade[4]} (P&L: $${trade[5]})
                </div>`
            ).join('');
        }
        
        // Initial data load
        refreshData();
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
        '''
    
    def start_server(self):
        """Start the production server"""
        print("🚀 Starting Production ES Trading Server...")
        print("📊 Dashboard: http://localhost:8000")
        print("🔌 API: http://localhost:8000/api/status")
        
        uvicorn.run(self.app, host="0.0.0.0", port=8000, log_level="info")

# NinjaTrader Integration Bridge
class NinjaTraderBridge:
    """Bridge between NinjaTrader AddOn and Production Server"""
    
    def __init__(self, server):
        self.server = server
        self.setup_tcp_listener()
    
    def setup_tcp_listener(self):
        """Listen for updates from NinjaTrader AddOn"""
        import socket
        import threading
        
        def listen_for_updates():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', 36975))  # Different port from order executor
            sock.listen(1)
            
            print("🔗 Listening for NinjaTrader updates on port 36975...")
            
            while True:
                conn, addr = sock.accept()
                try:
                    data = conn.recv(1024).decode()
                    if data:
                        update = json.loads(data)
                        self.process_ninjatrader_update(update)
                except:
                    pass
                finally:
                    conn.close()
        
        thread = threading.Thread(target=listen_for_updates, daemon=True)
        thread.start()
    
    def process_ninjatrader_update(self, update):
        """Process updates from NinjaTrader"""
        if update['type'] == 'trade':
            self.server.log_trade(
                update['action'],
                update['quantity'], 
                update['price'],
                update['pnl'],
                update['confidence'],
                update['signal_type']
            )
            self.server.trade_count += 1
            self.server.daily_pnl += update['pnl']
            
        elif update['type'] == 'signal':
            self.server.log_signal(
                update['signal'],
                update['confidence'],
                update['features'],
                update['market_data']
            )
            
        # Broadcast to web clients
        asyncio.create_task(self.server.broadcast_update(update))

def main():
    """Launch production system"""
    print("🎯 ES Trading System - Production Mode")
    print("=" * 50)
    
    # Create production server
    server = ProductionESTradingServer()
    
    # Create NinjaTrader bridge
    bridge = NinjaTraderBridge(server)
    
    # Start server
    server.start_server()

if __name__ == "__main__":
    main()