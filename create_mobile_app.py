"""
Mobile App Integration for ES Trading System
React Native / PWA companion app
"""

mobile_app_html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ES Trader Mobile</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #000;
            color: #fff;
            line-height: 1.6;
        }
        .app { max-width: 414px; margin: 0 auto; min-height: 100vh; }
        .header { 
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 20px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .status-card { 
            background: #1a1a1a;
            margin: 15px;
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #333;
        }
        .metric-row { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #999; font-size: 14px; }
        .metric-value { font-size: 18px; font-weight: bold; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .neutral { color: #FFA726; }
        .control-buttons { 
            display: flex;
            gap: 10px;
            margin: 15px;
        }
        .btn { 
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
        .btn:active { transform: scale(0.95); }
        .trades-list { margin: 15px; }
        .trade-item { 
            background: #1a1a1a;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .trade-header { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .trade-time { color: #999; font-size: 12px; }
        .notification { 
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            z-index: 1000;
            display: none;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>📱 ES Trader</h1>
            <p>Mobile Trading Dashboard</p>
        </div>
        
        <div class="notification" id="notification"></div>
        
        <div class="status-card">
            <h3>System Status</h3>
            <div class="metric-row">
                <span class="metric-label">Status</span>
                <span class="metric-value neutral" id="system-status">STOPPED</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Position</span>
                <span class="metric-value" id="position">0 contracts</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Today's P&L</span>
                <span class="metric-value neutral" id="daily-pnl">$0.00</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Trades</span>
                <span class="metric-value" id="trade-count">0/5</span>
            </div>
        </div>
        
        <div class="control-buttons">
            <button class="btn btn-start" onclick="controlSystem('start')">
                ▶️ Start
            </button>
            <button class="btn btn-stop" onclick="controlSystem('stop')">
                ⏹️ Stop
            </button>
        </div>
        
        <div class="status-card">
            <h3>Latest Signal</h3>
            <div class="metric-row">
                <span class="metric-label">AI Prediction</span>
                <span class="metric-value neutral" id="latest-signal">HOLD</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Confidence</span>
                <span class="metric-value" id="signal-confidence">0%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Last Update</span>
                <span class="metric-value" id="signal-time">--:--</span>
            </div>
        </div>
        
        <div class="trades-list">
            <h3 style="margin-bottom: 15px;">Recent Trades</h3>
            <div id="trades-container">
                <div class="trade-item">
                    <div class="trade-header">
                        <span>No trades yet</span>
                        <span class="trade-time">--:--</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = function() {
                console.log('Connected to trading server');
                showNotification('Connected to server', 'success');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateUI(data);
            };
            
            ws.onclose = function() {
                console.log('Disconnected from server');
                setTimeout(connectWebSocket, 5000); // Reconnect
            };
        }
        
        function updateUI(data) {
            if (data.type === 'status') {
                const statusEl = document.getElementById('system-status');
                statusEl.textContent = data.status;
                statusEl.className = data.status === 'RUNNING' ? 'metric-value positive' : 'metric-value neutral';
            }
            
            if (data.type === 'trade') {
                addTradeToList(data);
                showNotification(`${data.action} ${data.quantity} @ $${data.price}`, 'info');
            }
            
            if (data.type === 'signal') {
                updateSignal(data);
            }
        }
        
        function addTradeToList(trade) {
            const container = document.getElementById('trades-container');
            const tradeEl = document.createElement('div');
            tradeEl.className = 'trade-item';
            tradeEl.innerHTML = `
                <div class="trade-header">
                    <span class="${trade.pnl > 0 ? 'positive' : 'negative'}">
                        ${trade.action} ${trade.quantity} @ $${trade.price}
                    </span>
                    <span class="trade-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div style="font-size: 14px; color: #999;">
                    P&L: $${trade.pnl.toFixed(2)} | Confidence: ${(trade.confidence * 100).toFixed(1)}%
                </div>
            `;
            container.insertBefore(tradeEl, container.firstChild);
            
            // Keep only last 10 trades
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }
        
        function updateSignal(signal) {
            document.getElementById('latest-signal').textContent = signal.signal;
            document.getElementById('signal-confidence').textContent = `${(signal.confidence * 100).toFixed(1)}%`;
            document.getElementById('signal-time').textContent = new Date().toLocaleTimeString();
            
            const signalEl = document.getElementById('latest-signal');
            signalEl.className = signal.signal === 'BUY' ? 'metric-value positive' : 
                               signal.signal === 'SELL' ? 'metric-value negative' : 'metric-value neutral';
        }
        
        async function controlSystem(action) {
            try {
                const response = await fetch(`http://localhost:8000/api/control/${action}`, { 
                    method: 'POST' 
                });
                const result = await response.json();
                
                if (result.success) {
                    showNotification(result.message, 'success');
                    refreshData();
                } else {
                    showNotification(result.message, 'error');
                }
            } catch (error) {
                showNotification('Connection error', 'error');
            }
        }
        
        async function refreshData() {
            try {
                const response = await fetch('http://localhost:8000/api/status');
                const status = await response.json();
                
                document.getElementById('system-status').textContent = status.system_status;
                document.getElementById('position').textContent = `${status.current_position} contracts`;
                document.getElementById('daily-pnl').textContent = `$${status.daily_pnl.toFixed(2)}`;
                document.getElementById('trade-count').textContent = `${status.trade_count}/5`;
                
                // Update P&L color
                const pnlEl = document.getElementById('daily-pnl');
                pnlEl.className = status.daily_pnl > 0 ? 'metric-value positive' : 
                                 status.daily_pnl < 0 ? 'metric-value negative' : 'metric-value neutral';
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        function showNotification(message, type) {
            const notif = document.getElementById('notification');
            notif.textContent = message;
            notif.className = `notification ${type}`;
            notif.style.display = 'block';
            
            setTimeout(() => {
                notif.style.display = 'none';
            }, 3000);
        }
        
        // Initialize
        connectWebSocket();
        refreshData();
        setInterval(refreshData, 10000); // Refresh every 10 seconds
        
        // PWA Service Worker for offline functionality
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
</body>
</html>
'''

# Save mobile app
with open('mobile_dashboard.html', 'w') as f:
    f.write(mobile_app_html)

print("📱 Mobile dashboard created: mobile_dashboard.html")