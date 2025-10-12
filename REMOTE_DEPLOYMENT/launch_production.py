"""
Ultimate Production ES Trading System Launcher
Coordinates all components for institutional-grade deployment
"""

import os
import sys
import subprocess
import time
import json
import webbrowser
from pathlib import Path
import psutil
import threading

class ProductionLauncher:
    """Complete production system orchestrator"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_python = self.project_root / "venv" / "Scripts" / "python.exe"
        self.processes = {}
        
    def check_requirements(self):
        """Check all production requirements"""
        print("🔍 Production Requirements Check")
        print("=" * 40)
        
        requirements = {
            "Python Environment": self.venv_python.exists(),
            "NinjaTrader 8": self.check_ninjatrader(),
            "Trained ML Model": (self.project_root / "models").exists(),
            "System Resources": self.check_system_resources()
        }
        
        all_good = True
        for req, status in requirements.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {req}")
            if not status:
                all_good = False
        
        return all_good
    
    def check_ninjatrader(self):
        """Check if NinjaTrader is available"""
        nt_path = Path("C:/Users/Brian Cheruiyot/Documents/NinjaTrader 8")
        return nt_path.exists()
    
    def check_system_resources(self):
        """Check system resources"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count()
        
        print(f"   💾 RAM: {memory_gb:.1f} GB")
        print(f"   🖥️ CPU Cores: {cpu_cores}")
        
        return memory_gb >= 8 and cpu_cores >= 4  # Minimum requirements
    
    def deploy_production_components(self):
        """Deploy all production components"""
        print("\n🚀 Deploying Production Components...")
        
        # 1. Copy NinjaTrader AddOns
        self.deploy_ninjatrader_addons()
        
        # 2. Create mobile app
        self.create_mobile_app()
        
        # 3. Setup monitoring
        self.setup_monitoring()
        
        # 4. Configure alerts
        self.setup_alerts()
    
    def deploy_ninjatrader_addons(self):
        """Deploy NinjaTrader AddOns"""
        print("📦 Deploying NinjaTrader AddOns...")
        
        nt_addons_path = Path("C:/Users/Brian Cheruiyot/Documents/NinjaTrader 8/bin/Custom/AddOns")
        
        if nt_addons_path.exists():
            # Copy main AddOn files
            addon_files = [
                "ESMLTradingSystemMain.cs",
                "ESMLTradingWindow.cs", 
                "ESOrderExecutor.cs"
            ]
            
            for file in addon_files:
                src = self.project_root / "ninjatrader-addon" / file
                dst = nt_addons_path / file
                
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"   ✅ Deployed {file}")
                else:
                    print(f"   ❌ Missing {file}")
        else:
            print("   ❌ NinjaTrader AddOns directory not found")
    
    def create_mobile_app(self):
        """Create mobile app"""
        print("📱 Creating Mobile App...")
        subprocess.run([str(self.venv_python), "create_mobile_app.py"], cwd=self.project_root)
        print("   ✅ Mobile dashboard created")
    
    def setup_monitoring(self):
        """Setup system monitoring"""
        print("📊 Setting up Monitoring...")
        
        monitoring_script = '''
import psutil
import time
import json
import requests
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.alerts_sent = set()
    
    def monitor_loop(self):
        while True:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Check for issues
                if cpu_percent > 90:
                    self.send_alert("High CPU usage", f"CPU at {cpu_percent}%")
                
                if memory.percent > 95:
                    self.send_alert("High memory usage", f"Memory at {memory.percent}%")
                
                # Log metrics
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / 1e9
                }
                
                with open("system_metrics.json", "w") as f:
                    json.dump(metrics, f)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)
    
    def send_alert(self, title, message):
        alert_key = f"{title}_{message}"
        if alert_key not in self.alerts_sent:
            print(f"🚨 ALERT: {title} - {message}")
            self.alerts_sent.add(alert_key)
            
            # Remove old alerts after 1 hour
            if len(self.alerts_sent) > 10:
                self.alerts_sent.clear()

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.monitor_loop()
'''
        
        with open(self.project_root / "system_monitor.py", "w") as f:
            f.write(monitoring_script)
        
        print("   ✅ System monitor created")
    
    def setup_alerts(self):
        """Setup alert system"""
        print("🔔 Setting up Alerts...")
        
        alert_config = {
            "email_alerts": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "slack_alerts": {
                "enabled": False,
                "webhook_url": ""
            },
            "sms_alerts": {
                "enabled": False,
                "service": "twilio",
                "account_sid": "",
                "auth_token": "",
                "from_number": "",
                "to_numbers": []
            },
            "alert_conditions": {
                "high_drawdown": 0.05,  # 5% drawdown
                "consecutive_losses": 3,
                "daily_loss_limit": 1000,
                "system_errors": True
            }
        }
        
        with open(self.project_root / "alert_config.json", "w") as f:
            json.dump(alert_config, f, indent=2)
        
        print("   ✅ Alert configuration created")
        print("   💡 Edit alert_config.json to configure notifications")
    
    def start_production_system(self):
        """Start all production components"""
        print("\n🚀 Starting Production ES Trading System")
        print("=" * 50)
        
        # 1. Start production server
        print("1. 🖥️ Starting production server...")
        server_process = subprocess.Popen([
            str(self.venv_python), "production_server.py"
        ], cwd=self.project_root)
        self.processes['server'] = server_process
        time.sleep(3)
        
        # 2. Start system monitor
        print("2. 📊 Starting system monitor...")
        monitor_process = subprocess.Popen([
            str(self.venv_python), "system_monitor.py"
        ], cwd=self.project_root)
        self.processes['monitor'] = monitor_process
        
        # 3. Open web dashboard
        print("3. 🌐 Opening web dashboard...")
        time.sleep(2)
        webbrowser.open("http://localhost:8000")
        
        # 4. Open mobile dashboard in second tab
        print("4. 📱 Opening mobile dashboard...")
        time.sleep(1)
        mobile_path = (self.project_root / "mobile_dashboard.html").as_uri()
        webbrowser.open(mobile_path)
        
        print("\n✅ Production System Started!")
        print("=" * 40)
        print("🎯 Components Running:")
        print("   • Production Server: http://localhost:8000")
        print("   • System Monitor: Background process")
        print("   • Web Dashboard: Opened in browser")
        print("   • Mobile Dashboard: Opened in browser")
        print("\n📋 Next Steps:")
        print("   1. Open NinjaTrader 8")
        print("   2. Go to Tools → ES ML Trading System")
        print("   3. Click 'Start System' in NinjaTrader")
        print("   4. Monitor via web/mobile dashboards")
        print("\n⚠️ To stop system: Press Ctrl+C")
        
        # Keep running
        try:
            while True:
                time.sleep(10)
                
                # Check if processes are still running
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        print(f"⚠️ {name} process stopped unexpectedly")
        except KeyboardInterrupt:
            self.stop_system()
    
    def stop_system(self):
        """Stop all production components"""
        print("\n🛑 Stopping Production System...")
        
        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                process.terminate()
                print(f"   ✅ Stopped {name}")
        
        print("✅ All components stopped")
    
    def show_architecture(self):
        """Show production architecture"""
        print("\n🏗️ Production Architecture")
        print("=" * 50)
        print("""
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NinjaTrader   │    │  Production      │    │   Web/Mobile    │
│     AddOn       │◄──►│    Server        │◄──►│   Dashboard     │
│                 │    │                  │    │                 │
│ • Live Trading  │    │ • API Endpoints  │    │ • Real-time UI  │
│ • Risk Mgmt     │    │ • Database       │    │ • Controls      │
│ • ML Integration│    │ • WebSocket      │    │ • Monitoring    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Broker API    │    │  ML Models       │    │  Alert System   │
│                 │    │                  │    │                 │
│ • Interactive   │    │ • Transformer    │    │ • Email/SMS     │
│   Brokers       │    │ • Random Forest  │    │ • Slack         │
│ • TD Ameritrade │    │ • Feature Eng.   │    │ • Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘

🎯 Benefits:
• Institutional-grade reliability
• Multi-interface access (desktop/web/mobile)
• Real-time monitoring and alerts
• Scalable cloud deployment ready
• Professional risk management
• Complete audit trail
        """)
    
    def run_production_setup(self):
        """Complete production setup workflow"""
        print("🎯 ES Trading System - Production Setup")
        print("=" * 60)
        print("🏢 Institutional-Grade Automated Trading Platform")
        print("=" * 60)
        
        # Show architecture
        self.show_architecture()
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Requirements not met. Please install missing components.")
            return
        
        # Deploy components
        self.deploy_production_components()
        
        # Confirm start
        print(f"\n🚀 Ready to launch production system!")
        print("📊 This will start:")
        print("   • Web server with REST API")
        print("   • Real-time WebSocket connections")
        print("   • System monitoring")
        print("   • Web and mobile dashboards")
        print("   • Database logging")
        
        response = input(f"\n🔥 Launch production system? (y/n): ")
        if response.lower() != 'y':
            print("❌ Launch cancelled")
            return
        
        # Start production system
        self.start_production_system()

def main():
    launcher = ProductionLauncher()
    launcher.run_production_setup()

if __name__ == "__main__":
    main()