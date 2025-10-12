"""
Remote Deployment Package Creator
Creates a complete portable package for TeamViewer deployment
"""

import os
import shutil
import zipfile
import json
from pathlib import Path
import subprocess

class RemoteDeploymentPackager:
    """Creates portable deployment package for remote TeamViewer access"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deploy_dir = self.project_root / "REMOTE_DEPLOYMENT"
        self.package_name = "ES_Trading_System_Remote_Package.zip"
        
    def create_deployment_package(self):
        """Create complete deployment package"""
        print("📦 Creating Remote Deployment Package for TeamViewer")
        print("=" * 60)
        
        # Clean previous deployment
        if self.deploy_dir.exists():
            shutil.rmtree(self.deploy_dir)
        self.deploy_dir.mkdir()
        
        # 1. Copy essential Python files
        self.copy_python_files()
        
        # 2. Create requirements file
        self.create_requirements()
        
        # 3. Create auto-setup script
        self.create_auto_setup()
        
        # 4. Create one-click launcher
        self.create_one_click_launcher()
        
        # 5. Copy NinjaTrader files
        self.copy_ninjatrader_files()
        
        # 6. Create deployment instructions
        self.create_deployment_instructions()
        
        # 7. Package everything
        self.create_zip_package()
        
        print(f"\n✅ Deployment package created: {self.package_name}")
        print("📁 Ready for TeamViewer transfer!")
        
    def copy_python_files(self):
        """Copy all necessary Python files"""
        print("📋 Copying Python files...")
        
        essential_files = [
            # Core ML files
            "gpu_transformer_trainer.py",
            "train_ultimate_transformer.py", 
            "production_server.py",
            "launch_production.py",
            "create_mobile_app.py",
            
            # Data pipeline
            "data-pipeline/ingestion/ninjatrader_connector.py",
            "data-pipeline/processing/realtime_data_manager.py",
            
            # Feature engineering
            "feature-store/realtime/feature_engineering.py",
            
            # ML models
            "ml-models/training/trading_model.py",
            "ml-models/inference/model_predictor.py",
            
            # Trading engine
            "trading-engine/order_executor.py",
            "trading-engine/signal_generator.py",
            
            # Risk management
            "risk-engine/risk_manager.py",
            
            # Configuration
            "config/settings.py",
            "config/system_config.json",
            
            # Documentation
            "PRODUCTION_GUIDE.md",
            "QUICK_START_GUIDE.md"
        ]
        
        # Copy files maintaining directory structure
        for file_path in essential_files:
            src = self.project_root / file_path
            dst = self.deploy_dir / file_path
            
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️ Missing: {file_path}")
    
    def create_requirements(self):
        """Create comprehensive requirements file"""
        print("📝 Creating requirements file...")
        
        requirements = """# ES Trading System - Remote Deployment Requirements
# GPU-optimized for high-performance training

# Core ML/AI
torch>=2.0.0
transformers>=4.30.0
xgboost>=1.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Trading & Finance
yfinance>=0.2.0
ta-lib>=0.4.25
alpaca-trade-api>=3.0.0

# Web & API
fastapi>=0.100.0
uvicorn>=0.22.0
websockets>=11.0.0
aiofiles>=23.0.0
python-multipart>=0.0.6

# Database & Storage
sqlite3
sqlalchemy>=2.0.0
alembic>=1.11.0

# Data Processing
scipy>=1.10.0
plotly>=5.15.0
dash>=2.11.0
streamlit>=1.25.0

# GPU Acceleration (CUDA)
accelerate>=0.20.0
datasets>=2.13.0

# Monitoring & Alerts
psutil>=5.9.0
requests>=2.31.0
slack-sdk>=3.21.0
twilio>=8.5.0

# Development & Testing
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0

# System Integration
pywin32>=306  # Windows integration
keyboard>=0.13.5  # Hotkeys
plyer>=2.1  # Notifications

# Optional: Jupyter for analysis
jupyter>=1.0.0
ipykernel>=6.25.0
"""
        
        req_file = self.deploy_dir / "requirements.txt"
        req_file.write_text(requirements)
        print("   ✅ requirements.txt created")
    
    def create_auto_setup(self):
        """Create automatic setup script"""
        print("🔧 Creating auto-setup script...")
        
        setup_script = '''@echo off
echo ========================================
echo ES Trading System - Remote Setup
echo ========================================
echo.
echo Setting up on remote California PC...
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created
echo.

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python packages...
echo This may take 5-10 minutes depending on internet speed...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!
echo.

REM Create necessary directories
echo 📁 Creating directories...
mkdir logs 2>nul
mkdir models 2>nul
mkdir data 2>nul
mkdir backups 2>nul

echo ✅ Setup completed successfully!
echo.
echo 🚀 Ready to run the trading system!
echo.
echo Next steps:
echo 1. Run: start_system.bat
echo 2. Or run: python launch_production.py
echo.
pause
'''
        
        setup_file = self.deploy_dir / "setup_remote.bat"
        with open(setup_file, 'w', encoding='utf-8') as f:
            f.write(setup_script)
        print("   ✅ setup_remote.bat created")
    
    def create_one_click_launcher(self):
        """Create one-click system launcher"""
        print("🚀 Creating one-click launcher...")
        
        launcher_script = '''@echo off
echo ========================================
echo ES Trading System - Quick Launch
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\\Scripts\\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_remote.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Check if in correct directory
if not exist "launch_production.py" (
    echo ERROR: launch_production.py not found!
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

echo 🔥 Starting ES Trading System...
echo.
echo This will open:
echo • Web dashboard
echo • Mobile interface  
echo • System monitoring
echo • Production server
echo.

REM Launch the system
python launch_production.py

echo.
echo System stopped.
pause
'''
        
        launcher_file = self.deploy_dir / "start_system.bat"
        with open(launcher_file, 'w', encoding='utf-8') as f:
            f.write(launcher_script)
        print("   ✅ start_system.bat created")
    
    def copy_ninjatrader_files(self):
        """Copy NinjaTrader integration files"""
        print("🥷 Copying NinjaTrader files...")
        
        nt_dir = self.deploy_dir / "ninjatrader-addon"
        nt_dir.mkdir(exist_ok=True)
        
        nt_files = [
            "ESMLTradingSystemMain.cs",
            "ESMLTradingWindow.cs",
            "ESOrderExecutor.cs"
        ]
        
        for file_name in nt_files:
            src = self.project_root / "ninjatrader-addon" / file_name
            dst = nt_dir / file_name
            
            if src.exists():
                shutil.copy2(src, dst)
                print(f"   ✅ {file_name}")
            else:
                print(f"   ⚠️ Missing: {file_name}")
    
    def create_deployment_instructions(self):
        """Create detailed deployment instructions"""
        print("📖 Creating deployment instructions...")
        
        instructions = """# 🚀 TeamViewer Remote Deployment Instructions

## Quick Setup (5 minutes)

### Step 1: Transfer Files
1. Connect via TeamViewer to California PC
2. Copy the entire extracted folder to Desktop
3. Navigate to the folder in Command Prompt

### Step 2: Auto Setup
```cmd
# Run the automatic setup
setup_remote.bat
```
This will:
- Create Python virtual environment
- Install all required packages
- Set up directory structure
- Configure the system

### Step 3: Launch System
```cmd
# Start the complete trading system
start_system.bat
```

## Manual Setup (if auto-setup fails)

### Prerequisites Check
```cmd
# Check Python version (need 3.9+)
python --version

# Check pip
pip --version

# Check GPU (optional but recommended)
nvidia-smi
```

### Manual Installation
```cmd
# Create virtual environment
python -m venv venv

# Activate environment
venv\\Scripts\\activate.bat

# Install packages
pip install -r requirements.txt
```

### Launch Options

#### Option 1: Complete System (Recommended)
```cmd
python launch_production.py
```
- Starts web server, monitoring, and dashboards
- Opens browser automatically
- Full production setup

#### Option 2: Server Only
```cmd
python production_server.py
```
- Just the API server
- Access via http://localhost:8000

#### Option 3: Train Models
```cmd
python train_ultimate_transformer.py
```
- Train the GPU-optimized transformer
- Takes 2-4 hours on good GPU

## System URLs (once running)

- **Main Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Real-time Data**: WebSocket at ws://localhost:8000/ws
- **Mobile Dashboard**: Open mobile_dashboard.html

## Troubleshooting

### Common Issues

**"Python not found"**
- Install Python 3.9+ from python.org
- Make sure it's added to PATH

**"Package installation failed"**
- Check internet connection
- Try: `pip install --upgrade pip`
- Run: `pip install -r requirements.txt --verbose`

**"Port already in use"**
- Change port in production_server.py
- Or kill existing process: `taskkill /f /im python.exe`

**"GPU not detected"**
- Install NVIDIA drivers
- Install CUDA toolkit
- Use CPU mode if needed

### Performance Tips

**For GPU Training:**
- Ensure CUDA is installed
- Monitor GPU usage: `nvidia-smi`
- Adjust batch size if out of memory

**For Live Trading:**
- Stable internet connection required
- Close unnecessary programs
- Monitor system resources

## Directory Structure
```
ES_Trading_System/
├── setup_remote.bat          # Auto-setup script
├── start_system.bat          # One-click launcher  
├── requirements.txt          # Python dependencies
├── launch_production.py      # Main system launcher
├── production_server.py      # Web server
├── gpu_transformer_trainer.py # ML training
├── ninjatrader-addon/        # NinjaTrader files
├── data-pipeline/           # Data processing
├── ml-models/              # AI models
├── trading-engine/         # Trading logic
└── risk-engine/           # Risk management
```

## Remote Access Tips

### Via TeamViewer:
1. Use "View Only" mode for monitoring
2. "Full Control" for setup and trading
3. Enable file transfer for updates
4. Use clipboard sharing for quick commands

### Security Notes:
- Change default ports in production
- Use strong passwords
- Enable firewall rules
- Monitor access logs

## Support

### Logs Location:
- System logs: `logs/`
- Error logs: `error.log`
- Trading logs: Database

### Getting Help:
1. Check logs for error messages
2. Verify internet connection
3. Ensure all services are running
4. Check system resources (CPU/RAM/Disk)

---

## 🎯 Quick Commands Reference

```cmd
# Setup
setup_remote.bat

# Start system
start_system.bat

# Activate environment manually
venv\\Scripts\\activate.bat

# Check system status
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Test web server
curl http://localhost:8000/health

# Monitor logs
tail -f logs/system.log
```

Ready to trade! 🚀"""

        instructions_file = self.deploy_dir / "REMOTE_DEPLOYMENT_GUIDE.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        print("   ✅ Deployment guide created")
    
    def create_zip_package(self):
        """Create final zip package"""
        print("🗜️ Creating zip package...")
        
        zip_path = self.project_root / self.package_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.deploy_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.deploy_dir)
                    zipf.write(file_path, arcname)
        
        # Get package size
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Package size: {size_mb:.1f} MB")
        
        return zip_path

def main():
    """Create remote deployment package"""
    packager = RemoteDeploymentPackager()
    packager.create_deployment_package()
    
    print("\n" + "="*60)
    print("🎯 TEAMVIEWER DEPLOYMENT READY!")
    print("="*60)
    print(f"📦 Package: ES_Trading_System_Remote_Package.zip")
    print(f"📁 Location: Current directory")
    print("\n🚀 TeamViewer Deployment Steps:")
    print("1. Transfer zip file to California PC")
    print("2. Extract to Desktop")
    print("3. Open Command Prompt in extracted folder")
    print("4. Run: setup_remote.bat")
    print("5. Run: start_system.bat")
    print("\n✅ Complete system will be running in 5 minutes!")
    print("🌐 Access via: http://localhost:8000")

if __name__ == "__main__":
    main()