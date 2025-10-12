# 🚀 TeamViewer Remote Deployment Instructions

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
venv\Scripts\activate.bat

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
venv\Scripts\activate.bat

# Check system status
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Test web server
curl http://localhost:8000/health

# Monitor logs
tail -f logs/system.log
```

Ready to trade! 🚀