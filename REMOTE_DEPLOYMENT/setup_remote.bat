@echo off
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
call venv\Scripts\activate.bat

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
