@echo off
REM === ES Trading System - One-Click Setup ===
echo ========================================
echo ES Trading System - Automated Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo Python found.

REM Create virtual environment if not exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo Setup complete!
echo.
echo To start the trading system, double-click 'start_system.bat'.
pause
