@echo off
echo ========================================
echo ES Trading System - Quick Launch
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_remote.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

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
