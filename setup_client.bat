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

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

REM Try main requirements first
echo Installing packages from main requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Main installation failed. Trying minimal requirements...
    echo This may be due to compatibility issues with some packages.
    echo.
    pip install -r requirements_minimal.txt
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Installation failed completely.
        echo Trying individual package installation...
        pip install pandas numpy yfinance fastapi uvicorn matplotlib requests
        if %errorlevel% neq 0 (
            echo CRITICAL ERROR: Cannot install basic packages
            echo Please check your Python installation and internet connection
            pause
            exit /b 1
        )
    )
    echo.
    echo NOTICE: Minimal installation completed.
    echo Some advanced features may not work without all packages.
    echo You can try installing missing packages later with:
    echo   pip install scikit-learn xgboost torch
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start the trading system, double-click 'start_system.bat'.
echo.
pause
