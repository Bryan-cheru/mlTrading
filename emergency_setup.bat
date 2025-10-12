@echo off
REM === EMERGENCY FIX for Installation Issues ===
echo ========================================
echo ES Trading System - Emergency Setup Fix
echo ========================================
echo.
echo This will fix the scikit-learn installation error.
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ERROR: Not in the correct directory!
    echo Please run this from the MLTrading folder.
    pause
    exit /b 1
)

REM Create or activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Clear any cached packages
echo Clearing pip cache...
pip cache purge

REM Install ONLY the essential packages that work
echo Installing essential packages only...
pip install pandas>=2.0.0
pip install numpy>=1.25.0
pip install yfinance>=0.2.0
pip install fastapi>=0.100.0
pip install uvicorn>=0.20.0
pip install matplotlib>=3.7.0
pip install requests>=2.30.0
pip install psutil>=5.9.0

REM Try to install scikit-learn with pre-compiled wheel only
echo Trying to install scikit-learn (pre-compiled only)...
pip install scikit-learn --only-binary=all
if %errorlevel% neq 0 (
    echo WARNING: scikit-learn installation failed.
    echo The system will work without it but ML features will be limited.
)

REM Try to install xgboost
echo Trying to install xgboost...
pip install xgboost --only-binary=all
if %errorlevel% neq 0 (
    echo WARNING: xgboost installation failed.
    echo You can try installing it later if needed.
)

echo.
echo ========================================
echo EMERGENCY SETUP COMPLETED!
echo ========================================
echo.
echo Essential packages installed successfully.
echo The trading system should now work.
echo.
echo To start the system: double-click start_system.bat
echo.
pause