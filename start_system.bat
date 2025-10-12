@echo off
REM === ES Trading System - One-Click Launch ===
echo ========================================
echo ES Trading System - Launching System
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Choose your interface:
echo 1. Enhanced Trading UI (with training progress tracking)
echo 2. Production Server (web/mobile dashboards)
echo 3. Data Source Analysis
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Starting Enhanced Trading UI...
    python enhanced_trading_ui.py
) else if "%choice%"=="2" (
    echo Starting Production Server...
    python launch_production.py
) else if "%choice%"=="3" (
    echo Starting Data Source Analysis...
    python data_source_analyzer.py
) else (
    echo Starting Enhanced Trading UI (default)...
    python enhanced_trading_ui.py
)

echo.
echo System stopped.
pause
