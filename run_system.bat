@echo off
echo Professional Institutional Trading System
echo =============================================

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the system
echo Starting system...
python run_institutional_system.py

pause