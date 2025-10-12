@echo off
echo ================================================================
echo ES ML Trading System - Professional NinjaScript AddOn Installer
echo ================================================================
echo.

echo [1/4] Checking NinjaTrader installation...
if not exist "C:\Users\%USERNAME%\Documents\NinjaTrader 8\bin\Custom\AddOns" (
    echo ERROR: NinjaTrader 8 AddOns folder not found!
    echo Please ensure NinjaTrader 8 is installed properly.
    pause
    exit /b 1
)
echo ✓ NinjaTrader 8 found

echo.
echo [2/4] Installing AddOn files...
copy /Y "ninjatrader-addon\ESMLTradingSystem.cs" "C:\Users\%USERNAME%\Documents\NinjaTrader 8\bin\Custom\AddOns\" >nul
copy /Y "ninjatrader-addon\ESMLTradingWindow.cs" "C:\Users\%USERNAME%\Documents\NinjaTrader 8\bin\Custom\AddOns\" >nul
echo ✓ AddOn files copied successfully

echo.
echo [3/4] Verifying installation...
if exist "C:\Users\%USERNAME%\Documents\NinjaTrader 8\bin\Custom\AddOns\ESMLTradingSystem.cs" (
    echo ✓ ESMLTradingSystem.cs installed
) else (
    echo ✗ ESMLTradingSystem.cs missing
)

if exist "C:\Users\%USERNAME%\Documents\NinjaTrader 8\bin\Custom\AddOns\ESMLTradingWindow.cs" (
    echo ✓ ESMLTradingWindow.cs installed  
) else (
    echo ✗ ESMLTradingWindow.cs missing
)

echo.
echo [4/4] Installation complete!
echo.
echo ================================================================
echo NEXT STEPS:
echo ================================================================
echo 1. Open NinjaTrader 8
echo 2. Go to Tools → Edit NinjaScript → AddOn
echo 3. Press F5 to compile (or click Compile button)
echo 4. If compilation is successful, close the editor
echo 5. Go to Tools → ES ML Trading System
echo 6. Your professional trading interface will open!
echo 7. Click "Start System" to begin automated trading
echo ================================================================
echo.
echo Features of your new professional interface:
echo ✓ Real-time ES futures market data
echo ✓ Live ML signal generation and display  
echo ✓ Automated order execution with risk management
echo ✓ Beautiful dark theme professional UI
echo ✓ Complete performance tracking and metrics
echo ✓ Trade history and activity logging
echo ✓ Account information and position monitoring
echo.
echo Ready to launch your institutional-grade trading system!
echo.
pause