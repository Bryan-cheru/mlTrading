@echo off
echo ===============================================
echo NINJATRADER ADDON COMPILATION TEST
echo ===============================================

echo.
echo Checking NinjaTrader installation...
if exist "C:\Program Files\NinjaTrader 8" (
    echo ✅ NinjaTrader 8 found at default location
    set "NT8_PATH=C:\Program Files\NinjaTrader 8"
) else if exist "C:\Program Files (x86)\NinjaTrader 8" (
    echo ✅ NinjaTrader 8 found at x86 location
    set "NT8_PATH=C:\Program Files (x86)\NinjaTrader 8"
) else (
    echo ❌ NinjaTrader 8 not found in default locations
    echo Please ensure NinjaTrader 8 is installed
    pause
    exit /b 1
)

echo.
echo Checking AddOn file...
if exist "ModernInstitutionalAddOn.cs" (
    echo ✅ ModernInstitutionalAddOn.cs found
) else (
    echo ❌ ModernInstitutionalAddOn.cs not found
    echo Please ensure you are in the ninjatrader-addon directory
    pause
    exit /b 1
)

echo.
echo Copying AddOn to NinjaTrader...
set "ADDON_DIR=%USERPROFILE%\Documents\NinjaTrader 8\bin\Custom\AddOns"
if not exist "%ADDON_DIR%" mkdir "%ADDON_DIR%"

copy "ModernInstitutionalAddOn.cs" "%ADDON_DIR%\" > nul
if %ERRORLEVEL% == 0 (
    echo ✅ AddOn copied successfully to: %ADDON_DIR%
) else (
    echo ❌ Failed to copy AddOn file
    pause
    exit /b 1
)

echo.
echo ===============================================
echo NEXT STEPS:
echo ===============================================
echo 1. Open NinjaTrader 8
echo 2. Go to Tools ^> Edit NinjaScript ^> AddOn
echo 3. Find "Modern Institutional Trading" and open it
echo 4. Press F5 to compile
echo 5. If compilation succeeds, go to New ^> Modern Institutional Trading
echo 6. Monitor the Python server logs for connections and signals
echo.
echo Python ML Server Status: 
echo - Running on ws://localhost:8000
echo - Trained ML model ready
echo - Sending periodic trading signals
echo.
echo ===============================================

pause