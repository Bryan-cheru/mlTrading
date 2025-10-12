# Installation Troubleshooting Guide

## Common Installation Issues & Solutions

### Issue 1: scikit-learn Compilation Error
**Error**: "Cython.Compiler.Errors.CompileError" when installing scikit-learn

**Solutions**:
1. **Use pre-compiled wheels**: `pip install scikit-learn --only-binary=all`
2. **Try minimal install**: Use `requirements_minimal.txt` instead
3. **Install individually**: `pip install scikit-learn==1.4.0` (newer version)

### Issue 2: Visual Studio Build Tools Missing
**Error**: "Microsoft Visual C++ 14.0 is required"

**Solutions**:
1. Install "Microsoft C++ Build Tools" from Microsoft website
2. Or install Visual Studio Community (free)
3. Use pre-compiled packages: `pip install --only-binary=all -r requirements.txt`

### Issue 3: Torch/PyTorch Installation Issues
**Error**: Torch installation fails or takes too long

**Solutions**:
1. Install CPU version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Skip torch for now: Use `requirements_minimal.txt`
3. Install later when needed for ML training

### Issue 4: TA-Lib Installation Error
**Error**: "Failed building wheel for TA-Lib"

**Solutions**:
1. **Windows**: Download TA-Lib wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
2. **Install manually**: `pip install TA_Lib-0.4.25-cp311-cp311-win_amd64.whl`
3. **Skip for now**: System works without TA-Lib (reduced features)

## Alternative Installation Methods

### Method 1: Minimal Installation
```cmd
pip install -r requirements_minimal.txt
```
This installs only essential packages for basic functionality.

### Method 2: Individual Package Installation
```cmd
pip install pandas numpy yfinance fastapi uvicorn matplotlib requests
pip install scikit-learn --only-binary=all
pip install xgboost --only-binary=all
```

### Method 3: Skip Problematic Packages
```cmd
# Install essentials first
pip install pandas numpy yfinance fastapi uvicorn

# Add packages one by one
pip install matplotlib
pip install requests
pip install psutil

# Try problematic ones last
pip install scikit-learn
pip install xgboost
```

## System Requirements

### Minimum:
- Python 3.9+
- 4GB RAM
- Windows 10/11
- Internet connection

### Recommended:
- Python 3.11+
- 8GB+ RAM
- SSD storage
- Visual Studio Build Tools (for compilation)

## Quick Fixes

### Fix 1: Clear pip cache
```cmd
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

### Fix 2: Use different index
```cmd
pip install -i https://pypi.org/simple/ -r requirements.txt
```

### Fix 3: Force reinstall
```cmd
pip install --force-reinstall --no-deps -r requirements.txt
```

## What Packages Are Actually Required?

### Essential (System won't work without):
- pandas, numpy, yfinance, fastapi, uvicorn, requests

### Important (Reduced functionality without):
- matplotlib, scikit-learn, xgboost

### Optional (Advanced features only):
- torch, transformers, TA-Lib

## If All Else Fails

1. **Use Python 3.11**: Most compatible version
2. **Install Anaconda**: Comes with pre-compiled packages
3. **Use Docker**: Container with all dependencies
4. **Contact support**: Send error logs for help

The system is designed to work with minimal packages if needed!