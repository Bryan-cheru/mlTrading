"""
Real-time ML Signal Prediction Script
Called by NinjaTrader AddOn for live trading signals
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

# Import the ML model loader
try:
    from ml_models.integration.ml_model_loader import ESMLModelLoader, predict_ml_signal
except ImportError as e:
    print(json.dumps({
        "error": f"Failed to import ML components: {e}",
        "signal": "HOLD",
        "confidence": 0.0,
        "ml_enabled": False
    }))
    sys.exit(1)

def load_market_data(json_file):
    """Load market data from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

def convert_to_dataframe(market_data):
    """Convert market data to pandas DataFrame for feature engineering"""
    try:
        # Create DataFrame from historical data
        historical_data = []
        
        closes = market_data.get('historical_closes', [market_data['close']])
        volumes = market_data.get('historical_volumes', [market_data['volume']])
        
        # Ensure we have enough data
        if len(closes) < 10:
            # Pad with current values if insufficient history
            current_close = market_data['close']
            current_volume = market_data['volume']
            closes = [current_close] * (10 - len(closes)) + closes
            volumes = [current_volume] * (10 - len(volumes)) + volumes
        
        # Create OHLCV data (simplified - using close price for all OHLC)
        for i, (close, volume) in enumerate(zip(closes, volumes)):
            historical_data.append({
                'Open': close * 0.999,    # Approximate open
                'High': close * 1.001,    # Approximate high  
                'Low': close * 0.999,     # Approximate low
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(historical_data)
        return df
        
    except Exception as e:
        print(json.dumps({
            "error": f"Error converting data: {e}",
            "signal": "HOLD",
            "confidence": 0.0,
            "ml_enabled": False
        }))
        return None

def main():
    """Main prediction function"""
    if len(sys.argv) != 2:
        print(json.dumps({
            "error": "Usage: python predict_signal.py <market_data_json_file>",
            "signal": "HOLD",
            "confidence": 0.0,
            "ml_enabled": False
        }))
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        # Load market data
        market_data = load_market_data(json_file)
        if market_data is None:
            print(json.dumps({
                "error": "Failed to load market data",
                "signal": "HOLD",
                "confidence": 0.0,
                "ml_enabled": False
            }))
            sys.exit(1)
        
        # Convert to DataFrame for ML processing
        df = convert_to_dataframe(market_data)
        if df is None:
            sys.exit(1)
        
        # Get ML prediction
        try:
            model_loader = ESMLModelLoader()
            
            if not model_loader.is_loaded:
                print(json.dumps({
                    "error": "ML model not loaded",
                    "signal": "HOLD", 
                    "confidence": 0.0,
                    "ml_enabled": False
                }))
                sys.exit(1)
            
            # Generate prediction
            result = model_loader.predict_signal(df)
            
            # Return result as JSON
            output = {
                "signal": result['signal'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities'],
                "ml_enabled": result['ml_enabled'],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            print(json.dumps(output))
            
        except Exception as e:
            print(json.dumps({
                "error": f"ML prediction failed: {e}",
                "signal": "HOLD",
                "confidence": 0.0,
                "ml_enabled": False
            }))
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({
            "error": f"Unexpected error: {e}",
            "signal": "HOLD",
            "confidence": 0.0,
            "ml_enabled": False
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()