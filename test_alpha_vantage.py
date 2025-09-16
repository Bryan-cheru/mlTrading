#!/usr/bin/env python3
"""
Simple Alpha Vantage Test
Quick test to verify your API key and data retrieval
"""

import requests
import json
from datetime import datetime

def test_alpha_vantage_simple():
    """Simple test of Alpha Vantage API"""
    
    api_key = "144PSYGR7L4K5GZV"
    symbol = "AAPL"
    
    print("🔑 Testing Alpha Vantage API with your key...")
    print(f"📊 Symbol: {symbol}")
    print(f"🔑 API Key: ****{api_key[-4:]}")
    print("-" * 50)
    
    # Test 1: Quote endpoint (real-time quote)
    print("\n1️⃣ Testing GLOBAL_QUOTE endpoint...")
    quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    
    try:
        response = requests.get(quote_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response JSON:")
            print(json.dumps(data, indent=2))
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                print(f"\n✅ Current Quote for {symbol}:")
                print(f"   Price: ${quote.get('05. price', 'N/A')}")
                print(f"   Change: {quote.get('09. change', 'N/A')}")
                print(f"   Volume: {quote.get('06. volume', 'N/A')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Intraday data
    print(f"\n2️⃣ Testing TIME_SERIES_INTRADAY endpoint...")
    intraday_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"
    
    try:
        response = requests.get(intraday_url, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
            elif "Note" in data:
                print(f"⚠️ Rate Limit: {data['Note']}")
            elif "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
                print(f"✅ Retrieved {len(time_series)} data points")
                
                # Show latest few data points
                latest_times = sorted(time_series.keys(), reverse=True)[:3]
                print(f"\n📊 Latest data points:")
                for timestamp in latest_times:
                    ohlcv = time_series[timestamp]
                    print(f"   {timestamp}: Close=${ohlcv['4. close']}, Volume={ohlcv['5. volume']}")
            else:
                print("❌ Unexpected response format:")
                print(json.dumps(data, indent=2))
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print(f"\n🏁 Alpha Vantage API test completed at {datetime.now()}")

if __name__ == "__main__":
    test_alpha_vantage_simple()
