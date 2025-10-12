"""
Live ML Signal Generation Demo
Shows exactly how our trading system generates signals in real-time
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class SignalDemo:
    """Demonstrate live signal generation"""
    
    def __init__(self):
        self.symbol = "ES=F"
    
    def get_market_data(self):
        """Get current ES market data"""
        print("📊 FETCHING LIVE ES MARKET DATA")
        print("=" * 50)
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period="5d", interval="15m")
        
        if data.empty:
            print("❌ No market data available")
            return None
        
        print(f"✅ Retrieved {len(data)} bars of ES data")
        print(f"📈 Latest price: ${data['Close'].iloc[-1]:.2f}")
        print(f"⏰ Last update: {data.index[-1]}")
        
        return data
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        print(f"\n🔧 CALCULATING TECHNICAL INDICATORS")
        print("=" * 50)
        
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        print(f"✅ SMA 20: ${data['SMA_20'].iloc[-1]:.2f}")
        print(f"✅ SMA 50: ${data['SMA_50'].iloc[-1]:.2f}")
        
        # RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        print(f"✅ RSI: {data['RSI'].iloc[-1]:.1f}")
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
        print(f"✅ Bollinger Upper: ${data['BB_Upper'].iloc[-1]:.2f}")
        print(f"✅ Bollinger Lower: ${data['BB_Lower'].iloc[-1]:.2f}")
        
        return data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def generate_signals(self, data):
        """Generate trading signals with detailed explanation"""
        print(f"\n🤖 GENERATING AI TRADING SIGNALS")
        print("=" * 50)
        
        if len(data) < 50:
            print("❌ Insufficient data for signal generation")
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = []
        reasons = []
        
        print(f"📊 CURRENT MARKET STATE:")
        print(f"   Price: ${latest['Close']:.2f}")
        print(f"   Previous: ${prev['Close']:.2f}")
        
        # Signal 1: Moving Average Crossover
        print(f"\n🔍 SIGNAL 1: MOVING AVERAGE CROSSOVER")
        print(f"   SMA 20: ${latest['SMA_20']:.2f}")
        print(f"   SMA 50: ${latest['SMA_50']:.2f}")
        print(f"   Previous SMA 20: ${prev['SMA_20']:.2f}")
        print(f"   Previous SMA 50: ${prev['SMA_50']:.2f}")
        
        if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals.append('BUY')
            reasons.append('SMA20 crossed above SMA50')
            print("   🟢 BULLISH CROSSOVER - BUY SIGNAL")
        elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals.append('SELL')
            reasons.append('SMA20 crossed below SMA50')
            print("   🔴 BEARISH CROSSOVER - SELL SIGNAL")
        else:
            print("   ⚪ NO CROSSOVER - NO SIGNAL")
        
        # Signal 2: RSI Oscillator
        print(f"\n🔍 SIGNAL 2: RSI MOMENTUM")
        print(f"   RSI: {latest['RSI']:.1f}")
        print(f"   Oversold threshold: 30")
        print(f"   Overbought threshold: 70")
        
        if latest['RSI'] < 30:
            signals.append('BUY')
            reasons.append(f'RSI oversold ({latest["RSI"]:.1f})')
            print("   🟢 OVERSOLD - BUY SIGNAL")
        elif latest['RSI'] > 70:
            signals.append('SELL')
            reasons.append(f'RSI overbought ({latest["RSI"]:.1f})')
            print("   🔴 OVERBOUGHT - SELL SIGNAL")
        else:
            print("   ⚪ NEUTRAL ZONE - NO SIGNAL")
        
        # Signal 3: Bollinger Bands
        print(f"\n🔍 SIGNAL 3: BOLLINGER BANDS")
        print(f"   Price: ${latest['Close']:.2f}")
        print(f"   Upper Band: ${latest['BB_Upper']:.2f}")
        print(f"   Lower Band: ${latest['BB_Lower']:.2f}")
        
        if latest['Close'] < latest['BB_Lower']:
            signals.append('BUY')
            reasons.append('Price below lower Bollinger Band')
            print("   🟢 PRICE BELOW LOWER BAND - BUY SIGNAL")
        elif latest['Close'] > latest['BB_Upper']:
            signals.append('SELL')
            reasons.append('Price above upper Bollinger Band')
            print("   🔴 PRICE ABOVE UPPER BAND - SELL SIGNAL")
        else:
            print("   ⚪ PRICE WITHIN BANDS - NO SIGNAL")
        
        # Signal 4: Price Momentum
        price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        print(f"\n🔍 SIGNAL 4: PRICE MOMENTUM")
        print(f"   Price change: {price_change:.2f}%")
        print(f"   Bullish threshold: +0.5%")
        print(f"   Bearish threshold: -0.5%")
        
        if price_change > 0.5:
            signals.append('BUY')
            reasons.append(f'Strong upward momentum ({price_change:.2f}%)')
            print("   🟢 STRONG UPWARD MOMENTUM - BUY SIGNAL")
        elif price_change < -0.5:
            signals.append('SELL')
            reasons.append(f'Strong downward momentum ({price_change:.2f}%)')
            print("   🔴 STRONG DOWNWARD MOMENTUM - SELL SIGNAL")
        else:
            print("   ⚪ WEAK MOMENTUM - NO SIGNAL")
        
        # Aggregate signals
        print(f"\n📈 SIGNAL AGGREGATION")
        print("=" * 30)
        print(f"Total signals generated: {len(signals)}")
        print(f"Signals: {signals}")
        
        if not signals:
            final_signal = {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No clear signal'}
            print("🔘 FINAL DECISION: HOLD (No signals)")
        else:
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            total_signals = len(signals)
            
            print(f"BUY signals: {buy_signals}")
            print(f"SELL signals: {sell_signals}")
            
            if buy_signals > sell_signals:
                confidence = buy_signals / total_signals
                final_signal = {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': '; '.join(reasons),
                    'price': latest['Close'],
                    'signals_count': f'{buy_signals}/{total_signals}'
                }
                print(f"🟢 FINAL DECISION: BUY (Confidence: {confidence:.1%})")
            elif sell_signals > buy_signals:
                confidence = sell_signals / total_signals
                final_signal = {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': '; '.join(reasons),
                    'price': latest['Close'],
                    'signals_count': f'{sell_signals}/{total_signals}'
                }
                print(f"🔴 FINAL DECISION: SELL (Confidence: {confidence:.1%})")
            else:
                final_signal = {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conflicting signals'}
                print("🔘 FINAL DECISION: HOLD (Conflicting signals)")
        
        return final_signal
    
    def trading_decision(self, signal):
        """Make final trading decision based on signal"""
        print(f"\n💡 TRADING DECISION ENGINE")
        print("=" * 40)
        
        confidence_threshold = 0.7  # 70%
        
        print(f"Signal: {signal['action']}")
        print(f"Confidence: {signal['confidence']:.1%}")
        print(f"Minimum threshold: {confidence_threshold:.1%}")
        print(f"Reason: {signal['reason']}")
        
        if signal['action'] == 'HOLD':
            print(f"\n⏸️ DECISION: HOLD POSITION")
            print(f"   No trading action required")
            return False
        elif signal['confidence'] >= confidence_threshold:
            print(f"\n✅ DECISION: EXECUTE {signal['action']} ORDER")
            print(f"   Signal confidence meets threshold")
            print(f"   Would place: {signal['action']} 1 ES contract")
            return True
        else:
            print(f"\n❌ DECISION: HOLD POSITION")
            print(f"   Signal confidence below threshold")
            print(f"   Required: {confidence_threshold:.1%}, Got: {signal['confidence']:.1%}")
            return False

def demonstrate_ml_system():
    """Complete demonstration of ML trading system"""
    print("🤖 LIVE ML TRADING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 Instrument: ES Futures (December 2024)")
    print(f"🎯 Analysis: Multi-signal technical analysis")
    
    demo = SignalDemo()
    
    # Step 1: Get market data
    data = demo.get_market_data()
    if data is None:
        return
    
    # Step 2: Calculate indicators
    data = demo.calculate_indicators(data)
    
    # Step 3: Generate signals
    signal = demo.generate_signals(data)
    if signal is None:
        return
    
    # Step 4: Trading decision
    would_trade = demo.trading_decision(signal)
    
    # Summary
    print(f"\n📊 DEMONSTRATION SUMMARY")
    print("=" * 40)
    print(f"✅ Market data: Retrieved and processed")
    print(f"✅ Technical indicators: Calculated")
    print(f"✅ AI signals: Generated with {signal['confidence']:.1%} confidence")
    print(f"✅ Trading decision: {'Execute order' if would_trade else 'Hold position'}")
    
    if would_trade:
        print(f"\n🎯 IN LIVE SYSTEM:")
        print(f"   This signal would trigger a real {signal['action']} order")
        print(f"   Order would be sent to NinjaTrader via AddOn")
        print(f"   Risk management would verify position limits")
        print(f"   Trade would be logged to database")

if __name__ == "__main__":
    try:
        demonstrate_ml_system()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    except Exception as e:
        print(f"\nError: {e}")
    
    input("\nPress Enter to exit...")