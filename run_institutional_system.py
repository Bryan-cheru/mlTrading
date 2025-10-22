#!/usr/bin/env python3
"""
Run the Professional Institutional Trading System
Integrates Rithmic R|API with ML trading algorithms
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

async def main():
    print("Professional Institutional Trading System")
    print("=" * 60)
    
    try:
        from institutional_trading_system import MathematicalMLTradingSystem
        
        # Initialize and start system
        system = MathematicalMLTradingSystem()
        await system.start_system()
        
    except KeyboardInterrupt:
        print("System shutdown requested")
    except Exception as e:
        print(f"System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
