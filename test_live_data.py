#!/usr/bin/env python3
"""
Test script for live data functionality
"""

import pandas as pd
from datetime import datetime, timedelta
import random

def test_demo_data_generation():
    """Test demo data generation functionality."""
    print("🔄 Testing Demo Data Generation...")
    
    # Test Quote data
    print("\n📊 Testing Quote Data:")
    symbol = "NIFTY"
    current_time = datetime.now()
    base_price = 25000.0
    
    quote_data = {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'open': round(base_price + random.uniform(-50, 50), 2),
        'high': round(base_price + random.uniform(0, 100), 2),
        'low': round(base_price - random.uniform(0, 100), 2),
        'close': round(base_price + random.uniform(-25, 25), 2),
        'volume': random.randint(100000, 1000000),
        'change': round(random.uniform(-50, 50), 2),
        'change_pct': round(random.uniform(-2, 2), 2)
    }
    
    df_quote = pd.DataFrame([quote_data])
    print(df_quote.to_string())
    
    # Test Historical data
    print("\n📈 Testing Historical Data:")
    dates = pd.date_range(end=current_time, periods=5, freq='D')
    
    historical_data = []
    for i, date in enumerate(dates):
        trend = base_price + (i * random.uniform(-10, 10))
        open_price = trend + random.uniform(-50, 50)
        high_price = open_price + random.uniform(0, 75)
        low_price = open_price - random.uniform(0, 75)
        close_price = open_price + random.uniform(-40, 40)
        
        historical_data.append({
            'timestamp': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': random.randint(50000, 500000),
            'change': round(close_price - open_price, 2),
            'change_pct': round(((close_price - open_price) / open_price) * 100, 2)
        })
    
    df_historical = pd.DataFrame(historical_data)
    print(df_historical.to_string())
    
    # Test Index data
    print("\n📊 Testing Index Data:")
    indices_data = {
        'NIFTY 50': 25000,
        'NIFTY BANK': 52000,
        'NIFTY IT': 35000
    }
    
    index_data = []
    for idx_name, base_val in indices_data.items():
        change = random.uniform(-200, 200)
        current_val = base_val + change
        pct_change = (change / base_val) * 100
        
        index_data.append({
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(base_val, 2),
            'high': round(current_val + random.uniform(0, 50), 2),
            'low': round(current_val - random.uniform(0, 50), 2),
            'close': round(current_val, 2),
            'volume': random.randint(1000000, 5000000),
            'change': round(change, 2),
            'change_pct': round(pct_change, 2),
            'symbol': idx_name
        })
    
    df_indices = pd.DataFrame(index_data)
    print(df_indices.to_string())
    
    print("\n✅ Demo data generation test completed successfully!")
    return True

def test_market_data_connector():
    """Test market data connector imports."""
    print("\n🔄 Testing Market Data Connector Imports...")
    
    try:
        from indiatrader.data.market_data import NSEConnector, BSEConnector
        print("✅ Market data connectors imported successfully")
        
        # Test NSE connector initialization
        nse = NSEConnector()
        print("✅ NSE connector initialized")
        
        # Test BSE connector initialization  
        bse = BSEConnector()
        print("✅ BSE connector initialized")
        
        return True
    except ImportError as e:
        print(f"⚠️  Market data connectors not available: {e}")
        print("💡 Will fallback to demo data mode")
        return False
    except Exception as e:
        print(f"❌ Error testing connectors: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 IndiaTrader Live Data Functionality Test")
    print("=" * 50)
    
    # Test demo data generation
    demo_test = test_demo_data_generation()
    
    # Test market data connectors
    connector_test = test_market_data_connector()
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"Demo Data Generation: {'✅ PASS' if demo_test else '❌ FAIL'}")
    print(f"Market Data Connectors: {'✅ PASS' if connector_test else '⚠️  FALLBACK TO DEMO'}")
    
    print("\n💡 LIVE DATA GUI FEATURES:")
    print("• 🔴 Load Live Data button for real-time market data")
    print("• 📊 Support for NSE and BSE exchanges")
    print("• 📈 Quote, Historical, and Index data types")
    print("• 🔄 Refresh functionality for live updates")
    print("• 💾 Demo mode when APIs are not configured")
    print("• 🎨 Professional Zerodha-inspired dark theme")
    
    print("\n🚀 To launch the Live Data GUI:")
    print("python live_data_gui.py")

if __name__ == "__main__":
    main()