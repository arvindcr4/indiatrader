# 🔴 IndiaTrader Live Market Data

Enhanced GUI application with real-time market data loading capabilities for Indian stock exchanges (NSE/BSE).

## 🚀 Features

### Live Data Loading
- **🔴 Load Live Data Button**: One-click access to real-time market data
- **📊 Multiple Data Types**: Quote, Historical, and Index data
- **🔄 Refresh Functionality**: Update live data with a single click
- **🎯 Exchange Support**: NSE and BSE market data

### Data Types Available

#### 1. Quote Data 📈
- Real-time price information
- Current open, high, low, close (OHLC)
- Volume and trading statistics
- Price change and percentage change

#### 2. Historical Data 📊
- Last 30 days of daily OHLCV data
- Trend analysis and historical patterns
- Volume and price movement tracking

#### 3. Index Data 📉
- Major index values (NIFTY 50, BANK NIFTY, etc.)
- Real-time index movements
- Sector-wise index performance

## 🎨 GUI Features

### Professional Interface
- **Zerodha-inspired dark theme**
- **Responsive layout** with tabbed interface
- **Real-time status updates**
- **Professional data visualization**

### Enhanced Controls
- Load CSV Data (existing functionality)
- **🔴 Load Live Data** (new feature)
- **🔄 Refresh** live data
- Run Strategy analysis
- Export functionality

## 🛠️ How to Use

### 1. Launch the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Launch the live data GUI
python live_data_gui.py
```

### 2. Load Live Data
1. Click the **"🔴 Load Live Data"** button
2. Enter symbol details in the dialog:
   - **Symbol**: NIFTY, RELIANCE, TCS, etc.
   - **Exchange**: NSE or BSE
   - **Data Type**: Quote, Historical, or Indices
3. Click **"Load Data"** to fetch live market data

### 3. Refresh Data
- Use the **"🔄 Refresh"** button to update current live data
- Maintains the same symbol and parameters from last load

## 📊 Supported Symbols

### NSE Symbols
- **Indices**: NIFTY, BANKNIFTY
- **Stocks**: RELIANCE, TCS, HDFC, INFY, ICICIBANK
- **Any NSE-listed stock symbol**

### BSE Symbols
- Use **scrip codes** (e.g., 500325 for Reliance)
- BSE-listed stock symbols

## 🔧 Technical Implementation

### Architecture
- **Threading Support**: Non-blocking data loading
- **Error Handling**: Graceful fallback to demo data
- **API Integration**: NSE/BSE market data connectors
- **Demo Mode**: Simulated data when APIs unavailable

### Data Processing
- **Real-time Updates**: Background thread processing
- **Data Validation**: Automatic data type conversion
- **Performance Optimization**: Efficient data display (500 records limit)

## 🎯 Demo Mode

When live APIs are not configured, the application automatically switches to **demo mode**:

- **Realistic Data**: Simulated market data with proper price movements
- **Multiple Symbols**: Support for various symbols and indices
- **Time-based Data**: Historical trends and patterns
- **Visual Indicators**: Clear indication of demo mode usage

## 🔗 Integration

### Strategy Analysis
- Loaded data is automatically available for strategy analysis
- **Run Strategy** button processes live or historical data
- Compatible with existing Adam Mancini trading strategies

### Export Functionality
- Export live data to CSV/Excel formats
- Maintain data for offline analysis
- Integration with existing export workflows

## 📱 User Interface

### Main Controls
```
📊 Load CSV Data | 🔴 Load Live Data | 🔄 Refresh | 📈 Run Strategy
```

### Status Indicators
- **🟢 Live Data Loaded**: Successfully loaded live data
- **🔴 Market Closed**: Market status indicator
- **🔄 Loading**: Data fetch in progress
- **❌ Error**: Error state with details

### Data Display
- **Professional table**: OHLCV data with formatting
- **Color coding**: Price movements and changes
- **Scrollable interface**: Handle large datasets
- **Real-time updates**: Live data refresh capability

## 🚀 Getting Started

### Prerequisites
```bash
# Install required dependencies
pip install pandas numpy tkinter requests datetime threading
```

### Quick Start
```bash
# 1. Navigate to project directory
cd indiatrader

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch live data GUI
python live_data_gui.py

# 4. Click "🔴 Load Live Data" and enter:
#    Symbol: NIFTY
#    Exchange: NSE  
#    Data Type: Quote

# 5. Click "Load Data" to see live market data!
```

## 🎓 Example Usage

### Loading NIFTY Live Quote
1. Symbol: `NIFTY`
2. Exchange: `NSE`
3. Data Type: `Quote`
4. Result: Real-time NIFTY price, volume, and change data

### Loading Historical Data
1. Symbol: `RELIANCE`
2. Exchange: `NSE`
3. Data Type: `Historical`
4. Result: Last 30 days of RELIANCE OHLCV data

### Loading Index Data
1. Symbol: `Any` (ignored for indices)
2. Exchange: `NSE`
3. Data Type: `Indices`
4. Result: All major NSE indices with current values

## 🔧 Configuration

### API Setup (Optional)
- Configure NSE/BSE API credentials in `indiatrader/data/config.py`
- Without API setup, demo mode will be used automatically
- Demo mode provides full functionality with simulated data

### Customization
- Modify color themes in `_setup_zerodha_theme()`
- Adjust data refresh intervals
- Configure supported exchanges and symbols

## 📋 Testing

### Run Tests
```bash
# Test live data functionality
python test_live_data.py
```

### Test Coverage
- ✅ Demo data generation
- ✅ Market data connector imports
- ✅ GUI component functionality
- ✅ Error handling and fallbacks

## 🎯 Next Steps

### Planned Enhancements
- **Auto-refresh**: Configurable automatic data refresh
- **Multiple symbols**: Load data for multiple symbols simultaneously
- **Advanced charting**: Integration with plotting libraries
- **Real-time alerts**: Price-based notifications
- **WebSocket support**: True real-time streaming data

---

## 📞 Support

For issues or questions about the live data functionality:
1. Check the demo mode works with `python test_live_data.py`
2. Verify GUI launches with `python live_data_gui.py`
3. Review error messages in the status bar
4. Fallback to CSV data loading if live data fails

**Happy Trading!** 🚀📈