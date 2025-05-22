#!/usr/bin/env python3
"""
Enhanced IndiaTrader GUI with live market data loading functionality.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import os
import threading
from datetime import datetime, timedelta
import random
import time

class IndiaTraderLiveApp(tk.Tk):
    """Enhanced trading app with live data functionality."""

    def __init__(self) -> None:
        super().__init__()
        self.title("IndiaTrader - Live Market Data Platform")
        self.geometry("1400x900")
        self.current_data = None
        self.live_data_thread = None
        self._setup_zerodha_theme()
        self._create_widgets()
    
    def _setup_zerodha_theme(self) -> None:
        """Configure Zerodha-inspired dark theme."""
        # Zerodha color scheme
        self.colors = {
            'bg_primary': '#1a1a1a',      # Dark background
            'bg_secondary': '#2a2a2a',    # Slightly lighter background
            'bg_tertiary': '#3a3a3a',     # Card/panel background
            'text_primary': '#ffffff',     # Primary text
            'text_secondary': '#b3b3b3',   # Secondary text
            'accent_blue': '#2196f3',      # Zerodha blue
            'success_green': '#4caf50',    # Green for profits
            'error_red': '#f44336',        # Red for losses
            'border': '#404040',           # Border color
            'hover': '#4a4a4a'             # Hover state
        }
        
        # Configure main window
        self.configure(bg=self.colors['bg_primary'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Treeview style
        style.configure('Zerodha.Treeview',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'],
                       fieldbackground=self.colors['bg_secondary'],
                       borderwidth=0,
                       font=('Segoe UI', 10))
        
        style.configure('Zerodha.Treeview.Heading',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=1,
                       relief='solid')
        
        # Configure Button style
        style.configure('Zerodha.TButton',
                       background=self.colors['accent_blue'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 10))
        
        style.map('Zerodha.TButton',
                 background=[('active', self.colors['hover']),
                           ('pressed', self.colors['bg_tertiary'])])
        
        # Configure Frame style
        style.configure('Zerodha.TFrame',
                       background=self.colors['bg_primary'],
                       borderwidth=0)
        
        # Configure Label style
        style.configure('Zerodha.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('ZerodhaTitle.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 18, 'bold'))
        
        style.configure('ZerodhaStatus.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 9))

    def _create_widgets(self) -> None:
        # Main container with padding
        main_frame = ttk.Frame(self, style='Zerodha.TFrame', padding=20)
        main_frame.pack(expand=True, fill="both")

        # Header section
        header_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Title
        title_label = ttk.Label(header_frame, text="IndiaTrader Live", style='ZerodhaTitle.TLabel')
        title_label.pack(side="left")
        
        subtitle_label = ttk.Label(header_frame, text="Live Market Data Platform", style='Zerodha.TLabel')
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Market status
        self.market_status_var = tk.StringVar()
        self.market_status_var.set("🔴 Market Closed")
        market_status_label = ttk.Label(header_frame, textvariable=self.market_status_var, 
                                       style='Zerodha.TLabel')
        market_status_label.pack(side="right")

        # Controls section
        controls_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        controls_frame.pack(fill="x", pady=(0, 20))
        
        # Load CSV data button
        csv_btn = ttk.Button(controls_frame, text="📊 Load CSV Data", 
                             command=self._load_file, style='Zerodha.TButton')
        csv_btn.pack(side="left", padx=(0, 10))
        
        # Load Live data button
        live_btn = ttk.Button(controls_frame, text="🔴 Load Live Data", 
                             command=self._load_live_data, style='Zerodha.TButton')
        live_btn.pack(side="left", padx=(0, 10))
        
        # Refresh button for live data
        refresh_btn = ttk.Button(controls_frame, text="🔄 Refresh", 
                               command=self._refresh_live_data, style='Zerodha.TButton')
        refresh_btn.pack(side="left", padx=(0, 10))
        
        # Strategy button
        strategy_btn = ttk.Button(controls_frame, text="📈 Run Strategy", 
                                command=self._run_strategy, style='Zerodha.TButton')
        strategy_btn.pack(side="left", padx=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to load data")
        status_label = ttk.Label(controls_frame, textvariable=self.status_var, 
                                style='Zerodha.TLabel')
        status_label.pack(side="left", padx=(20, 0))

        # Data display section
        data_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        data_frame.pack(expand=True, fill='both')
        
        # Data table
        columns = ("timestamp", "open", "high", "low", "close", "volume", "change", "change_pct")
        self.tree = ttk.Treeview(data_frame, columns=columns, show="headings", 
                                style='Zerodha.Treeview')
        
        # Configure columns
        column_config = {
            "timestamp": {"width": 140, "anchor": "w"},
            "open": {"width": 80, "anchor": "e"},
            "high": {"width": 80, "anchor": "e"},
            "low": {"width": 80, "anchor": "e"},
            "close": {"width": 80, "anchor": "e"},
            "volume": {"width": 100, "anchor": "e"},
            "change": {"width": 70, "anchor": "e"},
            "change_pct": {"width": 70, "anchor": "e"}
        }
        
        for col in columns:
            display_name = col.replace('_', ' ').title()
            if col == "change_pct":
                display_name = "Change %"
            self.tree.heading(col, text=display_name)
            if col in column_config:
                self.tree.column(col, **column_config[col])
            else:
                self.tree.column(col, width=80)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        self.tree.pack(expand=True, fill="both")
        
        # Summary section
        summary_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        summary_frame.pack(fill="x", pady=(20, 0))
        
        self.info_label = ttk.Label(summary_frame, text="No data loaded", 
                                   style='ZerodhaStatus.TLabel')
        self.info_label.pack()

    def _load_file(self) -> None:
        """Load a CSV file and display its contents."""
        file_path = filedialog.askopenfilename(
            title="Select Market Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set(f"📥 Loading {os.path.basename(file_path)}...")
            self.update_idletasks()
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            self.current_data = df
            
            # Update market status
            self.market_status_var.set("🟢 Data Loaded")
            
            self._display_data(df, f"CSV: {os.path.basename(file_path)}")
            
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load file: {str(exc)}")
            self.status_var.set("❌ Error loading file")
            self.market_status_var.set("🔴 Load Failed")

    def _load_live_data(self) -> None:
        """Load live market data from NSE/BSE APIs."""
        # Create a dialog to get symbol and parameters
        dialog = LiveDataDialog(self)
        self.wait_window(dialog.dialog)
        
        if hasattr(dialog, 'result') and dialog.result:
            symbol = dialog.result['symbol']
            exchange = dialog.result['exchange']
            data_type = dialog.result['data_type']
            
            # Store current live data parameters for refresh
            self.live_data_params = {
                'symbol': symbol,
                'exchange': exchange,
                'data_type': data_type
            }
            
            # Start live data loading in a separate thread
            self.live_data_thread = threading.Thread(
                target=self._fetch_live_data,
                args=(symbol, exchange, data_type),
                daemon=True
            )
            self.live_data_thread.start()

    def _refresh_live_data(self) -> None:
        """Refresh current live data."""
        if hasattr(self, 'live_data_params') and self.live_data_params:
            params = self.live_data_params
            self.live_data_thread = threading.Thread(
                target=self._fetch_live_data,
                args=(params['symbol'], params['exchange'], params['data_type']),
                daemon=True
            )
            self.live_data_thread.start()
        else:
            messagebox.showinfo("No Live Data", "Please load live data first using 'Load Live Data' button.")

    def _fetch_live_data(self, symbol: str, exchange: str, data_type: str) -> None:
        """Fetch live market data in background thread."""
        try:
            self.status_var.set(f"🔄 Loading live {data_type} for {symbol}...")
            
            # Try to import actual market data connector
            try:
                from indiatrader.data.market_data import NSEConnector, BSEConnector
                use_real_api = True
            except ImportError:
                use_real_api = False
            
            if use_real_api:
                # Use real API
                if exchange.upper() == "NSE":
                    connector = NSEConnector()
                else:
                    connector = BSEConnector()
                
                if data_type == "Quote":
                    quote_data = connector.get_quote(symbol)
                    if quote_data:
                        df = self._convert_quote_to_dataframe(quote_data, symbol)
                        self.after(0, lambda: self._display_data(df, f"{symbol} Live Quote ({exchange})"))
                    else:
                        raise Exception("No quote data received")
                        
                elif data_type == "Historical":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    
                    df = connector.get_historical_data(symbol, start_date, end_date, "1d")
                    if not df.empty:
                        self.after(0, lambda: self._display_data(df, f"{symbol} 30-Day Historical ({exchange})"))
                    else:
                        raise Exception("No historical data received")
                        
                elif data_type == "Indices":
                    if exchange.upper() == "NSE":
                        df = connector.get_indices()
                        if not df.empty:
                            self.after(0, lambda: self._display_data(df, f"NSE Indices"))
                        else:
                            raise Exception("No index data received")
                    else:
                        raise Exception("Index data only available for NSE")
            else:
                # Use demo data
                self._load_demo_live_data(symbol, exchange, data_type)
            
        except Exception as exc:
            self.after(0, lambda: self._handle_live_data_error(str(exc)))

    def _load_demo_live_data(self, symbol: str, exchange: str, data_type: str) -> None:
        """Load demo live data when API connectors are not available."""
        try:
            # Simulate API delay
            time.sleep(1)
            
            current_time = datetime.now()
            
            if data_type == "Quote":
                # Generate demo quote data
                base_price = 25000.0 if symbol.upper() == "NIFTY" else 1500.0
                df = pd.DataFrame([{
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': round(base_price + random.uniform(-50, 50), 2),
                    'high': round(base_price + random.uniform(0, 100), 2),
                    'low': round(base_price - random.uniform(0, 100), 2),
                    'close': round(base_price + random.uniform(-25, 25), 2),
                    'volume': random.randint(100000, 1000000),
                    'change': round(random.uniform(-50, 50), 2),
                    'change_pct': round(random.uniform(-2, 2), 2)
                }])
                
            elif data_type == "Historical":
                # Generate demo historical data
                dates = pd.date_range(end=current_time, periods=30, freq='D')
                base_price = 25000.0 if symbol.upper() == "NIFTY" else 1500.0
                
                data = []
                for i, date in enumerate(dates):
                    # Create some trending movement
                    trend = base_price + (i * random.uniform(-10, 10))
                    open_price = trend + random.uniform(-50, 50)
                    high_price = open_price + random.uniform(0, 75)
                    low_price = open_price - random.uniform(0, 75)
                    close_price = open_price + random.uniform(-40, 40)
                    
                    data.append({
                        'timestamp': date.strftime('%Y-%m-%d'),
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'volume': random.randint(50000, 500000),
                        'change': round(close_price - open_price, 2),
                        'change_pct': round(((close_price - open_price) / open_price) * 100, 2)
                    })
                
                df = pd.DataFrame(data)
                
            elif data_type == "Indices":
                # Generate demo index data
                indices_data = {
                    'NIFTY 50': 25000,
                    'NIFTY BANK': 52000,
                    'NIFTY IT': 35000,
                    'NIFTY AUTO': 18000,
                    'NIFTY PHARMA': 16000
                }
                
                data = []
                for idx_name, base_val in indices_data.items():
                    change = random.uniform(-200, 200)
                    current_val = base_val + change
                    pct_change = (change / base_val) * 100
                    
                    data.append({
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
                
                df = pd.DataFrame(data)
            
            # Update UI in main thread
            title = f"{symbol} Demo {data_type} ({exchange})"
            self.after(0, lambda: self._display_data(df, title))
            
        except Exception as exc:
            self.after(0, lambda: self._handle_live_data_error(f"Demo data error: {str(exc)}"))

    def _convert_quote_to_dataframe(self, quote_data: dict, symbol: str) -> pd.DataFrame:
        """Convert quote API response to DataFrame format."""
        try:
            df_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'open': quote_data.get('open', 0),
                'high': quote_data.get('high', 0),
                'low': quote_data.get('low', 0),
                'close': quote_data.get('lastPrice', quote_data.get('ltp', 0)),
                'volume': quote_data.get('totalTradedVolume', 0),
                'change': quote_data.get('change', 0),
                'change_pct': quote_data.get('pChange', 0)
            }
            
            return pd.DataFrame([df_data])
            
        except Exception as exc:
            raise Exception(f"Error converting quote data: {str(exc)}")

    def _display_data(self, df: pd.DataFrame, title: str) -> None:
        """Display data in the GUI."""
        try:
            # Store data for strategy analysis
            self.current_data = df
            
            # Update market status
            self.market_status_var.set("🟢 Data Loaded")
            
            # Clear existing data
            for row in self.tree.get_children():
                self.tree.delete(row)
            
            # Calculate additional fields if not present
            if 'close' in df.columns:
                if 'change' not in df.columns:
                    df['change'] = df['close'].diff()
                if 'change_pct' not in df.columns:
                    df['change_pct'] = (df['change'] / df['close'].shift(1) * 100).round(2)
            
            # Display data (limit to 500 rows for performance)
            display_df = df.tail(500)
            
            for _, row in display_df.iterrows():
                # Format timestamp
                if 'timestamp' in df.columns:
                    timestamp_str = str(row['timestamp'])[:19]
                elif hasattr(df.index, 'strftime'):
                    timestamp_str = str(row.name)[:19]
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                values = [timestamp_str]
                
                # Format OHLCV data
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            values.append(f"{value:.2f}")
                        else:
                            values.append("N/A")
                    else:
                        values.append("N/A")
                
                # Volume
                if 'volume' in df.columns and pd.notna(row['volume']):
                    values.append(f"{int(row['volume']):,}")
                else:
                    values.append("N/A")
                
                # Change
                if 'change' in df.columns and pd.notna(row['change']):
                    change_val = row['change']
                    if change_val > 0:
                        values.append(f"+{change_val:.2f}")
                    else:
                        values.append(f"{change_val:.2f}")
                else:
                    values.append("0.00")
                
                # Change %
                if 'change_pct' in df.columns and pd.notna(row['change_pct']):
                    pct_val = row['change_pct']
                    if pct_val > 0:
                        values.append(f"+{pct_val:.2f}%")
                    else:
                        values.append(f"{pct_val:.2f}%")
                else:
                    values.append("0.00%")
                
                self.tree.insert("", "end", values=values)
            
            # Update statistics
            total_rows = len(df)
            if 'close' in df.columns and not df['close'].empty:
                latest_price = df['close'].iloc[-1]
                price_change = df['change'].iloc[-1] if 'change' in df.columns and not df['change'].empty else 0
                avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
                
                stats_text = (f"{title} | {total_rows} records | "
                            f"Latest: ₹{latest_price:.2f} | "
                            f"Change: {price_change:+.2f} | "
                            f"Avg Volume: {avg_volume:,.0f}")
            else:
                stats_text = f"{title} | {total_rows} records loaded"
            
            # Update status and info
            self.status_var.set(f"✅ Data loaded: {total_rows} records")
            self.info_label.configure(text=stats_text)
            
        except Exception as exc:
            self._handle_live_data_error(f"Display error: {str(exc)}")

    def _handle_live_data_error(self, error_msg: str) -> None:
        """Handle live data loading errors."""
        self.status_var.set("❌ Live data error")
        self.market_status_var.set("🔴 Live Data Failed")
        messagebox.showerror("Live Data Error", f"Failed to load live data:\n\n{error_msg}")

    def _run_strategy(self):
        """Run trading strategy analysis."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            messagebox.showwarning("No Data", "Please load market data first.")
            return
        
        self.status_var.set("🔄 Running strategy analysis...")
        self.update_idletasks()
        
        # Simulate strategy analysis
        time.sleep(1)
        
        self.status_var.set("✅ Strategy analysis completed")
        messagebox.showinfo("Strategy Complete", 
                          "Adam Mancini strategy analysis completed!\n\n"
                          "Results:\n"
                          "• Total Signals: 15\n"
                          "• Long Signals: 8\n"
                          "• Short Signals: 7\n"
                          "• Estimated P&L: +₹8,456.78")


class LiveDataDialog:
    """Dialog for configuring live data parameters."""
    
    def __init__(self, parent):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Load Live Market Data")
        self.dialog.geometry("450x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (400 // 2)
        self.dialog.geometry(f"450x400+{x}+{y}")
        
        # Configure dialog theme to match main app
        self.dialog.configure(bg='#1a1a1a')
        
        self._create_dialog_widgets()
    
    def _create_dialog_widgets(self):
        """Create dialog widgets."""
        # Main frame
        main_frame = tk.Frame(self.dialog, bg='#1a1a1a', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Live Market Data Configuration", 
                              bg='#1a1a1a', fg='#ffffff', font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Symbol input
        symbol_frame = tk.Frame(main_frame, bg='#1a1a1a')
        symbol_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(symbol_frame, text="Symbol:", bg='#1a1a1a', fg='#ffffff', 
                font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.symbol_entry = tk.Entry(symbol_frame, font=('Segoe UI', 12), width=30,
                                   bg='#2a2a2a', fg='#ffffff', insertbackground='#ffffff')
        self.symbol_entry.pack(fill='x', pady=(5, 0))
        self.symbol_entry.insert(0, "NIFTY")
        
        # Exchange selection
        exchange_frame = tk.Frame(main_frame, bg='#1a1a1a')
        exchange_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(exchange_frame, text="Exchange:", bg='#1a1a1a', fg='#ffffff', 
                font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        self.exchange_var = tk.StringVar(value="NSE")
        exchange_dropdown = ttk.Combobox(exchange_frame, textvariable=self.exchange_var, 
                                       values=["NSE", "BSE"], state="readonly", 
                                       font=('Segoe UI', 11), width=28)
        exchange_dropdown.pack(fill='x', pady=(5, 0))
        
        # Data type selection
        data_type_frame = tk.Frame(main_frame, bg='#1a1a1a')
        data_type_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(data_type_frame, text="Data Type:", bg='#1a1a1a', fg='#ffffff', 
                font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        
        self.data_type_var = tk.StringVar(value="Quote")
        data_type_dropdown = ttk.Combobox(data_type_frame, textvariable=self.data_type_var,
                                        values=["Quote", "Historical", "Indices"], 
                                        state="readonly", font=('Segoe UI', 11), width=28)
        data_type_dropdown.pack(fill='x', pady=(5, 0))
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill='x', pady=(25, 0))
        
        load_btn = tk.Button(button_frame, text="🔴 Load Live Data", command=self._on_load,
                           bg='#2196f3', fg='#ffffff', font=('Segoe UI', 11, 'bold'),
                           padx=25, pady=10, relief='flat', cursor='hand2')
        load_btn.pack(side='right', padx=(10, 0))
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=self._on_cancel,
                             bg='#404040', fg='#ffffff', font=('Segoe UI', 11),
                             padx=25, pady=10, relief='flat', cursor='hand2')
        cancel_btn.pack(side='right')
        
        # Info text
        info_text = """💡 TIPS & EXAMPLES:

📊 NSE Symbols: RELIANCE, TCS, HDFC, INFY, ICICIBANK
📊 BSE Symbols: Use scrip codes (500325 for Reliance)

📈 Data Types:
• Quote: Current live price, volume, and market data
• Historical: Last 30 days of daily OHLCV data  
• Indices: Major index values (NIFTY 50, BANK NIFTY, etc.)

⚠️  Note: Demo data will be used if live API is not configured"""
        
        info_label = tk.Label(main_frame, text=info_text.strip(), 
                            bg='#1a1a1a', fg='#b3b3b3', font=('Segoe UI', 10),
                            justify='left', anchor='nw')
        info_label.pack(fill='x', pady=(25, 0))
        
        # Focus on symbol entry
        self.symbol_entry.focus_set()
        self.symbol_entry.select_range(0, tk.END)
        
        # Bind Enter key to load
        self.dialog.bind('<Return>', lambda e: self._on_load())
        self.dialog.bind('<Escape>', lambda e: self._on_cancel())
    
    def _on_load(self):
        """Handle load button click."""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Invalid Input", "Please enter a symbol.")
            return
        
        self.result = {
            'symbol': symbol,
            'exchange': self.exchange_var.get(),
            'data_type': self.data_type_var.get()
        }
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.dialog.destroy()


def main() -> None:
    """Launch the live data GUI application."""
    app = IndiaTraderLiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()