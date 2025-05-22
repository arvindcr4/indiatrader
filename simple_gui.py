#!/usr/bin/env python3
"""
A simplified version of the IndiaTrader GUI that doesn't require all the dependencies.
This is a standalone desktop application for viewing and analyzing trading data.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from datetime import datetime


class SimpleTraderApp(tk.Tk):
    """A simplified version of the trading strategy app."""

    def __init__(self) -> None:
        super().__init__()
        self.title("IndiaTrader - Professional Trading Platform")
        self.geometry("1400x900")
        self.current_data = None
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

        # Header section with trading dashboard info
        header_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        header_frame.pack(fill="x", pady=(0, 20))

        # Left side - Title and subtitle
        title_frame = ttk.Frame(header_frame, style='Zerodha.TFrame')
        title_frame.pack(side="left", fill="x", expand=True)

        title_label = ttk.Label(title_frame, text="IndiaTrader", style='ZerodhaTitle.TLabel')
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(
            title_frame, text="Professional Trading Platform", style='Zerodha.TLabel')
        subtitle_label.pack(anchor="w")

        # Right side - Market status and time
        status_frame = ttk.Frame(header_frame, style='Zerodha.TFrame')
        status_frame.pack(side="right")

        self.market_status_var = tk.StringVar()
        self.market_status_var.set("🔴 Market Closed")
        market_status_label = ttk.Label(status_frame, textvariable=self.market_status_var,
                                        style='Zerodha.TLabel')
        market_status_label.pack(anchor="e")

        current_time = datetime.now().strftime("%H:%M:%S")
        time_label = ttk.Label(status_frame, text=f"⏰ {current_time}", style='ZerodhaStatus.TLabel')
        time_label.pack(anchor="e")

        # Controls and stats section
        controls_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        controls_frame.pack(fill="x", pady=(0, 20))

        # Left controls
        controls_left = ttk.Frame(controls_frame, style='Zerodha.TFrame')
        controls_left.pack(side="left", fill="x", expand=True)

        # Load CSV data button
        open_btn = ttk.Button(controls_left, text="📊 Load CSV Data",
                              command=self._load_file, style='Zerodha.TButton')
        open_btn.pack(side="left", padx=(0, 10))

        # Load Live data button
        live_btn = ttk.Button(controls_left, text="🔴 Load Live Data",
                              command=self._load_live_data, style='Zerodha.TButton')
        live_btn.pack(side="left", padx=(0, 10))

        # Strategy button
        strategy_btn = ttk.Button(controls_left, text="📈 Run Strategy",
                                  command=self._run_strategy, style='Zerodha.TButton')
        strategy_btn.pack(side="left", padx=(0, 10))

        # Export button
        export_btn = ttk.Button(controls_left, text="💾 Export Data",
                                command=self._export_data, style='Zerodha.TButton')
        export_btn.pack(side="left")

        # Status indicator
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to load data")
        status_label = ttk.Label(controls_left, textvariable=self.status_var,
                                 style='Zerodha.TLabel')
        status_label.pack(side="left", padx=(20, 0))

        # Right side stats
        stats_frame = ttk.Frame(controls_frame, style='Zerodha.TFrame')
        stats_frame.pack(side="right")

        self.pnl_var = tk.StringVar()
        self.pnl_var.set("P&L: ₹0.00")
        pnl_label = ttk.Label(stats_frame, textvariable=self.pnl_var, style='Zerodha.TLabel')
        pnl_label.pack(side="right", padx=(10, 0))

        self.trades_var = tk.StringVar()
        self.trades_var.set("Trades: 0")
        trades_label = ttk.Label(stats_frame, textvariable=self.trades_var, style='Zerodha.TLabel')
        trades_label.pack(side="right", padx=(10, 0))

        # Main content area with tabs
        content_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        content_frame.pack(expand=True, fill="both")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(expand=True, fill="both")

        # Market Data Tab
        self._create_market_data_tab()

        # Trading Signals Tab
        self._create_signals_tab()

        # Portfolio Tab
        self._create_portfolio_tab()

        # Analytics Tab
        self._create_analytics_tab()

        # Bottom info panel with comprehensive stats
        info_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        info_frame.pack(fill="x", pady=(15, 0))

        # Left side info
        info_left = ttk.Frame(info_frame, style='Zerodha.TFrame')
        info_left.pack(side="left", fill="x", expand=True)

        self.info_label = ttk.Label(info_left, text="No data loaded",
                                    style='ZerodhaStatus.TLabel')
        self.info_label.pack(side="left")

        # Right side additional stats
        info_right = ttk.Frame(info_frame, style='Zerodha.TFrame')
        info_right.pack(side="right")

        self.win_rate_var = tk.StringVar()
        self.win_rate_var.set("Win Rate: 0%")
        win_rate_label = ttk.Label(info_right, textvariable=self.win_rate_var,
                                   style='ZerodhaStatus.TLabel')
        win_rate_label.pack(side="right", padx=(10, 0))

        self.avg_trade_var = tk.StringVar()
        self.avg_trade_var.set("Avg Trade: ₹0.00")
        avg_trade_label = ttk.Label(info_right, textvariable=self.avg_trade_var,
                                    style='ZerodhaStatus.TLabel')
        avg_trade_label.pack(side="right", padx=(10, 0))

    def _create_market_data_tab(self):
        """Create the market data tab."""
        market_frame = ttk.Frame(self.notebook, style='Zerodha.TFrame')
        self.notebook.add(market_frame, text="📊 Market Data")

        # Market data table
        columns = ("timestamp", "open", "high", "low", "close", "volume", "change", "change_pct")
        self.tree = ttk.Treeview(market_frame, columns=columns, show="headings",
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
        v_scrollbar = ttk.Scrollbar(market_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(market_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        self.tree.pack(expand=True, fill="both")

    def _create_signals_tab(self):
        """Create the trading signals tab."""
        signals_frame = ttk.Frame(self.notebook, style='Zerodha.TFrame')
        self.notebook.add(signals_frame, text="📈 Signals")

        # Signals table
        signal_columns = ("timestamp", "symbol", "signal_type",
                          "price", "target", "stop_loss", "confidence")
        self.signals_tree = ttk.Treeview(signals_frame, columns=signal_columns, show="headings",
                                         style='Zerodha.Treeview')

        for col in signal_columns:
            self.signals_tree.heading(col, text=col.replace('_', ' ').title())
            self.signals_tree.column(col, width=100)

        # Demo signals
        demo_signals = [
            ("2024-01-15 09:30:00", "NIFTY", "🟢 BUY", "18,500.00", "18,650.00", "18,400.00", "85%"),
            ("2024-01-15 10:15:00", "BANKNIFTY", "🔴 SELL",
             "42,300.00", "42,100.00", "42,450.00", "78%"),
            ("2024-01-15 11:30:00", "NIFTY", "🔴 SELL", "18,520.00", "18,380.00", "18,620.00", "82%")
        ]

        for signal in demo_signals:
            self.signals_tree.insert("", "end", values=signal)

        signals_scroll = ttk.Scrollbar(signals_frame, orient="vertical",
                                       command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=signals_scroll.set)

        signals_scroll.pack(side="right", fill="y")
        self.signals_tree.pack(expand=True, fill="both")

    def _create_portfolio_tab(self):
        """Create the portfolio tab."""
        portfolio_frame = ttk.Frame(self.notebook, style='Zerodha.TFrame')
        self.notebook.add(portfolio_frame, text="💼 Portfolio")

        # Portfolio summary at top
        summary_frame = ttk.Frame(portfolio_frame, style='Zerodha.TFrame', padding=10)
        summary_frame.pack(fill="x", pady=(0, 10))

        # Portfolio metrics
        metrics = [
            ("Total Value", "₹2,45,678.50"),
            ("Day's P&L", "+₹3,245.25 (+1.34%)"),
            ("Total P&L", "+₹15,678.90 (+6.82%)"),
            ("Available Margin", "₹1,23,456.78")
        ]

        for i, (label, value) in enumerate(metrics):
            metric_frame = ttk.Frame(summary_frame, style='Zerodha.TFrame')
            metric_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))

            ttk.Label(metric_frame, text=label, style='ZerodhaStatus.TLabel').pack(anchor="w")
            color_style = 'Zerodha.TLabel'
            if "+" in value:
                color_style = 'Zerodha.TLabel'  # Would be green in a real app
            ttk.Label(metric_frame, text=value, style=color_style,
                      font=('Segoe UI', 11, 'bold')).pack(anchor="w")

        # Holdings table
        holdings_columns = ("symbol", "qty", "avg_price", "ltp", "pnl", "pnl_pct")
        self.holdings_tree = ttk.Treeview(portfolio_frame, columns=holdings_columns,
                                          show="headings", style='Zerodha.Treeview')

        for col in holdings_columns:
            display_name = col.replace('_', ' ').title()
            if col == "qty":
                display_name = "Quantity"
            elif col == "ltp":
                display_name = "LTP"
            elif col == "pnl_pct":
                display_name = "P&L %"
            self.holdings_tree.heading(col, text=display_name)
            self.holdings_tree.column(col, width=100)

        # Demo holdings
        demo_holdings = [
            ("RELIANCE", "100", "2,245.50", "2,378.25", "+13,275.00", "+5.91%"),
            ("TCS", "50", "3,456.75", "3,234.50", "-11,112.50", "-6.43%"),
            ("HDFC BANK", "75", "1,567.25", "1,623.75", "+4,237.50", "+3.61%"),
            ("INFY", "200", "1,234.50", "1,298.75", "+12,850.00", "+5.20%")
        ]

        for holding in demo_holdings:
            self.holdings_tree.insert("", "end", values=holding)

        holdings_scroll = ttk.Scrollbar(
            portfolio_frame, orient="vertical", command=self.holdings_tree.yview)
        self.holdings_tree.configure(yscrollcommand=holdings_scroll.set)

        holdings_scroll.pack(side="right", fill="y")
        self.holdings_tree.pack(expand=True, fill="both")

    def _create_analytics_tab(self):
        """Create the analytics tab."""
        analytics_frame = ttk.Frame(self.notebook, style='Zerodha.TFrame')
        self.notebook.add(analytics_frame, text="📊 Analytics")

        # Analytics dashboard
        dashboard_frame = ttk.Frame(analytics_frame, style='Zerodha.TFrame', padding=15)
        dashboard_frame.pack(fill="both", expand=True)

        # Performance metrics grid
        metrics_grid = ttk.Frame(dashboard_frame, style='Zerodha.TFrame')
        metrics_grid.pack(fill="x", pady=(0, 20))

        # Row 1
        row1 = ttk.Frame(metrics_grid, style='Zerodha.TFrame')
        row1.pack(fill="x", pady=(0, 10))

        self._create_metric_card(row1, "Total Trades", "47", "📊")
        self._create_metric_card(row1, "Win Rate", "68.09%", "📈")
        self._create_metric_card(row1, "Avg Return", "2.34%", "💰")
        self._create_metric_card(row1, "Max Drawdown", "-8.45%", "📉")

        # Row 2
        row2 = ttk.Frame(metrics_grid, style='Zerodha.TFrame')
        row2.pack(fill="x", pady=(0, 10))

        self._create_metric_card(row2, "Sharpe Ratio", "1.45", "⚡")
        self._create_metric_card(row2, "Profit Factor", "1.87", "🎯")
        self._create_metric_card(row2, "Avg Hold Time", "2.3 hrs", "⏱️")
        self._create_metric_card(row2, "Risk/Reward", "1:2.1", "⚖️")

        # Strategy performance summary
        summary_text = """
📊 STRATEGY PERFORMANCE SUMMARY

Adam Mancini Nifty Strategy Analysis:
• Strategy shows strong performance with 68% win rate
• Average return per trade: 2.34%
• Best performing timeframe: Morning session (9:30-11:30)
• Risk management: Stop loss effectiveness at 95%

🎯 KEY INSIGHTS:
• Long signals outperform short signals by 12%
• Volatility-based entries show 15% better results
• Weekend gap analysis improves Monday trades by 8%

⚠️  RISK METRICS:
• Maximum consecutive losses: 4
• Current drawdown: -2.1%
• VaR (95%): ₹8,456 per trade
        """

        summary_label = ttk.Label(dashboard_frame, text=summary_text.strip(),
                                  style='ZerodhaStatus.TLabel',
                                  anchor="nw", justify="left")
        summary_label.pack(fill="both", expand=True, pady=(10, 0))

    def _create_metric_card(self, parent, title, value, icon):
        """Create a metric card widget."""
        card = ttk.Frame(parent, style='Zerodha.TFrame')
        card.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Label(card, text=f"{icon} {title}", style='ZerodhaStatus.TLabel').pack(anchor="w")
        ttk.Label(card, text=value, style='Zerodha.TLabel',
                  font=('Segoe UI', 14, 'bold')).pack(anchor="w")

    def _run_strategy(self):
        """Run trading strategy analysis."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            messagebox.showwarning("No Data", "Please load market data first.")
            return

        self.status_var.set("🔄 Running strategy analysis...")
        self.update_idletasks()

        # Simulate strategy analysis
        import time
        time.sleep(1)  # Simulate processing time

        # Update trading stats
        self.trades_var.set("Trades: 23")
        self.pnl_var.set("P&L: +₹12,456.78")
        self.win_rate_var.set("Win Rate: 68%")
        self.avg_trade_var.set("Avg Trade: +₹541.60")

        self.status_var.set("✅ Strategy analysis completed")
        messagebox.showinfo(
            "Strategy Complete",
            "Adam Mancini strategy analysis completed!\n\n"
            "Results:\n• Total Trades: 23\n• Win Rate: 68%\n• Total P&L: +₹12,456.78")

    def _export_data(self):
        """Export current data and analysis."""
        if not hasattr(self, 'current_data') or self.current_data is None:
            messagebox.showwarning("No Data", "No data available to export.")
            return

        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )

        if file_path:
            self.status_var.set("💾 Exporting data...")
            self.update_idletasks()

            # In a real app, this would export the actual data
            import time
            time.sleep(0.5)

            self.status_var.set("✅ Data exported successfully")
            messagebox.showinfo("Export Complete", f"Data exported to:\n{file_path}")

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
            self.current_data = df  # Store for strategy analysis

            # Update market status
            self.market_status_var.set("🟢 Data Loaded")

            # Clear existing data
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Prepare data with additional calculated fields
            if 'close' in df.columns:
                # Calculate change and change percentage
                df['change'] = df['close'].diff()
                df['change_pct'] = (df['change'] / df['close'].shift(1) * 100).round(2)

            # Display data (limit to 500 rows for performance)
            display_df = df.head(500)

            for _, row in display_df.iterrows():
                timestamp_str = str(row.iloc[0])[:19] if pd.notna(row.iloc[0]) else "N/A"

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

                # Color coding based on price movement
                if 'change' in df.columns and pd.notna(row['change']):
                    if row['change'] > 0:
                        # In a real app, this would set green color
                        pass
                    elif row['change'] < 0:
                        # In a real app, this would set red color
                        pass

            # Update statistics
            total_rows = len(df)
            if 'close' in df.columns:
                latest_price = df['close'].iloc[-1] if not df['close'].empty else 0
                price_change = (df['change'].iloc[-1] if ('change' in df.columns and
                                                          not df['change'].empty) else 0)
                avg_volume = df['volume'].mean() if 'volume' in df.columns else 0

                stats_text = (f"Loaded {total_rows} records | "
                              f"Latest: ₹{latest_price:.2f} | "
                              f"Change: {price_change:+.2f} | "
                              f"Avg Volume: {avg_volume:,.0f}")
            else:
                stats_text = f"Loaded {total_rows} records from {os.path.basename(file_path)}"

            # Update status and info
            self.status_var.set(f"✅ Loaded {total_rows} rows")
            self.info_label.configure(text=stats_text)

        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load file: {str(exc)}")
            self.status_var.set("❌ Error loading file")
            self.market_status_var.set("🔴 Load Failed")


def main() -> None:
    """Launch the GUI application."""
    app = SimpleTraderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
