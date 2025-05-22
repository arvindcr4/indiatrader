"""Simple desktop application for running trading strategies."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from indiatrader.strategies import AdamManciniNiftyStrategy


class StrategyApp(tk.Tk):
    """Tkinter-based GUI for the Adam Mancini Nifty strategy."""

    def __init__(self, open_range_minutes: int = 15) -> None:
        super().__init__()
        self.title("IndiaTrader Strategy")
        self.geometry("900x700")
        self.strategy = AdamManciniNiftyStrategy(open_range_minutes=open_range_minutes)
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
                        font=('Segoe UI', 16, 'bold'))

    def _create_widgets(self) -> None:
        # Main container
        main_frame = ttk.Frame(self, style='Zerodha.TFrame', padding=20)
        main_frame.pack(expand=True, fill='both')

        # Header section
        header_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))

        # Title
        title_label = ttk.Label(header_frame, text="IndiaTrader Strategy",
                                style='ZerodhaTitle.TLabel')
        title_label.pack(side='left')

        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="Adam Mancini Nifty Strategy",
                                   style='Zerodha.TLabel')
        subtitle_label.pack(side='left', padx=(10, 0))

        # Controls section
        controls_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        controls_frame.pack(fill='x', pady=(0, 20))

        # Open CSV button
        open_btn = ttk.Button(controls_frame, text="📁 Load CSV Data",
                              command=self._load_file, style='Zerodha.TButton')
        open_btn.pack(side='left')

        # Status label
        self.status_label = ttk.Label(controls_frame, text="Ready to load data",
                                      style='Zerodha.TLabel')
        self.status_label.pack(side='left', padx=(20, 0))

        # Data display section
        data_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        data_frame.pack(expand=True, fill='both')

        # Data table
        columns = ("timestamp", "close", "long_signal", "short_signal")
        self.tree = ttk.Treeview(data_frame, columns=columns, show="headings",
                                 style='Zerodha.Treeview')

        # Configure columns
        column_config = {
            "timestamp": {"width": 180, "anchor": "w"},
            "close": {"width": 100, "anchor": "e"},
            "long_signal": {"width": 120, "anchor": "center"},
            "short_signal": {"width": 120, "anchor": "center"}
        }

        for col in columns:
            self.tree.heading(col, text=col.replace('_', ' ').title())
            self.tree.column(col, **column_config[col])

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(data_frame, orient='vertical', command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(data_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and treeview
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        self.tree.pack(expand=True, fill='both')

        # Summary section
        summary_frame = ttk.Frame(main_frame, style='Zerodha.TFrame')
        summary_frame.pack(fill='x', pady=(20, 0))

        self.summary_label = ttk.Label(summary_frame, text="No data loaded",
                                       style='Zerodha.TLabel')
        self.summary_label.pack()

    def _load_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.status_label.configure(text="Loading data...")
            self.update_idletasks()

            df = pd.read_csv(file_path, parse_dates=[0])
            df.set_index(df.columns[0], inplace=True)
            out = self.strategy.generate_signals(df)
            self._display_results(out)

            self.status_label.configure(text=f"✓ Loaded {len(out)} data points")
        except Exception as exc:  # pragma: no cover - UI error handling
            self.status_label.configure(text="✗ Error loading file")
            messagebox.showerror("Error", str(exc))

    def _display_results(self, df: pd.DataFrame) -> None:
        # Clear existing data
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Display last 200 rows for performance
        tail_df = df.tail(200)
        long_signals = (tail_df['long_signal'] == 1).sum()
        short_signals = (tail_df['short_signal'] == 1).sum()

        for ts, row in tail_df.iterrows():
            # Format timestamp
            timestamp_str = str(ts)[:19] if pd.notna(ts) else "N/A"

            # Format close price
            close_str = f"{row['close']:.2f}" if pd.notna(row['close']) else "N/A"

            # Format signals with indicators
            long_str = "🟢 LONG" if row['long_signal'] == 1 else "-"
            short_str = "🔴 SHORT" if row['short_signal'] == 1 else "-"

            values = (timestamp_str, close_str, long_str, short_str)

            # Add color coding based on signals
            item = self.tree.insert("", "end", values=values)
            if row['long_signal'] == 1:
                self.tree.set(item, 'long_signal', '🟢 LONG')
            elif row['short_signal'] == 1:
                self.tree.set(item, 'short_signal', '🔴 SHORT')

        # Update summary
        summary_text = (f"Displaying last 200 of {len(df)} records | "
                        f"Long Signals: {long_signals} | Short Signals: {short_signals}")
        self.summary_label.configure(text=summary_text)


def main(open_range_minutes: int = 15) -> None:
    """Launch the GUI application."""
    app = StrategyApp(open_range_minutes=open_range_minutes)
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
