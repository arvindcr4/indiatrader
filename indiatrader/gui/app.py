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
        self.geometry("700x500")
        self.strategy = AdamManciniNiftyStrategy(open_range_minutes=open_range_minutes)
        self._create_widgets()

    def _create_widgets(self) -> None:
        open_btn = tk.Button(self, text="Open CSV", command=self._load_file)
        open_btn.pack(pady=10)

        columns = ("timestamp", "close", "long", "short")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.pack(expand=True, fill="both")

    def _load_file(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path, parse_dates=[0])
            df.set_index(df.columns[0], inplace=True)
            out = self.strategy.generate_signals(df)
            self._display_results(out)
        except Exception as exc:  # pragma: no cover - UI error handling
            messagebox.showerror("Error", str(exc))

    def _display_results(self, df: pd.DataFrame) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        tail_df = df.tail(200)
        for ts, row in tail_df.iterrows():
            self.tree.insert("", "end", values=(ts, row["close"], row["long_signal"], row["short_signal"]))


def main(open_range_minutes: int = 15) -> None:
    """Launch the GUI application."""
    app = StrategyApp(open_range_minutes=open_range_minutes)
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
