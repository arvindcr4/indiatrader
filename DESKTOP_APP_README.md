# IndiaTrader Desktop Application

This document provides instructions for running IndiaTrader as a desktop application.

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Required Python packages (see below)

## Installation

1. Clone the repository or ensure you have the code locally
2. Set up a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install pytorch_lightning  # Additional dependency
   ```

## Running the Application

There are two ways to run the desktop application:

### Option 1: Simplified GUI (Recommended for first-time users)

This option provides a basic data viewer that allows you to load and view CSV files:

```bash
python3 simple_gui.py
```

### Option 2: Full Trading Application (Requires complete setup)

This option provides the full trading application with all features:

```bash
python3 run_desktop_app.py
```

Note: The full application requires all dependencies to be properly installed and configured, including data sources and API credentials.

## Usage

1. Launch the application using one of the options above
2. For the simplified GUI:
   - Click "Open CSV File" to select and load a data file
   - Browse to a CSV file containing market data (e.g., from the `data/` directory)
   - The data will be displayed in the table

3. For the full trading application:
   - Click "Open CSV" to load a data file
   - The application will analyze the data using the Adam Mancini strategy
   - Results will be displayed in the table

## Troubleshooting

If you encounter issues:

1. Ensure all required packages are installed
2. Check that your Python version is 3.10 or higher
3. Verify that the data file format is correct (CSV with appropriate columns)

## Data Files

Sample data files can be found in the `data/` directory:
- `nifty_5min_7days.csv`
- `nifty_5min_30days.csv`
- Other data files as available