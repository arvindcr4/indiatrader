#!/bin/bash
# Build script for macOS and Linux

set -e

echo "Building IndiaTrader applications..."

# Create build_configs directory if it doesn't exist
mkdir -p build_configs
mkdir -p dist

# Activate virtual environment
source venv/bin/activate

# Install build dependencies if not already installed
pip install pyinstaller

echo "Building Simple GUI application..."
cd build_configs
pyinstaller simple_gui.spec --clean --noconfirm
cd ..

echo "Building Full Trading application..."
cd build_configs
pyinstaller full_app.spec --clean --noconfirm
cd ..

echo "Build completed!"
echo "Applications are available in the dist/ directory:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "- IndiaTrader DataViewer.app"
    echo "- IndiaTrader.app"
else
    echo "- IndiaTrader-DataViewer"
    echo "- IndiaTrader-Full"
fi

echo ""
echo "To run the applications:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  open 'dist/IndiaTrader DataViewer.app'"
    echo "  open 'dist/IndiaTrader.app'"
else
    echo "  ./dist/IndiaTrader-DataViewer"
    echo "  ./dist/IndiaTrader-Full"
fi