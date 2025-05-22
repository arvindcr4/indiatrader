@echo off
REM Build script for Windows

echo Building IndiaTrader applications...

REM Create build_configs directory if it doesn't exist
if not exist build_configs mkdir build_configs
if not exist dist mkdir dist

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install build dependencies if not already installed
pip install pyinstaller

echo Building Simple GUI application...
cd build_configs
pyinstaller simple_gui.spec --clean --noconfirm
cd ..

echo Building Full Trading application...
cd build_configs
pyinstaller full_app.spec --clean --noconfirm
cd ..

echo Build completed!
echo Applications are available in the dist\ directory:
echo - IndiaTrader-DataViewer.exe
echo - IndiaTrader-Full.exe

echo.
echo To run the applications:
echo   dist\IndiaTrader-DataViewer.exe
echo   dist\IndiaTrader-Full.exe

pause