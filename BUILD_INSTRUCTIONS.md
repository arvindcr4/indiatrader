# IndiaTrader Build Instructions

This document provides instructions for building IndiaTrader applications for Windows, macOS, and Linux.

## Prerequisites

- Python 3.10 or higher
- Virtual environment set up with all dependencies
- PyInstaller (automatically installed by build scripts)

## Quick Build

### Option 1: Cross-Platform Build Script (Recommended)

```bash
python3 build_cross_platform.py
```

This script will:
- Set up the build environment
- Build both applications (Simple GUI and Full Trading Platform)
- Create a distribution package with all necessary files
- Generate platform-specific startup scripts

### Option 2: Platform-Specific Scripts

#### macOS/Linux:
```bash
./build.sh
```

#### Windows:
```batch
build.bat
```

## Manual Build Process

If you prefer to build manually:

### 1. Install Build Dependencies

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pyinstaller
```

### 2. Build Applications

#### Simple GUI Application:
```bash
cd build_configs
pyinstaller simple_gui.spec --clean --noconfirm
```

#### Full Trading Application:
```bash
cd build_configs
pyinstaller full_app.spec --clean --noconfirm
```

## Build Outputs

After building, you'll find the applications in the `dist/` directory:

### macOS:
- `IndiaTrader DataViewer.app` - Simple data viewer application
- `IndiaTrader.app` - Full trading platform

### Windows:
- `IndiaTrader-DataViewer.exe` - Simple data viewer application
- `IndiaTrader-Full.exe` - Full trading platform

### Linux:
- `IndiaTrader-DataViewer` - Simple data viewer application
- `IndiaTrader-Full` - Full trading platform

## Distribution Package

The cross-platform build script creates a complete distribution package at `dist/IndiaTrader_Package/` containing:

- Built applications
- Sample data files
- Documentation
- Startup scripts for easy launching

## Build Configuration

### PyInstaller Spec Files

The build process uses two spec files:

1. `build_configs/simple_gui.spec` - For the lightweight data viewer
2. `build_configs/full_app.spec` - For the complete trading platform

These files can be customized to:
- Add/remove dependencies
- Include additional data files
- Modify application metadata
- Change build options

### Key Features

- **One-file executables**: Each application is built as a single executable file
- **No external dependencies**: All required libraries are bundled
- **Cross-platform compatibility**: Same source builds for all platforms
- **Optimized size**: Excludes unnecessary dependencies where possible

## Troubleshooting

### Common Issues

1. **Missing modules**: Add any missing imports to the `hiddenimports` list in the spec files
2. **Large file sizes**: Use the `excludes` list to remove unnecessary dependencies
3. **Runtime errors**: Check the console output for missing data files or dependencies

### Build Optimization

To reduce application size:
- Remove unused dependencies from `hiddenimports`
- Add unnecessary modules to `excludes`
- Use UPX compression (enabled by default)

### Platform-Specific Notes

#### macOS:
- Applications are built as `.app` bundles
- Code signing can be added for distribution
- Notarization required for Gatekeeper approval

#### Windows:
- Applications are built as `.exe` files
- Can be signed for Windows SmartScreen approval
- Consider creating an installer for distribution

#### Linux:
- Applications are built as executable binaries
- May require additional runtime libraries on some distributions
- Consider creating AppImage or Flatpak packages

## Distribution

For distributing the applications:

1. **macOS**: Create a DMG file or use the App Store
2. **Windows**: Create an MSI installer or use the Microsoft Store
3. **Linux**: Create DEB/RPM packages or use AppImage/Flatpak

## Performance Notes

- First launch may be slower due to unpacking
- Subsequent launches are faster
- Full application is larger due to ML dependencies
- Simple GUI is lightweight and fast

## Security Considerations

- Applications are self-contained and don't require internet access for core functionality
- Trading features require API credentials (not included in build)
- Consider code signing for production distribution