#!/usr/bin/env python3
"""
Cross-platform build script for IndiaTrader applications.
Supports Windows, macOS, and Linux builds.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {cmd}: {e}")
        return False

def setup_environment():
    """Set up the build environment."""
    print("Setting up build environment...")
    
    # Create necessary directories
    os.makedirs("build_configs", exist_ok=True)
    os.makedirs("dist", exist_ok=True)
    
    # Install PyInstaller if not present
    if not run_command("pip show pyinstaller"):
        print("Installing PyInstaller...")
        if not run_command("pip install pyinstaller"):
            print("Failed to install PyInstaller")
            return False
    
    return True

def build_application(spec_file, app_name):
    """Build a single application using PyInstaller."""
    print(f"Building {app_name}...")
    
    spec_path = Path("build_configs") / spec_file
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}")
        return False
    
    cmd = f"pyinstaller {spec_path} --clean --noconfirm"
    if not run_command(cmd, cwd="build_configs"):
        print(f"Failed to build {app_name}")
        return False
    
    print(f"Successfully built {app_name}")
    return True

def create_distribution_package():
    """Create a distribution package with all necessary files."""
    print("Creating distribution package...")
    
    dist_dir = Path("dist")
    package_dir = dist_dir / "IndiaTrader_Package"
    
    # Clean and create package directory
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    
    # Copy applications
    system = platform.system().lower()
    if system == "darwin":  # macOS
        apps = ["IndiaTrader DataViewer.app", "IndiaTrader.app"]
        for app in apps:
            app_path = dist_dir / app
            if app_path.exists():
                shutil.copytree(app_path, package_dir / app)
    elif system == "windows":
        apps = ["IndiaTrader-DataViewer.exe", "IndiaTrader-Full.exe"]
        for app in apps:
            app_path = dist_dir / app
            if app_path.exists():
                shutil.copy2(app_path, package_dir / app)
    else:  # Linux
        apps = ["IndiaTrader-DataViewer", "IndiaTrader-Full"]
        for app in apps:
            app_path = dist_dir / app
            if app_path.exists():
                shutil.copy2(app_path, package_dir / app)
                # Make executable
                os.chmod(package_dir / app, 0o755)
    
    # Copy documentation and sample data
    files_to_copy = [
        "DESKTOP_APP_README.md",
        "README.md",
        "requirements.txt"
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, package_dir / file)
    
    # Copy sample data
    data_dir = Path("data")
    if data_dir.exists():
        shutil.copytree(data_dir, package_dir / "sample_data")
    
    # Create a startup script
    if system == "windows":
        startup_script = package_dir / "start_indiatrader.bat"
        with open(startup_script, "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting IndiaTrader...\n")
            f.write("echo Choose an application:\n")
            f.write("echo 1. Data Viewer (Simple)\n")
            f.write("echo 2. Full Trading Platform\n")
            f.write("set /p choice=Enter your choice (1 or 2): \n")
            f.write("if %choice%==1 start IndiaTrader-DataViewer.exe\n")
            f.write("if %choice%==2 start IndiaTrader-Full.exe\n")
            f.write("pause\n")
    else:
        startup_script = package_dir / "start_indiatrader.sh"
        with open(startup_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Starting IndiaTrader...'\n")
            f.write("echo 'Choose an application:'\n")
            f.write("echo '1. Data Viewer (Simple)'\n")
            f.write("echo '2. Full Trading Platform'\n")
            f.write("read -p 'Enter your choice (1 or 2): ' choice\n")
            
            if system == "darwin":
                f.write("if [ $choice -eq 1 ]; then\n")
                f.write("    open 'IndiaTrader DataViewer.app'\n")
                f.write("elif [ $choice -eq 2 ]; then\n")
                f.write("    open 'IndiaTrader.app'\n")
                f.write("fi\n")
            else:
                f.write("if [ $choice -eq 1 ]; then\n")
                f.write("    ./IndiaTrader-DataViewer\n")
                f.write("elif [ $choice -eq 2 ]; then\n")
                f.write("    ./IndiaTrader-Full\n")
                f.write("fi\n")
        
        os.chmod(startup_script, 0o755)
    
    print(f"Distribution package created at: {package_dir}")
    return True

def main():
    """Main build function."""
    print("=" * 60)
    print("IndiaTrader Cross-Platform Build Script")
    print("=" * 60)
    print(f"Building for: {platform.system()} {platform.machine()}")
    print()
    
    # Setup environment
    if not setup_environment():
        print("Failed to setup environment")
        return 1
    
    # Build applications
    success = True
    
    # Build simple GUI application
    if not build_application("simple_gui.spec", "Data Viewer"):
        success = False
    
    # Build full application
    if not build_application("full_app.spec", "Full Trading Platform"):
        success = False
    
    if not success:
        print("Some builds failed. Check the output above.")
        return 1
    
    # Create distribution package
    if not create_distribution_package():
        print("Failed to create distribution package")
        return 1
    
    print()
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)
    
    # Show what was created
    dist_dir = Path("dist")
    print(f"Applications built in: {dist_dir.absolute()}")
    print(f"Distribution package: {dist_dir / 'IndiaTrader_Package'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())