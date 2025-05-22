# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Add the project root to Python path
project_root = os.path.abspath('..')
sys.path.insert(0, project_root)

a = Analysis(
    [os.path.join(project_root, 'simple_gui.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'data'), 'data'),
        (os.path.join(project_root, 'DESKTOP_APP_README.md'), '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'pandas',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'pytorch_lightning',
        'gymnasium',
        'stable_baselines3',
        'matplotlib',
        'seaborn',
        'plotly',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='IndiaTrader-DataViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='IndiaTrader DataViewer.app',
        icon=None,
        bundle_identifier='com.indiatrader.dataviewer',
        info_plist={
            'CFBundleName': 'IndiaTrader DataViewer',
            'CFBundleDisplayName': 'IndiaTrader DataViewer',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'LSBackgroundOnly': False,
        },
    )