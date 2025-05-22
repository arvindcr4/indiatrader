# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

a = Analysis(
    [os.path.join(project_root, 'run_desktop_app.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'data'), 'data'),
        (os.path.join(project_root, 'config'), 'config'),
        (os.path.join(project_root, 'DESKTOP_APP_README.md'), '.'),
        (os.path.join(project_root, 'requirements.txt'), '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'pandas',
        'numpy',
        'torch',
        'pytorch_lightning',
        'gymnasium',
        'stable_baselines3',
        'indiatrader.gui.app',
        'indiatrader.strategies.adam_mancini',
        'indiatrader.strategies.mancini_trader',
        'indiatrader.brokers.dhan',
        'indiatrader.brokers.icici',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        'scipy.special.cython_special',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='IndiaTrader-Full',
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
        name='IndiaTrader.app',
        icon=None,
        bundle_identifier='com.indiatrader.full',
        info_plist={
            'CFBundleName': 'IndiaTrader',
            'CFBundleDisplayName': 'IndiaTrader Trading Platform',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'LSBackgroundOnly': False,
        },
    )