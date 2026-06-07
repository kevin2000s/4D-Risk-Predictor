# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('svd128_extratrees_models.joblib', '.'), ('svd128_svd_transformer.joblib', '.'), ('svd128_env_scaler.joblib', '.'), ('svd128_model_metadata.joblib', '.'), ('svd128_extratrees_feature_importance.csv', '.'), ('prediction_toolkit', 'prediction_toolkit')],
    hiddenimports=['sklearn', 'sklearn.ensemble', 'sklearn.decomposition', 'sklearn.preprocessing', 'pandas', 'numpy', 'scipy', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'ttkbootstrap', 'ttkbootstrap.dialogs', 'ttkbootstrap.scrolled', 'PIL'],
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
    [],
    exclude_binaries=True,
    name='4D_Risk_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='4D_Risk_Predictor',
)
