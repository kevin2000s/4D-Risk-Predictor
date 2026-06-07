@echo off
chcp 65001 >nul
title Build 4D Risk Predictor
echo ============================================
echo  4D Risk Predictor - Build Executable
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ first.
    echo          Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Python detected:
python --version
echo.

:: Create virtual environment
echo [2/5] Creating build environment...
if exist venv_build rmdir /s /q venv_build
python -m venv venv_build
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate and install deps
echo [3/5] Installing dependencies...
call venv_build\Scripts\activate.bat
pip install -q pyinstaller pandas numpy scikit-learn scipy matplotlib joblib ttkbootstrap Pillow
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Build
echo [4/5] Building executable with PyInstaller...
python -m PyInstaller 4D_Risk_Predictor.spec
if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

:: Cleanup
echo [5/5] Cleaning up...
if exist build rmdir /s /q build
if exist venv_build rmdir /s /q venv_build

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo Output: dist\4D_Risk_Predictor\4D_Risk_Predictor.exe
echo.
echo Double-click the .exe to launch the GUI.
echo.
pause
