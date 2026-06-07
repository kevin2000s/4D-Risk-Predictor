#!/usr/bin/env python3
"""
4D Transmission Risk Predictor - GUI Application
Main entry point for PyInstaller packaging.

Double-click 4D_Risk_Predictor.exe to launch.
"""
import sys
import os

# When frozen with PyInstaller, sys.executable points to the .exe file.
# We need to ensure the prediction_toolkit package is importable.
exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, exe_dir)

from prediction_toolkit.gui.app import main

if __name__ == '__main__':
    main()
