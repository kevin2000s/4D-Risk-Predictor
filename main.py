#!/usr/bin/env python3
# Copyright (c) 2025 Peking University People's Hospital Hui Lab
# SPDX-License-Identifier: MIT
"""GUI entry point for PyInstaller packaging."""
import sys
import os

exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, exe_dir)

from prediction_toolkit.gui.app import main

if __name__ == '__main__':
    main()
