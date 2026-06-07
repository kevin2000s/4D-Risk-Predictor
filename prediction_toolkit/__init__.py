"""
SVD(128) + ExtraTrees 4D Transmission Risk Prediction Toolkit

Usage:
    from prediction_toolkit import TransmissionRiskPredictor, SNPDataLoader, EnvDataLoader

    predictor = TransmissionRiskPredictor()
    # ... load data and predict
"""

from .model import TransmissionRiskPredictor
from .data_loader import SNPDataLoader, EnvDataLoader, align_samples

__version__ = '1.0.0'
__all__ = ['TransmissionRiskPredictor', 'SNPDataLoader', 'EnvDataLoader', 'align_samples']
