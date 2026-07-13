import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.20: shape-safe v0.18 core plus cross-sectional residual ridge."""

    _ALPHA = 8.0
    _CS_ALPHA = 14.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _BASE_WEIGHT = 0.92
    _CS_WEIGHT = 0.08
    _RIDGE_WEIGHT = 0.72
    _XGB