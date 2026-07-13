import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.15: temporal-CV guarded v0.8 core."""

    _BASE_ALPHA = 8.0
    _BASE_DECAY = 0.20
    _RIDGE_WEIGHT = 0.75
    _XGB_WEIGHT = 0.25
    _RANK_ALPHA = 20.0
    _MIN_GAIN = 0.015

    _CANDIDATES = (
        (8.0, 0.20, 0.00),
        (8.0, 0.16, 0.04),