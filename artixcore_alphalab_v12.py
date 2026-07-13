import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.12, leaderboard-hardened v0.8 core."""

    _ALPHA = 8.0
    _DECAY = 0.20
    _RIDGE_WEIGHT = 0.75
    _XGB_WEIGHT = 0.25
    _PRIORITY = (
        "Feature.1__raw