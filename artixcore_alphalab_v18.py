import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.18: reliable v0.15 refinement with 15 tree rounds."""

    _ALPHA = 8.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _RIDGE_WEIGHT = 0.72
    _XGB_WEIGHT = 0.24
    _RANK_WEIGHT = 0.