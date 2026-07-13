import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.18: v0.15 core with the proven 15-round tree fit."""

    _ALPHA = 8.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _RIDGE_WEIGHT = 0.72
    _XGB_WEIGHT = 0.24
    _RANK_WEIGHT = 0.04
    _PRIORITY = (
        "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
        "Feature.4__raw", "Feature.5__raw", "Feature.6__raw",
        "Feature.1__cs_rank", "Feature.2__cs_rank", "Feature.3__cs_rank",
        "Feature.1__ma5", "Feature.1__ma20", "Feature.1__ma60",
        "Feature.1__sd20", "Feature.1__ewma5",
        "Feature.1__ma5_rank", "Feature.1__ma20_rank", "Feature.1__ma60_rank",
        "Feature.1__diff_1", "Feature.2__diff_1", "Feature.3__diff_1",
        "Feature.1__roll_z", "Feature.1__mom_spread",
        "Feature.1__cs_demean", "interaction__rank_spread",
    )
    _XGB = {
        "objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "min_child_weight": 200, "reg_alpha": 0.02, "reg_lambda": 1.5,
        "tree_method": "hist", "verbosity": 0, "nthread": 2, "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows = 80000
        self.max_features = 35
        self.n_xgb_rounds = 15
        self.raw_coef_ = self.rank_coef_ = None
        self.raw_intercept_ = self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None
        self.selected_features_ = None
        self.impute_ = self.low_ = self.high_ = None
        self.mean_ = self.scale_ = None
        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.feature_count_ = self.training_rows_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    def _levels(self, columns):
        names = [str(v).lower