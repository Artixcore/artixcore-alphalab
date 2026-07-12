import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):

    # Artixcore AlphaLab v0.2
    # Fast leakage-safe cross-sectional forecaster.
    # Fixed Ridge + light XGBoost blend with recency-weighted training.

    _ALPHA = 8.0
    _DECAY = 0.20
    _RIDGE_WEIGHT = 0.75
    _XGB_WEIGHT = 0.25
    _FEATURE_PRIORITY = (
        "Feature.1__raw",
        "Feature.2__raw",
        "Feature.3__raw",
        "Feature.4__raw",
        "Feature.5__raw",
        "Feature.6__raw",
        "Feature.1__cs_rank",
        "Feature.2__cs_rank",
        "Feature.3__cs_rank",
        "Feature.1__ma5",
        "Feature.1__ma20",
        "Feature.1__ma60",
        "Feature.1__sd20",
        "Feature.1__ewma5",
        "Feature.1__ma5_rank",
        "Feature.1__ma20_rank",
        "Feature.1__ma60_rank",
        "Feature.1__diff_1",
        "Feature.2__diff_1",
        "Feature.3__diff_1",
        "Feature.1__roll_z",
        "Feature.1__mom_spread",
        "Feature.1__cs_demean",
        "interaction__rank_spread",
    )
    _XGB_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 200,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 2,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.alpha = self._ALPHA
        self.coef_ = None
        self.intercept_ = 0.0
        self.xgb_model_ = None

        self.x_mean_ = None
        self.x_scale_ = None
        self.impute_values_ = None
        self.clip_low_ = None
        self.clip_high_ = None
        self.selected_features_ = None

        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.max_train_rows = 80_000
        self.max_features = 35
        self.n_xgb_rounds = 15
        self.decay_fraction_ = self._DECAY

        self.training_error_ = None
        self.feature_count_ = 0
        self.training_rows_ = 0
        self.fallback_used_ = False

        self.feature_time_ = 0.0
        self.fit_time_ = 0.0
        self.predict_feature_time_ = 0.0
        self.predict_model_time_ = 0.0

    def _infer_column_levels(self, columns):
        if not isinstance(columns, pd.MultiIndex):
            return None, None

        names = [str(name).lower() if name is not None else "" for name in columns.names]
        unique_counts = [
            len(pd.Index(columns.get_level_values(i)).unique())
            for i in range(columns.nlevels)
        ]

        feature_level = None
        asset_level = None
        for i, name in enumerate(names):
            if "feature" in name or "factor" in name:
                feature_level = i
            if "ticker" in name or "asset" in name or "symbol" in name:
                asset_level = i

        if feature_level is None:
            feature_level = int(np.argmin(unique_counts))

        if asset_level is None:
            remaining = [i for i in range(columns.nlevels) if i != feature_level]
            asset_level = (
                max(remaining, key=lambda i: unique_counts[i])
                if remaining
                else feature_level
            )

        return feature_level, asset_level

    def _ordered_unique(self, values):
        seen = set()
        ordered = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _get_feature_names_and_tickers(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            feature_level, asset_level = self._infer_column_levels(features.columns)
            feature_names = self._ordered_unique(
                features.columns.get_level_values(feature_level)
            )
            tickers = self._ordered_unique(
                features.columns.get_level_values(asset_level)
            )
            return list(feature_names), list(tickers), feature_level, asset_level

        return ["feature.1"], list(features.columns), None, None

    def _safe_numeric_frame(self, frame):
        out = frame.apply(pd.to_numeric, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.astype(np.float32, copy=False)

    def _extract_feature_frames(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            feature_values, asset_values, feature_level, asset_level = (
                self._get_feature_names_and_tickers(features)
            )
            frames = {}
            for feature_name in feature_values:
                selected_columns = [
                    col for col in features.columns if col[feature_level] == feature_name
                ]
                if not selected_columns:
                    continue
                block = features.loc[:, selected_columns]
                block.columns = [col[asset_level] for col in selected_columns]
                block = block.loc[:, ~pd.Index(block.columns).duplicated()]
                block = block.reindex(columns=asset_values)
                frames[str(feature_name)] = self._safe_numeric_frame(block)
            return frames, list(asset_values)

        asset_values = list(features.columns)
        return {"feature.1": self._safe_numeric_frame(features)}, asset_values

    def _rolling_stat(self, frame, window, min_periods, stat):
        roller = frame.rolling(window=window, min_periods=min_periods)
        if stat == "mean":
            return roller.mean()
        if stat == "std":
            return roller.std(ddof=0)
        return roller.mean()

    def _lag_frame(self, frame, lag):
        return frame.shift(lag)

    def _ewma_frame(self, frame, span):
        return frame.ewm(span=span, adjust=False, min_periods=2).mean()

    def _cross_sectional_rank(self, frame, n_assets):
        rank = frame.rank(axis=1, method="average")
        return ((rank - 0.5 * (n_assets + 1)) / (0.5 * max(n_assets - 1, 1))).astype(
            np.float32, copy=False
        )

    def _cross_sectional_demean_median(self, frame):
        return frame.sub(frame.median(axis=1), axis=0).astype(np.float32, copy=False)

    def _as_engineered_block(self, name, frame):
        block = frame.replace([np.inf, -np.inf], np.nan)
        block.columns = pd.MultiIndex.from_product(
            [[name], block.columns], names=["feature", "asset"]
        )
        return block.astype(np.float32, copy=False)

    def _make_features(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        frames, assets = self._extract_feature_frames(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty

        feature_names = list(frames.keys())
        n_assets = len(assets)
        blocks = []

        for fi, fname in enumerate(feature_names):
            raw = frames[fname]
            blocks.append(self._as_engineered_block(f"{fname}__raw", raw))

            if fi < 3:
                cs_rank = self._cross_sectional_rank(raw, n_assets)
                blocks.append(self._as_engineered_block(f"{fname}__cs_rank", cs_rank))
                blocks.append(
                    self._as_engineered_block(
                        f"{fname}__diff_1", raw - self._lag_frame(raw, 1)
                    )
                )

            if fi == 0:
                roll_mean_5 = self._rolling_stat(raw, 5, 2, "mean")
                roll_mean_20 = self._rolling_stat(raw, 20, 3, "mean")
                roll_mean_60 = self._rolling_stat(raw, 60, 5, "mean")
                roll_std_20 = self._rolling_stat(raw, 20, 3, "std").fillna(0.0)
                ewma_5 = self._ewma_frame(raw, 5)
                blocks.extend(
                    [
                        self._as_engineered_block(f"{fname}__ma5", roll_mean_5),
                        self._as_engineered_block(f"{fname}__ma20", roll_mean_20),
                        self._as_engineered_block(f"{fname}__ma60", roll_mean_60),
                        self._as_engineered_block(f"{fname}__sd20", roll_std_20),
                        self._as_engineered_block(f"{fname}__ewma5", ewma_5),
                        self._as_engineered_block(
                            f"{fname}__ma5_rank",
                            self._cross_sectional_rank(roll_mean_5, n_assets),
                        ),
                        self._as_engineered_block(
                            f"{fname}__ma20_rank",
                            self._cross_sectional_rank(roll_mean_20, n_assets),
                        ),
                        self._as_engineered_block(
                            f"{fname}__ma60_rank",
                            self._cross_sectional_rank(roll_mean_60, n_assets),
                        ),
                        self._as_engineered_block(
                            f"{fname}__roll_z",
                            (raw - roll_mean_5) / (roll_std_20 + 1.0e-6),
                        ),
                        self._as_engineered_block(
                            f"{fname}__mom_spread", roll_mean_5 - roll_mean_60
                        ),
                        self._as_engineered_block(
                            f"{fname}__cs_demean", self._cross_sectional_demean_median(raw)
                        ),
                    ]
                )

        if len(feature_names) >= 2:
            f1 = frames[feature_names[0]]
            f2 = frames[feature_names[1]]
            rank_spread = self._cross_sectional_rank(f1, n_assets) - self._cross_sectional_rank(
                f2, n_assets
            )
            blocks.append(
                self._as_engineered_block("interaction__rank_spread", rank_spread)
            )

        engineered = pd.concat(blocks, axis=1)
        feat_names = engineered.columns.get_level_values("feature").unique()
        engineered = engineered.reindex(
            columns=pd.MultiIndex.from_product([feat_names, assets], names=["feature", "asset"])
        )
        return engineered.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)

    def _prepare_target_frame(self, target, index, assets):
        if isinstance(target, pd.Series):
            target_frame = (
                target.unstack(level=-1)
                if isinstance(target.index, pd.MultiIndex)
                else target.to_frame()
            )
        elif isinstance(target, pd.DataFrame):
            target_frame = target.copy()
        else:
            target_frame = pd.DataFrame(target, index=index)

        if isinstance(target_frame.columns, pd.MultiIndex):
            _, asset_level = self._infer_column_levels(target_frame.columns)
            target_frame.columns = target_frame.columns.get_level_values(asset_level)

        return self._safe_numeric_frame(target_frame.reindex(index=index, columns=assets))

    def _panel_to_matrix(self, engineered_features, target=None):
        assets = list(engineered_features.columns.get_level_values("asset").unique())
        x_long = engineered_features.stack(level="asset", future_stack=True)
        x_long = x_long.replace([np.inf, -np.inf], np.nan)

        if target is None:
            return x_long, None, assets

        target_frame = self._prepare_target_frame(target, engineered_features.index, assets)
        y_long = target_frame.stack(future_stack=True)
        x_long, y_long = x_long.align(y_long, join="inner", axis=0)

        mask = y_long.replace([np.inf, -np.inf], np.nan).notna()
        x_long = x_long.loc[mask]
        y_long = y_long.loc[mask].astype(np.float32, copy=False)
        return x_long, y_long, assets

    def _filter_features_fast(self, x_train):
        if x_train.empty:
            return x_train

        columns = list(x_train.columns)
        x_values = x_train.to_numpy(dtype=np.float32, copy=False)
        keep = []

        for j, col in enumerate(columns):
            col_values = x_values[:, j]
            finite = np.isfinite(col_values)
            if finite.mean() < 0.05:
                continue
            valid = col_values[finite]
            if valid.size == 0 or np.nanstd(valid) < 1.0e-8:
                continue
            keep.append(col)

        if not keep:
            return x_train.iloc[:, :0]

        priority = {name: i for i, name in enumerate(self._FEATURE_PRIORITY)}
        keep.sort(key=lambda col: priority.get(col, len(priority)))
        selected = keep[: self.max_features]
        return x_train.loc[:, selected]

    def _fit_preprocessor(self, x_train):
        x = x_train.to_numpy(dtype=np.float32, copy=True)
        x = np.where(np.isfinite(x), x, np.nan)

        self.impute_values_ = np.nanmedian(x, axis=0).astype(np.float32)
        self.impute_values_ = np.where(
            np.isfinite(self.impute_values_), self.impute_values_, 0.0
        ).astype(np.float32)

        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]

        self.clip_low_ = np.nanquantile(x, 0.005, axis=0).astype(np.float32)
        self.clip_high_ = np.nanquantile(x, 0.995, axis=0).astype(np.float32)
        invalid = ~np.isfinite(self.clip_low_) | ~np.isfinite(self.clip_high_) | (
            self.clip_low_ >= self.clip_high_
        )
        self.clip_low_[invalid] = -10.0
        self.clip_high_[invalid] = 10.0
        x = np.clip(x, self.clip_low_, self.clip_high_)

        self.x_mean_ = x.mean(axis=0).astype(np.float32)
        centered = x - self.x_mean_
        mad = np.nanmedian(np.abs(centered), axis=0).astype(np.float32)
        self.x_scale_ = (1.4826 * mad).astype(np.float32)
        fallback = x.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)
        self.x_scale_[bad] = fallback[bad]
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)] = 1.0

    def _transform_matrix(self, x_data):
        if self.impute_values_ is None:
            x = x_data.to_numpy(dtype=np.float32, copy=True)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = x_data.to_numpy(dtype=np.float32, copy=True)
        x = np.where(np.isfinite(x), x, np.nan)
        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]
        x = np.clip(x, self.clip_low_, self.clip_high_)
        x = (x - self.x_mean_) / self.x_scale_
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_imputation_frame(self, x_data):
        out = x_data.reindex(columns=self.selected_features_).copy()
        for j, col in enumerate(self.selected_features_):
            if col not in out.columns:
                out[col] = self.impute_values_[j]
            else:
                out[col] = out[col].fillna(self.impute_values_[j])
        return out[self.selected_features_]

    def _solve_ridge(self, x_scaled, y_centered, alpha, sample_weights=None):
        if sample_weights is None:
            gram = x_scaled.T @ x_scaled
            rhs = x_scaled.T @ y_centered
        else:
            sw = np.sqrt(sample_weights).astype(np.float32)
            xw = x_scaled * sw[:, None]
            yw = y_centered * sw
            gram = xw.T @ xw
            rhs = xw.T @ yw

        gram = gram.copy()
        gram.flat[:: gram.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ rhs

    def _fit_ridge_core(self, x_train, y_train, alpha, sample_weights=None):
        x_scaled = self._transform_matrix(x_train)
        y = np.nan_to_num(y_train.to_numpy(dtype=np.float32, copy=True), nan=0.0)
        y_mean = float(y.mean()) if y.size else 0.0
        coef = self._solve_ridge(x_scaled, y - y_mean, alpha, sample_weights)
        return coef.astype(np.float32), y_mean

    def _fit_xgb(self, x_train, y_train, sample_weights=None):
        x_scaled = self._transform_matrix(x_train)
        y = y_train.to_numpy(dtype=np.float32, copy=False)
        dtrain = xgb.DMatrix(x_scaled, label=y, weight=sample_weights)
        return xgb.train(self._XGB_PARAMS, dtrain, num_boost_round=self.n_xgb_rounds)

    def _predict_ridge(self, x_data):
        if self.coef_ is None:
            return np.zeros(len(x_data), dtype=np.float32)
        return (
            self.intercept_
            + self._transform_matrix(x_data) @ self.coef_
        ).astype(np.float32, copy=False)

    def _predict_xgb(self, x_data):
        if self.xgb_model_ is None:
            return np.zeros(len(x_data), dtype=np.float32)
        x_scaled = self._transform_matrix(x_data)
        return self.xgb_model_.predict(xgb.DMatrix(x_scaled)).astype(np.float32, copy=False)

    def _sample_training_rows(self, x_train, y_train):
        n_rows = len(x_train)
        if n_rows <= self.max_train_rows:
            return x_train, y_train, np.arange(n_rows, dtype=np.int64)

        recent_count = int(self.max_train_rows * 0.60)
        older_budget = self.max_train_rows - recent_count
        recent_start = n_rows - recent_count
        recent_positions = np.arange(recent_start, n_rows, dtype=np.int64)

        if older_budget <= 0 or recent_start <= 0:
            positions = recent_positions
        else:
            older_positions = np.linspace(0, recent_start - 1, older_budget, dtype=np.int64)
            positions = np.unique(np.concatenate([older_positions, recent_positions]))

        return x_train.iloc[positions], y_train.iloc[positions], positions

    def _compute_sample_weights(self, positions, total_rows):
        if total_rows <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total_rows - 1) - positions.astype(np.float32)
        halflife = max(1.0, total_rows * self.decay_fraction_)
        weights = np.exp(-ages / halflife)
        return (weights / np.mean(weights)).astype(np.float32)

    def _zero_prediction(self, features):
        _, assets = self._extract_feature_frames(features)
        pred = pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

    def _finalize_prediction(self, pred):
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clip_value = float(np.clip(self.prediction_clip_, 1.0e-6, 10.0))
        pred = pred.clip(lower=-clip_value, upper=clip_value)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred.astype(np.float32, copy=False)

    def _fallback_prediction(self, features):
        try:
            frames, assets = self._extract_feature_frames(features)
            if not frames:
                return self._zero_prediction(features)
            fname = next(iter(frames))
            primary = self._safe_numeric_frame(frames[fname])
            n_assets = len(assets)
            score = self._cross_sectional_rank(primary, n_assets)
            mom = self._rolling_stat(primary, 5, 2, "mean")
            score = score + 0.3 * self._cross_sectional_rank(mom, n_assets)
            pred = score.reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finalize_prediction(pred)
        except Exception:
            return self._zero_prediction(features)

    def train(self, features, target):
        self.training_error_ = None
        self.is_trained_ = False
        self.fallback_used_ = False
        self.xgb_model_ = None
        self.coef_ = None
        self.feature_time_ = 0.0
        self.fit_time_ = 0.0

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            t0 = time.perf_counter()
            engineered = self._make_features(features)
            self.feature_time_ = time.perf_counter() - t0

            x_all, y_all, assets = self._panel_to_matrix(engineered, target)
            del engineered

            if x_all.empty or len(y_all) < 40:
                self.selected_features_ = list(x_all.columns)
                self.feature_count_ = len(self.selected_features_ or [])
                return self

            filtered = self._filter_features_fast(x_all)
            if filtered.empty or filtered.shape[1] == 0:
                return self

            self.selected_features_ = list(filtered.columns)
            self._fit_preprocessor(filtered)

            x_all_f = self._apply_imputation_frame(x_all)
            x_sampled, y_sampled, positions = self._sample_training_rows(x_all_f, y_all)
            sample_weights = self._compute_sample_weights(positions, len(x_all_f))
            self.training_rows_ = len(x_sampled)

            abs_y = np.abs(y_sampled.to_numpy(dtype=np.float32, copy=False))
            q = np.nanquantile(abs_y, 0.995) if abs_y.size else 1.0
            self.prediction_clip_ = float(
                np.clip(3.0 * q if np.isfinite(q) and q > 0 else 1.0, 1e-6, 10.0)
            )

            t0 = time.perf_counter()
            self.coef_, self.intercept_ = self._fit_ridge_core(
                x_sampled, y_sampled, self.alpha, sample_weights
            )
            self.xgb_model_ = self._fit_xgb(x_sampled, y_sampled, sample_weights)
            self.fit_time_ = time.perf_counter() - t0

            self.feature_count_ = len(self.selected_features_)
            self.is_trained_ = True
            return self
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
            return self

    def predict(self, features):
        self.fallback_used_ = False
        self.predict_feature_time_ = 0.0
        self.predict_model_time_ = 0.0

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback_prediction(features)

            t0 = time.perf_counter()
            engineered = self._make_features(features)
            self.predict_feature_time_ = time.perf_counter() - t0

            x_long, _, assets = self._panel_to_matrix(engineered, target=None)
            del engineered

            if x_long.empty:
                self.fallback_used_ = True
                return self._fallback_prediction(features)

            x_long = self._apply_imputation_frame(x_long)
            n_rows = len(features.index)
            n_assets = len(assets)

            t0 = time.perf_counter()
            ridge_pred = self._predict_ridge(x_long)
            xgb_pred = self._predict_xgb(x_long)
            raw_pred = (
                self._RIDGE_WEIGHT * ridge_pred + self._XGB_WEIGHT * xgb_pred
            )
            raw_pred = np.nan_to_num(raw_pred, nan=0.0, posinf=0.0, neginf=0.0)
            self.predict_model_time_ = time.perf_counter() - t0

            pred = pd.DataFrame(
                raw_pred.reshape(n_rows, n_assets),
                index=features.index,
                columns=assets,
                dtype=np.float32,
            )
            return self._finalize_prediction(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback_prediction(features)
