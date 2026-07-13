import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.9, robust cross-sectional residual ensemble."""

    _RIDGE_ALPHA = 8.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _RAW_WEIGHT = 0.88
    _RANK_WEIGHT = 0.12
    _RESIDUAL_WEIGHT = 0.12
    _HUBER_C = 1.50

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
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.04,
        "subsample": 0.82,
        "colsample_bytree": 0.80,
        "min_child_weight": 220,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 2,
        "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.max_train_rows = 80_000
        self.max_features = 35
        self.n_xgb_rounds = 12

        self.raw_coef_ = None
        self.raw_intercept_ = 0.0
        self.rank_coef_ = None
        self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None

        self.selected_features_ = None
        self.impute_ = None
        self.low_ = None
        self.high_ = None
        self.mean_ = None
        self.scale_ = None
        self.prediction_clip_ = 1.0

        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.feature_count_ = 0
        self.training_rows_ = 0
        self.feature_time_ = 0.0
        self.fit_time_ = 0.0
        self.predict_feature_time_ = 0.0
        self.predict_model_time_ = 0.0

    def _levels(self, columns):
        names = [str(value).lower() if value is not None else "" for value in columns.names]
        counts = [
            len(pd.Index(columns.get_level_values(level)).unique())
            for level in range(columns.nlevels)
        ]
        feature_level = next(
            (level for level, name in enumerate(names) if "feature" in name or "factor" in name),
            None,
        )
        asset_level = next(
            (
                level
                for level, name in enumerate(names)
                if "ticker" in name or "asset" in name or "symbol" in name
            ),
            None,
        )
        if feature_level is None:
            feature_level = int(np.argmin(counts))
        if asset_level is None:
            remaining = [level for level in range(columns.nlevels) if level != feature_level]
            asset_level = max(remaining, key=lambda level: counts[level]) if remaining else feature_level
        return feature_level, asset_level

    def _numeric(self, frame):
        return (
            frame.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            numeric = self._numeric(features)
            return {"Feature.1": numeric}, list(numeric.columns)

        feature_level, asset_level = self._levels(features.columns)
        feature_names = list(dict.fromkeys(features.columns.get_level_values(feature_level)))
        assets = list(dict.fromkeys(features.columns.get_level_values(asset_level)))
        frames = {}

        for feature_name in feature_names:
            columns = [
                column
                for column in features.columns
                if column[feature_level] == feature_name
            ]
            if not columns:
                continue
            block = features.loc[:, columns].copy()
            block.columns = [column[asset_level] for column in columns]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()]
            frames[str(feature_name)] = self._numeric(block.reindex(columns=assets))

        return frames, assets

    def _rank(self, frame):
        n_assets = max(frame.shape[1], 1)
        rank = frame.rank(axis=1, method="average")
        return (
            (rank - 0.5 * (n_assets + 1)) / (0.5 * max(n_assets - 1, 1))
        ).astype(np.float32)

    def _block(self, name, frame):
        output = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        output.columns = pd.MultiIndex.from_product(
            [[name], output.columns], names=["feature", "asset"]
        )
        return output

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty

        names = list(frames)
        blocks = []
        for index, name in enumerate(names):
            raw = frames[name]
            blocks.append(self._block(f"{name}__raw", raw))

            if index < 3:
                blocks.append(self._block(f"{name}__cs_rank", self._rank(raw)))
                blocks.append(self._block(f"{name}__diff_1", raw - raw.shift(1)))

            if index == 0:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma20 = raw.rolling(20, min_periods=3).mean()
                ma60 = raw.rolling(60, min_periods=5).mean()
                sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
                blocks.extend(
                    [
                        self._block(f"{name}__ma5", ma5),
                        self._block(f"{name}__ma20", ma20),
                        self._block(f"{name}__ma60", ma60),
                        self._block(f"{name}__sd20", sd20),
                        self._block(
                            f"{name}__ewma5",
                            raw.ewm(span=5, adjust=False, min_periods=2).mean(),
                        ),
                        self._block(f"{name}__ma5_rank", self._rank(ma5)),
                        self._block(f"{name}__ma20_rank", self._rank(ma20)),
                        self._block(f"{name}__ma60_rank", self._rank(ma60)),
                        self._block(f"{name}__roll_z", (raw - ma5) / (sd20 + 1.0e-6)),
                        self._block(f"{name}__mom_spread", ma5 - ma60),
                        self._block(
                            f"{name}__cs_demean",
                            raw.sub(raw.median(axis=1), axis=0),
                        ),
                    ]
                )

        if len(names) >= 2:
            blocks.append(
                self._block(
                    "interaction__rank_spread",
                    self._rank(frames[names[0]]) - self._rank(frames[names[1]]),
                )
            )

        panel = pd.concat(blocks, axis=1)
        feature_names = panel.columns.get_level_values("feature").unique()
        panel = panel.reindex(
            columns=pd.MultiIndex.from_product(
                [feature_names, assets], names=["feature", "asset"]
            )
        )
        return panel.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def _target(self, target, index, assets):
        if isinstance(target, pd.Series):
            frame = (
                target.unstack(level=-1)
                if isinstance(target.index, pd.MultiIndex)
                else target.to_frame()
            )
        elif isinstance(target, pd.DataFrame):
            frame = target.copy()
        else:
            frame = pd.DataFrame(target, index=index)

        if isinstance(frame.columns, pd.MultiIndex):
            _, asset_level = self._levels(frame.columns)
            frame.columns = frame.columns.get_level_values(asset_level)

        frame = self._numeric(frame.reindex(index=index, columns=assets))
        return frame.sub(frame.mean(axis=1), axis=0).astype(np.float32)

    def _long(self, panel, target=None):
        assets = list(panel.columns.get_level_values("asset").unique())
        x = panel.stack(level="asset", future_stack=True).replace(
            [np.inf, -np.inf], np.nan
        )
        if target is None:
            return x, None, assets

        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.notna()
        return x.loc[valid], y.loc[valid].astype(np.float32), assets

    def _select(self, x):
        if x.empty:
            return []

        probe = x if len(x) <= 40_000 else x.iloc[
            np.linspace(0, len(x) - 1, 40_000, dtype=np.int64)
        ]
        values = probe.to_numpy(dtype=np.float32, copy=False)
        keep = []
        for column_index, column in enumerate(probe.columns):
            column_values = values[:, column_index]
            finite = np.isfinite(column_values)
            if finite.mean() < 0.05 or not finite.any():
                continue
            if np.nanstd(column_values[finite]) < 1.0e-8:
                continue
            keep.append(column)

        priority = {name: index for index, name in enumerate(self._PRIORITY)}
        keep.sort(key=lambda name: priority.get(name, len(priority)))
        return keep[: self.max_features]

    def _sample(self, n_rows):
        if n_rows <= self.max_train_rows:
            return np.arange(n_rows, dtype=np.int64)

        recent = int(self.max_train_rows * 0.60)
        older = self.max_train_rows - recent
        start = n_rows - recent
        old_positions = np.linspace(0, start - 1, older, dtype=np.int64)
        return np.unique(
            np.concatenate([old_positions, np.arange(start, n_rows, dtype=np.int64)])
        )

    def _fit_preprocessor(self, x):
        values = x.to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        self.impute_ = np.nanmedian(values, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0

        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = np.take(self.impute_, np.where(bad)[1])

        self.low_ = np.nanquantile(values, 0.005, axis=0).astype(np.float32)
        self.high_ = np.nanquantile(values, 0.995, axis=0).astype(np.float32)
        invalid = (
            ~np.isfinite(self.low_)
            | ~np.isfinite(self.high_)
            | (self.low_ >= self.high_)
        )
        self.low_[invalid] = -10.0
        self.high_[invalid] = 10.0

        values = np.clip(values, self.low_, self.high_)
        self.mean_ = values.mean(axis=0).astype(np.float32)
        centered = values - self.mean_
        mad = np.nanmedian(np.abs(centered), axis=0).astype(np.float32)
        self.scale_ = (1.4826 * mad).astype(np.float32)
        fallback = values.std(axis=0).astype(np.float32)
        bad_scale = ~np.isfinite(self.scale_) | (self.scale_ < 1.0e-8)
        self.scale_[bad_scale] = fallback[bad_scale]
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1.0e-8)] = 1.0

    def _transform(self, x):
        values = x.reindex(columns=self.selected_features_).to_numpy(
            dtype=np.float32, copy=True
        )
        values[~np.isfinite(values)] = np.nan
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = np.take(self.impute_, np.where(bad)[1])
        values = np.clip(values, self.low_, self.high_)
        return np.nan_to_num(
            (values - self.mean_) / self.scale_,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    def _weights(self, positions, total_rows):
        if total_rows <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total_rows - 1) - positions.astype(np.float32)
        weights = np.exp(-ages / max(1.0, total_rows * self._DECAY))
        return (weights / weights.mean()).astype(np.float32)

    def _ridge(self, matrix, target, weights, alpha):
        root = np.sqrt(weights).astype(np.float32)
        weighted_x = matrix * root[:, None]
        weighted_y = target * root
        gram = weighted_x.T @ weighted_x
        gram.flat[:: gram.shape[0] + 1] += alpha
        rhs = weighted_x.T @ weighted_y
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _finish(self, prediction):
        prediction = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prediction = prediction.clip(-self.prediction_clip_, self.prediction_clip_)
        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        return prediction.astype(np.float32)

    def _to_frame(self, values, index, assets):
        series = pd.Series(values, index=index, dtype=np.float32)
        if isinstance(series.index, pd.MultiIndex):
            frame = series.unstack(level=-1)
        else:
            frame = series.to_frame()
        return frame.reindex(columns=assets).fillna(0.0).astype(np.float32)

    def _fallback(self, features):
        try:
            frames, assets = self._extract(features)
            first = next(iter(frames.values()))
            prediction = self._rank(first) + 0.3 * self._rank(
                first.rolling(5, min_periods=2).mean()
            )
            return self._finish(
                prediction.reindex(index=features.index, columns=assets).fillna(0.0)
            )
        except Exception:
            try:
                _, assets = self._extract(features)
            except Exception:
                assets = []
            return pd.DataFrame(
                0.0, index=features.index, columns=assets, dtype=np.float32
            )

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.raw_coef_ = None
        self.rank_coef_ = None
        self.xgb_model_ = None

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            started = time.perf_counter()
            panel = self._features(features)
            self.feature_time_ = time.perf_counter() - started
            x, y, _ = self._long(panel, target)
            del panel

            if x.empty or len(y) < 40:
                return self

            self.selected_features_ = self._select(x)
            if not self.selected_features_:
                return self

            positions = self._sample(len(x))
            sampled_x = x.iloc[positions][self.selected_features_]
            sampled_y = y.iloc[positions]

            self._fit_preprocessor(sampled_x)
            matrix = self._transform(sampled_x)
            recency_weights = self._weights(positions, len(x))

            target_values = np.nan_to_num(
                sampled_y.to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target_limit = (
                np.nanquantile(np.abs(target_values), 0.995)
                if target_values.size
                else 1.0
            )
            if not np.isfinite(target_limit) or target_limit <= 0:
                target_limit = 1.0
            target_values = np.clip(target_values, -target_limit, target_limit)
            self.prediction_clip_ = float(np.clip(3.0 * target_limit, 1.0e-6, 10.0))

            started = time.perf_counter()
            self.raw_intercept_ = float(target_values.mean()) if target_values.size else 0.0
            centered_target = target_values - self.raw_intercept_

            initial_coef = self._ridge(
                matrix, centered_target, recency_weights, self._RIDGE_ALPHA
            )
            initial_prediction = self.raw_intercept_ + matrix @ initial_coef
            initial_residual = target_values - initial_prediction
            residual_median = float(np.median(initial_residual))
            residual_mad = float(
                np.median(np.abs(initial_residual - residual_median))
            )
            robust_scale = max(1.4826 * residual_mad, 1.0e-6)
            huber_weights = np.minimum(
                1.0,
                (self._HUBER_C * robust_scale)
                / (np.abs(initial_residual - residual_median) + 1.0e-6),
            ).astype(np.float32)
            robust_weights = (recency_weights * huber_weights).astype(np.float32)
            robust_weights /= max(float(robust_weights.mean()), 1.0e-6)

            self.raw_coef_ = self._ridge(
                matrix, centered_target, robust_weights, self._RIDGE_ALPHA
            )

            ranked_target = sampled_y.groupby(level=0).rank(
                method="average", pct=True
            ) if isinstance(sampled_y.index, pd.MultiIndex) else sampled_y.rank(
                method="average", pct=True
            )
            rank_values = ((ranked_target - 0.5) * 2.0).to_numpy(dtype=np.float32)
            self.rank_intercept_ = float(rank_values.mean()) if rank_values.size else 0.0
            self.rank_coef_ = self._ridge(
                matrix,
                rank_values - self.rank_intercept_,
                recency_weights,
                self._RANK_ALPHA,
            )

            target_std = float(np.std(target_values)) if target_values.size else 1.0
            rank_std = float(np.std(rank_values)) if rank_values.size else 1.0
            self.rank_scale_ = (
                target_std / rank_std
                if target_std > 1.0e-8 and rank_std > 1.0e-8
                else 1.0
            )

            raw_prediction = self.raw_intercept_ + matrix @ self.raw_coef_
            rank_prediction = self.rank_scale_ * (
                self.rank_intercept_ + matrix @ self.rank_coef_
            )
            base_prediction = (
                self._RAW_WEIGHT * raw_prediction
                + self._RANK_WEIGHT * rank_prediction
            )
            residual_target = target_values - base_prediction

            self.xgb_model_ = xgb.train(
                self._XGB,
                xgb.DMatrix(
                    matrix,
                    label=residual_target,
                    weight=robust_weights,
                ),
                num_boost_round=self.n_xgb_rounds,
            )
            self.fit_time_ = time.perf_counter() - started

            self.training_rows_ = len(sampled_x)
            self.feature_count_ = len(self.selected_features_)
            self.is_trained_ = True
            return self

        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
            return self

    def predict(self, features):
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback(features)

            started = time.perf_counter()
            panel = self._features(features)
            self.predict_feature_time_ = time.perf_counter() - started
            x, _, assets = self._long(panel)
            del panel

            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)

            started = time.perf_counter()
            matrix = self._transform(x)
            raw_prediction = self.raw_intercept_ + matrix @ self.raw_coef_
            rank_prediction = self.rank_scale_ * (
                self.rank_intercept_ + matrix @ self.rank_coef_
            )
            residual_prediction = self.xgb_model_.predict(
                xgb.DMatrix(matrix)
            ).astype(np.float32)
            combined = (
                self._RAW_WEIGHT * raw_prediction
                + self._RANK_WEIGHT * rank_prediction
                + self._RESIDUAL_WEIGHT * residual_prediction
            )
            combined = np.nan_to_num(
                combined, nan=0.0, posinf=0.0, neginf=0.0
            )
            prediction = self._to_frame(combined, x.index, assets).reindex(
                index=features.index, columns=assets
            ).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(prediction)

        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
