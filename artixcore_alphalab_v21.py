import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.21: permutation-invariant cross-sectional ensemble."""

    _RIDGE_ALPHA = 10.0
    _RANK_ALPHA = 24.0
    _DECAY = 0.22

    _RIDGE_WEIGHT = 0.70
    _XGB_WEIGHT = 0.26
    _RANK_WEIGHT = 0.04

    _XGB = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 220,
        "reg_alpha": 0.03,
        "reg_lambda": 1.8,
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

        self.max_train_rows = 80000
        self.max_features = 30
        self.n_xgb_rounds = 15

        self.selected_features_ = None
        self.assets_ = []

        self.impute_ = None
        self.low_ = None
        self.high_ = None
        self.mean_ = None
        self.scale_ = None

        self.ridge_coef_ = None
        self.rank_coef_ = None
        self.ridge_intercept_ = 0.0
        self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None

        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False

        self.training_rows_ = 0
        self.feature_count_ = 0
        self.feature_time_ = 0.0
        self.fit_time_ = 0.0
        self.predict_feature_time_ = 0.0
        self.predict_model_time_ = 0.0

    @staticmethod
    def _numeric(frame):
        return (
            frame.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )

    def _levels(self, columns):
        names = [str(value).lower() if value is not None else "" for value in columns.names]
        counts = [
            len(pd.Index(columns.get_level_values(level)).unique())
            for level in range(columns.nlevels)
        ]

        feature_level = next(
            (
                level
                for level, name in enumerate(names)
                if "feature" in name or "factor" in name
            ),
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
            block = block.reindex(columns=assets)
            frames[str(feature_name)] = self._numeric(block)

        return frames, assets

    @staticmethod
    def _rank(frame):
        asset_count = max(frame.shape[1], 1)
        ranks = frame.rank(axis=1, method="average")
        denominator = 0.5 * max(asset_count - 1, 1)
        return ((ranks - 0.5 * (asset_count + 1)) / denominator).astype(np.float32)

    @staticmethod
    def _robust_z(frame):
        median = frame.median(axis=1)
        centered = frame.sub(median, axis=0)
        mad = centered.abs().median(axis=1)
        scale = (1.4826 * mad).replace(0.0, np.nan)
        z = centered.div(scale + 1e-6, axis=0)
        return z.clip(-5.0, 5.0).astype(np.float32)

    @staticmethod
    def _block(name, frame):
        output = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        output.columns = pd.MultiIndex.from_product(
            [[name], output.columns],
            names=["feature", "asset"],
        )
        return output

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty, assets

        names = list(frames)
        blocks = []
        ranks = {}
        zscores = {}

        for name in names:
            raw = frames[name]
            rank = self._rank(raw)
            zscore = self._robust_z(raw)
            ranks[name] = rank
            zscores[name] = zscore

            blocks.append(self._block(f"{name}__raw", raw))
            blocks.append(self._block(f"{name}__cs_rank", rank))
            blocks.append(self._block(f"{name}__robust_z", zscore))
            blocks.append(
                self._block(
                    f"{name}__cs_demean",
                    raw.sub(raw.median(axis=1), axis=0),
                )
            )

        interaction_names = names[: min(3, len(names))]
        for left_index in range(len(interaction_names)):
            for right_index in range(left_index + 1, len(interaction_names)):
                left = interaction_names[left_index]
                right = interaction_names[right_index]
                blocks.append(
                    self._block(
                        f"interaction__rank_spread__{left}__{right}",
                        ranks[left] - ranks[right],
                    )
                )
                blocks.append(
                    self._block(
                        f"interaction__rank_product__{left}__{right}",
                        ranks[left] * ranks[right],
                    )
                )

        rank_stack = np.stack([ranks[name].to_numpy(dtype=np.float32) for name in names], axis=0)
        z_stack = np.stack([zscores[name].to_numpy(dtype=np.float32) for name in names], axis=0)

        rank_mean = pd.DataFrame(
            np.nanmean(rank_stack, axis=0),
            index=features.index,
            columns=assets,
        )
        rank_std = pd.DataFrame(
            np.nanstd(rank_stack, axis=0),
            index=features.index,
            columns=assets,
        )
        rank_range = pd.DataFrame(
            np.nanmax(rank_stack, axis=0) - np.nanmin(rank_stack, axis=0),
            index=features.index,
            columns=assets,
        )
        z_mean = pd.DataFrame(
            np.nanmean(z_stack, axis=0),
            index=features.index,
            columns=assets,
        )
        z_std = pd.DataFrame(
            np.nanstd(z_stack, axis=0),
            index=features.index,
            columns=assets,
        )

        blocks.extend(
            [
                self._block("consensus__rank_mean", rank_mean),
                self._block("consensus__rank_std", rank_std),
                self._block("consensus__rank_range", rank_range),
                self._block("consensus__z_mean", z_mean),
                self._block("consensus__z_std", z_std),
            ]
        )

        panel = pd.concat(blocks, axis=1)
        feature_names = panel.columns.get_level_values("feature").unique()
        columns = pd.MultiIndex.from_product(
            [feature_names, assets],
            names=["feature", "asset"],
        )

        return (
            panel.reindex(columns=columns)
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32),
            assets,
        )

    def _target(self, target, index, assets):
        if isinstance(target, pd.Series):
            if isinstance(target.index, pd.MultiIndex):
                frame = target.unstack(level=-1)
            else:
                frame = target.to_frame()
        elif isinstance(target, pd.DataFrame):
            frame = target.copy()
        else:
            frame = pd.DataFrame(target, index=index)

        if isinstance(frame.columns, pd.MultiIndex):
            _, asset_level = self._levels(frame.columns)
            frame.columns = frame.columns.get_level_values(asset_level)

        frame = self._numeric(frame.reindex(index=index, columns=assets))
        return frame.sub(frame.mean(axis=1), axis=0).astype(np.float32)

    def _long(self, panel, target=None, assets=None):
        x = panel.stack(level="asset", future_stack=True)
        x = x.replace([np.inf, -np.inf], np.nan)

        if target is None:
            return x

        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.notna()
        return x.loc[valid], y.loc[valid].astype(np.float32)

    def _stable_select(self, x, y):
        if x.empty:
            return []

        usable = []
        values = x.to_numpy(dtype=np.float32, copy=False)
        for column_index, column_name in enumerate(x.columns):
            finite = np.isfinite(values[:, column_index])
            if finite.mean() < 0.05:
                continue
            if np.nanstd(values[finite, column_index]) < 1e-8:
                continue
            usable.append(column_name)

        if len(usable) <= self.max_features:
            return usable

        probe = x[usable]
        probe_y = y
        if len(probe) > 60000:
            positions = np.linspace(0, len(probe) - 1, 60000, dtype=np.int64)
            probe = probe.iloc[positions]
            probe_y = probe_y.iloc[positions]

        matrix = probe.to_numpy(dtype=np.float32, copy=True)
        target = probe_y.to_numpy(dtype=np.float32, copy=True)

        matrix[~np.isfinite(matrix)] = np.nan
        medians = np.nanmedian(matrix, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(medians, np.where(missing)[1])
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        periods = probe.index.get_level_values(0)
        unique_periods = pd.Index(periods).unique()
        correlations = []

        for period in unique_periods:
            mask = np.asarray(periods == period)
            if int(mask.sum()) < 5:
                continue
            period_x = matrix[mask]
            period_y = target[mask]
            period_x = period_x - period_x.mean(axis=0, keepdims=True)
            period_y = period_y - period_y.mean()
            numerator = period_x.T @ period_y
            denominator = np.sqrt(
                np.sum(period_x * period_x, axis=0) * np.sum(period_y * period_y)
            )
            corr = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float32),
                where=denominator > 1e-12,
            )
            correlations.append(np.clip(corr, -1.0, 1.0))

        if not correlations:
            return usable[: self.max_features]

        corr_matrix = np.vstack(correlations)
        mean_corr = np.nanmean(corr_matrix, axis=0)
        std_corr = np.nanstd(corr_matrix, axis=0)
        nonzero = np.sign(corr_matrix)
        expected_sign = np.sign(mean_corr)
        consistency = np.mean(nonzero == expected_sign[None, :], axis=0)
        score = np.abs(mean_corr) * (0.50 + 0.50 * consistency) / (0.05 + std_corr)

        order = np.argsort(-np.nan_to_num(score, nan=-1.0))
        selected = [usable[index] for index in order[: self.max_features]]
        return selected

    def _sample(self, row_count):
        if row_count <= self.max_train_rows:
            return np.arange(row_count, dtype=np.int64)

        recent_count = int(self.max_train_rows * 0.60)
        recent_start = row_count - recent_count
        older_count = self.max_train_rows - recent_count
        older = np.linspace(0, recent_start - 1, older_count, dtype=np.int64)
        recent = np.arange(recent_start, row_count, dtype=np.int64)
        return np.unique(np.concatenate([older, recent]))

    def _fit_preprocessor(self, x):
        values = x.to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan

        self.impute_ = np.nanmedian(values, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0

        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.impute_, np.where(missing)[1])

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
        self.scale_ = values.std(axis=0).astype(np.float32)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        values = x.reindex(columns=self.selected_features_).to_numpy(
            dtype=np.float32,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan

        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.impute_, np.where(missing)[1])

        values = np.clip(values, self.low_, self.high_)
        values = (values - self.mean_) / self.scale_
        return np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    @staticmethod
    def _weights(positions, total_rows, decay):
        if total_rows <= 1:
            return np.ones(len(positions), dtype=np.float32)

        ages = (total_rows - 1) - positions.astype(np.float32)
        denominator = max(1.0, total_rows * decay)
        weights = np.exp(-ages / denominator).astype(np.float32)
        mean_weight = float(weights.mean())
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            return np.ones(len(positions), dtype=np.float32)
        return weights / mean_weight

    @staticmethod
    def _ridge(matrix, target, weights, alpha):
        root = np.sqrt(weights).astype(np.float32)
        weighted_matrix = matrix * root[:, None]
        weighted_target = target * root

        gram = weighted_matrix.T @ weighted_matrix
        rhs = weighted_matrix.T @ weighted_target
        gram.flat[:: gram.shape[0] + 1] += alpha

        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _finish(self, prediction):
        prediction = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prediction = prediction.clip(-self.prediction_clip_, self.prediction_clip_)

        row_median = prediction.median(axis=1)
        centered = prediction.sub(row_median, axis=0)
        row_mad = centered.abs().median(axis=1)
        cap = (4.0 * 1.4826 * row_mad).replace(0.0, np.nan)
        lower = row_median - cap
        upper = row_median + cap
        prediction = prediction.clip(lower=lower, upper=upper, axis=0)

        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        return prediction.astype(np.float32)

    @staticmethod
    def _zero_prediction(index, assets):
        return pd.DataFrame(0.0, index=index, columns=assets, dtype=np.float32)

    def _fallback(self, features, assets):
        try:
            frames, extracted_assets = self._extract(features)
            if extracted_assets:
                assets = extracted_assets
            if not frames:
                return self._zero_prediction(features.index, assets)

            first = next(iter(frames.values()))
            prediction = self._rank(first)
            prediction = prediction.reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finish(prediction)
        except Exception:
            return self._zero_prediction(features.index, assets)

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.ridge_coef_ = None
        self.rank_coef_ = None
        self.xgb_model_ = None

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            feature_started = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_ = time.perf_counter() - feature_started
            self.assets_ = list(assets)

            x, y = self._long(panel, target, assets)
            del panel

            if x.empty or len(y) < 40:
                self.training_error_ = "Insufficient usable training observations"
                return self

            self.selected_features_ = self._stable_select(x, y)
            if not self.selected_features_:
                self.training_error_ = "No usable features were selected"
                return self

            positions = self._sample(len(x))
            xs = x.iloc[positions][self.selected_features_]
            ys = y.iloc[positions]

            self._fit_preprocessor(xs)
            matrix = self._transform(xs)

            y_values = np.nan_to_num(
                ys.to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target_limit = np.nanquantile(np.abs(y_values), 0.995) if y_values.size else 1.0
            if not np.isfinite(target_limit) or target_limit <= 0.0:
                target_limit = 1.0
            y_values = np.clip(y_values, -target_limit, target_limit).astype(np.float32)
            self.prediction_clip_ = float(np.clip(3.0 * target_limit, 1e-6, 10.0))

            weights = self._weights(positions, len(x), self._DECAY)
            fit_started = time.perf_counter()

            self.ridge_intercept_ = float(np.average(y_values, weights=weights))
            self.ridge_coef_ = self._ridge(
                matrix,
                y_values - self.ridge_intercept_,
                weights,
                self._RIDGE_ALPHA,
            )

            ranked = ys.groupby(level=0).rank(method="average", pct=True)
            rank_values = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
            rank_values = np.nan_to_num(rank_values, nan=0.0, posinf=0.0, neginf=0.0)

            self.rank_intercept_ = float(np.average(rank_values, weights=weights))
            self.rank_coef_ = self._ridge(
                matrix,
                rank_values - self.rank_intercept_,
                weights,
                self._RANK_ALPHA,
            )

            target_std = float(np.std(y_values))
            rank_std = float(np.std(rank_values))
            self.rank_scale_ = target_std / rank_std if target_std > 1e-8 and rank_std > 1e-8 else 1.0

            self.xgb_model_ = xgb.train(
                self._XGB,
                xgb.DMatrix(matrix, label=y_values, weight=weights),
                num_boost_round=self.n_xgb_rounds,
            )

            self.fit_time_ = time.perf_counter() - fit_started
            self.training_rows_ = len(xs)
            self.feature_count_ = len(self.selected_features_)
            self.is_trained_ = True
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False

        return self

    def predict(self, features):
        self.fallback_used_ = False

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            feature_started = time.perf_counter()
            panel, assets = self._features(features)
            self.predict_feature_time_ = time.perf_counter() - feature_started

            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)

            x = self._long(panel)
            del panel
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)

            model_started = time.perf_counter()
            matrix = self._transform(x)

            ridge = self.ridge_intercept_ + matrix @ self.ridge_coef_
            tree = self.xgb_model_.predict(xgb.DMatrix(matrix)).astype(np.float32)
            rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)

            raw = (
                self._RIDGE_WEIGHT * ridge
                + self._XGB_WEIGHT * tree
                + self._RANK_WEIGHT * rank
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            series = pd.Series(raw, index=x.index, dtype=np.float32)
            prediction = (
                series.unstack(level=-1)
                .reindex(index=features.index, columns=assets)
                .fillna(0.0)
            )

            self.predict_model_time_ = time.perf_counter() - model_started
            return self._finish(prediction)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
