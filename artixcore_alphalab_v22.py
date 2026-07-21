import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.22: history-bridged, group-balanced v0.18 ensemble."""

    _RIDGE_ALPHA = 8.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20

    _RIDGE_WEIGHT = 0.72
    _TREE_A_WEIGHT = 0.12
    _TREE_B_WEIGHT = 0.12
    _RANK_WEIGHT = 0.04

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

    _XGB_BASE = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.05,
        "min_child_weight": 220,
        "reg_alpha": 0.03,
        "reg_lambda": 1.8,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 2,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.max_train_rows = 80000
        self.max_features = 35
        self.n_xgb_rounds = 12
        self.history_rows = 96

        self.selected_features_ = None
        self.assets_ = []
        self.history_tail_ = None

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
        self.xgb_model_a_ = None
        self.xgb_model_b_ = None

        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False

        self.training_rows_ = 0
        self.training_times_ = 0
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
            asset_level = (
                max(remaining, key=lambda level: counts[level])
                if remaining
                else feature_level
            )

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

        for feature_index, name in enumerate(names):
            raw = frames[name]
            blocks.append(self._block(f"{name}__raw", raw))

            if feature_index < 3:
                blocks.append(self._block(f"{name}__cs_rank", self._rank(raw)))
                blocks.append(self._block(f"{name}__diff_1", raw - raw.shift(1)))

            if feature_index == 0:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma20 = raw.rolling(20, min_periods=3).mean()
                ma60 = raw.rolling(60, min_periods=5).mean()
                sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
                ewma5 = raw.ewm(span=5, adjust=False, min_periods=2).mean()

                blocks.extend(
                    [
                        self._block(f"{name}__ma5", ma5),
                        self._block(f"{name}__ma20", ma20),
                        self._block(f"{name}__ma60", ma60),
                        self._block(f"{name}__sd20", sd20),
                        self._block(f"{name}__ewma5", ewma5),
                        self._block(f"{name}__ma5_rank", self._rank(ma5)),
                        self._block(f"{name}__ma20_rank", self._rank(ma20)),
                        self._block(f"{name}__ma60_rank", self._rank(ma60)),
                        self._block(f"{name}__roll_z", (raw - ma5) / (sd20 + 1e-6)),
                        self._block(f"{name}__mom_spread", ma5 - ma60),
                        self._block(
                            f"{name}__cs_demean",
                            raw.sub(raw.median(axis=1), axis=0),
                        ),
                    ]
                )

        if len(names) >= 2:
            rank_spread = self._rank(frames[names[0]]) - self._rank(frames[names[1]])
            blocks.append(self._block("interaction__rank_spread", rank_spread))

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

    def _prediction_features(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if self.history_tail_ is None or self.history_tail_.empty:
            return self._features(features)

        try:
            history = self.history_tail_.reindex(columns=features.columns)
            if history.shape[1] != features.shape[1]:
                return self._features(features)

            combined = pd.concat([history, features], axis=0, ignore_index=True)
            panel, assets = self._features(combined)
            panel = panel.iloc[-len(features) :].copy()
            panel.index = features.index
            return panel, assets
        except Exception:
            return self._features(features)

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

    def _long(self, panel, target=None, assets=None):
        x = panel.stack(level="asset", future_stack=True)
        x = x.replace([np.inf, -np.inf], np.nan)

        if target is None:
            return x

        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.replace([np.inf, -np.inf], np.nan).notna()
        return x.loc[valid], y.loc[valid].astype(np.float32)

    def _select(self, x):
        if x.empty:
            return []

        probe = x
        if len(probe) > 40000:
            positions = np.linspace(0, len(probe) - 1, 40000, dtype=np.int64)
            probe = probe.iloc[positions]

        values = probe.to_numpy(dtype=np.float32, copy=False)
        usable = []
        for column_index, column_name in enumerate(probe.columns):
            finite = np.isfinite(values[:, column_index])
            if finite.mean() < 0.05:
                continue
            if np.nanstd(values[finite, column_index]) < 1e-8:
                continue
            usable.append(column_name)

        priority = {name: index for index, name in enumerate(self._FEATURE_PRIORITY)}
        usable.sort(key=lambda name: priority.get(name, len(priority)))
        return usable[: self.max_features]

    def _sample_complete_times(self, index):
        if not isinstance(index, pd.MultiIndex):
            row_count = len(index)
            if row_count <= self.max_train_rows:
                positions = np.arange(row_count, dtype=np.int64)
            else:
                recent_count = int(self.max_train_rows * 0.60)
                start = row_count - recent_count
                older = np.linspace(
                    0,
                    max(start - 1, 0),
                    self.max_train_rows - recent_count,
                    dtype=np.int64,
                )
                positions = np.unique(
                    np.concatenate([older, np.arange(start, row_count, dtype=np.int64)])
                )
            return positions, positions.astype(np.int64), row_count

        time_values = index.get_level_values(0)
        time_codes, unique_times = pd.factorize(time_values, sort=False)
        total_times = len(unique_times)
        if total_times == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0

        group_sizes = np.bincount(time_codes, minlength=total_times)
        typical_assets = max(1, int(np.median(group_sizes[group_sizes > 0])))
        max_times = max(1, self.max_train_rows // typical_assets)

        if total_times <= max_times:
            selected_codes = np.arange(total_times, dtype=np.int64)
        else:
            recent_times = int(max_times * 0.60)
            older_times = max_times - recent_times
            recent_start = total_times - recent_times
            older_codes = np.linspace(
                0,
                recent_start - 1,
                older_times,
                dtype=np.int64,
            )
            selected_codes = np.unique(
                np.concatenate(
                    [older_codes, np.arange(recent_start, total_times, dtype=np.int64)]
                )
            )

        selected_mask = np.isin(time_codes, selected_codes)
        positions = np.flatnonzero(selected_mask).astype(np.int64)
        return positions, time_codes[positions].astype(np.int64), total_times

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
        mad = np.nanmedian(np.abs(values - self.mean_), axis=0).astype(np.float32)
        self.scale_ = (1.4826 * mad).astype(np.float32)
        fallback = values.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(self.scale_) | (self.scale_ < 1e-8)
        self.scale_[bad] = fallback[bad]
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
    def _weights(time_codes, total_times, decay):
        if len(time_codes) == 0 or total_times <= 1:
            return np.ones(len(time_codes), dtype=np.float32)

        ages = (total_times - 1) - time_codes.astype(np.float32)
        denominator = max(1.0, total_times * decay)
        weights = np.exp(-ages / denominator).astype(np.float32)
        mean_weight = float(weights.mean())
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            return np.ones(len(time_codes), dtype=np.float32)
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
        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        return prediction.astype(np.float32)

    @staticmethod
    def _zero_prediction(index, assets):
        return pd.DataFrame(0.0, index=index, columns=assets, dtype=np.float32)

    def _fallback(self, features, assets):
        try:
            panel, extracted_assets = self._prediction_features(features)
            if extracted_assets:
                assets = extracted_assets
            if panel.empty:
                return self._zero_prediction(features.index, assets)

            feature_names = list(panel.columns.get_level_values("feature").unique())
            preferred = next(
                (name for name in feature_names if name.endswith("__cs_rank")),
                None,
            )
            if preferred is None:
                preferred = feature_names[0]

            score = panel[preferred]
            score = score.reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finish(score)
        except Exception:
            return self._zero_prediction(features.index, assets)

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.ridge_coef_ = None
        self.rank_coef_ = None
        self.xgb_model_a_ = None
        self.xgb_model_b_ = None

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            self.history_tail_ = features.tail(self.history_rows).copy()

            feature_started = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_ = time.perf_counter() - feature_started
            self.assets_ = list(assets)

            x, y = self._long(panel, target, assets)
            del panel

            if x.empty or len(y) < 40:
                self.training_error_ = "Insufficient usable training observations"
                return self

            self.selected_features_ = self._select(x)
            if not self.selected_features_:
                self.training_error_ = "No usable features were selected"
                return self

            positions, time_codes, total_times = self._sample_complete_times(x.index)
            if len(positions) == 0:
                self.training_error_ = "No complete timestamps were sampled"
                return self

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

            weights = self._weights(time_codes, total_times, self._DECAY)
            fit_started = time.perf_counter()

            self.ridge_intercept_ = float(np.average(y_values, weights=weights))
            self.ridge_coef_ = self._ridge(
                matrix,
                y_values - self.ridge_intercept_,
                weights,
                self._RIDGE_ALPHA,
            )

            if isinstance(ys.index, pd.MultiIndex):
                ranked = ys.groupby(level=0).rank(method="average", pct=True)
            else:
                ranked = ys.rank(method="average", pct=True)
            rank_values = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
            rank_values = np.nan_to_num(
                rank_values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            self.rank_intercept_ = float(np.average(rank_values, weights=weights))
            self.rank_coef_ = self._ridge(
                matrix,
                rank_values - self.rank_intercept_,
                weights,
                self._RANK_ALPHA,
            )

            target_std = float(np.std(y_values))
            rank_std = float(np.std(rank_values))
            self.rank_scale_ = (
                target_std / rank_std
                if target_std > 1e-8 and rank_std > 1e-8
                else 1.0
            )

            params_a = dict(self._XGB_BASE)
            params_a.update(
                {
                    "subsample": 0.82,
                    "colsample_bytree": 0.82,
                    "seed": 42,
                }
            )
            params_b = dict(self._XGB_BASE)
            params_b.update(
                {
                    "subsample": 0.90,
                    "colsample_bytree": 0.72,
                    "seed": 137,
                }
            )

            dtrain = xgb.DMatrix(matrix, label=y_values, weight=weights)
            self.xgb_model_a_ = xgb.train(
                params_a,
                dtrain,
                num_boost_round=self.n_xgb_rounds,
            )
            self.xgb_model_b_ = xgb.train(
                params_b,
                dtrain,
                num_boost_round=self.n_xgb_rounds,
            )

            self.fit_time_ = time.perf_counter() - fit_started
            self.training_rows_ = len(xs)
            self.training_times_ = len(pd.Index(xs.index.get_level_values(0)).unique())
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
            panel, assets = self._prediction_features(features)
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
            tree_a = self.xgb_model_a_.predict(xgb.DMatrix(matrix)).astype(np.float32)
            tree_b = self.xgb_model_b_.predict(xgb.DMatrix(matrix)).astype(np.float32)
            rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)

            raw = (
                self._RIDGE_WEIGHT * ridge
                + self._TREE_A_WEIGHT * tree_a
                + self._TREE_B_WEIGHT * tree_b
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
