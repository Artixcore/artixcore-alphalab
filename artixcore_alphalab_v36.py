import time
import numpy as np
import pandas as pd
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """AlphaLab v0.36: sparse causal secondary signal from Features 4 to 6."""

    RIDGE_ALPHA = 22.0
    RANK_ALPHA = 42.0
    DECAY = 0.38

    RIDGE_WEIGHT = 0.65
    RANK_WEIGHT = 0.35

    PRIMARY_FEATURES = ("Feature.4", "Feature.5", "Feature.6")

    FEATURE_PRIORITY = (
        "Feature.4__raw",
        "Feature.5__raw",
        "Feature.6__raw",
        "Feature.4__rank",
        "Feature.5__rank",
        "Feature.6__rank",
        "Feature.4__demean",
        "Feature.5__demean",
        "Feature.6__demean",
        "Feature.4__diff1",
        "Feature.5__diff1",
        "Feature.6__diff1",
        "Feature.4__diff3",
        "Feature.5__diff3",
        "Feature.6__diff3",
        "Feature.4__reversal5",
        "Feature.4__ewma_reversal",
        "Feature.4__z10",
        "Feature.4__rank_persist",
        "Feature.4__vol_ratio",
        "interaction__rank45",
        "interaction__rank46",
        "Feature.5__reversal5",
        "Feature.5__ewma_reversal",
        "Feature.5__z10",
        "interaction__rank56",
        "interaction__relative456",
        "Feature.4__rank_turnover",
        "Feature.4__lag1",
        "Feature.5__lag1",
        "Feature.6__lag1",
    )

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.max_train_rows = 80000
        self.max_features = 22
        self.history_rows = 32

        self.selected_features_ = None
        self.primary_features_ = []
        self.assets_ = []
        self.history_tail_ = None

        self.impute_ = None
        self.low_ = None
        self.high_ = None
        self.center_ = None
        self.scale_ = None

        self.ridge_coef_ = None
        self.rank_coef_ = None
        self.ridge_intercept_ = 0.0
        self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0

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

    @staticmethod
    def _rank(frame):
        asset_count = max(frame.shape[1], 1)
        denominator = 0.5 * max(asset_count - 1, 1)
        return (
            (frame.rank(axis=1, method="average") - 0.5 * (asset_count + 1))
            / denominator
        ).astype(np.float32)

    @staticmethod
    def _block(name, frame):
        output = frame.replace([np.inf, -np.inf], np.nan).astype(
            np.float32, copy=False
        )
        output.columns = pd.MultiIndex.from_product(
            [[name], output.columns], names=["feature", "asset"]
        )
        return output

    @staticmethod
    def _stack(frame, level=None):
        try:
            if level is None:
                return frame.stack(future_stack=True)
            return frame.stack(level=level, future_stack=True)
        except TypeError:
            if level is None:
                return frame.stack(dropna=False)
            return frame.stack(level=level, dropna=False)

    def _levels(self, columns):
        names = [
            str(value).lower() if value is not None else ""
            for value in columns.names
        ]
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
                if "asset" in name or "ticker" in name or "symbol" in name
            ),
            None,
        )

        if feature_level is None:
            feature_level = int(np.argmin(counts))
        if asset_level is None:
            remaining = [
                level for level in range(columns.nlevels)
                if level != feature_level
            ]
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
            return {"Feature.4": numeric}, list(numeric.columns)

        feature_level, asset_level = self._levels(features.columns)
        assets = list(
            dict.fromkeys(features.columns.get_level_values(asset_level))
        )
        frames = {}

        for feature_name in dict.fromkeys(
            features.columns.get_level_values(feature_level)
        ):
            columns = [
                column
                for column in features.columns
                if column[feature_level] == feature_name
            ]
            block = features.loc[:, columns].copy()
            block.columns = [column[asset_level] for column in columns]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()]
            block = block.reindex(columns=assets)
            frames[str(feature_name)] = self._numeric(block)

        return frames, assets

    def _choose_primary(self, names):
        exact = [name for name in self.PRIMARY_FEATURES if name in names]
        if len(exact) == 3:
            return exact

        normalized = {str(name).lower(): name for name in names}
        selected = []
        for wanted in self.PRIMARY_FEATURES:
            match = normalized.get(wanted.lower())
            if match is not None:
                selected.append(match)

        for name in reversed(names):
            if name not in selected:
                selected.append(name)
            if len(selected) == 3:
                break
        return list(reversed(selected)) if len(exact) == 0 else selected[:3]

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays(
                [[], []], names=["feature", "asset"]
            )
            return empty, assets

        primary = self._choose_primary(list(frames))
        self.primary_features_ = list(primary)
        ranks = {name: self._rank(frames[name]) for name in primary}
        blocks = []

        for index, name in enumerate(primary):
            raw = frames[name]
            rank = ranks[name]
            median = raw.median(axis=1)

            blocks.extend(
                [
                    self._block(name + "__raw", raw),
                    self._block(name + "__rank", rank),
                    self._block(
                        name + "__demean", raw.sub(median, axis=0)
                    ),
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__diff1", raw.diff(1)),
                    self._block(name + "__diff3", raw.diff(3)),
                ]
            )

            if index < 2:
                mean5 = raw.rolling(5, min_periods=2).mean()
                mean10 = raw.rolling(10, min_periods=3).mean()
                ewma8 = raw.ewm(
                    span=8, adjust=False, min_periods=3
                ).mean()
                vol10 = raw.rolling(10, min_periods=3).std(ddof=0)

                blocks.extend(
                    [
                        self._block(
                            name + "__reversal5", mean5 - raw
                        ),
                        self._block(
                            name + "__ewma_reversal", ewma8 - raw
                        ),
                        self._block(
                            name + "__z10",
                            (raw - mean10) / (vol10 + 1e-6),
                        ),
                    ]
                )

            if index == 0:
                rank_lag1 = rank.shift(1)
                vol5 = raw.rolling(5, min_periods=2).std(ddof=0)
                vol20 = raw.rolling(20, min_periods=4).std(ddof=0)
                blocks.extend(
                    [
                        self._block(
                            name + "__rank_persist", rank * rank_lag1
                        ),
                        self._block(
                            name + "__rank_turnover",
                            (rank - rank_lag1).abs(),
                        ),
                        self._block(
                            name + "__vol_ratio",
                            vol5 / (vol20 + 1e-6),
                        ),
                    ]
                )

        if len(primary) >= 2:
            first = ranks[primary[0]]
            second = ranks[primary[1]]
            blocks.append(
                self._block("interaction__rank45", first - second)
            )

        if len(primary) >= 3:
            first = ranks[primary[0]]
            second = ranks[primary[1]]
            third = ranks[primary[2]]
            blocks.extend(
                [
                    self._block(
                        "interaction__rank46", first - third
                    ),
                    self._block(
                        "interaction__rank56", second - third
                    ),
                    self._block(
                        "interaction__relative456",
                        first + second - 2.0 * third,
                    ),
                ]
            )

        panel = pd.concat(blocks, axis=1)
        feature_names = panel.columns.get_level_values("feature").unique()
        columns = pd.MultiIndex.from_product(
            [feature_names, assets], names=["feature", "asset"]
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
            combined = pd.concat([history, features], ignore_index=True)
            panel, assets = self._features(combined)
            panel = panel.tail(len(features)).copy()
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
        matrix_frame = self._stack(panel, "asset").replace(
            [np.inf, -np.inf], np.nan
        )
        if target is None:
            return matrix_frame

        response = self._stack(self._target(target, panel.index, assets))
        matrix_frame, response = matrix_frame.align(
            response, join="inner", axis=0
        )
        valid = response.replace([np.inf, -np.inf], np.nan).notna()
        return matrix_frame.loc[valid], response.loc[valid].astype(np.float32)

    def _select(self, matrix_frame):
        if matrix_frame.empty:
            return []

        probe = matrix_frame
        if len(probe) > 40000:
            positions = np.linspace(
                0, len(probe) - 1, 40000, dtype=np.int64
            )
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

        priority = {
            name: index for index, name in enumerate(self.FEATURE_PRIORITY)
        }
        usable.sort(key=lambda name: priority.get(name, len(priority)))
        return usable[: self.max_features]

    def _sample_complete_times(self, index):
        if not isinstance(index, pd.MultiIndex):
            positions = np.arange(len(index), dtype=np.int64)
            if len(positions) > self.max_train_rows:
                positions = positions[-self.max_train_rows :]
            return positions, positions, len(index)

        time_codes, unique_times = pd.factorize(
            index.get_level_values(0), sort=False
        )
        total_times = len(unique_times)
        sizes = np.bincount(time_codes, minlength=max(total_times, 1))
        positive_sizes = sizes[sizes > 0]
        typical_assets = max(
            1, int(np.median(positive_sizes)) if len(positive_sizes) else 1
        )
        max_times = max(1, self.max_train_rows // typical_assets)

        if total_times <= max_times:
            chosen = np.arange(total_times, dtype=np.int64)
        else:
            recent = max(1, int(max_times * 0.50))
            older = max_times - recent
            recent_start = total_times - recent
            older_codes = (
                np.linspace(
                    0,
                    recent_start - 1,
                    older,
                    dtype=np.int64,
                )
                if older
                else np.empty(0, dtype=np.int64)
            )
            chosen = np.unique(
                np.concatenate(
                    [
                        older_codes,
                        np.arange(recent_start, total_times),
                    ]
                )
            )

        positions = np.flatnonzero(
            np.isin(time_codes, chosen)
        ).astype(np.int64)
        return (
            positions,
            time_codes[positions].astype(np.int64),
            total_times,
        )

    def _fit_preprocessor(self, matrix_frame):
        values = matrix_frame.to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan

        self.impute_ = np.nanmedian(values, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0

        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.impute_, np.where(missing)[1])

        self.low_ = np.nanquantile(values, 0.01, axis=0).astype(np.float32)
        self.high_ = np.nanquantile(values, 0.99, axis=0).astype(np.float32)
        invalid = (
            ~np.isfinite(self.low_)
            | ~np.isfinite(self.high_)
            | (self.low_ >= self.high_)
        )
        self.low_[invalid] = -10.0
        self.high_[invalid] = 10.0

        values = np.clip(values, self.low_, self.high_)
        self.center_ = np.nanmedian(values, axis=0).astype(np.float32)
        self.scale_ = (
            1.4826
            * np.nanmedian(np.abs(values - self.center_), axis=0)
        ).astype(np.float32)
        fallback = np.nanstd(values, axis=0).astype(np.float32)
        invalid = ~np.isfinite(self.scale_) | (self.scale_ < 1e-8)
        self.scale_[invalid] = fallback[invalid]
        self.scale_[
            ~np.isfinite(self.scale_) | (self.scale_ < 1e-8)
        ] = 1.0

    def _transform(self, matrix_frame):
        values = matrix_frame.reindex(
            columns=self.selected_features_
        ).to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan

        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.impute_, np.where(missing)[1])

        values = np.clip(values, self.low_, self.high_)
        values = (values - self.center_) / self.scale_
        return np.nan_to_num(values).astype(np.float32)

    @staticmethod
    def _weights(time_codes, total_times):
        if len(time_codes) == 0 or total_times <= 1:
            return np.ones(len(time_codes), dtype=np.float32)

        ages = (total_times - 1) - time_codes.astype(np.float32)
        denominator = max(
            1.0, total_times * ArtixcoreAlphaLabPredictor.DECAY
        )
        weights = np.exp(-ages / denominator).astype(np.float32)
        return weights / max(float(weights.mean()), 1e-8)

    @staticmethod
    def _ridge(matrix, target, weights, alpha):
        root = np.sqrt(np.maximum(weights, 1e-8)).astype(np.float32)
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
        prediction = prediction.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        prediction = prediction.clip(
            -self.prediction_clip_, self.prediction_clip_
        )
        prediction = prediction.sub(
            prediction.mean(axis=1), axis=0
        ).fillna(0.0)
        return prediction.astype(np.float32)

    @staticmethod
    def _zero_prediction(index, assets):
        return pd.DataFrame(
            0.0, index=index, columns=assets, dtype=np.float32
        )

    def _fallback(self, features, assets):
        try:
            panel, extracted_assets = self._prediction_features(features)
            if extracted_assets:
                assets = extracted_assets
            if panel.empty:
                return self._zero_prediction(features.index, assets)

            preferred = next(
                (
                    name
                    for name in self.FEATURE_PRIORITY
                    if name in panel.columns.get_level_values("feature")
                    and name.endswith("__rank")
                ),
                None,
            )
            if preferred is None:
                names = list(
                    panel.columns.get_level_values("feature").unique()
                )
                preferred = names[0]

            score = panel[preferred].reindex(
                index=features.index, columns=assets
            ).fillna(0.0)
            return self._finish(score)
        except Exception:
            return self._zero_prediction(features.index, assets)

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.ridge_coef_ = None
        self.rank_coef_ = None

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            self.history_tail_ = features.tail(self.history_rows).copy()

            started = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_ = time.perf_counter() - started
            self.assets_ = list(assets)

            matrix_frame, response = self._long(panel, target, assets)
            if matrix_frame.empty or len(response) < 60:
                self.training_error_ = (
                    "Insufficient usable training observations"
                )
                return self

            self.selected_features_ = self._select(matrix_frame)
            if not self.selected_features_:
                self.training_error_ = "No usable features selected"
                return self

            positions, time_codes, total_times = (
                self._sample_complete_times(matrix_frame.index)
            )
            if len(positions) == 0:
                self.training_error_ = "No timestamps sampled"
                return self

            sampled_features = matrix_frame.iloc[positions][
                self.selected_features_
            ]
            sampled_target = response.iloc[positions]

            self._fit_preprocessor(sampled_features)
            matrix = self._transform(sampled_features)

            raw_target = np.nan_to_num(
                sampled_target.to_numpy(dtype=np.float32)
            )
            target_limit = (
                float(np.nanquantile(np.abs(raw_target), 0.995))
                if raw_target.size
                else 1.0
            )
            if not np.isfinite(target_limit) or target_limit <= 0.0:
                target_limit = 1.0
            raw_target = np.clip(
                raw_target, -target_limit, target_limit
            ).astype(np.float32)
            self.prediction_clip_ = float(
                np.clip(3.0 * target_limit, 1e-6, 10.0)
            )

            if isinstance(sampled_target.index, pd.MultiIndex):
                ranked_target = sampled_target.groupby(level=0).rank(
                    method="average", pct=True
                )
            else:
                ranked_target = sampled_target.rank(
                    method="average", pct=True
                )
            rank_target = np.nan_to_num(
                ((ranked_target - 0.5) * 2.0).to_numpy(
                    dtype=np.float32
                )
            )

            weights = self._weights(time_codes, total_times)
            started = time.perf_counter()

            self.ridge_intercept_ = float(
                np.average(raw_target, weights=weights)
            )
            self.ridge_coef_ = self._ridge(
                matrix,
                raw_target - self.ridge_intercept_,
                weights,
                self.RIDGE_ALPHA,
            )

            self.rank_intercept_ = float(
                np.average(rank_target, weights=weights)
            )
            self.rank_coef_ = self._ridge(
                matrix,
                rank_target - self.rank_intercept_,
                weights,
                self.RANK_ALPHA,
            )

            raw_std = float(np.std(raw_target))
            rank_std = float(np.std(rank_target))
            self.rank_scale_ = (
                raw_std / rank_std
                if raw_std > 1e-8 and rank_std > 1e-8
                else 1.0
            )

            self.fit_time_ = time.perf_counter() - started
            self.training_rows_ = len(sampled_features)
            self.training_times_ = (
                len(
                    pd.Index(
                        sampled_features.index.get_level_values(0)
                    ).unique()
                )
                if isinstance(sampled_features.index, pd.MultiIndex)
                else len(sampled_features)
            )
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

            started = time.perf_counter()
            panel, assets = self._prediction_features(features)
            self.predict_feature_time_ = time.perf_counter() - started

            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback(
                    features, assets or self.assets_
                )

            matrix_frame = self._long(panel)
            if matrix_frame.empty:
                self.fallback_used_ = True
                return self._fallback(
                    features, assets or self.assets_
                )

            started = time.perf_counter()
            matrix = self._transform(matrix_frame)

            ridge = (
                self.ridge_intercept_ + matrix @ self.ridge_coef_
            )
            rank = self.rank_scale_ * (
                self.rank_intercept_ + matrix @ self.rank_coef_
            )
            raw = self.RIDGE_WEIGHT * ridge + self.RANK_WEIGHT * rank

            series = pd.Series(
                np.nan_to_num(raw),
                index=matrix_frame.index,
                dtype=np.float32,
            )
            prediction = series.unstack(level=-1).reindex(
                index=features.index, columns=assets
            ).fillna(0.0)

            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(prediction)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
