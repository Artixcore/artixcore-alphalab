import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.5, stability-selected residual ensemble."""

    _ALPHA = 8.0
    _MID_DECAY = 0.20
    _SLOW_DECAY = 0.45
    _RANK_DECAY = 0.30

    _BLEND_PRESETS = (
        (0.70, 0.20, 0.10, 0.10, 0.00),
        (0.62, 0.23, 0.15, 0.15, 0.05),
        (0.58, 0.22, 0.20, 0.15, 0.08),
        (0.68, 0.17, 0.15, 0.22, 0.05),
    )

    _FEATURE_PRIORITY = (
        "Feature.1__raw",
        "Feature.2__raw",
        "Feature.3__raw",
        "Feature.4__raw",
        "Feature.5__raw",
        "Feature.6__raw",
        "Feature.1__rank",
        "Feature.2__rank",
        "Feature.3__rank",
        "Feature.4__rank",
        "interaction__rank_mean",
        "interaction__rank_dispersion",
        "interaction__rank_spread_12",
        "interaction__rank_spread_13",
        "Feature.1__diff1",
        "Feature.2__diff1",
        "Feature.3__diff1",
        "Feature.1__ma5",
        "Feature.2__ma5",
        "Feature.3__ma5",
        "Feature.1__ma20",
        "Feature.2__ma20",
        "Feature.3__ma20",
        "Feature.1__sd20",
        "Feature.2__sd20",
        "Feature.3__sd20",
        "Feature.1__rankchg",
        "Feature.2__rankchg",
        "Feature.3__rankchg",
        "Feature.1__voladj",
        "Feature.2__voladj",
        "Feature.3__voladj",
        "Feature.1__ma60",
        "Feature.1__ewma5",
        "Feature.1__rollz",
        "Feature.1__momspread",
        "interaction__raw_rank",
        "interaction__rank_accel",
    )

    _XGB_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.045,
        "subsample": 0.82,
        "colsample_bytree": 0.78,
        "min_child_weight": 240,
        "reg_alpha": 0.15,
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

        self.max_train_rows = 85_000
        self.max_validation_rows = 30_000
        self.max_features = 38
        self.n_xgb_rounds = 16
        self.validation_xgb_rounds = 8

        self.selected_features_ = None
        self.impute_ = None
        self.low_ = None
        self.high_ = None
        self.mean_ = None
        self.scale_ = None

        self.coef_mid_ = None
        self.coef_slow_ = None
        self.coef_rank_ = None
        self.intercept_mid_ = 0.0
        self.intercept_slow_ = 0.0
        self.intercept_rank_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None

        self.model_weights_ = {
            "mid": 0.70,
            "slow": 0.20,
            "rank": 0.10,
            "residual": 0.10,
            "rank_blend": 0.00,
        }
        self.validation_score_ = None
        self.validation_metrics_ = None

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
            len(pd.Index(columns.get_level_values(i)).unique())
            for i in range(columns.nlevels)
        ]
        feature_level = next(
            (i for i, name in enumerate(names) if "feature" in name or "factor" in name),
            None,
        )
        asset_level = next(
            (
                i
                for i, name in enumerate(names)
                if "ticker" in name or "asset" in name or "symbol" in name
            ),
            None,
        )
        if feature_level is None:
            feature_level = int(np.argmin(counts))
        if asset_level is None:
            remaining = [i for i in range(columns.nlevels) if i != feature_level]
            asset_level = (
                max(remaining, key=lambda i: counts[i])
                if remaining
                else feature_level
            )
        return feature_level, asset_level

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if not isinstance(features.columns, pd.MultiIndex):
            frame = (
                features.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )
            return {"Feature.1": frame}, list(frame.columns)

        feature_level, asset_level = self._levels(features.columns)
        names = list(dict.fromkeys(features.columns.get_level_values(feature_level)))
        assets = list(dict.fromkeys(features.columns.get_level_values(asset_level)))
        frames = {}

        for name in names:
            columns = [col for col in features.columns if col[feature_level] == name]
            if not columns:
                continue
            block = features.loc[:, columns].copy()
            block.columns = [col[asset_level] for col in columns]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()]
            block = block.reindex(columns=assets)
            block = (
                block.apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )
            frames[str(name)] = block

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

    def _make_features(self, features):
        frames, assets = self._extract(features)
        blocks = []
        ranks = []
        rank_changes = []

        for index, (name, raw) in enumerate(frames.items()):
            blocks.append(self._block(f"{name}__raw", raw))
            if index >= 4:
                continue

            rank = self._rank(raw)
            ranks.append(rank)
            blocks.append(self._block(f"{name}__rank", rank))

            if index >= 3:
                continue

            diff1 = raw - raw.shift(1)
            ma5 = raw.rolling(5, min_periods=2).mean()
            ma20 = raw.rolling(20, min_periods=3).mean()
            sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
            rank_change = rank - rank.shift(1)
            rank_changes.append(rank_change)

            blocks.extend(
                [
                    self._block(f"{name}__diff1", diff1),
                    self._block(f"{name}__ma5", ma5),
                    self._block(f"{name}__ma20", ma20),
                    self._block(f"{name}__sd20", sd20),
                    self._block(f"{name}__rankchg", rank_change),
                    self._block(f"{name}__voladj", diff1 / (sd20 + 1.0e-6)),
                ]
            )

            if index == 0:
                ma60 = raw.rolling(60, min_periods=5).mean()
                blocks.extend(
                    [
                        self._block(f"{name}__ma60", ma60),
                        self._block(
                            f"{name}__ewma5",
                            raw.ewm(span=5, adjust=False, min_periods=2).mean(),
                        ),
                        self._block(
                            f"{name}__rollz", (raw - ma5) / (sd20 + 1.0e-6)
                        ),
                        self._block(f"{name}__momspread", ma5 - ma60),
                    ]
                )

        if ranks:
            rank_mean = sum(ranks) / float(len(ranks))
            rank_square_mean = sum(rank * rank for rank in ranks) / float(len(ranks))
            rank_dispersion = np.sqrt(
                (rank_square_mean - rank_mean * rank_mean).clip(lower=0.0)
            )
            blocks.append(self._block("interaction__rank_mean", rank_mean))
            blocks.append(self._block("interaction__rank_dispersion", rank_dispersion))

            raw_frames = list(frames.values())[: len(ranks)]
            raw_rank = sum(raw * rank for raw, rank in zip(raw_frames, ranks)) / float(
                len(ranks)
            )
            blocks.append(self._block("interaction__raw_rank", raw_rank))

        if len(ranks) >= 2:
            blocks.append(self._block("interaction__rank_spread_12", ranks[0] - ranks[1]))
        if len(ranks) >= 3:
            blocks.append(self._block("interaction__rank_spread_13", ranks[0] - ranks[2]))

        if rank_changes:
            blocks.append(
                self._block(
                    "interaction__rank_accel",
                    sum(rank_changes) / float(len(rank_changes)),
                )
            )

        if not blocks:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays(
                [[], []], names=["feature", "asset"]
            )
            return empty, assets

        panel = pd.concat(blocks, axis=1)
        feature_names = panel.columns.get_level_values("feature").unique()
        panel = panel.reindex(
            columns=pd.MultiIndex.from_product(
                [feature_names, assets], names=["feature", "asset"]
            )
        )
        return panel.replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

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

        return (
            frame.reindex(index=index, columns=assets)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )

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

    def _select_features(self, x):
        if len(x) <= 40_000:
            probe = x
        else:
            positions = np.linspace(0, len(x) - 1, 40_000, dtype=np.int64)
            probe = x.iloc[positions]

        values = probe.to_numpy(dtype=np.float32, copy=False)
        keep = []
        for column_index, column in enumerate(probe.columns):
            values_column = values[:, column_index]
            finite = np.isfinite(values_column)
            if finite.mean() < 0.05 or not finite.any():
                continue
            if np.nanstd(values_column[finite]) < 1.0e-8:
                continue
            keep.append(column)

        priority = {name: index for index, name in enumerate(self._FEATURE_PRIORITY)}
        keep.sort(key=lambda name: priority.get(name, len(priority)))
        return keep[: self.max_features]

    def _sample_positions(self, n_rows, limit=None):
        limit = int(limit or self.max_train_rows)
        if n_rows <= limit:
            return np.arange(n_rows, dtype=np.int64)

        recent = int(limit * 0.65)
        older = limit - recent
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
        self.scale_ = values.std(axis=0).astype(np.float32)
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

    def _weights(self, positions, total, decay):
        if total <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total - 1) - positions.astype(np.float32)
        weights = np.exp(-ages / max(1.0, total * decay))
        return (weights / weights.mean()).astype(np.float32)

    def _ridge(self, x, y, alpha, weights):
        root = np.sqrt(weights).astype(np.float32)
        weighted_x = x * root[:, None]
        weighted_y = y * root
        gram = weighted_x.T @ weighted_x
        gram.flat[:: gram.shape[0] + 1] += alpha
        rhs = weighted_x.T @ weighted_y
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _fit_models(self, matrix, y_series, positions, total_rows, xgb_rounds):
        target = np.nan_to_num(
            y_series.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        mid_weights = self._weights(positions, total_rows, self._MID_DECAY)
        slow_weights = self._weights(positions, total_rows, self._SLOW_DECAY)
        rank_weights = self._weights(positions, total_rows, self._RANK_DECAY)

        intercept_mid = float(target.mean()) if target.size else 0.0
        intercept_slow = intercept_mid
        centered = target - intercept_mid

        coef_mid = self._ridge(matrix, centered, self._ALPHA, mid_weights)
        coef_slow = self._ridge(matrix, centered, self._ALPHA, slow_weights)

        ranked = (
            y_series.groupby(level=0).rank(method="average", pct=True)
            if isinstance(y_series.index, pd.MultiIndex)
            else y_series.rank(method="average", pct=True)
        )
        rank_target = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
        intercept_rank = float(rank_target.mean()) if rank_target.size else 0.0
        coef_rank = self._ridge(
            matrix,
            rank_target - intercept_rank,
            self._ALPHA * 2.0,
            rank_weights,
        )

        target_std = float(np.nanstd(target)) if target.size else 1.0
        rank_std = float(np.nanstd(rank_target)) if rank_target.size else 1.0
        if not np.isfinite(target_std) or target_std < 1.0e-8:
            target_std = 1.0
        if not np.isfinite(rank_std) or rank_std < 1.0e-8:
            rank_std = 1.0
        rank_scale = target_std / rank_std

        mid_prediction = intercept_mid + matrix @ coef_mid
        slow_prediction = intercept_slow + matrix @ coef_slow
        residual = target - (0.72 * mid_prediction + 0.28 * slow_prediction)
        residual_weights = (0.65 * mid_weights + 0.35 * slow_weights).astype(np.float32)
        dtrain = xgb.DMatrix(matrix, label=residual, weight=residual_weights)
        residual_model = xgb.train(
            self._XGB_PARAMS,
            dtrain,
            num_boost_round=int(xgb_rounds),
        )

        return {
            "coef_mid": coef_mid,
            "coef_slow": coef_slow,
            "coef_rank": coef_rank,
            "intercept_mid": intercept_mid,
            "intercept_slow": intercept_slow,
            "intercept_rank": intercept_rank,
            "rank_scale": rank_scale,
            "xgb": residual_model,
        }

    def _predict_components(self, matrix, models):
        mid = models["intercept_mid"] + matrix @ models["coef_mid"]
        slow = models["intercept_slow"] + matrix @ models["coef_slow"]
        rank = models["rank_scale"] * (
            models["intercept_rank"] + matrix @ models["coef_rank"]
        )
        residual = models["xgb"].predict(xgb.DMatrix(matrix)).astype(np.float32)
        return mid, slow, rank, residual

    def _to_matrix(self, values, index, assets):
        series = pd.Series(values, index=index, dtype=np.float32)
        if isinstance(series.index, pd.MultiIndex):
            output = series.unstack(level=-1)
        else:
            output = series.to_frame()
        return output.reindex(columns=assets).fillna(0.0).astype(np.float32)

    def _blend_components(self, components, index, assets, preset):
        mid, slow, rank, residual = components
        mid_weight, slow_weight, rank_weight, residual_weight, rank_blend = preset
        base_total = mid_weight + slow_weight + rank_weight
        if base_total <= 0:
            base_total = 1.0
        raw = (
            mid_weight * mid + slow_weight * slow + rank_weight * rank
        ) / base_total
        raw = raw + residual_weight * residual

        frame = self._to_matrix(raw, index, assets)
        frame = frame.sub(frame.mean(axis=1), axis=0).fillna(0.0)
        if rank_blend > 0:
            rank_frame = frame.rank(axis=1, method="average", pct=True).sub(0.5)
            frame = (1.0 - rank_blend) * frame + rank_blend * rank_frame
            frame = frame.sub(frame.mean(axis=1), axis=0).fillna(0.0)
        return frame.astype(np.float32)

    def _validation_score(self, prediction, target):
        target = target.reindex(index=prediction.index, columns=prediction.columns)
        correlations = []
        rank_correlations = []

        for row in range(len(prediction)):
            predicted_values = prediction.iloc[row].to_numpy(dtype=np.float64)
            target_values = target.iloc[row].to_numpy(dtype=np.float64)
            valid = np.isfinite(predicted_values) & np.isfinite(target_values)
            if valid.sum() < 3:
                continue
            predicted_valid = predicted_values[valid]
            target_valid = target_values[valid]
            if np.std(predicted_valid) < 1.0e-12 or np.std(target_valid) < 1.0e-12:
                continue
            correlation = np.corrcoef(predicted_valid, target_valid)[0, 1]
            if np.isfinite(correlation):
                correlations.append(correlation)

            predicted_rank = pd.Series(predicted_valid).rank(method="average").to_numpy()
            target_rank = pd.Series(target_valid).rank(method="average").to_numpy()
            rank_correlation = np.corrcoef(predicted_rank, target_rank)[0, 1]
            if np.isfinite(rank_correlation):
                rank_correlations.append(rank_correlation)

        if not correlations:
            return -np.inf, {}

        correlations_array = np.asarray(correlations, dtype=np.float64)
        ic_mean = float(correlations_array.mean())
        ic_std = float(correlations_array.std())
        rank_ic = float(np.mean(rank_correlations)) if rank_correlations else 0.0
        positive_rate = float(np.mean(correlations_array > 0.0))

        portfolio = (prediction.shift(1) * target).sum(axis=1).iloc[1:]
        portfolio = portfolio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(portfolio) >= 5 and float(portfolio.std()) > 1.0e-12:
            lag_sharpe = float(portfolio.mean() / portfolio.std())
        else:
            lag_sharpe = -1.0

        block_means = []
        if len(correlations_array) >= 4:
            for block in np.array_split(correlations_array, 4):
                if len(block):
                    block_means.append(float(block.mean()))
        worst_block = min(block_means) if block_means else ic_mean
        block_std = float(np.std(block_means)) if len(block_means) > 1 else 0.0

        score = (
            0.42 * ic_mean
            + 0.18 * rank_ic
            + 0.25 * lag_sharpe
            + 0.10 * positive_rate
            + 0.12 * worst_block
            - 0.16 * ic_std
            - 0.10 * block_std
        )
        metrics = {
            "ic": ic_mean,
            "rank_ic": rank_ic,
            "lag_sharpe": lag_sharpe,
            "positive_rate": positive_rate,
            "worst_block": worst_block,
            "ic_std": ic_std,
        }
        return float(score), metrics

    def _choose_blend(self, x, y, assets):
        if not isinstance(x.index, pd.MultiIndex):
            return self._BLEND_PRESETS[0], None, None

        timestamps = pd.Index(x.index.get_level_values(0).unique())
        if len(timestamps) < 30:
            return self._BLEND_PRESETS[0], None, None

        split = max(1, min(len(timestamps) - 1, int(len(timestamps) * 0.82)))
        early_timestamps = set(timestamps[:split])
        late_timestamps = set(timestamps[split:])
        early_mask = x.index.get_level_values(0).isin(early_timestamps)
        late_mask = x.index.get_level_values(0).isin(late_timestamps)
        early_x, early_y = x.loc[early_mask], y.loc[early_mask]
        late_x, late_y = x.loc[late_mask], y.loc[late_mask]

        if len(early_x) < 100 or len(late_x) < 100:
            return self._BLEND_PRESETS[0], None, None

        early_positions = self._sample_positions(
            len(early_x), min(45_000, self.max_train_rows)
        )
        early_sample = early_x.iloc[early_positions][self.selected_features_]
        early_target = early_y.iloc[early_positions]

        late_limit = min(len(late_x), self.max_validation_rows)
        late_positions = np.linspace(0, len(late_x) - 1, late_limit, dtype=np.int64)
        late_sample = late_x.iloc[late_positions][self.selected_features_]
        late_target = late_y.iloc[late_positions]

        self._fit_preprocessor(early_sample)
        early_matrix = self._transform(early_sample)
        validation_models = self._fit_models(
            early_matrix,
            early_target,
            early_positions,
            len(early_x),
            self.validation_xgb_rounds,
        )
        late_matrix = self._transform(late_sample)
        components = self._predict_components(late_matrix, validation_models)

        validation_target = self._to_matrix(
            late_target.to_numpy(dtype=np.float32), late_target.index, assets
        )

        best_preset = self._BLEND_PRESETS[0]
        best_score = -np.inf
        best_metrics = None
        for preset in self._BLEND_PRESETS:
            prediction = self._blend_components(
                components, late_target.index, assets, preset
            )
            score, metrics = self._validation_score(prediction, validation_target)
            if score > best_score:
                best_score = score
                best_preset = preset
                best_metrics = metrics

        return best_preset, best_score, best_metrics

    def _finish(self, prediction):
        prediction = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prediction = prediction.clip(-self.prediction_clip_, self.prediction_clip_)
        prediction = prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0)
        return prediction.astype(np.float32)

    def _fallback(self, features):
        try:
            frames, assets = self._extract(features)
            first = next(iter(frames.values()))
            prediction = self._rank(first)
            smoothed = self._rank(first.rolling(5, min_periods=2).mean())
            prediction = 0.75 * prediction + 0.25 * smoothed
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
        self.xgb_model_ = None

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            started = time.perf_counter()
            panel, assets = self._make_features(features)
            self.feature_time_ = time.perf_counter() - started
            x, y, assets = self._long(panel, target)
            del panel

            if x.empty or len(y) < 40:
                return self

            self.selected_features_ = self._select_features(x)
            if not self.selected_features_:
                return self

            preset, score, metrics = self._choose_blend(x, y, assets)
            self.model_weights_ = {
                "mid": preset[0],
                "slow": preset[1],
                "rank": preset[2],
                "residual": preset[3],
                "rank_blend": preset[4],
            }
            self.validation_score_ = score
            self.validation_metrics_ = metrics

            positions = self._sample_positions(len(x))
            sampled_x = x.iloc[positions][self.selected_features_]
            sampled_y = y.iloc[positions]

            self._fit_preprocessor(sampled_x)
            matrix = self._transform(sampled_x)

            target_values = np.nan_to_num(
                sampled_y.to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target_abs = np.abs(target_values)
            quantile = np.nanquantile(target_abs, 0.995) if target_abs.size else 1.0
            self.prediction_clip_ = float(
                np.clip(
                    3.0 * quantile
                    if np.isfinite(quantile) and quantile > 0
                    else 1.0,
                    1.0e-6,
                    10.0,
                )
            )

            started = time.perf_counter()
            models = self._fit_models(
                matrix,
                sampled_y,
                positions,
                len(x),
                self.n_xgb_rounds,
            )
            self.coef_mid_ = models["coef_mid"]
            self.coef_slow_ = models["coef_slow"]
            self.coef_rank_ = models["coef_rank"]
            self.intercept_mid_ = models["intercept_mid"]
            self.intercept_slow_ = models["intercept_slow"]
            self.intercept_rank_ = models["intercept_rank"]
            self.rank_scale_ = models["rank_scale"]
            self.xgb_model_ = models["xgb"]
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
            panel, assets = self._make_features(features)
            self.predict_feature_time_ = time.perf_counter() - started
            x, _, assets = self._long(panel)
            del panel

            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)

            started = time.perf_counter()
            matrix = self._transform(x)
            models = {
                "coef_mid": self.coef_mid_,
                "coef_slow": self.coef_slow_,
                "coef_rank": self.coef_rank_,
                "intercept_mid": self.intercept_mid_,
                "intercept_slow": self.intercept_slow_,
                "intercept_rank": self.intercept_rank_,
                "rank_scale": self.rank_scale_,
                "xgb": self.xgb_model_,
            }
            components = self._predict_components(matrix, models)
            preset = (
                self.model_weights_["mid"],
                self.model_weights_["slow"],
                self.model_weights_["rank"],
                self.model_weights_["residual"],
                self.model_weights_["rank_blend"],
            )
            prediction = self._blend_components(components, x.index, assets, preset)
            prediction = prediction.reindex(
                index=features.index, columns=assets
            ).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(prediction)

        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
