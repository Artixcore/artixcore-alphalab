import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """AlphaLab v0.42: protected chronological residual stack on the v33 base."""

    RIDGE_ALPHA = 12.0
    RANK_ALPHA = 28.0
    RESIDUAL_ALPHA = 24.0
    DECAY = 0.30
    RIDGE_WEIGHT = 0.62
    RANK_WEIGHT = 0.23
    TREE_WEIGHT = 0.15
    MAX_RESIDUAL_WEIGHT = 0.30

    FEATURE_PRIORITY = (
        "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
        "Feature.4__raw", "Feature.5__raw", "Feature.6__raw",
        "Feature.1__rank", "Feature.2__rank", "Feature.3__rank",
        "Feature.4__rank", "Feature.5__rank", "Feature.6__rank",
        "Feature.1__lag1", "Feature.1__lag3",
        "Feature.2__lag1", "Feature.3__lag1",
        "Feature.1__diff1", "Feature.1__diff3",
        "Feature.2__diff1", "Feature.3__diff1",
        "Feature.1__ma5", "Feature.1__ma20",
        "Feature.1__sd20", "Feature.1__ewma5",
        "Feature.1__roll_z", "Feature.1__mom_spread",
        "Feature.1__demean", "interaction__rank12",
        "interaction__rank13",
    )

    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.04,
        "min_child_weight": 280,
        "reg_alpha": 0.08,
        "reg_lambda": 2.5,
        "subsample": 0.82,
        "colsample_bytree": 0.70,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": 2,
        "seed": 3301,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows = 80000
        self.max_features = 29
        self.history_rows = 64
        self.selected_features_ = None
        self.assets_ = []
        self.history_tail_ = None

        self.impute_ = self.low_ = self.high_ = None
        self.center_ = self.scale_ = None
        self.ridge_coef_ = self.rank_coef_ = None
        self.ridge_intercept_ = self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.tree_model_ = None

        self.residual_columns_ = None
        self.residual_impute_ = self.residual_low_ = self.residual_high_ = None
        self.residual_center_ = self.residual_scale_values_ = None
        self.residual_coef_ = None
        self.residual_intercept_ = 0.0
        self.residual_weight_ = 0.0
        self.residual_ic_ = 0.0
        self.residual_positive_fraction_ = 0.0
        self.residual_improvement_ = 0.0
        self.residual_enabled_ = False

        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.training_rows_ = self.training_times_ = self.feature_count_ = 0
        self.feature_time_ = self.fit_time_ = self.residual_fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    @staticmethod
    def _numeric(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype(np.float32)

    @staticmethod
    def _rank(frame):
        n = max(frame.shape[1], 1)
        denom = 0.5 * max(n - 1, 1)
        return ((frame.rank(axis=1, method="average") - 0.5 * (n + 1)) / denom).astype(np.float32)

    @staticmethod
    def _block(name, frame):
        out = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        out.columns = pd.MultiIndex.from_product(
            [[name], out.columns], names=["feature", "asset"]
        )
        return out

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
        names = [str(value).lower() if value is not None else "" for value in columns.names]
        counts = [len(pd.Index(columns.get_level_values(i)).unique()) for i in range(columns.nlevels)]
        feature_level = next(
            (i for i, name in enumerate(names) if "feature" in name or "factor" in name), None
        )
        asset_level = next(
            (i for i, name in enumerate(names) if "asset" in name or "ticker" in name or "symbol" in name), None
        )
        if feature_level is None:
            feature_level = int(np.argmin(counts))
        if asset_level is None:
            remaining = [i for i in range(columns.nlevels) if i != feature_level]
            asset_level = max(remaining, key=lambda i: counts[i]) if remaining else feature_level
        return feature_level, asset_level

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            numeric = self._numeric(features)
            return {"Feature.1": numeric}, list(numeric.columns)
        feature_level, asset_level = self._levels(features.columns)
        assets = list(dict.fromkeys(features.columns.get_level_values(asset_level)))
        frames = {}
        for name in dict.fromkeys(features.columns.get_level_values(feature_level)):
            columns = [c for c in features.columns if c[feature_level] == name]
            block = features.loc[:, columns].copy()
            block.columns = [c[asset_level] for c in columns]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()].reindex(columns=assets)
            frames[str(name)] = self._numeric(block)
        return frames, assets

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty, assets
        names = list(frames)
        ranks = {name: self._rank(frames[name]) for name in names}
        blocks = []
        for i, name in enumerate(names):
            raw = frames[name]
            blocks.append(self._block(name + "__raw", raw))
            blocks.append(self._block(name + "__rank", ranks[name]))
            if i < 3:
                blocks.extend([
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__diff1", raw.diff(1)),
                ])
            if i == 0:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma20 = raw.rolling(20, min_periods=4).mean()
                sd20 = raw.rolling(20, min_periods=4).std(ddof=0)
                ewma5 = raw.ewm(span=5, adjust=False, min_periods=2).mean()
                blocks.extend([
                    self._block(name + "__lag3", raw.shift(3)),
                    self._block(name + "__diff3", raw.diff(3)),
                    self._block(name + "__ma5", ma5),
                    self._block(name + "__ma20", ma20),
                    self._block(name + "__sd20", sd20),
                    self._block(name + "__ewma5", ewma5),
                    self._block(name + "__roll_z", (raw - ma5) / (sd20 + 1e-6)),
                    self._block(name + "__mom_spread", ma5 - ma20),
                    self._block(name + "__demean", raw.sub(raw.median(axis=1), axis=0)),
                ])
        if len(names) >= 2:
            blocks.append(self._block("interaction__rank12", ranks[names[0]] - ranks[names[1]]))
        if len(names) >= 3:
            blocks.append(self._block("interaction__rank13", ranks[names[0]] - ranks[names[2]]))
        panel = pd.concat(blocks, axis=1)
        feature_names = panel.columns.get_level_values("feature").unique()
        columns = pd.MultiIndex.from_product([feature_names, assets], names=["feature", "asset"])
        return panel.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

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
            frame = target.unstack(-1) if isinstance(target.index, pd.MultiIndex) else target.to_frame()
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
        x = self._stack(panel, "asset").replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x
        y = self._stack(self._target(target, panel.index, assets))
        x, y = x.align(y, join="inner", axis=0)
        valid = y.replace([np.inf, -np.inf], np.nan).notna()
        return x.loc[valid], y.loc[valid].astype(np.float32)

    def _select(self, x):
        if x.empty:
            return []
        probe = x if len(x) <= 40000 else x.iloc[
            np.linspace(0, len(x) - 1, 40000, dtype=np.int64)
        ]
        values = probe.to_numpy(np.float32, copy=False)
        usable = []
        for i, name in enumerate(probe.columns):
            finite = np.isfinite(values[:, i])
            if finite.mean() >= 0.05 and finite.any() and np.nanstd(values[finite, i]) >= 1e-8:
                usable.append(name)
        priority = {name: i for i, name in enumerate(self.FEATURE_PRIORITY)}
        usable.sort(key=lambda name: priority.get(name, len(priority)))
        return usable[: self.max_features]

    def _sample_times(self, index):
        if not isinstance(index, pd.MultiIndex):
            positions = np.arange(len(index), dtype=np.int64)
            if len(positions) > self.max_train_rows:
                positions = positions[-self.max_train_rows :]
            return positions, positions, len(index)
        codes, times = pd.factorize(index.get_level_values(0), sort=False)
        total = len(times)
        sizes = np.bincount(codes, minlength=max(total, 1))
        typical = max(1, int(np.median(sizes[sizes > 0])))
        max_times = max(1, self.max_train_rows // typical)
        if total <= max_times:
            chosen = np.arange(total, dtype=np.int64)
        else:
            recent = max(1, int(max_times * 0.55))
            older = max_times - recent
            start = total - recent
            head = np.linspace(0, start - 1, older, dtype=np.int64) if older else np.empty(0, np.int64)
            chosen = np.unique(np.r_[head, np.arange(start, total)])
        positions = np.flatnonzero(np.isin(codes, chosen)).astype(np.int64)
        return positions, codes[positions].astype(np.int64), total

    @staticmethod
    def _fit_stats(frame):
        values = frame.to_numpy(np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        impute = np.nanmedian(values, axis=0).astype(np.float32)
        impute[~np.isfinite(impute)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(impute, np.where(missing)[1])
        low = np.nanquantile(values, 0.01, axis=0).astype(np.float32)
        high = np.nanquantile(values, 0.99, axis=0).astype(np.float32)
        bad = ~np.isfinite(low) | ~np.isfinite(high) | (low >= high)
        low[bad], high[bad] = -10.0, 10.0
        values = np.clip(values, low, high)
        center = np.nanmedian(values, axis=0).astype(np.float32)
        scale = (1.4826 * np.nanmedian(np.abs(values - center), axis=0)).astype(np.float32)
        fallback = np.nanstd(values, axis=0).astype(np.float32)
        bad = ~np.isfinite(scale) | (scale < 1e-8)
        scale[bad] = fallback[bad]
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return impute, low, high, center, scale

    @staticmethod
    def _apply_stats(frame, columns, stats):
        impute, low, high, center, scale = stats
        values = frame.reindex(columns=columns).to_numpy(np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(impute, np.where(missing)[1])
        values = (np.clip(values, low, high) - center) / scale
        return np.nan_to_num(values).astype(np.float32)

    def _fit_preprocessor(self, x):
        stats = self._fit_stats(x)
        self.impute_, self.low_, self.high_, self.center_, self.scale_ = stats

    def _transform(self, x):
        stats = (self.impute_, self.low_, self.high_, self.center_, self.scale_)
        return self._apply_stats(x, self.selected_features_, stats)

    @staticmethod
    def _weights(codes, total):
        if len(codes) == 0 or total <= 1:
            return np.ones(len(codes), np.float32)
        ages = (total - 1) - codes.astype(np.float32)
        weights = np.exp(-ages / max(1.0, total * ArtixcoreAlphaLabPredictor.DECAY)).astype(np.float32)
        return weights / max(float(weights.mean()), 1e-8)

    @staticmethod
    def _ridge(matrix, target, weights, alpha):
        root = np.sqrt(np.maximum(weights, 1e-8)).astype(np.float32)
        xw, yw = matrix * root[:, None], target * root
        gram, rhs = xw.T @ xw, xw.T @ yw
        gram.flat[:: gram.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    @staticmethod
    def _rank_target(series):
        if isinstance(series.index, pd.MultiIndex):
            ranked = series.groupby(level=0).rank(method="average", pct=True)
        else:
            ranked = series.rank(method="average", pct=True)
        return np.nan_to_num(((ranked - 0.5) * 2.0).to_numpy(np.float32))

    def _fit_base_bundle(self, matrix, raw, rank, weights):
        ridge_intercept = float(np.average(raw, weights=weights))
        ridge_coef = self._ridge(matrix, raw - ridge_intercept, weights, self.RIDGE_ALPHA)
        rank_intercept = float(np.average(rank, weights=weights))
        rank_coef = self._ridge(matrix, rank - rank_intercept, weights, self.RANK_ALPHA)
        raw_std = float(np.std(raw))
        rank_std = float(np.std(rank))
        rank_scale = raw_std / rank_std if raw_std > 1e-8 and rank_std > 1e-8 else 1.0
        tree = xgb.train(
            dict(self.XGB_PARAMS),
            xgb.DMatrix(matrix, label=raw, weight=weights),
            num_boost_round=9,
        )
        return ridge_intercept, ridge_coef, rank_intercept, rank_coef, rank_scale, tree

    def _predict_base_bundle(self, bundle, matrix):
        ridge_intercept, ridge_coef, rank_intercept, rank_coef, rank_scale, tree = bundle
        ridge = ridge_intercept + matrix @ ridge_coef
        rank = rank_scale * (rank_intercept + matrix @ rank_coef)
        tree_prediction = tree.predict(xgb.DMatrix(matrix)).astype(np.float32)
        return (
            self.RIDGE_WEIGHT * ridge
            + self.RANK_WEIGHT * rank
            + self.TREE_WEIGHT * tree_prediction
        ).astype(np.float32)

    def _residual_features(self, x):
        index = x.index

        def column(name):
            if name in x.columns:
                return x[name].astype(np.float32)
            return pd.Series(0.0, index=index, dtype=np.float32)

        r1 = column("Feature.1__rank")
        r2 = column("Feature.2__rank")
        r3 = column("Feature.3__rank")
        d1 = column("Feature.1__diff1")
        d2 = column("Feature.2__diff1")
        d3 = column("Feature.3__diff1")
        l1 = column("Feature.1__lag1")
        l2 = column("Feature.2__lag1")
        l3 = column("Feature.3__lag1")
        z1 = column("Feature.1__roll_z")
        momentum = column("Feature.1__mom_spread")

        residual = pd.DataFrame(index=index)
        residual["r1"] = r1
        residual["r2"] = r2
        residual["r3"] = r3
        residual["r12"] = r1 - r2
        residual["r13"] = r1 - r3
        residual["r23"] = r2 - r3
        residual["p12"] = r1 * r2
        residual["p13"] = r1 * r3
        residual["p23"] = r2 * r3
        residual["q1"] = r1 * np.abs(r1)
        residual["q2"] = r2 * np.abs(r2)
        residual["q3"] = r3 * np.abs(r3)
        residual["curve"] = (r1 - r2) * (r1 - r3)
        residual["r1d1"] = r1 * d1
        residual["r2d2"] = r2 * d2
        residual["r3d3"] = r3 * d3
        residual["r1l1"] = r1 * l1
        residual["r2l2"] = r2 * l2
        residual["r3l3"] = r3 * l3
        residual["z1"] = z1
        residual["r1z1"] = r1 * z1
        residual["momentum"] = momentum
        residual = residual.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        if isinstance(residual.index, pd.MultiIndex):
            residual = residual.sub(residual.groupby(level=0).transform("mean"))
        return residual.astype(np.float32)

    @staticmethod
    def _orthogonalize(addition, base, codes):
        output = np.asarray(addition, dtype=np.float32).copy()
        base_values = np.asarray(base, dtype=np.float32)
        if len(output) == 0:
            return output
        for code in np.unique(codes):
            positions = np.flatnonzero(codes == code)
            if positions.size == 0:
                continue
            a = output[positions]
            b = base_values[positions]
            a = a - float(np.mean(a))
            denominator = float(np.dot(b, b))
            if denominator > 1e-10:
                a = a - float(np.dot(a, b) / denominator) * b
            output[positions] = a - float(np.mean(a))
        return output.astype(np.float32)

    @staticmethod
    def _cross_sectional_diagnostics(prediction, target, codes):
        correlations = []
        for code in np.unique(codes):
            positions = np.flatnonzero(codes == code)
            if positions.size < 4:
                continue
            p = prediction[positions]
            y = target[positions]
            valid = np.isfinite(p) & np.isfinite(y)
            if valid.sum() < 4:
                continue
            p = p[valid] - float(np.mean(p[valid]))
            y = y[valid] - float(np.mean(y[valid]))
            denominator = float(np.sqrt(np.dot(p, p) * np.dot(y, y)))
            if denominator > 1e-10:
                correlations.append(float(np.dot(p, y) / denominator))
        if not correlations:
            return 0.0, 0.0
        values = np.asarray(correlations, dtype=np.float64)
        return float(np.mean(values)), float(np.mean(values > 0.0))

    def _fit_residual_stack(self, sampled_x, sampled_y, sampled_codes, total_times):
        self.residual_enabled_ = False
        self.residual_weight_ = 0.0
        if not isinstance(sampled_x.index, pd.MultiIndex):
            return
        local_codes, local_times = pd.factorize(sampled_x.index.get_level_values(0), sort=False)
        n_times = len(local_times)
        if n_times < 36:
            return

        base_cut = max(18, int(n_times * 0.62))
        residual_cut = max(base_cut + 8, int(n_times * 0.82))
        residual_cut = min(residual_cut, n_times - 8)
        if residual_cut <= base_cut or residual_cut >= n_times:
            return

        base_mask = local_codes < base_cut
        residual_mask = (local_codes >= base_cut) & (local_codes < residual_cut)
        calibration_mask = local_codes >= residual_cut
        if base_mask.sum() < 120 or residual_mask.sum() < 80 or calibration_mask.sum() < 80:
            return

        columns = list(self.selected_features_)
        base_frame = sampled_x.loc[base_mask, columns]
        temp_stats = self._fit_stats(base_frame)
        base_matrix = self._apply_stats(base_frame, columns, temp_stats)
        raw_all = np.nan_to_num(sampled_y.to_numpy(np.float32))
        early_raw = raw_all[base_mask]
        limit = float(np.nanquantile(np.abs(early_raw), 0.995)) if early_raw.size else 1.0
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        clipped_all = np.clip(raw_all, -limit, limit).astype(np.float32)
        rank_all = self._rank_target(sampled_y)
        base_codes = local_codes[base_mask]
        base_weights = self._weights(base_codes, base_cut)
        temp_bundle = self._fit_base_bundle(
            base_matrix,
            clipped_all[base_mask],
            rank_all[base_mask],
            base_weights,
        )

        oof_mask = residual_mask | calibration_mask
        oof_matrix = self._apply_stats(sampled_x.loc[oof_mask, columns], columns, temp_stats)
        oof_base = self._predict_base_bundle(temp_bundle, oof_matrix)
        oof_target = clipped_all[oof_mask]
        oof_residual_target = oof_target - oof_base
        oof_codes = local_codes[oof_mask]

        residual_frame_all = self._residual_features(sampled_x.loc[oof_mask])
        residual_train_local = oof_codes < residual_cut
        calibration_local = oof_codes >= residual_cut
        residual_train_frame = residual_frame_all.loc[residual_train_local]
        self.residual_columns_ = list(residual_train_frame.columns)
        residual_stats = self._fit_stats(residual_train_frame)
        (
            self.residual_impute_, self.residual_low_, self.residual_high_,
            self.residual_center_, self.residual_scale_values_,
        ) = residual_stats
        residual_train_matrix = self._apply_stats(
            residual_train_frame, self.residual_columns_, residual_stats
        )
        residual_train_target = oof_residual_target[residual_train_local]
        residual_train_codes = oof_codes[residual_train_local] - base_cut
        residual_train_total = max(1, residual_cut - base_cut)
        residual_weights = self._weights(residual_train_codes, residual_train_total)
        self.residual_intercept_ = float(
            np.average(residual_train_target, weights=residual_weights)
        )
        self.residual_coef_ = self._ridge(
            residual_train_matrix,
            residual_train_target - self.residual_intercept_,
            residual_weights,
            self.RESIDUAL_ALPHA,
        )

        calibration_frame = residual_frame_all.loc[calibration_local]
        calibration_matrix = self._apply_stats(
            calibration_frame, self.residual_columns_, residual_stats
        )
        calibration_addition = (
            self.residual_intercept_ + calibration_matrix @ self.residual_coef_
        ).astype(np.float32)
        calibration_base = oof_base[calibration_local]
        calibration_codes = oof_codes[calibration_local]
        calibration_addition = self._orthogonalize(
            calibration_addition, calibration_base, calibration_codes
        )
        calibration_target = oof_residual_target[calibration_local]
        calibration_relative_codes = calibration_codes - residual_cut
        calibration_total = max(1, n_times - residual_cut)
        calibration_weights = self._weights(calibration_relative_codes, calibration_total)

        denominator = float(np.sum(calibration_weights * calibration_addition * calibration_addition))
        numerator = float(np.sum(calibration_weights * calibration_addition * calibration_target))
        gamma = numerator / denominator if denominator > 1e-10 else 0.0
        gamma = float(np.clip(gamma, 0.0, self.MAX_RESIDUAL_WEIGHT))
        ic_mean, positive_fraction = self._cross_sectional_diagnostics(
            calibration_addition, calibration_target, calibration_codes
        )
        baseline_error = float(np.average(calibration_target * calibration_target, weights=calibration_weights))
        adjusted_error = float(np.average(
            (calibration_target - gamma * calibration_addition) ** 2,
            weights=calibration_weights,
        ))
        improvement = (
            (baseline_error - adjusted_error) / max(baseline_error, 1e-10)
        )

        self.residual_ic_ = ic_mean
        self.residual_positive_fraction_ = positive_fraction
        self.residual_improvement_ = float(improvement)
        if (
            gamma >= 0.02
            and ic_mean >= 0.005
            and positive_fraction >= 0.50
            and improvement >= 0.001
        ):
            self.residual_weight_ = gamma
            self.residual_enabled_ = True
        else:
            self.residual_weight_ = 0.0
            self.residual_enabled_ = False

    def _finish(self, prediction):
        prediction = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prediction = prediction.clip(-self.prediction_clip_, self.prediction_clip_)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features, assets):
        try:
            panel, extracted = self._prediction_features(features)
            assets = extracted or assets
            names = list(panel.columns.get_level_values("feature").unique())
            name = next((n for n in names if n.endswith("__rank")), names[0])
            return self._finish(panel[name].reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.ridge_coef_ = self.rank_coef_ = self.tree_model_ = None
        self.residual_coef_ = None
        self.residual_enabled_ = False
        self.residual_weight_ = 0.0
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            self.history_tail_ = features.tail(self.history_rows).copy()
            started = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_ = time.perf_counter() - started
            self.assets_ = list(assets)
            x, y = self._long(panel, target, assets)
            if x.empty or len(y) < 60:
                self.training_error_ = "Insufficient usable training observations"
                return self
            self.selected_features_ = self._select(x)
            if not self.selected_features_:
                self.training_error_ = "No usable features selected"
                return self
            positions, codes, total = self._sample_times(x.index)
            sampled_x = x.iloc[positions]
            sampled_y = y.iloc[positions]

            residual_started = time.perf_counter()
            self._fit_residual_stack(sampled_x, sampled_y, codes, total)
            self.residual_fit_time_ = time.perf_counter() - residual_started

            xs = sampled_x.loc[:, self.selected_features_]
            ys = sampled_y
            self._fit_preprocessor(xs)
            matrix = self._transform(xs)
            raw = np.nan_to_num(ys.to_numpy(np.float32))
            limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
            if not np.isfinite(limit) or limit <= 0:
                limit = 1.0
            raw = np.clip(raw, -limit, limit).astype(np.float32)
            self.prediction_clip_ = float(np.clip(3.0 * limit, 1e-6, 10.0))
            rank = self._rank_target(ys)
            weights = self._weights(codes, total)

            started = time.perf_counter()
            bundle = self._fit_base_bundle(matrix, raw, rank, weights)
            (
                self.ridge_intercept_, self.ridge_coef_,
                self.rank_intercept_, self.rank_coef_,
                self.rank_scale_, self.tree_model_,
            ) = bundle
            self.fit_time_ = time.perf_counter() - started
            self.training_rows_ = len(xs)
            self.training_times_ = (
                len(pd.Index(xs.index.get_level_values(0)).unique())
                if isinstance(xs.index, pd.MultiIndex) else len(xs)
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
                return self._fallback(features, assets or self.assets_)
            x = self._long(panel)
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)

            started = time.perf_counter()
            matrix = self._transform(x)
            bundle = (
                self.ridge_intercept_, self.ridge_coef_,
                self.rank_intercept_, self.rank_coef_,
                self.rank_scale_, self.tree_model_,
            )
            base = self._predict_base_bundle(bundle, matrix)
            raw = base.copy()
            if (
                self.residual_enabled_
                and self.residual_coef_ is not None
                and self.residual_columns_
            ):
                residual_frame = self._residual_features(x)
                residual_stats = (
                    self.residual_impute_, self.residual_low_, self.residual_high_,
                    self.residual_center_, self.residual_scale_values_,
                )
                residual_matrix = self._apply_stats(
                    residual_frame, self.residual_columns_, residual_stats
                )
                addition = (
                    self.residual_intercept_ + residual_matrix @ self.residual_coef_
                ).astype(np.float32)
                if isinstance(x.index, pd.MultiIndex):
                    codes, _ = pd.factorize(x.index.get_level_values(0), sort=False)
                else:
                    codes = np.arange(len(x), dtype=np.int64)
                addition = self._orthogonalize(addition, base, codes)
                raw = base + self.residual_weight_ * addition

            series = pd.Series(np.nan_to_num(raw), index=x.index, dtype=np.float32)
            prediction = series.unstack(-1).reindex(index=features.index, columns=assets).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(prediction)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
