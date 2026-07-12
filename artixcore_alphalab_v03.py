import time

import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.3, fast dual-horizon leakage-safe predictor."""

    _ALPHA = 8.0
    _FAST_DECAY = 0.12
    _SLOW_DECAY = 0.45
    _FAST_WEIGHT = 0.35
    _SLOW_WEIGHT = 0.35
    _RANK_WEIGHT = 0.05
    _XGB_WEIGHT = 0.25
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
        "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows = 80000
        self.max_features = 36
        self.n_xgb_rounds = 15
        self.selected_features_ = None
        self.impute_ = None
        self.low_ = None
        self.high_ = None
        self.mean_ = None
        self.scale_ = None
        self.coef_fast_ = None
        self.coef_slow_ = None
        self.coef_rank_ = None
        self.intercept_fast_ = 0.0
        self.intercept_slow_ = 0.0
        self.intercept_rank_ = 0.0
        self.xgb_model_ = None
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
        names = [str(v).lower() if v is not None else "" for v in columns.names]
        counts = [len(pd.Index(columns.get_level_values(i)).unique()) for i in range(columns.nlevels)]
        feature_level = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        asset_level = next((i for i, n in enumerate(names) if "ticker" in n or "asset" in n or "symbol" in n), None)
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
            frame = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            return {"Feature.1": frame.astype(np.float32)}, list(frame.columns)
        feature_level, asset_level = self._levels(features.columns)
        names = list(dict.fromkeys(features.columns.get_level_values(feature_level)))
        assets = list(dict.fromkeys(features.columns.get_level_values(asset_level)))
        frames = {}
        for name in names:
            cols = [c for c in features.columns if c[feature_level] == name]
            block = features.loc[:, cols].copy()
            block.columns = [c[asset_level] for c in cols]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()].reindex(columns=assets)
            block = block.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            frames[str(name)] = block.astype(np.float32)
        return frames, assets

    def _rank(self, frame):
        n = max(frame.shape[1], 1)
        rank = frame.rank(axis=1, method="average")
        return ((rank - 0.5 * (n + 1)) / (0.5 * max(n - 1, 1))).astype(np.float32)

    def _block(self, name, frame):
        out = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        out.columns = pd.MultiIndex.from_product([[name], out.columns], names=["feature", "asset"])
        return out

    def _make_features(self, features):
        frames, assets = self._extract(features)
        blocks = []
        ranks = []
        for i, (name, raw) in enumerate(frames.items()):
            blocks.append(self._block(f"{name}__raw", raw))
            if i >= 3:
                continue
            rank = self._rank(raw)
            ranks.append(rank)
            diff1 = raw - raw.shift(1)
            ma5 = raw.rolling(5, min_periods=2).mean()
            ma20 = raw.rolling(20, min_periods=3).mean()
            sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
            blocks.extend([
                self._block(f"{name}__rank", rank),
                self._block(f"{name}__diff1", diff1),
                self._block(f"{name}__ma5", ma5),
                self._block(f"{name}__ma20", ma20),
                self._block(f"{name}__sd20", sd20),
                self._block(f"{name}__rankchg", rank - rank.shift(1)),
                self._block(f"{name}__voladj", diff1 / (sd20 + 1e-6)),
            ])
            if i == 0:
                ma60 = raw.rolling(60, min_periods=5).mean()
                blocks.extend([
                    self._block(f"{name}__ma60", ma60),
                    self._block(f"{name}__ewma5", raw.ewm(span=5, adjust=False, min_periods=2).mean()),
                    self._block(f"{name}__rollz", (raw - ma5) / (sd20 + 1e-6)),
                    self._block(f"{name}__momspread", ma5 - ma60),
                ])
        if ranks:
            blocks.append(self._block("interaction__rank_mean", sum(ranks) / float(len(ranks))))
        if len(ranks) >= 2:
            blocks.append(self._block("interaction__rank_spread", ranks[0] - ranks[1]))
        if not blocks:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty, assets
        panel = pd.concat(blocks, axis=1)
        names = panel.columns.get_level_values("feature").unique()
        panel = panel.reindex(columns=pd.MultiIndex.from_product([names, assets], names=["feature", "asset"]))
        return panel.replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

    def _target(self, target, index, assets):
        if isinstance(target, pd.Series):
            frame = target.unstack(level=-1) if isinstance(target.index, pd.MultiIndex) else target.to_frame()
        elif isinstance(target, pd.DataFrame):
            frame = target.copy()
        else:
            frame = pd.DataFrame(target, index=index)
        if isinstance(frame.columns, pd.MultiIndex):
            _, asset_level = self._levels(frame.columns)
            frame.columns = frame.columns.get_level_values(asset_level)
        return frame.reindex(index=index, columns=assets).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def _long(self, panel, target=None):
        assets = list(panel.columns.get_level_values("asset").unique())
        x = panel.stack(level="asset", future_stack=True).replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x, None, assets
        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.notna()
        return x.loc[valid], y.loc[valid].astype(np.float32), assets

    def _positions(self, n):
        if n <= self.max_train_rows:
            return np.arange(n, dtype=np.int64)
        recent = int(self.max_train_rows * 0.60)
        start = n - recent
        old = np.linspace(0, start - 1, self.max_train_rows - recent, dtype=np.int64)
        return np.unique(np.concatenate([old, np.arange(start, n, dtype=np.int64)]))

    def _select_features(self, x):
        probe = x if len(x) <= 40000 else x.iloc[np.linspace(0, len(x) - 1, 40000, dtype=np.int64)]
        values = probe.to_numpy(dtype=np.float32, copy=False)
        keep = []
        for j, col in enumerate(probe.columns):
            arr = values[:, j]
            finite = np.isfinite(arr)
            if finite.mean() >= 0.05 and finite.any() and np.nanstd(arr[finite]) >= 1e-8:
                keep.append(col)
        return keep[: self.max_features]

    def _fit_preprocessor(self, x):
        arr = x.to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        self.impute_ = np.nanmedian(arr, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.take(self.impute_, np.where(bad)[1])
        self.low_ = np.nanquantile(arr, 0.005, axis=0).astype(np.float32)
        self.high_ = np.nanquantile(arr, 0.995, axis=0).astype(np.float32)
        invalid = ~np.isfinite(self.low_) | ~np.isfinite(self.high_) | (self.low_ >= self.high_)
        self.low_[invalid], self.high_[invalid] = -10.0, 10.0
        arr = np.clip(arr, self.low_, self.high_)
        self.mean_ = arr.mean(axis=0).astype(np.float32)
        self.scale_ = arr.std(axis=0).astype(np.float32)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        arr = x.reindex(columns=self.selected_features_).to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.take(self.impute_, np.where(bad)[1])
        arr = np.clip(arr, self.low_, self.high_)
        return np.nan_to_num((arr - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0)

    def _weights(self, positions, total, decay):
        if total <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total - 1) - positions.astype(np.float32)
        w = np.exp(-ages / max(1.0, total * decay))
        return (w / w.mean()).astype(np.float32)

    def _ridge(self, x, y, alpha, weights):
        root = np.sqrt(weights).astype(np.float32)
        xw, yw = x * root[:, None], y * root
        gram = xw.T @ xw
        gram.flat[:: gram.shape[0] + 1] += alpha
        rhs = xw.T @ yw
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _finish(self, pred):
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = pred.clip(-self.prediction_clip_, self.prediction_clip_)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features):
        try:
            frames, assets = self._extract(features)
            first = next(iter(frames.values()))
            pred = self._rank(first) + 0.3 * self._rank(first.rolling(5, min_periods=2).mean())
            return self._finish(pred.reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=getattr(features, "columns", []), dtype=np.float32)

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            started = time.perf_counter()
            panel, assets = self._make_features(features)
            self.feature_time_ = time.perf_counter() - started
            x, y, assets = self._long(panel, target)
            if x.empty or len(y) < 40:
                return self
            self.selected_features_ = self._select_features(x)
            if not self.selected_features_:
                return self
            positions = self._positions(len(x))
            xs = x.iloc[positions][self.selected_features_]
            ys = y.iloc[positions]
            self._fit_preprocessor(xs)
            matrix = self._transform(xs)
            yv = np.nan_to_num(ys.to_numpy(dtype=np.float32), nan=0.0)
            fast_w = self._weights(positions, len(x), self._FAST_DECAY)
            slow_w = self._weights(positions, len(x), self._SLOW_DECAY)
            mid_w = (0.5 * fast_w + 0.5 * slow_w).astype(np.float32)
            self.training_rows_ = len(xs)
            self.feature_count_ = len(self.selected_features_)
            q = np.nanquantile(np.abs(yv), 0.995) if yv.size else 1.0
            self.prediction_clip_ = float(np.clip(3.0 * q if np.isfinite(q) and q > 0 else 1.0, 1e-6, 10.0))
            started = time.perf_counter()
            self.intercept_fast_ = float(yv.mean()) if yv.size else 0.0
            self.intercept_slow_ = self.intercept_fast_
            self.coef_fast_ = self._ridge(matrix, yv - self.intercept_fast_, self._ALPHA, fast_w)
            self.coef_slow_ = self._ridge(matrix, yv - self.intercept_slow_, self._ALPHA, slow_w)
            ranked = ys.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else ys.rank(method="average", pct=True)
            rank_y = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
            self.intercept_rank_ = float(rank_y.mean()) if rank_y.size else 0.0
            self.coef_rank_ = self._ridge(matrix, rank_y - self.intercept_rank_, self._ALPHA * 2.0, slow_w)
            self.xgb_model_ = xgb.train(self._XGB_PARAMS, xgb.DMatrix(matrix, label=yv, weight=mid_w), num_boost_round=self.n_xgb_rounds)
            self.fit_time_ = time.perf_counter() - started
            self.is_trained_ = True
            return self
        except Exception as exc:
            self.training_error_ = repr(exc)
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
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)
            started = time.perf_counter()
            matrix = self._transform(x)
            fast = self.intercept_fast_ + matrix @ self.coef_fast_
            slow = self.intercept_slow_ + matrix @ self.coef_slow_
            rank = self.intercept_rank_ + matrix @ self.coef_rank_
            tree = self.xgb_model_.predict(xgb.DMatrix(matrix))
            raw = self._FAST_WEIGHT * fast + self._SLOW_WEIGHT * slow + self._RANK_WEIGHT * rank + self._XGB_WEIGHT * tree
            self.predict_model_time_ = time.perf_counter() - started
            pred = pd.DataFrame(np.nan_to_num(raw).reshape(len(features.index), len(assets)), index=features.index, columns=assets, dtype=np.float32)
            return self._finish(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
