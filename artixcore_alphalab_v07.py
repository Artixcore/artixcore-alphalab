import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.7: direction-corrected multi-horizon ensemble."""

    _ALPHA = 8.0
    _DECAYS = (0.16, 0.30, 0.55)
    _BLEND = (0.52, 0.22, 0.10, 0.10, 0.06)
    _OUTPUT_SIGN = -1.0
    _XGB = {
        "objective": "reg:squarederror", "max_depth": 2, "eta": 0.04,
        "subsample": 0.82, "colsample_bytree": 0.78,
        "min_child_weight": 220, "reg_alpha": 0.15, "reg_lambda": 3.5,
        "tree_method": "hist", "verbosity": 0, "nthread": 2, "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows = 85000
        self.max_features = 40
        self.n_xgb_rounds = 16
        self.selected_features_ = None
        self.impute_ = self.low_ = self.high_ = self.mean_ = self.scale_ = None
        self.coefs_, self.intercepts_ = [], []
        self.rank_coef_, self.xgb_model_ = None, None
        self.rank_intercept_, self.rank_scale_ = 0.0, 1.0
        self.output_sign_ = self._OUTPUT_SIGN
        self.prediction_clip_, self.is_trained_ = 1.0, False
        self.training_error_, self.fallback_used_ = None, False
        self.feature_count_, self.training_rows_ = 0, 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    def _levels(self, cols):
        names = [str(v).lower() if v is not None else "" for v in cols.names]
        counts = [len(pd.Index(cols.get_level_values(i)).unique()) for i in range(cols.nlevels)]
        fl = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        al = next((i for i, n in enumerate(names) if "ticker" in n or "asset" in n or "symbol" in n), None)
        if fl is None:
            fl = int(np.argmin(counts))
        if al is None:
            remain = [i for i in range(cols.nlevels) if i != fl]
            al = max(remain, key=lambda i: counts[i]) if remain else fl
        return fl, al

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            frame = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
            return {"Feature.1": frame}, list(frame.columns)
        fl, al = self._levels(features.columns)
        names = list(dict.fromkeys(features.columns.get_level_values(fl)))
        assets = list(dict.fromkeys(features.columns.get_level_values(al)))
        frames = {}
        for name in names:
            cols = [c for c in features.columns if c[fl] == name]
            if not cols:
                continue
            block = features.loc[:, cols].copy()
            block.columns = [c[al] for c in cols]
            block = block.loc[:, ~pd.Index(block.columns).duplicated()].reindex(columns=assets)
            frames[str(name)] = block.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
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
        blocks, ranks, rank_changes = [], [], []
        for i, (name, raw) in enumerate(frames.items()):
            blocks.append(self._block(f"{name}__raw", raw))
            if i >= 4:
                continue
            rank = self._rank(raw)
            ranks.append(rank)
            blocks.append(self._block(f"{name}__rank", rank))
            if i >= 3:
                continue
            diff1 = raw - raw.shift(1)
            ma5 = raw.rolling(5, min_periods=2).mean()
            ma20 = raw.rolling(20, min_periods=3).mean()
            sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
            rank_change = rank - rank.shift(1)
            rank_changes.append(rank_change)
            blocks += [
                self._block(f"{name}__diff1", diff1), self._block(f"{name}__ma5", ma5),
                self._block(f"{name}__ma20", ma20), self._block(f"{name}__sd20", sd20),
                self._block(f"{name}__rankchg", rank_change),
                self._block(f"{name}__voladj", diff1 / (sd20 + 1e-6)),
            ]
            if i == 0:
                ma60 = raw.rolling(60, min_periods=5).mean()
                blocks += [
                    self._block(f"{name}__ma60", ma60),
                    self._block(f"{name}__ewma5", raw.ewm(span=5, adjust=False, min_periods=2).mean()),
                    self._block(f"{name}__rollz", (raw - ma5) / (sd20 + 1e-6)),
                    self._block(f"{name}__momspread", ma5 - ma60),
                ]
        if ranks:
            mean_rank = sum(ranks) / float(len(ranks))
            square_mean = sum(r * r for r in ranks) / float(len(ranks))
            dispersion = np.sqrt((square_mean - mean_rank * mean_rank).clip(lower=0.0))
            blocks += [self._block("interaction__rank_mean", mean_rank), self._block("interaction__rank_dispersion", dispersion)]
        if len(ranks) >= 2:
            blocks.append(self._block("interaction__rank_spread12", ranks[0] - ranks[1]))
        if len(ranks) >= 3:
            blocks.append(self._block("interaction__rank_spread13", ranks[0] - ranks[2]))
        if rank_changes:
            blocks.append(self._block("interaction__rank_accel", sum(rank_changes) / float(len(rank_changes))))
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
            _, al = self._levels(frame.columns)
            frame.columns = frame.columns.get_level_values(al)
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

    def _sample_positions(self, n, limit=None):
        limit = int(limit or self.max_train_rows)
        if n <= limit:
            return np.arange(n, dtype=np.int64)
        recent = int(limit * 0.65)
        start = n - recent
        old = np.linspace(0, start - 1, limit - recent, dtype=np.int64)
        return np.unique(np.concatenate([old, np.arange(start, n, dtype=np.int64)]))

    def _select_features(self, x, y):
        positions = np.linspace(0, len(x) - 1, min(len(x), 40000), dtype=np.int64)
        xs, ys = x.iloc[positions], y.iloc[positions].to_numpy(dtype=np.float32)
        values, scored = xs.to_numpy(dtype=np.float32, copy=False), []
        for j, col in enumerate(xs.columns):
            v = values[:, j]
            valid = np.isfinite(v) & np.isfinite(ys)
            if valid.sum() < 20 or valid.mean() < 0.05 or np.nanstd(v[valid]) < 1e-8:
                continue
            corr = np.corrcoef(v[valid], ys[valid])[0, 1]
            scored.append((col, abs(corr) if np.isfinite(corr) else 0.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [col for col, _ in scored[:self.max_features]]

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
        invalid = ~np.isfinite(self.low_) | ~np.isfinite(self.high_) | (self.low_ >= self.high_)
        self.low_[invalid], self.high_[invalid] = -10.0, 10.0
        values = np.clip(values, self.low_, self.high_)
        self.mean_ = values.mean(axis=0).astype(np.float32)
        self.scale_ = values.std(axis=0).astype(np.float32)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        values = x.reindex(columns=self.selected_features_).to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = np.take(self.impute_, np.where(bad)[1])
        values = np.clip(values, self.low_, self.high_)
        return np.nan_to_num((values - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _weights(self, positions, total, decay):
        if total <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total - 1) - positions.astype(np.float32)
        weights = np.exp(-ages / max(1.0, total * decay))
        return (weights / weights.mean()).astype(np.float32)

    def _ridge(self, x, y, alpha, weights):
        root = np.sqrt(weights).astype(np.float32)
        xw, yw = x * root[:, None], y * root
        gram, rhs = xw.T @ xw, xw.T @ yw
        gram.flat[::gram.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _fit_models(self, matrix, y, positions, total):
        target = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        intercept = float(target.mean()) if target.size else 0.0
        weights = [self._weights(positions, total, decay) for decay in self._DECAYS]
        self.coefs_, self.intercepts_, ridge_predictions = [], [], []
        for recency_weight in weights:
            coef = self._ridge(matrix, target - intercept, self._ALPHA, recency_weight)
            self.coefs_.append(coef)
            self.intercepts_.append(intercept)
            ridge_predictions.append(intercept + matrix @ coef)
        ranked = y.groupby(level=0).rank(method="average", pct=True) if isinstance(y.index, pd.MultiIndex) else y.rank(method="average", pct=True)
        rank_target = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
        self.rank_intercept_ = float(rank_target.mean()) if rank_target.size else 0.0
        self.rank_coef_ = self._ridge(matrix, rank_target - self.rank_intercept_, self._ALPHA * 2.0, weights[1])
        target_std, rank_std = float(np.std(target)), float(np.std(rank_target))
        self.rank_scale_ = target_std / rank_std if target_std > 1e-8 and rank_std > 1e-8 else 1.0
        base = 0.60 * ridge_predictions[0] + 0.25 * ridge_predictions[1] + 0.15 * ridge_predictions[2]
        residual = target - base
        residual_weights = (0.65 * weights[0] + 0.35 * weights[1]).astype(np.float32)
        self.xgb_model_ = xgb.train(self._XGB, xgb.DMatrix(matrix, label=residual, weight=residual_weights), num_boost_round=self.n_xgb_rounds)

    def _components(self, matrix):
        fast = self.intercepts_[0] + matrix @ self.coefs_[0]
        medium = self.intercepts_[1] + matrix @ self.coefs_[1]
        slow = self.intercepts_[2] + matrix @ self.coefs_[2]
        rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)
        residual = self.xgb_model_.predict(xgb.DMatrix(matrix)).astype(np.float32)
        return fast, medium, slow, rank, residual

    def _finish(self, prediction):
        prediction = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prediction = prediction.clip(-self.prediction_clip_, self.prediction_clip_)
        return prediction.sub(prediction.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features):
        try:
            frames, assets = self._extract(features)
            first = next(iter(frames.values()))
            prediction = self.output_sign_ * (0.75 * self._rank(first) + 0.25 * self._rank(first.rolling(5, min_periods=2).mean()))
            return self._finish(prediction.reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            try:
                _, assets = self._extract(features)
            except Exception:
                assets = []
            return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
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
            self.selected_features_ = self._select_features(x, y)
            if not self.selected_features_:
                return self
            positions = self._sample_positions(len(x))
            xs, ys = x.iloc[positions][self.selected_features_], y.iloc[positions]
            self._fit_preprocessor(xs)
            matrix = self._transform(xs)
            values = ys.to_numpy(dtype=np.float32)
            q = np.nanquantile(np.abs(values), 0.995) if values.size else 1.0
            self.prediction_clip_ = float(np.clip(3.0 * q if np.isfinite(q) and q > 0 else 1.0, 1e-6, 10.0))
            started = time.perf_counter()
            self._fit_models(matrix, ys, positions, len(x))
            self.fit_time_ = time.perf_counter() - started
            self.training_rows_, self.feature_count_ = len(xs), len(self.selected_features_)
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
            del panel
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)
            started = time.perf_counter()
            matrix = self._transform(x)
            fast, medium, slow, rank, residual = self._components(matrix)
            wf, wm, ws, wr, wx = self._BLEND
            raw = self.output_sign_ * (wf * fast + wm * medium + ws * slow + wr * rank + wx * residual)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            prediction = pd.DataFrame(raw.reshape(len(features.index), len(assets)), index=features.index, columns=assets, dtype=np.float32)
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(prediction)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
