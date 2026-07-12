import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.6, robust multi-horizon ensemble."""

    _ALPHA = 8.0
    _DECAYS = (0.16, 0.30, 0.55)
    _PRESETS = (
        (0.52, 0.22, 0.10, 0.10, 0.06),
        (0.46, 0.26, 0.12, 0.10, 0.06),
        (0.58, 0.18, 0.08, 0.10, 0.06),
    )
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
        self.coefs_ = []
        self.intercepts_ = []
        self.rank_coef_ = None
        self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None
        self.weights_ = self._PRESETS[0]
        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.feature_count_ = 0
        self.training_rows_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    def _levels(self, cols):
        names = [str(x).lower() if x is not None else "" for x in cols.names]
        counts = [len(pd.Index(cols.get_level_values(i)).unique()) for i in range(cols.nlevels)]
        fl = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        al = next((i for i, n in enumerate(names) if "ticker" in n or "asset" in n or "symbol" in n), None)
        if fl is None:
            fl = int(np.argmin(counts))
        if al is None:
            rem = [i for i in range(cols.nlevels) if i != fl]
            al = max(rem, key=lambda i: counts[i]) if rem else fl
        return fl, al

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            f = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
            return {"Feature.1": f}, list(f.columns)
        fl, al = self._levels(features.columns)
        names = list(dict.fromkeys(features.columns.get_level_values(fl)))
        assets = list(dict.fromkeys(features.columns.get_level_values(al)))
        out = {}
        for name in names:
            cols = [c for c in features.columns if c[fl] == name]
            b = features.loc[:, cols].copy()
            b.columns = [c[al] for c in cols]
            b = b.loc[:, ~pd.Index(b.columns).duplicated()].reindex(columns=assets)
            out[str(name)] = b.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return out, assets

    def _rank(self, frame):
        n = max(frame.shape[1], 1)
        r = frame.rank(axis=1, method="average")
        return ((r - 0.5 * (n + 1)) / (0.5 * max(n - 1, 1))).astype(np.float32)

    def _block(self, name, frame):
        b = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        b.columns = pd.MultiIndex.from_product([[name], b.columns], names=["feature", "asset"])
        return b

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
            rc = rank - rank.shift(1)
            rank_changes.append(rc)
            blocks.extend([
                self._block(f"{name}__diff1", diff1),
                self._block(f"{name}__ma5", ma5),
                self._block(f"{name}__ma20", ma20),
                self._block(f"{name}__sd20", sd20),
                self._block(f"{name}__rankchg", rc),
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
            mean_rank = sum(ranks) / float(len(ranks))
            sq = sum(r * r for r in ranks) / float(len(ranks))
            dispersion = np.sqrt((sq - mean_rank * mean_rank).clip(lower=0.0))
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
            f = target.unstack(level=-1) if isinstance(target.index, pd.MultiIndex) else target.to_frame()
        elif isinstance(target, pd.DataFrame):
            f = target.copy()
        else:
            f = pd.DataFrame(target, index=index)
        if isinstance(f.columns, pd.MultiIndex):
            _, al = self._levels(f.columns)
            f.columns = f.columns.get_level_values(al)
        return f.reindex(index=index, columns=assets).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def _long(self, panel, target=None):
        assets = list(panel.columns.get_level_values("asset").unique())
        x = panel.stack(level="asset", future_stack=True).replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x, None, assets
        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        ok = y.notna()
        return x.loc[ok], y.loc[ok].astype(np.float32), assets

    def _sample_positions(self, n, limit=None):
        limit = int(limit or self.max_train_rows)
        if n <= limit:
            return np.arange(n, dtype=np.int64)
        recent = int(limit * 0.65)
        start = n - recent
        old = np.linspace(0, start - 1, limit - recent, dtype=np.int64)
        return np.unique(np.concatenate([old, np.arange(start, n, dtype=np.int64)]))

    def _select_features(self, x, y):
        pos = np.linspace(0, len(x) - 1, min(len(x), 40000), dtype=np.int64)
        xs, ys = x.iloc[pos], y.iloc[pos].to_numpy(dtype=np.float32)
        values = xs.to_numpy(dtype=np.float32, copy=False)
        scored = []
        for j, col in enumerate(xs.columns):
            v = values[:, j]
            ok = np.isfinite(v) & np.isfinite(ys)
            if ok.mean() < 0.05 or np.nanstd(v[ok]) < 1e-8:
                continue
            c = np.corrcoef(v[ok], ys[ok])[0, 1] if ok.sum() >= 20 else 0.0
            scored.append((col, abs(c) if np.isfinite(c) else 0.0))
        scored.sort(key=lambda z: z[1], reverse=True)
        return [c for c, _ in scored[:self.max_features]]

    def _fit_preprocessor(self, x):
        a = x.to_numpy(dtype=np.float32, copy=True)
        a[~np.isfinite(a)] = np.nan
        self.impute_ = np.nanmedian(a, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(self.impute_, np.where(bad)[1])
        self.low_ = np.nanquantile(a, 0.005, axis=0).astype(np.float32)
        self.high_ = np.nanquantile(a, 0.995, axis=0).astype(np.float32)
        invalid = ~np.isfinite(self.low_) | ~np.isfinite(self.high_) | (self.low_ >= self.high_)
        self.low_[invalid], self.high_[invalid] = -10.0, 10.0
        a = np.clip(a, self.low_, self.high_)
        self.mean_ = a.mean(axis=0).astype(np.float32)
        self.scale_ = a.std(axis=0).astype(np.float32)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        a = x.reindex(columns=self.selected_features_).to_numpy(dtype=np.float32, copy=True)
        a[~np.isfinite(a)] = np.nan
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(self.impute_, np.where(bad)[1])
        a = np.clip(a, self.low_, self.high_)
        return np.nan_to_num((a - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _weights(self, positions, total, decay):
        ages = (total - 1) - positions.astype(np.float32)
        w = np.exp(-ages / max(1.0, total * decay))
        return (w / w.mean()).astype(np.float32)

    def _ridge(self, x, y, alpha, w):
        root = np.sqrt(w).astype(np.float32)
        xw, yw = x * root[:, None], y * root
        gram = xw.T @ xw
        gram.flat[::gram.shape[0] + 1] += alpha
        rhs = xw.T @ yw
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def _fit_models(self, matrix, y, positions, total):
        target = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0)
        intercept = float(target.mean()) if target.size else 0.0
        self.coefs_, self.intercepts_ = [], []
        ridge_preds = []
        for decay in self._DECAYS:
            w = self._weights(positions, total, decay)
            coef = self._ridge(matrix, target - intercept, self._ALPHA, w)
            self.coefs_.append(coef)
            self.intercepts_.append(intercept)
            ridge_preds.append(intercept + matrix @ coef)
        ranked = y.groupby(level=0).rank(method="average", pct=True) if isinstance(y.index, pd.MultiIndex) else y.rank(method="average", pct=True)
        rank_y = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
        self.rank_intercept_ = float(rank_y.mean()) if rank_y.size else 0.0
        rw = self._weights(positions, total, self._RANK_DECAY)
        self.rank_coef_ = self._ridge(matrix, rank_y - self.rank_intercept_, self._ALPHA * 2.0, rw)
        tstd, rstd = float(np.std(target)), float(np.std(rank_y))
        self.rank_scale_ = tstd / rstd if tstd > 1e-8 and rstd > 1e-8 else 1.0
        base = 0.60 * ridge_preds[0] + 0.25 * ridge_preds[1] + 0.15 * ridge_preds[2]
        residual = target - base
        xw = 0.65 * self._weights(positions, total, self._DECAYS[0]) + 0.35 * self._weights(positions, total, self._DECAYS[1])
        self.xgb_model_ = xgb.train(self._XGB, xgb.DMatrix(matrix, label=residual, weight=xw), num_boost_round=self.n_xgb_rounds)

    def _components(self, matrix):
        fast = self.intercepts_[0] + matrix @ self.coefs_[0]
        mid = self.intercepts_[1] + matrix @ self.coefs_[1]
        slow = self.intercepts_[2] + matrix @ self.coefs_[2]
        rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)
        residual = self.xgb_model_.predict(xgb.DMatrix(matrix)).astype(np.float32)
        return fast, mid, slow, rank, residual

    def _finish(self, pred):
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = pred.clip(-self.prediction_clip_, self.prediction_clip_)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features):
        try:
            frames, assets = self._extract(features)
            first = next(iter(frames.values()))
            pred = 0.75 * self._rank(first) + 0.25 * self._rank(first.rolling(5, min_periods=2).mean())
            return self._finish(pred.reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=[], dtype=np.float32)

    def train(self, features, target):
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            t = time.perf_counter()
            panel, assets = self._make_features(features)
            self.feature_time_ = time.perf_counter() - t
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
            q = np.nanquantile(np.abs(ys.to_numpy(dtype=np.float32)), 0.995)
            self.prediction_clip_ = float(np.clip(3.0 * q if np.isfinite(q) and q > 0 else 1.0, 1e-6, 10.0))
            t = time.perf_counter()
            self._fit_models(matrix, ys, positions, len(x))
            self.fit_time_ = time.perf_counter() - t
            self.training_rows_ = len(xs)
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
            t = time.perf_counter()
            panel, assets = self._make_features(features)
            self.predict_feature_time_ = time.perf_counter() - t
            x, _, assets = self._long(panel)
            del panel
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)
            t = time.perf_counter()
            matrix = self._transform(x)
            fast, mid, slow, rank, residual = self._components(matrix)
            wf, wm, ws, wr, wx = self.weights_
            raw = wf * fast + wm * mid + ws * slow + wr * rank + wx * residual
            pred = pd.DataFrame(np.nan_to_num(raw).reshape(len(features.index), len(assets)), index=features.index, columns=assets, dtype=np.float32)
            self.predict_model_time_ = time.perf_counter() - t
            return self._finish(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
