import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """AlphaLab v0.30: robust dual-horizon cross-sectional ensemble."""

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_rows = 80000
        self.history_rows = 96
        self.features_ = None
        self.assets_ = []
        self.history_ = None
        self.med_ = self.lo_ = self.hi_ = self.mu_ = self.sd_ = None
        self.ridge_ = self.rank_ = self.forward_ = None
        self.ridge_b_ = self.rank_b_ = self.forward_b_ = 0.0
        self.rank_scale_ = 1.0
        self.tree_ = self.rank_tree_ = self.forward_tree_ = None
        self.clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None
        self.fallback_used_ = False
        self.training_rows_ = self.training_times_ = self.feature_count_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    @staticmethod
    def _num(x):
        return x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

    @staticmethod
    def _rank(x):
        n = max(x.shape[1], 1)
        return ((x.rank(axis=1, method="average") - 0.5 * (n + 1)) / (0.5 * max(n - 1, 1))).astype(np.float32)

    @staticmethod
    def _block(name, x):
        y = x.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        y.columns = pd.MultiIndex.from_product([[name], y.columns], names=["feature", "asset"])
        return y

    def _levels(self, cols):
        names = [str(v).lower() if v is not None else "" for v in cols.names]
        counts = [len(pd.Index(cols.get_level_values(i)).unique()) for i in range(cols.nlevels)]
        fl = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        al = next((i for i, n in enumerate(names) if "asset" in n or "ticker" in n or "symbol" in n), None)
        fl = int(np.argmin(counts)) if fl is None else fl
        if al is None:
            rest = [i for i in range(cols.nlevels) if i != fl]
            al = max(rest, key=lambda i: counts[i]) if rest else fl
        return fl, al

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            x = self._num(features)
            return {"Feature.1": x}, list(x.columns)
        fl, al = self._levels(features.columns)
        assets = list(dict.fromkeys(features.columns.get_level_values(al)))
        frames = {}
        for name in dict.fromkeys(features.columns.get_level_values(fl)):
            cols = [c for c in features.columns if c[fl] == name]
            x = features.loc[:, cols].copy()
            x.columns = [c[al] for c in cols]
            frames[str(name)] = self._num(x.loc[:, ~pd.Index(x.columns).duplicated()].reindex(columns=assets))
        return frames, assets

    def _make(self, features):
        frames, assets = self._extract(features)
        if not frames:
            out = pd.DataFrame(index=features.index)
            out.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return out, assets
        names, blocks = list(frames), []
        for i, name in enumerate(names):
            raw = frames[name]
            blocks.append(self._block(name + "__raw", raw))
            if i < 3:
                blocks.append(self._block(name + "__rank", self._rank(raw)))
                blocks.append(self._block(name + "__d1", raw.diff()))
            if i == 0:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma20 = raw.rolling(20, min_periods=3).mean()
                ma60 = raw.rolling(60, min_periods=5).mean()
                vol = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
                ew5 = raw.ewm(span=5, adjust=False, min_periods=2).mean()
                blocks += [
                    self._block(name + "__ma5", ma5), self._block(name + "__ma20", ma20),
                    self._block(name + "__ma60", ma60), self._block(name + "__vol20", vol),
                    self._block(name + "__ew5", ew5), self._block(name + "__ma5rank", self._rank(ma5)),
                    self._block(name + "__ma20rank", self._rank(ma20)), self._block(name + "__ma60rank", self._rank(ma60)),
                    self._block(name + "__z5", (raw - ma5) / (vol + 1e-6)),
                    self._block(name + "__trend", ma5 - ma60),
                    self._block(name + "__demean", raw.sub(raw.median(axis=1), axis=0)),
                ]
        if len(names) >= 2:
            blocks.append(self._block("interaction__rank12", self._rank(frames[names[0]]) - self._rank(frames[names[1]])))
        panel = pd.concat(blocks, axis=1)
        fns = panel.columns.get_level_values("feature").unique()
        cols = pd.MultiIndex.from_product([fns, assets], names=["feature", "asset"])
        return panel.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

    def _predict_panel(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if self.history_ is None or self.history_.empty:
            return self._make(features)
        try:
            hist = self.history_.reindex(columns=features.columns)
            both = pd.concat([hist, features], ignore_index=True)
            panel, assets = self._make(both)
            panel = panel.iloc[-len(features):].copy()
            panel.index = features.index
            return panel, assets
        except Exception:
            return self._make(features)

    def _target(self, target, index, assets):
        if isinstance(target, pd.Series):
            y = target.unstack(-1) if isinstance(target.index, pd.MultiIndex) else target.to_frame()
        elif isinstance(target, pd.DataFrame):
            y = target.copy()
        else:
            y = pd.DataFrame(target, index=index)
        if isinstance(y.columns, pd.MultiIndex):
            _, al = self._levels(y.columns)
            y.columns = y.columns.get_level_values(al)
        y = self._num(y.reindex(index=index, columns=assets))
        return y.sub(y.mean(axis=1), axis=0).astype(np.float32)

    @staticmethod
    def _stack(x):
        try:
            return x.stack(level="asset", future_stack=True)
        except TypeError:
            return x.stack(level="asset", dropna=False)

    def _long(self, panel, target=None, assets=None):
        x = self._stack(panel).replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x
        try:
            y = self._target(target, panel.index, assets).stack(future_stack=True)
        except TypeError:
            y = self._target(target, panel.index, assets).stack(dropna=False)
        x, y = x.align(y, join="inner", axis=0)
        ok = y.replace([np.inf, -np.inf], np.nan).notna()
        return x.loc[ok], y.loc[ok].astype(np.float32)

    def _sample(self, index):
        if not isinstance(index, pd.MultiIndex):
            p = np.arange(len(index), dtype=np.int64)[-self.max_rows:]
            return p, p, len(index)
        codes, times = pd.factorize(index.get_level_values(0), sort=False)
        n = len(times)
        sizes = np.bincount(codes, minlength=max(n, 1))
        cap = max(1, self.max_rows // max(1, int(np.median(sizes[sizes > 0]))))
        if n <= cap:
            chosen = np.arange(n)
        else:
            recent = int(cap * 0.6)
            chosen = np.unique(np.r_[np.linspace(0, n - recent - 1, cap - recent, dtype=np.int64), np.arange(n - recent, n)])
        p = np.flatnonzero(np.isin(codes, chosen)).astype(np.int64)
        return p, codes[p].astype(np.int64), n

    def _select(self, x):
        if x.empty:
            return []
        q = x.iloc[np.linspace(0, len(x) - 1, min(len(x), 40000), dtype=np.int64)]
        v, out = q.to_numpy(np.float32, copy=False), []
        for i, name in enumerate(q.columns):
            good = np.isfinite(v[:, i])
            if good.mean() >= 0.05 and np.nanstd(v[good, i]) >= 1e-8:
                out.append(name)
        return out[:35]

    def _fit_scale(self, x):
        v = x.to_numpy(np.float32, copy=True)
        v[~np.isfinite(v)] = np.nan
        self.med_ = np.nanmedian(v, axis=0).astype(np.float32)
        self.med_[~np.isfinite(self.med_)] = 0.0
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.take(self.med_, np.where(bad)[1])
        self.lo_ = np.nanquantile(v, 0.005, axis=0).astype(np.float32)
        self.hi_ = np.nanquantile(v, 0.995, axis=0).astype(np.float32)
        bad = ~np.isfinite(self.lo_) | ~np.isfinite(self.hi_) | (self.lo_ >= self.hi_)
        self.lo_[bad], self.hi_[bad] = -10.0, 10.0
        v = np.clip(v, self.lo_, self.hi_)
        self.mu_ = v.mean(axis=0).astype(np.float32)
        self.sd_ = (1.4826 * np.nanmedian(np.abs(v - self.mu_), axis=0)).astype(np.float32)
        std = v.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(self.sd_) | (self.sd_ < 1e-8)
        self.sd_[bad] = std[bad]
        self.sd_[~np.isfinite(self.sd_) | (self.sd_ < 1e-8)] = 1.0

    def _transform(self, x):
        v = x.reindex(columns=self.features_).to_numpy(np.float32, copy=True)
        v[~np.isfinite(v)] = np.nan
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.take(self.med_, np.where(bad)[1])
        return np.nan_to_num((np.clip(v, self.lo_, self.hi_) - self.mu_) / self.sd_).astype(np.float32)

    @staticmethod
    def _weights(codes, total):
        if len(codes) == 0 or total <= 1:
            return np.ones(len(codes), np.float32)
        w = np.exp(-((total - 1) - codes.astype(np.float32)) / max(1.0, total * 0.2)).astype(np.float32)
        return w / max(float(w.mean()), 1e-8)

    @staticmethod
    def _ridge(x, y, w, alpha):
        r = np.sqrt(w).astype(np.float32)
        a, b = x * r[:, None], y * r
        g, rhs = a.T @ a, a.T @ b
        g.flat[::g.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(g, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(g) @ rhs).astype(np.float32)

    def _finish(self, p):
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-self.clip_, self.clip_)
        return p.sub(p.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features, assets):
        try:
            panel, got = self._predict_panel(features)
            assets = got or assets
            names = list(panel.columns.get_level_values("feature").unique())
            name = next((n for n in names if n.endswith("__rank")), names[0])
            return self._finish(panel[name].reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.ridge_ = self.rank_ = self.forward_ = None
        self.tree_ = self.rank_tree_ = self.forward_tree_ = None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            self.history_ = features.tail(self.history_rows).copy()
            t = time.perf_counter()
            panel, assets = self._make(features)
            self.feature_time_, self.assets_ = time.perf_counter() - t, list(assets)
            x, y = self._long(panel, target, assets)
            if x.empty or len(y) < 40:
                self.training_error_ = "Insufficient training observations"
                return self
            self.features_ = self._select(x)
            pos, codes, total = self._sample(x.index)
            xs, ys = x.iloc[pos][self.features_], y.iloc[pos]
            yf_all = y.groupby(level=-1).shift(-1) if isinstance(y.index, pd.MultiIndex) else y.shift(-1)
            yf = yf_all.iloc[pos].to_numpy(np.float32)
            self._fit_scale(xs)
            m = self._transform(xs)
            w = self._weights(codes, total)
            cur = np.nan_to_num(ys.to_numpy(np.float32))
            lim = float(np.nanquantile(np.abs(cur), 0.995)) if cur.size else 1.0
            lim = lim if np.isfinite(lim) and lim > 0 else 1.0
            cur = np.clip(cur, -lim, lim).astype(np.float32)
            self.clip_ = float(np.clip(3 * lim, 1e-6, 10.0))
            t = time.perf_counter()
            self.ridge_b_ = float(np.average(cur, weights=w))
            self.ridge_ = self._ridge(m, cur - self.ridge_b_, w, 8.0)
            ranks = ys.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else ys.rank(method="average", pct=True)
            ry = np.nan_to_num(((ranks - 0.5) * 2).to_numpy(np.float32))
            self.rank_b_ = float(np.average(ry, weights=w))
            self.rank_ = self._ridge(m, ry - self.rank_b_, w, 20.0)
            self.rank_scale_ = float(np.std(cur) / np.std(ry)) if np.std(cur) > 1e-8 and np.std(ry) > 1e-8 else 1.0
            base = dict(objective="reg:squarederror", max_depth=2, eta=0.05, min_child_weight=220, reg_alpha=0.03, reg_lambda=1.8, tree_method="hist", verbosity=0, nthread=2)
            pa, pb = dict(base, subsample=0.82, colsample_bytree=0.82, seed=42), dict(base, subsample=0.9, colsample_bytree=0.72, seed=137)
            self.tree_ = xgb.train(pa, xgb.DMatrix(m, label=cur, weight=w), num_boost_round=12)
            self.rank_tree_ = xgb.train(pb, xgb.DMatrix(m, label=ry, weight=w), num_boost_round=12)
            ok = np.isfinite(yf)
            if ok.sum() >= 40:
                fm, fw, fy = m[ok], w[ok], np.clip(yf[ok], -lim, lim).astype(np.float32)
                self.forward_b_ = float(np.average(fy, weights=fw))
                self.forward_ = self._ridge(fm, fy - self.forward_b_, fw, 12.0)
                pf = dict(base, subsample=0.86, colsample_bytree=0.78, seed=3030)
                self.forward_tree_ = xgb.train(pf, xgb.DMatrix(fm, label=fy, weight=fw), num_boost_round=10)
            else:
                self.forward_b_, self.forward_ = self.ridge_b_, self.ridge_.copy()
            self.fit_time_ = time.perf_counter() - t
            self.training_rows_, self.training_times_, self.feature_count_ = len(xs), len(pd.Index(xs.index.get_level_values(0)).unique()), len(self.features_)
            self.is_trained_ = True
        except Exception as exc:
            self.training_error_ = repr(exc)
        return self

    def predict(self, features):
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            t = time.perf_counter()
            panel, assets = self._predict_panel(features)
            self.predict_feature_time_ = time.perf_counter() - t
            if not self.is_trained_ or not self.features_:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)
            x = self._long(panel)
            t = time.perf_counter()
            m, dm = self._transform(x), None
            dm = xgb.DMatrix(m)
            ridge = self.ridge_b_ + m @ self.ridge_
            tree = self.tree_.predict(dm).astype(np.float32)
            rank = self.rank_scale_ * (0.8 * self.rank_tree_.predict(dm).astype(np.float32) + 0.2 * (self.rank_b_ + m @ self.rank_))
            current = 0.58 * ridge + 0.17 * tree + 0.25 * rank
            forward_ridge = self.forward_b_ + m @ self.forward_
            forward = 0.72 * forward_ridge + 0.28 * self.forward_tree_.predict(dm).astype(np.float32) if self.forward_tree_ is not None else forward_ridge
            raw = np.nan_to_num(0.5 * current + 0.5 * forward)
            p = pd.Series(raw, index=x.index, dtype=np.float32).unstack(-1).reindex(index=features.index, columns=assets).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - t
            return self._finish(p)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
