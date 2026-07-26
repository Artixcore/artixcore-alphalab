import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """AlphaLab v0.31: robust multi-horizon component ensemble."""

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_rows = 80000
        self.history_rows = 112
        self.max_features = 44
        self.features_ = None
        self.assets_ = []
        self.history_ = None
        self.med_ = self.lo_ = self.hi_ = self.mu_ = self.sd_ = None

        self.current_ridge_ = self.current_rank_ = None
        self.forward_ridge_ = self.slow_ridge_ = None
        self.current_b_ = self.current_rank_b_ = 0.0
        self.forward_b_ = self.slow_b_ = 0.0
        self.rank_scale_ = 1.0
        self.current_tree_ = self.current_rank_tree_ = None
        self.forward_tree_ = self.slow_tree_ = None

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
        denom = 0.5 * max(n - 1, 1)
        return ((x.rank(axis=1, method="average") - 0.5 * (n + 1)) / denom).astype(np.float32)

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
            x = x.loc[:, ~pd.Index(x.columns).duplicated()].reindex(columns=assets)
            frames[str(name)] = self._num(x)
        return frames, assets

    def _make(self, features):
        frames, assets = self._extract(features)
        if not frames:
            out = pd.DataFrame(index=features.index)
            out.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return out, assets

        names = list(frames)
        blocks = []
        ranks = {name: self._rank(frames[name]) for name in names}

        for i, name in enumerate(names):
            raw = frames[name]
            rank = ranks[name]
            blocks.append(self._block(name + "__raw", raw))
            blocks.append(self._block(name + "__rank", rank))
            if i < 3:
                blocks.append(self._block(name + "__d1", raw.diff(1)))
                blocks.append(self._block(name + "__d3", raw.diff(3)))
                blocks.append(self._block(name + "__rank_d1", rank.diff(1)))

            if i == 0:
                ma3 = raw.rolling(3, min_periods=2).mean()
                ma8 = raw.rolling(8, min_periods=3).mean()
                ma21 = raw.rolling(21, min_periods=5).mean()
                ma60 = raw.rolling(60, min_periods=8).mean()
                ew3 = raw.ewm(span=3, adjust=False, min_periods=2).mean()
                ew8 = raw.ewm(span=8, adjust=False, min_periods=3).mean()
                ew21 = raw.ewm(span=21, adjust=False, min_periods=5).mean()
                vol8 = raw.rolling(8, min_periods=3).std(ddof=0)
                vol21 = raw.rolling(21, min_periods=5).std(ddof=0)
                rank_l1 = rank.shift(1)
                mom3 = raw - raw.shift(3)
                mom5 = raw - raw.shift(5)
                blocks += [
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__lag3", raw.shift(3)),
                    self._block(name + "__rank_lag1", rank_l1),
                    self._block(name + "__rank_lag3", rank.shift(3)),
                    self._block(name + "__ma3", ma3),
                    self._block(name + "__ma8", ma8),
                    self._block(name + "__ma21", ma21),
                    self._block(name + "__ma60", ma60),
                    self._block(name + "__ew3", ew3),
                    self._block(name + "__ew8", ew8),
                    self._block(name + "__ew21", ew21),
                    self._block(name + "__vol8", vol8),
                    self._block(name + "__vol21", vol21),
                    self._block(name + "__z8", (raw - ma8) / (vol8 + 1e-6)),
                    self._block(name + "__z21", (raw - ma21) / (vol21 + 1e-6)),
                    self._block(name + "__trend3_21", ma3 - ma21),
                    self._block(name + "__ewspread", ew3 - ew21),
                    self._block(name + "__accel", raw.diff(1) - raw.diff(1).shift(1)),
                    self._block(name + "__mom3vol", mom3 / (vol8 + 1e-6)),
                    self._block(name + "__mom5vol", mom5 / (vol21 + 1e-6)),
                    self._block(name + "__rank_persist", rank + rank_l1),
                    self._block(name + "__rank_reversal", rank_l1 - rank),
                    self._block(name + "__demean", raw.sub(raw.median(axis=1), axis=0)),
                ]
            elif i == 1:
                ma5 = raw.rolling(5, min_periods=2).mean()
                vol10 = raw.rolling(10, min_periods=3).std(ddof=0)
                blocks += [
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__ma5", ma5),
                    self._block(name + "__reversal", ma5 - raw),
                    self._block(name + "__z10", (raw - raw.rolling(10, min_periods=3).mean()) / (vol10 + 1e-6)),
                ]
            elif i == 2:
                ma5 = raw.rolling(5, min_periods=2).mean()
                blocks += [
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__ma5", ma5),
                    self._block(name + "__rank_ma5", self._rank(ma5)),
                ]

        if len(names) >= 2:
            r1, r2 = ranks[names[0]], ranks[names[1]]
            blocks += [
                self._block("interaction__rank12_spread", r1 - r2),
                self._block("interaction__rank12_sum", r1 + r2),
                self._block("interaction__rank12_product", r1 * r2),
            ]
        if len(names) >= 3:
            r1, r2, r3 = ranks[names[0]], ranks[names[1]], ranks[names[2]]
            blocks += [
                self._block("interaction__rank13_spread", r1 - r3),
                self._block("interaction__rank13_product", r1 * r3),
                self._block("interaction__core", 0.55 * r1 - 0.30 * r2 + 0.15 * r3),
            ]

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
        typical = int(np.median(sizes[sizes > 0])) if np.any(sizes > 0) else 1
        cap = max(1, self.max_rows // max(1, typical))
        if n <= cap:
            chosen = np.arange(n)
        else:
            recent = max(1, int(cap * 0.65))
            old = max(0, cap - recent)
            head = np.linspace(0, max(0, n - recent - 1), old, dtype=np.int64) if old else np.array([], dtype=np.int64)
            chosen = np.unique(np.r_[head, np.arange(n - recent, n)])
        p = np.flatnonzero(np.isin(codes, chosen)).astype(np.int64)
        return p, codes[p].astype(np.int64), n

    def _select(self, x):
        if x.empty:
            return []
        q = x.iloc[np.linspace(0, len(x) - 1, min(len(x), 40000), dtype=np.int64)]
        v = q.to_numpy(np.float32, copy=False)
        scored = []
        priority = (
            "interaction__core", "__rank", "__lag1", "__rank_lag1", "__raw",
            "__rank12_spread", "__ma", "__ew", "__mom", "__trend", "__z",
        )
        for i, name in enumerate(q.columns):
            good = np.isfinite(v[:, i])
            if good.mean() < 0.05 or np.nanstd(v[good, i]) < 1e-8:
                continue
            label = str(name)
            bonus = next((len(priority) - j for j, token in enumerate(priority) if token in label), 0)
            scored.append((bonus, i, name))
        scored.sort(key=lambda z: (-z[0], z[1]))
        return [z[2] for z in scored[: self.max_features]]

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
        invalid = ~np.isfinite(self.lo_) | ~np.isfinite(self.hi_) | (self.lo_ >= self.hi_)
        self.lo_[invalid], self.hi_[invalid] = -10.0, 10.0
        v = np.clip(v, self.lo_, self.hi_)
        self.mu_ = np.nanmedian(v, axis=0).astype(np.float32)
        self.sd_ = (1.4826 * np.nanmedian(np.abs(v - self.mu_), axis=0)).astype(np.float32)
        std = np.nanstd(v, axis=0).astype(np.float32)
        invalid = ~np.isfinite(self.sd_) | (self.sd_ < 1e-8)
        self.sd_[invalid] = std[invalid]
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
        half_life = max(1.0, total * 0.28)
        w = np.exp(-np.log(2.0) * ((total - 1) - codes.astype(np.float32)) / half_life).astype(np.float32)
        return w / max(float(w.mean()), 1e-8)

    @staticmethod
    def _ridge(x, y, w, alpha):
        r = np.sqrt(np.maximum(w, 1e-8)).astype(np.float32)
        a, b = x * r[:, None], y * r
        g, rhs = a.T @ a, a.T @ b
        g.flat[:: g.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(g, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(g) @ rhs).astype(np.float32)

    def _finish(self, p):
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = p.to_numpy(np.float32, copy=True)
        if arr.size:
            med = np.median(arr, axis=1, keepdims=True)
            mad = 1.4826 * np.median(np.abs(arr - med), axis=1, keepdims=True)
            std = np.std(arr, axis=1, keepdims=True)
            scale = np.where(mad > 1e-7, mad, np.where(std > 1e-7, std, 1.0))
            arr = np.clip(arr, med - 3.5 * scale, med + 3.5 * scale)
        out = pd.DataFrame(arr, index=p.index, columns=p.columns)
        out = out.sub(out.mean(axis=1), axis=0).clip(-self.clip_, self.clip_)
        return out.fillna(0.0).astype(np.float32)

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
        self.current_ridge_ = self.current_rank_ = None
        self.forward_ridge_ = self.slow_ridge_ = None
        self.current_tree_ = self.current_rank_tree_ = None
        self.forward_tree_ = self.slow_tree_ = None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            self.history_ = features.tail(self.history_rows).copy()
            t = time.perf_counter()
            panel, assets = self._make(features)
            self.feature_time_, self.assets_ = time.perf_counter() - t, list(assets)
            x, y = self._long(panel, target, assets)
            if x.empty or len(y) < 80:
                self.training_error_ = "Insufficient training observations"
                return self

            self.features_ = self._select(x)
            pos, codes, total = self._sample(x.index)
            xs, ys = x.iloc[pos][self.features_], y.iloc[pos]
            self._fit_scale(xs)
            m = self._transform(xs)
            w = self._weights(codes, total)

            cur = np.nan_to_num(ys.to_numpy(np.float32), nan=0.0)
            lim = float(np.nanquantile(np.abs(cur), 0.995)) if cur.size else 1.0
            lim = lim if np.isfinite(lim) and lim > 0 else 1.0
            cur = np.clip(cur, -lim, lim).astype(np.float32)
            self.clip_ = float(np.clip(3.2 * lim, 1e-6, 10.0))

            if isinstance(y.index, pd.MultiIndex):
                fwd_all = y.groupby(level=-1).shift(-1)
                fwd2_all = y.groupby(level=-1).shift(-2)
            else:
                fwd_all, fwd2_all = y.shift(-1), y.shift(-2)
            fwd = fwd_all.iloc[pos].to_numpy(np.float32)
            fwd2 = fwd2_all.iloc[pos].to_numpy(np.float32)
            slow = 0.72 * fwd + 0.28 * fwd2

            t = time.perf_counter()
            self.current_b_ = float(np.average(cur, weights=w))
            self.current_ridge_ = self._ridge(m, cur - self.current_b_, w, 9.0)

            ranks = ys.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else ys.rank(method="average", pct=True)
            rank_y = np.nan_to_num(((ranks - 0.5) * 2.0).to_numpy(np.float32))
            self.current_rank_b_ = float(np.average(rank_y, weights=w))
            self.current_rank_ = self._ridge(m, rank_y - self.current_rank_b_, w, 22.0)
            self.rank_scale_ = float(np.std(cur) / np.std(rank_y)) if np.std(cur) > 1e-8 and np.std(rank_y) > 1e-8 else 1.0

            base = dict(
                objective="reg:squarederror", max_depth=2, eta=0.045,
                min_child_weight=180, reg_alpha=0.04, reg_lambda=2.1,
                tree_method="hist", verbosity=0, nthread=2,
            )
            pc = dict(base, subsample=0.84, colsample_bytree=0.78, seed=3101)
            pr = dict(base, subsample=0.90, colsample_bytree=0.70, seed=3119)
            self.current_tree_ = xgb.train(pc, xgb.DMatrix(m, label=cur, weight=w), num_boost_round=11)
            self.current_rank_tree_ = xgb.train(pr, xgb.DMatrix(m, label=rank_y, weight=w), num_boost_round=10)

            ok_f = np.isfinite(fwd)
            if ok_f.sum() >= 80:
                fy, fm, fw = np.clip(fwd[ok_f], -lim, lim).astype(np.float32), m[ok_f], w[ok_f]
                self.forward_b_ = float(np.average(fy, weights=fw))
                self.forward_ridge_ = self._ridge(fm, fy - self.forward_b_, fw, 13.0)
                pf = dict(base, subsample=0.87, colsample_bytree=0.76, seed=3137)
                self.forward_tree_ = xgb.train(pf, xgb.DMatrix(fm, label=fy, weight=fw), num_boost_round=11)
            else:
                self.forward_b_, self.forward_ridge_ = self.current_b_, self.current_ridge_.copy()

            ok_s = np.isfinite(slow)
            if ok_s.sum() >= 80:
                sy, sm, sw = np.clip(slow[ok_s], -lim, lim).astype(np.float32), m[ok_s], w[ok_s]
                self.slow_b_ = float(np.average(sy, weights=sw))
                self.slow_ridge_ = self._ridge(sm, sy - self.slow_b_, sw, 18.0)
                ps = dict(base, subsample=0.90, colsample_bytree=0.68, seed=3163)
                self.slow_tree_ = xgb.train(ps, xgb.DMatrix(sm, label=sy, weight=sw), num_boost_round=8)
            else:
                self.slow_b_, self.slow_ridge_ = self.forward_b_, self.forward_ridge_.copy()

            self.fit_time_ = time.perf_counter() - t
            self.training_rows_ = len(xs)
            self.training_times_ = len(pd.Index(xs.index.get_level_values(0)).unique()) if isinstance(xs.index, pd.MultiIndex) else len(xs)
            self.feature_count_ = len(self.features_)
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
            m = self._transform(x)
            dm = xgb.DMatrix(m)

            def as_frame(values):
                return pd.Series(
                    np.nan_to_num(values), index=x.index, dtype=np.float32
                ).unstack(-1).reindex(index=features.index, columns=assets).fillna(0.0)

            components = [
                as_frame(self.current_b_ + m @ self.current_ridge_),
                as_frame(self.current_tree_.predict(dm).astype(np.float32)),
                as_frame(self.rank_scale_ * (self.current_rank_b_ + m @ self.current_rank_)),
                as_frame(self.rank_scale_ * self.current_rank_tree_.predict(dm).astype(np.float32)),
                as_frame(self.forward_b_ + m @ self.forward_ridge_),
                as_frame(self.forward_tree_.predict(dm).astype(np.float32)),
                as_frame(self.slow_b_ + m @ self.slow_ridge_),
                as_frame(self.slow_tree_.predict(dm).astype(np.float32)),
            ]

            feature_names = list(panel.columns.get_level_values("feature").unique())
            rank_names = [
                name for name in feature_names
                if str(name).endswith("__rank") and not str(name).startswith("interaction")
            ]
            lag_rank_name = next(
                (name for name in feature_names if str(name).endswith("__rank_lag1")), None
            )
            core_name = next(
                (name for name in feature_names if str(name) == "interaction__core"), None
            )

            def panel_signal(name):
                if name is None:
                    return pd.DataFrame(0.0, index=features.index, columns=assets)
                return panel[name].reindex(index=features.index, columns=assets).fillna(0.0)

            components.extend([
                panel_signal(rank_names[0] if len(rank_names) > 0 else None),
                panel_signal(lag_rank_name),
                panel_signal(rank_names[1] if len(rank_names) > 1 else None),
                panel_signal(rank_names[2] if len(rank_names) > 2 else None),
                panel_signal(core_name),
            ])

            weights = np.array([
                -0.03787056, -0.01682583, 0.05846655, 0.08021127,
                -0.07269948, 0.15893598, 0.19806513, -0.14585455,
                0.15628094, 0.02679317, 0.01993833, 0.00849446, -0.01956375,
            ], dtype=np.float32)
            raw = np.zeros((len(features), len(assets)), dtype=np.float32)
            for weight, component in zip(weights, components):
                raw += weight * component.to_numpy(np.float32)

            p = pd.DataFrame(np.nan_to_num(raw), index=features.index, columns=assets)
            self.predict_model_time_ = time.perf_counter() - t
            return self._finish(p)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
