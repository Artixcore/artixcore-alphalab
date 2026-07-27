import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """AlphaLab v0.32: leakage-safe robust cross-sectional ensemble."""

    XGB = {
        "objective": "reg:squarederror", "max_depth": 2, "eta": 0.045,
        "min_child_weight": 210, "reg_alpha": 0.04, "reg_lambda": 2.0,
        "tree_method": "hist", "verbosity": 0, "nthread": 2,
    }
    PRIORITY = (
        "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
        "Feature.4__raw", "Feature.5__raw", "Feature.6__raw",
        "Feature.1__rank", "Feature.2__rank", "Feature.3__rank",
        "Feature.4__rank", "Feature.5__rank", "Feature.6__rank",
        "Feature.1__lag1", "Feature.1__lag2", "Feature.1__lag3",
        "Feature.2__lag1", "Feature.3__lag1",
        "Feature.1__d1", "Feature.1__d3", "Feature.1__d5",
        "Feature.2__d1", "Feature.2__d3", "Feature.3__d1",
        "Feature.1__rank_lag1", "Feature.1__rank_persist",
        "Feature.1__rank_reversal", "Feature.1__ma3", "Feature.1__ma8",
        "Feature.1__ma21", "Feature.1__ma60", "Feature.1__ew4",
        "Feature.1__ew12", "Feature.1__ew32", "Feature.1__vol8",
        "Feature.1__vol21", "Feature.1__z8", "Feature.1__z21",
        "Feature.1__mom3vol", "Feature.1__mom5vol",
        "Feature.1__trend3_21", "Feature.1__trend8_60",
        "Feature.1__accel", "Feature.1__demean",
        "Feature.2__reversal5", "Feature.2__z10",
        "Feature.3__ma5rank", "interaction__rank12",
        "interaction__rank13", "interaction__product12",
        "interaction__core", "interaction__dispersion",
    )

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_rows, self.max_features, self.history_rows = 80000, 44, 112
        self.features_, self.assets_, self.history_ = None, [], None
        self.med_ = self.lo_ = self.hi_ = self.center_ = self.scale_ = None
        self.raw_coef_ = self.rank_coef_ = None
        self.raw_b_ = self.rank_b_ = 0.0
        self.rank_scale_, self.rank_share_ = 1.0, 0.30
        self.raw_tree_ = self.rank_tree_ = None
        self.clip_ = 1.0
        self.is_trained_, self.fallback_used_, self.training_error_ = False, False, None
        self.training_rows_ = self.training_times_ = self.feature_count_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    @staticmethod
    def _num(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype(np.float32)

    def _levels(self, columns):
        names = [str(v).lower() if v is not None else "" for v in columns.names]
        counts = [len(pd.Index(columns.get_level_values(i)).unique()) for i in range(columns.nlevels)]
        fl = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        al = next((i for i, n in enumerate(names) if "asset" in n or "ticker" in n or "symbol" in n), None)
        fl = int(np.argmin(counts)) if fl is None else fl
        if al is None:
            rest = [i for i in range(columns.nlevels) if i != fl]
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

    @staticmethod
    def _rank(frame):
        n = max(frame.shape[1], 1)
        return ((frame.rank(axis=1, method="average") - 0.5 * (n + 1)) /
                (0.5 * max(n - 1, 1))).astype(np.float32)

    @staticmethod
    def _block(name, frame):
        out = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        out.columns = pd.MultiIndex.from_product([[name], out.columns], names=["feature", "asset"])
        return out

    def _make(self, features):
        frames, assets = self._extract(features)
        if not frames:
            out = pd.DataFrame(index=features.index)
            out.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return out, assets
        names = list(frames)
        ranks = {n: self._rank(frames[n]) for n in names}
        blocks = []
        for i, name in enumerate(names):
            raw, rank = frames[name], ranks[name]
            blocks += [self._block(name + "__raw", raw), self._block(name + "__rank", rank)]
            if i < 3:
                blocks += [
                    self._block(name + "__lag1", raw.shift(1)),
                    self._block(name + "__lag2", raw.shift(2)),
                    self._block(name + "__lag3", raw.shift(3)),
                    self._block(name + "__d1", raw.diff(1)),
                    self._block(name + "__d3", raw.diff(3)),
                    self._block(name + "__d5", raw.diff(5)),
                ]
            if i == 0:
                ma3 = raw.rolling(3, min_periods=2).mean()
                ma8 = raw.rolling(8, min_periods=3).mean()
                ma21 = raw.rolling(21, min_periods=5).mean()
                ma60 = raw.rolling(60, min_periods=8).mean()
                ew4 = raw.ewm(span=4, adjust=False, min_periods=2).mean()
                ew12 = raw.ewm(span=12, adjust=False, min_periods=3).mean()
                ew32 = raw.ewm(span=32, adjust=False, min_periods=5).mean()
                vol8 = raw.rolling(8, min_periods=3).std(ddof=0)
                vol21 = raw.rolling(21, min_periods=5).std(ddof=0)
                lag_rank = rank.shift(1)
                d1 = raw.diff(1)
                blocks += [
                    self._block(name + "__rank_lag1", lag_rank),
                    self._block(name + "__rank_persist", rank * lag_rank),
                    self._block(name + "__rank_reversal", lag_rank - rank),
                    self._block(name + "__ma3", ma3), self._block(name + "__ma8", ma8),
                    self._block(name + "__ma21", ma21), self._block(name + "__ma60", ma60),
                    self._block(name + "__ew4", ew4), self._block(name + "__ew12", ew12),
                    self._block(name + "__ew32", ew32), self._block(name + "__vol8", vol8),
                    self._block(name + "__vol21", vol21),
                    self._block(name + "__z8", (raw - ma8) / (vol8 + 1e-6)),
                    self._block(name + "__z21", (raw - ma21) / (vol21 + 1e-6)),
                    self._block(name + "__mom3vol", (raw - raw.shift(3)) / (vol8 + 1e-6)),
                    self._block(name + "__mom5vol", (raw - raw.shift(5)) / (vol21 + 1e-6)),
                    self._block(name + "__trend3_21", ma3 - ma21),
                    self._block(name + "__trend8_60", ma8 - ma60),
                    self._block(name + "__accel", d1 - d1.shift(1)),
                    self._block(name + "__demean", raw.sub(raw.median(axis=1), axis=0)),
                ]
            elif i == 1:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma10 = raw.rolling(10, min_periods=3).mean()
                vol10 = raw.rolling(10, min_periods=3).std(ddof=0)
                blocks += [self._block(name + "__reversal5", ma5 - raw),
                           self._block(name + "__z10", (raw - ma10) / (vol10 + 1e-6))]
            elif i == 2:
                ma5 = raw.rolling(5, min_periods=2).mean()
                blocks.append(self._block(name + "__ma5rank", self._rank(ma5)))
        if len(names) >= 2:
            r1, r2 = ranks[names[0]], ranks[names[1]]
            blocks += [self._block("interaction__rank12", r1 - r2),
                       self._block("interaction__product12", r1 * r2)]
        if len(names) >= 3:
            r1, r2, r3 = ranks[names[0]], ranks[names[1]], ranks[names[2]]
            blocks += [self._block("interaction__rank13", r1 - r3),
                       self._block("interaction__core", 0.55 * r1 - 0.30 * r2 + 0.15 * r3)]
        stack = np.stack([ranks[n].to_numpy(np.float32) for n in names], axis=0)
        disp = pd.DataFrame(np.nanstd(stack, axis=0), index=features.index, columns=assets)
        blocks.append(self._block("interaction__dispersion", self._rank(disp)))
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
            panel, assets = self._make(pd.concat([hist, features], ignore_index=True))
            panel = panel.tail(len(features)).copy()
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
    def _stack(frame, level=None):
        try:
            return frame.stack(level=level, future_stack=True) if level is not None else frame.stack(future_stack=True)
        except TypeError:
            return frame.stack(level=level, dropna=False) if level is not None else frame.stack(dropna=False)

    def _long(self, panel, target=None, assets=None):
        x = self._stack(panel, "asset").replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x
        y = self._stack(self._target(target, panel.index, assets))
        x, y = x.align(y, join="inner", axis=0)
        good = y.replace([np.inf, -np.inf], np.nan).notna()
        return x.loc[good], y.loc[good].astype(np.float32)

    def _select(self, x):
        q = x if len(x) <= 40000 else x.iloc[np.linspace(0, len(x) - 1, 40000, dtype=np.int64)]
        values, usable = q.to_numpy(np.float32, copy=False), []
        for i, name in enumerate(q.columns):
            finite = np.isfinite(values[:, i])
            if finite.mean() >= 0.05 and np.nanstd(values[finite, i]) >= 1e-8:
                usable.append(name)
        order = {name: i for i, name in enumerate(self.PRIORITY)}
        usable.sort(key=lambda name: order.get(name, len(order)))
        return usable[:self.max_features]

    def _sample(self, index):
        if not isinstance(index, pd.MultiIndex):
            pos = np.arange(len(index), dtype=np.int64)
            if len(pos) > self.max_rows:
                pos = pos[len(pos) - self.max_rows:]
            return pos, pos, len(index)
        codes, times = pd.factorize(index.get_level_values(0), sort=False)
        total = len(times)
        sizes = np.bincount(codes, minlength=max(total, 1))
        typical = max(1, int(np.median(sizes[sizes > 0])))
        cap = max(1, self.max_rows // typical)
        if total <= cap:
            chosen = np.arange(total)
        else:
            recent = max(1, int(cap * 0.65)); old = cap - recent; start = total - recent
            head = np.linspace(0, start - 1, old, dtype=np.int64) if old else np.empty(0, np.int64)
            chosen = np.unique(np.r_[head, np.arange(start, total)])
        pos = np.flatnonzero(np.isin(codes, chosen)).astype(np.int64)
        return pos, codes[pos].astype(np.int64), total

    def _fit_scale(self, x):
        v = x.to_numpy(np.float32, copy=True); v[~np.isfinite(v)] = np.nan
        self.med_ = np.nanmedian(v, axis=0).astype(np.float32); self.med_[~np.isfinite(self.med_)] = 0.0
        bad = ~np.isfinite(v)
        if bad.any(): v[bad] = np.take(self.med_, np.where(bad)[1])
        self.lo_ = np.nanquantile(v, 0.005, axis=0).astype(np.float32)
        self.hi_ = np.nanquantile(v, 0.995, axis=0).astype(np.float32)
        bad = ~np.isfinite(self.lo_) | ~np.isfinite(self.hi_) | (self.lo_ >= self.hi_)
        self.lo_[bad], self.hi_[bad] = -10.0, 10.0
        v = np.clip(v, self.lo_, self.hi_)
        self.center_ = np.nanmedian(v, axis=0).astype(np.float32)
        self.scale_ = (1.4826 * np.nanmedian(np.abs(v - self.center_), axis=0)).astype(np.float32)
        std = np.nanstd(v, axis=0).astype(np.float32)
        bad = ~np.isfinite(self.scale_) | (self.scale_ < 1e-8); self.scale_[bad] = std[bad]
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        v = x.reindex(columns=self.features_).to_numpy(np.float32, copy=True); v[~np.isfinite(v)] = np.nan
        bad = ~np.isfinite(v)
        if bad.any(): v[bad] = np.take(self.med_, np.where(bad)[1])
        return np.nan_to_num((np.clip(v, self.lo_, self.hi_) - self.center_) / self.scale_).astype(np.float32)

    @staticmethod
    def _weights(codes, total):
        if len(codes) == 0 or total <= 1: return np.ones(len(codes), np.float32)
        w = np.exp(-((total - 1) - codes.astype(np.float32)) / max(1.0, total * 0.20)).astype(np.float32)
        return w / max(float(w.mean()), 1e-8)

    @staticmethod
    def _ridge(x, y, w, alpha):
        r = np.sqrt(np.maximum(w, 1e-8)).astype(np.float32); a, b = x * r[:, None], y * r
        g, rhs = a.T @ a, a.T @ b; g.flat[::g.shape[0] + 1] += alpha
        try: return np.linalg.solve(g, rhs).astype(np.float32)
        except np.linalg.LinAlgError: return (np.linalg.pinv(g) @ rhs).astype(np.float32)

    def _choose_share(self, matrix, raw, rank, weights, codes):
        unique = np.unique(codes)
        if len(unique) < 18: return 0.30
        cut = len(unique) - max(4, int(np.ceil(len(unique) * 0.18)))
        train, valid = np.isin(codes, unique[:cut]), np.isin(codes, unique[cut:])
        if train.sum() < 80 or valid.sum() < 40: return 0.30
        xb, xv, w = matrix[train], matrix[valid], weights[train]
        rb, kb = raw[train], rank[train]
        ri, ki = float(np.average(rb, weights=w)), float(np.average(kb, weights=w))
        rc, kc = self._ridge(xb, rb - ri, w, 9.0), self._ridge(xb, kb - ki, w, 24.0)
        rp, kp = ri + xv @ rc, ki + xv @ kc
        if np.std(kb) > 1e-8: kp *= float(np.std(rb) / np.std(kb))
        y, vc = raw[valid], codes[valid]; scores = {}
        for share in (0.22, 0.30, 0.38):
            pred = (1 - share) * rp + share * kp; vals = []
            for code in np.unique(vc):
                p = np.flatnonzero(vc == code); a, b = pred[p], y[p]; a = a - a.mean(); b = b - b.mean()
                d = np.linalg.norm(a) * np.linalg.norm(b)
                if d > 1e-12: vals.append(float(np.dot(a, b) / d))
            scores[share] = float(np.mean(vals) - 0.20 * np.std(vals)) if vals else -np.inf
        best = max(scores, key=scores.get)
        return best if best == 0.30 or scores[best] >= scores[0.30] + 0.002 else 0.30

    @staticmethod
    def _corr(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        n = np.sum(a * b, 1); d = np.sqrt(np.sum(a * a, 1) * np.sum(b * b, 1))
        return np.divide(n, d, out=np.zeros_like(n), where=d > 1e-12)

    def _finish(self, p):
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        v = p.to_numpy(np.float32, copy=True)
        if v.size:
            med = np.median(v, 1, keepdims=True); mad = 1.4826 * np.median(np.abs(v - med), 1, keepdims=True)
            std = np.std(v, 1, keepdims=True); scale = np.where(mad > 1e-7, mad, np.where(std > 1e-7, std, 1.0))
            v = np.clip(v, med - 3.5 * scale, med + 3.5 * scale)
        out = pd.DataFrame(v, index=p.index, columns=p.columns).clip(-self.clip_, self.clip_)
        return out.sub(out.mean(1), axis=0).fillna(0.0).astype(np.float32)

    def _fallback(self, features, assets):
        try:
            panel, got = self._predict_panel(features); assets = got or assets
            names = list(panel.columns.get_level_values("feature").unique())
            name = next((n for n in names if n.endswith("__rank")), names[0])
            return self._finish(panel[name].reindex(index=features.index, columns=assets).fillna(0.0))
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        try:
            if not isinstance(features, pd.DataFrame): features = pd.DataFrame(features)
            self.history_ = features.tail(self.history_rows).copy(); started = time.perf_counter()
            panel, assets = self._make(features); self.feature_time_ = time.perf_counter() - started; self.assets_ = list(assets)
            x, y = self._long(panel, target, assets)
            if x.empty or len(y) < 80: self.training_error_ = "Insufficient training observations"; return self
            self.features_ = self._select(x); pos, codes, total = self._sample(x.index)
            xs, ys = x.iloc[pos][self.features_], y.iloc[pos]; self._fit_scale(xs); matrix = self._transform(xs); w = self._weights(codes, total)
            raw = np.nan_to_num(ys.to_numpy(np.float32)); limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
            limit = limit if np.isfinite(limit) and limit > 0 else 1.0; raw = np.clip(raw, -limit, limit).astype(np.float32); self.clip_ = float(np.clip(3.2 * limit, 1e-6, 10.0))
            ranked = ys.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else ys.rank(method="average", pct=True)
            rank = np.nan_to_num(((ranked - 0.5) * 2.0).to_numpy(np.float32)); self.rank_share_ = self._choose_share(matrix, raw, rank, w, codes)
            started = time.perf_counter(); initial_b = float(np.average(raw, weights=w)); initial_c = self._ridge(matrix, raw - initial_b, w, 9.0)
            resid = raw - (initial_b + matrix @ initial_c); scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
            if not np.isfinite(scale) or scale < 1e-8: scale = max(float(np.std(resid)), 1.0)
            huber = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(resid), 1e-8)).astype(np.float32); rw = w * huber; rw /= max(float(rw.mean()), 1e-8)
            self.raw_b_ = float(np.average(raw, weights=rw)); self.raw_coef_ = self._ridge(matrix, raw - self.raw_b_, rw, 9.0)
            self.rank_b_ = float(np.average(rank, weights=w)); self.rank_coef_ = self._ridge(matrix, rank - self.rank_b_, w, 24.0)
            self.rank_scale_ = float(np.std(raw) / np.std(rank)) if np.std(raw) > 1e-8 and np.std(rank) > 1e-8 else 1.0
            pa = dict(self.XGB, subsample=0.84, colsample_bytree=0.80, seed=3201); pb = dict(self.XGB, subsample=0.90, colsample_bytree=0.72, seed=3229)
            self.raw_tree_ = xgb.train(pa, xgb.DMatrix(matrix, label=raw, weight=rw), num_boost_round=12)
            self.rank_tree_ = xgb.train(pb, xgb.DMatrix(matrix, label=rank, weight=w), num_boost_round=12)
            self.fit_time_ = time.perf_counter() - started; self.training_rows_ = len(xs)
            self.training_times_ = len(pd.Index(xs.index.get_level_values(0)).unique()) if isinstance(xs.index, pd.MultiIndex) else len(xs)
            self.feature_count_, self.is_trained_ = len(self.features_), True
        except Exception as exc: self.training_error_ = repr(exc); self.is_trained_ = False
        return self

    def predict(self, features):
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame): features = pd.DataFrame(features)
            started = time.perf_counter(); panel, assets = self._predict_panel(features); self.predict_feature_time_ = time.perf_counter() - started
            if not self.is_trained_ or not self.features_: self.fallback_used_ = True; return self._fallback(features, assets or self.assets_)
            x = self._long(panel); started = time.perf_counter(); matrix = self._transform(x); dm = xgb.DMatrix(matrix)
            def frame(values):
                return pd.Series(np.nan_to_num(values), index=x.index, dtype=np.float32).unstack(-1).reindex(index=features.index, columns=assets).fillna(0.0)
            ridge = frame(self.raw_b_ + matrix @ self.raw_coef_).to_numpy(np.float32); tree = frame(self.raw_tree_.predict(dm)).to_numpy(np.float32)
            rank_ridge = frame(self.rank_scale_ * (self.rank_b_ + matrix @ self.rank_coef_)).to_numpy(np.float32)
            rank_tree = frame(self.rank_scale_ * self.rank_tree_.predict(dm)).to_numpy(np.float32)
            ga = np.clip(0.5 * (1 + self._corr(ridge, tree)), 0.15, 1.0); gb = np.clip(0.5 * (1 + self._corr(rank_ridge, rank_tree)), 0.15, 1.0)
            raw_family = ridge + 0.28 * ga[:, None] * (tree - ridge); rank_family = rank_ridge + 0.72 * gb[:, None] * (rank_tree - rank_ridge)
            agreement = np.clip(0.5 * (1 + self._corr(raw_family, rank_family)), 0.0, 1.0)
            share = np.clip(self.rank_share_ + 0.08 * (agreement - 0.5), 0.18, 0.42)
            values = (1 - share[:, None]) * raw_family + share[:, None] * rank_family
            low = agreement < 0.30; values[low] = 0.82 * ridge[low] + 0.18 * values[low]
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(pd.DataFrame(np.nan_to_num(values), index=features.index, columns=assets))
        except Exception: self.fallback_used_ = True; return self._fallback(features, self.assets_)
