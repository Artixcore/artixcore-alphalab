import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """v0.29: residual alpha with proxy-neutralized predictions."""

    PROXIES = ("proxy__f1_rank", "proxy__f2_rank", "proxy__f3_rank", "proxy__f1_mom", "proxy__f1_rev")
    PRIORITY = (
        "Feature.4__rank", "Feature.5__rank", "Feature.6__rank",
        "Feature.4__d1", "Feature.5__d1", "Feature.6__d1",
        "Feature.4__rev", "Feature.5__rev", "Feature.6__rev",
        "Feature.4__mom", "Feature.5__mom", "Feature.6__mom",
        "Feature.2__d1", "Feature.3__d1", "Feature.2__rev", "Feature.3__rev",
        "Feature.2__mom", "Feature.3__mom", "Feature.1__d1",
        "Feature.1__vd1", "Feature.2__vd1", "Feature.3__vd1",
        "Feature.4__vd1", "Feature.5__vd1", "Feature.6__vd1",
        "Feature.1__demean", "Feature.2__demean", "Feature.3__demean",
        "Feature.4__demean", "Feature.5__demean", "Feature.6__demean",
        "ix__spread12", "ix__spread34", "ix__spread56",
        "ix__product12", "ix__product34", "ix__product56",
        "ix__mom12", "ix__mom34", "ix__mom56", "ix__revdisp", "ix__momdisp",
    )
    XGB = dict(
        objective="reg:pseudohubererror", max_depth=2, eta=0.035,
        min_child_weight=260, subsample=0.84, colsample_bytree=0.76,
        reg_alpha=0.08, reg_lambda=2.6, tree_method="hist",
        verbosity=0, nthread=2, seed=2901,
    )

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows, self.max_features = 90000, 42
        self.n_xgb_rounds, self.history_rows = 18, 128
        self.selected_features_, self.assets_, self.history_tail_ = None, [], None
        self.impute_ = self.low_ = self.high_ = self.mean_ = self.scale_ = None
        self.ridge_coef_ = self.rank_coef_ = self.xgb_model_ = None
        self.ridge_intercept_ = self.rank_intercept_ = 0.0
        self.rank_scale_, self.prediction_clip_ = 1.0, 1.0
        self.is_trained_, self.fallback_used_, self.training_error_ = False, False, None
        self.training_rows_ = self.training_times_ = self.feature_count_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0
        self.target_proxy_correlation_ = self.last_proxy_correlation_ = np.nan

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
            x = self._num(features)
            return {"Feature.1": x}, list(x.columns)
        fl, al = self._levels(features.columns)
        names = list(dict.fromkeys(features.columns.get_level_values(fl)))
        assets = list(dict.fromkeys(features.columns.get_level_values(al)))
        out = {}
        for name in names:
            cols = [c for c in features.columns if c[fl] == name]
            if not cols:
                continue
            x = features.loc[:, cols].copy()
            x.columns = [c[al] for c in cols]
            out[str(name)] = self._num(x.loc[:, ~pd.Index(x.columns).duplicated()].reindex(columns=assets))
        return out, assets

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            x = pd.DataFrame(index=features.index)
            x.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return x, assets
        names, blocks, ranks, moms, revs = list(frames), [], {}, {}, {}
        for name, raw in frames.items():
            rank = self._rank(raw)
            d1 = raw.diff()
            ema4 = raw.ewm(span=4, adjust=False, min_periods=2).mean()
            ema16 = raw.ewm(span=16, adjust=False, min_periods=4).mean()
            mom, rev = self._rank(ema4 - ema16), self._rank(raw - ema4)
            vd1 = self._rank(d1 / (d1.rolling(12, min_periods=3).std(ddof=0) + 1e-6))
            ranks[name], moms[name], revs[name] = rank, mom, rev
            blocks += [
                self._block(name + "__rank", rank), self._block(name + "__d1", self._rank(d1)),
                self._block(name + "__rev", rev), self._block(name + "__mom", mom),
                self._block(name + "__vd1", vd1),
                self._block(name + "__demean", raw.sub(raw.median(axis=1), axis=0)),
            ]
        zero = pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
        at = lambda cache, i: cache[names[i]] if i < len(names) else zero
        blocks += [self._block(self.PROXIES[i], v) for i, v in enumerate((at(ranks, 0), at(ranks, 1), at(ranks, 2), at(moms, 0), at(revs, 0)))]
        for a, b, s in ((0, 1, "12"), (2, 3, "34"), (4, 5, "56")):
            ra, rb, ma, mb = at(ranks, a), at(ranks, b), at(moms, a), at(moms, b)
            blocks += [self._block("ix__spread" + s, ra - rb), self._block("ix__product" + s, ra * rb), self._block("ix__mom" + s, ma - mb)]
        rs = np.stack([revs[n].to_numpy(np.float32) for n in names], axis=0)
        ms = np.stack([moms[n].to_numpy(np.float32) for n in names], axis=0)
        blocks += [
            self._block("ix__revdisp", self._rank(pd.DataFrame(np.nanstd(rs, axis=0), index=features.index, columns=assets))),
            self._block("ix__momdisp", self._rank(pd.DataFrame(np.nanstd(ms, axis=0), index=features.index, columns=assets))),
        ]
        panel = pd.concat(blocks, axis=1)
        fns = panel.columns.get_level_values("feature").unique()
        cols = pd.MultiIndex.from_product([fns, assets], names=["feature", "asset"])
        return panel.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

    def _prediction_features(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if self.history_tail_ is None or self.history_tail_.empty:
            return self._features(features)
        try:
            hist = self.history_tail_.reindex(columns=features.columns)
            if hist.shape[1] != features.shape[1]:
                return self._features(features)
            panel, assets = self._features(pd.concat([hist, features], ignore_index=True))
            panel = panel.iloc[-len(features):].copy()
            panel.index = features.index
            return panel, assets
        except Exception:
            return self._features(features)

    @staticmethod
    def _stack(x, level=None):
        try:
            return x.stack(level=level, future_stack=True) if level is not None else x.stack(future_stack=True)
        except TypeError:
            return x.stack(level=level, dropna=False) if level is not None else x.stack(dropna=False)

    def _target(self, target, index, assets):
        if isinstance(target, pd.Series):
            x = target.unstack(level=-1) if isinstance(target.index, pd.MultiIndex) else target.to_frame()
        elif isinstance(target, pd.DataFrame):
            x = target.copy()
        else:
            x = pd.DataFrame(target, index=index)
        if isinstance(x.columns, pd.MultiIndex):
            _, al = self._levels(x.columns)
            x.columns = x.columns.get_level_values(al)
        x = self._num(x.reindex(index=index, columns=assets))
        return x.sub(x.mean(axis=1), axis=0).astype(np.float32)

    def _long(self, panel, target=None, assets=None):
        x = self._stack(panel, "asset").replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x
        y = self._stack(self._target(target, panel.index, assets))
        x, y = x.align(y, join="inner", axis=0)
        ok = y.replace([np.inf, -np.inf], np.nan).notna()
        return x.loc[ok], y.loc[ok].astype(np.float32)

    def _select(self, x):
        probe = x.iloc[np.linspace(0, len(x) - 1, min(len(x), 40000), dtype=np.int64)] if len(x) else x
        vals, allowed, out = probe.to_numpy(np.float32, copy=False), set(self.PRIORITY), []
        for i, name in enumerate(probe.columns):
            if name not in allowed:
                continue
            finite = np.isfinite(vals[:, i])
            if finite.mean() >= 0.05 and np.nanstd(vals[finite, i]) >= 1e-8:
                out.append(name)
        order = {n: i for i, n in enumerate(self.PRIORITY)}
        return sorted(out, key=lambda n: order[n])[:self.max_features]

    def _sample(self, index):
        if not isinstance(index, pd.MultiIndex):
            pos = np.arange(len(index), dtype=np.int64)
            return pos[-self.max_train_rows:], pos[-self.max_train_rows:], len(index)
        codes, times = pd.factorize(index.get_level_values(0), sort=False)
        n = len(times)
        sizes = np.bincount(codes, minlength=max(n, 1))
        cap = max(1, self.max_train_rows // max(1, int(np.median(sizes[sizes > 0]))))
        if n <= cap:
            chosen = np.arange(n)
        else:
            recent = int(cap * 0.65)
            chosen = np.unique(np.r_[np.linspace(0, n - recent - 1, cap - recent, dtype=np.int64), np.arange(n - recent, n)])
        pos = np.flatnonzero(np.isin(codes, chosen)).astype(np.int64)
        return pos, codes[pos].astype(np.int64), n

    def _fit_pre(self, x):
        v = x.to_numpy(np.float32, copy=True)
        v[~np.isfinite(v)] = np.nan
        self.impute_ = np.nanmedian(v, axis=0).astype(np.float32)
        self.impute_[~np.isfinite(self.impute_)] = 0.0
        miss = ~np.isfinite(v)
        if miss.any():
            v[miss] = np.take(self.impute_, np.where(miss)[1])
        self.low_, self.high_ = np.nanquantile(v, 0.01, axis=0).astype(np.float32), np.nanquantile(v, 0.99, axis=0).astype(np.float32)
        bad = ~np.isfinite(self.low_) | ~np.isfinite(self.high_) | (self.low_ >= self.high_)
        self.low_[bad], self.high_[bad] = -10.0, 10.0
        v = np.clip(v, self.low_, self.high_)
        self.mean_ = np.nanmedian(v, axis=0).astype(np.float32)
        self.scale_ = (1.4826 * np.nanmedian(np.abs(v - self.mean_), axis=0)).astype(np.float32)
        std = v.std(axis=0).astype(np.float32)
        bad = ~np.isfinite(self.scale_) | (self.scale_ < 1e-8)
        self.scale_[bad] = std[bad]
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        v = x.reindex(columns=self.selected_features_).to_numpy(np.float32, copy=True)
        v[~np.isfinite(v)] = np.nan
        miss = ~np.isfinite(v)
        if miss.any():
            v[miss] = np.take(self.impute_, np.where(miss)[1])
        return np.nan_to_num((np.clip(v, self.low_, self.high_) - self.mean_) / self.scale_).astype(np.float32)

    @staticmethod
    def _weights(codes, total, decay=0.14):
        if len(codes) == 0 or total <= 1:
            return np.ones(len(codes), np.float32)
        w = np.exp(-((total - 1) - codes.astype(np.float32)) / max(1.0, total * decay)).astype(np.float32)
        return w / max(float(w.mean()), 1e-8)

    @staticmethod
    def _ridge(x, y, w, alpha):
        r = np.sqrt(w).astype(np.float32)
        xw, yw = x * r[:, None], y * r
        g, b = xw.T @ xw, xw.T @ yw
        g.flat[::g.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(g, b).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(g) @ b).astype(np.float32)

    def _proxy_matrix(self, x):
        cols = [n for n in self.PROXIES if n in x.columns]
        return np.nan_to_num(x[cols].to_numpy(np.float32, copy=True)) if cols else np.zeros((len(x), 0), np.float32)

    def _neutralize(self, y, p, index):
        y = np.nan_to_num(np.asarray(y, np.float32))
        if p.size == 0:
            return y
        p, out = np.nan_to_num(np.asarray(p, np.float32)), y.copy()
        groups = pd.factorize(index.get_level_values(0), sort=False)[0] if isinstance(index, pd.MultiIndex) else np.zeros(len(y), np.int64)
        for g in np.unique(groups):
            pos = np.flatnonzero(groups == g)
            yy = out[pos].astype(np.float64)
            pp = p[pos].astype(np.float64)
            yy -= yy.mean()
            pp -= pp.mean(axis=0, keepdims=True)
            keep = np.std(pp, axis=0) > 1e-8
            pp = pp[:, keep]
            if pp.shape[1]:
                a = pp.T @ pp
                a.flat[::a.shape[0] + 1] += 0.35
                try:
                    beta = np.linalg.solve(a, pp.T @ yy)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(a) @ (pp.T @ yy)
                yy -= pp @ beta
            out[pos] = yy.astype(np.float32)
        return out

    @staticmethod
    def _maxcorr(y, p):
        if p.size == 0 or len(y) < 3:
            return np.nan
        y = np.asarray(y, float) - np.mean(y)
        sy, best = np.std(y), 0.0
        if sy < 1e-12:
            return 0.0
        for i in range(p.shape[1]):
            x = np.asarray(p[:, i], float) - np.mean(p[:, i])
            sx = np.std(x)
            if sx > 1e-12:
                best = max(best, abs(float(np.mean((x / sx) * (y / sy)))))
        return best

    def _proxy_frames(self, panel, index, assets):
        names = set(panel.columns.get_level_values("feature"))
        return [panel[n].reindex(index=index, columns=assets).fillna(0.0).astype(np.float32) for n in self.PROXIES if n in names]

    def _finish(self, pred, proxy_frames=None):
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-self.prediction_clip_, self.prediction_clip_)
        pred = pred.sub(pred.mean(axis=1), axis=0)
        med = pred.median(axis=1)
        mad = pred.sub(med, axis=0).abs().median(axis=1).replace(0.0, np.nan)
        pred = pred.clip(lower=med - 3 * mad, upper=med + 3 * mad, axis=0).fillna(0.0)
        sd = pred.std(axis=1, ddof=0).replace(0.0, np.nan)
        pred = 0.72 * pred.sub(pred.mean(axis=1), axis=0).div(sd, axis=0).fillna(0.0) + 0.28 * self._rank(pred)
        if proxy_frames:
            for i in range(len(pred)):
                p = np.column_stack([f.iloc[i].reindex(pred.columns).to_numpy(np.float32) for f in proxy_frames])
                pred.iloc[i] = self._neutralize(pred.iloc[i].to_numpy(np.float32), p, pd.RangeIndex(pred.shape[1]))
        pred = pred.sub(pred.mean(axis=1), axis=0)
        pred = pred.div(pred.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0).fillna(0.0).clip(-3.5, 3.5)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    @staticmethod
    def _zero(index, assets):
        return pd.DataFrame(0.0, index=index, columns=assets, dtype=np.float32)

    def _fallback(self, features, assets):
        try:
            panel, got = self._prediction_features(features)
            assets = got or assets
            if panel.empty:
                return self._zero(features.index, assets)
            names = set(panel.columns.get_level_values("feature"))
            name = next((n for n in ("ix__spread56", "Feature.6__rev", "Feature.4__d1", "Feature.1__vd1") if n in names), sorted(names)[0])
            return self._finish(panel[name].reindex(index=features.index, columns=assets).fillna(0.0), self._proxy_frames(panel, features.index, assets))
        except Exception:
            return self._zero(features.index, assets)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.ridge_coef_ = self.rank_coef_ = self.xgb_model_ = None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            self.history_tail_ = features.tail(self.history_rows).copy()
            t = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_, self.assets_ = time.perf_counter() - t, list(assets)
            x, y = self._long(panel, target, assets)
            if x.empty or len(y) < 40:
                self.training_error_ = "Insufficient usable training observations"
                return self
            self.selected_features_ = self._select(x)
            if not self.selected_features_:
                self.training_error_ = "No usable orthogonal features were selected"
                return self
            pos, codes, total = self._sample(x.index)
            xa, ys = x.iloc[pos], y.iloc[pos]
            xs, p = xa[self.selected_features_], self._proxy_matrix(xa)
            yt = self._neutralize(ys.to_numpy(np.float32), p, ys.index)
            self.target_proxy_correlation_ = self._maxcorr(yt, p)
            lim = float(np.nanquantile(np.abs(yt), 0.995)) if yt.size else 1.0
            lim = lim if np.isfinite(lim) and lim > 0 else 1.0
            yt = np.clip(yt, -lim, lim).astype(np.float32)
            self.prediction_clip_ = float(np.clip(3 * lim, 1e-6, 10.0))
            self._fit_pre(xs)
            m, w = self._transform(xs), self._weights(codes, total)
            t = time.perf_counter()
            self.ridge_intercept_ = float(np.average(yt, weights=w))
            self.ridge_coef_ = self._ridge(m, yt - self.ridge_intercept_, w, 14.0)
            s = pd.Series(yt, index=ys.index)
            rr = s.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else s.rank(method="average", pct=True)
            ry = np.nan_to_num(((rr - 0.5) * 2.0).to_numpy(np.float32))
            self.rank_intercept_ = float(np.average(ry, weights=w))
            self.rank_coef_ = self._ridge(m, ry - self.rank_intercept_, w, 30.0)
            self.rank_scale_ = float(np.std(yt) / np.std(ry)) if np.std(yt) > 1e-8 and np.std(ry) > 1e-8 else 1.0
            self.xgb_model_ = xgb.train(dict(self.XGB), xgb.DMatrix(m, label=yt, weight=w), num_boost_round=self.n_xgb_rounds)
            self.fit_time_ = time.perf_counter() - t
            self.training_rows_, self.training_times_, self.feature_count_ = len(xs), len(pd.Index(xs.index.get_level_values(0)).unique()), len(self.selected_features_)
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
            panel, assets = self._prediction_features(features)
            self.predict_feature_time_ = time.perf_counter() - t
            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)
            frames = self._proxy_frames(panel, features.index, assets)
            x = self._long(panel)
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)
            t = time.perf_counter()
            m, p = self._transform(x), self._proxy_matrix(x)
            ridge = self.ridge_intercept_ + m @ self.ridge_coef_
            tree = self.xgb_model_.predict(xgb.DMatrix(m)).astype(np.float32)
            rank = self.rank_scale_ * (self.rank_intercept_ + m @ self.rank_coef_)
            raw = self._neutralize(0.52 * ridge + 0.28 * tree + 0.20 * rank, p, x.index)
            self.last_proxy_correlation_ = self._maxcorr(raw, p)
            pred = pd.Series(raw, index=x.index, dtype=np.float32).unstack(level=-1).reindex(index=features.index, columns=assets).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - t
            return self._finish(pred, frames)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
