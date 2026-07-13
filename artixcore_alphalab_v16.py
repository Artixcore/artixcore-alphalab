import time
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.16: temporal-CV guarded v0.8 core."""

    _ALPHA = 8.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _BASE_RIDGE_WEIGHT = 0.75
    _BASE_XGB_WEIGHT = 0.25
    _RANK_CANDIDATES = (0.00, 0.02, 0.04, 0.06)
    _PRIORITY = (
        "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
        "Feature.4__raw", "Feature.5__raw", "Feature.6__raw",
        "Feature.1__cs_rank", "Feature.2__cs_rank", "Feature.3__cs_rank",
        "Feature.1__ma5", "Feature.1__ma20", "Feature.1__ma60",
        "Feature.1__sd20", "Feature.1__ewma5",
        "Feature.1__ma5_rank", "Feature.1__ma20_rank", "Feature.1__ma60_rank",
        "Feature.1__diff_1", "Feature.2__diff_1", "Feature.3__diff_1",
        "Feature.1__roll_z", "Feature.1__mom_spread",
        "Feature.1__cs_demean", "interaction__rank_spread",
    )
    _XGB = {
        "objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "min_child_weight": 200, "tree_method": "hist",
        "verbosity": 0, "nthread": 2, "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows, self.max_features, self.n_xgb_rounds = 80000, 35, 15
        self.raw_coef_ = self.rank_coef_ = None
        self.raw_intercept_ = self.rank_intercept_ = 0.0
        self.rank_scale_, self.rank_weight_ = 1.0, 0.0
        self.cv_scores_, self.xgb_model_ = None, None
        self.selected_features_ = None
        self.impute_ = self.low_ = self.high_ = None
        self.mean_ = self.scale_ = None
        self.prediction_clip_ = 1.0
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.feature_count_ = self.training_rows_ = 0
        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

    def _levels(self, columns):
        names = [str(v).lower() if v is not None else "" for v in columns.names]
        counts = [len(pd.Index(columns.get_level_values(i)).unique()) for i in range(columns.nlevels)]
        fl = next((i for i, n in enumerate(names) if "feature" in n or "factor" in n), None)
        al = next((i for i, n in enumerate(names) if "ticker" in n or "asset" in n or "symbol" in n), None)
        if fl is None:
            fl = int(np.argmin(counts))
        if al is None:
            rest = [i for i in range(columns.nlevels) if i != fl]
            al = max(rest, key=lambda i: counts[i]) if rest else fl
        return fl, al

    @staticmethod
    def _numeric(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            numeric = self._numeric(features)
            return {"Feature.1": numeric}, list(numeric.columns)
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
            frames[str(name)] = self._numeric(block)
        return frames, assets

    @staticmethod
    def _rank(frame):
        n = max(frame.shape[1], 1)
        rank = frame.rank(axis=1, method="average")
        return ((rank - 0.5 * (n + 1)) / (0.5 * max(n - 1, 1))).astype(np.float32)

    @staticmethod
    def _block(name, frame):
        out = frame.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)
        out.columns = pd.MultiIndex.from_product([[name], out.columns], names=["feature", "asset"])
        return out

    def _features(self, features):
        frames, assets = self._extract(features)
        if not frames:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty
        names, blocks = list(frames), []
        for i, name in enumerate(names):
            raw = frames[name]
            blocks.append(self._block(f"{name}__raw", raw))
            if i < 3:
                blocks.append(self._block(f"{name}__cs_rank", self._rank(raw)))
                blocks.append(self._block(f"{name}__diff_1", raw - raw.shift(1)))
            if i == 0:
                ma5 = raw.rolling(5, min_periods=2).mean()
                ma20 = raw.rolling(20, min_periods=3).mean()
                ma60 = raw.rolling(60, min_periods=5).mean()
                sd20 = raw.rolling(20, min_periods=3).std(ddof=0).fillna(0.0)
                blocks.extend([
                    self._block(f"{name}__ma5", ma5), self._block(f"{name}__ma20", ma20),
                    self._block(f"{name}__ma60", ma60), self._block(f"{name}__sd20", sd20),
                    self._block(f"{name}__ewma5", raw.ewm(span=5, adjust=False, min_periods=2).mean()),
                    self._block(f"{name}__ma5_rank", self._rank(ma5)),
                    self._block(f"{name}__ma20_rank", self._rank(ma20)),
                    self._block(f"{name}__ma60_rank", self._rank(ma60)),
                    self._block(f"{name}__roll_z", (raw - ma5) / (sd20 + 1e-6)),
                    self._block(f"{name}__mom_spread", ma5 - ma60),
                    self._block(f"{name}__cs_demean", raw.sub(raw.median(axis=1), axis=0)),
                ])
        if len(names) >= 2:
            spread = self._rank(frames[names[0]]) - self._rank(frames[names[1]])
            blocks.append(self._block("interaction__rank_spread", spread))
        panel = pd.concat(blocks, axis=1)
        feat_names = panel.columns.get_level_values("feature").unique()
        cols = pd.MultiIndex.from_product([feat_names, assets], names=["feature", "asset"])
        return panel.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).astype(np.float32)

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
        return self._numeric(frame.reindex(index=index, columns=assets))

    def _long(self, panel, target=None):
        assets = list(panel.columns.get_level_values("asset").unique())
        x = panel.stack(level="asset", future_stack=True).replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x, None, assets
        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.notna()
        return x.loc[valid], y.loc[valid].astype(np.float32), assets

    def _select(self, x):
        if x.empty:
            return []
        probe = x if len(x) <= 40000 else x.iloc[np.linspace(0, len(x) - 1, 40000, dtype=np.int64)]
        values, keep = probe.to_numpy(dtype=np.float32, copy=False), []
        for j, col in enumerate(probe.columns):
            finite = np.isfinite(values[:, j])
            if finite.any() and finite.mean() >= 0.05 and np.nanstd(values[finite, j]) >= 1e-8:
                keep.append(col)
        priority = {name: i for i, name in enumerate(self._PRIORITY)}
        keep.sort(key=lambda col: priority.get(col, len(priority)))
        return keep[:self.max_features]

    def _sample(self, n):
        if n <= self.max_train_rows:
            return np.arange(n, dtype=np.int64)
        recent = int(self.max_train_rows * 0.60)
        start = n - recent
        old = np.linspace(0, start - 1, self.max_train_rows - recent, dtype=np.int64)
        return np.unique(np.concatenate([old, np.arange(start, n, dtype=np.int64)]))

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
        mad = np.nanmedian(np.abs(a - self.mean_), axis=0).astype(np.float32)
        self.scale_ = (1.4826 * mad).astype(np.float32)
        fallback = a.std(axis=0).astype(np.float32)
        bad_scale = ~np.isfinite(self.scale_) | (self.scale_ < 1e-8)
        self.scale_[bad_scale] = fallback[bad_scale]
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0

    def _transform(self, x):
        a = x.reindex(columns=self.selected_features_).to_numpy(dtype=np.float32, copy=True)
        a[~np.isfinite(a)] = np.nan
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(self.impute_, np.where(bad)[1])
        a = np.clip(a, self.low_, self.high_)
        return np.nan_to_num((a - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _weights(self, positions, total):
        if total <= 1:
            return np.ones(len(positions), dtype=np.float32)
        ages = (total - 1) - positions.astype(np.float32)
        weights = np.exp(-ages / max(1.0, total * self._DECAY))
        return (weights / weights.mean()).astype(np.float32)

    @staticmethod
    def _ridge(matrix, target, weights, alpha):
        root = np.sqrt(weights).astype(np.float32)
        xw, yw = matrix * root[:, None], target * root
        gram, rhs = xw.T @ xw, xw.T @ yw
        gram.flat[::gram.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    @staticmethod
    def _rank_values(series):
        ranked = series.groupby(level=0).rank(method="average", pct=True) if isinstance(series.index, pd.MultiIndex) else series.rank(method="average", pct=True)
        return ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)

    @staticmethod
    def _fold_score(prediction, target, index):
        if len(prediction) < 20:
            return -np.inf
        if not isinstance(index, pd.MultiIndex):
            ps, ts = float(np.std(prediction)), float(np.std(target))
            return float(np.corrcoef(prediction, target)[0, 1]) if ps > 1e-8 and ts > 1e-8 else -np.inf
        frame = pd.DataFrame({"p": prediction, "y": target}, index=index).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            return -np.inf
        frame["p"] = frame["p"] - frame.groupby(level=0)["p"].transform("mean")
        returns = (frame["p"] * frame["y"]).groupby(level=0).mean()
        rs = float(returns.std(ddof=0))
        sharpe = float(returns.mean() / rs) if rs > 1e-8 else -np.inf
        ics = []
        for _, group in frame.groupby(level=0, sort=False):
            if len(group) >= 3 and group["p"].std(ddof=0) > 1e-8 and group["y"].std(ddof=0) > 1e-8:
                corr = float(group["p"].corr(group["y"]))
                if np.isfinite(corr):
                    ics.append(corr)
        return sharpe + 0.35 * (float(np.mean(ics)) if ics else 0.0)

    def _choose_rank_weight(self, matrix, target, weights, index):
        if not isinstance(index, pd.MultiIndex):
            return 0.02, None
        periods = pd.Index(index.get_level_values(0)).drop_duplicates()
        if len(periods) < 18:
            return 0.0, None
        rank_target = self._rank_values(pd.Series(target, index=index))
        period_values = index.get_level_values(0)
        scores = {candidate: [] for candidate in self._RANK_CANDIDATES}
        for train_fraction, validation_fraction in ((0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
            train_end = max(8, int(len(periods) * train_fraction))
            validation_end = min(len(periods), max(train_end + 2, int(len(periods) * validation_fraction)))
            if validation_end <= train_end:
                continue
            train_mask = np.asarray(period_values.isin(periods[:train_end]), dtype=bool)
            valid_mask = np.asarray(period_values.isin(periods[train_end:validation_end]), dtype=bool)
            if train_mask.sum() < 200 or valid_mask.sum() < 40:
                continue
            tx, vx = matrix[train_mask], matrix[valid_mask]
            ty, vy = target[train_mask], target[valid_mask]
            tw = weights[train_mask]
            raw_intercept = float(ty.mean())
            raw_coef = self._ridge(tx, ty - raw_intercept, tw, self._ALPHA)
            rank_train = rank_target[train_mask]
            rank_intercept = float(rank_train.mean())
            rank_coef = self._ridge(tx, rank_train - rank_intercept, tw, self._RANK_ALPHA)
            raw_prediction = raw_intercept + vx @ raw_coef
            rank_prediction = rank_intercept + vx @ rank_coef
            ystd, rstd = float(np.std(ty)), float(np.std(rank_train))
            rank_prediction *= ystd / rstd if ystd > 1e-8 and rstd > 1e-8 else 1.0
            valid_index = index[valid_mask]
            for candidate in self._RANK_CANDIDATES:
                blend = (1.0 - candidate) * raw_prediction + candidate * rank_prediction
                scores[candidate].append(self._fold_score(blend, vy, valid_index))
        valid = {k: v for k, v in scores.items() if len(v) >= 2 and np.all(np.isfinite(v))}
        if 0.0 not in valid:
            return 0.02, scores
        baseline = np.asarray(valid[0.0], dtype=np.float64)
        baseline_robust = float(baseline.mean() - 0.5 * baseline.std(ddof=0))
        eligible = []
        for candidate, values in valid.items():
            if candidate == 0.0:
                continue
            current = np.asarray(values, dtype=np.float64)
            robust = float(current.mean() - 0.5 * current.std(ddof=0))
            wins = int(np.sum(current > baseline + 1e-4))
            if wins >= 2 and robust > baseline_robust + 5e-4:
                eligible.append((robust, -candidate, candidate))
        if eligible:
            eligible.sort(reverse=True)
            return float(eligible[0][2]), scores
        small = valid.get(0.02)
        if small is not None:
            current = np.asarray(small, dtype=np.float64)
            if float(current.mean()) >= float(baseline.mean()) and int(np.sum(current > baseline)) >= 1:
                return 0.02, scores
        return 0.0, scores

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
            try:
                _, assets = self._extract(features)
            except Exception:
                assets = []
            return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.raw_coef_ = self.rank_coef_ = None
        self.xgb_model_, self.rank_weight_, self.cv_scores_ = None, 0.0, None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            started = time.perf_counter()
            panel = self._features(features)
            self.feature_time_ = time.perf_counter() - started
            x, y, _ = self._long(panel, target)
            del panel
            if x.empty or len(y) < 40:
                return self
            self.selected_features_ = self._select(x)
            if not self.selected_features_:
                return self
            positions = self._sample(len(x))
            xs, ys = x.iloc[positions][self.selected_features_], y.iloc[positions]
            self._fit_preprocessor(xs)
            matrix = self._transform(xs)
            weights = self._weights(positions, len(x))
            yv = np.nan_to_num(ys.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            limit = np.nanquantile(np.abs(yv), 0.995) if yv.size else 1.0
            self.prediction_clip_ = float(np.clip(3.0 * limit if np.isfinite(limit) and limit > 0 else 1.0, 1e-6, 10.0))
            started = time.perf_counter()
            self.rank_weight_, self.cv_scores_ = self._choose_rank_weight(matrix, yv, weights, ys.index)
            self.raw_intercept_ = float(yv.mean()) if yv.size else 0.0
            self.raw_coef_ = self._ridge(matrix, yv - self.raw_intercept_, weights, self._ALPHA)
            rv = self._rank_values(ys)
            self.rank_intercept_ = float(rv.mean()) if rv.size else 0.0
            self.rank_coef_ = self._ridge(matrix, rv - self.rank_intercept_, weights, self._RANK_ALPHA)
            ystd, rstd = float(np.std(yv)), float(np.std(rv))
            self.rank_scale_ = ystd / rstd if ystd > 1e-8 and rstd > 1e-8 else 1.0
            self.xgb_model_ = xgb.train(self._XGB, xgb.DMatrix(matrix, label=yv, weight=weights), num_boost_round=self.n_xgb_rounds)
            self.fit_time_ = time.perf_counter() - started
            self.training_rows_, self.feature_count_, self.is_trained_ = len(xs), len(self.selected_features_), True
        except Exception as exc:
            self.training_error_, self.is_trained_ = repr(exc), False
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
            panel = self._features(features)
            self.predict_feature_time_ = time.perf_counter() - started
            x, _, assets = self._long(panel)
            del panel
            if x.empty:
                self.fallback_used_ = True
                return self._fallback(features)
            started = time.perf_counter()
            matrix = self._transform(x)
            ridge = self.raw_intercept_ + matrix @ self.raw_coef_
            tree = self.xgb_model_.predict(xgb.DMatrix(matrix)).astype(np.float32)
            base = self._BASE_RIDGE_WEIGHT * ridge + self._BASE_XGB_WEIGHT * tree
            rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)
            raw = (1.0 - self.rank_weight_) * base + self.rank_weight_ * rank
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            series = pd.Series(raw, index=x.index, dtype=np.float32)
            pred = series.unstack(level=-1).reindex(index=features.index, columns=assets).fillna(0.0)
            self.predict_model_time_ = time.perf_counter() - started
            return self._finish(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback(features)
