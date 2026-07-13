import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """Artixcore AlphaLab v0.19: tail-aware refinement of v0.18."""

    _ALPHA = 8.0
    _TAIL_ALPHA = 14.0
    _RANK_ALPHA = 20.0
    _DECAY = 0.20
    _MAIN_WEIGHT = 0.62
    _TAIL_WEIGHT = 0.10
    _XGB_WEIGHT = 0.24
    _RANK_WEIGHT = 0.04
    _XGB = {
        "objective": "reg:squarederror", "max_depth": 2, "eta": 0.05,
        "subsample": 0.80, "colsample_bytree": 0.80,
        "min_child_weight": 200, "reg_alpha": 0.02, "reg_lambda": 1.5,
        "tree_method": "hist", "verbosity": 0, "nthread": 2, "seed": 42,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass
        self.max_train_rows = 80000
        self.max_features = 35
        self.n_xgb_rounds = 15
        self.selected_features_ = None
        self.impute_ = self.low_ = self.high_ = None
        self.mean_ = self.scale_ = None
        self.raw_coef_ = self.tail_coef_ = self.rank_coef_ = None
        self.raw_intercept_ = self.tail_intercept_ = self.rank_intercept_ = 0.0
        self.rank_scale_ = 1.0
        self.xgb_model_ = None
        self.assets_ = []
        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.training_error_ = None

    @staticmethod
    def _numeric(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype(np.float32)

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

    def _extract(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        if not isinstance(features.columns, pd.MultiIndex):
            frame = self._numeric(features)
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
            return empty, assets
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
                    self._block(f"{name}__ma5", ma5),
                    self._block(f"{name}__ma20", ma20),
                    self._block(f"{name}__ma60", ma60),
                    self._block(f"{name}__sd20", sd20),
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
        return panel.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).astype(np.float32), assets

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

    def _long(self, panel, target=None, assets=None):
        x = panel.stack(level="asset", future_stack=True).replace([np.inf, -np.inf], np.nan)
        if target is None:
            return x
        y = self._target(target, panel.index, assets).stack(future_stack=True)
        x, y = x.align(y, join="inner", axis=0)
        valid = y.notna()
        return x.loc[valid], y.loc[valid].astype(np.float32)

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

    def _finish(self, pred):
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred = pred.clip(-self.prediction_clip_, self.prediction_clip_)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_ = False, None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            panel, assets = self._features(features)
            self.assets_ = assets
            x, y = self._long(panel, target, assets)
            del panel
            if x.empty or len(y) < 40:
                return self
            self.selected_features_ = list(x.columns[:self.max_features])
            n = len(x)
            if n <= self.max_train_rows:
                positions = np.arange(n, dtype=np.int64)
            else:
                recent = int(self.max_train_rows * 0.60)
                start = n - recent
                old = np.linspace(0, start - 1, self.max_train_rows - recent, dtype=np.int64)
                positions = np.unique(np.concatenate([old, np.arange(start, n, dtype=np.int64)]))
            xs, ys = x.iloc[positions][self.selected_features_], y.iloc[positions]
            self._fit_preprocessor(xs)
            matrix = self._transform(xs)
            ages = (n - 1) - positions.astype(np.float32)
            weights = np.exp(-ages / max(1.0, n * self._DECAY)).astype(np.float32)
            weights /= weights.mean()
            full_y = np.nan_to_num(ys.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            main_limit = np.nanquantile(np.abs(full_y), 0.995) if full_y.size else 1.0
            if not np.isfinite(main_limit) or main_limit <= 0:
                main_limit = 1.0
            main_y = np.clip(full_y, -main_limit, main_limit)
            tail_limit = np.nanquantile(np.abs(full_y), 0.9995) if full_y.size else main_limit
            if not np.isfinite(tail_limit) or tail_limit <= main_limit:
                tail_limit = main_limit
            tail_y = np.clip(full_y, -tail_limit, tail_limit)
            self.prediction_clip_ = float(np.clip(3.0 * main_limit, 1e-6, 10.0))
            self.raw_intercept_ = float(main_y.mean())
            self.raw_coef_ = self._ridge(matrix, main_y - self.raw_intercept_, weights, self._ALPHA)
            self.tail_intercept_ = float(tail_y.mean())
            self.tail_coef_ = self._ridge(matrix, tail_y - self.tail_intercept_, weights, self._TAIL_ALPHA)
            ranked = ys.groupby(level=0).rank(method="average", pct=True) if isinstance(ys.index, pd.MultiIndex) else ys.rank(method="average", pct=True)
            rv = ((ranked - 0.5) * 2.0).to_numpy(dtype=np.float32)
            self.rank_intercept_ = float(rv.mean())
            self.rank_coef_ = self._ridge(matrix, rv - self.rank_intercept_, weights, self._RANK_ALPHA)
            self.rank_scale_ = float(np.std(main_y) / max(np.std(rv), 1e-8))
            self.xgb_model_ = xgb.train(
                self._XGB,
                xgb.DMatrix(matrix, label=main_y, weight=weights),
                num_boost_round=self.n_xgb_rounds,
            )
            self.is_trained_ = True
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
        return self

    def predict(self, features):
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            panel, assets = self._features(features)
            if not self.is_trained_ or not self.selected_features_:
                return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
            x = self._long(panel)
            del panel
            if x.empty:
                return pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
            matrix = self._transform(x)
            main = self.raw_intercept_ + matrix @ self.raw_coef_
            tail = self.tail_intercept_ + matrix @ self.tail_coef_
            tree = self.xgb_model_.predict(xgb.DMatrix(matrix)).astype(np.float32)
            rank = self.rank_scale_ * (self.rank_intercept_ + matrix @ self.rank_coef_)
            raw = (
                self._MAIN_WEIGHT * main
                + self._TAIL_WEIGHT * tail
                + self._XGB_WEIGHT * tree
                + self._RANK_WEIGHT * rank
            )
            series = pd.Series(np.nan_to_num(raw), index=x.index, dtype=np.float32)
            pred = series.unstack(level=-1).reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finish(pred)
        except Exception:
            return pd.DataFrame(0.0, index=features.index, columns=self.assets_, dtype=np.float32)
