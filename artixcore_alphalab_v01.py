import numpy as np
import pandas as pd
import xgboost as xgb

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """
    Artixcore AlphaLab v0.2
    Hybrid leakage-safe walk-forward cross-sectional signal forecaster.

    Ridge + Huber + rank linear ensemble blended with conservative XGBoost.
    Chronological timestamp validation with lag-aware Sharpe model selection.
    """

    _ALPHA_CANDIDATES = (2.0, 4.0, 8.0, 16.0, 32.0)
    _ENSEMBLE_PRESETS = (
        {"linear": 0.00, "xgb": 1.00, "rank": 0.00},
        {"linear": 0.25, "xgb": 0.70, "rank": 0.05},
        {"linear": 0.55, "xgb": 0.40, "rank": 0.05},
        {"linear": 0.45, "xgb": 0.45, "rank": 0.10},
    )
    _XGB_PARAMS = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 200,
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.alpha = 8.0
        self.selected_alpha_ = 8.0
        self.selected_params_ = {"alpha": 8.0}
        self.selected_model_ = "hybrid"

        self.coef_primary_ = None
        self.coef_huber_ = None
        self.coef_rank_ = None
        self.intercept_primary_ = 0.0
        self.intercept_huber_ = 0.0
        self.intercept_rank_ = 0.0
        self.xgb_model_ = None

        self.x_mean_ = None
        self.x_scale_ = None
        self.impute_values_ = None
        self.clip_low_ = None
        self.clip_high_ = None
        self.feature_columns_ = None
        self.selected_features_ = None

        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.max_train_rows = 250_000
        self.max_features = 180
        self.max_val_rows = 40_000
        self.n_xgb_rounds = 40

        self.training_error_ = None
        self.validation_score_ = None
        self.feature_count_ = 0
        self.model_weights_ = {"linear": 0.55, "xgb": 0.40, "rank": 0.05}
        self.training_rows_ = 0
        self.fallback_used_ = False
        self.use_rank_blend_ = False
        self.rank_blend_weight_ = 0.25

    def _infer_column_levels(self, columns):
        if not isinstance(columns, pd.MultiIndex):
            return None, None

        names = [str(name).lower() if name is not None else "" for name in columns.names]
        unique_counts = [
            len(pd.Index(columns.get_level_values(i)).unique())
            for i in range(columns.nlevels)
        ]

        feature_level = None
        asset_level = None
        for i, name in enumerate(names):
            if "feature" in name or "factor" in name:
                feature_level = i
            if "ticker" in name or "asset" in name or "symbol" in name:
                asset_level = i

        if feature_level is None:
            feature_level = int(np.argmin(unique_counts))

        if asset_level is None:
            remaining = [i for i in range(columns.nlevels) if i != feature_level]
            asset_level = (
                max(remaining, key=lambda i: unique_counts[i])
                if remaining
                else feature_level
            )

        return feature_level, asset_level

    def _ordered_unique(self, values):
        seen = set()
        ordered = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _get_feature_names_and_tickers(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            feature_level, asset_level = self._infer_column_levels(features.columns)
            feature_names = self._ordered_unique(
                features.columns.get_level_values(feature_level)
            )
            tickers = self._ordered_unique(
                features.columns.get_level_values(asset_level)
            )
            return list(feature_names), list(tickers), feature_level, asset_level

        return ["feature.1"], list(features.columns), None, None

    def _safe_numeric_frame(self, frame):
        out = frame.apply(pd.to_numeric, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.astype(np.float32, copy=False)

    def _extract_feature_frames(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            feature_values, asset_values, feature_level, asset_level = (
                self._get_feature_names_and_tickers(features)
            )
            frames = {}
            for feature_name in feature_values:
                selected_columns = [
                    col for col in features.columns if col[feature_level] == feature_name
                ]
                if not selected_columns:
                    continue
                block = features.loc[:, selected_columns]
                block.columns = [col[asset_level] for col in selected_columns]
                block = block.loc[:, ~pd.Index(block.columns).duplicated()]
                block = block.reindex(columns=asset_values)
                frames[str(feature_name)] = self._safe_numeric_frame(block)
            return frames, list(asset_values)

        asset_values = list(features.columns)
        return {"feature.1": self._safe_numeric_frame(features)}, asset_values

    def _rolling_stat(self, frame, window, min_periods, stat):
        roller = frame.rolling(window=window, min_periods=min_periods)
        if stat == "mean":
            return roller.mean()
        if stat == "std":
            return roller.std(ddof=0)
        return roller.mean()

    def _lag_frame(self, frame, lag):
        return frame.shift(lag)

    def _ewma_frame(self, frame, span):
        return frame.ewm(span=span, adjust=False, min_periods=2).mean()

    def _cross_sectional_rank(self, frame, n_assets):
        rank = frame.rank(axis=1, method="average")
        return ((rank - 0.5 * (n_assets + 1)) / (0.5 * max(n_assets - 1, 1))).astype(
            np.float32, copy=False
        )

    def _cross_sectional_demean_median(self, frame):
        return frame.sub(frame.median(axis=1), axis=0).astype(np.float32, copy=False)

    def _as_engineered_block(self, name, frame):
        block = frame.replace([np.inf, -np.inf], np.nan)
        block.columns = pd.MultiIndex.from_product(
            [[name], block.columns], names=["feature", "asset"]
        )
        return block.astype(np.float32, copy=False)

    def _broadcast_scalar(self, values, frame):
        arr = np.repeat(values.to_numpy(dtype=np.float32)[:, None], frame.shape[1], axis=1)
        return pd.DataFrame(arr, index=frame.index, columns=frame.columns, dtype=np.float32)

    def _build_feature_blocks(self, frames, assets):
        n_assets = len(assets)
        blocks = []
        feature_names = list(frames.keys())

        for fi, fname in enumerate(feature_names):
            raw = self._safe_numeric_frame(frames[fname])
            cs_rank = self._cross_sectional_rank(raw, n_assets)
            cs_demean = self._cross_sectional_demean_median(raw)

            roll_mean_5 = self._rolling_stat(raw, 5, 2, "mean")
            roll_mean_20 = self._rolling_stat(raw, 20, 3, "mean")
            roll_mean_60 = self._rolling_stat(raw, 60, 5, "mean")
            roll_std_20 = self._rolling_stat(raw, 20, 3, "std").fillna(0.0)
            ewma_5 = self._ewma_frame(raw, 5)

            blocks.extend(
                [
                    self._as_engineered_block(f"{fname}__raw", raw),
                    self._as_engineered_block(f"{fname}__cs_rank", cs_rank),
                    self._as_engineered_block(f"{fname}__cs_demean", cs_demean),
                    self._as_engineered_block(f"{fname}__raw_x_rank", raw * cs_rank),
                    self._as_engineered_block(f"{fname}__ma5", roll_mean_5),
                    self._as_engineered_block(f"{fname}__ma20", roll_mean_20),
                    self._as_engineered_block(f"{fname}__sd20", roll_std_20),
                ]
            )

            if fi == 0:
                blocks.extend(
                    [
                        self._as_engineered_block(f"{fname}__ma60", roll_mean_60),
                        self._as_engineered_block(f"{fname}__ewma5", ewma_5),
                        self._as_engineered_block(
                            f"{fname}__ma5_rank", self._cross_sectional_rank(roll_mean_5, n_assets)
                        ),
                        self._as_engineered_block(
                            f"{fname}__ma20_rank", self._cross_sectional_rank(roll_mean_20, n_assets)
                        ),
                        self._as_engineered_block(
                            f"{fname}__ma60_rank", self._cross_sectional_rank(roll_mean_60, n_assets)
                        ),
                    ]
                )

            if fi < 3:
                for lag in (1, 2, 5):
                    blocks.append(
                        self._as_engineered_block(
                            f"{fname}__diff_{lag}", raw - self._lag_frame(raw, lag)
                        )
                    )
                roll_z = (raw - roll_mean_5) / (roll_std_20 + 1.0e-6)
                diff_1 = raw - self._lag_frame(raw, 1)
                vol_adj = diff_1 / (roll_std_20 + 1.0e-6)
                rank_lag_1 = self._lag_frame(cs_rank, 1)
                blocks.extend(
                    [
                        self._as_engineered_block(f"{fname}__roll_z", roll_z),
                        self._as_engineered_block(f"{fname}__vol_adj", vol_adj),
                        self._as_engineered_block(
                            f"{fname}__mom_spread", roll_mean_5 - roll_mean_60
                        ),
                        self._as_engineered_block(f"{fname}__rank_chg1", cs_rank - rank_lag_1),
                    ]
                )
            else:
                blocks.append(
                    self._as_engineered_block(
                        f"{fname}__diff_1", raw - self._lag_frame(raw, 1)
                    )
                )

        if feature_names:
            primary = self._safe_numeric_frame(frames[feature_names[0]])
            prefix = feature_names[0]
            row_mean = primary.mean(axis=1)
            row_std = primary.std(axis=1, ddof=0).replace(0.0, np.nan)
            pct_above = (primary.gt(primary.median(axis=1), axis=0)).mean(axis=1)
            blocks.extend(
                [
                    self._as_engineered_block(
                        f"{prefix}__uni_mean", self._broadcast_scalar(row_mean, primary)
                    ),
                    self._as_engineered_block(
                        f"{prefix}__uni_std", self._broadcast_scalar(row_std.fillna(0.0), primary)
                    ),
                    self._as_engineered_block(
                        f"{prefix}__uni_pct_above",
                        self._broadcast_scalar(pct_above, primary),
                    ),
                ]
            )

        if len(feature_names) >= 2:
            f1 = frames[feature_names[0]]
            f2 = frames[feature_names[1]]
            rank_spread = self._cross_sectional_rank(f1, n_assets) - self._cross_sectional_rank(
                f2, n_assets
            )
            blocks.append(
                self._as_engineered_block("interaction__rank_spread", rank_spread)
            )

        return blocks, assets

    def _make_features(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        frames, assets = self._extract_feature_frames(features)
        blocks, assets = self._build_feature_blocks(frames, assets)
        if not blocks:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty

        engineered = pd.concat(blocks, axis=1)
        feat_names = engineered.columns.get_level_values("feature").unique()
        engineered = engineered.reindex(
            columns=pd.MultiIndex.from_product([feat_names, assets], names=["feature", "asset"])
        )
        return engineered.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)

    def _prepare_target_frame(self, target, index, assets):
        if isinstance(target, pd.Series):
            target_frame = (
                target.unstack(level=-1)
                if isinstance(target.index, pd.MultiIndex)
                else target.to_frame()
            )
        elif isinstance(target, pd.DataFrame):
            target_frame = target.copy()
        else:
            target_frame = pd.DataFrame(target, index=index)

        if isinstance(target_frame.columns, pd.MultiIndex):
            _, asset_level = self._infer_column_levels(target_frame.columns)
            target_frame.columns = target_frame.columns.get_level_values(asset_level)

        return self._safe_numeric_frame(target_frame.reindex(index=index, columns=assets))

    def _panel_to_matrix(self, engineered_features, target=None):
        assets = list(engineered_features.columns.get_level_values("asset").unique())
        x_long = engineered_features.stack(level="asset", future_stack=True)
        x_long = x_long.replace([np.inf, -np.inf], np.nan)

        if target is None:
            return x_long, None, assets

        target_frame = self._prepare_target_frame(target, engineered_features.index, assets)
        y_long = target_frame.stack(future_stack=True)
        x_long, y_long = x_long.align(y_long, join="inner", axis=0)

        mask = y_long.replace([np.inf, -np.inf], np.nan).notna()
        x_long = x_long.loc[mask]
        y_long = y_long.loc[mask].astype(np.float32, copy=False)
        return x_long, y_long, assets

    def _timestamp_split(self, index, train_fraction=0.80):
        n_ts = len(index)
        if n_ts < 5:
            split = max(1, n_ts - 1)
        else:
            split = max(1, min(n_ts - 1, int(n_ts * train_fraction)))
        return index[:split], index[split:]

    def _split_panel_by_timestamps(self, x_long, y_long, early_idx, late_idx):
        if isinstance(x_long.index, pd.MultiIndex):
            ts_level = 0
            early_mask = x_long.index.get_level_values(ts_level).isin(early_idx)
            late_mask = x_long.index.get_level_values(ts_level).isin(late_idx)
        else:
            early_mask = x_long.index.isin(early_idx)
            late_mask = x_long.index.isin(late_idx)

        return x_long.loc[early_mask], y_long.loc[early_mask], x_long.loc[late_mask], y_long.loc[
            late_mask
        ]

    def _filter_features(self, x_train, y_train):
        if x_train.empty:
            return x_train

        columns = list(x_train.columns)
        keep = []
        x_values = x_train.to_numpy(dtype=np.float64, copy=False)
        y_values = y_train.to_numpy(dtype=np.float64, copy=False)

        for j, col in enumerate(columns):
            col_values = x_values[:, j]
            finite = np.isfinite(col_values)
            if finite.mean() < 0.05:
                continue
            valid = col_values[finite]
            if valid.size == 0 or np.nanstd(valid) < 1.0e-8:
                continue
            keep.append(col)

        if not keep:
            return x_train.iloc[:, :0]

        filtered = x_train.loc[:, keep]
        scores = []
        for col in filtered.columns:
            col_values = filtered[col].to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(col_values) & np.isfinite(y_values)
            if mask.sum() < 10:
                scores.append((col, 0.0))
                continue
            corr = np.corrcoef(col_values[mask], y_values[mask])[0, 1]
            scores.append((col, abs(corr) if np.isfinite(corr) else 0.0))

        scores.sort(key=lambda item: item[1], reverse=True)
        selected = [col for col, _ in scores[: self.max_features]]
        if len(selected) < min(20, len(scores)):
            for col, _ in scores:
                if col not in selected:
                    selected.append(col)
                if len(selected) >= min(20, self.max_features):
                    break
        return filtered.loc[:, selected]

    def _fit_preprocessor(self, x_train):
        x = x_train.to_numpy(dtype=np.float64, copy=True)
        x = np.where(np.isfinite(x), x, np.nan)

        self.impute_values_ = np.nanmedian(x, axis=0)
        self.impute_values_ = np.where(
            np.isfinite(self.impute_values_), self.impute_values_, 0.0
        )

        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]

        self.clip_low_ = np.nanquantile(x, 0.005, axis=0)
        self.clip_high_ = np.nanquantile(x, 0.995, axis=0)
        invalid = ~np.isfinite(self.clip_low_) | ~np.isfinite(self.clip_high_) | (
            self.clip_low_ >= self.clip_high_
        )
        self.clip_low_[invalid] = -10.0
        self.clip_high_[invalid] = 10.0
        x = np.clip(x, self.clip_low_, self.clip_high_)

        self.x_mean_ = x.mean(axis=0)
        centered = x - self.x_mean_
        mad = np.nanmedian(np.abs(centered), axis=0)
        self.x_scale_ = 1.4826 * mad
        fallback = x.std(axis=0)
        bad = ~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)
        self.x_scale_[bad] = fallback[bad]
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)] = 1.0

    def _transform_matrix(self, x_data):
        if self.impute_values_ is None:
            x = x_data.to_numpy(dtype=np.float64, copy=True)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = x_data.to_numpy(dtype=np.float64, copy=True)
        x = np.where(np.isfinite(x), x, np.nan)
        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]
        x = np.clip(x, self.clip_low_, self.clip_high_)
        x = (x - self.x_mean_) / self.x_scale_
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_imputation_frame(self, x_data):
        out = x_data.reindex(columns=self.selected_features_).copy()
        for j, col in enumerate(self.selected_features_):
            if col not in out.columns:
                out[col] = self.impute_values_[j]
            else:
                out[col] = out[col].fillna(self.impute_values_[j])
        return out[self.selected_features_]

    def _solve_ridge(self, x_scaled, y_centered, alpha, sample_weights=None):
        if sample_weights is None:
            gram = x_scaled.T @ x_scaled
            rhs = x_scaled.T @ y_centered
        else:
            sw = np.sqrt(sample_weights)
            xw = x_scaled * sw[:, None]
            yw = y_centered * sw
            gram = xw.T @ xw
            rhs = xw.T @ yw

        gram = gram.copy()
        gram.flat[:: gram.shape[0] + 1] += alpha
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ rhs

    def _fit_ridge_core(self, x_train, y_train, alpha, sample_weights=None):
        x_scaled = self._transform_matrix(x_train)
        y = np.nan_to_num(y_train.to_numpy(dtype=np.float64, copy=True), nan=0.0)
        y_mean = float(y.mean()) if y.size else 0.0
        coef = self._solve_ridge(x_scaled, y - y_mean, alpha, sample_weights)
        return coef, y_mean

    def _fit_huber_ridge(self, x_train, y_train, alpha, sample_weights=None, n_iter=2, delta=1.0):
        coef, intercept = self._fit_ridge_core(x_train, y_train, alpha, sample_weights)
        y = np.nan_to_num(y_train.to_numpy(dtype=np.float64, copy=True), nan=0.0)
        weights = (
            np.asarray(sample_weights, dtype=np.float64)
            if sample_weights is not None
            else np.ones(len(y), dtype=np.float64)
        )

        for _ in range(n_iter):
            x_scaled = self._transform_matrix(x_train)
            residuals = y - (intercept + x_scaled @ coef)
            abs_res = np.abs(residuals)
            huber_w = np.ones_like(abs_res)
            heavy = abs_res > delta
            huber_w[heavy] = delta / np.maximum(abs_res[heavy], 1.0e-8)
            coef, intercept = self._fit_ridge_core(
                x_train, y_train, alpha * 2.0, weights * huber_w
            )
        return coef, intercept

    def _rank_targets(self, y_train):
        if isinstance(y_train.index, pd.MultiIndex):
            grouped = y_train.groupby(level=0)
            return grouped.rank(method="average", pct=True).astype(np.float32)
        return y_train.rank(method="average", pct=True).astype(np.float32)

    def _fit_rank_ridge(self, x_train, y_train, alpha, sample_weights=None):
        ranked = self._rank_targets(y_train)
        return self._fit_ridge_core(x_train, ranked, alpha, sample_weights)

    def _fit_xgb(self, x_train, y_train, sample_weights=None):
        x_scaled = self._transform_matrix(x_train).astype(np.float32, copy=False)
        y = y_train.to_numpy(dtype=np.float32, copy=False)
        dtrain = xgb.DMatrix(x_scaled, label=y, weight=sample_weights)
        return xgb.train(self._XGB_PARAMS, dtrain, num_boost_round=self.n_xgb_rounds)

    def _predict_linear(self, x_data, coef, intercept):
        if coef is None:
            return np.zeros(len(x_data), dtype=np.float64)
        return intercept + self._transform_matrix(x_data) @ coef

    def _predict_xgb(self, x_data):
        if self.xgb_model_ is None:
            return np.zeros(len(x_data), dtype=np.float64)
        x_scaled = self._transform_matrix(x_data).astype(np.float32, copy=False)
        return self.xgb_model_.predict(xgb.DMatrix(x_scaled))

    def _long_preds_to_matrix(self, preds, index, assets):
        series = pd.Series(preds, index=index, dtype=np.float64)
        if isinstance(series.index, pd.MultiIndex):
            matrix = series.unstack(level=-1)
            row_index = matrix.index
        else:
            matrix = series.to_frame()
            row_index = matrix.index
        return matrix.reindex(index=row_index, columns=assets).fillna(0.0)

    def _target_long_to_matrix(self, y_long, assets):
        if isinstance(y_long.index, pd.MultiIndex):
            return y_long.unstack(level=-1).reindex(columns=assets).fillna(0.0)
        return y_long.to_frame().reindex(columns=assets).fillna(0.0)

    def _lag_sharpe_from_long(self, pred_arr, y_long, assets):
        pred_series = pd.Series(pred_arr, index=y_long.index, dtype=np.float64)
        pred_matrix = self._long_preds_to_matrix(pred_series, y_long.index, assets)
        target_matrix = self._target_long_to_matrix(y_long, assets)

        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0).fillna(0.0)
        pf = (pred_matrix.shift(1) * target_matrix).sum(axis=1).iloc[1:]
        if len(pf) < 5 or pf.std() == 0:
            return -1.0
        return float(pf.mean() / pf.std())

    def _cross_sectional_ic_score(self, y_long, pred_long):
        aligned = pd.concat(
            [y_long.rename("y"), pd.Series(pred_long, index=y_long.index).rename("p")],
            axis=1,
        ).dropna()
        if aligned.empty:
            return -1.0, 0.0

        corrs = []
        if isinstance(aligned.index, pd.MultiIndex):
            groups = aligned.index.get_level_values(0)
            for g in pd.unique(groups):
                block = aligned.loc[groups == g]
                if len(block) < 3:
                    continue
                yv, pv = block["y"].values, block["p"].values
                if np.nanstd(yv) < 1e-12 or np.nanstd(pv) < 1e-12:
                    continue
                c = np.corrcoef(yv, pv)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        else:
            c = np.corrcoef(aligned["y"], aligned["p"])[0, 1]
            if np.isfinite(c):
                corrs.append(c)

        if not corrs:
            return -1.0, 0.0
        return float(np.mean(corrs)), float(np.std(corrs)) if len(corrs) > 1 else 0.0

    def _validation_score(self, pred_long, y_long, assets):
        if isinstance(pred_long, pd.Series):
            pred_arr = pred_long.reindex(y_long.index).to_numpy(dtype=np.float64)
        else:
            pred_arr = np.asarray(pred_long, dtype=np.float64)
        lag_sh = self._lag_sharpe_from_long(pred_arr, y_long, assets)
        ic_mean, ic_std = self._cross_sectional_ic_score(y_long, pred_arr)
        pred_matrix = self._long_preds_to_matrix(
            pd.Series(pred_arr, index=y_long.index), y_long.index, assets
        )
        row_std = pred_matrix.std(axis=1, ddof=0)
        flat_penalty = 0.5 if row_std.median() < 1e-8 else 0.0
        return lag_sh + 0.25 * ic_mean - 0.15 * ic_std - flat_penalty

    def _ensemble_predict_long(self, x_data, coef_p, int_p, coef_h, int_h, coef_r, int_r, weights):
        pred = np.zeros(len(x_data), dtype=np.float64)
        w_lin = weights.get("linear", 0.55)
        w_xgb = weights.get("xgb", 0.40)
        w_rank = weights.get("rank", 0.05)

        if coef_p is not None:
            pred += w_lin * 0.70 * self._predict_linear(x_data, coef_p, int_p)
        if coef_h is not None:
            pred += w_lin * 0.30 * self._predict_linear(x_data, coef_h, int_h)
        if coef_r is not None and w_rank > 0:
            pred += w_rank * self._predict_linear(x_data, coef_r, int_r)
        if self.xgb_model_ is not None and w_xgb > 0:
            pred += w_xgb * self._predict_xgb(x_data)
        return pred

    def _sample_training_rows(self, x_train, y_train):
        n_rows = len(x_train)
        if n_rows <= self.max_train_rows:
            return x_train, y_train, np.arange(n_rows, dtype=np.int64)

        recent_count = int(self.max_train_rows * 0.60)
        older_budget = self.max_train_rows - recent_count
        recent_start = n_rows - recent_count
        recent_positions = np.arange(recent_start, n_rows, dtype=np.int64)

        if older_budget <= 0 or recent_start <= 0:
            positions = recent_positions
        else:
            older_positions = np.linspace(0, recent_start - 1, older_budget, dtype=np.int64)
            positions = np.unique(np.concatenate([older_positions, recent_positions]))

        return x_train.iloc[positions], y_train.iloc[positions], positions

    def _compute_sample_weights(self, positions, total_rows):
        if total_rows <= 1:
            return np.ones(len(positions), dtype=np.float64)
        ages = (total_rows - 1) - positions.astype(np.float64)
        halflife = max(1.0, total_rows * 0.25)
        weights = np.exp(-ages / halflife)
        return (weights / np.mean(weights)).astype(np.float64)

    def _subsample_long_panel(self, x_long, y_long, max_rows=None):
        max_rows = max_rows or self.max_val_rows
        n_rows = len(x_long)
        if n_rows <= max_rows:
            return x_long, y_long
        positions = np.linspace(0, n_rows - 1, max_rows, dtype=np.int64)
        return x_long.iloc[positions], y_long.iloc[positions]

    def _select_alpha(self, x_early, y_early, x_late, y_late, assets):
        x_e, y_e = self._subsample_long_panel(x_early, y_early)
        x_l, y_l = self._subsample_long_panel(x_late, y_late)
        best_alpha = 8.0
        best_score = -np.inf
        for alpha in self._ALPHA_CANDIDATES:
            coef, intercept = self._fit_ridge_core(x_e, y_e, alpha)
            preds = self._predict_linear(x_l, coef, intercept)
            score = self._validation_score(preds, y_l, assets)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        return best_alpha, best_score

    def _select_ensemble_weights(
        self, x_early, y_early, x_late, y_late, assets, alpha, xgb_model
    ):
        x_e, y_e = self._subsample_long_panel(x_early, y_early)
        x_l, y_l = self._subsample_long_panel(x_late, y_late)

        coef_p, int_p = self._fit_ridge_core(x_e, y_e, alpha)
        coef_h, int_h = self._fit_huber_ridge(x_e, y_e, alpha)
        coef_r, int_r = self._fit_rank_ridge(x_e, y_e, alpha)

        saved_xgb = self.xgb_model_
        self.xgb_model_ = xgb_model

        best_weights = self._ENSEMBLE_PRESETS[0]
        best_score = -np.inf
        for preset in self._ENSEMBLE_PRESETS:
            weights = dict(preset)
            preds = self._ensemble_predict_long(
                x_l, coef_p, int_p, coef_h, int_h, coef_r, int_r, weights
            )
            score = self._validation_score(preds, y_l, assets)
            if score > best_score:
                best_score = score
                best_weights = weights

        self.xgb_model_ = saved_xgb
        return best_weights, best_score

    def _zero_prediction(self, features):
        _, assets = self._extract_feature_frames(features)
        pred = pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

    def _finalize_prediction(self, pred):
        pred = pred.astype(np.float64, copy=False)
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clip_value = float(np.clip(self.prediction_clip_, 1.0e-6, 10.0))
        pred = pred.clip(lower=-clip_value, upper=clip_value)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred.astype(np.float32, copy=False)

    def _apply_rank_blend(self, pred):
        if not self.use_rank_blend_:
            return pred
        rank_signal = pred.rank(axis=1, pct=True).sub(0.5)
        w = self.rank_blend_weight_
        blended = (1.0 - w) * pred + w * rank_signal
        return blended.fillna(0.0)

    def _fallback_prediction(self, features):
        try:
            frames, assets = self._extract_feature_frames(features)
            if not frames:
                return self._zero_prediction(features)
            fname = next(iter(frames))
            primary = self._safe_numeric_frame(frames[fname])
            n_assets = len(assets)
            score = self._cross_sectional_rank(primary, n_assets)
            mom = self._rolling_stat(primary, 5, 2, "mean")
            score = score + 0.3 * self._cross_sectional_rank(mom, n_assets)
            pred = score.reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finalize_prediction(pred)
        except Exception:
            return self._zero_prediction(features)

    def train(self, features, target):
        self.training_error_ = None
        self.is_trained_ = False
        self.fallback_used_ = False
        self.use_rank_blend_ = False
        self.validation_score_ = None
        self.xgb_model_ = None
        self.model_weights_ = dict(self._ENSEMBLE_PRESETS[0])

        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            engineered = self._make_features(features)
            x_all, y_all, assets = self._panel_to_matrix(engineered, target)
            del engineered

            if x_all.empty or len(y_all) < 40:
                self.selected_features_ = list(x_all.columns)
                self.feature_columns_ = self.selected_features_
                self.feature_count_ = len(self.selected_features_ or [])
                return self

            early_idx, late_idx = self._timestamp_split(features.index)
            x_early, y_early, x_late, y_late = self._split_panel_by_timestamps(
                x_all, y_all, early_idx, late_idx
            )

            if len(x_early) < 20 or len(x_late) < 10:
                x_early, y_early = x_all, y_all
                x_late, y_late = x_all.iloc[:0], y_all.iloc[:0]

            x_early_fit, y_early_fit = self._subsample_long_panel(x_early, y_early)
            filtered = self._filter_features(x_early_fit, y_early_fit)
            if filtered.empty or filtered.shape[1] == 0:
                return self

            self.selected_features_ = list(filtered.columns)
            self.feature_columns_ = self.selected_features_
            self._fit_preprocessor(filtered)

            x_early = self._apply_imputation_frame(x_early)
            x_late = self._apply_imputation_frame(x_late) if len(x_late) else x_late
            x_all_f = self._apply_imputation_frame(x_all)

            selected_alpha, alpha_score = self._select_alpha(
                x_early, y_early, x_late, y_late, assets
            )
            self.alpha = selected_alpha
            self.selected_alpha_ = selected_alpha
            self.selected_params_ = {"alpha": selected_alpha}

            xgb_early = self._fit_xgb(
                *self._subsample_long_panel(x_early, y_early)
            )
            if len(x_late):
                tuned_weights, val_score = self._select_ensemble_weights(
                    x_early, y_early, x_late, y_late, assets, selected_alpha, xgb_early
                )
                self.model_weights_ = tuned_weights
                self.validation_score_ = val_score
            else:
                self.validation_score_ = alpha_score

            x_sampled, y_sampled, positions = self._sample_training_rows(x_all_f, y_all)
            sample_weights = self._compute_sample_weights(positions, len(x_all_f))
            self.training_rows_ = len(x_sampled)

            abs_y = np.abs(y_sampled.to_numpy(dtype=np.float64, copy=False))
            q = np.nanquantile(abs_y, 0.995) if abs_y.size else 1.0
            self.prediction_clip_ = float(np.clip(3.0 * q if np.isfinite(q) and q > 0 else 1.0, 1e-6, 10.0))

            weights = self.model_weights_
            if weights.get("xgb", 0.0) >= 0.99:
                self.coef_primary_ = None
                self.coef_huber_ = None
                self.coef_rank_ = None
                self.xgb_model_ = self._fit_xgb(x_sampled, y_sampled, sample_weights)
            else:
                self.coef_primary_, self.intercept_primary_ = self._fit_ridge_core(
                    x_sampled, y_sampled, selected_alpha, sample_weights
                )
                self.coef_huber_, self.intercept_huber_ = self._fit_huber_ridge(
                    x_sampled, y_sampled, selected_alpha, sample_weights
                )
                self.coef_rank_, self.intercept_rank_ = self._fit_rank_ridge(
                    x_sampled, y_sampled, selected_alpha, sample_weights
                )
                self.xgb_model_ = self._fit_xgb(x_sampled, y_sampled, sample_weights)
            self.feature_count_ = len(self.selected_features_)

            if len(x_late) >= 10:
                raw_preds = self._ensemble_predict_long(
                    x_late,
                    self.coef_primary_,
                    self.intercept_primary_,
                    self.coef_huber_,
                    self.intercept_huber_,
                    self.coef_rank_,
                    self.intercept_rank_,
                    self.model_weights_,
                )
                base_score = self._validation_score(raw_preds, y_late, assets)
                pred_frame = self._long_preds_to_matrix(raw_preds, y_late.index, assets)
                blended = self._apply_rank_blend(
                    pred_frame.sub(pred_frame.mean(axis=1), axis=0).fillna(0.0)
                )
                blend_long = blended.stack(future_stack=True)
                blend_score = self._validation_score(blend_long, y_late, assets)
                if blend_score > base_score:
                    self.use_rank_blend_ = True

            self.is_trained_ = True
            return self
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
            return self

    def predict(self, features):
        self.fallback_used_ = False

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            if not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback_prediction(features)

            engineered = self._make_features(features)
            x_long, _, assets = self._panel_to_matrix(engineered, target=None)
            del engineered

            if x_long.empty:
                self.fallback_used_ = True
                return self._fallback_prediction(features)

            x_long = self._apply_imputation_frame(x_long)
            raw_pred = self._ensemble_predict_long(
                x_long,
                self.coef_primary_,
                self.intercept_primary_,
                self.coef_huber_,
                self.intercept_huber_,
                self.coef_rank_,
                self.intercept_rank_,
                self.model_weights_,
            )
            raw_pred = np.nan_to_num(raw_pred, nan=0.0, posinf=0.0, neginf=0.0)

            pred = self._long_preds_to_matrix(raw_pred, x_long.index, assets)
            pred = pred.reindex(index=features.index, columns=assets).fillna(0.0)
            pred = self._apply_rank_blend(pred)
            return self._finalize_prediction(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback_prediction(features)
