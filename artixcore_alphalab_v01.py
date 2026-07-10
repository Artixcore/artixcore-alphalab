import numpy as np
import pandas as pd

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """
    Artixcore AlphaLab v0.1
    Leakage-safe walk-forward cross-sectional signal forecasting model.

    Combines causal feature engineering, training-only preprocessing,
    chronological validation, and a compact Ridge-based ensemble.
    """

    _ALPHA_CANDIDATES = (2.0, 4.0, 8.0, 16.0, 32.0)
    _DEFAULT_ENSEMBLE_WEIGHTS = {"primary": 0.70, "huber": 0.25, "rank": 0.05}

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.alpha = 8.0
        self.selected_alpha_ = 8.0
        self.coef_ = None
        self.coef_primary_ = None
        self.coef_huber_ = None
        self.coef_rank_ = None
        self.intercept_ = 0.0
        self.intercept_primary_ = 0.0
        self.intercept_huber_ = 0.0
        self.intercept_rank_ = 0.0
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
        self.max_features = 600
        self.training_error_ = None
        self.validation_score_ = None
        self.feature_count_ = 0
        self.model_weights_ = dict(self._DEFAULT_ENSEMBLE_WEIGHTS)
        self.training_rows_ = 0
        self.fallback_used_ = False
        self.use_rank_blend_ = False
        self.rank_blend_weight_ = 0.30

    def _infer_column_levels(self, columns):
        if not isinstance(columns, pd.MultiIndex):
            return None, None

        names = [str(name).lower() if name is not None else "" for name in columns.names]
        unique_counts = [len(pd.Index(columns.get_level_values(i)).unique()) for i in range(columns.nlevels)]

        feature_level = None
        asset_level = None

        for i, name in enumerate(names):
            if "feature" in name or "factor" in name:
                feature_level = i
            if "ticker" in name or "asset" in name or "symbol" in name:
                asset_level = i

        if feature_level is None:
            feature_like_scores = []
            for i in range(columns.nlevels):
                values = pd.Index(columns.get_level_values(i)).astype(str).str.lower()
                score = int(values.str.contains("feature|factor|signal").sum())
                feature_like_scores.append(score)
            if max(feature_like_scores) > 0:
                feature_level = int(np.argmax(feature_like_scores))

        if feature_level is None:
            feature_level = int(np.argmin(unique_counts))

        if asset_level is None:
            remaining = [i for i in range(columns.nlevels) if i != feature_level]
            if remaining:
                asset_level = max(remaining, key=lambda i: unique_counts[i])
            else:
                asset_level = feature_level

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
            feature_names = self._ordered_unique(features.columns.get_level_values(feature_level))
            tickers = self._ordered_unique(features.columns.get_level_values(asset_level))
            return list(feature_names), list(tickers), feature_level, asset_level

        tickers = list(features.columns)
        return ["feature.1"], tickers, None, None

    def _safe_numeric_frame(self, frame):
        out = frame.apply(pd.to_numeric, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.astype(np.float32, copy=False)

    def _extract_feature_frames(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            feature_values, asset_values, feature_level, asset_level = self._get_feature_names_and_tickers(
                features
            )
            frames = {}

            for feature_name in feature_values:
                selected_columns = [
                    col for col in features.columns if col[feature_level] == feature_name
                ]
                if not selected_columns:
                    continue

                block = features.loc[:, selected_columns].copy()
                renamed_columns = [col[asset_level] for col in selected_columns]
                block.columns = renamed_columns
                block = block.loc[:, ~pd.Index(block.columns).duplicated()]
                block = block.reindex(columns=asset_values)
                frames[str(feature_name)] = self._safe_numeric_frame(block)

            return frames, list(asset_values)

        asset_values = list(features.columns)
        frames = {"feature.1": self._safe_numeric_frame(features.copy())}
        return frames, asset_values

    def _infer_period_groups(self, index):
        if not isinstance(index, pd.MultiIndex) or len(index) == 0:
            return None

        names = [str(name).lower() if name is not None else "" for name in index.names]
        candidate_levels = []

        for i, name in enumerate(names):
            if "period" in name or "fold" in name or "era" in name or "window" in name:
                candidate_levels.append(i)

        if not candidate_levels and index.nlevels >= 2:
            first_values = pd.Index(index.get_level_values(0))
            counts = first_values.value_counts(dropna=False)
            if 1 < len(counts) <= max(1, len(index) // 5) and counts.max() > 1:
                candidate_levels.append(0)

        for level in candidate_levels:
            values = np.asarray(index.get_level_values(level))
            if len(pd.Index(values).unique()) < len(values):
                return values

        return None

    def _apply_by_period(self, frame, compute_fn):
        groups = self._infer_period_groups(frame.index)
        if groups is None:
            return compute_fn(frame)

        result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=np.float32)
        group_values = pd.Series(groups)

        for group_value in pd.unique(group_values):
            positions = np.flatnonzero(group_values.to_numpy() == group_value)
            if positions.size == 0:
                continue
            block = frame.iloc[positions]
            computed = compute_fn(block)
            result.iloc[positions, :] = computed.to_numpy(dtype=np.float32, copy=False)

        return result

    def _rolling_stat(self, frame, window, min_periods, stat):
        def compute(block):
            roller = block.rolling(window=window, min_periods=min_periods)
            if stat == "mean":
                return roller.mean()
            if stat == "median":
                return roller.median()
            return roller.std(ddof=0)

        return self._apply_by_period(frame, compute)

    def _lag_frame(self, frame, lag):
        def compute(block):
            return block.shift(lag)

        return self._apply_by_period(frame, compute)

    def _ewma_frame(self, frame, span):
        def compute(block):
            return block.ewm(span=span, adjust=False, min_periods=2).mean()

        return self._apply_by_period(frame, compute)

    def _cross_sectional_rank(self, frame):
        return frame.rank(axis=1, pct=True).sub(0.5).astype(np.float32, copy=False)

    def _cross_sectional_zscore(self, frame):
        row_mean = frame.mean(axis=1)
        row_std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
        return frame.sub(row_mean, axis=0).div(row_std, axis=0).astype(np.float32, copy=False)

    def _cross_sectional_demean_median(self, frame):
        row_median = frame.median(axis=1)
        return frame.sub(row_median, axis=0).astype(np.float32, copy=False)

    def _as_engineered_block(self, name, frame):
        block = frame.replace([np.inf, -np.inf], np.nan)
        block.columns = pd.MultiIndex.from_product([[name], block.columns], names=["feature", "asset"])
        return block.astype(np.float32, copy=False)

    def _broadcast_scalar_to_frame(self, values, frame):
        return pd.DataFrame(
            np.repeat(values.to_numpy(dtype=np.float32)[:, None], frame.shape[1], axis=1),
            index=frame.index,
            columns=frame.columns,
            dtype=np.float32,
        )

    def _broadcast_universe_features(self, primary, prefix):
        row_mean = primary.mean(axis=1)
        row_median = primary.median(axis=1)
        row_std = primary.std(axis=1, ddof=0).replace(0.0, np.nan)
        pct_above = (primary.gt(row_median, axis=0)).mean(axis=1)
        mean_abs = primary.abs().mean(axis=1)

        blocks = [
            self._as_engineered_block(f"{prefix}__uni_mean", self._broadcast_scalar_to_frame(row_mean, primary)),
            self._as_engineered_block(
                f"{prefix}__uni_median", self._broadcast_scalar_to_frame(row_median, primary)
            ),
            self._as_engineered_block(f"{prefix}__uni_std", self._broadcast_scalar_to_frame(row_std, primary)),
            self._as_engineered_block(
                f"{prefix}__uni_pct_above_med", self._broadcast_scalar_to_frame(pct_above, primary)
            ),
            self._as_engineered_block(
                f"{prefix}__uni_mean_abs", self._broadcast_scalar_to_frame(mean_abs, primary)
            ),
        ]
        return blocks

    def _build_feature_blocks(self, frames, assets):
        engineered_blocks = []
        feature_names = list(frames.keys())
        primary_name = feature_names[0] if feature_names else None

        for feature_name, raw_frame in frames.items():
            raw = self._safe_numeric_frame(raw_frame)
            safe_name = str(feature_name)
            cs_rank = self._cross_sectional_rank(raw)
            cs_z = self._cross_sectional_zscore(raw)

            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__raw", raw))
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__is_nan", raw.isna().astype(np.float32))
            )
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__cs_rank", cs_rank))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__cs_z", cs_z))
            engineered_blocks.append(
                self._as_engineered_block(
                    f"{safe_name}__cs_demean_med", self._cross_sectional_demean_median(raw)
                )
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__raw_x_cs_rank", raw * cs_rank)
            )

            roll_mean_3 = self._rolling_stat(raw, 3, 2, "mean")
            roll_mean_5 = self._rolling_stat(raw, 5, 2, "mean")
            roll_mean_10 = self._rolling_stat(raw, 10, 3, "mean")
            roll_std_5 = self._rolling_stat(raw, 5, 3, "std")
            roll_std_10 = self._rolling_stat(raw, 10, 5, "std")
            roll_med_5 = self._rolling_stat(raw, 5, 3, "median")
            roll_med_10 = self._rolling_stat(raw, 10, 5, "median")
            ewma_5 = self._ewma_frame(raw, 5)

            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_mean_3", roll_mean_3))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_mean_5", roll_mean_5))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_mean_10", roll_mean_10))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_std_5", roll_std_5))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_std_10", roll_std_10))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_med_5", roll_med_5))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_med_10", roll_med_10))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__ewma_5", ewma_5))

            for lag in (1, 2, 3, 5):
                diff_frame = raw - self._lag_frame(raw, lag)
                engineered_blocks.append(
                    self._as_engineered_block(f"{safe_name}__diff_{lag}", diff_frame)
                )

            roll_z_5 = (raw - roll_mean_5) / (roll_std_5 + 1.0e-6)
            diff_1 = raw - self._lag_frame(raw, 1)
            vol_adj_1 = diff_1 / (roll_std_5 + 1.0e-6)
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__roll_z_5", roll_z_5))
            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__vol_adj_1", vol_adj_1))
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__mom_spread_3_10", roll_mean_3 - roll_mean_10)
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__minus_roll_med_5", raw - roll_med_5)
            )

            rank_lag_1 = self._lag_frame(cs_rank, 1)
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__rank_change_1", cs_rank - rank_lag_1)
            )

            z_lag_1 = self._lag_frame(cs_z, 1)
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__z_change_1", cs_z - z_lag_1)
            )

        if primary_name is not None:
            primary = self._safe_numeric_frame(frames[primary_name])
            prefix = str(primary_name)
            engineered_blocks.extend(self._broadcast_universe_features(primary, prefix))

            ret_vol_5 = self._rolling_stat(primary, 5, 3, "std")
            ret_vol_20 = self._rolling_stat(primary, 20, 5, "std")
            ret_mom_3 = self._rolling_stat(primary, 3, 2, "mean")
            ret_mom_5 = self._rolling_stat(primary, 5, 2, "mean")
            ret_mom_20 = self._rolling_stat(primary, 20, 5, "mean")

            engineered_blocks.append(
                self._as_engineered_block(f"{prefix}__mean_rev_vol5", -primary / (ret_vol_5 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{prefix}__mom5_over_vol20", ret_mom_5 / (ret_vol_20 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{prefix}__mom3_minus_mom20", ret_mom_3 - ret_mom_20)
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{prefix}__vol5_over_vol20", ret_vol_5 / (ret_vol_20 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{prefix}__cs_mom5", self._cross_sectional_zscore(ret_mom_5))
            )

        if len(feature_names) >= 2:
            first = self._cross_sectional_zscore(self._safe_numeric_frame(frames[feature_names[0]]))
            second = self._cross_sectional_zscore(self._safe_numeric_frame(frames[feature_names[1]]))
            interaction = first * second
            rank_spread = self._cross_sectional_rank(frames[feature_names[0]]) - self._cross_sectional_rank(
                frames[feature_names[1]]
            )
            engineered_blocks.append(
                self._as_engineered_block("interaction__f1_f2_cs_z_product", interaction)
            )
            engineered_blocks.append(
                self._as_engineered_block("interaction__f1_f2_rank_spread", rank_spread)
            )

        return engineered_blocks, assets

    def _make_features(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        frames, assets = self._extract_feature_frames(features)
        engineered_blocks, assets = self._build_feature_blocks(frames, assets)

        if not engineered_blocks:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty

        engineered = pd.concat(engineered_blocks, axis=1)
        feature_names = engineered.columns.get_level_values("feature").unique()
        engineered = engineered.reindex(
            columns=pd.MultiIndex.from_product([feature_names, assets], names=["feature", "asset"])
        )
        return engineered.replace([np.inf, -np.inf], np.nan).astype(np.float32, copy=False)

    def _prepare_target_frame(self, target, index, assets):
        if isinstance(target, pd.Series):
            if isinstance(target.index, pd.MultiIndex):
                target_frame = target.unstack(level=-1)
            else:
                target_frame = target.to_frame()
        elif isinstance(target, pd.DataFrame):
            target_frame = target.copy()
        else:
            target_frame = pd.DataFrame(target, index=index)

        if isinstance(target_frame.columns, pd.MultiIndex):
            selected = {}
            for asset in assets:
                matches = [
                    col
                    for col in target_frame.columns
                    if any(level_value == asset for level_value in col)
                ]
                if matches:
                    selected[asset] = matches[0]
            if selected:
                target_frame = target_frame.loc[:, list(selected.values())].copy()
                target_frame.columns = list(selected.keys())
            else:
                _, asset_level = self._infer_column_levels(target_frame.columns)
                target_frame.columns = target_frame.columns.get_level_values(asset_level)

        target_frame = target_frame.reindex(index=index)
        target_frame = target_frame.reindex(columns=assets)
        return self._safe_numeric_frame(target_frame)

    def _panel_to_matrix(self, engineered_features, target=None):
        assets = list(engineered_features.columns.get_level_values("asset").unique())
        x_long = engineered_features.stack(level="asset")
        x_long = x_long.replace([np.inf, -np.inf], np.nan)

        if target is None:
            return x_long, None

        target_frame = self._prepare_target_frame(target, engineered_features.index, assets)
        y_long = target_frame.stack()
        x_long, y_long = x_long.align(y_long, join="inner", axis=0)

        finite_y = y_long.replace([np.inf, -np.inf], np.nan).notna()
        if finite_y.any():
            x_long = x_long.loc[finite_y]
            y_long = y_long.loc[finite_y].astype(np.float32, copy=False)
        else:
            return pd.DataFrame(columns=x_long.columns), pd.Series(dtype=np.float32)

        return x_long, y_long

    def _feature_family(self, column_name):
        name = str(column_name)
        if "__" in name:
            return name.split("__", 1)[0]
        return name

    def _filter_features(self, x_train, y_train):
        if x_train.empty:
            return x_train

        columns = list(x_train.columns)
        keep = []

        x_values = x_train.to_numpy(dtype=np.float64, copy=False)
        n_rows = x_values.shape[0]

        for j, col in enumerate(columns):
            col_values = x_values[:, j]
            finite = np.isfinite(col_values)
            if not finite.any():
                continue
            if finite.mean() < 0.05:
                continue
            valid = col_values[finite]
            if valid.size == 0:
                continue
            if np.nanstd(valid) < 1.0e-8:
                continue
            if np.nanmin(valid) == np.nanmax(valid):
                continue
            keep.append(col)

        if not keep:
            return x_train.iloc[:, :0]

        filtered = x_train.loc[:, keep]
        if filtered.shape[1] <= 1:
            return filtered

        sample_size = min(n_rows, 20_000)
        if n_rows > sample_size:
            positions = np.linspace(0, n_rows - 1, sample_size, dtype=np.int64)
            sample = filtered.iloc[positions]
        else:
            sample = filtered

        corr_matrix = sample.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = set()
        for col in upper.columns:
            if col in drop_cols:
                continue
            high_corr = upper.index[upper[col] > 0.995].tolist()
            for other in high_corr:
                if other not in drop_cols:
                    drop_cols.add(other)

        deduped_cols = [col for col in filtered.columns if col not in drop_cols]
        filtered = filtered.loc[:, deduped_cols]

        if len(y_train) == len(filtered):
            y_values = y_train.to_numpy(dtype=np.float64, copy=False)
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
        else:
            scores = [(col, 0.0) for col in filtered.columns]

        if len(scores) <= self.max_features:
            selected = [col for col, _ in scores]
            return filtered.loc[:, selected]

        family_counts = {}
        selected = []
        for col, score in scores:
            if score <= 0.0 and len(selected) >= min(50, self.max_features // 4):
                continue
            family = self._feature_family(col)
            count = family_counts.get(family, 0)
            max_per_family = max(8, self.max_features // max(1, len(set(self._feature_family(c) for c, _ in scores))))
            if count >= max_per_family:
                continue
            selected.append(col)
            family_counts[family] = count + 1
            if len(selected) >= self.max_features:
                break

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
        self.impute_values_ = np.where(np.isfinite(self.impute_values_), self.impute_values_, 0.0)

        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]

        self.clip_low_ = np.nanquantile(x, 0.005, axis=0)
        self.clip_high_ = np.nanquantile(x, 0.995, axis=0)
        invalid_bounds = ~np.isfinite(self.clip_low_) | ~np.isfinite(self.clip_high_) | (
            self.clip_low_ >= self.clip_high_
        )
        self.clip_low_[invalid_bounds] = -10.0
        self.clip_high_[invalid_bounds] = 10.0

        x = np.clip(x, self.clip_low_, self.clip_high_)

        self.x_mean_ = x.mean(axis=0)
        centered = x - self.x_mean_
        mad = np.nanmedian(np.abs(centered), axis=0)
        self.x_scale_ = 1.4826 * mad
        fallback_std = x.std(axis=0)
        small_scale = ~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)
        self.x_scale_[small_scale] = fallback_std[small_scale]
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)] = 1.0

    def _transform_matrix(self, x_data):
        if self.impute_values_ is None or self.x_mean_ is None or self.x_scale_ is None:
            x = x_data.to_numpy(dtype=np.float64, copy=True)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = x_data.to_numpy(dtype=np.float64, copy=True)
        x = np.where(np.isfinite(x), x, np.nan)

        for j in range(x.shape[1]):
            mask = ~np.isfinite(x[:, j])
            if mask.any():
                x[mask, j] = self.impute_values_[j]

        if self.clip_low_ is not None and self.clip_high_ is not None:
            x = np.clip(x, self.clip_low_, self.clip_high_)

        x = (x - self.x_mean_) / self.x_scale_
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _chronological_split(self, x_data, y_data, train_fraction=0.80):
        n_rows = len(x_data)
        if n_rows < 40:
            split = max(1, int(n_rows * train_fraction))
        else:
            split = int(n_rows * train_fraction)

        split = min(max(split, 1), n_rows - 1)
        x_early = x_data.iloc[:split]
        y_early = y_data.iloc[:split]
        x_late = x_data.iloc[split:]
        y_late = y_data.iloc[split:]
        return x_early, y_early, x_late, y_late

    def _infer_timestamp_level(self, index):
        if not isinstance(index, pd.MultiIndex):
            return None
        if index.nlevels >= 2:
            return 0
        return None

    def _cross_sectional_corr_score(self, y_true, y_pred):
        if len(y_true) == 0:
            return -1.0

        y_pred_series = pd.Series(y_pred, index=y_true.index)
        aligned = pd.concat([y_true.rename("y"), y_pred_series.rename("p")], axis=1).dropna()
        if aligned.empty:
            return -1.0

        timestamp_level = self._infer_timestamp_level(aligned.index)
        correlations = []

        if timestamp_level is not None:
            groups = aligned.index.get_level_values(timestamp_level)
            for group_value in pd.unique(groups):
                block = aligned.loc[groups == group_value]
                if len(block) < 3:
                    continue
                y_vals = block["y"].to_numpy(dtype=np.float64)
                p_vals = block["p"].to_numpy(dtype=np.float64)
                if np.nanstd(y_vals) < 1.0e-12 or np.nanstd(p_vals) < 1.0e-12:
                    continue
                corr = np.corrcoef(y_vals, p_vals)[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)
        else:
            y_vals = aligned["y"].to_numpy(dtype=np.float64)
            p_vals = aligned["p"].to_numpy(dtype=np.float64)
            if np.nanstd(y_vals) >= 1.0e-12 and np.nanstd(p_vals) >= 1.0e-12:
                corr = np.corrcoef(y_vals, p_vals)[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)

        if not correlations:
            return -1.0

        mean_corr = float(np.mean(correlations))
        instability = float(np.std(correlations)) if len(correlations) > 1 else 0.0
        if timestamp_level is not None:
            n_groups = len(pd.unique(aligned.index.get_level_values(timestamp_level)))
            failure_rate = 1.0 - (len(correlations) / max(1, n_groups))
        else:
            failure_rate = 0.0
        score = mean_corr - 0.25 * instability - 0.10 * failure_rate
        return score

    def _solve_ridge(self, x_scaled, y_centered, alpha, sample_weights=None):
        if sample_weights is None:
            gram = x_scaled.T @ x_scaled
            rhs = x_scaled.T @ y_centered
        else:
            sw = np.sqrt(sample_weights)
            x_weighted = x_scaled * sw[:, None]
            y_weighted = y_centered * sw
            gram = x_weighted.T @ x_weighted
            rhs = x_weighted.T @ y_weighted

        gram = gram.copy()
        gram.flat[:: gram.shape[0] + 1] += alpha

        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs

        return coef

    def _fit_ridge_core(self, x_train, y_train, alpha, sample_weights=None):
        x_scaled = self._transform_matrix(x_train)
        y = y_train.to_numpy(dtype=np.float64, copy=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y_mean = float(y.mean()) if y.size else 0.0
        y_centered = y - y_mean

        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            sample_weights = np.clip(sample_weights, 1.0e-8, None)
        else:
            sample_weights = None

        coef = self._solve_ridge(x_scaled, y_centered, alpha, sample_weights)
        return coef, y_mean

    def _fit_huber_ridge(self, x_train, y_train, alpha, sample_weights=None, n_iter=4, delta=1.0):
        coef, intercept = self._fit_ridge_core(x_train, y_train, alpha, sample_weights)
        x_scaled = self._transform_matrix(x_train)
        y = y_train.to_numpy(dtype=np.float64, copy=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        weights = sample_weights
        if weights is None:
            weights = np.ones(len(y), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).copy()

        for _ in range(n_iter):
            residuals = y - (intercept + x_scaled @ coef)
            abs_res = np.abs(residuals)
            huber_w = np.ones_like(abs_res)
            heavy = abs_res > delta
            huber_w[heavy] = delta / np.maximum(abs_res[heavy], 1.0e-8)
            combined = weights * huber_w
            coef, intercept = self._fit_ridge_core(x_train, y_train, alpha * 2.0, combined)
            x_scaled = self._transform_matrix(x_train)

        return coef, intercept

    def _rank_targets(self, y_train):
        if isinstance(y_train.index, pd.MultiIndex) and y_train.index.nlevels >= 2:
            timestamp_level = self._infer_timestamp_level(y_train.index)
            if timestamp_level is not None:
                grouped = y_train.groupby(level=timestamp_level)
                ranked = grouped.rank(method="average", pct=True)
                return ranked.astype(np.float32)

        ranked = y_train.rank(method="average", pct=True)
        return ranked.astype(np.float32)

    def _fit_rank_ridge(self, x_train, y_train, alpha, sample_weights=None):
        ranked = self._rank_targets(y_train)
        return self._fit_ridge_core(x_train, ranked, alpha, sample_weights)

    def _predict_with_coef(self, x_data, coef, intercept):
        if coef is None:
            return np.zeros(len(x_data), dtype=np.float64)
        x_scaled = self._transform_matrix(x_data)
        return intercept + x_scaled @ coef

    def _sample_training_rows(self, x_train, y_train):
        n_rows = len(x_train)
        if n_rows <= self.max_train_rows:
            return x_train, y_train, np.arange(n_rows, dtype=np.int64)

        recent_count = int(self.max_train_rows * 0.60)
        older_budget = self.max_train_rows - recent_count
        recent_start = n_rows - recent_count
        recent_positions = np.arange(recent_start, n_rows, dtype=np.int64)

        older_end = recent_start
        if older_end <= 0 or older_budget <= 0:
            positions = recent_positions
        else:
            older_positions = np.linspace(0, older_end - 1, older_budget, dtype=np.int64)
            positions = np.concatenate([older_positions, recent_positions])

        positions = np.unique(positions)
        return x_train.iloc[positions], y_train.iloc[positions], positions

    def _compute_sample_weights(self, positions, total_rows):
        if total_rows <= 1:
            return np.ones(len(positions), dtype=np.float64)

        ages = (total_rows - 1) - positions.astype(np.float64)
        halflife = max(1.0, total_rows * 0.25)
        weights = np.exp(-ages / halflife)
        weights = weights / np.mean(weights)
        return weights.astype(np.float64)

    def _select_hyperparams(self, x_train, y_train):
        x_early, y_early, x_late, y_late = self._chronological_split(x_train, y_train)

        if len(x_early) < 20 or len(x_late) < 10:
            return 8.0, -1.0

        best_alpha = 8.0
        best_score = -np.inf

        for alpha in self._ALPHA_CANDIDATES:
            coef, intercept = self._fit_ridge_core(x_early, y_early, alpha)
            preds = self._predict_with_coef(x_late, coef, intercept)
            score = self._cross_sectional_corr_score(y_late, preds)
            if score > best_score:
                best_score = score
                best_alpha = alpha

        return best_alpha, best_score

    def _evaluate_ensemble_on_validation(self, x_early, y_early, x_late, y_late, alpha, sample_weights=None):
        coef_primary, intercept_primary = self._fit_ridge_core(x_early, y_early, alpha, sample_weights)
        coef_huber, intercept_huber = self._fit_huber_ridge(x_early, y_early, alpha, sample_weights)
        coef_rank, intercept_rank = self._fit_rank_ridge(x_early, y_early, alpha, sample_weights)

        pred_primary = self._predict_with_coef(x_late, coef_primary, intercept_primary)
        pred_huber = self._predict_with_coef(x_late, coef_huber, intercept_huber)
        pred_rank = self._predict_with_coef(x_late, coef_rank, intercept_rank)

        weights = dict(self._DEFAULT_ENSEMBLE_WEIGHTS)
        pred_ensemble = (
            weights["primary"] * pred_primary
            + weights["huber"] * pred_huber
            + weights["rank"] * pred_rank
        )
        score = self._cross_sectional_corr_score(y_late, pred_ensemble)

        rank_only_score = self._cross_sectional_corr_score(y_late, pred_rank)
        if rank_only_score > score + 0.01:
            weights["rank"] = min(0.15, weights["rank"] + 0.05)
            weights["primary"] = 1.0 - weights["huber"] - weights["rank"]

        return score, weights

    def _zero_prediction(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            _, assets = self._extract_feature_frames(features)
        except Exception:
            assets = list(features.columns)

        pred = pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

    def _finalize_prediction(self, pred):
        pred = pred.astype(np.float64, copy=False)
        pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clip_value = self.prediction_clip_ if np.isfinite(self.prediction_clip_) else 1.0
        clip_value = float(np.clip(clip_value, 1.0e-6, 10.0))
        pred = pred.clip(lower=-clip_value, upper=clip_value)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        pred = pred.clip(lower=-clip_value, upper=clip_value)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        return pred

    def _apply_rank_blend(self, pred):
        if not self.use_rank_blend_:
            return pred

        rank_signal = pred.rank(axis=1, pct=True).sub(0.5)
        row_std = pred.std(axis=1, ddof=0).replace(0.0, np.nan)
        standardized = pred.div(row_std, axis=0).fillna(0.0)
        blend_weight = self.rank_blend_weight_
        blended = (1.0 - blend_weight) * standardized + blend_weight * rank_signal
        return blended.fillna(0.0)

    def _fallback_prediction(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            frames, assets = self._extract_feature_frames(features)
            if not frames:
                return self._zero_prediction(features)

            primary_name = next(iter(frames))
            primary = self._safe_numeric_frame(frames[primary_name])
            score = -0.50 * self._cross_sectional_zscore(primary)

            mom_5 = self._rolling_stat(primary, 5, 2, "mean")
            mom_20 = self._rolling_stat(primary, 20, 5, "mean")
            vol_20 = self._rolling_stat(primary, 20, 5, "std")

            score = score + 0.35 * self._cross_sectional_zscore(mom_5)
            score = score + 0.20 * self._cross_sectional_zscore(mom_20)
            score = score - 0.15 * self._cross_sectional_zscore(primary / (vol_20 + 1.0e-6))

            for feature_name, frame in frames.items():
                if feature_name == primary_name:
                    continue
                score = score + 0.05 * self._cross_sectional_rank(self._safe_numeric_frame(frame))

            pred = score.reindex(index=features.index, columns=assets).fillna(0.0)
            return self._finalize_prediction(pred)
        except Exception:
            return self._zero_prediction(features)

    def _fit_ridge(self, x_train, y_train, sample_weights=None):
        coef, intercept = self._fit_ridge_core(x_train, y_train, self.alpha, sample_weights)
        self.coef_ = coef
        self.intercept_ = intercept

    def _predict_ridge(self, x_data):
        return self._predict_with_coef(x_data, self.coef_, self.intercept_)

    def _predict_ensemble(self, x_data):
        weights = self.model_weights_ or dict(self._DEFAULT_ENSEMBLE_WEIGHTS)
        pred = np.zeros(len(x_data), dtype=np.float64)

        if self.coef_primary_ is not None:
            pred += weights.get("primary", 0.0) * self._predict_with_coef(
                x_data, self.coef_primary_, self.intercept_primary_
            )
        if self.coef_huber_ is not None:
            pred += weights.get("huber", 0.0) * self._predict_with_coef(
                x_data, self.coef_huber_, self.intercept_huber_
            )
        if self.coef_rank_ is not None:
            pred += weights.get("rank", 0.0) * self._predict_with_coef(
                x_data, self.coef_rank_, self.intercept_rank_
            )

        if self.coef_primary_ is None and self.coef_ is not None:
            pred = self._predict_ridge(x_data)

        return pred

    def train(self, features, target):
        self.training_error_ = None
        self.is_trained_ = False
        self.fallback_used_ = False
        self.use_rank_blend_ = False
        self.validation_score_ = None
        self.model_weights_ = dict(self._DEFAULT_ENSEMBLE_WEIGHTS)

        try:
            engineered = self._make_features(features)
            x_train, y_train = self._panel_to_matrix(engineered, target)
            del engineered

            if x_train.empty or len(y_train) < 20:
                self.feature_columns_ = list(x_train.columns)
                self.selected_features_ = self.feature_columns_
                self.feature_count_ = len(self.feature_columns_ or [])
                return self

            x_train = self._filter_features(x_train, y_train)
            if x_train.empty or x_train.shape[1] == 0:
                self.feature_columns_ = []
                self.selected_features_ = []
                self.feature_count_ = 0
                return self

            self._fit_preprocessor(x_train)

            selected_alpha, validation_score = self._select_hyperparams(x_train, y_train)
            self.selected_alpha_ = selected_alpha
            self.alpha = selected_alpha
            self.validation_score_ = validation_score

            x_early, y_early, x_late, y_late = self._chronological_split(x_train, y_train)
            _, tuned_weights = self._evaluate_ensemble_on_validation(
                x_early, y_early, x_late, y_late, selected_alpha
            )
            self.model_weights_ = tuned_weights

            total_rows = len(x_train)
            x_sampled, y_sampled, positions = self._sample_training_rows(x_train, y_train)
            sample_weights = self._compute_sample_weights(positions, total_rows)
            self.training_rows_ = len(x_sampled)

            abs_y = np.abs(y_sampled.to_numpy(dtype=np.float64, copy=False))
            robust_quantile = np.nanquantile(abs_y, 0.995) if abs_y.size else 1.0
            if not np.isfinite(robust_quantile) or robust_quantile <= 0.0:
                robust_quantile = 1.0
            self.prediction_clip_ = float(np.clip(3.0 * robust_quantile, 1.0e-6, 10.0))

            self.coef_primary_, self.intercept_primary_ = self._fit_ridge_core(
                x_sampled, y_sampled, selected_alpha, sample_weights
            )
            self.coef_huber_, self.intercept_huber_ = self._fit_huber_ridge(
                x_sampled, y_sampled, selected_alpha, sample_weights
            )
            self.coef_rank_, self.intercept_rank_ = self._fit_rank_ridge(
                x_sampled, y_sampled, selected_alpha, sample_weights
            )

            self.coef_ = self.coef_primary_
            self.intercept_ = self.intercept_primary_
            self.feature_columns_ = list(x_sampled.columns)
            self.selected_features_ = self.feature_columns_
            self.feature_count_ = len(self.feature_columns_)

            if validation_score > -0.5 and len(x_late) >= 10:
                raw_preds = self._predict_ensemble(x_late)
                if isinstance(y_late.index, pd.MultiIndex):
                    pred_frame = pd.Series(raw_preds, index=y_late.index).unstack(level=-1)
                    blended_frame = self._apply_rank_blend(pred_frame)
                    blended_score = self._cross_sectional_corr_score(
                        y_late,
                        blended_frame.stack(future_stack=True),
                    )
                    if blended_score > validation_score:
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
            engineered = self._make_features(features)
            assets = list(engineered.columns.get_level_values("asset").unique())

            if engineered.empty or not self.is_trained_ or not self.selected_features_:
                self.fallback_used_ = True
                return self._fallback_prediction(features)

            x_long, _ = self._panel_to_matrix(engineered, target=None)
            del engineered

            x_long = x_long.reindex(columns=self.selected_features_)
            if self.impute_values_ is not None:
                for idx, col in enumerate(self.selected_features_):
                    if col in x_long.columns:
                        x_long[col] = x_long[col].fillna(self.impute_values_[idx])
                    else:
                        x_long[col] = self.impute_values_[idx]
            else:
                x_long = x_long.fillna(0.0)
            x_long = x_long[self.selected_features_]

            raw_pred = self._predict_ensemble(x_long)
            raw_pred = np.asarray(raw_pred, dtype=np.float64)
            raw_pred = np.nan_to_num(raw_pred, nan=0.0, posinf=0.0, neginf=0.0)

            pred_series = pd.Series(raw_pred, index=x_long.index)
            pred = pred_series.unstack(level=-1)
            pred = pred.reindex(index=features.index, columns=assets).fillna(0.0)
            pred = self._apply_rank_blend(pred)

            return self._finalize_prediction(pred)
        except Exception:
            self.fallback_used_ = True
            return self._fallback_prediction(features)
