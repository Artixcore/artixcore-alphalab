import numpy as np
import pandas as pd

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """
    Artixcore AlphaLab v0.1
    Leakage-safe walk-forward cross-sectional signal forecasting baseline.

    The model avoids ticker identity features and learns from timestamp-asset
    observations built from past/current rolling and cross-sectional features.
    """

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.alpha = 8.0
        self.coef_ = None
        self.intercept_ = 0.0
        self.x_mean_ = None
        self.x_scale_ = None
        self.feature_columns_ = None
        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.max_train_rows = 250_000
        self.training_error_ = None

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

    def _rolling_stat(self, frame, window, min_periods, stat):
        groups = self._infer_period_groups(frame.index)

        if groups is None:
            roller = frame.rolling(window=window, min_periods=min_periods)
            if stat == "mean":
                return roller.mean()
            return roller.std(ddof=0)

        result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=np.float32)
        group_values = pd.Series(groups)

        for group_value in pd.unique(group_values):
            positions = np.flatnonzero(group_values.to_numpy() == group_value)
            if positions.size == 0:
                continue
            block = frame.iloc[positions]
            roller = block.rolling(window=window, min_periods=min_periods)
            if stat == "mean":
                computed = roller.mean()
            else:
                computed = roller.std(ddof=0)
            result.iloc[positions, :] = computed.to_numpy(dtype=np.float32, copy=False)

        return result

    def _lag_frame(self, frame, lag):
        groups = self._infer_period_groups(frame.index)

        if groups is None:
            return frame.shift(lag)

        result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=np.float32)
        group_values = pd.Series(groups)

        for group_value in pd.unique(group_values):
            positions = np.flatnonzero(group_values.to_numpy() == group_value)
            if positions.size == 0:
                continue
            result.iloc[positions, :] = frame.iloc[positions].shift(lag).to_numpy(
                dtype=np.float32,
                copy=False,
            )

        return result

    def _cross_sectional_rank(self, frame):
        return frame.rank(axis=1, pct=True).sub(0.5).astype(np.float32, copy=False)

    def _cross_sectional_zscore(self, frame):
        row_mean = frame.mean(axis=1)
        row_std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
        return frame.sub(row_mean, axis=0).div(row_std, axis=0).astype(np.float32, copy=False)

    def _as_engineered_block(self, name, frame):
        block = frame.copy()
        block.columns = pd.MultiIndex.from_product([[name], block.columns], names=["feature", "asset"])
        return block

    def _make_features(self, features):
        frames, assets = self._extract_feature_frames(features)
        engineered_blocks = []

        for feature_name, raw_frame in frames.items():
            raw = self._safe_numeric_frame(raw_frame)
            safe_name = str(feature_name)

            engineered_blocks.append(self._as_engineered_block(f"{safe_name}__raw", raw))
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__cs_rank", self._cross_sectional_rank(raw))
            )
            engineered_blocks.append(
                self._as_engineered_block(f"{safe_name}__cs_z", self._cross_sectional_zscore(raw))
            )

            for window in (3, 5, 10, 20):
                min_periods = 2 if window <= 5 else max(3, window // 2)
                roll_mean = self._rolling_stat(raw, window, min_periods, "mean")
                engineered_blocks.append(
                    self._as_engineered_block(f"{safe_name}__roll_mean_{window}", roll_mean)
                )

            for window in (5, 10, 20):
                min_periods = 3 if window <= 10 else 5
                roll_std = self._rolling_stat(raw, window, min_periods, "std")
                engineered_blocks.append(
                    self._as_engineered_block(f"{safe_name}__roll_std_{window}", roll_std)
                )

            for lag in (1, 3, 5):
                diff_frame = raw - self._lag_frame(raw, lag)
                engineered_blocks.append(
                    self._as_engineered_block(f"{safe_name}__diff_{lag}", diff_frame)
                )

        primary_name = "feature.1" if "feature.1" in frames else next(iter(frames), None)
        if primary_name is not None:
            ret = self._safe_numeric_frame(frames[primary_name])
            ret_vol_5 = self._rolling_stat(ret, 5, 3, "std")
            ret_vol_20 = self._rolling_stat(ret, 20, 5, "std")
            ret_mom_3 = self._rolling_stat(ret, 3, 2, "mean")
            ret_mom_5 = self._rolling_stat(ret, 5, 2, "mean")
            ret_mom_20 = self._rolling_stat(ret, 20, 5, "mean")

            engineered_blocks.append(
                self._as_engineered_block("feature1__mean_reversion_vol5", -ret / (ret_vol_5 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block("feature1__mom5_over_vol20", ret_mom_5 / (ret_vol_20 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block("feature1__mom3_minus_mom20", ret_mom_3 - ret_mom_20)
            )
            engineered_blocks.append(
                self._as_engineered_block("feature1__vol5_over_vol20", ret_vol_5 / (ret_vol_20 + 1.0e-6))
            )
            engineered_blocks.append(
                self._as_engineered_block("feature1__cs_mom5", self._cross_sectional_zscore(ret_mom_5))
            )

        feature_names = list(frames.keys())
        if len(feature_names) >= 2:
            first = self._cross_sectional_zscore(self._safe_numeric_frame(frames[feature_names[0]]))
            second = self._cross_sectional_zscore(self._safe_numeric_frame(frames[feature_names[1]]))
            interaction = first * second
            rank_spread = self._cross_sectional_rank(frames[feature_names[0]]) - self._cross_sectional_rank(
                frames[feature_names[1]]
            )
            engineered_blocks.append(self._as_engineered_block("interaction__f1_f2_cs_z_product", interaction))
            engineered_blocks.append(self._as_engineered_block("interaction__f1_f2_rank_spread", rank_spread))

        if not engineered_blocks:
            empty = pd.DataFrame(index=features.index)
            empty.columns = pd.MultiIndex.from_arrays([[], []], names=["feature", "asset"])
            return empty

        engineered = pd.concat(engineered_blocks, axis=1)
        engineered = engineered.reindex(
            columns=pd.MultiIndex.from_product(
                [engineered.columns.get_level_values("feature").unique(), assets],
                names=["feature", "asset"],
            )
        )
        engineered = engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return engineered.astype(np.float32, copy=False)

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
        x_long = x_long.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

        x_long = x_long.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return x_long, y_long

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

    def _fallback_prediction(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            frames, assets = self._extract_feature_frames(features)
            if not frames:
                return self._zero_prediction(features)

            primary_name = "feature.1" if "feature.1" in frames else next(iter(frames))
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

    def _fit_ridge(self, x_train, y_train):
        x = x_train.to_numpy(dtype=np.float64, copy=True)
        y = y_train.to_numpy(dtype=np.float64, copy=True)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ < 1.0e-8)] = 1.0

        x_scaled = (x - self.x_mean_) / self.x_scale_
        y_mean = float(y.mean()) if y.size else 0.0
        y_centered = y - y_mean

        gram = x_scaled.T @ x_scaled
        gram.flat[:: gram.shape[0] + 1] += self.alpha
        rhs = x_scaled.T @ y_centered

        try:
            self.coef_ = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(gram) @ rhs

        self.intercept_ = y_mean

    def _predict_ridge(self, x_data):
        if self.coef_ is None or self.x_mean_ is None or self.x_scale_ is None:
            return np.zeros(len(x_data), dtype=np.float64)

        x = x_data.to_numpy(dtype=np.float64, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = (x - self.x_mean_) / self.x_scale_
        return self.intercept_ + x_scaled @ self.coef_

    def train(self, features, target):
        self.training_error_ = None
        self.is_trained_ = False

        try:
            engineered = self._make_features(features)
            x_train, y_train = self._panel_to_matrix(engineered, target)

            if x_train.empty or len(y_train) < 20:
                self.feature_columns_ = list(x_train.columns)
                return self

            if len(x_train) > self.max_train_rows:
                take = np.linspace(0, len(x_train) - 1, self.max_train_rows, dtype=np.int64)
                x_train = x_train.iloc[take]
                y_train = y_train.iloc[take]

            abs_y = np.abs(y_train.to_numpy(dtype=np.float64, copy=False))
            robust_quantile = np.nanquantile(abs_y, 0.995) if abs_y.size else 1.0
            if not np.isfinite(robust_quantile) or robust_quantile <= 0.0:
                robust_quantile = 1.0
            self.prediction_clip_ = float(np.clip(3.0 * robust_quantile, 1.0e-6, 10.0))

            self.feature_columns_ = list(x_train.columns)
            self._fit_ridge(x_train, y_train)
            self.is_trained_ = True
            return self
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
            return self

    def predict(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        try:
            engineered = self._make_features(features)
            assets = list(engineered.columns.get_level_values("asset").unique())

            if engineered.empty or not self.is_trained_ or self.feature_columns_ is None:
                return self._fallback_prediction(features)

            x_long, _ = self._panel_to_matrix(engineered, target=None)
            x_long = x_long.reindex(columns=self.feature_columns_, fill_value=0.0)
            x_long = x_long.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            raw_pred = self._predict_ridge(x_long)
            raw_pred = np.asarray(raw_pred, dtype=np.float64)
            raw_pred = np.nan_to_num(raw_pred, nan=0.0, posinf=0.0, neginf=0.0)

            pred_series = pd.Series(raw_pred, index=x_long.index)
            pred = pred_series.unstack(level=-1)
            pred = pred.reindex(index=features.index, columns=assets).fillna(0.0)

            return self._finalize_prediction(pred)
        except Exception:
            return self._fallback_prediction(features)
