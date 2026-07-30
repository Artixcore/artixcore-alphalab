from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src_path = ROOT / 'artixcore_alphalab_v43.py'
out_path = ROOT / 'artixcore_alphalab_v44.py'
src = src_path.read_text()
src = src.replace('"""AlphaLab v0.43: v33 ensemble with a stable low-dimensional core anchor."""', '"""AlphaLab v0.44: dual-horizon rank-transport ensemble."""')
needle = '    CORE_RANK_ALPHA = 46.0\n'
insert = '''    CORE_RANK_ALPHA = 46.0\n\n    HORIZON_WEIGHT = 0.35\n    HORIZON_POWER = 0.60\n    HORIZON_RIDGE_ALPHA = 12.0\n    HORIZON_RANK_ALPHA = 28.0\n'''
if needle not in src:
    raise SystemExit('constant needle missing')
src = src.replace(needle, insert, 1)
needle = '        self.base_core_corr_ = 0.0\n'
insert = '''        self.base_core_corr_ = 0.0\n\n        self.horizon_features_ = []\n        self.horizon_impute_ = self.horizon_low_ = self.horizon_high_ = None\n        self.horizon_center_ = self.horizon_scale_values_ = None\n        self.horizon_ridge_coef_ = self.horizon_rank_coef_ = None\n        self.horizon_ridge_intercept_ = self.horizon_rank_intercept_ = 0.0\n        self.horizon_rank_scale_ = 1.0\n        self.horizon_tree_model_ = None\n        self.horizon_enabled_ = False\n        self.horizon_error_ = None\n        self.horizon_fit_time_ = 0.0\n        self.base_horizon_corr_ = 0.0\n'''
if needle not in src:
    raise SystemExit('init needle missing')
src = src.replace(needle, insert, 1)
src = src.replace('    def train(self, features, target):\n', '    def _train_v43(self, features, target):\n', 1)
src = src.replace('    def predict(self, features):\n', '    def _predict_v43(self, features):\n', 1)
append = r'''

    def _transform_horizon(self, frame):
        values = frame.reindex(columns=self.horizon_features_).to_numpy(
            np.float32, copy=True
        )
        values[~np.isfinite(values)] = np.nan
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(
                self.horizon_impute_, np.where(missing)[1]
            )
        values = (
            np.clip(values, self.horizon_low_, self.horizon_high_)
            - self.horizon_center_
        ) / self.horizon_scale_values_
        return np.nan_to_num(values).astype(np.float32)

    def _forward_long(self, panel, target, assets):
        target_frame = self._target(target, panel.index, assets)
        if len(panel) < 2:
            return self._long(panel, target, assets)
        feature_panel = panel.iloc[:-1]
        response_frame = target_frame.iloc[1:].copy()
        response_frame.index = feature_panel.index
        matrix_frame = self._stack(feature_panel, "asset").replace(
            [np.inf, -np.inf], np.nan
        )
        response = self._stack(response_frame)
        matrix_frame, response = matrix_frame.align(
            response, join="inner", axis=0
        )
        valid = response.replace([np.inf, -np.inf], np.nan).notna()
        return (
            matrix_frame.loc[valid],
            response.loc[valid].astype(np.float32),
        )

    def _fit_horizon_branch(self, features, target):
        started = time.perf_counter()
        panel, assets = self._features(features)
        long_x, long_y = self._forward_long(panel, target, assets)
        if long_x.empty or len(long_y) < 60:
            raise RuntimeError("Insufficient forward-horizon observations")

        self.horizon_features_ = self._select(long_x)
        if not self.horizon_features_:
            raise RuntimeError("No forward-horizon features selected")
        positions, codes, total = self._sample_times(long_x.index)
        horizon_x = long_x.iloc[positions].loc[:, self.horizon_features_]
        horizon_y = long_y.iloc[positions]
        (
            self.horizon_impute_,
            self.horizon_low_,
            self.horizon_high_,
            self.horizon_center_,
            self.horizon_scale_values_,
        ) = self._fit_core_stats(horizon_x)
        matrix = self._transform_horizon(horizon_x)

        raw = np.nan_to_num(horizon_y.to_numpy(np.float32))
        limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        raw = np.clip(raw, -limit, limit).astype(np.float32)
        if isinstance(horizon_y.index, pd.MultiIndex):
            ranked = horizon_y.groupby(level=0).rank(
                method="average", pct=True
            )
        else:
            ranked = horizon_y.rank(method="average", pct=True)
        rank_target = np.nan_to_num(
            ((ranked - 0.5) * 2.0).to_numpy(np.float32)
        )
        weights = self._weights(codes, total)

        self.horizon_ridge_intercept_ = float(
            np.average(raw, weights=weights)
        )
        self.horizon_ridge_coef_ = self._ridge(
            matrix,
            raw - self.horizon_ridge_intercept_,
            weights,
            self.HORIZON_RIDGE_ALPHA,
        )
        self.horizon_rank_intercept_ = float(
            np.average(rank_target, weights=weights)
        )
        self.horizon_rank_coef_ = self._ridge(
            matrix,
            rank_target - self.horizon_rank_intercept_,
            weights,
            self.HORIZON_RANK_ALPHA,
        )
        raw_std = float(np.std(raw))
        rank_std = float(np.std(rank_target))
        self.horizon_rank_scale_ = (
            raw_std / rank_std
            if raw_std > 1e-8 and rank_std > 1e-8
            else 1.0
        )
        self.horizon_tree_model_ = xgb.train(
            dict(self.XGB_PARAMS),
            xgb.DMatrix(matrix, label=raw, weight=weights),
            num_boost_round=9,
        )
        self.horizon_fit_time_ = time.perf_counter() - started
        self.horizon_enabled_ = True

    def _predict_horizon(self, features):
        panel, assets = self._prediction_features(features)
        long_x = self._long(panel)
        if long_x.empty:
            raise RuntimeError("Empty forward-horizon prediction matrix")
        matrix = self._transform_horizon(long_x)
        ridge = (
            self.horizon_ridge_intercept_
            + matrix @ self.horizon_ridge_coef_
        )
        rank = self.horizon_rank_scale_ * (
            self.horizon_rank_intercept_
            + matrix @ self.horizon_rank_coef_
        )
        tree = self.horizon_tree_model_.predict(
            xgb.DMatrix(matrix)
        ).astype(np.float32)
        raw = (
            self.RIDGE_WEIGHT * ridge
            + self.RANK_WEIGHT * rank
            + self.TREE_WEIGHT * tree
        )
        series = pd.Series(
            np.nan_to_num(raw), index=long_x.index, dtype=np.float32
        )
        prediction = series.unstack(-1).reindex(
            index=features.index, columns=assets
        ).fillna(0.0)
        return self._finish(prediction)

    def _rank_transport(self, base, horizon):
        base_values = base.to_numpy(np.float64)
        horizon_values = horizon.to_numpy(np.float64)
        transported = np.empty_like(horizon_values)
        for row in range(len(base_values)):
            base_order = np.argsort(
                base_values[row], kind="mergesort"
            )
            sorted_horizon = np.sort(
                horizon_values[row], kind="mergesort"
            )
            scale = max(float(np.std(sorted_horizon)), 1e-8)
            shaped = (
                np.sign(sorted_horizon)
                * np.power(
                    np.abs(sorted_horizon) / scale,
                    self.HORIZON_POWER,
                )
                * scale
            )
            transported[row, base_order] = shaped
        return self._finish(
            pd.DataFrame(
                transported, index=base.index, columns=base.columns
            )
        )

    def train(self, features, target):
        self.horizon_enabled_ = False
        self.horizon_error_ = None
        self.horizon_features_ = []
        self.horizon_ridge_coef_ = self.horizon_rank_coef_ = None
        self.horizon_tree_model_ = None
        self.base_horizon_corr_ = 0.0
        self._train_v43(features, target)
        if not self.is_trained_:
            return self
        try:
            self._fit_horizon_branch(features, target)
        except Exception as exc:
            self.horizon_error_ = repr(exc)
            self.training_error_ = (
                "Forward-horizon training failed: " + repr(exc)
            )
            self.horizon_enabled_ = False
            self.is_trained_ = False
        return self

    def predict(self, features):
        self.fallback_used_ = False
        try:
            base = self._predict_v43(features)
            if self.fallback_used_:
                return base
            if not self.is_trained_ or not self.horizon_enabled_:
                self.fallback_used_ = True
                return base
            horizon = self._predict_horizon(features)
            self.base_horizon_corr_ = self._array_correlation(
                base.to_numpy(), horizon.to_numpy()
            )
            transported = self._rank_transport(base, horizon)
            return self._finish(
                (1.0 - self.HORIZON_WEIGHT) * base
                + self.HORIZON_WEIGHT * transported
            )
        except Exception as exc:
            self.horizon_error_ = repr(exc)
            self.fallback_used_ = True
            try:
                return self._predict_v43(features)
            except Exception:
                return self._fallback(features, self.assets_)
'''
src = src.rstrip() + append + '\n'
out_path.write_text(src)
print(out_path, len(src.splitlines()))
