from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "artixcore_alphalab_v33.py"
out_path = ROOT / "artixcore_alphalab_v43.py"
text = source_path.read_text(encoding="utf-8")

text = text.replace(
    '"""AlphaLab v0.33: compact leakage-safe generalization ensemble."""',
    '"""AlphaLab v0.43: v33 ensemble with a stable low-dimensional core anchor."""',
    1,
)

old_weights = '''    RIDGE_WEIGHT = 0.62
    RANK_WEIGHT = 0.23
    TREE_WEIGHT = 0.15
'''
new_weights = '''    RIDGE_WEIGHT = 0.62
    RANK_WEIGHT = 0.23
    TREE_WEIGHT = 0.15

    BASE_WEIGHT = 0.72
    CORE_WEIGHT = 0.28
    CORE_RAW_WEIGHT = 0.62
    CORE_RANK_WEIGHT = 0.38
    CORE_RAW_ALPHA = 30.0
    CORE_RANK_ALPHA = 46.0

    CORE_FEATURES = (
        "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
        "Feature.1__rank", "Feature.2__rank", "Feature.3__rank",
        "Feature.1__lag1", "Feature.2__lag1", "Feature.3__lag1",
        "interaction__rank12", "interaction__rank13",
    )
'''
if old_weights not in text:
    raise RuntimeError("weights marker not found")
text = text.replace(old_weights, new_weights, 1)

old_init_end = '''        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0
'''
new_init_end = '''        self.feature_time_ = self.fit_time_ = 0.0
        self.predict_feature_time_ = self.predict_model_time_ = 0.0

        self.core_features_ = []
        self.core_impute_ = self.core_low_ = self.core_high_ = None
        self.core_center_ = self.core_scale_values_ = None
        self.core_raw_coef_ = self.core_rank_coef_ = None
        self.core_raw_intercept_ = self.core_rank_intercept_ = 0.0
        self.core_rank_scale_ = 1.0
        self.core_enabled_ = False
        self.core_error_ = None
        self.core_fit_time_ = 0.0
        self.base_core_corr_ = 0.0
'''
if old_init_end not in text:
    raise RuntimeError("init marker not found")
text = text.replace(old_init_end, new_init_end, 1)

text = text.replace(
    "    def train(self, features, target):\n",
    "    def _train_base(self, features, target):\n",
    1,
)
text = text.replace(
    "    def predict(self, features):\n",
    "    def _predict_base(self, features):\n",
    1,
)

addition = r'''

    @staticmethod
    def _fit_core_stats(frame):
        values = frame.to_numpy(np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        impute = np.nanmedian(values, axis=0).astype(np.float32)
        impute[~np.isfinite(impute)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(impute, np.where(missing)[1])
        low = np.nanquantile(values, 0.01, axis=0).astype(np.float32)
        high = np.nanquantile(values, 0.99, axis=0).astype(np.float32)
        bad = ~np.isfinite(low) | ~np.isfinite(high) | (low >= high)
        low[bad], high[bad] = -10.0, 10.0
        values = np.clip(values, low, high)
        center = np.nanmedian(values, axis=0).astype(np.float32)
        scale = (
            1.4826 * np.nanmedian(np.abs(values - center), axis=0)
        ).astype(np.float32)
        fallback = np.nanstd(values, axis=0).astype(np.float32)
        bad = ~np.isfinite(scale) | (scale < 1e-8)
        scale[bad] = fallback[bad]
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return impute, low, high, center, scale

    def _transform_core(self, frame):
        values = frame.reindex(columns=self.core_features_).to_numpy(
            np.float32, copy=True
        )
        values[~np.isfinite(values)] = np.nan
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(
                self.core_impute_, np.where(missing)[1]
            )
        values = (
            np.clip(values, self.core_low_, self.core_high_)
            - self.core_center_
        ) / self.core_scale_values_
        return np.nan_to_num(values).astype(np.float32)

    def _fit_core_anchor(self, features, target):
        started = time.perf_counter()
        panel, assets = self._features(features)
        long_x, long_y = self._long(panel, target, assets)
        positions, codes, total = self._sample_times(long_x.index)
        self.core_features_ = [
            name for name in self.CORE_FEATURES if name in long_x.columns
        ]
        if len(self.core_features_) < 6:
            raise RuntimeError("Insufficient stable core features")

        core_x = long_x.iloc[positions].loc[:, self.core_features_]
        core_y = long_y.iloc[positions]
        (
            self.core_impute_,
            self.core_low_,
            self.core_high_,
            self.core_center_,
            self.core_scale_values_,
        ) = self._fit_core_stats(core_x)
        matrix = self._transform_core(core_x)

        raw = np.nan_to_num(core_y.to_numpy(np.float32))
        limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        raw = np.clip(raw, -limit, limit).astype(np.float32)
        if isinstance(core_y.index, pd.MultiIndex):
            ranked = core_y.groupby(level=0).rank(method="average", pct=True)
        else:
            ranked = core_y.rank(method="average", pct=True)
        rank_target = np.nan_to_num(
            ((ranked - 0.5) * 2.0).to_numpy(np.float32)
        )
        weights = self._weights(codes, total)

        self.core_raw_intercept_ = float(np.average(raw, weights=weights))
        self.core_raw_coef_ = self._ridge(
            matrix,
            raw - self.core_raw_intercept_,
            weights,
            self.CORE_RAW_ALPHA,
        )
        self.core_rank_intercept_ = float(
            np.average(rank_target, weights=weights)
        )
        self.core_rank_coef_ = self._ridge(
            matrix,
            rank_target - self.core_rank_intercept_,
            weights,
            self.CORE_RANK_ALPHA,
        )
        rank_std = float(np.std(rank_target))
        raw_std = float(np.std(raw))
        self.core_rank_scale_ = (
            raw_std / rank_std
            if raw_std > 1e-8 and rank_std > 1e-8
            else 1.0
        )
        self.core_fit_time_ = time.perf_counter() - started
        self.core_enabled_ = True

    @staticmethod
    def _array_correlation(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = float(
            np.sqrt(np.sum(left * left) * np.sum(right * right))
        )
        if denominator <= 1e-15:
            return 0.0
        return float(np.sum(left * right) / denominator)

    def train(self, features, target):
        self.core_enabled_ = False
        self.core_error_ = None
        self.core_features_ = []
        self.core_raw_coef_ = self.core_rank_coef_ = None
        self.base_core_corr_ = 0.0
        self._train_base(features, target)
        if not self.is_trained_:
            return self
        try:
            self._fit_core_anchor(features, target)
        except Exception as exc:
            self.core_error_ = repr(exc)
            self.training_error_ = "Core anchor training failed: " + repr(exc)
            self.core_enabled_ = False
            self.is_trained_ = False
        return self

    def predict(self, features):
        self.fallback_used_ = False
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            if not self.is_trained_ or not self.core_enabled_:
                self.fallback_used_ = True
                return self._fallback(features, self.assets_)

            base = self._predict_base(features)
            if self.fallback_used_:
                return base

            panel, assets = self._prediction_features(features)
            long_x = self._long(panel)
            if long_x.empty:
                self.fallback_used_ = True
                return self._fallback(features, assets or self.assets_)
            matrix = self._transform_core(long_x)
            raw = self.core_raw_intercept_ + matrix @ self.core_raw_coef_
            rank = self.core_rank_scale_ * (
                self.core_rank_intercept_ + matrix @ self.core_rank_coef_
            )
            core_values = (
                self.CORE_RAW_WEIGHT * raw
                + self.CORE_RANK_WEIGHT * rank
            )
            series = pd.Series(
                np.nan_to_num(core_values),
                index=long_x.index,
                dtype=np.float32,
            )
            core = series.unstack(-1).reindex(
                index=features.index, columns=assets
            ).fillna(0.0)
            core = self._finish(core)
            self.base_core_corr_ = self._array_correlation(
                base.to_numpy(), core.to_numpy()
            )
            return self._finish(
                self.BASE_WEIGHT * base + self.CORE_WEIGHT * core
            )
        except Exception as exc:
            self.core_error_ = repr(exc)
            self.fallback_used_ = True
            return self._fallback(features, self.assets_)
'''

text = text.rstrip() + addition + "\n"
out_path.write_text(text, encoding="utf-8")
print(f"Built {out_path.name} with {len(text.splitlines())} lines")
