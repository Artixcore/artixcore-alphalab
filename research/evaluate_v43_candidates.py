from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Type

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "files"
for path in (str(ROOT), str(FILES)):
    if path not in sys.path:
        sys.path.insert(0, path)

from artixcore_alphalab_v33 import ArtixcoreAlphaLabPredictor as V33
from city_tools import compute_gauge_fixed, make_gauge_matrices
from demo_engineered import make_dataset
from runner import compute_statistics
from walkforward import WalkForwardResult


DEV_SEEDS = (1729, 3141, 5003, 7919, 12007)


def finish_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.sub(frame.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)


def gauge_frame(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    gauges = make_gauge_matrices(
        n_rows=len(frame), n_assets=frame.shape[1], seed=314159 + period
    )
    fixed = compute_gauge_fixed(frame.to_numpy(dtype=np.float64), gauges)
    rebuilt = np.einsum("tij,ti->tj", gauges, fixed)
    return finish_frame(pd.DataFrame(rebuilt, index=frame.index, columns=frame.columns))


def row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.nanmean(left, axis=1, keepdims=True)
    right = right - np.nanmean(right, axis=1, keepdims=True)
    numerator = np.nansum(left * right, axis=1)
    denominator = np.sqrt(
        np.nansum(left * left, axis=1) * np.nansum(right * right, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-15,
    )


def subset_result(result: WalkForwardResult, periods: set[int]) -> WalkForwardResult:
    mask = np.isin(result.period_ids, list(periods))
    positions = np.flatnonzero(mask)
    return WalkForwardResult(
        predictions=result.predictions.iloc[positions],
        targets=result.targets.iloc[positions],
        period_ids=result.period_ids[mask],
        train_seconds=0.0,
        predict_seconds=0.0,
        failures=[],
    )


def fold_statistics(result: WalkForwardResult, width: int = 7) -> list[dict[str, float | int]]:
    periods = sorted(np.unique(result.period_ids).astype(int).tolist())
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(periods), width):
        fold = periods[start : start + width]
        stats = compute_statistics(subset_result(result, set(fold)))
        rows.append(
            {
                "period_start": int(fold[0]),
                "period_end": int(fold[-1]),
                "sharpe": float(stats["sharpe"]),
                "ic": float(stats["ic"]),
                "concentration": float(stats["concentration"]),
            }
        )
    return rows


class CoreAnchorPredictor(V33):
    """v33 plus a strongly regularized low-dimensional core anchor."""

    BASE_WEIGHT = 0.72
    CORE_WEIGHT = 0.28
    CORE_RAW_WEIGHT = 0.62
    CORE_RANK_WEIGHT = 0.38
    CORE_RAW_ALPHA = 30.0
    CORE_RANK_ALPHA = 46.0
    CORE_FEATURES = (
        "Feature.1__raw",
        "Feature.2__raw",
        "Feature.3__raw",
        "Feature.1__rank",
        "Feature.2__rank",
        "Feature.3__rank",
        "Feature.1__lag1",
        "Feature.2__lag1",
        "Feature.3__lag1",
        "interaction__rank12",
        "interaction__rank13",
    )

    def __init__(self):
        super().__init__()
        self.core_features_ = []
        self.core_impute_ = self.core_low_ = self.core_high_ = None
        self.core_center_ = self.core_scale_values_ = None
        self.core_raw_coef_ = self.core_rank_coef_ = None
        self.core_raw_intercept_ = self.core_rank_intercept_ = 0.0
        self.core_rank_scale_ = 1.0
        self.core_enabled_ = False
        self.core_error_ = None

    @staticmethod
    def _fit_core_stats(frame: pd.DataFrame):
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

    def _transform_core(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=self.core_features_).to_numpy(
            np.float32, copy=True
        )
        values[~np.isfinite(values)] = np.nan
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.core_impute_, np.where(missing)[1])
        values = (
            np.clip(values, self.core_low_, self.core_high_) - self.core_center_
        ) / self.core_scale_values_
        return np.nan_to_num(values).astype(np.float32)

    def train(self, features, target):
        super().train(features, target)
        self.core_enabled_ = False
        self.core_error_ = None
        if not self.is_trained_:
            return self
        try:
            panel, assets = self._features(features)
            long_x, long_y = self._long(panel, target, assets)
            positions, codes, total = self._sample_times(long_x.index)
            self.core_features_ = [
                name for name in self.CORE_FEATURES if name in long_x.columns
            ]
            if len(self.core_features_) < 6:
                raise RuntimeError("insufficient core features")
            core_x = long_x.iloc[positions].loc[:, self.core_features_]
            core_y = long_y.iloc[positions]
            stats = self._fit_core_stats(core_x)
            (
                self.core_impute_,
                self.core_low_,
                self.core_high_,
                self.core_center_,
                self.core_scale_values_,
            ) = stats
            matrix = self._transform_core(core_x)
            raw = np.nan_to_num(core_y.to_numpy(np.float32))
            limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
            if not np.isfinite(limit) or limit <= 0.0:
                limit = 1.0
            raw = np.clip(raw, -limit, limit).astype(np.float32)
            ranked = core_y.groupby(level=0).rank(method="average", pct=True)
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
                raw_std / rank_std if raw_std > 1e-8 and rank_std > 1e-8 else 1.0
            )
            self.core_enabled_ = True
        except Exception as exc:
            self.core_error_ = repr(exc)
            self.core_enabled_ = False
        return self

    def predict(self, features):
        base = super().predict(features)
        if not self.core_enabled_:
            self.fallback_used_ = True
            return base
        try:
            panel, assets = self._prediction_features(features)
            long_x = self._long(panel)
            matrix = self._transform_core(long_x)
            raw = self.core_raw_intercept_ + matrix @ self.core_raw_coef_
            rank = self.core_rank_scale_ * (
                self.core_rank_intercept_ + matrix @ self.core_rank_coef_
            )
            core_values = self.CORE_RAW_WEIGHT * raw + self.CORE_RANK_WEIGHT * rank
            series = pd.Series(
                np.nan_to_num(core_values), index=long_x.index, dtype=np.float32
            )
            core = series.unstack(-1).reindex(
                index=features.index, columns=assets
            ).fillna(0.0)
            core = self._finish(core)
            return self._finish(self.BASE_WEIGHT * base + self.CORE_WEIGHT * core)
        except Exception as exc:
            self.core_error_ = repr(exc)
            self.fallback_used_ = True
            return base


class GroupShrinkagePredictor(V33):
    """v33 feature engine with explicit group-specific ridge penalties."""

    @staticmethod
    def _penalty_multiplier(name: str) -> float:
        if name in {
            "Feature.1__raw", "Feature.2__raw", "Feature.3__raw",
            "Feature.1__rank", "Feature.2__rank", "Feature.3__rank",
        }:
            return 0.55
        if name in {
            "Feature.1__lag1", "Feature.2__lag1", "Feature.3__lag1",
            "interaction__rank12", "interaction__rank13",
        }:
            return 0.80
        if name.startswith(("Feature.4__", "Feature.5__", "Feature.6__")):
            return 2.80
        if "__diff" in name or "__ma" in name or "__sd" in name:
            return 1.80
        if "__ewma" in name or "__roll" in name or "__mom" in name:
            return 1.55
        return 1.25

    @classmethod
    def _group_ridge(cls, matrix, target, weights, alpha, names):
        root = np.sqrt(np.maximum(weights, 1e-8)).astype(np.float32)
        xw = matrix * root[:, None]
        yw = target * root
        gram = xw.T @ xw
        rhs = xw.T @ yw
        penalties = np.asarray(
            [alpha * cls._penalty_multiplier(str(name)) for name in names],
            dtype=np.float32,
        )
        gram.flat[:: gram.shape[0] + 1] += penalties
        try:
            return np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(gram) @ rhs).astype(np.float32)

    def train(self, features, target):
        self.is_trained_, self.training_error_, self.fallback_used_ = False, None, False
        self.ridge_coef_ = self.rank_coef_ = self.tree_model_ = None
        try:
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            self.history_tail_ = features.tail(self.history_rows).copy()
            started = time.perf_counter()
            panel, assets = self._features(features)
            self.feature_time_ = time.perf_counter() - started
            self.assets_ = list(assets)
            long_x, long_y = self._long(panel, target, assets)
            if long_x.empty or len(long_y) < 60:
                self.training_error_ = "Insufficient usable training observations"
                return self
            self.selected_features_ = self._select(long_x)
            positions, codes, total = self._sample_times(long_x.index)
            sampled_x = long_x.iloc[positions].loc[:, self.selected_features_]
            sampled_y = long_y.iloc[positions]
            self._fit_preprocessor(sampled_x)
            matrix = self._transform(sampled_x)
            raw = np.nan_to_num(sampled_y.to_numpy(np.float32))
            limit = float(np.nanquantile(np.abs(raw), 0.995)) if raw.size else 1.0
            if not np.isfinite(limit) or limit <= 0.0:
                limit = 1.0
            raw = np.clip(raw, -limit, limit).astype(np.float32)
            self.prediction_clip_ = float(np.clip(3.0 * limit, 1e-6, 10.0))
            ranked = sampled_y.groupby(level=0).rank(method="average", pct=True)
            rank_target = np.nan_to_num(
                ((ranked - 0.5) * 2.0).to_numpy(np.float32)
            )
            weights = self._weights(codes, total)
            started = time.perf_counter()
            self.ridge_intercept_ = float(np.average(raw, weights=weights))
            self.ridge_coef_ = self._group_ridge(
                matrix,
                raw - self.ridge_intercept_,
                weights,
                self.RIDGE_ALPHA,
                self.selected_features_,
            )
            self.rank_intercept_ = float(np.average(rank_target, weights=weights))
            self.rank_coef_ = self._group_ridge(
                matrix,
                rank_target - self.rank_intercept_,
                weights,
                self.RANK_ALPHA,
                self.selected_features_,
            )
            rank_std = float(np.std(rank_target))
            raw_std = float(np.std(raw))
            self.rank_scale_ = (
                raw_std / rank_std if raw_std > 1e-8 and rank_std > 1e-8 else 1.0
            )
            self.tree_model_ = xgb.train(
                dict(self.XGB_PARAMS),
                xgb.DMatrix(matrix, label=raw, weight=weights),
                num_boost_round=9,
            )
            self.fit_time_ = time.perf_counter() - started
            self.training_rows_ = len(sampled_x)
            self.training_times_ = len(
                pd.Index(sampled_x.index.get_level_values(0)).unique()
            )
            self.feature_count_ = len(self.selected_features_)
            self.is_trained_ = True
        except Exception as exc:
            self.training_error_ = repr(exc)
            self.is_trained_ = False
        return self


def evaluate_class(
    predictor_class: Type[V33], dataset
) -> tuple[WalkForwardResult, int, list[str]]:
    predictions: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    period_ids: list[np.ndarray] = []
    failures: list[str] = []
    fallback_count = 0
    train_seconds = 0.0
    predict_seconds = 0.0

    for period_data in dataset:
        features = period_data.features
        target = period_data.target
        n_valid = max(2, min(800, int(np.ceil(len(features) * 0.20))))
        split = len(features) - n_valid
        train_x, train_y = features.iloc[:split], target.iloc[:split]
        valid_x, valid_y = features.iloc[split:], target.iloc[split:].copy()
        model = predictor_class()
        started = time.perf_counter()
        model.train(train_x, train_y)
        train_seconds += time.perf_counter() - started
        if not getattr(model, "is_trained_", False):
            failures.append(
                f"period {period_data.period} train: {getattr(model, 'training_error_', None)}"
            )
        started = time.perf_counter()
        try:
            prediction = model.predict(valid_x)
        except Exception as exc:
            failures.append(f"period {period_data.period} predict: {exc!r}")
            prediction = pd.DataFrame(
                0.0, index=valid_x.index, columns=valid_y.columns, dtype=np.float32
            )
        predict_seconds += time.perf_counter() - started
        fallback_count += int(bool(getattr(model, "fallback_used_", False)))
        prediction = finish_frame(
            prediction.reindex(index=valid_x.index, columns=valid_y.columns)
        )
        prediction = gauge_frame(prediction, period_data.period)
        keyed_index = pd.MultiIndex.from_arrays(
            [np.full(len(prediction), period_data.period), prediction.index],
            names=["period", "time"],
        )
        prediction.index = keyed_index
        valid_y.index = keyed_index
        predictions.append(prediction)
        targets.append(valid_y.astype(np.float32))
        period_ids.append(
            np.full(len(prediction), period_data.period, dtype=np.int32)
        )

    return (
        WalkForwardResult(
            predictions=pd.concat(predictions),
            targets=pd.concat(targets),
            period_ids=np.concatenate(period_ids),
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            failures=failures,
        ),
        fallback_count,
        failures,
    )


def summarize_seed(
    seed: int,
    model_name: str,
    result: WalkForwardResult,
    baseline: WalkForwardResult | None,
    fallback_count: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    stats = compute_statistics(result)
    folds = fold_statistics(result)
    row: dict[str, float | int | str] = {
        "seed": seed,
        "model": model_name,
        "sharpe": float(stats["sharpe"]),
        "ic": float(stats["ic"]),
        "ic_std": float(stats["ic_std"]),
        "concentration": float(stats["concentration"]),
        "compression_loss": float(stats["compression_loss"]),
        "city_novelty": float(stats["city_novelty"]),
        "train_seconds": float(stats["train_seconds"]),
        "predict_seconds": float(stats["predict_seconds"]),
        "fallback_periods": int(fallback_count),
        "failures": int(len(result.failures)),
        "median_fold_sharpe": float(np.median([x["sharpe"] for x in folds])),
        "worst_fold_sharpe": float(np.min([x["sharpe"] for x in folds])),
        "median_fold_ic": float(np.median([x["ic"] for x in folds])),
        "worst_fold_ic": float(np.min([x["ic"] for x in folds])),
    }
    if baseline is not None:
        corr = row_corr(
            result.predictions.to_numpy(dtype=np.float64),
            baseline.predictions.to_numpy(dtype=np.float64),
        )
        row["prediction_corr_v33"] = float(np.mean(corr))
    else:
        row["prediction_corr_v33"] = 1.0
    fold_rows = [
        {"seed": seed, "model": model_name, **fold} for fold in folds
    ]
    return row, fold_rows


def aggregate(results: pd.DataFrame, candidate: str) -> dict[str, float | int]:
    base = results.loc[results["model"] == "v33"].set_index("seed")
    current = results.loc[results["model"] == candidate].set_index("seed")
    aligned = current.join(base, lsuffix="_candidate", rsuffix="_v33")
    sharpe_delta = aligned["sharpe_candidate"] - aligned["sharpe_v33"]
    ic_delta = aligned["ic_candidate"] - aligned["ic_v33"]
    worst_fold_delta = (
        aligned["worst_fold_sharpe_candidate"]
        - aligned["worst_fold_sharpe_v33"]
    )
    return {
        "sharpe_delta_mean": float(sharpe_delta.mean()),
        "sharpe_delta_median": float(sharpe_delta.median()),
        "sharpe_wins": int((sharpe_delta > 0.0).sum()),
        "ic_delta_mean": float(ic_delta.mean()),
        "ic_delta_median": float(ic_delta.median()),
        "ic_wins": int((ic_delta > 0.0).sum()),
        "worst_seed_sharpe_delta": float(sharpe_delta.min()),
        "worst_fold_sharpe_delta_median": float(worst_fold_delta.median()),
        "mean_prediction_corr_v33": float(
            current["prediction_corr_v33"].mean()
        ),
        "max_concentration": float(current["concentration"].max()),
        "fallback_periods": int(current["fallback_periods"].sum()),
        "failures": int(current["failures"].sum()),
    }


def main() -> None:
    output_dir = ROOT / "research" / "output" / "v43_development"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    classes: tuple[tuple[str, Type[V33]], ...] = (
        ("v33", V33),
        ("core_anchor", CoreAnchorPredictor),
        ("group_shrinkage", GroupShrinkagePredictor),
    )

    for seed in DEV_SEEDS:
        dataset = list(make_dataset(n_periods=77, n_rows=160, seed=seed))
        baseline_result: WalkForwardResult | None = None
        for model_name, predictor_class in classes:
            result, fallback_count, failures = evaluate_class(
                predictor_class, dataset
            )
            if failures:
                print(f"{model_name} seed={seed} failures={failures[:3]}")
            if model_name == "v33":
                baseline_result = result
            row, folds = summarize_seed(
                seed,
                model_name,
                result,
                baseline_result if model_name != "v33" else None,
                fallback_count,
            )
            rows.append(row)
            fold_rows.extend(folds)
            print(
                f"seed={seed} model={model_name} "
                f"sharpe={row['sharpe']:.6f} ic={row['ic']:.6f} "
                f"corr={row['prediction_corr_v33']:.6f}"
            )

    results = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    results.to_csv(output_dir / "development_seed_results.csv", index=False)
    folds.to_csv(output_dir / "development_fold_results.csv", index=False)
    summary = {
        "development_seeds": list(DEV_SEEDS),
        "core_anchor": aggregate(results, "core_anchor"),
        "group_shrinkage": aggregate(results, "group_shrinkage"),
        "holdout_evaluated": False,
    }
    (output_dir / "development_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("V43_DEVELOPMENT_JSON_START")
    print(json.dumps(summary, indent=2))
    print("V43_DEVELOPMENT_JSON_END")


if __name__ == "__main__":
    main()
