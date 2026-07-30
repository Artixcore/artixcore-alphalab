from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "files"
for path in (str(ROOT), str(FILES)):
    if path not in sys.path:
        sys.path.insert(0, path)

from city_tools import compute_gauge_fixed, make_gauge_matrices
from demo_engineered import make_dataset
from runner import compute_statistics
from walkforward import WalkForwardResult


def load_predictor(path: Path):
    name = f"diagnostic_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArtixcoreAlphaLabPredictor


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


def rank_rows(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(
        axis=1, method="average", pct=True
    ).to_numpy(dtype=np.float64)


def finish_frame(frame: pd.DataFrame, clip: float | None = None) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip is not None:
        frame = frame.clip(-clip, clip)
    return frame.sub(frame.mean(axis=1), axis=0).fillna(0.0).astype(np.float32)


def gauge_frame(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    gauges = make_gauge_matrices(
        n_rows=len(frame), n_assets=frame.shape[1], seed=314159 + period
    )
    fixed = compute_gauge_fixed(frame.to_numpy(), gauges)
    rebuilt = np.einsum("tij,ti->tj", gauges, fixed)
    return finish_frame(pd.DataFrame(rebuilt, index=frame.index, columns=frame.columns))


def frame_from_long(values: np.ndarray, index: pd.MultiIndex, rows, assets) -> pd.DataFrame:
    series = pd.Series(np.nan_to_num(values), index=index, dtype=np.float32)
    return series.unstack(-1).reindex(index=rows, columns=assets).fillna(0.0)


def strategy_returns(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(pred) < 2:
        return np.zeros(0, dtype=np.float64)
    scale = np.sum(np.abs(pred[:-1]), axis=1)
    pnl = np.sum(pred[:-1] * target[1:], axis=1)
    return np.divide(pnl, scale, out=np.zeros_like(pnl), where=scale > 1e-12)


def safe_sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / std) if std > 1e-15 else 0.0


def period_metrics(period: int, pred: pd.DataFrame, target: pd.DataFrame) -> dict[str, float | int]:
    p = pred.to_numpy(dtype=np.float64)
    y = target.to_numpy(dtype=np.float64)
    returns = strategy_returns(p, y)
    ic_rows = row_corr(rank_rows(p), rank_rows(y))
    abs_pred = np.abs(p)
    concentration_rows = np.sum(abs_pred * abs_pred, axis=1) / np.maximum(
        np.sum(abs_pred, axis=1) ** 2, 1e-15
    )
    return {
        "period": int(period),
        "return_mean": float(np.mean(returns)) if len(returns) else 0.0,
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "sharpe": safe_sharpe(returns),
        "ic": float(np.mean(ic_rows)),
        "ic_std": float(np.std(ic_rows, ddof=1)) if len(ic_rows) > 1 else 0.0,
        "dispersion": float(np.mean(np.std(y, axis=1))),
        "concentration": float(np.mean(concentration_rows)),
        "pred_scale": float(np.mean(np.std(p, axis=1))),
        "rows": int(len(pred)),
    }


def feature_ic(model, valid_x: pd.DataFrame, valid_y: pd.DataFrame) -> dict[str, float]:
    frames, assets = model._extract(valid_x)
    target = valid_y.reindex(columns=assets).to_numpy(dtype=np.float64)
    target_rank = rank_rows(target)
    output: dict[str, float] = {}
    for name in ("Feature.1", "Feature.2", "Feature.3"):
        if name not in frames:
            continue
        values = frames[name].reindex(columns=assets).to_numpy(dtype=np.float64)
        output[f"{name}_raw_ic"] = float(np.mean(row_corr(rank_rows(values), target_rank)))
        output[f"{name}_linear_corr"] = float(np.mean(row_corr(values, target)))
    return output


def component_predictions(model, valid_x: pd.DataFrame, period: int):
    panel, assets = model._prediction_features(valid_x)
    long_x = model._long(panel)
    matrix = model._transform(long_x)
    ridge = model.ridge_intercept_ + matrix @ model.ridge_coef_
    rank = model.rank_scale_ * (model.rank_intercept_ + matrix @ model.rank_coef_)
    tree = model.tree_model_.predict(__import__("xgboost").DMatrix(matrix)).astype(np.float32)
    raw = model.RIDGE_WEIGHT * ridge + model.RANK_WEIGHT * rank + model.TREE_WEIGHT * tree
    linear = model.RIDGE_WEIGHT * ridge + model.RANK_WEIGHT * rank

    frames = {
        "ridge": frame_from_long(ridge, long_x.index, valid_x.index, assets),
        "rank": frame_from_long(rank, long_x.index, valid_x.index, assets),
        "tree": frame_from_long(tree, long_x.index, valid_x.index, assets),
        "linear": frame_from_long(linear, long_x.index, valid_x.index, assets),
        "blend_unclipped": frame_from_long(raw, long_x.index, valid_x.index, assets),
    }
    finished = {
        name: gauge_frame(finish_frame(frame, model.prediction_clip_), period)
        for name, frame in frames.items()
    }
    raw_values = frames["blend_unclipped"].to_numpy(dtype=np.float64)
    clip_fraction = float(np.mean(np.abs(raw_values) >= model.prediction_clip_))
    return finished, clip_fraction


def evaluate(model_path: Path, dataset, inspect_components: bool):
    predictor_class = load_predictor(model_path)
    predictions = []
    targets = []
    period_ids = []
    period_rows = []
    component_store: dict[str, list[pd.DataFrame]] = {}
    feature_rows = []
    failures = []
    train_seconds = 0.0
    predict_seconds = 0.0

    for period_data in dataset:
        features = period_data.features
        target = period_data.target
        n_valid = max(2, min(800, int(np.ceil(len(features) * 0.20))))
        split = len(features) - n_valid
        train_x = features.iloc[:split]
        train_y = target.iloc[:split]
        valid_x = features.iloc[split:]
        valid_y = target.iloc[split:].copy()

        model = predictor_class()
        started = time.perf_counter()
        model.train(train_x, train_y)
        train_seconds += time.perf_counter() - started
        if not getattr(model, "is_trained_", False):
            failures.append(f"period {period_data.period}: {getattr(model, 'training_error_', None)}")

        started = time.perf_counter()
        pred = model.predict(valid_x)
        predict_seconds += time.perf_counter() - started
        pred = finish_frame(pred.reindex(index=valid_x.index, columns=valid_y.columns))
        pred = gauge_frame(pred, period_data.period)

        metrics = period_metrics(period_data.period, pred, valid_y)
        metrics.update(feature_ic(model, valid_x, valid_y))
        metrics["fallback"] = int(bool(getattr(model, "fallback_used_", False)))

        if inspect_components:
            components, clip_fraction = component_predictions(model, valid_x, period_data.period)
            metrics["clip_fraction"] = clip_fraction
            metrics["ridge_rank_corr"] = float(np.mean(row_corr(
                components["ridge"].to_numpy(), components["rank"].to_numpy()
            )))
            metrics["linear_tree_corr"] = float(np.mean(row_corr(
                components["linear"].to_numpy(), components["tree"].to_numpy()
            )))
            for name, frame in components.items():
                keyed = frame.copy()
                keyed.index = pd.MultiIndex.from_arrays(
                    [np.full(len(frame), period_data.period), frame.index],
                    names=["period", "time"],
                )
                component_store.setdefault(name, []).append(keyed)

        keyed_index = pd.MultiIndex.from_arrays(
            [np.full(len(pred), period_data.period), pred.index],
            names=["period", "time"],
        )
        pred.index = keyed_index
        valid_y.index = keyed_index
        predictions.append(pred)
        targets.append(valid_y.astype(np.float32))
        period_ids.append(np.full(len(pred), period_data.period, dtype=np.int32))
        period_rows.append(metrics)

    result = WalkForwardResult(
        predictions=pd.concat(predictions),
        targets=pd.concat(targets),
        period_ids=np.concatenate(period_ids),
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        failures=failures,
    )
    components = {
        name: pd.concat(frames).reindex(index=result.predictions.index)
        for name, frames in component_store.items()
    }
    return result, pd.DataFrame(period_rows), components


def contiguous_runs(periods: list[int]) -> list[list[int]]:
    if not periods:
        return []
    runs = [[periods[0]]]
    for period in periods[1:]:
        if period == runs[-1][-1] + 1:
            runs[-1].append(period)
        else:
            runs.append([period])
    return runs


def subset_result(result: WalkForwardResult, allowed: set[int]) -> WalkForwardResult:
    mask = np.isin(result.period_ids, list(allowed))
    return WalkForwardResult(
        predictions=result.predictions.iloc[np.flatnonzero(mask)],
        targets=result.targets.iloc[np.flatnonzero(mask)],
        period_ids=result.period_ids[mask],
        train_seconds=0.0,
        predict_seconds=0.0,
        failures=[],
    )


def fold_stats(result: WalkForwardResult, width: int = 7):
    periods = sorted(np.unique(result.period_ids).tolist())
    output = []
    for start in range(0, len(periods), width):
        fold = periods[start:start + width]
        stats = compute_statistics(subset_result(result, set(fold)))
        output.append({
            "period_start": int(fold[0]),
            "period_end": int(fold[-1]),
            "sharpe": float(stats["sharpe"]),
            "ic": float(stats["ic"]),
            "concentration": float(stats["concentration"]),
        })
    return output


def component_summary(components: dict[str, pd.DataFrame], target: pd.DataFrame, period_ids):
    output = {}
    for name, pred in components.items():
        result = WalkForwardResult(
            predictions=pred,
            targets=target,
            period_ids=period_ids,
            train_seconds=0.0,
            predict_seconds=0.0,
            failures=[],
        )
        stats = compute_statistics(result)
        output[name] = {
            "sharpe": float(stats["sharpe"]),
            "ic": float(stats["ic"]),
            "concentration": float(stats["concentration"]),
        }
    return output


def main():
    dataset = list(make_dataset(n_periods=77, n_rows=160, seed=1729))
    v33_result, v33_periods, components = evaluate(
        ROOT / "artixcore_alphalab_v33.py", dataset, inspect_components=True
    )
    v42_result, v42_periods, _ = evaluate(
        ROOT / "artixcore_alphalab_v42.py", dataset, inspect_components=False
    )

    v33_stats = compute_statistics(v33_result)
    v42_stats = compute_statistics(v42_result)
    merged = v33_periods.merge(v42_periods, on="period", suffixes=("_v33", "_v42"))
    merged["return_delta_v42_minus_v33"] = merged["return_mean_v42"] - merged["return_mean_v33"]
    merged["ic_delta_v42_minus_v33"] = merged["ic_v42"] - merged["ic_v33"]

    pred_corr = row_corr(
        v33_result.predictions.to_numpy(), v42_result.predictions.to_numpy()
    )
    negative_periods = sorted(
        v33_periods.loc[v33_periods["return_mean"] < 0, "period"].astype(int).tolist()
    )

    numeric = v33_periods.select_dtypes(include=[np.number])
    correlations = numeric.corr()["return_mean"].sort_values().to_dict()

    summary = {
        "v33": v33_stats,
        "v42": v42_stats,
        "v33_v42_prediction_corr_mean": float(np.mean(pred_corr)),
        "v33_v42_prediction_corr_min": float(np.min(pred_corr)),
        "v33_v42_prediction_corr_p10": float(np.quantile(pred_corr, 0.10)),
        "v33_component_summary": component_summary(
            components, v33_result.targets, v33_result.period_ids
        ),
        "v33_fold_stats": fold_stats(v33_result),
        "v42_fold_stats": fold_stats(v42_result),
        "v33_worst_periods": v33_periods.nsmallest(10, "return_mean").to_dict("records"),
        "v33_best_periods": v33_periods.nlargest(10, "return_mean").to_dict("records"),
        "v42_biggest_improvements": merged.nlargest(
            10, "return_delta_v42_minus_v33"
        ).to_dict("records"),
        "v42_biggest_regressions": merged.nsmallest(
            10, "return_delta_v42_minus_v33"
        ).to_dict("records"),
        "v33_negative_period_runs": contiguous_runs(negative_periods),
        "v33_return_correlations": {
            key: float(value) for key, value in correlations.items() if np.isfinite(value)
        },
        "v33_failures": v33_result.failures,
        "v42_failures": v42_result.failures,
    }

    output_dir = ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    v33_periods.to_csv(output_dir / "v33_periods.csv", index=False)
    v42_periods.to_csv(output_dir / "v42_periods.csv", index=False)
    merged.to_csv(output_dir / "v33_v42_comparison.csv", index=False)
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    print("DIAGNOSTICS_JSON_START")
    print(json.dumps(summary, indent=2, default=float))
    print("DIAGNOSTICS_JSON_END")


if __name__ == "__main__":
    main()
