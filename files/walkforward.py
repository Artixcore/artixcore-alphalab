"""Leakage-safe per-period walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Type

import numpy as np
import pandas as pd

from city_tools import compute_gauge_fixed, make_gauge_matrices
from demo_engineered import PeriodData
from predictor import Predictor


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    targets: pd.DataFrame
    period_ids: np.ndarray
    train_seconds: float
    predict_seconds: float
    failures: list[str]


def _validation_size(n_rows: int, fraction: float = 0.20, maximum: int = 800) -> int:
    return max(2, min(maximum, int(np.ceil(n_rows * fraction))))


def _shape_safe_prediction(
    prediction,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if isinstance(prediction, pd.Series):
        if isinstance(prediction.index, pd.MultiIndex):
            prediction = prediction.unstack(level=-1)
        else:
            prediction = prediction.to_frame()
    elif not isinstance(prediction, pd.DataFrame):
        prediction = pd.DataFrame(prediction)

    output = prediction.reindex(index=index, columns=columns)
    output = output.apply(pd.to_numeric, errors="coerce")
    output = output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output = output.sub(output.mean(axis=1), axis=0).fillna(0.0)
    return output.astype(np.float32)


def walkforward_full(
    predictor_class: Type[Predictor],
    periods: Iterable[PeriodData],
    validation_fraction: float = 0.20,
    max_validation_rows: int = 800,
    gauge_fix: bool = False,
    verbose: bool = True,
) -> WalkForwardResult:
    predictions: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    period_ids: list[np.ndarray] = []
    failures: list[str] = []
    train_seconds = 0.0
    predict_seconds = 0.0

    period_list = list(periods)
    for offset, period in enumerate(period_list, start=1):
        features = period.features
        target = period.target
        n_valid = _validation_size(len(features), validation_fraction, max_validation_rows)
        split = len(features) - n_valid
        if split < 10:
            failures.append(f"period {period.period}: insufficient training rows")
            continue

        train_x = features.iloc[:split]
        train_y = target.iloc[:split]
        valid_x = features.iloc[split:]
        valid_y = target.iloc[split:]

        model = predictor_class()
        started = time.perf_counter()
        try:
            model.train(train_x, train_y)
        except Exception as exc:
            failures.append(f"period {period.period} train: {exc!r}")
        train_seconds += time.perf_counter() - started

        started = time.perf_counter()
        try:
            raw_prediction = model.predict(valid_x)
            prediction = _shape_safe_prediction(raw_prediction, valid_x.index, valid_y.columns)
        except Exception as exc:
            failures.append(f"period {period.period} predict: {exc!r}")
            prediction = pd.DataFrame(
                0.0, index=valid_x.index, columns=valid_y.columns, dtype=np.float32
            )
        predict_seconds += time.perf_counter() - started

        if gauge_fix:
            gauges = make_gauge_matrices(
                n_rows=len(prediction),
                n_assets=prediction.shape[1],
                seed=314159 + period.period,
            )
            fixed = compute_gauge_fixed(prediction.to_numpy(), gauges)
            prediction = pd.DataFrame(
                np.einsum("tji,ti->tj", gauges, fixed),
                index=prediction.index,
                columns=prediction.columns,
            )
            prediction = _shape_safe_prediction(prediction, valid_x.index, valid_y.columns)

        keyed_index = pd.MultiIndex.from_arrays(
            [np.full(len(prediction), period.period), prediction.index],
            names=["period", "time"],
        )
        prediction.index = keyed_index
        valid_y = valid_y.copy()
        valid_y.index = keyed_index
        predictions.append(prediction)
        targets.append(valid_y.astype(np.float32))
        period_ids.append(np.full(len(prediction), period.period, dtype=np.int32))

        if verbose:
            print(
                f"Period {offset:03d}/{len(period_list):03d} "
                f"train={split} validate={n_valid}"
            )

    if not predictions:
        raise RuntimeError("walk-forward produced no valid periods")

    return WalkForwardResult(
        predictions=pd.concat(predictions, axis=0),
        targets=pd.concat(targets, axis=0),
        period_ids=np.concatenate(period_ids),
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        failures=failures,
    )
