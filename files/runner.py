"""Reconstructed local AlphaLab validation runner.

This runner is an open, self-contained compatibility tool. It is not a copy of
AlphaNova's private evaluator and does not reproduce its hidden dataset, city
library, or leaderboard score exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Type

import numpy as np
import pandas as pd

from city_tools import nearest_city
from demo_engineered import make_dataset
from predictor import Predictor
from walkforward import WalkForwardResult, walkforward_full


def _load_predictor_class(path: str | Path) -> Type[Predictor]:
    submission = Path(path).expanduser().resolve()
    if not submission.is_file():
        raise FileNotFoundError(f"submission not found: {submission}")

    files_dir = Path(__file__).resolve().parent
    root_dir = submission.parent
    for location in (str(files_dir), str(root_dir)):
        if location not in sys.path:
            sys.path.insert(0, location)

    module_name = f"alphalab_submission_{abs(hash(str(submission)))}"
    spec = importlib.util.spec_from_file_location(module_name, submission)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import submission: {submission}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Predictor:
            continue
        try:
            if issubclass(obj, Predictor) or (
                callable(getattr(obj, "train", None))
                and callable(getattr(obj, "predict", None))
            ):
                candidates.append(obj)
        except TypeError:
            continue
    if not candidates:
        raise TypeError("submission contains no Predictor-compatible class")
    candidates.sort(key=lambda cls: ("Artixcore" not in cls.__name__, cls.__name__))
    return candidates[0]


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _rank_rows(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    return frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float64)


def compute_statistics(result: WalkForwardResult) -> dict[str, float | str | int]:
    pred = result.predictions.to_numpy(dtype=np.float64)
    target = result.targets.to_numpy(dtype=np.float64)

    strategy_returns = []
    for period in np.unique(result.period_ids):
        mask = result.period_ids == period
        p = pred[mask]
        y = target[mask]
        if len(p) < 2:
            continue
        scale = np.sum(np.abs(p[:-1]), axis=1)
        lagged = np.sum(p[:-1] * y[1:], axis=1)
        lagged = np.divide(lagged, scale, out=np.zeros_like(lagged), where=scale > 1e-12)
        strategy_returns.append(lagged)
    returns = np.concatenate(strategy_returns) if strategy_returns else np.zeros(1)
    return_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / return_std) if return_std > 1e-15 else 0.0

    pred_rank = _rank_rows(pred)
    target_rank = _rank_rows(target)
    ic_values = _row_correlation(pred_rank, target_rank)
    ic = float(np.nanmean(ic_values))
    ic_std = float(np.nanstd(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    dispersion = np.nanstd(target, axis=1)
    ic_dispersion = (
        float(np.corrcoef(ic_values, dispersion)[0, 1])
        if np.std(ic_values) > 1e-15 and np.std(dispersion) > 1e-15
        else 0.0
    )

    abs_pred = np.abs(pred)
    concentration = float(
        np.mean(
            np.sum(abs_pred * abs_pred, axis=1)
            / np.maximum(np.sum(abs_pred, axis=1) ** 2, 1e-15)
        )
    )

    compressed = np.tanh(pred / np.maximum(np.nanstd(pred, axis=1, keepdims=True), 1e-6))
    similarity = _row_correlation(pred, compressed)
    compression_loss = float(np.nanmean(similarity) - 1.0)

    novelty, nearest = nearest_city(pred)
    return {
        "sharpe": sharpe,
        "periods": int(len(np.unique(result.period_ids))),
        "observations": int(len(result.predictions)),
        "ic": ic,
        "ic_std": ic_std,
        "ic_dispersion_corr": ic_dispersion,
        "concentration": concentration,
        "compression_loss": compression_loss,
        "city_novelty": novelty,
        "nearest_city": nearest,
        "train_seconds": float(result.train_seconds),
        "predict_seconds": float(result.predict_seconds),
        "failures": int(len(result.failures)),
    }


def validate_submission(
    submission: str | Path,
    *,
    full: bool = False,
    gauge_fix: bool = False,
    quiet: bool = False,
    periods: int | None = None,
    rows: int = 160,
    seed: int = 1729,
) -> dict[str, float | str | int]:
    predictor_class = _load_predictor_class(submission)
    period_count = periods if periods is not None else (77 if full else 12)
    dataset = make_dataset(n_periods=period_count, n_rows=rows, seed=seed)

    if not quiet:
        print("=" * 60)
        print("Artixcore reconstructed AlphaLab local validator")
        print(f"Submission: {Path(submission).name}")
        print(f"Periods: {period_count}")
        print("Validation: min(800, 20%) per period")
        print("NOTE: synthetic local statistics; official test results will differ.")
        print("=" * 60)

    result = walkforward_full(
        predictor_class,
        dataset,
        gauge_fix=gauge_fix,
        verbose=not quiet,
    )
    stats = compute_statistics(result)

    if not quiet:
        print("\n" + "=" * 60)
        print(f"Overall Sharpe: {stats['sharpe']:.4f}")
        print(f"Periods: {stats['periods']}, Observations: {stats['observations']}")
        print(f"IC: {stats['ic']:.4f} (std={stats['ic_std']:.4f})")
        print(f"IC-dispersion corr: {stats['ic_dispersion_corr']:.4f}")
        print(f"Concentration: {stats['concentration']:.4f}")
        print(f"Compression loss: {stats['compression_loss']:.4f}")
        print(
            f"City novelty: {stats['city_novelty']:.1f} deg "
            f"(nearest: {stats['nearest_city']})"
        )
        print("Global novelty: computed only by the official platform")
        print(f"Train time: {stats['train_seconds']:.2f}s")
        print(f"Predict time: {stats['predict_seconds']:.2f}s")
        if result.failures:
            print(f"Warnings: {len(result.failures)}")
            for message in result.failures[:10]:
                print(f"  - {message}")

    pd.DataFrame([stats]).to_csv("results.csv", index=False)
    if not quiet:
        print("\nResults written to results.csv")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Path to a submission .py file")
    parser.add_argument("--full", action="store_true", help="Run all 77 synthetic periods")
    parser.add_argument("--gauge-fix", action="store_true", help="Apply local gauge projection")
    parser.add_argument("--quiet", action="store_true", help="Suppress period progress")
    parser.add_argument("--periods", type=int, default=None, help="Override period count")
    parser.add_argument("--rows", type=int, default=160, help="Rows per synthetic period")
    parser.add_argument("--seed", type=int, default=1729, help="Synthetic data seed")
    args = parser.parse_args()

    validate_submission(
        args.submission,
        full=args.full,
        gauge_fix=args.gauge_fix,
        quiet=args.quiet,
        periods=args.periods,
        rows=args.rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
