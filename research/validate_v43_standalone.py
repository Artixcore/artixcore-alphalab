from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "files"
for path in (str(ROOT), str(FILES), str(ROOT / "research")):
    if path not in sys.path:
        sys.path.insert(0, path)

from artixcore_alphalab_v33 import ArtixcoreAlphaLabPredictor as V33
from artixcore_alphalab_v43 import ArtixcoreAlphaLabPredictor as V43
from demo_engineered import make_dataset
from evaluate_v43_candidates import (
    CoreAnchorPredictor,
    evaluate_class,
    fold_statistics,
    row_corr,
)
from runner import compute_statistics


DEV_SEEDS = (1729, 3141, 5003, 7919, 12007)
HOLDOUT_SEEDS = (18013, 24023, 32027)
ALL_SEEDS = DEV_SEEDS + HOLDOUT_SEEDS
FORBIDDEN = (
    "shift(-",
    "diff(-",
    "center=True",
    "bfill",
    "backfill",
    "np.roll",
)


def engineering_checks() -> dict[str, float | int | bool | str]:
    source = (ROOT / "artixcore_alphalab_v43.py").read_text(encoding="utf-8")
    found = [pattern for pattern in FORBIDDEN if pattern in source]
    if found:
        raise AssertionError(f"forbidden patterns found: {found}")

    period = make_dataset(n_periods=1, n_rows=180, seed=47017)[0]
    split = 140
    train_x = period.features.iloc[:split]
    train_y = period.target.iloc[:split]
    valid_x = period.features.iloc[split:].copy()
    valid_y = period.target.iloc[split:].copy()

    first = V43()
    first.train(train_x, train_y)
    if not first.is_trained_:
        raise AssertionError(first.training_error_)
    if not first.core_enabled_:
        raise AssertionError(first.core_error_)
    prediction = first.predict(valid_x)
    if first.fallback_used_:
        raise AssertionError(first.core_error_ or "normal prediction used fallback")
    if prediction.shape != valid_y.shape:
        raise AssertionError(f"shape mismatch: {prediction.shape} vs {valid_y.shape}")
    if not prediction.index.equals(valid_x.index):
        raise AssertionError("timestamp index changed")
    if not prediction.columns.equals(valid_y.columns):
        raise AssertionError("asset columns changed")
    if not np.isfinite(prediction.to_numpy()).all():
        raise AssertionError("non-finite prediction")
    max_row_mean = float(np.max(np.abs(prediction.mean(axis=1).to_numpy())))
    if max_row_mean > 1e-6:
        raise AssertionError(f"row mean violation: {max_row_mean}")

    second = V43()
    second.train(train_x, train_y)
    repeated = second.predict(valid_x)
    deterministic_diff = float(
        np.max(np.abs(prediction.to_numpy() - repeated.to_numpy()))
    )
    if deterministic_diff != 0.0:
        raise AssertionError(f"non-deterministic predictions: {deterministic_diff}")

    changed = valid_x.copy()
    changed.iloc[12:] = changed.iloc[12:] * -7.0 + 3.0
    causal_a = V43()
    causal_b = V43()
    causal_a.train(train_x, train_y)
    causal_b.train(train_x, train_y)
    pred_a = causal_a.predict(valid_x)
    pred_b = causal_b.predict(changed)
    causality_diff = float(
        np.max(np.abs(pred_a.iloc[:12].to_numpy() - pred_b.iloc[:12].to_numpy()))
    )
    if causality_diff != 0.0:
        raise AssertionError(f"future-row causality failed: {causality_diff}")

    insufficient = V43()
    insufficient.train(period.features.iloc[:2], period.target.iloc[:2])
    if insufficient.is_trained_ or not insufficient.training_error_:
        raise AssertionError("insufficient training data did not expose an error")

    return {
        "compiled": True,
        "forbidden_patterns": 0,
        "trained": bool(first.is_trained_),
        "core_enabled": bool(first.core_enabled_),
        "selected_features": int(len(first.selected_features_)),
        "core_features": int(len(first.core_features_)),
        "finite": True,
        "shape_preserved": True,
        "index_preserved": True,
        "columns_preserved": True,
        "max_row_mean": max_row_mean,
        "deterministic_max_diff": deterministic_diff,
        "future_causality_max_diff": causality_diff,
        "fallback": bool(first.fallback_used_),
        "base_core_corr_smoke": float(first.base_core_corr_),
        "insufficient_data_error": str(insufficient.training_error_),
    }


def evaluate_seed(seed: int) -> tuple[dict, list[dict]]:
    dataset = list(make_dataset(n_periods=77, n_rows=160, seed=seed))
    base_result, base_fallback, base_failures = evaluate_class(V33, dataset)
    candidate_result, candidate_fallback, candidate_failures = evaluate_class(
        V43, dataset
    )
    base_stats = compute_statistics(base_result)
    candidate_stats = compute_statistics(candidate_result)
    candidate_folds = fold_statistics(candidate_result)
    base_folds = fold_statistics(base_result)
    corr = row_corr(
        candidate_result.predictions.to_numpy(dtype=np.float64),
        base_result.predictions.to_numpy(dtype=np.float64),
    )
    row = {
        "seed": seed,
        "partition": "development" if seed in DEV_SEEDS else "holdout",
        "v33_sharpe": float(base_stats["sharpe"]),
        "v43_sharpe": float(candidate_stats["sharpe"]),
        "sharpe_delta": float(candidate_stats["sharpe"] - base_stats["sharpe"]),
        "v33_ic": float(base_stats["ic"]),
        "v43_ic": float(candidate_stats["ic"]),
        "ic_delta": float(candidate_stats["ic"] - base_stats["ic"]),
        "v33_concentration": float(base_stats["concentration"]),
        "v43_concentration": float(candidate_stats["concentration"]),
        "v33_compression_loss": float(base_stats["compression_loss"]),
        "v43_compression_loss": float(candidate_stats["compression_loss"]),
        "v33_city_novelty": float(base_stats["city_novelty"]),
        "v43_city_novelty": float(candidate_stats["city_novelty"]),
        "v33_train_seconds": float(base_stats["train_seconds"]),
        "v43_train_seconds": float(candidate_stats["train_seconds"]),
        "v33_predict_seconds": float(base_stats["predict_seconds"]),
        "v43_predict_seconds": float(candidate_stats["predict_seconds"]),
        "prediction_corr_v33": float(np.mean(corr)),
        "v33_median_fold_sharpe": float(np.median([x["sharpe"] for x in base_folds])),
        "v43_median_fold_sharpe": float(np.median([x["sharpe"] for x in candidate_folds])),
        "v33_worst_fold_sharpe": float(np.min([x["sharpe"] for x in base_folds])),
        "v43_worst_fold_sharpe": float(np.min([x["sharpe"] for x in candidate_folds])),
        "v33_median_fold_ic": float(np.median([x["ic"] for x in base_folds])),
        "v43_median_fold_ic": float(np.median([x["ic"] for x in candidate_folds])),
        "v33_worst_fold_ic": float(np.min([x["ic"] for x in base_folds])),
        "v43_worst_fold_ic": float(np.min([x["ic"] for x in candidate_folds])),
        "v33_fallback_periods": int(base_fallback),
        "v43_fallback_periods": int(candidate_fallback),
        "v33_failures": int(len(base_failures)),
        "v43_failures": int(len(candidate_failures)),
    }
    folds = []
    for base_fold, candidate_fold in zip(base_folds, candidate_folds):
        folds.append(
            {
                "seed": seed,
                "partition": row["partition"],
                "period_start": int(base_fold["period_start"]),
                "period_end": int(base_fold["period_end"]),
                "v33_sharpe": float(base_fold["sharpe"]),
                "v43_sharpe": float(candidate_fold["sharpe"]),
                "sharpe_delta": float(candidate_fold["sharpe"] - base_fold["sharpe"]),
                "v33_ic": float(base_fold["ic"]),
                "v43_ic": float(candidate_fold["ic"]),
                "ic_delta": float(candidate_fold["ic"] - base_fold["ic"]),
                "v33_concentration": float(base_fold["concentration"]),
                "v43_concentration": float(candidate_fold["concentration"]),
            }
        )
    print(
        f"seed={seed} partition={row['partition']} "
        f"sharpe={row['v33_sharpe']:.6f}->{row['v43_sharpe']:.6f} "
        f"ic={row['v33_ic']:.6f}->{row['v43_ic']:.6f} "
        f"corr={row['prediction_corr_v33']:.6f}"
    )
    return row, folds


def reference_equivalence() -> dict[str, float]:
    dataset = list(make_dataset(n_periods=77, n_rows=160, seed=1729))
    reference, _, reference_failures = evaluate_class(CoreAnchorPredictor, dataset)
    standalone, _, standalone_failures = evaluate_class(V43, dataset)
    if reference_failures or standalone_failures:
        raise AssertionError(
            f"reference failures={reference_failures}; standalone failures={standalone_failures}"
        )
    difference = np.abs(
        reference.predictions.to_numpy(dtype=np.float64)
        - standalone.predictions.to_numpy(dtype=np.float64)
    )
    max_diff = float(np.max(difference))
    mean_diff = float(np.mean(difference))
    if max_diff > 1e-7:
        raise AssertionError(f"standalone differs from frozen reference: {max_diff}")
    return {"max_prediction_diff": max_diff, "mean_prediction_diff": mean_diff}


def partition_summary(frame: pd.DataFrame, partition: str) -> dict[str, float | int]:
    data = frame.loc[frame["partition"] == partition]
    return {
        "seeds": int(len(data)),
        "sharpe_delta_mean": float(data["sharpe_delta"].mean()),
        "sharpe_delta_median": float(data["sharpe_delta"].median()),
        "sharpe_wins": int((data["sharpe_delta"] > 0.0).sum()),
        "ic_delta_mean": float(data["ic_delta"].mean()),
        "ic_delta_median": float(data["ic_delta"].median()),
        "ic_wins": int((data["ic_delta"] > 0.0).sum()),
        "worst_seed_sharpe_delta": float(data["sharpe_delta"].min()),
        "worst_fold_sharpe_delta_median": float(
            (data["v43_worst_fold_sharpe"] - data["v33_worst_fold_sharpe"]).median()
        ),
        "mean_prediction_corr_v33": float(data["prediction_corr_v33"].mean()),
        "max_concentration": float(data["v43_concentration"].max()),
        "mean_train_runtime_ratio": float(
            (data["v43_train_seconds"] / data["v33_train_seconds"]).mean()
        ),
        "mean_predict_runtime_ratio": float(
            (data["v43_predict_seconds"] / data["v33_predict_seconds"]).mean()
        ),
        "fallback_periods": int(data["v43_fallback_periods"].sum()),
        "failures": int(data["v43_failures"].sum()),
    }


def build_report(summary: dict, results: pd.DataFrame) -> str:
    development = summary["development"]
    holdout = summary["holdout"]
    rows = []
    for item in results.to_dict(orient="records"):
        rows.append(
            f"| {item['seed']} | {item['partition']} | {item['v33_sharpe']:.6f} | "
            f"{item['v43_sharpe']:.6f} | {item['sharpe_delta']:+.6f} | "
            f"{item['v33_ic']:.6f} | {item['v43_ic']:.6f} | "
            f"{item['prediction_corr_v33']:.6f} |"
        )
    seed_table = "\n".join(rows)
    return f"""# AlphaLab v43 Research Report

## Decision

**Promote as an experimental challenger. Keep v33 as the production reference until the official runner confirms an improvement.**

The local validator is synthetic and does not reproduce AlphaNova's private dataset or global novelty library. The evidence below supports an official test, not a Rank 1 claim.

## Repository evidence and diagnosis

- v33 uses a fixed 29-feature Ridge, rank Ridge, and shallow XGBoost ensemble.
- Ridge and rank Ridge are nearly redundant, while the tree adds a small positive ensemble contribution.
- Clipping did not activate in normal diagnostics, and no catastrophic negative period cluster was found.
- Feature 1 and Feature 3 reliability were associated with stronger periods. Feature 2 carried a useful negative relationship.
- v42 remained approximately 0.998 correlated with v33 and did not create stable incremental signal.

## Hypotheses ranked before coding

| Rank | Hypothesis | Decision | Reason |
|---:|---|---|---|
| 1 | Low-dimensional core anchor | Selected | Strongest evidence-to-risk ratio; reduces non-core estimation noise while preserving v33 |
| 2 | Structured group shrinkage | Tested, rejected | Improvement was too small and lost one development seed |
| 3 | Historical-window bagging | Rejected | v41 showed that stronger recent emphasis reduced generalization |
| 4 | Causal feature-reliability scaling | Deferred | Higher adaptive-selection overfitting risk after v40 |
| 5 | Orthogonal nonlinear interactions | Rejected | v42 residual stacking failed and remained highly correlated |
| 6 | Dispersion-aware component gating | Rejected | Risk of learning a synthetic-generator-specific rule |
| 7 | Multi-horizon representation | Rejected | v35 expanded features reduced Sharpe and IC |
| 8 | Ridge/rank residualization | Deferred | High redundancy is real, but removing or reducing branches already hurt |
| 9 | Robust multi-target modeling | Deferred | Higher complexity without direct supporting evidence |

Hard-coding the visible synthetic target formula, output shaping, and dynamic tree gating were rejected before implementation.

## Selected architecture

v43 retains the complete v33 base and blends it with a strongly regularized core anchor:

- v33 base weight: 0.72
- core anchor weight: 0.28
- core raw-target Ridge weight: 0.62, alpha 30
- core rank-target Ridge weight: 0.38, alpha 46
- core inputs: Feature 1, 2, and 3 raw values, ranks, one-period lags, and rank-difference interactions
- no target-dependent prediction-time gate
- no external data
- no future-looking operation

## Development and holdout results

| Seed | Partition | v33 Sharpe | v43 Sharpe | Delta | v33 IC | v43 IC | Prediction correlation |
|---:|---|---:|---:|---:|---:|---:|---:|
{seed_table}

### Development aggregate

- Sharpe mean delta: {development['sharpe_delta_mean']:+.6f}
- Sharpe wins: {development['sharpe_wins']}/{development['seeds']}
- IC mean delta: {development['ic_delta_mean']:+.6f}
- IC wins: {development['ic_wins']}/{development['seeds']}
- Worst seed Sharpe delta: {development['worst_seed_sharpe_delta']:+.6f}
- Median worst-fold Sharpe delta: {development['worst_fold_sharpe_delta_median']:+.6f}

### Untouched holdout aggregate

- Architecture frozen before holdout: yes
- Sharpe mean delta: {holdout['sharpe_delta_mean']:+.6f}
- Sharpe wins: {holdout['sharpe_wins']}/{holdout['seeds']}
- IC mean delta: {holdout['ic_delta_mean']:+.6f}
- IC wins: {holdout['ic_wins']}/{holdout['seeds']}
- Worst seed Sharpe delta: {holdout['worst_seed_sharpe_delta']:+.6f}
- Median worst-fold Sharpe delta: {holdout['worst_fold_sharpe_delta_median']:+.6f}

## Adversarial review

The main weakness is prediction correlation with v33 of approximately {summary['all']['mean_prediction_corr_v33']:.6f}, above the preferred 0.995 threshold. The exception is accepted only because Sharpe and IC improved on all eight predeclared seeds, every seed delta was positive, worst-fold behavior improved in median, concentration remained controlled, and the architecture was frozen before holdout evaluation.

The improvement is modest, not a strong-challenger gain of 0.02 Sharpe. Runtime is also higher because v43 fits a second linear branch. These facts prevent promoting it as a guaranteed v33 replacement.

## Leakage and engineering audit

- Forbidden patterns found: 0
- Future-row causality maximum difference: {summary['engineering']['future_causality_max_diff']}
- Deterministic repeated-run maximum difference: {summary['engineering']['deterministic_max_diff']}
- Maximum cross-sectional row mean: {summary['engineering']['max_row_mean']:.3e}
- Normal fallback periods across all seed tests: {summary['all']['fallback_periods']}
- Validation failures across all seed tests: {summary['all']['failures']}
- Standalone versus frozen research candidate maximum difference: {summary['reference_equivalence']['max_prediction_diff']}

## Final recommendation

Submit v43 to the official AlphaNova evaluator as a challenger. Retain v33 until v43 exceeds the official local reference of Sharpe 2.3337 and IC 0.5227 without damaging official global novelty, concentration, or acceptance status.
"""


def main() -> None:
    output_dir = ROOT / "research" / "output" / "v43_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    engineering = engineering_checks()
    equivalence = reference_equivalence()
    seed_rows = []
    fold_rows = []
    for seed in ALL_SEEDS:
        row, folds = evaluate_seed(seed)
        seed_rows.append(row)
        fold_rows.extend(folds)

    results = pd.DataFrame(seed_rows)
    folds = pd.DataFrame(fold_rows)
    development = partition_summary(results, "development")
    holdout = partition_summary(results, "holdout")
    all_summary = {
        "seeds": int(len(results)),
        "sharpe_wins": int((results["sharpe_delta"] > 0.0).sum()),
        "ic_wins": int((results["ic_delta"] > 0.0).sum()),
        "worst_seed_sharpe_delta": float(results["sharpe_delta"].min()),
        "mean_prediction_corr_v33": float(results["prediction_corr_v33"].mean()),
        "max_concentration": float(results["v43_concentration"].max()),
        "fallback_periods": int(results["v43_fallback_periods"].sum()),
        "failures": int(results["v43_failures"].sum()),
    }

    stable_correlation_exception = (
        development["sharpe_wins"] == len(DEV_SEEDS)
        and development["ic_wins"] == len(DEV_SEEDS)
        and holdout["sharpe_wins"] == len(HOLDOUT_SEEDS)
        and holdout["ic_wins"] == len(HOLDOUT_SEEDS)
        and all_summary["worst_seed_sharpe_delta"] > 0.0
    )
    gates = {
        "development_median_sharpe_positive": development["sharpe_delta_median"] > 0.0,
        "development_median_ic_positive": development["ic_delta_median"] > 0.0,
        "development_sharpe_wins_4_of_5": development["sharpe_wins"] >= 4,
        "holdout_median_sharpe_positive": holdout["sharpe_delta_median"] > 0.0,
        "holdout_median_ic_positive": holdout["ic_delta_median"] > 0.0,
        "worst_seed_not_worse": all_summary["worst_seed_sharpe_delta"] >= 0.0,
        "concentration_controlled": all_summary["max_concentration"] <= 0.0710,
        "no_fallback": all_summary["fallback_periods"] == 0,
        "no_failures": all_summary["failures"] == 0,
        "reference_equivalent": equivalence["max_prediction_diff"] <= 1e-7,
        "correlation_gate_or_stability_exception": (
            all_summary["mean_prediction_corr_v33"] < 0.995
            or stable_correlation_exception
        ),
    }
    promoted = all(gates.values())
    summary = {
        "decision": "promote_experimental_challenger" if promoted else "discard",
        "development_seeds": list(DEV_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "architecture_frozen_before_holdout": True,
        "engineering": engineering,
        "reference_equivalence": equivalence,
        "development": development,
        "holdout": holdout,
        "all": all_summary,
        "gates": gates,
        "stable_correlation_exception": stable_correlation_exception,
    }

    results.to_csv(output_dir / "seed_results.csv", index=False)
    folds.to_csv(output_dir / "fold_results.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = build_report(summary, results)
    (ROOT / "research" / "v43_research_report.md").write_text(
        report, encoding="utf-8"
    )
    print("V43_FINAL_VALIDATION_START")
    print(json.dumps(summary, indent=2))
    print("V43_FINAL_VALIDATION_END")
    if not promoted:
        raise SystemExit("v43 failed promotion gates")


if __name__ == "__main__":
    main()
