from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_v43_candidates import (
    CoreAnchorPredictor,
    V33,
    evaluate_class,
    summarize_seed,
)
from demo_engineered import make_dataset


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEEDS = (18013, 24023, 32027)


def aggregate(results: pd.DataFrame) -> dict[str, float | int]:
    base = results.loc[results["model"] == "v33"].set_index("seed")
    candidate = results.loc[results["model"] == "v43_core_anchor"].set_index("seed")
    aligned = candidate.join(base, lsuffix="_candidate", rsuffix="_v33")
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
            candidate["prediction_corr_v33"].mean()
        ),
        "max_concentration": float(candidate["concentration"].max()),
        "fallback_periods": int(candidate["fallback_periods"].sum()),
        "failures": int(candidate["failures"].sum()),
    }


def main() -> None:
    output_dir = ROOT / "research" / "output" / "v43_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_rows = []

    for seed in HOLDOUT_SEEDS:
        dataset = list(make_dataset(n_periods=77, n_rows=160, seed=seed))
        base_result, base_fallback, _ = evaluate_class(V33, dataset)
        candidate_result, candidate_fallback, _ = evaluate_class(
            CoreAnchorPredictor, dataset
        )
        base_row, base_folds = summarize_seed(
            seed, "v33", base_result, None, base_fallback
        )
        candidate_row, candidate_folds = summarize_seed(
            seed,
            "v43_core_anchor",
            candidate_result,
            base_result,
            candidate_fallback,
        )
        rows.extend([base_row, candidate_row])
        fold_rows.extend(base_folds + candidate_folds)
        print(
            f"seed={seed} v33_sharpe={base_row['sharpe']:.6f} "
            f"v43_sharpe={candidate_row['sharpe']:.6f} "
            f"v33_ic={base_row['ic']:.6f} v43_ic={candidate_row['ic']:.6f}"
        )

    results = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    results.to_csv(output_dir / "holdout_seed_results.csv", index=False)
    folds.to_csv(output_dir / "holdout_fold_results.csv", index=False)
    summary = {
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "architecture_frozen_before_holdout": True,
        "v43_core_anchor": aggregate(results),
    }
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("V43_HOLDOUT_JSON_START")
    print(json.dumps(summary, indent=2))
    print("V43_HOLDOUT_JSON_END")


if __name__ == "__main__":
    main()
