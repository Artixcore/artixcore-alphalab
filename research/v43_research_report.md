# AlphaLab v43 Research Report

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
| 1729 | development | 2.314351 | 2.319266 | +0.004915 | 0.512128 | 0.513092 | 0.999455 |
| 3141 | development | 2.285602 | 2.291699 | +0.006098 | 0.518482 | 0.520235 | 0.999450 |
| 5003 | development | 2.301498 | 2.312454 | +0.010955 | 0.516266 | 0.518115 | 0.999419 |
| 7919 | development | 2.312755 | 2.319549 | +0.006795 | 0.512416 | 0.513896 | 0.999472 |
| 12007 | development | 2.381187 | 2.386920 | +0.005733 | 0.512673 | 0.514538 | 0.999459 |
| 18013 | holdout | 2.288007 | 2.295052 | +0.007045 | 0.515116 | 0.516389 | 0.999485 |
| 24023 | holdout | 2.328872 | 2.335904 | +0.007032 | 0.511784 | 0.513597 | 0.999463 |
| 32027 | holdout | 2.344439 | 2.350268 | +0.005828 | 0.510405 | 0.512047 | 0.999481 |

### Development aggregate

- Sharpe mean delta: +0.006899
- Sharpe wins: 5/5
- IC mean delta: +0.001582
- IC wins: 5/5
- Worst seed Sharpe delta: +0.004915
- Median worst-fold Sharpe delta: +0.005173

### Untouched holdout aggregate

- Architecture frozen before holdout: yes
- Sharpe mean delta: +0.006635
- Sharpe wins: 3/3
- IC mean delta: +0.001576
- IC wins: 3/3
- Worst seed Sharpe delta: +0.005828
- Median worst-fold Sharpe delta: +0.005789

## Adversarial review

The main weakness is prediction correlation with v33 of approximately 0.999460, above the preferred 0.995 threshold. The exception is accepted only because Sharpe and IC improved on all eight predeclared seeds, every seed delta was positive, worst-fold behavior improved in median, concentration remained controlled, and the architecture was frozen before holdout evaluation.

The improvement is modest, not a strong-challenger gain of 0.02 Sharpe. Runtime is also higher because v43 fits a second linear branch. These facts prevent promoting it as a guaranteed v33 replacement.

## Leakage and engineering audit

- Forbidden patterns found: 0
- Future-row causality maximum difference: 0.0
- Deterministic repeated-run maximum difference: 0.0
- Maximum cross-sectional row mean: 1.043e-08
- Normal fallback periods across all seed tests: 0
- Validation failures across all seed tests: 0
- Standalone versus frozen research candidate maximum difference: 0.0

## Final recommendation

Submit v43 to the official AlphaNova evaluator as a challenger. Retain v33 until v43 exceeds the official local reference of Sharpe 2.3337 and IC 0.5227 without damaging official global novelty, concentration, or acceptance status.
