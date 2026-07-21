# Reconstructed AlphaLab local toolkit

This directory restores a self-contained local compatibility runner for the Artixcore AlphaLab predictor files.

It is not AlphaNova's private evaluator and does not include the competition's private dataset, hidden gauge matrices, city library, or server-side scoring. Its purpose is to catch imports, shape errors, NaNs, gauge-dimension failures, and obvious walk-forward regressions before submission.

## Files

- `runner.py`: command-line validator and metric report
- `walkforward.py`: leakage-safe per-period train/validation loop
- `city_tools.py`: shape-safe gauge projection and local novelty helpers
- `demo_engineered.py`: deterministic 6-feature, 20-asset synthetic dataset
- `predictor.py`: minimal base class expected by submissions
- `requirements.txt`: local dependencies

## Commands

From the repository root:

```bash
cd files
python runner.py ../artixcore_alphalab_v18.py
python runner.py ../artixcore_alphalab_v18.py --full
python runner.py ../artixcore_alphalab_v18.py --full --gauge-fix
```

Use a shorter smoke test while developing:

```bash
python runner.py ../artixcore_alphalab_v21.py --periods 3 --rows 80
```

The generated `results.csv` contains synthetic local statistics. Do not compare the absolute values directly with AlphaNova's official validation or leaderboard results.
