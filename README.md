# Artixcore AlphaLab v0.1

**Leakage-Safe Walk-Forward Cross-Sectional Signal Forecasting Model**

Author: **Ismam Tabriz**  
Built with: **ALI 1.0.0**  
Organization: **Artixcore**  
Project Type: **AI / Quant Research / Machine Learning Competition Prototype**  
Competition Target: **AlphaNova Competition 5  Walk-Forward Cross-Sectional Signal Forecasting**  
Status: **Portfolio-grade research prototype** — local venv smoke test passing; AlphaNova official validation pending

---

## 1. Project Overview

**Artixcore AlphaLab v0.1** is a quantitative machine learning research project created by **Ismam Tabriz and his Agent** under the **Artixcore** brand.

The project is designed for participation in AlphaNova's AI trading prediction competition, where participants build models that generate cross-sectional signals for multiple assets. The goal is not to claim guaranteed profit or to present a finished trading bot. The goal is to demonstrate that Artixcore can design, implement, validate, and document a disciplined AI-powered market signal forecasting system.

This project focuses on:

- Cross-sectional prediction
- Walk-forward validation
- Leakage-safe feature engineering
- CPU-efficient machine learning
- Competition-ready model submission
- Professional research documentation
- Portfolio-ready technical presentation

In simple terms:

> Artixcore AlphaLab turns competition-provided financial features into machine-learning-generated asset ranking signals while following strict validation and anti-leakage rules.

---

## 2. Why This Project Exists

Artixcore is building long-term capabilities in AI systems, quant research, trading automation, and intelligent financial infrastructure.

This project was created as a practical stepping stone toward that larger vision.

Instead of making unrealistic claims like "AI can always beat the market," this project takes a more serious engineering approach:

1. Understand the competition rules.
2. Build a valid model.
3. Avoid leakage and overfitting traps.
4. Generate cross-sectional predictions.
5. Validate locally.
6. Submit a compliant model.
7. Document the process clearly for portfolio viewers.

The main objective is:

> To show that Artixcore can participate in a real AI/quant research environment with a clean, rule-compliant, reproducible model.

---

## 3. Competition Context

AlphaNova's competition asks participants to develop a predictive signal that forecasts the relative returns of a set of assets.

The challenge is cross-sectional. That means the model is not simply asking:

> "Will the market go up or down?"

Instead, it asks something closer to:

> "Which assets are likely to outperform or underperform other assets in the same universe?"

The competition uses walk-forward evaluation. In this structure, the model is trained on past periods and evaluated on a later unseen period. This makes the setup closer to real-world forecasting than a random train/test split.

Important competition characteristics:

- The data is split into sequential periods.
- Each period is independently obfuscated.
- Ticker identities are not stable across periods.
- The model must learn general market patterns, not memorize individual tickers.
- The prediction output must be cross-sectionally de-meaned.
- External data is not allowed.
- Submissions must pass automated checks.
- Overfit or leaking models may be rejected.

---

## 4. What the Model Does

The model is designed to implement a `Predictor` subclass compatible with AlphaNova's submission system.

The core model responsibilities are:

1. Receive competition-provided features.
2. Engineer additional leakage-safe features.
3. Train a conservative machine learning model.
4. Predict one signal value per asset at each timestamp.
5. Reconstruct predictions into the required output format.
6. Cross-sectionally de-mean each prediction row.
7. Return a valid prediction DataFrame.

The model is intentionally conservative. It favors reliability over aggressive leaderboard chasing.

---

## 5. Technical Approach

### 5.1 Data Input

The competition provides a feature DataFrame containing multiple features across multiple assets.

Expected structure:

```text
(feature_name, ticker)
```

Example:

```text
(feature.1, ticker.1)
(feature.1, ticker.2)
(feature.2, ticker.1)
(feature.2, ticker.2)
...
```

The model must handle this panel-style financial dataset and transform it into a machine-learning-ready table.

---

### 5.2 Feature Engineering

Feature engineering is performed using only the competition-provided data.

No external data is used.

Possible engineered features include:

- Raw feature values
- Cross-sectional ranks
- Cross-sectional z-scores
- Rolling means
- Rolling standard deviations
- Short-term momentum features
- Medium-term momentum features
- Mean-reversion-style features
- Volatility-adjusted features
- Simple interaction features

Feature engineering must be past-only and leakage-safe.

Forbidden operations include:

- Future shifts such as `shift(-1)`
- Backward filling with `bfill`
- Centered rolling windows
- Target usage inside prediction
- External market data
- Web APIs
- Social media data
- News data
- Crypto exchange data outside the competition dataset

---

### 5.3 Model Choice

The preferred baseline model is a regularized model such as:

- Ridge Regression
- ElasticNet
- Conservative tree model
- Small LightGBM model, if CPU-safe

For the first version, a regularized model is preferred because it is:

- Fast
- Stable
- Less likely to overfit
- Easier to validate
- More suitable for a portfolio-grade baseline

The aim is not to create the most complex model. The aim is to create a model that runs, validates, and explains well.

---

### 5.4 Prediction Output

The model outputs a signal matrix where:

- Rows represent timestamps.
- Columns represent assets/tickers.
- Values represent predicted relative attractiveness of each asset.

Every row must be cross-sectionally de-meaned:

```python
pred = pred.sub(pred.mean(axis=1), axis=0)
```

This means the sum of predictions across assets at each timestamp should be approximately zero.

This is essential for competition compliance.

---

## 6. Leakage-Safety Principles

A major focus of this project is avoiding data leakage.

Data leakage happens when a model accidentally uses future information that would not have been available at prediction time. In financial modeling, leakage can make a model look powerful in testing but useless in real life.

Artixcore AlphaLab follows these principles:

- Use only current and past feature values.
- Never use future target values.
- Never use target values in `predict()`.
- Avoid future-looking shifts.
- Avoid backward filling.
- Avoid centered rolling windows.
- Avoid external data.
- Keep all model logic inside the submitted `Predictor` class.

The model is designed with competition rejection checks in mind.

---

## 7. Competition Compliance Checklist

The submission should satisfy the following requirements:

- [ ] Uses only one AlphaNova account.
- [ ] Uses no external data.
- [ ] Contains a valid `Predictor` subclass.
- [ ] Implements `train(features, target)`.
- [ ] Implements `predict(features)`.
- [ ] Keeps all custom logic inside the Predictor subclass.
- [ ] Avoids helper functions outside the class.
- [ ] Avoids extra classes outside the class.
- [ ] Does not modify competition-provided files.
- [ ] Does not use future-looking operations.
- [ ] Does not use target data in prediction.
- [ ] Returns a pandas DataFrame.
- [ ] Preserves timestamp index.
- [ ] Preserves asset/ticker columns.
- [ ] Cross-sectionally de-means every prediction row.
- [ ] Handles NaN and infinite values safely.
- [ ] Trains within CPU time limits.
- [ ] Predicts within time limits.
- [ ] Avoids excessive model complexity.
- [ ] Passes local validation before submission.

---

## 8. Repository File Structure

### 8.1 Current repository (implemented)

```text
artixcore-alphalab/
├── .gitignore                 # ignores .venv/
├── README.md
├── artixcore_alphalab_v01.py  # competition submission file (Predictor subclass)
├── predictor.py               # local dev stub only (replace with AlphaNova SDK for real validation)
├── requirements.txt           # numpy, pandas
├── run_local.py               # local smoke test (synthetic train/predict)
└── .venv/                     # local virtual environment (not committed)
```

**Submission rule:** Only `artixcore_alphalab_v01.py` is uploaded to AlphaNova. Keep all model logic inside that file. `predictor.py`, `run_local.py`, and `requirements.txt` exist for local development and are not part of the competition submission.

### 8.2 Optional portfolio additions (not yet in repo)

```text
artixcore-alphalab/
├── reports/
│   └── Artixcore_AlphaLab_v01_Research_Note.md
├── screenshots/
│   ├── local_validation.png
│   ├── full_walk_forward.png
│   └── submission_confirmation.png
├── notes/
│   └── competition_rules_summary.md
└── LICENSE
```

Important:

Do not publish AlphaNova's private competition dataset unless the competition rules explicitly allow it.

---

## 9. Local Validation

### 9.1 Local venv smoke test (current repo)

Use this first to confirm imports, training, and prediction work on your machine before AlphaNova validation.

**Requirements:** Python 3.11+ (tested on 3.14). If `pip install` fails, recreate the venv with Python 3.12.

**Git Bash (Windows):**

```bash
cd artixcore-alphalab
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -r requirements.txt
python run_local.py
```

**PowerShell (Windows):**

```powershell
cd artixcore-alphalab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python run_local.py
```

**Expected output:**

```text
is_trained_: True
prediction shape: (20, 3)
prediction head:
asset           AAPL      MSFT      GOOG
2020-04-10  0.283401  0.369604 -0.653005
...
```

**Quick import check:**

```bash
python -c "from artixcore_alphalab_v01 import ArtixcoreAlphaLabPredictor; print(ArtixcoreAlphaLabPredictor())"
```

#### `requirements.txt`

```text
numpy>=2.0
pandas>=2.2
```

#### `predictor.py` (local dev stub)

AlphaNova provides `predictor.Predictor` in their competition toolkit. For offline development, the repo includes a minimal stub:

```python
class Predictor:
    """Local dev stub for AlphaLab platform Predictor."""

    pass
```

Replace or remove this stub when running inside the official AlphaNova runner environment.

#### `run_local.py` (local smoke test)

```python
import numpy as np
import pandas as pd

from artixcore_alphalab_v01 import ArtixcoreAlphaLabPredictor


def make_synthetic_panel(n_dates=120, seed=42):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG"]

    features = pd.DataFrame(
        {ticker: rng.standard_normal(n_dates).astype(np.float32) for ticker in tickers},
        index=index,
    )
    target = pd.DataFrame(
        {
            ticker: (
                0.4 * features[ticker]
                + 0.3 * features[ticker].shift(1)
                + 0.1 * rng.standard_normal(n_dates)
            ).astype(np.float32)
            for ticker in tickers
        },
        index=index,
    )
    return features, target


def main():
    features, target = make_synthetic_panel()
    train_features = features.iloc[:-20]
    train_target = target.iloc[:-20]
    predict_features = features.iloc[-20:]

    model = ArtixcoreAlphaLabPredictor()
    model.train(train_features, train_target)
    predictions = model.predict(predict_features)

    print(f"is_trained_: {model.is_trained_}")
    if model.training_error_:
        print(f"training_error_: {model.training_error_}")
    print(f"prediction shape: {predictions.shape}")
    print("prediction head:")
    print(predictions.head())

    if not model.is_trained_:
        raise SystemExit(1)
    if predictions.shape != predict_features.shape:
        raise SystemExit(
            f"unexpected prediction shape: {predictions.shape} vs {predict_features.shape}"
        )


if __name__ == "__main__":
    main()
```

### 9.2 AlphaNova official validation (before submission)

After local smoke tests pass, validate with AlphaNova's official runner and competition dataset.

Recommended commands (from the AlphaNova toolkit directory):

```bash
python runner.py artixcore_alphalab_v01.py
python runner.py artixcore_alphalab_v01.py --full
python runner.py artixcore_alphalab_v01.py --gauge-fix
```

The goal is to confirm:

- The code runs.
- The output shape is correct.
- Predictions are de-meaned.
- There are no avoidable runtime errors.
- The model can complete walk-forward validation.
- The signal is not obviously broken.
- The signal has acceptable novelty characteristics.

---

## 10. What This Project Is Not

This project is **not**:

- A guaranteed profitable trading system
- Financial advice
- A live trading bot
- A hedge fund strategy
- A promise of AlphaNova leaderboard victory
- A claim that Artixcore can predict markets perfectly

This project is:

- A research prototype
- A machine learning competition submission
- A quant engineering demonstration
- A portfolio case study
- A step toward more advanced AI trading infrastructure

---

## 11. Portfolio Description

**Artixcore AlphaLab v0.1** is a leakage-safe quantitative signal forecasting prototype developed by **Ismam Tabriz and his Agent** under **Artixcore**.

The project was created for AlphaNova's walk-forward cross-sectional signal forecasting competition. It demonstrates how Artixcore approaches AI/quant research through disciplined data handling, feature engineering, conservative modeling, validation, and competition-compliant submission design.

The project highlights Artixcore's ability to build:

- AI-powered financial research systems
- Machine learning model pipelines
- Cross-sectional prediction engines
- Walk-forward validation workflows
- Competition-ready technical submissions
- Portfolio-grade research documentation

---

## 12. Artixcore Vision Connection

Artixcore AlphaLab is part of a broader technical vision.

Artixcore aims to build intelligent systems across software, AI, automation, trading, SaaS, and research infrastructure. This project supports that direction by proving a small but meaningful capability:

> Artixcore can build structured AI research tools that operate under real-world constraints.

The market is noisy. The rules are strict. The signal must be clean.

That is exactly why this project matters.

It is not about shouting that the model is unbeatable. It is about showing that Artixcore can step into a serious technical arena and build something valid, disciplined, and measurable.

---

## 13. Key Takeaways

- The project focuses on cross-sectional signal forecasting.
- The model uses only competition-provided data.
- The system avoids future-looking leakage.
- Predictions are cross-sectionally de-meaned.
- The model is designed to be CPU-efficient.
- The submission is built for validation and compliance.
- The project is suitable for Artixcore's AI trading portfolio.
- Local venv development setup is documented (`requirements.txt`, `run_local.py`, `predictor.py` stub).
- AlphaNova official runner validation is the next gate before submission.

---

## 14. Disclaimer

This project is for research, education, competition participation, and portfolio demonstration only.

It does not provide financial advice.

It does not guarantee profit.

It does not guarantee competition ranking.

Any trading or investment decision involves risk. The project should be understood as a technical research prototype, not as a production trading system.

---

## 15. Credits

**Author:** Ismam Tabriz  
**AI / Research Partner:** ALI 1.0 Agent  
**Brand / Organization:** Artixcore  
**Project:** Artixcore AlphaLab v0.1  
**Domain:** AI, Quant Research, Signal Forecasting, Machine Learning, Financial Engineering

---

## 16. References

- AlphaNova: https://www.alphanova.tech/
- AlphaNova Competitions: https://www.alphanova.tech/competition
- AlphaNova Competition 5: https://www.alphanova.tech/competition/competition-5

---

## 17. Suggested Repository Short Description

```text
Artixcore AlphaLab v0.1  A leakage-safe walk-forward cross-sectional signal forecasting model built by Ismam Tabriz and his Agent for AlphaNova's AI/quant competition.
```

---

## 18. Suggested GitHub Topics

```text
machine-learning
quant-research
signal-forecasting
cross-sectional-modeling
walk-forward-validation
financial-engineering
artixcore
alphalab
ai-trading-research
portfolio-project
```

---

## 19. ChatGPT / AI Assistant Context

Copy the block below into ChatGPT (or another AI assistant) to continue development with accurate project context.

```text
PROJECT: Artixcore AlphaLab v0.1
AUTHOR: Ismam Tabriz
ORG: Artixcore
COMPETITION: AlphaNova Competition 5 — Walk-Forward Cross-Sectional Signal Forecasting
URL: https://www.alphanova.tech/competition/competition-5

GOAL:
Build a leakage-safe cross-sectional signal forecasting Predictor for AlphaNova.
Submit ONLY artixcore_alphalab_v01.py. All model logic must stay inside the Predictor subclass.

MAIN CLASS:
- File: artixcore_alphalab_v01.py
- Class: ArtixcoreAlphaLabPredictor(Predictor)
- Methods: train(features, target), predict(features)
- Model: Ridge regression (alpha=8.0), CPU-efficient, max_train_rows=250_000
- Output: pandas DataFrame, rows=timestamps, columns=assets, cross-sectionally de-meaned per row

DATA SHAPE:
- Features: panel DataFrame with MultiIndex columns (feature_name, ticker) or simple ticker columns
- Target: aligned DataFrame/Series for training only (never used in predict)

LOCAL DEV FILES (not submitted):
- requirements.txt → numpy>=2.0, pandas>=2.2
- predictor.py → local Predictor stub (AlphaNova provides the real module)
- run_local.py → synthetic smoke test
- .venv/ → virtual environment

LOCAL RUN (Git Bash on Windows):
  cd artixcore-alphalab
  python -m venv .venv
  source .venv/Scripts/activate
  pip install -r requirements.txt
  python run_local.py

OFFICIAL VALIDATION (AlphaNova toolkit):
  python runner.py artixcore_alphalab_v01.py
  python runner.py artixcore_alphalab_v01.py --full
  python runner.py artixcore_alphalab_v01.py --gauge-fix

COMPLIANCE RULES:
- No external data
- No future-looking ops (no shift(-1), bfill, centered rolling)
- No target in predict()
- No helper functions or extra classes outside the Predictor subclass in the submission file
- Handle NaN/inf safely
- Cross-sectionally de-mean every prediction row

CURRENT STATUS:
- Local venv smoke test passes (is_trained_=True, prediction shape verified)
- Next: run AlphaNova official runner, then submit artixcore_alphalab_v01.py

WHEN HELPING ME:
- Prefer minimal, focused diffs in artixcore_alphalab_v01.py only for submission changes
- Keep leakage-safety and competition compliance as top priorities
- Do not add external data sources or complexity without CPU/time justification
```

### 19.1 Submission entry point (reference)

```python
import numpy as np
import pandas as pd

from predictor import Predictor


class ArtixcoreAlphaLabPredictor(Predictor):
    """
    Artixcore AlphaLab v0.1
    Leakage-safe walk-forward cross-sectional signal forecasting baseline.
    """

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self.alpha = 8.0
        self.coef_ = None
        self.intercept_ = 0.0
        self.x_mean_ = None
        self.x_scale_ = None
        self.feature_columns_ = None
        self.prediction_clip_ = 1.0
        self.is_trained_ = False
        self.max_train_rows = 250_000
        self.training_error_ = None

    def train(self, features, target):
        ...

    def predict(self, features):
        ...
```

The full implementation lives in `artixcore_alphalab_v01.py` (~500 lines). Ask the assistant to read or patch that file rather than duplicating helpers outside the class.
