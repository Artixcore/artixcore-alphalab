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
