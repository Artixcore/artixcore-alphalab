"""Deterministic synthetic panel data for local AlphaLab smoke testing.

This is not AlphaNova's private competition dataset. It only reproduces the
shape and broad cross-sectional forecasting interface used by the local tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeriodData:
    period: int
    features: pd.DataFrame
    target: pd.DataFrame


def _rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float32)
    denominator = max(values.shape[1] - 1, 1)
    return 2.0 * ranks / denominator - 1.0


def make_period(
    period: int,
    n_rows: int = 160,
    n_assets: int = 20,
    n_features: int = 6,
    seed: int = 1729,
) -> PeriodData:
    """Create one independently obfuscated synthetic period."""

    rng = np.random.default_rng(seed + period * 104729)
    assets = [f"Ticker.{i + 1:02d}" for i in range(n_assets)]
    feature_names = [f"Feature.{i + 1}" for i in range(n_features)]

    permutation = rng.permutation(n_assets)
    latent = np.zeros((n_rows, n_assets), dtype=np.float32)
    innovations = rng.standard_normal((n_rows, n_assets)).astype(np.float32)
    for t in range(1, n_rows):
        latent[t] = 0.82 * latent[t - 1] + 0.55 * innovations[t]

    market = rng.standard_normal((n_rows, 1)).astype(np.float32)
    regime = np.sin(np.linspace(0.0, 4.0 * np.pi, n_rows, dtype=np.float32))[:, None]

    raw_features = [
        latent + 0.20 * market,
        -0.35 * latent + 0.65 * innovations + 0.15 * market,
        np.tanh(latent) + 0.18 * rng.standard_normal(latent.shape),
        regime * latent + 0.45 * rng.standard_normal(latent.shape),
        np.abs(innovations) + 0.10 * rng.standard_normal(latent.shape),
        0.45 * np.roll(latent, 1, axis=0) + 0.55 * rng.standard_normal(latent.shape),
    ]

    blocks = []
    for i, name in enumerate(feature_names):
        values = np.asarray(raw_features[i], dtype=np.float32)[:, permutation]
        block = pd.DataFrame(values, columns=assets)
        block.columns = pd.MultiIndex.from_product(
            [[name], assets], names=["feature", "asset"]
        )
        blocks.append(block)

    index = pd.RangeIndex(n_rows, name="time")
    features = pd.concat(blocks, axis=1)
    features.index = index

    f1 = np.asarray(raw_features[0], dtype=np.float32)
    f2 = np.asarray(raw_features[1], dtype=np.float32)
    f3 = np.asarray(raw_features[2], dtype=np.float32)
    rank1 = _rank_rows(f1)
    rank2 = _rank_rows(f2)
    noise = 0.55 * rng.standard_normal((n_rows, n_assets)).astype(np.float32)
    target_values = (
        0.34 * rank1
        - 0.18 * rank2
        + 0.13 * f3
        + 0.10 * np.roll(rank1, 1, axis=0)
        + noise
    )
    target_values[0] = noise[0]
    target_values -= target_values.mean(axis=1, keepdims=True)
    target = pd.DataFrame(target_values[:, permutation], index=index, columns=assets)
    target.columns.name = "asset"

    return PeriodData(period=period, features=features, target=target.astype(np.float32))


def make_dataset(
    n_periods: int = 77,
    n_rows: int = 160,
    n_assets: int = 20,
    n_features: int = 6,
    seed: int = 1729,
) -> list[PeriodData]:
    return [
        make_period(
            period=p,
            n_rows=n_rows,
            n_assets=n_assets,
            n_features=n_features,
            seed=seed,
        )
        for p in range(n_periods)
    ]


def iter_periods(**kwargs) -> Iterator[PeriodData]:
    yield from make_dataset(**kwargs)
