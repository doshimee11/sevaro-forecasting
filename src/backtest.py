"""Shared error metrics and rolling-origin backtesting utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE is undefined near zero, which happens often for sparse hospitals.
    nonzero = y_true > 0.5
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")

    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)

    return dict(mae=mae, rmse=rmse, mape=mape, smape=smape, n=int(len(y_true)))


def rolling_origin_backtest(
    series: pd.Series,
    forecast_fn: Callable[[pd.Series, int], np.ndarray],
    horizon: int,
    n_splits: int = 3,
    min_train_size: int = 30,
    step: int | None = None,
) -> dict:
    """Rolling-origin (expanding window) backtest. series must be sorted, no gaps."""
    n = len(series)
    step = step or horizon

    max_splits_possible = (n - min_train_size) // step
    if max_splits_possible < 1:
        return dict(status="insufficient_history", n_obs=n, metrics=None, splits=[])

    n_splits = max(1, min(n_splits, max_splits_possible))

    splits = []
    all_true, all_pred = [], []
    for i in range(n_splits):
        train_end = n - (n_splits - i) * step
        train_end = max(train_end, min_train_size)
        test_end = min(train_end + horizon, n)
        if test_end <= train_end:
            continue

        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]

        try:
            preds = forecast_fn(train, len(test))
        except Exception as exc:  # keep backtesting robust to a single model failing on a fold
            splits.append(dict(train_end=train_end, test_end=test_end, error=str(exc)))
            continue

        preds = np.asarray(preds)[: len(test)]
        m = compute_metrics(test.values, preds)
        splits.append(dict(train_end=train_end, test_end=test_end, metrics=m))
        all_true.extend(test.values.tolist())
        all_pred.extend(preds.tolist())

    if not all_true:
        return dict(status="all_folds_failed", n_obs=n, metrics=None, splits=splits)

    overall = compute_metrics(np.array(all_true), np.array(all_pred))
    return dict(status="ok", n_obs=n, metrics=overall, splits=splits)
