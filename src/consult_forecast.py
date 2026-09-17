"""6-month consult forecast per hospital, at monthly granularity.

Model picked by how much monthly history a hospital has: seasonal Holt-Winters
at 24+ months, trend-only below that, naive-with-drift under 8 months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

RNG = np.random.default_rng(7)


def to_monthly_series(daily_counts: pd.DataFrame, facility: str, value_col: str) -> pd.Series:
    sub = daily_counts[daily_counts["facility"] == facility].copy()
    sub["local_date"] = pd.to_datetime(sub["local_date"])
    monthly = sub.set_index("local_date")[value_col].resample("MS").sum()

    # Drop a trailing partial month so a mid-month cutoff doesn't read as a volume crash.
    last_data_date = sub["local_date"].max()
    last_month_start = monthly.index[-1]
    last_day_of_month = last_month_start + pd.offsets.MonthEnd(0)
    if last_data_date < last_day_of_month:
        monthly = monthly.iloc[:-1]

    return monthly


def _naive_with_drift(series: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = series.values.astype(float)
    n = len(values)
    mean = values.mean() if n else 0.0
    if n >= 2:
        drift = (values[-1] - values[0]) / max(n - 1, 1)
    else:
        drift = 0.0
    point = np.array([max(values[-1] + drift * (i + 1), 0) for i in range(horizon)]) if n else np.zeros(horizon)
    # Blend toward the historical mean so a single noisy last month can't dominate.
    point = 0.5 * point + 0.5 * mean

    std = values.std() if n > 1 else max(mean * 0.5, 1.0)
    # Confidence widens with horizon and shrinks with more observed history.
    widen = np.sqrt(np.arange(1, horizon + 1)) * (std / max(np.sqrt(n), 1))
    lower = np.clip(point - 1.28 * widen - 0.5 * std, 0, None)
    upper = point + 1.28 * widen + 0.5 * std
    return point, lower, upper


def _holt_winters(series: pd.Series, horizon: int, seasonal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kwargs = dict(trend="add", damped_trend=True, initialization_method="estimated")
    if seasonal:
        kwargs.update(seasonal="add", seasonal_periods=12)
    fit = ExponentialSmoothing(series, **kwargs).fit()

    point = np.asarray(fit.forecast(horizon))
    sims = fit.simulate(horizon, repetitions=300, error="add", rng=RNG)
    lower = sims.quantile(0.1, axis=1).values
    upper = sims.quantile(0.9, axis=1).values
    point = np.clip(point, 0, None)
    lower = np.clip(lower, 0, None)
    upper = np.clip(upper, 0, None)
    return point, lower, upper


def choose_tier(n_months: int) -> str:
    if n_months >= 24:
        return "holt_winters_seasonal"
    if n_months >= 8:
        return "holt_winters_trend"
    return "naive_with_drift"


def forecast_monthly_consults(series: pd.Series, horizon: int = 6) -> dict:
    """Returns point/lower/upper forecast arrays plus which model tier was used."""
    n = len(series)
    tier = choose_tier(n)

    try:
        if tier == "holt_winters_seasonal":
            point, lower, upper = _holt_winters(series, horizon, seasonal=True)
        elif tier == "holt_winters_trend":
            point, lower, upper = _holt_winters(series, horizon, seasonal=False)
        else:
            point, lower, upper = _naive_with_drift(series, horizon)
    except Exception:
        # Fall back to naive rather than crash on a short or unstable series.
        tier = "naive_with_drift (fallback)"
        point, lower, upper = _naive_with_drift(series, horizon)

    future_index = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    return dict(tier=tier, index=future_index, point=point, lower=lower, upper=upper)


def point_forecast_fn_factory(tier_override: str | None = None):
    """Adapter for backtest.rolling_origin_backtest, which expects forecast_fn(train, h) -> array."""
    def _fn(train: pd.Series, horizon: int) -> np.ndarray:
        result = forecast_monthly_consults(train, horizon)
        return result["point"]
    return _fn
