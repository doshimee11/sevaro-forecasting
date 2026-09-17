"""Daily call forecast per hospital, 60-day horizon.

Holt-Winters with weekly seasonality when there's enough daily history and
volume; a day-of-week seasonal average otherwise, since Holt-Winters gets
unstable on small, mostly-zero daily counts.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

RNG = np.random.default_rng(11)

MIN_DAYS_FOR_SEASONAL = 90
MIN_AVG_VOLUME_FOR_SEASONAL = 5.0


def to_daily_series(daily_counts: pd.DataFrame, facility: str, value_col: str) -> pd.Series:
    sub = daily_counts[daily_counts["facility"] == facility].copy()
    sub["local_date"] = pd.to_datetime(sub["local_date"])
    return sub.set_index("local_date")[value_col].asfreq("D", fill_value=0)


def _dow_seasonal_average(series: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    df = series.reset_index()
    df.columns = ["date", "y"]
    df["dow"] = df["date"].dt.dayofweek
    dow_means = df.groupby("dow")["y"].mean()
    overall_mean = df["y"].mean()

    if len(df) >= 28:
        recent = df["y"].tail(28).mean()
        older = df["y"].iloc[:-28].mean() if len(df) > 28 else overall_mean
        trend_factor = float(np.clip(recent / max(older, 0.1), 0.5, 1.5))
    else:
        trend_factor = 1.0

    future_dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    point = np.array([dow_means.get(d.dayofweek, overall_mean) * trend_factor for d in future_dates])

    resid = df["y"] - df["dow"].map(dow_means)
    resid_std = float(resid.std()) if len(resid) > 1 else max(overall_mean * 0.5, 1.0)
    lower = np.clip(point - 1.28 * resid_std, 0, None)
    upper = point + 1.28 * resid_std
    return point, lower, upper, future_dates


def _holt_winters_daily(series: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    fit = ExponentialSmoothing(
        series, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit()
    point = np.clip(np.asarray(fit.forecast(horizon)), 0, None)
    sims = fit.simulate(horizon, repetitions=300, error="add", rng=RNG)
    lower = np.clip(sims.quantile(0.1, axis=1).values, 0, None)
    upper = np.clip(sims.quantile(0.9, axis=1).values, 0, None)
    future_dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=horizon, freq="D")
    return point, lower, upper, future_dates


def choose_tier(series: pd.Series) -> str:
    if len(series) >= MIN_DAYS_FOR_SEASONAL and series.mean() >= MIN_AVG_VOLUME_FOR_SEASONAL:
        return "holt_winters_weekly"
    return "dow_seasonal_average"


def forecast_daily_calls(series: pd.Series, horizon: int = 60) -> dict:
    tier = choose_tier(series)
    try:
        if tier == "holt_winters_weekly":
            point, lower, upper, future_dates = _holt_winters_daily(series, horizon)
        else:
            point, lower, upper, future_dates = _dow_seasonal_average(series, horizon)
    except Exception:
        tier = "dow_seasonal_average (fallback)"
        point, lower, upper, future_dates = _dow_seasonal_average(series, horizon)

    return dict(tier=tier, index=future_dates, point=point, lower=lower, upper=upper)


def point_forecast_fn(train: pd.Series, horizon: int) -> np.ndarray:
    return forecast_daily_calls(train, horizon)["point"]
