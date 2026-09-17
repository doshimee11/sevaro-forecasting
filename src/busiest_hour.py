"""Busiest local hour per hospital, plus a bootstrap confidence and drift check.

Not a forecast: the modal hour of historical calls, resampled 500 times to see
how solid that mode is, and compared across weekday/weekend and first-half vs
second-half of history for drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(23)
N_BOOTSTRAP = 500


def _mode_hour(hours: np.ndarray) -> int:
    counts = np.bincount(hours, minlength=24)
    return int(np.argmax(counts))


def bootstrap_confidence(hours: np.ndarray, observed_mode: int, n_boot: int = N_BOOTSTRAP) -> float:
    if len(hours) == 0:
        return float("nan")
    matches = 0
    n = len(hours)
    for _ in range(n_boot):
        sample = RNG.choice(hours, size=n, replace=True)
        if _mode_hour(sample) == observed_mode:
            matches += 1
    return matches / n_boot


def analyze_busiest_hour(calls_local: pd.DataFrame, facility: str) -> dict:
    sub = calls_local[calls_local["facility"] == facility]
    hours = sub["local_hour"].to_numpy()

    if len(hours) == 0:
        return dict(facility=facility, status="no_data")

    overall_hour = _mode_hour(hours)
    confidence = bootstrap_confidence(hours, overall_hour)

    hour_counts = np.bincount(hours, minlength=24)
    hour_share = (hour_counts / hour_counts.sum()).tolist()

    dow = sub["local_dow"].to_numpy()
    weekday_hours = hours[dow < 5]
    weekend_hours = hours[dow >= 5]
    weekday_hour = _mode_hour(weekday_hours) if len(weekday_hours) else None
    weekend_hour = _mode_hour(weekend_hours) if len(weekend_hours) else None

    dates = pd.to_datetime(sub["local_date"])
    sorted_idx = np.argsort(dates.values)
    hours_sorted = hours[sorted_idx]
    midpoint = len(hours_sorted) // 2
    first_half_hour = _mode_hour(hours_sorted[:midpoint]) if midpoint > 0 else None
    second_half_hour = _mode_hour(hours_sorted[midpoint:]) if len(hours_sorted) - midpoint > 0 else None

    months = dates.dt.month.to_numpy()
    by_month = {}
    for m in sorted(set(months)):
        by_month[int(m)] = _mode_hour(hours[months == m])
    distinct_month_hours = set(by_month.values())

    heatmap = np.zeros((7, 24), dtype=int)
    for d, h in zip(dow, hours):
        heatmap[int(d), int(h)] += 1

    drift_flag = (
        weekday_hour != weekend_hour
        or first_half_hour != second_half_hour
        or len(distinct_month_hours) > 3
    )

    return dict(
        facility=facility,
        status="ok",
        n_calls=int(len(hours)),
        overall_busiest_hour=overall_hour,
        confidence=confidence,
        hour_share=hour_share,
        weekday_busiest_hour=weekday_hour,
        weekend_busiest_hour=weekend_hour,
        first_half_busiest_hour=first_half_hour,
        second_half_busiest_hour=second_half_hour,
        by_month_busiest_hour=by_month,
        drift_flag=bool(drift_flag),
        heatmap=heatmap.tolist(),
    )
