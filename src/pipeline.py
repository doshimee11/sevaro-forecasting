"""Load -> clean/join -> forecast -> backtest -> write outputs/*.json.

Run: python -m src.pipeline
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src import backtest, busiest_hour, call_forecast, consult_forecast, data_loader

OUT_DIR = "outputs"
CONSULT_HORIZON_MONTHS = 6
CALL_HORIZON_DAYS = 60


def _series_to_records(index, values, date_fmt: str) -> list[dict]:
    return [
        {"date": pd.Timestamp(d).strftime(date_fmt), "value": round(float(v), 2)}
        for d, v in zip(index, values)
    ]


def run_consult_forecasts(tables: dict, caselogs_local: pd.DataFrame) -> dict:
    daily = data_loader.daily_consult_counts(caselogs_local)
    result = {}
    for facility in sorted(daily["facility"].unique()):
        monthly = consult_forecast.to_monthly_series(daily, facility, "consults")
        if len(monthly) < 2:
            continue

        fc = consult_forecast.forecast_monthly_consults(monthly, horizon=CONSULT_HORIZON_MONTHS)
        bt = backtest.rolling_origin_backtest(
            monthly, consult_forecast.point_forecast_fn_factory(), horizon=min(3, len(monthly) // 2 or 1),
            n_splits=3, min_train_size=max(4, len(monthly) // 3),
        )

        result[facility] = dict(
            tier=fc["tier"],
            history=_series_to_records(monthly.index, monthly.values, "%Y-%m"),
            forecast=[
                {"date": pd.Timestamp(d).strftime("%Y-%m"), "point": round(float(p), 2),
                 "lower": round(float(lo), 2), "upper": round(float(hi), 2)}
                for d, p, lo, hi in zip(fc["index"], fc["point"], fc["lower"], fc["upper"])
            ],
            backtest=bt,
        )
    return result


def run_call_forecasts(tables: dict, calls_local: pd.DataFrame) -> dict:
    daily = data_loader.daily_call_counts(calls_local)
    result = {}
    for facility in sorted(daily["facility"].unique()):
        series = call_forecast.to_daily_series(daily, facility, "calls")
        if len(series) < 5:
            continue

        fc = call_forecast.forecast_daily_calls(series, horizon=CALL_HORIZON_DAYS)
        bt = backtest.rolling_origin_backtest(
            series, call_forecast.point_forecast_fn, horizon=min(14, len(series) // 3 or 1),
            n_splits=3, min_train_size=max(14, len(series) // 3),
        )

        history_tail = series.tail(365)  # keep the JSON light; full series still used for modeling
        result[facility] = dict(
            tier=fc["tier"],
            history=_series_to_records(history_tail.index, history_tail.values, "%Y-%m-%d"),
            forecast=[
                {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "point": round(float(p), 2),
                 "lower": round(float(lo), 2), "upper": round(float(hi), 2)}
                for d, p, lo, hi in zip(fc["index"], fc["point"], fc["lower"], fc["upper"])
            ],
            backtest=bt,
        )
    return result


def compute_missed_rate_trend(caselogs_local: pd.DataFrame, calls_local: pd.DataFrame) -> dict:
    """Monthly missed-consult and missed-call rate per hospital."""
    def monthly_rate(df: pd.DataFrame, missed_col: str) -> pd.DataFrame:
        d = df[["facility", "local_date", missed_col]].copy()
        d["month"] = pd.to_datetime(d["local_date"]).dt.to_period("M").dt.to_timestamp()
        grouped = d.groupby(["facility", "month"])[missed_col].agg(missed="sum", total="size")
        grouped["rate"] = grouped["missed"] / grouped["total"]
        return grouped.reset_index()

    consult_rates = monthly_rate(caselogs_local, "isMissed")
    call_rates = monthly_rate(calls_local, "is_call_missed")

    def to_records(rates: pd.DataFrame, facility: str) -> list[dict]:
        sub = rates[rates["facility"] == facility]
        return [
            {"date": m.strftime("%Y-%m"), "missed": int(mi), "total": int(t), "rate": round(float(r), 4)}
            for m, mi, t, r in zip(sub["month"], sub["missed"], sub["total"], sub["rate"])
        ]

    facilities = sorted(set(consult_rates["facility"]) | set(call_rates["facility"]))
    return {
        fac: dict(consult=to_records(consult_rates, fac), call=to_records(call_rates, fac))
        for fac in facilities
    }


def compute_weekday_pattern(daily_consults: pd.DataFrame, daily_calls: pd.DataFrame) -> dict:
    """Average consults/calls by local day-of-week per hospital, zero-count days included."""
    def avg_by_weekday(df: pd.DataFrame, value_col: str) -> dict:
        d = df.copy()
        d["dow"] = pd.to_datetime(d["local_date"]).dt.dayofweek
        grouped = d.groupby(["facility", "dow"])[value_col].mean()
        result = {}
        for fac in d["facility"].unique():
            result[fac] = [round(float(grouped.get((fac, dow), 0.0)), 2) for dow in range(7)]
        return result

    return dict(
        consults=avg_by_weekday(daily_consults, "consults"),
        calls=avg_by_weekday(daily_calls, "calls"),
    )


def run_busiest_hour(calls_local: pd.DataFrame) -> dict:
    result = {}
    for facility in sorted(calls_local["facility"].unique()):
        result[facility] = busiest_hour.analyze_busiest_hour(calls_local, facility)
    return result


def compute_data_health(tables: dict, caselogs_local: pd.DataFrame, calls_local: pd.DataFrame,
                         consult_results: dict, call_results: dict) -> dict:
    """Network-wide and per-hospital data quality/coverage summary."""
    caselogs_total = len(tables["caselogs"])
    calls_total = len(tables["calls"])
    caselogs_dropped = caselogs_total - len(caselogs_local)
    calls_dropped = calls_total - len(calls_local)

    network = dict(
        caselogs_total=int(caselogs_total),
        caselogs_dropped=int(caselogs_dropped),
        caselogs_dropped_pct=round(caselogs_dropped / caselogs_total * 100, 2) if caselogs_total else 0,
        calls_total=int(calls_total),
        calls_dropped=int(calls_dropped),
        calls_dropped_pct=round(calls_dropped / calls_total * 100, 2) if calls_total else 0,
    )

    by_facility = {}
    all_facilities = sorted(set(caselogs_local["facility"].unique()) | set(calls_local["facility"].unique()))
    for fac in all_facilities:
        fac_consults = caselogs_local[caselogs_local["facility"] == fac]
        fac_calls = calls_local[calls_local["facility"] == fac]
        by_facility[fac] = dict(
            consult_history_start=str(fac_consults["local_date"].min()) if len(fac_consults) else None,
            consult_history_end=str(fac_consults["local_date"].max()) if len(fac_consults) else None,
            total_consults=int(len(fac_consults)),
            call_history_start=str(fac_calls["local_date"].min()) if len(fac_calls) else None,
            call_history_end=str(fac_calls["local_date"].max()) if len(fac_calls) else None,
            total_calls=int(len(fac_calls)),
            consult_tier=consult_results.get(fac, {}).get("tier"),
            call_tier=call_results.get(fac, {}).get("tier"),
        )

    return dict(network=network, by_facility=by_facility)


def compute_network_trend(consult_results: dict, call_results: dict) -> dict:
    """Sum each hospital's history/forecast into one network-wide series.

    Combined interval assumes independent errors: half-width is the
    root-sum-of-squares of each hospital's half-width, not a flat sum.
    """
    def combine(results: dict) -> dict:
        hist_totals: dict[str, float] = {}
        hist_contributors: dict[str, int] = {}
        for res in results.values():
            for rec in res["history"]:
                hist_totals[rec["date"]] = hist_totals.get(rec["date"], 0) + rec["value"]
                hist_contributors[rec["date"]] = hist_contributors.get(rec["date"], 0) + 1

        # Drop trailing dates with partial participation (a stray call past local
        # midnight can leave one hospital a day ahead of the rest).
        dates_sorted = sorted(hist_totals)
        full_participation = max(hist_contributors.values()) if hist_contributors else 0
        while dates_sorted and hist_contributors[dates_sorted[-1]] < full_participation:
            dates_sorted.pop()

        history = [{"date": d, "value": round(hist_totals[d], 2)} for d in dates_sorted]

        fc_point: dict[str, float] = {}
        fc_lower_sq: dict[str, float] = {}
        fc_upper_sq: dict[str, float] = {}
        for res in results.values():
            for rec in res["forecast"]:
                d = rec["date"]
                fc_point[d] = fc_point.get(d, 0) + rec["point"]
                fc_lower_sq[d] = fc_lower_sq.get(d, 0) + (rec["point"] - rec["lower"]) ** 2
                fc_upper_sq[d] = fc_upper_sq.get(d, 0) + (rec["upper"] - rec["point"]) ** 2

        forecast = []
        for d in sorted(fc_point):
            point = fc_point[d]
            lower = max(point - fc_lower_sq[d] ** 0.5, 0)
            upper = point + fc_upper_sq[d] ** 0.5
            forecast.append({"date": d, "point": round(point, 2), "lower": round(lower, 2), "upper": round(upper, 2)})

        return dict(history=history, forecast=forecast, n_hospitals=len(results))

    return dict(consults=combine(consult_results), calls=combine(call_results))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = data_loader.load_raw_tables()
    caselogs_local = data_loader.clean_caselogs(tables)
    calls_local = data_loader.clean_calls_with_facility(tables)

    facilities_meta = tables["facilities"].set_index("_id").to_dict(orient="index")

    consult_results = run_consult_forecasts(tables, caselogs_local)
    call_results = run_call_forecasts(tables, calls_local)
    busiest_hour_results = run_busiest_hour(calls_local)
    data_health = compute_data_health(tables, caselogs_local, calls_local, consult_results, call_results)
    network_trend = compute_network_trend(consult_results, call_results)
    missed_rate = compute_missed_rate_trend(caselogs_local, calls_local)
    weekday_pattern = compute_weekday_pattern(
        data_loader.daily_consult_counts(caselogs_local), data_loader.daily_call_counts(calls_local)
    )

    with open(f"{OUT_DIR}/facilities.json", "w") as f:
        json.dump(facilities_meta, f, indent=2, default=str)
    with open(f"{OUT_DIR}/consult_forecast.json", "w") as f:
        json.dump(consult_results, f, indent=2, default=str)
    with open(f"{OUT_DIR}/call_forecast.json", "w") as f:
        json.dump(call_results, f, indent=2, default=str)
    with open(f"{OUT_DIR}/busiest_hour.json", "w") as f:
        json.dump(busiest_hour_results, f, indent=2, default=str)
    with open(f"{OUT_DIR}/data_health.json", "w") as f:
        json.dump(data_health, f, indent=2, default=str)
    with open(f"{OUT_DIR}/network_trend.json", "w") as f:
        json.dump(network_trend, f, indent=2, default=str)
    with open(f"{OUT_DIR}/missed_rate.json", "w") as f:
        json.dump(missed_rate, f, indent=2, default=str)
    with open(f"{OUT_DIR}/weekday_pattern.json", "w") as f:
        json.dump(weekday_pattern, f, indent=2, default=str)

    print(f"Wrote outputs for {len(consult_results)} hospitals (consults), "
          f"{len(call_results)} (calls), {len(busiest_hour_results)} (busiest hour) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
