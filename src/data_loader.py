"""Loads the raw tables and produces timezone-correct, hospital-joined views.

Raw timestamps are UTC; every function here converts to local time before
bucketing by day or hour.
"""

from __future__ import annotations

import pandas as pd

RAW_DIR = "data/raw"


def load_raw_tables(raw_dir: str = RAW_DIR) -> dict[str, pd.DataFrame]:
    facilities = pd.read_csv(f"{raw_dir}/facilities.csv")
    doctors = pd.read_csv(f"{raw_dir}/doctors.csv")
    caselogs = pd.read_csv(f"{raw_dir}/caselogs.csv", parse_dates=["createdAt", "lastUpdatedAt"])
    calls = pd.read_csv(
        f"{raw_dir}/calls.csv",
        parse_dates=["call_initiated_at", "call_accepted_at", "call_ended_at"],
    )
    schedules = pd.read_csv(f"{raw_dir}/schedules.csv", parse_dates=["startTime", "endTime"])

    for col in ["createdAt", "lastUpdatedAt"]:
        caselogs[col] = pd.to_datetime(caselogs[col], utc=True, format="ISO8601")
    for col in ["call_initiated_at", "call_accepted_at", "call_ended_at"]:
        calls[col] = pd.to_datetime(calls[col], utc=True, format="ISO8601")
    for col in ["startTime", "endTime"]:
        schedules[col] = pd.to_datetime(schedules[col], utc=True, format="ISO8601")

    return dict(
        facilities=facilities, doctors=doctors,
        caselogs=caselogs, calls=calls, schedules=schedules,
    )


def _localize(df: pd.DataFrame, utc_col: str, tz_map: pd.Series, out_col: str) -> pd.DataFrame:
    """Convert a UTC timestamp column to each row's hospital-local time.

    Rows with no known facility timezone are dropped rather than guessed at.
    """
    df = df.copy()
    df["_tz"] = df["facility"].map(tz_map)
    df = df.dropna(subset=["_tz", utc_col])

    # Drop the tz after converting, since different hospitals' local timestamps
    # can't share one tz-aware column, but the wall-clock value is all we need.
    local_ts = []
    for tz, group in df.groupby("_tz"):
        local_ts.append(group[utc_col].dt.tz_convert(tz).dt.tz_localize(None))
    df[out_col] = pd.concat(local_ts).sort_index()
    df = df.drop(columns="_tz")
    return df


def clean_caselogs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Caselogs with a known facility, localized to that facility's timezone."""
    caselogs = tables["caselogs"]
    facilities = tables["facilities"]
    tz_map = facilities.set_index("_id")["timeZone"]

    known_facilities = set(tz_map.index)
    valid = caselogs[caselogs["facility"].isin(known_facilities)].copy()
    dropped = len(caselogs) - len(valid)
    if dropped:
        print(f"[data_loader] dropped {dropped} caselogs with missing/unknown facility "
              f"({dropped / len(caselogs):.1%} of rows)")

    valid = _localize(valid, "createdAt", tz_map, "createdAt_local")
    valid["local_date"] = valid["createdAt_local"].dt.date
    valid["local_hour"] = valid["createdAt_local"].dt.hour
    valid["local_dow"] = valid["createdAt_local"].dt.dayofweek
    return valid


def clean_calls_with_facility(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calls joined through caselogs -> facility, localized to hospital time."""
    calls = tables["calls"]
    caselogs = tables["caselogs"]
    facilities = tables["facilities"]

    case_to_facility = caselogs.set_index("_id")["facility"]
    calls = calls.copy()
    calls["facility"] = calls["case_id"].map(case_to_facility)

    dropped = calls["facility"].isna().sum()
    if dropped:
        print(f"[data_loader] dropped {dropped} calls with no matching case/facility "
              f"({dropped / len(calls):.1%} of rows)")
    calls = calls.dropna(subset=["facility"])

    tz_map = facilities.set_index("_id")["timeZone"]
    calls = _localize(calls, "call_initiated_at", tz_map, "call_initiated_at_local")
    calls["local_date"] = calls["call_initiated_at_local"].dt.date
    calls["local_hour"] = calls["call_initiated_at_local"].dt.hour
    calls["local_dow"] = calls["call_initiated_at_local"].dt.dayofweek
    return calls


def daily_consult_counts(caselogs_local: pd.DataFrame) -> pd.DataFrame:
    """One row per (facility, local_date) with a consult count, gaps filled with 0."""
    counts = (
        caselogs_local.groupby(["facility", "local_date"]).size().rename("consults").reset_index()
    )
    return _fill_date_gaps(counts, "consults")


def daily_call_counts(calls_local: pd.DataFrame) -> pd.DataFrame:
    """One row per (facility, local_date) with a call count, gaps filled with 0."""
    counts = (
        calls_local.groupby(["facility", "local_date"]).size().rename("calls").reset_index()
    )
    return _fill_date_gaps(counts, "calls")


def _fill_date_gaps(counts: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reindex each facility to a full daily range so gaps become explicit zeros."""
    out = []
    for facility, group in counts.groupby("facility"):
        series = group.set_index("local_date")[value_col].sort_index()
        full_range = pd.date_range(series.index.min(), series.index.max(), freq="D").date
        series = series.reindex(full_range, fill_value=0)
        filled = pd.DataFrame({"facility": facility, "local_date": full_range, value_col: series.values})
        out.append(filled)
    return pd.concat(out, ignore_index=True)[["facility", "local_date", value_col]]
