"""Synthetic data generator: facilities, doctors, caselogs, calls, schedules.

Covers mixed volume tiers, mixed time zones, a brand-new hospital, an
intermittent hospital, and some unjoinable/messy rows, so the pipeline has
real quirks to handle.

Run: python data/generate_synthetic_data.py
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

OUT_DIR = "data/raw"

# Fixed "as of" date so the dataset stays reproducible regardless of wall-clock time.
AS_OF_DATE = date(2025, 9, 14)
HISTORY_DAYS = 730  # ~24 months of history ending at AS_OF_DATE
HISTORY_START = AS_OF_DATE - timedelta(days=HISTORY_DAYS)

US_HOLIDAYS = holidays.US(years=range(HISTORY_START.year, AS_OF_DATE.year + 2))

CONSULT_TYPES = ["Stroke", "TIA", "General Neuro", "Post-Stroke Follow-up", "Seizure"]
CALL_TYPES = ["Consult Request", "Follow-up", "Nurse Line", "Callback"]


def hour_of_day_distribution(peaks: list[tuple[int, float]], spread: float = 2.5) -> np.ndarray:
    """Build a 24-length probability vector from one or more circular Gaussian bumps."""
    hours = np.arange(24)
    probs = np.full(24, 0.02)  # small floor so every hour is possible
    for peak_hour, weight in peaks:
        d = np.minimum(np.abs(hours - peak_hour), 24 - np.abs(hours - peak_hour))
        probs = probs + weight * np.exp(-0.5 * (d / spread) ** 2)
    probs = probs / probs.sum()
    return probs


FACILITIES = [
    dict(
        _id="FAC001", name="Riverside Regional Medical Center", shortName="Riverside",
        timeZone="America/New_York", state="NY", city="Albany", isER=True,
        base_rate=18.0, growth_per_year=0.05, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.05],
        hour_peaks=[(2, 1.0), (14, 0.6)], winter_boost=0.15, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC002", name="Lakeside Community Hospital", shortName="Lakeside",
        timeZone="America/New_York", state="NY", city="Buffalo", isER=True,
        base_rate=10.0, growth_per_year=0.02, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.9],
        hour_peaks=[(20, 1.0), (9, 0.5)], winter_boost=0.10, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC003", name="Oakview General Hospital", shortName="Oakview",
        timeZone="America/Chicago", state="IL", city="Peoria", isER=True,
        base_rate=7.0, growth_per_year=0.0, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        hour_peaks=[(19, 1.0)], winter_boost=0.20, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC004", name="Pinecrest Health System", shortName="Pinecrest",
        timeZone="America/New_York", state="NC", city="Asheville", isER=False,
        base_rate=5.0, growth_per_year=0.03, dow_mult=[1.1, 1.1, 1.1, 1.1, 1.0, 0.6, 0.5],
        hour_peaks=[(10, 1.0), (14, 0.8)], winter_boost=0.05, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC005", name="Summit Care Medical Center", shortName="Summit",
        timeZone="America/Los_Angeles", state="CA", city="Fresno", isER=True,
        base_rate=8.0, growth_per_year=0.04, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        hour_peaks=[(1, 1.0), (16, 0.5)], winter_boost=0.10, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC006", name="Brookhaven Hospital", shortName="Brookhaven",
        timeZone="America/New_York", state="VA", city="Roanoke", isER=False,
        base_rate=2.0, growth_per_year=0.0, dow_mult=[1.1, 1.1, 1.0, 1.0, 1.0, 0.7, 0.6],
        hour_peaks=[(11, 1.0)], winter_boost=0.05, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC007", name="Cedar Valley Regional", shortName="Cedar Valley",
        timeZone="America/New_York", state="OH", city="Dayton", isER=True,
        base_rate=4.0, growth_per_year=0.02, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        hour_peaks=[(3, 1.0)], winter_boost=0.15, start_offset_days=0,
        gap=(date(2024, 6, 1), date(2024, 7, 20)),  # onboarding/outage gap
    ),
    dict(
        _id="FAC008", name="Maple Grove Hospital", shortName="Maple Grove",
        timeZone="America/New_York", state="PA", city="Erie", isER=False,
        base_rate=1.0, growth_per_year=0.0, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7],
        hour_peaks=[(13, 1.0)], winter_boost=0.0, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC009", name="Harborview Medical Center", shortName="Harborview",
        timeZone="America/New_York", state="MA", city="Worcester", isER=True,
        base_rate=15.0, growth_per_year=0.12, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 1.0],
        hour_peaks=[(21, 1.0), (8, 0.6)], winter_boost=0.15, start_offset_days=0, gap=None,
    ),
    dict(
        _id="FAC010", name="Northgate Hospital", shortName="Northgate",
        timeZone="America/New_York", state="GA", city="Savannah", isER=True,
        base_rate=6.0, growth_per_year=0.0, dow_mult=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        hour_peaks=[(4, 1.0)], winter_boost=0.10,
        start_offset_days=HISTORY_DAYS - 95,  # only ~95 days of history: brand-new hospital
        gap=None,
    ),
]

for fac in FACILITIES:
    fac["hour_probs"] = hour_of_day_distribution(fac["hour_peaks"])


def month_seasonal_mult(d: date, winter_boost: float) -> float:
    # Winter (Dec-Feb) is busier, summer (Jun-Aug) quieter: stroke-incidence-style seasonality.
    month = d.month
    if month in (12, 1, 2):
        return 1.0 + winter_boost
    if month in (6, 7, 8):
        return 1.0 - winter_boost * 0.6
    return 1.0


def holiday_mult(d: date, is_er: bool) -> float:
    if d in US_HOLIDAYS:
        return 0.92 if is_er else 0.6  # ERs dip less than elective/non-ER volume
    return 1.0


def active_date_range(fac: dict) -> tuple[date, date]:
    start = HISTORY_START + timedelta(days=fac["start_offset_days"])
    return start, AS_OF_DATE


def daily_lambda(fac: dict, d: date) -> float:
    start, _ = active_date_range(fac)
    days_since_start = (d - start).days
    years_since_start = days_since_start / 365.25
    trend = 1.0 + fac["growth_per_year"] * years_since_start

    # Ramp-up curve for brand-new hospitals: first ~60 days scale up from ~25% to 100%.
    ramp = 1.0
    if fac["start_offset_days"] > 0:
        days_live = (d - start).days
        ramp = min(1.0, 0.25 + 0.75 * (days_live / 60.0)) if days_live < 60 else 1.0

    dow = d.weekday()
    lam = (
        fac["base_rate"]
        * trend
        * ramp
        * fac["dow_mult"][dow]
        * month_seasonal_mult(d, fac["winter_boost"])
        * holiday_mult(d, fac["isER"])
    )
    return max(lam, 0.05)


def local_datetime_to_utc(d: date, hour: int, minute: int, tz_name: str) -> pd.Timestamp:
    local = datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(tz_name))
    return pd.Timestamp(local).tz_convert("UTC")


def generate_caselogs_and_calls():
    caselog_rows = []
    call_rows = []

    for fac in FACILITIES:
        start, end = active_date_range(fac)
        if fac["gap"] is not None:
            gap_start, gap_end = fac["gap"]

        d = start
        while d <= end:
            if fac["gap"] is not None and gap_start <= d <= gap_end:
                d += timedelta(days=1)
                continue

            lam = daily_lambda(fac, d)
            n_consults = RNG.poisson(lam)
            if n_consults == 0:
                d += timedelta(days=1)
                continue

            hours = RNG.choice(24, size=n_consults, p=fac["hour_probs"])
            for hour in hours:
                minute = int(RNG.integers(0, 60))
                created_utc = local_datetime_to_utc(d, int(hour), minute, fac["timeZone"])

                case_id = str(uuid.uuid4())
                is_missed = RNG.random() < 0.03
                is_stroke = RNG.random() < (0.35 if fac["isER"] else 0.08)
                consult_type = "Stroke" if is_stroke else RNG.choice(
                    [c for c in CONSULT_TYPES if c != "Stroke"]
                )
                is_video = RNG.random() < 0.55
                handling_minutes = RNG.gamma(shape=3.0, scale=12.0)  # ~36 min avg
                last_updated = created_utc + timedelta(minutes=float(handling_minutes))

                # ~2% missing facility link, so the pipeline has to tolerate it.
                facility_value = fac["_id"] if RNG.random() > 0.02 else None

                caselog_rows.append(dict(
                    _id=case_id,
                    createdAt=created_utc,
                    lastUpdatedAt=last_updated,
                    facility=facility_value,
                    doctor=f"DOC{int(RNG.integers(1, 31)):03d}",
                    status="Closed" if RNG.random() > 0.05 else "Open",
                    active=bool(RNG.random() > 0.02),
                    isMissed=bool(is_missed),
                    consultType=consult_type,
                    childConsultType=consult_type,
                    type="New" if RNG.random() > 0.3 else "Follow-up",
                    isStrokeAlert=bool(is_stroke),
                    phoneConsult=bool(not is_video),
                    phoneConsultVideo=bool(is_video),
                ))

                # 1-3 calls per case, clustered tightly around case creation.
                n_calls = int(RNG.integers(1, 4))
                for _ in range(n_calls):
                    jitter_min = float(RNG.normal(0, 4))
                    call_init = created_utc + timedelta(minutes=jitter_min)
                    call_missed = RNG.random() < 0.06
                    call_type = "Consult Request" if _ == 0 else RNG.choice(CALL_TYPES)

                    if call_missed:
                        call_accept = pd.NaT
                        call_end = call_init + timedelta(seconds=float(RNG.integers(5, 30)))
                    else:
                        accept_delay_s = float(RNG.gamma(shape=2.0, scale=6.0))
                        call_accept = call_init + timedelta(seconds=accept_delay_s)
                        talk_min = float(RNG.gamma(shape=2.0, scale=4.0))
                        call_end = call_accept + timedelta(minutes=talk_min)

                    call_rows.append(dict(
                        uuid=str(uuid.uuid4()),
                        call_initiated_at=call_init,
                        call_accepted_at=call_accept,
                        call_ended_at=call_end,
                        case_id=case_id,
                        callType=call_type,
                        doctor=f"DOC{int(RNG.integers(1, 31)):03d}",
                        is_call_missed=bool(call_missed),
                    ))

            d += timedelta(days=1)

    caselogs = pd.DataFrame(caselog_rows)
    calls = pd.DataFrame(call_rows)

    # ~2% orphan calls: no matching case_id, so they can't be attributed to a hospital.
    n_orphan = int(len(calls) * 0.02)
    orphan_rows = []
    any_fac = FACILITIES[0]
    for _ in range(n_orphan):
        random_day_offset = int(RNG.integers(0, HISTORY_DAYS))
        d = HISTORY_START + timedelta(days=random_day_offset)
        hour = int(RNG.integers(0, 24))
        minute = int(RNG.integers(0, 60))
        call_init = local_datetime_to_utc(d, hour, minute, any_fac["timeZone"])
        orphan_rows.append(dict(
            uuid=str(uuid.uuid4()),
            call_initiated_at=call_init,
            call_accepted_at=pd.NaT,
            call_ended_at=call_init + timedelta(seconds=10),
            case_id=None,
            callType="Misc",
            doctor=None,
            is_call_missed=True,
        ))
    calls = pd.concat([calls, pd.DataFrame(orphan_rows)], ignore_index=True)

    return caselogs, calls


def generate_facilities_table() -> pd.DataFrame:
    rows = []
    for fac in FACILITIES:
        rows.append(dict(
            _id=fac["_id"], name=fac["name"], shortName=fac["shortName"],
            timeZone=fac["timeZone"], state=fac["state"], city=fac["city"],
            isER=fac["isER"], active=True,
        ))
    return pd.DataFrame(rows)


def generate_doctors_table() -> pd.DataFrame:
    rows = []
    fac_ids = [f["_id"] for f in FACILITIES]
    for i in range(1, 31):
        n_fac = int(RNG.integers(1, 4))
        covered = list(RNG.choice(fac_ids, size=n_fac, replace=False))
        rows.append(dict(
            _id=f"DOC{i:03d}",
            contractType=RNG.choice(["W2", "1099"]),
            facilities=";".join(covered),
            active=True,
        ))
    return pd.DataFrame(rows)


def generate_schedules_table() -> pd.DataFrame:
    """Coarse two-shift-per-day coverage per facility, for context only."""
    rows = []
    doctors_by_fac = {f["_id"]: [] for f in FACILITIES}
    doctors_df = generate_doctors_table()
    for _, row in doctors_df.iterrows():
        for fac_id in row["facilities"].split(";"):
            doctors_by_fac[fac_id].append(row["_id"])

    for fac in FACILITIES:
        start, end = active_date_range(fac)
        d = start
        while d <= end:
            covering = doctors_by_fac.get(fac["_id"]) or ["DOC001"]
            day_doc = RNG.choice(covering)
            night_doc = RNG.choice(covering)
            day_start = local_datetime_to_utc(d, 7, 0, fac["timeZone"])
            day_end = local_datetime_to_utc(d, 19, 0, fac["timeZone"])
            night_start = day_end
            night_end = local_datetime_to_utc(d + timedelta(days=1), 7, 0, fac["timeZone"])
            rows.append(dict(doctor=day_doc, facility=fac["_id"], startTime=day_start,
                              endTime=day_end, callType="primary", shiftType="day", active=True))
            rows.append(dict(doctor=night_doc, facility=fac["_id"], startTime=night_start,
                              endTime=night_end, callType="primary", shiftType="night", active=True))
            d += timedelta(days=7)  # weekly cadence is enough for a context signal
    return pd.DataFrame(rows)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    facilities = generate_facilities_table()
    doctors = generate_doctors_table()
    caselogs, calls = generate_caselogs_and_calls()
    schedules = generate_schedules_table()

    facilities.to_csv(f"{OUT_DIR}/facilities.csv", index=False)
    doctors.to_csv(f"{OUT_DIR}/doctors.csv", index=False)
    caselogs.to_csv(f"{OUT_DIR}/caselogs.csv", index=False)
    calls.to_csv(f"{OUT_DIR}/calls.csv", index=False)
    schedules.to_csv(f"{OUT_DIR}/schedules.csv", index=False)

    print(f"facilities: {len(facilities)} rows")
    print(f"doctors:    {len(doctors)} rows")
    print(f"caselogs:   {len(caselogs)} rows")
    print(f"calls:      {len(calls)} rows")
    print(f"schedules:  {len(schedules)} rows")
    print(f"date range: {HISTORY_START} .. {AS_OF_DATE}")


if __name__ == "__main__":
    main()
