# Sevaro Forecasting: Consults & Calls

Take-home submission: forecasting models for hospital-level consult volume, call volume, and busiest-hour-of-day,
plus a Streamlit dashboard for browsing the results per hospital.

**Live dashboard:** [sevaro-forecasting-production.up.railway.app](https://sevaro-forecasting-production.up.railway.app)
**Methodology notebook:** [`notebooks/forecasting.ipynb`](notebooks/forecasting.ipynb), start here for the full
write-up (problem framing, model choices and rejects, backtested accuracy, charts, limitations).

## What's in this repo

```
data/generate_synthetic_data.py   # synthetic data generator (see note below)
src/
  data_loader.py                  # load raw tables, timezone-correct join to facility, fill date gaps
  consult_forecast.py             # 6-month monthly consult forecast, tiered by history length
  call_forecast.py                # 60-day daily call forecast, tiered by history length & volume
  busiest_hour.py                 # busiest local hour per hospital + bootstrap confidence + stability checks
  backtest.py                     # rolling-origin backtesting + MAE/RMSE/MAPE/sMAPE
  pipeline.py                     # orchestrates the above, writes outputs/*.json
notebooks/forecasting.ipynb       # the graded deliverable: narrative + methodology + charts, runs end-to-end
outputs/*.json                    # forecast results consumed by the Streamlit app (generated, not committed)
app/streamlit_app.py              # dashboard: network overview + per-hospital forecasts/busiest-hour heatmap
```

## A note on the data

The real `caselogs` / `calls` / `schedules` / `doctors` / `facilities` tables were not provided for this exercise,
so `data/generate_synthetic_data.py` generates a synthetic dataset matching the spec's schema exactly (same
columns, same join keys) and deliberately includes the operational realities the spec calls out: multiple hospital
time zones, a brand-new hospital with ~3 months of history, an intermittent hospital with a multi-week coverage
gap, weekly/hourly/seasonal patterns that differ per hospital, and messy joins (a small fraction of calls with no
matching case, and a small fraction of consults with a missing facility). The generation seed is fixed, so it's
fully reproducible. Every module in `src/` reads only from the schema described in the spec, so pointing
`data/raw/*.csv` at real exports of the same shape is enough to re-run everything on real data.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate_synthetic_data.py   # writes data/raw/*.csv (deterministic, seeded)
python -m src.pipeline                   # writes outputs/*.json

jupyter notebook notebooks/forecasting.ipynb   # methodology write-up + charts
streamlit run app/streamlit_app.py             # interactive dashboard, http://localhost:8501
```

`data/raw/` and `outputs/` are both generated, not committed, since they're fully reproducible from the seeded
generator and the pipeline. Run the two commands above before `streamlit run` or the app will just point you at
them. The deployed version does this automatically on every boot (see `Procfile`).

## Summary of approach

- **6-month consult forecast (monthly granularity):** Holt-Winters (trend + yearly seasonal) for hospitals with
  ≥24 months of history, trend-only Holt-Winters for 8-23 months, naive-with-drift for newer/sparser hospitals.
- **Daily call forecast (60-day horizon):** Holt-Winters with weekly seasonality for hospitals with ≥90 days of
  history and ≥5 calls/day on average; a day-of-week seasonal-average model otherwise.
- **Busiest hour of day:** the modal local hour of historical call volume, with a bootstrap confidence score and
  an explicit weekday/weekend + first-half/second-half + month-to-month stability check.
- **Validation:** rolling-origin (expanding-window) backtesting for both forecasts, reporting MAE/RMSE/MAPE/sMAPE
  on held-out folds, not in-sample fit.

Full reasoning, rejected alternatives, and limitations are in the notebook.
