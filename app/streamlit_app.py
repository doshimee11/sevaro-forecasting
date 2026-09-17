"""Reads pre-computed forecasts from outputs/*.json and renders the dashboard."""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP_DIR, "..", "outputs")

st.set_page_config(page_title="Sevaro Forecasting Dashboard", layout="wide", page_icon="\U0001F4C8")


@st.cache_data
def load_outputs():
    with open(os.path.join(OUT_DIR, "facilities.json")) as f:
        facilities = json.load(f)
    with open(os.path.join(OUT_DIR, "consult_forecast.json")) as f:
        consult = json.load(f)
    with open(os.path.join(OUT_DIR, "call_forecast.json")) as f:
        calls = json.load(f)
    with open(os.path.join(OUT_DIR, "busiest_hour.json")) as f:
        busiest = json.load(f)
    with open(os.path.join(OUT_DIR, "data_health.json")) as f:
        data_health = json.load(f)
    with open(os.path.join(OUT_DIR, "network_trend.json")) as f:
        network_trend = json.load(f)
    return facilities, consult, calls, busiest, data_health, network_trend


def missing_outputs_message():
    st.error(
        "No forecast outputs found under `outputs/`. Generate them first:\n\n"
        "```\npython data/generate_synthetic_data.py\npython -m src.pipeline\n```"
    )


def history_forecast_chart(history, forecast, y_label, title):
    hist_df = pd.DataFrame(history)
    fc_df = pd.DataFrame(forecast)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    fc_df["date"] = pd.to_datetime(fc_df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["value"], mode="lines+markers", name="Historical",
        line=dict(color="#2f4b7c", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["point"], mode="lines+markers", name="Forecast",
        line=dict(color="#d1495b", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
        y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(209, 73, 91, 0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="80% interval", hoverinfo="skip",
    ))
    fig.update_layout(
        title=title, yaxis_title=y_label, xaxis_title=None, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def volume_comparison_chart(consult: dict, calls: dict, name_by_id: dict):
    """Trailing 12-month totals per hospital, so history length doesn't skew the comparison."""
    rows = []
    for fac_id, res in consult.items():
        total = sum(r["value"] for r in res["history"][-12:])
        rows.append(dict(facility=name_by_id.get(fac_id, fac_id), metric="Consults (trailing 12mo)", total=total))
    for fac_id, res in calls.items():
        total = sum(r["value"] for r in res["history"])  # already trimmed to <=365 days upstream
        rows.append(dict(facility=name_by_id.get(fac_id, fac_id), metric="Calls (trailing 12mo)", total=total))
    df = pd.DataFrame(rows)

    order = (
        df[df["metric"] == "Calls (trailing 12mo)"].sort_values("total")["facility"].tolist()
    )

    fig = go.Figure()
    for metric, color in [("Consults (trailing 12mo)", "#2f4b7c"), ("Calls (trailing 12mo)", "#118ab2")]:
        sub = df[df["metric"] == metric].set_index("facility").reindex(order).reset_index()
        fig.add_trace(go.Bar(
            y=sub["facility"], x=sub["total"], name=metric, orientation="h",
            marker=dict(color=color),
        ))
    fig.update_layout(
        barmode="group", height=420, xaxis_title="Total volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def data_health_table(data_health: dict, name_by_id: dict) -> pd.DataFrame:
    rows = []
    for fac_id, h in data_health["by_facility"].items():
        rows.append(dict(
            Hospital=name_by_id.get(fac_id, fac_id),
            **{"Consult history": f"{h['consult_history_start']} to {h['consult_history_end']}"},
            **{"Call history": f"{h['call_history_start']} to {h['call_history_end']}"},
            **{"Total consults": h["total_consults"], "Total calls": h["total_calls"]},
            **{"Consult model": h["consult_tier"], "Call model": h["call_tier"]},
        ))
    return pd.DataFrame(rows).sort_values("Hospital")


def backtest_badge(bt: dict):
    status = bt.get("status")
    if status != "ok":
        st.warning(f"Backtest: **{status}**, not enough history for a reliable held-out accuracy estimate.")
        return
    m = bt["metrics"]
    cols = st.columns(4)
    cols[0].metric("MAE", f"{m['mae']:.1f}")
    cols[1].metric("RMSE", f"{m['rmse']:.1f}")
    mape_display = f"{m['mape']:.1f}%" if m["mape"] == m["mape"] else "n/a"
    cols[2].metric("MAPE", mape_display)
    cols[3].metric("sMAPE", f"{m['smape']:.1f}%")


def main():
    if not os.path.exists(os.path.join(OUT_DIR, "facilities.json")):
        missing_outputs_message()
        st.stop()

    facilities, consult, calls, busiest, data_health, network_trend = load_outputs()

    st.title("Sevaro Forecasting Dashboard")
    st.caption(
        "Consult & call volume forecasts and busiest-hour analysis per hospital. "
        "All timestamps are bucketed in each hospital's local time zone."
    )

    facility_ids = sorted(consult.keys())
    name_by_id = {fid: facilities.get(fid, {}).get("name", fid) for fid in facility_ids}
    options = [f"{name_by_id[fid]} ({fid})" for fid in facility_ids]

    with st.sidebar:
        st.header("Hospital")
        choice = st.selectbox("Select a hospital", options, index=0)
        fac_id = facility_ids[options.index(choice)]
        meta = facilities.get(fac_id, {})
        st.markdown(
            f"**Time zone:** {meta.get('timeZone', 'n/a')}  \n"
            f"**Type:** {'ER' if meta.get('isER') else 'Non-ER'}  \n"
            f"**Location:** {meta.get('city', '')}, {meta.get('state', '')}"
        )
        st.divider()
        st.caption(
            "Forecasts are pre-computed by `src/pipeline.py` (Holt-Winters / seasonal models with a "
            "naive fallback for sparse or brand-new hospitals) and backtested via rolling-origin "
            "cross-validation. See the methodology notebook for full details."
        )

    tab0, tab1, tab2, tab3 = st.tabs(
        ["Overview", "6-Month Consult Forecast", "Daily Call Forecast (60 days)", "Busiest Hour"]
    )

    with tab0:
        st.subheader("Volume by hospital")
        st.plotly_chart(volume_comparison_chart(consult, calls, name_by_id), width="stretch")

        st.subheader("Network-wide trend")
        col1, col2 = st.columns(2)
        with col1:
            nc = network_trend["consults"]
            st.plotly_chart(
                history_forecast_chart(nc["history"], nc["forecast"], "Consults / month",
                                        f"All {nc['n_hospitals']} hospitals: monthly consults"),
                width="stretch",
            )
        with col2:
            ncalls = network_trend["calls"]
            st.plotly_chart(
                history_forecast_chart(ncalls["history"], ncalls["forecast"], "Calls / day",
                                        f"All {ncalls['n_hospitals']} hospitals: daily calls"),
                width="stretch",
            )
        st.caption(
            "The combined forecast interval assumes each hospital's error is independent of the "
            "others', so it's narrower than just adding up each hospital's own interval."
        )

        st.subheader("Data health")
        net = data_health["network"]
        c1, c2 = st.columns(2)
        c1.metric("Caselogs dropped (no facility match)",
                   f"{net['caselogs_dropped']:,} ({net['caselogs_dropped_pct']}%)")
        c2.metric("Calls dropped (no case match)",
                   f"{net['calls_dropped']:,} ({net['calls_dropped_pct']}%)")
        st.dataframe(data_health_table(data_health, name_by_id), width="stretch", hide_index=True)

    with tab1:
        res = consult.get(fac_id)
        if res is None:
            st.info("Not enough history to forecast consults for this hospital.")
        else:
            st.plotly_chart(
                history_forecast_chart(res["history"], res["forecast"], "Consults / month",
                                        f"{name_by_id[fac_id]}: monthly consults (model: {res['tier']})"),
                width="stretch",
            )
            backtest_badge(res["backtest"])

    with tab2:
        res = calls.get(fac_id)
        if res is None:
            st.info("Not enough history to forecast calls for this hospital.")
        else:
            st.plotly_chart(
                history_forecast_chart(res["history"], res["forecast"], "Calls / day",
                                        f"{name_by_id[fac_id]}: daily calls (model: {res['tier']})"),
                width="stretch",
            )
            backtest_badge(res["backtest"])

    with tab3:
        res = busiest.get(fac_id)
        if res is None or res.get("status") != "ok":
            st.info("Not enough call data to analyze busiest hour for this hospital.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Busiest local hour", f"{res['overall_busiest_hour']:02d}:00")
            c2.metric("Bootstrap confidence", f"{res['confidence']:.0%}")
            c3.metric("Stable over time?", "No, drifts" if res["drift_flag"] else "Yes")

            st.markdown(
                f"- **Weekday busiest hour:** {res['weekday_busiest_hour']:02d}:00 &nbsp;|&nbsp; "
                f"**Weekend busiest hour:** {res['weekend_busiest_hour']:02d}:00\n"
                f"- **First half of history:** {res['first_half_busiest_hour']:02d}:00 &nbsp;|&nbsp; "
                f"**Second half:** {res['second_half_busiest_hour']:02d}:00\n"
                f"- Based on **{res['n_calls']:,}** historical calls."
            )

            heatmap = res["heatmap"]
            fig = go.Figure(data=go.Heatmap(
                z=heatmap, x=list(range(24)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                colorscale="YlOrRd", colorbar=dict(title="Calls"),
            ))
            fig.update_layout(
                title=f"{name_by_id[fac_id]}: call volume by day x local hour",
                xaxis_title="Local hour of day", height=360,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Confidence is the fraction of 500 bootstrap resamples of historical call hours whose mode "
                "matches the observed busiest hour. Low values mean the busiest hour is a close contest "
                "between adjacent hours, not a sharp fact. 'Stable over time' flags whether weekday/weekend, "
                "first-half/second-half, or month-to-month busiest hours disagree with the overall pick."
            )


if __name__ == "__main__":
    main()
