"""
FM Energy Dashboard  v3  — Simulated live feed + Prediction
------------------------------------------------------------
Based on app_v2.py.  Adds:
  • "Predict next hour" button in the Prediction section.
  • Uses models/gbr_model.joblib + models/feature_order.json (no retraining).
  • Builds features from each meter's own revealed readings and predicts
    the next recorded hour individually.
  • Shows kWh prediction and target timestamp per meter, or a clear
    message when history is insufficient (needs ≥ 168 readings).
"""

import json
import os
import re
import time

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v3",
    page_icon="⚡",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_HERE, "..", "models", "gbr_model.joblib")
_FEAT_PATH  = os.path.join(_HERE, "..", "models", "feature_order.json")
_ENV_PATH   = os.path.join(_HERE, "..", ".env")

# ── Load model + feature order once (module-level cache) ─────────────────────
@st.cache_resource
def _load_model():
    model    = joblib.load(_MODEL_PATH)
    with open(_FEAT_PATH) as f:
        features = json.load(f)
    return model, features

GBR_MODEL, FEATURE_ORDER = _load_model()
# Deepest lag determines minimum history required
_MIN_HISTORY = max(
    int(f.replace("lag_", "").replace("h", ""))
    for f in FEATURE_ORDER if f.startswith("lag_")
)   # = 168

# ── Neon connection ───────────────────────────────────────────────────────────
def _load_db_url() -> str:
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DATABASE_URL"):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                return re.sub(r"[&?]channel_binding=require", "", url)
    raise RuntimeError("DATABASE_URL not found in .env")

_DB_URL  = _load_db_url()
_HOST    = re.match(r"postgresql://[^@]+@([^/?]+)", _DB_URL).group(1)
_SQL_URL = f"https://{_HOST}/sql"
_HEADERS = {
    "Content-Type":           "application/json",
    "Neon-Connection-String": _DB_URL,
}

def _query(sql: str, params: list | None = None) -> list[dict]:
    payload = {"query": sql}
    if params:
        payload["params"] = params
    r = requests.post(_SQL_URL, json=payload, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["rows"]

# ── Data helpers ──────────────────────────────────────────────────────────────
SIM_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
SIM_SEED  = pd.Timestamp("2016-02-01 00:00:00", tz="UTC")

METER_COLOURS = {
    "METER_01": "#3B82F6",
    "METER_02": "#10B981",
    "METER_03": "#F59E0B",
}


@st.cache_data(show_spinner="Loading January 2016 baseline…")
def _load_january() -> pd.DataFrame:
    rows = _query("""
        SELECT meter_id, timestamp, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= '2016-01-01 00:00:00+00'
          AND  timestamp <  '2016-02-01 00:00:00+00'
        ORDER  BY timestamp, meter_id
    """)
    return _to_df(rows)


def _fetch_hour(ts: pd.Timestamp) -> pd.DataFrame:
    rows = _query(
        """
        SELECT meter_id, timestamp, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp = $1
        ORDER  BY meter_id
        """,
        [ts.isoformat()],
    )
    return _to_df(rows)


def _to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["meter_id", "timestamp", "building_id", "electricity_kwh"])
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


def _last_db_timestamp() -> pd.Timestamp:
    rows = _query("SELECT MAX(timestamp) AS ts FROM meter_readings_source")
    return pd.to_datetime(rows[0]["ts"], utc=True)


# ── Prediction helper ─────────────────────────────────────────────────────────
def _predict_meter(meter_df: pd.DataFrame) -> dict:
    """
    Given a single meter's revealed readings (sorted by timestamp),
    build the feature row for the next hour and return a result dict:
      { ok: bool, kwh: float, target_ts: Timestamp, message: str }
    """
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    n        = len(meter_df)

    if n < _MIN_HISTORY:
        return {
            "ok":      False,
            "message": (
                f"Not enough history — need {_MIN_HISTORY} readings, "
                f"only {n} available so far."
            ),
        }

    # Target timestamp = last known + 1 hour
    last_ts    = meter_df["timestamp"].iloc[-1]
    target_ts  = last_ts + pd.Timedelta(hours=1)
    kwh_series = meter_df["electricity_kwh"]

    row = {
        "hour_of_day": target_ts.hour,
        "day_of_week": target_ts.weekday(),
        "lag_1h":      kwh_series.iloc[-1],
        "lag_2h":      kwh_series.iloc[-2],
        "lag_3h":      kwh_series.iloc[-3],
        "lag_24h":     kwh_series.iloc[-24],
        "lag_168h":    kwh_series.iloc[-168],
    }

    X      = pd.DataFrame([row])[FEATURE_ORDER]
    pred   = float(GBR_MODEL.predict(X)[0])

    return {
        "ok":        True,
        "kwh":       round(pred, 2),
        "target_ts": target_ts,
        "message":   "",
    }


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state():
    st.session_state.df         = _load_january()
    st.session_state.next_ts    = SIM_SEED
    st.session_state.last_tick  = time.time()
    st.session_state.sim_done   = False
    st.session_state.last_db_ts = _last_db_timestamp()
    st.session_state.predictions = {}   # meter_id → result dict


if "df" not in st.session_state:
    _init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ FM Energy v3")
    st.caption("Simulated live feed — 1 real minute = 1 data hour")
    st.divider()

    if st.button("🔁  Restart simulation", use_container_width=True, type="primary"):
        _load_january.clear()
        _init_state()
        st.rerun()

    st.divider()

    if st.session_state.sim_done:
        st.success("✅ Simulation complete — all records shown.")
    else:
        elapsed   = time.time() - st.session_state.last_tick
        remaining = max(0, 60 - elapsed)
        st.metric(label="⏱ Next data fetch in", value=f"{int(remaining):02d}s")

    st.divider()
    show_raw = st.toggle("Show raw data table", value=False)

# ── Simulation tick ───────────────────────────────────────────────────────────
if not st.session_state.sim_done:
    elapsed = time.time() - st.session_state.last_tick
    if elapsed >= 60:
        new_rows = _fetch_hour(st.session_state.next_ts)
        if not new_rows.empty:
            st.session_state.df = pd.concat(
                [st.session_state.df, new_rows], ignore_index=True
            )
        st.session_state.last_tick  = time.time()
        st.session_state.next_ts   += pd.Timedelta(hours=1)
        # Clear stale predictions so user re-runs against fresh data
        st.session_state.predictions = {}
        if st.session_state.next_ts > st.session_state.last_db_ts:
            st.session_state.sim_done = True

# ── Working data ──────────────────────────────────────────────────────────────
df        = st.session_state.df
meter_ids = sorted(df["meter_id"].unique())
date_min  = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
date_max  = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
n_rows    = len(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard")
st.caption(
    f"Historical data · {n_rows:,} readings · "
    f"{date_min} → {date_max} · "
    f"{len(meter_ids)} meters"
)

# ── Row 1 — three meter charts ────────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — by Meter")
cols = st.columns(3)

for col, meter in zip(cols, meter_ids):
    meter_df = df[df["meter_id"] == meter]
    building = meter_df["building_id"].iloc[0]
    colour   = METER_COLOURS.get(meter, "#6366F1")

    fig = px.line(
        meter_df,
        x="timestamp",
        y="electricity_kwh",
        title=f"{meter}<br><sup>{building}</sup>",
        labels={"timestamp": "", "electricity_kwh": "kWh"},
        color_discrete_sequence=[colour],
    )
    fig.update_traces(line_width=1.5)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=56, b=10),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#F3F4F6"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
    )
    col.plotly_chart(fig, use_container_width=True)

# ── Row 2 — total energy chart ────────────────────────────────────────────────
st.subheader("Total Electricity Usage — All Meters Combined")

total_df = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .rename(columns={"electricity_kwh": "total_kwh"})
)

fig_total = px.area(
    total_df,
    x="timestamp",
    y="total_kwh",
    labels={"timestamp": "", "total_kwh": "Total kWh"},
    color_discrete_sequence=["#6366F1"],
)
fig_total.update_traces(line_width=1.5, fillcolor="rgba(99,102,241,0.12)")
fig_total.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=16, b=10),
    hovermode="x unified",
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#F3F4F6"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12),
)
st.plotly_chart(fig_total, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
if show_raw:
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True, height=300)

st.divider()

# ── Prediction section ────────────────────────────────────────────────────────
with st.container():
    st.subheader("🔮 Prediction")
    st.caption(
        "Uses the saved GBR model with lag features built from each meter's "
        "revealed readings. Requires at least 168 readings (7 days) per meter."
    )

    if st.button("🔮  Predict next hour", type="primary"):
        results = {}
        for meter in meter_ids:
            meter_df = df[df["meter_id"] == meter].copy()
            results[meter] = _predict_meter(meter_df)
        st.session_state.predictions = results

    # Display results if available
    if st.session_state.predictions:
        pred_cols = st.columns(len(meter_ids))
        for col, meter in zip(pred_cols, meter_ids):
            result  = st.session_state.predictions.get(meter, {})
            colour  = METER_COLOURS.get(meter, "#6366F1")
            with col:
                st.markdown(
                    f"<div style='border-left: 4px solid {colour}; padding-left: 12px;'>"
                    f"<strong>{meter}</strong></div>",
                    unsafe_allow_html=True,
                )
                if not result:
                    st.caption("No prediction run yet.")
                elif result["ok"]:
                    target = result["target_ts"].strftime("%Y-%m-%d %H:%M UTC")
                    st.metric(
                        label=f"📅 {target}",
                        value=f"{result['kwh']:,.2f} kWh",
                    )
                else:
                    st.warning(result["message"], icon="⚠️")
    else:
        st.caption("Press **Predict next hour** to generate forecasts for all three meters.")

st.divider()

# ── LLM Analysis placeholder ──────────────────────────────────────────────────
with st.container():
    st.subheader("🤖 LLM Analysis")
    st.info(
        "**Coming soon.** This section will send a summary of the meter readings "
        "to an LLM and stream back a plain-English energy usage analysis.",
        icon="🤖",
    )

# ── Auto-rerun (1-second tick for countdown) ──────────────────────────────────
if not st.session_state.sim_done:
    time.sleep(1)
    st.rerun()
