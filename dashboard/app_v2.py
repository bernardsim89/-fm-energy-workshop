"""
FM Energy Dashboard  v2  — Simulated live feed
----------------------------------------------
• Loads January 2016 (all three meters) at startup.
• Every real minute, fetches the next recorded hour from Neon (3 rows —
  one per meter) and appends it to the charts.
• Restart button resets to the January baseline and restarts the timer.
• When the last record in the dataset is reached, the simulation stops.
• Placeholder sections for Prediction and LLM Analysis are preserved.
"""

import os
import re
import time
from datetime import timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

# ── Neon connection ───────────────────────────────────────────────────────────
def _load_db_url() -> str:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DATABASE_URL"):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                return re.sub(r"[&?]channel_binding=require", "", url)
    raise RuntimeError("DATABASE_URL not found in .env")

_DB_URL   = _load_db_url()
_HOST     = re.match(r"postgresql://[^@]+@([^/?]+)", _DB_URL).group(1)
_SQL_URL  = f"https://{_HOST}/sql"
_HEADERS  = {
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
SIM_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")  # first displayed hour
SIM_SEED  = pd.Timestamp("2016-02-01 00:00:00", tz="UTC")  # first "live" hour

METER_COLOURS = {
    "METER_01": "#3B82F6",
    "METER_02": "#10B981",
    "METER_03": "#F59E0B",
}


@st.cache_data(show_spinner="Loading January 2016 baseline…")
def _load_january() -> pd.DataFrame:
    """Full January 2016 — all three meters. Cached permanently for the session."""
    rows = _query("""
        SELECT meter_id, timestamp, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= '2016-01-01 00:00:00+00'
          AND  timestamp <  '2016-02-01 00:00:00+00'
        ORDER  BY timestamp, meter_id
    """)
    return _to_df(rows)


def _fetch_hour(ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch exactly one hourly slice (3 rows) from Neon."""
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


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state():
    """Set up (or reset) all simulation state."""
    st.session_state.df           = _load_january()
    st.session_state.next_ts      = SIM_SEED          # next hour to fetch from Neon
    st.session_state.last_tick    = time.time()        # wall-clock of last fetch
    st.session_state.sim_done     = False
    st.session_state.last_db_ts   = _last_db_timestamp()


if "df" not in st.session_state:
    _init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ FM Energy v2")
    st.caption("Simulated live feed — 1 real minute = 1 data hour")
    st.divider()

    if st.button("🔁  Restart simulation", use_container_width=True, type="primary"):
        _load_january.clear()          # bust the January cache so it re-fetches cleanly
        _init_state()
        st.rerun()

    st.divider()

    # Simulation status indicator
    if st.session_state.sim_done:
        st.success("✅ Simulation complete — all records shown.")
    else:
        elapsed   = time.time() - st.session_state.last_tick
        remaining = max(0, 60 - elapsed)
        st.metric(label="⏱ Next data fetch in", value=f"{int(remaining):02d}s")

    st.divider()
    show_raw = st.toggle("Show raw data table", value=False)

# ── Simulation tick ───────────────────────────────────────────────────────────
# If a full minute has elapsed and there are still records to fetch, grab the
# next hour now (before we render, so the charts include the new point).
if not st.session_state.sim_done:
    elapsed = time.time() - st.session_state.last_tick
    if elapsed >= 60:
        new_rows = _fetch_hour(st.session_state.next_ts)
        if not new_rows.empty:
            st.session_state.df = pd.concat(
                [st.session_state.df, new_rows], ignore_index=True
            )
        st.session_state.last_tick = time.time()
        # Advance pointer by one hour
        st.session_state.next_ts += pd.Timedelta(hours=1)
        # Stop when we have passed the last record in the DB
        if st.session_state.next_ts > st.session_state.last_db_ts:
            st.session_state.sim_done = True

# ── Working data for this render ──────────────────────────────────────────────
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

# ── Placeholder — Prediction ──────────────────────────────────────────────────
with st.container():
    st.subheader("🔮 Prediction")
    st.info(
        "**Coming soon.** This section will display a next-24-hour energy forecast "
        "generated by the trained GBR model in `models/gbr_model.joblib`.",
        icon="🔮",
    )

st.divider()

# ── Placeholder — LLM Analysis ────────────────────────────────────────────────
with st.container():
    st.subheader("🤖 LLM Analysis")
    st.info(
        "**Coming soon.** This section will send a summary of the meter readings "
        "to an LLM and stream back a plain-English energy usage analysis.",
        icon="🤖",
    )

# ── Auto-rerun while simulation is running ────────────────────────────────────
# Sleep until the next tick is due, then trigger a rerun so the countdown in
# the sidebar stays live and the next fetch fires on time.
if not st.session_state.sim_done:
    time.sleep(1)
    st.rerun()
