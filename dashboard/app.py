"""
FM Energy Dashboard
-------------------
Reads historical meter readings from Neon and displays:
  - Three side-by-side hourly electricity charts (one per meter)
  - A combined total energy chart (all meters summed by timestamp)
  - Placeholder sections for Prediction and LLM Analysis
"""

import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Load DATABASE_URL from .env ───────────────────────────────────────────────
def _load_env() -> str:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DATABASE_URL"):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                # Strip the channel_binding param that blocks the HTTP API
                url = re.sub(r"[&?]channel_binding=require", "", url)
                return url
    raise RuntimeError("DATABASE_URL not found in .env")

DATABASE_URL = _load_env()
_host_match  = re.match(r"postgresql://[^@]+@([^/?]+)", DATABASE_URL)
SQL_URL      = f"https://{_host_match.group(1)}/sql"
_HEADERS     = {
    "Content-Type":           "application/json",
    "Neon-Connection-String": DATABASE_URL,
}

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Loading meter readings…")
def load_readings() -> pd.DataFrame:
    resp = requests.post(
        SQL_URL,
        json={
            "query": """
                SELECT meter_id, timestamp, building_id, electricity_kwh
                FROM   meter_readings_source
                ORDER  BY timestamp, meter_id
            """
        },
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = payload["rows"]
    df = pd.DataFrame(rows, columns=["meter_id", "timestamp", "building_id", "electricity_kwh"])
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ FM Energy")
    st.caption("Historical sensor data from Neon PostgreSQL")
    st.divider()
    if st.button("🔄  Refresh data", use_container_width=True):
        st.cache_data.clear()
    st.divider()
    show_raw = st.toggle("Show raw data table", value=False)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_readings()

meter_ids  = sorted(df["meter_id"].unique())
date_min   = df["timestamp"].min().strftime("%Y-%m-%d")
date_max   = df["timestamp"].max().strftime("%Y-%m-%d")
total_rows = len(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard")
st.caption(
    f"Historical data · {total_rows:,} readings · "
    f"{date_min} → {date_max} · "
    f"{len(meter_ids)} meters"
)

# ── Row 1 — three meter charts side by side ───────────────────────────────────
st.subheader("Hourly Electricity Usage — by Meter")

cols = st.columns(3)

METER_COLOURS = {
    "METER_01": "#3B82F6",   # blue
    "METER_02": "#10B981",   # green
    "METER_03": "#F59E0B",   # amber
}

for col, meter in zip(cols, meter_ids):
    meter_df  = df[df["meter_id"] == meter]
    building  = meter_df["building_id"].iloc[0]
    colour    = METER_COLOURS.get(meter, "#6366F1")

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

# ── Raw data table (optional) ─────────────────────────────────────────────────
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
