"""
FM Energy Dashboard  v4  — Simulated live feed + Prediction + LLM Analysis
---------------------------------------------------------------------------
Based on app_v3.py.  Replaces the LLM Analysis placeholder with:

  • A read-only prompt: "Give me a summary of the energy usage for
    METER_02 for the past 2 days".
  • A Send button that (only on press) calls the OpenAI API.
  • The last 48 revealed hours for METER_02 are packaged into a JSON
    object (meter name, time range, total / avg / min / max kWh, and
    the 48 hourly readings) and attached as data context in the user
    message.
  • The API response is stored in session state — the 1-second rerun
    loop and the 1-minute simulation tick never trigger a new request.
  • The response is displayed with:
      - "Based on readings up to <latest timestamp>"
      - Number of LLM requests made this session
      - A collapsed expander showing the exact JSON sent to the LLM
  • OPENAI_API_KEY and OPENAI_MODEL are read from .env; the key is
    never displayed.
"""

import json
import os
import re
import time

import joblib
import pandas as pd
import plotly.express as px
import requests as http_requests
import streamlit as st
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v4",
    page_icon="⚡",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_HERE, "..", "models", "gbr_model.joblib")
_FEAT_PATH  = os.path.join(_HERE, "..", "models", "feature_order.json")
_ENV_PATH   = os.path.join(_HERE, "..", ".env")

# ── Load config: .env locally, st.secrets on Streamlit Cloud ─────────────────
def _read_env() -> dict:
    # Prefer .env when it exists (local development)
    if os.path.exists(_ENV_PATH):
        vals = {}
        with open(_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
        return vals
    # Fall back to Streamlit secrets (Streamlit Cloud deployment)
    return {k: str(v) for k, v in st.secrets.items()}

_ENV = _read_env()

_DB_URL_RAW   = _ENV.get("DATABASE_URL", "")
_DB_URL       = re.sub(r"[&?]channel_binding=require", "", _DB_URL_RAW)
_OPENAI_KEY   = _ENV.get("OPENAI_API_KEY", "")
_OPENAI_MODEL = _ENV.get("OPENAI_MODEL", "gpt-4o")

# Fail fast with a readable message if secrets are not configured
if not _DB_URL_RAW:
    st.error("DATABASE_URL is not set. Add it to .env (local) or Streamlit secrets (cloud).")
    st.stop()
if not _OPENAI_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to .env (local) or Streamlit secrets (cloud).")
    st.stop()

# ── Neon connection ───────────────────────────────────────────────────────────
_HOST    = re.match(r"postgresql://[^@]+@([^/?]+)", _DB_URL).group(1)
_SQL_URL = f"https://{_HOST}/sql"
_NEON_HEADERS = {
    "Content-Type":           "application/json",
    "Neon-Connection-String": _DB_URL,
}

def _query(sql: str, params: list | None = None) -> list[dict]:
    payload = {"query": sql}
    if params:
        payload["params"] = params
    r = http_requests.post(_SQL_URL, json=payload, headers=_NEON_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["rows"]

# ── OpenAI client (loaded once) ───────────────────────────────────────────────
@st.cache_resource
def _openai_client() -> OpenAI:
    return OpenAI(api_key=_OPENAI_KEY)

# ── GBR model (loaded once) ───────────────────────────────────────────────────
@st.cache_resource
def _load_model():
    model = joblib.load(_MODEL_PATH)
    with open(_FEAT_PATH) as f:
        features = json.load(f)
    return model, features

GBR_MODEL, FEATURE_ORDER = _load_model()
_MIN_HISTORY = max(
    int(f.replace("lag_", "").replace("h", ""))
    for f in FEATURE_ORDER if f.startswith("lag_")
)  # 168

# ── Simulation constants ──────────────────────────────────────────────────────
SIM_SEED = pd.Timestamp("2016-02-01 00:00:00", tz="UTC")

METER_COLOURS = {
    "METER_01": "#3B82F6",
    "METER_02": "#10B981",
    "METER_03": "#F59E0B",
}

LLM_METER   = "METER_02"
LLM_HOURS   = 48          # "past 2 days"
FIXED_PROMPT = f"Give me a summary of the energy usage for {LLM_METER} for the past 2 days"

# ── Data helpers ──────────────────────────────────────────────────────────────
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
        "SELECT meter_id, timestamp, building_id, electricity_kwh "
        "FROM meter_readings_source WHERE timestamp = $1 ORDER BY meter_id",
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
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    n = len(meter_df)
    if n < _MIN_HISTORY:
        return {"ok": False, "message": (
            f"Not enough history — need {_MIN_HISTORY} readings, only {n} available."
        )}
    last_ts   = meter_df["timestamp"].iloc[-1]
    target_ts = last_ts + pd.Timedelta(hours=1)
    kwh       = meter_df["electricity_kwh"]
    row = {
        "hour_of_day": target_ts.hour,
        "day_of_week": target_ts.weekday(),
        "lag_1h":      kwh.iloc[-1],
        "lag_2h":      kwh.iloc[-2],
        "lag_3h":      kwh.iloc[-3],
        "lag_24h":     kwh.iloc[-24],
        "lag_168h":    kwh.iloc[-168],
    }
    pred = float(GBR_MODEL.predict(pd.DataFrame([row])[FEATURE_ORDER])[0])
    return {"ok": True, "kwh": round(pred, 2), "target_ts": target_ts, "message": ""}

# ── LLM data builder ──────────────────────────────────────────────────────────
def _build_llm_payload(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (json_string, latest_ts_label) for the last LLM_HOURS rows of LLM_METER.
    """
    m2 = (
        df[df["meter_id"] == LLM_METER]
        .sort_values("timestamp")
        .tail(LLM_HOURS)
        .reset_index(drop=True)
    )
    readings = [
        {"timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
         "electricity_kwh": round(r["electricity_kwh"], 2)}
        for _, r in m2.iterrows()
    ]
    kwh_vals   = m2["electricity_kwh"]
    latest_ts  = m2["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    earliest_ts = m2["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")

    payload = {
        "meter":        LLM_METER,
        "time_range":   {"from": earliest_ts, "to": latest_ts},
        "statistics": {
            "total_kwh":   round(float(kwh_vals.sum()), 2),
            "average_kwh": round(float(kwh_vals.mean()), 2),
            "min_kwh":     round(float(kwh_vals.min()), 2),
            "max_kwh":     round(float(kwh_vals.max()), 2),
        },
        "hourly_readings": readings,
    }
    return json.dumps(payload, indent=2), latest_ts

# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state():
    st.session_state.df           = _load_january()
    st.session_state.next_ts      = SIM_SEED
    st.session_state.last_tick    = time.time()
    st.session_state.sim_done     = False
    st.session_state.last_db_ts   = _last_db_timestamp()
    st.session_state.predictions  = {}
    # LLM state — persists across reruns, cleared only on Restart
    if "llm_response" not in st.session_state:
        st.session_state.llm_response    = None   # str | None
        st.session_state.llm_latest_ts   = None   # str | None
        st.session_state.llm_json_sent   = None   # str | None
        st.session_state.llm_call_count  = 0      # int

if "df" not in st.session_state:
    _init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ FM Energy v4")
    st.caption("Simulated live feed — 1 real minute = 1 data hour")
    st.divider()

    if st.button("🔁  Restart simulation", use_container_width=True, type="primary"):
        _load_january.clear()
        # Reset LLM state on restart too
        st.session_state.llm_response   = None
        st.session_state.llm_latest_ts  = None
        st.session_state.llm_json_sent  = None
        st.session_state.llm_call_count = 0
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
        st.session_state.last_tick = time.time()
        st.session_state.next_ts  += pd.Timedelta(hours=1)
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
        meter_df, x="timestamp", y="electricity_kwh",
        title=f"{meter}<br><sup>{building}</sup>",
        labels={"timestamp": "", "electricity_kwh": "kWh"},
        color_discrete_sequence=[colour],
    )
    fig.update_traces(line_width=1.5)
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=56, b=10),
        hovermode="x unified",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6"),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12),
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
    total_df, x="timestamp", y="total_kwh",
    labels={"timestamp": "", "total_kwh": "Total kWh"},
    color_discrete_sequence=["#6366F1"],
)
fig_total.update_traces(line_width=1.5, fillcolor="rgba(99,102,241,0.12)")
fig_total.update_layout(
    height=320, margin=dict(l=10, r=10, t=16, b=10),
    hovermode="x unified",
    xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6"),
    plot_bgcolor="white", paper_bgcolor="white", font=dict(size=12),
)
st.plotly_chart(fig_total, use_container_width=True)

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
            results[meter] = _predict_meter(df[df["meter_id"] == meter].copy())
        st.session_state.predictions = results

    if st.session_state.predictions:
        pred_cols = st.columns(len(meter_ids))
        for col, meter in zip(pred_cols, meter_ids):
            result = st.session_state.predictions.get(meter, {})
            colour = METER_COLOURS.get(meter, "#6366F1")
            with col:
                st.markdown(
                    f"<div style='border-left:4px solid {colour};padding-left:12px'>"
                    f"<strong>{meter}</strong></div>",
                    unsafe_allow_html=True,
                )
                if not result:
                    st.caption("No prediction run yet.")
                elif result["ok"]:
                    target = result["target_ts"].strftime("%Y-%m-%d %H:%M UTC")
                    st.metric(label=f"📅 {target}", value=f"{result['kwh']:,.2f} kWh")
                else:
                    st.warning(result["message"], icon="⚠️")
    else:
        st.caption("Press **Predict next hour** to generate forecasts for all three meters.")

st.divider()

# ── LLM Analysis section ──────────────────────────────────────────────────────
with st.container():
    st.subheader("🤖 LLM Analysis")

    # Read-only prompt display
    st.text_input(
        label="Prompt",
        value=FIXED_PROMPT,
        disabled=True,
        label_visibility="collapsed",
    )

    send_clicked = st.button("📤  Send", type="primary")

    if send_clicked:
        # Build the data payload from currently revealed readings
        json_str, latest_ts = _build_llm_payload(df)

        # Compose the user message: fixed prompt + JSON data context
        user_message = (
            f"{FIXED_PROMPT}\n\n"
            f"Here is the data for your analysis:\n\n"
            f"```json\n{json_str}\n```"
        )

        with st.spinner("Calling OpenAI…"):
            client = _openai_client()
            response = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an energy analyst. "
                            "Analyse the provided electricity meter data and give a clear, "
                            "concise plain-English summary. Include observations about "
                            "usage patterns, peaks, and any notable trends."
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        # Store results — rerun loop will never re-trigger this block
        st.session_state.llm_response   = response.choices[0].message.content
        st.session_state.llm_latest_ts  = latest_ts
        st.session_state.llm_json_sent  = json_str
        st.session_state.llm_call_count += 1

    # Display stored response (survives all reruns)
    if st.session_state.llm_response:
        st.caption(
            f"Based on readings up to **{st.session_state.llm_latest_ts}** · "
            f"LLM requests this session: **{st.session_state.llm_call_count}**"
        )
        st.markdown(st.session_state.llm_response)

        with st.expander("📄 Data sent to the LLM", expanded=False):
            st.code(st.session_state.llm_json_sent, language="json")
    else:
        st.caption("Press **Send** to analyse the latest 48 hours of METER_02 with the LLM.")

# ── Auto-rerun (1-second tick for countdown) ──────────────────────────────────
if not st.session_state.sim_done:
    time.sleep(1)
    st.rerun()
