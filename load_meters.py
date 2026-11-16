"""
Upload three_meters.csv to Neon table meter_readings_source via the HTTP SQL API.
Uses the requests library; avoids psycopg2 / port 5432.
"""

import csv
import os
import re
import requests

# ── config ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ENV_FILE  = os.path.join(BASE_DIR, ".env")
CSV_FILE  = os.path.join(BASE_DIR, "data", "three_meters.csv")
BATCH     = 500          # rows per INSERT

# ── load DATABASE_URL from .env ──────────────────────────────────────────────
db_url = None
with open(ENV_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith("DATABASE_URL"):
            db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not db_url:
    raise RuntimeError("DATABASE_URL not found in .env")

# Strip channel_binding param (firewall-hostile)
db_url = re.sub(r"[&?]channel_binding=require", "", db_url)

# Build the HTTP endpoint from the connection string
# postgresql://user:pass@host/db?... → https://host/sql
match = re.match(r"postgresql://[^@]+@([^/?]+)", db_url)
if not match:
    raise RuntimeError(f"Cannot parse host from DATABASE_URL: {db_url}")
host     = match.group(1)
sql_url  = f"https://{host}/sql"
headers  = {
    "Content-Type":           "application/json",
    "Neon-Connection-String": db_url,
}

print(f"Neon SQL endpoint : {sql_url}")

# ── helper ───────────────────────────────────────────────────────────────────
def run_sql(statement: str, params: list | None = None) -> dict:
    payload = {"query": statement}
    if params:
        payload["params"] = params
    resp = requests.post(sql_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── create table ─────────────────────────────────────────────────────────────
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    meter_id        TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    building_id     TEXT        NOT NULL,
    electricity_kwh NUMERIC(10, 2) NOT NULL,
    CONSTRAINT meter_readings_source_pk PRIMARY KEY (meter_id, timestamp)
)
""")
print("Table ready.")

# ── load CSV ─────────────────────────────────────────────────────────────────
with open(CSV_FILE, newline="") as f:
    reader  = csv.DictReader(f)
    rows    = list(reader)

total = len(rows)
print(f"CSV rows to upload: {total}")

# ── batch INSERT with ON CONFLICT DO NOTHING ──────────────────────────────────
inserted = 0
for start in range(0, total, BATCH):
    batch = rows[start : start + BATCH]
    # Build a VALUES list with positional placeholders $1, $2, …
    placeholders = []
    params       = []
    idx          = 1
    for row in batch:
        placeholders.append(f"(${idx}, ${idx+1}, ${idx+2}, ${idx+3})")
        params.extend([
            row["meter_id"],
            row["timestamp"],
            row["building_id"],
            row["electricity_kwh"],
        ])
        idx += 4

    sql = (
        "INSERT INTO meter_readings_source "
        "(meter_id, timestamp, building_id, electricity_kwh) VALUES "
        + ", ".join(placeholders)
        + " ON CONFLICT (meter_id, timestamp) DO NOTHING"
    )
    run_sql(sql, params)
    inserted += len(batch)
    print(f"  uploaded {inserted}/{total} rows …")

print(f"\nDone. {total} rows processed.")

# ── summary query ─────────────────────────────────────────────────────────────
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)                                   AS row_count,
    MIN(timestamp) AT TIME ZONE 'UTC'          AS earliest,
    MAX(timestamp) AT TIME ZONE 'UTC'          AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

print("\n── meter_readings_source summary ──────────────────────────────")
cols = [c["name"] for c in result["fields"]]
print(f"  {'meter_id':<12}  {'row_count':>10}  {'earliest':<25}  {'latest':<25}")
print(f"  {'-'*12}  {'-'*10}  {'-'*25}  {'-'*25}")
for r in result["rows"]:
    vals = list(r.values())
    print(f"  {vals[0]:<12}  {vals[1]:>10}  {str(vals[2]):<25}  {str(vals[3]):<25}")
