import re, requests

with open(".env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DATABASE_URL"):
            db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

db_url = re.sub(r"[&?]channel_binding=require", "", db_url)
m = re.match(r"postgresql://[^@]+@([^/?]+)", db_url)
sql_url = f"https://{m.group(1)}/sql"
headers = {"Content-Type": "application/json", "Neon-Connection-String": db_url}

def sql(q):
    r = requests.post(sql_url, json={"query": q}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

# Total row count
total = sql("SELECT COUNT(*) AS n FROM meter_readings_source")
print("Total rows in table:", list(total["rows"][0].values())[0])

# Per-meter summary
result = sql("""
SELECT
    meter_id,
    COUNT(*)                              AS row_count,
    MIN(timestamp AT TIME ZONE 'UTC')     AS earliest,
    MAX(timestamp AT TIME ZONE 'UTC')     AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

print()
print(f"  {'meter_id':<12}  {'row_count':>10}  {'earliest':<26}  {'latest':<26}")
print(f"  {'-'*12}  {'-'*10}  {'-'*26}  {'-'*26}")
for r in result["rows"]:
    v = list(r.values())
    print(f"  {v[0]:<12}  {v[1]:>10}  {str(v[2]):<26}  {str(v[3]):<26}")
