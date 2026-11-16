import re, requests, sys

with open(".env") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DATABASE_URL"):
            db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

db_url = re.sub(r"[&?]channel_binding=require", "", db_url)
print("URL prefix:", db_url[:70], flush=True)

m = re.match(r"postgresql://[^@]+@([^/?]+)", db_url)
host = m.group(1)
sql_url = f"https://{host}/sql"
headers = {"Content-Type": "application/json", "Neon-Connection-String": db_url}

try:
    r = requests.post(sql_url, json={"query": "SELECT 1 AS ok"}, headers=headers, timeout=15)
    print("HTTP status:", r.status_code, flush=True)
    print("Body:", r.text[:400], flush=True)
except Exception as e:
    print("ERROR:", e, flush=True)
    sys.exit(1)
