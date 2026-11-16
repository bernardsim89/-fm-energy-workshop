import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
import joblib

log = open("step6_result.txt", "w")
def p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

# ── Reproduce dataset ────────────────────────────────────────────────────────
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df = df[["timestamp", "electricity_kwh"]]

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

split_time = df["timestamp"].max() - pd.Timedelta(days=14)
train = df[df["timestamp"] <= split_time].reset_index(drop=True)
test  = df[df["timestamp"] >  split_time].reset_index(drop=True)

features = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target   = "electricity_kwh"

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# ── Train ────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(test["timestamp"], y_test.values,  label="Actual",    color="#1f77b4", linewidth=1.2)
ax.plot(test["timestamp"], y_pred,          label="Predicted", color="#ff7f0e", linewidth=1.0, linestyle="--")

ax.set_title("FM Energy Workshop — Actual vs Predicted (14-day test window)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
fig.autofmt_xdate()
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "step6_forecast_plot.png"
plt.savefig(plot_path, dpi=150)
plt.close()
p(f"Plot saved → {plot_path}")

# ── Save model and feature order ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

model_path   = "models/gbr_model.joblib"
feature_path = "models/feature_order.json"

joblib.dump(model, model_path)
p(f"Model saved → {model_path}")

with open(feature_path, "w") as f:
    json.dump(features, f, indent=2)
p(f"Feature order saved → {feature_path}")

# ── Verify files ─────────────────────────────────────────────────────────────
p()
p("=== Saved files ===")
for path in [model_path, feature_path, plot_path]:
    size = os.path.getsize(path)
    p(f"  {path:<35}  {size:>8,} bytes")

p()
p("Step 6 complete.")
log.close()
