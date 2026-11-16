import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Write output to file as well as stdout
log = open("step5_result.txt", "w")
def p(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

# Reproduce prepared dataset
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

# Reproduce train/test split
split_time = df["timestamp"].max() - pd.Timedelta(days=14)
train = df[df["timestamp"] <= split_time].reset_index(drop=True)
test  = df[df["timestamp"] >  split_time].reset_index(drop=True)

features = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target   = "electricity_kwh"

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# --- Model predictions ---
y_pred = model.predict(X_test)

# --- Baseline: predict using the previous hour's reading (lag_1h) ---
y_baseline = test["lag_1h"].values

# --- Metrics ---
model_mae  = mean_absolute_error(y_test, y_pred)
model_rmse = root_mean_squared_error(y_test, y_pred)

base_mae   = mean_absolute_error(y_test, y_baseline)
base_rmse  = root_mean_squared_error(y_test, y_baseline)

p("=== Evaluation on 14-day test set (336 hours) ===")
p()
p(f"{'':30s}  {'MAE (kWh)':>10}  {'RMSE (kWh)':>10}")
p(f"{'Baseline (lag_1h)':30s}  {base_mae:>10.3f}  {base_rmse:>10.3f}")
p(f"{'GradientBoostingRegressor':30s}  {model_mae:>10.3f}  {model_rmse:>10.3f}")
p()

mae_improvement  = (base_mae  - model_mae)  / base_mae  * 100
rmse_improvement = (base_rmse - model_rmse) / base_rmse * 100
p(f"MAE improvement over baseline : {mae_improvement:+.1f}%")
p(f"RMSE improvement over baseline: {rmse_improvement:+.1f}%")
p()

# Worst predictions
errors = np.abs(y_pred - y_test.values)
worst_idx = np.argsort(errors)[-5:][::-1]
p("=== Five largest individual errors ===")
p(f"  {'Timestamp':<22}  {'Actual':>8}  {'Predicted':>9}  {'Error':>7}")
for i in worst_idx:
    ts  = test["timestamp"].iloc[i]
    act = y_test.iloc[i]
    prd = y_pred[i]
    err = errors[i]
    p(f"  {str(ts):<22}  {act:>8.2f}  {prd:>9.2f}  {err:>7.2f}")

log.close()
