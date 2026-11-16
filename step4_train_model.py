import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Write output both to stdout and to a file so it isn't lost in shell redirection
log = open("step4_result.txt", "w")
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

features = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target   = "electricity_kwh"

X_train = train[features]
y_train = train[target]

# --- Train ---
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

p("=== Model trained ===")
p(f"Type           : {type(model).__name__}")
p(f"Estimators     : {model.n_estimators}")
p(f"Learning rate  : {model.learning_rate}")
p(f"Max tree depth : {model.max_depth}")
p()

# Feature importances
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
p("=== Feature importances (higher = more useful) ===")
for feat, imp in importances.items():
    bar = "#" * int(imp * 50)
    p(f"  {feat:<12}  {imp:.4f}  {bar}")

p()
p("Training complete — model is ready for evaluation in Step 5.")
log.close()
