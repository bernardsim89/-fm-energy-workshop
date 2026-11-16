import pandas as pd

# Reproduce the prepared dataset from Step 2
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

# --- Split: hold back last 14 days as test set ---
split_time = df["timestamp"].max() - pd.Timedelta(days=14)

train = df[df["timestamp"] <= split_time].reset_index(drop=True)
test  = df[df["timestamp"] >  split_time].reset_index(drop=True)

features = ["hour_of_day", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target   = "electricity_kwh"

X_train = train[features]
y_train = train[target]
X_test  = test[features]
y_test  = test[target]

print("=== Split point ===")
print(f"Everything up to and including: {split_time}")
print()

print("=== Training set ===")
print(f"Rows : {len(train)}")
print(f"From : {train['timestamp'].min()}")
print(f"To   : {train['timestamp'].max()}")
print()

print("=== Test set ===")
print(f"Rows : {len(test)}")
print(f"From : {test['timestamp'].min()}")
print(f"To   : {test['timestamp'].max()}")
print()

print("=== Feature matrix shapes ===")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
