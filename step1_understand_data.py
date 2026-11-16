import pandas as pd

# Load the data
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

print("=== Shape ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns\n")

print("=== Columns & types ===")
print(df.dtypes, "\n")

print("=== First 3 rows ===")
print(df.head(3), "\n")

print("=== Date range ===")
print(f"From: {df['timestamp'].min()}")
print(f"To:   {df['timestamp'].max()}\n")

print("=== electricity_kwh summary ===")
print(df["electricity_kwh"].describe().round(2), "\n")

print("=== Missing values ===")
print(df.isnull().sum(), "\n")

# Check for duplicate timestamps
dupes = df["timestamp"].duplicated().sum()
print(f"=== Duplicate timestamps: {dupes} ===\n")

# Check for gaps in the hourly sequence
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
expected_hours = pd.date_range(
    start=df_sorted["timestamp"].min(),
    end=df_sorted["timestamp"].max(),
    freq="h"
)
missing_hours = expected_hours.difference(df_sorted["timestamp"])
print(f"=== Missing hours in sequence: {len(missing_hours)} ===")
if len(missing_hours) > 0:
    print(missing_hours)
