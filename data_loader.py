import glob
import gc
import numpy as np 
import os
import pandas as pd


def load_smart_meter_dataset(data_dir: str, num_blocks: int = 15) -> pd.DataFrame:
    """
    Loads smart meter CSV blocks with float32 downcasting to prevent RAM overload.
    """
    block_pattern = os.path.join(
        data_dir, "halfhourly_dataset", "halfhourly_dataset", "block_*.csv"
    )
    block_files = sorted(glob.glob(block_pattern))[:num_blocks]

    if not block_files:
        block_files = sorted(
            glob.glob(os.path.join(data_dir, "halfhourly_dataset", "block_*.csv"))
        )[:num_blocks]

    dfs = []
    for file in block_files:
        df = pd.read_csv(file, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        energy_col = [c for c in df.columns if "kwh" in c or "energy" in c][0]
        time_col = [c for c in df.columns if "tstp" in c or "time" in c][0]

        df["energy"] = pd.to_numeric(df[energy_col], errors="coerce").astype(np.float32)
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce").astype("datetime64[s]")
        dfs.append(df[["timestamp", "energy"]].dropna(subset=["timestamp"]))

    full_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    grid_df = (
        full_df.groupby("timestamp")["energy"]
        .sum()
        .astype(np.float32)
        .reset_index()
        .rename(columns={"energy": "total_load_kwh"})
        .sort_values("timestamp")
    )
    del full_df
    gc.collect()

    # Merge Weather Observations
    weather_path = os.path.join(data_dir, "weather_hourly_darksky.csv")
    if os.path.exists(weather_path):
        weather = pd.read_csv(weather_path)
        weather["time"] = pd.to_datetime(weather["time"]).astype("datetime64[s]")
        weather = weather.sort_values("time")
        for col in ["temperature", "humidity", "windSpeed"]:
            weather[col] = weather[col].astype(np.float32)

        grid_df = pd.merge_asof(
            grid_df,
            weather[["time", "temperature", "humidity", "windSpeed"]],
            left_on="timestamp",
            right_on="time",
            direction="nearest",
        )
    else:
        grid_df["temperature"] = np.float32(12.0)
        grid_df["humidity"] = np.float32(0.7)
        grid_df["windSpeed"] = np.float32(5.0)

    # Merge Bank Holiday Indicator
    holidays_path = os.path.join(data_dir, "uk_bank_holidays.csv")
    if os.path.exists(holidays_path):
        holidays_df = pd.read_csv(holidays_path)
        date_col = holidays_df.columns[0]
        holiday_dates = set(
            pd.to_datetime(holidays_df[date_col], format="mixed", errors="coerce").dt.date.dropna()
        )
        grid_df["is_bank_holiday"] = grid_df["timestamp"].dt.date.isin(holiday_dates).astype(np.int8)
    else:
        grid_df["is_bank_holiday"] = np.int8(0)

    return grid_df