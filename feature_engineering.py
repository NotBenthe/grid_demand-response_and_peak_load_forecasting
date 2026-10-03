import pandas as pd
import numpy as np


def build_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers temporal, interaction, cyclical, lag, and rolling features in float32."""
    data = df.copy().sort_values("timestamp").reset_index(drop=True)

    # 1. Temporal Features
    data["hour"] = data["timestamp"].dt.hour.astype(np.float32)
    data["minute"] = data["timestamp"].dt.minute.astype(np.float32)
    data["dayofweek"] = data["timestamp"].dt.dayofweek.astype(np.float32)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.float32)
    data["month"] = data["timestamp"].dt.month.astype(np.float32)

    if "is_bank_holiday" not in data.columns:
        data["is_bank_holiday"] = np.float32(0)

    # 2. Cyclical Encodings
    half_hour_slot = data["hour"] * 2 + (data["minute"] // 30)
    data["sin_slot"] = np.sin(2 * np.pi * half_hour_slot / 48.0).astype(np.float32)
    data["cos_slot"] = np.cos(2 * np.pi * half_hour_slot / 48.0).astype(np.float32)

    weekly_slot = (data["dayofweek"] * 48) + half_hour_slot
    data["sin_weekly"] = np.sin(2 * np.pi * weekly_slot / 336.0).astype(np.float32)
    data["cos_weekly"] = np.cos(2 * np.pi * weekly_slot / 336.0).astype(np.float32)

    # 3. Domain Interaction Terms
    data["temp_x_hour"] = (data["temperature"] * data["hour"]).astype(np.float32)

    # 4. Historical Lags
    data["lag_1"] = data["total_load_kwh"].shift(1).astype(np.float32)
    data["lag_2"] = data["total_load_kwh"].shift(2).astype(np.float32)
    data["lag_48"] = data["total_load_kwh"].shift(48).astype(np.float32)
    data["lag_336"] = data["total_load_kwh"].shift(336).astype(np.float32)

    # 5. Short & Long Rolling Statistics
    data["rolling_mean_6h"] = data["total_load_kwh"].shift(1).rolling(window=12).mean().astype(np.float32)
    data["rolling_std_6h"] = data["total_load_kwh"].shift(1).rolling(window=12).std().astype(np.float32)
    data["rolling_mean_24h"] = data["total_load_kwh"].shift(1).rolling(window=48).mean().astype(np.float32)
    data["rolling_std_24h"] = data["total_load_kwh"].shift(1).rolling(window=48).std().astype(np.float32)
    data["rolling_max_24h"] = data["total_load_kwh"].shift(1).rolling(window=48).max().astype(np.float32)
    data["ema_12h"] = data["total_load_kwh"].shift(1).ewm(span=24).mean().astype(np.float32)

    return data.dropna().reset_index(drop=True)