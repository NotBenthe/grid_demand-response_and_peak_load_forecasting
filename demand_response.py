import pandas as pd
import numpy as np


def simulate_demand_response(
    predictions_df: pd.DataFrame,
    peak_percentile: float = 0.90,
    dr_shaving_factor: float = 0.15,
):
    """Simulates Demand Response with peak load shaving and off-peak energy redistribution."""
    df = predictions_df.copy()

    # Determine peak load threshold from predictions
    peak_threshold = df["predicted_load_kwh"].quantile(peak_percentile)

    # Calculate initial Peak-to-Average Ratio (PAR)
    initial_peak = df["actual_load_kwh"].max()
    initial_mean = df["actual_load_kwh"].mean()
    initial_par = initial_peak / initial_mean if initial_mean > 0 else 1.0

    # Identify peak slots
    peak_mask = df["predicted_load_kwh"] >= peak_threshold

    # Shave load during peak slots
    df["shaved_load_kwh"] = 0.0
    df.loc[peak_mask, "shaved_load_kwh"] = (
        df.loc[peak_mask, "actual_load_kwh"] * dr_shaving_factor
    )

    # Total energy shaved from peaks
    total_shaved_energy = df["shaved_load_kwh"].sum()

    # Identify off-peak slots for load shifting (01:00 AM to 05:00 AM)
    off_peak_mask = df["timestamp"].dt.hour.isin([1, 2, 3, 4])
    off_peak_count = off_peak_mask.sum()

    # Redistribute shaved energy evenly across off-peak slots
    df["redistributed_load_kwh"] = 0.0
    if off_peak_count > 0 and total_shaved_energy > 0:
        df.loc[off_peak_mask, "redistributed_load_kwh"] = total_shaved_energy / off_peak_count

    # Final optimized grid load profile
    df["optimized_load_kwh"] = (
        df["actual_load_kwh"] - df["shaved_load_kwh"] + df["redistributed_load_kwh"]
    )

    # Calculate optimized Peak-to-Average Ratio
    optimized_peak = df["optimized_load_kwh"].max()
    optimized_mean = df["optimized_load_kwh"].mean()
    optimized_par = optimized_peak / optimized_mean if optimized_mean > 0 else 1.0

    max_peak_shaved = df["shaved_load_kwh"].max()
    peak_slots_count = int(peak_mask.sum())

    summary = {
        "peak_threshold_kwh": peak_threshold,
        "peak_slots_identified": peak_slots_count,
        "initial_par": initial_par,
        "optimized_par": optimized_par,
        "peak_load_reduction_kwh": max_peak_shaved,
        "total_shaved_energy_kwh": total_shaved_energy,
    }

    return df, summary