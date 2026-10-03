import os
import gc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_all_results(
    gb_results_df: pd.DataFrame,
    pt_results_df: pd.DataFrame,
    dr_gb_df: pd.DataFrame,
    output_dir: str = "./plots",
):
    """Generates evaluation plots while enforcing low memory usage."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Forecast Comparison (Sliced to 7 days: 336 slots)
    plt.figure(figsize=(12, 4.5))
    sample_window = gb_results_df.iloc[: 48 * 7]
    pt_sample = pt_results_df.iloc[: 48 * 7]

    plt.plot(sample_window["timestamp"], sample_window["actual_load_kwh"], label="Actual", color="black", linewidth=1.5)
    plt.plot(sample_window["timestamp"], sample_window["predicted_load_kwh"], label="Gradient Boosting", color="#005A9C", linestyle="--")
    plt.plot(pt_sample["timestamp"], pt_sample["predicted_load_kwh"], label="PyTorch NN", color="#E30613", linestyle=":")

    plt.title("Smart Grid Forecast Comparison (7-Day Horizon)")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (kWh)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast_comparison_7day.png"), dpi=200)
    plt.close()

    # 2. Demand-Response Shifting Profile (Sliced to 3 days: 144 slots)
    plt.figure(figsize=(12, 4.5))
    dr_sample = dr_gb_df.iloc[: 48 * 3]

    plt.plot(dr_sample["timestamp"], dr_sample["actual_load_kwh"], label="Original Load", color="#E30613", linestyle="--")
    plt.plot(dr_sample["timestamp"], dr_sample["optimized_load_kwh"], label="Optimized Load", color="#00875A", linewidth=1.8)

    plt.fill_between(dr_sample["timestamp"], dr_sample["actual_load_kwh"], dr_sample["optimized_load_kwh"], where=(dr_sample["actual_load_kwh"] > dr_sample["optimized_load_kwh"]), color="#E30613", alpha=0.3, label="Shaved Peak")
    plt.fill_between(dr_sample["timestamp"], dr_sample["actual_load_kwh"], dr_sample["optimized_load_kwh"], where=(dr_sample["actual_load_kwh"] < dr_sample["optimized_load_kwh"]), color="#00875A", alpha=0.3, label="Shifted Off-Peak")

    plt.title("Demand-Response Peak Shaving & Redistribution")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand (kWh)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demand_response_shifting.png"), dpi=200)
    plt.close()

    # 3. Residual Error Distribution (Sampled to max 10,000 points)
    plt.figure(figsize=(9, 4))
    sample_size = min(10000, len(gb_results_df))
    gb_res = (gb_results_df["actual_load_kwh"] - gb_results_df["predicted_load_kwh"]).sample(sample_size, random_state=42)
    pt_res = (pt_results_df["actual_load_kwh"] - pt_results_df["predicted_load_kwh"]).sample(sample_size, random_state=42)

    sns.kdeplot(gb_res, label="HistGradientBoosting", color="#005A9C", fill=True, alpha=0.25)
    sns.kdeplot(pt_res, label="PyTorch NN", color="#E30613", fill=True, alpha=0.25)

    plt.axvline(0, color="black", linestyle="--")
    plt.title("Residual Error Distribution")
    plt.xlabel("Error (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_error_distribution.png"), dpi=200)
    plt.close()

    # 4. Parity Scatter Plot (Sampled to max 5,000 points)
    plt.figure(figsize=(5.5, 5.5))
    gb_sub = gb_results_df.sample(min(5000, len(gb_results_df)), random_state=42)
    pt_sub = pt_results_df.sample(min(5000, len(pt_results_df)), random_state=42)

    plt.scatter(gb_sub["actual_load_kwh"], gb_sub["predicted_load_kwh"], alpha=0.15, s=8, color="#005A9C", label="GB")
    plt.scatter(pt_sub["actual_load_kwh"], pt_sub["predicted_load_kwh"], alpha=0.15, s=8, color="#E30613", label="PyTorch")

    max_v = max(gb_sub["actual_load_kwh"].max(), pt_sub["actual_load_kwh"].max())
    plt.plot([0, max_v], [0, max_v], 'k--', label="Ideal")

    plt.title("Actual vs Predicted Parity Plot")
    plt.xlabel("Actual Demand (kWh)")
    plt.ylabel("Predicted Demand (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted_parity.png"), dpi=200)
    plt.close()

    plt.close('all')
    gc.collect()
    print(f"  -> Generated 4 plots saved in '{output_dir}/'.")