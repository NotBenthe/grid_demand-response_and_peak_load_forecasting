import sys
import gc
from data_loader import load_smart_meter_dataset
from feature_engineering import build_time_series_features
from model import train_grid_load_model, train_pytorch_load_model
from demand_response import simulate_demand_response
from visualization import plot_all_results


class TeeLogger:
    """Duplicates stdout output to both terminal console and results.txt."""
    def __init__(self, filename="results.txt"):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def run_pipeline():
    logger = TeeLogger("results.txt")
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("==================================================")
        print(" E.ON Smart Grid Forecasting & DR Pipeline ")
        print("==================================================")

        data_dir = "./data"
        print("\n[1/5] Ingesting smart meter, weather, and holiday data (num_blocks=15)...")
        raw_df = load_smart_meter_dataset(data_dir, num_blocks=15)

        print("\n[2/5] Engineering interaction, Fourier seasonal, & lag features...")
        featured_df = build_time_series_features(raw_df)
        del raw_df
        gc.collect()

        print("\n[3/5] Training Models...")
        
        # Original Gradient Boosting Method
        print("  -> Training Original Method (HistGradientBoosting)...")
        gb_model, gb_preds, gb_metrics = train_grid_load_model(featured_df)
        print(f"     MAE: {gb_metrics['MAE']:.3f} | RMSE: {gb_metrics['RMSE']:.3f} | R²: {gb_metrics['R2']:.3f}")

        # PyTorch Neural Network Addition
        print("  -> Training Addition (PyTorch Residual NN)...")
        pt_model, pt_preds, pt_metrics = train_pytorch_load_model(
            featured_df, epochs=40, batch_size=128, learning_rate=0.001
        )
        print(f"     MAE: {pt_metrics['MAE']:.3f} | RMSE: {pt_metrics['RMSE']:.3f} | R²: {pt_metrics['R2']:.3f}")

        del featured_df
        gc.collect()

        print("\n[4/5] Executing Demand-Response Load-Shifting Comparison...")
        
        gb_dr_df, gb_dr_summary = simulate_demand_response(
            gb_preds, peak_percentile=0.90, dr_shaving_factor=0.15
        )
        pt_dr_df, pt_dr_summary = simulate_demand_response(
            pt_preds, peak_percentile=0.90, dr_shaving_factor=0.15
        )

        print("\n--- Demand Response Optimization Comparison ---")
        print(f"{'Metric':<32} | {'Original (GB)':<15} | {'PyTorch NN':<15}")
        print("-" * 68)
        print(f"{'Initial PAR':<32} | {gb_dr_summary['initial_par']:<15.3f} | {pt_dr_summary['initial_par']:<15.3f}")
        print(f"{'Optimized PAR':<32} | {gb_dr_summary['optimized_par']:<15.3f} | {pt_dr_summary['optimized_par']:<15.3f}")
        print(f"{'Peak Trigger Threshold (kWh)':<32} | {gb_dr_summary['peak_threshold_kwh']:<15.2f} | {pt_dr_summary['peak_threshold_kwh']:<15.2f}")
        print(f"{'Max Peak Shaved (kWh)':<32} | {gb_dr_summary['peak_load_reduction_kwh']:<15.2f} | {pt_dr_summary['peak_load_reduction_kwh']:<15.2f}")
        print(f"{'Total Energy Shifted (kWh)':<32} | {gb_dr_summary['total_shaved_energy_kwh']:<15.2f} | {pt_dr_summary['total_shaved_energy_kwh']:<15.2f}")

        print("\n[5/5] Generating Visual Evaluation Plots...")
        plot_all_results(gb_preds, pt_preds, gb_dr_df, output_dir="./plots")

        print("\nPipeline execution completed successfully.")
        print("Summary metrics logged to 'results.txt'. Plots saved in './plots/'.")

    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":
    run_pipeline()