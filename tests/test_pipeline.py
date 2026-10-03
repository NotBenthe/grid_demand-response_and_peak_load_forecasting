import pytest
import pandas as pd
import numpy as np
import torch

from feature_engineering import build_time_series_features
from demand_response import simulate_demand_response
from model import PyTorchGridNN

@pytest.fixture
def mock_raw_dataframe():
    """Generates 500 half-hour rows (~10.4 days) to cover 7-day lag requirements."""
    timestamps = pd.date_range(start="2026-01-01", periods=500, freq="30min")
    
    # Use structured sine pattern to simulate realistic daily peak/off-peak cycles
    time_hours = np.arange(500) * 0.5
    load = 30.0 + 15.0 * np.sin(2 * np.pi * time_hours / 24.0) + np.random.normal(0, 2, size=500)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "total_load_kwh": load.astype(np.float32),
        "temperature": np.random.uniform(5, 20, size=500).astype(np.float32),
        "humidity": np.full(500, 0.7, dtype=np.float32),
        "windSpeed": np.full(500, 5.0, dtype=np.float32),
        "is_bank_holiday": np.zeros(500, dtype=np.int8),
    })


def test_feature_engineering_output_types_and_nans(mock_raw_dataframe):
    """Verifies feature output contains no NaNs and enforces float32 types."""
    featured_df = build_time_series_features(mock_raw_dataframe)

    # 500 rows - 336 lag slots = 164 valid rows
    assert not featured_df.empty, "Featured dataframe should not be empty."
    assert len(featured_df) == 164, f"Expected 164 rows after lag_336 dropna, got {len(featured_df)}"
    assert "sin_slot" in featured_df.columns, "Missing cyclical slot feature."
    assert "rolling_mean_6h" in featured_df.columns, "Missing short rolling feature."
    assert featured_df["lag_1"].dtype == np.float32, "Feature float precision must be float32."
    assert featured_df.isnull().sum().sum() == 0, "Engineered features should not contain NaNs."


def test_demand_response_par_reduction(mock_raw_dataframe):
    """Verifies Demand Response logic shifts load and calculates valid PAR scores."""
    predictions_df = pd.DataFrame({
        "timestamp": mock_raw_dataframe["timestamp"],
        "actual_load_kwh": mock_raw_dataframe["total_load_kwh"],
        "predicted_load_kwh": mock_raw_dataframe["total_load_kwh"],
    })

    optimized_df, summary = simulate_demand_response(
        predictions_df, peak_percentile=0.85, dr_shaving_factor=0.15
    )

    assert "optimized_load_kwh" in optimized_df.columns
    assert summary["optimized_par"] <= summary["initial_par"], (
        f"Optimized PAR ({summary['optimized_par']:.3f}) should be <= Initial PAR ({summary['initial_par']:.3f})"
    )
    assert summary["total_shaved_energy_kwh"] > 0, "Energy shaving must be greater than 0."


def test_pytorch_model_forward_pass():
    """Verifies PyTorch network tensor dimensional outputs."""
    input_dim = 23
    batch_size = 16
    model = PyTorchGridNN(input_dim=input_dim)
    model.eval()

    dummy_input = torch.randn(batch_size, input_dim, dtype=torch.float32)
    output = model(dummy_input)

    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}."