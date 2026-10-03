import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Gradient Boosting Optimization
# ==========================================
def train_grid_load_model(df: pd.DataFrame):
    """Tuned HistGradientBoosting with L2 Regularization & Early Stopping."""
    feature_cols = [
        "hour", "dayofweek", "is_weekend", "is_bank_holiday", "month",
        "sin_slot", "cos_slot", "sin_weekly", "cos_weekly",
        "temperature", "humidity", "windSpeed", "temp_x_hour",
        "lag_1", "lag_2", "lag_48", "lag_336",
        "rolling_mean_6h", "rolling_std_6h",
        "rolling_mean_24h", "rolling_std_24h", "rolling_max_24h", "ema_12h",
    ]
    target_col = "total_load_kwh"

    X = df[feature_cols].astype(np.float32)
    y = df[target_col].astype(np.float32)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_timestamps = df["timestamp"].iloc[split_idx:]

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test).astype(np.float32)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual_load_kwh": y_test,
        "predicted_load_kwh": preds,
    })

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, results_df, metrics


# ==========================================
# 2. PyTorch Optimization
# ==========================================
class PyTorchGridNN(nn.Module):
    """Residual Neural Network with Layer Normalization for faster convergence."""

    def __init__(self, input_dim: int):
        super(PyTorchGridNN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.dense_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = x + self.res_block(x)
        x = self.dense_block(x)
        return self.head(x)


def train_pytorch_load_model(
    df: pd.DataFrame, epochs: int = 60, batch_size: int = 128, learning_rate: float = 0.0015
):
    """Trains PyTorch NN with AdamW and dynamic Learning Rate Schedule."""
    torch.manual_seed(42)
    np.random.seed(42)

    feature_cols = [
        "hour", "dayofweek", "is_weekend", "is_bank_holiday", "month",
        "sin_slot", "cos_slot", "sin_weekly", "cos_weekly",
        "temperature", "humidity", "windSpeed", "temp_x_hour",
        "lag_1", "lag_2", "lag_48", "lag_336",
        "rolling_mean_6h", "rolling_std_6h",
        "rolling_mean_24h", "rolling_std_24h", "rolling_max_24h", "ema_12h",
    ]
    target_col = "total_load_kwh"

    X = df[feature_cols].astype(np.float32).values
    y = df[target_col].astype(np.float32).values.reshape(-1, 1)

    split_idx = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]
    test_timestamps = df["timestamp"].iloc[split_idx:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler_X.transform(X_test_raw).astype(np.float32)
    y_train_scaled = scaler_y.fit_transform(y_train_raw).astype(np.float32)

    del X_train_raw, y_train_raw, X
    gc.collect()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PyTorchGridNN(input_dim=X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step(epoch_loss)

    model.eval()
    with torch.no_grad():
        scaled_preds = model(X_test_tensor).numpy()

    unscaled_preds = scaler_y.inverse_transform(scaled_preds).flatten().astype(np.float32)
    y_test = y_test_raw.flatten().astype(np.float32)

    mae = mean_absolute_error(y_test, unscaled_preds)
    rmse = np.sqrt(mean_squared_error(y_test, unscaled_preds))
    r2 = r2_score(y_test, unscaled_preds)

    results_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "actual_load_kwh": y_test,
        "predicted_load_kwh": unscaled_preds,
    })

    del X_train_tensor, y_train_tensor, X_test_tensor, train_dataset, train_loader
    gc.collect()

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, results_df, metrics