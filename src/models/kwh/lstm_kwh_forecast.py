from __future__ import annotations

# Handle file paths for input datasets and saved figures
from pathlib import Path

# Use pandas and numpy for data handling
import numpy as np
import pandas as pd

# Use matplotlib for training and forecast visualization
import matplotlib.pyplot as plt

# Use sklearn metrics and scaling for regression evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Use Keras for the LSTM sequence model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping


# Hourly split datasets for sequence forecasting
TRAIN_FILE = Path("data/processed/train_hourly.csv")
VAL_FILE = Path("data/processed/val_hourly.csv")
TEST_FILE = Path("data/processed/test_hourly.csv")

# Save kWh LSTM figures here
FIGURE_DIR = Path("reports/figures/kwh")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one hourly dataset for sequence forecasting.
    """
    df = pd.read_csv(file_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


def create_sequences(series: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D time series into supervised LSTM sequences.

    Each input sequence uses the previous sequence_length values,
    and the target is the next value.
    """
    X, y = [], []

    for i in range(sequence_length, len(series)):
        X.append(series[i - sequence_length:i])
        y.append(series[i])

    return np.array(X), np.array(y)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> dict[str, float]:
    """
    Compute and print standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} results")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_loss(history, output_file: Path) -> None:
    """
    Plot training and validation loss over epochs.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM (kWh) Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved loss plot to: {output_file}")


def plot_actual_vs_predicted(
    hours: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    output_file: Path,
    n_points: int = 300,
) -> None:
    """
    Plot actual vs predicted hourly total_kwh for a selected time window.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = pd.DataFrame({
        "hour": hours.reset_index(drop=True),
        "actual": y_true,
        "predicted": y_pred,
    }).iloc[:n_points]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["hour"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["hour"], plot_df["predicted"], label="Predicted")
    plt.title(f"LSTM (kWh): Actual vs Predicted ({dataset_name})")
    plt.xlabel("Time")
    plt.ylabel("Total kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_file}")


def main() -> None:
    """
    Train and evaluate the LSTM model for hourly total_kwh forecasting.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    target_col = "total_kwh"

    # Main LSTM tuning knobs
    sequence_length = 24
    lstm_units = 64
    epochs = 30
    batch_size = 32

    print(
        f"Using LSTM for kWh with sequence_length={sequence_length}, "
        f"lstm_units={lstm_units}, epochs={epochs}, batch_size={batch_size}"
    )

    # Extract the target time series
    train_series = train[target_col].values.reshape(-1, 1)
    val_series = val[target_col].values.reshape(-1, 1)
    test_series = test[target_col].values.reshape(-1, 1)

    # Scale the target values so LSTM training is more stable
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)

    # Build supervised sequences from the scaled target series
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    # Reshape into [samples, timesteps, features] for Keras LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    # Build the LSTM sequence model
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(lstm_units),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")

    # Stop early if validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping],
    )

    # Generate predictions on each split
    train_pred_scaled = model.predict(X_train, verbose=0)
    val_pred_scaled = model.predict(X_val, verbose=0)
    test_pred_scaled = model.predict(X_test, verbose=0)

    # Convert predictions back to the original kWh scale
    train_pred = scaler.inverse_transform(train_pred_scaled).flatten()
    val_pred = scaler.inverse_transform(val_pred_scaled).flatten()
    test_pred = scaler.inverse_transform(test_pred_scaled).flatten()

    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluate model performance
    evaluate_regression(y_train_true, train_pred, "Train")
    evaluate_regression(y_val_true, val_pred, "Validation")
    evaluate_regression(y_test_true, test_pred, "Test")

    # Save training-loss figure
    plot_loss(
        history,
        FIGURE_DIR / "lstm_kwh_training_loss.png",
    )

    # Align timestamps to the sequence targets
    val_hours = val["hour"].iloc[sequence_length:].reset_index(drop=True)
    test_hours = test["hour"].iloc[sequence_length:].reset_index(drop=True)

    # Save validation and test forecast plots
    plot_actual_vs_predicted(
        val_hours,
        y_val_true,
        val_pred,
        "Validation",
        FIGURE_DIR / "lstm_kwh_val_actual_vs_pred.png",
        n_points=300,
    )

    plot_actual_vs_predicted(
        test_hours,
        y_test_true,
        test_pred,
        "Test",
        FIGURE_DIR / "lstm_kwh_test_actual_vs_pred.png",
        n_points=300,
    )


if __name__ == "__main__":
    main()