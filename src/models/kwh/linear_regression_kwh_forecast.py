from __future__ import annotations

# Handle file paths for the model input data and output figures
from pathlib import Path

# Use pandas for loading the feature datasets
import pandas as pd

# Use matplotlib for model result visualization
import matplotlib.pyplot as plt

# Linear Regression model and evaluation metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Feature datasets created from the feature engineering step
TRAIN_FILE = Path("data/processed/train_features.csv")
VAL_FILE = Path("data/processed/val_features.csv")
TEST_FILE = Path("data/processed/test_features.csv")

# Save result figures here for the report and presentation
FIGURE_DIR = Path("reports/figures/kwh")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset for training or evaluation.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert the hour column to datetime so it can be used in time-based plots
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")

    return df


def get_feature_columns() -> list[str]:
    """
    Define the predictor columns for the linear regression kWh model.

    These are feature-selection tuning knobs.
    """
    return [
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "session_lag_1",
        "session_lag_2",
        "session_lag_24",
        "session_lag_168",
        "session_rolling_mean_24",
        "kwh_lag_1",
        "kwh_lag_2",
        "kwh_lag_24",
        "kwh_lag_168",
        "kwh_rolling_mean_24",
    ]


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series, dataset_name: str) -> dict[str, float]:
    """
    Compute and print standard regression metrics.
    """
    # MAE shows the average absolute prediction error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE gives more penalty to larger prediction errors
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # R² measures how much variance in the target is explained by the model
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} results")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    # Return metrics in case we want to store or compare them later
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    dataset_name: str,
    output_file: Path,
    n_points: int = 300,
) -> None:
    """
    Plot actual vs predicted hourly total_kwh for a selected time window.

    n_points is a presentation tuning knob.
    """
    # Make sure the figure output folder exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Build a plotting table that includes timestamps, actual values, and predictions
    plot_df = df.copy()
    plot_df["actual"] = y_true.values
    plot_df["predicted"] = y_pred

    # Keep only the first part of the dataset for a cleaner time-series figure
    plot_df = plot_df.iloc[:n_points]

    # Plot actual and predicted energy demand on the same axes
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["hour"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["hour"], plot_df["predicted"], label="Predicted")
    plt.title(f"Linear Regression (kWh): Actual vs Predicted ({dataset_name})")
    plt.xlabel("Time")
    plt.ylabel("Total kWh")
    plt.legend()
    plt.tight_layout()

    # Save the figure to disk for report and presentation use
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_file}")


def plot_coefficients(model: LinearRegression, feature_cols: list[str], output_file: Path) -> None:
    """
    Plot the learned coefficients so we can see how each feature influences kWh prediction.
    """
    # Make sure the figure output folder exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Match each feature with its learned coefficient
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient")

    # Plot coefficients as a horizontal bar chart for easier reading
    plt.figure(figsize=(10, 5))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.title("Linear Regression Coefficients for kWh Forecasting")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    # Save the coefficient plot
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved coefficient plot to: {output_file}")


def main() -> None:
    """
    Train and evaluate the linear regression model for hourly total_kwh forecasting.
    """
    # Load the train, validation, and test feature datasets
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    # Define the predictors and the forecasting target
    feature_cols = get_feature_columns()
    target_col = "total_kwh"

    # Split each dataset into predictors and target
    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # Create and fit the linear regression model using the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict total_kwh for train, validation, and test sets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Evaluate model performance on each split
    evaluate_regression(y_train, train_pred, "Train")
    evaluate_regression(y_val, val_pred, "Validation")
    evaluate_regression(y_test, test_pred, "Test")

    # Save a validation forecast figure
    plot_actual_vs_predicted(
        val,
        y_val,
        val_pred,
        "Validation",
        FIGURE_DIR / "linear_regression_kwh_val_actual_vs_pred.png",
        n_points=300,
    )

    # Save a test forecast figure
    plot_actual_vs_predicted(
        test,
        y_test,
        test_pred,
        "Test",
        FIGURE_DIR / "linear_regression_kwh_test_actual_vs_pred.png",
        n_points=300,
    )

    # Save the coefficient plot to interpret the learned feature weights
    plot_coefficients(
        model,
        feature_cols,
        FIGURE_DIR / "linear_regression_kwh_coefficients.png",
    )


if __name__ == "__main__":
    main()