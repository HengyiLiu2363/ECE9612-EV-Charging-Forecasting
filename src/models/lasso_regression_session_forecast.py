from __future__ import annotations

# Handle file paths for data loading and figure saving
from pathlib import Path

# Use pandas for loading the feature datasets
import pandas as pd

# Use matplotlib for result visualization
import matplotlib.pyplot as plt

# Lasso Regression model and evaluation metrics
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Feature datasets created from the feature engineering step
TRAIN_FILE = Path("data/processed/train_features.csv")
VAL_FILE = Path("data/processed/val_features.csv")
TEST_FILE = Path("data/processed/test_features.csv")

# Save result figures here for the report and presentation
FIGURE_DIR = Path("reports/figures")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset for training or evaluation.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert the hour column into datetime format for plotting
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")

    return df


def get_feature_columns() -> list[str]:
    """
    Define the predictor columns used by the Lasso Regression model.

    These are feature-selection tuning knobs.
    """
    return [
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
    ]


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series, dataset_name: str) -> dict[str, float]:
    """
    Compute and print standard regression metrics.
    """
    # MAE measures average absolute error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE penalizes larger errors more strongly
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # R² measures explained variance
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} results")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

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
    Plot actual vs predicted session counts for a selected time window.

    n_points is a presentation tuning knob.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Build a plotting table with timestamps, actual values, and predictions
    plot_df = df.copy()
    plot_df["actual"] = y_true.values
    plot_df["predicted"] = y_pred

    # Limit the plot to a cleaner time window
    plot_df = plot_df.iloc[:n_points]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["hour"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["hour"], plot_df["predicted"], label="Predicted")
    plt.title(f"Lasso Regression: Actual vs Predicted ({dataset_name})")
    plt.xlabel("Time")
    plt.ylabel("Session Count")
    plt.legend()
    plt.tight_layout()

    # Save figure for later use
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_file}")


def plot_coefficients(model: Lasso, feature_cols: list[str], output_file: Path) -> None:
    """
    Plot the learned Lasso coefficients to compare feature influence.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
    }).sort_values("coefficient")

    plt.figure(figsize=(10, 5))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.title("Lasso Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    # Save coefficient figure
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved coefficient plot to: {output_file}")


def main() -> None:
    """
    Train and evaluate the Lasso Regression forecasting model.

    The target variable is hourly session_count.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    feature_cols = get_feature_columns()
    target_col = "session_count"

    # This is the main Lasso tuning knob.
    # Larger alpha means stronger coefficient shrinkage.
    alpha = 0.001

    # Split each dataset into predictors and target
    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # Train the Lasso Regression model
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)

    # Generate predictions on all datasets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print(f"Using Lasso alpha = {alpha}")

    # Evaluate model performance
    evaluate_regression(y_train, train_pred, "Train")
    evaluate_regression(y_val, val_pred, "Validation")
    evaluate_regression(y_test, test_pred, "Test")

    # Save validation and test prediction plots
    plot_actual_vs_predicted(
        val,
        y_val,
        val_pred,
        "Validation",
        FIGURE_DIR / "lasso_regression_val_actual_vs_pred.png",
        n_points=300,
    )

    plot_actual_vs_predicted(
        test,
        y_test,
        test_pred,
        "Test",
        FIGURE_DIR / "lasso_regression_test_actual_vs_pred.png",
        n_points=300,
    )

    # Save coefficient plot
    plot_coefficients(
        model,
        feature_cols,
        FIGURE_DIR / "lasso_regression_coefficients.png",
    )


if __name__ == "__main__":
    main()