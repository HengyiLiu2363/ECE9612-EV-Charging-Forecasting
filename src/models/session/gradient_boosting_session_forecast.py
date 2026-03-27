from __future__ import annotations

# Handle file paths for model inputs and saved figures
from pathlib import Path

# Use pandas for loading the feature datasets
import pandas as pd

# Use matplotlib for result visualization
import matplotlib.pyplot as plt

# Gradient Boosting model and regression metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Feature datasets created from the feature engineering step
TRAIN_FILE = Path("data/processed/train_features.csv")
VAL_FILE = Path("data/processed/val_features.csv")
TEST_FILE = Path("data/processed/test_features.csv")

# Save result figures here for the report and presentation
FIGURE_DIR = Path("reports/figures/session")


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
    Define the predictor columns used by the Gradient Boosting model.

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
    ]


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series, dataset_name: str) -> dict[str, float]:
    """
    Compute and print standard regression metrics.
    """
    # MAE measures average absolute prediction error
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE penalizes larger prediction errors more strongly
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

    # Use only the first part of the dataset for a clearer plot
    plot_df = plot_df.iloc[:n_points]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["hour"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["hour"], plot_df["predicted"], label="Predicted")
    plt.title(f"Gradient Boosting: Actual vs Predicted ({dataset_name})")
    plt.xlabel("Time")
    plt.ylabel("Session Count")
    plt.legend()
    plt.tight_layout()

    # Save the figure to disk
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_file}")


def plot_feature_importance(
    model: GradientBoostingRegressor,
    feature_cols: list[str],
    output_file: Path,
) -> None:
    """
    Plot feature importance to show which inputs matter most to the Gradient Boosting model.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance")

    plt.figure(figsize=(10, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Gradient Boosting Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    # Save the feature importance plot
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved feature importance plot to: {output_file}")


def main() -> None:
    """
    Train and evaluate the Gradient Boosting forecasting model.

    The target variable is hourly session_count.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    feature_cols = get_feature_columns()
    target_col = "session_count"

    # These are the main Gradient Boosting tuning knobs.
    n_estimators = 200
    learning_rate = 0.05
    max_depth = 3
    random_state = 42

    # Split each dataset into predictors and target
    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # Train the Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Generate predictions on each dataset
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print(
        f"Using Gradient Boosting with n_estimators={n_estimators}, "
        f"learning_rate={learning_rate}, max_depth={max_depth}"
    )

    # Evaluate model performance
    evaluate_regression(y_train, train_pred, "Train")
    evaluate_regression(y_val, val_pred, "Validation")
    evaluate_regression(y_test, test_pred, "Test")

    # Save validation and test forecast plots
    plot_actual_vs_predicted(
        val,
        y_val,
        val_pred,
        "Validation",
        FIGURE_DIR / "gradient_boosting_val_actual_vs_pred.png",
        n_points=300,
    )

    plot_actual_vs_predicted(
        test,
        y_test,
        test_pred,
        "Test",
        FIGURE_DIR / "gradient_boosting_test_actual_vs_pred.png",
        n_points=300,
    )

    # Save feature importance plot
    plot_feature_importance(
        model,
        feature_cols,
        FIGURE_DIR / "gradient_boosting_feature_importance.png",
    )


if __name__ == "__main__":
    main()