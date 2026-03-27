from __future__ import annotations

# Handle file paths for model inputs and saved figures
from pathlib import Path

# Use pandas for loading the feature datasets
import pandas as pd

# Use matplotlib for result visualization
import matplotlib.pyplot as plt

# Random Forest model and regression metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree


# Feature datasets created from the feature engineering step
TRAIN_FILE = Path("data/processed/train_features.csv")
VAL_FILE = Path("data/processed/val_features.csv")
TEST_FILE = Path("data/processed/test_features.csv")

# Save kWh result figures here
FIGURE_DIR = Path("reports/figures/kwh")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset for training or evaluation.
    """
    df = pd.read_csv(file_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


def get_feature_columns() -> list[str]:
    """
    Define the predictor columns used by the Random Forest kWh model.
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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
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
    Plot actual vs predicted hourly total_kwh for a selected time window.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["actual"] = y_true.values
    plot_df["predicted"] = y_pred
    plot_df = plot_df.iloc[:n_points]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["hour"], plot_df["actual"], label="Actual")
    plt.plot(plot_df["hour"], plot_df["predicted"], label="Predicted")
    plt.title(f"Random Forest (kWh): Actual vs Predicted ({dataset_name})")
    plt.xlabel("Time")
    plt.ylabel("Total kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved figure to: {output_file}")


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_cols: list[str],
    output_file: Path,
) -> None:
    """
    Plot feature importance for the Random Forest kWh model.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance")

    plt.figure(figsize=(10, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Random Forest Feature Importance for kWh Forecasting")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved feature importance plot to: {output_file}")


def plot_sample_tree(
    model: RandomForestRegressor,
    feature_cols: list[str],
    output_file: Path,
    tree_index: int = 0,
    max_depth: int = 3,
) -> None:
    """
    Plot one sample decision tree from the Random Forest model.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    tree = model.estimators_[tree_index]

    plt.figure(figsize=(16, 8))
    plot_tree(
        tree,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
    )
    plt.title(f"Sample Tree from Random Forest for kWh (tree_index={tree_index}, max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved sample tree plot to: {output_file}")


def main() -> None:
    """
    Train and evaluate the Random Forest model for hourly total_kwh forecasting.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)
    test = load_data(TEST_FILE)

    feature_cols = get_feature_columns()
    target_col = "total_kwh"

    # Main Random Forest tuning knobs
    n_estimators = 200
    max_depth = 10
    min_samples_split = 5
    random_state = 42

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print(
        f"Using Random Forest for kWh with n_estimators={n_estimators}, "
        f"max_depth={max_depth}, min_samples_split={min_samples_split}"
    )

    evaluate_regression(y_train, train_pred, "Train")
    evaluate_regression(y_val, val_pred, "Validation")
    evaluate_regression(y_test, test_pred, "Test")

    plot_actual_vs_predicted(
        val,
        y_val,
        val_pred,
        "Validation",
        FIGURE_DIR / "random_forest_kwh_val_actual_vs_pred.png",
        n_points=300,
    )

    plot_actual_vs_predicted(
        test,
        y_test,
        test_pred,
        "Test",
        FIGURE_DIR / "random_forest_kwh_test_actual_vs_pred.png",
        n_points=300,
    )

    plot_feature_importance(
        model,
        feature_cols,
        FIGURE_DIR / "random_forest_kwh_feature_importance.png",
    )

    plot_sample_tree(
        model,
        feature_cols,
        FIGURE_DIR / "random_forest_kwh_sample_tree.png",
        tree_index=0,
        max_depth=3,
    )


if __name__ == "__main__":
    main()