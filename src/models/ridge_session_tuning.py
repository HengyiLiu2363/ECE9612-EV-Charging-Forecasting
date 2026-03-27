from __future__ import annotations

# Handle file paths
from pathlib import Path

# Use pandas and numpy for data handling and parameter sweep
import pandas as pd
import numpy as np

# Use matplotlib for tuning visualizations
import matplotlib.pyplot as plt

# Ridge Regression model and evaluation metric
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Feature datasets
TRAIN_FILE = Path("data/processed/train_features.csv")
VAL_FILE = Path("data/processed/val_features.csv")
TEST_FILE = Path("data/processed/test_features.csv")

# Save tuning figures here
FIGURE_DIR = Path("reports/figures")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset for tuning and evaluation.
    """
    df = pd.read_csv(file_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df


def get_feature_columns() -> list[str]:
    """
    Define the predictor columns used by the Ridge model.
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


def main() -> None:
    """
    Sweep across several Ridge alpha values and visualize
    how model performance and coefficients change.
    """
    train = load_data(TRAIN_FILE)
    val = load_data(VAL_FILE)

    feature_cols = get_feature_columns()
    target_col = "session_count"

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    # These are the Ridge tuning knob values we want to compare
    alpha_values = np.logspace(-3, 2, 10)

    results = []
    coefficients = []

    for alpha in alpha_values:
        # Train one Ridge model for each alpha value
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict on the validation set
        val_pred = model.predict(X_val)

        # Compute validation RMSE
        rmse = mean_squared_error(y_val, val_pred) ** 0.5
        results.append({"alpha": alpha, "val_rmse": rmse})

        # Store coefficients so we can plot how they change
        coef_row = {"alpha": alpha}
        for feature, coef in zip(feature_cols, model.coef_):
            coef_row[feature] = coef
        coefficients.append(coef_row)

    results_df = pd.DataFrame(results)
    coef_df = pd.DataFrame(coefficients)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Plot validation RMSE versus alpha
    plt.figure(figsize=(8, 5))
    plt.semilogx(results_df["alpha"], results_df["val_rmse"], marker="o")
    plt.title("Ridge Tuning: Validation RMSE vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Validation RMSE")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ridge_tuning_rmse_vs_alpha.png", dpi=200)
    plt.close()

    print(f"Saved figure to: {FIGURE_DIR / 'ridge_tuning_rmse_vs_alpha.png'}")

    # Plot coefficient paths versus alpha
    plt.figure(figsize=(10, 6))
    for feature in feature_cols:
        plt.semilogx(coef_df["alpha"], coef_df[feature], label=feature)

    plt.title("Ridge Tuning: Coefficient Paths vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ridge_tuning_coefficient_paths.png", dpi=200)
    plt.close()

    print(f"Saved figure to: {FIGURE_DIR / 'ridge_tuning_coefficient_paths.png'}")

    print("\nValidation RMSE by alpha:")
    print(results_df)
    

if __name__ == "__main__":
    main()