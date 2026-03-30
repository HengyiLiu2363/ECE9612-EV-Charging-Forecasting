import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clustering_model import (
    load_and_preprocess_data,
    evaluate_k_values,
    train_kmeans,
    build_cluster_summary,
    get_cluster_centers_original_units,
    plot_silhouette_scores,
    plot_clusters,
)


def main():
    # Adjust this path if your processed CSV is stored elsewhere
    csv_path = PROJECT_ROOT / "data" / "processed" / "acn_sessions_flat.csv"
    final_k = 4

    # 1. Load and preprocess data
    df_clean, X_scaled, scaler, features = load_and_preprocess_data(csv_path)

    print("Data preprocessing completed.")
    print(f"Total valid sessions for clustering: {len(df_clean)}")

    # 2. Evaluate k from 2 to 6
    print("\nEvaluating K-Means for k = 2 to 6...")
    sil_scores = evaluate_k_values(X_scaled, k_values=range(2, 7))

    for k, score in sil_scores.items():
        print(f"k = {k}, Silhouette Score = {score:.4f}")

    # 3. Train final model with fixed k = 4
    model, labels, final_score = train_kmeans(X_scaled, n_clusters=final_k)
    df_clean["Cluster"] = labels

    print(f"\nFinal K-Means model uses k = {final_k}")
    print(f"Silhouette Score for k = {final_k}: {final_score:.4f}")

    # 4. Cluster summary
    cluster_summary = build_cluster_summary(df_clean, cluster_col="Cluster")
    print("\nCluster Summary:")
    print(cluster_summary)

    # 5. Cluster centers in original units
    centers_df = get_cluster_centers_original_units(model, scaler, features)
    print("\nCluster Centers:")
    print(centers_df)

    # 6. Plots
    plot_silhouette_scores(sil_scores, chosen_k=final_k)
    plot_clusters(
        df_clean,
        cluster_col="Cluster",
        final_k=final_k,
        final_score=final_score
    )


if __name__ == "__main__":
    main()