import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_and_preprocess_data(csv_path):
    """
    Load data, create duration feature, clean invalid rows, and scale features.
    Returns:
        df_clean: cleaned dataframe
        X_scaled: scaled feature matrix
        scaler: fitted StandardScaler
        features: feature names used for clustering
    """
    df = pd.read_csv(csv_path)

    df["connectionTime"] = pd.to_datetime(df["connectionTime"])
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"])

    df["duration_hours"] = (
        df["disconnectTime"] - df["connectionTime"]
    ).dt.total_seconds() / 3600.0

    # Keep only valid sessions
    df_clean = df[
        (df["duration_hours"] > 0) &
        (df["kWhDelivered"] > 0) &
        (df["duration_hours"] <= 48)
    ].copy()

    features = ["duration_hours", "kWhDelivered"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    return df_clean, X_scaled, scaler, features


def evaluate_k_values(X_scaled, k_values=range(2, 7), random_state=42, n_init=10):
    """
    Evaluate silhouette score for multiple k values.
    """
    sil_scores = {}

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores[k] = score

    return sil_scores


def train_kmeans(X_scaled, n_clusters=4, random_state=42, n_init=10):
    """
    Train final KMeans model with fixed k.
    Returns:
        model, labels, silhouette score
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    return model, labels, score


def build_cluster_summary(df_clean, cluster_col="Cluster"):
    """
    Build cluster summary table.
    """
    summary = df_clean.groupby(cluster_col).agg(
        Average_Duration_Hours=("duration_hours", "mean"),
        Median_Duration_Hours=("duration_hours", "median"),
        Average_kWh_Delivered=("kWhDelivered", "mean"),
        Median_kWh_Delivered=("kWhDelivered", "median"),
        Session_Count=("duration_hours", "count")
    ).round(2)

    summary = summary.sort_values(by="Average_Duration_Hours")
    return summary


def get_cluster_centers_original_units(model, scaler, features):
    """
    Convert cluster centers from scaled space back to original feature units.
    """
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=features)
    centers_df["Cluster"] = range(len(centers_df))
    return centers_df.round(2)


def plot_silhouette_scores(sil_scores, chosen_k=4):
    """
    Plot silhouette scores for different k values.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Different K-Means Cluster Numbers")
    plt.xticks(list(sil_scores.keys()))
    plt.tight_layout()
    plt.show()


def plot_clusters(df_clean, cluster_col="Cluster", final_k=4, final_score=0.0):
    """
    Plot clustering result in original feature space.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    palette = sns.color_palette("tab10", n_colors=final_k)

    sns.scatterplot(
        data=df_clean,
        x="duration_hours",
        y="kWhDelivered",
        hue=cluster_col,
        palette=palette,
        alpha=0.6,
        s=45
    )

    plt.title(f"K-Means Clustering of EV Charging Sessions (k={final_k})")
    plt.xlabel("Connection Duration (Hours)")
    plt.ylabel("Energy Delivered (kWh)")
    plt.legend(title="Cluster", loc="upper right")
    plt.tight_layout()
    plt.show()